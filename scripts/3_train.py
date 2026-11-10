"""
BUOC 3 - Train model nho: MobileNetV2 (alpha 0.35), anh 64x64, TensorFlow/Keras.

  - Giai doan 1: dong bang MobileNetV2, chi train lop cuoi (5 epoch, lr 1e-3)
  - Giai doan 2: mo khoa toan bo, tinh chinh (toi da 30 epoch, lr 1e-4), dung som sau 8 epoch khong tot hon
  - BatchNorm giu che do inference (on dinh voi anh nho); class weights bu du lieu lech
  - Moi epoch augment lai tap train (gia lap chup bang camera)
  - Luu model co AUC tren VAL cao nhat: models/model.keras
  - Ghi ra: models/probs_float_{val,test}.npy, results/training_log.csv, training_curves.png, eval_float.txt

Chay:  venv-tf\\Scripts\\python scripts\\3_train.py        (~15 phut tren CPU)
"""
import argparse
import csv
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import common as C


def make_epoch(x, seed):
    """1 ban augment moi cho ca tap train."""
    seeds = np.random.SeedSequence(seed).spawn(len(x))
    with ThreadPoolExecutor(8) as ex:
        out = list(ex.map(lambda i: C.augment(x[i], np.random.default_rng(seeds[i])), range(len(x))))
    return C.to_model_input(np.stack(out))


def build_model(tf, alpha, dropout):
    L = tf.keras.layers
    inp = L.Input((C.SIZE, C.SIZE, 3), name="image")                 # pixel 0..255
    x = L.Rescaling(1 / 127.5, offset=-1, name="normalize")(inp)     # chuan hoa NAM TRONG model
    base = tf.keras.applications.MobileNetV2(input_shape=(C.SIZE, C.SIZE, 3), alpha=alpha,
                                             include_top=False, weights="imagenet")
    x = base(x, training=False)                                       # BatchNorm luon o che do inference
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(dropout)(x)
    out = L.Dense(2, activation="softmax", name="prob")(x)            # [NORMAL, PNEUMONIA]
    return tf.keras.Model(inp, out), base


def save_curves(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="val")
    ax[0].set_title("Loss")
    ax[1].plot(ep, [h["train_acc"] for h in history], label="train acc")
    ax[1].plot(ep, [h["val_acc"] for h in history], label="val acc")
    ax[1].plot(ep, [h["val_auc"] for h in history], ":", label="val AUC")
    ax[1].set_title("Val metrics")
    for a in ax:
        a.set_xlabel("epoch"); a.legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Buoc 3: train model nho")
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--epochs-head", type=int, default=5)
    ap.add_argument("--epochs-ft", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import tensorflow as tf
    tf.keras.utils.set_random_seed(args.seed)

    d = np.load(C.DATASET)
    xtr, ytr = d["x_train"], d["y_train"].astype(np.int64)
    xva, yva = C.to_model_input(d["x_val"]), d["y_val"].astype(np.int64)
    xte, yte = C.to_model_input(d["x_test"]), d["y_test"].astype(np.int64)
    w = len(ytr) / (2.0 * np.bincount(ytr, minlength=2))
    class_weight = {0: float(w[0]), 1: float(w[1])}
    print(f"Train {len(ytr)} | Val {len(yva)} | Test {len(yte)} | class weights {w.round(2)}")

    model, base = build_model(tf, args.alpha, args.dropout)
    best_path = C.MODELS / "model.keras"
    history, best, best_loss, best_epoch, epoch, t0 = [], -1.0, 1e9, 0, 0, time.time()

    for phase, n_ep, lr in [("head", args.epochs_head, 1e-3), ("finetune", args.epochs_ft, 1e-4)]:
        if n_ep <= 0:
            continue
        base.trainable = phase == "finetune"
        sched = tf.keras.optimizers.schedules.CosineDecay(lr, decay_steps=int(np.ceil(len(ytr) / args.batch)) * n_ep)
        model.compile(optimizer=tf.keras.optimizers.AdamW(sched, weight_decay=1e-4),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        print(f"\n=== {phase} ({n_ep} epoch, lr={lr}) ===")
        bad = 0
        for _ in range(n_ep):
            epoch += 1
            h = model.fit(make_epoch(xtr, args.seed * 1000 + epoch), ytr, batch_size=args.batch, epochs=1,
                          shuffle=True, class_weight=class_weight, verbose=0)
            pva = model.predict(xva, batch_size=128, verbose=0)
            m = C.metrics(yva, pva[:, 1])
            vloss = float(tf.keras.losses.sparse_categorical_crossentropy(yva, pva).numpy().mean())
            history.append({"epoch": epoch, "phase": phase, "train_loss": float(h.history["loss"][0]),
                            "train_acc": float(h.history["accuracy"][0]), "val_loss": vloss,
                            "val_acc": m["acc"], "val_auc": m["auc"]})
            improved = m["auc"] > best + 1e-9 or (abs(m["auc"] - best) <= 1e-9 and vloss < best_loss)
            if improved:
                best, best_loss, best_epoch, bad = m["auc"], vloss, epoch, 0
                model.save(best_path)
            else:
                bad += 1
            print(f"Epoch {epoch:2d} | val loss {vloss:.3f} acc {m['acc']:.3f} AUC {m['auc']:.3f}"
                  f"{'  <-- luu' if improved else ''}", flush=True)
            if phase == "finetune" and bad >= args.patience:
                print("Dung som.")
                break

    with open(C.RESULTS / "training_log.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(history[0]))
        wr.writeheader(); wr.writerows(history)
    save_curves(history, C.RESULTS / "training_curves.png")

    best_model = tf.keras.models.load_model(best_path)
    pva = best_model.predict(xva, batch_size=128, verbose=0)[:, 1]
    pte = best_model.predict(xte, batch_size=128, verbose=0)[:, 1]
    np.save(C.MODELS / "probs_float_val.npy", pva); np.save(C.MODELS / "probs_float_test.npy", pte)
    text = "\n".join([
        "BAO CAO MODEL FLOAT32 (buoc 3)", "=" * 60,
        f"MobileNetV2 alpha={args.alpha}, anh {C.SIZE}x{C.SIZE}, {best_model.count_params():,} tham so",
        f"Epoch tot nhat: {best_epoch} | thoi gian train {(time.time() - t0) / 60:.1f} phut", "",
        C.TABLE_HEAD, C.table_row("VAL", yva, pva, 0.5), C.table_row("TEST", yte, pte, 0.5)])
    (C.RESULTS / "eval_float.txt").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
