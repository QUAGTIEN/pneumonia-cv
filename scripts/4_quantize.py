"""
BUOC 4 - Luong tu hoa int8, danh gia tren laptop, chon nguong va vung "khong chac chan".

  1. models/model.keras -> models/model_int8.tflite (full int8; 300 anh train lam representative)
  2. Chay ban int8 tren VAL + TEST bang DUNG cach ESP32 se lam (kernel tham chieu BUILTIN_REF)
  3. Chon nguong tren VAL: balanced accuracy cao nhat, giu recall PNEUMONIA >= 0.95
  4. Chon vung khong chac [lo, hi) quanh nguong: doi xung theo log-odds, toi da 5% anh VAL roi vao
  - Ghi ra: models/threshold.json, models/probs_int8_{val,test}.npy, results/int8_report.txt

Chay:  venv-tf\\Scripts\\python scripts\\4_quantize.py
"""
import argparse
import json

import numpy as np

import common as C


def quantize(tf, model, x_train, y_train, n_rep, seed):
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.flatnonzero(y_train == c), n_rep // 2, replace=False) for c in (0, 1)])
    rng.shuffle(idx)
    rep = C.to_model_input(x_train[idx])

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([rep[i:i + 1]] for i in range(len(rep)))
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def run_int8(tf, blob, x):
    """Xac suat PNEUMONIA cho tung anh, giong het ESP32:
         dau vao q = round(pixel / scale) + zero_point;  dau ra p = (q - zero_point) * scale"""
    it = tf.lite.Interpreter(model_content=blob,
                             experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    it.allocate_tensors()
    i_d, o_d = it.get_input_details()[0], it.get_output_details()[0]
    (i_s, i_z), (o_s, o_z) = i_d["quantization"], o_d["quantization"]
    xs = C.to_model_input(x)
    probs = np.empty(len(xs), dtype=np.float32)
    for k in range(len(xs)):
        it.set_tensor(i_d["index"], np.clip(np.round(xs[k:k + 1] / i_s) + i_z, -128, 127).astype(np.int8))
        it.invoke()
        probs[k] = (float(it.get_tensor(o_d["index"])[0][1]) - o_z) * o_s
    ops = sorted({op["op_name"] for op in it._get_ops_details()} - {"DELEGATE"})
    return probs, ops, (i_s, i_z, o_s, o_z)


def pick_threshold(y, p, min_recall):
    rows = [C.metrics(y, p, t) for t in np.round(np.arange(0.05, 0.991, 0.01), 2)]
    ok = [r for r in rows if r["rec_pneu"] >= min_recall] or rows
    return max(ok, key=lambda r: (round(r["bal"], 6), -abs(r["thr"] - 0.5)))["thr"]


def pick_uncertain(y, p, thr, max_frac):
    """Khoang rong nhat logit(thr) +- m sao cho <= max_frac anh VAL roi vao."""
    logit = lambda v: np.log(v / (1 - v))
    sigm = lambda z: 1 / (1 + np.exp(-z))
    best = (thr, thr)
    for m in np.arange(0.05, 3.001, 0.05):
        lo, hi = round(float(sigm(logit(thr) - m)), 3), round(float(sigm(logit(thr) + m)), 3)
        if np.mean((p >= lo) & (p < hi)) <= max_frac:
            best = (lo, hi)
    return best


def main():
    ap = argparse.ArgumentParser(description="Buoc 4: luong tu hoa + chon nguong")
    ap.add_argument("--n-rep", type=int, default=300, help="so anh representative")
    ap.add_argument("--min-recall", type=float, default=0.95, help="recall PNEUMONIA toi thieu tren VAL")
    ap.add_argument("--max-uncertain", type=float, default=0.05, help="toi da ti le anh VAL 'khong chac'")
    args = ap.parse_args()

    import tensorflow as tf
    d = np.load(C.DATASET)
    yv, yt = d["y_val"].astype(int), d["y_test"].astype(int)

    model = tf.keras.models.load_model(C.MODELS / "model.keras")
    float_kb = len(tf.lite.TFLiteConverter.from_keras_model(model).convert()) / 1024
    blob = quantize(tf, model, d["x_train"], d["y_train"], args.n_rep, seed=0)
    (C.MODELS / "model_int8.tflite").write_bytes(blob)

    print("Chay ban int8 tren VAL + TEST ...")
    pv, ops, (i_s, i_z, o_s, o_z) = run_int8(tf, blob, d["x_val"])
    pt, _, _ = run_int8(tf, blob, d["x_test"])
    np.save(C.MODELS / "probs_int8_val.npy", pv); np.save(C.MODELS / "probs_int8_test.npy", pt)
    pv_f, pt_f = np.load(C.MODELS / "probs_float_val.npy"), np.load(C.MODELS / "probs_float_test.npy")

    thr = pick_threshold(yv, pv, args.min_recall)
    lo, hi = pick_uncertain(yv, pv, thr, args.max_uncertain)
    cfg = {"threshold": thr, "uncertain_lo": lo, "uncertain_hi": hi,
           "min_recall_val": args.min_recall, "uncertain_max_val": args.max_uncertain, "chosen_on": "val"}
    (C.MODELS / "threshold.json").write_text(json.dumps(cfg, indent=2) + "\n")

    unc = lambda p: np.mean((p >= lo) & (p < hi))
    L = ["BAO CAO MODEL INT8 (buoc 4)", "=" * 78,
         f"Kich thuoc : float32 {float_kb:.0f} KB -> int8 {len(blob) / 1024:.0f} KB",
         f"Dau vao    : scale {i_s:.6f}, zero_point {i_z}   |   Dau ra: scale {o_s:.6f}, zero_point {o_z}",
         f"Phep toan  : {', '.join(ops)}  (dang ky trong firmware/src/model.cpp)",
         "", f"Nguong (chon tren VAL, recall PNEUMONIA >= {args.min_recall}): {thr:.2f}",
         f"Vung khong chac chan: [{lo}, {hi})  -> VAL {unc(pv):.1%} anh, TEST {unc(pt):.1%} anh",
         "", "[VAL]", C.TABLE_HEAD,
         C.table_row("float32", yv, pv_f, 0.5), C.table_row("int8", yv, pv, 0.5), C.table_row("int8 @ nguong", yv, pv, thr),
         "", "[TEST]", C.TABLE_HEAD,
         C.table_row("float32", yt, pt_f, 0.5), C.table_row("int8", yt, pt, 0.5), C.table_row("int8 @ nguong", yt, pt, thr),
         "", f"float32 vs int8 tren TEST: lech xac suat trung binh {np.abs(pt - pt_f).mean():.4f}, "
             f"cung nhan {np.mean((pt >= thr) == (pt_f >= thr)):.1%} (nguong {thr:.2f})",
         "FP = NORMAL bi bao benh, FN = PNEUMONIA bi bo sot. Du an hoc tap - KHONG dung de chan doan."]
    text = "\n".join(L)
    (C.RESULTS / "int8_report.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
