"""
BUOC 2 - Tao bo du lieu 64x64 cho model nho.

Doc data_split/, ap dung common.prep64 (khong augment) va luu 1 file models/data64.npz:
    x_train, y_train, x_val, y_val, x_test, y_test   (x: uint8 [N,64,64], y: 0/1)
    f_train, f_val, f_test                            (ten file, de doi chieu)
Ghi them: results/data_report.txt, results/samples64.png

Chay:  venv-tf\\Scripts\\python scripts\\2_make_dataset.py
"""
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

import common as C

EXPECTED = {"train": 4470, "val": 762, "test": 624}     # so anh sau buoc 1


def load_split(split):
    items = C.list_split(split)
    with ThreadPoolExecutor(8) as ex:
        x = np.stack(list(ex.map(lambda it: C.prep64(C.read_image(it[0])), items))).astype(np.uint8)
    y = np.array([lbl for _, lbl in items], dtype=np.uint8)
    names = np.array([f"{C.CLASSES[lbl]}/{p.name}" for p, lbl in items])
    return x, y, names


def samples_grid(x, y, out, per_class=8, scale=3):
    rng, rows = np.random.default_rng(0), []
    for label, cls in enumerate(C.CLASSES):
        tiles = []
        for i in rng.choice(np.flatnonzero(y == label), per_class, replace=False):
            t = cv2.cvtColor(cv2.resize(x[i], None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST),
                             cv2.COLOR_GRAY2BGR)
            cv2.putText(t, cls[:4], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(cv2.copyMakeBorder(t, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
        rows.append(np.hstack(tiles))
    cv2.imencode(".png", np.vstack(rows))[1].tofile(str(out))


def main():
    C.MODELS.mkdir(exist_ok=True); C.RESULTS.mkdir(exist_ok=True)
    data, rep = {}, ["BAO CAO DU LIEU 64x64 (buoc 2)", "=" * 60,
                     "Tien xu ly: xam -> thu nho canh dai ve 512 -> resize 64x64 INTER_AREA", "",
                     f"{'Tap':<7}{'NORMAL':>9}{'PNEUMONIA':>11}{'Tong':>8}   Kiem tra"]
    for split in ["train", "val", "test"]:
        print(f"Doc tap {split} ...")
        x, y, names = load_split(split)
        data[f"x_{split}"], data[f"y_{split}"], data[f"f_{split}"] = x, y, names
        ok = len(y) == EXPECTED[split]
        rep.append(f"{split:<7}{int((y == 0).sum()):>9}{int((y == 1).sum()):>11}{len(y):>8}   "
                   + ("OK" if ok else f"KHAC ({EXPECTED[split]})"))

    np.savez_compressed(C.DATASET, **data, size=np.int32(C.SIZE))
    samples_grid(data["x_train"], data["y_train"], C.RESULTS / "samples64.png")
    rep.append(f"\nFile: {C.DATASET.relative_to(C.ROOT)} ({C.DATASET.stat().st_size / 1e6:.1f} MB)")
    text = "\n".join(rep)
    (C.RESULTS / "data_report.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
