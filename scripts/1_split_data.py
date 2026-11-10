"""
BUOC 1 - Chia du lieu X-quang goc (Kaggle chest_xray) thanh train / val / test.

  - test : giu nguyen tap test goc (624 anh)
  - train + val goc gop lai roi chia lai THEO BENH NHAN (1 nguoi khong nam o ca 2 tap)
  - Ghi ra: data_split/{train,val,test}/{NORMAL,PNEUMONIA}/, results/split_report.txt, results/samples.png

Chay:  venv-tf\\Scripts\\python scripts\\1_split_data.py --src data
"""
import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import common as C


def find_dataset_root(src: Path) -> Path:
    """Tim thu muc chua train/ va test/ (bo Kaggle hay long 2 lop chest_xray/chest_xray)."""
    for p in [src] + [p for p in src.rglob("*") if p.is_dir() and "__MACOSX" not in p.parts]:
        if (p / "train").is_dir() and (p / "test").is_dir():
            return p
    raise SystemExit(f"Khong thay thu muc co train/ va test/ trong '{src}'.")


def patient_key(path: Path, cls: str) -> str:
    """person123_bacteria_456.jpeg -> 'person123' | IM-0115-0001.jpeg -> 'IM-0115'."""
    m = re.match(r"(person\d+)", path.stem, re.I) if cls == "PNEUMONIA" else re.search(r"(IM-\d+)", path.stem, re.I)
    return m.group(1).lower() if m else path.stem


def split_by_patient(files, cls, val_ratio, rng):
    groups = defaultdict(list)
    for f in files:
        groups[patient_key(f, cls)].append(f)
    keys = sorted(groups)
    rng.shuffle(keys)
    val_keys = set(keys[:max(1, round(len(keys) * val_ratio))])
    train = [f for k in keys if k not in val_keys for f in groups[k]]
    val = [f for k in keys if k in val_keys for f in groups[k]]
    return train, val


def copy_files(files, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for f in files:
        target, i = dst / f.name, 1
        while target.exists():                       # tranh trung ten khi gop train + val goc
            target, i = dst / f"{f.stem}_dup{i}{f.suffix}", i + 1
        shutil.copy2(f, target)


def samples_grid(out: Path, per_class=6, size=200, seed=0):
    rng, rows = random.Random(seed), []
    for cls in C.CLASSES:
        tiles = []
        for p in rng.sample(C.list_images(C.DATA_SPLIT / "train" / cls), per_class):
            t = cv2.cvtColor(cv2.resize(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), (size, size)), cv2.COLOR_GRAY2BGR)
            cv2.rectangle(t, (0, 0), (size, 26), (0, 0, 0), -1)
            cv2.putText(t, cls, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(t)
        rows.append(np.hstack(tiles))
    cv2.imwrite(str(out), np.vstack(rows))


def main():
    ap = argparse.ArgumentParser(description="Buoc 1: chia du lieu")
    ap.add_argument("--src", default="data", help="thu muc chua bo du lieu goc da giai nen")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="xoa data_split cu roi lam lai")
    args = ap.parse_args()

    root, dst = find_dataset_root(Path(args.src)), C.DATA_SPLIT
    if dst.exists() and any(dst.iterdir()):
        if not args.force:
            raise SystemExit(f"'{dst}' da co du lieu. Them --force neu muon lam lai.")
        shutil.rmtree(dst)
    C.RESULTS.mkdir(exist_ok=True)
    rng, counts = random.Random(args.seed), {"train": {}, "val": {}, "test": {}}

    for cls in C.CLASSES:
        test = C.list_images(root / "test" / cls)
        copy_files(test, dst / "test" / cls)
        pool = C.list_images(root / "train" / cls) + C.list_images(root / "val" / cls)
        train, val = split_by_patient(pool, cls, args.val_ratio, rng)
        copy_files(train, dst / "train" / cls)
        copy_files(val, dst / "val" / cls)
        counts["test"][cls], counts["train"][cls], counts["val"][cls] = len(test), len(train), len(val)

    lines = ["BAO CAO CHIA DU LIEU", "=" * 44, f"seed={args.seed}, val_ratio={args.val_ratio}", "",
             f"{'Tap':<8}{'NORMAL':>10}{'PNEUMONIA':>12}{'Tong':>10}"]
    for s in ["train", "val", "test"]:
        n, p = counts[s]["NORMAL"], counts[s]["PNEUMONIA"]
        lines.append(f"{s:<8}{n:>10}{p:>12}{n + p:>10}")
    lines.append(f"\nTi le PNEUMONIA/NORMAL o tap train: {counts['train']['PNEUMONIA'] / max(counts['train']['NORMAL'], 1):.2f}")
    report = "\n".join(lines)
    (C.RESULTS / "split_report.txt").write_text(report, encoding="utf-8")
    samples_grid(C.RESULTS / "samples.png")
    print(report)


if __name__ == "__main__":
    main()
