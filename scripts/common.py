"""
Ham dung chung cho cac buoc 1-6: duong dan, tien xu ly anh, augment, chi so danh gia.

TIEN XU LY CHUAN (trang web JavaScript phai lam giong het):
    1. Anh mau -> anh xam (cong thuc cua OpenCV)
    2. Neu canh dai > 512 px: thu nho ve 512 (INTER_AREA)
    3. Resize ve 64 x 64 (INTER_AREA, khong giu ti le)
Model nhan pixel tho 0..255; viec chuan hoa nam TRONG model.
"""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_SPLIT = ROOT / "data_split"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIRMWARE_SRC = ROOT / "firmware" / "src"
DATASET = MODELS / "data64.npz"

SIZE = 64
MAX_SIDE = 512
CLASSES = ["NORMAL", "PNEUMONIA"]          # NORMAL = 0, PNEUMONIA = 1
IMG_EXTS = {".jpeg", ".jpg", ".png"}


# ----------------------------------------------------------------------------
# Doc anh / tien xu ly
# ----------------------------------------------------------------------------
def list_images(folder: Path):
    """Anh trong 1 thu muc, bo qua file rac cua macOS (._abc.jpeg, .DS_Store)."""
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.suffix.lower() in IMG_EXTS and not p.name.startswith("."))


def list_split(split: str):
    """[(duong_dan, nhan)] cua 1 tap (train / val / test) trong data_split/."""
    items = []
    for label, cls in enumerate(CLASSES):
        folder = DATA_SPLIT / split / cls
        if not folder.is_dir():
            raise SystemExit(f"Khong thay {folder}. Chay buoc 1 (1_split_data.py) truoc.")
        items += [(p, label) for p in list_images(folder)]
    return items


def read_image(path) -> np.ndarray:
    """Doc anh BGR. imdecode doc duoc duong dan co dau / dau cach tren Windows."""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Khong doc duoc anh: {path}")
    return img


def prep64(img: np.ndarray) -> np.ndarray:
    """HAM CHINH: anh bat ky (BGR hoac xam) -> anh xam uint8 [64, 64]."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        gray = cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return cv2.resize(gray, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def to_model_input(x: np.ndarray) -> np.ndarray:
    """uint8 [..,64,64] -> float32 [..,64,64,3] (R = G = B), gia tri 0..255."""
    return np.repeat(x.astype(np.float32)[..., None], 3, axis=-1)


# ----------------------------------------------------------------------------
# Augment (chi dung luc train) - gia lap anh chup bang camera.
# KHONG lat ngang: lat se doi vi tri tim trai <-> phai, sai y hoc.
# ----------------------------------------------------------------------------
def _affine(g, rng):                         # xoay +-10 do, zoom 0.9-1.1, dich 5%
    h, w = g.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-10, 10), rng.uniform(0.9, 1.1))
    m[:, 2] += (rng.uniform(-0.05, 0.05) * w, rng.uniform(-0.05, 0.05) * h)
    return cv2.warpAffine(g, m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _brightness_contrast(g, rng):
    return cv2.convertScaleAbs(g, alpha=rng.uniform(0.8, 1.2), beta=rng.uniform(-20, 20))


def _perspective(g, rng, strength=0.06):     # chup lech goc
    h, w = g.shape
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = (src + rng.uniform(-strength, strength, size=(4, 2)) * np.float32([w, h])).astype(np.float32)
    return cv2.warpPerspective(g, cv2.getPerspectiveTransform(src, dst), (w, h),
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _gamma(g, rng):
    lut = (np.linspace(0, 1, 256) ** rng.uniform(0.7, 1.5) * 255).astype(np.uint8)
    return cv2.LUT(g, lut)


def _blur(g, rng):
    return cv2.GaussianBlur(g, (3, 3), rng.uniform(0.3, 1.2))


def _noise(g, rng):
    return np.clip(g.astype(np.float32) + rng.normal(0, rng.uniform(3, 12), g.shape), 0, 255).astype(np.uint8)


def _moire(g, rng):                          # van song khi chup man hinh
    h, w = g.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    theta, freq, amp = rng.uniform(0, np.pi), rng.uniform(0.15, 0.6), rng.uniform(4, 12)
    wave = np.sin((xx * np.cos(theta) + yy * np.sin(theta)) * freq) * amp
    return np.clip(g.astype(np.float32) + wave, 0, 255).astype(np.uint8)


def _glare(g, rng):                          # vet choi sang
    h, w = g.shape
    cx, cy = rng.uniform(0.1, 0.9) * w, rng.uniform(0.1, 0.9) * h
    r = rng.uniform(0.10, 0.25) * max(h, w)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2)))
    return np.clip(g.astype(np.float32) + blob * rng.uniform(30, 90), 0, 255).astype(np.uint8)


def augment(g: np.ndarray, rng) -> np.ndarray:
    out = _brightness_contrast(_affine(g, rng), rng)
    for fn, p in [(_perspective, 0.5), (_gamma, 0.5), (_blur, 0.4),
                  (_noise, 0.5), (_moire, 0.3), (_glare, 0.2)]:
        if rng.random() < p:
            out = fn(out, rng)
    return out


# ----------------------------------------------------------------------------
# Chi so danh gia
# ----------------------------------------------------------------------------
def metrics(y, p, thr=0.5) -> dict:
    """y: nhan that (0/1), p: xac suat PNEUMONIA. FP = NORMAL bi bao benh, FN = PNEUMONIA bi bo sot."""
    from sklearn.metrics import roc_auc_score
    y, pred = np.asarray(y), (np.asarray(p) >= thr).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum()); tp = int(((y == 1) & (pred == 1)).sum())
    rn, rp = tn / max(tn + fp, 1), tp / max(tp + fn, 1)
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    return {"thr": float(thr), "acc": (tp + tn) / len(y), "auc": auc, "rec_norm": rn, "rec_pneu": rp,
            "bal": (rn + rp) / 2, "tn": tn, "fp": fp, "fn": fn, "tp": tp}


TABLE_HEAD = f"{'':24s}{'nguong':>7s}{'acc':>8s}{'AUC':>8s}{'rec PNEU':>10s}{'rec NORM':>10s}{'bal':>8s}{'FP':>6s}{'FN':>6s}"


def table_row(name, y, p, thr) -> str:
    m = metrics(y, p, thr)
    return (f"{name:24s}{thr:7.2f}{m['acc']:8.3f}{m['auc']:8.3f}{m['rec_pneu']:10.3f}"
            f"{m['rec_norm']:10.3f}{m['bal']:8.3f}{m['fp']:6d}{m['fn']:6d}")
