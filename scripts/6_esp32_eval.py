"""
BUOC 6 - Danh gia model tren ESP32 THAT qua cap USB.

  - Gui lan luot 624 anh test xuong ESP32:
        laptop -> ESP32 : "IMG0" + 64*64 byte anh xam
        ESP32 -> laptop : {"p":0.8730,"ms":346.2,"label":"PNEUMONIA"}
  - So voi ket qua int8 tren laptop (buoc 4): yeu cau >= 99% anh cung nhan
  - Ghi ra: results/esp32_report.txt, esp32_preds.csv, esp32_confusion.png

Chay (tat Serial Monitor truoc - cong COM chi mo duoc 1 noi):
    .\\eval_esp32.bat               (ca 624 anh, ~5 phut)
    .\\eval_esp32.bat --limit 20    (chay thu 20 anh)
"""
import argparse
import csv
import json
import sys
import time

import numpy as np

import common as C


class Esp32:
    def __init__(self, port, baud):
        import serial
        self.ser = serial.Serial()
        self.ser.port, self.ser.baudrate, self.ser.timeout = port, baud, 0.5
        self.ser.dtr = self.ser.rts = False          # khong vo tinh vao che do nap
        try:
            self.ser.open()
        except serial.SerialException as e:
            sys.exit(f"LOI: khong mo duoc {port}: {e}\n-> Tat Serial Monitor roi chay lai.")
        self.ser.rts = True; time.sleep(0.15); self.ser.rts = False     # reset board (nut EN)
        t0 = time.time()
        while time.time() - t0 < 25 and self.ser.readline().strip() != b"READY":
            pass
        self.ser.reset_input_buffer()

    def _reply(self, timeout=5.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            line = self.ser.readline().decode("utf-8", "replace").strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    return {"err": line}
        return {"err": "khong tra loi"}

    def info(self):
        self.ser.write(b"INFO\n")
        return self._reply()

    def predict(self, img):
        for _ in range(3):                           # thu lai toi da 3 lan
            self.ser.write(b"IMG0" + img.tobytes())
            r = self._reply()
            if "p" in r:
                return r
            time.sleep(0.2); self.ser.reset_input_buffer()
        return r


def save_confusion(path, m, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4.4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], C.CLASSES); ax.set_yticks([0, 1], C.CLASSES)
    ax.set_xlabel("Du doan (ESP32)"); ax.set_ylabel("Nhan that"); ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Buoc 6: danh gia tren ESP32")
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--limit", type=int, default=0, help="chi chay N anh (trai deu 2 lop)")
    args = ap.parse_args()

    d = np.load(C.DATASET)
    x, y, f = d["x_test"], d["y_test"].astype(int), d["f_test"]
    p_lap = np.load(C.MODELS / "probs_int8_test.npy")
    thr = json.loads((C.MODELS / "threshold.json").read_text())["threshold"]
    idx = np.arange(len(y))
    if 0 < args.limit < len(y):
        idx = np.unique(np.linspace(0, len(y) - 1, args.limit).round().astype(int))

    print(f"Mo {args.port}, reset board, cho 'READY' ...")
    dev = Esp32(args.port, args.baud)
    info = dev.info()
    print("ESP32:", info)
    if not info.get("ready"):
        sys.exit("LOI: ESP32 khong tra loi INFO hoac model chua san sang. Mo Serial Monitor xem log.")

    p_esp, ms = np.full(len(idx), np.nan), np.full(len(idx), np.nan)
    errors, t0 = [], time.time()
    try:
        for k, i in enumerate(idx):
            r = dev.predict(np.ascontiguousarray(x[i], dtype=np.uint8))
            if "p" in r:
                p_esp[k], ms[k] = r["p"], r.get("ms", np.nan)
            else:
                errors.append(f"{f[i]}: {r}")
            print(f"\r{k + 1}/{len(idx)} anh | {time.time() - t0:.0f}s | loi {len(errors)}   ", end="", flush=True)
    except KeyboardInterrupt:
        print("\nDung giua chung - bao cao tren cac anh da chay.")
    dev.ser.close()
    total = time.time() - t0
    print()

    ok = ~np.isnan(p_esp)
    if not ok.any():
        sys.exit(f"LOI: khong nhan duoc ket qua nao. {errors[:1]}")
    yi, pe, pl, fi, msv = y[idx][ok], p_esp[ok], p_lap[idx][ok], f[idx][ok], ms[ok]

    with open(C.RESULTS / "esp32_preds.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "label", "p_esp32", "p_laptop", "diff", "pred_esp32", "ms"])
        for row in zip(fi, yi, pe, pl, msv):
            w.writerow([row[0], C.CLASSES[row[1]], f"{row[2]:.4f}", f"{row[3]:.4f}", f"{abs(row[2] - row[3]):.4f}",
                        C.CLASSES[int(row[2] >= thr)], f"{row[4]:.1f}"])

    m = C.metrics(yi, pe, thr)
    same = float(np.mean((pe >= thr) == (pl >= thr)))
    diff = np.abs(pe - pl)
    mism = np.flatnonzero((pe >= thr) != (pl >= thr))
    L = ["BAO CAO DANH GIA TREN ESP32 THAT (buoc 6)", "=" * 78,
         f"Thoi diem : {time.strftime('%Y-%m-%d %H:%M')} | chip {info.get('chip')} @ {info.get('mhz')} MHz",
         f"So anh    : {ok.sum()}/{len(idx)} thanh cong, loi {len(errors)} | nguong {thr}", "",
         C.TABLE_HEAD, C.table_row("ESP32", yi, pe, thr), C.table_row("laptop int8", yi, pl, thr), "",
         f"Ma tran nham lan (ESP32): that NORMAL -> {m['tn']} dung, {m['fp']} bao benh | "
         f"that PNEUMONIA -> {m['tp']} dung, {m['fn']} bo sot",
         f"ESP32 vs laptop: cung nhan {same:.2%} (yeu cau >= 99%: {'DAT' if same >= 0.99 else 'KHONG DAT'}), "
         f"lech xac suat TB {diff.mean():.4f}, lon nhat {diff.max():.4f}"]
    L += [f"   khac nhan: {fi[j]}  ESP32 {pe[j]:.4f} | laptop {pl[j]:.4f}" for j in mism[:20]]
    L += [f"Thoi gian : Invoke TB {np.nanmean(msv):.1f} ms | ca vong (truyen + chay) {total / len(idx) * 1000:.0f} ms/anh"]
    L += [f"Loi: {e}" for e in errors[:20]]
    text = "\n".join(L)
    (C.RESULTS / "esp32_report.txt").write_text(text + "\n", encoding="utf-8")
    save_confusion(C.RESULTS / "esp32_confusion.png", m, f"ESP32 - test ({ok.sum()} anh), nguong {thr}")
    print(text)


if __name__ == "__main__":
    main()
