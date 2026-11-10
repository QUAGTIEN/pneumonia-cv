"""
BUOC 5 - Xuat model + trang web thanh file C++ cho firmware (firmware/src/).

  - model_data.h : model int8 (mang byte), kich thuoc anh, nguong, vung khong chac
  - test_image.h : 2 anh test 64x64 (1 NORMAL, 1 PNEUMONIA) + xac suat tren laptop,
                   ESP32 tu kiem tra khi khoi dong (lech phai < 0.01)
  - web_page.h   : trang web firmware/web/index.html (sua giao dien o index.html roi chay lai buoc nay)

Chay:  venv-tf\\Scripts\\python scripts\\5_export_firmware.py          (ca 3 file)
       venv-tf\\Scripts\\python scripts\\5_export_firmware.py --web    (chi trang web)
"""
import argparse
import json

import numpy as np

import common as C

HEADER = "// Tu sinh boi scripts/5_export_firmware.py - KHONG sua tay.\n#pragma once\n"


def c_bytes(data, per_line=16):
    return "\n".join("  " + ", ".join(f"0x{b:02x}" for b in data[i:i + per_line]) + ","
                     for i in range(0, len(data), per_line))


def export_model():
    blob = (C.MODELS / "model_int8.tflite").read_bytes()
    cfg = json.loads((C.MODELS / "threshold.json").read_text())
    (C.FIRMWARE_SRC / "model_data.h").write_text(
        HEADER + "#include <cstdint>\n\n"
        f"constexpr int IMG_SIZE = {C.SIZE};                      // anh xam {C.SIZE}x{C.SIZE}\n\n"
        "// Ket luan:  p < UNC_LO -> BINH THUONG | UNC_LO <= p < UNC_HI -> KHONG CHAC | p >= UNC_HI -> NGHI VIEM PHOI\n"
        f"constexpr float THRESHOLD = {cfg['threshold']:.2f}f;   // nguong 2 lop (chon tren VAL)\n"
        f"constexpr float UNC_LO = {cfg['uncertain_lo']:.3f}f;\n"
        f"constexpr float UNC_HI = {cfg['uncertain_hi']:.3f}f;\n\n"
        f"constexpr unsigned int g_model_len = {len(blob)};\n"
        f"alignas(16) const unsigned char g_model[] = {{\n{c_bytes(blob)}\n}};\n", encoding="utf-8")
    print(f"model_data.h : model {len(blob) / 1024:.0f} KB, nguong {cfg['threshold']:.2f}, "
          f"vung khong chac [{cfg['uncertain_lo']}, {cfg['uncertain_hi']})")
    return cfg["threshold"]


def export_test_images(thr):
    """Chon 2 anh test du doan DUNG nhung xac suat khong sat 0 / 1, de kiem tra phep tinh that su."""
    d = np.load(C.DATASET)
    x, y, f = d["x_test"], d["y_test"], d["f_test"]
    p = np.load(C.MODELS / "probs_int8_test.npy")
    ok_n, ok_p = np.flatnonzero((y == 0) & (p < thr)), np.flatnonzero((y == 1) & (p >= thr))
    picks = [("TEST_NORMAL", int(ok_n[np.argmin(np.abs(p[ok_n] - thr / 2))])),
             ("TEST_PNEUMONIA", int(ok_p[np.argmin(np.abs(p[ok_p] - (1 + thr) / 2))]))]
    parts = [HEADER + "#include <cstdint>\n"]
    for name, i in picks:
        parts += [f"// {f[i]}  ->  P(PNEUMONIA) tren laptop = {p[i]:.4f}",
                  f"constexpr float {name}_EXPECTED_P = {p[i]:.4f}f;",
                  f"const uint8_t {name}[{x[i].size}] = {{", c_bytes(x[i].tobytes()), "};", ""]
    (C.FIRMWARE_SRC / "test_image.h").write_text("\n".join(parts), encoding="utf-8")
    print("test_image.h : " + " | ".join(f"{f[i]} p={p[i]:.4f}" for _, i in picks))


def export_web_page():
    html = (C.ROOT / "firmware" / "web" / "index.html").read_text(encoding="utf-8")
    assert ')rawhtml"' not in html
    (C.FIRMWARE_SRC / "web_page.h").write_text(
        HEADER + 'static const char WEB_PAGE[] PROGMEM = R"rawhtml(' + html + ')rawhtml";\n', encoding="utf-8")
    print(f"web_page.h   : {len(html.encode('utf-8')) / 1024:.1f} KB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Buoc 5: xuat file cho firmware")
    ap.add_argument("--web", action="store_true", help="chi dong goi lai trang web (sau khi sua index.html)")
    if not ap.parse_args().web:
        export_test_images(export_model())
    export_web_page()
