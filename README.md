# Nhận diện viêm phổi từ ảnh X-quang trên ESP32

Một mô hình AI nhỏ (MobileNetV2, 630 KB, int8) chạy ngay trên chip **ESP32-WROOM-32D**.
ESP32 tự phát WiFi. Điện thoại kết nối vào, mở trang web, chụp ảnh phim X-quang và nhận kết quả:
**Bình thường**, **Không chắc chắn** hoặc **Nghi viêm phổi**.

---

## 1. Cách hệ thống hoạt động

```
 Điện thoại (trình duyệt)                         ESP32
 ┌───────────────────────────────┐   WiFi    ┌──────────────────────────────┐
 │ 1. Chụp / chọn ảnh X-quang    │           │ web.cpp                      │
 │ 2. JavaScript: ảnh xám,       │  POST     │   nhận 4096 byte (base64)    │
 │    thu về 64x64               │ ───────►  │ model.cpp                    │
 │    (= 4096 byte)              │ /predict  │   chạy model int8 (~350 ms)  │
 │ 4. Hiện kết quả               │ ◄───────  │   -> xác suất p (0..1)       │
 │                               │   JSON    │ 3. so p với ngưỡng           │
 └───────────────────────────────┘           └──────────────────────────────┘
```

- Điện thoại làm phần nặng: đọc ảnh lớn, chuyển xám, thu nhỏ. ESP32 chỉ chạy mô hình.
- Ra quyết định dựa trên xác suất `p`:
  - `p < 0.033` là **Bình thường**.
  - `0.033 ≤ p < 0.22` là **Không chắc chắn** (nên chụp lại).
  - `p ≥ 0.22` là **Nghi viêm phổi**.
- Cả hai ngưỡng được chọn trên tập VAL, không dùng tập TEST để chọn.

## 2. Cấu trúc thư mục

```
PNEUMONIA CV/
├── verify.bat          Biên dịch firmware (giống nút Verify của Arduino IDE)
├── upload.bat          Biên dịch + nạp vào board + mở Serial Monitor
├── monitor.bat         Chỉ mở Serial Monitor
├── eval_esp32.bat      Đánh giá 624 ảnh test trên ESP32 thật qua cáp USB
│
├── firmware/           Chương trình chạy trên ESP32 (PlatformIO)
│   ├── platformio.ini    cấu hình board, cổng COM3
│   ├── web/index.html    giao diện trang web (sửa giao diện ở đây)
│   └── src/
│       ├── main.cpp      khởi động + nhận lệnh qua USB
│       ├── model.cpp/.h  chạy mô hình AI
│       ├── web.cpp/.h    phát WiFi + trang web
│       └── model_data.h, test_image.h, web_page.h   <- TỰ SINH, không sửa tay
│
├── scripts/            Code Python trên laptop, chạy lần lượt từ 1 đến 6
│   ├── common.py         hàm dùng chung: đường dẫn, xử lý ảnh, augment, chỉ số
│   ├── 1_split_data.py   chia dữ liệu gốc thành train / val / test (theo bệnh nhân)
│   ├── 2_make_dataset.py thu ảnh về 64x64, lưu models/data64.npz
│   ├── 3_train.py        train mô hình -> models/model.keras
│   ├── 4_quantize.py     nén sang int8, chọn ngưỡng + vùng không chắc chắn
│   ├── 5_export_firmware.py  xuất mô hình + trang web thành file .h cho firmware
│   └── 6_esp32_eval.py   gửi ảnh test xuống ESP32, so với laptop
│
├── models/             Dữ liệu 64x64, mô hình, ngưỡng (threshold.json)
├── results/            Báo cáo và biểu đồ của từng bước
├── data_split/         Ảnh X-quang đã chia (không có trên GitHub, tạo bằng bước 1)
├── docs/               Tài liệu hướng dẫn dự án
├── requirements.txt    Danh sách thư viện Python
└── venv-tf/            Môi trường Python (không có trên GitHub, tạo theo mục 3)
```

## 3. Cài đặt (máy mới)

Phần cứng cần có:
- Board ESP32 có 4 MB Flash (ví dụ ESP32-WROOM-32D) và cáp USB.
- Một điện thoại có trình duyệt.

Phần mềm cần cài:

1. **Python 3.12** và các thư viện:
   ```
   python -m venv venv-tf
   venv-tf\Scripts\pip install -r requirements.txt
   ```
2. **VS Code** và extension **PlatformIO IDE**.
   PlatformIO tự tải thư viện TensorFlow Lite Micro khi build lần đầu.
   Nếu board nhận cổng khác COM3, sửa cổng trong `firmware/platformio.ini`.
3. Nạp firmware bằng `.\upload.bat`.
   Model đã train sẵn nằm trong `firmware/src/model_data.h`, nên không cần train lại.

Dữ liệu dùng để train:
- Bộ [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) trên Kaggle, giấy phép CC BY 4.0.
- Ảnh gốc khoảng 1.2 GB nên không đưa lên repo. Repo đã có sẵn bản rút gọn 64×64 là `models/data64.npz`, đủ để chạy lại từ bước 4 trở đi.
- Muốn làm lại từ bước 1: tải bộ ảnh gốc, giải nén vào thư mục `data/`.

## 4. Các lệnh hay dùng

Chạy trong terminal của VS Code, ở thư mục gốc `PNEUMONIA CV`.

| Việc cần làm | Lệnh |
|---|---|
| Nạp firmware vào board | `.\upload.bat` |
| Xem log của board | `.\monitor.bat` (thoát bằng Ctrl + C) |
| Chỉ kiểm tra code biên dịch được | `.\verify.bat` |
| Đánh giá trên ESP32 (tắt Monitor trước) | `.\eval_esp32.bat` hoặc `.\eval_esp32.bat --limit 20` |
| Sau khi sửa giao diện `firmware/web/index.html` | `venv-tf\Scripts\python scripts\5_export_firmware.py --web` rồi `.\upload.bat` |

Dùng trên điện thoại:
1. Kết nối WiFi **Pneumonia-Checker**, mật khẩu `xquang2026`.
2. Mở `http://192.168.4.1`.

## 5. Làm lại từ đầu (pipeline đầy đủ)

```
venv-tf\Scripts\python scripts\1_split_data.py --src data   (data = bộ Kaggle chest_xray đã giải nén)
venv-tf\Scripts\python scripts\2_make_dataset.py
venv-tf\Scripts\python scripts\3_train.py                    (~15 phút)
venv-tf\Scripts\python scripts\4_quantize.py
venv-tf\Scripts\python scripts\5_export_firmware.py
.\upload.bat
.\eval_esp32.bat
```

| Bước | Đọc vào | Ghi ra |
|---|---|---|
| 1 | ảnh gốc Kaggle | `data_split/`, `results/split_report.txt` |
| 2 | `data_split/` | `models/data64.npz`, `results/data_report.txt` |
| 3 | `data64.npz` | `models/model.keras`, `results/eval_float.txt`, `training_curves.png` |
| 4 | `model.keras` | `models/model_int8.tflite`, `threshold.json`, `results/int8_report.txt` |
| 5 | `.tflite`, `threshold.json`, `index.html` | `firmware/src/model_data.h`, `test_image.h`, `web_page.h` |
| 6 | board qua USB | `results/esp32_report.txt`, `esp32_confusion.png` |

## 6. Vài điểm kỹ thuật đáng chú ý

- **Bộ nhớ.** Mô hình cần khoảng 125 KB RAM, nhưng ESP32 không có vùng RAM liền mạch nào đủ lớn.
  `model.cpp` nên chia làm 2 vùng:
  - 66 KB RAM tĩnh cho dữ liệu tạm.
  - Khoảng 72 KB lấy từ heap cho phần cố định.
- **Chuẩn hóa nằm trong mô hình.** ESP32 chỉ đưa pixel thô vào: `q = pixel − 128`, rồi chép ra 3 kênh R = G = B.
- **Tự kiểm tra khi khởi động.** ESP32 chạy 2 ảnh mẫu và so với kết quả trên laptop.
  - Lệch dưới 0.01 thì in ra `KHỚP`.
  - Kiểm tra này nằm trong `test_image.h`.
- **Kết quả.** Với 624 ảnh test ở ngưỡng 0.09:
  - Accuracy 0.846, AUC 0.950.
  - Recall viêm phổi 0.985, recall bình thường 0.615.
  - ESP32 cho cùng nhãn với laptop ở 99.2% số ảnh.
  - Thời gian chạy khoảng 346 ms/ảnh.
