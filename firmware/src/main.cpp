// =============================================================================
// Nhan dien viem phoi tu anh X-quang tren ESP32 (du an hoc tap - KHONG dung de chan doan).
//
//   model.cpp : chay model int8 (TensorFlow Lite Micro)
//   web.cpp   : phat WiFi + trang web cho dien thoai
//   main.cpp  : khoi dong + nhan lenh qua cap USB (Serial 921600 baud)
//
// Lenh qua USB (dung boi scripts/6_esp32_eval.py):
//   INFO\n            -> 1 dong JSON thong tin
//   "IMG0" + 4096 byte anh xam 64x64 -> {"p":0.8730,"ms":346.2,"label":"PNEUMONIA"}
// =============================================================================
#include <Arduino.h>
#include <esp_heap_caps.h>

#include "model.h"
#include "web.h"

static void print_info() {
  Serial.printf("{\"chip\":\"%s\",\"mhz\":%lu,\"img\":%d,\"thr\":%.4f,\"model_kb\":%u,\"arena_kb\":%u,"
                "\"ready\":%s,\"free_kb\":%lu}\n",
                ESP.getChipModel(), (unsigned long)ESP.getCpuFreqMHz(), IMG_SIZE, THRESHOLD, g_model_len / 1024,
                (unsigned)model_arena_kb(), model_ready() ? "true" : "false",
                (unsigned long)(heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024));
}

// Sau khi gap "IMG0": doc 4096 byte anh, chay model, tra 1 dong JSON.
static void handle_image() {
  uint8_t* img = image_buffer();
  if (!img || !model_ready()) { Serial.println("{\"err\":\"model\"}"); return; }
  const size_t got = Serial.readBytes(img, IMG_BYTES);   // cho toi da 2 giay
  if (got != IMG_BYTES) { Serial.printf("{\"err\":\"timeout\",\"got\":%u}\n", (unsigned)got); return; }
  uint32_t us = 0;
  const float p = model_predict(img, &us);
  if (p < 0.0f) { Serial.println("{\"err\":\"invoke\"}"); return; }
  Serial.printf("{\"p\":%.4f,\"ms\":%.1f,\"label\":\"%s\"}\n", p, us / 1000.0f, label_of(p));
}

void setup() {
  Serial.setRxBufferSize(8192);   // du chua tron 1 anh khi truyen nhanh
  Serial.begin(921600);
  Serial.setTimeout(2000);
  delay(500);
  Serial.printf("\n=== Pneumonia X-ray tren %s (anh %dx%d, int8) ===\n", ESP.getChipModel(), IMG_SIZE, IMG_SIZE);

  if (model_init() && image_buffer()) {
    Serial.printf("Model OK: %u KB, RAM model %u KB, nguong %.2f\n", g_model_len / 1024,
                  (unsigned)model_arena_kb(), THRESHOLD);
    Serial.println(model_self_test() ? "Tu kiem tra: KHOP voi laptop." : "Tu kiem tra: KHONG khop!");
  } else {
    Serial.println("LOI: khoi tao model that bai.");
  }
  web_start();
  Serial.println("READY");
}

// Doc tung byte tu USB: gap "IMG0" -> nhan anh; gap xuong dong -> lenh chu (INFO).
void loop() {
  web_loop();
  static String line;
  while (Serial.available()) {
    const char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      line.trim();
      if (line.equalsIgnoreCase("INFO")) print_info();
      line = "";
      continue;
    }
    if (line.length() >= 32) line.remove(0, 1);
    line += c;
    if (line.endsWith("IMG0")) {
      line = "";
      handle_image();
    }
  }
  delay(1);
}
