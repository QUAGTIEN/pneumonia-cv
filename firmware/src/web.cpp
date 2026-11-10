// Cac dia chi:
//   GET  /         trang web (web/index.html, nhung san trong web_page.h)
//   GET  /info     thong tin model / bo nho (JSON)
//   POST /predict  than = base64 cua 64x64 byte anh xam -> JSON ket qua
// Dien thoai tu tien xu ly anh (JavaScript) roi chi gui 4096 byte, ESP32 chi chay model.
#include "web.h"

#include <WiFi.h>
#include <WebServer.h>
#include <esp_heap_caps.h>
#include "mbedtls/base64.h"

#include "model.h"
#include "web_page.h"

static const char* AP_SSID = "Pneumonia-Checker";
static const char* AP_PASS = "xquang2026";   // toi thieu 8 ky tu

static WebServer g_server(80);
static bool g_on = false;
static uint32_t g_count = 0;                 // so anh da nhan dien

static unsigned long free_kb() { return heap_caps_get_free_size(MALLOC_CAP_INTERNAL) / 1024; }

static void send_json(int code, const char* body) {
  g_server.sendHeader("Cache-Control", "no-store");
  g_server.send(code, "application/json", body);
}

static void on_root() {
  g_server.sendHeader("Cache-Control", "no-store");
  g_server.send_P(200, "text/html; charset=utf-8", WEB_PAGE, sizeof(WEB_PAGE) - 1);   // gui thang tu Flash
}

static void on_info() {
  char buf[200];
  snprintf(buf, sizeof(buf),
           "{\"chip\":\"%s\",\"mhz\":%lu,\"img\":%d,\"thr\":%.4f,\"lo\":%.4f,\"hi\":%.4f,\"ready\":%s,"
           "\"free_kb\":%lu,\"uptime_s\":%lu,\"count\":%lu}",
           ESP.getChipModel(), (unsigned long)ESP.getCpuFreqMHz(), IMG_SIZE, THRESHOLD, UNC_LO, UNC_HI, model_ready() ? "true" : "false", free_kb(),
           (unsigned long)(millis() / 1000), (unsigned long)g_count);
  send_json(200, buf);
}

static void on_predict() {
  uint8_t* img = image_buffer();
  if (!model_ready() || !img) return send_json(503, "{\"err\":\"model chua san sang\"}");
  const String& body = g_server.arg("plain");
  size_t len = 0;
  if (mbedtls_base64_decode(img, IMG_BYTES, &len, (const unsigned char*)body.c_str(), body.length()) != 0 ||
      len != IMG_BYTES)
    return send_json(400, "{\"err\":\"anh sai kich thuoc\"}");

  uint32_t us = 0;
  const float p = model_predict(img, &us);
  if (p < 0.0f) return send_json(500, "{\"err\":\"invoke\"}");
  ++g_count;
  char buf[200];
  snprintf(buf, sizeof(buf),
           "{\"p\":%.4f,\"ms\":%.1f,\"label\":\"%s\",\"decision\":\"%s\",\"thr\":%.4f,\"lo\":%.4f,\"hi\":%.4f}",
           p, us / 1000.0f, label_of(p), decide(p), THRESHOLD, UNC_LO, UNC_HI);
  send_json(200, buf);
  Serial.printf("[WEB] #%lu p=%.4f %s %.0f ms | RAM trong %lu KB\n", (unsigned long)g_count, p, decide(p),
                us / 1000.0f, free_kb());
}

static void on_not_found() {   // dia chi la -> ve trang chinh
  g_server.sendHeader("Location", "http://192.168.4.1/", true);
  g_server.send(302, "text/plain", "");
}

void web_start() {
  WiFi.mode(WIFI_AP);
  g_on = WiFi.softAP(AP_SSID, AP_PASS, 6, 0, 4);   // kenh 6, hien ten, toi da 4 thiet bi
  if (!g_on) {
    Serial.println("LOI: khong bat duoc WiFi.");
    return;
  }
  WiFi.setSleep(false);   // phan hoi nhanh hon
  WiFi.onEvent([](WiFiEvent_t e, WiFiEventInfo_t) {
    if (e == ARDUINO_EVENT_WIFI_AP_STACONNECTED) Serial.println("[WiFi] co thiet bi ket noi");
    if (e == ARDUINO_EVENT_WIFI_AP_STADISCONNECTED) Serial.println("[WiFi] thiet bi ngat ket noi");
  });
  g_server.on("/", HTTP_GET, on_root);
  g_server.on("/info", HTTP_GET, on_info);
  g_server.on("/predict", HTTP_POST, on_predict);
  g_server.onNotFound(on_not_found);
  g_server.begin();
  Serial.printf("WiFi \"%s\" (mat khau \"%s\") -> mo http://%s | RAM trong %lu KB\n", AP_SSID, AP_PASS,
                WiFi.softAPIP().toString().c_str(), free_kb());
}

void web_loop() {
  if (g_on) g_server.handleClient();
}
