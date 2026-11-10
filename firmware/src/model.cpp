#include "model.h"

#include <esp_heap_caps.h>
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "test_image.h"

// -----------------------------------------------------------------------------
// Tensor arena = RAM model dung khi tinh toan. Model can ~125 KB, nhung ESP32 khong
// co khoi RAM lien mach nao du lon, nen CHIA LAM 2 VUNG:
//   - vung tam (anh trung gian giua cac lop, ~60 KB): mang tinh trong RAM
//   - vung co dinh (thong tin tensor, ~65 KB): cap tu heap, thu tu nho -> lon
// -----------------------------------------------------------------------------
constexpr size_t SCRATCH_KB = 66;
alignas(16) static uint8_t g_scratch[SCRATCH_KB * 1024];
static const size_t PERSIST_KB[] = {72, 80, 88, 104};
static size_t g_persist_kb = 0;

static tflite::MicroMutableOpResolver<7> g_resolver;   // 7 phep toan cua model (results/int8_report.txt)
static tflite::MicroInterpreter* g_interpreter = nullptr;
static TfLiteTensor* g_input = nullptr;
static TfLiteTensor* g_output = nullptr;

bool model_init() {
  const tflite::Model* model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("LOI: phien ban model khong khop thu vien TFLite.");
    return false;
  }
  g_resolver.AddAdd();
  g_resolver.AddConv2D();
  g_resolver.AddDepthwiseConv2D();
  g_resolver.AddFullyConnected();
  g_resolver.AddMean();
  g_resolver.AddMul();
  g_resolver.AddSoftmax();

  for (size_t kb : PERSIST_KB) {
    uint8_t* persist = (uint8_t*)heap_caps_aligned_alloc(16, kb * 1024, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!persist) continue;
    auto* alloc = tflite::MicroAllocator::Create(persist, kb * 1024, g_scratch, sizeof(g_scratch));
    auto* it = alloc ? new tflite::MicroInterpreter(model, g_resolver, alloc) : nullptr;
    if (it && it->AllocateTensors() == kTfLiteOk) {
      g_interpreter = it;
      g_persist_kb = kb;
      break;
    }
    delete it;
    heap_caps_free(persist);
  }
  if (!g_interpreter) {
    Serial.println("LOI: khong du RAM cho model.");
    return false;
  }
  g_input = g_interpreter->input(0);
  g_output = g_interpreter->output(0);
  return g_input->type == kTfLiteInt8 && g_input->dims->data[1] == IMG_SIZE;
}

bool model_ready() { return g_interpreter != nullptr; }
size_t model_arena_kb() { return SCRATCH_KB + g_persist_kb; }

// Cap tu heap (khong de mang tinh): RAM tinh da gan day vi vung tam cua model.
uint8_t* image_buffer() {
  static uint8_t* buf = (uint8_t*)malloc(IMG_BYTES);
  return buf;
}

// Anh xam (uint8, theo hang) -> xac suat PNEUMONIA. Doc scale / zero_point tu model.
float model_predict(const uint8_t* gray, uint32_t* us) {
  const float scale = g_input->params.scale;
  const int zp = g_input->params.zero_point;
  int8_t* dst = g_input->data.int8;
  for (size_t i = 0; i < IMG_BYTES; ++i) {
    const int q = constrain((int)lroundf(gray[i] / scale) + zp, -128, 127);   // model hien tai: pixel - 128
    dst[3 * i] = dst[3 * i + 1] = dst[3 * i + 2] = (int8_t)q;                   // xam -> R = G = B
  }
  const uint32_t t0 = micros();
  const bool ok = g_interpreter->Invoke() == kTfLiteOk;
  if (us) *us = micros() - t0;
  if (!ok) return -1.0f;
  // dau ra [NORMAL, PNEUMONIA]
  return (g_output->data.int8[1] - g_output->params.zero_point) * g_output->params.scale;
}

const char* label_of(float p) { return p >= THRESHOLD ? "PNEUMONIA" : "NORMAL"; }

const char* decide(float p) {
  if (p >= UNC_LO && p < UNC_HI) return "uncertain";
  return p >= THRESHOLD ? "pneu" : "normal";
}

bool model_self_test() {
  struct Case { const char* name; const uint8_t* img; float expected; };
  const Case cases[] = {{"NORMAL   ", TEST_NORMAL, TEST_NORMAL_EXPECTED_P},
                        {"PNEUMONIA", TEST_PNEUMONIA, TEST_PNEUMONIA_EXPECTED_P}};
  bool all_ok = true;
  for (const Case& c : cases) {
    uint32_t us = 0;
    const float p = model_predict(c.img, &us);
    const bool ok = p >= 0.0f && fabsf(p - c.expected) < 0.01f;
    all_ok &= ok;
    Serial.printf("Anh mau %s: p = %.4f (laptop %.4f) %lu ms -> %s\n", c.name, p, c.expected,
                  (unsigned long)(us / 1000), ok ? "DAT" : "KHONG DAT");
  }
  return all_ok;
}
