// Chay model nhan dien viem phoi (TensorFlow Lite Micro, int8) tren ESP32.
#pragma once
#include <Arduino.h>
#include "model_data.h"   // IMG_SIZE, THRESHOLD, UNC_LO, UNC_HI, g_model (sinh boi scripts/5_export_firmware.py)

constexpr size_t IMG_BYTES = IMG_SIZE * IMG_SIZE;   // 1 anh xam 64x64 = 4096 byte

bool model_init();                                  // goi 1 lan trong setup(); true = san sang
bool model_ready();
float model_predict(const uint8_t* gray, uint32_t* us = nullptr);   // xac suat PNEUMONIA 0..1, loi -> -1
size_t model_arena_kb();
uint8_t* image_buffer();                            // 1 bo dem anh 4096 byte dung chung cho USB va WiFi

const char* label_of(float p);   // "NORMAL" / "PNEUMONIA" (theo THRESHOLD)
const char* decide(float p);     // "normal" / "uncertain" / "pneu" (3 muc, cho trang web)
bool model_self_test();          // chay 2 anh mau, so voi laptop
