#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Result {
  float a;
  float b;
} Result;

// Ví dụ: tính mean + std đơn giản trên buffer grayscale
Result process_gray(const uint8_t* data, int32_t width, int32_t height, int32_t rowStride);

#ifdef __cplusplus
}
#endif
