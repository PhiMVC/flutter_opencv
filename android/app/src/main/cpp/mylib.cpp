#include "mylib.h"
#include <cmath>

extern "C" Result process_gray(const uint8_t* data, int32_t width, int32_t height, int32_t rowStride) {
  Result r{0, 0};
  if (!data || width <= 0 || height <= 0 || rowStride <= 0) return r;

  // mean
  double sum = 0.0;
  int64_t n = (int64_t)width * height;
  for (int y = 0; y < height; y++) {
    const uint8_t* row = data + (int64_t)y * rowStride;
    for (int x = 0; x < width; x++) sum += row[x];
  }
  double mean = sum / (double)n;

  // std
  double var = 0.0;
  for (int y = 0; y < height; y++) {
    const uint8_t* row = data + (int64_t)y * rowStride;
    for (int x = 0; x < width; x++) {
      double d = row[x] - mean;
      var += d * d;
    }
  }
  var /= (double)n;

  r.a = (float)mean;
  r.b = (float)std::sqrt(var);
  return r;
}
