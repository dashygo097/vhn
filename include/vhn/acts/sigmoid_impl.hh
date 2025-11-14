#pragma once

#include "../base.hh"
#include <cmath>

#ifdef __VITIS_HLS__
#include <hls_math.h>
#endif

namespace vhn {
template <typename DType, int N> class SigmoidImpl {
public:
  using dtype = DType;
  static constexpr int n = N;

#ifndef __VITIS_HLS__
  static std::string type() { return "Sigmoid"; }
  static json hparams() {
    json j;

    j["n"] = n;

    return j;
  }
#endif

  static void kernel(dtype output, const dtype input) {
#ifdef __VITIS_HLS__
    output = dtype(1.0f) / (dtype(1.0f) + hls::exp(-input));
#else
    output = dtype(1.0f) / (dtype(1.0f) + std::exp(-input));
#endif
  }
};

} // namespace vhn
