#pragma once

#include <cmath>

#ifdef __VITIS_HLS__
#include <hls_math.h>
#endif

namespace hls_nn {
template <typename DType, int N> class SigmoidImpl {
public:
  using dtype = DType;
  static constexpr int n = N;

  static void kernel(dtype output, const dtype input) {
#ifdef __VITIS_HLS__
    output = dtype(1.0f) / (dtype(1.0f) + hls::exp(-input));
#else
    output = dtype(1.0f) / (dtype(1.0f) + std::exp(-input));
#endif
  }
};

} // namespace hls_nn
