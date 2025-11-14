#pragma once

#include <cmath>

#ifdef __VITIS_HLS__
#include <hls_math.h>
#endif

namespace vhn {
template <typename Dtype, int N> class GeLUImpl {
public:
  using dtype = Dtype;
  static constexpr int n = N;

  static dtype kernel(dtype x) {
    // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    dtype x3 = x * x * x;
    dtype inner = dtype(0.7978845608f) * (x + dtype(0.044715f) * x3);
#ifdef __VITIS_HLS__
    dtype tanh_val = hls::tanh(inner);
#else
    dtype tanh_val = std::tanh(float(inner));
#endif
    return dtype(0.5f) * x * (dtype(1.0f) + tanh_val);
  }
};

} // namespace vhn
