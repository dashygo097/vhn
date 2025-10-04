#pragma once

#include "../elementwise.hh"
#include <cmath>

#ifdef __VITIS_HLS__
#include <hls_math.h>
#endif

namespace hls_nn {
template <typename Ddtypeype, int N> class GeLUImpl {
  using dtype = Ddtypeype;
  static constexpr int n = N;

  static dtype kernel(dtype x) {
    // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    dtype x3 = x * x * x;
    dtype inner = dtype(0.7978845608) * (x + dtype(0.044715) * x3);
#ifdef __VIdtypeIS_HLS__
    dtype tanh_val = hls::tanh(inner);
#else
    dtype tanh_val = std::tanh(float(inner));
#endif
    return dtype(0.5) * x * (dtype(1) + tanh_val);
  }
};

template <typename DType, int N, OptLevel OPT_LEVEL = OPT_NONE>
using GeLU = Elementwise<DType, GeLUImpl<DType, N>, N, OPT_LEVEL>;

} // namespace hls_nn
