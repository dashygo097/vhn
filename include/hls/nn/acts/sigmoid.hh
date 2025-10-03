#pragma once

#include "../elementwise.hh"
#include <cmath>

#ifdef __VITIS_HLS__
#include <hls_math.h>
#endif

namespace hls_nn {
template <typename DType, int N> class SigmoidImpl {
  using dtype = DType;
  static constexpr int n = N;

  static void kernel(dtype output, const dtype input) {
#ifdef __VITIS_HLS__
    output = 1 / (1 + hls::exp(-input));
#else
    output = 1 / (1 + std::exp(-input));
#endif
  }
};

template <typename DType, int N>
using Sigmoid = Elementwise<SigmoidImpl<DType, N>, DType, N>;

} // namespace hls_nn
