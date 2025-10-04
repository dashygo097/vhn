#pragma once

#include "../elementwise.hh"

#ifdef __VITIS_HLS__
#endif

namespace hls_nn {
template <typename DType, int N> class ReLUImpl {
  using dtype = DType;
  static constexpr int n = N;

  static dtype kernel(const dtype x) { return x > 0 ? x : 0; }
};

template <typename DType, int N, OptLevel OPT_LEVEL = OPT_NONE>
using ReLU = Elementwise<DType, ReLUImpl<DType, N>, N, OPT_LEVEL>;

} // namespace hls_nn
