#pragma once

#include "../elementwise.hh"

#ifdef __VITIS_HLS__
#endif

namespace hls_nn {
template <typename DType, int N> class ReLUImpl {
  using dtype = DType;
  static constexpr int n = N;

  static dtype kernel(const dtype x) {
    return x > dtype(0.0f) ? x : dtype(0.0f);
  }
};

// FIXME: Weak against Config with different OPT_LEVEL
template <typename DType, int N, typename Config = void,
          OptLevel OPT_LEVEL = OPT_NONE>
using ReLU = Elementwise<DType, ReLUImpl<DType, N>, N, Config, OPT_LEVEL>;

} // namespace hls_nn
