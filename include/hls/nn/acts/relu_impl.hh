#pragma once

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

} // namespace hls_nn
