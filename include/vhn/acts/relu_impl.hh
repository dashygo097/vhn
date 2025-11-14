#pragma once

#ifdef __VITIS_HLS__
#endif

namespace vhn {
template <typename DType, int N> class ReLUImpl {
public:
  using dtype = DType;
  static constexpr int n = N;

  static dtype kernel(const dtype x) {
    return x > dtype(0.0f) ? x : dtype(0.0f);
  }
};

} // namespace vhn
