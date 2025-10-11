#pragma once

#include "./elementwise.hh"

#ifdef __VITIS_HLS__
#endif

namespace hls_nn {
template <typename DType, int N> class AddImpl {
public:
  using dtype = DType;
  static constexpr int n = N;

  static dtype kernel(const dtype x, const dtype y) { return x + y; }
};

template <typename DType, int N> class SubImpl {
public:
  using dtype = DType;
  static constexpr int n = N;

  static dtype kernel(const dtype x, const dtype y) { return x - y; }
};

template <typename DType, int N> class MaxImpl {
public:
  using dtype = DType;
  static constexpr int n = N;

  static dtype kernel(const dtype x, const dtype y) { return (x > y) ? x : y; }
};

template <typename DType, int N> class MinImpl {
public:
  using dtype = DType;
  static constexpr int n = N;

  static dtype kernel(const dtype x, const dtype y) { return (x < y) ? x : y; }
};

} // namespace hls_nn
