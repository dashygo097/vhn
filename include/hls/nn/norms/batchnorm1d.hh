#pragma once

#include "../../opt_level.hh"

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace hls_nn {
template <typename DType, int N, OptLevel OPT_LEVEL = OPT_NONE>
class BatchNorm1d {
public:
  using dtype = DType;
  static constexpr int n = N;
  static constexpr OptLevel opt_level = OPT_LEVEL;

  BatchNorm1d() = default;
  ~BatchNorm1d() = default;

  static void forward(dtype output[n], const dtype input[N],
                      const dtype weight[N], const dtype bias[N],
                      const dtype running_mean[N], const dtype running_var[N],
                      const dtype eps = dtype(1e-5));

private:
};

template <typename DType, int N> class BatchNorm1d<DType, N, OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int n = N;
  static constexpr OptLevel opt_level = OPT_NONE;

  BatchNorm1d() = default;
  ~BatchNorm1d() = default;

  static void forward(dtype output[n], const dtype input[N],
                      const dtype weight[N], const dtype bias[N],
                      const dtype running_mean[N], const dtype running_var[N],
                      const dtype eps = dtype(1e-5)) {}

private:
};

template <typename DType, int N> class BatchNorm1d<DType, N, OPT_LATENCY> {
public:
  using dtype = DType;
  static constexpr int n = N;
  static constexpr OptLevel opt_level = OPT_LATENCY;

  BatchNorm1d() = default;
  ~BatchNorm1d() = default;

  static void forward(dtype output[n], const dtype input[N],
                      const dtype weight[N], const dtype bias[N],
                      const dtype running_mean[N], const dtype running_var[N],
                      const dtype eps = dtype(1e-5)) {}

private:
};

template <typename DType, int N> class BatchNorm1d<DType, N, OPT_THROUGHPUT> {
public:
  using dtype = DType;
  static constexpr int n = N;
  static constexpr OptLevel opt_level = OPT_THROUGHPUT;
  BatchNorm1d() = default;
  ~BatchNorm1d() = default;
  static void forward(dtype output[n], const dtype input[N],
                      const dtype weight[N], const dtype bias[N],
                      const dtype running_mean[N], const dtype running_var[N],
                      const dtype eps = dtype(1e-5)) {}

private:
};

} // namespace hls_nn
