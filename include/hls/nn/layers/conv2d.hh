#pragma once
#include "../../opt_level.hh"

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace hls_nn {
template <typename DType, const int IN_CHANNELS, const int OUT_CHANNELS,
          const int KernelSize, const int N, OptLevel OPT_LEVEL = OPT_NONE>
class Conv2d {
public:
  using dtype = DType;
  static constexpr int in_channels = IN_CHANNELS;
  static constexpr int out_channels = OUT_CHANNELS;
  static constexpr int kernel_size = KernelSize;
  static constexpr int n = N;
  static constexpr OptLevel opt_level = OPT_LEVEL;

  Conv2d() = default;
  ~Conv2d() = default;

  static void forward(
      dtype output[out_channels][n - kernel_size + 1][n - kernel_size + 1],
      const dtype input[in_channels][n][n],
      const dtype weight[out_channels][in_channels][kernel_size][kernel_size],
      const dtype bias[out_channels]);

private:
};

template <typename DType, const int IN_CHANNELS, const int OUT_CHANNELS,
          const int KernelSize, const int N>
class Conv2d<DType, IN_CHANNELS, OUT_CHANNELS, KernelSize, N, OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int in_channels = IN_CHANNELS;
  static constexpr int out_channels = OUT_CHANNELS;
  static constexpr int kernel_size = KernelSize;
  static constexpr int n = N;
  static constexpr OptLevel opt_level = OPT_NONE;

  Conv2d() = default;
  ~Conv2d() = default;

  static void forward(
      dtype output[out_channels][n - kernel_size + 1][n - kernel_size + 1],
      const dtype input[in_channels][n][n],
      const dtype weight[out_channels][in_channels][kernel_size][kernel_size],
      const dtype bias[out_channels]) {}

private:
};

template <typename DType, const int IN_CHANNELS, const int OUT_CHANNELS,
          const int KernelSize, const int N>
class Conv2d<DType, IN_CHANNELS, OUT_CHANNELS, KernelSize, N, OPT_LATENCY> {
public:
  using dtype = DType;
  static constexpr int in_channels = IN_CHANNELS;
  static constexpr int out_channels = OUT_CHANNELS;
  static constexpr int kernel_size = KernelSize;
  static constexpr int n = N;
  static constexpr OptLevel opt_level = OPT_LATENCY;

  Conv2d() = default;
  ~Conv2d() = default;

  static void forward(
      dtype output[out_channels][n - kernel_size + 1][n - kernel_size + 1],
      const dtype input[in_channels][n][n],
      const dtype weight[out_channels][in_channels][kernel_size][kernel_size],
      const dtype bias[out_channels]) {}

private:
};

template <typename DType, const int IN_CHANNELS, const int OUT_CHANNELS,
          const int KernelSize, const int N>
class Conv2d<DType, IN_CHANNELS, OUT_CHANNELS, KernelSize, N, OPT_THROUGHPUT> {
public:
  using dtype = DType;
  static constexpr int in_channels = IN_CHANNELS;
  static constexpr int out_channels = OUT_CHANNELS;
  static constexpr int kernel_size = KernelSize;
  static constexpr int n = N;
  static constexpr OptLevel opt_level = OPT_THROUGHPUT;

  Conv2d() = default;
  ~Conv2d() = default;

  static void forward(
      dtype output[out_channels][n - kernel_size + 1][n - kernel_size + 1],
      const dtype input[in_channels][n][n],
      const dtype weight[out_channels][in_channels][kernel_size][kernel_size],
      const dtype bias[out_channels]) {}

private:
};

} // namespace hls_nn
