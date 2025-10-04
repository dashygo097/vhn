#pragma once

#include "../../opt_level.hh"

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace hls_nn {
template <typename DType, const int IN_CHANNELS, const int OUT_CHANNELS,
          const int KernelSize, const int N, OptLevel OPT_LEVEL = OPT_NONE>
class Conv1d {
public:
  using dtype = DType;
  static constexpr int in_channels = IN_CHANNELS;
  static constexpr int out_channels = OUT_CHANNELS;
  static constexpr int kernel_size = KernelSize;
  static constexpr int n = N;
  static constexpr OptLevel opt_level = OPT_LEVEL;

  Conv1d() = default;
  ~Conv1d() = default;

  static void
  forward(dtype output[out_channels][n - kernel_size + 1],
          const dtype input[in_channels][n],
          const dtype weight[out_channels][in_channels][kernel_size],
          const dtype bias[out_channels]);

private:
};

template <typename DType, const int IN_CHANNELS, const int OUT_CHANNELS,
          const int KernelSize, const int N>
class Conv1d<DType, IN_CHANNELS, OUT_CHANNELS, KernelSize, N, OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int in_channels = IN_CHANNELS;
  static constexpr int out_channels = OUT_CHANNELS;
  static constexpr int kernel_size = KernelSize;
  static constexpr int n = N;
  static constexpr OptLevel opt_level = OPT_NONE;

  Conv1d() = default;
  ~Conv1d() = default;

  static void
  forward(dtype output[out_channels][n - kernel_size + 1],
          const dtype input[in_channels][n],
          const dtype weight[out_channels][in_channels][kernel_size],
          const dtype bias[out_channels]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif

  OUT_CHANNEL_LOOP:
    for (int oc = 0; oc < out_channels; oc++) {
    OUT_POS_LOOP:
      for (int pos = 0; pos < n - kernel_size + 1; pos++) {
        dtype acc = 0;
      IN_CHANNEL_LOOP:
        for (int ic = 0; ic < in_channels; ic++) {
        KERNEL_LOOP:
          for (int k = 0; k < kernel_size; k++) {
            acc += input[ic][pos + k] * weight[oc][ic][k];
          }
        }
        output[oc][pos] = acc + bias[oc];
      }
    }
  }

private:
};

template <typename DType, const int IN_CHANNELS, const int OUT_CHANNELS,
          const int KernelSize, const int N>
class Conv1d<DType, IN_CHANNELS, OUT_CHANNELS, KernelSize, N, OPT_LATENCY> {
public:
  using dtype = DType;
  static constexpr int in_channels = IN_CHANNELS;
  static constexpr int out_channels = OUT_CHANNELS;
  static constexpr int kernel_size = KernelSize;
  static constexpr int n = N;
  static constexpr OptLevel opt_level = OPT_LATENCY;

  Conv1d() = default;
  ~Conv1d() = default;

  static void
  forward(dtype output[out_channels][n - kernel_size + 1],
          const dtype input[in_channels][n],
          const dtype weight[out_channels][in_channels][kernel_size],
          const dtype bias[out_channels]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = input complete dim = 1
#pragma HLS ARRAY_PARTITION variable = weight block factor = 16 dim = 3
#endif

  OUT_CHANNEL_LOOP:
    for (int oc = 0; oc < out_channels; oc++) {
    OUT_POS_LOOP:
      for (int pos = 0; pos < n - kernel_size + 1; pos++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = 1 rewind
#endif
        dtype acc = 0;
      IN_CHANNEL_LOOP:
        for (int ic = 0; ic < in_channels; ic++) {
        KERNEL_LOOP:
          for (int k = 0; k < kernel_size; k++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL
#endif
            acc += input[ic][pos + k] * weight[oc][ic][k];
          }
        }
        output[oc][pos] = acc + bias[oc];
      }
    }
  }
};

template <typename DType, const int IN_CHANNELS, const int OUT_CHANNELS,
          const int KernelSize, const int N>
class Conv1d<DType, IN_CHANNELS, OUT_CHANNELS, KernelSize, N, OPT_THROUGHPUT> {
public:
  using dtype = DType;
  static constexpr int in_channels = IN_CHANNELS;
  static constexpr int out_channels = OUT_CHANNELS;
  static constexpr int kernel_size = KernelSize;
  static constexpr int n = N;
  static constexpr OptLevel opt_level = OPT_THROUGHPUT;

  Conv1d() = default;
  ~Conv1d() = default;

  static void
  forward(dtype output[out_channels][n - kernel_size + 1],
          const dtype input[in_channels][n],
          const dtype weight[out_channels][in_channels][kernel_size],
          const dtype bias[out_channels]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = input cyclic factor = 8 dim = 1
#pragma HLS ARRAY_PARTITION variable = weight cyclic factor = 8 dim = 3
#endif

  OUT_CHANNEL_LOOP:
    for (int oc = 0; oc < out_channels; oc++) {
    OUT_POS_LOOP:
      for (int pos = 0; pos < n - kernel_size + 1; pos++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = 1
#endif
        dtype acc = 0;
      IN_CHANNEL_LOOP:
        for (int ic = 0; ic < in_channels; ic++) {
        KERNEL_LOOP:
          for (int k = 0; k < kernel_size; k++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = 4
#endif
            acc += input[ic][pos + k] * weight[oc][ic][k];
          }
        }
        output[oc][pos] = acc + bias[oc];
      }
    }
  }

private:
};

} // namespace hls_nn
