#pragma once

#include "../../opt_level.hh"

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace hls_nn {
template <const int UNROLL_FACTOR, const int PARTITION_FACTOR,
          const int PIPELINE_II>
struct Conv1dHLSConfig {
  static constexpr int _unroll_factor = UNROLL_FACTOR;
  static constexpr int _partition_factor = PARTITION_FACTOR;
  static constexpr int _pipeline_ii = PIPELINE_II;
};

template <typename DType, const int IN_CHANNELS, const int OUT_CHANNELS,
          const int KERNEL_SIZE, const int PADDING, const int N,
          typename Config, OptLevel OPT_LEVEL = OPT_NONE>
class Conv1d {
public:
  using dtype = DType;
  static constexpr int in_channels = IN_CHANNELS;
  static constexpr int out_channels = OUT_CHANNELS;
  static constexpr int kernel_size = KERNEL_SIZE;
  static constexpr int padding = PADDING;
  static constexpr int n = N;
  static constexpr OptLevel opt_level = OPT_LEVEL;

  Conv1d() = default;
  ~Conv1d() = default;

  static void
  forward(dtype output[out_channels][n - kernel_size + 1 + 2 * padding],
          const dtype input[in_channels][n],
          const dtype weight[out_channels][in_channels][kernel_size],
          const dtype bias[out_channels]);

private:
};

template <typename DType, const int IN_CHANNELS, const int OUT_CHANNELS,
          const int KERNEL_SIZE, const int PADDING, const int N>
class Conv1d<DType, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, PADDING, N, void,
             OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int in_channels = IN_CHANNELS;
  static constexpr int out_channels = OUT_CHANNELS;
  static constexpr int kernel_size = KERNEL_SIZE;
  static constexpr int padding = PADDING;
  static constexpr int n = N;
  static constexpr OptLevel opt_level = OPT_NONE;

  Conv1d() = default;
  ~Conv1d() = default;

  static void
  forward(dtype output[out_channels][n - kernel_size + 1 + 2 * padding],
          const dtype input[in_channels][n],
          const dtype weight[out_channels][in_channels][kernel_size],
          const dtype bias[out_channels]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
  OUT_CHANNEL_LOOP:
    for (int oc = 0; oc < out_channels; oc++) {
    OUT_POS_LOOP:
      for (int pos = 0; pos < n - kernel_size + 1 + 2 * padding; pos++) {
        dtype acc = dtype(0.0f);
      IN_CHANNEL_LOOP:
        for (int ic = 0; ic < in_channels; ic++) {
        KERNEL_LOOP:
          for (int k = 0; k < kernel_size; k++) {
            int in_pos = pos + k - padding;
            if (in_pos >= 0 && in_pos < n) {
              acc += input[ic][in_pos] * weight[oc][ic][k];
            }
          }
        }
        output[oc][pos] = acc + bias[oc];
      }
    }
  }

private:
};

template <typename DType, const int IN_CHANNELS, const int OUT_CHANNELS,
          const int KERNEL_SIZE, const int PADDING, const int N,
          typename Config>
class Conv1d<DType, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, PADDING, N, Config,
             OPT_ENABLED> {
public:
  using dtype = DType;
  static constexpr int in_channels = IN_CHANNELS;
  static constexpr int out_channels = OUT_CHANNELS;
  static constexpr int kernel_size = KERNEL_SIZE;
  static constexpr int padding = PADDING;
  static constexpr int n = N;
  static constexpr OptLevel opt_level = OPT_ENABLED;

  static constexpr int unroll_factor = Config::_unroll_factor;
  static constexpr int partition_factor = Config::_partition_factor;
  static constexpr int pipeline_ii = Config::_pipeline_ii;

  Conv1d() = default;
  ~Conv1d() = default;

  static void
  forward(dtype output[out_channels][n - kernel_size + 1 + 2 * padding],
          const dtype input[in_channels][n],
          const dtype weight[out_channels][in_channels][kernel_size],
          const dtype bias[out_channels]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = input block factor =                    \
    partition_factor dim = 1
#pragma HLS ARRAY_PARTITION variable = weight block factor =                   \
    partition_factor dim = 3
#endif

  OUT_CHANNEL_LOOP:
    for (int oc = 0; oc < out_channels; oc++) {
    OUT_POS_LOOP:
      for (int pos = 0; pos < n - kernel_size + 1 + 2 * padding; pos++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#endif
        dtype acc = dtype(0.0f);
      IN_CHANNEL_LOOP:
        for (int ic = 0; ic < in_channels; ic++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = unroll_factor
#endif
        KERNEL_LOOP:
          for (int k = 0; k < kernel_size; k++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL
#endif
            int in_pos = pos + k - padding;
            if (in_pos >= 0 && in_pos < n) {
              acc += input[ic][pos + k] * weight[oc][ic][k];
            }
          }
        }
        output[oc][pos] = acc + bias[oc];
      }
    }
  }
};

template <typename DType, const int IN_CHANNELS, const int OUT_CHANNELS,
          const int KERNEL_SIZE, const int PADDING, const int N,
          typename Config, const OptLevel OPT_LEVEL = OPT_NONE>
class Conv1dBatched {
public:
  using dtype = DType;
  static constexpr int in_channels = IN_CHANNELS;
  static constexpr int out_channels = OUT_CHANNELS;
  static constexpr int kernel_size = KERNEL_SIZE;
  static constexpr int padding = PADDING;
  static constexpr int n = N;
  static constexpr OptLevel opt_level = OPT_LEVEL;

  Conv1dBatched() = default;
  ~Conv1dBatched() = default;

  static void
  forward(dtype output[][out_channels][n - kernel_size + 1 + 2 * padding],
          const dtype input[][in_channels][n],
          const dtype weight[out_channels][in_channels][kernel_size],
          const dtype bias[out_channels], int batch_size) {
#ifdef __VITIS_HLS__
#pragma HLS DATAFLOW
#endif

  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      Conv1d<DType, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, PADDING, N, Config,
             OPT_LEVEL>::forward(output[b], input[b], weight, bias);
    }
  }

private:
};

} // namespace hls_nn
