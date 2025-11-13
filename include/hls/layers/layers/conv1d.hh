#pragma once

#include "../../opt_level.hh"

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace hls_nn {

template <typename DType, const int IN_CHANNELS, const int OUT_CHANNELS,
          const int KERNEL_SIZE, const int PADDING, const int N,
          typename Config = void, OptLevel OPT_LEVEL = OPT_NONE>
class Conv1d;

// ============================================================================
// Non-optimized version (OPT_NONE)
// ============================================================================
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
  static constexpr int out_length = N - KERNEL_SIZE + 1 + 2 * PADDING;
  static constexpr OptLevel opt_level = OPT_NONE;

  using Weight_t = dtype[OUT_CHANNELS][IN_CHANNELS][KERNEL_SIZE];
  using Bias_t = dtype[OUT_CHANNELS];
  using Input_t = dtype[IN_CHANNELS][N];
  using Output_t = dtype[OUT_CHANNELS][out_length];

  Conv1d() = default;
  ~Conv1d() = default;

  static void forward(Output_t output, const Input_t input,
                      const Weight_t weight, const Bias_t bias) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    forward_2d_impl(output, input, weight, bias);
  }

  static void forward(dtype output[][OUT_CHANNELS][out_length],
                      const dtype input[][IN_CHANNELS][N],
                      const Weight_t weight, const Bias_t bias,
                      const int batch_size) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_2d_impl(output[b], input[b], weight, bias);
    }
  }

  static void forward(dtype *output, const dtype *input, const Weight_t weight,
                      const Bias_t bias, const int batch_size) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
    constexpr int input_size = IN_CHANNELS * N;
    constexpr int output_size = OUT_CHANNELS * out_length;

  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      auto out_ptr = reinterpret_cast<Output_t *>(&output[b * output_size]);
      auto in_ptr = reinterpret_cast<const Input_t *>(&input[b * input_size]);
      forward_2d_impl(*out_ptr, *in_ptr, weight, bias);
    }
  }

private:
  static void forward_2d_impl(dtype output[OUT_CHANNELS][out_length],
                              const dtype input[IN_CHANNELS][N],
                              const Weight_t weight, const Bias_t bias) {
  OUT_CHANNEL_LOOP:
    for (int oc = 0; oc < OUT_CHANNELS; oc++) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    OUT_POS_LOOP:
      for (int pos = 0; pos < out_length; pos++) {
        dtype acc = dtype(0);
      IN_CHANNEL_LOOP:
        for (int ic = 0; ic < IN_CHANNELS; ic++) {
        KERNEL_LOOP:
          for (int k = 0; k < KERNEL_SIZE; k++) {
            int in_pos = pos + k - PADDING;
            if (in_pos >= 0 && in_pos < N) {
              acc += input[ic][in_pos] * weight[oc][ic][k];
            }
          }
        }
        output[oc][pos] = acc + bias[oc];
      }
    }
  }
};

// ============================================================================
// Optimized version (OPT_ENABLED)
// ============================================================================
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
  static constexpr int out_length = N - KERNEL_SIZE + 1 + 2 * PADDING;
  static constexpr OptLevel opt_level = OPT_ENABLED;

  static constexpr int unroll_factor = Config::_unroll_factor;
  static constexpr int partition_factor = Config::_partition_factor;
  static constexpr int pipeline_ii = Config::_pipeline_ii;

  using Weight_t = dtype[OUT_CHANNELS][IN_CHANNELS][KERNEL_SIZE];
  using Bias_t = dtype[OUT_CHANNELS];
  using Input_t = dtype[IN_CHANNELS][N];
  using Output_t = dtype[OUT_CHANNELS][out_length];

  Conv1d() = default;
  ~Conv1d() = default;

  static void forward(Output_t output, const Input_t input,
                      const Weight_t weight, const Bias_t bias) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    forward_2d_impl(output, input, weight, bias);
  }

  static void forward(dtype output[][OUT_CHANNELS][out_length],
                      const dtype input[][IN_CHANNELS][N],
                      const Weight_t weight, const Bias_t bias,
                      const int batch_size) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_2d_impl(output[b], input[b], weight, bias);
    }
  }

  static void forward(dtype *output, const dtype *input, const Weight_t weight,
                      const Bias_t bias, const int batch_size) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
    constexpr int input_size = IN_CHANNELS * N;
    constexpr int output_size = OUT_CHANNELS * out_length;

  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      auto out_ptr = reinterpret_cast<Output_t *>(&output[b * output_size]);
      auto in_ptr = reinterpret_cast<const Input_t *>(&input[b * input_size]);
      forward_2d_impl(*out_ptr, *in_ptr, weight, bias);
    }
  }

private:
  static void forward_2d_impl(dtype output[OUT_CHANNELS][out_length],
                              const dtype input[IN_CHANNELS][N],
                              const Weight_t weight, const Bias_t bias) {
#ifdef __VITIS_HLS__
#pragma HLS ARRAY_PARTITION variable = input cyclic factor =                   \
    partition_factor dim = 1
#pragma HLS ARRAY_PARTITION variable = weight cyclic factor =                  \
    partition_factor dim = 3
#pragma HLS ARRAY_PARTITION variable = bias cyclic factor = partition_factor
#endif

  OUT_CHANNEL_LOOP:
    for (int oc = 0; oc < OUT_CHANNELS; oc++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = unroll_factor
#endif
    OUT_POS_LOOP:
      for (int pos = 0; pos < out_length; pos++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#endif
        dtype acc = dtype(0);
      IN_CHANNEL_LOOP:
        for (int ic = 0; ic < IN_CHANNELS; ic++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = unroll_factor
#endif
        KERNEL_LOOP:
          for (int k = 0; k < KERNEL_SIZE; k++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL
#endif
            int in_pos = pos + k - PADDING;
            if (in_pos >= 0 && in_pos < N) {
              acc += input[ic][in_pos] * weight[oc][ic][k];
            }
          }
        }
        output[oc][pos] = acc + bias[oc];
      }
    }
  }
};

} // namespace hls_nn
