#pragma once

#include "../opt_level.hh"

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace vhn {

template <typename DType, typename HParams, typename Config = void,
          OptLevel OPT_LEVEL = OPT_NONE>
class Conv1d;

template <int IN_CHANNELS, int OUT_CHANNELS, int KERNEL_SIZE, int PADDING,
          int N>
struct Conv1dHParams {
  static constexpr int in_channels = IN_CHANNELS;
  static constexpr int out_channels = OUT_CHANNELS;
  static constexpr int kernel_size = KERNEL_SIZE;
  static constexpr int padding = PADDING;
  static constexpr int n = N;
  static constexpr int out_length = N - KERNEL_SIZE + 1 + 2 * PADDING;
};

// ============================================================================
// Non-optimized version (OPT_NONE)
// ============================================================================
template <typename DType, typename HParams>
class Conv1d<DType, HParams, void, OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int in_channels = HParams::in_channels;
  static constexpr int out_channels = HParams::out_channels;
  static constexpr int kernel_size = HParams::kernel_size;
  static constexpr int padding = HParams::padding;
  static constexpr int n = HParams::n;
  static constexpr int out_length = HParams::out_length;
  static constexpr OptLevel opt_level = OPT_NONE;

  using Weight_t = dtype[out_channels][in_channels][kernel_size];
  using Bias_t = dtype[out_channels];
  using Input_t = dtype[in_channels][n];
  using Output_t = dtype[out_channels][out_length];

  Conv1d() = default;
  ~Conv1d() = default;

  static void conv1d(Output_t output, const Input_t input,
                     const Weight_t weight, const Bias_t bias) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    conv1d_2d_impl(output, input, weight, bias);
  }

  static void conv1d(dtype output[][out_channels][out_length],
                     const dtype input[][in_channels][n], const Weight_t weight,
                     const Bias_t bias, const int batch_size) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 32
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      conv1d_2d_impl(output[b], input[b], weight, bias);
    }
  }

  static void conv1d(dtype *output, const dtype *input, const Weight_t weight,
                     const Bias_t bias, const int batch_size) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 32
#endif
    constexpr int input_size = in_channels * n;
    constexpr int output_size = out_channels * out_length;

  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      auto out_ptr = reinterpret_cast<Output_t *>(&output[b * output_size]);
      auto in_ptr = reinterpret_cast<const Input_t *>(&input[b * input_size]);
      conv1d_2d_impl(*out_ptr, *in_ptr, weight, bias);
    }
  }

#ifdef __VITIS_HLS__
  static void conv1d(hls::stream<dtype> &output_stream,
                     hls::stream<dtype> &input_stream, const Weight_t weight,
                     const Bias_t bias, const int batch_size) {
#pragma HLS INLINE off
    conv1d_stream_impl(output_stream, input_stream, weight, bias, batch_size);
  }
#endif

private:
  static void conv1d_2d_impl(dtype output[out_channels][out_length],
                             const dtype input[in_channels][n],
                             const Weight_t weight, const Bias_t bias) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif

  OUT_CHANNEL_LOOP:
    for (int oc = 0; oc < out_channels; oc++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#endif
    OUT_POS_LOOP:
      for (int pos = 0; pos < out_length; pos++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#endif
        dtype acc = dtype(0);
      IN_CHANNEL_LOOP:
        for (int ic = 0; ic < in_channels; ic++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#endif
        KERNEL_LOOP:
          for (int k = 0; k < kernel_size; k++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 7
#endif
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

#ifdef __VITIS_HLS__
  static void conv1d_stream_impl(hls::stream<dtype> &output_stream,
                                 hls::stream<dtype> &input_stream,
                                 const Weight_t weight, const Bias_t bias,
                                 const int batch_size) {
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 32
      dtype input_buffer[in_channels][n];
    READ_INPUT:
      for (int ic = 0; ic < in_channels; ic++) {
        for (int i = 0; i < n; i++) {
          input_buffer[ic][i] = input_stream.read();
        }
      }

      dtype output_buffer[out_channels][out_length];
      conv1d_2d_impl(output_buffer, input_buffer, weight, bias);

    WRITE_OUTPUT:
      for (int oc = 0; oc < out_channels; oc++) {
        for (int pos = 0; pos < out_length; pos++) {
          output_stream.write(output_buffer[oc][pos]);
        }
      }
    }
  }
#endif
};

template <bool DATAFLOW_ENABLED, int PIPELINE_II, int UNROLL_FACTOR,
          int PARTITION_FACTOR, int KERNEL_UNROLL, int IC_UNROLL>
struct Conv1dConfig {
  static constexpr bool dataflow_enabled = DATAFLOW_ENABLED;
  static constexpr int pipeline_ii = PIPELINE_II;
  static constexpr int unroll_factor = UNROLL_FACTOR;
  static constexpr int partition_factor = PARTITION_FACTOR;
  static constexpr int kernel_unroll = KERNEL_UNROLL;
  static constexpr int ic_unroll = IC_UNROLL;
};

// ============================================================================
// Optimized version (OPT_ENABLED)
// ============================================================================
template <typename DType, typename HParams, typename Config>
class Conv1d<DType, HParams, Config, OPT_ENABLED> {
public:
  using dtype = DType;
  static constexpr int in_channels = HParams::in_channels;
  static constexpr int out_channels = HParams::out_channels;
  static constexpr int kernel_size = HParams::kernel_size;
  static constexpr int padding = HParams::padding;
  static constexpr int n = HParams::n;
  static constexpr int out_length = HParams::out_length;
  static constexpr OptLevel opt_level = OPT_ENABLED;

  static constexpr int dataflow_enabled = Config::dataflow_enabled;
  static constexpr int pipeline_ii = Config::pipeline_ii;
  static constexpr int unroll_factor = Config::unroll_factor;
  static constexpr int partition_factor = Config::partition_factor;
  static constexpr int kernel_unroll = Config::kernel_unroll;
  static constexpr int ic_unroll = Config::ic_unroll;

  using Weight_t = dtype[out_channels][in_channels][kernel_size];
  using Bias_t = dtype[out_channels];
  using Input_t = dtype[in_channels][n];
  using Output_t = dtype[out_channels][out_length];

  Conv1d() = default;
  ~Conv1d() = default;

  static void conv1d(Output_t output, const Input_t input,
                     const Weight_t weight, const Bias_t bias) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    conv1d_2d_impl(output, input, weight, bias);
  }

  static void conv1d(dtype output[][out_channels][out_length],
                     const dtype input[][in_channels][n], const Weight_t weight,
                     const Bias_t bias, const int batch_size) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 32
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
      if constexpr (dataflow_enabled) {
#pragma HLS PIPELINE II = 1
      }
#endif
      conv1d_2d_impl(output[b], input[b], weight, bias);
    }
  }

  static void conv1d(dtype *output, const dtype *input, const Weight_t weight,
                     const Bias_t bias, const int batch_size) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 32
#endif
    constexpr int input_size = in_channels * n;
    constexpr int output_size = out_channels * out_length;

  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
      if constexpr (dataflow_enabled) {
#pragma HLS PIPELINE II = 1
      }
#endif
      auto out_ptr = reinterpret_cast<Output_t *>(&output[b * output_size]);
      auto in_ptr = reinterpret_cast<const Input_t *>(&input[b * input_size]);
      conv1d_2d_impl(*out_ptr, *in_ptr, weight, bias);
    }
  }

#ifdef __VITIS_HLS__
  static void conv1d(hls::stream<dtype> &output_stream,
                     hls::stream<dtype> &input_stream, const Weight_t weight,
                     const Bias_t bias, const int batch_size) {
#pragma HLS INLINE off
    conv1d_stream_impl(output_stream, input_stream, weight, bias, batch_size);
  }
#endif

private:
  static void conv1d_2d_impl(dtype output[out_channels][out_length],
                             const dtype input[in_channels][n],
                             const Weight_t weight, const Bias_t bias) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off

    constexpr bool should_partition =
        (partition_factor > 1) && (in_channels <= 512) && (out_length <= 1024);
    if constexpr (should_partition) {
#pragma HLS ARRAY_PARTITION variable = input type = cyclic factor =            \
    partition_factor dim = 1
#pragma HLS ARRAY_PARTITION variable = weight type = cyclic factor =           \
    partition_factor dim = 3
#pragma HLS ARRAY_PARTITION variable = bias type = cyclic factor =             \
    partition_factor
    } else {
#pragma HLS BIND_STORAGE variable = weight type = rom_2p impl = bram
#pragma HLS BIND_STORAGE variable = input type = ram_1p impl = bram
#pragma HLS BIND_STORAGE variable = bias type = rom_1p impl = bram
    }
#endif

  OUT_CHANNEL_LOOP:
    for (int oc = 0; oc < out_channels; oc++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
      if constexpr (unroll_factor > 1 && out_channels <= 256) {
#pragma HLS UNROLL factor = unroll_factor
      }
#endif
    OUT_POS_LOOP:
      for (int pos = 0; pos < out_length; pos++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1024
#endif
        dtype acc = dtype(0);
#ifdef __VITIS_HLS__
#pragma HLS BIND_OP variable = acc op = add impl = dsp
#endif

      IN_CHANNEL_LOOP:
        for (int ic = 0; ic < in_channels; ic++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
          if constexpr (ic_unroll > 1 && in_channels <= 256) {
#pragma HLS UNROLL factor = ic_unroll
          }
#endif
        KERNEL_LOOP:
          for (int k = 0; k < kernel_size; k++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 7
            if constexpr (kernel_unroll > 1) {
#pragma HLS UNROLL factor = kernel_unroll
            } else {
#pragma HLS UNROLL
            }
#pragma HLS BIND_OP variable = acc op = mul impl = dsp
#endif
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

#ifdef __VITIS_HLS__
  static void conv1d_stream_impl(hls::stream<dtype> &output_stream,
                                 hls::stream<dtype> &input_stream,
                                 const Weight_t weight, const Bias_t bias,
                                 const int batch_size) {
#pragma HLS INLINE off

    constexpr bool should_partition =
        (partition_factor > 1) && (in_channels <= 512) && (out_length <= 1024);

  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 32
      dtype input_buffer[in_channels][n];
      if constexpr (should_partition) {
#pragma HLS ARRAY_PARTITION variable = input_buffer type = cyclic factor =     \
    partition_factor dim = 1
      }

    READ_INPUT:
      for (int ic = 0; ic < in_channels; ic++) {
        for (int i = 0; i < n; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
          input_buffer[ic][i] = input_stream.read();
        }
      }

      dtype output_buffer[out_channels][out_length];
      if constexpr (should_partition) {
#pragma HLS ARRAY_PARTITION variable = output_buffer type = cyclic factor =    \
    partition_factor dim = 1
      }

      conv1d_2d_impl(output_buffer, input_buffer, weight, bias);

    WRITE_OUTPUT:
      for (int oc = 0; oc < out_channels; oc++) {
        for (int pos = 0; pos < out_length; pos++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1024
#pragma HLS PIPELINE II = pipeline_ii
          output_stream.write(output_buffer[oc][pos]);
        }
      }
    }
  }
#endif
};

} // namespace vhn
