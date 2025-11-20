#pragma once

#include "../opt_level.hh"

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace vhn {

template <typename DType, typename HParams, typename Config = void,
          OptLevel OPT_LEVEL = OPT_NONE>
class Conv2d;

template <int IN_CHANNELS, int OUT_CHANNELS, int KERNEL_SIZE, int PADDING,
          int WIDTH, int HEIGHT>
struct Conv2dHParams {
  static constexpr int in_channels = IN_CHANNELS;
  static constexpr int out_channels = OUT_CHANNELS;
  static constexpr int kernel_size = KERNEL_SIZE;
  static constexpr int padding = PADDING;
  static constexpr int width = WIDTH;
  static constexpr int height = HEIGHT;
  static constexpr int out_width = WIDTH - KERNEL_SIZE + 1 + 2 * PADDING;
  static constexpr int out_height = HEIGHT - KERNEL_SIZE + 1 + 2 * PADDING;
};

// ============================================================================
// Non-optimized version (OPT_NONE)
// ============================================================================
template <typename DType, typename HParams>
class Conv2d<DType, HParams, void, OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int in_channels = HParams::in_channels;
  static constexpr int out_channels = HParams::out_channels;
  static constexpr int kernel_size = HParams::kernel_size;
  static constexpr int padding = HParams::padding;
  static constexpr int width = HParams::width;
  static constexpr int height = HParams::height;
  static constexpr int out_width = HParams::out_width;
  static constexpr int out_height = HParams::out_height;
  static constexpr OptLevel opt_level = OPT_NONE;

  using Weight_t = dtype[out_channels][in_channels][kernel_size][kernel_size];
  using Bias_t = dtype[out_channels];
  using Input_t = dtype[in_channels][width][height];
  using Output_t = dtype[out_channels][out_width][out_height];

  Conv2d() = default;
  ~Conv2d() = default;

  static void forward(Output_t output, const Input_t input,
                      const Weight_t weight, const Bias_t bias) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    forward_3d_impl(output, input, weight, bias);
  }

  static void forward(dtype output[][out_channels][out_width][out_height],
                      const dtype input[][in_channels][width][height],
                      const int batch_size, const Weight_t weight,
                      const Bias_t bias) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 32
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_3d_impl(output[b], input[b], weight, bias);
    }
  }

#ifdef __VITIS_HLS__
  static void forward(hls::stream<dtype> &output_stream,
                      hls::stream<dtype> &input_stream, const Weight_t weight,
                      const Bias_t bias, const int batch_size) {
#pragma HLS INLINE off
    forward_stream_impl(output_stream, input_stream, weight, bias, batch_size);
  }
#endif

private:
  static void forward_3d_impl(dtype output[out_channels][out_width][out_height],
                              const dtype input[in_channels][width][height],
                              const Weight_t weight, const Bias_t bias) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = input type = cyclic factor = 4 dim = 1
#pragma HLS ARRAY_PARTITION variable = weight type = cyclic factor = 4 dim = 2
#pragma HLS ARRAY_PARTITION variable = bias type = cyclic factor = 4
#endif

  OUT_CHANNEL_LOOP:
    for (int oc = 0; oc < out_channels; oc++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#endif
    OUT_POS_Y_LOOP:
      for (int pos_y = 0; pos_y < out_width; pos_y++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#endif
      OUT_POS_X_LOOP:
        for (int pos_x = 0; pos_x < out_height; pos_x++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#endif
          dtype acc = dtype(0.0f);
        IN_CHANNEL_LOOP:
          for (int ic = 0; ic < in_channels; ic++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#endif
          KERNEL_Y_LOOP:
            for (int ky = 0; ky < kernel_size; ky++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 7
#endif
            KERNEL_X_LOOP:
              for (int kx = 0; kx < kernel_size; kx++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 7
#endif
                int in_pos_y = pos_y + ky - padding;
                int in_pos_x = pos_x + kx - padding;
                if (in_pos_y >= 0 && in_pos_y < width && in_pos_x >= 0 &&
                    in_pos_x < height) {
                  acc += input[ic][in_pos_y][in_pos_x] * weight[oc][ic][ky][kx];
                }
              }
            }
          }
          output[oc][pos_y][pos_x] = acc + bias[oc];
        }
      }
    }
  }

#ifdef __VITIS_HLS__
  static void forward_stream_impl(hls::stream<dtype> &output_stream,
                                  hls::stream<dtype> &input_stream,
                                  const Weight_t weight, const Bias_t bias,
                                  const int batch_size) {
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 32
      dtype input_buffer[in_channels][width][height];
    READ_INPUT:
      for (int ic = 0; ic < in_channels; ic++) {
        for (int w = 0; w < width; w++) {
          for (int h = 0; h < height; h++) {
            input_buffer[ic][w][h] = input_stream.read();
          }
        }
      }

      dtype output_buffer[out_channels][out_width][out_height];
      forward_3d_impl(output_buffer, input_buffer, weight, bias);

    WRITE_OUTPUT:
      for (int oc = 0; oc < out_channels; oc++) {
        for (int w = 0; w < out_width; w++) {
          for (int h = 0; h < out_height; h++) {
            output_stream.write(output_buffer[oc][w][h]);
          }
        }
      }
    }
  }
#endif
};

template <bool DATAFLOW_ENABLED, int PIPELINE_II, int UNROLL_FACTOR,
          int PARTITION_FACTOR, int KERNEL_UNROLL, int IC_UNROLL>
struct Conv2dConfig {
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
class Conv2d<DType, HParams, Config, OPT_ENABLED> {
public:
  using dtype = DType;
  static constexpr int in_channels = HParams::in_channels;
  static constexpr int out_channels = HParams::out_channels;
  static constexpr int kernel_size = HParams::kernel_size;
  static constexpr int padding = HParams::padding;
  static constexpr int width = HParams::width;
  static constexpr int height = HParams::height;
  static constexpr int out_width = HParams::out_width;
  static constexpr int out_height = HParams::out_height;
  static constexpr OptLevel opt_level = OPT_ENABLED;

  static constexpr int dataflow_enabled = Config::dataflow_enabled;
  static constexpr int pipeline_ii = Config::pipeline_ii;
  static constexpr int unroll_factor = Config::unroll_factor;
  static constexpr int partition_factor = Config::partition_factor;
  static constexpr int kernel_unroll = Config::kernel_unroll;
  static constexpr int ic_unroll = Config::ic_unroll;

  using Weight_t = dtype[out_channels][in_channels][kernel_size][kernel_size];
  using Bias_t = dtype[out_channels];
  using Input_t = dtype[in_channels][width][height];
  using Output_t = dtype[out_channels][out_width][out_height];

  Conv2d() = default;
  ~Conv2d() = default;

  static void forward(Output_t output, const Input_t input,
                      const Weight_t weight, const Bias_t bias) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    forward_3d_impl(output, input, weight, bias);
  }

  static void forward(dtype output[][out_channels][out_width][out_height],
                      const dtype input[][in_channels][width][height],
                      const int batch_size, const Weight_t weight,
                      const Bias_t bias) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
    if constexpr (dataflow_enabled) {
#pragma HLS DATAFLOW
    }
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
      forward_3d_impl(output[b], input[b], weight, bias);
    }
  }

#ifdef __VITIS_HLS__
  static void forward(hls::stream<dtype> &output_stream,
                      hls::stream<dtype> &input_stream, const Weight_t weight,
                      const Bias_t bias, const int batch_size) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
    forward_stream_impl(output_stream, input_stream, weight, bias, batch_size);
  }
#endif

private:
  static void forward_3d_impl(dtype output[out_channels][out_width][out_height],
                              const dtype input[in_channels][width][height],
                              const Weight_t weight, const Bias_t bias) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off

    constexpr bool should_partition = (partition_factor > 1) &&
                                      (in_channels <= 512) &&
                                      (out_width <= 512) && (out_height <= 512);
    if constexpr (should_partition) {
#pragma HLS ARRAY_PARTITION variable = input type = cyclic factor =            \
    partition_factor dim = 1
#pragma HLS ARRAY_PARTITION variable = weight type = cyclic factor =           \
    partition_factor dim = 2
#pragma HLS ARRAY_PARTITION variable = output type = cyclic factor =           \
    partition_factor dim = 1
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
    OUT_POS_Y_LOOP:
      for (int pos_y = 0; pos_y < out_width; pos_y++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#endif
      OUT_POS_X_LOOP:
        for (int pos_x = 0; pos_x < out_height; pos_x++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#endif
          dtype acc = dtype(0.0f);
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
          KERNEL_Y_LOOP:
            for (int ky = 0; ky < kernel_size; ky++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 7
              if constexpr (kernel_unroll > 1) {
#pragma HLS UNROLL factor = kernel_unroll
              } else {
#pragma HLS UNROLL
              }
#endif
            KERNEL_X_LOOP:
              for (int kx = 0; kx < kernel_size; kx++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 7
#pragma HLS UNROLL
#pragma HLS BIND_OP variable = acc op = mul impl = dsp
#endif
                int in_pos_y = pos_y + ky - padding;
                int in_pos_x = pos_x + kx - padding;
                if (in_pos_y >= 0 && in_pos_y < width && in_pos_x >= 0 &&
                    in_pos_x < height) {
                  acc += input[ic][in_pos_y][in_pos_x] * weight[oc][ic][ky][kx];
                }
              }
            }
          }
          output[oc][pos_y][pos_x] = acc + bias[oc];
        }
      }
    }
  }

#ifdef __VITIS_HLS__
  static void forward_stream_impl(hls::stream<dtype> &output_stream,
                                  hls::stream<dtype> &input_stream,
                                  const Weight_t weight, const Bias_t bias,
                                  const int batch_size) {
#pragma HLS INLINE off

    constexpr bool should_partition = (partition_factor > 1) &&
                                      (in_channels <= 512) &&
                                      (out_width <= 512) && (out_height <= 512);

  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 32
      dtype input_buffer[in_channels][width][height];
      if constexpr (should_partition) {
#pragma HLS ARRAY_PARTITION variable = input_buffer type = cyclic factor =     \
    partition_factor dim = 1
      }

    READ_INPUT:
      for (int ic = 0; ic < in_channels; ic++) {
        for (int w = 0; w < width; w++) {
          for (int h = 0; h < height; h++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
            input_buffer[ic][w][h] = input_stream.read();
          }
        }
      }

      dtype output_buffer[out_channels][out_width][out_height];
      if constexpr (should_partition) {
#pragma HLS ARRAY_PARTITION variable = output_buffer type = cyclic factor =    \
    partition_factor dim = 1
      }

      forward_3d_impl(output_buffer, input_buffer, weight, bias);

    WRITE_OUTPUT:
      for (int oc = 0; oc < out_channels; oc++) {
        for (int w = 0; w < out_width; w++) {
          for (int h = 0; h < out_height; h++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#pragma HLS PIPELINE II = pipeline_ii
            output_stream.write(output_buffer[oc][w][h]);
          }
        }
      }
    }
  }
#endif
};

} // namespace vhn
