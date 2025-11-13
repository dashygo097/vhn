#pragma once
#include "../opt_level.hh"

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace vhn {

template <typename DType, const int IN_CHANNELS, const int OUT_CHANNELS,
          const int KERNEL_SIZE, const int PADDING, const int WIDTH,
          const int HEIGHT, typename Config = void,
          OptLevel OPT_LEVEL = OPT_NONE>
class Conv2d;

// ============================================================================
// Non-optimized version (OPT_NONE)
// ============================================================================
template <typename DType, const int IN_CHANNELS, const int OUT_CHANNELS,
          const int KERNEL_SIZE, const int PADDING, const int WIDTH,
          const int HEIGHT>
class Conv2d<DType, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, PADDING, WIDTH,
             HEIGHT, void, OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int in_channels = IN_CHANNELS;
  static constexpr int out_channels = OUT_CHANNELS;
  static constexpr int kernel_size = KERNEL_SIZE;
  static constexpr int padding = PADDING;
  static constexpr int width = WIDTH;
  static constexpr int height = HEIGHT;
  static constexpr int out_width = WIDTH - KERNEL_SIZE + 1 + 2 * PADDING;
  static constexpr int out_height = HEIGHT - KERNEL_SIZE + 1 + 2 * PADDING;
  static constexpr OptLevel opt_level = OPT_NONE;

  using Weight_t = dtype[OUT_CHANNELS][IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
  using Bias_t = dtype[OUT_CHANNELS];

  Conv2d() = default;
  ~Conv2d() = default;

  static void forward(dtype output[OUT_CHANNELS][out_width][out_height],
                      const dtype input[IN_CHANNELS][WIDTH][HEIGHT],
                      const Weight_t weight, const Bias_t bias) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    forward_3d_impl(output, input, weight, bias);
  }

  static void forward(dtype output[][OUT_CHANNELS][out_width][out_height],
                      const dtype input[][IN_CHANNELS][WIDTH][HEIGHT],
                      const int batch_size, const Weight_t weight,
                      const Bias_t bias) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_3d_impl(output[b], input[b], weight, bias);
    }
  }

private:
  static void forward_3d_impl(dtype output[OUT_CHANNELS][out_width][out_height],
                              const dtype input[IN_CHANNELS][WIDTH][HEIGHT],
                              const Weight_t weight, const Bias_t bias) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
  OUT_CHANNEL_LOOP:
    for (int oc = 0; oc < out_channels; oc++) {
    OUT_POS_Y_LOOP:
      for (int pos_y = 0; pos_y < out_width; pos_y++) {
      OUT_POS_X_LOOP:
        for (int pos_x = 0; pos_x < out_height; pos_x++) {
          dtype acc = dtype(0.0f);
        IN_CHANNEL_LOOP:
          for (int ic = 0; ic < in_channels; ic++) {
          KERNEL_Y_LOOP:
            for (int ky = 0; ky < kernel_size; ky++) {
            KERNEL_X_LOOP:
              for (int kx = 0; kx < kernel_size; kx++) {
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
};

// ============================================================================
// Optimized version (OPT_ENABLED)
// ============================================================================
template <typename DType, const int IN_CHANNELS, const int OUT_CHANNELS,
          const int KERNEL_SIZE, const int PADDING, const int WIDTH,
          const int HEIGHT, typename Config>
class Conv2d<DType, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, PADDING, WIDTH,
             HEIGHT, Config, OPT_ENABLED> {
public:
  using dtype = DType;
  static constexpr int in_channels = IN_CHANNELS;
  static constexpr int out_channels = OUT_CHANNELS;
  static constexpr int kernel_size = KERNEL_SIZE;
  static constexpr int padding = PADDING;
  static constexpr int width = WIDTH;
  static constexpr int height = HEIGHT;
  static constexpr int out_width = WIDTH - KERNEL_SIZE + 1 + 2 * PADDING;
  static constexpr int out_height = HEIGHT - KERNEL_SIZE + 1 + 2 * PADDING;
  static constexpr OptLevel opt_level = OPT_ENABLED;

  static constexpr int unroll_factor = Config::_unroll_factor;
  static constexpr int partition_factor = Config::_partition_factor;
  static constexpr int pipeline_ii = Config::_pipeline_ii;

  using Weight_t = dtype[OUT_CHANNELS][IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
  using Bias_t = dtype[OUT_CHANNELS];

  Conv2d() = default;
  ~Conv2d() = default;

  static void forward(dtype output[OUT_CHANNELS][out_width][out_height],
                      const dtype input[IN_CHANNELS][WIDTH][HEIGHT],
                      const Weight_t weight, const Bias_t bias) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    forward_3d_impl(output, input, weight, bias);
  }

  static void forward(dtype output[][OUT_CHANNELS][out_width][out_height],
                      const dtype input[][IN_CHANNELS][WIDTH][HEIGHT],
                      const int batch_size, const Weight_t weight,
                      const Bias_t bias) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_3d_impl(output[b], input[b], weight, bias);
    }
  }

private:
  static void forward_3d_impl(dtype output[OUT_CHANNELS][out_width][out_height],
                              const dtype input[IN_CHANNELS][WIDTH][HEIGHT],
                              const Weight_t weight, const Bias_t bias) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = input cyclic factor =                   \
    partition_factor dim = 1
#pragma HLS ARRAY_PARTITION variable = weight cyclic factor =                  \
    partition_factor dim = 2
#pragma HLS ARRAY_PARTITION variable = weight cyclic factor =                  \
    kernel_size dim = 4
#pragma HLS ARRAY_PARTITION variable = output cyclic factor =                  \
    partition_factor dim = 1
#pragma HLS ARRAY_PARTITION variable = bias cyclic factor = partition_factor
#endif
  OUT_CHANNEL_LOOP:
    for (int oc = 0; oc < out_channels; oc++) {
    OUT_POS_Y_LOOP:
      for (int pos_y = 0; pos_y < out_width; pos_y++) {
      OUT_POS_X_LOOP:
        for (int pos_x = 0; pos_x < out_height; pos_x++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#endif
          dtype acc = dtype(0.0f);
        IN_CHANNEL_LOOP:
          for (int ic = 0; ic < in_channels; ic++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = unroll_factor
#endif
          KERNEL_Y_LOOP:
            for (int ky = 0; ky < kernel_size; ky++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL
#endif
            KERNEL_X_LOOP:
              for (int kx = 0; kx < kernel_size; kx++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL
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
};

} // namespace vhn
