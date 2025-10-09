#pragma once

#include "../../opt_level.hh"
#include <cmath>

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace hls_nn {
template <typename DType, const int CHANNELS, const int WIDTH, const int HEIGHT,
          typename Config, OptLevel OPT_LEVEL = OPT_NONE>
class BatchNorm2d {
public:
  using dtype = DType;
  static constexpr int channels = CHANNELS;
  static constexpr int width = WIDTH;
  static constexpr int height = HEIGHT;
  static constexpr OptLevel opt_level = OPT_LEVEL;

  BatchNorm2d() = default;
  ~BatchNorm2d() = default;

  static void forward(dtype output[][CHANNELS][HEIGHT][WIDTH],
                      const dtype input[][CHANNELS][HEIGHT][WIDTH],
                      const dtype weight[CHANNELS], const dtype bias[CHANNELS],
                      const dtype running_mean[CHANNELS],
                      const dtype running_var[CHANNELS], const int batch_size,
                      const float epsilon = 1e-5);

private:
};

template <typename DType, const int CHANNELS, const int WIDTH, const int HEIGHT,
          typename Config>
class BatchNorm2d<DType, CHANNELS, WIDTH, HEIGHT, Config, OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int channels = CHANNELS;
  static constexpr int width = WIDTH;
  static constexpr int height = HEIGHT;
  static constexpr OptLevel opt_level = OPT_NONE;

  BatchNorm2d() = default;
  ~BatchNorm2d() = default;

  static void forward(dtype output[][CHANNELS][HEIGHT][WIDTH],
                      const dtype input[][CHANNELS][HEIGHT][WIDTH],
                      const dtype weight[CHANNELS], const dtype bias[CHANNELS],
                      const dtype running_mean[CHANNELS],
                      const dtype running_var[CHANNELS], const int batch_size,
                      const float epsilon = 1e-5) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif

  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif

    CHANNEL_LOOP:
      for (int c = 0; c < channels; c++) {
#ifdef __VITIS_HLS__
        dtype inv_std = hls::rsqrt(running_var[c] + dtype(epsilon));
#else
        dtype inv_std = dtype(1.0) / sqrt(running_var[c] + dtype(epsilon));
#endif

      HEIGHT_LOOP:
        for (int h = 0; h < height; h++) {
        WIDTH_LOOP:
          for (int w = 0; w < width; w++) {
            output[b][c][h][w] =
                weight[c] * (input[b][c][h][w] - running_mean[c]) * inv_std +
                bias[c];
          }
        }
      }
    }
  }

private:
};

template <typename DType, const int CHANNELS, const int H, const int W,
          typename Config>
class BatchNorm2d<DType, CHANNELS, H, W, Config, OPT_ENABLED> {
public:
  using dtype = DType;
  static constexpr int channels = CHANNELS;
  static constexpr int height = H;
  static constexpr int width = W;
  static constexpr OptLevel opt_level = OPT_ENABLED;

  static constexpr int unroll_factor = Config::_unroll_factor;
  static constexpr int partition_factor = Config::_partition_factor;
  static constexpr int pipeline_ii = Config::_pipeline_ii;

  BatchNorm2d() = default;
  ~BatchNorm2d() = default;

  static void forward(dtype output[][CHANNELS][H][W],
                      const dtype input[][CHANNELS][H][W],
                      const dtype weight[CHANNELS], const dtype bias[CHANNELS],
                      const dtype running_mean[CHANNELS],
                      const dtype running_var[CHANNELS], const int batch_size,
                      const float epsilon = 1e-5) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = input cyclic factor =                   \
    partition_factor dim = 2
#pragma HLS ARRAY_PARTITION variable = output cyclic factor =                  \
    partition_factor dim = 2
#pragma HLS ARRAY_PARTITION variable = weight cyclic factor = partition_factor
#pragma HLS ARRAY_PARTITION variable = bias cyclic factor = partition_factor
#pragma HLS ARRAY_PARTITION variable = running_mean cyclic factor =            \
    partition_factor
#pragma HLS ARRAY_PARTITION variable = running_var cyclic factor =             \
    partition_factor
#endif

  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif

    CHANNEL_LOOP:
      for (int c = 0; c < channels; c++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = unroll_factor
#endif
#ifdef __VITIS_HLS__
        dtype inv_std = hls::rsqrt(running_var[c] + dtype(epsilon));
#else
        dtype inv_std = dtype(1.0) / std::sqrt(running_var[c] + dtype(epsilon));
#endif

      HEIGHT_LOOP:
        for (int h = 0; h < height; h++) {
        WIDTH_LOOP:
          for (int w = 0; w < width; w++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#endif
            output[b][c][h][w] =
                weight[c] * (input[b][c][h][w] - running_mean[c]) * inv_std +
                bias[c];
          }
        }
      }
    }
  }

private:
};

} // namespace hls_nn
