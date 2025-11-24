#pragma once

#include "../opt_level.hh"
#include <cmath>

#ifdef __VITIS_HLS__
#include <hls_math.h>
#include <hls_stream.h>
#endif

namespace vhn {

template <typename DType, const int CHANNELS, const int WIDTH, const int HEIGHT,
          typename Config = void, OptLevel OPT_LEVEL = OPT_NONE>
class BatchNorm2d;

// ============================================================================
// Non-optimized version (OPT_NONE)
// ============================================================================
template <typename DType, const int CHANNELS, const int WIDTH, const int HEIGHT>
class BatchNorm2d<DType, CHANNELS, WIDTH, HEIGHT, void, OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int channels = CHANNELS;
  static constexpr int width = WIDTH;
  static constexpr int height = HEIGHT;
  static constexpr int spatial_size = WIDTH * HEIGHT;
  static constexpr OptLevel opt_level = OPT_NONE;

  using Tensor_2d_t = dtype[CHANNELS][spatial_size];
  using Tensor_3d_t = dtype[CHANNELS][HEIGHT][WIDTH];
  using Weight_t = dtype[CHANNELS];
  using Bias_t = dtype[CHANNELS];
  using RunningMean_t = dtype[CHANNELS];
  using RunningVar_t = dtype[CHANNELS];

  BatchNorm2d() = default;
  ~BatchNorm2d() = default;

  static void bn2d(Tensor_3d_t output, const Tensor_3d_t input,
                   const Weight_t weight, const Bias_t bias,
                   const RunningMean_t running_mean,
                   const RunningVar_t running_var, const float epsilon = 1e-5) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    bn2d_3d_impl(output, input, weight, bias, running_mean, running_var,
                 epsilon);
  }

  static void bn2d(Tensor_2d_t output, const Tensor_2d_t input,
                   const Weight_t weight, const Bias_t bias,
                   const RunningMean_t running_mean,
                   const RunningVar_t running_var, const float epsilon = 1e-5) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    bn2d_2d_impl(output, input, weight, bias, running_mean, running_var,
                 epsilon);
  }

  static void bn2d(dtype output[][CHANNELS][HEIGHT][WIDTH],
                   const dtype input[][CHANNELS][HEIGHT][WIDTH],
                   const int batch_size, const Weight_t weight,
                   const Bias_t bias, const RunningMean_t running_mean,
                   const RunningVar_t running_var, const float epsilon = 1e-5) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      bn2d_3d_impl(output[b], input[b], weight, bias, running_mean, running_var,
                   epsilon);
    }
  }

  static void bn2d(dtype output[][CHANNELS][spatial_size],
                   const dtype input[][CHANNELS][spatial_size],
                   const int batch_size, const Weight_t weight,
                   const Bias_t bias, const RunningMean_t running_mean,
                   const RunningVar_t running_var, const float epsilon = 1e-5) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      bn2d_2d_impl(output[b], input[b], weight, bias, running_mean, running_var,
                   epsilon);
    }
  }

#ifdef __VITIS_HLS__
  static void bn2d(hls::stream<dtype> &output_stream,
                   hls::stream<dtype> &input_stream, const Weight_t weight,
                   const Bias_t bias, const RunningMean_t running_mean,
                   const RunningVar_t running_var, const float epsilon = 1e-5) {
#pragma HLS INLINE off
    bn2d_stream_impl(output_stream, input_stream, weight, bias, running_mean,
                     running_var, epsilon);
  }

#endif

private:
  static void bn2d_2d_impl(Tensor_2d_t output, const Tensor_2d_t input,
                           const Weight_t weight, const Bias_t bias,
                           const RunningMean_t running_mean,
                           const RunningVar_t running_var,
                           const float epsilon) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
  CHANNEL_LOOP:
    for (int c = 0; c < channels; c++) {
#ifdef __VITIS_HLS__
      dtype inv_std = hls::rsqrt(running_var[c] + dtype(epsilon));
#else
      dtype inv_std = dtype(1.0) / std::sqrt(running_var[c] + dtype(epsilon));
#endif

    SPATIAL_LOOP:
      for (int s = 0; s < spatial_size; s++) {
        output[c][s] =
            weight[c] * (input[c][s] - running_mean[c]) * inv_std + bias[c];
      }
    }
  }

  static void bn2d_3d_impl(Tensor_3d_t output, const Tensor_3d_t input,
                           const Weight_t weight, const Bias_t bias,
                           const RunningMean_t running_mean,
                           const RunningVar_t running_var,
                           const float epsilon) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
  CHANNEL_LOOP:
    for (int c = 0; c < channels; c++) {
#ifdef __VITIS_HLS__
      dtype inv_std = hls::rsqrt(running_var[c] + dtype(epsilon));
#else
      dtype inv_std = dtype(1.0) / std::sqrt(running_var[c] + dtype(epsilon));
#endif

    HEIGHT_LOOP:
      for (int h = 0; h < height; h++) {
      WIDTH_LOOP:
        for (int w = 0; w < width; w++) {
          output[c][h][w] =
              weight[c] * (input[c][h][w] - running_mean[c]) * inv_std +
              bias[c];
        }
      }
    }
  }

#ifdef __VITIS_HLS__
  static void bn2d_stream_impl(hls::stream<dtype> &output_stream,
                               hls::stream<dtype> &input_stream,
                               const Weight_t weight, const Bias_t bias,
                               const RunningMean_t running_mean,
                               const RunningVar_t running_var,
                               const float epsilon) {
    dtype input_buffer[spatial_size];

  READ_INPUT:
    for (int s = 0; s < spatial_size; s++) {
      input_buffer[s] = input_stream.read();
    }

  CHANNEL_STREAM_LOOP:
    for (int c = 0; c < channels; c++) {
      dtype inv_std = hls::rsqrt(running_var[c] + dtype(epsilon));

    SPATIAL_STREAM_LOOP:
      for (int s = 0; s < spatial_size; s++) {
        dtype output_val =
            weight[c] * (input_buffer[s] - running_mean[c]) * inv_std + bias[c];
        output_stream.write(output_val);
      }
    }
  }
#endif
};

template <int PIPELINE_II, int UNROLL_FACTOR, int PARTITION_FACTOR>
struct BatchNorm2dConfig {
  static constexpr int pipeline_ii = PIPELINE_II;
  static constexpr int unroll_factor = UNROLL_FACTOR;
  static constexpr int partition_factor = PARTITION_FACTOR;
};

// ============================================================================
// Optimized version (OPT_ENABLED)
// ============================================================================
template <typename DType, const int CHANNELS, const int WIDTH, const int HEIGHT,
          typename Config>
class BatchNorm2d<DType, CHANNELS, WIDTH, HEIGHT, Config, OPT_ENABLED> {
public:
  using dtype = DType;
  static constexpr int channels = CHANNELS;
  static constexpr int width = WIDTH;
  static constexpr int height = HEIGHT;
  static constexpr int spatial_size = WIDTH * HEIGHT;
  static constexpr OptLevel opt_level = OPT_ENABLED;

  static constexpr int pipeline_ii = Config::pipeline_ii;
  static constexpr int unroll_factor = Config::unroll_factor;
  static constexpr int partition_factor = Config::partition_factor;

  using Tensor_2d_t = dtype[CHANNELS][spatial_size];
  using Tensor_3d_t = dtype[CHANNELS][HEIGHT][WIDTH];
  using Weight_t = dtype[CHANNELS];
  using Bias_t = dtype[CHANNELS];
  using RunningMean_t = dtype[CHANNELS];
  using RunningVar_t = dtype[CHANNELS];

  BatchNorm2d() = default;
  ~BatchNorm2d() = default;

  static void bn2d(Tensor_3d_t output, const Tensor_3d_t input,
                   const Weight_t weight, const Bias_t bias,
                   const RunningMean_t running_mean,
                   const RunningVar_t running_var, const float epsilon = 1e-5) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    bn2d_3d_impl(output, input, weight, bias, running_mean, running_var,
                 epsilon);
  }

  static void bn2d(Tensor_2d_t output, const Tensor_2d_t input,
                   const Weight_t weight, const Bias_t bias,
                   const RunningMean_t running_mean,
                   const RunningVar_t running_var, const float epsilon = 1e-5) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    bn2d_2d_impl(output, input, weight, bias, running_mean, running_var,
                 epsilon);
  }

  static void bn2d(dtype output[][CHANNELS][HEIGHT][WIDTH],
                   const dtype input[][CHANNELS][HEIGHT][WIDTH],
                   const int batch_size, const Weight_t weight,
                   const Bias_t bias, const RunningMean_t running_mean,
                   const RunningVar_t running_var, const float epsilon = 1e-5) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      bn2d_3d_impl(output[b], input[b], weight, bias, running_mean, running_var,
                   epsilon);
    }
  }

  static void bn2d(dtype output[][CHANNELS][spatial_size],
                   const dtype input[][CHANNELS][spatial_size],
                   const int batch_size, const Weight_t weight,
                   const Bias_t bias, const RunningMean_t running_mean,
                   const RunningVar_t running_var, const float epsilon = 1e-5) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      bn2d_2d_impl(output[b], input[b], weight, bias, running_mean, running_var,
                   epsilon);
    }
  }

#ifdef __VITIS_HLS__
  static void bn2d(hls::stream<dtype> &output_stream,
                   hls::stream<dtype> &input_stream, const Weight_t weight,
                   const Bias_t bias, const RunningMean_t running_mean,
                   const RunningVar_t running_var, const float epsilon = 1e-5) {
#pragma HLS INLINE off
    bn2d_stream_impl(output_stream, input_stream, weight, bias, running_mean,
                     running_var, epsilon);
  }

#endif

private:
  static void bn2d_2d_impl(Tensor_2d_t output, const Tensor_2d_t input,
                           const Weight_t weight, const Bias_t bias,
                           const RunningMean_t running_mean,
                           const RunningVar_t running_var,
                           const float epsilon) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = input cyclic factor =                   \
    partition_factor dim = 1
#pragma HLS ARRAY_PARTITION variable = output cyclic factor =                  \
    partition_factor dim = 1
#pragma HLS ARRAY_PARTITION variable = weight cyclic factor = partition_factor
#pragma HLS ARRAY_PARTITION variable = bias cyclic factor = partition_factor
#pragma HLS ARRAY_PARTITION variable = running_mean cyclic factor =            \
    partition_factor
#pragma HLS ARRAY_PARTITION variable = running_var cyclic factor =             \
    partition_factor
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

    SPATIAL_LOOP:
      for (int s = 0; s < spatial_size; s++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#endif
        output[c][s] =
            weight[c] * (input[c][s] - running_mean[c]) * inv_std + bias[c];
      }
    }
  }

  static void bn2d_3d_impl(Tensor_3d_t output, const Tensor_3d_t input,
                           const Weight_t weight, const Bias_t bias,
                           const RunningMean_t running_mean,
                           const RunningVar_t running_var,
                           const float epsilon) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = input cyclic factor =                   \
    partition_factor dim = 1
#pragma HLS ARRAY_PARTITION variable = output cyclic factor =                  \
    partition_factor dim = 1
#pragma HLS ARRAY_PARTITION variable = weight cyclic factor = partition_factor
#pragma HLS ARRAY_PARTITION variable = bias cyclic factor = partition_factor
#pragma HLS ARRAY_PARTITION variable = running_mean cyclic factor =            \
    partition_factor
#pragma HLS ARRAY_PARTITION variable = running_var cyclic factor =             \
    partition_factor
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
          output[c][h][w] =
              weight[c] * (input[c][h][w] - running_mean[c]) * inv_std +
              bias[c];
        }
      }
    }
  }

#ifdef __VITIS_HLS__
  static void bn2d_stream_impl(hls::stream<dtype> &output_stream,
                               hls::stream<dtype> &input_stream,
                               const Weight_t weight, const Bias_t bias,
                               const RunningMean_t running_mean,
                               const RunningVar_t running_var,
                               const float epsilon) {
    dtype input_buffer[spatial_size];
#pragma HLS ARRAY_PARTITION variable = input_buffer cyclic factor =            \
    partition_factor

  READ_INPUT:
    for (int s = 0; s < spatial_size; s++) {
#pragma HLS PIPELINE II = pipeline_ii
      input_buffer[s] = input_stream.read();
    }

  CHANNEL_STREAM_LOOP:
    for (int c = 0; c < channels; c++) {
#pragma HLS UNROLL factor = unroll_factor
      dtype inv_std = hls::rsqrt(running_var[c] + dtype(epsilon));

    SPATIAL_STREAM_LOOP:
      for (int s = 0; s < spatial_size; s++) {
#pragma HLS PIPELINE II = pipeline_ii
        dtype output_val =
            weight[c] * (input_buffer[s] - running_mean[c]) * inv_std + bias[c];
        output_stream.write(output_val);
      }
    }
  }
#endif
};

} // namespace vhn
