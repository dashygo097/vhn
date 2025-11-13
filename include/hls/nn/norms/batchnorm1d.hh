#pragma once

#include "../../opt_level.hh"
#include <cmath>

#ifdef __VITIS_HLS__
#include <hls_math.h>
#include <hls_stream.h>
#endif

namespace hls_nn {

template <typename DType, const int CHANNELS, typename Config = void,
          OptLevel OPT_LEVEL = OPT_NONE>
class BatchNorm1d;

// ============================================================================
// Non-optimized version (OPT_NONE)
// ============================================================================
template <typename DType, const int CHANNELS>
class BatchNorm1d<DType, CHANNELS, void, OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int channels = CHANNELS;
  static constexpr OptLevel opt_level = OPT_NONE;

  using Weight_t = dtype[CHANNELS];
  using Bias_t = dtype[CHANNELS];
  using RunningMean_t = dtype[CHANNELS];
  using RunningVar_t = dtype[CHANNELS];

  BatchNorm1d() = default;
  ~BatchNorm1d() = default;

  static void forward(dtype output[CHANNELS], const dtype input[CHANNELS],
                      const Weight_t weight, const Bias_t bias,
                      const RunningMean_t running_mean,
                      const RunningVar_t running_var,
                      const float epsilon = 1e-5) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    forward_1d_impl(output, input, weight, bias, running_mean, running_var,
                    epsilon);
  }

  static void forward(dtype output[][CHANNELS], const dtype input[][CHANNELS],
                      const int batch_size, const Weight_t weight,
                      const Bias_t bias, const RunningMean_t running_mean,
                      const RunningVar_t running_var,
                      const float epsilon = 1e-5) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_1d_impl(output[b], input[b], weight, bias, running_mean,
                      running_var, epsilon);
    }
  }

  static void forward(dtype *output, const dtype *input, const int batch_size,
                      const Weight_t weight, const Bias_t bias,
                      const RunningMean_t running_mean,
                      const RunningVar_t running_var,
                      const float epsilon = 1e-5) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_1d_impl(&output[b * CHANNELS], &input[b * CHANNELS], weight, bias,
                      running_mean, running_var, epsilon);
    }
  }

#ifdef __VITIS_HLS__
  static void forward(hls::stream<dtype> &output_stream,
                      hls::stream<dtype> &input_stream, const Weight_t weight,
                      const Bias_t bias, const RunningMean_t running_mean,
                      const RunningVar_t running_var,
                      const float epsilon = 1e-5) {
#pragma HLS INLINE off
    forward_1d_stream_impl(output_stream, input_stream, weight, bias,
                           running_mean, running_var, epsilon);
  }

#endif

private:
  static void forward_1d_impl(dtype *output, const dtype *input,
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
      output[c] = weight[c] * (input[c] - running_mean[c]) * inv_std + bias[c];
    }
  }

#ifdef __VITIS_HLS__
  static void forward_1d_stream_impl(hls::stream<dtype> &output_stream,
                                     hls::stream<dtype> &input_stream,
                                     const Weight_t weight, const Bias_t bias,
                                     const RunningMean_t running_mean,
                                     const RunningVar_t running_var,
                                     const float epsilon) {
    dtype input_buffer[CHANNELS];

  READ_INPUT:
    for (int c = 0; c < CHANNELS; c++) {
      input_buffer[c] = input_stream.read();
    }

  CHANNEL_STREAM_LOOP:
    for (int c = 0; c < CHANNELS; c++) {
      dtype inv_std = hls::rsqrt(running_var[c] + dtype(epsilon));
      dtype output_val =
          weight[c] * (input_buffer[c] - running_mean[c]) * inv_std + bias[c];
      output_stream.write(output_val);
    }
  }
#endif
};

// ============================================================================
// Optimized version (OPT_ENABLED)
// ============================================================================
template <typename DType, const int CHANNELS, typename Config>
class BatchNorm1d<DType, CHANNELS, Config, OPT_ENABLED> {
public:
  using dtype = DType;
  static constexpr int channels = CHANNELS;
  static constexpr OptLevel opt_level = OPT_ENABLED;

  static constexpr int unroll_factor = Config::_unroll_factor;
  static constexpr int partition_factor = Config::_partition_factor;
  static constexpr int pipeline_ii = Config::_pipeline_ii;

  using Weight_t = dtype[CHANNELS];
  using Bias_t = dtype[CHANNELS];
  using RunningMean_t = dtype[CHANNELS];
  using RunningVar_t = dtype[CHANNELS];

  BatchNorm1d() = default;
  ~BatchNorm1d() = default;

  static void forward(dtype output[CHANNELS], const dtype input[CHANNELS],
                      const Weight_t weight, const Bias_t bias,
                      const RunningMean_t running_mean,
                      const RunningVar_t running_var,
                      const float epsilon = 1e-5) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    forward_1d_impl(output, input, weight, bias, running_mean, running_var,
                    epsilon);
  }

  static void forward(dtype output[][CHANNELS], const dtype input[][CHANNELS],
                      const int batch_size, const Weight_t weight,
                      const Bias_t bias, const RunningMean_t running_mean,
                      const RunningVar_t running_var,
                      const float epsilon = 1e-5) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_1d_impl(output[b], input[b], weight, bias, running_mean,
                      running_var, epsilon);
    }
  }

  static void forward(dtype *output, const dtype *input, const int batch_size,
                      const Weight_t weight, const Bias_t bias,
                      const RunningMean_t running_mean,
                      const RunningVar_t running_var,
                      const float epsilon = 1e-5) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_1d_impl(&output[b * CHANNELS], &input[b * CHANNELS], weight, bias,
                      running_mean, running_var, epsilon);
    }
  }

#ifdef __VITIS_HLS__
  static void forward(hls::stream<dtype> &output_stream,
                      hls::stream<dtype> &input_stream, const Weight_t weight,
                      const Bias_t bias, const RunningMean_t running_mean,
                      const RunningVar_t running_var,
                      const float epsilon = 1e-5) {
#pragma HLS INLINE off
    forward_1d_stream_impl(output_stream, input_stream, weight, bias,
                           running_mean, running_var, epsilon);
  }

#endif

private:
  static void forward_1d_impl(dtype *output, const dtype *input,
                              const Weight_t weight, const Bias_t bias,
                              const RunningMean_t running_mean,
                              const RunningVar_t running_var,
                              const float epsilon) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = input cyclic factor = partition_factor
#pragma HLS ARRAY_PARTITION variable = output cyclic factor = partition_factor
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
#pragma HLS PIPELINE II = pipeline_ii
#endif
#ifdef __VITIS_HLS__
      dtype inv_std = hls::rsqrt(running_var[c] + dtype(epsilon));
#else
      dtype inv_std = dtype(1.0) / std::sqrt(running_var[c] + dtype(epsilon));
#endif
      output[c] = weight[c] * (input[c] - running_mean[c]) * inv_std + bias[c];
    }
  }

#ifdef __VITIS_HLS__
  static void forward_1d_stream_impl(hls::stream<dtype> &output_stream,
                                     hls::stream<dtype> &input_stream,
                                     const Weight_t weight, const Bias_t bias,
                                     const RunningMean_t running_mean,
                                     const RunningVar_t running_var,
                                     const float epsilon) {
    dtype input_buffer[CHANNELS];
#pragma HLS ARRAY_PARTITION variable = input_buffer cyclic factor =            \
    partition_factor

  READ_INPUT:
    for (int c = 0; c < CHANNELS; c++) {
#pragma HLS PIPELINE II = pipeline_ii
      input_buffer[c] = input_stream.read();
    }

  CHANNEL_STREAM_LOOP:
    for (int c = 0; c < CHANNELS; c++) {
#pragma HLS PIPELINE II = pipeline_ii
#pragma HLS UNROLL factor = unroll_factor
      dtype inv_std = hls::rsqrt(running_var[c] + dtype(epsilon));
      dtype output_val =
          weight[c] * (input_buffer[c] - running_mean[c]) * inv_std + bias[c];
      output_stream.write(output_val);
    }
  }
#endif
};

} // namespace hls_nn
