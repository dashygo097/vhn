#pragma once

#include "../opt_level.hh"
#include <cmath>

#ifdef __VITIS_HLS__
#include <hls_math.h>
#include <hls_stream.h>
#endif

namespace vhn {

template <typename DType, typename HParams, typename Config = void,
          OptLevel OPT_LEVEL = OPT_NONE>
class BatchNorm1d;

template <int CHANNELS> struct BatchNorm1dHParams {
  static constexpr int channels = CHANNELS;
};

// ============================================================================
// Non-optimized version (OPT_NONE)
// ============================================================================
template <typename DType, typename HParams>
class BatchNorm1d<DType, HParams, void, OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int channels = HParams::channels;
  static constexpr OptLevel opt_level = OPT_NONE;

  using Weight_t = dtype[channels];
  using Bias_t = dtype[channels];
  using RunningMean_t = dtype[channels];
  using RunningVar_t = dtype[channels];

  BatchNorm1d() = default;
  ~BatchNorm1d() = default;

  static void forward(dtype output[channels], const dtype input[channels],
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

  static void forward(dtype output[][channels], const dtype input[][channels],
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
      forward_1d_impl(&output[b * channels], &input[b * channels], weight, bias,
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
    dtype input_buffer[channels];

  READ_INPUT:
    for (int c = 0; c < channels; c++) {
      input_buffer[c] = input_stream.read();
    }

  CHANNEL_STREAM_LOOP:
    for (int c = 0; c < channels; c++) {
      dtype inv_std = hls::rsqrt(running_var[c] + dtype(epsilon));
      dtype output_val =
          weight[c] * (input_buffer[c] - running_mean[c]) * inv_std + bias[c];
      output_stream.write(output_val);
    }
  }
#endif
};

template <int PIPELINE_II, int UNROLL_FACTOR, int PARTITION_FACTOR>
struct BatchNorm1dConfig {
  static constexpr int pipeline_ii = PIPELINE_II;
  static constexpr int unroll_factor = UNROLL_FACTOR;
  static constexpr int partition_factor = PARTITION_FACTOR;
};

// ============================================================================
// Optimized version (OPT_ENABLED)
// ============================================================================
template <typename DType, typename HParams, typename Config>
class BatchNorm1d<DType, HParams, Config, OPT_ENABLED> {
public:
  using dtype = DType;
  static constexpr int channels = HParams::channels;
  static constexpr OptLevel opt_level = OPT_ENABLED;

  static constexpr int pipeline_ii = Config::pipeline_ii;
  static constexpr int unroll_factor = Config::unroll_factor;
  static constexpr int partition_factor = Config::partition_factor;

  using Weight_t = dtype[channels];
  using Bias_t = dtype[channels];
  using RunningMean_t = dtype[channels];
  using RunningVar_t = dtype[channels];

  BatchNorm1d() = default;
  ~BatchNorm1d() = default;

  static void forward(dtype output[channels], const dtype input[channels],
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

  static void forward(dtype output[][channels], const dtype input[][channels],
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
      forward_1d_impl(&output[b * channels], &input[b * channels], weight, bias,
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
    dtype input_buffer[channels];
#pragma HLS ARRAY_PARTITION variable = input_buffer cyclic factor =            \
    partition_factor

  READ_INPUT:
    for (int c = 0; c < channels; c++) {
#pragma HLS PIPELINE II = pipeline_ii
      input_buffer[c] = input_stream.read();
    }

  CHANNEL_STREAM_LOOP:
    for (int c = 0; c < channels; c++) {
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

} // namespace vhn
