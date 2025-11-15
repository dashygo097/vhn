#pragma once

#include "../opt_level.hh"

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace vhn {

template <typename DType, typename HParams, typename Config, OptLevel OPT_LEVEL>
class Linear;

template <int IN_FEATURES, int OUT_FEATURES> struct LinearHParams {
  static constexpr int in_features = IN_FEATURES;
  static constexpr int out_features = OUT_FEATURES;
};

template <int UNROLL_FACTOR = 1, int PARTITION_FACTOR = 1, int PIPELINE_II = 1>
struct LinearConfig {
  static constexpr int unroll_factor = UNROLL_FACTOR;
  static constexpr int partition_factor = PARTITION_FACTOR;
  static constexpr int pipeline_ii = PIPELINE_II;
};

// ============================================================================
// Non-optimized version (OPT_NONE)
// ============================================================================
template <typename DType, typename HParams>
class Linear<DType, HParams, void, OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int in_features = HParams::in_features;
  static constexpr int out_features = HParams::out_features;
  static constexpr OptLevel opt_level = OPT_NONE;

  using Weight_t = dtype[out_features][in_features];
  using Bias_t = dtype[out_features];

  Linear() = default;
  ~Linear() = default;

  static void forward(dtype output[out_features],
                      const dtype input[in_features], const Weight_t weight,
                      const Bias_t bias) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    forward_1d_impl(output, input, weight, bias);
  }

  static void forward(dtype output[][out_features],
                      const dtype input[][in_features], const int batch_size,
                      const Weight_t weight, const Bias_t bias) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_1d_impl(output[b], input[b], weight, bias);
    }
  }

  static void forward(dtype *output, const dtype *input, const int batch_size,
                      const Weight_t weight, const Bias_t bias) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_1d_impl(&output[b * out_features], &input[b * in_features],
                      weight, bias);
    }
  }

#ifdef __VITIS_HLS__
  static void forward(hls::stream<dtype> &output_stream,
                      hls::stream<dtype> &input_stream, const Weight_t weight,
                      const Bias_t bias) {
#pragma HLS INLINE off
    forward_1d_stream_impl(output_stream, input_stream, weight, bias);
  }
#endif

private:
  static void forward_1d_impl(dtype *output, const dtype *input,
                              const Weight_t weight, const Bias_t bias) {
  OUTER_LOOP:
    for (int i = 0; i < out_features; i++) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
      dtype acc = dtype(0);

    INNER_LOOP:
      for (int j = 0; j < in_features; j++) {
        acc += input[j] * weight[i][j];
      }
      output[i] = acc + bias[i];
    }
  }

#ifdef __VITIS_HLS__
  static void forward_1d_stream_impl(hls::stream<dtype> &output_stream,
                                     hls::stream<dtype> &input_stream,
                                     const Weight_t weight, const Bias_t bias) {
    dtype input_buffer[in_features];

  READ_INPUT:
    for (int j = 0; j < in_features; j++) {
      input_buffer[j] = input_stream.read();
    }

  OUTER_LOOP:
    for (int i = 0; i < out_features; i++) {
      dtype acc = dtype(0);

    INNER_LOOP:
      for (int j = 0; j < in_features; j++) {
        acc += input_buffer[j] * weight[i][j];
      }
      output_stream.write(acc + bias[i]);
    }
  }
#endif
};

// ============================================================================
// Optimized version (OPT_ENABLED)
// ============================================================================
template <typename DType, typename HParams, typename Config>
class Linear<DType, HParams, Config, OPT_ENABLED> {
public:
  using dtype = DType;
  static constexpr int in_features = HParams::in_features;
  static constexpr int out_features = HParams::out_features;
  static constexpr OptLevel opt_level = OPT_ENABLED;

  static constexpr int unroll_factor = Config::unroll_factor;
  static constexpr int partition_factor = Config::partition_factor;
  static constexpr int pipeline_ii = Config::pipeline_ii;

  using Weight_t = dtype[out_features][in_features];
  using Bias_t = dtype[out_features];

  Linear() = default;
  ~Linear() = default;

  static void forward(dtype output[out_features],
                      const dtype input[in_features], const Weight_t weight,
                      const Bias_t bias) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    forward_1d_impl(output, input, weight, bias);
  }

  static void forward(dtype output[][out_features],
                      const dtype input[][in_features], const int batch_size,
                      const Weight_t weight, const Bias_t bias) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_1d_impl(output[b], input[b], weight, bias);
    }
  }

  static void forward(dtype *output, const dtype *input, const int batch_size,
                      const Weight_t weight, const Bias_t bias) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_1d_impl(&output[b * out_features], &input[b * in_features],
                      weight, bias);
    }
  }

#ifdef __VITIS_HLS__
  static void forward(hls::stream<dtype> &output_stream,
                      hls::stream<dtype> &input_stream, const Weight_t weight,
                      const Bias_t bias) {
#pragma HLS INLINE off
    forward_1d_stream_impl(output_stream, input_stream, weight, bias);
  }
#endif

private:
  static void forward_1d_impl(dtype *output, const dtype *input,
                              const Weight_t weight, const Bias_t bias) {
#ifdef __VITIS_HLS__
#pragma HLS ARRAY_PARTITION variable = input cyclic factor = partition_factor
#pragma HLS ARRAY_PARTITION variable = weight cyclic factor =                  \
    partition_factor dim = 2
#pragma HLS ARRAY_PARTITION variable = bias cyclic factor = partition_factor
#endif

  OUTER_LOOP:
    for (int i = 0; i < out_features; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#endif
      dtype acc = dtype(0);

    INNER_LOOP:
      for (int j = 0; j < in_features; j++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = unroll_factor
#endif
        acc += input[j] * weight[i][j];
      }
      output[i] = acc + bias[i];
    }
  }

#ifdef __VITIS_HLS__
  static void forward_1d_stream_impl(hls::stream<dtype> &output_stream,
                                     hls::stream<dtype> &input_stream,
                                     const Weight_t weight, const Bias_t bias) {
    dtype input_buffer[in_features];
#pragma HLS ARRAY_PARTITION variable = input_buffer cyclic factor =            \
    partition_factor
#pragma HLS ARRAY_PARTITION variable = weight cyclic factor =                  \
    partition_factor dim = 2
#pragma HLS ARRAY_PARTITION variable = bias cyclic factor = partition_factor

  READ_INPUT:
    for (int j = 0; j < in_features; j++) {
#pragma HLS PIPELINE II = pipeline_ii
      input_buffer[j] = input_stream.read();
    }

  OUTER_LOOP:
    for (int i = 0; i < out_features; i++) {
#pragma HLS PIPELINE II = pipeline_ii
      dtype acc = dtype(0);

    INNER_LOOP:
      for (int j = 0; j < in_features; j++) {
#pragma HLS UNROLL factor = unroll_factor
        acc += input_buffer[j] * weight[i][j];
      }
      output_stream.write(acc + bias[i]);
    }
  }
#endif
};
} // namespace vhn
