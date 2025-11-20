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
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 32
#endif
      forward_1d_impl(output[b], input[b], weight, bias);
    }
  }

  static void forward(dtype *output, const dtype *input, const int batch_size,
                      const Weight_t weight, const Bias_t bias) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 32
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
#pragma HLS INLINE off
#endif
  OUTER_LOOP:
    for (int i = 0; i < out_features; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4096
#endif
      dtype acc = dtype(0);

    INNER_LOOP:
      for (int j = 0; j < in_features; j++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4096
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
#pragma HLS INLINE off
    dtype input[in_features];
  READ_LOOP:
    for (int i = 0; i < in_features; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4096
      input[i] = input_stream.read();
    }

  OUTER_LOOP:
    for (int i = 0; i < out_features; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4096
      dtype acc = dtype(0);
    INNER_LOOP:
      for (int j = 0; j < in_features; j++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4096
        acc += input[j] * weight[i][j];
      }

      output_stream.write(acc + bias[i]);
    }
  }
#endif
}; // namespace vhn

template <int UNROLL_FACTOR, int PARTITION_FACTOR> struct LinearConfig {
  static constexpr int unroll_factor = UNROLL_FACTOR;
  static constexpr int partition_factor = PARTITION_FACTOR;
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
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 32
#endif
      forward_1d_impl(output[b], input[b], weight, bias);
    }
  }

  static void forward(dtype *output, const dtype *input, const int batch_size,
                      const Weight_t weight, const Bias_t bias) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 32
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
#pragma HLS INLINE off

    constexpr bool should_partition = (partition_factor > 1) &&
                                      (in_features <= 1024) &&
                                      (out_features <= 512);

    if constexpr (should_partition) {
#pragma HLS ARRAY_PARTITION variable = input cyclic factor = partition_factor
#pragma HLS ARRAY_PARTITION variable = weight cyclic factor =                  \
    partition_factor dim = 2
#pragma HLS ARRAY_PARTITION variable = bias cyclic factor = partition_factor
    } else {
#pragma HLS BIND_STORAGE variable = weight type = rom_2p impl = bram
#pragma HLS BIND_STORAGE variable = input type = ram_1p impl = bram
#pragma HLS BIND_STORAGE variable = bias type = rom_1p impl = bram
    }
#endif

  OUTER_LOOP:
    for (int i = 0; i < out_features; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE style = flp
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4096
#endif
      dtype acc = dtype(0);
#ifdef __VITIS_HLS__
#pragma HLS BIND_OP variable = acc op = add impl = dsp
#endif

    INNER_LOOP:
      for (int j = 0; j < in_features; j++) {
#ifdef __VITIS_HLS__
        constexpr bool should_unroll =
            (unroll_factor > 1) && (in_features <= 512);
        if constexpr (should_unroll) {
#pragma HLS UNROLL factor = unroll_factor
        }
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4096
#pragma HLS BIND_OP variable = acc op = mul impl = dsp
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
#pragma HLS INLINE off
    dtype input[in_features];

  READ_LOOP:
    for (int i = 0; i < in_features; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4096
      input[i] = input_stream.read();
    }

  OUTER_LOOP:
    for (int i = 0; i < out_features; i++) {
#pragma HLS PIPELINE style = flp
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4096

      dtype acc = dtype(0);
#pragma HLS BIND_OP variable = acc op = add impl = dsp

    INNER_LOOP:
      for (int j = 0; j < in_features; j++) {
        constexpr bool should_unroll =
            (unroll_factor > 1) && (in_features <= 512);
        if constexpr (should_unroll) {
#pragma HLS UNROLL factor = unroll_factor
        }
#pragma HLS BIND_OP variable = acc op = mul impl = dsp
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4096
        acc += input[j] * weight[i][j];
      }
      output_stream.write(acc + bias[i]);
    }
  }

#endif
};

} // namespace vhn
