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

  static void lin(dtype output[out_features], const dtype input[in_features],
                  const Weight_t weight, const Bias_t bias) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    lin_1d_impl(output, input, weight, bias);
  }

  static void lin(dtype output[][out_features],
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
      lin_1d_impl(output[b], input[b], weight, bias);
    }
  }

  static void lin(dtype *output, const dtype *input, const int batch_size,
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
      lin_1d_impl(&output[b * out_features], &input[b * in_features], weight,
                  bias);
    }
  }

#ifdef __VITIS_HLS__
  static void lin(hls::stream<dtype> &output_stream,
                  hls::stream<dtype> &input_stream, const Weight_t weight,
                  const Bias_t bias) {
#pragma HLS INLINE off
    lin_1d_stream_impl(output_stream, input_stream, weight, bias);
  }

#endif

private:
  static void lin_1d_impl(dtype *output, const dtype *input,
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
  static void lin_1d_stream_impl(hls::stream<dtype> &output_stream,
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

template <int UNROLL_FACTOR, int PARTITION_FACTOR, int TILE_SIZE_OUT,
          int TILE_SIZE_IN, bool USE_SYSTOLIC>
struct LinearConfig {
  static constexpr int unroll_factor = UNROLL_FACTOR;
  static constexpr int partition_factor = PARTITION_FACTOR;
  static constexpr int tile_size_out = TILE_SIZE_OUT;
  static constexpr int tile_size_in = TILE_SIZE_IN;
  static constexpr bool use_systolic = USE_SYSTOLIC;
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
  static constexpr int tile_size_out = Config::tile_size_out;
  static constexpr int tile_size_in = Config::tile_size_in;
  static constexpr bool use_systolic = Config::use_systolic;

  using Weight_t = dtype[out_features][in_features];
  using Bias_t = dtype[out_features];

  Linear() = default;
  ~Linear() = default;

  static void lin(dtype output[out_features], const dtype input[in_features],
                  const Weight_t weight, const Bias_t bias) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    lin_1d_impl(output, input, weight, bias);
  }

  static void lin(dtype output[][out_features],
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
      lin_1d_impl(output[b], input[b], weight, bias);
    }
  }

  static void lin(dtype *output, const dtype *input, const int batch_size,
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
      lin_1d_impl(&output[b * out_features], &input[b * in_features], weight,
                  bias);
    }
  }

#ifdef __VITIS_HLS__
  static void lin(hls::stream<dtype> &output_stream,
                  hls::stream<dtype> &input_stream, const Weight_t weight,
                  const Bias_t bias) {
#pragma HLS INLINE off
    lin_1d_stream_impl(output_stream, input_stream, weight, bias);
  }
#endif

private:
  static void lin_1d_impl(dtype *output, const dtype *input,
                          const Weight_t weight, const Bias_t bias) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off

    constexpr bool should_partition = (partition_factor > 1) &&
                                      (in_features <= 2048) &&
                                      (out_features <= 1024);

    if constexpr (should_partition) {
#pragma HLS ARRAY_PARTITION variable = input cyclic factor = partition_factor
#pragma HLS ARRAY_PARTITION variable = weight cyclic factor =                  \
    partition_factor dim = 2
#pragma HLS ARRAY_PARTITION variable = bias cyclic factor = partition_factor
    }
#endif

    dtype output_buffer[out_features];

    constexpr int OUT_TILE = (tile_size_out > 0 && tile_size_out < out_features)
                                 ? tile_size_out
                                 : out_features;
    constexpr int IN_TILE = (tile_size_in > 0 && tile_size_in < in_features)
                                ? tile_size_in
                                : in_features;

    if constexpr (use_systolic && OUT_TILE < out_features &&
                  IN_TILE < in_features) {
      lin_tiled_systolic(output_buffer, input, weight, bias);
    } else if constexpr (OUT_TILE < out_features || IN_TILE < in_features) {
      lin_tiled(output_buffer, input, weight, bias);
    } else {
      lin_standard(output_buffer, input, weight, bias);
    }

#ifdef __VITIS_HLS__
    for (int i = 0; i < out_features; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4096
      output[i] = output_buffer[i];
    }
#else
    for (int i = 0; i < out_features; i++) {
      output[i] = output_buffer[i];
    }
#endif
  }

  static void lin_standard(dtype *output, const dtype *input,
                           const Weight_t weight, const Bias_t bias) {
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
            (unroll_factor > 1) && (in_features <= 1024);
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

  static void lin_tiled(dtype *output, const dtype *input,
                        const Weight_t weight, const Bias_t bias) {
    constexpr int OUT_TILE = (tile_size_out > 0) ? tile_size_out : 32;
    constexpr int IN_TILE = (tile_size_in > 0) ? tile_size_in : 64;

    for (int i = 0; i < out_features; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4096
#endif
      output[i] = bias[i];
    }

  OUT_TILE_LOOP:
    for (int i_tile = 0; i_tile < out_features; i_tile += OUT_TILE) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 128
#endif
      int i_end =
          (i_tile + OUT_TILE < out_features) ? i_tile + OUT_TILE : out_features;

    IN_TILE_LOOP:
      for (int j_tile = 0; j_tile < in_features; j_tile += IN_TILE) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 64
#pragma HLS LOOP_FLATTEN off
#endif
        int j_end =
            (j_tile + IN_TILE < in_features) ? j_tile + IN_TILE : in_features;

        // Local tile buffers
        dtype input_tile[IN_TILE];
#ifdef __VITIS_HLS__
#pragma HLS ARRAY_PARTITION variable = input_tile complete
#endif

        for (int j = 0; j < (j_end - j_tile); j++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 64
#endif
          input_tile[j] = input[j_tile + j];
        }

      OUT_LOOP:
        for (int i = i_tile; i < i_end; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 32
#endif
          dtype acc = dtype(0);
#ifdef __VITIS_HLS__
#pragma HLS BIND_OP variable = acc op = add impl = dsp
#endif

        IN_LOOP:
          for (int j = 0; j < (j_end - j_tile); j++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = unroll_factor
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 64
#pragma HLS BIND_OP variable = acc op = mul impl = dsp
#endif
            acc += input_tile[j] * weight[i][j_tile + j];
          }
          output[i] += acc;
        }
      }
    }
  }

  static void lin_tiled_systolic(dtype *output, const dtype *input,
                                 const Weight_t weight, const Bias_t bias) {
    constexpr int OUT_TILE = (tile_size_out > 0) ? tile_size_out : 16;
    constexpr int IN_TILE = (tile_size_in > 0) ? tile_size_in : 16;

    for (int i = 0; i < out_features; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = 1
#endif
      output[i] = bias[i];
    }

  OUT_TILE_LOOP:
    for (int i_tile = 0; i_tile < out_features; i_tile += OUT_TILE) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 256
#endif

    IN_TILE_LOOP:
      for (int j_tile = 0; j_tile < in_features; j_tile += IN_TILE) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 256
#endif

        dtype pe_acc[OUT_TILE][IN_TILE];
#ifdef __VITIS_HLS__
#pragma HLS ARRAY_PARTITION variable = pe_acc complete
#endif

        int i_end = (i_tile + OUT_TILE < out_features)
                        ? OUT_TILE
                        : (out_features - i_tile);
        int j_end =
            (j_tile + IN_TILE < in_features) ? IN_TILE : (in_features - j_tile);

        for (int i = 0; i < i_end; i++) {
          for (int j = 0; j < j_end; j++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = 1
#pragma HLS UNROLL factor = unroll_factor
#pragma HLS BIND_OP variable = pe_acc op = mul impl = dsp
#endif
            pe_acc[i][j] = input[j_tile + j] * weight[i_tile + i][j_tile + j];
          }
        }

        for (int i = 0; i < i_end; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = 1
#endif
          dtype sum = dtype(0);
#ifdef __VITIS_HLS__
#pragma HLS BIND_OP variable = sum op = add impl = dsp
#endif
          for (int j = 0; j < j_end; j++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = unroll_factor
#endif
            sum += pe_acc[i][j];
          }
          output[i_tile + i] += sum;
        }
      }
    }
  }

#ifdef __VITIS_HLS__
  static void lin_1d_stream_impl(hls::stream<dtype> &output_stream,
                                 hls::stream<dtype> &input_stream,
                                 const Weight_t weight, const Bias_t bias) {
#pragma HLS INLINE off

    dtype input[in_features];
#pragma HLS ARRAY_PARTITION variable = input cyclic factor = partition_factor

  READ_LOOP:
    for (int i = 0; i < in_features; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4096
      input[i] = input_stream.read();
    }

    constexpr int OUT_TILE = (tile_size_out > 0) ? tile_size_out : 32;

  OUTER_TILE_LOOP:
    for (int i_tile = 0; i_tile < out_features; i_tile += OUT_TILE) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 128

      int i_end =
          (i_tile + OUT_TILE < out_features) ? i_tile + OUT_TILE : out_features;

      dtype output_tile[OUT_TILE];
#pragma HLS ARRAY_PARTITION variable = output_tile complete

      for (int i = 0; i < (i_end - i_tile); i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 32

        dtype acc = dtype(0);
#pragma HLS BIND_OP variable = acc op = add impl = dsp

        for (int j = 0; j < in_features; j++) {
          constexpr bool should_unroll =
              (unroll_factor > 1) && (in_features <= 512);
          if constexpr (should_unroll) {
#pragma HLS UNROLL factor = unroll_factor
          }
#pragma HLS BIND_OP variable = acc op = mul impl = dsp
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4096
          acc += input[j] * weight[i_tile + i][j];
        }
        output_tile[i] = acc + bias[i_tile + i];
      }

      for (int i = 0; i < (i_end - i_tile); i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 32
        output_stream.write(output_tile[i]);
      }
    }
  }
#endif
};
} // namespace vhn
