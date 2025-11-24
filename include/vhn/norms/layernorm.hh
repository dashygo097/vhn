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
class LayerNorm;

template <int HIDDEN_DIM> struct LayerNormHParams {
  static constexpr int hidden_dim = HIDDEN_DIM;
};

// ============================================================================
// Non-optimized version (OPT_NONE)
// ============================================================================
template <typename DType, typename HParams>
class LayerNorm<DType, HParams, void, OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int hidden_dim = HParams::hidden_dim;
  static constexpr OptLevel opt_level = OPT_NONE;

  using Gamma_t = dtype[hidden_dim];
  using Beta_t = dtype[hidden_dim];

  LayerNorm() = default;
  ~LayerNorm() = default;

  static void ln(dtype output[hidden_dim], const dtype input[hidden_dim],
                 const Gamma_t gamma, const Beta_t beta,
                 const float epsilon = 1e-5) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    ln_1d_impl(output, input, gamma, beta, epsilon);
  }

  static void ln(dtype output[][hidden_dim], const dtype input[][hidden_dim],
                 const int seq_len, const Gamma_t gamma, const Beta_t beta,
                 const float epsilon = 1e-5) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
  SEQ_LOOP:
    for (int i = 0; i < seq_len; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#endif
      ln_1d_impl(output[i], input[i], gamma, beta, epsilon);
    }
  }

  static void ln(dtype *output, const dtype *input, const int seq_len,
                 const Gamma_t gamma, const Beta_t beta,
                 const float epsilon = 1e-5) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
  SEQ_LOOP:
    for (int i = 0; i < seq_len; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#endif
      ln_1d_impl(&output[i * hidden_dim], &input[i * hidden_dim], gamma, beta,
                 epsilon);
    }
  }

#ifdef __VITIS_HLS__
  static void ln(hls::stream<dtype> &output_stream,
                 hls::stream<dtype> &input_stream, const Gamma_t gamma,
                 const Beta_t beta, const float epsilon = 1e-5) {
#pragma HLS INLINE off
    ln_1d_stream_impl(output_stream, input_stream, gamma, beta, epsilon);
  }

#endif

private:
  static void ln_1d_impl(dtype *output, const dtype *input, const Gamma_t gamma,
                         const Beta_t beta, const float epsilon) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    dtype mean = dtype(0);
  CALC_MEAN:
    for (int j = 0; j < hidden_dim; j++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4096
#pragma HLS BIND_OP variable = mean op = add impl = dsp
#endif
      mean += input[j];
    }
    mean /= dtype(hidden_dim);

    dtype variance = dtype(0);
  CALC_VARIANCE:
    for (int j = 0; j < hidden_dim; j++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4096
#pragma HLS BIND_OP variable = variance op = add impl = dsp
#endif
      dtype diff = input[j] - mean;
      variance += diff * diff;
    }
    variance /= dtype(hidden_dim);

#ifdef __VITIS_HLS__
    dtype inv_std = hls::rsqrt(variance + dtype(epsilon));
#else
    dtype inv_std = dtype(1.0) / std::sqrt(variance + dtype(epsilon));
#endif

  NORMALIZE:
    for (int j = 0; j < hidden_dim; j++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4096
#pragma HLS BIND_OP variable = output op = mul impl = dsp
#endif
      output[j] = gamma[j] * (input[j] - mean) * inv_std + beta[j];
    }
  }

#ifdef __VITIS_HLS__
  static void ln_1d_stream_impl(hls::stream<dtype> &output_stream,
                                hls::stream<dtype> &input_stream,
                                const Gamma_t gamma, const Beta_t beta,
                                const float epsilon) {
#pragma HLS INLINE off
    dtype input_buffer[hidden_dim];

  READ_INPUT:
    for (int j = 0; j < hidden_dim; j++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4096
      input_buffer[j] = input_stream.read();
    }

    dtype mean = dtype(0);
  CALC_MEAN_STREAM:
    for (int j = 0; j < hidden_dim; j++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4096
#pragma HLS BIND_OP variable = mean op = add impl = dsp
      mean += input_buffer[j];
    }
    mean /= dtype(hidden_dim);

    dtype variance = dtype(0);
  CALC_VARIANCE_STREAM:
    for (int j = 0; j < hidden_dim; j++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4096
#pragma HLS BIND_OP variable = variance op = add impl = dsp
      dtype diff = input_buffer[j] - mean;
      variance += diff * diff;
    }
    variance /= dtype(hidden_dim);

    dtype inv_std = hls::rsqrt(variance + dtype(epsilon));

  NORMALIZE_WRITE:
    for (int j = 0; j < hidden_dim; j++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4096
#pragma HLS BIND_OP variable = inv_std op = mul impl = dsp
      dtype output_val =
          gamma[j] * (input_buffer[j] - mean) * inv_std + beta[j];
      output_stream.write(output_val);
    }
  }
#endif
};

template <int PIPELINE_II, int UNROLL_FACTOR, int PARTITION_FACTOR>
struct LayerNormConfig {
  static constexpr int pipeline_ii = PIPELINE_II;
  static constexpr int unroll_factor = UNROLL_FACTOR;
  static constexpr int partition_factor = PARTITION_FACTOR;
};

// ============================================================================
// Optimized version (OPT_ENABLED)
// ============================================================================
template <typename DType, typename HParams, typename Config>
class LayerNorm<DType, HParams, Config, OPT_ENABLED> {
public:
  using dtype = DType;
  static constexpr int hidden_dim = HParams::hidden_dim;
  static constexpr OptLevel opt_level = OPT_ENABLED;

  static constexpr int pipeline_ii = Config::pipeline_ii;
  static constexpr int unroll_factor = Config::unroll_factor;
  static constexpr int partition_factor = Config::partition_factor;

  using Gamma_t = dtype[hidden_dim];
  using Beta_t = dtype[hidden_dim];

  LayerNorm() = default;
  ~LayerNorm() = default;

  static void ln(dtype output[hidden_dim], const dtype input[hidden_dim],
                 const Gamma_t gamma, const Beta_t beta,
                 const float epsilon = 1e-5) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    ln_1d_impl(output, input, gamma, beta, epsilon);
  }

  static void ln(dtype output[][hidden_dim], const dtype input[][hidden_dim],
                 const int seq_len, const Gamma_t gamma, const Beta_t beta,
                 const float epsilon = 1e-5) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
  SEQ_LOOP:
    for (int i = 0; i < seq_len; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      ln_1d_impl(output[i], input[i], gamma, beta, epsilon);
    }
  }

  static void ln(dtype *output, const dtype *input, const int seq_len,
                 const Gamma_t gamma, const Beta_t beta,
                 const float epsilon = 1e-5) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
  SEQ_LOOP:
    for (int i = 0; i < seq_len; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#pragma HLS PIPELINE II = 1
#endif
      ln_1d_impl(&output[i * hidden_dim], &input[i * hidden_dim], gamma, beta,
                 epsilon);
    }
  }

#ifdef __VITIS_HLS__
  static void ln(hls::stream<dtype> &output_stream,
                 hls::stream<dtype> &input_stream, const Gamma_t gamma,
                 const Beta_t beta, const float epsilon = 1e-5) {
#pragma HLS INLINE off
    ln_1d_stream_impl(output_stream, input_stream, gamma, beta, epsilon);
  }

#endif

private:
  static void ln_1d_impl(dtype *output, const dtype *input, const Gamma_t gamma,
                         const Beta_t beta, const float epsilon) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off

    constexpr bool should_partition =
        (partition_factor > 1) && (hidden_dim <= 4096);
    if constexpr (should_partition) {
#pragma HLS ARRAY_PARTITION variable = input type = cyclic factor =            \
    partition_factor
#pragma HLS ARRAY_PARTITION variable = output type = cyclic factor =           \
    partition_factor
#pragma HLS ARRAY_PARTITION variable = gamma type = cyclic factor =            \
    partition_factor
#pragma HLS ARRAY_PARTITION variable = beta type = cyclic factor =             \
    partition_factor
    } else {
#pragma HLS BIND_STORAGE variable = input type = ram_1p impl = bram
#pragma HLS BIND_STORAGE variable = output type = ram_1p impl = bram
#pragma HLS BIND_STORAGE variable = gamma type = rom_1p impl = bram
#pragma HLS BIND_STORAGE variable = beta type = rom_1p impl = bram
    }
#endif

    dtype mean = dtype(0);
  CALC_MEAN:
    for (int i = 0; i < hidden_dim; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#pragma HLS UNROLL factor = unroll_factor
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4096
#pragma HLS BIND_OP variable = mean op = add impl = dsp
#endif
      mean += input[i];
    }
    mean /= dtype(hidden_dim);

    dtype variance = dtype(0);
  CALC_VARIANCE:
    for (int i = 0; i < hidden_dim; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#pragma HLS UNROLL factor = unroll_factor
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4096
#pragma HLS BIND_OP variable = variance op = add impl = dsp
#endif
      dtype diff = input[i] - mean;
      variance += diff * diff;
    }
    variance /= dtype(hidden_dim);

#ifdef __VITIS_HLS__
    dtype inv_std = hls::rsqrt(variance + dtype(epsilon));
#else
    dtype inv_std = dtype(1.0) / std::sqrt(variance + dtype(epsilon));
#endif

  NORMALIZE:
    for (int i = 0; i < hidden_dim; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#pragma HLS UNROLL factor = unroll_factor
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4096
#pragma HLS BIND_OP variable = output op = mul impl = dsp
#endif
      output[i] = gamma[i] * (input[i] - mean) * inv_std + beta[i];
    }
  }

#ifdef __VITIS_HLS__
  static void ln_1d_stream_impl(hls::stream<dtype> &output_stream,
                                hls::stream<dtype> &input_stream,
                                const Gamma_t gamma, const Beta_t beta,
                                const float epsilon) {
#pragma HLS INLINE off

    dtype input_buffer[hidden_dim];

    constexpr bool should_partition =
        (partition_factor > 1) && (hidden_dim <= 4096);
    if constexpr (should_partition) {
#pragma HLS ARRAY_PARTITION variable = input_buffer type = cyclic factor =     \
    partition_factor
    } else {
#pragma HLS BIND_STORAGE variable = input_buffer type = ram_1p impl = bram
    }

#pragma HLS ARRAY_PARTITION variable = input_buffer cyclic factor =            \
    partition_factor

  READ_INPUT:
    for (int j = 0; j < hidden_dim; j++) {
#pragma HLS PIPELINE II = pipeline_ii
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4096
      input_buffer[j] = input_stream.read();
    }

    dtype mean = dtype(0);
  CALC_MEAN_STREAM:
    for (int j = 0; j < hidden_dim; j++) {
#pragma HLS UNROLL factor = unroll_factor
#pragma HLS BIND_OP variable = mean op = add impl = dsp
      mean += input_buffer[j];
    }
    mean /= dtype(hidden_dim);

    dtype variance = dtype(0);
  CALC_VARIANCE_STREAM:
    for (int j = 0; j < hidden_dim; j++) {
#pragma HLS UNROLL factor = unroll_factor
#pragma HLS BIND_OP variable = variance op = add impl = dsp
      dtype diff = input_buffer[j] - mean;
      variance += diff * diff;
    }
    variance /= dtype(hidden_dim);

    dtype inv_std = hls::rsqrt(variance + dtype(epsilon));

  NORMALIZE_WRITE:
    for (int j = 0; j < hidden_dim; j++) {
#pragma HLS PIPELINE II = pipeline_ii
#pragma HLS UNROLL factor = unroll_factor
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4096
#pragma HLS BIND_OP variable = inv_std op = mul impl = dsp
      dtype output_val =
          gamma[j] * (input_buffer[j] - mean) * inv_std + beta[j];
      output_stream.write(output_val);
    }
  }
#endif
};

} // namespace vhn
