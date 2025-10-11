#pragma once

#include "../../opt_level.hh"
#include <cmath>

#ifdef __VITIS_HLS__
#include <hls_math.h>
#include <hls_stream.h>
#endif

namespace hls_nn {

template <typename DType, const int HIDDEN_DIM, typename Config = void,
          OptLevel OPT_LEVEL = OPT_NONE>
class LayerNorm;

// ============================================================================
// Non-optimized version (OPT_NONE)
// ============================================================================
template <typename DType, const int HIDDEN_DIM>
class LayerNorm<DType, HIDDEN_DIM, void, OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int hidden_dim = HIDDEN_DIM;
  static constexpr OptLevel opt_level = OPT_NONE;

  using Gamma_t = dtype[HIDDEN_DIM];
  using Beta_t = dtype[HIDDEN_DIM];

  LayerNorm() = default;
  ~LayerNorm() = default;

  static void forward(dtype output[HIDDEN_DIM], const dtype input[HIDDEN_DIM],
                      const Gamma_t gamma, const Beta_t beta,
                      const float epsilon = 1e-5) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    forward_1d_impl(output, input, gamma, beta, epsilon);
  }

  static void forward(dtype output[][HIDDEN_DIM],
                      const dtype input[][HIDDEN_DIM], const int seq_len,
                      const Gamma_t gamma, const Beta_t beta,
                      const float epsilon = 1e-5) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
  SEQ_LOOP:
    for (int i = 0; i < seq_len; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_1d_impl(output[i], input[i], gamma, beta, epsilon);
    }
  }

  static void forward(dtype *output, const dtype *input, const int seq_len,
                      const Gamma_t gamma, const Beta_t beta,
                      const float epsilon = 1e-5) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
  SEQ_LOOP:
    for (int i = 0; i < seq_len; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_1d_impl(&output[i * HIDDEN_DIM], &input[i * HIDDEN_DIM], gamma,
                      beta, epsilon);
    }
  }

private:
  static void forward_1d_impl(dtype *output, const dtype *input,
                              const Gamma_t gamma, const Beta_t beta,
                              const float epsilon) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    dtype mean = dtype(0);
  CALC_MEAN:
    for (int j = 0; j < hidden_dim; j++) {
      mean += input[j];
    }
    mean /= dtype(hidden_dim);

    dtype variance = dtype(0);
  CALC_VARIANCE:
    for (int j = 0; j < hidden_dim; j++) {
      dtype diff = input[j] - mean;
      variance += diff * diff;
    }
    variance /= dtype(hidden_dim);

#ifdef __VITIS_HLS__
    dtype inv_std = hls::rsqrt(variance + dtype(epsilon));
#else
    dtype inv_std = dtype(1.0) / std::sqrt(variance + dtype(epsilon));
#endif

    // Normalize and apply affine transformation
  NORMALIZE:
    for (int j = 0; j < hidden_dim; j++) {
      output[j] = gamma[j] * (input[j] - mean) * inv_std + beta[j];
    }
  }
};

// ============================================================================
// Optimized version (OPT_ENABLED)
// ============================================================================
template <typename DType, const int HIDDEN_DIM, typename Config>
class LayerNorm<DType, HIDDEN_DIM, Config, OPT_ENABLED> {
public:
  using dtype = DType;
  static constexpr int hidden_dim = HIDDEN_DIM;
  static constexpr OptLevel opt_level = OPT_ENABLED;

  static constexpr int unroll_factor = Config::_unroll_factor;
  static constexpr int partition_factor = Config::_partition_factor;
  static constexpr int pipeline_ii = Config::_pipeline_ii;

  using Gamma_t = dtype[HIDDEN_DIM];
  using Beta_t = dtype[HIDDEN_DIM];

  LayerNorm() = default;
  ~LayerNorm() = default;

  static void forward(dtype output[HIDDEN_DIM], const dtype input[HIDDEN_DIM],
                      const Gamma_t gamma, const Beta_t beta,
                      const float epsilon = 1e-5) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    forward_1d_impl(output, input, gamma, beta, epsilon);
  }

  static void forward(dtype output[][HIDDEN_DIM],
                      const dtype input[][HIDDEN_DIM], const int seq_len,
                      const Gamma_t gamma, const Beta_t beta,
                      const float epsilon = 1e-5) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
  SEQ_LOOP:
    for (int i = 0; i < seq_len; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_1d_impl(output[i], input[i], gamma, beta, epsilon);
    }
  }

  static void forward(dtype *output, const dtype *input, const int seq_len,
                      const Gamma_t gamma, const Beta_t beta,
                      const float epsilon = 1e-5) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
  SEQ_LOOP:
    for (int i = 0; i < seq_len; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_1d_impl(&output[i * HIDDEN_DIM], &input[i * HIDDEN_DIM], gamma,
                      beta, epsilon);
    }
  }

private:
  static void forward_1d_impl(dtype *output, const dtype *input,
                              const Gamma_t gamma, const Beta_t beta,
                              const float epsilon) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = input cyclic factor = partition_factor
#pragma HLS ARRAY_PARTITION variable = output cyclic factor = partition_factor
#pragma HLS ARRAY_PARTITION variable = gamma cyclic factor = partition_factor
#pragma HLS ARRAY_PARTITION variable = beta cyclic factor = partition_factor
#endif

    dtype mean = dtype(0);
  CALC_MEAN:
    for (int i = 0; i < hidden_dim; i++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = unroll_factor
#pragma HLS PIPELINE II = pipeline_ii
#endif
      mean += input[i];
    }
    mean /= dtype(hidden_dim);

    dtype variance = dtype(0);
  CALC_VARIANCE:
    for (int i = 0; i < hidden_dim; i++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = unroll_factor
#pragma HLS PIPELINE II = pipeline_ii
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
#pragma HLS UNROLL factor = unroll_factor
#pragma HLS PIPELINE II = pipeline_ii
#endif
      output[i] = gamma[i] * (input[i] - mean) * inv_std + beta[i];
    }
  }
};

} // namespace hls_nn
