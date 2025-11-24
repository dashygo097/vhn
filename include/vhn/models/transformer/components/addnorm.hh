#pragma once

#include "./postnorm.hh"
#include "./prenorm.hh"

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

enum NormType { POSTNORM, PRENORM };

namespace vhn {
template <typename DType, typename HParams, typename Config, OptLevel OPT_LEVEL>
class AddNorm;

template <typename NORM_HParams, NormType NORM_TYPE> struct AddNormHParams {
  using norm_hparams = NORM_HParams;

  static constexpr int d_model = NORM_HParams::hidden_dim;
  static constexpr NormType norm_type = NORM_TYPE;
};

// ============================================================================
// AddNorm specialization for OPT_NONE (using PreNorm, PostNorm)
// ============================================================================
template <typename DType, typename HParams>
class AddNorm<DType, HParams, void, OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int d_model = HParams::d_model;
  static constexpr NormType norm_type = HParams::norm_type;
  static constexpr OptLevel opt_level = OPT_NONE;

  using gamma_t = dtype[d_model];
  using beta_t = dtype[d_model];

  AddNorm() = default;
  ~AddNorm() = default;

  using addnorm =
      typename std::conditional<norm_type == POSTNORM,
                                PostNorm<DType, HParams, void, OPT_NONE>,
                                PreNorm<DType, HParams, void, OPT_NONE>>::type;

  static void forward(dtype output[d_model], const dtype input[d_model],
                      const dtype residual[d_model], const gamma_t gamma,
                      const beta_t beta) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    addnorm::addnorm(output, input, residual, gamma, beta);
  }

  static void forward(dtype output[][d_model], const dtype input[][d_model],
                      const dtype residual[][d_model], const int actual_len,
                      const gamma_t gamma, const beta_t beta) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    addnorm::addnorm(output, input, residual, actual_len, gamma, beta);
  }

  static void forward(dtype *output, const dtype *input, const dtype *residual,
                      const int actual_len, const gamma_t gamma,
                      const beta_t beta) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#endif
    addnorm::addnorm(output, input, residual, actual_len, gamma, beta);
  }
};

template <typename NORM_CONFIG, bool DATAFLOW_ENABLED, int PIPELINE_II,
          int PARTITION_FACTOR>
struct AddNormConfig {
  using norm_config = NORM_CONFIG;

  static constexpr bool dataflow_enabled = DATAFLOW_ENABLED;
  static constexpr int pipeline_ii = PIPELINE_II;
  static constexpr int partition_factor = PARTITION_FACTOR;
};

// ============================================================================
// AddNorm specialization for OPT_ENABLED
// ============================================================================
template <typename DType, typename HParams, typename Config>
class AddNorm<DType, HParams, Config, OPT_ENABLED> {
public:
  using dtype = DType;
  static constexpr int d_model = HParams::d_model;
  static constexpr NormType norm_type = HParams::norm_type;
  static constexpr OptLevel opt_level = OPT_ENABLED;

  static constexpr int partition_factor = Config::partition_factor;
  static constexpr int pipeline_ii = Config::pipeline_ii;
  static constexpr int dataflow_enabled = Config::dataflow_enabled;

  using norm_config = typename Config::norm_config;

  using gamma_t = dtype[d_model];
  using beta_t = dtype[d_model];

  AddNorm() = default;
  ~AddNorm() = default;

  static constexpr bool is_norm_optimized =
      !std::is_same<norm_config, void>::value;

  using addnorm = typename std::conditional<
      norm_type == POSTNORM,
      PostNorm<DType, HParams, norm_config,
               is_norm_optimized ? OPT_ENABLED : OPT_NONE>,
      PreNorm<DType, HParams, norm_config,
              is_norm_optimized ? OPT_ENABLED : OPT_NONE>>::type;

  static void forward(dtype output[d_model], const dtype input[d_model],
                      const dtype residual[d_model], const gamma_t gamma,
                      const beta_t beta) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS PIPELINE II = 1
#endif
    addnorm::addnorm(output, input, residual, gamma, beta);
  }

  static void forward(dtype output[][d_model], const dtype input[][d_model],
                      const dtype residual[][d_model], const int actual_len,
                      const gamma_t gamma, const beta_t beta) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = gamma type = cyclic factor =            \
    partition_factor
#pragma HLS ARRAY_PARTITION variable = beta type = cyclic factor =             \
    partition_factor
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#endif
    addnorm::addnorm(output, input, residual, actual_len, gamma, beta);
  }

  static void forward(dtype *output, const dtype *input, const dtype *residual,
                      const int actual_len, const gamma_t gamma,
                      const beta_t beta) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = gamma type = cyclic factor =            \
    partition_factor
#pragma HLS ARRAY_PARTITION variable = beta type = cyclic factor =             \
    partition_factor
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#endif
    addnorm::addnorm(output, input, residual, actual_len, gamma, beta);
  }
};

} // namespace vhn
