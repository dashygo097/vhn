#pragma once

#include "../../../norms/layernorm.hh"
#include "../../../operators/operators.hh"
#include "../../../opt_level.hh"

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace vhn {
template <typename DType, typename HParams, typename Config, OptLevel OPT_LEVEL>
class PreNorm;

template <typename NORM_HParams> struct PreNormHParams {
  using norm_hparams = NORM_HParams;
  static constexpr int d_model = norm_hparams::hidden_dim;
};

// ============================================================================
// PreNorm specialization for OPT_NONE
// ============================================================================
template <typename DType, typename HParams, typename Config>
class PreNorm<DType, HParams, Config, OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int d_model = HParams::d_model;
  static constexpr OptLevel opt_level = OPT_NONE;

  using gamma_t = dtype[d_model];
  using beta_t = dtype[d_model];

  PreNorm() = default;
  ~PreNorm() = default;

  using norm_hparams = typename HParams::norm_hparams;

  using add = Add<DType, d_model, void, OPT_NONE>;
  using norm = LayerNorm<DType, norm_hparams, void, OPT_NONE>;

  static void addnorm(dtype output[d_model], const dtype input[d_model],
                      const dtype residual[d_model], const gamma_t gamma,
                      const beta_t beta) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    dtype normed[d_model];

    norm::ln(normed, input, gamma, beta);
    add::elem(output, residual, normed);
  }

  static void addnorm(dtype output[][d_model], const dtype input[][d_model],
                      const dtype residual[][d_model], const int actual_len,
                      const gamma_t gamma, const beta_t beta) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif

    dtype normed[d_model];
  SEQ_LOOP:
    for (int i = 0; i < actual_len; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#endif
      norm::ln(normed, input[i], gamma, beta);
      add::elem(output[i], residual[i], normed);
    }
  }

  static void addnorm(dtype *output, const dtype *input, const dtype *residual,
                      const int actual_len, const gamma_t gamma,
                      const beta_t beta) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    dtype normed[d_model];

  SEQ_LOOP:
    for (int i = 0; i < actual_len; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#endif
      norm::ln(normed, input + i * d_model, gamma, beta);
      add::elem(output + i * d_model, residual + i * d_model, normed);
    }
  }
};

// ============================================================================
// PreNorm specialization for OPT_ENABLED
// ============================================================================
template <typename DType, typename HParams, typename Config>
class PreNorm<DType, HParams, Config, OPT_ENABLED> {
public:
  using dtype = DType;
  static constexpr int d_model = HParams::d_model;
  static constexpr OptLevel opt_level = OPT_ENABLED;
  static constexpr int memory_partition = Config::memory_partition;

  using gamma_t = dtype[d_model];
  using beta_t = dtype[d_model];

  PreNorm() = default;
  ~PreNorm() = default;

  using norm_hparams = typename HParams::norm_hparams;
  using norm_config = typename Config::norm_config;
  using add_config = typename Config::add_config;

  static constexpr bool norm_is_optimized =
      !std::is_same<norm_config, void>::value;
  static constexpr bool add_is_optimized =
      !std::is_same<add_config, void>::value;

  using add = Add<DType, d_model, add_config,
                  add_is_optimized ? OPT_ENABLED : OPT_NONE>;
  using norm = LayerNorm<DType, norm_hparams, norm_config,
                         norm_is_optimized ? OPT_ENABLED : OPT_NONE>;

  static void addnorm(dtype output[d_model], const dtype input[d_model],
                      const dtype residual[d_model], const gamma_t gamma,
                      const beta_t beta) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS PIPELINE II = 1
#endif
    dtype normed[d_model];
    norm::ln(normed, input, gamma, beta);
    add::elem(output, residual, normed);
  }

  static void addnorm(dtype output[][d_model], const dtype input[][d_model],
                      const dtype residual[][d_model], const int actual_len,
                      const gamma_t gamma, const beta_t beta) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = gamma type = cyclic factor =            \
    memory_partition dim = 1
#pragma HLS ARRAY_PARTITION variable = beta type = cyclic factor =             \
    memory_partition dim = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#endif
    dtype normed[d_model];

#ifdef __VITIS_HLS__
    constexpr int part_factor = (memory_partition > 1) ? memory_partition : 4;
    if constexpr (d_model > 512) {
#pragma HLS ARRAY_PARTITION variable = normed type = cyclic factor =           \
    part_factor dim = 1
    }
#endif

  SEQ_LOOP:
    for (int i = 0; i < actual_len; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#endif
      norm::ln(normed, input[i], gamma, beta);
      add::elem(output[i], residual[i], normed);
    }
  }

  static void addnorm(dtype *output, const dtype *input, const dtype *residual,
                      const int actual_len, const gamma_t gamma,
                      const beta_t beta) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = gamma type = cyclic factor =            \
    memory_partition dim = 1
#pragma HLS ARRAY_PARTITION variable = beta type = cyclic factor =             \
    memory_partition dim = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#endif
    dtype normed[d_model];

#ifdef __VITIS_HLS__
    constexpr int part_factor = (memory_partition > 1) ? memory_partition : 4;
    if constexpr (d_model > 512) {
#pragma HLS ARRAY_PARTITION variable = normed type = cyclic factor =           \
    part_factor dim = 1
    }
#endif

  SEQ_LOOP:
    for (int i = 0; i < actual_len; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#endif
      norm::ln(normed, input + i * d_model, gamma, beta);
      add::elem(output + i * d_model, residual + i * d_model, normed);
    }
  }
};

} // namespace vhn
