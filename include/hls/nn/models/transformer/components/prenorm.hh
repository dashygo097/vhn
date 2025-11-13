#pragma once

#include "../../../../opt_level.hh"
#include "../../../norms/layernorm.hh"
#include "../../../operators.hh"

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace hls_nn {
template <typename ADD_CONFIG = void, typename NORM_CONFIG = void>
struct PreNormConfig {
  using add = ADD_CONFIG;
  using norm = NORM_CONFIG;
};

template <typename PreNormConfig> struct PreNorm_AddConfig {
  using add = typename PreNormConfig::add;
};

template <typename PreNormConfig> struct PreNorm_LayerNormConfig {
  using norm = typename PreNormConfig::norm;
};

template <typename DType, const int D_MODEL, typename Config = PreNormConfig<>,
          OptLevel OPT_LEVEL = OPT_NONE>
class PreNorm;

// ============================================================================
// PreNorm specialization for OPT_NONE (using Add, LayerNorm)
// ============================================================================
template <typename DType, const int D_MODEL, typename Config>
class PreNorm<DType, D_MODEL, Config, OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int d_model = D_MODEL;
  static constexpr OptLevel opt_level = OPT_NONE;

  using gamma_t = dtype[D_MODEL];
  using beta_t = dtype[D_MODEL];

  PreNorm() = default;
  ~PreNorm() = default;

  using add = Add<DType, D_MODEL, void, OPT_NONE>;
  using norm = LayerNorm<DType, D_MODEL, void, OPT_NONE>;

  static void forward(dtype output[D_MODEL], const dtype input[D_MODEL],
                      const dtype residual[D_MODEL], const gamma_t gamma,
                      const beta_t beta) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    dtype normed[D_MODEL];

    norm::forward(normed, input, gamma, beta);
    add::forward(output, residual, normed);
  }

  static void forward(dtype output[][D_MODEL], const dtype input[][D_MODEL],
                      const dtype residual[][D_MODEL], const int actual_len,
                      const gamma_t gamma, const beta_t beta) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif

    dtype normed[D_MODEL];
  SEQ_LOOP:
    for (int i = 0; i < actual_len; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      norm::forward(normed, input[i], gamma, beta);
      add::forward(output[i], residual[i], normed);
    }
  }

  static void forward(dtype *output, const dtype *input, const dtype *residual,
                      const int actual_len, const gamma_t gamma,
                      const beta_t beta) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
    dtype normed[D_MODEL];

  SEQ_LOOP:
    for (int i = 0; i < actual_len; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      norm::forward(normed, input + i * D_MODEL, gamma, beta);
      add::forward(output + i * D_MODEL, residual + i * D_MODEL, normed);
    }
  }
};

// ============================================================================
// PreNorm specialization for OPT_ENABLED with Config (using Add, LayerNorm)
// ============================================================================
template <typename DType, const int D_MODEL, typename Config>
class PreNorm<DType, D_MODEL, Config, OPT_ENABLED> {
public:
  using dtype = DType;
  static constexpr int d_model = D_MODEL;
  static constexpr OptLevel opt_level = OPT_ENABLED;

  using gamma_t = dtype[D_MODEL];
  using beta_t = dtype[D_MODEL];

  using add_config = typename Config::add;
  using norm_config = typename Config::norm;

  static constexpr OptLevel norm_opt =
      std::is_same<norm_config, void>::value ? OPT_NONE : OPT_ENABLED;

  PreNorm() = default;
  ~PreNorm() = default;

  using add = Add<DType, D_MODEL, add_config, OPT_ENABLED>;
  using norm = LayerNorm<DType, D_MODEL, norm_config, norm_opt>;

  static void forward(dtype output[D_MODEL], const dtype input[D_MODEL],
                      const dtype residual[D_MODEL], const gamma_t gamma,
                      const beta_t beta) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    dtype normed[D_MODEL];

    norm::forward(normed, input, gamma, beta);
    add::forward(output, residual, normed);
  }

  static void forward(dtype output[][D_MODEL], const dtype input[][D_MODEL],
                      const dtype residual[][D_MODEL], const int actual_len,
                      const gamma_t gamma, const beta_t beta) {
#ifdef __VITIS_HLS__
#pragma HLS DATAFLOW
#pragma HLS INLINE off
#endif
    dtype normed[D_MODEL];

  SEQ_LOOP:
    for (int i = 0; i < actual_len; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      norm::forward(normed, input[i], gamma, beta);
      add::forward(output[i], residual[i], normed);
    }
  }

  static void forward(dtype *output, const dtype *input, const dtype *residual,
                      const int actual_len, const gamma_t gamma,
                      const beta_t beta) {
#ifdef __VITIS_HLS__
#pragma HLS DATAFLOW
#pragma HLS INLINE off
#endif
    dtype normed[D_MODEL];

  SEQ_LOOP:
    for (int i = 0; i < actual_len; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      norm::forward(normed, input + i * D_MODEL, gamma, beta);
      add::forward(output + i * D_MODEL, residual + i * D_MODEL, normed);
    }
  }
};

} // namespace hls_nn
