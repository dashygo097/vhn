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

template <int D_MODEl> struct PreNormHParams {
  static constexpr int d_model = D_MODEl;
};

// ============================================================================
// PreNorm specialization for OPT_NONE (using Add, LayerNorm)
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

  using add = Add<DType, d_model, void, OPT_NONE>;
  using norm = LayerNorm<DType, d_model, void, OPT_NONE>;

  static void forward(dtype output[d_model], const dtype input[d_model],
                      const dtype residual[d_model], const gamma_t gamma,
                      const beta_t beta) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    dtype normed[d_model];

    norm::forward(normed, input, gamma, beta);
    add::forward(output, residual, normed);
  }

  static void forward(dtype output[][d_model], const dtype input[][d_model],
                      const dtype residual[][d_model], const int actual_len,
                      const gamma_t gamma, const beta_t beta) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif

    dtype normed[d_model];
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
    dtype normed[d_model];

  SEQ_LOOP:
    for (int i = 0; i < actual_len; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      norm::forward(normed, input + i * d_model, gamma, beta);
      add::forward(output + i * d_model, residual + i * d_model, normed);
    }
  }
};
} // namespace vhn
