#pragma once

#include "../../../norms/layernorm.hh"
#include "../../../operators/operators.hh"
#include "../../../opt_level.hh"

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace vhn {
template <typename DType, typename HParams, typename Config, OptLevel OPT_LEVEL>
class PostNorm;

template <typename NORM_HParams> struct PostNormHParams {
  using norm_hparams = NORM_HParams;

  static constexpr int d_model = norm_hparams::hidden_dim;
};

// ============================================================================
// PostNorm specialization for OPT_NONE (using Add, LayerNorm)
// ============================================================================
template <typename DType, typename HParams, typename Config>
class PostNorm<DType, HParams, Config, OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int d_model = HParams::d_model;
  static constexpr OptLevel opt_level = OPT_NONE;

  using gamma_t = dtype[d_model];
  using beta_t = dtype[d_model];

  PostNorm() = default;
  ~PostNorm() = default;

  using norm_hparams = typename HParams::norm_hparams;

  using add = Add<DType, d_model, void, OPT_NONE>;
  using norm = LayerNorm<DType, norm_hparams, void, OPT_NONE>;

  static void addnorm(dtype output[d_model], const dtype input[d_model],
                      const dtype residual[d_model], const gamma_t gamma,
                      const beta_t beta) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    dtype sum[d_model];

    add::elem(sum, input, residual);
    norm::ln(output, sum, gamma, beta);
  }

  static void addnorm(dtype output[][d_model], const dtype input[][d_model],
                      const dtype residual[][d_model], const int actual_len,
                      const gamma_t gamma, const beta_t beta) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif

    dtype sum[d_model];
  SEQ_LOOP:
    for (int i = 0; i < actual_len; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      add::elem(sum, input[i], residual[i], actual_len);
      norm::ln(output[i], sum, gamma, beta, actual_len);
    }
  }

  static void addnorm(dtype *output, const dtype *input, const dtype *residual,
                      const int actual_len, const gamma_t gamma,
                      const beta_t beta) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    dtype sum[d_model];

  SEQ_LOOP:
    for (int i = 0; i < actual_len; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      add::elem(sum, input + i * d_model, residual + i * d_model, actual_len);
      norm::ln(output + i * d_model, sum, gamma, beta, actual_len);
    }
  }
};
} // namespace vhn
