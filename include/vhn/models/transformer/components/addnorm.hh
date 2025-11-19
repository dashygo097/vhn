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

template <int D_MODEL, NormType NORM_TYPE> struct AddNormHParams {
  static constexpr int d_model = D_MODEL;
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
  static constexpr NormType norm_type = HParams::norm_typ;
  static constexpr OptLevel opt_level = OPT_NONE;

  using gamma_t = dtype[d_model];
  using beta_t = dtype[d_model];

  AddNorm() = default;
  ~AddNorm() = default;

  using addnorm =
      typename std::conditional<norm_type == POSTNORM,
                                PostNorm<DType, d_model, void, OPT_NONE>,
                                PreNorm<DType, d_model, void, OPT_NONE>>::type;

  static void forward(dtype output[d_model], const dtype input[d_model],
                      const dtype residual[d_model], const gamma_t gamma,
                      const beta_t beta) {
    addnorm::forward(output, input, residual, gamma, beta);
  }

  static void forward(dtype output[][d_model], const dtype input[][d_model],
                      const dtype residual[][d_model], const int actual_len,
                      const gamma_t gamma, const beta_t beta) {
    addnorm::forward(output, input, residual, actual_len, gamma, beta);
  }

  static void forward(dtype *output, const dtype *input, const dtype *residual,
                      const int actual_len, const gamma_t gamma,
                      const beta_t beta) {
    addnorm::forward(output, input, residual, actual_len, gamma, beta);
  }

private:
};
} // namespace vhn
