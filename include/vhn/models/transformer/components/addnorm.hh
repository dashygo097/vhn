#pragma once

#include "./postnorm.hh"
#include "./prenorm.hh"

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

enum NormType { POSTNORM, PRENORM };

namespace vhn {
template <typename ADD_CONFIG = void, typename NORM_CONFIG = void>
struct AddNormConfig {
  using add = ADD_CONFIG;
  using norm = NORM_CONFIG;
};

template <typename DType, const int D_MODEL, NormType NORM_TYPE,
          typename Config = AddNormConfig<>, OptLevel OPT_LEVEL = OPT_NONE>
class AddNorm;

// ============================================================================
// AddNorm specialization for OPT_NONE (using PreNorm, PostNorm)
// ============================================================================
template <typename DType, const int D_MODEL, NormType NORM_TYPE>
class AddNorm<DType, D_MODEL, NORM_TYPE, void, OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int d_model = D_MODEL;
  static constexpr NormType norm_type = NORM_TYPE;
  static constexpr OptLevel opt_level = OPT_NONE;

  using gamma_t = dtype[D_MODEL];
  using beta_t = dtype[D_MODEL];

  AddNorm() = default;
  ~AddNorm() = default;

  using addnorm =
      typename std::conditional<NORM_TYPE == POSTNORM,
                                PostNorm<DType, D_MODEL, void, OPT_NONE>,
                                PreNorm<DType, D_MODEL, void, OPT_NONE>>::type;

  static void forward(dtype output[D_MODEL], const dtype input[D_MODEL],
                      const dtype residual[D_MODEL], const gamma_t gamma,
                      const beta_t beta) {
    addnorm::forward(output, input, residual, gamma, beta);
  }

  static void forward(dtype output[][D_MODEL], const dtype input[][D_MODEL],
                      const dtype residual[][D_MODEL], const int actual_len,
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
