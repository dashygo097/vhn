#pragma once

#include "../../../nn/nn.hh"
#include "../../../opt_level.hh"

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace hls_nn {
template <typename NORM_CONFIG = void> struct PostNormConfig {
  using norm = NORM_CONFIG;
};

template <typename DType, const int D_MODEL, typename Config = PostNormConfig<>,
          OptLevel OPT_LEVEL = OPT_NONE>
class PostNorm {
public:
  using dtype = DType;
  static constexpr int d_model = D_MODEL;

  using gamma_t = dtype[D_MODEL];
  using beta_t = dtype[D_MODEL];

  PostNorm() = default;
  ~PostNorm() = default;

private:
};

template <typename DType, const int D_MODEL>
class PostNorm<DType, D_MODEL, void, OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int d_model = D_MODEL;
  static constexpr OptLevel opt_level = OPT_NONE;

  using gamma_t = dtype[D_MODEL];
  using beta_t = dtype[D_MODEL];

  PostNorm() = default;
  ~PostNorm() = default;

  using norm = LayerNorm<DType, D_MODEL, void, OPT_NONE>;

  static void forward(dtype output[][D_MODEL], const dtype input[][D_MODEL],
                      const dtype residual[][D_MODEL], const int actual_len,
                      const gamma_t gamma, const beta_t beta) {
#ifdef __VITIS_HLS__
#pragma HLS DATAFLOW
#endif
    dtype sum[D_MODEL];
  SEQ_LOOP:
    for (int i = 0; i < actual_len; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
    ADD_LOOP:
      for (int j = 0; j < D_MODEL; j++) {
        sum[j] = input[i][j] + residual[i][j];
      }
      norm::forward(output[i], sum, gamma, beta, actual_len);
    }
  }

private:
};

template <typename DType, const int D_MODEL, typename Config>
class PostNorm<DType, D_MODEL, Config, OPT_ENABLED> {
public:
  using dtype = DType;
  static constexpr int d_model = D_MODEL;
  static constexpr OptLevel opt_level = OPT_ENABLED;

  using gamma_t = dtype[D_MODEL];
  using beta_t = dtype[D_MODEL];

  using norm_config = typename Config::norm;

  static constexpr OptLevel norm_opt =
      std::is_same<norm_config, void>::value ? OPT_NONE : OPT_ENABLED;

  PostNorm() = default;
  ~PostNorm() = default;

  using norm = LayerNorm<DType, D_MODEL, norm_config, norm_opt>;

  static void forward(dtype output[][D_MODEL], const dtype input[][D_MODEL],
                      const dtype residual[][D_MODEL], const int actual_len,
                      const gamma_t gamma, const beta_t beta) {
#ifdef __VITIS_HLS__
#pragma HLS DATAFLOW
#endif

    dtype sum[D_MODEL];
  SEQ_LOOP:
    for (int i = 0; i < actual_len; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
    ADD_LOOP:
      for (int j = 0; j < D_MODEL; j++) {
        sum[j] = input[i][j] + residual[i][j];
      }
      norm::forward(output[i], sum, gamma, beta, actual_len);
    }
  }

private:
};
} // namespace hls_nn
