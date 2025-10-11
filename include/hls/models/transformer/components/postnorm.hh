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

template <typename PostNormConfig> struct PostNormMLPConfig {
  using norm = typename PostNormConfig::norm;
};

template <typename DType, const int D_MODEL, typename Config = PostNormConfig<>,
          OptLevel OPT_LEVEL = OPT_NONE>
class PostNorm;

template <typename DType, const int D_MODEL, typename Config>
class PostNorm<DType, D_MODEL, Config, OPT_NONE> {
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
#pragma HLS INLINE off
#endif
    dtype sum[D_MODEL];

  SEQ_LOOP:
    for (int i = 0; i < actual_len; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE II = 1
#endif
    ADD_LOOP:
      for (int j = 0; j < D_MODEL; j++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = 4
#endif
        sum[j] = input[i][j] + residual[i][j];
      }
      norm::forward(output[i], sum, gamma, beta, actual_len);
    }
  }

  // 单样本处理接口
  static void forward(dtype output[D_MODEL], const dtype input[D_MODEL],
                      const dtype residual[D_MODEL], const gamma_t gamma,
                      const beta_t beta, const int actual_len = 1) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    dtype sum[D_MODEL];

  ADD_LOOP:
    for (int j = 0; j < D_MODEL; j++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = 4
#endif
      sum[j] = input[j] + residual[j];
    }
    norm::forward(output, sum, gamma, beta, actual_len);
  }

  // 指针接口
  static void forward(dtype *output, const dtype *input, const dtype *residual,
                      const int actual_len, const gamma_t gamma,
                      const beta_t beta) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    dtype sum[D_MODEL];

  SEQ_LOOP:
    for (int i = 0; i < actual_len; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE II = 1
#endif
    ADD_LOOP:
      for (int j = 0; j < D_MODEL; j++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = 4
#endif
        sum[j] = input[i * D_MODEL + j] + residual[i * D_MODEL + j];
      }
      norm::forward(output + i * D_MODEL, sum, gamma, beta, actual_len);
    }
  }
};

// ============================================================================
// PostNorm 特化：OPT_ENABLED (支持配置)
// ============================================================================
template <typename DType, const int D_MODEL, typename Config>
class PostNorm<DType, D_MODEL, Config, OPT_ENABLED> {
public:
  using dtype = DType;
  static constexpr int d_model = D_MODEL;
  static constexpr OptLevel opt_level = OPT_ENABLED;

  using gamma_t = dtype[D_MODEL];
  using beta_t = dtype[D_MODEL];

  using norm_config = typename Config::norm;

  // 根据配置决定 LayerNorm 的优化级别
  static constexpr OptLevel norm_opt =
      std::is_same<norm_config, void>::value ? OPT_NONE : OPT_ENABLED;

  PostNorm() = default;
  ~PostNorm() = default;

  // 使用配置的 LayerNorm
  using norm = LayerNorm<DType, D_MODEL, norm_config, norm_opt>;

  // 批量处理接口
  static void forward(dtype output[][D_MODEL], const dtype input[][D_MODEL],
                      const dtype residual[][D_MODEL], const int actual_len,
                      const gamma_t gamma, const beta_t beta) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    dtype sum[D_MODEL];

  SEQ_LOOP:
    for (int i = 0; i < actual_len; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE II = 1
#endif
    ADD_LOOP:
      for (int j = 0; j < D_MODEL; j++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = 8
#endif
        sum[j] = input[i][j] + residual[i][j];
      }
      norm::forward(output[i], sum, gamma, beta, actual_len);
    }
  }

  // 单样本处理接口
  static void forward(dtype output[D_MODEL], const dtype input[D_MODEL],
                      const dtype residual[D_MODEL], const gamma_t gamma,
                      const beta_t beta, const int actual_len = 1) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    dtype sum[D_MODEL];

  ADD_LOOP:
    for (int j = 0; j < D_MODEL; j++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = 8
#endif
      sum[j] = input[j] + residual[j];
    }
    norm::forward(output, sum, gamma, beta, actual_len);
  }

  // 指针接口
  static void forward(dtype *output, const dtype *input, const dtype *residual,
                      const int actual_len, const gamma_t gamma,
                      const beta_t beta) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    dtype sum[D_MODEL];

  SEQ_LOOP:
    for (int i = 0; i < actual_len; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE II = 1
#endif
    ADD_LOOP:
      for (int j = 0; j < D_MODEL; j++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = 8
#endif
        sum[j] = input[i * D_MODEL + j] + residual[i * D_MODEL + j];
      }
      norm::forward(output + i * D_MODEL, sum, gamma, beta, actual_len);
    }
  }
};

} // namespace hls_nn
