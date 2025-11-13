#pragma once
#include "../../../../opt_level.hh"
#include "../../../layers/linear.hh"
#include "../../../layers/softmax.hh"
#include <cmath>

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace hls_nn {
template <typename WQKV_CONFIG = void, typename SPLIT_CONFIG = void,
          typename ATTN_CONFIG = void, typename SOFTMAX_CONFIG = void,
          typename CONCAT_CONFIG = void, typename WO_CONFIG = void>
struct MHAConfig {
  using wqkv = WQKV_CONFIG;
  using split = SPLIT_CONFIG;
  using softmax = ATTN_CONFIG;
  using attn = ATTN_CONFIG;
  using concat = CONCAT_CONFIG;
  using wo = WO_CONFIG;
};

template <typename DType, const int D_MODEL, const int NUM_HEADS,
          const int MAX_SEQ_LEN, typename Config, OptLevel OPT_LEVEL = OPT_NONE>
class MulHeadAttn;

// ============================================================================
// Non-optimized version (OPT_NONE)
// ============================================================================
template <typename DType, const int D_MODEL, const int NUM_HEADS,
          const int MAX_SEQ_LEN>
class MulHeadAttn<DType, D_MODEL, NUM_HEADS, MAX_SEQ_LEN, void, OPT_NONE> {
public:
  static_assert(D_MODEL % NUM_HEADS == 0,
                "D_MODEL must be divisible by NUM_HEADS");

  using dtype = DType;
  static constexpr int d_model = D_MODEL;
  static constexpr int num_heads = NUM_HEADS;
  static constexpr int head_dim = D_MODEL / NUM_HEADS;
  static constexpr int max_seq_len = MAX_SEQ_LEN;
  static constexpr OptLevel opt_level = OPT_NONE;

  using Wqkv_t = dtype[3 * D_MODEL][D_MODEL];
  using bqkv_t = dtype[3 * D_MODEL];
  using Wo_t = dtype[D_MODEL][D_MODEL];
  using bo_t = dtype[D_MODEL];

  MulHeadAttn() = default;
  ~MulHeadAttn() = default;

  using wqkv = Linear<dtype, D_MODEL, 3 * D_MODEL, void, OPT_NONE>;
  using wo = Linear<dtype, D_MODEL, D_MODEL, void, OPT_NONE>;
  using softmax = Softmax<dtype, MAX_SEQ_LEN, void, OPT_NONE>;

  static void forward(dtype output[][D_MODEL], const dtype input[][D_MODEL],
                      const int actual_len, const Wqkv_t wqkv,
                      const bqkv_t bqkv, const Wo_t wo, const bo_t bo) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    dtype qkv[MAX_SEQ_LEN][3 * D_MODEL];
    wqkv::forward(qkv, input, actual_len, wqkv, bqkv);

    dtype q[MAX_SEQ_LEN][NUM_HEADS][head_dim];
    dtype k[MAX_SEQ_LEN][NUM_HEADS][head_dim];
    dtype v[MAX_SEQ_LEN][NUM_HEADS][head_dim];
    split_qkv(q, k, v, qkv, actual_len);

    dtype attn_output[MAX_SEQ_LEN][NUM_HEADS][head_dim];
    compute_attention(attn_output, q, k, v, actual_len);

    dtype concat[MAX_SEQ_LEN][D_MODEL];
    concat_heads(concat, attn_output, actual_len);

    wo::forward(output, concat, actual_len, wo, bo);
  }

private:
  static void split_qkv(dtype q[][NUM_HEADS][head_dim],
                        dtype k[][NUM_HEADS][head_dim],
                        dtype v[][NUM_HEADS][head_dim],
                        const dtype qkv[][3 * D_MODEL], const int actual_len) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    for (int i = 0; i < actual_len; i++) {
      for (int h = 0; h < NUM_HEADS; h++) {
        for (int d = 0; d < head_dim; d++) {
          q[i][h][d] = qkv[i][h * head_dim + d];
          k[i][h][d] = qkv[i][D_MODEL + h * head_dim + d];
          v[i][h][d] = qkv[i][2 * D_MODEL + h * head_dim + d];
        }
      }
    }
  }

  static void compute_attention(dtype attn_output[][NUM_HEADS][head_dim],
                                const dtype q[][NUM_HEADS][head_dim],
                                const dtype k[][NUM_HEADS][head_dim],
                                const dtype v[][NUM_HEADS][head_dim],
                                const int actual_len) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
#ifdef __VITIS_HLS__

    dtype scale = dtype(1.0) / hls::sqrt(dtype(head_dim));
#else
    dtype scale = dtype(1.0) / sqrt(dtype(head_dim));
#endif

    for (int h = 0; h < NUM_HEADS; h++) {
      dtype scores[MAX_SEQ_LEN][MAX_SEQ_LEN];
      for (int i = 0; i < actual_len; i++) {
        for (int j = 0; j < actual_len; j++) {
          dtype dot = dtype(0.0);
          for (int d = 0; d < head_dim; d++) {
            dot += q[i][h][d] * k[j][h][d];
          }
          scores[i][j] = dot * scale;
        }
      }

      dtype attn_weights[MAX_SEQ_LEN][MAX_SEQ_LEN];
      for (int i = 0; i < actual_len; i++) {
        softmax::forward(attn_weights[i], scores[i]);
      }

      for (int i = 0; i < actual_len; i++) {
        for (int d = 0; d < head_dim; d++) {
          dtype sum = dtype(0.0);
          for (int j = 0; j < actual_len; j++) {
            sum += attn_weights[i][j] * v[j][h][d];
          }
          attn_output[i][h][d] = sum;
        }
      }
    }
  }

  static void concat_heads(dtype concat[][D_MODEL],
                           const dtype attn_output[][NUM_HEADS][head_dim],
                           const int actual_len) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    for (int i = 0; i < actual_len; i++) {
      for (int h = 0; h < NUM_HEADS; h++) {
        for (int d = 0; d < head_dim; d++) {
          concat[i][h * head_dim + d] = attn_output[i][h][d];
        }
      }
    }
  }
};

// ============================================================================
// Optimized version (OPT_ENABLED)
// ============================================================================
template <typename DType, const int D_MODEL, const int NUM_HEADS,
          const int MAX_SEQ_LEN, typename Config>
class MulHeadAttn<DType, D_MODEL, NUM_HEADS, MAX_SEQ_LEN, Config, OPT_ENABLED> {
public:
  static_assert(D_MODEL % NUM_HEADS == 0,
                "D_MODEL must be divisible by NUM_HEADS");

  using dtype = DType;
  static constexpr int d_model = D_MODEL;
  static constexpr int num_heads = NUM_HEADS;
  static constexpr int head_dim = D_MODEL / NUM_HEADS;
  static constexpr int max_seq_len = MAX_SEQ_LEN;
  static constexpr OptLevel opt_level = OPT_ENABLED;

  using Wqkv_t = dtype[3 * D_MODEL][D_MODEL];
  using bqkv_t = dtype[3 * D_MODEL];
  using Wo_t = dtype[D_MODEL][D_MODEL];
  using bo_t = dtype[D_MODEL];

  using wqkv_config = typename Config::wqkv;
  using split_config = typename Config::split;
  using attn_config = typename Config::attn;
  using softmax_config = typename Config::softmax;
  using concat_config = typename Config::concat;
  using wo_config = typename Config::wo;

  static constexpr OptLevel wqkv_opt =
      std::is_same<wqkv_config, void>::value ? OPT_NONE : OPT_ENABLED;
  static constexpr OptLevel split_opt =
      std::is_same<split_config, void>::value ? OPT_NONE : OPT_ENABLED;
  static constexpr OptLevel attn_opt =
      std::is_same<attn_config, void>::value ? OPT_NONE : OPT_ENABLED;
  static constexpr OptLevel softmax_opt =
      std::is_same<softmax_config, void>::value ? OPT_NONE : OPT_ENABLED;
  static constexpr OptLevel concat_opt =
      std::is_same<concat_config, void>::value ? OPT_NONE : OPT_ENABLED;
  static constexpr OptLevel wo_opt =
      std::is_same<wo_config, void>::value ? OPT_NONE : OPT_ENABLED;

  static constexpr int split_unroll_factor =
      split_opt == OPT_ENABLED ? Config::_unroll_factor : -1;
  static constexpr int split_pipeline_ii =
      split_opt == OPT_ENABLED ? Config::_pipeline_ii : -1;
  static constexpr int attn_unroll_factor =
      attn_opt == OPT_ENABLED ? Config::_unroll_factor : -1;
  static constexpr int attn_partition_factor =
      attn_opt == OPT_ENABLED ? Config::_partition_factor : -1;
  static constexpr int attn_pipeline_ii =
      attn_opt == OPT_ENABLED ? Config::_pipeline_ii : -1;
  static constexpr int concat_unroll_factor =
      concat_opt == OPT_ENABLED ? Config::_unroll_factor : -1;
  static constexpr int concat_pipeline_ii =
      concat_opt == OPT_ENABLED ? Config::_pipeline_ii : -1;

  MulHeadAttn() = default;
  ~MulHeadAttn() = default;

  using wqkv = Linear<dtype, D_MODEL, 3 * D_MODEL, wqkv_config, wqkv_opt>;
  using softmax = Softmax<dtype, MAX_SEQ_LEN, softmax_config, softmax_opt>;
  using wo = Linear<dtype, D_MODEL, D_MODEL, wo_config, wo_opt>;

  static void forward(dtype output[][D_MODEL], const dtype input[][D_MODEL],
                      const int actual_len, const Wqkv_t wqkv,
                      const bqkv_t bqkv, const Wo_t wo, const bo_t bo) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
    dtype qkv[MAX_SEQ_LEN][3 * D_MODEL];
    wqkv::forward(qkv, input, actual_len, wqkv, bqkv);

    dtype q[MAX_SEQ_LEN][NUM_HEADS][head_dim];
    dtype k[MAX_SEQ_LEN][NUM_HEADS][head_dim];
    dtype v[MAX_SEQ_LEN][NUM_HEADS][head_dim];
    split_qkv(q, k, v, qkv, actual_len);

    dtype attn_output[MAX_SEQ_LEN][NUM_HEADS][head_dim];
    compute_attention(attn_output, q, k, v, actual_len);

    dtype concat[MAX_SEQ_LEN][D_MODEL];
    concat_heads(concat, attn_output, actual_len);

    wo::forward(output, concat, actual_len, wo, bo);
  }

private:
  static void split_qkv(dtype q[][NUM_HEADS][head_dim],
                        dtype k[][NUM_HEADS][head_dim],
                        dtype v[][NUM_HEADS][head_dim],
                        const dtype qkv[][3 * D_MODEL], const int actual_len) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    for (int i = 0; i < actual_len; i++) {
      if constexpr (split_opt == OPT_ENABLED) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = split_pipeline_ii
#endif
      }
      for (int h = 0; h < NUM_HEADS; h++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = split_unroll_factor
#endif
        for (int d = 0; d < head_dim; d++) {
          q[i][h][d] = qkv[i][h * head_dim + d];
          k[i][h][d] = qkv[i][D_MODEL + h * head_dim + d];
          v[i][h][d] = qkv[i][2 * D_MODEL + h * head_dim + d];
        }
      }
    }
  }

  static void compute_attention(dtype attn_output[][NUM_HEADS][head_dim],
                                const dtype q[][NUM_HEADS][head_dim],
                                const dtype k[][NUM_HEADS][head_dim],
                                const dtype v[][NUM_HEADS][head_dim],
                                const int actual_len) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
#ifdef __VITIS_HLS__
    dtype scale = dtype(1.0) / hls::sqrt(dtype(head_dim));
#else
    dtype scale = dtype(1.0) / sqrt(dtype(head_dim));
#endif

    for (int h = 0; h < NUM_HEADS; h++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = attn_unroll_factor
#endif
      dtype scores[MAX_SEQ_LEN][MAX_SEQ_LEN];
#ifdef __VITIS_HLS__
#pragma HLS ARRAY_PARTITION variable = scores cyclic factor =                  \
    attn_partition_factor dim = 2
#endif

      for (int i = 0; i < actual_len; i++) {
        for (int j = 0; j < actual_len; j++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = attn_pipeline_ii
#endif
          dtype dot = dtype(0.0);
          for (int d = 0; d < head_dim; d++) {
            dot += q[i][h][d] * k[j][h][d];
          }
          scores[i][j] = dot * scale;
        }
      }

      dtype attn_weights[MAX_SEQ_LEN][MAX_SEQ_LEN];
      using SoftmaxLayer =
          Softmax<dtype, MAX_SEQ_LEN, typename Config::softmax, OPT_ENABLED>;
      for (int i = 0; i < actual_len; i++) {
        SoftmaxLayer::forward(attn_weights[i], scores[i]);
      }

      for (int i = 0; i < actual_len; i++) {
        for (int d = 0; d < head_dim; d++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = attn_pipeline_ii
#endif
          dtype sum = dtype(0.0);
          for (int j = 0; j < actual_len; j++) {
            sum += attn_weights[i][j] * v[j][h][d];
          }
          attn_output[i][h][d] = sum;
        }
      }
    }
  }

  static void concat_heads(dtype concat[][D_MODEL],
                           const dtype attn_output[][NUM_HEADS][head_dim],
                           const int actual_len) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    for (int i = 0; i < actual_len; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = concat_pipeline_ii
#endif
      for (int h = 0; h < NUM_HEADS; h++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = concat_unroll_factor
#endif
        for (int d = 0; d < head_dim; d++) {
          concat[i][h * head_dim + d] = attn_output[i][h][d];
        }
      }
    }
  }
};

} // namespace hls_nn
