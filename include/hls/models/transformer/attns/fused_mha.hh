#pragma once

#include "../../../opt_level.hh"
#include <cmath>

#ifdef __VITIS_HLS__
#include <hls_math.h>
#include <hls_stream.h>
#endif

namespace hls_nn {
template <typename DType, const int D_MODEL, const int NUM_HEADS,
          typename Config, OptLevel OPT_LEVEL = OPT_NONE>
class FusedMHA;

template <typename DType, const int D_MODEL, const int NUM_HEADS>
class FusedMHA<DType, D_MODEL, NUM_HEADS, void, OPT_NONE> {
public:
  static_assert(D_MODEL % NUM_HEADS == 0,
                "D_MODEL must be divisible by NUM_HEADS");
  using dtype = DType;
  static constexpr int d_model = D_MODEL;
  static constexpr int num_heads = NUM_HEADS;
  static constexpr int head_dim = D_MODEL / NUM_HEADS;
  static constexpr OptLevel opt_level = OPT_NONE;

  using Wqkv_t = dtype[3 * D_MODEL][D_MODEL];
  using bqkv_t = dtype[3 * D_MODEL];
  using Wo_t = dtype[D_MODEL][D_MODEL];
  using bo_t = dtype[D_MODEL];

  FusedMHA() = default;
  ~FusedMHA() = default;

  static void qkv(dtype Q[][D_MODEL], dtype K[][D_MODEL], dtype V[][D_MODEL],
                  const dtype input[][D_MODEL], const int actual_len,
                  const Wqkv_t Wqkv, const bqkv_t bqkv) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
  QKV_OUTER_LOOP:
    for (int i = 0; i < actual_len; i++) {
    QKV_INNER_LOOP:
      for (int j = 0; j < 3 * d_model; j++) {
        dtype sum = dtype(0.0f);
      QKV_DOT_PRODUCT:
        for (int k = 0; k < d_model; k++) {
          sum += input[i][j] * Wqkv[j][k];
        }
        sum += bqkv[j];
        if (j < d_model) {
          Q[i][j] = sum;
        } else if (j < 2 * d_model) {
          K[i][j - d_model] = sum;
        } else {
          V[i][j - 2 * d_model] = sum;
        }
      }
    }
  }

  static void forward(dtype output[][D_MODEL], const dtype input[][D_MODEL],
                      const dtype mask[], const int actual_len,
                      const Wqkv_t Wqkv, const bqkv_t bqkv, const Wo_t Wo,
                      const bo_t bo) {
    dtype heads[num_heads][actual_len][head_dim];
    dtype attn_scores[num_heads][actual_len][actual_len];
    dtype attn_output[actual_len][d_model];
    dtype max_exp_score[num_heads][actual_len];
    dtype Q[actual_len][d_model];
    dtype K[actual_len][d_model];
    dtype V[actual_len][d_model];

#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif

    qkv(Q, K, V, input, actual_len, Wqkv, bqkv);
  ATTN_SCORES_HEAD_LOOP:
    for (int h = 0; h < num_heads; h++) {
    ATTN_SCORES_I_LOOP:
      for (int i = 0; i < actual_len; i++) {
        dtype row_max = dtype(-1e9);
      ATTN_SCORES_J_LOOP:
        for (int j = 0; j < actual_len; j++) {
          dtype score = dtype(0.0f);
        ATTN_SCORES_DOT_PRODUCT:
          for (int k = 0; k < head_dim; k++) {
            score += Q[i][h * head_dim + k] * K[j][h * head_dim + k];
          }
#ifdef __VITIS_HLS__
          score = score / hls::sqrt((dtype)head_dim);
#else
          score = score / std::sqrt((dtype)head_dim);
#endif
          if (mask != nullptr) {
            score += mask[i * actual_len + j]; // apply -inf for masking
          }
          if (score > row_max) {
            row_max = score;
          }
          attn_scores[h][i][j] = score;
        }
        max_exp_score[h][i] = row_max;
      }
    }

  ATTN_SOFTMAX_HEAD_LOOP:
    for (int h = 0; h < num_heads; h++) {
    ATTN_SOFTMAX_I_LOOP:
      for (int i = 0; i < actual_len; i++) {
        dtype sum_exp = dtype(0.0f);
      ATTN_SOFTMAX_J_LOOP:
        for (int j = 0; j < actual_len; j++) {
          attn_scores[h][i][j] =
#ifdef __VITIS_HLS__
              hls::exp(attn_scores[h][i][j] - max_exp_score[h][i]);
#else
              std::exp(attn_scores[h][i][j] - max_exp_score[h][i]);
#endif
          sum_exp += attn_scores[h][i][j];
        }
      ATTN_SOFTMAX_NORMALIZE_J_LOOP:
        for (int j = 0; j < actual_len; j++) {
          attn_scores[h][i][j] = attn_scores[h][i][j] / sum_exp;
        }
      }
    }

  ATTN_OUTPUT_HEAD_LOOP:
    for (int h = 0; h < num_heads; h++) {
    ATTN_OUTPUT_I_LOOP:
      for (int i = 0; i < actual_len; i++) {
      ATTN_OUTPUT_K_LOOP:
        for (int k = 0; k < head_dim; k++) {
          dtype sum = dtype(0.0f);
        ATTN_OUTPUT_J_LOOP:
          for (int j = 0; j < actual_len; j++) {
            sum += attn_scores[h][i][j] * V[j][h * head_dim + k];
          }
          heads[h][i][k] = sum;
        }
      }
    }

  CONCAT_LOOP:
    for (int i = 0; i < actual_len; i++) {
    CONCAT_HEADS_LOOP:
      for (int h = 0; h < num_heads; h++) {
      CONCAT_HEAD_DIM_LOOP:
        for (int k = 0; k < head_dim; k++) {
          attn_output[i][h * head_dim + k] = heads[h][i][k];
        }
      }
    }

  OUTPUT_LOOP:
    for (int i = 0; i < actual_len; i++) {
    OUTPUT_INNER_LOOP:
      for (int j = 0; j < d_model; j++) {
        dtype acc = dtype(0.0f);
      OUTPUT_DOT_PRODUCT:
        for (int k = 0; k < d_model; k++) {
          acc += attn_output[i][k] * Wo[j][k];
        }
        acc += bo[j];
        output[i][j] = acc;
      }
    }
  }

private:
};

template <typename DType, const int D_MODEL, const int NUM_HEADS,
          typename Config>
class FusedMHA<DType, D_MODEL, NUM_HEADS, Config, OPT_ENABLED> {
  static_assert(D_MODEL % NUM_HEADS == 0,
                "D_MODEL must be divisible by NUM_HEADS");
  using dtype = DType;
  static constexpr int d_model = D_MODEL;
  static constexpr int num_heads = NUM_HEADS;
  static constexpr int head_dim = D_MODEL / NUM_HEADS;
  static constexpr OptLevel opt_level = OPT_ENABLED;

  using Wqkv_t = dtype[3 * D_MODEL][D_MODEL];
  using bqkv_t = dtype[3 * D_MODEL];
  using Wo_t = dtype[D_MODEL][D_MODEL];
  using bo_t = dtype[D_MODEL];

  static constexpr int unroll_factor = Config::_unroll_factor;
  static constexpr int partition_factor = Config::_partition_factor;
  static constexpr int pipeline_ii = Config::_pipeline_ii;

  FusedMHA() = default;
  ~FusedMHA() = default;

  static void qkv(dtype Q[][D_MODEL], dtype K[][D_MODEL], dtype V[][D_MODEL],
                  const dtype input[][D_MODEL], const int actual_len,
                  const Wqkv_t Wqkv, const bqkv_t bqkv) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = Q cyclic factor =                       \
    partition_factor dim = 1
#pragma HLS ARRAY_PARTITION variable = K cyclic factor =                       \
    partition_factor dim = 1
#pragma HLS ARRAY_PARTITION variable = V cyclic factor =                       \
    partition_factor dim = 1
#pragma HLS ARRAY_PARTITION variable = input cyclic factor =                   \
    partition_factor dim = 1
#endif
  QKV_OUTER_LOOP:
    for (int i = 0; i < actual_len; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#endif
    QKV_INNER_LOOP:
      for (int j = 0; j < 3 * d_model; j++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#endif
        dtype sum = dtype(0.0f);
      QKV_DOT_PRODUCT:
        for (int k = 0; k < d_model; k++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = unroll_factor
#endif
          sum += input[i][k] * Wqkv[j][k];
        }
        sum += bqkv[j];
        if (j < d_model) {
          Q[i][j] = sum;
        } else if (j < 2 * d_model) {
          K[i][j - d_model] = sum;
        } else {
          V[i][j - 2 * d_model] = sum;
        }
      }
    }
  }

  void forward(dtype output[][D_MODEL], const dtype input[][D_MODEL],
               const dtype mask[], const int actual_len, const Wqkv_t Wqkv,
               const bqkv_t bqkv, const Wo_t Wo, const bo_t bo) {
    dtype heads[num_heads][actual_len][head_dim];
    dtype attn_scores[num_heads][actual_len][actual_len];
    dtype attn_output[actual_len][d_model];
    dtype max_exp_score[num_heads][actual_len];
    dtype Q[actual_len][d_model];
    dtype K[actual_len][d_model];
    dtype V[actual_len][d_model];

#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = heads cyclic factor =                   \
    partition_factor dim = 1
#pragma HLS ARRAY_PARTITION variable = attn_scores cyclic factor =             \
    partition_factor dim = 1
#endif

    qkv(Q, K, V, input, actual_len, Wqkv, bqkv);
  ATTN_SCORES_HEAD_LOOP:
    for (int h = 0; h < num_heads; h++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif

    ATTN_SCORES_I_LOOP:
      for (int i = 0; i < actual_len; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#endif
        dtype row_max = dtype(-1e9);
      ATTN_SCORES_J_LOOP:
        for (int j = 0; j < actual_len; j++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#endif
          dtype score = dtype(0.0f);
        ATTN_SCORES_DOT_PRODUCT:
          for (int k = 0; k < head_dim; k++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = unroll_factor
#endif
            score += Q[i][h * head_dim + k] * K[j][h * head_dim + k];
          }
          score /= sqrt((dtype)head_dim);
          score += mask[i * actual_len + j]; // apply -inf for masking

          if (score > row_max) {
            row_max = score;
          }
          attn_scores[h][i][j] = score;
        }
        max_exp_score[h][i] = row_max;
      }
    }

  ATTN_SOFTMAX_HEAD_LOOP:
    for (int h = 0; h < num_heads; h++) {
    ATTN_SOFTMAX_I_LOOP:
      for (int i = 0; i < actual_len; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#endif
        dtype sum_exp = dtype(0.0f);
      ATTN_SOFTMAX_J_LOOP:
        for (int j = 0; j < actual_len; j++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = unroll_factor
#endif
          attn_scores[h][i][j] =
              exp(attn_scores[h][i][j] - max_exp_score[h][i]);
          sum_exp += attn_scores[h][i][j];
        }
      ATTN_SOFTMAX_NORMALIZE_J_LOOP:
        for (int j = 0; j < actual_len; j++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = unroll_factor
#endif
          attn_scores[h][i][j] /= sum_exp;
        }
      }
    }

  ATTN_OUTPUT_HEAD_LOOP:
    for (int h = 0; h < num_heads; h++) {
    ATTN_OUTPUT_I_LOOP:
      for (int i = 0; i < actual_len; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#endif
      ATTN_OUTPUT_K_LOOP:
        for (int k = 0; k < head_dim; k++) {
          dtype sum = dtype(0.0f);
        ATTN_OUTPUT_J_LOOP:
          for (int j = 0; j < actual_len; j++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = unroll_factor
#endif
            sum += attn_scores[h][i][j] * V[j][h * head_dim + k];
          }
          heads[h][i][k] = sum;
        }
      }
    }

  CONCAT_LOOP:
    for (int i = 0; i < actual_len; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#endif
    CONCAT_HEADS_LOOP:
      for (int h = 0; h < num_heads; h++) {
      CONCAT_HEAD_DIM_LOOP:
        for (int k = 0; k < head_dim; k++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = unroll_factor
#endif
          attn_output[i][h * head_dim + k] = heads[h][i][k];
        }
      }
    }

  OUTPUT_LOOP:
    for (int i = 0; i < actual_len; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#endif
    OUTPUT_INNER_LOOP:
      for (int j = 0; j < d_model; j++) {
        dtype acc = dtype(0.0f);
      OUTPUT_DOT_PRODUCT:
        for (int k = 0; k < d_model; k++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = unroll_factor
#endif
          acc += attn_output[i][k] * Wo[j][k];
        }
        acc += bo[j];
        output[i][j] = acc;
      }
    }
  }

private:
};

} // namespace hls_nn
