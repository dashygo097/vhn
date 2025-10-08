#pragma once

#include "../../../opt_level.hh"

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace hls_nn {
template <const int UNROLL_FACTOR, const int PARTITION_FACTOR,
          const int PIPELINE_II>
struct FusedMHAHLSConfig {
  static constexpr int _unroll_factor = UNROLL_FACTOR;
  static constexpr int _partition_factor = PARTITION_FACTOR;
  static constexpr int _pipeline_ii = PIPELINE_II;
};

template <typename DType, const int D_MODEL, const int NUM_HEADS,
          const int MAX_SEQ_LEN, typename Config, OptLevel OPT_LEVEL = OPT_NONE>
class FusedMHA {
public:
  static_assert(D_MODEL % NUM_HEADS == 0,
                "D_MODEL must be divisible by NUM_HEADS");
  using dtype = DType;
  static constexpr int d_model = D_MODEL;
  static constexpr int num_heads = NUM_HEADS;
  static constexpr int head_dim = D_MODEL / NUM_HEADS;
  static constexpr int max_seq_len = MAX_SEQ_LEN;
  static constexpr OptLevel opt_level = OPT_LEVEL;

  using Wqkv_t = dtype[3 * D_MODEL][D_MODEL];
  using bqkv_t = dtype[3 * D_MODEL];
  using Wo_t = dtype[D_MODEL][D_MODEL];
  using bo_t = dtype[D_MODEL];

  FusedMHA() = default;
  FusedMHA(const Wqkv_t &Wqkv, const bqkv_t &bqkv, const Wo_t &Wo,
           const bo_t &bo)
      : _Wqkv(&Wqkv), _bqkv(&bqkv), _Wo(&Wo), _bo(&bo) {}
  ~FusedMHA() = default;

  void load_weights(const Wqkv_t &Wqkv, const bqkv_t &bqkv, const Wo_t &Wo,
                    const bo_t &bo) {
    _Wqkv = &Wqkv;
    _bqkv = &bqkv;
    _Wo = &Wo;
    _bo = &bo;
  }

  [[nodiscard]] const Wqkv_t &Wqkv() const { return *_Wqkv; }
  [[nodiscard]] const bqkv_t &bqkv() const { return *_bqkv; }
  [[nodiscard]] const Wo_t &Wo() const { return *_Wo; }
  [[nodiscard]] const bo_t &bo() const { return *_bo; }
  [[nodiscard]] bool isLoaded() const {

    return _Wqkv != nullptr && _bqkv != nullptr && _Wo != nullptr &&
           _bo != nullptr;
  }

  void qkv(dtype *Q, dtype *K, dtype *V, const dtype *input,
           const int actual_len);
  void forward(dtype *output, const dtype *input, const dtype *mask,
               const int actual_len);

private:
  const Wqkv_t *_Wqkv{nullptr};
  const bqkv_t *_bqkv{nullptr};
  const Wo_t *_Wo{nullptr};
  const bo_t *_bo{nullptr};
};

template <typename DType, const int D_MODEL, const int NUM_HEADS,
          const int MAX_SEQ_LEN>
class FusedMHA<DType, D_MODEL, NUM_HEADS, MAX_SEQ_LEN, void, OPT_NONE> {
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

  FusedMHA() = default;
  FusedMHA(const Wqkv_t &Wqkv, const bqkv_t &bqkv, const Wo_t &Wo,
           const bo_t &bo)
      : _Wqkv(&Wqkv), _bqkv(&bqkv), _Wo(&Wo), _bo(&bo) {}
  ~FusedMHA() = default;

  void load_weights(const Wqkv_t &Wqkv, const bqkv_t &bqkv, const Wo_t &Wo,
                    const bo_t &bo) {
    _Wqkv = &Wqkv;
    _bqkv = &bqkv;
    _Wo = &Wo;
    _bo = &bo;
  }

  [[nodiscard]] const Wqkv_t &Wqkv() const { return *_Wqkv; }
  [[nodiscard]] const bqkv_t &bqkv() const { return *_bqkv; }
  [[nodiscard]] const Wo_t &Wo() const { return *_Wo; }
  [[nodiscard]] const bo_t &bo() const { return *_bo; }
  [[nodiscard]] bool isLoaded() const {
    return _Wqkv != nullptr && _bqkv != nullptr && _Wo != nullptr &&
           _bo != nullptr;
  }

  void qkv(dtype *Q, dtype *K, dtype *V, const dtype *input,
           const int actual_len) {
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
          sum += input[i * d_model + k] * (*_Wqkv)[j][k];
        }
        sum += (*_bqkv)[j];
        if (j < d_model) {
          Q[i * d_model + j] = sum;
        } else if (j < 2 * d_model) {
          K[i * d_model + (j - d_model)] = sum;
        } else {
          V[i * d_model + (j - 2 * d_model)] = sum;
        }
      }
    }
  }

  void forward(dtype *output, const dtype *input, const dtype *mask,
               const int actual_len) {
    dtype heads[num_heads][max_seq_len][head_dim];
    dtype attn_scores[num_heads][actual_len][actual_len];
    dtype attn_output[actual_len * d_model];
    dtype max_exp_score[num_heads][actual_len];
    dtype Q[actual_len * d_model];
    dtype K[actual_len * d_model];
    dtype V[actual_len * d_model];

#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    qkv(&Q, &K, &V, &input, &actual_len);

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
            score += Q[i * d_model + h * head_dim + k] *
                     K[j * d_model + h * head_dim + k];
          }
          score /= sqrt((dtype)head_dim);
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
              exp(attn_scores[h][i][j] - max_exp_score[h][i]);
          sum_exp += attn_scores[h][i][j];
        }
      ATTN_SOFTMAX_NORMALIZE_J_LOOP:
        for (int j = 0; j < actual_len; j++) {
          attn_scores[h][i][j] /= sum_exp;
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
            sum += attn_scores[h][i][j] * V[j * d_model + h * head_dim + k];
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
          attn_output[i * d_model + h * head_dim + k] = heads[h][i][k];
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
          acc += attn_output[i * d_model + k] * (*_Wo)[j][k];
        }
        acc += (*_bo)[j];
        output[i * d_model + j] = acc;
      }
    }
  }

private:
  const Wqkv_t *_Wqkv{nullptr};
  const bqkv_t *_bqkv{nullptr};
  const Wo_t *_Wo{nullptr};
  const bo_t *_bo{nullptr};
};

template <typename DType, const int D_MODEL, const int NUM_HEADS,
          const int MAX_SEQ_LEN, typename Config>
class FusedMHA<DType, D_MODEL, NUM_HEADS, MAX_SEQ_LEN, Config, OPT_ENABLED> {
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

  static constexpr int unroll_factor = Config::_unroll_factor;
  static constexpr int partition_factor = Config::_partition_factor;
  static constexpr int pipeline_ii = Config::_pipeline_ii;

  FusedMHA() = default;
  FusedMHA(const Wqkv_t &Wqkv, const bqkv_t &bqkv, const Wo_t &Wo,
           const bo_t &bo)
      : _Wqkv(&Wqkv), _bqkv(&bqkv), _Wo(&Wo), _bo(&bo) {}
  ~FusedMHA() = default;

  void load_weights(const Wqkv_t &Wqkv, const bqkv_t &bqkv, const Wo_t &Wo,
                    const bo_t &bo) {
    _Wqkv = &Wqkv;
    _bqkv = &bqkv;
    _Wo = &Wo;
    _bo = &bo;
  }

  [[nodiscard]] const Wqkv_t &Wqkv() const { return *_Wqkv; }
  [[nodiscard]] const bqkv_t &bqkv() const { return *_bqkv; }
  [[nodiscard]] const Wo_t &Wo() const { return *_Wo; }
  [[nodiscard]] const bo_t &bo() const { return *_bo; }
  [[nodiscard]] bool isLoaded() const {
    return _Wqkv != nullptr && _bqkv != nullptr && _Wo != nullptr &&
           _bo != nullptr;
  }

  void qkv(dtype *Q, dtype *K, dtype *V, const dtype *input,
           const int actual_len) {
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
          sum += input[i * d_model + k] * (*_Wqkv)[j][k];
        }
        sum += (*_bqkv)[j];
        if (j < d_model) {
          Q[i * d_model + j] = sum;
        } else if (j < 2 * d_model) {
          K[i * d_model + (j - d_model)] = sum;
        } else {
          V[i * d_model + (j - 2 * d_model)] = sum;
        }
      }
    }
  }

  void forward(dtype *output, const dtype *input, const dtype *mask,
               const int actual_len) {
    dtype heads[num_heads][max_seq_len][head_dim];
    dtype attn_scores[num_heads][actual_len][actual_len];
    dtype attn_output[actual_len * d_model];
    dtype max_exp_score[num_heads][actual_len];
    dtype Q[actual_len * d_model];
    dtype K[actual_len * d_model];
    dtype V[actual_len * d_model];

#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = heads cyclic factor =                   \
    partition_factor dim = 1
#pragma HLS ARRAY_PARTITION variable = attn_scores cyclic factor =             \
    partition_factor dim = 1
#endif

    qkv(&Q, &K, &V, &input, &actual_len);
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
            score += Q[i * d_model + h * head_dim + k] *
                     K[j * d_model + h * head_dim + k];
          }
          score /= sqrt((dtype)head_dim);
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
            sum += attn_scores[h][i][j] * V[j * d_model + h * head_dim + k];
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
          attn_output[i * d_model + h * head_dim + k] = heads[h][i][k];
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
          acc += attn_output[i * d_model + k] * (*_Wo)[j][k];
        }
        acc += (*_bo)[j];
        output[i * d_model + j] = acc;
      }
    }
  }

private:
  const Wqkv_t *_Wqkv{nullptr};
  const bqkv_t *_bqkv{nullptr};
  const Wo_t *_Wo{nullptr};
  const bo_t *_bo{nullptr};
};

} // namespace hls_nn
