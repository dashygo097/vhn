#pragma once

#include "../../../layers/linear.hh"
#include "../../../layers/softmax.hh"
#include "../../../opt_level.hh"
#include <cmath>

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace vhn {

template <typename DType, typename HParams, typename Config,
          OptLevel OPT_LEVEL = OPT_NONE>
class MulHeadAttn;

template <typename WQKV_HParams, typename SOFTMAX_HParams, typename WO_HParams,
          int MAX_SEQ_LEN>
struct MulHeadAttnHParams {
  using wqkv_hparams = WQKV_HParams;
  using softmax_hparams = SOFTMAX_HParams;
  using wo_hparams = WO_HParams;

  static constexpr int d_model = WQKV_HParams::in_features;
  static constexpr int num_heads = WQKV_HParams::out_features / d_model;
  static constexpr int head_dim = d_model / num_heads;
  static constexpr int max_seq_len = MAX_SEQ_LEN;
};

// ============================================================================
// Non-optimized version (OPT_NONE)
// ============================================================================
template <typename DType, typename HParams>
class MulHeadAttn<DType, HParams, void, OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int d_model = HParams::d_model;
  static constexpr int num_heads = HParams::num_heads;
  static constexpr int head_dim = HParams::head_dim;
  static constexpr int max_seq_len = HParams::max_seq_len;
  static constexpr OptLevel opt_level = OPT_NONE;

  using Wqkv_t = dtype[3 * d_model][d_model];
  using bqkv_t = dtype[3 * d_model];
  using Wo_t = dtype[d_model][d_model];
  using bo_t = dtype[d_model];

  MulHeadAttn() = default;
  ~MulHeadAttn() = default;

  using wqkv_hparams = typename HParams::wqkv_hparams;
  using softmax_hparams = typename HParams::softmax_hparams;
  using wo_hparams = typename HParams::wo_hparams;

  using wqkv = Linear<dtype, wqkv_hparams, void, OPT_NONE>;
  using softmax = Softmax<dtype, softmax_hparams, void, OPT_NONE>;
  using wo = Linear<dtype, wo_hparams, void, OPT_NONE>;

  static void forward(dtype output[][d_model], const dtype input[][d_model],
                      const int actual_len, const Wqkv_t wqkv,
                      const bqkv_t bqkv, const Wo_t wo, const bo_t bo) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#endif
    dtype qkv[max_seq_len][3 * d_model];
    wqkv::forward(qkv, input, actual_len, wqkv, bqkv);

    dtype q[max_seq_len][num_heads][head_dim];
    dtype k[max_seq_len][num_heads][head_dim];
    dtype v[max_seq_len][num_heads][head_dim];
    split_qkv(q, k, v, qkv, actual_len);

    dtype attn_output[max_seq_len][num_heads][head_dim];
    compute_attention(attn_output, q, k, v, actual_len);

    dtype concat[max_seq_len][d_model];
    concat_heads(concat, attn_output, actual_len);

    wo::forward(output, concat, actual_len, wo, bo);
  }

private:
  static void split_qkv(dtype q[][num_heads][head_dim],
                        dtype k[][num_heads][head_dim],
                        dtype v[][num_heads][head_dim],
                        const dtype qkv[][3 * d_model], const int actual_len) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#endif
    for (int i = 0; i < actual_len; i++) {
      for (int h = 0; h < num_heads; h++) {
        for (int d = 0; d < head_dim; d++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 128
#endif
          q[i][h][d] = qkv[i][h * head_dim + d];
          k[i][h][d] = qkv[i][d_model + h * head_dim + d];
          v[i][h][d] = qkv[i][2 * d_model + h * head_dim + d];
        }
      }
    }
  }

  static void compute_attention(dtype attn_output[][num_heads][head_dim],
                                const dtype q[][num_heads][head_dim],
                                const dtype k[][num_heads][head_dim],
                                const dtype v[][num_heads][head_dim],
                                const int actual_len) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#endif
#ifdef __VITIS_HLS__

    dtype scale = dtype(1.0) / hls::sqrt(dtype(head_dim));
#else
    dtype scale = dtype(1.0) / sqrt(dtype(head_dim));
#endif

    for (int h = 0; h < num_heads; h++) {
      dtype scores[max_seq_len][max_seq_len];
      for (int i = 0; i < actual_len; i++) {
        for (int j = 0; j < actual_len; j++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#pragma HLS LOOP_FLATTEN off
#endif
          dtype dot = dtype(0.0);
          for (int d = 0; d < head_dim; d++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 128
#endif
            dot += q[i][h][d] * k[j][h][d];
          }
          scores[i][j] = dot * scale;
        }
      }

      dtype attn_weights[max_seq_len][max_seq_len];
      for (int i = 0; i < actual_len; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#endif
        softmax::forward(attn_weights[i], scores[i]);
      }

      for (int i = 0; i < actual_len; i++) {
        for (int d = 0; d < head_dim; d++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#endif
          dtype sum = dtype(0.0);
          for (int j = 0; j < actual_len; j++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#endif
            sum += attn_weights[i][j] * v[j][h][d];
          }
          attn_output[i][h][d] = sum;
        }
      }
    }
  }

  static void concat_heads(dtype concat[][d_model],
                           const dtype attn_output[][num_heads][head_dim],
                           const int actual_len) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    for (int i = 0; i < actual_len; i++) {
      for (int h = 0; h < num_heads; h++) {
        for (int d = 0; d < head_dim; d++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 128
#endif
          concat[i][h * head_dim + d] = attn_output[i][h][d];
        }
      }
    }
  }
};

template <typename WQKV_CONFIG, typename SOFTMAX_CONFIG, typename WO_CONFIG,
          bool DATAFLOW_ENABLED, int PIPELINE_II, int QKV_PARTITION_FACTOR,
          int ATTN_PARTITION_FACTOR, int ATTN_UNROLL_FACTOR,
          int HEAD_UNROLL_FACTOR>
struct MulHeadAttnConfig {
  using wqkv_config = WQKV_CONFIG;
  using softmax_config = SOFTMAX_CONFIG;
  using wo_config = WO_CONFIG;

  static constexpr bool dataflow_enabled = DATAFLOW_ENABLED;
  static constexpr int pipeline_ii = PIPELINE_II;
  static constexpr int qkv_partition_factor = QKV_PARTITION_FACTOR;
  static constexpr int attn_partition_factor = ATTN_PARTITION_FACTOR;
  static constexpr int attn_unroll_factor = ATTN_UNROLL_FACTOR;
  static constexpr int head_unroll_factor = HEAD_UNROLL_FACTOR;
};

// ============================================================================
// Optimized version (OPT_ENABLED)
// ============================================================================
template <typename DType, typename HParams, typename Config>
class MulHeadAttn<DType, HParams, Config, OPT_ENABLED> {
public:
  using dtype = DType;
  static constexpr int d_model = HParams::d_model;
  static constexpr int num_heads = HParams::num_heads;
  static constexpr int head_dim = HParams::head_dim;
  static constexpr int max_seq_len = HParams::max_seq_len;
  static constexpr OptLevel opt_level = OPT_ENABLED;

  static constexpr int dataflow_enabled = Config::dataflow_enabled;
  static constexpr int pipeline_ii = Config::pipeline_ii;
  static constexpr int qkv_partition_factor = Config::qkv_partition_factor;
  static constexpr int attn_partition_factor = Config::attn_partition_factor;
  static constexpr int attn_unroll_factor = Config::attn_unroll_factor;
  static constexpr int head_unroll_factor = Config::head_unroll_factor;

  using Wqkv_t = dtype[3 * d_model][d_model];
  using bqkv_t = dtype[3 * d_model];
  using Wo_t = dtype[d_model][d_model];
  using bo_t = dtype[d_model];

  MulHeadAttn() = default;
  ~MulHeadAttn() = default;

  using wqkv_hparams = typename HParams::wqkv_hparams;
  using softmax_hparams = typename HParams::softmax_hparams;
  using wo_hparams = typename HParams::wo_hparams;

  using wqkv_config = typename Config::wqkv_config;
  using softmax_config = typename Config::softmax_config;
  using wo_config = typename Config::wo_config;

  static constexpr bool is_wqkv_optimized =
      !std::is_same<wqkv_config, void>::value;
  static constexpr bool is_softmax_optimized =
      !std::is_same<softmax_config, void>::value;
  static constexpr bool is_wo_optimized = !std::is_same<wo_config, void>::value;

  using wqkv = Linear<dtype, wqkv_hparams, wqkv_config,
                      is_wqkv_optimized ? OPT_ENABLED : OPT_NONE>;
  using softmax = Softmax<dtype, softmax_hparams, softmax_config,
                          is_softmax_optimized ? OPT_ENABLED : OPT_NONE>;
  using wo = Linear<dtype, wo_hparams, wo_config,
                    is_wo_optimized ? OPT_ENABLED : OPT_NONE>;

  static void forward(dtype output[][d_model], const dtype input[][d_model],
                      const int actual_len, const Wqkv_t wqkv,
                      const bqkv_t bqkv, const Wo_t wo, const bo_t bo) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    dtype qkv[max_seq_len][3 * d_model];
#ifdef __VITIS_HLS__
    constexpr bool should_partition_qkv =
        (qkv_partition_factor > 1) && (d_model <= 2048);
    if constexpr (should_partition_qkv) {
#pragma HLS ARRAY_PARTITION variable = qkv type = cyclic factor =              \
    qkv_partition_factor dim = 2
    } else {
#pragma HLS ARRAY_PARTITION variable = qkv type = cyclic factor = 4 dim = 2
    }
#endif
    wqkv::forward(qkv, input, actual_len, wqkv, bqkv);

    dtype q[max_seq_len][num_heads][head_dim];
    dtype k[max_seq_len][num_heads][head_dim];
    dtype v[max_seq_len][num_heads][head_dim];
#ifdef __VITIS_HLS__
    constexpr bool should_partition_qkv_split =
        (attn_partition_factor > 1) && (head_dim <= 512);
    if constexpr (should_partition_qkv_split) {
#pragma HLS ARRAY_PARTITION variable = q type = cyclic factor =                \
    attn_partition_factor dim = 3
#pragma HLS ARRAY_PARTITION variable = k type = cyclic factor =                \
    attn_partition_factor dim = 3
#pragma HLS ARRAY_PARTITION variable = v type = cyclic factor =                \
    attn_partition_factor dim = 3
    } else {
#pragma HLS ARRAY_PARTITION variable = q type = cyclic factor = 4 dim = 3
#pragma HLS ARRAY_PARTITION variable = k type = cyclic factor = 4 dim = 3
#pragma HLS ARRAY_PARTITION variable = v type = cyclic factor = 4 dim = 3
    }
#endif
    split_qkv(q, k, v, qkv, actual_len);

    dtype attn_output[max_seq_len][num_heads][head_dim];
#ifdef __VITIS_HLS__
    if constexpr (should_partition_qkv_split) {
#pragma HLS ARRAY_PARTITION variable = attn_output type = cyclic factor =      \
    attn_partition_factor dim = 3
    } else {
#pragma HLS ARRAY_PARTITION variable = attn_output type = cyclic factor =      \
    4 dim = 3
    }
#endif
    compute_attention(attn_output, q, k, v, actual_len);

    dtype concat[max_seq_len][d_model];
#ifdef __VITIS_HLS__
    if constexpr (should_partition_qkv) {
#pragma HLS ARRAY_PARTITION variable = concat type = cyclic factor =           \
    qkv_partition_factor dim = 2
    } else {
#pragma HLS ARRAY_PARTITION variable = concat type = cyclic factor = 4 dim = 2
    }
#endif
    concat_heads(concat, attn_output, actual_len);

    wo::forward(output, concat, actual_len, wo, bo);
  }

private:
  static void split_qkv(dtype q[][num_heads][head_dim],
                        dtype k[][num_heads][head_dim],
                        dtype v[][num_heads][head_dim],
                        const dtype qkv[][3 * d_model], const int actual_len) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    for (int i = 0; i < actual_len; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#endif
      for (int h = 0; h < num_heads; h++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#pragma HLS UNROLL factor = head_unroll_factor
#endif
        for (int d = 0; d < head_dim; d++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 128
#pragma HLS UNROLL factor = attn_unroll_factor
#endif
          q[i][h][d] = qkv[i][h * head_dim + d];
          k[i][h][d] = qkv[i][d_model + h * head_dim + d];
          v[i][h][d] = qkv[i][2 * d_model + h * head_dim + d];
        }
      }
    }
  }

  static void compute_attention(dtype attn_output[][num_heads][head_dim],
                                const dtype q[][num_heads][head_dim],
                                const dtype k[][num_heads][head_dim],
                                const dtype v[][num_heads][head_dim],
                                const int actual_len) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
#ifdef __VITIS_HLS__
    dtype scale = dtype(1.0) / hls::sqrt(dtype(head_dim));
#else
    dtype scale = dtype(1.0) / sqrt(dtype(head_dim));
#endif

    for (int h = 0; h < num_heads; h++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = head_unroll_factor
#endif
      dtype scores[max_seq_len][max_seq_len];
#ifdef __VITIS_HLS__
      constexpr bool should_partition_scores =
          (attn_partition_factor > 1) && (max_seq_len <= 1024);
      if constexpr (should_partition_scores) {
#pragma HLS ARRAY_PARTITION variable = scores type = cyclic factor =           \
    attn_partition_factor
      } else {
#pragma HLS ARRAY_PARTITION variable = scores type = cyclic factor = 4
      }
#endif

      // Compute QK^T / sqrt(d_k)
      for (int i = 0; i < actual_len; i++) {
        for (int j = 0; j < actual_len; j++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE II = pipeline_ii
#endif
          dtype dot = dtype(0.0);
          for (int d = 0; d < head_dim; d++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 128
#pragma HLS UNROLL factor = attn_unroll_factor
#pragma HLS BIND_OP variable = dot op = add impl = dsp
#endif
            dot += q[i][h][d] * k[j][h][d];
          }
          scores[i][j] = dot * scale;
        }
      }

      // Apply softmax per row
      dtype attn_weights[max_seq_len][max_seq_len];
#ifdef __VITIS_HLS__
      if constexpr (should_partition_scores) {
#pragma HLS ARRAY_PARTITION variable = attn_weights type = cyclic factor =     \
    attn_partition_factor
      } else {
#pragma HLS ARRAY_PARTITION variable = attn_weights type = cyclic factor = 4
      }
#endif
      for (int i = 0; i < actual_len; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#pragma HLS PIPELINE II = pipeline_ii
#endif
        softmax::forward(attn_weights[i], scores[i]);
      }

      // Apply attention weights to values
      for (int i = 0; i < actual_len; i++) {
        for (int d = 0; d < head_dim; d++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE II = pipeline_ii
#endif
          dtype sum = dtype(0.0);
          for (int j = 0; j < actual_len; j++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#pragma HLS UNROLL factor = attn_unroll_factor
#pragma HLS BIND_OP variable = sum op = add impl = dsp
#endif
            sum += attn_weights[i][j] * v[j][h][d];
          }
          attn_output[i][h][d] = sum;
        }
      }
    }
  }

  static void concat_heads(dtype concat[][d_model],
                           const dtype attn_output[][num_heads][head_dim],
                           const int actual_len) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    for (int i = 0; i < actual_len; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#endif
      for (int h = 0; h < num_heads; h++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#pragma HLS UNROLL factor = head_unroll_factor
#endif
        for (int d = 0; d < head_dim; d++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 128
#pragma HLS UNROLL factor = attn_unroll_factor
#endif
          concat[i][h * head_dim + d] = attn_output[i][h][d];
        }
      }
    }
  }
};

} // namespace vhn
