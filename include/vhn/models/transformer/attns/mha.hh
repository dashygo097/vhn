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

template <int D_MODEL, int NUM_HEADS, int MAX_SEQ_LEN> struct MHAHParams {
  static constexpr int d_model = D_MODEL;
  static constexpr int num_heads = NUM_HEADS;
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
  static constexpr int head_dim = d_model / num_heads;
  static constexpr int max_seq_len = HParams::max_seq_len;
  static constexpr OptLevel opt_level = OPT_NONE;

  using Wqkv_t = dtype[3 * d_model][d_model];
  using bqkv_t = dtype[3 * d_model];
  using Wo_t = dtype[d_model][d_model];
  using bo_t = dtype[d_model];

  MulHeadAttn() = default;
  ~MulHeadAttn() = default;

  using wqkv =
      Linear<dtype, LinearHParams<d_model, d_model * 3>, void, OPT_NONE>;
  using wo = Linear<dtype, LinearHParams<d_model, d_model>, void, OPT_NONE>;
  using softmax = Softmax<dtype, max_seq_len, void, OPT_NONE>;

  static void forward(dtype output[][d_model], const dtype input[][d_model],
                      const int actual_len, const Wqkv_t wqkv,
                      const bqkv_t bqkv, const Wo_t wo, const bo_t bo) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
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
#endif
    for (int i = 0; i < actual_len; i++) {
      for (int h = 0; h < num_heads; h++) {
        for (int d = 0; d < head_dim; d++) {
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
      dtype scores[max_seq_len][max_seq_len];
      for (int i = 0; i < actual_len; i++) {
        for (int j = 0; j < actual_len; j++) {
          dtype dot = dtype(0.0);
          for (int d = 0; d < head_dim; d++) {
            dot += q[i][h][d] * k[j][h][d];
          }
          scores[i][j] = dot * scale;
        }
      }

      dtype attn_weights[max_seq_len][max_seq_len];
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

  static void concat_heads(dtype concat[][d_model],
                           const dtype attn_output[][num_heads][head_dim],
                           const int actual_len) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    for (int i = 0; i < actual_len; i++) {
      for (int h = 0; h < num_heads; h++) {
        for (int d = 0; d < head_dim; d++) {
          concat[i][h * head_dim + d] = attn_output[i][h][d];
        }
      }
    }
  }
};
} // namespace vhn
