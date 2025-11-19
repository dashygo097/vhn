#pragma once

#include "../../opt_level.hh"
#include "./attns/mha.hh"
#include "./components/addnorm.hh"
#include "./components/ffn.hh"

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace vhn {

template <typename DType, typename HParams, typename Config, OptLevel OPT_LEVEL>
class EncoderBlock;

template <typename MHA_HParams, typename ADDNORM1_HParams, typename FFN_HParams,
          typename ADDNORM2_HParams>
struct EncoderBlockHParams {
  using mha_hparams = MHA_HParams;
  using addnorm1_hparams = ADDNORM1_HParams;
  using ffn_hparams = FFN_HParams;
  using addnorm2_hparams = ADDNORM2_HParams;

  static constexpr int d_model = MHA_HParams::d_model;
  static constexpr int num_heads = MHA_HParams::num_heads;
  static constexpr int max_seq_len = MHA_HParams::max_seq_len;
  static constexpr int d_ff = FFN_HParams::d_ff;
  static constexpr NormType norm_type = ADDNORM1_HParams::norm_type;
  using act_type = typename FFN_HParams::act_type;
};

// ============================================================================
// EncoderBlock specialization for OPT_NONE
// ============================================================================
template <typename DType, typename HParams>
class EncoderBlock<DType, HParams, void, OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int d_model = HParams::d_model;
  static constexpr int num_heads = HParams::num_heads;
  static constexpr int max_seq_len = HParams::max_seq_len;
  static constexpr int d_ff = HParams::d_ff;
  static constexpr NormType norm_type = HParams::norm_type;
  static constexpr OptLevel opt_level = OPT_NONE;

  using Wqkv_t = dtype[3 * d_model][d_model];
  using bqkv_t = dtype[3 * d_model];
  using Wo_t = dtype[d_model][d_model];
  using bo_t = dtype[d_model];

  using W1_t = dtype[d_ff][d_model];
  using b1_t = dtype[d_ff];
  using W2_t = dtype[d_model][d_ff];
  using b2_t = dtype[d_model];

  using gamma_t = dtype[d_model];
  using beta_t = dtype[d_model];

  EncoderBlock() = default;
  ~EncoderBlock() = default;

  using mha_hparams = typename HParams::mha_hparams;
  using addnorm1_hparams = typename HParams::addnorm1_hparams;
  using ffn_hparams = typename HParams::ffn_hparams;
  using addnorm2_hparams = typename HParams::addnorm2_hparams;

  using mha = MulHeadAttn<dtype, mha_hparams, void, OPT_NONE>;
  using addnorm1 = AddNorm<dtype, addnorm1_hparams, void, OPT_NONE>;
  using ffn = FFN<dtype, ffn_hparams, void, OPT_NONE>;
  using addnorm2 = AddNorm<dtype, addnorm2_hparams, void, OPT_NONE>;

  static void forward(dtype output[][d_model], const dtype input[][d_model],
                      const int actual_len, const Wqkv_t wqkv,
                      const bqkv_t bqkv, const Wo_t wo, const bo_t bo,
                      const gamma_t gamma1, const beta_t beta1, const W1_t w1,
                      const b1_t b1, const W2_t w2, const b2_t b2,
                      const gamma_t gamma2, const beta_t beta2) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    dtype attn_out[max_seq_len][d_model];
    mha::forward(attn_out, input, actual_len, wqkv, bqkv, wo, bo);

    dtype addnorm1_out[max_seq_len][d_model];
    addnorm1::forward(addnorm1_out, attn_out, input, actual_len, gamma1, beta1);

    dtype ffn_out[max_seq_len][d_model];
    ffn::forward(ffn_out, addnorm1_out, actual_len, w1, b1, w2, b2);

    addnorm2::forward(output, ffn_out, addnorm1_out, actual_len, gamma2, beta2);
  }
};
} // namespace vhn
