#pragma once

#include "../../opt_level.hh"
#include "./attns/mha.hh"
#include "./components/addnorm.hh"
#include "./components/ffn.hh"

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace vhn {

template <typename DType, const int D_MODEL, const int NUM_HEADS,
          const int MAX_SEQ_LEN, const int D_FF,
          template <typename, int, typename, OptLevel> class ActLayer,
          NormType NORM_TYPE = POSTNORM, typename Config = void,
          OptLevel OPT_LEVEL = OPT_NONE>
class EncoderBlock;

// ============================================================================
// EncoderBlock specialization for OPT_NONE
// ============================================================================
template <typename DType, const int D_MODEL, const int NUM_HEADS,
          const int MAX_SEQ_LEN, const int D_FF,
          template <typename, int, typename, OptLevel> class ActLayer,
          NormType NORM_TYPE>
class EncoderBlock<DType, D_MODEL, NUM_HEADS, MAX_SEQ_LEN, D_FF, ActLayer,
                   NORM_TYPE, void, OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int d_model = D_MODEL;
  static constexpr int num_heads = NUM_HEADS;
  static constexpr int max_seq_len = MAX_SEQ_LEN;
  static constexpr int d_ff = D_FF;
  static constexpr OptLevel opt_level = OPT_NONE;
  static constexpr NormType norm_type = NORM_TYPE;

  using Wqkv_t = dtype[3 * D_MODEL][D_MODEL];
  using bqkv_t = dtype[3 * D_MODEL];
  using Wo_t = dtype[D_MODEL][D_MODEL];
  using bo_t = dtype[D_MODEL];

  using W1_t = dtype[D_FF][D_MODEL];
  using b1_t = dtype[D_FF];
  using W2_t = dtype[D_MODEL][D_FF];
  using b2_t = dtype[D_MODEL];

  using gamma_t = dtype[D_MODEL];
  using beta_t = dtype[D_MODEL];

  EncoderBlock() = default;
  ~EncoderBlock() = default;

  using mha =
      MulHeadAttn<dtype, D_MODEL, NUM_HEADS, MAX_SEQ_LEN, void, OPT_NONE>;
  using addnorm1 = AddNorm<dtype, D_MODEL, NORM_TYPE, void, OPT_NONE>;
  using ffn = FFN<dtype, D_MODEL, D_FF, ActLayer, void, OPT_NONE>;
  using addnorm2 = AddNorm<dtype, D_MODEL, NORM_TYPE, void, OPT_NONE>;

  static void forward(dtype output[][D_MODEL], const dtype input[][D_MODEL],
                      const int actual_len, const Wqkv_t wqkv,
                      const bqkv_t bqkv, const Wo_t wo, const bo_t bo,
                      const gamma_t gamma1, const beta_t beta1, const W1_t w1,
                      const b1_t b1, const W2_t w2, const b2_t b2,
                      const gamma_t gamma2, const beta_t beta2) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    dtype attn_out[MAX_SEQ_LEN][D_MODEL];
    mha::forward(attn_out, input, actual_len, wqkv, bqkv, wo, bo);

    dtype addnorm1_out[MAX_SEQ_LEN][D_MODEL];
    addnorm1::forward(addnorm1_out, attn_out, input, actual_len, gamma1, beta1);

    dtype ffn_out[MAX_SEQ_LEN][D_MODEL];
    ffn::forward(ffn_out, addnorm1_out, actual_len, w1, b1, w2, b2);

    addnorm2::forward(output, ffn_out, addnorm1_out, actual_len, gamma2, beta2);
  }
};
} // namespace vhn
