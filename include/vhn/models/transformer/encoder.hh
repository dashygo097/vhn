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
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
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

template <typename MHA_CONFIG, typename ADDNORM1_CONFIG, typename FFN_CONFIG,
          typename ADDNORM2_CONFIG, bool DATAFLOW_ENABLED, int PIPELINE_II,
          int INTERMEDIATE_PARTITION>
struct EncoderBlockConfig {
  using mha_config = MHA_CONFIG;
  using addnorm1_config = ADDNORM1_CONFIG;
  using ffn_config = FFN_CONFIG;
  using addnorm2_config = ADDNORM2_CONFIG;

  static constexpr bool dataflow_enabled = DATAFLOW_ENABLED;
  static constexpr int pipeline_ii = PIPELINE_II;
  static constexpr int intermediate_partition = INTERMEDIATE_PARTITION;
};

// ============================================================================
// EncoderBlock specialization for OPT_ENABLED
// ============================================================================
template <typename DType, typename HParams, typename Config>
class EncoderBlock<DType, HParams, Config, OPT_ENABLED> {
public:
  using dtype = DType;
  static constexpr int d_model = HParams::d_model;
  static constexpr int num_heads = HParams::num_heads;
  static constexpr int max_seq_len = HParams::max_seq_len;
  static constexpr int d_ff = HParams::d_ff;
  static constexpr NormType norm_type = HParams::norm_type;
  static constexpr OptLevel opt_level = OPT_ENABLED;

  static constexpr int dataflow_enabled = Config::dataflow_enabled;
  static constexpr int intermediate_partition = Config::intermediate_partition;
  static constexpr int pipeline_ii = Config::pipeline_ii;

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

  using mha_config = typename Config::mha_config;
  using addnorm1_config = typename Config::addnorm1_config;
  using ffn_config = typename Config::ffn_config;
  using addnorm2_config = typename Config::addnorm2_config;

  static constexpr bool is_mha_optimized =
      !std::is_same<mha_config, void>::value;
  static constexpr bool is_addnorm1_optimized =
      !std::is_same<addnorm1_config, void>::value;
  static constexpr bool is_ffn_optimized =
      !std::is_same<ffn_config, void>::value;
  static constexpr bool is_addnorm2_optimized =
      !std::is_same<addnorm2_config, void>::value;

  using mha = MulHeadAttn<dtype, mha_hparams, mha_config,
                          is_mha_optimized ? OPT_ENABLED : OPT_NONE>;
  using addnorm1 = AddNorm<dtype, addnorm1_hparams, addnorm1_config,
                           is_addnorm1_optimized ? OPT_ENABLED : OPT_NONE>;
  using ffn = FFN<dtype, ffn_hparams, ffn_config,
                  is_ffn_optimized ? OPT_ENABLED : OPT_NONE>;
  using addnorm2 = AddNorm<dtype, addnorm2_hparams, addnorm2_config,
                           is_addnorm2_optimized ? OPT_ENABLED : OPT_NONE>;

  static void forward(dtype output[][d_model], const dtype input[][d_model],
                      const int actual_len, const Wqkv_t wqkv,
                      const bqkv_t bqkv, const Wo_t wo, const bo_t bo,
                      const gamma_t gamma1, const beta_t beta1, const W1_t w1,
                      const b1_t b1, const W2_t w2, const b2_t b2,
                      const gamma_t gamma2, const beta_t beta2) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
    if constexpr (dataflow_enabled) {
#pragma HLS DATAFLOW
    }
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#endif

    dtype attn_out[max_seq_len][d_model];
#ifdef __VITIS_HLS__
    constexpr bool should_partition_attn =
        (intermediate_partition > 1) && (d_model <= 2048);
    if constexpr (should_partition_attn) {
#pragma HLS ARRAY_PARTITION variable = attn_out type = cyclic factor =         \
    intermediate_partition dim = 2
    } else {
#pragma HLS ARRAY_PARTITION variable = attn_out type = cyclic factor = 4 dim = 2
    }
#endif
    mha::forward(attn_out, input, actual_len, wqkv, bqkv, wo, bo);

    dtype addnorm1_out[max_seq_len][d_model];
#ifdef __VITIS_HLS__
    if constexpr (should_partition_attn) {
#pragma HLS ARRAY_PARTITION variable = addnorm1_out type = cyclic factor =     \
    intermediate_partition dim = 2
    } else {
#pragma HLS ARRAY_PARTITION variable = addnorm1_out type = cyclic factor =     \
    4 dim = 2
    }
#endif
    addnorm1::forward(addnorm1_out, attn_out, input, actual_len, gamma1, beta1);

    dtype ffn_out[max_seq_len][d_model];
#ifdef __VITIS_HLS__
    if constexpr (should_partition_attn) {
#pragma HLS ARRAY_PARTITION variable = ffn_out type = cyclic factor =          \
    intermediate_partition dim = 2
    } else {
#pragma HLS ARRAY_PARTITION variable = ffn_out type = cyclic factor = 4 dim = 2
    }
#endif
    ffn::forward(ffn_out, addnorm1_out, actual_len, w1, b1, w2, b2);

    addnorm2::forward(output, ffn_out, addnorm1_out, actual_len, gamma2, beta2);
  }

  static void forward_single(dtype output[d_model], const dtype input[d_model],
                             const Wqkv_t wqkv, const bqkv_t bqkv,
                             const Wo_t wo, const bo_t bo, const gamma_t gamma1,
                             const beta_t beta1, const W1_t w1, const b1_t b1,
                             const W2_t w2, const b2_t b2, const gamma_t gamma2,
                             const beta_t beta2) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS PIPELINE II = 1
#endif

    dtype attn_out[d_model];
#ifdef __VITIS_HLS__
#pragma HLS ARRAY_PARTITION variable = attn_out type = complete
#pragma HLS ARRAY_PARTITION variable = input type = complete
#pragma HLS ARRAY_PARTITION variable = output type = complete
#pragma HLS ARRAY_PARTITION variable = wqkv type = cyclic factor =             \
    mha_config::qkv_partition_factor dim = 2
#pragma HLS ARRAY_PARTITION variable = wo type = cyclic factor =               \
    mha_config::wo_config::partition_factor dim = 2
#pragma HLS ARRAY_PARTITION variable = gamma1 type = cyclic factor =           \
    intermediate_partition
#pragma HLS ARRAY_PARTITION variable = beta1 type = cyclic factor =            \
    intermediate_partition
#pragma HLS ARRAY_PARTITION variable = gamma2 type = cyclic factor =           \
    intermediate_partition
#pragma HLS ARRAY_PARTITION variable = beta2 type = cyclic factor =            \
    intermediate_partition
#endif
    mha::forward(attn_out, input, wqkv, bqkv, wo, bo);

    dtype addnorm1_out[d_model];
#ifdef __VITIS_HLS__
#pragma HLS ARRAY_PARTITION variable = addnorm1_out type = complete
#endif
    addnorm1::forward(addnorm1_out, attn_out, input, gamma1, beta1);

    dtype ffn_out[d_model];
#ifdef __VITIS_HLS__
#pragma HLS ARRAY_PARTITION variable = ffn_out type = complete
#endif
    ffn::forward(ffn_out, addnorm1_out, w1, b1, w2, b2);

    addnorm2::forward(output, ffn_out, addnorm1_out, gamma2, beta2);
  }
};

} // namespace vhn
