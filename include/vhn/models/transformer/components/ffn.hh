#pragma once

#include "../../../layers/linear.hh"
#include "../../../operators/elementwise.hh"

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace vhn {

template <typename DType, typename HParams, typename Config = void,
          OptLevel OPT_LEVEL = OPT_NONE>
class FFN;

template <typename FC1_HParams, typename ACT_HParams, typename FC2_HParams>
struct FFNHParams {
  using fc1_hparams = FC1_HParams;
  using act_hparams = ACT_HParams;
  using fc2_hparams = FC2_HParams;

  static constexpr int d_model = FC1_HParams::in_features;
  static constexpr int d_ff = FC1_HParams::out_features;
  using act_type = typename ACT_HParams::impl;
};

// ============================================================================
// FFN specialization for OPT_NONE
// ============================================================================
template <typename DType, typename HParams>
class FFN<DType, HParams, void, OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int d_model = HParams::d_model;
  static constexpr int d_ff = HParams::d_ff;
  static constexpr OptLevel opt_level = OPT_NONE;

  using W1_t = dtype[d_ff][d_model];
  using b1_t = dtype[d_ff];
  using W2_t = dtype[d_model][d_ff];
  using b2_t = dtype[d_model];

  using fc1_hparams = typename HParams::fc1_hparams;
  using act_hparams = typename HParams::act_hparams;
  using fc2_hparams = typename HParams::fc2_hparams;

  using fc1 = Linear<dtype, fc1_hparams, void, OPT_NONE>;
  using act = Elementwise<dtype, act_hparams, void, OPT_NONE>;
  using fc2 = Linear<dtype, fc2_hparams, void, OPT_NONE>;

  FFN() = default;
  ~FFN() = default;

  static void forward(dtype output[][d_model], const dtype input[][d_model],
                      const int actual_len, const W1_t w1, const b1_t b1,
                      const W2_t w2, const b2_t b2) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    dtype fc1_out[actual_len][d_ff];
    dtype act_out[actual_len][d_ff];

    fc1::forward(fc1_out, input, actual_len, w1, b1);
    act::forward(act_out, fc1_out, actual_len);
    fc2::forward(output, act_out, actual_len, w2, b2);
  }

  static void forward(dtype output[d_model], const dtype input[d_model],
                      const W1_t w1, const b1_t b1, const W2_t w2,
                      const b2_t b2) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    dtype fc1_out[d_ff];
    dtype act_out[d_ff];

    fc1::forward(fc1_out, input, w1, b1);
    act::forward(act_out, fc1_out);
    fc2::forward(output, act_out, w2, b2);
  }

  static void forward(dtype *output, const dtype *input, const int actual_len,
                      const W1_t w1, const b1_t b1, const W2_t w2,
                      const b2_t b2) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    dtype *fc1_out = new dtype[actual_len * d_ff];
    dtype *act_out = new dtype[actual_len * d_ff];
    fc1::forward(fc1_out, input, actual_len, w1, b1);
    act::forward(act_out, fc1_out, actual_len);
    fc2::forward(output, act_out, actual_len, w2, b2);
  }
};
} // namespace vhn
