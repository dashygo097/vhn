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

template <int D_MODEL, int D_FF, typename ActImpl> class FFNHParams {
  using act = ActImpl;
  static constexpr int d_model = D_MODEL;
  static constexpr int d_ff = D_FF;
};

// ============================================================================
// FFN specialization for OPT_NONE
// ============================================================================
template <typename DType, typename HParams>
class FFN<DType, HParams, void, OPT_NONE> {
public:
  using dtype = DType;
  using act = typename HParams::act;
  static constexpr int d_model = HParams::d_model;
  static constexpr int d_ff = HParams::d_ff;
  static constexpr OptLevel opt_level = OPT_NONE;

  using W1_t = dtype[d_ff][d_model];
  using b1_t = dtype[d_ff];
  using W2_t = dtype[d_model][d_ff];
  using b2_t = dtype[d_model];

  using fc1_config = LinearHParams<d_model, d_ff>;
  using act_config = ElementwiseHParams<d_ff, act>;
  using fc2_config = LinearHParams<d_ff, d_model>;

  using fc1 = Linear<dtype, fc1_config, void, OPT_NONE>;
  using act = Elementwise<dtype, act_config, void, OPT_NONE>;
  using fc2 = Linear<dtype, fc2_config, void, OPT_NONE>;

  FFN() = default;
  ~FFN() = default;

  static void forward(dtype output[][d_model], const dtype input[][d_model],
                      const int actual_len, const W1_t w1, const b1_t b1,
                      const W2_t w2, const b2_t b2) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    mlp::forward(output, input, actual_len, w1, b1, w2, b2);
  }

  static void forward(dtype output[d_model], const dtype input[d_model],
                      const W1_t w1, const b1_t b1, const W2_t w2,
                      const b2_t b2) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    mlp::forward(output, input, w1, b1, w2, b2);
  }

  static void forward(dtype *output, const dtype *input, const int actual_len,
                      const W1_t w1, const b1_t b1, const W2_t w2,
                      const b2_t b2) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    mlp::forward(output, input, actual_len, w1, b1, w2, b2);
  }
};
} // namespace vhn
