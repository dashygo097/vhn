#pragma once

#include "../../mlp.hh"

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace vhn {

template <typename DType, const int D_MODEL, const int D_FF,
          template <typename, int, typename, OptLevel> class ActLayer,
          typename Config = void, OptLevel OPT_LEVEL = OPT_NONE>
class FFN;

// ============================================================================
// FFN specialization for OPT_NONE (using MLP)
// ============================================================================
template <typename DType, const int D_MODEL, const int D_FF,
          template <typename, int, typename, OptLevel> class ActLayer>
class FFN<DType, D_MODEL, D_FF, ActLayer, void, OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int d_model = D_MODEL;
  static constexpr int d_ff = D_FF;
  static constexpr OptLevel opt_level = OPT_NONE;

  using W1_t = dtype[D_FF][D_MODEL];
  using b1_t = dtype[D_FF];
  using W2_t = dtype[D_MODEL][D_FF];
  using b2_t = dtype[D_MODEL];

  FFN() = default;
  ~FFN() = default;

  using mlp = MLP<DType, ActLayer, void, OPT_NONE, D_MODEL, D_FF, D_MODEL>;

  static void forward(dtype output[][D_MODEL], const dtype input[][D_MODEL],
                      const int actual_len, const W1_t w1, const b1_t b1,
                      const W2_t w2, const b2_t b2) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    mlp::forward(output, input, actual_len, w1, b1, w2, b2);
  }

  static void forward(dtype output[D_MODEL], const dtype input[D_MODEL],
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
