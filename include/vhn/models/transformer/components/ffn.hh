#pragma once

#include "../../mlp.hh"

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace vhn {
template <typename FC1_Config = void, typename ACT_Config = void,
          typename FC2_Config = void>
struct FFNConfig {
  using fc1 = FC1_Config;
  using act = ACT_Config;
  using fc2 = FC2_Config;
};

template <typename FFNConfig> struct FFN_MLPConfig {
  using act = typename FFNConfig::act;
  template <int LayerIdx>
  using layer =
      typename std::conditional<LayerIdx == 0, typename FFNConfig::fc1,
                                typename FFNConfig::fc2>::type;
};

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

// ============================================================================
// FFN specialization for OPT_ENABLED with Config
// ============================================================================
template <typename DType, const int D_MODEL, const int D_FF,
          template <typename, int, typename, OptLevel> class ActLayer,
          typename Config>
class FFN<DType, D_MODEL, D_FF, ActLayer, Config, OPT_ENABLED> {
public:
  using dtype = DType;
  static constexpr int d_model = D_MODEL;
  static constexpr int d_ff = D_FF;
  static constexpr OptLevel opt_level = OPT_ENABLED;

  using W1_t = dtype[D_FF][D_MODEL];
  using b1_t = dtype[D_FF];
  using W2_t = dtype[D_MODEL][D_FF];
  using b2_t = dtype[D_MODEL];

  FFN() = default;
  ~FFN() = default;

  using MLPConfig = FFN_MLPConfig<Config>;

  using mlp =
      MLP<DType, ActLayer, MLPConfig, OPT_ENABLED, D_MODEL, D_FF, D_MODEL>;

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
