#pragma once

#include "../../../hls_config.hh"
#include "../../../nn/nn.hh"
#include "../../../opt_level.hh"

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace hls_nn {
template <typename FC1_Config, typename ACT_Config, typename FC2_Config>
struct FFNConfig {
  using fc1 = FC1_Config;
  using act = ACT_Config;
  using fc2 = FC2_Config;
};

template <typename DType, const int D_MODEL, const int D_FF,
          template <typename, int, typename, OptLevel> class ActLayer,
          typename Config, OptLevel OPT_LEVEL = OPT_NONE>
class FFN {
public:
  using dtype = DType;
  static constexpr int d_model = D_MODEL;
  static constexpr int d_ff = D_FF;
  static constexpr OptLevel opt_level = OPT_LEVEL;

  using W1_t = dtype[D_FF][D_MODEL];
  using b1_t = dtype[D_FF];
  using W2_t = dtype[D_MODEL][D_FF];
  using b2_t = dtype[D_MODEL];

  FFN() = default;
  ~FFN() = default;

  static void forward(dtype output[][D_MODEL], const dtype input[][D_MODEL],
                      const W1_t w1, const b1_t b1, const W2_t w2,
                      const b2_t b2, const int actual_len);

private:
};

template <typename DType, const int D_MODEL, const int D_FF,
          template <typename, int, typename, OptLevel> class ActLayer>
class FFN<DType, D_MODEL, D_FF, ActLayer, void, OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int d_model = D_MODEL;
  static constexpr int d_ff = D_FF;
  static constexpr OptLevel opt_level = OPT_NONE;

  FFN() = default;
  ~FFN() = default;

  using W1_t = dtype[D_FF][D_MODEL];
  using b1_t = dtype[D_FF];
  using W2_t = dtype[D_MODEL][D_FF];
  using b2_t = dtype[D_MODEL];

  using fc1 = LinearBatched<DType, D_MODEL, D_FF, void, OPT_NONE>;
  using act = ActLayer<DType, D_FF, void, OPT_NONE>;
  using fc2 = LinearBatched<DType, D_FF, D_MODEL, void, OPT_NONE>;

  static void forward(dtype output[][D_MODEL], const dtype input[][D_MODEL],
                      const W1_t w1, const b1_t b1, const W2_t w2,
                      const b2_t b2, const int actual_len) {
#ifdef __VITIS_HLS_
#pragma HLS INLINE off
#endif
    dtype fc1_out[actual_len][d_ff];
    dtype act_out[actual_len][d_ff];

    fc1::forward(fc1_out, input, w1, b1, actual_len);
    act::forward(act_out, fc1_out, actual_len);
    fc2::forward(output, act_out, w2, b2, actual_len);
  }

private:
};

template <typename DType, const int D_MODEL, const int D_FF,
          template <typename, int, typename, OptLevel> class ActLayer,
          typename Config>
class FFN<DType, D_MODEL, D_FF, ActLayer, Config, OPT_ENABLED> {
public:
  using dtype = DType;
  static constexpr int d_model = D_MODEL;
  static constexpr int d_ff = D_FF;
  static constexpr OptLevel opt_level = OPT_ENABLED;

  using fc1_config = typename Config::fc1;
  using act_config = typename Config::act;
  using fc2_config = typename Config::fc2;

  FFN() = default;
  ~FFN() = default;

  using W1_t = dtype[D_FF][D_MODEL];
  using b1_t = dtype[D_FF];
  using W2_t = dtype[D_MODEL][D_FF];
  using b2_t = dtype[D_MODEL];

  using fc1 =
      LinearBatched<DType, D_MODEL, D_FF, decltype(Config::fc1), OPT_ENABLED>;
  using act = ActLayer<DType, D_FF, decltype(Config::act), OPT_ENABLED>;
  using fc2 =
      LinearBatched<DType, D_FF, D_MODEL, decltype(Config::fc2), OPT_ENABLED>;

  static void forward(dtype output[][D_MODEL], const dtype input[][D_MODEL],
                      const W1_t w1, const b1_t b1, const W2_t w2,
                      const b2_t b2, const int actual_len) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif

    dtype fc1_out[actual_len][d_ff];
    dtype act_out[actual_len][d_ff];

    fc1::forward(fc1_out, input, w1, b1, actual_len);
    act::forward(act_out, fc1_out, actual_len);
    fc2::forward(output, act_out, w2, b2, actual_len);
  }

private:
};

} // namespace hls_nn
