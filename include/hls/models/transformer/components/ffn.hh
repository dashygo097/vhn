#pragma once

#include "../../../nn/nn.hh"
#include "../../../opt_level.hh"

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace hls_nn {
template <typename DType, const int D_MODEL, const int D_FF, typename Config,
          OptLevel OPT_LEVEL = OPT_NONE>
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
                      W1_t w1, b1_t b1, W2_t w2, b2_t b2);

private:
};

template <typename DType, const int D_MODEL, const int D_FF>
class FFN<DType, D_MODEL, D_FF, void, OPT_NONE> {
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

  static void forward(dtype output[][D_MODEL], const dtype input[][D_MODEL],
                      W1_t w1, b1_t b1, W2_t w2, b2_t b2);

private:
};

template <typename DType, const int D_MODEL, const int D_FF, typename Config>
class FFN<DType, D_MODEL, D_FF, Config, OPT_ENABLED> {
public:
  using dtype = DType;
  static constexpr int d_model = D_MODEL;
  static constexpr int d_ff = D_FF;
  static constexpr OptLevel opt_level = OPT_ENABLED;

  FFN() = default;
  ~FFN() = default;

  using W1_t = dtype[D_FF][D_MODEL];
  using b1_t = dtype[D_FF];
  using W2_t = dtype[D_MODEL][D_FF];
  using b2_t = dtype[D_MODEL];

  static void forward(dtype output[][D_MODEL], const dtype input[][D_MODEL],
                      W1_t w1, b1_t b1, W2_t w2, b2_t b2);

private:
};

} // namespace hls_nn
