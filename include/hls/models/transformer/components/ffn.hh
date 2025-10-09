#pragma once

#include "../../../nn/nn.hh"
#include "../../../opt_level.hh"

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace hls_nn {
template <typename DType, const int D_MODEL, typename Config,
          OptLevel OPT_LEVEL = OPT_NONE>
class FFN {
public:
  using dtype = DType;
  static constexpr int d_model = D_MODEL;

  FFN() = default;
  ~FFN() = default;

  static void forward();

private:
};

} // namespace hls_nn
