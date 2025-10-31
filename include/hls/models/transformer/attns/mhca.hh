#pragma once
#include "../../../nn/nn.hh"
#include "../../../opt_level.hh"
#include <cmath>

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace hls_nn {
class MulHeadCrossAttn {};
} // namespace hls_nn
