#pragma once

#include "./decls.hh"
#include "./gelu_impl.hh"
#include "./relu_impl.hh"
#include "./sigmoid_impl.hh"

namespace hls_nn {
// Define Activation Functions
ACT_DEF(ReLU)
ACT_DEF(Sigmoid)
ACT_DEF(GeLU)
} // namespace hls_nn

namespace hls_tb {
// Define Testbenches for Activation Functions
ACT_TB_DEF(ReLU)
ACT_TB_DEF(Sigmoid)
ACT_TB_DEF(GeLU)
} // namespace hls_tb
