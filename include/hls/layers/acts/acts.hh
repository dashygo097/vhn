#pragma once

#include "../decls.hh"
#include "./gelu_impl.hh"
#include "./relu_impl.hh"
#include "./sigmoid_impl.hh"

namespace hls_nn {
// Define Activation Functions
ELEMENTWISE_DEF(ReLU)
ELEMENTWISE_DEF(Sigmoid)
ELEMENTWISE_DEF(GeLU)
} // namespace hls_nn

namespace hls_tb {
// Define Testbenches for Activation Functions
ELEMENTWISE_TB_DEF(ReLU)
ELEMENTWISE_TB_DEF(Sigmoid)
ELEMENTWISE_TB_DEF(GeLU)
} // namespace hls_tb
