#pragma once

#include "../decls.hh"
#include "./gelu_impl.hh"
#include "./relu_impl.hh"
#include "./sigmoid_impl.hh"

namespace vhn {
// Define Activation Functions
ELEMENTWISE_REGISTRY(ReLU)
ELEMENTWISE_REGISTRY(Sigmoid)
ELEMENTWISE_REGISTRY(GeLU)
} // namespace vhn

namespace vhn::tb {
// Define Testbenches for Activation Functions
ELEMENTWISE_TB_REGISTRY(ReLU)
ELEMENTWISE_TB_REGISTRY(Sigmoid)
ELEMENTWISE_TB_REGISTRY(GeLU)
} // namespace vhn::tb
