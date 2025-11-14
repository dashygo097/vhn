#pragma once

#include "./operator_impl.hh"
#include "./registry.hh"

namespace vhn {
// Define Common Operators
ELEMENTWISE_REGISTRY(Add)
ELEMENTWISE_REGISTRY(Sub)
ELEMENTWISE_REGISTRY(Mul)

REDUCE_REGISTRY(Max)
REDUCE_REGISTRY(Min)
REDUCE_REGISTRY(Sum)
REDUCE_REGISTRY(Mean)
} // namespace vhn

namespace vhn::tb {
// Define Testbenches for Common Operators
ELEMENTWISE_TB_REGISTRY(Add)
ELEMENTWISE_TB_REGISTRY(Sub)

REDUCE_TB_REGISTRY(Max)
REDUCE_TB_REGISTRY(Min)
REDUCE_TB_REGISTRY(Sum)
REDUCE_TB_REGISTRY(Mean)
} // namespace vhn::tb
