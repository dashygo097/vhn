#pragma once

#include "./decls.hh"
#include "./operator_impl.hh"

namespace hls_nn {
// Define Common Operators
ELEMENTWISE_DEF(Add)
ELEMENTWISE_DEF(Sub)
ELEMENTWISE_DEF(Mul)

REDUCE_DEF(Max)
REDUCE_DEF(Min)
REDUCE_DEF(Sum)
REDUCE_DEF(Mean)
} // namespace hls_nn

namespace hls_tb {
// Define Testbenches for Common Operators
ELEMENTWISE_TB_DEF(Add)
ELEMENTWISE_TB_DEF(Sub)

REDUCE_TB_DEF(Max)
REDUCE_TB_DEF(Min)
REDUCE_TB_DEF(Sum)
REDUCE_TB_DEF(Mean)
} // namespace hls_tb
