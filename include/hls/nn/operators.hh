#pragma once

#include "./decls.hh"
#include "./operator_impl.hh"

namespace hls_nn {
// Define Common Operators
ELEMENTWISE_DEF(Add);
ELEMENTWISE_DEF(Sub);
} // namespace hls_nn

namespace hls_tb {
// Define Testbenches for Common Operators
ELEMENTWISE_TB_DEF(Add);
ELEMENTWISE_TB_DEF(Sub);
} // namespace hls_tb
