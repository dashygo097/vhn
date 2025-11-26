#pragma once

#include "../tensor.hh"
#include <cmath>

namespace vhn::cpu {

template <typename T> class SiLU {
public:
  void forward(Tensor<T> &input) {
    for (auto &elem : input.data()) {
      elem = elem / (T(1) + std::exp(-elem));
    }
  }
};

} // namespace vhn::cpu
