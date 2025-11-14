#pragma once

#include "../base.hh"

#ifdef __VITIS_HLS__
#endif

namespace vhn {
template <typename DType, int N> class ReLUImpl {
public:
  using dtype = DType;
  static constexpr int n = N;

#ifndef __VITIS_HLS__
  static std::string type() { return "ReLU"; };
  static json hparams() {
    json j;

    j["n"] = n;

    return j;
  }
#endif

  static dtype kernel(const dtype x) {
    return x > dtype(0.0f) ? x : dtype(0.0f);
  }
};

} // namespace vhn
