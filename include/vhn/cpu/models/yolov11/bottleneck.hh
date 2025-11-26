#pragma once

#include "../../tensor.hh"
#include "cbs.hh"

namespace vhn::cpu::yolov11 {

template <typename T> class Bottleneck {
public:
  Bottleneck(size_t in_channels, size_t out_channels)
      : _cbs1(in_channels, in_channels / 2, 1, 1, 0),
        _cbs2(in_channels / 2, out_channels, 3, 1, 1) {}

  void load_weights(
      const std::vector<T> &cbs1_conv_weights,
      const std::vector<T> &cbs1_bn_weight, const std::vector<T> &cbs1_bn_bias,
      const std::vector<T> &cbs1_bn_mean, const std::vector<T> &cbs1_bn_var,
      const std::vector<T> &cbs2_conv_weights,
      const std::vector<T> &cbs2_bn_weight, const std::vector<T> &cbs2_bn_bias,
      const std::vector<T> &cbs2_bn_mean, const std::vector<T> &cbs2_bn_var) {
    _cbs1.load_weights(cbs1_conv_weights, cbs1_bn_weight, cbs1_bn_bias,
                       cbs1_bn_mean, cbs1_bn_var);
    _cbs2.load_weights(cbs2_conv_weights, cbs2_bn_weight, cbs2_bn_bias,
                       cbs2_bn_mean, cbs2_bn_var);
  }

  Tensor<T> forward(const Tensor<T> &input) {
    auto x = _cbs1.forward(input);
    x = _cbs2.forward(x);

    if (input.shape() == x.shape()) {
      for (size_t i = 0; i < x.size(); ++i) {
        x.data()[i] += input.data()[i];
      }
    }
    return x;
  }

  Tensor<T> forward_batched(const Tensor<T> &input) {
    auto x = _cbs1.forward_batched(input);
    x = _cbs2.forward_batched(x);

    if (input.shape() == x.shape()) {
      for (size_t i = 0; i < x.size(); ++i) {
        x.data()[i] += input.data()[i];
      }
    }
    return x;
  }

  [[nodiscard]] const CBS<T> &cbs1() const { return _cbs1; }
  [[nodiscard]] CBS<T> &cbs1() { return _cbs1; }

  [[nodiscard]] const CBS<T> &cbs2() const { return _cbs2; }
  [[nodiscard]] CBS<T> &cbs2() { return _cbs2; }

private:
  CBS<T> _cbs1;
  CBS<T> _cbs2;
};

} // namespace vhn::cpu::yolov11
