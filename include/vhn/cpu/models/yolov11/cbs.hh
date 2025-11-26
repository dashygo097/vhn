#pragma once

#include "../../layers/bn2d.hh"
#include "../../layers/conv2d.hh"
#include "../../layers/silu.hh"
#include "../../tensor.hh"

namespace vhn::cpu::yolov11 {
template <typename T> class CBS {
public:
  CBS(size_t in_channels, size_t out_channels, size_t kernel_size,
      size_t stride = 1, size_t padding = 0)
      : _conv(in_channels, out_channels, kernel_size, stride, padding, false),
        _bn(out_channels) {}

  void load_weights(const std::vector<T> &conv_weights,
                    const std::vector<T> &bn_weight,
                    const std::vector<T> &bn_bias,
                    const std::vector<T> &bn_mean,
                    const std::vector<T> &bn_var) {
    _conv.load_weights(conv_weights);
    _bn.load_weights(bn_weight, bn_bias, bn_mean, bn_var);
  }

  Tensor<T> forward(const Tensor<T> &input) {
    auto x = _conv.forward(input);
    x = _bn.forward(x);
    _silu.forward(x);
    return x;
  }

  Tensor<T> forward_batched(const Tensor<T> &input) {
    auto x = _conv.forward_batched(input);
    x = _bn.forward_batched(x);
    _silu.forward(x);
    return x;
  }

  [[nodiscard]] const Conv2d<T> &conv() const { return _conv; }
  [[nodiscard]] Conv2d<T> &conv() { return _conv; }

  [[nodiscard]] const BatchNorm2d<T> &bn() const { return _bn; }
  [[nodiscard]] BatchNorm2d<T> &bn() { return _bn; }

  [[nodiscard]] const SiLU<T> &silu() const { return _silu; }
  [[nodiscard]] SiLU<T> &silu() { return _silu; }

private:
  Conv2d<T> _conv;
  BatchNorm2d<T> _bn;
  SiLU<T> _silu;
};

} // namespace vhn::cpu::yolov11
