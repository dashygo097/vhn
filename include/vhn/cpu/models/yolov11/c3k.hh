#pragma once

#include "../../tensor.hh"
#include "bottleneck.hh"
#include "cbs.hh"
#include <vector>

namespace vhn::cpu::yolov11 {

template <typename T> class C3k {
public:
  C3k(size_t in_channels, size_t out_channels, size_t num_blocks = 1)
      : _cbs1(in_channels, in_channels / 2, 1, 1, 0),
        _cbs2(in_channels, in_channels / 2, 1, 1, 0),
        _cbs3(in_channels, out_channels, 1, 1, 0), _num_blocks(num_blocks) {
    for (size_t i = 0; i < num_blocks; ++i) {
      _bottlenecks.emplace_back(in_channels / 2, in_channels / 2);
    }
  }

  void load_weights(
      const std::vector<std::vector<T>> &cbs1_weights,
      const std::vector<std::vector<T>> &cbs2_weights,
      const std::vector<std::vector<std::vector<T>>> &bottleneck_weights,
      const std::vector<std::vector<T>> &cbs3_weights) {
    _cbs1.load_weights(cbs1_weights[0], cbs1_weights[1], cbs1_weights[2],
                       cbs1_weights[3], cbs1_weights[4]);
    _cbs2.load_weights(cbs2_weights[0], cbs2_weights[1], cbs2_weights[2],
                       cbs2_weights[3], cbs2_weights[4]);

    for (size_t i = 0; i < _num_blocks; ++i) {
      _bottlenecks[i].load_weights(
          bottleneck_weights[i][0], bottleneck_weights[i][1],
          bottleneck_weights[i][2], bottleneck_weights[i][3],
          bottleneck_weights[i][4], bottleneck_weights[i][5],
          bottleneck_weights[i][6], bottleneck_weights[i][7],
          bottleneck_weights[i][8], bottleneck_weights[i][9]);
    }

    _cbs3.load_weights(cbs3_weights[0], cbs3_weights[1], cbs3_weights[2],
                       cbs3_weights[3], cbs3_weights[4]);
  }

  Tensor<T> forward(const Tensor<T> &input) {
    auto x1 = _cbs1.forward(input);
    auto x2 = _cbs2.forward(input);

    for (auto &bn : _bottlenecks) {
      x1 = bn.forward(x1);
    }

    size_t c_in = input.shape()[0];
    size_t h = input.shape()[1];
    size_t w = input.shape()[2];

    Tensor<T> concat({c_in, h, w});
    for (size_t i = 0; i < x1.size(); ++i) {
      concat.data()[i] = x1.data()[i];
    }
    for (size_t i = 0; i < x2.size(); ++i) {
      concat.data()[x1.size() + i] = x2.data()[i];
    }

    return _cbs3.forward(concat);
  }

  Tensor<T> forward_batched(const Tensor<T> &input) {
    auto x1 = _cbs1.forward_batched(input);
    auto x2 = _cbs2.forward_batched(input);

    for (auto &bn : _bottlenecks) {
      x1 = bn.forward_batched(x1);
    }

    size_t batch = input.shape()[0];
    size_t c_in = input.shape()[1];
    size_t h = input.shape()[2];
    size_t w = input.shape()[3];

    Tensor<T> concat({batch, c_in, h, w});
    for (size_t i = 0; i < x1.size(); ++i) {
      concat.data()[i] = x1.data()[i];
    }
    for (size_t i = 0; i < x2.size(); ++i) {
      concat.data()[x1.size() + i] = x2.data()[i];
    }

    return _cbs3.forward_batched(concat);
  }

  [[nodiscard]] const CBS<T> &cbs1() const { return _cbs1; }
  [[nodiscard]] CBS<T> &cbs1() { return _cbs1; }

  [[nodiscard]] const CBS<T> &cbs2() const { return _cbs2; }
  [[nodiscard]] CBS<T> &cbs2() { return _cbs2; }

  [[nodiscard]] const CBS<T> &cbs3() const { return _cbs3; }
  [[nodiscard]] CBS<T> &cbs3() { return _cbs3; }

  [[nodiscard]] const std::vector<Bottleneck<T>> &bottlenecks() const {
    return _bottlenecks;
  }

private:
  CBS<T> _cbs1;
  CBS<T> _cbs2;
  CBS<T> _cbs3;
  std::vector<Bottleneck<T>> _bottlenecks;
  size_t _num_blocks;
};

} // namespace vhn::cpu::yolov11
