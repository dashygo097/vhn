#pragma once

#include "../../layers/maxpool2d.hh"
#include "../../tensor.hh"
#include "cbs.hh"
#include <vector>

namespace vhn::cpu::yolov11 {

template <typename T> class SPPF {
public:
  SPPF(size_t in_channels, size_t out_channels, size_t kernel_size = 5)
      : _cbs1(in_channels, in_channels / 2, 1, 1, 0),
        _cbs2(in_channels * 2, out_channels, 1, 1, 0),
        _maxpool(kernel_size, 1, kernel_size / 2) {}

  void load_weights(const std::vector<std::vector<T>> &cbs1_weights,
                    const std::vector<std::vector<T>> &cbs2_weights) {
    _cbs1.load_weights(cbs1_weights[0], cbs1_weights[1], cbs1_weights[2],
                       cbs1_weights[3], cbs1_weights[4]);
    _cbs2.load_weights(cbs2_weights[0], cbs2_weights[1], cbs2_weights[2],
                       cbs2_weights[3], cbs2_weights[4]);
  }

  Tensor<T> forward(const Tensor<T> &input) {
    auto x = _cbs1.forward(input);

    auto y1 = _maxpool.forward(x);
    auto y2 = _maxpool.forward(y1);
    auto y3 = _maxpool.forward(y2);

    size_t c = x.shape()[0];
    size_t h = x.shape()[1];
    size_t w = x.shape()[2];

    Tensor<T> concat({c * 4, h, w});

    size_t offset = 0;
    for (const auto &tensor : {x, y1, y2, y3}) {
      for (size_t i = 0; i < tensor.size(); ++i) {
        concat.data()[offset + i] = tensor.data()[i];
      }
      offset += tensor.size();
    }

    return _cbs2.forward(concat);
  }

  Tensor<T> forward_batched(const Tensor<T> &input) {
    auto x = _cbs1.forward_batched(input);

    auto y1 = _maxpool.forward_batched(x);
    auto y2 = _maxpool.forward_batched(y1);
    auto y3 = _maxpool.forward_batched(y2);

    size_t batch = x.shape()[0];
    size_t c = x.shape()[1];
    size_t h = x.shape()[2];
    size_t w = x.shape()[3];

    Tensor<T> concat({batch, c * 4, h, w});

    size_t offset = 0;
    for (const auto &tensor : {x, y1, y2, y3}) {
      for (size_t i = 0; i < tensor.size(); ++i) {
        concat.data()[offset + i] = tensor.data()[i];
      }
      offset += tensor.size();
    }

    return _cbs2.forward_batched(concat);
  }

  [[nodiscard]] const CBS<T> &cbs1() const { return _cbs1; }
  [[nodiscard]] CBS<T> &cbs1() { return _cbs1; }

  [[nodiscard]] const CBS<T> &cbs2() const { return _cbs2; }
  [[nodiscard]] CBS<T> &cbs2() { return _cbs2; }

  [[nodiscard]] const MaxPool2d<T> &maxpool() const { return _maxpool; }
  [[nodiscard]] MaxPool2d<T> &maxpool() { return _maxpool; }

private:
  CBS<T> _cbs1;
  CBS<T> _cbs2;
  MaxPool2d<T> _maxpool;
};

} // namespace vhn::cpu::yolov11
