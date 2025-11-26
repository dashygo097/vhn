#pragma once

#include "../../tensor.hh"
#include <vector>

namespace vhn::cpu::yolov11 {

template <typename T> class Concat {
public:
  explicit Concat(size_t axis = 1) : _axis(axis) {}

  Tensor<T> forward(const std::vector<Tensor<T>> &inputs) {
    if (inputs.empty()) {
      throw std::invalid_argument("Concat requires at least one input");
    }

    if (_axis == 1 && inputs[0].ndim() == 3) {
      size_t h = inputs[0].shape()[1];
      size_t w = inputs[0].shape()[2];
      size_t total_c = 0;

      for (const auto &input : inputs) {
        if (input.shape()[1] != h || input.shape()[2] != w) {
          throw std::invalid_argument("Concat: shape mismatch");
        }
        total_c += input.shape()[0];
      }

      Tensor<T> output({total_c, h, w});
      size_t offset = 0;

      for (const auto &input : inputs) {
        for (size_t i = 0; i < input.size(); ++i) {
          output.data()[offset + i] = input.data()[i];
        }
        offset += input.size();
      }

      return output;
    } else if (_axis == 1 && inputs[0].ndim() == 4) {
      size_t batch = inputs[0].shape()[0];
      size_t h = inputs[0].shape()[2];
      size_t w = inputs[0].shape()[3];
      size_t total_c = 0;

      for (const auto &input : inputs) {
        if (input.shape()[0] != batch || input.shape()[2] != h ||
            input.shape()[3] != w) {
          throw std::invalid_argument("Concat: shape mismatch");
        }
        total_c += input.shape()[1];
      }

      Tensor<T> output({batch, total_c, h, w});
      size_t offset = 0;

      for (const auto &input : inputs) {
        for (size_t i = 0; i < input.size(); ++i) {
          output.data()[offset + i] = input.data()[i];
        }
        offset += input.size();
      }

      return output;
    }

    throw std::invalid_argument("Concat: unsupported configuration");
  }

  [[nodiscard]] size_t axis() const { return _axis; }

private:
  size_t _axis;
};

} // namespace vhn::cpu::yolov11
