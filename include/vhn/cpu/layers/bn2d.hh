#pragma once

#include "../tensor.hh"
#include <cmath>

namespace vhn::cpu {

template <typename T> class BatchNorm2d {
public:
  BatchNorm2d(size_t num_features, T eps = 1e-5)
      : _num_features(num_features), _eps(eps), _weights({num_features}),
        _bias({num_features}), _running_mean({num_features}),
        _running_var({num_features}) {
    _weights.fill(T(1));
    _bias.fill(T(0));
    _running_mean.fill(T(0));
    _running_var.fill(T(1));
  }

  void load_weights(const std::vector<T> &weights, const std::vector<T> &bias,
                    const std::vector<T> &running_mean,
                    const std::vector<T> &running_var) {
    _weights.load_weights(weights);
    _bias.load_weights(bias);
    _running_mean.load_weights(running_mean);
    _running_var.load_weights(running_var);
  }

  Tensor<T> forward(const Tensor<T> &input) {
    if (input.ndim() != 3) {
      throw std::invalid_argument(
          "BatchNorm2d forward expects 3D input [C, H, W]");
    }

    const size_t channels = input.shape()[0];
    const size_t height = input.shape()[1];
    const size_t width = input.shape()[2];

    if (channels != _num_features) {
      throw std::invalid_argument("Input channels mismatch");
    }

    Tensor<T> output({channels, height, width});

    for (size_t c = 0; c < channels; ++c) {
      T normalized_val = std::sqrt(_running_var.at({c}) + _eps);
      for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
          T normalized =
              (input.at({c, h, w}) - _running_mean.at({c})) / normalized_val;
          output.at({c, h, w}) = _weights.at({c}) * normalized + _bias.at({c});
        }
      }
    }

    return output;
  }

  Tensor<T> forward_batched(const Tensor<T> &input) {
    if (input.ndim() != 4) {
      throw std::invalid_argument(
          "BatchNorm2d forward_batched expects 4D input [N, C, H, W]");
    }

    const size_t batch_size = input.shape()[0];
    const size_t channels = input.shape()[1];
    const size_t height = input.shape()[2];
    const size_t width = input.shape()[3];

    if (channels != _num_features) {
      throw std::invalid_argument("Input channels mismatch");
    }

    Tensor<T> output({batch_size, channels, height, width});

    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t c = 0; c < channels; ++c) {
        T normalized_val = std::sqrt(_running_var.at({c}) + _eps);
        for (size_t h = 0; h < height; ++h) {
          for (size_t w = 0; w < width; ++w) {
            T normalized = (input.at({n, c, h, w}) - _running_mean.at({c})) /
                           normalized_val;
            output.at({n, c, h, w}) =
                _weights.at({c}) * normalized + _bias.at({c});
          }
        }
      }
    }

    return output;
  }

  [[nodiscard]] const Tensor<T> &weights() const { return _weights; }
  [[nodiscard]] Tensor<T> &weights() { return _weights; }

  [[nodiscard]] const Tensor<T> &bias() const { return _bias; }
  [[nodiscard]] Tensor<T> &bias() { return _bias; }

  [[nodiscard]] const Tensor<T> &running_mean() const { return _running_mean; }
  [[nodiscard]] const Tensor<T> &running_var() const { return _running_var; }

private:
  size_t _num_features;
  T _eps;

  Tensor<T> _weights;
  Tensor<T> _bias;
  Tensor<T> _running_mean;
  Tensor<T> _running_var;
};

} // namespace vhn::cpu
