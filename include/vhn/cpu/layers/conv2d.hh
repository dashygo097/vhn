#pragma once

#include "../tensor.hh"

namespace vhn::cpu {
template <typename T> class Conv2d {
public:
  Conv2d(size_t in_channels, size_t out_channels, size_t kernel_size,
         size_t stride = 1, size_t padding = 0, bool bias = true)
      : _in_channels(in_channels), _out_channels(out_channels),
        _kernel_size(kernel_size), _stride(stride), _padding(padding),
        _weights({out_channels, in_channels, kernel_size, kernel_size}),
        _bias({out_channels}), _enable_bias(bias) {
    _weights.fill(T(1));
    _bias.fill(T(0));
  }

  void load_weights(const std::vector<T> &weights,
                    const std::vector<T> &bias = {}) {
    _weights.load_weights(weights);
    if (_enable_bias)
      _bias.load_weights(bias);
  }

  Tensor<T> forward(const Tensor<T> &input) {
    if (input.ndim() != 3) {
      throw std::invalid_argument(
          "Conv2d2d forward expects 3D input [C, H, W]");
    }

    const size_t in_length = input.shape()[2];
    const size_t out_length =
        (in_length + 2 * _padding - _kernel_size) / _stride + 1;
    Tensor<T> output({_out_channels, out_length, out_length});
    for (size_t oc = 0; oc < _out_channels; ++oc) {
      for (size_t ol = 0; ol < out_length; ++ol) {
        T sum = 0;
        for (size_t ic = 0; ic < _in_channels; ++ic) {
          for (size_t kl = 0; kl < _kernel_size; ++kl) {
            int il = static_cast<int>(ol * _stride + kl) -
                     static_cast<int>(_padding);
            if (il >= 0 && il < static_cast<int>(in_length)) {
              sum += input.at({ic, static_cast<size_t>(il)}) *
                     _weights.at({oc, ic, kl, kl});
            }
          }
        }
        if (_enable_bias) {
          sum += _bias.at({oc});
        }
        output.at({oc, ol, ol}) = sum;
      }
    }
    return output;
  }

  Tensor<T> forward_batched(const Tensor<T> &input) {
    if (input.ndim() != 4) {
      throw std::invalid_argument(
          "Conv2d forward_batched expects 4D input [N, C, H, W]");
    }

    const size_t batch_size = input.shape()[0];
    const size_t in_height = input.shape()[2];
    const size_t in_width = input.shape()[3];
    const size_t out_height =
        (in_height + 2 * _padding - _kernel_size) / _stride + 1;
    const size_t out_width =
        (in_width + 2 * _padding - _kernel_size) / _stride + 1;
    Tensor<T> output({batch_size, _out_channels, out_height, out_width});
    for (size_t n = 0; n < batch_size; ++n) {
      for (size_t oc = 0; oc < _out_channels; ++oc) {
        for (size_t oh = 0; oh < out_height; ++oh) {
          for (size_t ow = 0; ow < out_width; ++ow) {
            T sum = 0;
            for (size_t ic = 0; ic < _in_channels; ++ic) {
              for (size_t kh = 0; kh < _kernel_size; ++kh) {
                for (size_t kw = 0; kw < _kernel_size; ++kw) {
                  int ih = static_cast<int>(oh * _stride + kh) -
                           static_cast<int>(_padding);
                  int iw = static_cast<int>(ow * _stride + kw) -
                           static_cast<int>(_padding);
                  if (ih >= 0 && ih < static_cast<int>(in_height) && iw >= 0 &&
                      iw < static_cast<int>(in_width)) {
                    sum += input.at({n, ic, static_cast<size_t>(ih),
                                     static_cast<size_t>(iw)}) *
                           _weights.at({oc, ic, kh, kw});
                  }
                }
              }
            }
            if (_enable_bias) {
              sum += _bias.at({oc});
            }
            output.at({n, oc, oh, ow}) = sum;
          }
        }
      }
    }
    return output;
  }

  [[nodiscard]] const Tensor<T> &weighst() const { return _weights; }
  [[nodiscard]] Tensor<T> &weights() { return _weights; }

  [[nodiscard]] const Tensor<T> &bias() const { return _bias; }
  [[nodiscard]] Tensor<T> &bias() { return _bias; }

private:
  size_t _in_channels;
  size_t _out_channels;
  size_t _kernel_size;
  size_t _stride;
  size_t _padding;
  bool _enable_bias;

  Tensor<T> _weights;
  Tensor<T> _bias;
};
} // namespace vhn::cpu
