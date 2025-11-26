#pragma once

#include "../tensor.hh"
#include <algorithm>
#include <limits>

namespace vhn::cpu {

template <typename T> class MaxPool2d {
public:
  MaxPool2d(size_t kernel_size = 1, size_t stride = 1, size_t padding = 0)
      : _kernel_size(kernel_size), _stride(stride), _padding(padding) {}

  Tensor<T> forward(const Tensor<T> &input) {
    size_t channels = input.shape()[0];
    size_t in_h = input.shape()[1];
    size_t in_w = input.shape()[2];

    size_t out_h = (in_h + 2 * _padding - _kernel_size) / _stride + 1;
    size_t out_w = (in_w + 2 * _padding - _kernel_size) / _stride + 1;

    Tensor<T> output({channels, out_h, out_w});

    const auto &input_data = input.data();
    auto &output_data = output.data();

    for (size_t c = 0; c < channels; ++c) {
      for (size_t oh = 0; oh < out_h; ++oh) {
        for (size_t ow = 0; ow < out_w; ++ow) {
          T max_val = -std::numeric_limits<T>::max();

          for (size_t kh = 0; kh < _kernel_size; ++kh) {
            for (size_t kw = 0; kw < _kernel_size; ++kw) {
              int ih = static_cast<int>(oh * _stride + kh) -
                       static_cast<int>(_padding);
              int iw = static_cast<int>(ow * _stride + kw) -
                       static_cast<int>(_padding);

              if (ih >= 0 && ih < static_cast<int>(in_h) && iw >= 0 &&
                  iw < static_cast<int>(in_w)) {
                size_t in_idx = c * in_h * in_w + ih * in_w + iw;
                max_val = std::max(max_val, input_data[in_idx]);
              }
            }
          }

          size_t out_idx = c * out_h * out_w + oh * out_w + ow;
          output_data[out_idx] = max_val;
        }
      }
    }

    return output;
  }

  Tensor<T> forward_batched(const Tensor<T> &input) {
    size_t batch = input.shape()[0];
    size_t channels = input.shape()[1];
    size_t in_h = input.shape()[2];
    size_t in_w = input.shape()[3];

    size_t out_h = (in_h + 2 * _padding - _kernel_size) / _stride + 1;
    size_t out_w = (in_w + 2 * _padding - _kernel_size) / _stride + 1;

    Tensor<T> output({batch, channels, out_h, out_w});

    const auto &input_data = input.data();
    auto &output_data = output.data();

    for (size_t n = 0; n < batch; ++n) {
      for (size_t c = 0; c < channels; ++c) {
        for (size_t oh = 0; oh < out_h; ++oh) {
          for (size_t ow = 0; ow < out_w; ++ow) {
            T max_val = -std::numeric_limits<T>::max();

            for (size_t kh = 0; kh < _kernel_size; ++kh) {
              for (size_t kw = 0; kw < _kernel_size; ++kw) {
                int ih = static_cast<int>(oh * _stride + kh) -
                         static_cast<int>(_padding);
                int iw = static_cast<int>(ow * _stride + kw) -
                         static_cast<int>(_padding);

                if (ih >= 0 && ih < static_cast<int>(in_h) && iw >= 0 &&
                    iw < static_cast<int>(in_w)) {
                  size_t in_idx = n * channels * in_h * in_w + c * in_h * in_w +
                                  ih * in_w + iw;
                  max_val = std::max(max_val, input_data[in_idx]);
                }
              }
            }

            size_t out_idx = n * channels * out_h * out_w + c * out_h * out_w +
                             oh * out_w + ow;
            output_data[out_idx] = max_val;
          }
        }
      }
    }

    return output;
  }

  [[nodiscard]] size_t kernel_size() const { return _kernel_size; }
  [[nodiscard]] size_t stride() const { return _stride; }
  [[nodiscard]] size_t padding() const { return _padding; }

private:
  size_t _kernel_size;
  size_t _stride;
  size_t _padding;
};

} // namespace vhn::cpu
