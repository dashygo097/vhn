#pragma once

#include "../tensor.hh"

namespace vhn::cpu {

template <typename T> class Upsample {
public:
  Upsample(size_t scale = 2) : _scale(scale) {}

  Tensor<T> forward(const Tensor<T> &input) {
    size_t c = input.shape()[0];
    size_t in_h = input.shape()[1];
    size_t in_w = input.shape()[2];

    size_t out_h = in_h * _scale;
    size_t out_w = in_w * _scale;

    Tensor<T> output({c, out_h, out_w});

    const auto &input_data = input.data();
    auto &output_data = output.data();

    for (size_t c_idx = 0; c_idx < c; ++c_idx) {
      for (size_t oh = 0; oh < out_h; ++oh) {
        for (size_t ow = 0; ow < out_w; ++ow) {
          size_t ih = oh / _scale;
          size_t iw = ow / _scale;

          size_t in_idx = c_idx * in_h * in_w + ih * in_w + iw;
          size_t out_idx = c_idx * out_h * out_w + oh * out_w + ow;

          output_data[out_idx] = input_data[in_idx];
        }
      }
    }

    return output;
  }

  Tensor<T> forward_batched(const Tensor<T> &input) {
    size_t batch = input.shape()[0];
    size_t c = input.shape()[1];
    size_t in_h = input.shape()[2];
    size_t in_w = input.shape()[3];

    size_t out_h = in_h * _scale;
    size_t out_w = in_w * _scale;

    Tensor<T> output({batch, c, out_h, out_w});

    const auto &input_data = input.data();
    auto &output_data = output.data();

    for (size_t n = 0; n < batch; ++n) {
      for (size_t c_idx = 0; c_idx < c; ++c_idx) {
        for (size_t oh = 0; oh < out_h; ++oh) {
          for (size_t ow = 0; ow < out_w; ++ow) {
            size_t ih = oh / _scale;
            size_t iw = ow / _scale;

            size_t in_idx =
                n * c * in_h * in_w + c_idx * in_h * in_w + ih * in_w + iw;
            size_t out_idx =
                n * c * out_h * out_w + c_idx * out_h * out_w + oh * out_w + ow;

            output_data[out_idx] = input_data[in_idx];
          }
        }
      }
    }

    return output;
  }

  [[nodiscard]] size_t scale() const { return _scale; }

private:
  size_t _scale;
};

} // namespace vhn::cpu
