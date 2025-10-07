#pragma once

#include "../../opt_level.hh"

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace hls_nn {
template <typename DType, const int CHANNELS, const int H, const int W,
          OptLevel OPT_LEVEL = OPT_NONE>
class BatchNorm2d {
public:
  using dtype = DType;
  static constexpr int channels = CHANNELS;
  static constexpr int height = H;
  static constexpr int width = W;
  static constexpr OptLevel opt_level = OPT_LEVEL;

  BatchNorm2d() = default;
  ~BatchNorm2d() = default;

  static void forward(dtype output[][channels][height][width],
                      const dtype input[][channels][height][width],
                      const dtype weight[channels], const dtype bias[channels],
                      const dtype running_mean[channels],
                      const dtype running_var[channels],
                      const dtype eps = dtype(1e-5));

private:
};

template <typename DType, const int CHANNELS, const int H, const int W>
class BatchNorm2d<DType, CHANNELS, H, W, OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int channels = CHANNELS;
  static constexpr int height = H;
  static constexpr int width = W;
  static constexpr OptLevel opt_level = OPT_NONE;

  BatchNorm2d() = default;
  ~BatchNorm2d() = default;

  static void forward(dtype output[][channels][height][width],
                      const dtype input[][channels][height][width],
                      const dtype weight[channels], const dtype bias[channels],
                      const dtype running_mean[channels],
                      const dtype running_var[channels],
                      const dtype eps = dtype(1e-5)) {}

private:
};

template <typename DType, const int CHANNELS, const int H, const int W>
class BatchNorm2d<DType, CHANNELS, H, W, OPT_LATENCY> {
public:
  using dtype = DType;
  static constexpr int channels = CHANNELS;
  static constexpr int height = H;
  static constexpr int width = W;
  static constexpr OptLevel opt_level = OPT_LATENCY;

  BatchNorm2d() = default;
  ~BatchNorm2d() = default;

  static void forward(dtype output[][channels][height][width],
                      const dtype input[][channels][height][width],
                      const dtype weight[channels], const dtype bias[channels],
                      const dtype running_mean[channels],
                      const dtype running_var[channels],
                      const dtype eps = dtype(1e-5)) {}

private:
};

template <typename DType, const int CHANNELS, const int H, const int W>
class BatchNorm2d<DType, CHANNELS, H, W, OPT_THROUGHPUT> {
public:
  using dtype = DType;
  static constexpr int channels = CHANNELS;
  static constexpr int height = H;
  static constexpr int width = W;
  static constexpr OptLevel opt_level = OPT_THROUGHPUT;

  BatchNorm2d() = default;
  ~BatchNorm2d() = default;

  static void forward(dtype output[][channels][height][width],
                      const dtype input[][channels][height][width],
                      const dtype weight[channels], const dtype bias[channels],
                      const dtype running_mean[channels],
                      const dtype running_var[channels],
                      const dtype eps = dtype(1e-5)) {}

private:
};

template <typename DType, const int CHANNELS, const int H, const int W,
          const int LENGTH>
class SpacialBatchNorm2d {
public:
  using dtype = DType;
  static constexpr int channels = CHANNELS;
  static constexpr int height = H;
  static constexpr int width = W;
  static constexpr int length = LENGTH;
  static constexpr OptLevel opt_level = OPT_NONE;

  SpacialBatchNorm2d() = default;
  ~SpacialBatchNorm2d() = default;

  static void forward(dtype output[][channels][height][width][length],
                      const dtype input[][channels][height][width][length],
                      const dtype weight[channels], const dtype bias[channels],
                      const dtype running_mean[channels],
                      const dtype running_var[channels],
                      const dtype eps = dtype(1e-5)) {}

private:
};

} // namespace hls_nn
