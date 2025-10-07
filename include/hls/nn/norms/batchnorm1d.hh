#pragma once

#include "../../opt_level.hh"

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace hls_nn {
template <typename DType, const int CHANNELS, OptLevel OPT_LEVEL = OPT_NONE>
class BatchNorm1d {
public:
  using dtype = DType;
  static constexpr int channels = CHANNELS;
  static constexpr OptLevel opt_level = OPT_LEVEL;

  BatchNorm1d() = default;
  ~BatchNorm1d() = default;

  static void forward(dtype output[][channels], const dtype input[][channels],
                      const dtype weight[channels], const dtype bias[channels],
                      const dtype running_mean[channels],
                      const dtype running_var[channels],
                      const dtype eps = dtype(1e-5));

private:
};

template <typename DType, const int CHANNELS>
class BatchNorm1d<DType, CHANNELS, OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int channels = CHANNELS;
  static constexpr OptLevel opt_level = OPT_NONE;

  BatchNorm1d() = default;
  ~BatchNorm1d() = default;

  static void forward(dtype output[][channels], const dtype input[][channels],
                      const dtype weight[channels], const dtype bias[channels],
                      const dtype running_mean[channels],
                      const dtype running_var[channels],
                      const dtype eps = dtype(1e-5)) {}

private:
};

template <typename DType, const int CHANNELS>
class BatchNorm1d<DType, CHANNELS, OPT_LATENCY> {
public:
  using dtype = DType;
  static constexpr int channels = CHANNELS;
  static constexpr OptLevel opt_level = OPT_LATENCY;

  BatchNorm1d() = default;
  ~BatchNorm1d() = default;

  static void forward(dtype output[][channels], const dtype input[][channels],
                      const dtype weight[channels], const dtype bias[channels],
                      const dtype running_mean[channels],
                      const dtype running_var[channels],
                      const dtype eps = dtype(1e-5)) {}

private:
};

template <typename DType, const int CHANNELS>
class BatchNorm1d<DType, CHANNELS, OPT_THROUGHPUT> {
public:
  using dtype = DType;
  static constexpr int channels = CHANNELS;
  static constexpr OptLevel opt_level = OPT_THROUGHPUT;

  BatchNorm1d() = default;
  ~BatchNorm1d() = default;

  static void forward(dtype output[][channels], const dtype input[][channels],
                      const dtype weight[channels], const dtype bias[channels],
                      const dtype running_mean[channels],
                      const dtype running_var[channels],
                      const dtype eps = dtype(1e-5)) {}

private:
};

template <typename DType, const int CHANNELS, const int LENGTH>
class SpacialBatchNorm1d {
public:
  using dtype = DType;
  static constexpr int channels = CHANNELS;
  static constexpr int length = LENGTH;
  static constexpr OptLevel opt_level = OPT_NONE;

  static void forward(dtype output[][channels][length],
                      const dtype input[][channels][length],
                      const dtype weight[channels], const dtype bias[channels],
                      const dtype running_mean[channels],
                      const dtype running_var[channels],
                      const dtype eps = dtype(1e-5)) {}

private:
};

} // namespace hls_nn
