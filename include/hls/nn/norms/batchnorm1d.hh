#pragma once

#include "../../opt_level.hh"

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace hls_nn {
template <typename DType, const int CHANNELS, typename Config,
          OptLevel OPT_LEVEL = OPT_NONE>
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

template <typename DType, const int CHANNELS, typename Config>
class BatchNorm1d<DType, CHANNELS, Config, OPT_NONE> {
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

template <typename DType, const int CHANNELS, typename Config>
class BatchNorm1d<DType, CHANNELS, Config, OPT_ENABLED> {
public:
  using dtype = DType;
  static constexpr int channels = CHANNELS;
  static constexpr OptLevel opt_level = OPT_ENABLED;

  static constexpr int unroll_factor = Config::_unroll_factor;
  static constexpr int partition_factor = Config::_partition_factor;
  static constexpr int pipeline_ii = Config::_pipeline_ii;

  BatchNorm1d() = default;
  ~BatchNorm1d() = default;

  static void forward(dtype output[][channels], const dtype input[][channels],
                      const dtype weight[channels], const dtype bias[channels],
                      const dtype running_mean[channels],
                      const dtype running_var[channels],
                      const dtype eps = dtype(1e-5)) {}

private:
};

} // namespace hls_nn
