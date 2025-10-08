#pragma once

#include "../../opt_level.hh"

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace hls_nn {
template <const int UNROLL_FACTOR, const int PARTITION_FACTOR,
          const int PIPELINE_II>
struct LayerNormHLSConfig {
  static constexpr int _unroll_factor = UNROLL_FACTOR;
  static constexpr int _partition_factor = PARTITION_FACTOR;
  static constexpr int _pipeline_ii = PIPELINE_II;
};

template <typename DType, const int HIDDEN_DIM, typename Config,
          OptLevel OPT_LEVEL = OPT_NONE>
class LayerNorm {
public:
  using dtype = DType;
  static constexpr int hidden_dim = HIDDEN_DIM;
  static constexpr OptLevel opt_level = OPT_LEVEL;

  LayerNorm() = default;
  ~LayerNorm() = default;

  static void forward(dtype output[hidden_dim], dtype input[hidden_dim],
                      const dtype gamma[hidden_dim],
                      const dtype beta[hidden_dim]);

private:
};

template <typename DType, const int HIDDEN_DIM, typename Config>
class LayerNorm<DType, HIDDEN_DIM, Config, OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int hidden_dim = HIDDEN_DIM;
  static constexpr OptLevel opt_level = OPT_NONE;

  LayerNorm() = default;
  ~LayerNorm() = default;

  static void forward(dtype output[hidden_dim], dtype input[hidden_dim],
                      const dtype gamma[hidden_dim],
                      const dtype beta[hidden_dim]) {}

private:
};

template <typename DType, const int HIDDEN_DIM, typename Config>
class LayerNorm<DType, HIDDEN_DIM, Config, OPT_ENABLED> {
public:
  using dtype = DType;
  static constexpr int hidden_dim = HIDDEN_DIM;
  static constexpr OptLevel opt_level = OPT_ENABLED;

  static constexpr int unroll_factor = Config::_unroll_factor;
  static constexpr int partition_factor = Config::_partition_factor;
  static constexpr int pipeline_ii = Config::_pipeline_ii;

  LayerNorm() = default;
  ~LayerNorm() = default;

  static void forward(dtype output[hidden_dim], dtype input[hidden_dim],
                      const dtype gamma[hidden_dim],
                      const dtype beta[hidden_dim]) {}

private:
};

} // namespace hls_nn
