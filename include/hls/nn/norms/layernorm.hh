#pragma once

#include "../../opt_level.hh"

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace hls_nn {
template <typename DType, const int HIDDEN_DIM, OptLevel OPT_LEVEL = OPT_NONE>
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

template <typename DType, const int HIDDEN_DIM>
class LayerNorm<DType, HIDDEN_DIM, OPT_NONE> {
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

template <typename DType, const int HIDDEN_DIM>
class LayerNorm<DType, HIDDEN_DIM, OPT_LATENCY> {
public:
  using dtype = DType;
  static constexpr int hidden_dim = HIDDEN_DIM;
  static constexpr OptLevel opt_level = OPT_LATENCY;

  LayerNorm() = default;
  ~LayerNorm() = default;

  static void forward(dtype output[hidden_dim], dtype input[hidden_dim],
                      const dtype gamma[hidden_dim],
                      const dtype beta[hidden_dim]) {}

private:
};

template <typename DType, const int HIDDEN_DIM>
class LayerNorm<DType, HIDDEN_DIM, OPT_THROUGHPUT> {
public:
  using dtype = DType;
  static constexpr int hidden_dim = HIDDEN_DIM;
  static constexpr OptLevel opt_level = OPT_THROUGHPUT;

  LayerNorm() = default;
  ~LayerNorm() = default;

  static void forward(dtype output[hidden_dim], dtype input[hidden_dim],
                      const dtype gamma[hidden_dim],
                      const dtype beta[hidden_dim]) {}

private:
};

template <typename DType, const int SEQ_LENGTH, const int HIDDEN_DIM>
class SequentialLayerNorm {
public:
  using dtype = DType;
  static constexpr int seq_length = SEQ_LENGTH;
  static constexpr int hidden_dim = HIDDEN_DIM;

  SequentialLayerNorm() = default;
  ~SequentialLayerNorm() = default;

  static void forward(dtype output[seq_length][hidden_dim],
                      dtype input[seq_length][hidden_dim],
                      const dtype gamma[hidden_dim],
                      const dtype beta[hidden_dim]) {}

private:
};

} // namespace hls_nn
