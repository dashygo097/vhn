#pragma once

#include "../../opt_level.hh"

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace hls_nn {
template <typename DType, const int INPUT_DIM, const int OUTPUT_DIM,
          typename Config = void, OptLevel OPT_LEVEL = OPT_NONE>
class Embedding;

template <typename DType, const int INPUT_DIM, const int OUTPUT_DIM>
class Embedding<DType, INPUT_DIM, OUTPUT_DIM, void, OPT_NONE> {
  using dtype = DType;
  constexpr static int input_dim = INPUT_DIM;
  constexpr static int output_dim = OUTPUT_DIM;
  OptLevel opt_level = OPT_NONE;

  Embedding() = default;
  ~Embedding() = default;

private:
};

template <typename DType, const int INPUT_DIM, const int OUTPUT_DIM,
          typename Config>
class Embedding<DType, INPUT_DIM, OUTPUT_DIM, Config, OPT_ENABLED> {};
} // namespace hls_nn
