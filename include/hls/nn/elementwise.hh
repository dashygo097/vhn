#pragma once

#include "../opt_level.hh"

#ifdef __VITIS_HLS__
#endif

namespace hls_nn {
template <const int UNROLL_FACTOR, const int PARTITION_FACTOR,
          const int PIPELINE_II>
struct ElementwiseHLSConfig {
  static constexpr int _unroll_factor = UNROLL_FACTOR;
  static constexpr int _partition_factor = PARTITION_FACTOR;
  static constexpr int _pipeline_ii = PIPELINE_II;
};

template <typename DType, typename ImplType, int N, typename Config,
          OptLevel OPT_LEVEL = OPT_NONE>
class Elementwise {
public:
  using dtype = DType;
  using impl = ImplType;
  static constexpr int n = N;
  static constexpr OptLevel opt_level = OPT_LEVEL;

  Elementwise() = default;
  ~Elementwise() = default;

  static void forward(dtype output[n], const dtype input[n]);
  static void forward(dtype output[n], const dtype input1[n],
                      const dtype input2[n]);

private:
};

template <typename DType, typename ImplType, int N>
class Elementwise<DType, ImplType, N, void, OPT_NONE> {
public:
  using dtype = DType;
  using impl = ImplType;
  static constexpr int n = N;
  static constexpr OptLevel opt_level = OPT_NONE;

  Elementwise() = default;
  ~Elementwise() = default;

  static void forward(dtype output[n], const dtype input[n]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
  ELEMENTWISE_LOOP:
    for (int i = 0; i < n; i++) {
      output[i] = impl::kernel(input[i]);
    }
  }

  static void forward(dtype output[n], const dtype input1[n],
                      const dtype input2[n]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
  ELEMEMTWISE_LOOP_2:
    for (int i = 0; i < n; i++) {
      output[i] = impl::kernel(input1[i], input2[i]);
    }
  }

private:
};

template <typename DType, typename ImplType, int N, typename Config>
class Elementwise<DType, ImplType, N, Config, OPT_ENABLED> {
public:
  using dtype = DType;
  using impl = ImplType;
  static constexpr int n = N;
  static constexpr OptLevel opt_level = OPT_ENABLED;

  static constexpr int unroll_factor = Config::_unroll_factor;
  static constexpr int partition_factor = Config::_partition_factor;
  static constexpr int pipeline_ii = Config::_pipeline_ii;

  Elementwise() = default;
  ~Elementwise() = default;

  static void forward(dtype output[n], const dtype input[n]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = output factor = partition_factor dim = 1
#pragma HLS ARRAY_PARTITION variable = input factor = partition_factor dim = 1
#endif
  ELEMENTWISE_LOOP:
    for (int i = 0; i < n; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#pragma HLS UNROLL factor = unroll_factor
#endif
      output[i] = impl::kernel(input[i]);
    }
  }

  static void forward(dtype output[n], const dtype input1[n],
                      const dtype input2[n]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = output factor = partition_factor dim = 1
#pragma HLS ARRAY_PARTITION variable = input1 factor = partition_factor dim = 1
#endif
  ELEMEMTWISE_LOOP_2:
    for (int i = 0; i < n; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#pragma HLS UNROLL factor = unroll_factor
#endif
      output[i] = impl::kernel(input1[i], input2[i]);
    }
  }

private:
};

} // namespace hls_nn
