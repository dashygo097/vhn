#pragma once

#include "../../opt_level.hh"
#include <cmath>

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace hls_nn {
template <const int UNROLL_FACTOR, const int PARTITION_FACTOR,
          const int PIPELINE_II>
struct SoftmaxHLSConfig {
  static constexpr int _unroll_factor = UNROLL_FACTOR;
  static constexpr int _partition_factor = PARTITION_FACTOR;
  static constexpr int _pipeline_ii = PIPELINE_II;
};

template <typename DType, int N, typename Config, OptLevel OPT_LEVEL = OPT_NONE>
class Softmax {
public:
  using dtype = DType;
  static constexpr int n = N;
  static constexpr OptLevel opt_level = OPT_LEVEL;

  Softmax() = default;
  ~Softmax() = default;

  static void forward(dtype output[n], const dtype input[n]);

private:
};

template <typename DType, int N> class Softmax<DType, N, void, OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int n = N;
  static constexpr OptLevel opt_level = OPT_NONE;

  static void forward(dtype output[n], dtype input[n]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    dtype max_val = input[0];
  FIND_MAX:
    for (int i = 1; i < n; i++) {
      max_val = (input[i] > max_val) ? input[i] : max_val;
    }

    dtype sum = dtype(0.0f);
    dtype exp_val[n];
  CALC_EXP:
    for (int i = 0; i < n; i++) {
#ifdef __VITIS_HLS__
      exp_val[i] = hls::exp(input[i] - max_val);
#else
      exp_val[i] = std::exp(input[i] - max_val);
#endif
      sum += exp_val[i];
    }

    dtype inv_sum = dtype(1.0) / sum;
  NORMALIZE:
    for (int i = 0; i < n; i++) {
      output[i] = exp_val[i] * inv_sum;
    }
  }

private:
};

template <typename DType, int N, typename Config>
class Softmax<DType, N, Config, OPT_ENABLED> {
public:
  using dtype = DType;
  static constexpr int n = N;
  static constexpr OptLevel opt_level = OPT_ENABLED;

  static constexpr int unroll_factor = Config::_unroll_factor;
  static constexpr int partition_factor = Config::_partition_factor;
  static constexpr int pipeline_ii = Config::_pipeline_ii;

  Softmax() = default;
  ~Softmax() = default;

  static void forward(dtype output[n], dtype input[n]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = input cyclic factor = partition_factor
#pragma HLS ARRAY_PARTITION variable = output cyclic factor = partition_factor
#endif

    dtype max_val = input[0];
  FIND_MAX:
    for (int i = 1; i < n; i++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = unroll_factor
#endif
      max_val = (input[i] > max_val) ? input[i] : max_val;
    }

    dtype sum = dtype(0.0f);
    dtype exp_val[n];
  CALC_EXP:
    for (int i = 0; i < n; i++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = unroll_factor
#endif
#ifdef __VITIS_HLS__
      exp_val[i] = hls::exp(input[i] - max_val);
#else
      exp_val[i] = std::exp(input[i] - max_val);
#endif
      sum += exp_val[i];
    }

    dtype inv_sum = dtype(1.0) / sum;
  NORMALIZE:
    for (int i = 0; i < n; i++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = unroll_factor
#endif
      output[i] = exp_val[i] * inv_sum;
    }
  }
};

template <typename DType, int N, typename Config, OptLevel OPT_LEVEL = OPT_NONE>
class SoftmaxBatched {
public:
  using dtype = DType;
  static constexpr int n = N;
  static constexpr OptLevel opt_level = OPT_LEVEL;

  SoftmaxBatched() = default;
  ~SoftmaxBatched() = default;

  static void forward(dtype output[][n], const dtype input[][n],
                      int batch_size) {
#ifdef __VITIS_HLS__
#pragma HLS DATAFLOW
#endif

  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif

      Softmax<DType, N, Config, OPT_LEVEL>::forward(output[b], input[b]);
    }
  }
};

} // namespace hls_nn
