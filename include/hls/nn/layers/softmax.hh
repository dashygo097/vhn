#pragma once

#include "../../opt_level.hh"
#include <cmath>

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace hls_nn {
template <typename DType, int N, OptLevel OPT_LEVEL = OPT_NONE> class Softmax {
public:
  using dtype = DType;
  static const int n = N;
  static const OptLevel opt_level = OPT_LEVEL;

  Softmax() = default;
  ~Softmax() = default;

  static void forward(dtype output[n], const dtype input[n]);

private:
};

template <typename DType, int N> class Softmax<DType, N, OPT_NONE> {
public:
  using dtype = DType;
  static const int n = N;
  static const OptLevel opt_level = OPT_NONE;

  static void forward(dtype output[n], dtype input[n]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    dtype max_val = input[0];
  FIND_MAX:
    for (int i = 1; i < n; i++) {
      max_val = (input[i] > max_val) ? input[i] : max_val;
    }

    dtype sum = 0;
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

template <typename DType, int N> class Softmax<DType, N, OPT_LATENCY> {
public:
  using dtype = DType;
  static const int n = N;
  static const OptLevel opt_level = OPT_LATENCY;

  Softmax() = default;
  ~Softmax() = default;

  static void forward(dtype output[n], dtype input[n]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = input complete
#pragma HLS ARRAY_PARTITION variable = output complete
#endif

    dtype max_val = input[0];
  FIND_MAX:
    for (int i = 1; i < n; i++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL
#endif
      max_val = (input[i] > max_val) ? input[i] : max_val;
    }

    dtype sum = 0;
    dtype exp_val[n];
  CALC_EXP:
    for (int i = 0; i < n; i++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL
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
#pragma HLS UNROLL
#endif
      output[i] = exp_val[i] * inv_sum;
    }
  }
};

template <typename DType, int N> class Softmax<DType, N, OPT_THROUGHPUT> {
public:
  using dtype = DType;
  static const int n = N;
  static const OptLevel opt_level = OPT_THROUGHPUT;

  Softmax() = default;
  ~Softmax() = default;

  static void forward(dtype output[n], dtype input[n]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = input cyclic factor = 8
#pragma HLS ARRAY_PARTITION variable = output cyclic factor = 8
#endif

    dtype max_val = input[0];
  FIND_MAX:
    for (int i = 1; i < n; i++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = 4
#endif
      max_val = (input[i] > max_val) ? input[i] : max_val;
    }

    dtype sum = 0;
    dtype exp_val[n];
  CALC_EXP:
    for (int i = 0; i < n; i++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = 4
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
#pragma HLS UNROLL factor = 4
#endif
      output[i] = exp_val[i] * inv_sum;
    }
  }

private:
};
} // namespace hls_nn
