#pragma once

#include "../opt_level.hh"
#include "../tb/tb.hh"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace vhn {

template <typename DType, int N, typename Config = void,
          OptLevel OPT_LEVEL = OPT_NONE>
class Softmax;

// ============================================================================
// Non-optimized version (OPT_NONE)
// ============================================================================
template <typename DType, int N> class Softmax<DType, N, void, OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int n = N;
  static constexpr OptLevel opt_level = OPT_NONE;

  Softmax() = default;
  ~Softmax() = default;

  static void forward(dtype output[N], const dtype input[N]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    forward_1d_impl(output, input);
  }

  static void forward(dtype output[][N], const dtype input[][N],
                      const int batch_size) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_1d_impl(output[b], input[b]);
    }
  }

  static void forward(dtype *output, const dtype *input, const int batch_size) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_1d_impl(&output[b * N], &input[b * N]);
    }
  }

#ifdef __VITIS_HLS__
  static void forward(hls::stream<dtype> &output_stream,
                      hls::stream<dtype> &input_stream) {
#pragma HLS INLINE off
    forward_1d_stream_impl(output_stream, input_stream);
  }

#endif

private:
  static void forward_1d_impl(dtype *output, const dtype *input) {
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

    // Normalize by sum
    dtype inv_sum = dtype(1.0) / sum;
  NORMALIZE:
    for (int i = 0; i < n; i++) {
      output[i] = exp_val[i] * inv_sum;
    }
  }

#ifdef __VITIS_HLS__
  static void forward_1d_stream_impl(hls::stream<dtype> &output_stream,
                                     hls::stream<dtype> &input_stream) {
    dtype input_buffer[N];

  READ_INPUT:
    for (int i = 0; i < N; i++) {
      input_buffer[i] = input_stream.read();
    }

    dtype max_val = input_buffer[0];
  FIND_MAX:
    for (int i = 1; i < N; i++) {
      max_val = (input_buffer[i] > max_val) ? input_buffer[i] : max_val;
    }

    dtype sum = dtype(0.0f);
    dtype exp_val[N];
#pragma HLS ARRAY_PARTITION variable = exp_val complete

  CALC_EXP:
    for (int i = 0; i < N; i++) {
      exp_val[i] = hls::exp(input_buffer[i] - max_val);
      sum += exp_val[i];
    }

    dtype inv_sum = dtype(1.0) / sum;
  NORMALIZE_WRITE:
    for (int i = 0; i < N; i++) {
      output_stream.write(exp_val[i] * inv_sum);
    }
  }
#endif
};

// ============================================================================
// Optimized version (OPT_ENABLED)
// ============================================================================
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

  static void forward(dtype output[N], const dtype input[N]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    forward_1d_impl(output, input);
  }

  static void forward(dtype output[][N], const dtype input[][N],
                      const int batch_size) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_1d_impl(output[b], input[b]);
    }
  }

  static void forward(dtype *output, const dtype *input, const int batch_size) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_1d_impl(&output[b * N], &input[b * N]);
    }
  }

#ifdef __VITIS_HLS__
  static void forward(hls::stream<dtype> &output_stream,
                      hls::stream<dtype> &input_stream) {
#pragma HLS INLINE off
    forward_1d_stream_impl(output_stream, input_stream);
  }

#endif

private:
  static void forward_1d_impl(dtype *output, const dtype *input) {
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

#ifdef __VITIS_HLS__
  static void forward_1d_stream_impl(hls::stream<dtype> &output_stream,
                                     hls::stream<dtype> &input_stream) {
    dtype input_buffer[N];
#pragma HLS ARRAY_PARTITION variable = input_buffer cyclic factor =            \
    partition_factor

  READ_INPUT:
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II = pipeline_ii
      input_buffer[i] = input_stream.read();
    }

    dtype max_val = input_buffer[0];
  FIND_MAX:
    for (int i = 1; i < N; i++) {
#pragma HLS UNROLL factor = unroll_factor
      max_val = (input_buffer[i] > max_val) ? input_buffer[i] : max_val;
    }

    dtype sum = dtype(0.0f);
    dtype exp_val[N];
#pragma HLS ARRAY_PARTITION variable = exp_val cyclic factor = partition_factor

  CALC_EXP:
    for (int i = 0; i < N; i++) {
#pragma HLS UNROLL factor = unroll_factor
      exp_val[i] = hls::exp(input_buffer[i] - max_val);
      sum += exp_val[i];
    }

    dtype inv_sum = dtype(1.0) / sum;
  NORMALIZE_WRITE:
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II = pipeline_ii
      output_stream.write(exp_val[i] * inv_sum);
    }
  }
#endif
};
} // namespace vhn
