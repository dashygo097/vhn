#pragma once

#include "../opt_level.hh"
#include <cmath>

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace vhn {

template <typename DType, typename HParams, typename Config = void,
          OptLevel OPT_LEVEL = OPT_NONE>
class Softmax;

template <int N> struct SoftmaxHParams {
  static constexpr int n = N;
};

// ============================================================================
// Non-optimized version (OPT_NONE)
// ============================================================================
template <typename DType, typename HParams>
class Softmax<DType, HParams, void, OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int n = HParams::n;
  static constexpr OptLevel opt_level = OPT_NONE;

  Softmax() = default;
  ~Softmax() = default;

  static void sm(dtype output[n], const dtype input[n]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    sm_1d_impl(output, input);
  }

  static void sm(dtype output[][n], const dtype input[][n],
                 const int batch_size) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 32
#endif
      sm_1d_impl(output[b], input[b]);
    }
  }

  static void sm(dtype *output, const dtype *input, const int batch_size) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 32
#endif
      sm_1d_impl(&output[b * n], &input[b * n]);
    }
  }

#ifdef __VITIS_HLS__
  static void sm(hls::stream<dtype> &output_stream,
                 hls::stream<dtype> &input_stream) {
#pragma HLS INLINE off
    sm_1d_stream_impl(output_stream, input_stream);
  }

#endif

private:
  static void sm_1d_impl(dtype *output, const dtype *input) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif

    dtype max_val = input[0];
  FIND_MAX:
    for (int i = 1; i < n; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4096
#endif
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

#ifdef __VITIS_HLS__
  static void sm_1d_stream_impl(hls::stream<dtype> &output_stream,
                                hls::stream<dtype> &input_stream) {
    dtype input_buffer[n];

  READ_INPUT:
    for (int i = 0; i < n; i++) {
      input_buffer[i] = input_stream.read();
    }

    dtype max_val = input_buffer[0];
  FIND_MAX:
    for (int i = 1; i < n; i++) {
      max_val = (input_buffer[i] > max_val) ? input_buffer[i] : max_val;
    }

    dtype sum = dtype(0.0f);
    dtype exp_val[n];

  CALC_EXP:
    for (int i = 0; i < n; i++) {
      exp_val[i] = hls::exp(input_buffer[i] - max_val);
      sum += exp_val[i];
    }

    dtype inv_sum = dtype(1.0) / sum;
  NORMALIZE_WRITE:
    for (int i = 0; i < n; i++) {
      output_stream.write(exp_val[i] * inv_sum);
    }
  }
#endif
};

template <int PIPELINE_II, int UNROLL_FACTOR, int PARTITION_FACTOR>
struct SoftmaxConfig {
  static constexpr int pipeline_ii = PIPELINE_II;
  static constexpr int unroll_factor = UNROLL_FACTOR;
  static constexpr int partition_factor = PARTITION_FACTOR;
};

// ============================================================================
// Optimized version (OPT_ENABLED)
// ============================================================================
template <typename DType, typename HParams, typename Config>
class Softmax<DType, HParams, Config, OPT_ENABLED> {
public:
  using dtype = DType;
  static constexpr int n = HParams::n;
  static constexpr OptLevel opt_level = OPT_ENABLED;

  static constexpr int pipeline_ii = Config::pipeline_ii;
  static constexpr int unroll_factor = Config::unroll_factor;
  static constexpr int partition_factor = Config::partition_factor;

  Softmax() = default;
  ~Softmax() = default;

  static void sm(dtype output[n], const dtype input[n]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    sm_1d_impl(output, input);
  }

  static void sm(dtype output[][n], const dtype input[][n],
                 const int batch_size) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 32
#pragma HLS PIPELINE II = 1
#endif
      sm_1d_impl(output[b], input[b]);
    }
  }

  static void sm(dtype *output, const dtype *input, const int batch_size) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 32
#pragma HLS PIPELINE II = 1
#endif
      sm_1d_impl(&output[b * n], &input[b * n]);
    }
  }

#ifdef __VITIS_HLS__
  static void sm(hls::stream<dtype> &output_stream,
                 hls::stream<dtype> &input_stream) {
#pragma HLS INLINE off
    sm_1d_stream_impl(output_stream, input_stream);
  }

#endif

private:
  static void sm_1d_impl(dtype *output, const dtype *input) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = input cyclic factor = partition_factor
#pragma HLS ARRAY_PARTITION variable = output cyclic factor = partition_factor
#endif

    dtype max_val = input[0];
  FIND_MAX:
    for (int i = 1; i < n; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = partition_factor
#pragma HLS UNROLL factor = unroll_factor
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4096
#endif
      max_val = (input[i] > max_val) ? input[i] : max_val;
    }

    dtype sum = dtype(0.0f);
    dtype exp_val[n];

#ifdef __VITIS_HLS__
#pragma HLS ARRAY_PARTITION variable = exp_val cyclic factor = partition_factor
#endif

  CALC_EXP:
    for (int i = 0; i < n; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#pragma HLS UNROLL factor = unroll_factor
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4096
#pragma HLS BIND_OP variable = sum op = add impl = dsp
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
#pragma HLS PIPELINE II = pipeline_ii
#pragma HLS UNROLL factor = unroll_factor
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4096
#pragma HLS BIND_OP variable = output op = mul impl = dsp
#endif
      output[i] = exp_val[i] * inv_sum;
    }
  }

#ifdef __VITIS_HLS__
  static void sm_1d_stream_impl(hls::stream<dtype> &output_stream,
                                hls::stream<dtype> &input_stream) {
    dtype input_buffer[n];
#pragma HLS ARRAY_PARTITION variable = input_buffer cyclic factor =            \
    partition_factor

  READ_INPUT:
    for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II = pipeline_ii
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4096
      input_buffer[i] = input_stream.read();
    }

    dtype max_val = input_buffer[0];
  FIND_MAX:
    for (int i = 1; i < n; i++) {
#pragma HLS UNROLL factor = unroll_factor
      max_val = (input_buffer[i] > max_val) ? input_buffer[i] : max_val;
    }

    dtype sum = dtype(0.0f);
    dtype exp_val[n];
#pragma HLS ARRAY_PARTITION variable = exp_val cyclic factor = partition_factor

  CALC_EXP:
    for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II = pipeline_ii
#pragma HLS UNROLL factor = unroll_factor
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4096
#pragma HLS BIND_OP variable = sum op = add impl = dsp
      exp_val[i] = hls::exp(input_buffer[i] - max_val);
      sum += exp_val[i];
    }

    dtype inv_sum = dtype(1.0) / sum;
  NORMALIZE_WRITE:
    for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II = pipeline_ii
#pragma HLS UNROLL factor = unroll_factor
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 4096
#pragma HLS BIND_OP variable = inv_sum op = mul impl = dsp
#pragma HLS PIPELINE II = pipeline_ii
      output_stream.write(exp_val[i] * inv_sum);
    }
  }
#endif
};
} // namespace vhn
