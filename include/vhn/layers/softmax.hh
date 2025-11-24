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

template <int PIPELINE_II, int UNROLL_FACTOR, int PARTITION_FACTOR>
struct SoftmaxConfig {
  static constexpr int pipeline_ii = PIPELINE_II;
  static constexpr int unroll_factor = UNROLL_FACTOR;
  static constexpr int partition_factor = PARTITION_FACTOR;
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
#pragma HLS BIND_STORAGE variable = input type = ram_2p impl = bram
#pragma HLS BIND_STORAGE variable = output type = ram_2p impl = bram
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

#ifdef __VITIS_HLS__
  static void sm(hls::stream<dtype> &output_stream,
                 hls::stream<dtype> &input_stream) {
#pragma HLS INLINE off
    sm_1d_stream_impl(output_stream, input_stream);
  }

  static void sm_2d(hls::stream<dtype> &output_stream,
                    hls::stream<dtype> &input_stream, const int batch_size) {
#pragma HLS INLINE off
  BATCH_STREAM_LOOP:
    for (int b = 0; b < batch_size; b++) {
#pragma HLS LOOP_FLATTEN off
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 32
      sm_1d_stream_impl(output_stream, input_stream);
    }
  }
#endif

private:
  static void sm_1d_impl(dtype *output, const dtype *input) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS PIPELINE off
#endif

    dtype max_val = input[0];
  FIND_MAX:
    for (int i = 1; i < n; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 256
#pragma HLS PIPELINE II = 1
#endif
      if (input[i] > max_val) {
        max_val = input[i];
      }
    }

    dtype sum = dtype(0.0f);

  CALC_EXP_SUM:
    for (int i = 0; i < n; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 256
#pragma HLS PIPELINE II = 2
#pragma HLS BIND_OP variable = sum op = add impl = fabric latency = 3
      dtype exp_val = hls::exp(input[i] - max_val);
      output[i] = exp_val;
#else
      dtype exp_val = std::exp(input[i] - max_val);
      output[i] = exp_val;
#endif
      sum += exp_val;
    }

    dtype inv_sum = dtype(1.0) / sum;

  NORMALIZE:
    for (int i = 0; i < n; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 256
#pragma HLS PIPELINE II = 1
#pragma HLS BIND_OP variable = output op = mul impl = dsp
#endif
      output[i] = output[i] * inv_sum;
    }
  }

#ifdef __VITIS_HLS__
  static void sm_1d_stream_impl(hls::stream<dtype> &output_stream,
                                hls::stream<dtype> &input_stream) {
#pragma HLS INLINE off
#pragma HLS PIPELINE off

    dtype max_val = input_stream.read();
    dtype temp_input[n];

  READ_FIND_MAX:
    for (int i = 0; i < n; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 256
#pragma HLS PIPELINE II = 1
      dtype val = input_stream.read();
      temp_input[i] = val;
      if (val > max_val) {
        max_val = val;
      }
    }

    dtype sum = dtype(0.0f);

  CALC_EXP_STREAM:
    for (int i = 0; i < n; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 256
#pragma HLS PIPELINE II = 2
      dtype exp_val = hls::exp(temp_input[i] - max_val);
      temp_input[i] = exp_val;
      sum += exp_val;
    }

    dtype inv_sum = dtype(1.0) / sum;

  NORMALIZE_STREAM:
    for (int i = 0; i < n; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 256
#pragma HLS PIPELINE II = 1
      dtype normalized = temp_input[i] * inv_sum;
      output_stream.write(normalized);
    }
  }
#endif
};

// ============================================================================
// Optimized version (OPT_ENABLED) - Better timing
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
#pragma HLS BIND_STORAGE variable = input type = ram_2p impl = bram
#pragma HLS BIND_STORAGE variable = output type = ram_2p impl = bram
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

#ifdef __VITIS_HLS__
  static void sm(hls::stream<dtype> &output_stream,
                 hls::stream<dtype> &input_stream) {
#pragma HLS INLINE off
    sm_1d_stream_impl(output_stream, input_stream);
  }

  static void sm_2d(hls::stream<dtype> &output_stream,
                    hls::stream<dtype> &input_stream, const int batch_size) {
#pragma HLS INLINE off
  BATCH_STREAM_LOOP_OPT:
    for (int b = 0; b < batch_size; b++) {
#pragma HLS LOOP_FLATTEN off
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 32
      sm_1d_stream_impl(output_stream, input_stream);
    }
  }
#endif

private:
  static void sm_1d_impl(dtype *output, const dtype *input) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS ALLOCATION operation instances = fadd limit = 1
#pragma HLS ALLOCATION operation instances = fmul limit = 1
#pragma HLS ALLOCATION operation instances = fdiv limit = 1
#endif

    constexpr bool use_partition = (n <= 64) && (partition_factor > 1);

    dtype max_val;

#ifdef __VITIS_HLS__
    if constexpr (use_partition) {
      dtype max_buffer[n];
#pragma HLS ARRAY_PARTITION variable = max_buffer cyclic factor =              \
    partition_factor

      for (int i = 0; i < n; i++) {
#pragma HLS UNROLL factor = unroll_factor
        max_buffer[i] = input[i];
      }

      max_val = max_buffer[0];
      for (int i = 1; i < n; i++) {
#pragma HLS UNROLL factor = unroll_factor
        if (max_buffer[i] > max_val)
          max_val = max_buffer[i];
      }
    } else {
#endif
      max_val = input[0];
    FIND_MAX:
      for (int i = 1; i < n; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 256
#pragma HLS PIPELINE II = 1
#endif
        if (input[i] > max_val)
          max_val = input[i];
      }
#ifdef __VITIS_HLS__
    }
#endif

    dtype sum = dtype(0.0f);

  CALC_EXP:
    for (int i = 0; i < n; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 256
#pragma HLS PIPELINE II = 4
#pragma HLS BIND_OP variable = sum op = fadd impl = fabric latency = 4
      dtype exp_val = hls::exp(input[i] - max_val);
#else
      dtype exp_val = std::exp(input[i] - max_val);
#endif
      output[i] = exp_val;
      sum += exp_val;
    }

    dtype inv_sum;
#ifdef __VITIS_HLS__
#pragma HLS BIND_OP variable = inv_sum op = fdiv impl = fabric latency = 16
#endif
    inv_sum = dtype(1.0) / sum;

  NORMALIZE:
    for (int i = 0; i < n; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 256
#pragma HLS PIPELINE II = 2
#pragma HLS BIND_OP variable = output op = fmul impl = dsp latency = 3
#endif
      output[i] = output[i] * inv_sum;
    }
  }

#ifdef __VITIS_HLS__
  static void sm_1d_stream_impl(hls::stream<dtype> &output_stream,
                                hls::stream<dtype> &input_stream) {
#pragma HLS INLINE off
#pragma HLS PIPELINE off

    dtype max_val = -1e9;
    dtype buffer[n];

#pragma HLS ARRAY_PARTITION variable = buffer cyclic factor = partition_factor

  READ_MAX_STREAM:
    for (int i = 0; i < n; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 256
#pragma HLS PIPELINE II = 1
      dtype val = input_stream.read();
      buffer[i] = val;
      if (val > max_val) {
        max_val = val;
      }
    }

    dtype sum = dtype(0.0f);

  CALC_EXP_OPT_STREAM:
    for (int i = 0; i < n; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 256
#pragma HLS PIPELINE II = pipeline_ii
#pragma HLS BIND_OP variable = sum op = fadd impl = fabric latency = 4
      dtype exp_val = hls::exp(buffer[i] - max_val);
      buffer[i] = exp_val;
      sum += exp_val;
    }

    dtype inv_sum;
#pragma HLS BIND_OP variable = inv_sum op = fdiv impl = fabric latency = 16
    inv_sum = dtype(1.0) / sum;

  NORMALIZE_OPT_STREAM:
    for (int i = 0; i < n; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 256
#pragma HLS PIPELINE II = 2
#pragma HLS BIND_OP variable = output_stream op = write
      dtype normalized = buffer[i] * inv_sum;
      output_stream.write(normalized);
    }
  }
#endif
};

} // namespace vhn
