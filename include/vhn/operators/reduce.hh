#pragma once

#include "../opt_level.hh"

#ifdef __VITIS_HLS__
#endif

namespace vhn {
template <const int UNROLL_FACTOR, const int PARTITION_FACTOR,
          const int PIPELINE_II, const bool _USE_REDUCE_TREE = true>
struct ReduceConfig {
  static constexpr int _unroll_factor = UNROLL_FACTOR;
  static constexpr int _partition_factor = PARTITION_FACTOR;
  static constexpr int _pipeline_ii = PIPELINE_II;
  static constexpr bool _use_reduce_tree = _USE_REDUCE_TREE;
};

template <typename DType, typename ImplType, int N, typename Config = void,
          OptLevel OPT_LEVEL = OPT_NONE>
class Reduce;

constexpr int log2_ceil(int x) {
  return (x <= 1) ? 0 : 1 + log2_ceil((x + 1) / 2);
}
constexpr bool is_power_of_2(int x) { return (x & (x - 1)) == 0; }
constexpr int next_power_of_2(int x) {
  return is_power_of_2(x) ? x : 1 << log2_ceil(x);
}

// ============================================================================
// Non-optimized version (OPT_NONE) - Always Sequential
// ============================================================================
template <typename DType, typename ImplType, int N>
class Reduce<DType, ImplType, N, void, OPT_NONE> {
public:
  using dtype = DType;
  using impl = ImplType;
  static constexpr int n = N;
  static constexpr OptLevel opt_level = OPT_NONE;

  Reduce() = default;
  ~Reduce() = default;

  static dtype forward(const dtype input[N]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    return forward_sequential(input);
  }

  static void forward(dtype &output, const dtype input[N]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    output = forward_sequential(input);
  }

  static void forward(dtype output[], const dtype input[][N],
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
      output[b] = forward_sequential(input[b]);
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
      output[b] = forward_sequential(&input[b * N]);
    }
  }

#ifdef __VITIS_HLS__
  static dtype forward(hls::stream<dtype> &input_stream) {
#pragma HLS INLINE off
    return forward_stream_impl(input_stream);
  }

  static void forward(hls::stream<dtype> &output_stream,
                      hls::stream<dtype> &input_stream) {
#pragma HLS INLINE off
    output_stream.write(forward_stream_impl(input_stream));
  }
#endif

private:
  static dtype forward_sequential(const dtype *input) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    dtype acc = dtype(0);

  REDUCE_LOOP:
    for (int i = 0; i < N; i++) {
      acc = impl::kernel(acc, input[i]);
    }

    return impl::finalize(acc);
  }

#ifdef __VITIS_HLS__
  static dtype forward_stream_impl(hls::stream<dtype> &input_stream) {
    dtype input_buffer[N];
#pragma HLS ARRAY_PARTITION variable = input_buffer complete

  READ_INPUT:
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II = 1
      input_buffer[i] = input_stream.read();
    }

    dtype acc = dtype(0);
  REDUCE_STREAM_LOOP:
    for (int i = 0; i < N; i++) {
      acc = impl::kernel(acc, input_buffer[i]);
    }

    return impl::finalize(acc);
  }
#endif
};

// ============================================================================
// Optimized version (OPT_ENABLED) - Sequential or Tree based on Config
// ============================================================================
template <typename DType, typename ImplType, int N, typename Config>
class Reduce<DType, ImplType, N, Config, OPT_ENABLED> {
public:
  using dtype = DType;
  using impl = ImplType;
  static constexpr int n = N;
  static constexpr OptLevel opt_level = OPT_ENABLED;

  static constexpr int unroll_factor = Config::_unroll_factor;
  static constexpr int partition_factor = Config::_partition_factor;
  static constexpr int pipeline_ii = Config::_pipeline_ii;
  static constexpr bool use_tree = Config::_use_reduce_tree;

  static constexpr int num_stages = log2_ceil(N);
  static constexpr int padded_n = next_power_of_2(N);

  Reduce() = default;
  ~Reduce() = default;

  static dtype forward(const dtype input[N]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    if constexpr (use_tree) {
      return forward_tree(input);
    } else {
      return forward_sequential(input);
    }
  }

  static void forward(dtype &output, const dtype input[N]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    output = forward(input);
  }

  static void forward(dtype output[], const dtype input[][N],
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
      output[b] = forward(input[b]);
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
      output[b] = forward(&input[b * N]);
    }
  }

#ifdef __VITIS_HLS__
  static dtype forward(hls::stream<dtype> &input_stream) {
#pragma HLS INLINE off
    return forward_stream_impl(input_stream);
  }

  static void forward(hls::stream<dtype> &output_stream,
                      hls::stream<dtype> &input_stream) {
#pragma HLS INLINE off
    output_stream.write(forward_stream_impl(input_stream));
  }

#endif

private:
  static dtype forward_sequential(const dtype *input) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = input cyclic factor = partition_factor
#endif
    dtype acc = dtype(0);

  REDUCE_LOOP:
    for (int i = 0; i < N; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#pragma HLS UNROLL factor = unroll_factor
#endif
      acc = impl::kernel(acc, input[i]);
    }

    return impl::finalize(acc);
  }

  static dtype forward_tree(const dtype *input) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable = input cyclic factor = partition_factor
#endif

    // Pad input if not power of 2
    dtype padded_input[padded_n];
#ifdef __VITIS_HLS__
#pragma HLS ARRAY_PARTITION variable = padded_input cyclic factor =            \
    partition_factor
#endif

  PAD_LOOP:
    for (int i = 0; i < padded_n; i++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = unroll_factor
#endif
      if (i < N) {
        padded_input[i] = input[i];
      } else {
        padded_input[i] = dtype(0);
      }
    }

    dtype result = forward_tree_recursive<padded_n>(padded_input);
    return impl::finalize(result);
  }

  template <int SIZE>
  static dtype forward_tree_recursive(const dtype input[SIZE]) {
    if constexpr (SIZE == 1) {
      return input[0];
    } else {
      constexpr int HALF = SIZE / 2;
      dtype left_result[HALF];
      dtype right_result[HALF];

#ifdef __VITIS_HLS__
#pragma HLS ARRAY_PARTITION variable = left_result cyclic factor =             \
    partition_factor
#pragma HLS ARRAY_PARTITION variable = right_result cyclic factor =            \
    partition_factor
#endif

    TREE_LEVEL:
      for (int i = 0; i < HALF; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#pragma HLS UNROLL factor = unroll_factor
#endif
        left_result[i] = input[2 * i];
        right_result[i] = input[2 * i + 1];
      }

      dtype combined[HALF];
#ifdef __VITIS_HLS__
#pragma HLS ARRAY_PARTITION variable = combined cyclic factor = partition_factor
#endif

    COMBINE:
      for (int i = 0; i < HALF; i++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = unroll_factor
#endif
        combined[i] = impl::kernel(left_result[i], right_result[i]);
      }

      return forward_tree_recursive<HALF>(combined);
    }
  }

#ifdef __VITIS_HLS__
  static dtype forward_stream_impl(hls::stream<dtype> &input_stream) {
    dtype input_buffer[N];
#pragma HLS ARRAY_PARTITION variable = input_buffer cyclic factor =            \
    partition_factor

  READ_INPUT:
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II = pipeline_ii
      input_buffer[i] = input_stream.read();
    }

    if constexpr (use_tree) {
      return forward_tree_internal(input_buffer);
    } else {
      dtype acc = dtype(0);
    REDUCE_STREAM_LOOP:
      for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II = pipeline_ii
#pragma HLS UNROLL factor = unroll_factor
        acc = impl::kernel(acc, input_buffer[i]);
      }
      return impl::finalize(acc);
    }
  }
#endif
};

} // namespace vhn
