#pragma once

#include "../opt_level.hh"

#ifdef __VITIS_HLS__
#endif

namespace hls_nn {
template <typename DType, typename ImplType, int N, typename Config = void,
          OptLevel OPT_LEVEL = OPT_NONE>
class Reduce;

// ============================================================================
// Non-optimized version (OPT_NONE)
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
    return forward_1d_impl(input);
  }

  static void forward(dtype &output, const dtype input[N]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    output = forward_1d_impl(input);
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
      output[b] = forward_1d_impl(input[b]);
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
      output[b] = forward_1d_impl(&input[b * N]);
    }
  }

  template <int AXIS>
  static void forward_axis(dtype output[], const dtype input[][N],
                           const int batch_size) {
    static_assert(AXIS == 1,
                  "For 2D input, only AXIS=1 (reduce N) is supported");
    forward(output, input, batch_size);
  }

  static dtype forward_global(const dtype input[][N], const int batch_size) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    dtype global_acc = impl::init_value();
    int total_count = 0;

  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
    ELEMENT_LOOP:
      for (int i = 0; i < N; i++) {
        global_acc = impl::kernel(global_acc, input[b][i]);
        total_count++;
      }
    }

    return impl::finalize(global_acc, total_count);
  }

private:
  static dtype forward_1d_impl(const dtype *input) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    dtype acc = impl::init_value();

  REDUCE_LOOP:
    for (int i = 0; i < N; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE off
#endif
      acc = impl::kernel(acc, input[i]);
    }

    return impl::finalize(acc, N);
  }
};

// ============================================================================
// Optimized version (OPT_ENABLED)
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

  Reduce() = default;
  ~Reduce() = default;

  static dtype forward(const dtype input[N]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    return forward_1d_impl(input);
  }

  static void forward(dtype &output, const dtype input[N]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    output = forward_1d_impl(input);
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
      output[b] = forward_1d_impl(input[b]);
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
      output[b] = forward_1d_impl(&input[b * N]);
    }
  }

  template <int AXIS>
  static void forward_axis(dtype output[], const dtype input[][N],
                           const int batch_size) {
    static_assert(AXIS == 1, "For 2D input, only AXIS=1 is supported");
    forward(output, input, batch_size);
  }

  static dtype forward_global(const dtype input[][N], const int batch_size) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    dtype global_acc = impl::init_value();
    int total_count = 0;

  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
    ELEMENT_LOOP:
      for (int i = 0; i < N; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#pragma HLS UNROLL factor = unroll_factor
#endif
        global_acc = impl::kernel(global_acc, input[b][i]);
        total_count++;
      }
    }

    return impl::finalize(global_acc, total_count);
  }

private:
  // Core reduction with optimizations
  static dtype forward_1d_impl(const dtype *input) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = input cyclic factor = partition_factor
#endif
    dtype acc = impl::init_value();

  REDUCE_LOOP:
    for (int i = 0; i < N; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#pragma HLS UNROLL factor = unroll_factor
#endif
      acc = impl::kernel(acc, input[i]);
    }

    return impl::finalize(acc, N);
  }
};

} // namespace hls_nn
