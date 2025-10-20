#pragma once

#include "../opt_level.hh"

#ifdef __VITIS_HLS__
#endif

namespace hls_nn {

template <typename DType, typename ImplType, int N, typename Config = void,
          OptLevel OPT_LEVEL = OPT_NONE>
class Elementwise;

// ============================================================================
// Non-optimized version (OPT_NONE)
// ============================================================================
template <typename DType, typename ImplType, int N>
class Elementwise<DType, ImplType, N, void, OPT_NONE> {
public:
  using dtype = DType;
  using impl = ImplType;
  static constexpr int n = N;
  static constexpr OptLevel opt_level = OPT_NONE;

  Elementwise() = default;
  ~Elementwise() = default;

  static void forward(dtype output[N], const dtype input) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif

    forward_1d_impl(output, input);
  }

  static void forward(dtype output[N], const dtype input[N]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    forward_1d_impl(output, input);
  }

  static void forward(dtype output[N], const dtype input1[N],
                      const dtype input2) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif

    forward_1d_impl(output, input1, input2);
  }

  static void forward(dtype output[N], const dtype input1[N],
                      const dtype input2[N]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    forward_1d_impl(output, input1, input2);
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

  static void forward(dtype output[][N], const dtype input1[][N],
                      const dtype input2[][N], const int batch_size) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_1d_impl(output[b], input1[b], input2[b]);
    }
  }

  static void forward(dtype *output, const dtype input, const int batch_size) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif

  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_1d_impl(&output[b * N], input);
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

  static void forward(dtype *output, const dtype *input1, const dtype input2,
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

      forward_1d_impl(&output[b * N], &input1[b * N], input2);
    }
  }

  static void forward(dtype *output, const dtype *input1, const dtype *input2,
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
      forward_1d_impl(&output[b * N], &input1[b * N], &input2[b * N]);
    }
  }

#ifdef __VITIS_HLS__
  static void forward(hls::stream<dtype> &output_stream,
                      hls::stream<dtype> &input_stream) {
#pragma HLS INLINE off
    forward_1d_stream_impl(output_stream, input_stream);
  }

  static void forward(hls::stream<dtype> &output_stream,
                      hls::stream<dtype> &input_stream1,
                      hls::stream<dtype> &input_stream2) {
#pragma HLS INLINE off
    forward_1d_stream_impl(output_stream, input_stream1, input_stream2);
  }
#endif

private:
  static void forward_1d_impl(dtype *output, const dtype input) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
  ELEMENTWISE_LOOP:
    for (int i = 0; i < n; i++) {
      output[i] = impl::kernel(input);
    }
  }

  static void forward_1d_impl(dtype *output, const dtype *input) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
  ELEMENTWISE_LOOP:
    for (int i = 0; i < n; i++) {
      output[i] = impl::kernel(input[i]);
    }
  }

  static void forward_1d_impl(dtype *output, const dtype *input1,
                              const dtype *input2) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
  ELEMENTWISE_LOOP_2:
    for (int i = 0; i < n; i++) {
      output[i] = impl::kernel(input1[i], input2[i]);
    }
  }

  static void forward_1d_impl(dtype *output, const dtype *input1,
                              const dtype input2) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
  ELEMENTWISE_LOOP_2:
    for (int i = 0; i < n; i++) {
      output[i] = impl::kernel(input1[i], input2);
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
  ELEMENTWISE_STREAM_LOOP:
    for (int i = 0; i < N; i++) {
      output_stream.write(impl::kernel(input_buffer[i]));
    }
  }

  static void forward_1d_stream_impl(hls::stream<dtype> &output_stream,
                                     hls::stream<dtype> &input_stream1,
                                     hls::stream<dtype> &input_stream2) {
    dtype input_buffer1[N];
    dtype input_buffer2[N];

  READ_INPUT1:
    for (int i = 0; i < N; i++) {
      input_buffer1[i] = input_stream1.read();
    }

  READ_INPUT2:
    for (int i = 0; i < N; i++) {
      input_buffer2[i] = input_stream2.read();
    }

  ELEMENTWISE_STREAM_LOOP_2:
    for (int i = 0; i < N; i++) {
      output_stream.write(impl::kernel(input_buffer1[i], input_buffer2[i]));
    }
  }

#endif
};

// ============================================================================
// Optimized version (OPT_ENABLED)
// ============================================================================
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

  static void forward(dtype output[N], const dtype input) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif

    forward_1d_impl(output, input);
  }

  static void forward(dtype output[N], const dtype input[N]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    forward_1d_impl(output, input);
  }

  static void forward(dtype output[N], const dtype input1[N],
                      const dtype input2) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif

    forward_1d_impl(output, input1, input2);
  }

  static void forward(dtype output[N], const dtype input1[N],
                      const dtype input2[N]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    forward_1d_impl(output, input1, input2);
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

  static void forward(dtype output[][N], const dtype input1[][N],
                      const dtype input2[][N], const int batch_size) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_1d_impl(output[b], input1[b], input2[b]);
    }
  }

  static void forward(dtype *output, const dtype input, const int batch_size) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif

  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_1d_impl(&output[b * N], input);
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

  static void forward(dtype *output, const dtype *input1, const dtype input2,
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

      forward_1d_impl(&output[b * N], &input1[b * N], input2);
    }
  }

  static void forward(dtype *output, const dtype *input1, const dtype *input2,
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
      forward_1d_impl(&output[b * N], &input1[b * N], &input2[b * N]);
    }
  }

#ifdef __VITIS_HLS__
  static void forward(hls::stream<dtype> &output_stream,
                      hls::stream<dtype> &input_stream) {
#pragma HLS INLINE off
    forward_1d_stream_impl(output_stream, input_stream);
  }

  static void forward(hls::stream<dtype> &output_stream,
                      hls::stream<dtype> &input_stream1,
                      hls::stream<dtype> &input_stream2) {
#pragma HLS INLINE off
    forward_1d_stream_impl(output_stream, input_stream1, input_stream2);
  }
#endif

private:
  static void forward_1d_impl(dtype *output, const dtype input) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = output cyclic factor = partition_factor
#endif
  ELEMENTWISE_LOOP:
    for (int i = 0; i < n; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#pragma HLS UNROLL factor = unroll_factor
#endif
      output[i] = impl::kernel(input);
    }
  }

  static void forward_1d_impl(dtype *output, const dtype *input) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = output cyclic factor = partition_factor
#pragma HLS ARRAY_PARTITION variable = input cyclic factor = partition_factor
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

  static void forward_1d_impl(dtype *output, const dtype *input1,
                              const dtype input2) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = output cyclic factor = partition_factor
#pragma HLS ARRAY_PARTITION variable = input1 cyclic factor = partition_factor
#endif

  ELEMENTWISE_LOOP_2:
    for (int i = 0; i < n; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#pragma HLS UNROLL factor = unroll_factor
#endif
      output[i] = impl::kernel(input1[i], input2);
    }
  }

  static void forward_1d_impl(dtype *output, const dtype *input1,
                              const dtype *input2) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = output cyclic factor = partition_factor
#pragma HLS ARRAY_PARTITION variable = input1 cyclic factor = partition_factor
#pragma HLS ARRAY_PARTITION variable = input2 cyclic factor = partition_factor
#endif
  ELEMENTWISE_LOOP_2:
    for (int i = 0; i < n; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#pragma HLS UNROLL factor = unroll_factor
#endif
      output[i] = impl::kernel(input1[i], input2[i]);
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

  ELEMENTWISE_STREAM_LOOP:
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II = pipeline_ii
#pragma HLS UNROLL factor = unroll_factor
      output_stream.write(impl::kernel(input_buffer[i]));
    }
  }

  static void forward_1d_stream_impl(hls::stream<dtype> &output_stream,
                                     hls::stream<dtype> &input_stream1,
                                     hls::stream<dtype> &input_stream2) {
    dtype input_buffer1[N];
    dtype input_buffer2[N];
#pragma HLS ARRAY_PARTITION variable = input_buffer1 cyclic factor =           \
    partition_factor
#pragma HLS ARRAY_PARTITION variable = input_buffer2 cyclic factor =           \
    partition_factor

  READ_INPUT1:
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II = pipeline_ii
      input_buffer1[i] = input_stream1.read();
    }

  READ_INPUT2:
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II = pipeline_ii
      input_buffer2[i] = input_stream2.read();
    }

  ELEMENTWISE_STREAM_LOOP_2:
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II = pipeline_ii
#pragma HLS UNROLL factor = unroll_factor
      output_stream.write(impl::kernel(input_buffer1[i], input_buffer2[i]));
    }
  }
#endif
};

} // namespace hls_nn
