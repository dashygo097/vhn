#pragma once

#include "../opt_level.hh"

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace vhn {

template <typename DType, const int VOCAB_SIZE, const int EMBED_DIM,
          typename Config = void, OptLevel OPT_LEVEL = OPT_NONE>
class Embedding;

// ============================================================================
// Non-optimized version (OPT_NONE)
// ============================================================================
template <typename DType, const int VOCAB_SIZE, const int EMBED_DIM>
class Embedding<DType, VOCAB_SIZE, EMBED_DIM, void, OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int vocab_size = VOCAB_SIZE;
  static constexpr int embed_dim = EMBED_DIM;
  static constexpr OptLevel opt_level = OPT_NONE;

  using Weight_t = dtype[VOCAB_SIZE][EMBED_DIM];

  Embedding() = default;
  ~Embedding() = default;

  static void forward(dtype output[EMBED_DIM], const int input,
                      const Weight_t weight) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    forward_single(output, input, weight);
  }

  static void forward(dtype output[][EMBED_DIM], const int input[],
                      const int batch_size, const Weight_t weight) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
  BATCH_LOOP:
    for (int i = 0; i < batch_size; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_single(output[i], input[i], weight);
    }
  }

  static void forward(dtype *output, const int *input, const int length,
                      const Weight_t weight) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    forward_impl_1d(output, input, length, weight);
  }

#ifdef __VITIS_HLS__
  static void forward(hls::stream<dtype> &output, hls::stream<int> &input,
                      const int length, const Weight_t weight) {
#pragma HLS INLINE off
    forward_1d_stream_impl(output, input, length, weight);
  }
#endif

private:
  static void forward_single(dtype output[EMBED_DIM], const int input,
                             const Weight_t weight) {
  OUT_LOOP:
    for (int i = 0; i < EMBED_DIM; i++) {
      output[i] = weight[input][i];
    }
  }

  static void forward_impl_1d(dtype *output, const int *input, const int length,
                              const Weight_t weight) {
  LENGTH_LOOP:
    for (int i = 0; i < length; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
    OUT_LOOP:
      for (int j = 0; j < EMBED_DIM; j++) {
        output[i * EMBED_DIM + j] = weight[input[i]][j];
      }
    }
  }

#ifdef __VITIS_HLS__
  static void forward_1d_stream_impl(hls::stream<dtype> &output,
                                     hls::stream<int> &input, const int length,
                                     const Weight_t weight) {
  LENGTH_LOOP:
    for (int i = 0; i < length; i++) {
      int idx = input.read();
    OUT_LOOP:
      for (int j = 0; j < EMBED_DIM; j++) {
        output.write(weight[idx][j]);
      }
    }
  }
#endif
};

// ============================================================================
// Optimized version (OPT_ENABLED)
// ============================================================================
template <typename DType, const int VOCAB_SIZE, const int EMBED_DIM,
          typename Config>
class Embedding<DType, VOCAB_SIZE, EMBED_DIM, Config, OPT_ENABLED> {
public:
  using dtype = DType;
  static constexpr int vocab_size = VOCAB_SIZE;
  static constexpr int embed_dim = EMBED_DIM;
  static constexpr OptLevel opt_level = OPT_ENABLED;

  static constexpr int unroll_factor = Config::_unroll_factor;
  static constexpr int partition_factor = Config::_partition_factor;
  static constexpr int pipeline_ii = Config::_pipeline_ii;

  using Weight_t = dtype[VOCAB_SIZE][EMBED_DIM];

  Embedding() = default;
  ~Embedding() = default;

  static void forward(dtype output[EMBED_DIM], const int input,
                      const Weight_t weight) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    forward_single(output, input, weight);
  }

  static void forward(dtype output[][EMBED_DIM], const int input[],
                      const int batch_size, const Weight_t weight) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
  BATCH_LOOP:
    for (int i = 0; i < batch_size; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_single(output[i], input[i], weight);
    }
  }

  static void forward(dtype *output, const int *input, const int length,
                      const Weight_t weight) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    forward_impl_1d(output, input, length, weight);
  }

#ifdef __VITIS_HLS__
  static void forward(hls::stream<dtype> &output, hls::stream<int> &input,
                      const int length, const Weight_t weight) {
#pragma HLS INLINE off
    forward_1d_stream_impl(output, input, length, weight);
  }
#endif

private:
  static void forward_single(dtype output[EMBED_DIM], const int input,
                             const Weight_t weight) {
#ifdef __VITIS_HLS__
#pragma HLS ARRAY_PARTITION variable = output cyclic factor = partition_factor
#pragma HLS ARRAY_PARTITION variable = weight cyclic factor =                  \
    partition_factor dim = 2
#endif

  OUT_LOOP:
    for (int i = 0; i < EMBED_DIM; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#pragma HLS UNROLL factor = unroll_factor
#endif
      output[i] = weight[input][i];
    }
  }

  static void forward_impl_1d(dtype *output, const int *input, const int length,
                              const Weight_t weight) {
#ifdef __VITIS_HLS__
#pragma HLS ARRAY_PARTITION variable = weight cyclic factor =                  \
    partition_factor dim = 2
#endif

  LENGTH_LOOP:
    for (int i = 0; i < length; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      int idx = input[i];
    OUT_LOOP:
      for (int j = 0; j < EMBED_DIM; j++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#pragma HLS UNROLL factor = unroll_factor
#endif
        output[i * EMBED_DIM + j] = weight[idx][j];
      }
    }
  }

#ifdef __VITIS_HLS__
  static void forward_1d_stream_impl(hls::stream<dtype> &output,
                                     hls::stream<int> &input, const int length,
                                     const Weight_t weight) {
#pragma HLS ARRAY_PARTITION variable = weight cyclic factor =                  \
    partition_factor dim = 2

  LENGTH_LOOP:
    for (int i = 0; i < length; i++) {
      int idx = input.read();
    OUT_LOOP:
      for (int j = 0; j < EMBED_DIM; j++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#pragma HLS UNROLL factor = unroll_factor
#endif
        output.write(weight[idx][j]);
      }
    }
  }
#endif
};
} // namespace vhn
