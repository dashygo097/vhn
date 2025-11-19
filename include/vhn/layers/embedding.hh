#pragma once

#include "../opt_level.hh"

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace vhn {

template <typename DType, typename HParams, typename Config = void,
          OptLevel OPT_LEVEL = OPT_NONE>
class Embedding;

template <int VOCAB_SIZE, int EMBED_SIZE> struct EmbeddingHParams {
  static constexpr int vocab_size = VOCAB_SIZE;
  static constexpr int embed_size = EMBED_SIZE;
};

// ============================================================================
// Non-optimized version (OPT_NONE)
// ============================================================================
template <typename DType, typename HParams>
class Embedding<DType, HParams, void, OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int vocab_size = HParams::vocab_size;
  static constexpr int embed_size = HParams::embed_size;
  static constexpr OptLevel opt_level = OPT_NONE;

  using Weight_t = dtype[vocab_size][embed_size];

  Embedding() = default;
  ~Embedding() = default;

  static void forward(dtype output[embed_size], const int input,
                      const Weight_t weight) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    forward_single(output, input, weight);
  }

  static void forward(dtype output[][embed_size], const int input[],
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
  static void forward_single(dtype output[embed_size], const int input,
                             const Weight_t weight) {
  OUT_LOOP:
    for (int i = 0; i < embed_size; i++) {
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
      for (int j = 0; j < embed_size; j++) {
        output[i * embed_size + j] = weight[input[i]][j];
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
      for (int j = 0; j < embed_size; j++) {
        output.write(weight[idx][j]);
      }
    }
  }
#endif
};

// ============================================================================
// Optimized version (OPT_ENABLED)
// ============================================================================
template <typename DType, typename HParams, typename Config>
class Embedding<DType, HParams, Config, OPT_ENABLED> {
public:
  using dtype = DType;
  static constexpr int vocab_size = HParams::vocab_size;
  static constexpr int embed_size = HParams::embed_size;
  static constexpr OptLevel opt_level = OPT_ENABLED;

  static constexpr int unroll_factor = Config::unroll_factor;
  static constexpr int partition_factor = Config::partition_factor;
  static constexpr int pipeline_ii = Config::pipeline_ii;

  using Weight_t = dtype[vocab_size][embed_size];

  Embedding() = default;
  ~Embedding() = default;

  static void forward(dtype output[embed_size], const int input,
                      const Weight_t weight) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    forward_single(output, input, weight);
  }

  static void forward(dtype output[][embed_size], const int input[],
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
  static void forward_single(dtype output[embed_size], const int input,
                             const Weight_t weight) {
#ifdef __VITIS_HLS__
#pragma HLS ARRAY_PARTITION variable = output cyclic factor = partition_factor
#pragma HLS ARRAY_PARTITION variable = weight cyclic factor =                  \
    partition_factor dim = 2
#endif

  OUT_LOOP:
    for (int i = 0; i < embed_size; i++) {
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
      for (int j = 0; j < embed_size; j++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#pragma HLS UNROLL factor = unroll_factor
#endif
        output[i * embed_size + j] = weight[idx][j];
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
      for (int j = 0; j < embed_size; j++) {
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
