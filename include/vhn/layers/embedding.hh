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

  static void emb(dtype output[embed_size], const int input,
                  const Weight_t weight) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    emb_single(output, input, weight);
  }

  static void emb(dtype output[][embed_size], const int input[],
                  const int batch_size, const Weight_t weight) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
  BATCH_LOOP:
    for (int i = 0; i < batch_size; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 32
#endif
      emb_single(output[i], input[i], weight);
    }
  }

  static void emb(dtype *output, const int *input, const int length,
                  const Weight_t weight) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    emb_impl_1d(output, input, length, weight);
  }

#ifdef __VITIS_HLS__
  static void emb(hls::stream<dtype> &output, hls::stream<int> &input,
                  const int length, const Weight_t weight) {
#pragma HLS INLINE off
    emb_1d_stream_impl(output, input, length, weight);
  }
#endif

private:
  static void emb_single(dtype output[embed_size], const int input,
                         const Weight_t weight) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
  OUT_LOOP:
    for (int i = 0; i < embed_size; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1024
#endif
      output[i] = weight[input][i];
    }
  }

  static void emb_impl_1d(dtype *output, const int *input, const int length,
                          const Weight_t weight) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
  LENGTH_LOOP:
    for (int i = 0; i < length; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#endif
    OUT_LOOP:
      for (int j = 0; j < embed_size; j++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1024
#endif
        output[i * embed_size + j] = weight[input[i]][j];
      }
    }
  }

#ifdef __VITIS_HLS__
  static void emb_1d_stream_impl(hls::stream<dtype> &output,
                                 hls::stream<int> &input, const int length,
                                 const Weight_t weight) {
#pragma HLS INLINE off
  LENGTH_LOOP:
    for (int i = 0; i < length; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
      int idx = input.read();
    OUT_LOOP:
      for (int j = 0; j < embed_size; j++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1024
        output.write(weight[idx][j]);
      }
    }
  }
#endif
};

template <int PIPELINE_II, int UNROLL_FACTOR, int PARTITION_FACTOR>
struct EmbeddingConfig {
  static constexpr int pipeline_ii = PIPELINE_II;
  static constexpr int unroll_factor = UNROLL_FACTOR;
  static constexpr int partition_factor = PARTITION_FACTOR;
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

  static constexpr int pipeline_ii = Config::pipeline_ii;
  static constexpr int unroll_factor = Config::unroll_factor;
  static constexpr int partition_factor = Config::partition_factor;

  using Weight_t = dtype[vocab_size][embed_size];

  Embedding() = default;
  ~Embedding() = default;

  static void emb(dtype output[embed_size], const int input,
                  const Weight_t weight) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    emb_single(output, input, weight);
  }

  static void emb(dtype output[][embed_size], const int input[],
                  const int batch_size, const Weight_t weight) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
  BATCH_LOOP:
    for (int i = 0; i < batch_size; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 32
#pragma HLS PIPELINE II = 1
#endif
      emb_single(output[i], input[i], weight);
    }
  }

  static void emb(dtype *output, const int *input, const int length,
                  const Weight_t weight) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    emb_impl_1d(output, input, length, weight);
  }

#ifdef __VITIS_HLS__
  static void emb(hls::stream<dtype> &output, hls::stream<int> &input,
                  const int length, const Weight_t weight) {
#pragma HLS INLINE off
    emb_1d_stream_impl(output, input, length, weight);
  }
#endif

private:
  static void emb_single(dtype output[embed_size], const int input,
                         const Weight_t weight) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
    constexpr bool should_partition =
        (partition_factor > 1) && (embed_size <= 2048);
    if constexpr (should_partition) {
#pragma HLS ARRAY_PARTITION variable = output type = cyclic factor =           \
    partition_factor
#pragma HLS ARRAY_PARTITION variable = weight type = cyclic factor =           \
    partition_factor dim = 2
    } else {
#pragma HLS BIND_STORAGE variable = weight type = rom_2p impl = bram
#pragma HLS BIND_STORAGE variable = output type = ram_1p impl = bram
    }
#endif

  OUT_LOOP:
    for (int i = 0; i < embed_size; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#pragma HLS UNROLL factor = unroll_factor
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 2048
#endif
      output[i] = weight[input][i];
    }
  }

  static void emb_impl_1d(dtype *output, const int *input, const int length,
                          const Weight_t weight) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
    constexpr bool should_partition =
        (partition_factor > 1) && (embed_size <= 2048);
    if constexpr (should_partition) {
#pragma HLS ARRAY_PARTITION variable = weight type = cyclic factor =           \
    partition_factor dim = 2
    } else {
#pragma HLS BIND_STORAGE variable = weight type = rom_2p impl = bram
    }
#endif

  LENGTH_LOOP:
    for (int i = 0; i < length; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#endif
      int idx = input[i];
    OUT_LOOP:
      for (int j = 0; j < embed_size; j++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#pragma HLS UNROLL factor = unroll_factor
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 2048
#endif
        output[i * embed_size + j] = weight[idx][j];
      }
    }
  }

#ifdef __VITIS_HLS__
  static void emb_1d_stream_impl(hls::stream<dtype> &output,
                                 hls::stream<int> &input, const int length,
                                 const Weight_t weight) {
#pragma HLS INLINE off

    constexpr bool should_partition =
        (partition_factor > 1) && (embed_size <= 2048);
    if constexpr (should_partition) {
#pragma HLS ARRAY_PARTITION variable = weight type = cyclic factor =           \
    partition_factor dim = 2
    } else {
#pragma HLS BIND_STORAGE variable = weight type = rom_2p impl = bram
    }

  LENGTH_LOOP:
    for (int i = 0; i < length; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
      int idx = input.read();
    OUT_LOOP:
      for (int j = 0; j < embed_size; j++) {
#pragma HLS PIPELINE II = pipeline_ii
#pragma HLS UNROLL factor = unroll_factor
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 2048
        output.write(weight[idx][j]);
      }
    }
  }
#endif
};
} // namespace vhn
