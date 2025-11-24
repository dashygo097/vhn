#pragma once

#include "../../../layers/linear.hh"
#include "../../../operators/elementwise.hh"
#include <type_traits>

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace vhn {

template <typename DType, typename HParams, typename Config = void,
          OptLevel OPT_LEVEL = OPT_NONE>
class FFN;

template <typename FC1_HParams, typename ACT_HParams, typename FC2_HParams,
          int MAX_SEQ_LEN>
struct FFNHParams {
  using fc1_hparams = FC1_HParams;
  using act_hparams = ACT_HParams;
  using fc2_hparams = FC2_HParams;

  static constexpr int d_model = FC1_HParams::in_features;
  static constexpr int d_ff = FC1_HParams::out_features;
  static constexpr int max_seq_len = MAX_SEQ_LEN;
  using act_type = typename ACT_HParams::impl;
};

// ============================================================================
// FFN specialization for OPT_NONE
// ============================================================================
template <typename DType, typename HParams>
class FFN<DType, HParams, void, OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int d_model = HParams::d_model;
  static constexpr int d_ff = HParams::d_ff;
  static constexpr int max_seq_len = HParams::max_seq_len;
  static constexpr OptLevel opt_level = OPT_NONE;

  using W1_t = dtype[d_ff][d_model];
  using b1_t = dtype[d_ff];
  using W2_t = dtype[d_model][d_ff];
  using b2_t = dtype[d_model];

  using fc1_hparams = typename HParams::fc1_hparams;
  using act_hparams = typename HParams::act_hparams;
  using fc2_hparams = typename HParams::fc2_hparams;

  using fc1 = Linear<dtype, fc1_hparams, void, OPT_NONE>;
  using act = Elementwise<dtype, act_hparams, void, OPT_NONE>;
  using fc2 = Linear<dtype, fc2_hparams, void, OPT_NONE>;

  FFN() = default;
  ~FFN() = default;

  static void forward(dtype output[][d_model], const dtype input[][d_model],
                      const int actual_len, const W1_t w1, const b1_t b1,
                      const W2_t w2, const b2_t b2) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    dtype fc1_out[max_seq_len][d_ff];
    dtype act_out[max_seq_len][d_ff];

    fc1::forward(fc1_out, input, actual_len, w1, b1);
    act::forward(act_out, fc1_out, actual_len);
    fc2::forward(output, act_out, actual_len, w2, b2);
  }

  static void forward(dtype output[d_model], const dtype input[d_model],
                      const W1_t w1, const b1_t b1, const W2_t w2,
                      const b2_t b2) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    dtype fc1_out[d_ff];
    dtype act_out[d_ff];

    fc1::forward(fc1_out, input, w1, b1);
    act::forward(act_out, fc1_out);
    fc2::forward(output, act_out, w2, b2);
  }

  static void forward(dtype *output, const dtype *input, const int actual_len,
                      const W1_t w1, const b1_t b1, const W2_t w2,
                      const b2_t b2) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    dtype *fc1_out = new dtype[actual_len * d_ff];
    dtype *act_out = new dtype[actual_len * d_ff];

    fc1::forward(fc1_out, input, actual_len, w1, b1);
    act::forward(act_out, fc1_out, actual_len);
    fc2::forward(output, act_out, actual_len, w2, b2);

    delete[] fc1_out;
    delete[] act_out;
  }

#ifdef __VITIS_HLS__
  static void forward(hls::stream<dtype> &output_stream,
                      hls::stream<dtype> &input_stream, const int actual_len,
                      const W1_t w1, const b1_t b1, const W2_t w2,
                      const b2_t b2) {
#pragma HLS INLINE off

    hls::stream<dtype> fc1_stream("fc1_stream");
    hls::stream<dtype> act_stream("act_stream");

  PROCESS_STREAM:
    for (int i = 0; i < actual_len; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
      fc1::forward(fc1_stream, input_stream, w1, b1);
      act::forward(act_stream, fc1_stream);
      fc2::forward(output_stream, act_stream, w2, b2);
    }
  }
#endif
};

template <typename FC1_CONFIG, typename ACT_CONFIG, typename FC2_CONFIG,
          int DATAFLOW_DEPTH, int SEQ_UNROLL, int MEMORY_PARTITION>
struct FFNConfig {
  using fc1_config = FC1_CONFIG;
  using act_config = ACT_CONFIG;
  using fc2_config = FC2_CONFIG;

  static constexpr int dataflow_depth = DATAFLOW_DEPTH;
  static constexpr int seq_unroll = SEQ_UNROLL;
  static constexpr int memory_partition = MEMORY_PARTITION;
};

// ============================================================================
// FFN specialization for OPT_ENABLED
// ============================================================================
template <typename DType, typename HParams, typename Config>
class FFN<DType, HParams, Config, OPT_ENABLED> {
public:
  using dtype = DType;
  static constexpr int d_model = HParams::d_model;
  static constexpr int d_ff = HParams::d_ff;
  static constexpr int max_seq_len = HParams::max_seq_len;
  static constexpr OptLevel opt_level = OPT_ENABLED;

  static constexpr int dataflow_depth = Config::dataflow_depth;
  static constexpr int seq_unroll = Config::seq_unroll;
  static constexpr int memory_partition = Config::memory_partition;

  using W1_t = dtype[d_ff][d_model];
  using b1_t = dtype[d_ff];
  using W2_t = dtype[d_model][d_ff];
  using b2_t = dtype[d_model];

  using fc1_hparams = typename HParams::fc1_hparams;
  using act_hparams = typename HParams::act_hparams;
  using fc2_hparams = typename HParams::fc2_hparams;

  using fc1_config = typename Config::fc1_config;
  using act_config = typename Config::act_config;
  using fc2_config = typename Config::fc2_config;

  static constexpr bool fc1_is_optimized =
      !std::is_same<fc1_config, void>::value;
  static constexpr bool act_is_optimized =
      !std::is_same<act_config, void>::value;
  static constexpr bool fc2_is_optimized =
      !std::is_same<fc2_config, void>::value;

  using fc1 = Linear<dtype, fc1_hparams, fc1_config,
                     fc1_is_optimized ? OPT_ENABLED : OPT_NONE>;
  using act = Elementwise<dtype, act_hparams, act_config,
                          act_is_optimized ? OPT_ENABLED : OPT_NONE>;
  using fc2 = Linear<dtype, fc2_hparams, fc2_config,
                     fc2_is_optimized ? OPT_ENABLED : OPT_NONE>;

  FFN() = default;
  ~FFN() = default;

  static void forward(dtype output[][d_model], const dtype input[][d_model],
                      const int actual_len, const W1_t w1, const b1_t b1,
                      const W2_t w2, const b2_t b2) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    dtype fc1_out[max_seq_len][d_ff];
    dtype act_out[max_seq_len][d_ff];

#ifdef __VITIS_HLS__
    constexpr bool should_partition =
        (memory_partition > 1) && (d_ff <= 4096) && (d_model <= 2048);
    if constexpr (should_partition) {
#pragma HLS ARRAY_PARTITION variable = fc1_out type = cyclic factor =          \
    memory_partition dim = 2
#pragma HLS ARRAY_PARTITION variable = act_out type = cyclic factor =          \
    memory_partition dim = 2
    } else {
#pragma HLS ARRAY_PARTITION variable = fc1_out type = cyclic factor = 4 dim = 2
#pragma HLS ARRAY_PARTITION variable = act_out type = cyclic factor = 4 dim = 2
    }
#endif

    fc1::forward(fc1_out, input, actual_len, w1, b1);
    act::forward(act_out, fc1_out, actual_len);
    fc2::forward(output, act_out, actual_len, w2, b2);
  }

  static void forward(dtype output[d_model], const dtype input[d_model],
                      const W1_t w1, const b1_t b1, const W2_t w2,
                      const b2_t b2) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS PIPELINE II = 1
#endif
    dtype fc1_out[d_ff];
    dtype act_out[d_ff];

#ifdef __VITIS_HLS__
#pragma HLS ARRAY_PARTITION variable = fc1_out type = complete
#pragma HLS ARRAY_PARTITION variable = act_out type = complete
#pragma HLS ARRAY_PARTITION variable = input type = complete
#pragma HLS ARRAY_PARTITION variable = output type = complete
#pragma HLS ARRAY_PARTITION variable = w1 type = cyclic factor =               \
    fc1_config::partition_factor dim = 2
#pragma HLS ARRAY_PARTITION variable = w2 type = cyclic factor =               \
    fc2_config::partition_factor dim = 2
#pragma HLS ARRAY_PARTITION variable = b1 type = cyclic factor =               \
    fc1_config::partition_factor
#pragma HLS ARRAY_PARTITION variable = b2 type = cyclic factor =               \
    fc2_config::partition_factor
#endif

    fc1::forward(fc1_out, input, w1, b1);
    act::forward(act_out, fc1_out);
    fc2::forward(output, act_out, w2, b2);
  }

  // ========================================================================
  // Pointer-based forward (memory-efficient batch)
  // ========================================================================
  static void forward(dtype *output, const dtype *input, const int actual_len,
                      const W1_t w1, const b1_t b1, const W2_t w2,
                      const b2_t b2) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    dtype *fc1_out = new dtype[actual_len * d_ff];
    dtype *act_out = new dtype[actual_len * d_ff];

#ifdef __VITIS_HLS__
    constexpr bool should_partition = (memory_partition > 1);
    if constexpr (!should_partition) {
#pragma HLS BIND_STORAGE variable = w1 type = rom_2p impl = bram
#pragma HLS BIND_STORAGE variable = w2 type = rom_2p impl = bram
#pragma HLS BIND_STORAGE variable = b1 type = rom_1p impl = bram
#pragma HLS BIND_STORAGE variable = b2 type = rom_1p impl = bram
    }
#endif

    fc1::forward(fc1_out, input, actual_len, w1, b1);
    act::forward(act_out, fc1_out, actual_len);
    fc2::forward(output, act_out, actual_len, w2, b2);

    delete[] fc1_out;
    delete[] act_out;
  }

#ifdef __VITIS_HLS__
  static void forward(hls::stream<dtype> &output_stream,
                      hls::stream<dtype> &input_stream, const int actual_len,
                      const W1_t w1, const b1_t b1, const W2_t w2,
                      const b2_t b2) {
#pragma HLS INLINE off

    hls::stream<dtype> fc1_stream("fc1_stream");
    hls::stream<dtype> act_stream("act_stream");

#pragma HLS STREAM variable = fc1_stream depth = dataflow_depth
#pragma HLS STREAM variable = act_stream depth = dataflow_depth

  PROCESS_STREAM:
    for (int i = 0; i < actual_len; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
      fc1::forward(fc1_stream, input_stream, w1, b1);
      act::forward(act_stream, fc1_stream);
      fc2::forward(output_stream, act_stream, w2, b2);
    }
  }
#endif
};

} // namespace vhn
