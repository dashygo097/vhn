#pragma once

#include "./postnorm.hh"
#include "./prenorm.hh"

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

enum NormType { POSTNORM, PRENORM };

namespace vhn {
template <typename DType, typename HParams, typename Config, OptLevel OPT_LEVEL>
class AddNorm;

template <typename NORM_HParams, NormType NORM_TYPE> struct AddNormHParams {
  using norm_hparams = NORM_HParams;

  static constexpr int d_model = NORM_HParams::hidden_dim;
  static constexpr NormType norm_type = NORM_TYPE;
};

// ============================================================================
// AddNorm specialization for OPT_NONE (using PreNorm, PostNorm)
// ============================================================================
template <typename DType, typename HParams>
class AddNorm<DType, HParams, void, OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int d_model = HParams::d_model;
  static constexpr NormType norm_type = HParams::norm_type;
  static constexpr OptLevel opt_level = OPT_NONE;

  using gamma_t = dtype[d_model];
  using beta_t = dtype[d_model];

  AddNorm() = default;
  ~AddNorm() = default;

  using addnorm =
      typename std::conditional<norm_type == POSTNORM,
                                PostNorm<DType, HParams, void, OPT_NONE>,
                                PreNorm<DType, HParams, void, OPT_NONE>>::type;

  static void forward(dtype output[d_model], const dtype input[d_model],
                      const dtype residual[d_model], const gamma_t gamma,
                      const beta_t beta) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    addnorm::forward(output, input, residual, gamma, beta);
  }

  static void forward(dtype output[][d_model], const dtype input[][d_model],
                      const dtype residual[][d_model], const int actual_len,
                      const gamma_t gamma, const beta_t beta) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#endif
    addnorm::forward(output, input, residual, actual_len, gamma, beta);
  }

  static void forward(dtype *output, const dtype *input, const dtype *residual,
                      const int actual_len, const gamma_t gamma,
                      const beta_t beta) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#endif
    addnorm::forward(output, input, residual, actual_len, gamma, beta);
  }

#ifdef __VITIS_HLS__
  static void forward(hls::stream<dtype> &output_stream,
                      hls::stream<dtype> &input_stream,
                      hls::stream<dtype> &residual_stream, const int actual_len,
                      const gamma_t gamma, const beta_t beta) {
#pragma HLS INLINE off
    forward_stream_impl(output_stream, input_stream, residual_stream,
                        actual_len, gamma, beta);
  }
#endif

private:
#ifdef __VITIS_HLS__
  static void forward_stream_impl(hls::stream<dtype> &output_stream,
                                  hls::stream<dtype> &input_stream,
                                  hls::stream<dtype> &residual_stream,
                                  const int actual_len, const gamma_t gamma,
                                  const beta_t beta) {
  SEQ_LOOP:
    for (int i = 0; i < actual_len; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
      dtype input_buffer[d_model];
      dtype residual_buffer[d_model];

    READ_INPUT:
      for (int j = 0; j < d_model; j++) {
        input_buffer[j] = input_stream.read();
        residual_buffer[j] = residual_stream.read();
      }

      dtype output_buffer[d_model];
      addnorm::forward_single(output_buffer, input_buffer, residual_buffer,
                              gamma, beta);

    WRITE_OUTPUT:
      for (int j = 0; j < d_model; j++) {
        output_stream.write(output_buffer[j]);
      }
    }
  }
#endif
};

template <typename NORM_CONFIG, bool DATAFLOW_ENABLED, int PIPELINE_II,
          int PARTITION_FACTOR>
struct AddNormConfig {
  using norm_config = NORM_CONFIG;

  static constexpr int dataflow_enabled = DATAFLOW_ENABLED;
  static constexpr int pipeline_ii = PIPELINE_II;
  static constexpr int partition_factor = PARTITION_FACTOR;
};

// ============================================================================
// AddNorm specialization for OPT_ENABLED
// ============================================================================
template <typename DType, typename HParams, typename Config>
class AddNorm<DType, HParams, Config, OPT_ENABLED> {
public:
  using dtype = DType;
  static constexpr int d_model = HParams::d_model;
  static constexpr NormType norm_type = HParams::norm_type;
  static constexpr OptLevel opt_level = OPT_ENABLED;

  static constexpr int partition_factor = Config::partition_factor;
  static constexpr int pipeline_ii = Config::pipeline_ii;
  static constexpr int dataflow_enabled = Config::dataflow_enabled;

  using norm_config = typename Config::norm_config;

  using gamma_t = dtype[d_model];
  using beta_t = dtype[d_model];

  AddNorm() = default;
  ~AddNorm() = default;

  static constexpr bool is_norm_optimized =
      !std::is_same<norm_config, void>::value;

  using addnorm = typename std::conditional<
      norm_type == POSTNORM,
      PostNorm<DType, HParams, norm_config,
               is_norm_optimized ? OPT_ENABLED : OPT_NONE>,
      PreNorm<DType, HParams, norm_config,
              is_norm_optimized ? OPT_ENABLED : OPT_NONE>>::type;

  static void forward(dtype output[d_model], const dtype input[d_model],
                      const dtype residual[d_model], const gamma_t gamma,
                      const beta_t beta) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS PIPELINE II = 1
#endif
    addnorm::forward(output, input, residual, gamma, beta);
  }

  static void forward(dtype output[][d_model], const dtype input[][d_model],
                      const dtype residual[][d_model], const int actual_len,
                      const gamma_t gamma, const beta_t beta) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
    if constexpr (dataflow_enabled) {
#pragma HLS DATAFLOW
    }
#pragma HLS ARRAY_PARTITION variable = gamma type = cyclic factor =            \
    partition_factor
#pragma HLS ARRAY_PARTITION variable = beta type = cyclic factor =             \
    partition_factor
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#endif
    addnorm::forward(output, input, residual, actual_len, gamma, beta);
  }

  static void forward(dtype *output, const dtype *input, const dtype *residual,
                      const int actual_len, const gamma_t gamma,
                      const beta_t beta) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
    if constexpr (dataflow_enabled) {
#pragma HLS DATAFLOW
    }
#pragma HLS ARRAY_PARTITION variable = gamma type = cyclic factor =            \
    partition_factor
#pragma HLS ARRAY_PARTITION variable = beta type = cyclic factor =             \
    partition_factor
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#endif
    addnorm::forward(output, input, residual, actual_len, gamma, beta);
  }

#ifdef __VITIS_HLS__
  static void forward(hls::stream<dtype> &output_stream,
                      hls::stream<dtype> &input_stream,
                      hls::stream<dtype> &residual_stream, const int actual_len,
                      const gamma_t gamma, const beta_t beta) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#pragma HLS ARRAY_PARTITION variable = gamma type = cyclic factor =            \
    partition_factor
#pragma HLS ARRAY_PARTITION variable = beta type = cyclic factor =             \
    partition_factor
    forward_stream_impl(output_stream, input_stream, residual_stream,
                        actual_len, gamma, beta);
  }
#endif

private:
#ifdef __VITIS_HLS__
  static void forward_stream_impl(hls::stream<dtype> &output_stream,
                                  hls::stream<dtype> &input_stream,
                                  hls::stream<dtype> &residual_stream,
                                  const int actual_len, const gamma_t gamma,
                                  const beta_t beta) {
    dtype input_buffer[d_model];
    dtype residual_buffer[d_model];
    dtype output_buffer[d_model];

    constexpr bool should_partition =
        (partition_factor > 1) && (d_model <= 2048);
    if constexpr (should_partition) {
#pragma HLS ARRAY_PARTITION variable = input_buffer type = cyclic factor =     \
    partition_factor
#pragma HLS ARRAY_PARTITION variable = residual_buffer type = cyclic factor =  \
    partition_factor
#pragma HLS ARRAY_PARTITION variable = output_buffer type = cyclic factor =    \
    partition_factor
    }

  SEQ_LOOP:
    for (int i = 0; i < actual_len; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 512
#pragma HLS PIPELINE II = pipeline_ii

    READ_INPUT:
      for (int j = 0; j < d_model; j++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 2048
        input_buffer[j] = input_stream.read();
        residual_buffer[j] = residual_stream.read();
      }

      addnorm::forward_single(output_buffer, input_buffer, residual_buffer,
                              gamma, beta);

    WRITE_OUTPUT:
      for (int j = 0; j < d_model; j++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 2048
        output_stream.write(output_buffer[j]);
      }
    }
  }
#endif
};

} // namespace vhn
