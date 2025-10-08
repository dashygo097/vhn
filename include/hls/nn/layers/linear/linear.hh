#pragma once

#include "../../../opt_level.hh"

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace hls_nn {
template <const int UNROLL_FACTOR, const int PARTITION_FACTOR,
          const int PIPELINE_II>
struct LinearHLSConfig {
  static constexpr int _unroll_factor = UNROLL_FACTOR;
  static constexpr int _partition_factor = PARTITION_FACTOR;
  static constexpr int _pipeline_ii = PIPELINE_II;
};

template <typename DType, const int IN_FEATURES, const int OUT_FEATURES,
          typename Config, OptLevel OPT_LEVEL = OPT_NONE>
class Linear {
public:
  using dtype = DType;
  static constexpr int in_features = IN_FEATURES;
  static constexpr int out_features = OUT_FEATURES;
  static constexpr OptLevel opt_level = OPT_LEVEL;

  Linear() = default;
  ~Linear() = default;

  static void forward(dtype output[out_features],
                      const dtype input[in_features],
                      const dtype weight[out_features][in_features],
                      const dtype bias[out_features]);

private:
};

template <typename DType, const int IN_FEATURES, const int OUT_FEATURES>
class Linear<DType, IN_FEATURES, OUT_FEATURES, void, OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int in_features = IN_FEATURES;
  static constexpr int out_features = OUT_FEATURES;
  static constexpr OptLevel opt_level = OPT_NONE;
  Linear() = default;
  ~Linear() = default;
  static void forward(dtype output[out_features],
                      const dtype input[in_features],
                      const dtype weight[out_features][in_features],
                      const dtype bias[out_features]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif

  OUTER_LOOP:
    for (int i = 0; i < out_features; i++) {
      dtype acc = dtype(0.0f);
    INNER_LOOP:
      for (int j = 0; j < in_features; j++) {
        acc += input[j] * weight[i][j];
      }
      output[i] = acc + bias[i];
    }
  }

private:
};

template <typename DType, const int IN_FEATURES, const int OUT_FEATURES,
          typename Config>
class Linear<DType, IN_FEATURES, OUT_FEATURES, Config, OPT_ENABLED> {
public:
  using dtype = DType;
  static constexpr int in_features = IN_FEATURES;
  static constexpr int out_features = OUT_FEATURES;
  static constexpr OptLevel opt_level = OPT_ENABLED;

  static constexpr int unroll_factor = Config::_unroll_factor;
  static constexpr int partition_factor = Config::_partition_factor;
  static constexpr int pipeline_ii = Config::_pipeline_ii;

  Linear() = default;
  ~Linear() = default;

  static void forward(dtype output[out_features],
                      const dtype input[in_features],
                      const dtype weight[out_features][in_features],
                      const dtype bias[out_features]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = input cyclic factor = partition_factor
#pragma HLS ARRAY_PARTITION variable = weight cyclic factor =                  \
    partition_factor dim = 2
#pragma HLS ARRAY_PARTITION variable = bias cyclic factor = partition_factor
#endif

  OUTER_LOOP:
    for (int i = 0; i < out_features; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#endif

      dtype acc = dtype(0.0f);
    INNER_LOOP:
      for (int j = 0; j < in_features; j++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = unroll_factor
#endif
        acc += input[j] * weight[i][j];
      }

      output[i] = acc + bias[i];
    }
  }

private:
};

template <typename DType, const int IN_FEATURES, const int OUT_FEATURES,
          typename Config>
class LinearBatched {
public:
  using dtype = DType;
  static constexpr int in_features = IN_FEATURES;
  static constexpr int out_features = OUT_FEATURES;

  LinearBatched() = default;
  ~LinearBatched() = default;

  static void forward(dtype output[][out_features],
                      const dtype input[][in_features],
                      const dtype weight[out_features][in_features],
                      const dtype bias[out_features], int batch_size) {
#ifdef __VITIS_HLS__
#pragma HLS DATAFLOW
#endif

  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      Linear<DType, IN_FEATURES, OUT_FEATURES, Config>::forward(
          input[b], weight, bias, output[b]);
    }
  }

private:
};

} // namespace hls_nn
