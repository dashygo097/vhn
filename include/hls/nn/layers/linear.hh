#pragma once

#include "../../opt_level.hh"

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace hls_nn {
template <typename DType, const int IN_FEATURES, const int OUT_FEATURES,
          OptLevel OPT_LEVEL = OPT_NONE>
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
class Linear<DType, IN_FEATURES, OUT_FEATURES, OPT_NONE> {
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

  OUT_LOOP:
    for (int i = 0; i < out_features; i++) {
      dtype acc = 0;

    IN_LOOP:
      for (int j = 0; j < in_features; j++) {
        acc += input[j] * weight[i][j];
      }

      output[i] = acc + bias[i];
    }
  }

private:
};

template <typename DType, const int IN_FEATURES, const int OUT_FEATURES>
class Linear<DType, IN_FEATURES, OUT_FEATURES, OPT_LATENCY> {
public:
  using dtype = DType;
  static constexpr int in_features = IN_FEATURES;
  static constexpr int out_features = OUT_FEATURES;
  static constexpr OptLevel opt_level = OPT_LATENCY;

  Linear() = default;
  ~Linear() = default;

  static void forward(dtype output[out_features],
                      const dtype input[in_features],
                      const dtype weight[out_features][in_features],
                      const dtype bias[out_features]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = input complete
#pragma HLS ARRAY_PARTITION variable = weight block factor = 16 dim = 2
#endif

  OUT_LOOP:
    for (int i = 0; i < out_features; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = 1 rewind
#endif
      dtype acc = 0;

    IN_LOOP:
      for (int j = 0; j < in_features; j++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL
#endif
        acc += input[j] * weight[i][j];
      }

      output[i] = acc + bias[i];
    }
  }

private:
};

template <typename DType, const int IN_FEATURES, const int OUT_FEATURES>
class Linear<DType, IN_FEATURES, OUT_FEATURES, OPT_THROUGHPUT> {
public:
  using dtype = DType;
  static constexpr int in_features = IN_FEATURES;
  static constexpr int out_features = OUT_FEATURES;
  static constexpr OptLevel opt_level = OPT_THROUGHPUT;

  Linear() = default;
  ~Linear() = default;

  static void forward(dtype output[out_features],
                      const dtype input[in_features],
                      const dtype weight[out_features][in_features],
                      const dtype bias[out_features]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = input cyclic factor = 8
#pragma HLS ARRAY_PARTITION variable = weight cyclic factor = 8 dim = 2
#endif

  OUT_LOOP:
    for (int i = 0; i < out_features; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = 1
#endif
      dtype acc = 0;

    IN_LOOP:
      for (int j = 0; j < in_features; j++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = 4
#endif
        acc += input[j] * weight[i][j];
      }

      output[i] = acc + bias[i];
    }
  }

private:
};

template <typename DType, const int IN_FEATURES, const int OUT_FEATURES,
          OptLevel OPT_LEVEL = OPT_NONE>
class LinearBatched {
public:
  using dtype = DType;
  static constexpr int in_features = IN_FEATURES;
  static constexpr int out_features = OUT_FEATURES;
  static constexpr OptLevel opt_level = OPT_LEVEL;

  LinearBatched() = default;
  ~LinearBatched() = default;

  static void forward(dtype output[][out_features],
                      const dtype input[][in_features],
                      const dtype weight[out_features][in_features],
                      const dtype bias[out_features], int batch_size) {
#ifdef __VITIS_HLS__
#pragma HLS DATAFLOW
#pragma HLS ARRAY_PARTITION variable = weight cyclic factor = 8 dim = 2
#endif

  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      Linear<DType, IN_FEATURES, OUT_FEATURES, OPT_LEVEL>::forward(
          input[b], weight, bias, output[b]);
    }
  }

private:
};

} // namespace hls_nn
