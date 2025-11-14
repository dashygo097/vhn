#pragma once

#include "../base.hh"
#include "../layers/linear.hh"
#include "../opt_level.hh"

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace vhn {

template <typename DType, typename HParams, typename Config, OptLevel OPT_LEVEL>
class MLP;

template <int... HIDDEN_FEATURES> struct MLPHParams {
  static constexpr int n_layers = sizeof...(HIDDEN_FEATURES) - 1;
  static constexpr int hidden_features[sizeof...(HIDDEN_FEATURES)] = {
      HIDDEN_FEATURES...};
  static constexpr int in_features = hidden_features[0];
  static constexpr int out_features = hidden_features[n_layers];
};

// ============================================================================
// Non-optimized version (OPT_NONE)
// ============================================================================
template <typename DType, typename HParams>
class MLP<DType, HParams, void, OPT_NONE>
    : public BaseModule<MLP<DType, HParams, void, OPT_NONE>, DType, OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int n_layers = HParams::n_layers;
  static constexpr int in_features = HParams::in_features;
  static constexpr int out_features = HParams::out_features;
  static constexpr OptLevel opt_level = OPT_NONE;

  template <int IN_DIM, int OUT_DIM> using Weight_t = dtype[OUT_DIM][IN_DIM];
  template <int OUT_DIM> using Bias_t = dtype[OUT_DIM];

  MLP() = default;
  ~MLP() = default;

  template <typename... WeightBiasPairs>
  static void forward(dtype output[out_features],
                      const dtype input[in_features],
                      WeightBiasPairs... params) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    forward_1d_impl<0>(output, input, params...);
  }

  template <typename... WeightBiasPairs>
  static void forward(dtype output[][out_features],
                      const dtype input[][in_features], const int batch_size,
                      WeightBiasPairs... params) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_1d_impl<0>(output[b], input[b], params...);
    }
  }

  template <typename... WeightBiasPairs>
  static void forward(dtype *output, const dtype *input, const int batch_size,
                      WeightBiasPairs... params) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_1d_impl<0>(&output[b * out_features], &input[b * in_features],
                         params...);
    }
  }

private:
  template <int LayerIdx, typename W_t, typename B_t, typename... Rest>
  static void forward_1d_impl(dtype *output, const dtype *input, const W_t &w,
                              const B_t &b, Rest... rest) {
    static constexpr int in_dim = HParams::hidden_features[LayerIdx];
    static constexpr int out_dim = HParams::hidden_features[LayerIdx + 1];
    static constexpr bool is_last = (LayerIdx == n_layers - 2);

    dtype layer_out[out_dim];

    using fc = Linear<dtype, LinearHParams<in_dim, out_dim>, void, OPT_NONE>;
    fc::forward(layer_out, input, w, b);

    if constexpr (!is_last) {
      if constexpr (sizeof...(Rest) > 0) {
        forward_1d_impl<LayerIdx + 1>(output, layer_out, rest...);
      }
    } else {
    LAST_COPY_LOOP:
      for (int i = 0; i < out_dim; i++) {
        output[i] = layer_out[i];
      }
    }
  }
};
} // namespace vhn
