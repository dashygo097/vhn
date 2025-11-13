#pragma once

#include "../hls_config.hh"
#include "../layers/layers.hh"
#include "../opt_level.hh"

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace hls_nn {
template <typename DType, const int OUTPUT_DIM,
          template <typename, int, typename, OptLevel> class ActLayer,
          typename Config = void, OptLevel OPT_LEVEL = OPT_NONE,
          int... HiddenDims>
class MLP;

template <typename DType, const int OUTPUT_DIM,
          template <typename, int, typename, OptLevel> class ActLayer,
          int... HiddenDims>
class MLP<DType, OUTPUT_DIM, ActLayer, void, OPT_NONE, HiddenDims...> {
public:
  using dtype = DType;
  static constexpr int output_dim = OUTPUT_DIM;
  static constexpr int n_layers = sizeof...(HiddenDims) - 1;
  static constexpr int hidden_dims[sizeof...(HiddenDims)] = {HiddenDims...};
  static constexpr int input_dim = hidden_dims[0];
  static constexpr OptLevel opt_level = OPT_NONE;

  MLP() = default;
  ~MLP() = default;

  template <int IN_DIM, int OUT_DIM> using Weight_t = dtype[OUT_DIM][IN_DIM];
  template <int OUT_DIM> using Bias_t = dtype[OUT_DIM];

  template <typename... WeightBiasPairs>
  static void forward(dtype output[output_dim], const dtype input[input_dim],
                      WeightBiasPairs... params) {
    forward_1d_impl<0>(output, input, params...);
  }

  template <typename... WeightBiasPairs>
  static void forward(dtype output[][output_dim],
                      const dtype input[][input_dim], const int batch_size,
                      WeightBiasPairs... params) {
#ifdef __VITIS_HLS__
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
#pragma HLS DATAFLOW
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_1d_impl<0>(&output[b * output_dim], &input[b * input_dim],
                         params...);
    }
  }

private:
  template <int LayerIdx, typename W_t, typename B_t, typename... Rest>
  static void forward_1d_impl(dtype *output, const dtype *input, const W_t &w,
                              const B_t &b, Rest... rest) {
    static constexpr int in_dim = hidden_dims[LayerIdx];
    static constexpr int out_dim = hidden_dims[LayerIdx + 1];
    static constexpr bool is_last = (LayerIdx == n_layers - 1);

    dtype layer_out[out_dim];

    using fc = Linear<DType, in_dim, out_dim, void, OPT_NONE>;
    fc::forward(layer_out, input, w, b);

    if constexpr (!is_last) {
      dtype act_out[out_dim];
      using act = ActLayer<DType, out_dim, void, OPT_NONE>;
      act::forward(act_out, layer_out);

      if constexpr (sizeof...(Rest) > 0) {
        forward_1d_impl<LayerIdx + 1>(output, act_out, rest...);
      }
    } else {
    LAST_COPY_LOOP:
      for (int i = 0; i < out_dim; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = 1
#endif
        output[i] = layer_out[i];
      }
    }
  }
};

template <typename DType, const int OUTPUT_DIM,
          template <typename, int, typename, OptLevel> class ActLayer,
          typename Config, int... HiddenDims>
class MLP<DType, OUTPUT_DIM, ActLayer, Config, OPT_ENABLED, HiddenDims...> {
public:
  using dtype = DType;
  static constexpr int output_dim = OUTPUT_DIM;
  static constexpr int n_layers = sizeof...(HiddenDims) - 1;
  static constexpr int hidden_dims[sizeof...(HiddenDims)] = {HiddenDims...};
  static constexpr int input_dim = hidden_dims[0];
  static constexpr OptLevel opt_level = OPT_ENABLED;

  MLP() = default;
  ~MLP() = default;

  template <int IN_DIM, int OUT_DIM> using Weight_t = dtype[OUT_DIM][IN_DIM];
  template <int OUT_DIM> using Bias_t = dtype[OUT_DIM];

  template <int LayerIdx>
  using LayerConfig =
      typename std::conditional <
      LayerIdx<n_layers, typename Config::template layer<LayerIdx>, void>::type;

  template <int LayerIdx> static constexpr OptLevel get_layer_opt() {
    using layer_config = LayerConfig<LayerIdx>;
    return std::is_same<layer_config, void>::value ? OPT_NONE : OPT_ENABLED;
  }

  template <typename... WeightBiasPairs>
  static void forward(dtype output[output_dim], const dtype input[input_dim],
                      WeightBiasPairs... params) {
    forward_1d_impl<0>(output, input, params...);
  }

  template <typename... WeightBiasPairs>
  static void forward(dtype output[][output_dim],
                      const dtype input[][input_dim], const int batch_size,
                      WeightBiasPairs... params) {
#ifdef __VITIS_HLS__
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
#pragma HLS DATAFLOW
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_1d_impl<0>(&output[b * output_dim], &input[b * input_dim],
                         params...);
    }
  }

private:
  template <int LayerIdx, typename W_t, typename B_t, typename... Rest>
  static void forward_1d_impl(dtype *output, const dtype *input, const W_t &w,
                              const B_t &b, Rest... rest) {
    static constexpr int in_dim = hidden_dims[LayerIdx];
    static constexpr int out_dim = hidden_dims[LayerIdx + 1];
    static constexpr bool is_last = (LayerIdx == n_layers - 1);
    static constexpr OptLevel layer_opt = get_layer_opt<LayerIdx>();

    dtype layer_out[out_dim];

    using fc = Linear<DType, in_dim, out_dim, LayerConfig<LayerIdx>, layer_opt>;
    fc::forward(layer_out, input, w, b);

    if constexpr (!is_last) {
      dtype act_out[out_dim];
      using act_config = typename Config::act;
      static constexpr OptLevel act_opt =
          std::is_same<act_config, void>::value ? OPT_NONE : OPT_ENABLED;
      using act = ActLayer<DType, out_dim, act_config, act_opt>;
      act::forward(act_out, layer_out);

      if constexpr (sizeof...(Rest) > 0) {
        forward_1d_impl<LayerIdx + 1>(output, act_out, rest...);
      }
    } else {
    LAST_COPY_LOOP:
      for (int i = 0; i < out_dim; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = 1
#endif
        output[i] = layer_out[i];
      }
    }
  }
};

} // namespace hls_nn
