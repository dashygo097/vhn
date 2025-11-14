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

template <int... HIDDEN_DIMS> struct MLPHParams {
  static constexpr int n_layers = sizeof...(HIDDEN_DIMS);
  static constexpr int hidden_dims[sizeof...(HIDDEN_DIMS)] = {HIDDEN_DIMS...};
  static constexpr int input_dim = hidden_dims[0];
  static constexpr int output_dim = hidden_dims[n_layers - 1];
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
  static constexpr int input_dim = HParams::input_dim;
  static constexpr int output_dim = HParams::output_dim;
  static constexpr OptLevel opt_level = OPT_NONE;

  template <int IN_DIM, int OUT_DIM> using Weight_t = dtype[OUT_DIM][IN_DIM];
  template <int OUT_DIM> using Bias_t = dtype[OUT_DIM];

  MLP() = default;
  ~MLP() = default;

#ifndef __VITIS_HLS__
  static std::string type() { return "MLP"; }

  static json hparams() {
    json j;

    json hidden_dims_json = json::array();
    for (int i = 0; i < n_layers; i++) {
      hidden_dims_json.push_back(HParams::hidden_dims[i]);
    }
    j["hidden_dims"] = hidden_dims_json;

    return j;
  }

  static json hls_cfg() { return json::object(); }

  static std::vector<json> submodules() {
    std::vector<json> subs;

    for (int i = 0; i < n_layers - 1; i++) {
      json layer;
      layer["type"] = "Linear";
      layer["layer_idx"] = i;

      json layer_params;
      layer_params["in_features"] = HParams::hidden_dims[i];
      layer_params["out_features"] = HParams::hidden_dims[i + 1];
      layer["params"] = layer_params;

      layer["opt_level"] = "OPT_NONE";
      layer["hls_cfg"] = json::object();
      layer["submodules"] = json::array();

      subs.push_back(layer);
    }

    return subs;
  }
#endif

  template <typename... WeightBiasPairs>
  static void forward(dtype output[output_dim], const dtype input[input_dim],
                      WeightBiasPairs... params) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    forward_1d_impl<0>(output, input, params...);
  }

  template <typename... WeightBiasPairs>
  static void forward(dtype output[][output_dim],
                      const dtype input[][input_dim], const int batch_size,
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
      forward_1d_impl<0>(&output[b * output_dim], &input[b * input_dim],
                         params...);
    }
  }

private:
  template <int LayerIdx, typename W_t, typename B_t, typename... Rest>
  static void forward_1d_impl(dtype *output, const dtype *input, const W_t &w,
                              const B_t &b, Rest... rest) {
    static constexpr int in_dim = HParams::hidden_dims[LayerIdx];
    static constexpr int out_dim = HParams::hidden_dims[LayerIdx + 1];
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

// ============================================================================
// Optimized version (OPT_ENABLED)
// ============================================================================
template <typename DType, typename HParams, typename Config>
class MLP<DType, HParams, Config, OPT_ENABLED>
    : public BaseModule<MLP<DType, HParams, Config, OPT_ENABLED>, DType,
                        OPT_ENABLED> {
public:
  using dtype = DType;
  static constexpr int n_layers = HParams::n_layers;
  static constexpr int input_dim = HParams::input_dim;
  static constexpr int output_dim = HParams::output_dim;
  static constexpr OptLevel opt_level = OPT_ENABLED;

  template <int IN_DIM, int OUT_DIM> using Weight_t = dtype[OUT_DIM][IN_DIM];

  template <int OUT_DIM> using Bias_t = dtype[OUT_DIM];

  MLP() = default;
  ~MLP() = default;

#ifndef __VITIS_HLS__
  static std::string type() { return "MLP"; }

  static json hparams() {
    json j;
    j["input_dim"] = input_dim;
    j["output_dim"] = output_dim;
    j["n_layers"] = n_layers;

    json hidden_dims_json = json::array();
    for (int i = 0; i < n_layers; i++) {
      hidden_dims_json.push_back(HParams::hidden_dims[i]);
    }
    j["hidden_dims"] = hidden_dims_json;

    return j;
  }

  static json hls_cfg() {
    json j;
    return j;
  }

  static std::vector<json> submodules() {
    std::vector<json> subs;

    for (int i = 0; i < n_layers - 1; i++) {
      json layer;
      layer["type"] = "Linear";
      layer["layer_idx"] = i;

      json layer_hparams;
      layer_hparams["in_features"] = HParams::hidden_dims[i];
      layer_hparams["out_features"] = HParams::hidden_dims[i + 1];
      layer["hparams"] = layer_hparams;

      layer["opt_level"] = "OPT_ENABLED";

      json layer_hls_cfg;
      get_layer_hls_config<decltype(i)>(layer_hls_cfg, i);
      layer["hls_cfg"] = layer_hls_cfg;

      layer["submodules"] = json::array();

      subs.push_back(layer);
    }

    return subs;
  }

private:
  template <typename T> static void get_layer_hls_config(json &j, int idx) {
    j["unroll_factor"] = 1;
    j["partition_factor"] = 1;
    j["pipeline_ii"] = 1;
  }

public:
#endif

  template <int LayerIdx>
  using LayerConfig =
      typename std::conditional<(LayerIdx < n_layers),
                                typename Config::template layer<LayerIdx>,
                                void>::type;

  template <int LayerIdx> static constexpr OptLevel get_layer_opt() {
    using layer_config = LayerConfig<LayerIdx>;
    return std::is_same<layer_config, void>::value ? OPT_NONE : OPT_ENABLED;
  }

  template <typename... WeightBiasPairs>
  static void forward(dtype output[output_dim], const dtype input[input_dim],
                      WeightBiasPairs... params) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    forward_1d_impl<0>(output, input, params...);
  }

  template <typename... WeightBiasPairs>
  static void forward(dtype output[][output_dim],
                      const dtype input[][input_dim], const int batch_size,
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
      forward_1d_impl<0>(&output[b * output_dim], &input[b * input_dim],
                         params...);
    }
  }

private:
  template <int LayerIdx, typename W_t, typename B_t, typename... Rest>
  static void forward_1d_impl(dtype *output, const dtype *input, const W_t &w,
                              const B_t &b, Rest... rest) {
    static constexpr int in_dim = HParams::hidden_dims[LayerIdx];
    static constexpr int out_dim = HParams::hidden_dims[LayerIdx + 1];
    static constexpr bool is_last = (LayerIdx == n_layers - 2);
    static constexpr OptLevel layer_opt = get_layer_opt<LayerIdx>();

    dtype layer_out[out_dim];

    using fc = Linear<dtype, LinearHParams<in_dim, out_dim>,
                      LayerConfig<LayerIdx>, layer_opt>;
    fc::forward(layer_out, input, w, b);

    if constexpr (!is_last) {
      if constexpr (sizeof...(Rest) > 0) {
        forward_1d_impl<LayerIdx + 1>(output, layer_out, rest...);
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

} // namespace vhn
