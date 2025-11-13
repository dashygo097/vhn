#pragma once

#include "../../opt_level.hh"
#include "../../tb/tb.hh"
#include "../base.hh"
#include <algorithm>
#include <iostream>
#include <random>

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace hls_nn {
template <typename DType, const int IN_FEATURES, const int OUT_FEATURES,
          typename Config = void, OptLevel OPT_LEVEL = OPT_NONE>
class Linear;

// ============================================================================
// Non-optimized version (OPT_NONE)
// ============================================================================
template <typename DType, const int IN_FEATURES, const int OUT_FEATURES>
class Linear<DType, IN_FEATURES, OUT_FEATURES, void, OPT_NONE>
    : public BaseModule<DType> {
public:
  using dtype = DType;
  static constexpr int in_features = IN_FEATURES;
  static constexpr int out_features = OUT_FEATURES;
  static constexpr OptLevel opt_level = OPT_NONE;

  using Weight_t = dtype[OUT_FEATURES][IN_FEATURES];
  using Bias_t = dtype[OUT_FEATURES];

  Linear() = default;
  ~Linear() = default;

  // Json Serialization
  std::string module_name() const override { return "Linear"; }
  std::string module_type() const override { return "Linear_OPT_NONE"; }
  json to_json() const override {
    json j;
    j["module_type"] = module_type();
    j["module_name"] = module_name();
    j["opt_level"] = "OPT_NONE";
    j["hls_config"] = json::object();
    return j;
  }

  static void forward(dtype output[OUT_FEATURES],
                      const dtype input[IN_FEATURES], const Weight_t weight,
                      const Bias_t bias) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    forward_1d_impl(output, input, weight, bias);
  }

  static void forward(dtype output[][OUT_FEATURES],
                      const dtype input[][IN_FEATURES], const int batch_size,
                      const Weight_t weight, const Bias_t bias) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_1d_impl(output[b], input[b], weight, bias);
    }
  }

  static void forward(dtype *output, const dtype *input, const int batch_size,
                      const Weight_t weight, const Bias_t bias) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_1d_impl(&output[b * OUT_FEATURES], &input[b * IN_FEATURES],
                      weight, bias);
    }
  }

#ifdef __VITIS_HLS__
  static void forward(hls::stream<dtype> &output_stream,
                      hls::stream<dtype> &input_stream, const Weight_t weight,
                      const Bias_t bias) {
#pragma HLS INLINE off
    forward_1d_stream_impl(output_stream, input_stream, weight, bias);
  }

#endif

private:
  static void forward_1d_impl(dtype *output, const dtype *input,
                              const Weight_t weight, const Bias_t bias) {
  OUTER_LOOP:
    for (int i = 0; i < OUT_FEATURES; i++) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
      dtype acc = dtype(0);
    INNER_LOOP:
      for (int j = 0; j < IN_FEATURES; j++) {
        acc += input[j] * weight[i][j];
      }
      output[i] = acc + bias[i];
    }
  }

#ifdef __VITIS_HLS__
  static void forward_1d_stream_impl(hls::stream<dtype> &output_stream,
                                     hls::stream<dtype> &input_stream,
                                     const Weight_t weight, const Bias_t bias) {
    dtype input_buffer[IN_FEATURES];

  READ_INPUT:
    for (int j = 0; j < IN_FEATURES; j++) {
      input_buffer[j] = input_stream.read();
    }

  OUTER_LOOP:
    for (int i = 0; i < OUT_FEATURES; i++) {
      dtype acc = dtype(0);
    INNER_LOOP:
      for (int j = 0; j < IN_FEATURES; j++) {
        acc += input_buffer[j] * weight[i][j];
      }
      output_stream.write(acc + bias[i]);
    }
  }
#endif
};

// ============================================================================
// Optimized version (OPT_ENABLED)
// ===========================================================================
template <typename DType, const int IN_FEATURES, const int OUT_FEATURES,
          typename Config>
class Linear<DType, IN_FEATURES, OUT_FEATURES, Config, OPT_ENABLED>
    : public BaseModule<DType> {
public:
  using dtype = DType;
  static constexpr int in_features = IN_FEATURES;
  static constexpr int out_features = OUT_FEATURES;
  static constexpr OptLevel opt_level = OPT_ENABLED;

  static constexpr int unroll_factor = Config::_unroll_factor;
  static constexpr int partition_factor = Config::_partition_factor;
  static constexpr int pipeline_ii = Config::_pipeline_ii;

  using Weight_t = dtype[OUT_FEATURES][IN_FEATURES];
  using Bias_t = dtype[OUT_FEATURES];

  Linear() = default;
  ~Linear() = default;

  std::string module_name() const override { return "Linear"; }
  std::string module_type() const override { return "Linear_OPT_ENABLED"; }
  json to_json() const override {
    json j;
    j["module_type"] = module_type();
    j["module_name"] = module_name();
    j["in_features"] = in_features;
    j["out_features"] = out_features;
    j["opt_level"] = "OPT_ENABLED";

    json hls_config;
    hls_config["unroll_factor"] = unroll_factor;
    hls_config["partition_factor"] = partition_factor;
    hls_config["pipeline_ii"] = pipeline_ii;
    j["hls_config"] = hls_config;

    return j;
  }

  static void forward(dtype output[OUT_FEATURES],
                      const dtype input[IN_FEATURES], const Weight_t weight,
                      const Bias_t bias) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    forward_1d_impl(output, input, weight, bias);
  }

  static void forward(dtype output[][OUT_FEATURES],
                      const dtype input[][IN_FEATURES], const int batch_size,
                      const Weight_t weight, const Bias_t bias) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_1d_impl(output[b], input[b], weight, bias);
    }
  }

  static void forward(dtype *output, const dtype *input, const int batch_size,
                      const Weight_t weight, const Bias_t bias) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_1d_impl(&output[b * OUT_FEATURES], &input[b * IN_FEATURES],
                      weight, bias);
    }
  }

#ifdef __VITIS_HLS__
  static void forward(hls::stream<dtype> &output_stream,
                      hls::stream<dtype> &input_stream, const Weight_t weight,
                      const Bias_t bias) {
#pragma HLS INLINE off
    forward_1d_stream_impl(output_stream, input_stream, weight, bias);
  }
#endif

private:
  static void forward_1d_impl(dtype *output, const dtype *input,
                              const Weight_t weight, const Bias_t bias) {
#ifdef __VITIS_HLS__
#pragma HLS ARRAY_PARTITION variable = input cyclic factor = partition_factor
#pragma HLS ARRAY_PARTITION variable = weight cyclic factor =                  \
    partition_factor dim = 2
#pragma HLS ARRAY_PARTITION variable = bias cyclic factor = partition_factor
#endif

  OUTER_LOOP:
    for (int i = 0; i < OUT_FEATURES; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#endif
      dtype acc = dtype(0);
    INNER_LOOP:
      for (int j = 0; j < IN_FEATURES; j++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = unroll_factor
#endif
        acc += input[j] * weight[i][j];
      }
      output[i] = acc + bias[i];
    }
  }

#ifdef __VITIS_HLS__
  static void forward_1d_stream_impl(hls::stream<dtype> &output_stream,
                                     hls::stream<dtype> &input_stream,
                                     const Weight_t weight, const Bias_t bias) {
    dtype input_buffer[IN_FEATURES];
#pragma HLS ARRAY_PARTITION variable = input_buffer cyclic factor =            \
    partition_factor
#pragma HLS ARRAY_PARTITION variable = weight cyclic factor =                  \
    partition_factor dim = 2
#pragma HLS ARRAY_PARTITION variable = bias cyclic factor = partition_factor

  READ_INPUT:
    for (int j = 0; j < IN_FEATURES; j++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#endif
      input_buffer[j] = input_stream.read();
    }

  OUTER_LOOP:
    for (int i = 0; i < OUT_FEATURES; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#endif
      dtype acc = dtype(0);
    INNER_LOOP:
      for (int j = 0; j < IN_FEATURES; j++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = unroll_factor
#endif
        acc += input_buffer[j] * weight[i][j];
      }
      output_stream.write(acc + bias[i]);
    }
  }
#endif
};
} // namespace hls_nn

namespace hls_tb {
template <const int IN_FEATURES, const int OUT_FEATURES>
class LinearTestCase : public BaseTestCase {
public:
  LinearTestCase(unsigned seed = 42) : BaseTestCase(seed) {}

  void generate_random_input(float input[IN_FEATURES]) {
    generate_random_array(input, IN_FEATURES);
  }

  void generate_random_weight(float weight[OUT_FEATURES][IN_FEATURES]) {
    for (int i = 0; i < OUT_FEATURES; i++) {
      generate_random_array(weight[i], IN_FEATURES, 0.1f);
    }
  }

  void generate_random_bias(float bias[OUT_FEATURES]) {
    generate_random_array(bias, OUT_FEATURES, 0.01f);
  }

  void generate_ones_input(float input[IN_FEATURES]) {
    generate_ones_array(input, IN_FEATURES);
  }

  void generate_identity_weight(float weight[OUT_FEATURES][IN_FEATURES]) {
    for (int i = 0; i < OUT_FEATURES; i++) {
      for (int j = 0; j < IN_FEATURES; j++) {
        weight[i][j] =
            (i == j && i < std::min(IN_FEATURES, OUT_FEATURES)) ? 1.0f : 0.0f;
      }
    }
  }

  void generate_zero_bias(float bias[OUT_FEATURES]) {
    generate_zeros_array(bias, OUT_FEATURES);
  }
};

template <typename DType, const int IN_FEATURES, const int OUT_FEATURES,
          typename Config = void, OptLevel OPT_LEVEL = OPT_NONE>
class LinearTestbench : public BaseTestbench<DType, Config, OPT_LEVEL> {
public:
  LinearTestbench() : BaseTestbench<DType, Config, OPT_LEVEL>("Linear") {}

  void test_random_case(const std::string &case_name) override {
    _generator.generate_random_input(_input_ref);
    _generator.generate_random_weight(_weight_ref);
    _generator.generate_random_bias(_bias_ref);

    _convert_input(_input_ref, _input_dut);
    _convert_weight(_weight_ref, _weight_dut);
    _convert_bias(_bias_ref, _bias_dut);

    LinearDUT::forward(_output_dut, _input_dut, _weight_dut, _bias_dut);
    LinearRef::forward(_output_ref, _input_ref, _weight_ref, _bias_ref);

    float output_dut_float[OUT_FEATURES];
    _convert_output(_output_dut, output_dut_float);

    auto result =
        ResultComparator::compare(output_dut_float, _output_ref, OUT_FEATURES);
    ResultComparator::print_result(result, case_name);
  }

  void test_identity_case() override {
    _generator.generate_random_input(_input_ref);
    _generator.generate_identity_weight(_weight_ref);
    _generator.generate_zero_bias(_bias_ref);

    _convert_input(_input_ref, _input_dut);
    _convert_weight(_weight_ref, _weight_dut);
    _convert_bias(_bias_ref, _bias_dut);

    LinearDUT::forward(_output_dut, _input_dut, _weight_dut, _bias_dut);
    LinearRef::forward(_output_ref, _input_ref, _weight_ref, _bias_ref);

    float output_dut_float[OUT_FEATURES];
    _convert_output(_output_dut, output_dut_float);

    auto result =
        ResultComparator::compare(output_dut_float, _output_ref, OUT_FEATURES);
    ResultComparator::print_result(result, "Identity Matrix Test");
  }

  void test_ones_case() override {
    _generator.generate_ones_input(_input_ref);
    _generator.generate_random_weight(_weight_ref);
    _generator.generate_random_bias(_bias_ref);

    _convert_input(_input_ref, _input_dut);
    _convert_weight(_weight_ref, _weight_dut);
    _convert_bias(_bias_ref, _bias_dut);

    LinearDUT::forward(_output_dut, _input_dut, _weight_dut, _bias_dut);
    LinearRef::forward(_output_ref, _input_ref, _weight_ref, _bias_ref);

    float output_dut_float[OUT_FEATURES];
    _convert_output(_output_dut, output_dut_float);

    auto result =
        ResultComparator::compare(output_dut_float, _output_ref, OUT_FEATURES);
    ResultComparator::print_result(result, "All Ones Input Test");
  }

protected:
  void print_test_header() override {
    std::cout << "\n########################################" << std::endl;
    std::cout << "Testing Linear Layer" << std::endl;
    std::cout << "IN_FEATURES: " << IN_FEATURES << std::endl;
    std::cout << "OUT_FEATURES: " << OUT_FEATURES << std::endl;
    std::cout << "OPT_STATUS: " << OPT_LEVEL << std::endl;
    std::cout << "DType: " << typeid(DType).name() << std::endl;
    std::cout << "########################################" << std::endl;
  }

private:
  using LinearDUT =
      hls_nn::Linear<DType, IN_FEATURES, OUT_FEATURES, Config, OPT_LEVEL>;
  using LinearRef =
      hls_nn::Linear<float, IN_FEATURES, OUT_FEATURES, void, OPT_NONE>;
  LinearTestCase<IN_FEATURES, OUT_FEATURES> _generator;

  DType _input_dut[IN_FEATURES];
  DType _weight_dut[OUT_FEATURES][IN_FEATURES];
  DType _bias_dut[OUT_FEATURES];
  DType _output_dut[OUT_FEATURES];

  float _input_ref[IN_FEATURES];
  float _weight_ref[OUT_FEATURES][IN_FEATURES];
  float _bias_ref[OUT_FEATURES];
  float _output_ref[OUT_FEATURES];

  void _convert_input(const float src[IN_FEATURES], DType dst[IN_FEATURES]) {
    this->convert_array(src, dst, IN_FEATURES);
  }

  void _convert_weight(const float src[OUT_FEATURES][IN_FEATURES],
                       DType dst[OUT_FEATURES][IN_FEATURES]) {
    for (int i = 0; i < OUT_FEATURES; i++) {
      this->convert_array(src[i], dst[i], IN_FEATURES);
    }
  }

  void _convert_bias(const float src[OUT_FEATURES], DType dst[OUT_FEATURES]) {
    this->convert_array(src, dst, OUT_FEATURES);
  }

  void _convert_output(const DType src[OUT_FEATURES], float dst[OUT_FEATURES]) {
    this->convert_array(src, dst, OUT_FEATURES);
  }
};
} // namespace hls_tb
