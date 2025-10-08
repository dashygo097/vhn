#pragma once

#include "../../../tb/tb.hh"
#include "./linear.hh"
#include <algorithm>
#include <iostream>
#include <random>

namespace hls_tb {
template <const int IN_FEATURES, const int OUT_FEATURES> class LinearTestCase {
public:
  LinearTestCase(unsigned seed = 42) : _seed(seed), _dist(-1.0f, 1.0f) {}

  void generate_random_input(float input[IN_FEATURES]) {
    for (int i = 0; i < IN_FEATURES; i++) {
      input[i] = _dist(_seed);
    }
  }

  void generate_random_weight(float weight[OUT_FEATURES][IN_FEATURES]) {
    for (int i = 0; i < OUT_FEATURES; i++) {
      for (int j = 0; j < IN_FEATURES; j++) {
        weight[i][j] = _dist(_seed) * 0.1f;
      }
    }
  }

  void generate_random_bias(float bias[OUT_FEATURES]) {
    for (int i = 0; i < OUT_FEATURES; i++) {
      bias[i] = _dist(_seed) * 0.01f;
    }
  }

  void generate_ones_input(float input[IN_FEATURES]) {
    std::fill_n(input, IN_FEATURES, 1.0f);
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
    std::fill_n(bias, OUT_FEATURES, 0.0f);
  }

private:
  std::mt19937 _seed;
  std::uniform_real_distribution<float> _dist;
};

template <typename DType, const int IN_FEATURES, const int OUT_FEATURES,
          OptLevel OPT_LEVEL>
class LinearTestbench {
public:
  LinearTestbench() = default;
  ~LinearTestbench() = default;

  void test_random_case(const std::string &case_name) {
    // Generate random test data in float32
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

  void test_identity_case() {
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

  void test_ones_case() {
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

  void run_all_tests() {
    std::cout << "\n########################################" << std::endl;
    std::cout << "Testing Linear Layer" << std::endl;
    std::cout << "IN_FEATURES: " << IN_FEATURES << std::endl;
    std::cout << "OUT_FEATURES: " << OUT_FEATURES << std::endl;
    std::cout << "OPT_LEVEL: " << static_cast<int>(OPT_LEVEL) << std::endl;
    std::cout << "DType: " << typeid(DType).name() << std::endl;
    std::cout << "########################################" << std::endl;

    test_random_case("Random Test Case 1");
    test_random_case("Random Test Case 2");
    test_random_case("Random Test Case 3");
    test_identity_case();
    test_ones_case();
  }

private:
  using LinearDUT = hls_nn::Linear<DType, IN_FEATURES, OUT_FEATURES, OPT_LEVEL>;
  using LinearRef = hls_nn::Linear<float, IN_FEATURES, OUT_FEATURES>;
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
    for (int i = 0; i < IN_FEATURES; i++) {
      dst[i] = static_cast<DType>(src[i]);
    }
  }

  void _convert_weight(const float src[OUT_FEATURES][IN_FEATURES],
                       DType dst[OUT_FEATURES][IN_FEATURES]) {
    for (int i = 0; i < OUT_FEATURES; i++) {
      for (int j = 0; j < IN_FEATURES; j++) {
        dst[i][j] = static_cast<DType>(src[i][j]);
      }
    }
  }

  void _convert_bias(const float src[OUT_FEATURES], DType dst[OUT_FEATURES]) {
    for (int i = 0; i < OUT_FEATURES; i++) {
      dst[i] = static_cast<DType>(src[i]);
    }
  }

  void _convert_output(const DType src[OUT_FEATURES], float dst[OUT_FEATURES]) {
    for (int i = 0; i < OUT_FEATURES; i++) {
      dst[i] = static_cast<float>(src[i]);
    }
  }
};
} // namespace hls_tb
