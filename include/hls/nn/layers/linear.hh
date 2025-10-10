#pragma once

#include "../../opt_level.hh"
#include "../../tb/tb.hh"
#include <algorithm>
#include <iostream>
#include <random>

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace hls_nn {
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

  static void forward(dtype output[OUT_FEATURES],
                      const dtype input[IN_FEATURES],
                      const dtype weight[OUT_FEATURES][IN_FEATURES],
                      const dtype bias[OUT_FEATURES]);

  static void forward(dtype output[][OUT_FEATURES],
                      const dtype input[][IN_FEATURES],
                      const dtype weight[OUT_FEATURES][IN_FEATURES],
                      const dtype bias[OUT_FEATURES]);

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

  static void forward(dtype output[OUT_FEATURES],
                      const dtype input[IN_FEATURES],
                      const dtype weight[OUT_FEATURES][IN_FEATURES],
                      const dtype bias[OUT_FEATURES]) {
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

  static void forward(dtype output[][OUT_FEATURES],
                      const dtype input[][IN_FEATURES],
                      const dtype weight[OUT_FEATURES][IN_FEATURES],
                      const dtype bias[OUT_FEATURES], const int batch_size) {
#ifdef __VITIS_HLS__
#pragma HLS DATAFLOW
#endif

  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward(*reinterpret_cast<dtype(*)[OUT_FEATURES]>(&output[b]),
              *reinterpret_cast<const dtype(*)[IN_FEATURES]>(&input[b]), weight,
              bias);
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

  static void forward(dtype output[OUT_FEATURES],
                      const dtype input[IN_FEATURES],
                      const dtype weight[OUT_FEATURES][IN_FEATURES],
                      const dtype bias[OUT_FEATURES]) {
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

  static void forward(dtype output[][OUT_FEATURES],
                      const dtype input[][IN_FEATURES],
                      const dtype weight[OUT_FEATURES][IN_FEATURES],
                      const dtype bias[OUT_FEATURES], const int batch_size) {
#ifdef __VITIS_HLS__
#pragma HLS DATAFLOW
#endif

  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward(*reinterpret_cast<dtype(*)[OUT_FEATURES]>(&output[b]),
              *reinterpret_cast<const dtype(*)[IN_FEATURES]>(&input[b]), weight,
              bias);
    }
  }

private:
};

} // namespace hls_nn

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
          typename Config = void, OptLevel OPT_LEVEL = OPT_NONE>
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
    std::cout << "OPT_STATUS: " << OPT_LEVEL << std::endl
              << "DType: " << typeid(DType).name() << std::endl;
    std::cout << "########################################" << std::endl;

    test_random_case("Random Test Case 1");
    test_random_case("Random Test Case 2");
    test_random_case("Random Test Case 3");
    test_identity_case();
    test_ones_case();
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
