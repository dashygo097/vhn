#pragma once

#include "../../tb/tb.hh"
#include "../elementwise.hh"
#include <algorithm>
#include <iostream>
#include <random>

#ifdef __VITIS_HLS__
#include <hls_math.h>
#endif

namespace hls_nn {
template <typename DType, int N> class SigmoidImpl {
  using dtype = DType;
  static constexpr int n = N;

  static void kernel(dtype output, const dtype input) {
#ifdef __VITIS_HLS__
    output = dtype(1.0f) / (dtype(1.0f) + hls::exp(-input));
#else
    output = dtype(1.0f) / (dtype(1.0f) + std::exp(-input));
#endif
  }
};

// FIXME: Weak against Config with different OPT_LEVEL
template <typename DType, int N, typename Config = void,
          OptLevel OPT_LEVEL = OPT_NONE>
using Sigmoid = Elementwise<DType, SigmoidImpl<DType, N>, N, Config, OPT_LEVEL>;

} // namespace hls_nn

namespace hls_tb {
template <const int N> class SigmoidTestCase {
public:
  SigmoidTestCase(unsigned seed = 42) : _seed(seed), _dist(-1.0f, 1.0f) {}

  void generate_random_input(float input[N]) {
    for (int i = 0; i < N; i++) {
      input[i] = _dist(_seed);
    }
  }

  void generate_ones_input(float input[N]) { std::fill_n(input, N, 1.0f); }

private:
  std::mt19937 _seed;
  std::uniform_real_distribution<float> _dist;
};

// FIXME: Weak against Config with different OPT_LEVEL
template <typename DType, const int N, typename Config = void,
          OptLevel OPT_LEVEL = OPT_NONE>
class SigmoidTestbench {
public:
  SigmoidTestbench() = default;
  ~SigmoidTestbench() = default;

  void test_random_case(const std::string &case_name) {
    // Generate random test data in float32
    _generator.generate_random_input(_input_ref);

    _convert_input(_input_ref, _input_dut);

    SigmoidDUT::forward(_output_dut, _input_dut);
    SigmoidRef::forward(_output_ref, _input_ref);

    float output_dut_float[N];
    _convert_output(_output_dut, output_dut_float);

    auto result = ResultComparator::compare(output_dut_float, _output_ref, N);
    ResultComparator::print_result(result, case_name);
  }

  void test_identity_case() {
    _generator.generate_random_input(_input_ref);

    _convert_input(_input_ref, _input_dut);

    SigmoidDUT::forward(_output_dut, _input_dut);
    SigmoidRef::forward(_output_ref, _input_ref);

    float output_dut_float[N];
    _convert_output(_output_dut, output_dut_float);

    auto result = ResultComparator::compare(output_dut_float, _output_ref, N);
    ResultComparator::print_result(result, "Identity Matrix Test");
  }

  void test_ones_case() {
    _generator.generate_ones_input(_input_ref);

    _convert_input(_input_ref, _input_dut);

    SigmoidDUT::forward(_output_dut, _input_dut);
    SigmoidRef::forward(_output_ref, _input_ref);

    float output_dut_float[N];
    _convert_output(_output_dut, output_dut_float);

    auto result = ResultComparator::compare(output_dut_float, _output_ref, N);
    ResultComparator::print_result(result, "All Ones Input Test");
  }

  void run_all_tests() {
    std::cout << "\n########################################" << std::endl;
    std::cout << "Testing Sigmoid Layer" << std::endl;
    std::cout << "N: " << N << std::endl;
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
  using SigmoidDUT = hls_nn::Sigmoid<DType, N, Config, OPT_LEVEL>;
  using SigmoidRef = hls_nn::Sigmoid<float, N, void, OPT_NONE>;
  SigmoidTestCase<N> _generator;

  DType _input_dut[N];
  DType _output_dut[N];

  float _input_ref[N];
  float _output_ref[N];

  void _convert_input(const float src[N], DType dst[N]) {
    for (int i = 0; i < N; i++) {
      dst[i] = static_cast<DType>(src[i]);
    }
  }

  void _convert_output(const DType src[N], float dst[N]) {
    for (int i = 0; i < N; i++) {
      dst[i] = static_cast<float>(src[i]);
    }
  }
};
} // namespace hls_tb
