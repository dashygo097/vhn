#pragma once

#include "../../opt_level.hh"
#include "../../tb/tb.hh"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace hls_nn {

template <typename DType, int N, typename Config = void,
          OptLevel OPT_LEVEL = OPT_NONE>
class Softmax;

// ============================================================================
// Non-optimized version (OPT_NONE)
// ============================================================================
template <typename DType, int N> class Softmax<DType, N, void, OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int n = N;
  static constexpr OptLevel opt_level = OPT_NONE;

  Softmax() = default;
  ~Softmax() = default;

  static void forward(dtype output[N], const dtype input[N]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    forward_1d_impl(output, input);
  }

  static void forward(dtype output[][N], const dtype input[][N],
                      const int batch_size) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_1d_impl(output[b], input[b]);
    }
  }

  static void forward(dtype *output, const dtype *input, const int batch_size) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_1d_impl(&output[b * N], &input[b * N]);
    }
  }

private:
  static void forward_1d_impl(dtype *output, const dtype *input) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif

    dtype max_val = input[0];
  FIND_MAX:
    for (int i = 1; i < n; i++) {
      max_val = (input[i] > max_val) ? input[i] : max_val;
    }

    dtype sum = dtype(0.0f);
    dtype exp_val[n];
  CALC_EXP:
    for (int i = 0; i < n; i++) {
#ifdef __VITIS_HLS__
      exp_val[i] = hls::exp(input[i] - max_val);
#else
      exp_val[i] = std::exp(input[i] - max_val);
#endif
      sum += exp_val[i];
    }

    // Normalize by sum
    dtype inv_sum = dtype(1.0) / sum;
  NORMALIZE:
    for (int i = 0; i < n; i++) {
      output[i] = exp_val[i] * inv_sum;
    }
  }
};

// ============================================================================
// Optimized version (OPT_ENABLED)
// ============================================================================
template <typename DType, int N, typename Config>
class Softmax<DType, N, Config, OPT_ENABLED> {
public:
  using dtype = DType;
  static constexpr int n = N;
  static constexpr OptLevel opt_level = OPT_ENABLED;

  static constexpr int unroll_factor = Config::_unroll_factor;
  static constexpr int partition_factor = Config::_partition_factor;
  static constexpr int pipeline_ii = Config::_pipeline_ii;

  Softmax() = default;
  ~Softmax() = default;

  static void forward(dtype output[N], const dtype input[N]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    forward_1d_impl(output, input);
  }

  static void forward(dtype output[][N], const dtype input[][N],
                      const int batch_size) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_1d_impl(output[b], input[b]);
    }
  }

  static void forward(dtype *output, const dtype *input, const int batch_size) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_1d_impl(&output[b * N], &input[b * N]);
    }
  }

private:
  static void forward_1d_impl(dtype *output, const dtype *input) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = input cyclic factor = partition_factor
#pragma HLS ARRAY_PARTITION variable = output cyclic factor = partition_factor
#endif

    dtype max_val = input[0];
  FIND_MAX:
    for (int i = 1; i < n; i++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = unroll_factor
#endif
      max_val = (input[i] > max_val) ? input[i] : max_val;
    }

    dtype sum = dtype(0.0f);
    dtype exp_val[n];
  CALC_EXP:
    for (int i = 0; i < n; i++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = unroll_factor
#endif
#ifdef __VITIS_HLS__
      exp_val[i] = hls::exp(input[i] - max_val);
#else
      exp_val[i] = std::exp(input[i] - max_val);
#endif
      sum += exp_val[i];
    }

    dtype inv_sum = dtype(1.0) / sum;
  NORMALIZE:
    for (int i = 0; i < n; i++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = unroll_factor
#endif
      output[i] = exp_val[i] * inv_sum;
    }
  }
};

} // namespace hls_nn

namespace hls_tb {
template <const int N> class SoftmaxTestCase {
public:
  SoftmaxTestCase(unsigned seed = 42) : _seed(seed), _dist(-1.0f, 1.0f) {}

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
class SoftmaxTestbench {
public:
  SoftmaxTestbench() = default;
  ~SoftmaxTestbench() = default;

  void test_random_case(const std::string &case_name) {
    // Generate random test data in float32
    _generator.generate_random_input(_input_ref);

    _convert_input(_input_ref, _input_dut);

    SoftmaxDUT::forward(_output_dut, _input_dut);
    SoftmaxRef::forward(_output_ref, _input_ref);

    float output_dut_float[N];
    _convert_output(_output_dut, output_dut_float);

    auto result = ResultComparator::compare(output_dut_float, _output_ref, N);
    ResultComparator::print_result(result, case_name);
  }

  void test_identity_case() {
    _generator.generate_random_input(_input_ref);

    _convert_input(_input_ref, _input_dut);

    SoftmaxDUT::forward(_output_dut, _input_dut);
    SoftmaxRef::forward(_output_ref, _input_ref);

    float output_dut_float[N];
    _convert_output(_output_dut, output_dut_float);

    auto result = ResultComparator::compare(output_dut_float, _output_ref, N);
    ResultComparator::print_result(result, "Identity Matrix Test");
  }

  void test_ones_case() {
    _generator.generate_ones_input(_input_ref);

    _convert_input(_input_ref, _input_dut);

    SoftmaxDUT::forward(_output_dut, _input_dut);
    SoftmaxRef::forward(_output_ref, _input_ref);

    float output_dut_float[N];
    _convert_output(_output_dut, output_dut_float);

    auto result = ResultComparator::compare(output_dut_float, _output_ref, N);
    ResultComparator::print_result(result, "All Ones Input Test");
  }

  void run_all_tests() {
    std::cout << "\n########################################" << std::endl;
    std::cout << "Testing Softmax Layer" << std::endl;
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
  using SoftmaxDUT = hls_nn::Softmax<DType, N, Config, OPT_LEVEL>;
  using SoftmaxRef = hls_nn::Softmax<float, N, void, OPT_NONE>;
  SoftmaxTestCase<N> _generator;

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
