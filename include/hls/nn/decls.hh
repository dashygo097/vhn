#pragma once

#include "../tb/tb.hh"
#include "./elementwise.hh"
#include <algorithm>
#include <random>

#ifdef __VITIS_HLS__
#include <hls_math.h>
#endif

#define ELEMENTWISE_DEF(ELEMENTWISE_NAME)                                      \
  template <typename DType, int N, typename Config = void,                     \
            OptLevel OPT_LEVEL = OPT_NONE>                                     \
  using ELEMENTWISE_NAME =                                                     \
      Elementwise<DType, ELEMENTWISE_NAME##Impl<DType, N>, N, Config,          \
                  OPT_LEVEL>;

#define ELEMENTWISE_TB_DEF(ELEMENTWISE_NAME)                                   \
  template <const int N> class ELEMENTWISE_NAME##TestCase {                    \
  public:                                                                      \
    ELEMENTWISE_NAME##TestCase(unsigned seed = 42)                             \
        : _seed(seed), _dist(-1.0f, 1.0f) {}                                   \
                                                                               \
    void generate_random_input(float input[N]) {                               \
      for (int i = 0; i < N; i++) {                                            \
        input[i] = _dist(_seed);                                               \
      }                                                                        \
    }                                                                          \
                                                                               \
    void generate_ones_input(float input[N]) { std::fill_n(input, N, 1.0f); }  \
                                                                               \
  private:                                                                     \
    std::mt19937 _seed;                                                        \
    std::uniform_real_distribution<float> _dist;                               \
  };                                                                           \
                                                                               \
  template <typename DType, const int N, typename Config = void,               \
            OptLevel OPT_LEVEL = OPT_NONE>                                     \
  class ELEMENTWISE_NAME##Testbench {                                          \
  public:                                                                      \
    ELEMENTWISE_NAME##Testbench() = default;                                   \
    ~ELEMENTWISE_NAME##Testbench() = default;                                  \
    void test_random_case(const std::string &case_ELEMENTWISE_NAME) {          \
      _generator.generate_random_input(_input_ref);                            \
                                                                               \
      _convert_input(_input_ref, _input_dut);                                  \
                                                                               \
      ELEMENTWISE_NAME##DUT::forward(_output_dut, _input_dut);                 \
      ELEMENTWISE_NAME##Ref::forward(_output_ref, _input_ref);                 \
                                                                               \
      float output_dut_float[N];                                               \
      _convert_output(_output_dut, output_dut_float);                          \
                                                                               \
      auto result =                                                            \
          ResultComparator::compare(output_dut_float, _output_ref, N);         \
      ResultComparator::print_result(result, case_ELEMENTWISE_NAME);           \
    }                                                                          \
    void test_identity_case() {                                                \
      _generator.generate_random_input(_input_ref);                            \
                                                                               \
      _convert_input(_input_ref, _input_dut);                                  \
                                                                               \
      ELEMENTWISE_NAME##DUT::forward(_output_dut, _input_dut);                 \
      ELEMENTWISE_NAME##Ref::forward(_output_ref, _input_ref);                 \
                                                                               \
      float output_dut_float[N];                                               \
      _convert_output(_output_dut, output_dut_float);                          \
                                                                               \
      auto result =                                                            \
          ResultComparator::compare(output_dut_float, _output_ref, N);         \
      ResultComparator::print_result(result, "Identity Matrix Test");          \
    }                                                                          \
                                                                               \
    void test_ones_case() {                                                    \
      _generator.generate_ones_input(_input_ref);                              \
                                                                               \
      _convert_input(_input_ref, _input_dut);                                  \
                                                                               \
      ELEMENTWISE_NAME##DUT::forward(_output_dut, _input_dut);                 \
      ELEMENTWISE_NAME##Ref::forward(_output_ref, _input_ref);                 \
                                                                               \
      float output_dut_float[N];                                               \
      _convert_output(_output_dut, output_dut_float);                          \
                                                                               \
      auto result =                                                            \
          ResultComparator::compare(output_dut_float, _output_ref, N);         \
      ResultComparator::print_result(result, "All Ones Input Test");           \
    }                                                                          \
                                                                               \
    void run_all_tests() {                                                     \
      std::cout << "\n########################################" << std::endl;  \
      std::cout << "Testing" #ELEMENTWISE_NAME "Layer" << std::endl;           \
      std::cout << "N: " << N << std::endl;                                    \
      std::cout << "OPT_STATUS: " << OPT_LEVEL << std::endl                    \
                << "DType: " << typeid(DType).name() << std::endl;             \
      std::cout << "########################################" << std::endl;    \
                                                                               \
      test_random_case("Random Test Case 1");                                  \
      test_random_case("Random Test Case 2");                                  \
      test_random_case("Random Test Case 3");                                  \
      test_identity_case();                                                    \
      test_ones_case();                                                        \
    }                                                                          \
                                                                               \
  private:                                                                     \
    using ELEMENTWISE_NAME##DUT =                                              \
        hls_nn::ELEMENTWISE_NAME<DType, N, Config, OPT_LEVEL>;                 \
    using ELEMENTWISE_NAME##Ref =                                              \
        hls_nn::ELEMENTWISE_NAME<float, N, void, OPT_NONE>;                    \
    ELEMENTWISE_NAME##TestCase<N> _generator;                                  \
                                                                               \
    DType _input_dut[N];                                                       \
    DType _output_dut[N];                                                      \
                                                                               \
    float _input_ref[N];                                                       \
    float _output_ref[N];                                                      \
                                                                               \
    void _convert_input(const float src[N], DType dst[N]) {                    \
      for (int i = 0; i < N; i++) {                                            \
        dst[i] = static_cast<DType>(src[i]);                                   \
      }                                                                        \
    }                                                                          \
                                                                               \
    void _convert_output(const DType src[N], float dst[N]) {                   \
      for (int i = 0; i < N; i++) {                                            \
        dst[i] = static_cast<float>(src[i]);                                   \
      }                                                                        \
    }                                                                          \
  };
