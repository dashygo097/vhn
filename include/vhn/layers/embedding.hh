#pragma once

#include "../opt_level.hh"
#include "../tb/tb.hh"
#include <algorithm>
#include <iostream>
#include <random>

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace vhn {

template <typename DType, const int VOCAB_SIZE, const int EMBED_DIM,
          typename Config = void, OptLevel OPT_LEVEL = OPT_NONE>
class Embedding;

// ============================================================================
// Non-optimized version (OPT_NONE)
// ============================================================================
template <typename DType, const int VOCAB_SIZE, const int EMBED_DIM>
class Embedding<DType, VOCAB_SIZE, EMBED_DIM, void, OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int vocab_size = VOCAB_SIZE;
  static constexpr int embed_dim = EMBED_DIM;
  static constexpr OptLevel opt_level = OPT_NONE;

  using Weight_t = dtype[VOCAB_SIZE][EMBED_DIM];

  Embedding() = default;
  ~Embedding() = default;

  static void forward(dtype output[EMBED_DIM], const int input,
                      const Weight_t weight) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    forward_single(output, input, weight);
  }

  static void forward(dtype output[][EMBED_DIM], const int input[],
                      const int batch_size, const Weight_t weight) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
  BATCH_LOOP:
    for (int i = 0; i < batch_size; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_single(output[i], input[i], weight);
    }
  }

  static void forward(dtype *output, const int *input, const int length,
                      const Weight_t weight) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    forward_impl_1d(output, input, length, weight);
  }

#ifdef __VITIS_HLS__
  static void forward(hls::stream<dtype> &output, hls::stream<int> &input,
                      const int length, const Weight_t weight) {
#pragma HLS INLINE off
    forward_1d_stream_impl(output, input, length, weight);
  }
#endif

private:
  static void forward_single(dtype output[EMBED_DIM], const int input,
                             const Weight_t weight) {
  OUT_LOOP:
    for (int i = 0; i < EMBED_DIM; i++) {
      output[i] = weight[input][i];
    }
  }

  static void forward_impl_1d(dtype *output, const int *input, const int length,
                              const Weight_t weight) {
  LENGTH_LOOP:
    for (int i = 0; i < length; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
    OUT_LOOP:
      for (int j = 0; j < EMBED_DIM; j++) {
        output[i * EMBED_DIM + j] = weight[input[i]][j];
      }
    }
  }

#ifdef __VITIS_HLS__
  static void forward_1d_stream_impl(hls::stream<dtype> &output,
                                     hls::stream<int> &input, const int length,
                                     const Weight_t weight) {
  LENGTH_LOOP:
    for (int i = 0; i < length; i++) {
      int idx = input.read();
    OUT_LOOP:
      for (int j = 0; j < EMBED_DIM; j++) {
        output.write(weight[idx][j]);
      }
    }
  }
#endif
};

// ============================================================================
// Optimized version (OPT_ENABLED)
// ============================================================================
template <typename DType, const int VOCAB_SIZE, const int EMBED_DIM,
          typename Config>
class Embedding<DType, VOCAB_SIZE, EMBED_DIM, Config, OPT_ENABLED> {
public:
  using dtype = DType;
  static constexpr int vocab_size = VOCAB_SIZE;
  static constexpr int embed_dim = EMBED_DIM;
  static constexpr OptLevel opt_level = OPT_ENABLED;

  static constexpr int unroll_factor = Config::_unroll_factor;
  static constexpr int partition_factor = Config::_partition_factor;
  static constexpr int pipeline_ii = Config::_pipeline_ii;

  using Weight_t = dtype[VOCAB_SIZE][EMBED_DIM];

  Embedding() = default;
  ~Embedding() = default;

  static void forward(dtype output[EMBED_DIM], const int input,
                      const Weight_t weight) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    forward_single(output, input, weight);
  }

  static void forward(dtype output[][EMBED_DIM], const int input[],
                      const int batch_size, const Weight_t weight) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#endif
  BATCH_LOOP:
    for (int i = 0; i < batch_size; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      forward_single(output[i], input[i], weight);
    }
  }

  static void forward(dtype *output, const int *input, const int length,
                      const Weight_t weight) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
    forward_impl_1d(output, input, length, weight);
  }

#ifdef __VITIS_HLS__
  static void forward(hls::stream<dtype> &output, hls::stream<int> &input,
                      const int length, const Weight_t weight) {
#pragma HLS INLINE off
    forward_1d_stream_impl(output, input, length, weight);
  }
#endif

private:
  static void forward_single(dtype output[EMBED_DIM], const int input,
                             const Weight_t weight) {
#ifdef __VITIS_HLS__
#pragma HLS ARRAY_PARTITION variable = output cyclic factor = partition_factor
#pragma HLS ARRAY_PARTITION variable = weight cyclic factor =                  \
    partition_factor dim = 2
#endif

  OUT_LOOP:
    for (int i = 0; i < EMBED_DIM; i++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#pragma HLS UNROLL factor = unroll_factor
#endif
      output[i] = weight[input][i];
    }
  }

  static void forward_impl_1d(dtype *output, const int *input, const int length,
                              const Weight_t weight) {
#ifdef __VITIS_HLS__
#pragma HLS ARRAY_PARTITION variable = weight cyclic factor =                  \
    partition_factor dim = 2
#endif

  LENGTH_LOOP:
    for (int i = 0; i < length; i++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      int idx = input[i];
    OUT_LOOP:
      for (int j = 0; j < EMBED_DIM; j++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#pragma HLS UNROLL factor = unroll_factor
#endif
        output[i * EMBED_DIM + j] = weight[idx][j];
      }
    }
  }

#ifdef __VITIS_HLS__
  static void forward_1d_stream_impl(hls::stream<dtype> &output,
                                     hls::stream<int> &input, const int length,
                                     const Weight_t weight) {
#pragma HLS ARRAY_PARTITION variable = weight cyclic factor =                  \
    partition_factor dim = 2

  LENGTH_LOOP:
    for (int i = 0; i < length; i++) {
      int idx = input.read();
    OUT_LOOP:
      for (int j = 0; j < EMBED_DIM; j++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = pipeline_ii
#pragma HLS UNROLL factor = unroll_factor
#endif
        output.write(weight[idx][j]);
      }
    }
  }
#endif
};
} // namespace vhn

namespace vhn::tb {
template <const int VOCAB_SIZE, const int EMBED_DIM>
class EmbeddingTestCase : public BaseTestCase {
public:
  EmbeddingTestCase(unsigned seed = 42) : BaseTestCase(seed) {}

  void generate_random_indices(int *indices, int length) {
    std::uniform_int_distribution<int> idx_dist(0, VOCAB_SIZE - 1);
    for (int i = 0; i < length; i++) {
      indices[i] = idx_dist(_seed);
    }
  }

  void generate_sequential_indices(int *indices, int length) {
    for (int i = 0; i < length; i++) {
      indices[i] = i % VOCAB_SIZE;
    }
  }

  void generate_single_index(int &index) {
    std::uniform_int_distribution<int> idx_dist(0, VOCAB_SIZE - 1);
    index = idx_dist(_seed);
  }

  void generate_random_weight(float weight[VOCAB_SIZE][EMBED_DIM]) {
    for (int i = 0; i < VOCAB_SIZE; i++) {
      generate_random_array(weight[i], EMBED_DIM, 0.1f);
    }
  }

  void generate_identity_weight(float weight[VOCAB_SIZE][EMBED_DIM]) {
    for (int i = 0; i < VOCAB_SIZE; i++) {
      for (int j = 0; j < EMBED_DIM; j++) {
        weight[i][j] =
            (i == j && i < std::min(VOCAB_SIZE, EMBED_DIM)) ? 1.0f : 0.0f;
      }
    }
  }
};

template <typename DType, const int VOCAB_SIZE, const int EMBED_DIM,
          typename Config = void, OptLevel OPT_LEVEL = OPT_NONE>
class EmbeddingTestbench : public BaseTestbench<DType, Config, OPT_LEVEL> {
public:
  static constexpr int TEST_LENGTH = 16;

  EmbeddingTestbench() : BaseTestbench<DType, Config, OPT_LEVEL>("Embedding") {}

  void test_random_case(const std::string &case_name) override {
    _generator.generate_random_indices(_indices_ref, TEST_LENGTH);
    _generator.generate_random_weight(_weight_ref);

    // Convert indices (int to int, no conversion needed)
    for (int i = 0; i < TEST_LENGTH; i++) {
      _indices_dut[i] = _indices_ref[i];
    }

    // Convert weight
    _convert_weight(_weight_ref, _weight_dut);

    // Run forward
    EmbeddingDUT::forward(_output_dut, _indices_dut, TEST_LENGTH, _weight_dut);
    EmbeddingRef::forward(_output_ref, _indices_ref, TEST_LENGTH, _weight_ref);

    // Convert output
    float output_dut_float[TEST_LENGTH * EMBED_DIM];
    _convert_output(_output_dut, output_dut_float, TEST_LENGTH * EMBED_DIM);

    auto result = ResultComparator::compare(output_dut_float, &_output_ref[0],
                                            TEST_LENGTH * EMBED_DIM);
    ResultComparator::print_result(result, case_name);
  }

  void test_identity_case() override {
    _generator.generate_sequential_indices(_indices_ref, TEST_LENGTH);
    _generator.generate_identity_weight(_weight_ref);

    for (int i = 0; i < TEST_LENGTH; i++) {
      _indices_dut[i] = _indices_ref[i];
    }
    _convert_weight(_weight_ref, _weight_dut);

    EmbeddingDUT::forward(_output_dut, _indices_dut, TEST_LENGTH, _weight_dut);
    EmbeddingRef::forward(_output_ref, _indices_ref, TEST_LENGTH, _weight_ref);

    float output_dut_float[TEST_LENGTH * EMBED_DIM];
    _convert_output(_output_dut, output_dut_float, TEST_LENGTH * EMBED_DIM);

    auto result = ResultComparator::compare(output_dut_float, &_output_ref[0],
                                            TEST_LENGTH * EMBED_DIM);
    ResultComparator::print_result(result, "Identity Weight Test");
  }

  void test_ones_case() override {
    // For embedding, "ones case" means looking up the same index multiple times
    for (int i = 0; i < TEST_LENGTH; i++) {
      _indices_ref[i] = 0; // All lookup index 0
      _indices_dut[i] = 0;
    }
    _generator.generate_random_weight(_weight_ref);
    _convert_weight(_weight_ref, _weight_dut);

    EmbeddingDUT::forward(_output_dut, _indices_dut, TEST_LENGTH, _weight_dut);
    EmbeddingRef::forward(_output_ref, _indices_ref, TEST_LENGTH, _weight_ref);

    float output_dut_float[TEST_LENGTH * EMBED_DIM];
    _convert_output(_output_dut, output_dut_float, TEST_LENGTH * EMBED_DIM);

    auto result = ResultComparator::compare(output_dut_float, &_output_ref[0],
                                            TEST_LENGTH * EMBED_DIM);
    ResultComparator::print_result(result, "Same Index Lookup Test");
  }

  void test_single_lookup() {
    int idx_ref, idx_dut;
    _generator.generate_single_index(idx_ref);
    idx_dut = idx_ref;

    _generator.generate_random_weight(_weight_ref);
    _convert_weight(_weight_ref, _weight_dut);

    DType output_dut_single[EMBED_DIM];
    float output_ref_single[EMBED_DIM];

    EmbeddingDUT::forward(output_dut_single, idx_dut, _weight_dut);
    EmbeddingRef::forward(output_ref_single, idx_ref, _weight_ref);

    float output_dut_float[EMBED_DIM];
    _convert_output(output_dut_single, output_dut_float, EMBED_DIM);

    auto result = ResultComparator::compare(output_dut_float, output_ref_single,
                                            EMBED_DIM);
    ResultComparator::print_result(result, "Single Index Lookup Test");
  }

  void run_all_tests() override {
    this->print_test_header();
    test_random_case("Random Test Case 1");
    test_random_case("Random Test Case 2");
    test_random_case("Random Test Case 3");
    test_identity_case();
    test_ones_case();
    test_single_lookup();
  }

protected:
  void print_test_header() override {
    std::cout << "\n########################################" << std::endl;
    std::cout << "Testing Embedding Layer" << std::endl;
    std::cout << "VOCAB_SIZE: " << VOCAB_SIZE << std::endl;
    std::cout << "EMBED_DIM: " << EMBED_DIM << std::endl;
    std::cout << "TEST_LENGTH: " << TEST_LENGTH << std::endl;
    std::cout << "OPT_STATUS: " << OPT_LEVEL << std::endl;
    std::cout << "DType: " << typeid(DType).name() << std::endl;
    std::cout << "########################################" << std::endl;
  }

private:
  using EmbeddingDUT =
      vhn::Embedding<DType, VOCAB_SIZE, EMBED_DIM, Config, OPT_LEVEL>;
  using EmbeddingRef =
      vhn::Embedding<float, VOCAB_SIZE, EMBED_DIM, void, OPT_NONE>;
  EmbeddingTestCase<VOCAB_SIZE, EMBED_DIM> _generator;

  int _indices_dut[TEST_LENGTH];
  DType _weight_dut[VOCAB_SIZE][EMBED_DIM];
  DType _output_dut[TEST_LENGTH * EMBED_DIM];

  int _indices_ref[TEST_LENGTH];
  float _weight_ref[VOCAB_SIZE][EMBED_DIM];
  float _output_ref[TEST_LENGTH * EMBED_DIM];

  void _convert_weight(const float src[VOCAB_SIZE][EMBED_DIM],
                       DType dst[VOCAB_SIZE][EMBED_DIM]) {
    for (int i = 0; i < VOCAB_SIZE; i++) {
      for (int j = 0; j < EMBED_DIM; j++) {
        dst[i][j] = static_cast<DType>(src[i][j]);
      }
    }
  }

  void _convert_output(const DType *src, float *dst, int size) {
    for (int i = 0; i < size; i++) {
      dst[i] = static_cast<float>(src[i]);
    }
  }
};

} // namespace vhn::tb
