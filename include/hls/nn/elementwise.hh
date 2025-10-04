#pragma once

#include "../opt_level.hh"

#ifdef __VITIS_HLS__
#endif

namespace hls_nn {
template <typename DType, typename KernelType, int N,
          OptLevel OPT_LEVEL = OPT_NONE>
class Elementwise {
public:
  using dtype = DType;
  using kernel = KernelType;
  static constexpr int n = N;

  Elementwise() = default;
  ~Elementwise() = default;

  static void forward(dtype output[n], const dtype input[n]);
  static void forward(dtype output[n], const dtype input1[n],
                      const dtype input2[n]);

private:
};

template <typename DType, typename KernelType, int N>
class Elementwise<DType, KernelType, N, OPT_NONE> {
public:
  using dtype = DType;
  using kernel = KernelType;
  static constexpr int n = N;

  Elementwise() = default;
  ~Elementwise() = default;

  static void forward(dtype output[n], const dtype input[n]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
  ELEMENTWISE_LOOP:
    for (int i = 0; i < n; i++) {
      output[i] = kernel::compute(input[i]);
    }
  }

  static void forward(dtype output[n], const dtype input1[n],
                      const dtype input2[n]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
  ELEMEMTWISE_LOOP_2:
    for (int i = 0; i < n; i++) {
      output[i] = kernel::compute(input1[i], input2[i]);
    }
  }

private:
};

template <typename DType, typename KernelType, int N>
class Elementwise<DType, KernelType, N, OPT_LATENCY> {
public:
  using dtype = DType;
  using kernel = KernelType;
  static constexpr int n = N;

  Elementwise() = default;
  ~Elementwise() = default;

  static void forward(dtype output[n], const dtype input[n]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS PIPELINE II = 1 rewind
#pragma HLS ARRAY_PARTITION variable = input complete
#pragma HLS ARRAY_PARTITION variable = output complete
#endif
  ELEMENTWISE_LOOP:
    for (int i = 0; i < n; i++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL
#endif
      output[i] = kernel::compute(input[i]);
    }
  }

  static void forward(dtype output[n], const dtype input1[n],
                      const dtype input2[n]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS PIPELINE II = 1 rewind
#pragma HLS ARRAY_PARTITION variable = input1 complete
#pragma HLS ARRAY_PARTITION variable = input2 complete
#pragma HLS ARRAY_PARTITION variable = output complete
#endif
  ELEMEMTWISE_LOOP_2:
    for (int i = 0; i < n; i++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL
#endif
      output[i] = kernel::compute(input1[i], input2[i]);
    }
  }

private:
};

template <typename DType, typename KernelType, int N>
class Elementwise<DType, KernelType, N, OPT_THROUGHPUT> {
public:
  using dtype = DType;
  using kernel = KernelType;
  static constexpr int n = N;

  Elementwise() = default;
  ~Elementwise() = default;

  static void forward(dtype output[n], const dtype input[n]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS PIPELINE II = 1
#pragma HLS ARRAY_PARTITION variable = input cyclic factor = 4
#pragma HLS ARRAY_PARTITION variable = output cyclic factor = 4
#endif
  ELEMENTWISE_LOOP:
    for (int i = 0; i < n; i++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = 4
#endif
      output[i] = kernel::compute(input[i]);
    }
  }

  static void forward(dtype output[n], const dtype input1[n],
                      const dtype input2[n]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS PIPELINE II = 1
#pragma HLS ARRAY_PARTITION variable = input1 cyclic factor = 4
#pragma HLS ARRAY_PARTITION variable = input2 cyclic factor = 4
#pragma HLS ARRAY_PARTITION variable = output cyclic factor = 4
#endif
  ELEMEMTWISE_LOOP_2:
    for (int i = 0; i < n; i++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = 4

#endif
      output[i] = kernel::compute(input1[i], input2[i]);
    }
  }

private:
};

} // namespace hls_nn
