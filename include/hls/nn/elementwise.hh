#pragma once

#ifdef __VITIS_HLS__
#endif

namespace hls_nn {
template <typename DType, typename KernelType, int N> class Elementwise {
public:
  using dtype = DType;
  using kernel = KernelType;
  static constexpr int n = N;

  Elementwise() = default;
  ~Elementwise() = default;

  static void forward(dtype output[n], const dtype input[n]) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = 1
#endif
    for (int i = 0; i < n; i++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = 8
#endif
      output[i] = kernel(input[i]);
    }
  }

  static void forward(dtype output[n], const dtype input1[n],
                      const dtype input2[n]) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = 1
#endif
    for (int i = 0; i < n; i++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = 8
#endif
      output[i] = kernel(input1[i], input2[i]);
    }
  }

private:
};
} // namespace hls_nn
