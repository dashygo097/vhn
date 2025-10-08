#pragma once
#include "../../opt_level.hh"

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace hls_nn {
template <typename DType, const int IN_CHANNELS, const int OUT_CHANNELS,
          const int KERNEL_SIZE, const int PADDING, const int N,
          OptLevel OPT_LEVEL = OPT_NONE>
class Conv2d {
public:
  using dtype = DType;
  static constexpr int in_channels = IN_CHANNELS;
  static constexpr int out_channels = OUT_CHANNELS;
  static constexpr int kernel_size = KERNEL_SIZE;
  static constexpr int padding = PADDING;
  static constexpr int n = N;
  static constexpr OptLevel opt_level = OPT_LEVEL;

  Conv2d() = default;
  ~Conv2d() = default;

  static void forward(
      dtype output[out_channels][n - kernel_size + 1][n - kernel_size + 1],
      const dtype input[in_channels][n][n],
      const dtype weight[out_channels][in_channels][kernel_size][kernel_size],
      const dtype bias[out_channels]);

private:
};

template <typename DType, const int IN_CHANNELS, const int OUT_CHANNELS,
          const int KERNEL_SIZE, const int PADDING, const int N>
class Conv2d<DType, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, PADDING, N,
             OPT_NONE> {
public:
  using dtype = DType;
  static constexpr int in_channels = IN_CHANNELS;
  static constexpr int out_channels = OUT_CHANNELS;
  static constexpr int kernel_size = KERNEL_SIZE;
  static constexpr int padding = PADDING;
  static constexpr int n = N;
  static constexpr OptLevel opt_level = OPT_NONE;

  Conv2d() = default;
  ~Conv2d() = default;

  static void forward(
      dtype output[out_channels][n - kernel_size + 1 + 2 * padding]
                  [n - kernel_size + 1 + 2 * padding],
      const dtype input[in_channels][n][n],
      const dtype weight[out_channels][in_channels][kernel_size][kernel_size],
      const dtype bias[out_channels]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#endif
  OUT_CHANNEL_LOOP:
    for (int oc = 0; oc < out_channels; oc++) {
    OUT_POS_Y_LOOP:
      for (int pos_y = 0; pos_y < n - kernel_size + 1 + 2 * padding; pos_y++) {
      OUT_POS_X_LOOP:
        for (int pos_x = 0; pos_x < n - kernel_size + 1 + 2 * padding;
             pos_x++) {
          dtype acc = dtype(0.0f);
        IN_CHANNEL_LOOP:
          for (int ic = 0; ic < in_channels; ic++) {
          KERNEL_Y_LOOP:
            for (int ky = 0; ky < kernel_size; ky++) {
            KERNEL_X_LOOP:
              for (int kx = 0; kx < kernel_size; kx++) {
                int in_pos_y = pos_y + ky - padding;
                int in_pos_x = pos_x + kx - padding;
                if (in_pos_y >= 0 && in_pos_y < n && in_pos_x >= 0 &&
                    in_pos_x < n) {
                  acc += input[ic][in_pos_y][in_pos_x] * weight[oc][ic][ky][kx];
                }
              }
            }
          }
          output[oc][pos_y][pos_x] = acc + bias[oc];
        }
      }
    }
  }

private:
};

template <typename DType, const int IN_CHANNELS, const int OUT_CHANNELS,
          const int KERNEL_SIZE, const int PADDING, const int N>
class Conv2d<DType, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, PADDING, N,
             OPT_LATENCY> {
public:
  using dtype = DType;
  static constexpr int in_channels = IN_CHANNELS;
  static constexpr int out_channels = OUT_CHANNELS;
  static constexpr int kernel_size = KERNEL_SIZE;
  static constexpr int padding = PADDING;
  static constexpr int n = N;
  static constexpr OptLevel opt_level = OPT_LATENCY;

  Conv2d() = default;
  ~Conv2d() = default;

  static void forward(
      dtype output[out_channels][n - kernel_size + 1 + 2 * padding]
                  [n - kernel_size + 1 + 2 * padding],
      const dtype input[in_channels][n][n],
      const dtype weight[out_channels][in_channels][kernel_size][kernel_size],
      const dtype bias[out_channels]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = input complete dim = 1
#pragma HLS ARRAY_PARTITION variable = weight cyclic factor = 16 dim = 4
#endif
  OUT_CHANNEL_LOOP:
    for (int oc = 0; oc < out_channels; oc++) {
    OUT_POS_Y_LOOP:
      for (int pos_y = 0; pos_y < n - kernel_size + 1 + 2 * padding; pos_y++) {
      OUT_POS_X_LOOP:
        for (int pos_x = 0; pos_x < n - kernel_size + 1 + 2 * padding;
             pos_x++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = 1 rewind
#endif
          dtype acc = dtype(0.0f);
        IN_CHANNEL_LOOP:
          for (int ic = 0; ic < in_channels; ic++) {
          KERNEL_Y_LOOP:
            for (int ky = 0; ky < kernel_size; ky++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL
#endif
            KERNEL_X_LOOP:
#ifdef __VITIS_HLS__
#pragma HLS UNROLL
#endif
              for (int kx = 0; kx < kernel_size; kx++) {
                int in_pos_y = pos_y + ky - padding;
                int in_pos_x = pos_x + kx - padding;
                if (in_pos_y >= 0 && in_pos_y < n && in_pos_x >= 0 &&
                    in_pos_x < n) {
                  acc += input[ic][in_pos_y][in_pos_x] * weight[oc][ic][ky][kx];
                }
              }
            }
          }
          output[oc][pos_y][pos_x] = acc + bias[oc];
        }
      }
    }
  }

private:
};

template <typename DType, const int IN_CHANNELS, const int OUT_CHANNELS,
          const int KERNEL_SIZE, const int PADDING, const int N>
class Conv2d<DType, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, PADDING, N,
             OPT_THROUGHPUT> {
public:
  using dtype = DType;
  static constexpr int in_channels = IN_CHANNELS;
  static constexpr int out_channels = OUT_CHANNELS;
  static constexpr int kernel_size = KERNEL_SIZE;
  static constexpr int padding = PADDING;
  static constexpr int n = N;
  static constexpr OptLevel opt_level = OPT_THROUGHPUT;

  Conv2d() = default;
  ~Conv2d() = default;

  static void forward(
      dtype output[out_channels][n - kernel_size + 1 + 2 * padding]
                  [n - kernel_size + 1 + 2 * padding],
      const dtype input[in_channels][n][n],
      const dtype weight[out_channels][in_channels][kernel_size][kernel_size],
      const dtype bias[out_channels]) {
#ifdef __VITIS_HLS__
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = input cyclic factor = 8 dim = 1
#pragma HLS ARRAY_PARTITION variable = weight cyclic factor = 8 dim = 4
#endif
  OUT_CHANNEL_LOOP:
    for (int oc = 0; oc < out_channels; oc++) {
    OUT_POS_Y_LOOP:
      for (int pos_y = 0; pos_y < n - kernel_size + 1 + 2 * padding; pos_y++) {
      OUT_POS_X_LOOP:
        for (int pos_x = 0; pos_x < n - kernel_size + 1 + 2 * padding;
             pos_x++) {
#ifdef __VITIS_HLS__
#pragma HLS PIPELINE II = 1
#endif
          dtype acc = dtype(0.0f);
        IN_CHANNEL_LOOP:
          for (int ic = 0; ic < in_channels; ic++) {
          KERNEL_Y_LOOP:
            for (int ky = 0; ky < kernel_size; ky++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = kernel_size
#endif
            KERNEL_X_LOOP:
              for (int kx = 0; kx < kernel_size; kx++) {
#ifdef __VITIS_HLS__
#pragma HLS UNROLL factor = kernel_size
#endif
                int in_pos_y = pos_y + ky - padding;
                int in_pos_x = pos_x + kx - padding;
                if (in_pos_y >= 0 && in_pos_y < n && in_pos_x >= 0 &&
                    in_pos_x < n) {
                  acc += input[ic][in_pos_y][in_pos_x] * weight[oc][ic][ky][kx];
                }
              }
            }
          }
          output[oc][pos_y][pos_x] = acc + bias[oc];
        }
      }
    }
  }

private:
};

template <typename DType, const int IN_CHANNELS, const int OUT_CHANNELS,
          const int KERNEL_SIZE, const int PADDING, const int N,
          const OptLevel OPT_LEVEL = OPT_NONE>
class Conv2dBatched {
public:
  using dtype = DType;
  static constexpr int in_channels = IN_CHANNELS;
  static constexpr int out_channels = OUT_CHANNELS;
  static constexpr int kernel_size = KERNEL_SIZE;
  static constexpr int padding = PADDING;
  static constexpr int n = N;
  static constexpr OptLevel opt_level = OPT_LEVEL;

  Conv2dBatched() = default;
  ~Conv2dBatched() = default;

  static void
  forward(dtype output[][out_channels][n - kernel_size + 1 + 2 * padding]
                      [n - kernel_size + 1 + 2 * padding],
          const dtype input[][in_channels][n],
          const dtype weight[out_channels][in_channels][kernel_size],
          const dtype bias[out_channels], int batch_size) {
#ifdef __VITIS_HLS__
#pragma HLS DATAFLOW
#endif

  BATCH_LOOP:
    for (int b = 0; b < batch_size; b++) {
#ifdef __VITIS_HLS__
#pragma HLS LOOP_FLATTEN off
#endif
      Conv2d<dtype, in_channels, out_channels, kernel_size, padding, n,
             opt_level>::forward(output[b], input[b], weight, bias);
    }
  }

private:
};

} // namespace hls_nn
