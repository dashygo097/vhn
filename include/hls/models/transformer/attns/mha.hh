#pragma once

#include "../../../nn/nn.hh"
#include "../../../opt_level.hh"

#ifdef __VITIS_HLS__
#include <hls_stream.h>
#endif

namespace hls_nn {
template <typename DType, const int D_MODEL, const int NUM_HEADS,
          const int MAX_SEQ_LEN, typename Config, OptLevel OPT_LEVEL = OPT_NONE>
class MulHeadAttn {
public:
  static_assert(D_MODEL % NUM_HEADS == 0,
                "D_MODEL must be divisible by NUM_HEADS");
  using dtype = DType;
  static constexpr int d_model = D_MODEL;
  static constexpr int num_heads = NUM_HEADS;
  static constexpr int head_dim = D_MODEL / NUM_HEADS;
  static constexpr int max_seq_len = MAX_SEQ_LEN;
  static constexpr OptLevel opt_level = OPT_LEVEL;

  using Wqkv_t = dtype[3 * D_MODEL * D_MODEL];
  using bqkv_t = dtype[3 * D_MODEL];
  using Wo_t = dtype[D_MODEL * D_MODEL];
  using bo_t = dtype[D_MODEL];

  MulHeadAttn() = default;
  ~MulHeadAttn() = default;

private:
};

template <typename DType, const int D_MODEL, const int NUM_HEADS,
          const int MAX_SEQ_LEN>
class MulHeadAttn<DType, D_MODEL, NUM_HEADS, MAX_SEQ_LEN, void, OPT_NONE> {
public:
  static_assert(D_MODEL % NUM_HEADS == 0,
                "D_MODEL must be divisible by NUM_HEADS");
  using dtype = DType;
  static constexpr int d_model = D_MODEL;
  static constexpr int num_heads = NUM_HEADS;
  static constexpr int head_dim = D_MODEL / NUM_HEADS;
  static constexpr int max_seq_len = MAX_SEQ_LEN;
  static constexpr OptLevel opt_level = OPT_NONE;

  using Wqkv_t = dtype[3 * D_MODEL * D_MODEL];
  using bqkv_t = dtype[3 * D_MODEL];
  using Wo_t = dtype[D_MODEL * D_MODEL];
  using bo_t = dtype[D_MODEL];

  MulHeadAttn() = default;
  ~MulHeadAttn() = default;

private:
};

template <typename DType, const int D_MODEL, const int NUM_HEADS,
          const int MAX_SEQ_LEN, typename Config>
class MulHeadAttn<DType, D_MODEL, NUM_HEADS, MAX_SEQ_LEN, Config, OPT_ENABLED> {
public:
  static_assert(D_MODEL % NUM_HEADS == 0,
                "D_MODEL must be divisible by NUM_HEADS");
  using dtype = DType;
  static constexpr int d_model = D_MODEL;
  static constexpr int num_heads = NUM_HEADS;
  static constexpr int head_dim = D_MODEL / NUM_HEADS;
  static constexpr int max_seq_len = MAX_SEQ_LEN;
  static constexpr OptLevel opt_level = OPT_NONE;

  using Wqkv_t = dtype[3 * D_MODEL * D_MODEL];
  using bqkv_t = dtype[3 * D_MODEL];
  using Wo_t = dtype[D_MODEL * D_MODEL];
  using bo_t = dtype[D_MODEL];

  MulHeadAttn() = default;
  ~MulHeadAttn() = default;

private:
};

} // namespace hls_nn
