#pragma once

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

  using Wqkv_t = dtype[3 * D_MODEL][D_MODEL];
  using bqkv_t = dtype[3 * D_MODEL];
  using Wo_t = dtype[D_MODEL][D_MODEL];
  using bo_t = dtype[D_MODEL];

  MulHeadAttn() = default;
  MulHeadAttn(const Wqkv_t &Wqkv, const bqkv_t &bqkv, const Wo_t &Wo,
              const bo_t &bo)
      : _Wqkv(&Wqkv), _bqkv(&bqkv), _Wo(&Wo), _bo(&bo) {}
  ~MulHeadAttn() = default;

  void load_weights(const Wqkv_t &Wqkv, const bqkv_t &bqkv, const Wo_t &Wo,
                    const bo_t &bo) {
    _Wqkv = &Wqkv;
    _bqkv = &bqkv;
    _Wo = &Wo;
    _bo = &bo;
  }

  [[nodiscard]] const Wqkv_t &Wqkv() const { return *_Wqkv; }
  [[nodiscard]] const bqkv_t &bqkv() const { return *_bqkv; }
  [[nodiscard]] const Wo_t &Wo() const { return *_Wo; }
  [[nodiscard]] const bo_t &bo() const { return *_bo; }
  [[nodiscard]] bool isLoaded() const {

    return _Wqkv != nullptr && _bqkv != nullptr && _Wo != nullptr &&
           _bo != nullptr;
  }

  void qkv(dtype *Q, dtype *K, dtype *V, const dtype *input,
           const int actual_len);
  void forward(dtype *output, const dtype *input, const dtype *mask,
               const int actual_len);

private:
  const Wqkv_t *_Wqkv{nullptr};
  const bqkv_t *_bqkv{nullptr};
  const Wo_t *_Wo{nullptr};
  const bo_t *_bo{nullptr};
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

  using Wqkv_t = dtype[3 * D_MODEL][D_MODEL];
  using bqkv_t = dtype[3 * D_MODEL];
  using Wo_t = dtype[D_MODEL][D_MODEL];
  using bo_t = dtype[D_MODEL];

  MulHeadAttn() = default;
  MulHeadAttn(const Wqkv_t &Wqkv, const bqkv_t &bqkv, const Wo_t &Wo,
              const bo_t &bo)
      : _Wqkv(&Wqkv), _bqkv(&bqkv), _Wo(&Wo), _bo(&bo) {}
  ~MulHeadAttn() = default;

  void load_weights(const Wqkv_t &Wqkv, const bqkv_t &bqkv, const Wo_t &Wo,
                    const bo_t &bo) {
    _Wqkv = &Wqkv;
    _bqkv = &bqkv;
    _Wo = &Wo;
    _bo = &bo;
  }

  [[nodiscard]] const Wqkv_t &Wqkv() const { return *_Wqkv; }
  [[nodiscard]] const bqkv_t &bqkv() const { return *_bqkv; }
  [[nodiscard]] const Wo_t &Wo() const { return *_Wo; }
  [[nodiscard]] const bo_t &bo() const { return *_bo; }
  [[nodiscard]] bool isLoaded() const {

    return _Wqkv != nullptr && _bqkv != nullptr && _Wo != nullptr &&
           _bo != nullptr;
  }

  void qkv(dtype *Q, dtype *K, dtype *V, const dtype *input,
           const int actual_len) {}
  void forward(dtype *output, const dtype *input, const dtype *mask,
               const int actual_len) {}

private:
  const Wqkv_t *_Wqkv{nullptr};
  const bqkv_t *_bqkv{nullptr};
  const Wo_t *_Wo{nullptr};
  const bo_t *_bo{nullptr};
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

  using Wqkv_t = dtype[3 * D_MODEL][D_MODEL];
  using bqkv_t = dtype[3 * D_MODEL];
  using Wo_t = dtype[D_MODEL][D_MODEL];
  using bo_t = dtype[D_MODEL];

  MulHeadAttn() = default;
  MulHeadAttn(const Wqkv_t &Wqkv, const bqkv_t &bqkv, const Wo_t &Wo,
              const bo_t &bo)
      : _Wqkv(&Wqkv), _bqkv(&bqkv), _Wo(&Wo), _bo(&bo) {}
  ~MulHeadAttn() = default;

  void load_weights(const Wqkv_t &Wqkv, const bqkv_t &bqkv, const Wo_t &Wo,
                    const bo_t &bo) {
    _Wqkv = &Wqkv;
    _bqkv = &bqkv;
    _Wo = &Wo;
    _bo = &bo;
  }

  [[nodiscard]] const Wqkv_t &Wqkv() const { return *_Wqkv; }
  [[nodiscard]] const bqkv_t &bqkv() const { return *_bqkv; }
  [[nodiscard]] const Wo_t &Wo() const { return *_Wo; }
  [[nodiscard]] const bo_t &bo() const { return *_bo; }
  [[nodiscard]] bool isLoaded() const {

    return _Wqkv != nullptr && _bqkv != nullptr && _Wo != nullptr &&
           _bo != nullptr;
  }

  void qkv(dtype *Q, dtype *K, dtype *V, const dtype *input,
           const int actual_len) {}
  void forward(dtype *output, const dtype *input, const dtype *mask,
               const int actual_len) {}

private:
  const Wqkv_t *_Wqkv{nullptr};
  const bqkv_t *_bqkv{nullptr};
  const Wo_t *_Wo{nullptr};
  const bo_t *_bo{nullptr};
};

} // namespace hls_nn
