#pragma once

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "./utils/load_weights.hh"

namespace vhn::cpu {
template <typename T> class Tensor {
public:
  using dtype = T;
  using iterator = typename std::vector<T>::iterator;
  using const_iterator = typename std::vector<T>::const_iterator;

  explicit Tensor(const std::vector<size_t> &shape)
      : _shape(shape), _ndim(shape.size()) {
    if (_shape.empty()) {
      throw std::invalid_argument("Tensor shape cannot be empty");
    }
    _strides = compute_strides(_shape);
    _data.resize(compute_total_size(_shape));
  }

  Tensor(Tensor &&other) noexcept = default;
  Tensor &operator=(Tensor &&other) noexcept = default;

  Tensor(const Tensor &) = delete;
  Tensor &operator=(const Tensor &) = delete;

  [[nodiscard]] Tensor clone() const {
    Tensor temp(_shape);
    temp._data = _data;
    return temp;
  }

  T &operator[](size_t index) {
    if (index >= _data.size()) {
      throw std::out_of_range("Tensor 1D index " + std::to_string(index) +
                              " out of range [0, " +
                              std::to_string(_data.size()) + ")");
    }
    return _data[index];
  }

  const T &operator[](size_t index) const {
    if (index >= _data.size()) {
      throw std::out_of_range("Tensor 1D index " + std::to_string(index) +
                              " out of range [0, " +
                              std::to_string(_data.size()) + ")");
    }
    return _data[index];
  }

  T &at(const std::vector<size_t> &indices) {
    validate_indices(indices);
    return _data[flat_index(indices)];
  }

  const T &at(const std::vector<size_t> &indices) const {
    validate_indices(indices);
    return _data[flat_index(indices)];
  }

  [[nodiscard]] const std::vector<size_t> &shape() const noexcept {
    return _shape;
  }
  [[nodiscard]] const std::vector<size_t> &strides() const noexcept {
    return _strides;
  }
  [[nodiscard]] const std::vector<T> &data() const noexcept { return _data; }
  [[nodiscard]] std::vector<T> &data() noexcept { return _data; }
  [[nodiscard]] size_t size() const noexcept { return _data.size(); }
  [[nodiscard]] size_t ndim() const noexcept { return _ndim; }
  [[nodiscard]] size_t dim(size_t axis) const {
    if (axis >= _ndim) {
      throw std::out_of_range("Axis " + std::to_string(axis) +
                              " out of range for " + std::to_string(_ndim) +
                              "D tensor");
    }
    return _shape[axis];
  }

  iterator begin() noexcept { return _data.begin(); }
  iterator end() noexcept { return _data.end(); }
  const_iterator begin() const noexcept { return _data.begin(); }
  const_iterator end() const noexcept { return _data.end(); }
  const_iterator cbegin() const noexcept { return _data.cbegin(); }
  const_iterator cend() const noexcept { return _data.cend(); }

  void fill(T value) { std::fill(_data.begin(), _data.end(), value); }
  void zero() { std::fill(_data.begin(), _data.end(), T(0)); }

  bool is_contiguous() const noexcept {
    if (_ndim == 0)
      return true;

    size_t expected_stride = 1;
    for (int i = _ndim - 1; i >= 0; --i) {
      if (_strides[i] != expected_stride)
        return false;
      expected_stride *= _shape[i];
    }
    return true;
  }

  void load_weights(const std::vector<T> &weights) {
    if (weights.size() != _data.size()) {
      throw std::invalid_argument("Weights size mismatch");
    }
    for (size_t i = 0; i < weights.size(); ++i) {
      _data[i] = weights[i];
    }
  }

private:
  std::vector<size_t> _shape;
  std::vector<size_t> _strides;
  std::vector<T> _data;
  size_t _ndim;

  [[nodiscard]] static size_t
  compute_total_size(const std::vector<size_t> &shape) {
    return std::accumulate(shape.begin(), shape.end(), size_t(1),
                           std::multiplies<size_t>());
  }

  [[nodiscard]] static std::vector<size_t>
  compute_strides(const std::vector<size_t> &shape) {
    std::vector<size_t> strides(shape.size());
    if (shape.empty())
      return strides;

    strides.back() = 1;
    for (int i = shape.size() - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
  }

  [[nodiscard]] size_t flat_index(const std::vector<size_t> &indices) const {
    size_t index = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
      index += indices[i] * _strides[i];
    }
    return index;
  }

  void validate_indices(const std::vector<size_t> &indices) const {
    if (indices.size() != _ndim) {
      throw std::invalid_argument(
          "Index dimension " + std::to_string(indices.size()) +
          " mismatch with tensor dimension " + std::to_string(_ndim));
    }
    for (size_t i = 0; i < indices.size(); ++i) {
      if (indices[i] >= _shape[i]) {
        throw std::out_of_range(
            "Index [" + std::to_string(i) + "]=" + std::to_string(indices[i]) +
            " out of range [0, " + std::to_string(_shape[i]) + ")");
      }
    }
  }
};

} // namespace vhn::cpu
