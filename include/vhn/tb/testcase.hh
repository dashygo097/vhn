#pragma once

#include <algorithm>
#include <random>

class BaseTestCase {
public:
  BaseTestCase(unsigned seed = 42) : _seed(seed), _dist(-1.0f, 1.0f) {}

  float random_float(float scale = 1.0f) { return _dist(_seed) * scale; }

  void generate_random_array(float *arr, int size, float scale = 1.0f) {
    for (int i = 0; i < size; i++) {
      arr[i] = _dist(_seed) * scale;
    }
  }

  void generate_ones_array(float *arr, int size) {
    std::fill_n(arr, size, 1.0f);
  }

  void generate_zeros_array(float *arr, int size) {
    std::fill_n(arr, size, 0.0f);
  }

  void generate_identity_matrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        matrix[i * cols + j] =
            (i == j && i < std::min(rows, cols)) ? 1.0f : 0.0f;
      }
    }
  }

protected:
  std::mt19937 _seed;
  std::uniform_real_distribution<float> _dist;
};
