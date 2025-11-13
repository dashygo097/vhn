#pragma once
#include <cmath>
#include <iomanip>
#include <iostream>

class ResultComparator {
public:
  struct ComparisonResult {
    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    float avg_abs_error = 0.0f;
    float rmse = 0.0f;
    bool passed = false;
    int total_elements = 0;
  };

  static ComparisonResult compare(const float *result, const float *reference,
                                  int size) {
    ComparisonResult res;
    res.total_elements = size;

    float sum_abs_error = 0.0f;
    float sum_sq_error = 0.0f;

    for (int i = 0; i < size; i++) {
      float abs_error = std::abs(result[i] - reference[i]);
      float rel_error = (std::abs(reference[i]) > 1e-8f)
                            ? abs_error / std::abs(reference[i])
                            : 0.0f;

      res.max_abs_error = std::max(res.max_abs_error, abs_error);
      res.max_rel_error = std::max(res.max_rel_error, rel_error);
      sum_abs_error += abs_error;
      sum_sq_error += abs_error * abs_error;
    }

    res.avg_abs_error = sum_abs_error / size;
    res.rmse = std::sqrt(sum_sq_error / size);

    return res;
  }

  static void print_result(const ComparisonResult &res,
                           const std::string &test_name) {
    std::cout << "\n=== " << test_name << " ===" << std::endl;
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "Max Absolute Error: " << res.max_abs_error << std::endl;
    std::cout << "Max Relative Error: " << res.max_rel_error << std::endl;
    std::cout << "Avg Absolute Error: " << res.avg_abs_error << std::endl;
    std::cout << "RMSE:               " << res.rmse << std::endl;
  }
};
