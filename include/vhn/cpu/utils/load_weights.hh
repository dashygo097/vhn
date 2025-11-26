#pragma once

#include <fstream>
#include <iostream>

namespace vhn::cpu {
template <typename T>
bool load_weights(const std::string filename, T *data, size_t size) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to open " << filename << std::endl;
    return false;
  }
  file.read(reinterpret_cast<char *>(data), size * sizeof(T));
  file.close();
  return true;
}
} // namespace vhn::cpu
