#pragma once

#include "../opt_level.hh"
#include <iostream>

template <typename DType, typename Config = void, OptLevel OPT_LEVEL = OPT_NONE>
class BaseTestbench {
public:
  BaseTestbench(const std::string &layer_name) : _layer_name(layer_name) {}
  virtual ~BaseTestbench() = default;

  virtual void test_random_case(const std::string &case_name) = 0;
  virtual void test_identity_case() = 0;
  virtual void test_ones_case() = 0;

  virtual void run_all_tests() {
    print_test_header();
    test_random_case("Random Test Case 1");
    test_random_case("Random Test Case 2");
    test_random_case("Random Test Case 3");
    test_identity_case();
    test_ones_case();
  }

protected:
  std::string _layer_name;

  virtual void print_test_header() {
    std::cout << "\n########################################" << std::endl;
    std::cout << "Testing " << _layer_name << " Layer" << std::endl;
    std::cout << "OPT_STATUS: " << OPT_LEVEL << std::endl;
    std::cout << "DType: " << typeid(DType).name() << std::endl;
    std::cout << "########################################" << std::endl;
  }

  template <typename SrcType, typename DstType>
  void convert_array(const SrcType *src, DstType *dst, int size) {
    for (int i = 0; i < size; i++) {
      dst[i] = static_cast<DstType>(src[i]);
    }
  }
};
