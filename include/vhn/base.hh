#pragma once

#ifndef VITIS_HLS
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

namespace vhn {
using json = nlohmann::json;

class BaseLayer {
public:
  BaseLayer() = default;
  virtual ~BaseLayer() = default;

  static std::string type();
  static json params();
  static json to_json() {
    json j;
    j["type"] = type();
    j["params"] = params();
    j["opt_level"] = "OPT_NONE";
    j["hls_config"] = json::object();
    return j;
  }
};
} // namespace vhn
#endif
