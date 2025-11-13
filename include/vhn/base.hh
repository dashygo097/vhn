#pragma once

#ifndef VITIS_HLS
#include "./opt_level.hh"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

namespace vhn {
using json = nlohmann::json;

template <typename DType, OptLevel OPT_LEVEL> class BaseModule {
public:
  BaseModule() = default;
  virtual ~BaseModule() = default;

  static std::string type();
  static json hparams();
  static std::vector<json> submodules();
  static json to_json() {
    json j;
    j["type"] = type();
    j["params"] = hparams();
    j["submodules"] = submodules();
    if (OPT_LEVEL != OPT_NONE)
      j["opt_level"] = "OPT_ENABLED";
    else
      j["opt_level"] = "OPT_NONE";
    j["hls_config"] = json::object();
    return j;
  }
};
} // namespace vhn
#endif
