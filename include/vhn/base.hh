#pragma once

#include "./opt_level.hh"

#ifndef __VITIS_HLS__
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#endif

namespace vhn {
#ifndef __VITIS_HLS__
using json = nlohmann::json;
#endif

template <typename DType, OptLevel OPT_LEVEL> class BaseModule {
public:
  BaseModule() = default;
  virtual ~BaseModule() = default;

#ifndef __VITIS_HLS__
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
#endif
};

} // namespace vhn
