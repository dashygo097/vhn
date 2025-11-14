#pragma once

#include "./opt_level.hh"

#ifndef __VITIS_HLS__
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
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
  static std::string type() { return "BaseModule"; }
  static json hparams() { return json::object(); }
  static std::vector<json> submodules() { return std::vector<json>(); }

  static json to_json() {
    json j;
    j["type"] = type();
    j["params"] = hparams();

    auto subs = submodules();
    if (!subs.empty()) {
      j["submodules"] = json::array();
      for (const auto &sub : subs) {
        j["submodules"].push_back(sub);
      }
    } else {
      j["submodules"] = json::array();
    }

    if (OPT_LEVEL != OPT_NONE)
      j["opt_level"] = "OPT_ENABLED";
    else
      j["opt_level"] = "OPT_NONE";

    j["hls_config"] = json::object();

    return j;
  }

  static bool dump_config(const std::string &filepath, int indent = 2) {
    try {
      json config = to_json();
      std::ofstream out_file(filepath);

      if (!out_file.is_open()) {
        std::cerr << "Error: Cannot open file " << filepath << " for writing\n";
        return false;
      }

      out_file << config.dump(indent);
      out_file.close();

      std::cout << "✓ Successfully dumped config to " << filepath << "\n";
      return true;
    } catch (const std::exception &e) {
      std::cerr << "Error dumping config: " << e.what() << "\n";
      return false;
    }
  }

  static bool dump_network_config(const std::string &filepath,
                                  const std::string &network_name = "Network",
                                  const std::string &dtype_name = "float",
                                  int indent = 2) {
    try {
      json network_config;
      network_config["model"]["name"] = network_name;
      network_config["model"]["dtype"] = dtype_name;
      network_config["modules"] = json::array();
      network_config["modules"].push_back(to_json());

      std::ofstream out_file(filepath);
      if (!out_file.is_open()) {
        std::cerr << "Error: Cannot open file " << filepath << " for writing\n";
        return false;
      }

      out_file << network_config.dump(indent);
      out_file.close();

      std::cout << "✓ Successfully dumped network config to " << filepath
                << "\n";
      return true;
    } catch (const std::exception &e) {
      std::cerr << "Error dumping network config: " << e.what() << "\n";
      return false;
    }
  }
#endif
};

} // namespace vhn
