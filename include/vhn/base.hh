#pragma once

#include <fstream>
#include <iostream>

#ifndef __VITIS_HLS__
#include <nlohmann/json.hpp>
#endif

namespace vhn {

#ifndef __VITIS_HLS__
using json = nlohmann::json;
#endif

class BaseLayer {
public:
  BaseLayer() = default;
  virtual ~BaseLayer() = default;

#ifndef __VITIS_HLS__
  virtual std::string type() const = 0;
  virtual json params() const = 0;
  virtual json to_json() const = 0;

  bool dump(const std::string &filepath, int indent = 2) const {
    try {
      json config = to_json();
      std::ofstream out_file(filepath);

      if (!out_file.is_open()) {
        std::cerr << "Error: Cannot open file " << filepath << " for writing"
                  << std::endl;
        return false;
      }

      out_file << config.dump(indent);
      out_file.close();

      std::cout << "Successfully dumped config to " << filepath << std::endl;
      return true;
    } catch (const std::exception &e) {
      std::cerr << "Error dumping config: " << e.what() << std::endl;
      return false;
    }
  }
#endif
};

class BaseModule : public BaseLayer {
protected:
#ifndef __VITIS_HLS__
  std::vector<std::shared_ptr<BaseLayer>> submodules;
#endif

public:
  BaseModule() = default;
  virtual ~BaseModule() = default;

#ifndef __VITIS_HLS__
  void add_submodule(std::shared_ptr<BaseModule> submodule) {
    submodules.push_back(submodule);
  }

  int get_num_submodules() const { return submodules.size(); }

  std::shared_ptr<BaseLayer> get_submodule(int idx) const {
    if (idx >= 0 && idx < (int)submodules.size()) {
      return submodules[idx];
    }
    return nullptr;
  }

  json export_submodules_to_json() const {
    json submodules_json = json::array();
    for (const auto &submodule : submodules) {
      submodules_json.push_back(submodule->to_json());
    }
    return submodules_json;
  }
#endif
};

struct HLSConfig {
  int unroll_factor;
  int partition_factor;
  int pipeline_ii;

#ifndef __VITIS_HLS__
  json to_json() const {
    json j;

    if (unroll_factor != 0)
      j["unroll_factor"] = unroll_factor;
    if (partition_factor != 0)
      j["partition_factor"] = partition_factor;
    if (pipeline_ii != 0)
      j["pipeline_ii"] = pipeline_ii;

    return j;
  }

  static HLSConfig from_json(const json &j) {
    HLSConfig config;
    if (j.contains("unroll_factor"))
      config.unroll_factor = j["unroll_factor"];
    if (j.contains("partition_factor"))
      config.partition_factor = j["partition_factor"];
    if (j.contains("pipeline_ii"))
      config.pipeline_ii = j["pipeline_ii"];
    return config;
  }
#endif
};

} // namespace vhn
