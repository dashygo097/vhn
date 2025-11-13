#pragma once

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

namespace hls_nn {

using json = nlohmann::json;

template <typename DType> class BaseModule {
public:
  using dtype = DType;

  virtual ~BaseModule() = default;

  virtual json to_json() const = 0;
  virtual std::string module_name() const = 0;
  virtual std::string module_type() const = 0;

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
};

struct HLSConfig {
  int unroll_factor = 1;
  int partition_factor = 1;
  int pipeline_ii = 1;

  json to_json() const {
    json j;
    j["unroll_factor"] = unroll_factor;
    j["partition_factor"] = partition_factor;
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
};
} // namespace hls_nn
