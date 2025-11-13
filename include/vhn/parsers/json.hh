#pragma once

#ifndef __VITIS_HLS__
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <vector>

namespace vhn {

using json = nlohmann::json;

class ModelConfigGenerator {
public:
  static void generate_from_json(const std::string &json_path,
                                 const std::string &output_path) {
    std::ifstream file(json_path);
    json config = json::parse(file);

    std::ofstream out(output_path);

    // Header
    out << "#pragma once\n\n";
    out << "// Auto-generated from " << json_path << "\n";
    out << "// DO NOT EDIT MANUALLY\n\n";
    out << "#include \"linear.hh\"\n\n";

    std::string dtype = config["model"]["dtype"];

    // Generate config structs
    for (const auto &layer : config["layers"]) {
      generate_layer_config(out, layer);
    }

    // Generate type aliases
    out << "// Type aliases for convenience\n";
    for (const auto &layer : config["layers"]) {
      generate_layer_typedef(out, layer, dtype);
    }

    // Generate network info
    generate_network_info(out, config);
  }

private:
  static void generate_layer_config(std::ofstream &out, const json &layer) {
    std::string name = layer["name"];
    auto hls_cfg = layer["hls_config"];

    out << "// Configuration for " << name << "\n";
    out << "struct " << name << "_config {\n";
    out << "  static constexpr int _unroll_factor = "
        << hls_cfg.value("unroll_factor", 1) << ";\n";
    out << "  static constexpr int _partition_factor = "
        << hls_cfg.value("partition_factor", 1) << ";\n";
    out << "  static constexpr int _pipeline_ii = "
        << hls_cfg.value("pipeline_ii", 1) << ";\n";
    out << "};\n\n";
  }

  static void generate_layer_typedef(std::ofstream &out, const json &layer,
                                     const std::string &dtype) {
    std::string name = layer["name"];
    std::string type = layer["type"];

    if (type == "linear") {
      int in_features = layer["in_features"];
      int out_features = layer["out_features"];
      bool has_opt = layer.contains("hls_config");

      out << "using " << name << "_t = vhn::Linear<" << dtype << ", "
          << in_features << ", " << out_features << ", " << name << "_config, "
          << (has_opt ? "vhn::OPT_ENABLED" : "hls_nn::OPT_NONE") << ">;\n";
    }
  }

  static void generate_network_info(std::ofstream &out, const json &config) {
    out << "\n// Network information\n";
    out << "namespace network_info {\n";
    out << "  constexpr const char* name = \""
        << config["model"]["name"].get<std::string>() << "\";\n";
    out << "  constexpr int num_layers = " << config["layers"].size() << ";\n";
    out << "}\n";
  }
};

} // namespace vhn
#endif
