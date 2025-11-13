#pragma once

#ifndef __VITIS_HLS__
#include "./modulegen.hh"
#include "./registry.hh"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace vhn {
using json = nlohmann::json;

class ModelConfigGenerator {
public:
  static void generate_from_json(const std::string &json_path,
                                 const std::string &output_path) {
    std::ifstream file(json_path);
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open file: " + json_path);
    }

    json config;
    try {
      config = json::parse(file);
    } catch (const json::exception &e) {
      throw std::runtime_error("JSON parse error: " + std::string(e.what()));
    }

    validate_config(config);

    std::ofstream out(output_path);
    if (!out.is_open()) {
      throw std::runtime_error("Cannot create output file: " + output_path);
    }

    generate_header(out, json_path, config);

    std::cout << "✓ Successfully generated " << output_path << " with "
              << config["layers"].size() << " layers\n";
  }

private:
  static void validate_config(const json &config) {
    if (!config.contains("model")) {
      throw std::runtime_error("Config missing 'model' section");
    }

    if (!config.contains("layers")) {
      throw std::runtime_error("Config missing 'layers' section");
    }

    if (!config["layers"].is_array()) {
      throw std::runtime_error("'layers' must be an array");
    }

    for (size_t i = 0; i < config["layers"].size(); ++i) {
      const auto &layer = config["layers"][i];

      if (!layer.contains("name")) {
        throw std::runtime_error("Layer at index " + std::to_string(i) +
                                 " missing 'name'");
      }

      if (!layer.contains("type")) {
        throw std::runtime_error("Layer '" + layer["name"].get<std::string>() +
                                 "' missing 'type'");
      }
    }
  }

  static void generate_header(std::ofstream &out, const std::string &json_path,
                              const json &config) {
    out << "#pragma once\n\n";
    out << "// ============================================\n";
    out << "// AUTO-GENERATED CODE\n";
    out << "// Generated from: " << json_path << "\n";
    out << "// ============================================\n";
    out << "#include \"path/to/proj/include/vhn.hh\"\n\n";

    std::string dtype = config["model"].value("dtype", "float");

    out << "\n// Layer Configuration Structs\n";

    UniversalCodeGen universal_gen;

    for (const auto &layer : config["layers"]) {
      std::string name = layer["name"];
      out << universal_gen.generate_config_struct(name, layer);
    }

    out << "\n// Layer Type Aliases\n";

    for (const auto &layer : config["layers"]) {
      std::string name = layer["name"];
      out << universal_gen.generate_type_alias(name, layer, dtype);
    }

    generate_network_info(out, config);
  }

  static void generate_network_info(std::ofstream &out, const json &config) {
    out << "\n// Network Information\n";

    out << "namespace network_info {\n";
    out << "  constexpr const char* name = \""
        << config["model"].value("name", "Network") << "\";\n";
    out << "  constexpr const char* dtype = \""
        << config["model"].value("dtype", "float") << "\";\n";
    out << "  constexpr int num_layers = " << config["layers"].size() << ";\n";

    out << "\n  // Layer names\n";
    out << "  constexpr const char* layer_names[] = {\n";
    bool first = true;
    for (const auto &layer : config["layers"]) {
      if (!first)
        out << ",\n";
      out << "    \"" << layer["name"].get<std::string>() << "\"";
      first = false;
    }
    out << "\n  };\n";

    out << "\n  // Layer types\n";
    out << "  constexpr const char* layer_types[] = {\n";
    first = true;
    for (const auto &layer : config["layers"]) {
      if (!first)
        out << ",\n";
      out << "    \"" << layer["type"].get<std::string>() << "\"";
      first = false;
    }
    out << "\n  };\n";

    out << "\n  // Layer parameters (for debugging)\n";
    out << "  constexpr const char* layer_params[] = {\n";
    first = true;
    for (const auto &layer : config["layers"]) {
      if (!first)
        out << ",\n";

      std::string params_str = "";
      if (layer.contains("params")) {
        params_str = layer["params"].dump();
        size_t pos = 0;
        while ((pos = params_str.find("\"", pos)) != std::string::npos) {
          params_str.replace(pos, 1, "\\\"");
          pos += 2;
        }
      }
      out << "    \"" << params_str << "\"";
      first = false;
    }
    out << "\n  };\n";

    out << "}\n";
  }
};

} // namespace vhn
#endif // __VITIS_HLS__
