#pragma once

#ifndef __VITIS_HLS__
#include "./modulegen.hh"
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

    std::cout << "✓ Successfully generated " << output_path << "\n";
  }

private:
  static void validate_config(const json &config) {
    if (!config.contains("model")) {
      throw std::runtime_error("Config missing 'model' section");
    }

    if (!config.contains("modules")) {
      throw std::runtime_error("Config missing 'modules' section");
    }

    if (!config["modules"].is_array()) {
      throw std::runtime_error("'modules' must be an array");
    }
  }

  static void generate_header(std::ofstream &out, const std::string &json_path,
                              const json &config) {
    out << "#pragma once\n\n";
    out << "// AUTO-GENERATED CODE\n";
    out << "// Generated from: " << json_path << "\n";
    out << "#include \"path/to/proj/include/vhn.hh\"\n\n";
    std::string dtype = config["model"]["dtype"];

    out << "// Module Configuration Structs\n";

    UniversalCodeGen codegen;

    for (const auto &module : config["modules"]) {
      std::string module_name = module.value("name", "unnamed_module");
      out << codegen.generate_all_configs(module, module_name);
    }

    out << "\n// Module Type Aliases\n";

    for (const auto &module : config["modules"]) {
      std::string module_name = module.value("name", "unnamed_module");
      out << codegen.generate_all_type_aliases(module, dtype, module_name);
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
    out << "  constexpr int num_modules = " << config["modules"].size()
        << ";\n";

    out << "\n  // Module names\n";
    out << "  constexpr const char* module_names[] = {\n";
    bool first = true;
    for (const auto &module : config["modules"]) {
      if (!first)
        out << ",\n";
      out << "    \"" << module.value("name", "unnamed") << "\"";
      first = false;
    }
    out << "\n  };\n";

    out << "}\n";
  }
};

} // namespace vhn
#endif // __VITIS_HLS__
