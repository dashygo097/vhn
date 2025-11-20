#pragma once

#ifndef __VITIS_HLS__
#include "registry.hh"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace vhn {

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

    auto &registry = LayerRegistry::instance();
    for (const auto &module : config["modules"]) {
      std::string type = module.value("type", "");
      if (!registry.has_layer(type)) {
        throw std::runtime_error("Unknown module type: " + type);
      }
    }
  }

  static void generate_header(std::ofstream &out, const std::string &json_path,
                              const json &config) {
    out << "#pragma once\n\n";
    out << "// AUTO-GENERATED CODE\n";
    out << "// Generated from: " << json_path << "\n";
    out << "#include \"path/to/proj/include/vhn.hh\"\n\n";

    std::string dtype = config["model"].value("dtype", "float");

    out << "// Module Configurations\n\n";

    auto &registry = LayerRegistry::instance();

    // dfs
    for (const auto &module : config["modules"]) {
      generate_module_recursive(out, module, module.value("name", "module"),
                                dtype, registry);
    }

    out << "// Network Information\n";
    out << "namespace network_info {\n";
    out << "  constexpr const char* name = \""
        << config["model"].value("name", "Network") << "\";\n";
    out << "  constexpr const char* dtype = \"" << dtype << "\";\n";
    out << "  constexpr int num_modules = " << config["modules"].size()
        << ";\n";
    out << "}\n";
  }

  static void generate_module_recursive(std::ofstream &out, const json &module,
                                        const std::string &name,
                                        const std::string &dtype,
                                        LayerRegistry &registry) {
    std::string type = module.value("type", "");
    auto generator = registry.get_generator(type);

    if (!generator) {
      throw std::runtime_error("No generator for type: " + type);
    }

    out << "// Configuration for " << name << " (" << type << ")\n";
    out << generator->generate_hparams(name, dtype,
                                       module.value("hparams", json::object()));
    out << generator->generate_config(name,
                                      module.value("hls_cfg", json::object()));
    out << generator->generate_type_alias(name, dtype, module);
    out << "\n";
  }
};

} // namespace vhn
#endif
