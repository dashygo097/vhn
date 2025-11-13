#pragma once

#ifndef __VITIS_HLS__
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace vhn {
using json = nlohmann::json;

class UniversalCodeGen {
public:
  std::string generate_all_configs(const json &module,
                                   const std::string &parent_name = "") const {
    std::ostringstream oss;

    std::string module_name = get_module_name(module, parent_name);
    oss << generate_config_struct(module_name, module);

    if (module.contains("submodules") && module["submodules"].is_array()) {
      auto submodules = module["submodules"];
      for (size_t i = 0; i < submodules.size(); i++) {
        std::string submodule_name = module_name + "_sub" + std::to_string(i);
        oss << generate_all_configs(submodules[i], submodule_name);
      }
    }

    return oss.str();
  }

  std::string
  generate_all_type_aliases(const json &module, const std::string &dtype,
                            const std::string &parent_name = "") const {
    std::ostringstream oss;

    std::string module_name = get_module_name(module, parent_name);
    oss << generate_type_alias(module_name, module, dtype);

    if (module.contains("submodules") && module["submodules"].is_array()) {
      auto submodules = module["submodules"];
      for (size_t i = 0; i < submodules.size(); i++) {
        std::string submodule_name = module_name + "_sub" + std::to_string(i);
        oss << generate_all_type_aliases(submodules[i], dtype, submodule_name);
      }
    }

    return oss.str();
  }

  std::string generate_config_struct(const std::string &name,
                                     const json &module) const {
    std::ostringstream oss;

    std::string module_type = module.value("type", "Unknown");
    bool has_submodules = module.contains("submodules") &&
                          module["submodules"].is_array() &&
                          !module["submodules"].empty();

    oss << "// Configuration for " << name << " (" << module_type;
    if (has_submodules) {
      oss << " - composite module with " << module["submodules"].size()
          << " submodules";
    }
    oss << ")\n";

    oss << "struct " << name << "_config {\n";

    if (module.contains("hls_config") && !module["hls_config"].empty()) {
      auto hls_cfg = module["hls_config"];

      for (auto it = hls_cfg.begin(); it != hls_cfg.end(); ++it) {
        std::string internal_name = "_" + it.key();
        oss << "  static constexpr ";

        if (it.value().is_number_integer()) {
          oss << "int " << internal_name << " = " << it.value() << ";\n";
        } else if (it.value().is_number_float()) {
          oss << "double " << internal_name << " = " << it.value() << ";\n";
        } else if (it.value().is_boolean()) {
          oss << "bool " << internal_name << " = "
              << (it.value().get<bool>() ? "true" : "false") << ";\n";
        } else if (it.value().is_string()) {
          oss << "const char* " << internal_name << " = \""
              << it.value().get<std::string>() << "\";\n";
        }
      }
    }

    if (has_submodules) {
      auto submodules = module["submodules"];

      oss << "\n  // Submodule configurations\n";
      for (size_t i = 0; i < submodules.size(); i++) {
        oss << "  using submodule" << i << "_config = " << name << "_sub" << i
            << "_config;\n";
      }
    }

    oss << "};\n\n";
    return oss.str();
  }

  std::string generate_type_alias(const std::string &name, const json &module,
                                  const std::string &dtype) const {
    std::ostringstream oss;

    if (!module.contains("type")) {
      throw std::runtime_error("Module '" + name + "' missing 'type' field");
    }

    std::string module_type = module["type"];
    std::string opt_level = determine_opt_level(module);
    std::string template_hparams = build_template_hparams(module, dtype);

    oss << "using " << name << "_t = vhn::" << module_type << "<"
        << template_hparams << ", " << name << "_config, " << opt_level
        << ">;\n";

    return oss.str();
  }

private:
  std::string get_module_name(const json &module,
                              const std::string &parent_name) const {
    if (module.contains("name")) {
      return module["name"].get<std::string>();
    }
    return parent_name.empty() ? "module" : parent_name;
  }

  std::string determine_opt_level(const json &module) const {
    if (module.contains("opt_level")) {
      return module["opt_level"].get<std::string>();
    }

    if (module.contains("hls_config") && !module["hls_config"].empty()) {
      return "OPT_ENABLED";
    }

    return "OPT_NONE";
  }

  std::string build_template_hparams(const json &module,
                                     const std::string &dtype) const {
    std::string result = dtype;

    if (module.contains("hparams") && !module["hparams"].empty()) {
      result += format_hparams(module["hparams"]);
    }

    return result;
  }

  std::string format_hparams(const json &hparams) const {
    std::ostringstream oss;

    for (auto it = hparams.begin(); it != hparams.end(); ++it) {
      if (it.value().is_array() || it.value().is_object()) {
        continue;
      }

      oss << ", ";

      if (it.value().is_number_integer()) {
        oss << it.value().get<int>();
      } else if (it.value().is_number_float()) {
        oss << it.value().get<double>();
      } else if (it.value().is_boolean()) {
        oss << (it.value().get<bool>() ? "true" : "false");
      } else if (it.value().is_string()) {
        oss << it.value().get<std::string>();
      }
    }

    return oss.str();
  }

  static std::string to_lowercase(const std::string &str) {
    std::string result = str;
    for (char &c : result) {
      c = std::tolower(c);
    }
    return result;
  }
};

} // namespace vhn
#endif // __VITIS_HLS__
