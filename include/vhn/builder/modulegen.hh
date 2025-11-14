#pragma once

#ifndef __VITIS_HLS__
#include <iostream>
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

    if (module.contains("submodules") && module["submodules"].is_array()) {
      auto submodules = module["submodules"];
      for (size_t i = 0; i < submodules.size(); i++) {
        std::string submodule_name = module_name + "_sub" + std::to_string(i);
        oss << generate_all_configs(submodules[i], submodule_name);
      }
    }

    oss << generate_config_struct(module_name, module);

    return oss.str();
  }

  std::string
  generate_all_type_aliases(const json &module, const std::string &dtype,
                            const std::string &parent_name = "") const {
    std::ostringstream oss;

    std::string module_name = get_module_name(module, parent_name);

    if (module.contains("submodules") && module["submodules"].is_array()) {
      auto submodules = module["submodules"];
      for (size_t i = 0; i < submodules.size(); i++) {
        std::string submodule_name = module_name + "_sub" + std::to_string(i);
        oss << generate_all_type_aliases(submodules[i], dtype, submodule_name);
      }
    }

    oss << generate_type_alias(module_name, module, dtype);

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

    oss << "struct " << name << "_hparams {\n";
    if (module.contains("hparams") && !module["hparams"].empty()) {
      auto hparams = module["hparams"];
      for (auto it = hparams.begin(); it != hparams.end(); ++it) {
        std::string internal_name = it.key();
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
        } else if (it.value().is_array()) {
          oss << "int " << internal_name << "[] = {";
          for (size_t i = 0; i < it.value().size(); i++) {
            if (it.value()[i].is_string()) {
              oss << "\"" << it.value()[i].get<std::string>() << "\"";
            } else {
              oss << it.value()[i];
            }
            if (i < it.value().size() - 1) {
              oss << ", ";
            }
          }
          oss << "};\n";
        }
      }
    }

    if (has_submodules) {
      auto submodules = module["submodules"];
      oss << "\n  // Submodule hyperparameters\n";
      for (size_t i = 0; i < submodules.size(); i++) {
        oss << "  using submodule" << i << "_hparams = " << name << "_sub" << i
            << "_hparams;\n";
      }
    }

    oss << "};\n\n";

    oss << "struct " << name << "_cfg {\n";

    if (module.contains("hls_cfg") && !module["hls_cfg"].empty()) {
      auto hls_cfg = module["hls_cfg"];

      for (auto it = hls_cfg.begin(); it != hls_cfg.end(); ++it) {
        std::string internal_name = it.key();
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
        oss << "  using submodule" << i << "_cfg = " << name << "_sub" << i
            << "_cfg;\n";
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

    oss << "using " << name << "_t = vhn::" << module_type << "<" << dtype
        << ", " << name << "_hparams" << ", " << name << "_cfg, " << opt_level
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

    if (module.contains("hls_cfg") && !module["hls_cfg"].empty()) {
      return "OPT_ENABLED";
    }

    return "OPT_NONE";
  }
};

} // namespace vhn
#endif // __VITIS_HLS__
