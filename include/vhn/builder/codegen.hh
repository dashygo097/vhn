#pragma once

#ifndef __VITIS_HLS__
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>

namespace vhn {
using json = nlohmann::json;

class UniversalCodeGen {
public:
  std::string generate_config_struct(const std::string &name,
                                     const json &layer) const {
    std::ostringstream oss;

    std::string layer_type = layer.value("type", "Unknown");

    oss << "// Configuration for " << name << " (" << layer_type << " layer)\n";
    oss << "struct " << name << "_config {\n";

    if (layer.contains("hls_config") && !layer["hls_config"].empty()) {
      auto hls_cfg = layer["hls_config"];

      for (auto it = hls_cfg.begin(); it != hls_cfg.end(); ++it) {
        std::string field_name = it.key();

        std::string internal_name = "_" + field_name;

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

    oss << "};\n\n";
    return oss.str();
  }

  std::string generate_type_alias(const std::string &name, const json &layer,
                                  const std::string &dtype) const {
    std::ostringstream oss;

    if (!layer.contains("type")) {
      throw std::runtime_error("Layer '" + name + "' missing 'type' field");
    }

    std::string layer_type = layer["type"];

    std::string class_name = layer_type;
    if (!class_name.empty()) {
      class_name[0] = std::toupper(class_name[0]);
    }

    std::string opt_level = "vhn::OPT_NONE";
    if (layer.contains("opt_level")) {
      opt_level = "vhn::" + layer["opt_level"].get<std::string>();
    } else if (layer.contains("hls_config") && !layer["hls_config"].empty()) {
      opt_level = "vhn::OPT_ENABLED";
    }

    std::string template_params = dtype;

    if (layer.contains("params") && !layer["params"].empty()) {
      template_params += format_params(layer["params"]);
    }

    oss << "using " << name << "_t = vhn::" << class_name << "<"
        << template_params << ", " << name << "_config, " << opt_level
        << ">;\n";

    return oss.str();
  }

private:
  std::string format_params(const json &params) const {
    std::ostringstream oss;

    for (auto it = params.begin(); it != params.end(); ++it) {
      oss << ", ";

      if (it.value().is_number_integer()) {
        oss << it.value().get<int>();
      } else if (it.value().is_number_float()) {
        oss << it.value().get<double>();
      } else if (it.value().is_boolean()) {
        oss << (it.value().get<bool>() ? "true" : "false");
      } else if (it.value().is_string()) {
        oss << it.value().get<std::string>();
      } else {
        throw std::runtime_error("Unsupported parameter type for: " + it.key());
      }
    }

    return oss.str();
  }
};

} // namespace vhn
#endif // __VITIS_HLS__
