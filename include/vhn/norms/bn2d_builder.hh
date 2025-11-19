#pragma once

#ifndef __VITIS_HLS__
#include "../builder/builder.hh"
#include <sstream>

namespace vhn {

class BatchNorm2dBuilder : public BaseBuilder {
public:
  std::string generate_hparams(const std::string &name,
                               const std::string &dtype,
                               const json &module) const override {
    std::ostringstream oss;

    if (!module.contains("hparams")) {
      throw std::runtime_error("BatchNorm2d module '" + name +
                               "' missing hparams");
    }

    auto hparams = module["hparams"];

    if (!hparams.contains("channels")) {
      throw std::runtime_error("BatchNorm2d module '" + name +
                               "' missing channels param");
    }

    if (!hparams.contains("width")) {
      throw std::runtime_error("BatchNorm2d module '" + name +
                               "' missing width param");
    }

    if (!hparams.contains("height")) {
      throw std::runtime_error("BatchNorm2d module '" + name +
                               "' missing height param");
    }

    auto channels = hparams["channels"].get<int>();
    auto width = hparams["width"].get<int>();
    auto height = hparams["height"].get<int>();

    oss << "using " << name << "_hparams = vhn::BatchNorm2dHParams<";
    oss << channels << ", " << width << ", " << height;
    oss << ">;\n\n";

    return oss.str();
  }

  std::string generate_config(const std::string &name,
                              const json &module) const override {
    std::string opt_level = module.value("opt_level", "OPT_NONE");

    if (opt_level == "OPT_NONE") {
      return "";
    }

    std::ostringstream oss;

    oss << "struct " << name << "_cfg {\n";

    if (module.contains("hls_cfg") && !module["hls_cfg"].empty()) {
      auto hls_cfg = module["hls_cfg"];

      for (auto it = hls_cfg.begin(); it != hls_cfg.end(); ++it) {
        if (it.value().is_boolean()) {
          oss << "  static constexpr bool " << it.key() << " = "
              << (it.value().get<bool>() ? "true" : "false") << ";\n";
        } else if (it.value().is_number_integer()) {
          oss << "  static constexpr int " << it.key() << " = "
              << it.value().get<int>() << ";\n";
        }
      }
    }

    oss << "};\n\n";

    return oss.str();
  }

  std::string generate_type_alias(const std::string &name,
                                  const std::string &dtype,
                                  const json &module) const override {
    std::ostringstream oss;

    std::string opt_level = module.value("opt_level", "OPT_NONE");

    std::string config_type =
        (opt_level == "OPT_NONE") ? "void" : (name + "_cfg");

    oss << "using " << name << "_t = vhn::BatchNorm2d<" << dtype << ", " << name
        << "_hparams, " << config_type << ", " << opt_level << ">;\n";

    return oss.str();
  }
};

} // namespace vhn

#endif
