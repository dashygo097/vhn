#pragma once

#ifndef __VITIS_HLS__
#include "../builder/builder.hh"
#include <sstream>

namespace vhn {

class BatchNorm2dBuilder : public BaseBuilder {
public:
  std::string generate_hparams(const std::string &name,
                               const std::string &dtype,
                               const json &hparams) const override {
    std::ostringstream oss;
    NECESSARY_HPARAMS("BatchNorm2d", name, "channels")
    NECESSARY_HPARAMS("BatchNorm2d", name, "width")
    NECESSARY_HPARAMS("BatchNorm2d", name, "height")

    auto channels = hparams["channels"].get<int>();
    auto width = hparams["width"].get<int>();
    auto height = hparams["height"].get<int>();

    oss << "using " << name << "_hparams = vhn::BatchNorm2dHParams<";
    oss << channels << ", " << width << ", " << height;
    oss << ">;\n\n";

    return oss.str();
  }

  std::string generate_config(const std::string &name,
                              const json &hls_cfg) const override {
    if (hls_cfg.empty() || hls_cfg.is_null()) {
      return "";
    }

    std::ostringstream oss;

    oss << "struct " << name << "_cfg {\n";

    for (auto it = hls_cfg.begin(); it != hls_cfg.end(); ++it) {
      if (it.value().is_boolean()) {
        oss << "  static constexpr bool " << it.key() << " = "
            << (it.value().get<bool>() ? "true" : "false") << ";\n";
      } else if (it.value().is_number_integer()) {
        oss << "  static constexpr int " << it.key() << " = "
            << it.value().get<int>() << ";\n";
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
