#pragma once

#ifndef __VITIS_HLS__
#include "../builder/builder.hh"
#include <sstream>

namespace vhn {

class LinearBuilder : public BaseBuilder {
public:
  std::string generate_hparams(const std::string &name,
                               const json &module) const override {
    std::ostringstream oss;

    if (!module.contains("hparams")) {
      throw std::runtime_error("Linear module '" + name + "' missing params");
    }

    auto hparams = module["hparams"];

    if (!hparams.contains("in_features") ||
        !hparams["in_features"].is_number_unsigned()) {
      throw std::runtime_error("Linear module '" + name +
                               "' missing in_features param");
    }

    if (!hparams.contains("out_features") ||
        !hparams["out_features"].is_number_unsigned()) {
      throw std::runtime_error("Linear module '" + name +
                               "' missing out_features param");
    }

    auto in_features = hparams["in_features"];
    auto out_features = hparams["out_features"];

    oss << "using " << name << "_hparams = vhn::LinearHParams<";
    oss << in_features << ", " << out_features;
    oss << ">;\n\n";

    return oss.str();
  }

  std::string generate_config(const std::string &name,
                              const json &module) const override {
    std::ostringstream oss;

    oss << "struct " << name << "_cfg {\n";

    if (module.contains("hls_cfg") && !module["hls_cfg"].empty()) {
      auto hls_cfg = module["hls_cfg"];

      for (auto it = hls_cfg.begin(); it != hls_cfg.end(); ++it) {
        oss << "  static constexpr int " << it.key() << " = " << it.value()
            << ";\n";
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

    oss << "using " << name << "_t = vhn::Linear<" << dtype << ", " << name
        << "_hparams, " << name << "_cfg, " << opt_level << ">;\n";

    return oss.str();
  }
};
} // namespace vhn
#endif
