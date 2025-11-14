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
    int in_features = hparams.value("in_features", 0);
    int out_features = hparams.value("out_features", 0);

    oss << "struct " << name << "_hparams {\n";
    oss << "  static constexpr int in_features = " << in_features << ";\n";
    oss << "  static constexpr int out_features = " << out_features << ";\n";
    oss << "};\n\n";

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
