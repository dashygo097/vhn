#pragma once

#ifndef __VITIS_HLS__
#include "../builder/builder.hh"
#include <sstream>

namespace vhn {

class MLPBuilder : public BaseBuilder {
public:
  std::string generate_hparams(const std::string &name,
                               const json &module) const override {
    std::ostringstream oss;

    if (!module.contains("hparams")) {
      throw std::runtime_error("MLP module '" + name + "' missing params");
    }

    auto hparams = module["hparams"];

    if (!hparams.contains("hidden_features") ||
        !hparams["hidden_features"].is_array()) {
      throw std::runtime_error("MLP module '" + name +
                               "' missing hidden_features array");
    }

    auto hidden_features = hparams["hidden_features"];

    oss << "using " << name << "_hparams = vhn::MLPHParams<";
    for (size_t i = 0; i < hidden_features.size(); i++) {
      oss << hidden_features[i].get<int>();
      if (i < hidden_features.size() - 1) {
        oss << ", ";
      }
    }
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

    if (has_submodules(module)) {
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

  std::string generate_type_alias(const std::string &name,
                                  const std::string &dtype,
                                  const json &module) const override {
    std::ostringstream oss;

    std::string opt_level = module.value("opt_level", "OPT_NONE");

    oss << "using " << name << "_t = vhn::MLP<" << dtype << ", " << name
        << "_hparams, " << name << "_cfg, " << opt_level << ">;\n";

    return oss.str();
  }

  bool has_submodules(const json &module) const override {
    return module.contains("submodules") && module["submodules"].is_array() &&
           !module["submodules"].empty();
  }
};

} // namespace vhn
#endif
