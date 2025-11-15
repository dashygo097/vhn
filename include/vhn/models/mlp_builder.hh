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
    std::string opt_level = module.value("opt_level", "OPT_NONE");

    if (opt_level == "OPT_NONE") {
      return "";
    }

    std::ostringstream oss;

    oss << "struct " << name << "_cfg {\n";

    bool has_hls_cfg = module.contains("hls_cfg") && !module["hls_cfg"].empty();

    if (has_hls_cfg) {
      auto hls_cfg = module["hls_cfg"];

      if (hls_cfg.contains("batch_loop_ii")) {
        oss << "  static constexpr int batch_loop_ii = "
            << hls_cfg["batch_loop_ii"].get<int>() << ";\n";
      } else {
        oss << "  static constexpr int batch_loop_ii = 1;\n";
      }

    } else {
      oss << "  static constexpr int batch_loop_ii = 1;\n";
    }

    if (has_submodules(module)) {
      auto submodules = module["submodules"];

      oss << "\n  // Submodule configuration selector\n";
      oss << "  template <int LayerIdx>\n";
      oss << "  using submodule_cfg = typename std::conditional<LayerIdx == "
             "0, ";

      for (size_t i = 0; i < submodules.size(); i++) {
        std::string submodule_opt =
            submodules[i].value("opt_level", "OPT_NONE");

        if (i == 0) {
          if (submodules.size() == 1) {
            if (submodule_opt == "OPT_NONE") {
              oss << "void, void>::type";
            } else {
              oss << name << "_sub0_cfg, void>::type";
            }
          } else {
            if (submodule_opt == "OPT_NONE") {
              oss << "void, typename std::conditional<LayerIdx == 1, ";
            } else {
              oss << name
                  << "_sub0_cfg, typename std::conditional<LayerIdx == 1, ";
            }
          }
        } else if (i == submodules.size() - 1) {
          if (submodule_opt == "OPT_NONE") {
            oss << "void, void";
          } else {
            oss << name << "_sub" << i << "_cfg, void";
          }
          for (size_t j = 1; j < submodules.size(); j++) {
            oss << ">::type";
          }
        } else {
          if (submodule_opt == "OPT_NONE") {
            oss << "void, typename std::conditional<LayerIdx == " << (i + 1)
                << ", ";
          } else {
            oss << name << "_sub" << i
                << "_cfg, typename std::conditional<LayerIdx == " << (i + 1)
                << ", ";
          }
        }
      }

      oss << ";\n";
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

    oss << "using " << name << "_t = vhn::MLP<" << dtype << ", " << name
        << "_hparams, " << config_type << ", " << opt_level << ">;\n";

    return oss.str();
  }

  bool has_submodules(const json &module) const override {
    return module.contains("submodules") && module["submodules"].is_array() &&
           !module["submodules"].empty();
  }
};

} // namespace vhn
#endif
