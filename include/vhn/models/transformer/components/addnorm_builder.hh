#pragma once

#ifndef __VITIS_HLS__
#include "../../../builder/builder.hh"
#include "../../../norms/layernorm_builder.hh"
#include <sstream>

namespace vhn {

class AddNormBuilder : public BaseBuilder {
public:
  std::string generate_hparams(const std::string &name,
                               const std::string &dtype,
                               const json &module) const override {
    std::ostringstream oss;

    if (!module.contains("hparams")) {
      throw std::runtime_error("AddNorm module '" + name + "' missing hparams");
    }

    auto hparams = module["hparams"];

    if (!hparams.contains("d_model")) {
      throw std::runtime_error("AddNorm module '" + name +
                               "' missing d_model param");
    }

    if (!hparams.contains("norm_type")) {
      throw std::runtime_error("AddNorm module '" + name +
                               "' missing norm_type param");
    }

    auto d_model = hparams["d_model"].get<int>();
    auto norm_type_str = hparams["norm_type"].get<std::string>();

    json ln_module = {{"hparams", {{"hidden_dim", d_model}}},
                      {"opt_level", module.value("opt_level", "OPT_NONE")}};

    if (module.contains("hls_cfg")) {
      ln_module["hls_cfg"] = module["hls_cfg"];
    }

    LayerNormBuilder ln_builder;
    oss << ln_builder.generate_hparams(name + "_ln", dtype, ln_module);

    oss << "using " << name << "_hparams = vhn::AddNormHParams<";
    oss << name << "_ln_hparams, ";
    oss << (norm_type_str == "pre" ? "PRENORM" : "POSTNORM");
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

    if (!module.contains("hparams")) {
      throw std::runtime_error("AddNorm module '" + name + "' missing hparams");
    }

    auto hparams = module["hparams"];
    auto d_model = hparams["d_model"].get<int>();

    json ln_module = {{"hparams", {{"hidden_dim", d_model}}},
                      {"opt_level", opt_level}};

    if (module.contains("hls_cfg")) {
      ln_module["hls_cfg"] = module["hls_cfg"];
    }

    LayerNormBuilder ln_builder;
    oss << ln_builder.generate_config(name + "_ln", ln_module);

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

    if (!module.contains("hparams")) {
      throw std::runtime_error("AddNorm module '" + name + "' missing hparams");
    }

    auto hparams = module["hparams"];
    auto d_model = hparams["d_model"].get<int>();

    json ln_module = {{"hparams", {{"hidden_dim", d_model}}},
                      {"opt_level", opt_level}};

    if (module.contains("hls_cfg")) {
      ln_module["hls_cfg"] = module["hls_cfg"];
    }

    LayerNormBuilder ln_builder;
    oss << ln_builder.generate_type_alias(name + "_ln", dtype, ln_module);

    std::string config_type =
        (opt_level == "OPT_NONE") ? "void" : (name + "_cfg");

    oss << "using " << name << "_t = vhn::AddNorm<" << dtype << ", " << name
        << "_hparams, " << config_type << ", " << opt_level << ">;\n";

    return oss.str();
  }
};

} // namespace vhn

#endif
