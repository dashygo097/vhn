#pragma once

#ifndef __VITIS_HLS__
#include "../../../builder/builder.hh"
#include "../../../norms/ln_builder.hh"
#include <sstream>

namespace vhn {

class AddNormBuilder : public BaseBuilder {
public:
  std::string generate_hparams(const std::string &name,
                               const std::string &dtype,
                               const json &hparams) const override {
    std::ostringstream oss;

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

    json ln_hparams = {{"hidden_dim", d_model}};

    LayerNormBuilder ln_builder;
    oss << ln_builder.generate_hparams(name + "_ln", dtype, ln_hparams);

    oss << "using " << name << "_hparams = vhn::AddNormHParams<";
    oss << name << "_ln_hparams, ";
    oss << (norm_type_str == "pre" ? "PRENORM" : "POSTNORM");
    oss << ">;\n\n";

    return oss.str();
  }

  std::string generate_config(const std::string &name,
                              const json &hls_cfg) const override {
    if (hls_cfg.empty() || hls_cfg.is_null()) {
      return "";
    }

    std::ostringstream oss;

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
