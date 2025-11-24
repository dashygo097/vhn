#pragma once

#ifndef __VITIS_HLS__
#include "../../../builder/builder.hh"
#include "../../../norms/ln_builder.hh"
#include "../../../operators/elementwise_builder.hh"
#include <sstream>

namespace vhn {

class AddNormBuilder : public BaseBuilder {
public:
  std::string generate_hparams(const std::string &name,
                               const std::string &dtype,
                               const json &hparams) const override {
    std::ostringstream oss;
    NECESSARY_HPARAMS("AddNorm", name, "d_model")
    NECESSARY_HPARAMS("AddNorm", name, "norm_type")

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
    if (hls_cfg.is_null() && hls_cfg.empty()) {
      return "";
    }

    std::ostringstream oss;

    auto norm_cfg = hls_cfg.value("ln", json::object());
    auto add_cfg = hls_cfg.value("add", json::object());

    auto memory_partition = hls_cfg.value("memory_partition", 4);

    LayerNormBuilder layernorm_builder;
    ElementwiseBuilder elementwise_builder;

    if (hls_cfg.contains("ln") && !norm_cfg.empty())
      oss << layernorm_builder.generate_config(name + "_norm", norm_cfg);
    if (hls_cfg.contains("add") && !hls_cfg["add"].empty())
      oss << elementwise_builder.generate_config(name + "_add", hls_cfg["add"]);

    oss << "using " << name << "_cfg = vhn::AddNormConfig<";
    if (hls_cfg.contains("ln") && !norm_cfg.empty())
      oss << name << "_norm_cfg, ";
    else
      oss << "void, ";
    if (hls_cfg.contains("add") && !add_cfg.empty())
      oss << name << "_add_cfg, ";
    else
      oss << "void, ";
    oss << memory_partition;
    oss << ">;\n\n";

    return oss.str();
  }

  std::string generate_type_alias(const std::string &name,
                                  const std::string &dtype,
                                  const json &hls_cfg) const override {
    std::ostringstream oss;

    std::string opt_level = "OPT_NONE";

    if (!hls_cfg.empty() && !hls_cfg.is_null()) {
      opt_level = "OPT_ENABLED";
    }

    std::string config_type =
        (opt_level == "OPT_NONE") ? "void" : (name + "_cfg");

    GENERATE_TYPE_ALIAS(oss, "AddNorm", name, dtype, opt_level)
    return oss.str();
  }
};

} // namespace vhn

#endif
