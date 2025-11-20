#pragma once

#ifndef __VITIS_HLS__
#include "../builder/builder.hh"
#include <sstream>

namespace vhn {

class LayerNormBuilder : public BaseBuilder {
public:
  std::string generate_hparams(const std::string &name,
                               const std::string &dtype,
                               const json &hparams) const override {
    std::ostringstream oss;
    NECESSARY_HPARAMS("LayerNorm", name, "hidden_dim")

    auto hidden_dim = hparams["hidden_dim"];

    oss << "using " << name << "_hparams = vhn::LayerNormHParams<";
    oss << hidden_dim;
    oss << ">;\n\n";

    return oss.str();
  }

  std::string generate_config(const std::string &name,
                              const json &hls_cfg) const override {
    if (hls_cfg.empty() || hls_cfg.is_null()) {
      return "";
    }

    std::ostringstream oss;

    auto pipeline_ii = hls_cfg.value("pipeline_ii", 1);
    auto unroll_factor = hls_cfg.value("unroll_factor", 4);
    auto partition_factor = hls_cfg.value("partition_factor", 4);

    oss << "using " << name << "_cfg = vhn::LayerNormConfig<";
    oss << pipeline_ii << ", " << unroll_factor << ", " << partition_factor;
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

    GENERATE_TYPE_ALIAS(oss, "LayerNorm", name, dtype, opt_level)
    return oss.str();
  }
};

} // namespace vhn
#endif
