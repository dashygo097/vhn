#pragma once

#ifndef __VITIS_HLS__
#include "../builder/builder.hh"
#include <sstream>

namespace vhn {

class LinearBuilder : public BaseBuilder {
public:
  std::string generate_hparams(const std::string &name,
                               const std::string &dtype,
                               const json &hparams) const override {
    std::ostringstream oss;
    NECESSARY_HPARAMS("Linear", name, "in_features");
    NECESSARY_HPARAMS("Linear", name, "out_features");

    auto in_features = hparams["in_features"];
    auto out_features = hparams["out_features"];

    oss << "using " << name << "_hparams = vhn::LinearHParams<";
    oss << in_features << ", " << out_features;
    oss << ">;\n\n";

    return oss.str();
  }

  std::string generate_config(const std::string &name,
                              const json &hls_cfg) const override {
    if (hls_cfg.empty() || hls_cfg.is_null()) {
      return "";
    }

    std::ostringstream oss;

    auto unroll_factor = hls_cfg.value("unroll_factor", 4);
    auto partition_factor = hls_cfg.value("partition_factor", 4);
    auto tile_size_out = hls_cfg.value("tile_size_out", 16);
    auto tile_size_in = hls_cfg.value("tile_size_in", 16);
    auto use_systolic = hls_cfg.value("use_systolic", false);

    oss << "using " << name << "_cfg = vhn::LinearConfig<";
    oss << unroll_factor << ", " << partition_factor << ", ";
    oss << tile_size_out << ", " << tile_size_in << ", ";
    oss << (use_systolic ? "true" : "false");
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

    GENERATE_TYPE_ALIAS(oss, "Linear", name, dtype, opt_level)
    return oss.str();
  }
};

} // namespace vhn
#endif
