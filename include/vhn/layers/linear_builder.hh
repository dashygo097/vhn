#pragma once

#ifndef __VITIS_HLS__
#include "../builder/builder.hh"
#include <sstream>

namespace vhn {

class LinearBuilder : public BaseBuilder {
public:
  std::string generate_hparams(const std::string &name,
                               const std::string &dtype,
                               const json &module) const override {
    std::ostringstream oss;

    if (!module.contains("hparams")) {
      throw std::runtime_error("Linear module '" + name + "' missing params");
    }

    auto hparams = module["hparams"];

    if (!hparams.contains("in_features")) {
      throw std::runtime_error("Linear module '" + name +
                               "' missing in_features param");
    }

    if (!hparams.contains("out_features")) {
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
    std::string opt_level = module.value("opt_level", "OPT_NONE");

    if (opt_level == "OPT_NONE") {
      return "";
    }

    std::ostringstream oss;

    int unroll_factor = 1;
    int partition_factor = 1;

    if (module.contains("hls_cfg") && !module["hls_cfg"].empty()) {
      auto hls_cfg = module["hls_cfg"];

      if (hls_cfg.contains("unroll_factor")) {
        unroll_factor = hls_cfg["unroll_factor"].get<int>();
      }

      if (hls_cfg.contains("partition_factor")) {
        partition_factor = hls_cfg["partition_factor"].get<int>();
      }
    }

    oss << "using " << name << "_cfg = vhn::LinearConfig<";
    oss << unroll_factor << ", " << partition_factor;
    oss << ">;\n\n";

    return oss.str();
  }

  std::string generate_type_alias(const std::string &name,
                                  const std::string &dtype,
                                  const json &module) const override {
    std::ostringstream oss;

    std::string opt_level = module.value("opt_level", "OPT_NONE");

    std::string config_type =
        (opt_level == "OPT_NONE") ? "void" : (name + "_cfg");

    oss << "using " << name << "_t = vhn::Linear<" << dtype << ", " << name
        << "_hparams, " << config_type << ", " << opt_level << ">;\n";

    return oss.str();
  }
};

} // namespace vhn
#endif
