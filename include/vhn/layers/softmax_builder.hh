#pragma once

#ifndef __VITIS_HLS__
#include "../builder/builder.hh"
#include <sstream>

namespace vhn {

class SoftmaxBuilder : public BaseBuilder {
public:
  std::string generate_hparams(const std::string &name,
                               const std::string &dtype,
                               const json &hparams) const override {
    std::ostringstream oss;

    if (!hparams.contains("n")) {
      throw std::runtime_error("Softmax module '" + name + "' missing n param");
    }

    auto n = hparams["n"].get<int>();

    oss << "using " << name << "_hparams = vhn::SoftmaxHParams<";
    oss << n;
    oss << ">;\n\n";

    return oss.str();
  }

  std::string generate_config(const std::string &name,
                              const json &hls_cfg) const override {
    if (hls_cfg.empty() || hls_cfg.is_null()) {
      return "";
    }

    std::ostringstream oss;

    int unroll_factor = 1;
    int partition_factor = 1;
    int pipeline_ii = 1;

    if (hls_cfg.contains("unroll_factor")) {
      unroll_factor = hls_cfg["unroll_factor"].get<int>();
    }

    if (hls_cfg.contains("partition_factor")) {
      partition_factor = hls_cfg["partition_factor"].get<int>();
    }

    if (hls_cfg.contains("pipeline_ii")) {
      pipeline_ii = hls_cfg["pipeline_ii"].get<int>();
    }

    oss << "using " << name << "_cfg = vhn::SoftmaxConfig<";
    oss << unroll_factor << ", " << partition_factor << ", " << pipeline_ii;
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

    oss << "using " << name << "_t = vhn::Softmax<" << dtype << ", " << name
        << "_hparams, " << config_type << ", " << opt_level << ">;\n";

    return oss.str();
  }
};

} // namespace vhn

#endif
