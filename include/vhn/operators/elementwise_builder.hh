#pragma once

#ifndef __VITIS_HLS__
#include "../builder/builder.hh"
#include <sstream>

namespace vhn {

class ElementwiseBuilder : public BaseBuilder {
public:
  std::string generate_hparams(const std::string &name,
                               const std::string &dtype,
                               const json &hparams) const override {
    std::ostringstream oss;
    NECESSARY_HPARAMS("Elementwise", name, "n")
    NECESSARY_HPARAMS("Elementwise", name, "op")

    int n = hparams["n"].get<int>();
    std::string op = hparams["op"].get<std::string>();

    std::string impl_class;

    if (op == "relu") {
      impl_class = "vhn::ReLUImpl";
    } else if (op == "sigmoid") {
      impl_class = "vhn::SigmoidImpl";
    } else if (op == "gelu") {
      impl_class = "vhn::GeLUImpl";
    } else {
      throw std::runtime_error("Unsupported elementwise operation: " + op);
    }

    oss << "using " << name << "_hparams = vhn::ElementwiseHParams<"
        << impl_class << "<" << dtype << ", " << n << ">, " << n << ">;\n";

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

    oss << "using " << name << "_cfg = vhn::ElementwiseConfig<";
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

    GENERATE_TYPE_ALIAS(oss, "Elementwise", name, dtype, opt_level)
    return oss.str();
  }
};

} // namespace vhn
#endif
