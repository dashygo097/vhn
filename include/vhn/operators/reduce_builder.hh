#pragma once

#ifndef __VITIS_HLS__
#include "../builder/builder.hh"
#include <sstream>

namespace vhn {

class ReduceBuilder : public BaseBuilder {
public:
  std::string generate_hparams(const std::string &name,
                               const std::string &dtype,
                               const json &hparams) const override {
    std::ostringstream oss;
    NECESSARY_HPARAMS("Reduce", name, "n")
    NECESSARY_HPARAMS("Reduce", name, "op")

    int n = hparams["n"].get<int>();
    std::string op = hparams["op"].get<std::string>();

    std::string impl_class;

    if (op == "sum") {
      impl_class = "vhn::SumImpl";
    } else if (op == "mean") {
      impl_class = "vhn::MeanImpl";
    } else if (op == "max") {
      impl_class = "vhn::MaxImpl";
    } else if (op == "min") {
      impl_class = "vhn::MinImpl";
    } else {
      throw std::runtime_error("Unsupported reduce operation: " + op);
    }

    oss << "using " << name << "_hparams = vhn::ReduceHParams<" << impl_class
        << "<" << dtype << ", " << n << ">, " << n << ">;\n";

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
    auto use_tree = hls_cfg.value("use_tree", true);

    oss << "using " << name << "_cfg = vhn::ReduceConfig<";
    oss << pipeline_ii << ", " << unroll_factor << ", " << partition_factor
        << ", " << (use_tree ? "true" : "false");
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

    GENERATE_TYPE_ALIAS(oss, "Reduce", name, dtype, opt_level)
    return oss.str();
  }
};

} // namespace vhn
#endif
