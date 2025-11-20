#pragma once

#ifndef __VITIS_HLS__
#include "../builder/builder.hh"
#include <sstream>

namespace vhn {

class Conv1dBuilder : public BaseBuilder {
public:
  std::string generate_hparams(const std::string &name,
                               const std::string &dtype,
                               const json &hparams) const override {
    std::ostringstream oss;

    NECESSARY_HPARAMS("Conv1d", name, "in_channels")
    NECESSARY_HPARAMS("Conv1d", name, "out_channels")
    NECESSARY_HPARAMS("Conv1d", name, "kernel_size")
    NECESSARY_HPARAMS("Conv1d", name, "padding")
    NECESSARY_HPARAMS("Conv1d", name, "n")

    auto in_channels = hparams["in_channels"].get<int>();
    auto out_channels = hparams["out_channels"].get<int>();
    auto kernel_size = hparams["kernel_size"].get<int>();
    auto padding = hparams["padding"].get<int>();
    auto n = hparams["n"].get<int>();

    oss << "using " << name << "_hparams = vhn::Conv1dHParams<";
    oss << in_channels << ", " << out_channels << ", " << kernel_size << ", "
        << padding << ", " << n;
    oss << ">;\n\n";

    return oss.str();
  }

  std::string generate_config(const std::string &name,
                              const json &hls_cfg) const override {
    if (hls_cfg.empty() || hls_cfg.is_null()) {
      return "";
    }

    std::ostringstream oss;

    auto dataflow_enabled = hls_cfg.value("dataflow_enabled", true);
    auto pipeline_ii = hls_cfg.value("pipeline_ii", 1);
    auto unroll_factor = hls_cfg.value("unroll_factor", 1);
    auto partition_factor = hls_cfg.value("partition_factor", 4);
    auto kernel_unroll = hls_cfg.value("kernel_unroll", 1);
    auto ic_unroll = hls_cfg.value("ic_unroll", 1);

    oss << "using " << name << "_cfg = vhn::Conv1dConfig<";
    oss << (dataflow_enabled ? "true" : "false") << ", " << pipeline_ii << ", "
        << unroll_factor << ", " << partition_factor << ", " << kernel_unroll
        << ", " << ic_unroll;
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

    GENERATE_TYPE_ALIAS(oss, "Conv1d", name, dtype, opt_level)
    return oss.str();
  }
};

} // namespace vhn
#endif
