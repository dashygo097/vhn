#pragma once

#ifndef __VITIS_HLS__
#include "../builder/builder.hh"
#include <sstream>

namespace vhn {

class BatchNorm2dBuilder : public BaseBuilder {
public:
  std::string generate_hparams(const std::string &name,
                               const std::string &dtype,
                               const json &hparams) const override {
    std::ostringstream oss;
    NECESSARY_HPARAMS("BatchNorm2d", name, "channels")
    NECESSARY_HPARAMS("BatchNorm2d", name, "width")
    NECESSARY_HPARAMS("BatchNorm2d", name, "height")

    auto channels = hparams["channels"].get<int>();
    auto width = hparams["width"].get<int>();
    auto height = hparams["height"].get<int>();

    oss << "using " << name << "_hparams = vhn::BatchNorm2dHParams<";
    oss << channels << ", " << width << ", " << height;
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

    oss << "using " << name << "_cfg = vhn::BatchNorm2dConfig<";
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

    GENERATE_TYPE_ALIAS(oss, "BatchNorm2d", name, dtype, opt_level)
    return oss.str();
  }
};

} // namespace vhn

#endif
