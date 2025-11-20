#pragma once

#ifndef __VITIS_HLS__
#include "../../../builder/builder.hh"
#include "../../../layers/linear_builder.hh"
#include "../../../operators/elementwise_builder.hh"
#include <sstream>

namespace vhn {

class FFNBuilder : public BaseBuilder {
public:
  std::string generate_hparams(const std::string &name,
                               const std::string &dtype,
                               const json &hparams) const override {
    std::ostringstream oss;
    NECESSARY_HPARAMS("FFN", name, "d_model")
    NECESSARY_HPARAMS("FFN", name, "d_ff")
    NECESSARY_HPARAMS("FFN", name, "max_seq_len")
    NECESSARY_HPARAMS("FFN", name, "act")

    auto d_model = hparams["d_model"].get<int>();
    auto d_ff = hparams["d_ff"].get<int>();
    auto act = hparams["act"].get<std::string>();
    auto max_seq_len = hparams["max_seq_len"].get<int>();

    json fc1_hparams = {{"in_features", d_model}, {"out_features", d_ff}};
    json act_hparams = {{"op", act}, {"n", d_ff}};
    json fc2_hparams = {{"in_features", d_ff}, {"out_features", d_model}};

    LinearBuilder linear_builder;
    ElementwiseBuilder elementwise_builder;

    oss << linear_builder.generate_hparams(name + "_fc1", dtype, fc1_hparams);
    oss << elementwise_builder.generate_hparams(name + "_act", dtype,
                                                act_hparams);
    oss << linear_builder.generate_hparams(name + "_fc2", dtype, fc2_hparams);

    oss << "using " << name << "_hparams = vhn::FFNHParams<";
    oss << name << "_fc1_hparams, ";
    oss << name << "_act_hparams, ";
    oss << name << "_fc2_hparams, ";
    oss << max_seq_len;
    oss << ">;\n\n";

    return oss.str();
  }

  std::string generate_config(const std::string &name,
                              const json &hls_cfg) const override {
    if (hls_cfg.empty() || hls_cfg.is_null()) {
      return "";
    }
    std::ostringstream oss;

    auto fc1_cfg = hls_cfg.value("fc1", json::object());
    auto act_cfg = hls_cfg.value("act", json::object());
    auto fc2_cfg = hls_cfg.value("fc2", json::object());

    auto dataflow_depth = hls_cfg.value("dataflow_depth", 16);
    auto seq_unroll = hls_cfg.value("seq_unroll", 1);
    auto memory_partition = hls_cfg.value("memory_partition", 4);

    LinearBuilder linear_builder;
    ElementwiseBuilder elementwise_builder;

    if (hls_cfg.contains("fc1"))
      oss << linear_builder.generate_config(name + "_fc1", fc1_cfg);
    if (hls_cfg.contains("act"))
      oss << elementwise_builder.generate_config(name + "_act", act_cfg);
    if (hls_cfg.contains("fc2"))
      oss << linear_builder.generate_config(name + "_fc2", fc2_cfg);

    oss << "using " << name << "_cfg = vhn::FFNConfig<";
    if (hls_cfg.contains("fc1"))
      oss << "  " << name << "_fc1_cfg, ";
    else
      oss << "void, ";
    if (hls_cfg.contains("act"))
      oss << name << "_act_cfg, ";
    else
      oss << "void, ";
    if (hls_cfg.contains("fc2"))
      oss << name << "_fc2_cfg, ";
    else
      oss << "void, ";
    oss << dataflow_depth << ", ";
    oss << seq_unroll << ", ";
    oss << memory_partition;
    oss << ">;\n\n";

    return oss.str();
  }

  std::string generate_type_alias(const std::string &name,
                                  const std::string &dtype,
                                  const json &module) const override {
    std::ostringstream oss;

    std::string opt_level = module.value("opt_level", "OPT_NONE");

    if (!module.contains("hparams")) {
      throw std::runtime_error("FFN module '" + name + "' missing hparams");
    }

    auto hparams = module["hparams"];
    auto d_model = hparams["d_model"].get<int>();
    auto d_ff = hparams["d_ff"].get<int>();
    auto max_seq_len = hparams["max_seq_len"].get<int>();
    auto act = hparams["act"].get<std::string>();

    json fc1_module = {
        {"hparams", {{"in_features", d_model}, {"out_features", d_ff}}},
        {"opt_level", opt_level}};

    json act_module = {{"hparams", {{"op", act}, {"n", d_ff}}},
                       {"opt_level", opt_level}};

    json fc2_module = {
        {"hparams", {{"in_features", d_ff}, {"out_features", d_model}}},
        {"opt_level", opt_level}};

    if (module.contains("hls_cfg")) {
      fc1_module["hls_cfg"] = module["hls_cfg"];
      act_module["hls_cfg"] = module["hls_cfg"];
      fc2_module["hls_cfg"] = module["hls_cfg"];
    }

    LinearBuilder linear_builder;
    ElementwiseBuilder elementwise_builder;

    oss << linear_builder.generate_type_alias(name + "_fc1", dtype, fc1_module);
    oss << elementwise_builder.generate_type_alias(name + "_act", dtype,
                                                   act_module);
    oss << linear_builder.generate_type_alias(name + "_fc2", dtype, fc2_module);

    std::string config_type =
        (opt_level == "OPT_NONE") ? "void" : (name + "_cfg");

    oss << "using " << name << "_t = vhn::FFN<" << dtype << ", " << name
        << "_hparams, " << config_type << ", " << opt_level << ">;\n";

    return oss.str();
  }
};

} // namespace vhn

#endif
