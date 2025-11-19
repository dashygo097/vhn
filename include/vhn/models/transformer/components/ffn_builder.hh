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
                               const json &module) const override {
    std::ostringstream oss;

    if (!module.contains("hparams")) {
      throw std::runtime_error("FFN module '" + name + "' missing hparams");
    }

    auto hparams = module["hparams"];

    if (!hparams.contains("d_model")) {
      throw std::runtime_error("FFN module '" + name +
                               "' missing d_model param");
    }

    if (!hparams.contains("d_ff")) {
      throw std::runtime_error("FFN module '" + name + "' missing d_ff param");
    }

    if (!hparams.contains("max_seq_len")) {
      throw std::runtime_error("FFN module '" + name +
                               "' missing max_seq_len param");
    }

    if (!hparams.contains("act")) {
      throw std::runtime_error("FFN module '" + name + "' missing act param");
    }

    auto d_model = hparams["d_model"].get<int>();
    auto d_ff = hparams["d_ff"].get<int>();
    auto act = hparams["act"].get<std::string>();
    auto max_seq_len = hparams["max_seq_len"].get<int>();

    json fc1_module = {
        {"hparams", {{"in_features", d_model}, {"out_features", d_ff}}},
        {"opt_level", module.value("opt_level", "OPT_NONE")}};

    json act_module = {{"hparams", {{"op", act}, {"n", d_ff}}},
                       {"opt_level", module.value("opt_level", "OPT_NONE")}};

    json fc2_module = {
        {"hparams", {{"in_features", d_ff}, {"out_features", d_model}}},
        {"opt_level", module.value("opt_level", "OPT_NONE")}};

    if (module.contains("hls_cfg")) {
      fc1_module["hls_cfg"] = module["hls_cfg"];
      act_module["hls_cfg"] = module["hls_cfg"];
      fc2_module["hls_cfg"] = module["hls_cfg"];
    }

    LinearBuilder linear_builder;
    ElementwiseBuilder elementwise_builder;

    oss << linear_builder.generate_hparams(name + "_fc1", dtype, fc1_module);
    oss << elementwise_builder.generate_hparams(name + "_act", dtype,
                                                act_module);
    oss << linear_builder.generate_hparams(name + "_fc2", dtype, fc2_module);

    oss << "using " << name << "_hparams = vhn::FFNHParams<";
    oss << name << "_fc1_hparams, ";
    oss << name << "_act_hparams, ";
    oss << name << "_fc2_hparams, ";
    oss << max_seq_len;
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

    if (!module.contains("hparams")) {
      throw std::runtime_error("FFN module '" + name + "' missing hparams");
    }

    auto hparams = module["hparams"];
    auto d_model = hparams["d_model"].get<int>();
    auto d_ff = hparams["d_ff"].get<int>();
    auto max_seq_len = hparams["max_seq_len"].get<int>();
    auto act = hparams["act"].get<std::string>();

    // Create submodule JSON configurations
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

    oss << linear_builder.generate_config(name + "_fc1", fc1_module);
    oss << elementwise_builder.generate_config(name + "_act", act_module);
    oss << linear_builder.generate_config(name + "_fc2", fc2_module);

    oss << "struct " << name << "_cfg {\n";

    if (module.contains("hls_cfg") && !module["hls_cfg"].empty()) {
      auto hls_cfg = module["hls_cfg"];

      for (auto it = hls_cfg.begin(); it != hls_cfg.end(); ++it) {
        if (it.value().is_boolean()) {
          oss << "  static constexpr bool " << it.key() << " = "
              << (it.value().get<bool>() ? "true" : "false") << ";\n";
        } else if (it.value().is_number_integer()) {
          oss << "  static constexpr int " << it.key() << " = "
              << it.value().get<int>() << ";\n";
        }
      }
    }

    oss << "};\n\n";

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
