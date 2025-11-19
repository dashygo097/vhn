#pragma once

#ifndef __VITIS_HLS__
#include "../../../builder/builder.hh"
#include "../../../layers/linear_builder.hh"
#include "../../../layers/softmax_builder.hh"
#include <sstream>

namespace vhn {

class MulHeadAttnBuilder : public BaseBuilder {
public:
  std::string generate_hparams(const std::string &name,
                               const std::string &dtype,
                               const json &module) const override {
    std::ostringstream oss;

    if (!module.contains("hparams")) {
      throw std::runtime_error("MulHeadAttn module '" + name +
                               "' missing hparams");
    }

    auto hparams = module["hparams"];

    if (!hparams.contains("d_model")) {
      throw std::runtime_error("MulHeadAttn module '" + name +
                               "' missing d_model param");
    }

    if (!hparams.contains("num_heads")) {
      throw std::runtime_error("MulHeadAttn module '" + name +
                               "' missing num_heads param");
    }

    if (!hparams.contains("max_seq_len")) {
      throw std::runtime_error("MulHeadAttn module '" + name +
                               "' missing max_seq_len param");
    }

    auto d_model = hparams["d_model"].get<int>();
    auto num_heads = hparams["num_heads"].get<int>();
    auto max_seq_len = hparams["max_seq_len"].get<int>();
    auto head_dim = d_model / num_heads;

    json wqkv_module = {
        {"hparams", {{"in_features", d_model}, {"out_features", 3 * d_model}}},
        {"opt_level", module.value("opt_level", "OPT_NONE")}};

    json softmax_module = {
        {"hparams", {{"n", max_seq_len}}},
        {"opt_level", module.value("opt_level", "OPT_NONE")}};

    json wo_module = {
        {"hparams", {{"in_features", d_model}, {"out_features", d_model}}},
        {"opt_level", module.value("opt_level", "OPT_NONE")}};

    if (module.contains("hls_cfg")) {
      wqkv_module["hls_cfg"] = module["hls_cfg"];
      softmax_module["hls_cfg"] = module["hls_cfg"];
      wo_module["hls_cfg"] = module["hls_cfg"];
    }

    LinearBuilder linear_builder;
    SoftmaxBuilder softmax_builder;

    oss << linear_builder.generate_hparams(name + "_wqkv", dtype, wqkv_module);
    oss << softmax_builder.generate_hparams(name + "_softmax", dtype,
                                            softmax_module);
    oss << linear_builder.generate_hparams(name + "_wo", dtype, wo_module);

    oss << "using " << name << "_hparams = vhn::MulHeadAttnHParams<";
    oss << name << "_wqkv_hparams, ";
    oss << name << "_softmax_hparams, ";
    oss << name << "_wo_hparams, ";
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
      throw std::runtime_error("MulHeadAttn module '" + name +
                               "' missing hparams");
    }

    auto hparams = module["hparams"];
    auto d_model = hparams["d_model"].get<int>();
    auto num_heads = hparams["num_heads"].get<int>();
    auto max_seq_len = hparams["max_seq_len"].get<int>();

    json wqkv_module = {
        {"hparams", {{"in_features", d_model}, {"out_features", 3 * d_model}}},
        {"opt_level", opt_level}};

    json softmax_module = {{"hparams", {{"n", max_seq_len}}},
                           {"opt_level", opt_level}};

    json wo_module = {
        {"hparams", {{"in_features", d_model}, {"out_features", d_model}}},
        {"opt_level", opt_level}};

    if (module.contains("hls_cfg")) {
      wqkv_module["hls_cfg"] = module["hls_cfg"];
      softmax_module["hls_cfg"] = module["hls_cfg"];
      wo_module["hls_cfg"] = module["hls_cfg"];
    }

    LinearBuilder linear_builder;
    SoftmaxBuilder softmax_builder;

    oss << linear_builder.generate_config(name + "_wqkv", wqkv_module);
    oss << softmax_builder.generate_config(name + "_softmax", softmax_module);
    oss << linear_builder.generate_config(name + "_wo", wo_module);

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
      throw std::runtime_error("MulHeadAttn module '" + name +
                               "' missing hparams");
    }

    auto hparams = module["hparams"];
    auto d_model = hparams["d_model"].get<int>();
    auto num_heads = hparams["num_heads"].get<int>();
    auto max_seq_len = hparams["max_seq_len"].get<int>();

    json wqkv_module = {
        {"hparams", {{"in_features", d_model}, {"out_features", 3 * d_model}}},
        {"opt_level", opt_level}};

    json softmax_module = {{"hparams", {{"n", max_seq_len}}},
                           {"opt_level", opt_level}};

    json wo_module = {
        {"hparams", {{"in_features", d_model}, {"out_features", d_model}}},
        {"opt_level", opt_level}};

    if (module.contains("hls_cfg")) {
      wqkv_module["hls_cfg"] = module["hls_cfg"];
      softmax_module["hls_cfg"] = module["hls_cfg"];
      wo_module["hls_cfg"] = module["hls_cfg"];
    }

    LinearBuilder linear_builder;
    SoftmaxBuilder softmax_builder;

    oss << linear_builder.generate_type_alias(name + "_wqkv", dtype,
                                              wqkv_module);
    oss << softmax_builder.generate_type_alias(name + "_softmax", dtype,
                                               softmax_module);
    oss << linear_builder.generate_type_alias(name + "_wo", dtype, wo_module);

    std::string config_type =
        (opt_level == "OPT_NONE") ? "void" : (name + "_cfg");

    oss << "using " << name << "_t = vhn::MulHeadAttn<" << dtype << ", " << name
        << "_hparams, " << config_type << ", " << opt_level << ">;\n";

    return oss.str();
  }
};

} // namespace vhn

#endif
