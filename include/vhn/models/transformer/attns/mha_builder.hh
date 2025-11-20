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
                               const json &hparams) const override {
    std::ostringstream oss;
    NECESSARY_HPARAMS("MulHeadAttn", name, "d_model")
    NECESSARY_HPARAMS("MulHeadAttn", name, "num_heads")
    NECESSARY_HPARAMS("MulHeadAttn", name, "max_seq_len")

    auto d_model = hparams["d_model"].get<int>();
    auto num_heads = hparams["num_heads"].get<int>();
    auto max_seq_len = hparams["max_seq_len"].get<int>();
    auto head_dim = d_model / num_heads;

    json wqkv_hparams = {{"in_features", d_model},
                         {"out_features", 3 * d_model}};
    json softmax_hparams = {{"n", max_seq_len}};
    json wo_hparams = {{"in_features", d_model}, {"out_features", d_model}};

    LinearBuilder linear_builder;
    SoftmaxBuilder softmax_builder;

    oss << linear_builder.generate_hparams(name + "_wqkv", dtype, wqkv_hparams);
    oss << softmax_builder.generate_hparams(name + "_softmax", dtype,
                                            softmax_hparams);
    oss << linear_builder.generate_hparams(name + "_wo", dtype, wo_hparams);

    oss << "using " << name << "_hparams = vhn::MulHeadAttnHParams<";
    oss << name << "_wqkv_hparams, ";
    oss << name << "_softmax_hparams, ";
    oss << name << "_wo_hparams, ";
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

    auto wqkv_cfg = hls_cfg.value("wqkv", json::object());
    auto softmax_cfg = hls_cfg.value("softmax", json::object());
    auto wo_cfg = hls_cfg.value("wo", json::object());

    auto dataflow_enabled = hls_cfg.value("dataflow_enabled", true);
    auto pipeline_ii = hls_cfg.value("pipeline_ii", 1);
    auto qkv_partition_factor = hls_cfg.value("partition_factor", 4);
    auto attn_partition_factor = hls_cfg.value("partition_factor", 4);
    auto attn_unroll_factor = hls_cfg.value("unroll_factor", 4);
    auto head_unroll_factor = hls_cfg.value("unroll_factor", 4);

    LinearBuilder linear_builder;
    SoftmaxBuilder softmax_builder;

    if (hls_cfg.contains("wqkv"))
      oss << linear_builder.generate_config(name + "_wqkv", wqkv_cfg);
    if (hls_cfg.contains("softmax"))
      oss << softmax_builder.generate_config(name + "_softmax", softmax_cfg);
    if (hls_cfg.contains("wo"))
      oss << linear_builder.generate_config(name + "_wo", wo_cfg);

    oss << "using " << name << "_cfg = vhn::MulHeadAttnConfig<";
    if (hls_cfg.contains("wqkv"))
      oss << name << "_wqkv_cfg, ";
    else
      oss << "void, ";
    if (hls_cfg.contains("softmax"))
      oss << name << "_softmax_cfg, ";
    else
      oss << "void, ";
    if (hls_cfg.contains("wo"))
      oss << name << "_wo_cfg, ";
    else
      oss << "void, ";
    oss << dataflow_enabled << ", ";
    oss << pipeline_ii << ", ";
    oss << qkv_partition_factor << ", ";
    oss << attn_partition_factor << ", ";
    oss << attn_unroll_factor << ", ";
    oss << head_unroll_factor << ">;\n\n";

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
