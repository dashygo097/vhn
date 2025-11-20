#pragma once

#ifndef __VITIS_HLS__
#include "../../builder/builder.hh"
#include "./attns/mha_builder.hh"
#include "./components/addnorm_builder.hh"
#include "./components/ffn_builder.hh"
#include <sstream>

namespace vhn {

class EncoderBlockBuilder : public BaseBuilder {
public:
  std::string generate_hparams(const std::string &name,
                               const std::string &dtype,
                               const json &hparams) const override {
    std::ostringstream oss;
    NECESSARY_HPARAMS("EncoderBlock", name, "d_model")
    NECESSARY_HPARAMS("EncoderBlock", name, "num_heads")
    NECESSARY_HPARAMS("EncoderBlock", name, "d_ff")
    NECESSARY_HPARAMS("EncoderBlock", name, "max_seq_len")
    NECESSARY_HPARAMS("EncoderBlock", name, "norm_type")
    NECESSARY_HPARAMS("EncoderBlock", name, "act")

    auto d_model = hparams["d_model"].get<int>();
    auto num_heads = hparams["num_heads"].get<int>();
    auto d_ff = hparams["d_ff"].get<int>();
    auto max_seq_len = hparams["max_seq_len"].get<int>();
    auto norm_type_str = hparams["norm_type"].get<std::string>();
    auto act_str = hparams["act"].get<std::string>();

    json mha_hparams = {{"d_model", d_model},
                        {"num_heads", num_heads},
                        {"max_seq_len", max_seq_len}};
    json addnorm1_hparams = {{"d_model", d_model},
                             {"norm_type", norm_type_str}};
    json ffn_hparams = {{"d_model", d_model},
                        {"d_ff", d_ff},
                        {"act", act_str},
                        {"max_seq_len", max_seq_len}};
    json addnorm2_hparams = {{"d_model", d_model},
                             {"norm_type", norm_type_str}};

    MulHeadAttnBuilder mha_builder;
    AddNormBuilder addnorm_builder;
    FFNBuilder ffn_builder;

    oss << mha_builder.generate_hparams(name + "_mha", dtype, mha_hparams);
    oss << addnorm_builder.generate_hparams(name + "_addnorm1", dtype,
                                            addnorm1_hparams);
    oss << ffn_builder.generate_hparams(name + "_ffn", dtype, ffn_hparams);
    oss << addnorm_builder.generate_hparams(name + "_addnorm2", dtype,
                                            addnorm2_hparams);

    oss << "using " << name << "_hparams = vhn::EncoderBlockHParams<";
    oss << name << "_mha_hparams, ";
    oss << name << "_addnorm1_hparams, ";
    oss << name << "_ffn_hparams, ";
    oss << name << "_addnorm2_hparams";
    oss << ">;\n\n";

    return oss.str();
  }

  std::string generate_config(const std::string &name,
                              const json &hls_cfg) const override {
    if (hls_cfg.empty() || hls_cfg.is_null()) {
      return "";
    }

    std::ostringstream oss;

    auto mha_cfg = hls_cfg.value("mha", json::object());
    auto addnorm1_cfg = hls_cfg.value("addnorm1", json::object());
    auto ffn_cfg = hls_cfg.value("ffn", json::object());
    auto addnorm2_cfg = hls_cfg.value("addnorm2", json::object());

    auto dataflow_enabled = hls_cfg.value("dataflow_enabled", true);
    auto pipeline_ii = hls_cfg.value("pipeline_enabled", 1);
    auto intermediate_partition = hls_cfg.value("intermediate_partition", 4);

    MulHeadAttnBuilder mha_builder;
    AddNormBuilder addnorm_builder;
    FFNBuilder ffn_builder;

    if (hls_cfg.contains("mha"))
      oss << mha_builder.generate_config(name + "_mha", mha_cfg);
    if (hls_cfg.contains("addnorm1"))
      oss << addnorm_builder.generate_config(name + "_addnorm1", addnorm1_cfg);
    if (hls_cfg.contains("ffn"))
      oss << ffn_builder.generate_config(name + "_ffn", ffn_cfg);
    if (hls_cfg.contains("addnorm2"))
      oss << addnorm_builder.generate_config(name + "_addnorm2", addnorm2_cfg);

    oss << "using " << name << "_cfg = vhn::EncoderBlockConfig<";
    if (hls_cfg.contains("mha"))
      oss << name << "_mha_cfg, ";
    else
      oss << "void, ";
    if (hls_cfg.contains("addnorm1"))
      oss << name << "_addnorm1_cfg, ";
    else
      oss << "void, ";
    if (hls_cfg.contains("ffn"))
      oss << name << "_ffn_cfg, ";
    else
      oss << "void, ";
    if (hls_cfg.contains("addnorm2"))
      oss << name << "_addnorm2_cfg, ";
    else
      oss << "void, ";
    oss << (dataflow_enabled ? "true, " : "false, ");
    oss << pipeline_ii << ", ";
    oss << intermediate_partition << ">;\n\n";

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

    GENERATE_TYPE_ALIAS(oss, "EncoderBlock", name, dtype, opt_level)
    return oss.str();
  }
};

} // namespace vhn

#endif
