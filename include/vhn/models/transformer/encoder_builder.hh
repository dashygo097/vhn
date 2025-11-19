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

    if (!hparams.contains("d_model")) {
      throw std::runtime_error("EncoderBlock module '" + name +
                               "' missing d_model param");
    }

    if (!hparams.contains("num_heads")) {
      throw std::runtime_error("EncoderBlock module '" + name +
                               "' missing num_heads param");
    }

    if (!hparams.contains("d_ff")) {
      throw std::runtime_error("EncoderBlock module '" + name +
                               "' missing d_ff param");
    }

    if (!hparams.contains("max_seq_len")) {
      throw std::runtime_error("EncoderBlock module '" + name +
                               "' missing max_seq_len param");
    }

    if (!hparams.contains("norm_type")) {
      throw std::runtime_error("EncoderBlock module '" + name +
                               "' missing norm_type param");
    }

    if (!hparams.contains("act")) {
      throw std::runtime_error("EncoderBlock module '" + name +
                               "' missing act param");
    }

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

    return oss.str();
  }

  std::string generate_type_alias(const std::string &name,
                                  const std::string &dtype,
                                  const json &module) const override {
    std::ostringstream oss;

    std::string opt_level = module.value("opt_level", "OPT_NONE");

    if (!module.contains("hparams")) {
      throw std::runtime_error("EncoderBlock module '" + name +
                               "' missing hparams");
    }

    auto hparams = module["hparams"];
    auto d_model = hparams["d_model"].get<int>();
    auto num_heads = hparams["num_heads"].get<int>();
    auto d_ff = hparams["d_ff"].get<int>();
    auto max_seq_len = hparams["max_seq_len"].get<int>();
    auto norm_type_str = hparams["norm_type"].get<std::string>();
    auto act_str = hparams["act"].get<std::string>();

    json mha_module = {{"hparams",
                        {{"d_model", d_model},
                         {"num_heads", num_heads},
                         {"max_seq_len", max_seq_len}}},
                       {"opt_level", opt_level}};

    json addnorm1_module = {
        {"hparams", {{"d_model", d_model}, {"norm_type", norm_type_str}}},
        {"opt_level", opt_level}};

    json ffn_module = {{"hparams",
                        {{"d_model", d_model},
                         {"d_ff", d_ff},
                         {"act", act_str},
                         {"max_seq_len", max_seq_len}}},
                       {"opt_level", opt_level}};

    json addnorm2_module = {
        {"hparams", {{"d_model", d_model}, {"norm_type", norm_type_str}}},
        {"opt_level", opt_level}};

    if (module.contains("hls_cfg")) {
      mha_module["hls_cfg"] = module["hls_cfg"];
      addnorm1_module["hls_cfg"] = module["hls_cfg"];
      ffn_module["hls_cfg"] = module["hls_cfg"];
      addnorm2_module["hls_cfg"] = module["hls_cfg"];
    }

    MulHeadAttnBuilder mha_builder;
    AddNormBuilder addnorm_builder;
    FFNBuilder ffn_builder;

    oss << mha_builder.generate_type_alias(name + "_mha", dtype, mha_module);
    oss << addnorm_builder.generate_type_alias(name + "_addnorm1", dtype,
                                               addnorm1_module);
    oss << ffn_builder.generate_type_alias(name + "_ffn", dtype, ffn_module);
    oss << addnorm_builder.generate_type_alias(name + "_addnorm2", dtype,
                                               addnorm2_module);

    std::string config_type =
        (opt_level == "OPT_NONE") ? "void" : (name + "_cfg");

    oss << "using " << name << "_t = vhn::EncoderBlock<" << dtype << ", "
        << name << "_hparams, " << config_type << ", " << opt_level << ">;\n";

    return oss.str();
  }
};

} // namespace vhn

#endif
