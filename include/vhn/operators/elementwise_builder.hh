#pragma once

#ifndef __VITIS_HLS__
#include "../builder/builder.hh"
#include <sstream>

namespace vhn {

class ElementwiseBuilder : public BaseBuilder {
public:
  std::string generate_hparams(const std::string &name,
                               const std::string &dtype,
                               const json &module) const override {
    std::ostringstream oss;

    if (!module.contains("hparams")) {
      throw std::runtime_error("Elementwise module '" + name +
                               "' missing hparams");
    }

    auto hparams = module["hparams"];

    if (!hparams.contains("n")) {
      throw std::runtime_error("Elementwise module '" + name +
                               "' missing 'n' parameter");
    }

    if (!hparams.contains("op")) {
      throw std::runtime_error("Elementwise module '" + name +
                               "' missing 'op' parameter");
    }

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
                              const json &module) const override {
    std::string opt_level = module.value("opt_level", "OPT_NONE");

    if (opt_level == "OPT_NONE") {
      return "";
    }

    std::ostringstream oss;

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
    std::string config_type =
        (opt_level == "OPT_NONE") ? "void" : (name + "_cfg");

    oss << "using " << name << "_t = vhn::Elementwise<" << dtype << ", " << name
        << "_hparams, " << config_type << ", " << opt_level << ">;\n";

    return oss.str();
  }
};

} // namespace vhn
#endif
