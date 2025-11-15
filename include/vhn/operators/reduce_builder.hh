#pragma once

#ifndef __VITIS_HLS__
#include "../builder/builder.hh"
#include <sstream>

namespace vhn {

class ReduceBuilder : public BaseBuilder {
public:
  std::string generate_hparams(const std::string &name,
                               const std::string &dtype,
                               const json &module) const override {
    std::ostringstream oss;

    if (!module.contains("hparams")) {
      throw std::runtime_error("Reduce module '" + name + "' missing hparams");
    }

    auto hparams = module["hparams"];

    if (!hparams.contains("n")) {
      throw std::runtime_error("Reduce module '" + name +
                               "' missing 'n' parameter");
    }

    if (!hparams.contains("op")) {
      throw std::runtime_error("Reduce module '" + name +
                               "' missing 'op' parameter");
    }

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
          oss << "  static constexpr bool _" << it.key() << " = "
              << (it.value().get<bool>() ? "true" : "false") << ";\n";
        } else if (it.value().is_number_integer()) {
          oss << "  static constexpr int _" << it.key() << " = "
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

    oss << "using " << name << "_t = vhn::Reduce<" << dtype << ", " << name
        << "_hparams, " << config_type << ", " << opt_level << ">;\n";

    return oss.str();
  }
};

} // namespace vhn
#endif
