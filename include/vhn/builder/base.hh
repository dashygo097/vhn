#pragma once

#ifndef __VITIS_HLS__
#include <nlohmann/json.hpp>
#include <string>
#endif

#define NECESSARY_HPARAMS(type, name, hparam)                                  \
  if (!hparams.contains(hparam)) {                                             \
    throw std::runtime_error((std::string)type + " module '" + name +          \
                             "' missing " + hparam + " param");                \
  }

#ifndef __VITIS_HLS__
namespace vhn {
using json = nlohmann::json;

class BaseBuilder {
public:
  virtual ~BaseBuilder() = default;

  virtual std::string generate_hparams(const std::string &name,
                                       const std::string &dtype,
                                       const json &hparams) const = 0;

  virtual std::string generate_config(const std::string &name,
                                      const json &hls_cfg) const = 0;

  virtual std::string generate_type_alias(const std::string &name,
                                          const std::string &dtype,
                                          const json &module) const = 0;

  virtual bool has_submodules(const json &module) const {
    return module.contains("submodules") && module["submodules"].is_array() &&
           !module["submodules"].empty();
  }
};

} // namespace vhn
#endif
