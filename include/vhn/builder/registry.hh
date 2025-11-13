#pragma once

#ifndef __VITIS_HLS__
#include "./codegen.hh"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace vhn {
using json = nlohmann::json;
class LayerRegistry {
private:
  std::map<std::string, std::shared_ptr<UniversalCodeGen>> generators_;

  LayerRegistry() = default;

public:
  static LayerRegistry &instance() {
    static LayerRegistry registry;
    return registry;
  }

  void register_layer(const std::string &type,
                      std::shared_ptr<UniversalCodeGen> generator) {
    generators_[type] = generator;
  }

  std::shared_ptr<UniversalCodeGen>
  get_generator(const std::string &type) const {
    auto it = generators_.find(type);
    if (it != generators_.end()) {
      return it->second;
    }
    return nullptr;
  }

  bool has_layer(const std::string &type) const {
    return generators_.find(type) != generators_.end();
  }

  std::vector<std::string> get_registered_types() const {
    std::vector<std::string> types;
    for (const auto &pair : generators_) {
      types.push_back(pair.first);
    }
    return types;
  }
};

#define REGISTER_LAYER_CODEGEN(TYPE, CLASS)                                    \
  namespace {                                                                  \
  struct CLASS##Registrar {                                                    \
    CLASS##Registrar() {                                                       \
      vhn::LayerRegistry::instance().register_layer(                           \
          TYPE, std::make_shared<CLASS>());                                    \
    }                                                                          \
  };                                                                           \
  static CLASS##Registrar global_##CLASS##Registrar;                           \
  }

} // namespace vhn
#endif // __VITIS_HLS__
