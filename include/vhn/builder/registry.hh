#pragma once

#ifndef __VITIS_HLS__
#include "./base.hh"
#include <map>
#include <memory>
#include <string>
#include <vector>

#define REGISTER_LAYER_BUILDER(TYPE, CLASS)                                    \
  namespace {                                                                  \
  struct CLASS##Registrar {                                                    \
    CLASS##Registrar() {                                                       \
      vhn::LayerRegistry::instance().register_layer(                           \
          TYPE, std::make_shared<vhn::CLASS>());                               \
    }                                                                          \
  };                                                                           \
  static CLASS##Registrar global_##CLASS##Registrar;                           \
  }

namespace vhn {

class LayerRegistry {
private:
  std::map<std::string, std::shared_ptr<BaseBuilder>> generators_;

  LayerRegistry() = default;

public:
  static LayerRegistry &instance() {
    static LayerRegistry registry;
    return registry;
  }

  void register_layer(const std::string &type,
                      std::shared_ptr<BaseBuilder> generator) {
    generators_[type] = generator;
  }

  std::shared_ptr<BaseBuilder> get_generator(const std::string &type) const {
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

} // namespace vhn
#endif
