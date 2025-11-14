#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <vhn.hh>

using json = nlohmann::json;

void print_module_tree(const json &module, int depth = 0) {
  std::string indent(depth * 2, ' ');

  std::cout << indent << "└─ " << module.value("type", "Unknown") << " ["
            << module.value("opt_level", "OPT_NONE") << "]";

  if (module.contains("hparams") && !module["hparams"].empty()) {
    std::cout << " (";
    bool first = true;
    for (auto it = module["hparams"].begin(); it != module["hparams"].end();
         ++it) {
      if (!first)
        std::cout << ", ";
      std::cout << it.key() << "=" << it.value();
      first = false;
    }
    std::cout << ")";
  }
  std::cout << "\n";

  if (module.contains("submodules") && module["submodules"].is_array()) {
    for (const auto &submodule : module["submodules"]) {
      print_module_tree(submodule, depth + 1);
    }
  }
}

int count_modules(const json &module) {
  int count = 1;
  if (module.contains("submodules") && module["submodules"].is_array()) {
    for (const auto &submodule : module["submodules"]) {
      count += count_modules(submodule);
    }
  }
  return count;
}

void print_usage(const char *prog_name) {
  std::cout << "Usage: " << prog_name << " <command> [options]\n\n";
  std::cout << "Commands:\n";
  std::cout << "  inspect <config.json>              - Display module tree "
               "structure\n";
  std::cout << "  generate <config.json> <output.hh> - Generate C++ header "
               "from config\n";
  std::cout
      << "  validate <config.json>             - Validate config file\n\n";
  std::cout << "Examples:\n";
  std::cout << "  " << prog_name << " inspect network_config.json\n";
  std::cout << "  " << prog_name
            << " generate network_config.json generated_config.hh\n";
}

int main(int argc, char **argv) {
  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }

  std::string command = argv[1];

  try {
    if (command == "inspect") {
      if (argc != 3) {
        std::cerr << "Error: inspect requires config file path\n";
        return 1;
      }

      std::string config_path = argv[2];
      std::ifstream file(config_path);
      if (!file.is_open()) {
        std::cerr << "Error: Cannot open " << config_path << "\n";
        return 1;
      }

      json config = json::parse(file);

      std::cout << "=== Network Configuration ===\n";
      std::cout << "Name: " << config["model"].value("name", "Unknown") << "\n";
      std::cout << "DType: " << config["model"].value("dtype", "float")
                << "\n\n";

      if (config.contains("modules") && config["modules"].is_array()) {
        std::cout << "=== Module Tree ===\n";
        for (const auto &module : config["modules"]) {
          print_module_tree(module);
          std::cout << "\nTotal modules in tree: " << count_modules(module)
                    << "\n";
        }
      }

    } else if (command == "generate") {
      if (argc != 4) {
        std::cerr << "Error: generate requires <input.json> <output.hh>\n";
        return 1;
      }

      std::string input_path = argv[2];
      std::string output_path = argv[3];

      std::cout << "Generating C++ configuration...\n";
      vhn::ModelConfigGenerator::generate_from_json(input_path, output_path);

    } else if (command == "validate") {
      if (argc != 3) {
        std::cerr << "Error: validate requires config file path\n";
        return 1;
      }

      std::string config_path = argv[2];
      std::ifstream file(config_path);
      if (!file.is_open()) {
        std::cerr << "Error: Cannot open " << config_path << "\n";
        return 1;
      }

      json config = json::parse(file);

      if (!config.contains("model")) {
        std::cerr << "✗ Missing 'model' section\n";
        return 1;
      }

      if (!config.contains("modules")) {
        std::cerr << "✗ Missing 'modules' section\n";
        return 1;
      }

      if (!config["modules"].is_array() || config["modules"].empty()) {
        std::cerr << "✗ 'modules' must be a non-empty array\n";
        return 1;
      }

      std::cout << "✓ Configuration is valid\n";
      std::cout << "  Model: " << config["model"].value("name", "Unknown")
                << "\n";
      std::cout << "  Modules: " << config["modules"].size() << "\n";

    } else {
      std::cerr << "Error: Unknown command '" << command << "'\n\n";
      print_usage(argv[0]);
      return 1;
    }

  } catch (const json::exception &e) {
    std::cerr << "JSON Error: " << e.what() << "\n";
    return 1;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
