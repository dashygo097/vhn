#include <iostream>
#include <vhn.hh>
void print_usage(const char *prog_name) {
  std::cout << "Usage: " << prog_name << " <input.json> <output.hh>\n";
  std::cout << "Example: " << prog_name
            << " configs/network.json generated_config.hh\n";
}

int main(int argc, char **argv) {
  if (argc != 3) {
    print_usage(argv[0]);
    return 1;
  }

  try {
    vhn::ModelConfigGenerator::generate_from_json(argv[1], argv[2]);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "✗ Error: " << e.what() << "\n";
    return 1;
  }
}
