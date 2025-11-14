#include <vhn.hh>

using MyMLPHParams = vhn::MLPHParams<784, 256, 128, 10>;
using MyMLP = vhn::MLP<float, MyMLPHParams, void, OPT_NONE>;

struct fc1_config {
  static constexpr int unroll_factor = 2;
};

struct MLPConfig {
  using layer_0 = fc1_config;
};

using MyMLPOpt = vhn::MLP<float, MyMLPHParams, MLPConfig, OPT_ENABLED>;

int main() { MyMLP::dump_config("mlp_config.json"); }
