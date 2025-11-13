#include "../include/hls.hh"
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

using namespace hls_nn;
using json = nlohmann::json;

class LinearJsonTest : public ::testing::Test {
protected:
  using DType = float;
  static constexpr int IN_FEATURES = 16;
  static constexpr int OUT_FEATURES = 8;

  void ValidateBasicStructure(const json &j) {
    EXPECT_TRUE(j.contains("module_type"));
    EXPECT_TRUE(j.contains("module_name"));
    EXPECT_TRUE(j.contains("opt_level"));
    EXPECT_TRUE(j.contains("hls_config"));
  }
};

class LinearOPT_NONEJsonTest : public LinearJsonTest {
protected:
  using LinearLayer = Linear<DType, IN_FEATURES, OUT_FEATURES, void, OPT_NONE>;

  LinearLayer layer;
};

struct TestOptConfig {
  static constexpr int _unroll_factor = 4;
  static constexpr int _partition_factor = 2;
  static constexpr int _pipeline_ii = 2;
};

class LinearOPT_ENABLEDJsonTest : public LinearJsonTest {
protected:
  using LinearLayer =
      Linear<DType, IN_FEATURES, OUT_FEATURES, TestOptConfig, OPT_ENABLED>;

  LinearLayer layer;
};

// ============================================================================
// Tests for OPT_NONE Configuration
// ============================================================================
TEST_F(LinearOPT_NONEJsonTest, ToJsonBasicStructure) {
  auto config = layer.to_json();
  ValidateBasicStructure(config);
}

TEST_F(LinearOPT_NONEJsonTest, ToJsonModuleInfo) {
  auto config = layer.to_json();

  EXPECT_EQ(config["module_name"].get<std::string>(), "Linear");
  EXPECT_EQ(config["module_type"].get<std::string>(), "Linear_OPT_NONE");
  EXPECT_EQ(config["opt_level"].get<std::string>(), "OPT_NONE");
}

TEST_F(LinearOPT_NONEJsonTest, ToJsonHLSConfigEmpty) {
  auto config = layer.to_json();

  auto hls_config = config["hls_config"];
  EXPECT_TRUE(hls_config.is_object());
  EXPECT_EQ(hls_config.size(), 0);
}

TEST_F(LinearOPT_NONEJsonTest, ToJsonIsValid) {
  auto config = layer.to_json();

  std::string json_str = config.dump();
  EXPECT_FALSE(json_str.empty());

  auto parsed = json::parse(json_str);
  EXPECT_EQ(parsed["module_type"].get<std::string>(),
            config["module_type"].get<std::string>());
}

TEST_F(LinearOPT_NONEJsonTest, ToJsonDataTypes) {
  auto config = layer.to_json();

  EXPECT_TRUE(config["module_type"].is_string());
  EXPECT_TRUE(config["opt_level"].is_string());
  EXPECT_TRUE(config["hls_config"].is_object());
}

TEST_F(LinearOPT_NONEJsonTest, ToJsonPrettyPrint) {
  auto config = layer.to_json();

  std::string pretty = config.dump(2);
  EXPECT_FALSE(pretty.empty());
  EXPECT_GT(pretty.size(), config.dump().size());
}

// ============================================================================
// Tests for OPT_ENABLED Configuration
// ============================================================================
TEST_F(LinearOPT_ENABLEDJsonTest, ToJsonBasicStructure) {
  auto config = layer.to_json();
  ValidateBasicStructure(config);
}

TEST_F(LinearOPT_ENABLEDJsonTest, ToJsonModuleInfo) {
  auto config = layer.to_json();

  EXPECT_EQ(config["module_name"].get<std::string>(), "Linear");
  EXPECT_EQ(config["module_type"].get<std::string>(), "Linear_OPT_ENABLED");
  EXPECT_EQ(config["opt_level"].get<std::string>(), "OPT_ENABLED");
}

TEST_F(LinearOPT_ENABLEDJsonTest, ToJsonHLSConfigPresent) {
  auto config = layer.to_json();

  auto hls_config = config["hls_config"];
  EXPECT_TRUE(hls_config.is_object());
  EXPECT_GT(hls_config.size(), 0);
}

TEST_F(LinearOPT_ENABLEDJsonTest, ToJsonHLSUnrollFactor) {
  auto config = layer.to_json();

  auto hls_config = config["hls_config"];
  EXPECT_TRUE(hls_config.contains("unroll_factor"));
  EXPECT_EQ(hls_config["unroll_factor"].get<int>(),
            TestOptConfig::_unroll_factor);
}

TEST_F(LinearOPT_ENABLEDJsonTest, ToJsonHLSPartitionFactor) {
  auto config = layer.to_json();

  auto hls_config = config["hls_config"];
  EXPECT_TRUE(hls_config.contains("partition_factor"));
  EXPECT_EQ(hls_config["partition_factor"].get<int>(),
            TestOptConfig::_partition_factor);
}

TEST_F(LinearOPT_ENABLEDJsonTest, ToJsonHLSPipelineII) {
  auto config = layer.to_json();

  auto hls_config = config["hls_config"];
  EXPECT_TRUE(hls_config.contains("pipeline_ii"));
  EXPECT_EQ(hls_config["pipeline_ii"].get<int>(), TestOptConfig::_pipeline_ii);
}

TEST_F(LinearOPT_ENABLEDJsonTest, ToJsonAllHLSParameters) {
  auto config = layer.to_json();

  auto hls_config = config["hls_config"];
  EXPECT_EQ(hls_config["unroll_factor"].get<int>(), 4);
  EXPECT_EQ(hls_config["partition_factor"].get<int>(), 2);
  EXPECT_EQ(hls_config["pipeline_ii"].get<int>(), 2);
}
