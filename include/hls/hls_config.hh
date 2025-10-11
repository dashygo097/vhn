#pragma once

namespace hls_nn {
template <const int UNROLL_FACTOR, const int PARTITION_FACTOR,
          const int PIPELINE_II>
struct HLSConfig {
  static constexpr int _unroll_factor = UNROLL_FACTOR;
  static constexpr int _partition_factor = PARTITION_FACTOR;
  static constexpr int _pipeline_ii = PIPELINE_II;
};

template <const int UNROLL_FACTOR, const int PARTITION_FACTOR,
          const int PIPELINE_II, const bool _USE_REDUCE_TREE = true>
struct ReduceConfig {
  static constexpr int _unroll_factor = UNROLL_FACTOR;
  static constexpr int _partition_factor = PARTITION_FACTOR;
  static constexpr int _pipeline_ii = PIPELINE_II;
  static constexpr bool _use_reduce_tree = _USE_REDUCE_TREE;
};

} // namespace hls_nn
