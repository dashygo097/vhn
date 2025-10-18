#pragma once

namespace hls_nn {
template <const int UNROLL_FACTOR, const int PIPELINE_II> struct LoopConfig {
  static constexpr int _unroll_factor = UNROLL_FACTOR;
  static constexpr int _pipeline_ii = PIPELINE_II;
};

template <const int UNROLL_FACTOR, const int PARTITION_FACTOR,
          const int PIPELINE_II>
struct HLSConfig {
  static constexpr int _unroll_factor = UNROLL_FACTOR;
  static constexpr int _partition_factor = PARTITION_FACTOR;
  static constexpr int _pipeline_ii = PIPELINE_II;
};

} // namespace hls_nn
