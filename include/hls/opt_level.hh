#pragma once

enum OptLevel {
  OPT_NONE,       // No optimization (default)
  OPT_LATENCY,    // Minimize latency
  OPT_THROUGHPUT, // Maximize throughput
};
