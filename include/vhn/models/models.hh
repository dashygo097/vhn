#pragma once
// CNN

// Transformer
#include "./transformer/attns/mha.hh"
#include "./transformer/attns/mhca.hh"
#include "./transformer/components/addnorm.hh"
// #include "./transformer/components/ffn.hh"
#include "./transformer/components/postnorm.hh"
#include "./transformer/components/prenorm.hh"
#include "./transformer/decoder.hh"
// #include "./transformer/encoder.hh"

// Common
#include "./mlp.hh"

// Builders
#ifndef __VITIS_HLS__
#include "./mlp_builder.hh"

REGISTER_LAYER_BUILDER("mlp", MLPBuilder)
#endif
