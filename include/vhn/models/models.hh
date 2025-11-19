#pragma once
// CNN

// Transformer
#include "./transformer/attns/mha.hh"
#include "./transformer/attns/mhca.hh"
#include "./transformer/components/addnorm.hh"
#include "./transformer/components/ffn.hh"
#include "./transformer/components/postnorm.hh"
#include "./transformer/components/prenorm.hh"
#include "./transformer/decoder.hh"
// #include "./transformer/encoder.hh"

// Common

// Builders
#ifndef __VITIS_HLS__
#include "./transformer/attns/mha_builder.hh"
#include "./transformer/components/addnorm_builder.hh"
#include "./transformer/components/ffn_builder.hh"
#include "./transformer/encoder_builder.hh"

REGISTER_LAYER_BUILDER("ffn", FFNBuilder)
REGISTER_LAYER_BUILDER("addnorm", AddNormBuilder)
REGISTER_LAYER_BUILDER("mha", MulHeadAttnBuilder)
REGISTER_LAYER_BUILDER("enc_blk", EncoderBlockBuilder)
#endif
