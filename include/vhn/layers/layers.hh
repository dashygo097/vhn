#pragma once

// Layers
#include "./conv1d.hh"
#include "./conv2d.hh"
#include "./embedding.hh"
#include "./linear.hh"
#include "./softmax.hh"

// Builders
#ifndef __VITIS_HLS__
#include "./conv1d_builder.hh"
#include "./conv2d_builder.hh"
#include "./embedding_builder.hh"
#include "./linear_builder.hh"
#include "./softmax_builder.hh"

REGISTER_LAYER_BUILDER("linear", LinearBuilder)
REGISTER_LAYER_BUILDER("conv1d", Conv1dBuilder)
REGISTER_LAYER_BUILDER("conv2d", Conv2dBuilder)
REGISTER_LAYER_BUILDER("embedding", EmbeddingBuilder)
REGISTER_LAYER_BUILDER("softmax", SoftmaxBuilder)
#endif
