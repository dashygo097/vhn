#pragma once

// Layers
#include "./conv1d.hh"
#include "./conv2d.hh"
#include "./embedding.hh"
#include "./linear.hh"
#include "./softmax.hh"

// Builders
#ifndef __VITIS_HLS__
#include "./linear_builder.hh"

REGISTER_LAYER_BUILDER("linear", LinearBuilder)
#endif
