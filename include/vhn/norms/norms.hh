#pragma once

#include "./batchnorm1d.hh"
#include "./batchnorm2d.hh"
#include "./layernorm.hh"

#ifndef __VITIS_HLS__
#include "./bn1d_builder.hh"
#include "./bn2d_builder.hh"
#include "./ln_builder.hh"

REGISTER_LAYER_BUILDER("ln", LayerNormBuilder)
REGISTER_LAYER_BUILDER("bn1d", BatchNorm1dBuilder)
REGISTER_LAYER_BUILDER("bn2d", BatchNorm2dBuilder)
#endif
