#pragma once

#include "./batchnorm1d.hh"
#include "./batchnorm2d.hh"
#include "./layernorm.hh"

#ifndef __VITIS_HLS__
#include "./batchnorm1d_builder.hh"
#include "./layernorm_builder.hh"

REGISTER_LAYER_BUILDER("layernorm", LayerNormBuilder)
REGISTER_LAYER_BUILDER("batchnorm1d", BatchNorm1dBuilder)
#endif
