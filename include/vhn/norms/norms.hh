#pragma once

#include "./batchnorm1d.hh"
#include "./batchnorm2d.hh"
#include "./layernorm.hh"

#ifndef __VITIS_HLS__
#include "./layernorm_builder.hh"

REGISTER_LAYER_BUILDER("layernorm", LayerNormBuilder)
#endif
