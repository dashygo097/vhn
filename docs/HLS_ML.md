# HLS For ML `hls_nn`

###### This is a Vitis HLS(High Level Synthesis) library for ML implementations and accelerations.

## Key Features

- Support for only c++ env and cpu impls **without Vitis environment**
- Easy to use with **header-only structure**
- Lower overhead with **Modern C++ and generic programming**
- **Configurable** kernels for users.

## How To Use

## Quick Start

If you just want to use it with no user defined optimizations:

```c++
#ifndef __VITIS_HLS__
#define __VITIS_HLS__
#endif

#include "path/to/proj/include/hls.hh" // Because this is a header only proj, so you can directly include it.
FIXED(16, 8) // This defines a type called `fixed16_8` (basically it is `ap_fixed`)
using TestLinear = hls_nn::Linear<fixed16_8, 256, 10, void, OPT_NONE>;
using TestVarMLP = hls_nn::MLP<fixed16_8, 256, hls_nn::ReLU, void, OPT_NONE, 128, 64, 10>;

void linear(fixed16_8 output[10 * 10], const fixed16_8 input[10 * 256],
    const fixed16_8 w1[10][256], const fixed16_8 b1[10]) {
        TestLinear::forward(output, input, 10, w1, b1);
    } // If you are unfamiliar with the static method `forward`, you should check the code(just refer to its definition).

void mlp(fixed16_8 output[10 * 10], const fixed16_8 input[10 * 256],
    const fixed16_8 w1[128][256], const fixed16_8 b1[128],
    const fixed16_8 w2[64][128], const fixed16_8 b2[64],
    const fixed16_8 w3[10][64], const fixed16_8 b3[10]) {
        TestVarMLP::forward(output, input, 10, w1, b1, w2, b2, w3, b3);
}
```

### TODO List

##### User Interface

- [x] `Ready For Using`
- [ ] `Config File (.yaml .toml .json ...)`
- [ ] `Python Bindings`

##### Concurrency

- [x] `Hardware Paralleled`
- [ ] `Stream`

##### Basic Kernels

- [x] `MLP`
- [x] `Elementwise`
- [x] `Reduce`
- [x] `Layers`
  - [x] `Linear`
  - [x] `Softmax`
  - [x] `Conv1d`, `Conv2d`
  - [ ] ...
- [ ] `Norms`
  - [x] `BatchNorm1d`,`BatchNorm2d`
  - [x] `LayerNorm`
  - [ ] `RMSNorm`
  - [ ] ...
- [ ] ...

##### Configurable Transformer

- [x] `Common`
  - [x] `AddNorm`
  - [x] `FFN`
- [ ] `MulHeadSelfAttn`
  - [x] `Fused Kernel`
  - [ ] `Impled Kernel`
- [ ] `MulHeadCrossAttn`
