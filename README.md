# Vitis HLS For Neural Networking `vhn`

###### This is a Vitis HLS(High Level Synthesis) library for neural network implementations and accelerations.

## Key Features

- Support for only c++ env and cpu impls **without Vitis environment**
- Easy to use with **header-only structure**
- Lower overhead with **Modern C++ and generic programming**
- **Configurable** kernels for users.
- **Json** interpreter for kernel configurations.

## How To Use

### Build Essentials

The build script is provided in `scripts/` folder for `unix-like os`

```bash
./scripts/bootstrap.sh
```

Also, build it with any other cmake project:

```bash
mkdir -p build
cd build
cmake ..
make
```

The json parser is located in `build/bin`

### Quick Start

Here is an mlp example(name the file `model.json` for example):

```json
{
  "model": {
    "name": "MLPTestModel",
    "dtype": "float"
  },
  "modules": [
    {
      "name": "fc1",
      "type": "linear",
      "hparams": {
        "in_features": 784,
        "out_features": 256
      },
      "hls_cfg": {
        "partition_factor": 4
      }
    },
    {
      "name": "fc2",
      "type": "linear",
      "hparams": {
        "in_features": 256,
        "out_features": 10
      },
      "hls_cfg": {
        "partition_factor": 4
      }
    }
  ]
}
```

And use json parser built by `tools` to generate c configuration files for these modules:

```bash
parser generate ./model.json ./model_config.hh
```

This will generate a file like this `model_config.hh`:

```c++
#pragma once

// AUTO-GENERATED CODE
// Generated from: ./model.json
#include "path/to/proj/include/vhn.hh"

// Module Configurations

// Configuration for fc1 (linear)
using fc1_hparams = vhn::LinearHParams<784, 256>;

using fc1_cfg = vhn::LinearConfig<4, 4, 16, 16, false>;

using fc1_t = vhn::Linear<float, fc1_hparams, fc1_cfg, OPT_ENABLED>;

// Configuration for fc2 (linear)
using fc2_hparams = vhn::LinearHParams<256, 10>;

using fc2_cfg = vhn::LinearConfig<4, 4, 16, 16, false>;

using fc2_t = vhn::Linear<float, fc2_hparams, fc2_cfg, OPT_ENABLED>;

// Network Information
namespace network_info {
constexpr const char *name = "MLPTestModel";
constexpr const char *dtype = "float";
constexpr int num_modules = 2;
} // namespace network_info

```

And directly include this header file on your vitis source files, and it is done.

## TODO List

##### User Interface

- [x] `Ready For Using`
- [x] `Config File (json)`
- [ ] `Python Bindings`

##### Concurrency

- [x] `Hardware Paralleled`
- [x] `Stream`

##### Basic Kernels

- [ ] `GEMM`
- [x] `MLP`
- [x] `Elementwise`
- [x] `Reduce`
- [ ] `Layers`
  - [x] `Linear`
  - [x] `Softmax`
  - [x] `Conv1d`, `Conv2d`
  - [x] `Embedding`
  - [ ] `Conv1d`, `Conv2d(Winograd)`
  - [ ] `Poolings`
  - [ ] ...
- [ ] `Norms`
  - [x] `BatchNorm1d`,`BatchNorm2d`
  - [x] `LayerNorm`
  - [ ] `RMSNorm`
  - [ ] ...
- [x] `Activations`
  - [x] `Sigmoid`
  - [x] `ReLU`
  - [x] `GeLU`
  - [x] `User-defined`
- [ ] ...

##### Configurable Transformer

- [x] `Common`
  - [x] `AddNorm`
  - [x] `FFN`
- [x] `MulHeadSelfAttn`
  - [x] `Fused Kernel`
  - [x] `Impled Kernel`
- [ ] `MulHeadCrossAttn`
