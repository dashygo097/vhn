# Vitis HLS For Neural Networking `vhn`

###### This is a Vitis HLS(High Level Synthesis) library for neural network implementations and accelerations.

## Key Features

- Support for only c++ env and cpu impls **without Vitis environment**
- Easy to use with **header-only structure**
- Lower overhead with **Modern C++ and generic programming**
- **Configurable** kernels for users.
- Json interpreter for kernel configurations.

## How To Use

## Quick Start

If you just want to use it with no user defined optimizations:

```c++

```

### TODO List

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
  - [ ] ...
- [ ] ...

##### Configurable Transformer

- [x] `Common`
  - [x] `AddNorm`
  - [x] `FFN`
- [x] `MulHeadSelfAttn`
  - [x] `Fused Kernel`
  - [x] `Impled Kernel`
- [ ] `MulHeadCrossAttn`
