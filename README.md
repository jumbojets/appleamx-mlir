# Apple AMX Support for MLIR

https://github.com/corsix/amx

### Clone

```bash
git clone --recursive https://github.com/jumbojets/appleamx-mlir
```

### Install LLVM, MLIR, Clang, AppleAMX-MLIR

#### 1. Build LLVM, MLIR, Clang

```bash
mkdir llvm-project/build
cd llvm-project/build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG
ninja
ninja check-mlir
```

#### 2. Build and test AppleAMX-MLIR

```bash
mkdir build && cd build
cmake -G Ninja .. \
  -DMLIR_DIR=$PWD/../llvm-project/build/lib/cmake/mlir \
  -DCLANG_DIR=$PWD/../llvm-project/build/lib/cmake/clang \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG
ninja
ninja check-applemax
```

### TODO

- [ ] get MVP working
- [ ] find different home for the raise affine matmul stuff
- [ ] make some higher level constructors work for both buffers and tensors
- [ ] flesh out entire dialect
- [ ] high-level Apple AMX optimizations
