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

### References
- https://www.jeremykun.com/2023/09/11/mlir-folders/
- https://polygeist.llvm.org/getting_started/Use_Polygeist/
- https://discourse.llvm.org/t/help-lowering-linalg-matmul-to-vector-outer-product/81238
- https://github.com/llvm/llvm-project/blob/main/mlir/test/Integration/Dialect/Linalg/CPU/ArmSME/matmul.mlir

### TODO

- [ ] get MVP working
- [ ] find different home for the raise affine matmul stuff
- [ ] make some higher level constructors work for both buffers and tensors
- [ ] flesh out entire dialect
- [ ] high-level Apple AMX optimizations
