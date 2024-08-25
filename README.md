# Apple AMX Support for MLIR

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

#### 2. Build AppleAMX-MLIR

```bash
mkdir build
cd build
cmake -G Ninja .. \
  -DMLIR_DIR=$PWD/../llvm-project/build/lib/cmake/mlir \
  -DCLANG_DIR=$PWD/../llvm-project/build/lib/cmake/clang \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG
ninja
ninja check-appleamx-opt
```

```bash
mkdir build && cd build
cmake -G Ninja .. \
  -DMLIR_DIR=$PWD/../llvm-project/build/lib/cmake/mlir \
  -DCLANG_DIR=$PWD/../llvm-project/build/lib/cmake/clang \
  -DCMAKE_BUILD_TYPE=DEBUG
cmake --build . --target check-appleamx
```
