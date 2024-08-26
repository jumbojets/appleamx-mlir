//===-- Passes.td - AppleAMX dialect passes -----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef APPLEAMX_PASS
#define APPLEAMX_PASS

include "mlir/Pass/PassBase.td"

// def AppleAMXRaiseAffineMatmul: Pass<"appleamx-raise-affine-matmul", "::mlir::affine::AffineForOp"> {
def AppleAMXRaiseAffineMatmul: Pass<"appleamx-raise-affine-matmul", "::mlir::ModuleOp"> {
  let summary = "Converts nested affine for loops to matmuls";
  let description = [{
    Converts nested affine for loops to matmuls.
    ```
    func.func @matmul(%A: memref<100x100xf32>, %B: memref<100x100xf32>, %C: memref<100x100xf32>) {
      affine.for %i = 0 to 100 {
        affine.for %j = 0 to 100 {
          affine.for %k = 0 to 100 {
            %0 = affine.load %A[%i, %k] : memref<100x100xf32>
            %1 = affine.load %B[%k, %j] : memref<100x100xf32>
            %2 = arith.mulf %0, %1 : f32
            affine.store %2, %C[%i, %j] : memref<100x100xf32>
          }
        }
      }
    }
    // Gets transformed to:
    func.func @matmul(%A: memref<100x100xf32>, %B: memref<100x100xf32>, %C: memref<100x100xf32>) {
      %0 = appleamx.matmul %A, %B : memref<100x100xf32>, memref<100x100xf32>, memref<100x100xf32>
    }
    ```
  }];

  let dependentDialects = [
    "::mlir::linalg::LinalgDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::bufferization::BufferizationDialect"
  ];
}

#endif // APPLEAMX_PASS