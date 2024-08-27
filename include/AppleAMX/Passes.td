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

def AppleAMXRaiseAffineMatmul: Pass<"appleamx-raise-affine-matmul", "::mlir::ModuleOp"> {
  let summary = "Converts nested affine for loops to matmuls";
  let description = [{
    Converts nested affine for loops to matmuls.
    ```
    func.func @matmul(%A: memref<128x64xf16>, %B: memref<64x128xf16>, %C: memref<128x128xf16>) {
      affine.for %i = 0 to 128 {
        affine.for %j = 0 to 128 {
          affine.for %k = 0 to 64 {
            %a = memref.load %A[%i, %k] : memref<128x64xf16>
            %b = memref.load %B[%k, %j] : memref<64x128xf16>
            %c = memref.load %C[%i, %j] : memref<128x128xf16>
            %mul = arith.mulf %a, %b : f16
            %add = arith.addf %c, %mul : f16
            memref.store %add, %C[%i, %j] : memref<128x128xf16>
          }
        }
      } {appleamx.matmul.transform}
      return
    }
    // Gets transformed to:
    module {
      func.func @matmul(%arg0: memref<128x64xf16>, %arg1: memref<64x128xf16>, %arg2: memref<128x128xf16>) {
        linalg.matmul {appleamx.created} ins(%arg0, %arg1 : memref<128x64xf16>, memref<64x128xf16>) outs(%arg2 : memref<128x128xf16>)
        return
      }
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

def AppleAMXTileMatmul: Pass<"appleamx-tile-matmul", "::mlir::func::FuncOp"> {
  let summary = "Tiles matmuls";
  let description = [{
    Tiles matmuls.
  }];

  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::scf::SCFDialect"
  ];
}

#endif // APPLEAMX_PASS
