// RUN: mlir-opt %s --load-pass-plugin=%appleamx_libs/AppleAMXPlugin%shlibext --pass-pipeline="builtin.module(appleamx-raise-affine-matmul)" | FileCheck %s

module {
  func.func @matmul(%A: memref<128x64xf16>, %B: memref<64x128xf16>, %C: memref<128x128xf16>) {
    // CHECK: %0 = bufferization.to_tensor %arg0 : memref<128x64xf16>
    // CHECK: %1 = bufferization.to_tensor %arg1 : memref<64x128xf16>
    // CHECK: %2 = bufferization.to_tensor %arg2 : memref<128x128xf16>
    // CHECK: %3 = linalg.matmul {appleamx.created} ins(%0, %1 : tensor<128x64xf16>, tensor<64x128xf16>) outs(%2 : tensor<128x128xf16>) -> tensor<128x128xf16>
    // CHECK: %4 = bufferization.to_memref %3 : memref<128x128xf16>
    // CHECK: memref.copy %4, %arg2 : memref<128x128xf16> to memref<128x128xf16>
    // CHECK: return
    // CHECK-NOT: affine.for
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
}
