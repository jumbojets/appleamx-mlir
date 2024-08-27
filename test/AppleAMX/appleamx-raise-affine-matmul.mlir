// RUN: appleamx-opt --appleamx-raise-affine-matmul %s | FileCheck %s

module {
  func.func @matmul(%A: memref<128x64xf16>, %B: memref<64x128xf16>, %C: memref<128x128xf16>) {
    // CHECK: func.func @matmul(%[[A:.*]]: memref<128x64xf16>, %[[B:.*]]: memref<64x128xf16>, %[[C:.*]]: memref<128x128xf16>) {
    // CHECK-NEXT: linalg.matmul {appleamx.created} ins(%[[A]], %[[B]] : memref<128x64xf16>, memref<64x128xf16>) outs(%[[C]] : memref<128x128xf16>)
    // CHECK-NEXT: return
    // CHECK-NEXT: }
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
