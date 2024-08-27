// RUN: appleamx-opt --appleamx-tile-matmul %s | FileCheck %s

module {
  func.func @matmul(%arg0: memref<128x64xf16>, %arg1: memref<64x128xf16>, %arg2: memref<128x128xf16>) {
    linalg.matmul {appleamx.created} ins(%arg0, %arg1 : memref<128x64xf16>, memref<64x128xf16>) outs(%arg2 : memref<128x128xf16>)
    return
    // CHECK: %c32 = arith.constant 32 : index
    // CHECK: %c0 = arith.constant 0 : index
    // CHECK: %c128 = arith.constant 128 : index
    // CHECK: %c64 = arith.constant 64 : index
    // CHECK: scf.for %arg3 = %c0 to %c128 step %c32 {
    // CHECK:   scf.for %arg4 = %c0 to %c128 step %c32 {
    // CHECK:     scf.for %arg5 = %c0 to %c64 step %c32 {
    // CHECK:       %subview = memref.subview %arg0[%arg3, %arg5] [32, 32] [1, 1]
    // CHECK:       %subview_0 = memref.subview %arg1[%arg5, %arg4] [32, 32] [1, 1]
    // CHECK:       %subview_1 = memref.subview %arg2[%arg3, %arg4] [32, 32] [1, 1]
    // CHECK:       linalg.matmul {appleamx.created} ins(%subview, %subview_0 : memref<32x32xf16, strided<[64, 1], offset: ?>>, memref<32x32xf16, strided<[128, 1], offset: ?>>) outs(%subview_1 : memref<32x32xf16, strided<[128, 1], offset: ?>>)
    // CHECK:     }
    // CHECK:   }
    // CHECK: }
    // CHECK: return
  }
}

