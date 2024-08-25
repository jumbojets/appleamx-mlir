// RUN: mlir-opt %s --load-pass-plugin=%appleamx_libs/AppleAMXPlugin%shlibext --pass-pipeline="builtin.module(appleamx-switch-bar-foo)" | FileCheck %s

module {
  // CHECK-LABEL: func @foo()
  func.func @bar() {
    return
  }

  // CHECK-LABEL: func @abar()
  func.func @abar() {
    return
  }
}
