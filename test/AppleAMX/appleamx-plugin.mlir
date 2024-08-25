// RUN: mlir-opt %s --load-dialect-plugin=%appleamx_libs/AppleAMXPlugin%shlibext --pass-pipeline="builtin.module(appleamx-switch-bar-foo)" | FileCheck %s

module {
  // CHECK-LABEL: func @foo()
  func.func @bar() {
    return
  }

  // CHECK-LABEL: func @appleamx_types(%arg0: !appleamx.custom<"10">)
  func.func @appleamx_types(%arg0: !appleamx.custom<"10">) {
    return
  }
}
