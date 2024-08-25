// RUN: appleamx-opt %s | appleamx-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = arith.constant 1 : i32
        // CHECK: %{{.*}} = appleamx.foo %{{.*}} : i32
        %res = appleamx.foo %0 : i32
        return
    }

    // CHECK-LABEL: func @appleamx(%arg0: !appleamx.custom<"10">)
    func.func @appleamx(%arg0: !appleamx.custom<"10">) {
        return
    }
}
