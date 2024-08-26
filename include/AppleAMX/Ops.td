//===- AppleAMXOps.td - AppleAMX dialect ops ---------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef APPLEAMX_OPS
#define APPLEAMX_OPS

include "AppleAMX/Types.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def AppleAMX_FooOp : AppleAMX_Op<"foo", [Pure, SameOperandsAndResultType]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `appleamx.foo` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = arith.constant 2 : i32
        // Apply the foo operation to %0
        %1 = appleamx.foo %0 : i32
        ```
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

def AppleAMX_LoadX : AppleAMX_Op<"loadx"> {
    let summary = "Load 64 bytes from memory into the x memory pool";
    let arguments = (ins AnyMemRef : $source, Index : $index);
}

def AppleAMX_LoadY : AppleAMX_Op<"loady"> {
    let summary = "Load 64 bytes from memory into the y memory pool";
    let arguments = (ins AnyMemRef : $source, Index : $index);
}

def AppleAMX_StoreZ : AppleAMX_Op<"storez"> {
    let summary = "Store 64 bytes into memory from the zmemory pool";
    let arguments = (ins AnyMemRef : $source, Index : $index);
}

def AppleAMX_FMA16 : AppleAMX_Op<"matrix_fma16"> {
    let summary = "Perform float16 outer product on 32-length vectors from x and y and accumulate into z";
    let arguments = (ins Index : $offsetx, Index : $offsety);
}

#endif // APPLEAMX_OPS
