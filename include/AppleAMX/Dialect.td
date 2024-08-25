//===- Dialect.td - AppleAMX dialect -----------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef APPLEAMX_DIALECT
#define APPLEAMX_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// AppleAMX dialect definition.
//===----------------------------------------------------------------------===//

def AppleAMX_Dialect : Dialect {
    let name = "appleamx";
    let summary = "A standalone out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::appleamx";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base appleamx operation definition.
//===----------------------------------------------------------------------===//

class AppleAMX_Op<string mnemonic, list<Trait> traits = []> :
        Op<AppleAMX_Dialect, mnemonic, traits>;

#endif // APPLEAMX_DIALECT
