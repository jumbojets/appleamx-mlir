//===- Types.td - AppleAMX dialect types -------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef APPLEAMX_TYPES
#define APPLEAMX_TYPES

include "mlir/IR/AttrTypeBase.td"
include "AppleAMX/Dialect.td"

//===----------------------------------------------------------------------===//
// AppleAMX type definitions
//===----------------------------------------------------------------------===//

class AppleAMX_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<AppleAMX_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def AppleAMX_CustomType : AppleAMX_Type<"Custom", "custom"> {
    let summary = "Standalone custom type";
    let description = "Custom type in standalone dialect";
    let parameters = (ins StringRefParameter<"the custom value">:$value);
    let assemblyFormat = "`<` $value `>`";
}

#endif // APPLEAMX_TYPES
