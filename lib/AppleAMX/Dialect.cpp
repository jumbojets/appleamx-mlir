//===- Dialect.cpp - AppleAMX dialect ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AppleAMX/Dialect.h"
#include "AppleAMX/Ops.h"
#include "AppleAMX/Types.h"

using namespace mlir;
using namespace mlir::appleamx;

#include "AppleAMX/OpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// AppleAMX dialect.
//===----------------------------------------------------------------------===//

void AppleAMXDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "AppleAMX/Ops.cpp.inc"
      >();
  registerTypes();
}
