//===- Types.cpp - AppleAMX dialect types -----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AppleAMX/Types.h"

#include "AppleAMX/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::appleamx;

#define GET_TYPEDEF_CLASSES
#include "AppleAMX/OpsTypes.cpp.inc"

void AppleAMXDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "AppleAMX/OpsTypes.cpp.inc"
      >();
}
