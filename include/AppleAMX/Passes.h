//===- Passes.h - AppleAMX passes  ------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef APPLEAMX_PASSES_H
#define APPLEAMX_PASSES_H

#include "AppleAMX/Dialect.h"
#include "AppleAMX/Ops.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace appleamx {
#define GEN_PASS_DECL
#include "AppleAMX/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "AppleAMX/Passes.h.inc"
} // namespace appleamx 
} // namespace mlir

#endif
