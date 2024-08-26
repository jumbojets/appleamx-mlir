//===- appleamx-opt.cpp -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "AppleAMX/Dialect.h"
#include "AppleAMX/Passes.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::appleamx::registerPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::appleamx::AppleAMXDialect,
                  mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::memref::MemRefDialect,
                  mlir::affine::AffineDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "AppleAMX optimizer driver\n", registry));
}
