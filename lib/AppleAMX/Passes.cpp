//===- Passes.cpp - AppleAMX passes -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <optional>
#include <iostream>

#include "AppleAMX/Passes.h"

namespace mlir::appleamx {
#define GEN_PASS_DEF_APPLEAMXSWITCHBARFOO
#define GEN_PASS_DEF_APPLEAMXRAISEAFFINEMATMUL
#include "AppleAMX/Passes.h.inc"

namespace {
class AppleAMXRaiseAffineMatmulRewriter : public OpRewritePattern<affine::AffineForOp> {
public:
  using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(affine::AffineForOp op,
                                PatternRewriter &rewriter) const final {
    auto loopNest = detectMatmulPattern(op);
    if (!loopNest)
      return failure();

    Value A, B, C;
    int64_t M, N, K; // TODO: figure out how to uses these
    std::tie(A, B, C, M, N, K) = *loopNest;

    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<linalg::MatmulOp>(op, C.getType(), ValueRange{A, B, C});

    return success();
  }

private:
  using LoopNest = std::tuple<Value, Value, Value, int64_t, int64_t, int64_t>;

  std::optional<LoopNest> detectMatmulPattern(affine::AffineForOp rootLoop) const {
    // if (!isMatmulRootLoop(rootLoop)) {}
    //   return {};
      
    auto loops = getNestedLoops(rootLoop);
    if (loops.size() != 3)
      return {};
      
    auto [A, B, C] = extractOperands(loops);

    int64_t M = loops[0].getConstantUpperBound();
    int64_t N = loops[1].getConstantUpperBound();
    int64_t K = loops[2].getConstantUpperBound();

    if (A && B && C && M > 0 && N > 0 && K > 0)
      return LoopNest(A, B, C, M, N, K);

    return {};
  }

  bool isMatmulRootLoop(affine::AffineForOp loop) const {
    // Check for three nested loops
    if (!loop->hasOneUse() || !isa<affine::AffineForOp>(*loop.getBody()->begin()))
      return false;
    auto innerLoop1 = cast<affine::AffineForOp>(loop.getBody()->front());
    if (!innerLoop1->hasOneUse() || !isa<affine::AffineForOp>(*innerLoop1.getBody()->begin()))
      return false;
    auto innerLoop2 = cast<affine::AffineForOp>(innerLoop1.getBody()->front());
    
    // Check for matmul operations in the innermost loop
    for (auto &op : innerLoop2.getBody()->getOperations()) {
      if (isa<memref::LoadOp>(op) || isa<memref::StoreOp>(op) || 
          isa<arith::MulFOp>(op) || isa<arith::AddFOp>(op))
        continue;
      if (!isa<affine::AffineYieldOp>(op))
        return false;
    }
    return true;
  }

  SmallVector<affine::AffineForOp, 3> getNestedLoops(affine::AffineForOp rootLoop) const {
    SmallVector<affine::AffineForOp, 3> loops;
    loops.push_back(rootLoop);
    auto loop = rootLoop;

    for (int i = 0; i < 2; i++) {
      if (auto innerLoop = dyn_cast<affine::AffineForOp>(loop.getBody()->front())) {
        loops.push_back(innerLoop);
        loop = innerLoop;
      } else {
        break;
      }
    }

    return loops;
  }

  std::tuple<Value, Value, Value> extractOperands(SmallVector<affine::AffineForOp, 3> &loops) const {
    // Simplified example for illustration
    // You would analyze the body of the innermost loop to identify the load, mul, add operations
    // and extract the corresponding matrices A, B, and C.

    Value A, B, C;

    Value i = loops[0].getInductionVar();
    Value j = loops[1].getInductionVar();
    Value k = loops[2].getInductionVar();
    
    for (auto &op : loops[2].getBody()->getOperations()) {
      if (auto loadOp = dyn_cast<memref::LoadOp>(&op)) {
        // Check if load corresponds to A[i, k] or B[k, j]
        if (matchesLoadPattern(loadOp, i, k)) {
          A = loadOp.getMemRef();
        } else if (matchesLoadPattern(loadOp, k, j)) {
          B = loadOp.getMemRef();
        }
      } else if (auto storeOp = dyn_cast<memref::StoreOp>(&op)) {
        // Check if store corresponds to C[i, j]
        if (matchesStorePattern(storeOp, i, j)) {
          C = storeOp.getMemRef();
        }
      }
    }

    // Ensure all operands were found
    if (!A || !B || !C) {
      return std::make_tuple(Value(), Value(), Value()); // Return empty tuple on failure
    }

    return std::make_tuple(A, B, C);
  }

  bool matchesLoadPattern(memref::LoadOp loadOp, Value idx1, Value idx2) const {
    auto indices = loadOp.getIndices();
    // Ensure there are exactly two indices
    if (indices.size() != 2) {
      return false;
    }
    // Check that the indices match the expected loop induction variables
    return indices[0] == idx1 && indices[1] == idx2;
  }

  bool matchesStorePattern(memref::StoreOp storeOp, Value idx1, Value idx2) const {
    auto indices = storeOp.getIndices();
    // Ensure there are exactly two indices
    if (indices.size() != 2) {
      return false;
    }
    // Check that the indices match the expected loop induction variables
    return indices[0] == idx1 && indices[1] == idx2;
  }
};

class AppleAMXRaiseAffineMatmul
    : public impl::AppleAMXRaiseAffineMatmulBase<AppleAMXRaiseAffineMatmul> {
public:
  using impl::AppleAMXRaiseAffineMatmulBase<AppleAMXRaiseAffineMatmul>::AppleAMXRaiseAffineMatmulBase;
  void runOnOperation() final {
    getOperation()->emitRemark("Running AppleAMXRaiseAffineMatmul");

    RewritePatternSet patterns(&getContext());
    patterns.add<AppleAMXRaiseAffineMatmulRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};
} // namespace
} // namespace mlir::appleamx
