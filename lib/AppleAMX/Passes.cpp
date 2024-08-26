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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <optional>

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
    if (!op->hasAttr("appleamx.matmul.transform"))
      return failure();

    auto loopNest = detectMatmulPattern(op);
    if (!loopNest)
      return failure();

    Value A, B, C;
    int64_t M, N, K;
    std::tie(A, B, C, M, N, K) = *loopNest;

    auto AType = mlir::cast<MemRefType>(A.getType());
    auto BType = mlir::cast<MemRefType>(B.getType());
    auto CType = mlir::cast<MemRefType>(C.getType());
    auto elementType = AType.getElementType();

    auto ATensorType = RankedTensorType::get(AType.getShape(), elementType);
    auto BTensorType = RankedTensorType::get(BType.getShape(), elementType);
    auto CTensorType = RankedTensorType::get(CType.getShape(), elementType);

    rewriter.setInsertionPoint(op);
    Value ATensor = rewriter.create<bufferization::ToTensorOp>(op.getLoc(), ATensorType, A);
    Value BTensor = rewriter.create<bufferization::ToTensorOp>(op.getLoc(), BTensorType, B);
    Value CTensor = rewriter.create<bufferization::ToTensorOp>(op.getLoc(), CTensorType, C);

    auto matmulOp = rewriter.create<linalg::MatmulOp>(
      op.getLoc(),
      TypeRange{CTensorType},
      ValueRange{ATensor, BTensor},
      ValueRange{CTensor}
    );

    Value result = rewriter.create<bufferization::ToMemrefOp>(op.getLoc(), CType, matmulOp.getResult(0));

    rewriter.create<memref::CopyOp>(op.getLoc(), result, C);

    rewriter.eraseOp(op);

    return success();
  }

private:
  using LoopNest = std::tuple<Value, Value, Value, int64_t, int64_t, int64_t>;

  std::optional<LoopNest> detectMatmulPattern(affine::AffineForOp rootLoop) const {
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
    Value A, B, C;

    Value i = loops[0].getInductionVar();
    Value j = loops[1].getInductionVar();
    Value k = loops[2].getInductionVar();
    
    for (auto &op : loops[2].getBody()->getOperations()) {
      if (auto loadOp = dyn_cast<memref::LoadOp>(&op)) {
        // Check if load corresponds to A[i, k] or B[k, j]
        if (matchesLoadStorePattern(loadOp, i, k)) {
          A = loadOp.getMemRef();
        } else if (matchesLoadStorePattern(loadOp, k, j)) {
          B = loadOp.getMemRef();
        }
      } else if (auto storeOp = dyn_cast<memref::StoreOp>(&op)) {
        // Check if store corresponds to C[i, j]
        if (matchesLoadStorePattern(storeOp, i, j)) {
          C = storeOp.getMemRef();
        }
      }
    }

    // Ensure all operands were found
    if (!A || !B || !C) {
      return std::make_tuple(Value(), Value(), Value());
    }

    return std::make_tuple(A, B, C);
  }

  template <typename T>
  bool matchesLoadStorePattern(T loadStoreOp, Value idx1, Value idx2) const {
    auto indices = loadStoreOp.getIndices();
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
    RewritePatternSet patterns(&getContext());
    patterns.add<AppleAMXRaiseAffineMatmulRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};
} // namespace
} // namespace mlir::appleamx
