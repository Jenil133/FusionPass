#include "fusionpass/Dialect/FP/FPOps.h"
#include "fusionpass/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
using namespace mlir;
using namespace fusionpass::fp;

namespace {
bool isChainable(Operation *op) { return isa<AddOp, MulOp, ScaleOp, ReluOp, ExpOp, SoftmaxOp, FusedChainOp>(op); }

/// A plain softmax is fusible only if it reduces the last axis and the row fits the row-kernel cap.
bool isFusible(Operation *op) {
  if (auto sm = dyn_cast<SoftmaxOp>(op)) {
    auto t = cast<RankedTensorType>(sm.getType());
    return static_cast<int64_t>(sm.getAxis()) == t.getRank() - 1 &&
           t.getShape().back() <= fusionpass::kMaxRowSoftmaxCols;
  }
  return isChainable(op);
}

int softmaxCount(Operation *op) {
  if (isa<SoftmaxOp>(op)) return 1;
  int n = 0;
  if (auto f = dyn_cast<FusedChainOp>(op))
    for (Operation &o : f.getBody().front().without_terminator()) n += isa<SoftmaxOp>(o);
  return n;
}

/// Clone `op` (plain op or the body of a fused_chain) into the builder's current block.
/// `outer` maps values visible outside to values inside the new body. Returns the value
/// that represents op's single result inside the new body.
Value inlineOp(OpBuilder &b, Operation *op, IRMapping &outer) {
  if (auto f = dyn_cast<FusedChainOp>(op)) {
    IRMapping inner;
    Block &body = f.getBody().front();
    for (auto [arg, operand] : llvm::zip(body.getArguments(), f.getInputs()))
      inner.map(arg, outer.lookup(operand));
    for (Operation &o : body.without_terminator()) b.clone(o, inner);
    return inner.lookup(cast<YieldOp>(body.getTerminator()).getValue());
  }
  return b.clone(*op, outer)->getResult(0);
}

struct FuseChainPattern : RewritePattern {
  FuseChainPattern(MLIRContext *ctx) : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *c, PatternRewriter &rw) const override {
    if (!isChainable(c) || !isFusible(c)) return failure();
    // GreedyPatternRewriteDriver walks IsolatedFromAbove regions. Fusing inside an
    // existing fused_chain would nest fused_chain ops, which the region verifier rejects.
    if (isa<FusedChainOp>(c->getParentOp())) return failure();
    for (OpOperand &use : c->getOpOperands()) {
      Value pv = use.get();
      Operation *p = pv.getDefiningOp();
      if (!p || !isChainable(p) || !isFusible(p)) continue;
      if (!pv.hasOneUse()) continue;               // WHY: a multi-use producer must stay materialized
      if (p->getBlock() != c->getBlock()) continue;
      if (softmaxCount(p) + softmaxCount(c) > 1) continue;   // row kernel supports one reduction

      // External inputs of the merged region: producer operands + consumer operands (minus pv), deduped.
      SmallVector<Value> externals;
      auto addExt = [&](Value x) { if (!llvm::is_contained(externals, x)) externals.push_back(x); };
      for (Value x : p->getOperands()) addExt(x);
      for (Value x : c->getOperands()) if (x != pv) addExt(x);

      Type resTy = c->getResult(0).getType();
      rw.setInsertionPoint(c);
      auto fused = rw.create<FusedChainOp>(c->getLoc(), resTy, externals);
      SmallVector<Type> argTys;
      SmallVector<Location> locs;
      for (Value x : externals) { argTys.push_back(x.getType()); locs.push_back(c->getLoc()); }
      Block *blk = rw.createBlock(&fused.getBody(), fused.getBody().end(), argTys, locs);

      IRMapping map;
      for (auto [i, x] : llvm::enumerate(externals)) map.map(x, blk->getArgument(i));
      rw.setInsertionPointToStart(blk);
      Value pOut = inlineOp(rw, p, map);
      map.map(pv, pOut);
      Value cOut = inlineOp(rw, c, map);
      rw.create<YieldOp>(c->getLoc(), cOut);

      rw.replaceOp(c, fused.getResult());          // p becomes dead; the greedy driver erases it
      return success();
    }
    return failure();
  }
};

struct FuseChainsPass : public PassWrapper<FuseChainsPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseChainsPass)
  StringRef getArgument() const final { return "fp-fuse-chains"; }
  StringRef getDescription() const final { return "Fuse elementwise/softmax chains into fp.fused_chain"; }
  void getDependentDialects(DialectRegistry &r) const override { r.insert<FPDialect>(); }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseChainPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> fusionpass::createFuseChainsPass() { return std::make_unique<FuseChainsPass>(); }
