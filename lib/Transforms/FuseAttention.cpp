#include "fusionpass/Dialect/FP/FPOps.h"
#include "fusionpass/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
using namespace mlir;
using namespace fusionpass::fp;

namespace {
/// Root pattern at the SECOND matmul (P @ V), then walk backwards.
/// matmul(softmax(scale*(matmul(Q, transpose(K)))), V)
struct FuseAttentionPattern : OpRewritePattern<MatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulOp pv, PatternRewriter &rw) const override {
    Value p = pv.getLhs(), v = pv.getRhs();
    auto sm = p.getDefiningOp<SoftmaxOp>();
    // WHY single-use everywhere: if an intermediate (e.g. the probabilities P) is used elsewhere,
    // fusing would force us to still materialize it, defeating the point -> do not fuse.
    if (!sm || !p.hasOneUse()) return failure();
    auto st = cast<RankedTensorType>(sm.getType());
    if (static_cast<int64_t>(sm.getAxis()) != st.getRank() - 1) return failure();

    double scale = 1.0;
    Value cur = sm.getInput();
    while (auto sc = cur.getDefiningOp<ScaleOp>()) {   // zero or more scales; combine multiplicatively
      if (!cur.hasOneUse()) return failure();
      scale *= (double)sc.getFactor().convertToFloat();
      cur = sc.getInput();
    }
    auto qk = cur.getDefiningOp<MatmulOp>();
    if (!qk || !cur.hasOneUse()) return failure();
    auto kt = qk.getRhs().getDefiningOp<TransposeOp>();
    if (!kt || !qk.getRhs().hasOneUse()) return failure();

    Value q = qk.getLhs(), k = kt.getInput();
    // Self-attention shape contract: Q, K, V and output all identical, rank >= 3, D in {16,32,64,128}.
    auto qt = cast<RankedTensorType>(q.getType());
    if (qt != cast<RankedTensorType>(k.getType()) || qt != cast<RankedTensorType>(v.getType()) ||
        qt != cast<RankedTensorType>(pv.getType()) || qt.getRank() < 3)
      return failure();
    int64_t d = qt.getShape().back();
    if (d != 16 && d != 32 && d != 64 && d != 128) return failure();

    auto fused = rw.create<FusedAttentionOp>(pv.getLoc(), pv.getType(), q, k, v,
                                             rw.getF32FloatAttr((float)scale));
    rw.replaceOp(pv, fused.getResult());   // the dead softmax/scale/matmul/transpose are erased by the driver
    return success();
  }
};

struct FuseAttentionPass
    : public PassWrapper<FuseAttentionPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseAttentionPass)
  StringRef getArgument() const final { return "fp-fuse-attention"; }
  StringRef getDescription() const final { return "Fuse attention subgraphs into fp.fused_attention"; }
  void getDependentDialects(DialectRegistry &r) const override { r.insert<FPDialect>(); }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseAttentionPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> fusionpass::createFuseAttentionPass() { return std::make_unique<FuseAttentionPass>(); }
