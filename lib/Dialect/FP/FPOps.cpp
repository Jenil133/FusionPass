#include "fusionpass/Dialect/FP/FPOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <utility>
using namespace mlir;
using namespace fusionpass::fp;

#define GET_OP_CLASSES
#include "fusionpass/Dialect/FP/FPOps.cpp.inc"

LogicalResult SoftmaxOp::verify() {
  auto t = cast<RankedTensorType>(getInput().getType());
  // WHY non-negative: the frontend normalizes dim; passes compare axis == rank-1 directly.
  if (getAxis() < 0 || getAxis() >= t.getRank())
    return emitOpError("axis must be in [0, rank)");
  return success();
}

LogicalResult MatmulOp::verify() {
  auto l = cast<RankedTensorType>(getLhs().getType());
  auto r = cast<RankedTensorType>(getRhs().getType());
  auto o = cast<RankedTensorType>(getResult().getType());
  int64_t rank = l.getRank();
  if (rank < 2 || r.getRank() != rank || o.getRank() != rank)
    return emitOpError("operands and result must have equal rank >= 2");
  if (l.getElementType() != r.getElementType() || l.getElementType() != o.getElementType())
    return emitOpError("element types must match");
  SmallVector<int64_t> expected;
  for (int64_t i = 0; i < rank - 2; ++i) {
    if (l.getDimSize(i) != r.getDimSize(i))
      return emitOpError("batch dimensions must match");
    expected.push_back(l.getDimSize(i));
  }
  if (l.getDimSize(rank - 1) != r.getDimSize(rank - 2))
    return emitOpError("contraction dimension mismatch");
  expected.push_back(l.getDimSize(rank - 2));
  expected.push_back(r.getDimSize(rank - 1));
  if (o.getShape() != ArrayRef<int64_t>(expected))
    return emitOpError("result shape does not match matmul semantics");
  return success();
}

LogicalResult TransposeOp::verify() {
  auto i = cast<RankedTensorType>(getInput().getType());
  auto o = cast<RankedTensorType>(getResult().getType());
  if (i.getRank() < 2 || o.getRank() != i.getRank())
    return emitOpError("rank must be >= 2 and preserved");
  SmallVector<int64_t> expected(i.getShape().begin(), i.getShape().end());
  std::swap(expected[expected.size() - 1], expected[expected.size() - 2]);
  if (o.getShape() != ArrayRef<int64_t>(expected) || i.getElementType() != o.getElementType())
    return emitOpError("result must be the input with its last two dims swapped");
  return success();
}

LogicalResult FusedChainOp::verifyRegions() {
  Block &b = getBody().front();
  Type t = getResult().getType();
  if (b.getNumArguments() != getInputs().size())
    return emitOpError("block argument count must equal input count");
  for (auto [arg, in] : llvm::zip(b.getArguments(), getInputs()))
    if (arg.getType() != in.getType() || in.getType() != t)
      return emitOpError("inputs, block arguments and result must share one tensor type");
  int softmaxes = 0;
  int64_t rank = cast<RankedTensorType>(t).getRank();
  for (Operation &op : b.without_terminator()) {
    if (!isa<AddOp, MulOp, ScaleOp, ReluOp, ExpOp, SoftmaxOp>(op))
      return emitOpError("unsupported op in chain: ") << op.getName();
    if (auto sm = dyn_cast<SoftmaxOp>(op)) {
      ++softmaxes;
      if (static_cast<int64_t>(sm.getAxis()) != rank - 1)
        return emitOpError("softmax in a chain must reduce the last axis");
    }
  }
  if (softmaxes > 1) return emitOpError("at most one softmax per chain");
  auto y = dyn_cast<YieldOp>(b.getTerminator());
  if (!y || y.getValue().getType() != t) return emitOpError("body must yield a value of the result type");
  return success();
}

LogicalResult FusedAttentionOp::verify() {
  auto t = cast<RankedTensorType>(getQ().getType());
  if (t.getRank() < 3) return emitOpError("rank must be >= 3 ([batch..., S, D])");
  int64_t d = t.getShape().back();
  if (d != 16 && d != 32 && d != 64 && d != 128)
    return emitOpError("head dim must be one of 16, 32, 64, 128");  // WHY: tl.dot needs power-of-two K >= 16
  return success();
}
