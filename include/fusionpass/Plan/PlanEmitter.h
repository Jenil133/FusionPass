#pragma once
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/JSON.h"
namespace fusionpass {
class TritonModuleWriter;
/// Walks `func.func @main` in `module` and produces the plan.json value.
/// Fused ops become `"kind":"triton"` steps via `writer`; remaining fp ops stay `"kind":"torch"`.
mlir::LogicalResult emitPlan(mlir::ModuleOp module, TritonModuleWriter &writer,
                             llvm::json::Value &out);
} // namespace fusionpass
