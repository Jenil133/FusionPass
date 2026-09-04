#pragma once
#include <memory>
#include "mlir/Pass/Pass.h"

namespace fusionpass {
/// Largest softmax row (last-dim size) allowed inside a fused row kernel.
/// WHY: the row kernel keeps one whole row in registers; larger rows would spill. (⚠️ tune)
constexpr int64_t kMaxRowSoftmaxCols = 8192;

std::unique_ptr<mlir::Pass> createFuseAttentionPass();
std::unique_ptr<mlir::Pass> createFuseChainsPass();
void registerFusionPasses();
} // namespace fusionpass
