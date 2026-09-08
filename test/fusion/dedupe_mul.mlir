// RUN: %fusionpass-opt %s --fp-fuse-chains | FileCheck %s
// REQUIRES: fusion-passes

// Duplicate external operand must be deduplicated: the fused op takes ONE input.
// CHECK-LABEL: func.func @dedupe
// CHECK: "fp.fused_chain"(%arg0)
func.func @dedupe(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = fp.mul %a, %a : tensor<4x4xf32>
  %1 = fp.relu %0 : tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}
