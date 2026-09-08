// RUN: %fusionpass-opt %s --fp-fuse-chains | FileCheck %s
// REQUIRES: fusion-passes

// Two last-axis softmaxes cannot share one fused_chain (row kernel supports one reduction).
// CHECK-LABEL: func.func @two_sm
// CHECK-NOT: "fp.fused_chain"
func.func @two_sm(%a: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = fp.softmax %a {axis = 1 : i64} : tensor<4x8xf32>
  %1 = fp.softmax %0 {axis = 1 : i64} : tensor<4x8xf32>
  return %1 : tensor<4x8xf32>
}
