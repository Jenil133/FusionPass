// RUN: %fusionpass-opt %s --fp-fuse-chains | FileCheck %s
// REQUIRES: fusion-passes

// relu(scale(a + b)) -> exp : one fused chain with 4 body ops
// CHECK-LABEL: func.func @chain
// CHECK: "fp.fused_chain"
// CHECK: fp.add
// CHECK: fp.scale
// CHECK: fp.relu
// CHECK: fp.exp
// CHECK-NOT: "fp.fused_chain"
func.func @chain(%a: tensor<8x16xf32>, %b: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = fp.add %a, %b : tensor<8x16xf32>
  %1 = fp.scale %0 {factor = 2.000000e+00 : f32} : tensor<8x16xf32>
  %2 = fp.relu %1 : tensor<8x16xf32>
  %3 = fp.exp %2 : tensor<8x16xf32>
  return %3 : tensor<8x16xf32>
}

// Multi-use producer must NOT be fused into either consumer.
// CHECK-LABEL: func.func @multiuse
// CHECK-NOT: "fp.fused_chain"
func.func @multiuse(%a: tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
  %0 = fp.relu %a : tensor<4x4xf32>
  %1 = fp.exp %0 : tensor<4x4xf32>
  %2 = fp.scale %0 {factor = 3.000000e+00 : f32} : tensor<4x4xf32>
  return %1, %2 : tensor<4x4xf32>, tensor<4x4xf32>
}

// scale -> softmax(last axis) -> relu : one fused chain containing the softmax.
// CHECK-LABEL: func.func @softmax_chain
// CHECK: "fp.fused_chain"
// CHECK: fp.softmax
func.func @softmax_chain(%a: tensor<4x64xf32>) -> tensor<4x64xf32> {
  %0 = fp.scale %a {factor = 5.000000e-01 : f32} : tensor<4x64xf32>
  %1 = fp.softmax %0 {axis = 1 : i64} : tensor<4x64xf32>
  %2 = fp.relu %1 : tensor<4x64xf32>
  return %2 : tensor<4x64xf32>
}

// Softmax over a NON-last axis is not fusible.
// CHECK-LABEL: func.func @softmax_axis0
// CHECK-NOT: "fp.fused_chain"
func.func @softmax_axis0(%a: tensor<4x64xf32>) -> tensor<4x64xf32> {
  %0 = fp.scale %a {factor = 5.000000e-01 : f32} : tensor<4x64xf32>
  %1 = fp.softmax %0 {axis = 0 : i64} : tensor<4x64xf32>
  return %1 : tensor<4x64xf32>
}

// Duplicate external operand must be deduplicated: the fused op takes ONE input.
// CHECK-LABEL: func.func @dedupe
// CHECK: "fp.fused_chain"(%arg0)
func.func @dedupe(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = fp.mul %a, %a : tensor<4x4xf32>
  %1 = fp.relu %0 : tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}
