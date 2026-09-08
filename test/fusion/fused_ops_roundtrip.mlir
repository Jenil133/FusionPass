// RUN: %fusionpass-opt %s | %fusionpass-opt | FileCheck %s
// REQUIRES: fusion-passes

// CHECK-LABEL: func.func @attn_rt
// CHECK: fp.fused_attention
func.func @attn_rt(%q: tensor<1x2x16x16xf32>, %k: tensor<1x2x16x16xf32>, %v: tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32> {
  %o = fp.fused_attention %q, %k, %v {scale = 1.250000e-01 : f32} : tensor<1x2x16x16xf32>
  return %o : tensor<1x2x16x16xf32>
}

// CHECK-LABEL: func.func @chain_rt
// CHECK: "fp.fused_chain"
// CHECK: fp.add
// CHECK: fp.yield
func.func @chain_rt(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %r = "fp.fused_chain"(%a, %b) ({
  ^bb0(%x: tensor<4x4xf32>, %y: tensor<4x4xf32>):
    %0 = fp.add %x, %y : tensor<4x4xf32>
    fp.yield %0 : tensor<4x4xf32>
  }) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %r : tensor<4x4xf32>
}
