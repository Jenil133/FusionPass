// RUN: %fusionpass-opt %s --fp-fuse-attention | FileCheck %s
// REQUIRES: fusion-passes

// CHECK-LABEL: func.func @attn
// CHECK: fp.fused_attention
// CHECK-SAME: scale = 2.500000e-01
// CHECK-NOT: fp.softmax
// CHECK-NOT: fp.transpose
func.func @attn(%q: tensor<2x4x32x64xf16>, %k: tensor<2x4x32x64xf16>, %v: tensor<2x4x32x64xf16>) -> tensor<2x4x32x64xf16> {
  %kt = fp.transpose %k : tensor<2x4x32x64xf16> -> tensor<2x4x64x32xf16>
  %s = fp.matmul %q, %kt : (tensor<2x4x32x64xf16>, tensor<2x4x64x32xf16>) -> tensor<2x4x32x32xf16>
  %a = fp.scale %s {factor = 5.000000e-01 : f32} : tensor<2x4x32x32xf16>
  %b = fp.scale %a {factor = 5.000000e-01 : f32} : tensor<2x4x32x32xf16>
  %p = fp.softmax %b {axis = 3 : i64} : tensor<2x4x32x32xf16>
  %o = fp.matmul %p, %v : (tensor<2x4x32x32xf16>, tensor<2x4x32x64xf16>) -> tensor<2x4x32x64xf16>
  return %o : tensor<2x4x32x64xf16>
}

// Negative: the probabilities are also returned, so P must stay materialized -> no fusion.
// CHECK-LABEL: func.func @attn_multiuse
// CHECK-NOT: fp.fused_attention
// CHECK: fp.softmax
func.func @attn_multiuse(%q: tensor<1x2x16x16xf32>, %k: tensor<1x2x16x16xf32>, %v: tensor<1x2x16x16xf32>) -> (tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>) {
  %kt = fp.transpose %k : tensor<1x2x16x16xf32> -> tensor<1x2x16x16xf32>
  %s = fp.matmul %q, %kt : (tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
  %p = fp.softmax %s {axis = 3 : i64} : tensor<1x2x16x16xf32>
  %o = fp.matmul %p, %v : (tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>) -> tensor<1x2x16x16xf32>
  return %o, %p : tensor<1x2x16x16xf32>, tensor<1x2x16x16xf32>
}
