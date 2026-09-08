// RUN: %fusionpass-opt %s --fp-fuse-attention | FileCheck %s
// REQUIRES: fusion-passes

// Head dim 24 is not in {16, 32, 64, 128} -> pattern must not fuse.
// CHECK-LABEL: func.func @attn_d24
// CHECK-NOT: fp.fused_attention
// CHECK: fp.softmax
func.func @attn_d24(%q: tensor<1x2x8x24xf32>, %k: tensor<1x2x8x24xf32>, %v: tensor<1x2x8x24xf32>) -> tensor<1x2x8x24xf32> {
  %kt = fp.transpose %k : tensor<1x2x8x24xf32> -> tensor<1x2x24x8xf32>
  %s = fp.matmul %q, %kt : (tensor<1x2x8x24xf32>, tensor<1x2x24x8xf32>) -> tensor<1x2x8x8xf32>
  %sc = fp.scale %s {factor = 2.500000e-01 : f32} : tensor<1x2x8x8xf32>
  %p = fp.softmax %sc {axis = 3 : i64} : tensor<1x2x8x8xf32>
  %o = fp.matmul %p, %v : (tensor<1x2x8x8xf32>, tensor<1x2x8x24xf32>) -> tensor<1x2x8x24xf32>
  return %o : tensor<1x2x8x24xf32>
}
