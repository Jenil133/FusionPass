// RUN: %fusionpass-opt %s | %fusionpass-opt | FileCheck %s
// CHECK-LABEL: func.func @rt
// CHECK: fp.scale
// CHECK: fp.softmax
// CHECK: fp.matmul
func.func @rt(%a: tensor<2x4x4xf32>, %b: tensor<2x4x4xf32>) -> tensor<2x4x4xf32> {
  %s = fp.scale %a {factor = 5.000000e-01 : f32} : tensor<2x4x4xf32>
  %p = fp.softmax %s {axis = 2 : i64} : tensor<2x4x4xf32>
  %o = fp.matmul %p, %b : (tensor<2x4x4xf32>, tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
  return %o : tensor<2x4x4xf32>
}
