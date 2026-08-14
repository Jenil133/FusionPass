// RUN: %fusionpass-opt %s -split-input-file -verify-diagnostics

func.func @bad_matmul(%a: tensor<2x3xf32>, %b: tensor<4x5xf32>) -> tensor<2x5xf32> {
  // expected-error@+1 {{contraction dimension mismatch}}
  %o = fp.matmul %a, %b : (tensor<2x3xf32>, tensor<4x5xf32>) -> tensor<2x5xf32>
  return %o : tensor<2x5xf32>
}

// -----

func.func @bad_axis(%a: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // expected-error@+1 {{axis must be in [0, rank)}}
  %o = fp.softmax %a {axis = 5 : i64} : tensor<2x3xf32>
  return %o : tensor<2x3xf32>
}
