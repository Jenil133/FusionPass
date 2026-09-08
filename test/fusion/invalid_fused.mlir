// RUN: %fusionpass-opt %s -split-input-file -verify-diagnostics
// REQUIRES: fusion-passes

func.func @bad_head_dim(%q: tensor<1x2x8x24xf32>) -> tensor<1x2x8x24xf32> {
  // expected-error@+1 {{head dim must be one of 16, 32, 64, 128}}
  %o = fp.fused_attention %q, %q, %q {scale = 1.0e-01 : f32} : tensor<1x2x8x24xf32>
  return %o : tensor<1x2x8x24xf32>
}

// -----

func.func @two_softmax(%a: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // expected-error@+1 {{at most one softmax per chain}}
  %r = "fp.fused_chain"(%a) ({
  ^bb0(%x: tensor<4x8xf32>):
    %s1 = fp.softmax %x {axis = 1 : i64} : tensor<4x8xf32>
    %s2 = fp.softmax %s1 {axis = 1 : i64} : tensor<4x8xf32>
    fp.yield %s2 : tensor<4x8xf32>
  }) : (tensor<4x8xf32>) -> tensor<4x8xf32>
  return %r : tensor<4x8xf32>
}

// -----

func.func @softmax_not_last(%a: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // expected-error@+1 {{softmax in a chain must reduce the last axis}}
  %r = "fp.fused_chain"(%a) ({
  ^bb0(%x: tensor<4x8xf32>):
    %s = fp.softmax %x {axis = 0 : i64} : tensor<4x8xf32>
    fp.yield %s : tensor<4x8xf32>
  }) : (tensor<4x8xf32>) -> tensor<4x8xf32>
  return %r : tensor<4x8xf32>
}
