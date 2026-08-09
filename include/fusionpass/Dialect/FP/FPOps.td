#ifndef FP_OPS
#define FP_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def FP_Dialect : Dialect {
  let name = "fp";
  let summary = "Fusionpass tensor-level dialect";
  let cppNamespace = "::fusionpass::fp";
}

class FP_Op<string mnemonic, list<Trait> traits = []> :
    Op<FP_Dialect, mnemonic, traits>;

// WHY static shapes: Triton kernels are generated with concrete sizes; dynamic dims are out of scope.
def FP_Tensor : StaticShapeTensorOf<[F16, F32]>;

class FP_BinaryOp<string mnemonic> : FP_Op<mnemonic, [Pure, SameOperandsAndResultType]> {
  let arguments = (ins FP_Tensor:$lhs, FP_Tensor:$rhs);
  let results = (outs FP_Tensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}
def FP_AddOp : FP_BinaryOp<"add">;
def FP_MulOp : FP_BinaryOp<"mul">;

class FP_UnaryOp<string mnemonic> : FP_Op<mnemonic, [Pure, SameOperandsAndResultType]> {
  let arguments = (ins FP_Tensor:$input);
  let results = (outs FP_Tensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
}
def FP_ReluOp : FP_UnaryOp<"relu">;
def FP_ExpOp  : FP_UnaryOp<"exp">;

def FP_ScaleOp : FP_Op<"scale", [Pure, SameOperandsAndResultType]> {
  let arguments = (ins FP_Tensor:$input, F32Attr:$factor);
  let results = (outs FP_Tensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
}

def FP_SoftmaxOp : FP_Op<"softmax", [Pure, SameOperandsAndResultType]> {
  let arguments = (ins FP_Tensor:$input, I64Attr:$axis);
  let results = (outs FP_Tensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
  let hasVerifier = 1;
}

def FP_MatmulOp : FP_Op<"matmul", [Pure]> {
  let arguments = (ins FP_Tensor:$lhs, FP_Tensor:$rhs);
  let results = (outs FP_Tensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)";
  let hasVerifier = 1;
}

def FP_TransposeOp : FP_Op<"transpose", [Pure]> {
  let arguments = (ins FP_Tensor:$input);
  let results = (outs FP_Tensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def FP_YieldOp : FP_Op<"yield", [Pure, Terminator, HasParent<"FusedChainOp">]> {
  let arguments = (ins FP_Tensor:$value);
  let assemblyFormat = "$value attr-dict `:` type($value)";
}

def FP_FusedChainOp : FP_Op<"fused_chain", [Pure, IsolatedFromAbove, SingleBlock]> {
  let summary = "A fused region of elementwise ops and at most one row softmax";
  let arguments = (ins Variadic<FP_Tensor>:$inputs);
  let results = (outs FP_Tensor:$result);
  let regions = (region SizedRegion<1>:$body);
  // WHY region verifier: it runs after the body's own structure (terminator) is verified.
  let hasRegionVerifier = 1;
}

def FP_FusedAttentionOp : FP_Op<"fused_attention",
    [Pure, AllTypesMatch<["q", "k", "v", "result"]>]> {
  let arguments = (ins FP_Tensor:$q, FP_Tensor:$k, FP_Tensor:$v, F32Attr:$scale);
  let results = (outs FP_Tensor:$result);
  let assemblyFormat = "$q `,` $k `,` $v attr-dict `:` type($result)";
  let hasVerifier = 1;
}

#endif // FP_OPS
