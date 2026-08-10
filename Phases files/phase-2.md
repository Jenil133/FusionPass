# PHASE 2 — Core Feature Build: Fusion Passes + Triton Codegen (fusionpass)

---

## 1. Project Context

**Project:** `fusionpass` — an MLIR-based compiler pass that fuses attention subgraphs and elementwise/softmax chains of a PyTorch model into single generated Triton kernels.

**What it is.** A PyTorch `nn.Module` (static shapes, f16/f32, inference only) is traced with `torch.fx` into textual MLIR in a small custom dialect called `fp`. A C++ tool, `fusionpass-compile`, parses that MLIR, runs fusion passes, and emits two artifacts: `plan.json` (an ordered list of steps) and `kernels.py` (generated Triton source). A Python runtime executes the plan: fused regions run as Triton kernels, everything else as plain PyTorch ops. A verification harness (Phase 3) proves numerical equivalence against eager PyTorch before any speedup is reported.

**Target user:** ML-systems engineers / students who want a working, readable example of an MLIR fusion pass with real GPU codegen.

**Problem solved:** each unfused op writes its result to GPU global memory and the next op reads it back. Fusing removes those round trips. For attention, fusion also avoids ever materializing the S×S score matrix.

**Whole-project scope boundaries (do not exceed):** static shapes only; dtypes f16 and f32 only; no broadcasting (elementwise operands must have identical shapes); inference only; no causal/attention masks; one output value per graph; single device.

### Tech stack (identical in all three phase files)

| Layer | Choice |
|---|---|
| Compiler | C++17, **LLVM/MLIR 18.1.x** (tag `llvmorg-18.1.8` — ⚠️ verify), CMake ≥ 3.20, Ninja, clang or gcc |
| Compiler tests | `lit` (pip) + `FileCheck` (LLVM install with `-DLLVM_INSTALL_UTILS=ON`) |
| Frontend / runtime | Python ≥ 3.10, PyTorch ≥ 2.2 (⚠️ verify), `torch.fx` + `ShapeProp` |
| Kernel codegen target | **Triton** — the version bundled with the PyTorch pip wheel (do NOT `pip install triton` separately) |
| GPU | NVIDIA GPU + CUDA for benchmarks (compute capability ≥ 8.0 recommended — ⚠️ verify). CPU correctness via `TRITON_INTERPRET=1` (⚠️ verify interpreter supports the constructs used) |
| Python tests | `pytest` |

**Stack decisions (why this over the alternative):**
- Custom `fp` dialect **over torch-mlir** — torch-mlir has heavy build/version coupling; a small dialect keeps the pass surface under our control.
- Emit Triton source **over lowering to NVVM** — Triton already does tiling, shared-memory staging, tensor-core codegen; our pass focuses on *fusion decisions*.
- `torch.fx` **over ONNX** — reference is the same `nn.Module`; `ShapeProp` provides shapes.
- JSON plan + Python runtime **over embedding Python in C++** — decoupled, independently testable.
- Greedy pattern rewrites **over a custom graph algorithm** — idiomatic MLIR, small code.
- Hand-written `PassWrapper` passes **over TableGen pass definitions** — fewer generated-code API pitfalls across LLVM versions.

### Naming conventions (must match across all phases)
- Dialect `fp`, C++ namespace `fusionpass::fp`; libraries `MLIRFPDialect`, `MLIRFPPlan`, `MLIRFPTransforms`.
- Tools: `fusionpass-opt`, `fusionpass-compile`. Python package `fusionpass` under `python/`. Entry function `func.func @main`.
- Plan value names: args `arg0..`; op results `v0, v1, ...` in program order.
- Generated kernels: `fp_chain_<N>` (one per fused chain, N from 0) and one shared `fp_attention`.
- Out-dir files: `plan.json`, `kernels.py`, `optimized.mlir`.
- Pass flags: `--fp-fuse-attention`, `--fp-fuse-chains`. Env vars: `LLVM_PREFIX`, `FUSIONPASS_BUILD_DIR`, `FUSIONPASS_COMPILE`.

---

## 2. Architecture Overview

```
 PyTorch nn.Module
      │  torch.fx symbolic_trace + ShapeProp          python/fusionpass/frontend.py      (Phase 1, done)
      ▼
 model.mlir   (func.func @main + fp dialect ops)
      │  fusionpass-compile   (C++ / MLIR)
      │     parse → verify → fp-fuse-attention → fp-fuse-chains → emit plan + Triton src ◄ PHASE 2 (passes + emitter)
      ▼
 plan.json  +  kernels.py (Triton source)
      │  python/fusionpass/runtime.py                                                      ◄ PHASE 2 (launch triton steps)
      ▼
 PyTorch ops (unfused)  |  Triton kernels (fused)  → CUDA GPU (or Triton interpreter on CPU)
      │
 verify.py (equivalence gate) → bench.py (speedup)                                         ◄ Phase 3
```

**This phase's place:** the heart of the project. After Phase 2, a traced model's attention block becomes ONE `fp_attention` Triton kernel, and runs of elementwise/softmax ops become ONE generated kernel each, with correct results.

**Final project layout (reference):**
```
fusionpass/
├── CMakeLists.txt  README.md  pyproject.toml
├── include/fusionpass/Dialect/FP/{CMakeLists.txt,FPOps.td,FPDialect.h,FPOps.h}
├── include/fusionpass/Transforms/Passes.h
├── include/fusionpass/Plan/{PlanEmitter.h,TritonEmitter.h}
├── lib/Dialect/FP/{CMakeLists.txt,FPDialect.cpp,FPOps.cpp}
├── lib/Transforms/{CMakeLists.txt,FuseAttention.cpp,FuseChains.cpp,Passes.cpp}
├── lib/Plan/{CMakeLists.txt,PlanEmitter.cpp,TritonEmitter.cpp}
├── tools/{CMakeLists.txt,fusionpass-opt/fusionpass-opt.cpp,fusionpass-compile/fusionpass-compile.cpp}
├── python/fusionpass/{__init__.py,frontend.py,runtime.py,api.py,verify.py,bench.py,traffic.py,__main__.py}
├── test/{lit.cfg.py,*.mlir}
├── tests/{conftest.py,test_*.py}
└── examples/ docs/ .github/workflows/   (Phase 3)
```

---

## 3. Current State

The Phase 1 repository exists on disk and builds. **Rely on it; do not rebuild it.** Read the files you modify.

- **Build:** LLVM/MLIR 18.1.x installed at `$LLVM_PREFIX`; project configured in `build/`; `cmake --build build` succeeds.
- **Dialect `fp`** (`include/fusionpass/Dialect/FP/FPOps.td`, `lib/Dialect/FP/{FPDialect,FPOps}.cpp`): ops `fp.add, fp.mul, fp.scale{factor:f32}, fp.relu, fp.exp, fp.softmax{axis:i64}, fp.matmul, fp.transpose` with verifiers. Tensor type constraint `FP_Tensor = StaticShapeTensorOf<[F16, F32]>`. C++ classes: `AddOp, MulOp, ScaleOp, ReluOp, ExpOp, SoftmaxOp, MatmulOp, TransposeOp`; accessors `getLhs/getRhs/getInput/getFactor/getAxis`.
- **Library targets:** `MLIRFPDialect` (`lib/Dialect/FP`), `MLIRFPPlan` (`lib/Plan`). Generated-code target `MLIRFPOpsIncGen`.
- **`fusionpass-opt`** (`tools/fusionpass-opt/fusionpass-opt.cpp`): registers `func` + `fp` dialects; has a commented placeholder `// Phase 2 adds: fusionpass::registerFusionPasses();`.
- **`fusionpass-compile`** (`tools/fusionpass-compile/fusionpass-compile.cpp`): `fusionpass-compile <in.mlir> -o <outdir> [--pipeline none]`; parses, verifies, calls `fusionpass::emitPlan(module, json::Value&)`, writes `plan.json`. `--pipeline` currently errors for anything but `none`.
- **`PlanEmitter`** (`include/fusionpass/Plan/PlanEmitter.h`, `lib/Plan/PlanEmitter.cpp`): `mlir::LogicalResult fusionpass::emitPlan(ModuleOp, llvm::json::Value&)`. Walks `@main`, names args `argN` and results `vN`, emits only `"kind":"torch"` steps via a `TypeSwitch` (unknown op → error).
- **Python (`python/fusionpass/`):** `frontend.py` (`trace_to_mlir`, `UnsupportedOp`), `runtime.py` (`execute_plan(plan, args, kernels=None)`, `_run_torch`, raises `NotImplementedError` for non-torch steps), `api.py` (`compile_model(model, example_inputs, *, fuse=False, workdir=None)`; `fuse=True` raises `NotImplementedError`; `CompiledModel(plan, captured, workdir, mlir_text)`), `__init__.py` exporting `compile_model, CompiledModel, UnsupportedOp`.
- **Tests:** `test/lit.cfg.py` (+ `roundtrip.mlir`, `invalid.mlir`; substitution `%fusionpass-opt`), `tests/conftest.py` (skips if compiler missing), `tests/test_frontend.py`, `tests/test_e2e.py`.

---

## 4. Phase Goal & Scope

**Goal:** implement fusion. After this phase, `compile_model(..., fuse=True)` produces a plan where attention subgraphs are single `fp_attention` kernel launches and elementwise/softmax chains are single generated-kernel launches, and results match eager PyTorch within tolerance on CPU (interpreter) and GPU.

**In scope**
- New ops: `fp.yield`, `fp.fused_chain` (region op), `fp.fused_attention`, with verifiers.
- Pass `fp-fuse-attention`: matches `matmul(Q, transpose(K)) → scale* → softmax(last axis) → matmul(·, V)`.
- Pass `fp-fuse-chains`: greedily fuses single-use elementwise ops and (at most one) last-axis row softmax into `fp.fused_chain`.
- Triton source emitter (`TritonEmitter`): generated flat-elementwise kernel, row-softmax kernel, and the FlashAttention-style tiled online-softmax attention kernel.
- `PlanEmitter` extended with `"triton"` steps; `fusionpass-compile --pipeline fuse` writes `kernels.py` and `optimized.mlir`.
- Runtime: load `kernels.py`, launch kernels (flat / rows / attention).
- `compile_model(fuse=True)`.
- lit tests per pass (positive + negative) and pytest end-to-end tests (CPU interpreter; GPU when available).

**Out of scope (deferred to Phase 3)**
- Equivalence-gate harness with tolerance policy, benchmarking, memory-traffic estimate, CLI demos (`python -m fusionpass`), plan schema validator, fallback/edge-case test matrix, README/docs/CI, deployment.

---

## 5. Prerequisites & Setup

```bash
source .venv/bin/activate
export LLVM_PREFIX=$HOME/llvm-install
export FUSIONPASS_BUILD_DIR=build
export FUSIONPASS_COMPILE=$PWD/build/bin/fusionpass-compile
pip install -e .[dev]                 # torch wheel already includes Triton
python -c "import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.is_available())"
```
- **No GPU:** tests run in the Triton interpreter. `tests/conftest.py` (modified here) sets `TRITON_INTERPRET=1` automatically when CUDA is unavailable. This env var **must be set before `triton` is imported**.
- **GPU:** nothing extra; kernels compile on first launch.
- No secrets or external accounts are used.

---

## 6. File Structure for This Phase

```
fusionpass/
├── include/fusionpass/
│   ├── Dialect/FP/FPOps.td                          MODIFIED  (+yield, fused_chain, fused_attention)
│   ├── Transforms/Passes.h                          NEW
│   └── Plan/
│       ├── PlanEmitter.h                            MODIFIED  (new signature)
│       └── TritonEmitter.h                          NEW
├── lib/
│   ├── CMakeLists.txt                               MODIFIED  (+Transforms)
│   ├── Dialect/FP/FPOps.cpp                         MODIFIED  (+verifiers)
│   ├── Transforms/{CMakeLists.txt,FuseAttention.cpp,FuseChains.cpp,Passes.cpp}   NEW
│   └── Plan/{CMakeLists.txt,PlanEmitter.cpp,TritonEmitter.cpp}   MODIFIED / MODIFIED / NEW
├── tools/
│   ├── CMakeLists.txt                               MODIFIED  (link MLIRFPTransforms)
│   ├── fusionpass-opt/fusionpass-opt.cpp            MODIFIED  (register passes)
│   └── fusionpass-compile/fusionpass-compile.cpp    MODIFIED  (--pipeline fuse, kernels.py, optimized.mlir)
├── python/fusionpass/{runtime.py,api.py}            MODIFIED
├── test/{fuse_attention.mlir,fuse_chains.mlir}      NEW
└── tests/{conftest.py (MODIFIED), test_fusion_e2e.py (NEW)}
```

---

## 7. Data Model / Schema

No database. Schema = new dialect ops + extended `plan.json`.

### A. New ops (added to `FPOps.td`)

| Op | Form | Semantics / invariants |
|---|---|---|
| `fp.yield` | `fp.yield %v : T` | Terminator of a `fused_chain` body; parent must be `FusedChainOp`. |
| `fp.fused_chain` | generic form: `%r = "fp.fused_chain"(%a, %b) ({ ^bb0(%x: T, %y: T): ...; fp.yield %z : T }) : (T, T) -> T` | `Pure`, `IsolatedFromAbove`, single block, variadic inputs. **All inputs, block args and result share ONE tensor type T.** Body contains only `add, mul, scale, relu, exp, softmax` and the yield. **At most one `softmax`**, and it must reduce the last axis. |
| `fp.fused_attention` | `%o = fp.fused_attention %q, %k, %v {scale = 1.25e-01 : f32} : T` | `Pure`. `q,k,v,result` all type T (`AllTypesMatch`). Rank ≥ 3 (leading dims are batch dims, then `[S, D]`). `D = last dim ∈ {16, 32, 64, 128}`. Semantics: `softmax(scale · Q·Kᵀ, axis=-1) · V`. |

`fused_chain` intentionally has **no custom assembly format** (it prints/parses in MLIR generic form) — this avoids hand-written region parsers.

### B. `plan.json` schema (version 1, extended)
```json
{
  "version": 1,
  "inputs":  [ {"name": "arg0", "shape": [2,4,32,64], "dtype": "f16"} ],
  "steps": [
    {"kind": "torch",  "op": "matmul", "operands": ["arg0","v0"], "result": "v1",
     "shape": [..], "dtype": "f16", "attrs": {}},

    {"kind": "triton", "kernel": "fp_chain_0", "launch": "flat",
     "operands": ["arg0","arg1"], "result": "v2", "shape": [8,16], "dtype": "f32",
     "attrs": {"block": 1024}},

    {"kind": "triton", "kernel": "fp_chain_1", "launch": "rows",
     "operands": ["v2"], "result": "v3", "shape": [8,128], "dtype": "f32",
     "attrs": {"block_n": 128, "num_warps": 4}},

    {"kind": "triton", "kernel": "fp_attention", "launch": "attention",
     "operands": ["arg0","arg1","arg2"], "result": "v4", "shape": [2,4,32,64], "dtype": "f16",
     "attrs": {"scale": 0.125, "block_m": 64, "block_n": 64, "num_warps": 4, "num_stages": 2}}
  ],
  "outputs": ["v4"]
}
```
- `launch` ∈ `flat` | `rows` | `attention`.
- `rows`: `n_cols = shape[-1]`, `rows = numel / n_cols`, `block_n = next_pow2(n_cols)`; `num_warps = 4` if `block_n ≤ 2048` else `8`.
- `flat`: `n_elements = numel`, `block = 1024`.
- Attention tuning constants (`64/64/4/2`) are starting points (⚠️ tune per GPU).

---

## 8. Implementation Steps

### Step 1 — Extend the dialect: yield, fused_chain, fused_attention
Append to `include/fusionpass/Dialect/FP/FPOps.td` (before `#endif`):
```tablegen
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
```
Append to `lib/Dialect/FP/FPOps.cpp`:
```cpp
LogicalResult FusedChainOp::verifyRegions() {
  Block &b = getBody().front();
  Type t = getResult().getType();
  if (b.getNumArguments() != getInputs().size())
    return emitOpError("block argument count must equal input count");
  for (auto [arg, in] : llvm::zip(b.getArguments(), getInputs()))
    if (arg.getType() != in.getType() || in.getType() != t)
      return emitOpError("inputs, block arguments and result must share one tensor type");
  int softmaxes = 0;
  int64_t rank = cast<RankedTensorType>(t).getRank();
  for (Operation &op : b.without_terminator()) {
    if (!isa<AddOp, MulOp, ScaleOp, ReluOp, ExpOp, SoftmaxOp>(op))
      return emitOpError("unsupported op in chain: ") << op.getName();
    if (auto sm = dyn_cast<SoftmaxOp>(op)) {
      ++softmaxes;
      if (sm.getAxis() != rank - 1) return emitOpError("softmax in a chain must reduce the last axis");
    }
  }
  if (softmaxes > 1) return emitOpError("at most one softmax per chain");
  auto y = dyn_cast<YieldOp>(b.getTerminator());
  if (!y || y.getValue().getType() != t) return emitOpError("body must yield a value of the result type");
  return success();
}

LogicalResult FusedAttentionOp::verify() {
  auto t = cast<RankedTensorType>(getQ().getType());
  if (t.getRank() < 3) return emitOpError("rank must be >= 3 ([batch..., S, D])");
  int64_t d = t.getShape().back();
  if (d != 16 && d != 32 && d != 64 && d != 128)
    return emitOpError("head dim must be one of 16, 32, 64, 128");  // WHY: tl.dot needs power-of-two K >= 16
  return success();
}
```
**Verify:** `ninja -C build MLIRFPDialect` compiles.

### Step 2 — Pass declarations and CMake
`include/fusionpass/Transforms/Passes.h`:
```cpp
#pragma once
#include <memory>
#include "mlir/Pass/Pass.h"

namespace fusionpass {
/// Largest softmax row (last-dim size) allowed inside a fused row kernel.
/// WHY: the row kernel keeps one whole row in registers; larger rows would spill. (⚠️ tune)
constexpr int64_t kMaxRowSoftmaxCols = 8192;

std::unique_ptr<mlir::Pass> createFuseAttentionPass();
std::unique_ptr<mlir::Pass> createFuseChainsPass();
void registerFusionPasses();
} // namespace fusionpass
```
`lib/Transforms/CMakeLists.txt`:
```cmake
add_mlir_library(MLIRFPTransforms
  FuseAttention.cpp
  FuseChains.cpp
  Passes.cpp
  DEPENDS MLIRFPOpsIncGen
  LINK_LIBS PUBLIC MLIRFPDialect MLIRFuncDialect MLIRIR MLIRPass MLIRTransformUtils)
```
`lib/CMakeLists.txt`: add `add_subdirectory(Transforms)`.
In `tools/CMakeLists.txt`: add `MLIRFPTransforms` to the link libs of **both** tools.
**Verify:** configure still succeeds after Steps 3–5 add the sources.

### Step 3 — `fp-fuse-attention` pass
`lib/Transforms/FuseAttention.cpp`:
```cpp
#include "fusionpass/Dialect/FP/FPOps.h"
#include "fusionpass/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
using namespace mlir;
using namespace fusionpass::fp;

namespace {
/// Root pattern at the SECOND matmul (P @ V), then walk backwards.
/// matmul(softmax(scale*(matmul(Q, transpose(K)))), V)
struct FuseAttentionPattern : OpRewritePattern<MatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MatmulOp pv, PatternRewriter &rw) const override {
    Value p = pv.getLhs(), v = pv.getRhs();
    auto sm = p.getDefiningOp<SoftmaxOp>();
    // WHY single-use everywhere: if an intermediate (e.g. the probabilities P) is used elsewhere,
    // fusing would force us to still materialize it, defeating the point -> do not fuse.
    if (!sm || !p.hasOneUse()) return failure();
    auto st = cast<RankedTensorType>(sm.getType());
    if (sm.getAxis() != st.getRank() - 1) return failure();

    double scale = 1.0;
    Value cur = sm.getInput();
    while (auto sc = cur.getDefiningOp<ScaleOp>()) {   // zero or more scales; combine multiplicatively
      if (!cur.hasOneUse()) return failure();
      scale *= (double)sc.getFactor().convertToFloat();
      cur = sc.getInput();
    }
    auto qk = cur.getDefiningOp<MatmulOp>();
    if (!qk || !cur.hasOneUse()) return failure();
    auto kt = qk.getRhs().getDefiningOp<TransposeOp>();
    if (!kt || !qk.getRhs().hasOneUse()) return failure();

    Value q = qk.getLhs(), k = kt.getInput();
    // Self-attention shape contract: Q, K, V and output all identical, rank >= 3, D in {16,32,64,128}.
    auto qt = cast<RankedTensorType>(q.getType());
    if (qt != cast<RankedTensorType>(k.getType()) || qt != cast<RankedTensorType>(v.getType()) ||
        qt != cast<RankedTensorType>(pv.getType()) || qt.getRank() < 3)
      return failure();
    int64_t d = qt.getShape().back();
    if (d != 16 && d != 32 && d != 64 && d != 128) return failure();

    auto fused = rw.create<FusedAttentionOp>(pv.getLoc(), pv.getType(), q, k, v,
                                             rw.getF32FloatAttr((float)scale));
    rw.replaceOp(pv, fused.getResult());   // the dead softmax/scale/matmul/transpose are erased by the driver
    return success();
  }
};

struct FuseAttentionPass
    : public PassWrapper<FuseAttentionPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseAttentionPass)
  StringRef getArgument() const final { return "fp-fuse-attention"; }
  StringRef getDescription() const final { return "Fuse attention subgraphs into fp.fused_attention"; }
  void getDependentDialects(DialectRegistry &r) const override { r.insert<FPDialect>(); }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseAttentionPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> fusionpass::createFuseAttentionPass() { return std::make_unique<FuseAttentionPass>(); }
```
**Verify:** compiles; lit test in Step 8 passes.

### Step 4 — `fp-fuse-chains` pass
`lib/Transforms/FuseChains.cpp`:
```cpp
#include "fusionpass/Dialect/FP/FPOps.h"
#include "fusionpass/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
using namespace mlir;
using namespace fusionpass::fp;

namespace {
bool isChainable(Operation *op) { return isa<AddOp, MulOp, ScaleOp, ReluOp, ExpOp, SoftmaxOp, FusedChainOp>(op); }

/// A plain softmax is fusible only if it reduces the last axis and the row fits the row-kernel cap.
bool isFusible(Operation *op) {
  if (auto sm = dyn_cast<SoftmaxOp>(op)) {
    auto t = cast<RankedTensorType>(sm.getType());
    return sm.getAxis() == t.getRank() - 1 && t.getShape().back() <= fusionpass::kMaxRowSoftmaxCols;
  }
  return isChainable(op);
}

int softmaxCount(Operation *op) {
  if (isa<SoftmaxOp>(op)) return 1;
  int n = 0;
  if (auto f = dyn_cast<FusedChainOp>(op))
    for (Operation &o : f.getBody().front().without_terminator()) n += isa<SoftmaxOp>(o);
  return n;
}

/// Clone `op` (plain op or the body of a fused_chain) into the builder's current block.
/// `outer` maps values visible outside to values inside the new body. Returns the value
/// that represents op's single result inside the new body.
Value inlineOp(OpBuilder &b, Operation *op, IRMapping &outer) {
  if (auto f = dyn_cast<FusedChainOp>(op)) {
    IRMapping inner;
    Block &body = f.getBody().front();
    for (auto [arg, operand] : llvm::zip(body.getArguments(), f.getInputs()))
      inner.map(arg, outer.lookup(operand));
    for (Operation &o : body.without_terminator()) b.clone(o, inner);
    return inner.lookup(cast<YieldOp>(body.getTerminator()).getValue());
  }
  return b.clone(*op, outer)->getResult(0);
}

struct FuseChainPattern : RewritePattern {
  FuseChainPattern(MLIRContext *ctx) : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *c, PatternRewriter &rw) const override {
    if (!isChainable(c) || !isFusible(c)) return failure();
    for (OpOperand &use : c->getOpOperands()) {
      Value pv = use.get();
      Operation *p = pv.getDefiningOp();
      if (!p || !isChainable(p) || !isFusible(p)) continue;
      if (!pv.hasOneUse()) continue;               // WHY: a multi-use producer must stay materialized
      if (p->getBlock() != c->getBlock()) continue;
      if (softmaxCount(p) + softmaxCount(c) > 1) continue;   // row kernel supports one reduction

      // External inputs of the merged region: producer operands + consumer operands (minus pv), deduped.
      SmallVector<Value> externals;
      auto addExt = [&](Value x) { if (!llvm::is_contained(externals, x)) externals.push_back(x); };
      for (Value x : p->getOperands()) addExt(x);
      for (Value x : c->getOperands()) if (x != pv) addExt(x);

      Type resTy = c->getResult(0).getType();
      rw.setInsertionPoint(c);
      auto fused = rw.create<FusedChainOp>(c->getLoc(), resTy, externals);
      SmallVector<Type> argTys;
      SmallVector<Location> locs;
      for (Value x : externals) { argTys.push_back(x.getType()); locs.push_back(c->getLoc()); }
      Block *blk = rw.createBlock(&fused.getBody(), fused.getBody().end(), argTys, locs);

      IRMapping map;
      for (auto [i, x] : llvm::enumerate(externals)) map.map(x, blk->getArgument(i));
      rw.setInsertionPointToStart(blk);
      Value pOut = inlineOp(rw, p, map);
      map.map(pv, pOut);
      Value cOut = inlineOp(rw, c, map);
      rw.create<YieldOp>(c->getLoc(), cOut);

      rw.replaceOp(c, fused.getResult());          // p becomes dead; the greedy driver erases it
      return success();
    }
    return failure();
  }
};

struct FuseChainsPass : public PassWrapper<FuseChainsPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseChainsPass)
  StringRef getArgument() const final { return "fp-fuse-chains"; }
  StringRef getDescription() const final { return "Fuse elementwise/softmax chains into fp.fused_chain"; }
  void getDependentDialects(DialectRegistry &r) const override { r.insert<FPDialect>(); }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseChainPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> fusionpass::createFuseChainsPass() { return std::make_unique<FuseChainsPass>(); }
```
Why this terminates: every successful rewrite replaces two ops by one, strictly reducing op count.
**Verify:** compiles; lit test in Step 8.

### Step 5 — Pass registration
`lib/Transforms/Passes.cpp`:
```cpp
#include "fusionpass/Transforms/Passes.h"
#include "mlir/Pass/PassRegistry.h"

void fusionpass::registerFusionPasses() {
  // WHY registerPass(lambda): the pass's argument/description come from getArgument()/getDescription().
  mlir::registerPass([] { return createFuseAttentionPass(); });
  mlir::registerPass([] { return createFuseChainsPass(); });
}
```
(⚠️ verify `registerPass` signature in the pinned tag: `void registerPass(const PassAllocatorFunction &)` in `mlir/Pass/PassRegistry.h`.)
In `fusionpass-opt.cpp`: `#include "fusionpass/Transforms/Passes.h"`, and replace the placeholder comment with `fusionpass::registerFusionPasses();` **before** `MlirOptMain`.
**Verify:** `build/bin/fusionpass-opt --help | grep fp-fuse` lists both passes.

### Step 6 — Triton source emitter
`include/fusionpass/Plan/TritonEmitter.h`:
```cpp
#pragma once
#include <string>
#include "fusionpass/Dialect/FP/FPOps.h"
#include "llvm/Support/raw_ostream.h"

namespace fusionpass {
class TritonModuleWriter {
public:
  /// Appends a kernel for `op`; returns its name ("fp_chain_<N>").
  std::string addChainKernel(fp::FusedChainOp op);
  /// Ensures the shared FlashAttention-style kernel is present; returns "fp_attention".
  std::string addAttentionKernel();
  /// Writes the complete kernels.py source.
  void write(llvm::raw_ostream &os) const;
private:
  int chainCount = 0;
  bool hasAttention = false;
  std::string chains;   // concatenated chain-kernel sources
};
} // namespace fusionpass
```
`lib/Plan/TritonEmitter.cpp`:
```cpp
#include "fusionpass/Plan/TritonEmitter.h"
#include "llvm/ADT/SmallString.h"
using namespace mlir;
using namespace fusionpass;
using namespace fusionpass::fp;

static std::string pyFloat(llvm::APFloat f) {
  llvm::SmallString<32> s;
  f.toString(s);               // e.g. "0.125" or "1.0E-1"; both are valid Python float literals
  return s.str().str();
}

std::string TritonModuleWriter::addChainKernel(FusedChainOp op) {
  std::string name = "fp_chain_" + std::to_string(chainCount++);
  Block &body = op.getBody().front();
  bool rows = llvm::any_of(body.without_terminator(), [](Operation &o) { return isa<SoftmaxOp>(o); });

  std::string src;
  llvm::raw_string_ostream os(src);
  os << "@triton.jit\ndef " << name << "(";
  for (unsigned i = 0; i < body.getNumArguments(); ++i) os << "x" << i << ", ";
  os << "out, " << (rows ? "n_cols, BLOCK_N: tl.constexpr" : "n_elements, BLOCK: tl.constexpr") << "):\n";
  if (rows) {
    // One program per row; the whole row lives in registers (BLOCK_N = next_pow2(n_cols)).
    os << "    row = tl.program_id(0)\n"
          "    cols = tl.arange(0, BLOCK_N)\n"
          "    mask = cols < n_cols\n"
          "    offs = row.to(tl.int64) * n_cols + cols\n";
  } else {
    os << "    pid = tl.program_id(0)\n"
          "    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)\n"
          "    mask = offs < n_elements\n";
  }

  DenseMap<Value, std::string> v;
  for (unsigned i = 0; i < body.getNumArguments(); ++i) {
    v[body.getArgument(i)] = "a" + std::to_string(i);
    // WHY float32 compute: matches eager accuracy for fp16 inputs; cast back on store.
    os << "    a" << i << " = tl.load(x" << i << " + offs, mask=mask, other=0.0).to(tl.float32)\n";
  }
  int k = 0;
  std::string last;
  for (Operation &o : body.without_terminator()) {
    std::string t = "t" + std::to_string(k++);
    auto in = [&](unsigned i) { return v[o.getOperand(i)]; };
    if (isa<AddOp>(o))       os << "    " << t << " = " << in(0) << " + " << in(1) << "\n";
    else if (isa<MulOp>(o))  os << "    " << t << " = " << in(0) << " * " << in(1) << "\n";
    else if (auto s = dyn_cast<ScaleOp>(o))
                             os << "    " << t << " = " << in(0) << " * " << pyFloat(s.getFactor()) << "\n";
    else if (isa<ReluOp>(o)) os << "    " << t << " = tl.maximum(" << in(0) << ", 0.0)\n";
    else if (isa<ExpOp>(o))  os << "    " << t << " = tl.exp(" << in(0) << ")\n";
    else if (isa<SoftmaxOp>(o)) {
      // Numerically stable softmax over the row; padded lanes forced to -inf so they contribute 0.
      os << "    " << t << "_m = tl.where(mask, " << in(0) << ", float(\"-inf\"))\n"
         << "    " << t << "_e = tl.exp(" << t << "_m - tl.max(" << t << "_m, axis=0))\n"
         << "    " << t << " = " << t << "_e / tl.sum(" << t << "_e, axis=0)\n";
    }
    v[o.getResult(0)] = t;
    last = t;
  }
  Value y = cast<YieldOp>(body.getTerminator()).getValue();
  os << "    tl.store(out + offs, " << v[y] << ".to(out.dtype.element_ty), mask=mask)\n\n\n";
  os.flush();
  chains += src;
  return name;
}

std::string TritonModuleWriter::addAttentionKernel() { hasAttention = true; return "fp_attention"; }

// FlashAttention-style tiled attention: one program per (BLOCK_M query rows, batch*head).
// K/V are streamed in BLOCK_N tiles; softmax is computed ONLINE (running max m_i and sum l_i),
// so the S x S score matrix is never written to global memory. Triton stages tiles through
// shared memory when lowering tl.dot (num_stages controls software pipelining).
static const char *kAttentionKernel = R"PY(@triton.jit
def fp_attention(Q, K, V, O, S, scale,
                 HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    base = pid_bh.to(tl.int64) * S * HEAD_DIM          # tensors are contiguous [BH, S, D]
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
                mask=offs_m[:, None] < S, other=0.0)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    for start_n in range(0, S, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        # K tile loaded already transposed: shape [HEAD_DIM, BLOCK_N]
        k = tl.load(K + base + offs_n[None, :] * HEAD_DIM + offs_d[:, None],
                    mask=offs_n[None, :] < S, other=0.0)
        s = tl.dot(q, k) * scale                       # [BLOCK_M, BLOCK_N] in fp32
        s = tl.where(offs_n[None, :] < S, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)                    # rescale factor for previous partial results
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(V + base + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
                    mask=offs_n[:, None] < S, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(O + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
             acc.to(O.dtype.element_ty), mask=offs_m[:, None] < S)
)PY";

void TritonModuleWriter::write(llvm::raw_ostream &os) const {
  os << "# GENERATED by fusionpass-compile. Do not edit.\n"
        "import triton\nimport triton.language as tl\n\n\n";
  os << chains;
  if (hasAttention) os << kAttentionKernel;
}
```
Register `TritonEmitter.cpp` in `lib/Plan/CMakeLists.txt` (`add_mlir_library(MLIRFPPlan PlanEmitter.cpp TritonEmitter.cpp ...)`).
**Verify:** `ninja -C build MLIRFPPlan` compiles. Padded query rows (`offs_m ≥ S`) load zeros, so they stay finite and are masked on store. If `range(0, S, BLOCK_N)` with a runtime `S` is rejected in the interpreter, switch the loop to `tl.range(0, S, BLOCK_N)` (⚠️ verify).

### Step 7 — Extend `PlanEmitter` and `fusionpass-compile`
`PlanEmitter.h`: change the signature (and forward-declare `TritonModuleWriter`):
```cpp
namespace fusionpass {
class TritonModuleWriter;
mlir::LogicalResult emitPlan(mlir::ModuleOp module, TritonModuleWriter &writer, llvm::json::Value &out);
}
```
`PlanEmitter.cpp` — add `#include "fusionpass/Plan/TritonEmitter.h"` and `#include "llvm/Support/MathExtras.h"`; thread `writer` through; replace the `TypeSwitch` so that, **before** it, fused ops are handled as `triton` steps:
```cpp
// inside the loop, after computing `operands`/`res` (restructure so these are computed first):
if (auto fc = dyn_cast<FusedChainOp>(&op)) {
  auto rt = cast<RankedTensorType>(fc.getType());
  std::string kernel = writer.addChainKernel(fc);
  bool rows = llvm::any_of(fc.getBody().front().without_terminator(), [](Operation &o) { return isa<SoftmaxOp>(o); });
  json::Object a;
  if (rows) {
    uint64_t bn = llvm::PowerOf2Ceil((uint64_t)rt.getShape().back());
    a["block_n"] = (int64_t)bn;
    a["num_warps"] = (int64_t)(bn <= 2048 ? 4 : 8);
  } else {
    a["block"] = (int64_t)1024;
  }
  steps.push_back(tritonStep(kernel, rows ? "rows" : "flat", operands, res, rt, std::move(a)));
  continue;
}
if (auto fa = dyn_cast<FusedAttentionOp>(&op)) {
  auto rt = cast<RankedTensorType>(fa.getType());
  std::string kernel = writer.addAttentionKernel();
  json::Object a;
  a["scale"] = (double)fa.getScale().convertToFloat();
  a["block_m"] = (int64_t)64; a["block_n"] = (int64_t)64;
  a["num_warps"] = (int64_t)4; a["num_stages"] = (int64_t)2;
  steps.push_back(tritonStep(kernel, "attention", operands, res, rt, std::move(a)));
  continue;
}
```
with helper:
```cpp
static json::Object tritonStep(const std::string &kernel, const char *launch, json::Array operands,
                               const std::string &res, RankedTensorType rt, json::Object attrs) {
  return json::Object{{"kind", "triton"}, {"kernel", kernel}, {"launch", launch}, {"operands", std::move(operands)},
                      {"result", res}, {"shape", shapeArr(rt)}, {"dtype", dtypeStr(rt.getElementType())},
                      {"attrs", std::move(attrs)}};
}
```
Note: result names `vN` must still be assigned in program order for fused ops (increment `next` exactly as for torch ops).

`fusionpass-compile.cpp` changes:
```cpp
#include "fusionpass/Plan/TritonEmitter.h"
#include "fusionpass/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
// ...
static cl::opt<bool> emitMlir("emit-mlir", cl::desc("also write optimized.mlir"), cl::init(true));

// after verify(), replace the pipeline check:
if (pipeline == "fuse") {
  PassManager pm(&ctx);
  pm.addNestedPass<func::FuncOp>(fusionpass::createFuseAttentionPass());  // attention FIRST so its scale/softmax are absorbed
  pm.addNestedPass<func::FuncOp>(fusionpass::createFuseChainsPass());
  if (failed(pm.run(*module))) return 1;
} else if (pipeline != "none") { /* unchanged error, exit 2 */ }

fusionpass::TritonModuleWriter writer;
llvm::json::Value plan(nullptr);
if (failed(fusionpass::emitPlan(*module, writer, plan))) return 1;
// write plan.json (unchanged), then:
{ std::error_code e2; llvm::raw_fd_ostream ks(outDir + "/kernels.py", e2); if (e2) { llvm::errs() << "cannot write kernels.py\n"; return 1; } writer.write(ks); }
if (emitMlir) { std::error_code e3; llvm::raw_fd_ostream ms(outDir + "/optimized.mlir", e3); if (!e3) module->print(ms); }
```
Update the `--pipeline` description to `none | fuse`.
**Verify:** `build/bin/fusionpass-compile model.mlir -o /tmp/o --pipeline fuse` writes `plan.json`, `kernels.py`, `optimized.mlir`; for the Phase 1 attention example `plan.json` contains one `triton` step with `"launch": "attention"` and `kernels.py` contains `def fp_attention`.

### Step 8 — lit tests for the passes
`test/fuse_attention.mlir`:
```mlir
// RUN: %fusionpass-opt %s --fp-fuse-attention | FileCheck %s

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
```
Both functions live in one file/module; `CHECK-LABEL` keeps their checks independent.

`test/fuse_chains.mlir`:
```mlir
// RUN: %fusionpass-opt %s --fp-fuse-chains | FileCheck %s

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
```
Note: in `@softmax_axis0` the `scale` alone cannot form a chain (needs ≥ 2 ops), so no fused chain appears. In `@multiuse`, `%0` has two users.
**Verify:** `lit -v test/` → 4 tests pass (roundtrip, invalid, fuse_attention, fuse_chains).

### Step 9 — Runtime: load kernels and launch
Replace the `NotImplementedError` branch in `python/fusionpass/runtime.py`. Add:
```python
import importlib.util, os, sys
from pathlib import Path


def load_kernels(path):
    """Import the generated kernels.py. Must happen AFTER TRITON_INTERPRET is set (if used)."""
    path = Path(path)
    name = f"fusionpass_kernels_{abs(hash(str(path.resolve())))}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod      # WHY: triton.jit calls inspect.getsource -> needs a resolvable module
    spec.loader.exec_module(mod)
    return mod


def _check_device(t: torch.Tensor):
    if t.device.type == "cpu" and os.environ.get("TRITON_INTERPRET") != "1":
        raise RuntimeError("Triton kernels need a CUDA tensor, or set TRITON_INTERPRET=1 before importing "
                           "triton to run on CPU in interpreter mode")


def _run_triton(step: dict, ops: list[torch.Tensor], kernels) -> torch.Tensor:
    import triton                                  # lazy: keeps `import fusionpass` cheap and env-var-safe
    if kernels is None:
        raise RuntimeError("plan has triton steps but no kernels module was provided")
    ops = [o.contiguous() for o in ops]            # kernels assume contiguous row-major layouts
    _check_device(ops[0])
    kern = getattr(kernels, step["kernel"])
    a, launch = step["attrs"], step["launch"]
    out = torch.empty(step["shape"], dtype=_DTYPES[step["dtype"]], device=ops[0].device)
    if launch == "flat":
        n = out.numel()
        kern[(triton.cdiv(n, a["block"]),)](*ops, out, n, BLOCK=a["block"])
    elif launch == "rows":
        n_cols = step["shape"][-1]
        kern[(out.numel() // n_cols,)](*ops, out, n_cols, BLOCK_N=a["block_n"], num_warps=a["num_warps"])
    elif launch == "attention":
        q, k, v = ops
        S, D = step["shape"][-2], step["shape"][-1]
        bh = out.numel() // (S * D)
        kern[(triton.cdiv(S, a["block_m"]), bh)](q, k, v, out, S, a["scale"], HEAD_DIM=D,
                                                 BLOCK_M=a["block_m"], BLOCK_N=a["block_n"],
                                                 num_warps=a["num_warps"], num_stages=a["num_stages"])
    else:
        raise ValueError(f"unknown launch kind {launch}")
    return out
```
and in `execute_plan` replace the `else: raise NotImplementedError(...)` with:
```python
        elif step["kind"] == "triton":
            env[step["result"]] = _run_triton(step, ops, kernels)
        else:
            raise ValueError(f"unknown step kind {step['kind']}")
```
**Verify:** exercised in Step 11.

### Step 10 — `api.py`: enable `fuse=True`
In `python/fusionpass/api.py`:
- Remove the `NotImplementedError` for `fuse`.
- Pass `"--pipeline", "fuse" if fuse else "none"` to the subprocess.
- After reading `plan.json`, `kernels = load_kernels(workdir / "kernels.py")` **only if** any step has `kind == "triton"`, else `None`.
- `CompiledModel.__init__` gains a `kernels` attribute and `__call__` becomes `execute_plan(self.plan, list(inputs) + self._captured, self.kernels)`.
```python
from .runtime import execute_plan, load_kernels
# ...
has_triton = any(s["kind"] == "triton" for s in plan["steps"])
kernels = load_kernels(workdir / "kernels.py") if has_triton else None
return CompiledModel(plan, arg_tensors[len(example_inputs):], workdir, mlir_text, kernels)
```
**Verify:** Step 11 tests.

### Step 11 — pytest end-to-end
`tests/conftest.py` (MODIFIED) — add at the very top, before any test imports triton:
```python
import os
import torch
if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")   # CPU correctness via the Triton interpreter
# (keep the existing _require_compiler fixture)
```
`tests/test_fusion_e2e.py`:
```python
import pytest, torch
from torch import nn
from fusionpass import compile_model

DEV = "cuda" if torch.cuda.is_available() else "cpu"

class Attn(nn.Module):
    def forward(self, q, k, v):
        s = torch.matmul(q, k.transpose(-2, -1)) * 0.25
        return torch.matmul(torch.softmax(s, dim=-1), v)

class Chain(nn.Module):
    def __init__(self):
        super().__init__(); self.b = nn.Parameter(torch.randn(8, 16))
    def forward(self, x):
        return torch.exp(torch.relu(x * 2.0 + self.b) * 0.1)

class SoftmaxChain(nn.Module):
    def forward(self, x):
        return torch.relu(torch.softmax(x * 0.5, dim=-1))      # scale -> softmax -> relu

def kinds(c): return [s.get("launch", s["op"] if s["kind"] == "torch" else "?") for s in c.plan["steps"]]

def test_attention_fused_single_kernel():
    m = Attn().to(DEV)
    q, k, v = (torch.randn(1, 2, 20, 16, device=DEV) for _ in range(3))   # S=20: not a multiple of BLOCK
    c = compile_model(m, (q, k, v), fuse=True)
    assert kinds(c) == ["attention"]                       # whole subgraph -> one launch
    torch.testing.assert_close(c(q, k, v), m(q, k, v), atol=5e-3, rtol=5e-3)   # tf32 on GPU needs slack

def test_chain_fused_single_kernel():
    m = Chain().to(DEV).eval(); x = torch.randn(8, 16, device=DEV)
    c = compile_model(m, (x,), fuse=True)
    assert kinds(c) == ["flat"]
    torch.testing.assert_close(c(x), m(x), atol=1e-5, rtol=1e-5)

def test_softmax_chain_rows_kernel():
    m = SoftmaxChain().to(DEV); x = torch.randn(6, 100, device=DEV)          # 100 cols -> BLOCK_N=128 (masked)
    c = compile_model(m, (x,), fuse=True)
    assert kinds(c) == ["rows"]
    torch.testing.assert_close(c(x), m(x), atol=1e-5, rtol=1e-5)

def test_unfused_and_fused_agree():
    m = Chain().to(DEV).eval(); x = torch.randn(8, 16, device=DEV)
    torch.testing.assert_close(compile_model(m, (x,), fuse=False)(x), compile_model(m, (x,), fuse=True)(x))
```
**Verify:** `pytest -q` → all Phase 1 + Phase 2 tests pass (CPU interpreter or GPU).

---

## 9. API / Interface Contracts

**CLI — `fusionpass-compile <input.mlir> -o <outdir> [--pipeline none|fuse] [--emit-mlir=true|false]`**
- Writes `plan.json` and `kernels.py` (always; `kernels.py` has only the header when nothing was fused), and `optimized.mlir` unless disabled.
- Exit codes: 0 ok; 1 parse/verify/pass/emit/IO failure; 2 unknown `--pipeline`.

**Passes (`fusionpass-opt`)**
- `--fp-fuse-attention` (per `func.func`): rewrites attention subgraphs → `fp.fused_attention`.
- `--fp-fuse-chains` (per `func.func`): rewrites chains → `fp.fused_chain`.
- Both are no-ops (never fail) when nothing matches.

**C++**
- `fusionpass::emitPlan(ModuleOp, TritonModuleWriter&, llvm::json::Value&) -> LogicalResult`
- `TritonModuleWriter::{addChainKernel(FusedChainOp), addAttentionKernel(), write(raw_ostream&)}`
- `fusionpass::createFuseAttentionPass()`, `createFuseChainsPass()`, `registerFusionPasses()`, constant `kMaxRowSoftmaxCols = 8192`.

**Generated Triton kernels (signatures)**
- `fp_chain_N(x0..xk, out, n_elements, BLOCK: constexpr)` — grid `(cdiv(n_elements, BLOCK),)`.
- `fp_chain_N(x0..xk, out, n_cols, BLOCK_N: constexpr)` (row kernels) — grid `(rows,)`.
- `fp_attention(Q, K, V, O, S, scale, HEAD_DIM, BLOCK_M, BLOCK_N: constexpr)` — grid `(cdiv(S, BLOCK_M), batch*heads)`; inputs contiguous `[..., S, D]`.

**Python**
- `compile_model(model, example_inputs, *, fuse=False, workdir=None) -> CompiledModel` (now supports `fuse=True`); `CompiledModel` has `.plan`, `.workdir`, `.mlir_text`, `.kernels`.
- `load_kernels(path) -> module`; `execute_plan(plan, args, kernels=None)`.
- Errors: `RuntimeError` for triton steps on CPU without `TRITON_INTERPRET=1`; `ValueError` for unknown launch/step kinds.

---

## 10. Error Handling & Edge Cases

| Situation | Handling |
|---|---|
| Attention intermediate used elsewhere (P, scores, K^T) | Pattern returns `failure()`; ops stay unfused (still correct). Covered by `@attn_multiuse`. |
| Head dim not in {16,32,64,128}, or Q/K/V shapes differ (cross-attention) | Not fused (pattern fails); falls back to torch steps. |
| Sequence length not a multiple of 64 | Handled by masks on loads/stores and `-inf` on padded key columns. Tested with S=20. |
| Softmax over non-last axis, or last dim > 8192 | Not fusible (`isFusible`); stays a torch op. |
| Two softmaxes in one candidate chain | Merge rejected (`softmaxCount > 1`); they remain separate. |
| Producer used by two consumers (or twice by one) | Not fused (`hasOneUse`). |
| Same value used twice as external input (`x*x`) | Externals are deduplicated; kernel gets one pointer, used twice. |
| fp16 inputs | Kernels compute in fp32, cast on store. |
| Padded lanes in row kernel | Loaded as 0, forced to `-inf` before softmax, masked on store. |
| CPU tensors without `TRITON_INTERPRET=1` | `RuntimeError` with remediation text. |
| Non-contiguous operands (e.g., a transposed view) | Runtime calls `.contiguous()` before launch. |
| Verifier failure after a pass | `pm.run` fails → exit 1 with MLIR diagnostics. |
| Dead ops left behind | Removed by the greedy driver (all `fp` ops are `Pure`). |

---

## 11. Testing & Verification

```bash
cmake --build build
lit -v test/                  # roundtrip, invalid, fuse_attention, fuse_chains
pytest -q                     # includes tests/test_fusion_e2e.py
```
Manual inspection:
```bash
python - <<'EOF'
import torch, json
from fusionpass import compile_model
class A(torch.nn.Module):
    def forward(self, q, k, v):
        return torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-2,-1)) * 0.125, dim=-1), v)
dev = "cuda" if torch.cuda.is_available() else "cpu"
q = k = v = torch.randn(2, 4, 64, 32, device=dev, dtype=torch.float16 if dev == "cuda" else torch.float32)
c = compile_model(A(), (q, k, v), fuse=True)
print([s["kernel"] if s["kind"]=="triton" else s["op"] for s in c.plan["steps"]])   # ['fp_attention']
print(open(c.workdir/"optimized.mlir").read())
EOF
```
Expected: one step `fp_attention`; `optimized.mlir` contains `fp.fused_attention` and no `fp.softmax`.

---

## 12. Definition of Done

- [ ] `FPOps.td` has `fp.yield`, `fp.fused_chain`, `fp.fused_attention`; verifiers reject malformed fused ops.
- [ ] `fusionpass-opt --fp-fuse-attention` and `--fp-fuse-chains` exist and pass their lit tests (positive and negative cases).
- [ ] `fusionpass-compile --pipeline fuse` writes `plan.json`, `kernels.py`, `optimized.mlir`.
- [ ] Attention subgraph → exactly one `fp_attention` launch; elementwise chain → one `flat` launch; scale→softmax→relu → one `rows` launch.
- [ ] `pytest -q` passes (CPU interpreter and/or GPU): fused output matches eager within tolerance for attention (S=20), chain, softmax-chain.
- [ ] `fuse=False` path from Phase 1 still passes all Phase 1 tests.
- [ ] Generated `kernels.py` imports cleanly and Triton compiles/interprets every kernel.

---

## 13. Notes & Pitfalls for the Implementer

- **Order matters:** run `fp-fuse-attention` before `fp-fuse-chains`, otherwise the `scale`/`softmax` inside attention may be swallowed by a chain and the attention pattern no longer matches.
- **Set `TRITON_INTERPRET=1` before importing triton.** The runtime imports triton lazily and `kernels.py` imports it at load time — keep it that way.
- **fp32 attention on GPU** uses TF32 in `tl.dot` by default, so error vs. FP32 eager is ~1e-3, not 1e-6. Tests use 5e-3 for attention. A strict mode (`input_precision="ieee"` in newer Triton, ⚠️ verify the argument name for your Triton version) can be considered in Phase 3.
- "Shared-memory staging" is done by the Triton compiler when it lowers `tl.dot` on tiled loads; our code controls the tiling (`BLOCK_M/BLOCK_N`) and pipelining (`num_stages`) only. Do not claim hand-written shared-memory code.
- `fused_chain` prints in generic form — lit `CHECK`s use `"fp.fused_chain"` with quotes.
- `getFactor()`/`getScale()` return `llvm::APFloat`; `APFloat::toString` output is valid Python.
- `rw.createBlock` moves the insertion point into the new block; set it explicitly (as shown).
- API names for the greedy driver are LLVM-18-specific (`applyPatternsAndFoldGreedily`); do not change LLVM version.
- **Do NOT build here (Phase 3 owns it):** equivalence gate / tolerance policy, benchmarking and speedup reporting, traffic estimates, `python -m fusionpass` CLI, plan validator, extended fallback test matrix, README/docs/CI/deployment.

---

## 14. Handoff to Next Phase

**What now exists and works:** the full fusion pipeline — fused ops, two passes, Triton source emission (including the FlashAttention-style online-softmax kernel), `triton` plan steps, kernel loading and launching, and `compile_model(fuse=True)` — verified by lit tests and pytest against eager PyTorch.

**Next (Phase 3):** the numerical-equivalence gate that must pass before any speedup is reported, benchmarking and a memory-traffic estimate, the demo CLI, an edge-case/fallback test matrix, hardening, README/docs, CI, and delivery steps.
