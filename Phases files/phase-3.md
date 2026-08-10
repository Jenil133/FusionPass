# PHASE 3 — Completion, Hardening & Delivery (fusionpass)

---

## 1. Project Context

**Project:** `fusionpass` — an MLIR-based compiler pass that fuses attention subgraphs and elementwise/softmax chains of a PyTorch model into single generated Triton kernels.

**What it is.** A PyTorch `nn.Module` (static shapes, f16/f32, inference only) is traced with `torch.fx` into textual MLIR in a small custom dialect called `fp`. A C++ tool, `fusionpass-compile`, parses that MLIR, runs fusion passes, and emits `plan.json` (an ordered list of steps) and `kernels.py` (generated Triton source). A Python runtime executes the plan: fused regions run as Triton kernels, everything else as plain PyTorch ops. A verification harness proves numerical equivalence against eager PyTorch **before any speedup is reported** — a fusion that changes results is a miscompilation, not an optimization.

**Target user:** ML-systems engineers / students who want a working, readable example of an MLIR fusion pass with real GPU codegen.

**Problem solved:** each unfused op writes its result to GPU global memory and the next op reads it back. Fusing removes those round trips. For attention, fusion also avoids ever materializing the S×S score matrix.

**Whole-project scope boundaries (do not exceed):** static shapes only; dtypes f16 and f32 only; no broadcasting; inference only; no causal/attention masks; one output value per graph; single device.

### Tech stack (identical in all three phase files)

| Layer | Choice |
|---|---|
| Compiler | C++17, **LLVM/MLIR 18.1.x** (tag `llvmorg-18.1.8` — ⚠️ verify), CMake ≥ 3.20, Ninja, clang or gcc |
| Compiler tests | `lit` (pip) + `FileCheck` (LLVM install with `-DLLVM_INSTALL_UTILS=ON`) |
| Frontend / runtime | Python ≥ 3.10, PyTorch ≥ 2.2 (⚠️ verify), `torch.fx` + `ShapeProp` |
| Kernel codegen target | **Triton** — the version bundled with the PyTorch pip wheel (do NOT `pip install triton` separately) |
| GPU | NVIDIA GPU + CUDA for benchmarks (compute capability ≥ 8.0 recommended — ⚠️ verify). CPU correctness via `TRITON_INTERPRET=1` |
| Python tests | `pytest` |

**Stack decisions (why this over the alternative):**
- Custom `fp` dialect **over torch-mlir** — heavy build/version coupling; a small dialect keeps the pass surface under our control.
- Emit Triton source **over lowering to NVVM** — Triton already does tiling, shared-memory staging and tensor-core codegen; our pass focuses on *fusion decisions*.
- `torch.fx` **over ONNX** — reference is the same `nn.Module`; `ShapeProp` provides shapes.
- JSON plan + Python runtime **over embedding Python in C++** — decoupled, independently testable.
- Greedy pattern rewrites **over a custom graph algorithm** — idiomatic MLIR, small code.
- Verify-before-benchmark gate **over "just time it"** — an unverified speedup is meaningless.

### Naming conventions (must match across all phases)
- Dialect `fp`, namespace `fusionpass::fp`; libs `MLIRFPDialect`, `MLIRFPPlan`, `MLIRFPTransforms`; tools `fusionpass-opt`, `fusionpass-compile`.
- Python package `fusionpass` under `python/`. Entry function `func.func @main`.
- Plan names: args `arg0..`; results `v0, v1, ...`. Kernels: `fp_chain_<N>`, `fp_attention`.
- Out-dir files: `plan.json`, `kernels.py`, `optimized.mlir`.
- Passes: `--fp-fuse-attention`, `--fp-fuse-chains`. Env vars: `LLVM_PREFIX`, `FUSIONPASS_BUILD_DIR`, `FUSIONPASS_COMPILE`.

---

## 2. Architecture Overview

```
 PyTorch nn.Module
      │  torch.fx symbolic_trace + ShapeProp          python/fusionpass/frontend.py      (Phase 1, done)
      ▼
 model.mlir   (func.func @main + fp dialect ops)
      │  fusionpass-compile   (C++ / MLIR)
      │     parse → verify → fp-fuse-attention → fp-fuse-chains → emit plan + Triton src (Phase 2, done)
      ▼
 plan.json  +  kernels.py (Triton source)
      │  python/fusionpass/runtime.py  (+ validate_plan)                                 ◄ PHASE 3 hardens
      ▼
 PyTorch ops (unfused)  |  Triton kernels (fused)  → CUDA GPU (or Triton interpreter on CPU)
      │
 verify.py  (equivalence gate)  ──►  bench.py (speedup, only if verified)  + traffic.py    ◄ PHASE 3 builds
      │
 python -m fusionpass demo|check   (CLI)                                                   ◄ PHASE 3 builds
```

**This phase's place:** everything that makes the project trustworthy and deliverable — the equivalence gate, benchmarking, the CLI, edge-case coverage, docs, CI, and the delivery checklist.

**Final project layout:**
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
├── python/fusionpass/{__init__.py,frontend.py,runtime.py,api.py,verify.py,bench.py,traffic.py,demos.py,__main__.py}
├── test/{lit.cfg.py,roundtrip.mlir,invalid.mlir,invalid_fused.mlir,fuse_attention.mlir,fuse_chains.mlir}
├── tests/{conftest.py,test_frontend.py,test_e2e.py,test_fusion_e2e.py,test_verify_bench.py,test_edge_cases.py,test_plan_validation.py}
├── examples/{attention_demo.py,chain_demo.py}
├── docs/design.md
├── scripts/build.sh
└── .github/workflows/ci.yml
```

---

## 3. Current State

Phases 1–2 are complete and on disk. **Rely on them; do not rebuild.** Read files before modifying them.

**Build & tools**
- LLVM/MLIR 18.1.x at `$LLVM_PREFIX`; project builds in `build/`; binaries `build/bin/fusionpass-opt`, `build/bin/fusionpass-compile`.
- `fusionpass-opt` registers dialects `func`, `fp` and passes `fp-fuse-attention`, `fp-fuse-chains`.
- `fusionpass-compile <in.mlir> -o <outdir> [--pipeline none|fuse] [--emit-mlir=true|false]` writes `plan.json`, `kernels.py`, `optimized.mlir`. Exit codes 0 / 1 (failure) / 2 (bad `--pipeline`).

**Dialect `fp`:** ops `add, mul, scale{factor:f32}, relu, exp, softmax{axis:i64}, matmul, transpose, yield, fused_chain` (region op, generic print form; all inputs/result one type; body ⊂ {add,mul,scale,relu,exp,softmax}; ≤ 1 softmax on the last axis), `fused_attention{scale:f32}` (`q,k,v,result` same type; rank ≥ 3; last dim ∈ {16,32,64,128}). Verifiers exist for all.

**Passes:** `fp-fuse-attention` (root = second matmul; matches `matmul(softmax(scale*…(matmul(Q, transpose(K)))), V)`, all intermediates single-use, Q/K/V/out identical types, D ∈ {16,32,64,128}); `fp-fuse-chains` (greedy merge of single-use chainable producer into consumer; ≤ 1 softmax; softmax must be last-axis with `last_dim ≤ kMaxRowSoftmaxCols = 8192`; externals deduplicated).

**Plan format (v1):** `torch` steps (`op` ∈ add, mul, scale, relu, exp, softmax, matmul, transpose) and `triton` steps (`kernel`, `launch` ∈ flat/rows/attention, `attrs`). Attention attrs: `scale, block_m=64, block_n=64, num_warps=4, num_stages=2`. Flat: `block=1024`. Rows: `block_n=next_pow2(n_cols), num_warps∈{4,8}`.

**Python (`python/fusionpass/`)**
- `frontend.py`: `trace_to_mlir(model, example_inputs) -> (str, list[Tensor])`, `UnsupportedOp`.
- `runtime.py`: `execute_plan(plan, args, kernels=None)`, `load_kernels(path)`, `_run_torch`, `_run_triton`, `_check_device`, `_DTYPES`.
- `api.py`: `compile_model(model, example_inputs, *, fuse=False, workdir=None) -> CompiledModel`; `CompiledModel(plan, captured, workdir, mlir_text, kernels)` with `__call__`; `_find_compiler()`.
- `__init__.py` exports `compile_model, CompiledModel, UnsupportedOp`.

**Tests:** lit — `roundtrip.mlir, invalid.mlir, fuse_attention.mlir, fuse_chains.mlir`; pytest — `conftest.py` (sets `TRITON_INTERPRET=1` when no CUDA; skips if compiler missing), `test_frontend.py`, `test_e2e.py`, `test_fusion_e2e.py`. Test helper idiom: models `Attn`, `Chain`, `SoftmaxChain`; tolerance 5e-3 for attention.

---

## 4. Phase Goal & Scope

**Goal:** make the project correct-by-gate, measurable, hardened, documented, and deliverable.

**In scope**
- `verify.py`: numerical-equivalence harness with per-dtype tolerance policy; integrate into `compile_model` (`verify=True`, strict).
- `bench.py`: CUDA-event benchmark that **refuses to report a speedup unless verification passed**.
- `traffic.py`: rough global-memory-traffic estimate and plan summary.
- `runtime.validate_plan`.
- `demos.py` + `__main__.py` CLI (`python -m fusionpass demo attention|chain`, `check`) + `examples/`.
- Edge-case and fallback test matrix; negative lit tests for fused-op verifiers; dedupe lit test.
- `README.md`, `docs/design.md`, `scripts/build.sh`, CI workflow, packaging polish.
- Delivery steps and completion summary.

**Out of scope (explicitly NOT part of this project)**
- Causal/arbitrary masks, broadcasting, dynamic shapes, backward pass, bf16, multi-output graphs, autotuning, cross-attention fusion, Windows support. Document these as limitations; do not implement.

---

## 5. Prerequisites & Setup

```bash
source .venv/bin/activate
export LLVM_PREFIX=$HOME/llvm-install
export FUSIONPASS_BUILD_DIR=build
export FUSIONPASS_COMPILE=$PWD/build/bin/fusionpass-compile
pip install -e .[dev]
cmake --build build && lit -v test/ && pytest -q     # must be green before you start
```
GPU benchmarking additionally needs an NVIDIA GPU, a CUDA-enabled PyTorch wheel, and `torch.cuda.is_available() == True`. No secrets or accounts are used.

---

## 6. File Structure for This Phase

```
fusionpass/
├── python/fusionpass/
│   ├── verify.py                                   NEW
│   ├── bench.py                                    NEW
│   ├── traffic.py                                  NEW
│   ├── demos.py                                    NEW
│   ├── __main__.py                                 NEW
│   ├── runtime.py                                  MODIFIED  (+validate_plan)
│   ├── api.py                                      MODIFIED  (verify gate)
│   └── __init__.py                                 MODIFIED  (exports)
├── test/{invalid_fused.mlir (NEW), fuse_chains.mlir (MODIFIED: +dedupe case)}
├── tests/{test_verify_bench.py,test_edge_cases.py,test_plan_validation.py}   NEW
├── examples/{attention_demo.py,chain_demo.py}      NEW
├── docs/design.md                                  NEW
├── scripts/build.sh                                NEW
├── .github/workflows/ci.yml                        NEW
├── README.md                                       NEW
└── pyproject.toml                                  MODIFIED  (+console script)
```

---

## 7. Data Model / Schema

No database. Phase 3 adds two Python result types; `plan.json` v1 is unchanged.

```python
@dataclass(frozen=True)
class VerifyReport:           # verify.py
    ok: bool
    max_abs_err: float        # over all trials, finite elements only
    max_rel_err: float        # |out-ref| / max(|ref|, 1e-6)
    atol: float
    rtol: float
    dtype: str                # "f16" | "f32"
    trials: int
    reason: str               # "" if ok, else first failure description

@dataclass(frozen=True)
class BenchResult:            # bench.py
    eager_ms: float           # median over iters
    fused_ms: float
    speedup: float            # eager_ms / fused_ms
    iters: int
    eager_launches: int       # plan steps of the UNFUSED plan
    fused_launches: int       # plan steps of the fused plan
    est_bytes_unfused: int | None
    est_bytes_fused: int | None
    report: VerifyReport      # the verification that gated this benchmark
```

**Tolerance policy (`default_tolerances(dtype, has_attention)`):**

| dtype | has fused attention? | atol / rtol | why |
|---|---|---|---|
| f16 | any | 1e-2 / 1e-2 | eager rounds every op to fp16; our kernels accumulate in fp32 |
| f32 | yes | 5e-3 / 5e-3 | `tl.dot` uses TF32 by default on GPU |
| f32 | no | 1e-4 / 1e-4 | only reordering/fused-rounding differences |

These are starting values (⚠️ calibrate on your GPU; tighten if they pass with margin).

---

## 8. Implementation Steps

### Step 1 — Plan validation
Add to `python/fusionpass/runtime.py`:
```python
_TORCH_OPS = {"add", "mul", "scale", "relu", "exp", "softmax", "matmul", "transpose"}
_LAUNCHES = {"flat", "rows", "attention"}

def validate_plan(plan: dict) -> None:
    """Structural check so a corrupt/stale plan fails loudly BEFORE any kernel runs."""
    if plan.get("version") != 1:
        raise ValueError(f"unsupported plan version {plan.get('version')!r}")
    for key in ("inputs", "steps", "outputs"):
        if not isinstance(plan.get(key), list):
            raise ValueError(f"plan['{key}'] must be a list")
    defined = set()
    for spec in plan["inputs"]:
        if spec.get("dtype") not in _DTYPES or not isinstance(spec.get("shape"), list):
            raise ValueError(f"bad input spec: {spec}")
        defined.add(spec["name"])
    for i, st in enumerate(plan["steps"]):
        for o in st.get("operands", []):
            if o not in defined:
                raise ValueError(f"step {i}: operand '{o}' used before definition")
        res = st.get("result")
        if not res or res in defined:
            raise ValueError(f"step {i}: missing or duplicate result name '{res}'")
        if st.get("dtype") not in _DTYPES:
            raise ValueError(f"step {i}: bad dtype")
        if st.get("kind") == "torch":
            if st.get("op") not in _TORCH_OPS:
                raise ValueError(f"step {i}: unknown torch op '{st.get('op')}'")
        elif st.get("kind") == "triton":
            if st.get("launch") not in _LAUNCHES or not st.get("kernel"):
                raise ValueError(f"step {i}: bad triton step")
            if st["launch"] == "attention" and len(st["operands"]) != 3:
                raise ValueError(f"step {i}: attention needs exactly 3 operands")
        else:
            raise ValueError(f"step {i}: unknown kind '{st.get('kind')}'")
        defined.add(res)
    for o in plan["outputs"]:
        if o not in defined:
            raise ValueError(f"output '{o}' is never defined")
```
Change `execute_plan` signature to `execute_plan(plan, args, kernels=None, *, validate=True)` and call `validate_plan(plan)` first when `validate` is true (`CompiledModel` validates once at construction and passes `validate=False` per call).
**Verify:** Step 9 tests.

### Step 2 — Equivalence harness
`python/fusionpass/verify.py`:
```python
from __future__ import annotations
from dataclasses import dataclass
import torch

class EquivalenceError(AssertionError):
    """Raised when a compiled model does not match the eager reference."""

@dataclass(frozen=True)
class VerifyReport:
    ok: bool; max_abs_err: float; max_rel_err: float; atol: float; rtol: float
    dtype: str; trials: int; reason: str

def default_tolerances(dtype: torch.dtype, has_attention: bool) -> tuple[float, float]:
    if dtype == torch.float16: return 1e-2, 1e-2
    return (5e-3, 5e-3) if has_attention else (1e-4, 1e-4)

def _compare(ref: torch.Tensor, out: torch.Tensor, atol: float, rtol: float):
    if ref.shape != out.shape or ref.dtype != out.dtype:
        return False, float("inf"), float("inf"), f"shape/dtype mismatch: ref {ref.dtype}{tuple(ref.shape)} vs out {out.dtype}{tuple(out.shape)}"
    r, o = ref.detach().float(), out.detach().float()
    # WHY compare non-finite patterns separately: allclose-style math on inf/NaN is meaningless;
    # an overflow in eager must overflow identically in the fused path (or it's a behavior change).
    if not torch.equal(torch.isnan(r), torch.isnan(o)) or not torch.equal(torch.isinf(r), torch.isinf(o)):
        return False, float("inf"), float("inf"), "NaN/Inf pattern differs from reference"
    inf = torch.isinf(r)
    if inf.any() and not torch.equal(r[inf], o[inf]):
        return False, float("inf"), float("inf"), "Inf sign differs from reference"
    fin = torch.isfinite(r)
    if not fin.any():
        return True, 0.0, 0.0, ""
    rf, of = r[fin], o[fin]
    err = (of - rf).abs()
    ok = bool((err <= atol + rtol * rf.abs()).all())
    max_abs = float(err.max())
    max_rel = float((err / rf.abs().clamp_min(1e-6)).max())
    why = "" if ok else f"max_abs_err={max_abs:.3e}, max_rel_err={max_rel:.3e} exceed atol={atol}, rtol={rtol}"
    return ok, max_abs, max_rel, why

def verify_equivalence(model, compiled, inputs, *, atol=None, rtol=None,
                       extra_trials: int = 2, seed: int = 0) -> VerifyReport:
    """Compare compiled(*inputs) against model(*inputs) on the given inputs plus
    `extra_trials` randomized inputs (scales 2x, 4x) that stress softmax stability."""
    inputs = tuple(inputs)
    dtype = next(t.dtype for t in inputs if torch.is_tensor(t))
    has_attn = any(s.get("launch") == "attention" for s in compiled.plan["steps"])
    d_atol, d_rtol = default_tolerances(dtype, has_attn)
    atol = d_atol if atol is None else atol
    rtol = d_rtol if rtol is None else rtol

    g = torch.Generator(device="cpu").manual_seed(seed)
    trials = [inputs]
    for scale in (2.0, 4.0)[:extra_trials]:
        trials.append(tuple((torch.randn(x.shape, generator=g) * scale).to(device=x.device, dtype=x.dtype)
                            for x in inputs))
    worst_abs = worst_rel = 0.0
    with torch.no_grad():
        for i, xs in enumerate(trials):
            ok, a, r, why = _compare(model(*xs), compiled(*xs), atol, rtol)
            worst_abs, worst_rel = max(worst_abs, a), max(worst_rel, r)
            if not ok:
                return VerifyReport(False, worst_abs, worst_rel, atol, rtol, _dt(dtype), len(trials), f"trial {i}: {why}")
    return VerifyReport(True, worst_abs, worst_rel, atol, rtol, _dt(dtype), len(trials), "")

def _dt(d): return "f16" if d == torch.float16 else "f32"

def assert_equivalent(model, compiled, inputs, **kw) -> VerifyReport:
    rep = verify_equivalence(model, compiled, inputs, **kw)
    if not rep.ok:
        raise EquivalenceError(f"compiled model differs from eager reference: {rep.reason}")
    return rep
```
**Verify:** `test_verify_bench.py` (Step 9).

### Step 3 — Integrate the gate into `compile_model`
Modify `python/fusionpass/api.py`:
```python
from .runtime import execute_plan, load_kernels, validate_plan
from .verify import verify_equivalence, EquivalenceError

class CompiledModel:
    def __init__(self, plan, captured, workdir, mlir_text, kernels=None):
        validate_plan(plan)                      # fail early on a corrupt plan
        self.plan, self._captured, self.workdir, self.mlir_text, self.kernels = plan, captured, workdir, mlir_text, kernels
        self.report = None                       # VerifyReport once verification has run
        self.verified = False                    # True ONLY if verification ran and passed
    def __call__(self, *inputs):
        return execute_plan(self.plan, list(inputs) + self._captured, self.kernels, validate=False)

def compile_model(model, example_inputs, *, fuse=False, workdir=None, verify=True, strict=True,
                  atol=None, rtol=None) -> CompiledModel:
    # ... existing compile logic building `compiled` ...
    if verify:
        compiled.report = verify_equivalence(model, compiled, example_inputs, atol=atol, rtol=rtol)
        compiled.verified = compiled.report.ok
        if strict and not compiled.report.ok:
            raise EquivalenceError(f"fusion changed results: {compiled.report.reason}")
    return compiled
```
**Important:** `fuse` still defaults to `False` (Phase 1 tests rely on that). `verify` defaults to `True`; existing tests must still pass.
**Verify:** `pytest -q` — Phases 1–2 tests still green.

### Step 4 — Traffic estimate and plan summary
`python/fusionpass/traffic.py`:
```python
from __future__ import annotations
from math import prod

_BYTES = {"f16": 2, "f32": 4}

def summarize_plan(plan: dict) -> dict:
    kinds = [s["kind"] for s in plan["steps"]]
    return {"steps": len(kinds), "triton_steps": kinds.count("triton"), "torch_steps": kinds.count("torch")}

def estimate_bytes(plan: dict) -> int:
    """ROUGH model of global-memory traffic: every step reads all its operands and writes its result once.
    Fused attention re-reads K and V once per query tile (ceil(S/block_m) times).
    This is an estimate for explaining WHY fusion helps, NOT a measurement (ignores caches, fusion
    inside PyTorch, in-place ops, and L2 reuse)."""
    size = {s["name"]: prod(s["shape"]) * _BYTES[s["dtype"]] for s in plan["inputs"]}
    total = 0
    for st in plan["steps"]:
        out_b = prod(st["shape"]) * _BYTES[st["dtype"]]
        size[st["result"]] = out_b
        if st.get("launch") == "attention":
            q, k, v = (size[o] for o in st["operands"])
            tiles = -(-st["shape"][-2] // st["attrs"]["block_m"])
            total += q + tiles * (k + v) + out_b
        else:
            total += sum(size[o] for o in st["operands"]) + out_b
    return total
```
**Verify:** unit test: for the attention demo, `estimate_bytes(fused) < estimate_bytes(unfused)` when `S` is large (e.g., 1024).

### Step 5 — Benchmark with the verification gate
`python/fusionpass/bench.py`:
```python
from __future__ import annotations
from dataclasses import dataclass
import torch
from .traffic import estimate_bytes

class UnverifiedSpeedupError(RuntimeError):
    """Raised when a speedup is requested for a model that has not passed equivalence verification."""

@dataclass(frozen=True)
class BenchResult:
    eager_ms: float; fused_ms: float; speedup: float; iters: int
    eager_launches: int; fused_launches: int
    est_bytes_unfused: int | None; est_bytes_fused: int | None
    report: object

def _median_ms_cuda(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):          # also triggers Triton JIT compilation
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for s, e in zip(starts, ends):
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    ts = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return ts[len(ts) // 2]

def benchmark(model, compiled, inputs, *, unfused=None, warmup: int = 25, iters: int = 100) -> BenchResult:
    # WHY the gate comes first: never time (or report) a result that has not been proven equivalent.
    if not getattr(compiled, "verified", False) or compiled.report is None or not compiled.report.ok:
        raise UnverifiedSpeedupError("refusing to report a speedup: the compiled model has not passed "
                                     "equivalence verification (compile with verify=True)")
    inputs = tuple(inputs)
    if not inputs[0].is_cuda:
        raise RuntimeError("benchmarking requires CUDA tensors; the Triton CPU interpreter is not timed")
    with torch.no_grad():
        eager = _median_ms_cuda(lambda: model(*inputs), warmup, iters)
        fused = _median_ms_cuda(lambda: compiled(*inputs), warmup, iters)
    base_plan = unfused.plan if unfused is not None else None
    return BenchResult(
        eager_ms=eager, fused_ms=fused, speedup=eager / fused, iters=iters,
        eager_launches=len(base_plan["steps"]) if base_plan else -1,
        fused_launches=len(compiled.plan["steps"]),
        est_bytes_unfused=estimate_bytes(base_plan) if base_plan else None,
        est_bytes_fused=estimate_bytes(compiled.plan),
        report=compiled.report)
```
**Verify:** CPU test: `benchmark(...)` on a `verify=False` compile raises `UnverifiedSpeedupError` (gate is checked before the CUDA check).

### Step 6 — Demo models and CLI
`python/fusionpass/demos.py`:
```python
from __future__ import annotations
import torch
from torch import nn

class AttentionBlock(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__(); self.scale = head_dim ** -0.5     # float attribute -> traced as a constant -> fp.scale
    def forward(self, q, k, v):
        s = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        return torch.matmul(torch.softmax(s, dim=-1), v)

class RowChain(nn.Module):
    def forward(self, x, bias):                                 # bias has x's shape (no broadcasting supported)
        return torch.softmax(torch.relu(x * 0.5 + bias), dim=-1) * 2.0

_DT = {"f16": torch.float16, "f32": torch.float32}

def make_attention(batch, heads, seq, head_dim, dtype, device):
    m = AttentionBlock(head_dim).to(device)
    xs = tuple(torch.randn(batch, heads, seq, head_dim, device=device, dtype=_DT[dtype]) for _ in range(3))
    return m, xs

def make_chain(rows, cols, dtype, device):
    m = RowChain().to(device)
    xs = tuple(torch.randn(rows, cols, device=device, dtype=_DT[dtype]) for _ in range(2))
    return m, xs
```
`python/fusionpass/__main__.py`:
```python
from __future__ import annotations
import argparse, sys
import torch
from .api import compile_model, _find_compiler
from .bench import benchmark, UnverifiedSpeedupError
from .demos import make_attention, make_chain
from .verify import EquivalenceError
from .traffic import summarize_plan

def _cmd_check(_) -> int:
    try:
        print("compiler :", _find_compiler())
    except FileNotFoundError as e:
        print("compiler : MISSING —", e); return 3
    import triton
    print("torch    :", torch.__version__, "| triton:", triton.__version__)
    print("cuda     :", torch.cuda.is_available())
    return 0

def _cmd_demo(a) -> int:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.kind == "attention":
        model, xs = make_attention(a.batch, a.heads, a.seq, a.head_dim, a.dtype, dev)
    else:
        model, xs = make_chain(a.rows, a.cols, a.dtype, dev)
    try:
        unfused = compile_model(model, xs, fuse=False, verify=False)
        fused = compile_model(model, xs, fuse=True, verify=True, strict=True)
    except FileNotFoundError as e:
        print(e, file=sys.stderr); return 3
    except EquivalenceError as e:
        print(f"VERIFICATION FAILED: {e}", file=sys.stderr); return 2
    r = fused.report
    print(f"verified   : OK  (max_abs_err={r.max_abs_err:.2e}, max_rel_err={r.max_rel_err:.2e}, "
          f"atol={r.atol}, rtol={r.rtol}, trials={r.trials})")
    print(f"plan       : unfused {summarize_plan(unfused.plan)}  ->  fused {summarize_plan(fused.plan)}")
    if dev != "cuda":
        print("benchmark  : skipped (no CUDA; interpreter mode is correctness-only)"); return 0
    try:
        b = benchmark(model, fused, xs, unfused=unfused)
    except UnverifiedSpeedupError as e:
        print(e, file=sys.stderr); return 2
    print(f"eager      : {b.eager_ms:.3f} ms   fused: {b.fused_ms:.3f} ms   speedup: {b.speedup:.2f}x  (median of {b.iters})")
    print(f"est. traffic: unfused {b.est_bytes_unfused/1e6:.1f} MB  ->  fused {b.est_bytes_fused/1e6:.1f} MB  (rough model, not measured)")
    return 0

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="fusionpass")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").set_defaults(fn=_cmd_check)
    d = sub.add_parser("demo"); d.set_defaults(fn=_cmd_demo)
    d.add_argument("kind", choices=["attention", "chain"])
    d.add_argument("--dtype", choices=["f16", "f32"], default="f16")
    d.add_argument("--batch", type=int, default=4); d.add_argument("--heads", type=int, default=8)
    d.add_argument("--seq", type=int, default=1024); d.add_argument("--head-dim", type=int, default=64)
    d.add_argument("--rows", type=int, default=8192); d.add_argument("--cols", type=int, default=1024)
    a = p.parse_args(argv)
    if a.cmd == "demo" and not torch.cuda.is_available() and a.dtype == "f16":
        a.dtype = "f32"                      # fp16 on the CPU interpreter is slow/unsupported in places
        if a.kind == "attention": a.seq = min(a.seq, 64); a.batch = min(a.batch, 1); a.heads = min(a.heads, 2)
        else: a.rows = min(a.rows, 64); a.cols = min(a.cols, 128)
    return a.fn(a)

if __name__ == "__main__":
    raise SystemExit(main())
```
`pyproject.toml`: add
```toml
[project.scripts]
fusionpass = "fusionpass.__main__:main"
```
`__init__.py`: also export `verify_equivalence, assert_equivalent, EquivalenceError` (from `.verify`) and `benchmark, UnverifiedSpeedupError` (from `.bench`).
`examples/attention_demo.py`: `from fusionpass.__main__ import main; raise SystemExit(main(["demo","attention"]))`; `examples/chain_demo.py` likewise with `"chain"`.
**Verify:** `python -m fusionpass check` and `python -m fusionpass demo attention` succeed (on CPU it prints the verification line and the "benchmark skipped" line; on GPU also timing).

### Step 7 — Negative lit tests and dedupe case
`test/invalid_fused.mlir`:
```mlir
// RUN: %fusionpass-opt %s -split-input-file -verify-diagnostics

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
```
Append to `test/fuse_chains.mlir`:
```mlir
// Duplicate external operand must be deduplicated: the fused op takes ONE input.
// CHECK-LABEL: func.func @dedupe
// CHECK: "fp.fused_chain"(%arg0)
func.func @dedupe(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = fp.mul %a, %a : tensor<4x4xf32>
  %1 = fp.relu %0 : tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}
```
**Verify:** `lit -v test/` → 5 tests pass (roundtrip, invalid, invalid_fused, fuse_attention, fuse_chains).

### Step 8 — Python test matrix
`tests/test_plan_validation.py`:
```python
import copy, pytest
from fusionpass.runtime import validate_plan

GOOD = {"version": 1, "inputs": [{"name": "arg0", "shape": [2, 2], "dtype": "f32"}],
        "steps": [{"kind": "torch", "op": "relu", "operands": ["arg0"], "result": "v0",
                   "shape": [2, 2], "dtype": "f32", "attrs": {}}],
        "outputs": ["v0"]}

def test_good_plan_ok(): validate_plan(GOOD)

@pytest.mark.parametrize("mutate", [
    lambda p: p.update(version=2),
    lambda p: p["steps"][0].update(operands=["nope"]),
    lambda p: p["steps"][0].update(op="tanh"),
    lambda p: p["steps"][0].update(kind="cuda"),
    lambda p: p["steps"][0].update(result="arg0"),
    lambda p: p.update(outputs=["v9"]),
])
def test_bad_plans_rejected(mutate):
    p = copy.deepcopy(GOOD); mutate(p)
    with pytest.raises(ValueError): validate_plan(p)
```
`tests/test_verify_bench.py`:
```python
import copy, pytest, torch
from fusionpass import compile_model
from fusionpass.bench import benchmark, UnverifiedSpeedupError
from fusionpass.demos import make_attention, make_chain
from fusionpass.traffic import estimate_bytes
from fusionpass.verify import verify_equivalence, EquivalenceError

DEV = "cuda" if torch.cuda.is_available() else "cpu"

def test_verified_flag_and_report():
    m, xs = make_attention(1, 2, 20, 16, "f32", DEV)
    c = compile_model(m, xs, fuse=True)
    assert c.verified and c.report.ok and c.report.trials == 3

def test_benchmark_refuses_unverified():
    m, xs = make_chain(8, 32, "f32", DEV)
    c = compile_model(m, xs, fuse=True, verify=False)
    assert c.verified is False
    with pytest.raises(UnverifiedSpeedupError):
        benchmark(m, c, xs)

def test_gate_catches_miscompile():
    m, xs = make_attention(1, 2, 20, 16, "f32", DEV)
    c = compile_model(m, xs, fuse=True)
    bad = copy.deepcopy(c.plan)
    for s in bad["steps"]:
        if s.get("launch") == "attention":
            s["attrs"]["scale"] *= 1.5                      # deliberately wrong
    c.plan = bad
    rep = verify_equivalence(m, c, xs)
    assert rep.ok is False and "exceed" in rep.reason

def test_traffic_estimate_fused_smaller_for_long_sequences():
    m, xs = make_attention(1, 2, 1024, 64, "f16", "cpu")      # only traced/compiled, never executed
    u = compile_model(m, xs, fuse=False, verify=False)
    f = compile_model(m, xs, fuse=True, verify=False)
    assert estimate_bytes(f.plan) < estimate_bytes(u.plan)
```
(`compile_model(..., verify=False)` never runs the model, so f16/CPU is fine here.)

`tests/test_edge_cases.py`:
```python
import pytest, torch
from torch import nn
from fusionpass import compile_model, UnsupportedOp

DEV = "cuda" if torch.cuda.is_available() else "cpu"
launches = lambda c: [s.get("launch", "torch") for s in c.plan["steps"]]

class Attn(nn.Module):
    def forward(self, q, k, v):
        return torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * 0.25, dim=-1), v)

@pytest.mark.parametrize("S", [1, 7, 64, 65, 130])            # around the BLOCK_M/BLOCK_N = 64 boundary
def test_attention_sequence_lengths(S):
    q, k, v = (torch.randn(1, 2, S, 16, device=DEV) for _ in range(3))
    c = compile_model(Attn(), (q, k, v), fuse=True)            # verify=True raises if wrong
    assert launches(c) == ["attention"]

def test_unsupported_head_dim_falls_back():
    q, k, v = (torch.randn(1, 2, 8, 24, device=DEV) for _ in range(3))
    c = compile_model(Attn(), (q, k, v), fuse=True)
    assert "attention" not in launches(c)                      # D=24 not fusible -> still correct (verified)

def test_cross_attention_shapes_fall_back():
    class Cross(nn.Module):
        def forward(self, q, k, v):
            return torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-2, -1)), dim=-1), v)
    q = torch.randn(1, 2, 8, 16, device=DEV); k = torch.randn(1, 2, 12, 16, device=DEV); v = torch.randn(1, 2, 12, 16, device=DEV)
    c = compile_model(Cross(), (q, k, v), fuse=True)
    assert "attention" not in launches(c)

def test_multiuse_probabilities_not_fused():
    class Twice(nn.Module):
        def forward(self, q, k, v):
            p = torch.softmax(torch.matmul(q, k.transpose(-2, -1)), dim=-1)
            return torch.matmul(p, v) + torch.matmul(p, v)     # p has two users
    q, k, v = (torch.randn(1, 2, 16, 16, device=DEV) for _ in range(3))
    c = compile_model(Twice(), (q, k, v), fuse=True)
    assert "attention" not in launches(c)

def test_non_last_axis_softmax_not_fused():
    class M(nn.Module):
        def forward(self, x): return torch.relu(torch.softmax(x * 0.5, dim=0))
    x = torch.randn(6, 32, device=DEV)
    assert "rows" not in launches(compile_model(M(), (x,), fuse=True))

def test_duplicate_operand_chain():
    class M(nn.Module):
        def forward(self, x): return torch.relu(x * x)
    x = torch.randn(5, 7, device=DEV)
    c = compile_model(M(), (x,), fuse=True)
    assert launches(c) == ["flat"] and c.plan["steps"][0]["operands"] == ["arg0"]

def test_large_magnitude_softmax_is_stable():
    class M(nn.Module):
        def forward(self, x): return torch.softmax(x * 1.0, dim=-1) * 2.0
    x = torch.randn(4, 50, device=DEV) * 50                    # exp would overflow without max-subtraction
    c = compile_model(M(), (x,), fuse=True)
    torch.testing.assert_close(c(x), M()(x), atol=1e-4, rtol=1e-4)

def test_noncontiguous_input_handled():
    class M(nn.Module):
        def forward(self, x): return torch.exp(torch.relu(x * 2.0) * 0.1)
    base = torch.randn(16, 8, device=DEV)
    x = base.t()                                               # shape (8,16), non-contiguous
    c = compile_model(M(), (x.contiguous(),), fuse=True)
    torch.testing.assert_close(c(x), M()(x), atol=1e-5, rtol=1e-5)

def test_wrong_shape_at_call_time():
    class M(nn.Module):
        def forward(self, x): return torch.relu(x * 2.0)
    x = torch.randn(4, 4, device=DEV)
    c = compile_model(M(), (x,), fuse=True)
    with pytest.raises(ValueError):
        c(torch.randn(4, 5, device=DEV))

def test_unsupported_broadcast_still_raises():
    class M(nn.Module):
        def forward(self, x, b): return x + b
    with pytest.raises(UnsupportedOp):
        compile_model(M(), (torch.randn(4, 8, device=DEV), torch.randn(8, device=DEV)), fuse=True)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp16 kernels need CUDA")
def test_fp16_attention_and_chain():
    class Chain(nn.Module):
        def forward(self, x): return torch.relu(x * 1.5) * 0.5
    q, k, v = (torch.randn(2, 4, 100, 64, device="cuda", dtype=torch.float16) for _ in range(3))
    assert launches(compile_model(Attn(), (q, k, v), fuse=True)) == ["attention"]
    x = torch.randn(64, 256, device="cuda", dtype=torch.float16)
    assert launches(compile_model(Chain(), (x,), fuse=True)) == ["flat"]
```
**Verify:** `pytest -q` → all tests pass (fp16 test skipped without CUDA).

### Step 9 — Documentation
`README.md` (write these sections): **What it is** (one paragraph as in Section 1) · **Quick start** (build LLVM → `scripts/build.sh` → `pip install -e .[dev]` → `python -m fusionpass check` → `python -m fusionpass demo attention`) · **How it works** (the Section 2 diagram + 5 bullet explanation) · **Supported subset** (ops, dtypes, static shapes, no broadcasting/masks) · **Results** (a table with columns `model | shape | dtype | eager ms | fused ms | speedup | verified`; **populate only with numbers measured by `python -m fusionpass demo ...` on your hardware, and state the GPU model; never include numbers you did not measure**) · **Verification policy** (the tolerance table from Section 7; "no speedup is reported without passing equivalence") · **Limitations** (below) · **Testing** (`lit`, `pytest`).

`docs/design.md` (write these sections): (1) dialect and invariants; (2) the two passes with the matching rules and why single-use is required; (3) the chain-merge algorithm (externals dedupe, IRMapping inlining, termination argument); (4) the generated kernels — flat, row-softmax, and the online-softmax attention kernel with the m_i/l_i/acc recurrence written out; (5) plan format; (6) **Scaling vs. breaking points**:
  - *Scales:* passes are linear in op count (greedy, each rewrite removes an op); attention memory is O(S·D) instead of O(S²); kernels are O(1) in registers per tile.
  - *Breaks / limits:* per-shape recompilation (static shapes); row kernel requires the row in registers (cap 8192); attention tile sizes are fixed (no autotune); no mask support; fp32 attention uses TF32 on GPU; Python runtime adds per-step launch overhead for unfused steps; one output per graph; compared against *unfused eager* — a fused library kernel such as `torch.nn.functional.scaled_dot_product_attention` is a stronger baseline and is not compared here.

`scripts/build.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
: "${LLVM_PREFIX:?set LLVM_PREFIX to your LLVM/MLIR install}"
export FUSIONPASS_BUILD_DIR="${FUSIONPASS_BUILD_DIR:-build}"
cmake -S . -B "$FUSIONPASS_BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR="$LLVM_PREFIX/lib/cmake/mlir" -DLLVM_DIR="$LLVM_PREFIX/lib/cmake/llvm"
cmake --build "$FUSIONPASS_BUILD_DIR"
export FUSIONPASS_COMPILE="$PWD/$FUSIONPASS_BUILD_DIR/bin/fusionpass-compile"
lit -v test/
pytest -q
```
**Verify:** `bash scripts/build.sh` ends with all lit and pytest tests passing.

### Step 10 — CI
`.github/workflows/ci.yml` (CPU-only: lit + pytest in Triton interpreter mode; no GPU in CI):
```yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-22.04
    env:
      LLVM_PREFIX: ${{ github.workspace }}/llvm-install
      TRITON_INTERPRET: "1"
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - name: Cache LLVM/MLIR install
        id: llvm-cache
        uses: actions/cache@v4
        with:
          path: llvm-install
          key: llvm-18.1.8-mlir-${{ runner.os }}
      - name: Build LLVM/MLIR (cache miss only; slow)
        if: steps.llvm-cache.outputs.cache-hit != 'true'
        run: |
          sudo apt-get update && sudo apt-get install -y ninja-build
          git clone --depth 1 --branch llvmorg-18.1.8 https://github.com/llvm/llvm-project.git
          cmake -S llvm-project/llvm -B llvm-project/build -G Ninja -DLLVM_ENABLE_PROJECTS=mlir \
            -DLLVM_TARGETS_TO_BUILD=host -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON \
            -DLLVM_INSTALL_UTILS=ON -DLLVM_BUILD_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX=$LLVM_PREFIX
          cmake --build llvm-project/build --target install
      - name: Python deps
        run: |
          sudo apt-get install -y ninja-build
          pip install torch --index-url https://download.pytorch.org/whl/cpu
          pip install pytest lit && pip install -e .
      - name: Build, lit, pytest
        run: bash scripts/build.sh
```
(⚠️ verify action versions and that the CPU torch wheel includes a Triton build usable by the interpreter; if not, install a compatible `triton` for CPU interpretation and record the pin in README.)

---

## 9. API / Interface Contracts

**Python (final public API)**
- `compile_model(model, example_inputs, *, fuse=False, workdir=None, verify=True, strict=True, atol=None, rtol=None) -> CompiledModel`
  - Raises: `UnsupportedOp`, `FileNotFoundError` (no compiler), `RuntimeError` (compiler failure), `EquivalenceError` (when `verify and strict` and results differ).
- `CompiledModel`: `.plan`, `.workdir`, `.mlir_text`, `.kernels`, `.report: VerifyReport | None`, `.verified: bool`; `__call__(*inputs)` raises `ValueError` on shape/dtype mismatch, `RuntimeError` for CPU triton without interpreter.
- `verify_equivalence(model, compiled, inputs, *, atol=None, rtol=None, extra_trials=2, seed=0) -> VerifyReport` (never raises on mismatch); `assert_equivalent(...)` raises `EquivalenceError`.
- `benchmark(model, compiled, inputs, *, unfused=None, warmup=25, iters=100) -> BenchResult` — raises `UnverifiedSpeedupError` if `compiled.verified` is not True; `RuntimeError` for non-CUDA inputs.
- `validate_plan(plan) -> None` raises `ValueError`.
- `estimate_bytes(plan) -> int`, `summarize_plan(plan) -> dict`.

**CLI**
- `python -m fusionpass check` → 0 ok, 3 compiler missing.
- `python -m fusionpass demo {attention|chain} [--dtype f16|f32] [--batch --heads --seq --head-dim] [--rows --cols]` → 0 ok, 2 verification failed / unverified, 3 compiler missing. Prints verification line, plan summary, and (CUDA only) timing and the estimated-traffic line.
- Console script `fusionpass` mirrors `python -m fusionpass`.

---

## 10. Error Handling & Edge Cases

| Situation | Handling |
|---|---|
| Fused result differs from eager | `EquivalenceError` (strict) or `verified=False` (non-strict); `benchmark` then raises `UnverifiedSpeedupError` |
| Reference overflows to Inf/NaN | Fused path must reproduce the same non-finite pattern, otherwise reported as a mismatch |
| Corrupt/stale `plan.json` | `validate_plan` → `ValueError` at `CompiledModel` construction |
| Unfusable structure (multi-use P, D=24, cross-attention shapes, non-last softmax) | Not fused; falls back to torch steps; still verified correct |
| Seq len 1 / non-multiple of 64 | Masked loads/stores; tested for S ∈ {1,7,64,65,130} |
| Duplicate operands (`x*x`) | Deduplicated externals; plan lists one operand |
| Non-contiguous call-time inputs | `.contiguous()` before kernel launch |
| Benchmark on CPU / interpreter | `RuntimeError` (only after the verification gate); demo prints "benchmark skipped" |
| Missing compiler binary | `FileNotFoundError` with remediation; CLI exit 3 |
| fp16 on CPU interpreter | Not tested; CLI demo downgrades to f32 on CPU |
| Triton JIT compile time in timing | Excluded by warmup (≥ 25 iterations) |
| Timing noise | Median of 100 CUDA-event timings; document GPU model and clocks state |

---

## 11. Testing & Verification

```bash
bash scripts/build.sh                      # build + lit (5) + pytest (all)
python -m fusionpass check
python -m fusionpass demo attention        # CPU: verifies; GPU: verifies + benchmarks
python -m fusionpass demo chain
```
GPU spot check (expected: `verified: OK`, fused plan has fewer steps than unfused, speedup line printed):
```bash
python -m fusionpass demo attention --dtype f16 --batch 4 --heads 8 --seq 1024 --head-dim 64
```
Negative check of the gate: run `pytest -q tests/test_verify_bench.py::test_gate_catches_miscompile` and `::test_benchmark_refuses_unverified` — both must pass.

---

## 12. Definition of Done

- [ ] `lit -v test/` passes: roundtrip, invalid, invalid_fused, fuse_attention, fuse_chains (incl. dedupe case).
- [ ] `pytest -q` passes on CPU interpreter (fp16 test skipped) and on GPU (all, incl. fp16).
- [ ] `compile_model(..., verify=True)` fails loudly (`EquivalenceError`) on a deliberately corrupted plan in tests; `benchmark` raises `UnverifiedSpeedupError` for unverified models.
- [ ] `python -m fusionpass demo attention` and `demo chain` run end to end; on GPU they print timing and traffic lines.
- [ ] README results table contains only numbers you measured (with GPU model) — or is left clearly marked "not yet measured".
- [ ] `docs/design.md`, `README.md`, `scripts/build.sh`, `.github/workflows/ci.yml` exist and are accurate.
- [ ] A clean-clone build (`bash scripts/build.sh`) succeeds from scratch given an LLVM install.
- [ ] Phase 1 and Phase 2 tests still pass unchanged (no regressions; `fuse` still defaults to `False`).

---

## 13. Notes & Pitfalls for the Implementer

- **Never report an unverified speedup** — do not add any code path that prints timings without `compiled.verified is True`.
- **Do not change `fuse`'s default** to `True`: Phase 1 tests assert all-`torch` plans by default.
- `verify=True` runs the model during `compile_model` (3 trials). Pass `verify=False` only for trace-only use (e.g., the traffic test).
- Verification trials use fixed seeds on CPU then move to the device, so results are reproducible across devices.
- Compare against **eager in the same dtype**; the tolerance table encodes the expected fp16/TF32 deviations. Tighten only with evidence.
- Speedup is relative to *unfused eager PyTorch*; do not market it as faster than `scaled_dot_product_attention`.
- The traffic number is a rough model, always labeled "estimate".
- CI cannot benchmark (no GPU). Do not add timing assertions to CI.
- Do not add features listed under "Out of scope" (masks, broadcasting, dynamic shapes, bf16, autotune, backward).
- Keep names/paths exactly as defined; `README` and CI reference them.

---

## 14. Deployment Steps & Project Completion Summary

### Delivery steps
1. **Install toolchain once:** build/install LLVM/MLIR 18.1.x to `$LLVM_PREFIX` (commands in `README`/CI).
2. **Build & test:** `bash scripts/build.sh` — must end green.
3. **Install Python package:** `pip install -e .` (or `pip install .` in a venv that already has a CUDA-enabled PyTorch).
4. **Point at the compiler:** `export FUSIONPASS_COMPILE=$PWD/build/bin/fusionpass-compile` (or keep `build/bin/` layout so the default lookup works).
5. **Smoke test on target machine:** `python -m fusionpass check` then `python -m fusionpass demo attention` and `demo chain`.
6. **Record results:** on the GPU machine, run the demos at your chosen shapes, copy the printed lines into the README results table together with the GPU model, PyTorch and Triton versions.
7. **Tag a release:** `git tag v0.1.0` after CI is green.

### Completion summary
`fusionpass` is a complete, working pipeline: a PyTorch model is traced to a custom MLIR dialect; two MLIR rewrite passes fuse attention subgraphs and elementwise/softmax chains; a C++ emitter generates Triton kernels (including a FlashAttention-style tiled kernel with online softmax) and an execution plan; a Python runtime runs the plan; and a verification harness gates every claim of speedup on numerical equivalence with eager PyTorch. It is covered by lit tests (passes, verifiers, negatives), pytest end-to-end and edge-case tests, a CLI with demos, documentation, and CI. Known limits (static shapes, no masks/broadcasting, fixed tile sizes, TF32 on fp32 GPU attention, unfused-eager baseline) are documented in the README and `docs/design.md`.
