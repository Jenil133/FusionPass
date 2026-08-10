# PHASE 1 — Foundation & Core Skeleton (fusionpass)

---

## 1. Project Context

**Project:** `fusionpass` — an MLIR-based compiler pass that fuses attention subgraphs and elementwise/softmax chains of a PyTorch model into single generated Triton kernels.

**What it is.** A PyTorch `nn.Module` (static shapes, f16/f32, inference only) is traced with `torch.fx` into textual MLIR in a small custom dialect called `fp`. A C++ tool, `fusionpass-compile`, parses that MLIR, optionally runs fusion passes (Phase 2), and emits two artifacts: `plan.json` (an ordered list of steps to execute) and `kernels.py` (generated Triton source). A Python runtime executes the plan: fused regions run as Triton kernels, everything else runs as plain PyTorch ops. A verification harness (Phase 3) proves numerical equivalence against eager PyTorch before any speedup is reported.

**Target user:** ML-systems engineers / students who want a working, readable example of an MLIR fusion pass with real GPU codegen.

**Problem solved:** each unfused op (matmul, scale, softmax, relu, ...) writes its result to GPU global memory and the next op reads it back. Fusing them into one kernel removes those round trips. For attention, fusion also avoids ever materializing the S×S score matrix.

**Whole-project scope boundaries (do not exceed):** static shapes only; dtypes f16 and f32 only; no broadcasting (elementwise operands must have identical shapes); inference only (no autograd); no causal/attention masks; one output value per graph; single device.

### Tech stack (identical in all three phase files)

| Layer | Choice |
|---|---|
| Compiler | C++17, **LLVM/MLIR 18.1.x** (tag `llvmorg-18.1.8` — ⚠️ verify tag exists), CMake ≥ 3.20, Ninja, clang or gcc |
| Compiler tests | `lit` (pip) + `FileCheck` (from LLVM install with `-DLLVM_INSTALL_UTILS=ON`) |
| Frontend / runtime | Python ≥ 3.10, PyTorch ≥ 2.2 (⚠️ verify), `torch.fx` + `torch.fx.passes.shape_prop.ShapeProp` |
| Kernel codegen target | **Triton** — use the version bundled with the PyTorch pip wheel (do NOT `pip install triton` separately) |
| GPU | NVIDIA GPU with CUDA for benchmarks (compute capability ≥ 8.0 recommended — ⚠️ verify). CPU correctness via `TRITON_INTERPRET=1` |
| Python tests | `pytest` |

**Stack decisions (why this over the alternative):**
- Custom `fp` dialect **over torch-mlir** — torch-mlir has a heavy build and tight LLVM version coupling; a small dialect keeps the pass surface tiny and fully under our control.
- Emit Triton source **over lowering to NVVM/LLVM** — Triton already does tiling, shared-memory staging, and tensor-core codegen; our pass focuses on *fusion decisions*.
- `torch.fx` frontend **over ONNX** — the reference model is the same `nn.Module`, and `ShapeProp` gives shapes/dtypes for free.
- JSON plan + Python runtime **over embedding Python in C++** — decouples stages; each is testable alone.
- Greedy pattern rewrites **over a custom graph algorithm** — idiomatic MLIR, small code.

### Naming conventions (must match across all phases)
- Dialect `fp`, C++ namespace `fusionpass::fp`; library `MLIRFPDialect`; plan library `MLIRFPPlan`.
- Tools: `fusionpass-opt` (mlir-opt-style driver), `fusionpass-compile` (MLIR → plan.json + kernels.py).
- Python package: `fusionpass` under `python/`. Entry MLIR function is always `func.func @main`.
- Plan value names: function args `arg0, arg1, ...`; op results `v0, v1, ...` in program order.
- Generated kernel names (Phase 2): `fp_chain_<N>`, and a single shared `fp_attention`.
- Output files in an out-dir: `plan.json`, `kernels.py`, `optimized.mlir` (Phase 2).
- Env vars: `LLVM_PREFIX` (LLVM install dir), `FUSIONPASS_BUILD_DIR` (default `build`), `FUSIONPASS_COMPILE` (path to `fusionpass-compile`).

---

## 2. Architecture Overview

```
 PyTorch nn.Module
      │  torch.fx symbolic_trace + ShapeProp          python/fusionpass/frontend.py      ◄ PHASE 1
      ▼
 model.mlir   (func.func @main + fp dialect ops)
      │  fusionpass-compile   (C++ / MLIR)
      │     parse → verify → [fp-fuse-attention → fp-fuse-chains] (Phase 2) → emit plan   ◄ PHASE 1 (parse/verify/emit only)
      ▼
 plan.json  (+ kernels.py in Phase 2)
      │  python/fusionpass/runtime.py                                                     ◄ PHASE 1 (torch steps only)
      ▼
 PyTorch ops (unfused)  |  Triton kernels (fused, Phase 2)  → CUDA GPU (or Triton interpreter on CPU)
      │
 verify.py (equivalence gate) → bench.py (speedup)                                        ◄ Phase 3
```

**This phase's place:** Phase 1 builds everything left of "fusion": the dialect, the C++ tools, the frontend, the plan format, and a runtime that executes an *unfused* plan with PyTorch. End state: a PyTorch model round-trips through MLIR and produces output identical to eager PyTorch. No fusion yet, no Triton yet, no GPU needed.

**Final project layout (reference — Phase 1 creates only part of this):**
```
fusionpass/
├── CMakeLists.txt  README.md  pyproject.toml
├── include/fusionpass/Dialect/FP/{CMakeLists.txt,FPOps.td,FPDialect.h,FPOps.h}
├── include/fusionpass/Transforms/Passes.h                      (Phase 2)
├── include/fusionpass/Plan/{PlanEmitter.h,TritonEmitter.h}     (PlanEmitter.h: P1; TritonEmitter.h: P2)
├── lib/Dialect/FP/{CMakeLists.txt,FPDialect.cpp,FPOps.cpp}
├── lib/Transforms/{CMakeLists.txt,FuseAttention.cpp,FuseChains.cpp,Passes.cpp}   (Phase 2)
├── lib/Plan/{CMakeLists.txt,PlanEmitter.cpp,TritonEmitter.cpp} (TritonEmitter.cpp: P2)
├── tools/{CMakeLists.txt,fusionpass-opt/fusionpass-opt.cpp,fusionpass-compile/fusionpass-compile.cpp}
├── python/fusionpass/{__init__.py,frontend.py,runtime.py,api.py,verify.py,bench.py,traffic.py,__main__.py}
├── test/{lit.cfg.py,*.mlir}            (lit tests)
├── tests/{conftest.py,test_*.py}       (pytest)
└── examples/ docs/ .github/workflows/  (Phase 3)
```

---

## 3. Current State

**Empty project / nothing exists yet.** Create a new directory `fusionpass/` and `git init` it. LLVM/MLIR is also not yet built — Step 1 does that.

---

## 4. Phase Goal & Scope

**Goal:** a runnable, verified skeleton: (a) the `fp` dialect parses/prints/verifies; (b) a `torch.fx` frontend turns a small PyTorch model into valid `fp` MLIR; (c) `fusionpass-compile` converts that MLIR to `plan.json`; (d) the Python runtime executes the plan and matches eager PyTorch.

**In scope**
- Build/install LLVM+MLIR; CMake project; ODS dialect with ops: `add, mul, scale, relu, exp, softmax, matmul, transpose`, with verifiers.
- `fusionpass-opt` and `fusionpass-compile` (pipeline `none` only).
- `PlanEmitter` producing `plan.json` with `torch` steps only.
- Python: `frontend.py`, `runtime.py` (torch steps), `api.py` (`compile_model`, `fuse=False` only).
- lit tests (round-trip + verifier errors) and pytest end-to-end tests (CPU only).

**Out of scope**
- Fused ops, fusion passes, Triton codegen, triton plan steps → **Phase 2**.
- Equivalence-gate harness, benchmarking, CLI demos, docs, CI, packaging polish → **Phase 3**.

---

## 5. Prerequisites & Setup

```bash
# Tools: git, cmake>=3.20, ninja, clang or gcc (C++17), python>=3.10
python -m venv .venv && source .venv/bin/activate
pip install torch pytest lit          # torch wheel bundles Triton (used from Phase 2)

# Environment (add to your shell profile)
export LLVM_PREFIX=$HOME/llvm-install
export FUSIONPASS_BUILD_DIR=build
```

**Env/config values (placeholders, no secrets are used in this project):** `LLVM_PREFIX`, `FUSIONPASS_BUILD_DIR`, `FUSIONPASS_COMPILE` (set after the build: `export FUSIONPASS_COMPILE=$PWD/build/bin/fusionpass-compile`).

**No external accounts or API keys are required.**

---

## 6. File Structure for This Phase

```
fusionpass/
├── CMakeLists.txt                                   NEW
├── pyproject.toml                                   NEW
├── .gitignore                                       NEW   (build/, .venv/, __pycache__/, *.egg-info)
├── include/
│   ├── CMakeLists.txt                               NEW
│   └── fusionpass/
│       ├── Dialect/FP/
│       │   ├── CMakeLists.txt                       NEW
│       │   ├── FPOps.td                             NEW
│       │   ├── FPDialect.h                          NEW
│       │   └── FPOps.h                              NEW
│       └── Plan/PlanEmitter.h                       NEW
├── lib/
│   ├── CMakeLists.txt                               NEW
│   ├── Dialect/FP/{CMakeLists.txt,FPDialect.cpp,FPOps.cpp}   NEW
│   └── Plan/{CMakeLists.txt,PlanEmitter.cpp}        NEW
├── tools/
│   ├── CMakeLists.txt                               NEW
│   ├── fusionpass-opt/fusionpass-opt.cpp            NEW
│   └── fusionpass-compile/fusionpass-compile.cpp    NEW
├── python/fusionpass/{__init__.py,frontend.py,runtime.py,api.py}   NEW
├── test/{lit.cfg.py,roundtrip.mlir,invalid.mlir}    NEW
└── tests/{conftest.py,test_frontend.py,test_e2e.py} NEW
```

---

## 7. Data Model / Schema

There is no database. The "schema" is (A) the `fp` dialect's op definitions and (B) the `plan.json` format.

### A. Dialect ops (all tensors are static-shape ranked tensors of `f16`/`f32`)

| Op | Syntax | Semantics |
|---|---|---|
| `fp.add` | `%r = fp.add %a, %b : T` | elementwise sum; `a,b,r` same type `T` (no broadcasting) |
| `fp.mul` | `%r = fp.mul %a, %b : T` | elementwise product; same type |
| `fp.scale` | `%r = fp.scale %x {factor = 0.125 : f32} : T` | `x * factor` |
| `fp.relu` | `%r = fp.relu %x : T` | `max(x, 0)` |
| `fp.exp` | `%r = fp.exp %x : T` | `exp(x)` |
| `fp.softmax` | `%r = fp.softmax %x {axis = 3 : i64} : T` | softmax along a **non-negative** axis |
| `fp.matmul` | `%r = fp.matmul %a, %b : (Ta, Tb) -> Tr` | batched matmul; equal rank ≥ 2, identical batch dims, `[...,M,K]x[...,K,N]->[...,M,N]` |
| `fp.transpose` | `%r = fp.transpose %x : Tx -> Tr` | swaps the **last two** dims |

Example module:
```mlir
module {
  func.func @main(%arg0: tensor<1x2x8x16xf32>, %arg1: tensor<1x2x8x16xf32>, %arg2: tensor<1x2x8x16xf32>) -> tensor<1x2x8x16xf32> {
    %kt = fp.transpose %arg1 : tensor<1x2x8x16xf32> -> tensor<1x2x16x8xf32>
    %s = fp.matmul %arg0, %kt : (tensor<1x2x8x16xf32>, tensor<1x2x16x8xf32>) -> tensor<1x2x8x8xf32>
    %sc = fp.scale %s {factor = 2.500000000e-01 : f32} : tensor<1x2x8x8xf32>
    %p = fp.softmax %sc {axis = 3 : i64} : tensor<1x2x8x8xf32>
    %o = fp.matmul %p, %arg2 : (tensor<1x2x8x8xf32>, tensor<1x2x8x16xf32>) -> tensor<1x2x8x16xf32>
    return %o : tensor<1x2x8x16xf32>
  }
}
```

### B. `plan.json` schema (version 1)
```json
{
  "version": 1,
  "inputs":  [ {"name": "arg0", "shape": [1,2,8,16], "dtype": "f32"} ],
  "steps": [
    {"kind": "torch", "op": "matmul", "operands": ["arg0", "v0"],
     "result": "v1", "shape": [1,2,8,8], "dtype": "f32", "attrs": {}}
  ],
  "outputs": ["v4"]
}
```
- `kind`: Phase 1 only emits `"torch"`. (`"triton"` is added in Phase 2.)
- `op` ∈ `add, mul, scale, relu, exp, softmax, matmul, transpose`.
- `attrs`: `scale` → `{"factor": <float>}`; `softmax` → `{"axis": <int>}`; others `{}`.
- `dtype` ∈ `"f16"`, `"f32"`.

---

## 8. Implementation Steps

### Step 1 — Build and install LLVM/MLIR
```bash
git clone --depth 1 --branch llvmorg-18.1.8 https://github.com/llvm/llvm-project.git
cd llvm-project && mkdir build && cd build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD="host" \
  -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_INSTALL_UTILS=ON -DLLVM_BUILD_EXAMPLES=OFF \
  -DCMAKE_INSTALL_PREFIX=$LLVM_PREFIX
ninja install            # long (tens of minutes); only done once
```
**Verify:** `$LLVM_PREFIX/bin/mlir-opt --version` prints `LLVM version 18.1.x`; `ls $LLVM_PREFIX/bin/FileCheck` exists.

### Step 2 — CMake skeleton
`CMakeLists.txt`:
```cmake
cmake_minimum_required(VERSION 3.20)
project(fusionpass LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)   # sets -fno-rtti etc. to match the LLVM build

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
```
`include/CMakeLists.txt`: `add_subdirectory(fusionpass/Dialect/FP)`
`include/fusionpass/Dialect/FP/CMakeLists.txt`: `add_mlir_dialect(FPOps fp)`  (generates `FPOps.h.inc`, `FPOps.cpp.inc`, `FPOpsDialect.h.inc`, `FPOpsDialect.cpp.inc`, target `MLIRFPOpsIncGen`)
`lib/CMakeLists.txt`: `add_subdirectory(Dialect/FP)` and `add_subdirectory(Plan)`
`lib/Dialect/FP/CMakeLists.txt`:
```cmake
add_mlir_dialect_library(MLIRFPDialect
  FPDialect.cpp
  FPOps.cpp
  ADDITIONAL_HEADER_DIRS ${PROJECT_SOURCE_DIR}/include/fusionpass
  DEPENDS MLIRFPOpsIncGen
  LINK_LIBS PUBLIC MLIRIR MLIRSideEffectInterfaces)
```
`lib/Plan/CMakeLists.txt`:
```cmake
add_mlir_library(MLIRFPPlan
  PlanEmitter.cpp
  LINK_LIBS PUBLIC MLIRFPDialect MLIRFuncDialect MLIRIR LLVMSupport)
```
`tools/CMakeLists.txt`:
```cmake
add_llvm_executable(fusionpass-opt fusionpass-opt/fusionpass-opt.cpp)
llvm_update_compile_flags(fusionpass-opt)
target_link_libraries(fusionpass-opt PRIVATE MLIROptLib MLIRFuncDialect MLIRFPDialect)

add_llvm_executable(fusionpass-compile fusionpass-compile/fusionpass-compile.cpp)
llvm_update_compile_flags(fusionpass-compile)
target_link_libraries(fusionpass-compile PRIVATE MLIRParser MLIRPass MLIRFuncDialect MLIRFPDialect MLIRFPPlan)
```
Mirror the structure of `mlir/examples/standalone` in the pinned LLVM tag if any CMake detail differs (⚠️ verify). Configure command used from now on:
```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=$LLVM_PREFIX/lib/cmake/mlir -DLLVM_DIR=$LLVM_PREFIX/lib/cmake/llvm
cmake --build build
```
**Verify:** configure succeeds (build will fail until later steps add sources — that's expected).

### Step 3 — Dialect definition (ODS)
`include/fusionpass/Dialect/FP/FPOps.td`:
```tablegen
#ifndef FP_OPS
#define FP_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

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

#endif // FP_OPS
```
**Verify:** `ninja -C build MLIRFPOpsIncGen` generates the `.inc` files under `build/include/fusionpass/Dialect/FP/`.

### Step 4 — Dialect C++ glue and verifiers
`include/fusionpass/Dialect/FP/FPDialect.h`:
```cpp
#pragma once
#include "mlir/IR/Dialect.h"
#include "fusionpass/Dialect/FP/FPOpsDialect.h.inc"
```
`include/fusionpass/Dialect/FP/FPOps.h`:
```cpp
#pragma once
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "fusionpass/Dialect/FP/FPDialect.h"
#define GET_OP_CLASSES
#include "fusionpass/Dialect/FP/FPOps.h.inc"
```
`lib/Dialect/FP/FPDialect.cpp`:
```cpp
#include "fusionpass/Dialect/FP/FPDialect.h"
#include "fusionpass/Dialect/FP/FPOps.h"
using namespace fusionpass::fp;
#include "fusionpass/Dialect/FP/FPOpsDialect.cpp.inc"

void FPDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "fusionpass/Dialect/FP/FPOps.cpp.inc"
      >();
}
```
`lib/Dialect/FP/FPOps.cpp`:
```cpp
#include "fusionpass/Dialect/FP/FPOps.h"
#include "mlir/IR/Builders.h"
using namespace mlir;
using namespace fusionpass::fp;

#define GET_OP_CLASSES
#include "fusionpass/Dialect/FP/FPOps.cpp.inc"

LogicalResult SoftmaxOp::verify() {
  auto t = cast<RankedTensorType>(getInput().getType());
  // WHY non-negative: the frontend normalizes dim; passes compare axis == rank-1 directly.
  if (getAxis() < 0 || getAxis() >= t.getRank())
    return emitOpError("axis must be in [0, rank)");
  return success();
}

LogicalResult MatmulOp::verify() {
  auto l = cast<RankedTensorType>(getLhs().getType());
  auto r = cast<RankedTensorType>(getRhs().getType());
  auto o = cast<RankedTensorType>(getResult().getType());
  int64_t rank = l.getRank();
  if (rank < 2 || r.getRank() != rank || o.getRank() != rank)
    return emitOpError("operands and result must have equal rank >= 2");
  if (l.getElementType() != r.getElementType() || l.getElementType() != o.getElementType())
    return emitOpError("element types must match");
  SmallVector<int64_t> expected;
  for (int64_t i = 0; i < rank - 2; ++i) {
    if (l.getDimSize(i) != r.getDimSize(i))
      return emitOpError("batch dimensions must match");
    expected.push_back(l.getDimSize(i));
  }
  if (l.getDimSize(rank - 1) != r.getDimSize(rank - 2))
    return emitOpError("contraction dimension mismatch");
  expected.push_back(l.getDimSize(rank - 2));
  expected.push_back(r.getDimSize(rank - 1));
  if (o.getShape() != ArrayRef<int64_t>(expected))
    return emitOpError("result shape does not match matmul semantics");
  return success();
}

LogicalResult TransposeOp::verify() {
  auto i = cast<RankedTensorType>(getInput().getType());
  auto o = cast<RankedTensorType>(getResult().getType());
  if (i.getRank() < 2 || o.getRank() != i.getRank())
    return emitOpError("rank must be >= 2 and preserved");
  SmallVector<int64_t> expected(i.getShape().begin(), i.getShape().end());
  std::swap(expected[expected.size() - 1], expected[expected.size() - 2]);
  if (o.getShape() != ArrayRef<int64_t>(expected) || i.getElementType() != o.getElementType())
    return emitOpError("result must be the input with its last two dims swapped");
  return success();
}
```
**Verify:** `ninja -C build MLIRFPDialect` compiles.

### Step 5 — `fusionpass-opt` driver
`tools/fusionpass-opt/fusionpass-opt.cpp`:
```cpp
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "fusionpass/Dialect/FP/FPDialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, fusionpass::fp::FPDialect>();
  // Phase 2 adds: fusionpass::registerFusionPasses();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "fusionpass optimizer driver\n", registry));
}
```
**Verify:** `build/bin/fusionpass-opt --help` runs.

### Step 6 — lit tests for the dialect
`test/lit.cfg.py`:
```python
import os, lit.formats
build = os.environ.get("FUSIONPASS_BUILD_DIR", "build")
config.name = "fusionpass"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(os.path.abspath(build), "test")
config.substitutions.append(("%fusionpass-opt", os.path.join(os.path.abspath(build), "bin", "fusionpass-opt")))
config.environment["PATH"] = os.path.join(os.environ["LLVM_PREFIX"], "bin") + os.pathsep + os.environ["PATH"]
```
`test/roundtrip.mlir`:
```mlir
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
```
`test/invalid.mlir`:
```mlir
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
```
**Verify:** `lit -v test/` → 2 passed (needs `LLVM_PREFIX` and `FUSIONPASS_BUILD_DIR` set).

### Step 7 — Plan emitter (C++)
`include/fusionpass/Plan/PlanEmitter.h`:
```cpp
#pragma once
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/JSON.h"
namespace fusionpass {
/// Walks `func.func @main` in `module` and produces the plan.json value.
mlir::LogicalResult emitPlan(mlir::ModuleOp module, llvm::json::Value &out);
} // namespace fusionpass
```
`lib/Plan/PlanEmitter.cpp`:
```cpp
#include "fusionpass/Plan/PlanEmitter.h"
#include "fusionpass/Dialect/FP/FPOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/TypeSwitch.h"
using namespace mlir;
using namespace fusionpass::fp;
namespace json = llvm::json;

static std::string dtypeStr(Type t) { return t.isF16() ? "f16" : t.isF32() ? "f32" : "?"; }

static json::Array shapeArr(RankedTensorType t) {
  json::Array a;
  for (int64_t d : t.getShape()) a.push_back(d);
  return a;
}

LogicalResult fusionpass::emitPlan(ModuleOp module, json::Value &out) {
  auto fn = module.lookupSymbol<func::FuncOp>("main");
  if (!fn) return module.emitError("expected func.func @main");
  DenseMap<Value, std::string> names;
  json::Array inputs, steps, outputs;
  int next = 0;

  for (auto [i, arg] : llvm::enumerate(fn.getArguments())) {
    std::string n = "arg" + std::to_string(i);
    names[arg] = n;
    auto t = cast<RankedTensorType>(arg.getType());
    inputs.push_back(json::Object{{"name", n}, {"shape", shapeArr(t)}, {"dtype", dtypeStr(t.getElementType())}});
  }

  for (Operation &op : fn.getBody().front()) {
    if (auto ret = dyn_cast<func::ReturnOp>(op)) {
      for (Value v : ret.getOperands()) outputs.push_back(names[v]);
      continue;
    }
    if (op.getNumResults() != 1)
      return op.emitError("unsupported: op must have exactly one result");

    json::Object attrs;
    StringRef opName = op.getName().stripDialect();
    // WHY TypeSwitch: only fp ops are legal in Phase 1; anything else is a hard error, never silently skipped.
    bool ok = llvm::TypeSwitch<Operation *, bool>(&op)
        .Case<AddOp, MulOp, ReluOp, ExpOp, MatmulOp, TransposeOp>([&](auto) { return true; })
        .Case<ScaleOp>([&](ScaleOp s) { attrs["factor"] = (double)s.getFactor().convertToFloat(); return true; })
        .Case<SoftmaxOp>([&](SoftmaxOp s) { attrs["axis"] = (int64_t)s.getAxis(); return true; })
        .Default([](Operation *) { return false; });
    if (!ok) return op.emitError("plan emitter: unsupported op");

    json::Array operands;
    for (Value v : op.getOperands()) operands.push_back(names[v]);
    std::string res = "v" + std::to_string(next++);
    names[op.getResult(0)] = res;
    auto rt = cast<RankedTensorType>(op.getResult(0).getType());
    steps.push_back(json::Object{{"kind", "torch"}, {"op", opName.str()}, {"operands", std::move(operands)},
                                 {"result", res}, {"shape", shapeArr(rt)}, {"dtype", dtypeStr(rt.getElementType())},
                                 {"attrs", std::move(attrs)}});
  }
  out = json::Object{{"version", 1}, {"inputs", std::move(inputs)}, {"steps", std::move(steps)}, {"outputs", std::move(outputs)}};
  return success();
}
```
**Verify:** `ninja -C build MLIRFPPlan` compiles. (`json::Object` initializer-list construction with `std::string`/`json::Array` values is standard; if a conversion ambiguity appears for `int64_t`, cast explicitly — ⚠️ verify.)

### Step 8 — `fusionpass-compile` driver
`tools/fusionpass-compile/fusionpass-compile.cpp`:
```cpp
#include "fusionpass/Dialect/FP/FPDialect.h"
#include "fusionpass/Plan/PlanEmitter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input.mlir>"), cl::Required);
static cl::opt<std::string> outDir("o", cl::desc("output directory"), cl::Required);
static cl::opt<std::string> pipeline("pipeline", cl::desc("none | fuse (fuse arrives in Phase 2)"), cl::init("none"));

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "fusionpass compiler driver\n");
  MLIRContext ctx;
  ctx.loadDialect<func::FuncDialect, fusionpass::fp::FPDialect>();
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(inputFile, &ctx);
  if (!module) return 1;
  if (failed(verify(*module))) return 1;
  if (pipeline != "none") { llvm::errs() << "error: unknown --pipeline '" << pipeline << "'\n"; return 2; }

  llvm::json::Value plan(nullptr);
  if (failed(fusionpass::emitPlan(*module, plan))) return 1;

  if (auto ec = llvm::sys::fs::create_directories(outDir)) { llvm::errs() << "cannot create " << outDir << ": " << ec.message() << "\n"; return 1; }
  std::error_code ec;
  llvm::raw_fd_ostream os(llvm::Twine(outDir + "/plan.json").str(), ec);
  if (ec) { llvm::errs() << "cannot write plan.json: " << ec.message() << "\n"; return 1; }
  os << llvm::formatv("{0:2}\n", plan);
  return 0;
}
```
**Verify:** build; then
`build/bin/fusionpass-compile test_input.mlir -o /tmp/out && cat /tmp/out/plan.json` (use the example module from Section 7) shows 5 steps, each `"kind": "torch"`, `outputs: ["v4"]`.

### Step 9 — Python packaging
`pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "fusionpass"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["torch>=2.2"]

[project.optional-dependencies]
dev = ["pytest", "lit"]

[tool.setuptools.packages.find]
where = ["python"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```
`python/fusionpass/__init__.py`:
```python
from .api import compile_model, CompiledModel
from .frontend import UnsupportedOp
__all__ = ["compile_model", "CompiledModel", "UnsupportedOp"]
```
**Verify:** `pip install -e .[dev]` then `python -c "import fusionpass"` — will fail until Steps 10–12 exist; do them, then re-check.

### Step 10 — Frontend: `torch.fx` → textual MLIR
`python/fusionpass/frontend.py`:
```python
from __future__ import annotations
import operator
import torch
import torch.fx as fx
import torch.nn.functional as F
from torch.fx.passes.shape_prop import ShapeProp


class UnsupportedOp(Exception):
    """Raised when the traced graph uses something outside the supported subset.
    WHY raise instead of falling back: a silent fallback would hide that the model was not compiled."""


_DT = {torch.float32: "f32", torch.float16: "f16"}
_MATMUL = {torch.matmul, operator.matmul}
_SOFTMAX = {torch.softmax, F.softmax}
_RELU = {torch.relu, F.relu}
_ADD = {operator.add, torch.add}
_MUL = {operator.mul, torch.mul}
_DIV = {operator.truediv, torch.div}


def _ty(shape, dtype) -> str:
    if dtype not in _DT:
        raise UnsupportedOp(f"unsupported dtype {dtype}")
    if len(shape) == 0:
        raise UnsupportedOp("rank-0 tensors are unsupported")
    return "tensor<" + "x".join(str(int(d)) for d in shape) + "x" + _DT[dtype] + ">"


def _f32(x: float) -> str:
    # WHY .9e: MLIR float literals require a '.', and repr(1e-05) has none.
    return f"{float(x):.9e}"


def _fetch_attr(root, target: str):
    obj = root
    for part in target.split("."):
        obj = getattr(obj, part)
    return obj


def trace_to_mlir(model: torch.nn.Module, example_inputs):
    """Returns (mlir_text, arg_tensors). arg_tensors = example inputs followed by captured
    parameters/buffers, in the same order as func.func @main's arguments."""
    gm = fx.symbolic_trace(model)
    ShapeProp(gm).propagate(*example_inputs)

    arg_nodes, arg_tensors = [], []
    ph = iter(example_inputs)
    for n in gm.graph.nodes:
        if n.op == "placeholder":
            arg_nodes.append(n); arg_tensors.append(next(ph))
    for n in gm.graph.nodes:
        if n.op == "get_attr":
            t = _fetch_attr(gm, n.target)
            if not isinstance(t, torch.Tensor):
                raise UnsupportedOp(f"get_attr {n.target} is not a tensor")
            arg_nodes.append(n); arg_tensors.append(t.detach())

    names, types = {}, {}
    for i, (n, t) in enumerate(zip(arg_nodes, arg_tensors)):
        names[n] = f"%arg{i}"
        types[n] = _ty(t.shape, t.dtype)

    def meta_ty(n):
        m = n.meta.get("tensor_meta")
        if m is None:
            raise UnsupportedOp(f"node {n.name} has no tensor shape info")
        return _ty(m.shape, m.dtype)

    lines, out_node = [], None
    for n in gm.graph.nodes:
        if n.op in ("placeholder", "get_attr"):
            continue
        if n.op == "output":
            out = n.args[0]
            if not isinstance(out, fx.Node):
                raise UnsupportedOp("exactly one tensor output is supported")
            out_node = out
            continue
        if n.op not in ("call_function", "call_method"):
            raise UnsupportedOp(f"node op {n.op} ({n.target}) is unsupported (call_module is not inlined)")
        names[n] = f"%{n.name}"
        types[n] = meta_ty(n)
        lines.append("    " + _emit(n, names, types))

    if out_node is None:
        raise UnsupportedOp("graph has no output")
    sig = ", ".join(f"%arg{i}: {types[n]}" for i, n in enumerate(arg_nodes))
    body = "\n".join(lines)
    text = (f"module {{\n  func.func @main({sig}) -> {types[out_node]} {{\n{body}\n"
            f"    return {names[out_node]} : {types[out_node]}\n  }}\n}}\n")
    return text, arg_tensors


def _shape(n: fx.Node):
    return tuple(n.meta["tensor_meta"].shape)


def _norm_axis(dim, rank):
    if dim is None:
        raise UnsupportedOp("softmax requires an explicit dim")
    if not -rank <= dim < rank:
        raise UnsupportedOp(f"softmax dim {dim} out of range")
    return dim % rank


def _emit(n: fx.Node, names, types) -> str:
    tgt, args, kwargs = n.target, n.args, n.kwargs
    res, rty = names[n], types[n]
    is_method = n.op == "call_method"

    def ref(x): return names[x]

    if (not is_method and tgt in _MATMUL) or (is_method and tgt == "matmul"):
        a, b = args[:2]
        if len(_shape(a)) != len(_shape(b)):
            raise UnsupportedOp("matmul with differing ranks (broadcast) is unsupported")
        return f"{res} = fp.matmul {ref(a)}, {ref(b)} : ({types[a]}, {types[b]}) -> {rty}"

    if (not is_method and tgt in _SOFTMAX) or (is_method and tgt == "softmax"):
        x = args[0]
        dim = kwargs.get("dim", args[1] if len(args) > 1 else None)
        if kwargs.get("dtype") is not None:
            raise UnsupportedOp("softmax with dtype= is unsupported")
        return f"{res} = fp.softmax {ref(x)} {{axis = {_norm_axis(dim, len(_shape(x)))} : i64}} : {rty}"

    if (not is_method and tgt in _RELU) or (is_method and tgt == "relu"):
        return f"{res} = fp.relu {ref(args[0])} : {rty}"

    if (not is_method and tgt is torch.exp) or (is_method and tgt == "exp"):
        return f"{res} = fp.exp {ref(args[0])} : {rty}"

    if is_method and tgt == "transpose":
        x, d0, d1 = args[:3]
        r = len(_shape(x))
        if {d0 % r, d1 % r} != {r - 2, r - 1}:
            raise UnsupportedOp("only transpose of the last two dims is supported")
        return f"{res} = fp.transpose {ref(x)} : {types[x]} -> {rty}"

    kind = None
    if not is_method:
        kind = "add" if tgt in _ADD else "mul" if tgt in _MUL else "div" if tgt in _DIV else None
    else:
        kind = {"add": "add", "mul": "mul", "div": "div", "__add__": "add", "__mul__": "mul", "__truediv__": "div"}.get(tgt)
    if kind is not None:
        a, b = args[:2]
        a_t, b_t = isinstance(a, fx.Node), isinstance(b, fx.Node)
        if a_t and b_t:
            if kind == "div":
                raise UnsupportedOp("tensor/tensor division is unsupported")
            if _shape(a) != _shape(b):
                raise UnsupportedOp(f"broadcasting is unsupported: {_shape(a)} vs {_shape(b)}")
            return f"{res} = fp.{kind} {ref(a)}, {ref(b)} : {rty}"
        if kind in ("mul", "div") and (a_t != b_t):
            x, c = (a, b) if a_t else (b, a)
            if kind == "div" and not a_t:
                raise UnsupportedOp("scalar / tensor is unsupported")
            if not isinstance(c, (int, float)):
                raise UnsupportedOp("scale factor must be a Python number")
            factor = float(c) if kind == "mul" else 1.0 / float(c)
            return f"{res} = fp.scale {ref(x)} {{factor = {_f32(factor)} : f32}} : {rty}"
        raise UnsupportedOp(f"{kind} with a Python scalar is unsupported (only mul/div by a constant)")

    raise UnsupportedOp(f"unsupported op: {n.op} {tgt}")
```
**Verify:** Step 12 tests exercise it; interactively `trace_to_mlir(Chain(), (torch.randn(8,16),))[0]` prints a `func.func @main` with `fp.scale`, `fp.add`, `fp.relu`.

### Step 11 — Runtime (torch steps only)
`python/fusionpass/runtime.py`:
```python
from __future__ import annotations
import torch

_DTYPES = {"f32": torch.float32, "f16": torch.float16}


def _run_torch(step: dict, ops: list[torch.Tensor]) -> torch.Tensor:
    op, attrs = step["op"], step["attrs"]
    if op == "add": return torch.add(ops[0], ops[1])
    if op == "mul": return torch.mul(ops[0], ops[1])
    if op == "scale": return ops[0] * attrs["factor"]
    if op == "relu": return torch.relu(ops[0])
    if op == "exp": return torch.exp(ops[0])
    if op == "softmax": return torch.softmax(ops[0], dim=attrs["axis"])
    if op == "matmul": return torch.matmul(ops[0], ops[1])
    if op == "transpose": return ops[0].transpose(-1, -2)
    raise ValueError(f"unknown torch op in plan: {op}")


def execute_plan(plan: dict, args: list[torch.Tensor], kernels=None):
    if plan.get("version") != 1:
        raise ValueError(f"unsupported plan version {plan.get('version')}")
    if len(args) != len(plan["inputs"]):
        raise ValueError(f"expected {len(plan['inputs'])} inputs, got {len(args)}")
    env = {}
    for spec, t in zip(plan["inputs"], args):
        if list(t.shape) != spec["shape"] or t.dtype != _DTYPES[spec["dtype"]]:
            # WHY strict: the plan (and later, kernels) are specialized to these static shapes.
            raise ValueError(f"{spec['name']}: expected {spec['dtype']}{spec['shape']}, got {t.dtype}{list(t.shape)}")
        env[spec["name"]] = t
    for step in plan["steps"]:
        ops = [env[o] for o in step["operands"]]
        if step["kind"] == "torch":
            env[step["result"]] = _run_torch(step, ops)
        else:
            raise NotImplementedError(f"step kind '{step['kind']}' is added in Phase 2")
    outs = [env[o] for o in plan["outputs"]]
    return outs[0] if len(outs) == 1 else tuple(outs)
```
**Verify:** imported by Step 12.

### Step 12 — `api.py` and tests
`python/fusionpass/api.py`:
```python
from __future__ import annotations
import json, os, subprocess, tempfile
from pathlib import Path
import torch
from .frontend import trace_to_mlir
from .runtime import execute_plan


def _find_compiler() -> str:
    p = os.environ.get("FUSIONPASS_COMPILE")
    if not p:
        p = str(Path(os.environ.get("FUSIONPASS_BUILD_DIR", "build")) / "bin" / "fusionpass-compile")
    if not Path(p).exists():
        raise FileNotFoundError(f"fusionpass-compile not found at {p}; build the C++ project or set FUSIONPASS_COMPILE")
    return p


class CompiledModel:
    def __init__(self, plan: dict, captured: list[torch.Tensor], workdir: Path, mlir_text: str):
        self.plan, self._captured, self.workdir, self.mlir_text = plan, captured, workdir, mlir_text

    def __call__(self, *inputs):
        return execute_plan(self.plan, list(inputs) + self._captured)


def compile_model(model, example_inputs, *, fuse: bool = False, workdir=None) -> CompiledModel:
    if fuse:
        raise NotImplementedError("fusion arrives in Phase 2")
    example_inputs = tuple(example_inputs)
    mlir_text, arg_tensors = trace_to_mlir(model, example_inputs)
    workdir = Path(workdir or tempfile.mkdtemp(prefix="fusionpass_"))
    workdir.mkdir(parents=True, exist_ok=True)
    src = workdir / "model.mlir"
    src.write_text(mlir_text)
    proc = subprocess.run([_find_compiler(), str(src), "-o", str(workdir), "--pipeline", "none"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"fusionpass-compile failed:\n{proc.stderr}\nMLIR:\n{mlir_text}")
    plan = json.loads((workdir / "plan.json").read_text())
    return CompiledModel(plan, arg_tensors[len(example_inputs):], workdir, mlir_text)
```
`tests/conftest.py`:
```python
import os, shutil, pytest
from pathlib import Path

@pytest.fixture(scope="session", autouse=True)
def _require_compiler():
    p = os.environ.get("FUSIONPASS_COMPILE") or str(Path(os.environ.get("FUSIONPASS_BUILD_DIR", "build")) / "bin" / "fusionpass-compile")
    if not Path(p).exists():
        pytest.skip("fusionpass-compile not built", allow_module_level=False)
```
`tests/test_frontend.py`:
```python
import pytest, torch
from torch import nn
from fusionpass.frontend import trace_to_mlir, UnsupportedOp

class Chain(nn.Module):
    def __init__(self):
        super().__init__(); self.b = nn.Parameter(torch.randn(8, 16))
    def forward(self, x):
        return torch.exp(torch.relu(x * 2.0 + self.b))

def test_chain_emits_fp_ops():
    text, args = trace_to_mlir(Chain(), (torch.randn(8, 16),))
    for tok in ("func.func @main", "fp.scale", "fp.add", "fp.relu", "fp.exp"):
        assert tok in text
    assert len(args) == 2          # input + captured parameter

def test_broadcast_rejected():
    class B(nn.Module):
        def forward(self, x, y): return x + y
    with pytest.raises(UnsupportedOp):
        trace_to_mlir(B(), (torch.randn(4, 8), torch.randn(8)))
```
`tests/test_e2e.py`:
```python
import torch
from torch import nn
from fusionpass import compile_model

class Chain(nn.Module):
    def __init__(self):
        super().__init__(); self.b = nn.Parameter(torch.randn(8, 16))
    def forward(self, x):
        return torch.exp(torch.relu(x * 2.0 + self.b))

class Attn(nn.Module):
    def forward(self, q, k, v):
        s = torch.matmul(q, k.transpose(-2, -1)) * 0.25
        return torch.matmul(torch.softmax(s, dim=-1), v)

def test_chain_matches_eager():
    m, x = Chain().eval(), torch.randn(8, 16)
    c = compile_model(m, (x,))
    assert all(s["kind"] == "torch" for s in c.plan["steps"])
    torch.testing.assert_close(c(x), m(x))

def test_attention_matches_eager():
    m = Attn()
    q, k, v = (torch.randn(1, 2, 8, 16) for _ in range(3))
    c = compile_model(m, (q, k, v))
    torch.testing.assert_close(c(q, k, v), m(q, k, v))
    assert len(c.plan["steps"]) == 5
```
**Verify:** `pytest -q` → all pass.

---

## 9. API / Interface Contracts

**CLI — `fusionpass-compile <input.mlir> -o <outdir> [--pipeline none]`**
- Reads MLIR text (module containing `func.func @main`), verifies it, writes `<outdir>/plan.json`.
- Exit 0 success; 1 parse/verify/emit/IO failure (message on stderr); 2 unknown `--pipeline`.

**CLI — `fusionpass-opt`**: standard `mlir-opt` flags; dialects `func` and `fp` registered.

**Python**
- `trace_to_mlir(model, example_inputs) -> (str, list[Tensor])` — raises `UnsupportedOp(msg)`.
- `compile_model(model, example_inputs, *, fuse=False, workdir=None) -> CompiledModel` — raises `FileNotFoundError` (compiler missing), `RuntimeError` (compiler failed; message includes stderr + MLIR), `NotImplementedError` (`fuse=True`).
- `CompiledModel.__call__(*inputs) -> Tensor` — raises `ValueError` on input count/shape/dtype mismatch.
- `execute_plan(plan: dict, args: list[Tensor], kernels=None) -> Tensor | tuple[Tensor,...]`.

**plan.json**: as in Section 7B.

---

## 10. Error Handling & Edge Cases

| Situation | Handling |
|---|---|
| Op outside the supported set (e.g., `torch.tanh`, `nn.Linear` as a module) | `UnsupportedOp` with the offending target in the message. Note `symbolic_trace` keeps `nn.Module` leaves as `call_module`; these are rejected — write models functionally. |
| Broadcasting (`x + bias_vector`) | `UnsupportedOp("broadcasting is unsupported ...")` |
| Python-scalar add (`x + 1.0`) | `UnsupportedOp` (only mul/div by a constant → `fp.scale`) |
| Dynamic/unsupported dtype (bf16, int) | `UnsupportedOp("unsupported dtype ...")` |
| Tensor output tuple / multiple outputs | `UnsupportedOp("exactly one tensor output ...")` |
| Malformed MLIR / verifier failure | `fusionpass-compile` prints diagnostics and exits 1; `compile_model` surfaces it as `RuntimeError` |
| `plan.json` version mismatch | runtime raises `ValueError` |
| Wrong input shape/dtype at call time | runtime raises `ValueError` naming the argument |
| Compiler binary missing | `FileNotFoundError` with remediation text |
| Output dir not writable | exit 1 with message |

---

## 11. Testing & Verification

```bash
cmake --build build
export FUSIONPASS_COMPILE=$PWD/build/bin/fusionpass-compile
lit -v test/            # 2 lit tests pass
pytest -q               # frontend + e2e pass (CPU only, no GPU)
```
Manual smoke test:
```bash
python - <<'EOF'
import torch
from fusionpass import compile_model
class M(torch.nn.Module):
    def forward(self, x): return torch.relu(x * 3.0)
x = torch.randn(4, 4)
c = compile_model(M(), (x,))
print(c.workdir, len(c.plan["steps"]))   # 2 steps
torch.testing.assert_close(c(x), M()(x))
EOF
```
Expected: no exception; `cat <workdir>/plan.json` shows two `torch` steps (`scale`, `relu`).

---

## 12. Definition of Done

- [ ] LLVM/MLIR 18.1.x installed at `$LLVM_PREFIX`; `FileCheck` present.
- [ ] `cmake --build build` succeeds; `fusionpass-opt` and `fusionpass-compile` exist in `build/bin/`.
- [ ] `lit -v test/` passes (round-trip and both verifier-error cases).
- [ ] `fusionpass-compile` converts the Section 7 example module into a 5-step `plan.json`.
- [ ] `pytest -q` passes: chain and attention models match eager PyTorch (`assert_close`) through the MLIR round trip.
- [ ] Unsupported constructs (broadcast add) raise `UnsupportedOp`.
- [ ] `git commit` with a clean tree (no build artifacts committed).

---

## 13. Notes & Pitfalls for the Implementer

- **LLVM API drift:** the pin is 18.1.x. Pattern-driver and pass APIs differ in other releases (e.g., `applyPatternsAndFoldGreedily` is renamed in later LLVM). Do not upgrade LLVM.
- **Build flags:** if you see RTTI / `typeinfo` link errors, ensure `include(HandleLLVMOptions)` is present and you did not add `-frtti`.
- **ODS naming:** `def FP_AddOp` becomes C++ class `AddOp` (the `FP_` prefix is stripped). Accessors follow argument names: `getLhs()`, `getRhs()`, `getInput()`, `getFactor()`, `getAxis()`.
- **`F32Attr` accessor returns `llvm::APFloat`** — use `.convertToFloat()`.
- **Float literals in MLIR need a `.`** (`1.0e-1` ok, `1e-1` not) — hence `_f32()` in the frontend.
- `get_attr` parameters are captured **by reference** (via `.detach()` sharing storage); later in-place updates to the model's weights are visible to the compiled model.
- **Do NOT build here:** fused ops (`fp.fused_chain`, `fp.fused_attention`), any pass, Triton emission, `triton` plan steps (Phase 2); equivalence-gate harness, benchmarks, CLI demos, README/CI (Phase 3).
- Keep op names, plan field names and file paths exactly as written — later phases depend on them.

---

## 14. Handoff to Next Phase

**What now exists and works:** the `fp` dialect (8 ops + verifiers), `fusionpass-opt`, `fusionpass-compile` (`--pipeline none`), `PlanEmitter` (torch steps), Python `frontend.py` / `runtime.py` / `api.py`, lit and pytest suites. A PyTorch model round-trips through MLIR and matches eager execution on CPU.

**Next (Phase 2):** add the fused ops, the `fp-fuse-attention` and `fp-fuse-chains` passes, the Triton source emitter (FlashAttention-style online-softmax kernel + generated chain kernels), `triton` plan steps, and the runtime that launches them.
