# fusionpass design

This document describes the compiler internals: the `fp` dialect, the two fusion passes, Triton kernel generation, the execution plan, and the verify-before-bench gate. User-facing setup, CLI, and (when measured) results live in [README.md](../README.md).

**No speedup numbers appear here.** Timing belongs in the README results table, populated only from `python -m fusionpass demo ...` on a given GPU. This host has not recorded those numbers.

## Architecture

```
 PyTorch nn.Module
      │  torch.fx symbolic_trace + ShapeProp          python/fusionpass/frontend.py
      ▼
 model.mlir   (func.func @main + fp dialect ops)
      │  fusionpass-compile   (C++ / MLIR 18.1.8)
      │     parse → verify → fp-fuse-attention → fp-fuse-chains → emit
      ▼
 plan.json  +  kernels.py  +  optimized.mlir
      │  python/fusionpass/runtime.py  (+ validate_plan)
      ▼
 PyTorch ops (unfused)  |  Triton kernels (fused)
      │
 verify.py  ──►  bench.py (only if verified)  +  traffic.py
```

A static-shape `nn.Module` (f16/f32, inference, one output, no broadcasting) is traced to textual MLIR. `fusionpass-compile` (`--pipeline fuse` or `none`) writes an ordered plan and generated Triton source. The Python runtime executes that plan. Equivalence against eager PyTorch is proven **before** any speedup is reported.

Attention fusion runs **before** chain fusion so the scale/softmax inside `softmax(scale · Q Kᵀ) V` are absorbed into `fp.fused_attention` rather than a generic chain.

Naming: dialect `fp` / namespace `fusionpass::fp`; tools `fusionpass-opt`, `fusionpass-compile`; Python package `fusionpass`; kernels `fp_chain_<N>`, `fp_attention`; plan names `arg0..` and `v0, v1, ...`.

---

## 1. Dialect and invariants

The `fp` dialect (`include/fusionpass/Dialect/FP/FPOps.td`, verifiers in `lib/Dialect/FP/FPOps.cpp`) is a small tensor-level IR. Element type is f16 or f32; every tensor has a **static** ranked shape (`FP_Tensor`). Dynamic dims, bf16, integer types, and broadcasting are out of scope.

### Primitive ops

| Op | Form | Invariants |
|---|---|---|
| `fp.add`, `fp.mul` | binary, same type | `Pure`, `SameOperandsAndResultType` |
| `fp.relu`, `fp.exp` | unary, same type | `Pure` |
| `fp.scale` | unary + `factor:f32` | `Pure`; factor is a constant attribute (frontend folds `x * c`) |
| `fp.softmax` | unary + `axis:i64` | axis ∈ `[0, rank)` |
| `fp.matmul` | `%r = fp.matmul %a, %b : (T, U) -> V` | equal rank ≥ 2, matching element type and batch dims, contraction `lhs[-1] == rhs[-2]`, result shape is the matmul result |
| `fp.transpose` | last two dims swapped | rank ≥ 2, element type preserved |

All primitive ops are `Pure` (no side effects). Elementwise ops require identical operand shapes — there is no broadcast.

### Region / fused ops

**`fp.yield`** — terminator of a `fused_chain` body (`HasParent<"FusedChainOp">`).

**`fp.fused_chain`** — `Pure`, `IsolatedFromAbove`, single block, variadic inputs. Prints in MLIR generic form (no custom assembly; avoids a hand-written region parser).

- All inputs, block arguments, and the result share **one** tensor type `T`.
- Body ops ⊂ `{add, mul, scale, relu, exp, softmax}` plus the yield.
- At most one `softmax`, and it must reduce the **last** axis.
- Block argument count equals input count.

**`fp.fused_attention`** — `Pure`, `AllTypesMatch` on `q, k, v, result`.

- Rank ≥ 3: leading dims are batch, then `[S, D]`.
- Last dim `D ∈ {16, 32, 64, 128}` (`tl.dot` wants a power-of-two K ≥ 16).
- Semantics: `softmax(scale · Q·Kᵀ, axis=-1) · V`.

An unfusable graph is still legal IR: it stays as primitive ops and becomes `torch` plan steps.

---

## 2. The two fusion passes

Both passes are greedy `OpRewritePattern`s on `func.func` (`applyPatternsAndFoldGreedily`). They never invent ops that the verifiers would reject; a failed match leaves the graph unchanged.

### `--fp-fuse-attention` (`lib/Transforms/FuseAttention.cpp`)

Root is the **second** matmul (`P @ V`). The pattern walks backwards and matches:

```
matmul(softmax(scale*…(matmul(Q, transpose(K)))), V)
```

Zero or more `fp.scale` ops between the first matmul and softmax are folded multiplicatively into the fused `scale` attribute.

Match requires:

1. Softmax on the last axis.
2. **Single-use** on every intermediate: probabilities `P`, each scale result, the QK matmul result, and `transpose(K)`.
3. Self-attention shape contract: `Q`, `K`, `V`, and the output have identical types; rank ≥ 3; `D ∈ {16, 32, 64, 128}`.

On success the second matmul is replaced by `fp.fused_attention`; the now-dead softmax / scale / first matmul / transpose are DCE'd by the greedy driver.

**Why single-use is required.** Fusion is only a win if the intermediates can disappear. If `P` (or QK scores, or `Kᵀ`) has another user, the fused kernel still would not produce that tensor, so the compiler would have to materialize it anyway — extra work for no saved traffic. Multi-use probabilities, cross-attention shapes (`S_q ≠ S_kv`), and unsupported head dims therefore **fall back** to unfused `torch` steps. That fallback is still verified correct.

### `--fp-fuse-chains` (`lib/Transforms/FuseChains.cpp`)

Chainable ops: `add, mul, scale, relu, exp, softmax, fused_chain`. A plain softmax is fusible only if it reduces the last axis **and** `last_dim ≤ kMaxRowSoftmaxCols` (8192) — the row kernel keeps one whole row in registers.

A consumer `C` merges with a producer `P` when:

- both are chainable and fusible;
- `C` is not already inside a `fused_chain` (the driver walks `IsolatedFromAbove` regions; nesting would fail the region verifier);
- the producer result has **one use**;
- `P` and `C` are in the same block;
- `softmaxCount(P) + softmaxCount(C) ≤ 1` (the row kernel supports one reduction).

**Why single-use is required (chains).** A producer with two users must remain a named value in the plan so both users can read it. Fusing it into one consumer would either duplicate work or force a materialization of the fused result for the other user. Same rationale as attention.

---

## 3. Chain-merge algorithm

Each successful rewrite of consumer `C` with producer `P`:

1. **Externals, deduped.** Collect `P`'s operands plus `C`'s operands except the edge `P → C`. Insertion into the list skips values already present (`llvm::is_contained`). Duplicate uses of the same SSA value (e.g. `relu(x * x)`) therefore become **one** `fused_chain` input, and the plan lists a single operand.

2. **New `fused_chain`.** Created at `C`'s insertion point, with one block argument per external.

3. **`IRMapping` inlining.** Map each external to the corresponding block argument. `inlineOp`:
   - for a plain op: `clone` into the new body;
   - for an existing `fused_chain`: map its block args to the already-mapped outer operands, clone the inner body (without terminator), and return the mapped yield value.

   The producer is inlined first; its result is mapped onto `C`'s corresponding operand; then the consumer is inlined. A `fp.yield` of the consumer result closes the body.

4. **Replace.** `C` is replaced by the fused op. `P` becomes dead and is erased by the greedy driver.

### Termination

Let *N* be the number of chainable operations at function scope (not inside an existing `fused_chain` body). Each successful rewrite strictly decreases *N* by one: `P` and `C` become a single `fused_chain`, and the driver does not rewrite inside `IsolatedFromAbove` regions. *N* is finite and non-negative, so the greedy iteration terminates. There is no rewrite that splits a fused op back into primitives, so there is no cycle.

Work is linear in op count: each op is a producer in at most one merge (after that it is inside an isolated region or dead).

---

## 4. Generated Triton kernels

`lib/Plan/TritonEmitter.cpp` writes `kernels.py`. Compute inside kernels is **float32**; the store casts back to the output element type. Loads use a mask; out-of-range lanes contribute `0.0` (or `-inf` for softmax padding).

### Flat elementwise (`launch: "flat"`)

Used when a `fused_chain` has **no** softmax. One program processes `BLOCK = 1024` linear elements:

```
pid = tl.program_id(0)
offs = pid * BLOCK + tl.arange(0, BLOCK)
mask = offs < n_elements
# load each input at offs, compute the inlined body, store
```

Grid: `(cdiv(n_elements, BLOCK),)`. Body ops lower as `+`, `*`, `tl.maximum(..., 0.0)`, `tl.exp`.

### Row softmax (`launch: "rows"`)

Used when the chain contains a softmax. One program **per row**; the whole row lives in registers with `BLOCK_N = next_pow2(n_cols)`:

```
row = tl.program_id(0)
cols = tl.arange(0, BLOCK_N)
mask = cols < n_cols
offs = row * n_cols + cols
```

Stable softmax over the row:

```
m = tl.where(mask, x, -inf)
e = exp(m - max(m))
y = e / sum(e)
```

Padded lanes are forced to `-inf` so they contribute 0 to the sum. `num_warps` is 4 if `block_n ≤ 2048`, else 8.

### Online-softmax attention (`launch: "attention"`, kernel `fp_attention`)

FlashAttention-style tiling: one program per `(BLOCK_M` query rows, batch×head). Tensors are contiguous `[BH, S, D]`. K and V are streamed in `BLOCK_N` tiles. The S×S score matrix is **never written to global memory**.

Fixed launch attrs (not autotuned): `block_m=64`, `block_n=64`, `num_warps=4`, `num_stages=2`. Grid: `(cdiv(S, BLOCK_M), batch*heads)`.

**Online softmax recurrence** (Dao et al. / FlashAttention). For each query tile, in fp32:

Initialize:

- \(m_i \leftarrow -\infty\) (running row-max)
- \(l_i \leftarrow 0\) (running row-sum of unnormalized exp)
- \(\mathrm{acc} \leftarrow 0\) (running weighted \(V\), shape `[BLOCK_M, D]`)

For each K/V tile \(n = 0, BLOCK_N, 2\,BLOCK_N, \ldots\):

\[
\begin{aligned}
s &\leftarrow (Q_{\text{tile}} \cdot K_n^\top) \cdot \mathrm{scale} \\
m_{\mathrm{new}} &\leftarrow \max(m_i,\; \mathrm{rowmax}(s)) \\
\alpha &\leftarrow \exp(m_i - m_{\mathrm{new}}) \\
p &\leftarrow \exp(s - m_{\mathrm{new}}) \\
l_i &\leftarrow l_i \cdot \alpha + \mathrm{rowsum}(p) \\
\mathrm{acc} &\leftarrow \mathrm{acc} \cdot \alpha + p \cdot V_n \\
m_i &\leftarrow m_{\mathrm{new}}
\end{aligned}
\]

After the loop: \(O \leftarrow \mathrm{acc} / l_i\).

`α` rescales previous partial numerators and the running sum onto the new max so the algebra equals a full softmax. Sequence lengths not a multiple of 64 (including `S = 1`) are handled by masked loads/stores; out-of-range K columns are set to `-inf` in `s`.

On GPU, `tl.dot` uses **TF32** by default for fp32 — that is why the verify tolerance for fp32+attention is looser than for elementwise chains.

---

## 5. Plan format

`lib/Plan/PlanEmitter.h` / `PlanEmitter.cpp` walk `func.func @main` in program order. Schema version **1**:

```json
{
  "version": 1,
  "inputs":  [ {"name": "arg0", "shape": [2, 4, 32, 64], "dtype": "f16"} ],
  "steps":   [ /* torch or triton */ ],
  "outputs": ["v0"]
}
```

- Inputs are `arg0, arg1, ...` (function arguments, including captured parameters/buffers from the frontend).
- Each op with one result is a step named `v0, v1, ...` in program order.
- Exactly one graph output is the supported subset; the emitter will list whatever `@main` returns.

**`torch` step** — unfused primitive:

```json
{"kind": "torch", "op": "matmul", "operands": ["arg0", "v0"], "result": "v1",
 "shape": [...], "dtype": "f16", "attrs": {}}
```

`op ∈ {add, mul, scale, relu, exp, softmax, matmul, transpose}`. `scale` carries `factor`; `softmax` carries `axis`.

**`triton` step** — fused region:

| `launch` | `kernel` | attrs |
|---|---|---|
| `flat` | `fp_chain_<N>` | `block=1024` |
| `rows` | `fp_chain_<N>` | `block_n=next_pow2(n_cols)`, `num_warps ∈ {4,8}` |
| `attention` | `fp_attention` | `scale`, `block_m=64`, `block_n=64`, `num_warps=4`, `num_stages=2` |

Attention steps have exactly three operands (Q, K, V).

`validate_plan` (`python/fusionpass/validate.py`, called from `runtime.execute_plan` and `CompiledModel` construction) rejects unsupported version, non-list fields, unknown dtypes/ops/launches, use-before-def, duplicate result names, and undefined outputs — **before** any kernel runs. After construction, `CompiledModel.__call__` passes `validate=False` so the structural check is not repeated every invocation.

---

## 6. Verify and bench

### Equivalence gate (`python/fusionpass/verify.py`)

`compile_model(..., verify=True, strict=True)` (the default) runs `verify_equivalence` against eager PyTorch on the example inputs plus two extra randomized trials (scales 2× and 4×, fixed CPU seed, then moved to device) that stress softmax stability.

`default_tolerances(dtype, has_attention)`:

| dtype | fused attention? | atol / rtol | why |
|---|---|---|---|
| f16 | any | 1e-2 / 1e-2 | eager rounds every op to fp16; kernels accumulate in fp32 |
| f32 | yes | 5e-3 / 5e-3 | `tl.dot` uses TF32 by default on GPU |
| f32 | no | 1e-4 / 1e-4 | only reordering / fused-rounding differences |

Finite elements use `err ≤ atol + rtol · |ref|`. NaN/Inf **patterns** (and Inf sign) must match the reference; allclose on non-finites is meaningless. A mismatch with `strict=True` raises `EquivalenceError`. `fuse` still defaults to `False` so Phase 1 unfused tests stay valid.

### Benchmark (`python/fusionpass/bench.py`)

`benchmark()` **refuses** to time or report a speedup unless `compiled.verified is True` and `compiled.report.ok` (`UnverifiedSpeedupError`). That check runs **before** the CUDA requirement. Timing is median of CUDA events over `iters` (default 100) after `warmup` (default 25, which also covers Triton JIT). The Triton CPU interpreter is correctness-only and is never timed.

Speedup is `eager_ms / fused_ms` versus **unfused eager PyTorch in the same dtype**. It is not a comparison against `torch.nn.functional.scaled_dot_product_attention`.

### Traffic estimate (`python/fusionpass/traffic.py`)

`estimate_bytes` is a **rough model**, not a measurement: each step reads every operand once and writes its result once; fused attention re-reads K and V once per query tile (`ceil(S / block_m)`). It ignores caches, PyTorch internal fusion, in-place ops, and L2 reuse. Always label it as an estimate.

---

## 7. Scaling vs. breaking points

### Scales

- **Passes are linear in op count.** Greedy rewrites; each successful chain merge removes one function-scope op; attention replaces a whole subgraph with one op. No quadratic matching over the function.
- **Attention memory is O(S·D), not O(S²).** Online softmax never materializes the score matrix. Unfused eager writes QK scores of size `S × S` per head.
- **Kernels are O(1) registers per tile.** Flat: `BLOCK` lanes. Rows: one row of `BLOCK_N`. Attention: `BLOCK_M × D` accumulator plus one `BLOCK_N` K/V tile. Tile sizes do not grow with `S`.

### Breaks / limits

- **Per-shape recompilation.** Shapes are static and baked into the plan and into Triton `constexpr`s. A different shape needs a new `compile_model`.
- **Row kernel register cap.** Last dim ≤ 8192 (`kMaxRowSoftmaxCols`); larger softmax stays unfused.
- **Fixed attention tiles.** `BLOCK_M=64`, `BLOCK_N=64`, `num_warps=4`, `num_stages=2`. No autotune.
- **No masks.** Causal / arbitrary attention masks are out of scope; mismatched Q/K/V shapes (cross-attention) are not fused.
- **TF32 on GPU fp32 attention.** `tl.dot` default; encoded in the verify table, not a bug.
- **Python per-step launch overhead** for unfused `torch` steps. Fusion helps because it removes launches *and* global-memory round trips; a graph that does not fuse still pays one Python/PyTorch launch per op.
- **One output per graph.** Multi-result modules are unsupported.
- **Baseline is unfused eager.** A fused library kernel such as `torch.nn.functional.scaled_dot_product_attention` is a stronger baseline and is **not** compared here.

Also out of scope (documented, not implemented): broadcasting, dynamic shapes, backward, bf16, native Windows C++ builds. Sequential lengths around the 64-tile boundary (`S ∈ {1, 7, 64, 65, 130}`) are tested with masked loads; unfusable structures fall back and still pass verification.
