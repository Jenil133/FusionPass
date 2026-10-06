from __future__ import annotations

from dataclasses import dataclass

import torch

from .traffic import estimate_bytes
from .verify import VerifyReport


class UnverifiedSpeedupError(RuntimeError):
    """Raised when a speedup is requested for a model that has not passed equivalence verification."""


@dataclass(frozen=True)
class BenchResult:
    eager_ms: float
    fused_ms: float
    speedup: float
    iters: int
    eager_launches: int
    fused_launches: int
    est_bytes_unfused: int | None
    est_bytes_fused: int | None
    report: VerifyReport


def _median_ms_cuda(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):          # also triggers Triton JIT compilation
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for s, e in zip(starts, ends):
        s.record()
        fn()
        e.record()
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
