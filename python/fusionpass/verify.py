from __future__ import annotations

from dataclasses import dataclass

import torch


class EquivalenceError(AssertionError):
    """Raised when a compiled model does not match the eager reference."""


@dataclass(frozen=True)
class VerifyReport:
    ok: bool
    max_abs_err: float
    max_rel_err: float
    atol: float
    rtol: float
    dtype: str
    trials: int
    reason: str


def default_tolerances(dtype: torch.dtype, has_attention: bool) -> tuple[float, float]:
    if dtype == torch.float16:
        return 1e-2, 1e-2
    return (5e-3, 5e-3) if has_attention else (1e-4, 1e-4)


def _compare(ref: torch.Tensor, out: torch.Tensor, atol: float, rtol: float):
    if ref.shape != out.shape or ref.dtype != out.dtype:
        return False, float("inf"), float("inf"), (
            f"shape/dtype mismatch: ref {ref.dtype}{tuple(ref.shape)} vs out {out.dtype}{tuple(out.shape)}"
        )
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


def _dt(d):
    return "f16" if d == torch.float16 else "f32"


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
    # Match the attention kernel's IEEE fp32 dots, not Ampere TF32, so the
    # 2x/4x softmax stress trials compare like-for-like against eager matmul.
    tf32_matmul = tf32_cudnn = None
    if torch.cuda.is_available():
        tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        tf32_cudnn = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.no_grad():
            for i, xs in enumerate(trials):
                ok, a, r, why = _compare(model(*xs), compiled(*xs), atol, rtol)
                worst_abs, worst_rel = max(worst_abs, a), max(worst_rel, r)
                if not ok:
                    return VerifyReport(
                        False, worst_abs, worst_rel, atol, rtol, _dt(dtype),
                        len(trials), f"trial {i}: {why}",
                    )
        return VerifyReport(True, worst_abs, worst_rel, atol, rtol, _dt(dtype), len(trials), "")
    finally:
        if tf32_matmul is not None:
            torch.backends.cuda.matmul.allow_tf32 = tf32_matmul
            torch.backends.cudnn.allow_tf32 = tf32_cudnn


def assert_equivalent(model, compiled, inputs, **kw) -> VerifyReport:
    rep = verify_equivalence(model, compiled, inputs, **kw)
    if not rep.ok:
        raise EquivalenceError(f"compiled model differs from eager reference: {rep.reason}")
    return rep
