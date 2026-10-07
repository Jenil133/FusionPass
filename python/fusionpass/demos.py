from __future__ import annotations

import argparse
import os
import sys

import torch
from torch import nn

__all__ = [
    "AttentionBlock",
    "RowChain",
    "make_attention",
    "make_chain",
    "run_demo",
]


class AttentionBlock(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        self.scale = head_dim ** -0.5     # float attribute -> traced as a constant -> fp.scale

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


def _summarize_plan(plan: dict) -> dict:
    try:
        from .traffic import summarize_plan
        return summarize_plan(plan)
    except ImportError:
        kinds = [s["kind"] for s in plan["steps"]]
        return {
            "steps": len(kinds),
            "triton_steps": kinds.count("triton"),
            "torch_steps": kinds.count("torch"),
        }


def _fuse_unsupported(exc: BaseException) -> bool:
    if isinstance(exc, NotImplementedError):
        return True
    msg = str(exc).lower()
    return "unknown --pipeline" in msg


def _compiler_unusable(exc: BaseException) -> bool:
    """True when the C++ binary is missing or not runnable on this OS.

    A Linux `fusionpass-compile` sitting in `build/bin/` makes `Path.exists()`
    true on Windows, but `subprocess.run` then raises WinError 2 or 193.
    """
    if isinstance(exc, FileNotFoundError):
        return True
    return isinstance(exc, OSError) and getattr(exc, "winerror", None) in (2, 193)


def _compile(model, xs, **kwargs):
    """Public-API compile; drops kwargs `compile_model` does not yet accept."""
    from .api import compile_model

    try:
        return compile_model(model, xs, **kwargs)
    except TypeError as e:
        if "unexpected keyword argument" not in str(e):
            raise
        import inspect
        params = inspect.signature(compile_model).parameters
        filtered = {k: v for k, v in kwargs.items() if k in params}
        if filtered == kwargs:
            raise
        return compile_model(model, xs, **filtered)


def _parse_demo_args(kind: str, argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog=f"examples/{kind}_demo.py")
    p.add_argument("--dtype", choices=["f16", "f32"], default="f16")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--rows", type=int, default=8192)
    p.add_argument("--cols", type=int, default=1024)
    return p.parse_args(argv)


def run_demo(kind: str, argv: list[str] | None = None) -> int:
    """Compile a sample model through `compile_model` and print the demo report.

    Tries ``fuse=True`` first; if fusion has not landed (NotImplementedError or
    unknown ``--pipeline fuse``), retries with ``fuse=False`` so the examples
    run on the unfused Phase 1 pipeline. Never prints a speedup unless
    ``compiled.verified`` is True.

    Exit codes match the CLI: 0 ok, 2 verification failed, 3 compiler missing.
    """
    if kind not in ("attention", "chain"):
        print(f"unknown demo kind {kind!r}; expected 'attention' or 'chain'", file=sys.stderr)
        return 2

    a = _parse_demo_args(kind, sys.argv[1:] if argv is None else argv)
    cuda = torch.cuda.is_available()
    dev = "cuda" if cuda else "cpu"
    if not cuda:
        os.environ.setdefault("TRITON_INTERPRET", "1")
        if a.dtype == "f16":
            a.dtype = "f32"
            if kind == "attention":
                a.seq = min(a.seq, 64)
                a.batch = min(a.batch, 1)
                a.heads = min(a.heads, 2)
            else:
                a.rows = min(a.rows, 64)
                a.cols = min(a.cols, 128)
            print("note       : no CUDA; using f32 and smaller shapes (fp16 interpreter is slow)")

    if kind == "attention":
        model, xs = make_attention(a.batch, a.heads, a.seq, a.head_dim, a.dtype, dev)
        print(
            f"model      : AttentionBlock  "
            f"B={a.batch} H={a.heads} S={a.seq} D={a.head_dim}  dtype={a.dtype}  device={dev}"
        )
    else:
        model, xs = make_chain(a.rows, a.cols, a.dtype, dev)
        print(f"model      : RowChain  rows={a.rows} cols={a.cols}  dtype={a.dtype}  device={dev}")

    try:
        from .verify import EquivalenceError
    except ImportError:
        EquivalenceError = None  # type: ignore[misc, assignment]

    try:
        unfused = _compile(model, xs, fuse=False, verify=False)
    except (FileNotFoundError, OSError) as e:
        if not _compiler_unusable(e):
            raise
        print(e, file=sys.stderr)
        print(
            "hint: fusionpass-compile must be executable on this OS; "
            "build the C++ project or set FUSIONPASS_COMPILE.",
            file=sys.stderr,
        )
        from .frontend import trace_to_mlir
        print("traced MLIR (compiler not required for this dump):", file=sys.stderr)
        print(trace_to_mlir(model, xs)[0], file=sys.stderr)
        return 3

    fused = None
    fusion_note = ""
    try:
        fused = _compile(model, xs, fuse=True, verify=True, strict=True)
    except (FileNotFoundError, OSError) as e:
        if not _compiler_unusable(e):
            raise
        print(e, file=sys.stderr)
        return 3
    except Exception as e:
        if EquivalenceError is not None and isinstance(e, EquivalenceError):
            print(f"VERIFICATION FAILED: {e}", file=sys.stderr)
            return 2
        if _fuse_unsupported(e):
            fusion_note = "fusion not available yet; compiled with fuse=False"
            print(f"fusion     : skipped ({e})")
        else:
            raise

    compiled = fused if fused is not None else unfused
    did_fuse = fused is not None

    r = getattr(compiled, "report", None)
    if r is not None and r.ok:
        print(
            f"verified   : OK  (max_abs_err={r.max_abs_err:.2e}, max_rel_err={r.max_rel_err:.2e}, "
            f"atol={r.atol}, rtol={r.rtol}, trials={r.trials})"
        )
    elif r is not None:
        print(f"VERIFICATION FAILED: {r.reason}", file=sys.stderr)
        return 2
    else:
        with torch.no_grad():
            ref, out = model(*xs), compiled(*xs)
        dtype = next(t.dtype for t in xs if torch.is_tensor(t))
        atol = rtol = 1e-2 if dtype == torch.float16 else 1e-4
        torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)
        print(f"verified   : OK  (eager match, atol={atol}, rtol={rtol}; verify.py report not attached)")

    u_sum = _summarize_plan(unfused.plan)
    if did_fuse:
        print(f"plan       : unfused {u_sum}  ->  fused {_summarize_plan(fused.plan)}")
    elif fusion_note:
        print(f"plan       : unfused {u_sum}  ({fusion_note})")
    else:
        print(f"plan       : unfused {u_sum}")

    if not did_fuse:
        print("benchmark  : skipped (fusion not available; nothing to time against eager)")
        return 0
    if not cuda:
        print("benchmark  : skipped (no CUDA; interpreter mode is correctness-only)")
        return 0
    if not getattr(compiled, "verified", False):
        print("benchmark  : skipped (compiled.verified is not True; no unverified speedup)")
        return 0

    try:
        from .bench import UnverifiedSpeedupError, benchmark
    except ImportError:
        print("benchmark  : skipped (bench.py not yet available)")
        return 0
    try:
        b = benchmark(model, compiled, xs, unfused=unfused)
    except UnverifiedSpeedupError as e:
        print(e, file=sys.stderr)
        return 2
    print(
        f"eager      : {b.eager_ms:.3f} ms   fused: {b.fused_ms:.3f} ms   "
        f"speedup: {b.speedup:.2f}x  (median of {b.iters})"
    )
    if b.est_bytes_unfused is not None and b.est_bytes_fused is not None:
        print(
            f"est. traffic: unfused {b.est_bytes_unfused / 1e6:.1f} MB  ->  "
            f"fused {b.est_bytes_fused / 1e6:.1f} MB  (rough model, not measured)"
        )
    return 0
