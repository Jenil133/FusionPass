from __future__ import annotations

import argparse
import inspect
import os
import sys

import torch

from .api import _find_compiler, compile_model


def _compile(model, xs, **kwargs):
    """Call compile_model, forwarding fuse/verify/strict/atol/rtol even while those kwargs land."""
    try:
        return compile_model(model, xs, **kwargs)
    except TypeError as e:
        if "unexpected keyword argument" not in str(e):
            raise
        params = inspect.signature(compile_model).parameters
        filtered = {k: v for k, v in kwargs.items() if k in params}
        if filtered == kwargs:
            raise
        return compile_model(model, xs, **filtered)


def _cmd_check(_) -> int:
    try:
        print("compiler :", _find_compiler())
    except FileNotFoundError as e:
        print("compiler : MISSING —", e)
        return 3
    try:
        import triton
        triton_ver = triton.__version__
    except ImportError:
        triton_ver = "MISSING"
    print("torch    :", torch.__version__, "| triton:", triton_ver)
    print("cuda     :", torch.cuda.is_available())
    return 0


def _cmd_demo(a) -> int:
    try:
        from .bench import UnverifiedSpeedupError, benchmark
        from .demos import make_attention, make_chain
        from .traffic import summarize_plan
        from .verify import EquivalenceError
    except ImportError as e:
        print(e, file=sys.stderr)
        return 1

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev != "cuda":
        os.environ.setdefault("TRITON_INTERPRET", "1")
    if a.kind == "attention":
        model, xs = make_attention(a.batch, a.heads, a.seq, a.head_dim, a.dtype, dev)
    else:
        model, xs = make_chain(a.rows, a.cols, a.dtype, dev)
    try:
        unfused = _compile(model, xs, fuse=False, verify=False)
        fused = _compile(
            model, xs,
            fuse=a.fuse, verify=a.verify, strict=a.strict,
            atol=a.atol, rtol=a.rtol,
        )
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 3
    except EquivalenceError as e:
        print(f"VERIFICATION FAILED: {e}", file=sys.stderr)
        return 2
    r = getattr(fused, "report", None)
    if r is not None and r.ok:
        print(f"verified   : OK  (max_abs_err={r.max_abs_err:.2e}, max_rel_err={r.max_rel_err:.2e}, "
              f"atol={r.atol}, rtol={r.rtol}, trials={r.trials})")
    elif r is not None:
        print(f"VERIFICATION FAILED: {r.reason}", file=sys.stderr)
        if a.strict:
            return 2
        print(f"verified   : FAIL (max_abs_err={r.max_abs_err:.2e}, max_rel_err={r.max_rel_err:.2e}, "
              f"atol={r.atol}, rtol={r.rtol}, trials={r.trials})")
    elif a.verify:
        print("verified   : not reported (compile_model did not attach a VerifyReport)")
    else:
        print("verified   : skipped (--no-verify)")
    print(f"plan       : unfused {summarize_plan(unfused.plan)}  ->  fused {summarize_plan(fused.plan)}")
    want_bench = a.bench if a.bench is not None else (dev == "cuda")
    if not want_bench:
        if dev != "cuda":
            print("benchmark  : skipped (no CUDA; interpreter mode is correctness-only)")
        else:
            print("benchmark  : skipped (--no-bench)")
        return 0
    try:
        b = benchmark(model, fused, xs, unfused=unfused, warmup=a.warmup, iters=a.iters)
    except UnverifiedSpeedupError as e:
        print(e, file=sys.stderr)
        return 2
    print(f"eager      : {b.eager_ms:.3f} ms   fused: {b.fused_ms:.3f} ms   speedup: {b.speedup:.2f}x  (median of {b.iters})")
    print(f"est. traffic: unfused {b.est_bytes_unfused/1e6:.1f} MB  ->  fused {b.est_bytes_fused/1e6:.1f} MB  (rough model, not measured)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="fusionpass")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="print compiler / torch / triton / CUDA status").set_defaults(fn=_cmd_check)
    d = sub.add_parser("demo", help="compile, verify, and optionally benchmark an attention or chain model")
    d.set_defaults(fn=_cmd_demo)
    d.add_argument("kind", choices=["attention", "chain"])
    d.add_argument("--dtype", choices=["f16", "f32"], default="f16")
    d.add_argument("--batch", type=int, default=4)
    d.add_argument("--heads", type=int, default=8)
    d.add_argument("--seq", type=int, default=1024)
    d.add_argument("--head-dim", type=int, default=64)
    d.add_argument("--rows", type=int, default=8192)
    d.add_argument("--cols", type=int, default=1024)
    d.add_argument("--fuse", action=argparse.BooleanOptionalAction, default=True,
                   help="run the fused compile (default: True; unfused compile always runs for the plan summary)")
    d.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True,
                   help="run equivalence verification on the fused compile (default: True)")
    d.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True,
                   help="treat a verification mismatch as a hard failure (default: True)")
    d.add_argument("--bench", action=argparse.BooleanOptionalAction, default=None,
                   help="run the CUDA-event benchmark (default: on when CUDA is available)")
    d.add_argument("--atol", type=float, default=None, help="override absolute tolerance")
    d.add_argument("--rtol", type=float, default=None, help="override relative tolerance")
    d.add_argument("--warmup", type=int, default=25, help="benchmark warmup iterations")
    d.add_argument("--iters", type=int, default=100, help="benchmark timed iterations")
    a = p.parse_args(argv)
    if a.cmd == "demo" and not torch.cuda.is_available() and a.dtype == "f16":
        a.dtype = "f32"                      # fp16 on the CPU interpreter is slow/unsupported in places
        if a.kind == "attention":
            a.seq = min(a.seq, 64)
            a.batch = min(a.batch, 1)
            a.heads = min(a.heads, 2)
        else:
            a.rows = min(a.rows, 64)
            a.cols = min(a.cols, 128)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
