"""Phase 3 verification gate and benchmark tests.

Compile-dependent cases skip if fusionpass-compile is missing or not
runnable on this host (e.g. a Docker Linux ELF on Windows).
"""

from __future__ import annotations

import copy
import os
import subprocess

import pytest
import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

from fusionpass import compile_model
from fusionpass.bench import UnverifiedSpeedupError, benchmark
from fusionpass.demos import make_attention, make_chain
from fusionpass.traffic import estimate_bytes
from fusionpass.verify import EquivalenceError, assert_equivalent, default_tolerances, verify_equivalence

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _compiler_runnable(path: str) -> bool:
    try:
        subprocess.run([path, "--help"], capture_output=True, text=True, timeout=15)
        return True
    except (OSError, subprocess.TimeoutExpired):
        return False


@pytest.fixture
def need_compiler(fusionpass_compile):
    if not _compiler_runnable(fusionpass_compile):
        pytest.skip("fusionpass-compile is not runnable on this host")
    return fusionpass_compile


def _compile(*args, **kwargs):
    try:
        return compile_model(*args, **kwargs)
    except (FileNotFoundError, OSError) as e:
        pytest.skip(f"fusionpass-compile not runnable: {e}")


def test_default_tolerances():
    assert default_tolerances(torch.float16, False) == (1e-2, 1e-2)
    assert default_tolerances(torch.float16, True) == (1e-2, 1e-2)
    assert default_tolerances(torch.float32, True) == (5e-3, 5e-3)
    assert default_tolerances(torch.float32, False) == (1e-4, 1e-4)


def test_verify_equivalence_matches_and_mismatches_without_compiler():
    class M(torch.nn.Module):
        def forward(self, x):
            return torch.relu(x)

    m = M()
    x = torch.randn(4, 4)
    plan = {
        "version": 1,
        "inputs": [{"name": "arg0", "shape": [4, 4], "dtype": "f32"}],
        "steps": [{"kind": "torch", "op": "relu", "operands": ["arg0"], "result": "v0",
                   "shape": [4, 4], "dtype": "f32", "attrs": {}}],
        "outputs": ["v0"],
    }

    class Dummy:
        def __init__(self, fn):
            self.plan = plan
            self._fn = fn

        def __call__(self, *xs):
            return self._fn(*xs)

    good = Dummy(lambda t: torch.relu(t))
    rep = verify_equivalence(m, good, (x,), extra_trials=2)
    assert rep.ok and rep.trials == 3 and rep.reason == ""
    assert_equivalent(m, good, (x,), extra_trials=0)

    bad = Dummy(lambda t: torch.relu(t) + 1.0)
    rep = verify_equivalence(m, bad, (x,), extra_trials=0)
    assert rep.ok is False and "exceed" in rep.reason
    with pytest.raises(EquivalenceError):
        assert_equivalent(m, bad, (x,), extra_trials=0)


def test_benchmark_refuses_unverified():
    """Gate is checked before the CUDA requirement; no compiler needed."""
    class Dummy:
        verified = False
        report = None
        plan = {"steps": []}

    m, xs = make_chain(8, 32, "f32", DEV)
    with pytest.raises(UnverifiedSpeedupError):
        benchmark(m, Dummy(), xs)


def test_verified_flag_and_report(need_compiler, require_triton):
    m, xs = make_attention(1, 2, 20, 16, "f32", DEV)
    c = _compile(m, xs, fuse=True)
    assert c.verified and c.report.ok and c.report.trials == 3


def test_benchmark_refuses_unverified_after_compile(need_compiler, require_triton):
    m, xs = make_chain(8, 32, "f32", DEV)
    c = _compile(m, xs, fuse=True, verify=False)
    assert c.verified is False
    with pytest.raises(UnverifiedSpeedupError):
        benchmark(m, c, xs)


def test_gate_catches_miscompile(need_compiler, require_triton):
    m, xs = make_attention(1, 2, 20, 16, "f32", DEV)
    c = _compile(m, xs, fuse=True)
    if not any(s.get("launch") == "attention" for s in c.plan["steps"]):
        pytest.skip("attention was not fused")
    bad = copy.deepcopy(c.plan)
    for s in bad["steps"]:
        if s.get("launch") == "attention":
            s["attrs"]["scale"] *= 1.5
    c.plan = bad
    rep = verify_equivalence(m, c, xs)
    assert rep.ok is False and "exceed" in rep.reason


def test_traffic_estimate_fused_smaller_for_long_sequences(need_compiler, require_triton):
    m, xs = make_attention(1, 2, 1024, 64, "f16", "cpu")
    u = _compile(m, xs, fuse=False, verify=False)
    f = _compile(m, xs, fuse=True, verify=False)
    assert estimate_bytes(f.plan) < estimate_bytes(u.plan)


def test_verify_false_skips_gate_on_unfused(need_compiler):
    m, xs = make_chain(8, 32, "f32", DEV)
    c = _compile(m, xs, fuse=False, verify=False)
    assert c.verified is False
    assert c.report is None


def test_unfused_verify_sets_flag(need_compiler):
    m, xs = make_chain(8, 32, "f32", DEV)
    c = _compile(m, xs, fuse=False, verify=True)
    assert c.verified and c.report.ok and c.report.trials == 3


def test_assert_equivalent_raises_on_corrupt_plan(need_compiler, require_triton):
    m, xs = make_attention(1, 2, 20, 16, "f32", DEV)
    c = _compile(m, xs, fuse=True, verify=False)
    if not any(s.get("launch") == "attention" for s in c.plan["steps"]):
        pytest.skip("attention was not fused")
    for s in c.plan["steps"]:
        if s.get("launch") == "attention":
            s["attrs"]["scale"] *= 1.5
    with pytest.raises(EquivalenceError):
        assert_equivalent(m, c, xs)
