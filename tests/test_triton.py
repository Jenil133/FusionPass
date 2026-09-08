"""Phase 2 Triton runtime and kernel-contract tests.

Kernel source is generated with the Python twin of TritonEmitter
(``fusionpass.kernels.KernelModuleWriter``). Execution tests skip if triton
is missing or the interpreter rejects a construct.
"""

from __future__ import annotations

import os

import pytest
import torch

from fusionpass.kernels import (
    ATTENTION_KERNEL_SRC,
    KERNEL_ATTENTION,
    KernelModuleWriter,
    attention_launch_attrs,
    chain_launch_attrs,
    next_pow2,
)
from fusionpass.runtime import _check_device, execute_plan, load_kernels

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _maybe_skip_kernel_error(exc: BaseException) -> None:
    if isinstance(exc, AssertionError):
        raise exc
    msg = str(exc).lower()
    needles = (
        "not supported in interpreter",
        "interpreter",
        "tl.dot",
        "nvcc",
        "cuda_home",
        "no kernel image",
        "cannot find compiler",
        "libcuda",
        "triton is not installed",
        "no module named 'triton'",
    )
    if any(n in msg for n in needles):
        pytest.skip(f"Triton/interpreter cannot run this kernel here: {exc}")
    raise exc


def _write_chain(tmp_path, n_args, body, *, rows=None):
    writer = KernelModuleWriter()
    name = writer.add_chain_kernel(n_args, body, rows=rows)
    path = tmp_path / "kernels.py"
    path.write_text(writer.write(), encoding="utf-8")
    try:
        return name, load_kernels(path)
    except Exception as exc:
        _maybe_skip_kernel_error(exc)


def test_load_kernels_imports_plain_module(tmp_path):
    path = tmp_path / "kernels.py"
    path.write_text("VALUE = 42\n", encoding="utf-8")
    mod = load_kernels(path)
    assert mod.VALUE == 42


def test_kernel_module_writer_emits_fp_chain_and_attention():
    writer = KernelModuleWriter()
    name = writer.add_chain_kernel(
        2,
        [
            {"op": "add", "operands": ["a0", "a1"]},
            {"op": "scale", "operands": ["t0"], "factor": 2.0},
            {"op": "relu", "operands": ["t1"]},
            {"op": "exp", "operands": ["t2"]},
        ],
    )
    assert name == "fp_chain_0"
    assert writer.add_attention_kernel() == KERNEL_ATTENTION
    src = writer.write()
    assert "def fp_chain_0" in src
    assert "def fp_attention" in src
    assert "import triton" in src
    assert "n_elements, BLOCK: tl.constexpr" in src
    assert "tl.maximum" in src


def test_rows_kernel_uses_block_n_signature():
    writer = KernelModuleWriter()
    writer.add_chain_kernel(
        1,
        [
            {"op": "scale", "operands": ["a0"], "factor": 0.5},
            {"op": "softmax", "operands": ["t0"]},
            {"op": "relu", "operands": ["t1"]},
        ],
    )
    src = writer.write()
    assert "n_cols, BLOCK_N: tl.constexpr" in src
    assert "float(\"-inf\")" in src
    assert "def fp_attention" not in src


def test_launch_attr_contract():
    assert chain_launch_attrs([8, 16], rows=False) == {"block": 1024}
    assert chain_launch_attrs([6, 100], rows=True) == {"block_n": 128, "num_warps": 4}
    assert chain_launch_attrs([4, 3000], rows=True)["num_warps"] == 8
    assert next_pow2(100) == 128
    attrs = attention_launch_attrs(0.25)
    assert attrs["scale"] == pytest.approx(0.25)
    assert attrs["block_m"] == 64 and attrs["block_n"] == 64
    assert "def fp_attention" in ATTENTION_KERNEL_SRC


def test_cpu_without_interpret_raises(monkeypatch):
    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    with pytest.raises(RuntimeError, match="TRITON_INTERPRET"):
        _check_device(torch.zeros(2, 2))


def test_unknown_launch_kind_raises(tmp_path, require_triton):
    _, kernels = _write_chain(
        tmp_path,
        1,
        [{"op": "relu", "operands": ["a0"]}],
        rows=False,
    )
    plan = {
        "version": 1,
        "inputs": [{"name": "arg0", "shape": [2, 2], "dtype": "f32"}],
        "steps": [
            {
                "kind": "triton",
                "kernel": "fp_chain_0",
                "launch": "bogus",
                "operands": ["arg0"],
                "result": "v0",
                "shape": [2, 2],
                "dtype": "f32",
                "attrs": {"block": 1024},
            }
        ],
        "outputs": ["v0"],
    }
    with pytest.raises(ValueError, match="unknown launch kind"):
        execute_plan(plan, [torch.randn(2, 2, device=DEV)], kernels, validate=False)


def test_triton_step_without_kernels_raises(require_triton):
    plan = {
        "version": 1,
        "inputs": [{"name": "arg0", "shape": [2, 2], "dtype": "f32"}],
        "steps": [
            {
                "kind": "triton",
                "kernel": "fp_chain_0",
                "launch": "flat",
                "operands": ["arg0"],
                "result": "v0",
                "shape": [2, 2],
                "dtype": "f32",
                "attrs": {"block": 1024},
            }
        ],
        "outputs": ["v0"],
    }
    with pytest.raises(RuntimeError, match="no kernels module"):
        execute_plan(plan, [torch.randn(2, 2, device=DEV)], kernels=None)


def test_unknown_step_kind_raises():
    plan = {
        "version": 1,
        "inputs": [{"name": "arg0", "shape": [2, 2], "dtype": "f32"}],
        "steps": [
            {
                "kind": "cuda",
                "op": "relu",
                "operands": ["arg0"],
                "result": "v0",
                "shape": [2, 2],
                "dtype": "f32",
                "attrs": {},
            }
        ],
        "outputs": ["v0"],
    }
    with pytest.raises(ValueError, match="unknown kind"):
        execute_plan(plan, [torch.randn(2, 2)])


def test_flat_launch_matches_eager(tmp_path, require_triton):
    _, kernels = _write_chain(
        tmp_path,
        2,
        [
            {"op": "add", "operands": ["a0", "a1"]},
            {"op": "scale", "operands": ["t0"], "factor": 2.0},
            {"op": "relu", "operands": ["t1"]},
            {"op": "exp", "operands": ["t2"]},
        ],
    )
    a = torch.randn(8, 16, device=DEV)
    b = torch.randn(8, 16, device=DEV)
    plan = {
        "version": 1,
        "inputs": [
            {"name": "arg0", "shape": [8, 16], "dtype": "f32"},
            {"name": "arg1", "shape": [8, 16], "dtype": "f32"},
        ],
        "steps": [
            {
                "kind": "triton",
                "kernel": "fp_chain_0",
                "launch": "flat",
                "operands": ["arg0", "arg1"],
                "result": "v0",
                "shape": [8, 16],
                "dtype": "f32",
                "attrs": chain_launch_attrs([8, 16], rows=False),
            }
        ],
        "outputs": ["v0"],
    }
    try:
        out = execute_plan(plan, [a, b], kernels)
    except Exception as exc:
        _maybe_skip_kernel_error(exc)
    expected = torch.exp(torch.relu((a + b) * 2.0))
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_rows_softmax_launch_matches_eager(tmp_path, require_triton):
    _, kernels = _write_chain(
        tmp_path,
        1,
        [
            {"op": "scale", "operands": ["a0"], "factor": 0.5},
            {"op": "softmax", "operands": ["t0"]},
            {"op": "relu", "operands": ["t1"]},
        ],
    )
    x = torch.randn(6, 100, device=DEV)
    plan = {
        "version": 1,
        "inputs": [{"name": "arg0", "shape": [6, 100], "dtype": "f32"}],
        "steps": [
            {
                "kind": "triton",
                "kernel": "fp_chain_0",
                "launch": "rows",
                "operands": ["arg0"],
                "result": "v0",
                "shape": [6, 100],
                "dtype": "f32",
                "attrs": chain_launch_attrs([6, 100], rows=True),
            }
        ],
        "outputs": ["v0"],
    }
    try:
        out = execute_plan(plan, [x], kernels)
    except Exception as exc:
        _maybe_skip_kernel_error(exc)
    expected = torch.relu(torch.softmax(x * 0.5, dim=-1))
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_attention_launch_matches_eager(tmp_path, require_triton):
    writer = KernelModuleWriter()
    writer.add_attention_kernel()
    path = tmp_path / "kernels.py"
    path.write_text(writer.write(), encoding="utf-8")
    try:
        kernels = load_kernels(path)
    except Exception as exc:
        _maybe_skip_kernel_error(exc)
    q, k, v = (torch.randn(1, 2, 20, 16, device=DEV) for _ in range(3))
    plan = {
        "version": 1,
        "inputs": [
            {"name": "arg0", "shape": [1, 2, 20, 16], "dtype": "f32"},
            {"name": "arg1", "shape": [1, 2, 20, 16], "dtype": "f32"},
            {"name": "arg2", "shape": [1, 2, 20, 16], "dtype": "f32"},
        ],
        "steps": [
            {
                "kind": "triton",
                "kernel": "fp_attention",
                "launch": "attention",
                "operands": ["arg0", "arg1", "arg2"],
                "result": "v0",
                "shape": [1, 2, 20, 16],
                "dtype": "f32",
                "attrs": attention_launch_attrs(0.25),
            }
        ],
        "outputs": ["v0"],
    }
    try:
        out = execute_plan(plan, [q, k, v], kernels)
    except Exception as exc:
        _maybe_skip_kernel_error(exc)
    eager = torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * 0.25, dim=-1), v)
    torch.testing.assert_close(out, eager, atol=5e-3, rtol=5e-3)


def test_flat_kernel_handles_noncontiguous_input(tmp_path, require_triton):
    _, kernels = _write_chain(
        tmp_path,
        2,
        [
            {"op": "add", "operands": ["a0", "a1"]},
            {"op": "scale", "operands": ["t0"], "factor": 2.0},
            {"op": "relu", "operands": ["t1"]},
            {"op": "exp", "operands": ["t2"]},
        ],
    )
    base = torch.randn(16, 8, device=DEV)
    a = base.t()
    assert list(a.shape) == [8, 16] and not a.is_contiguous()
    b = torch.randn(8, 16, device=DEV)
    plan = {
        "version": 1,
        "inputs": [
            {"name": "arg0", "shape": [8, 16], "dtype": "f32"},
            {"name": "arg1", "shape": [8, 16], "dtype": "f32"},
        ],
        "steps": [
            {
                "kind": "triton",
                "kernel": "fp_chain_0",
                "launch": "flat",
                "operands": ["arg0", "arg1"],
                "result": "v0",
                "shape": [8, 16],
                "dtype": "f32",
                "attrs": {"block": 1024},
            }
        ],
        "outputs": ["v0"],
    }
    try:
        out = execute_plan(plan, [a, b], kernels)
    except Exception as exc:
        _maybe_skip_kernel_error(exc)
    expected = torch.exp(torch.relu((a + b) * 2.0))
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_generated_kernel_names_contract():
    assert KERNEL_ATTENTION == "fp_attention"
    assert os.environ.get("TRITON_INTERPRET") == "1" or torch.cuda.is_available()
