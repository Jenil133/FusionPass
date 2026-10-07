"""Phase 3 edge-case and fallback matrix.

Skips if fuse=True or fusionpass-compile is not ready.
"""

from __future__ import annotations

import os
import subprocess

import pytest
import torch
from torch import nn

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

from fusionpass import UnsupportedOp, compile_model

DEV = "cuda" if torch.cuda.is_available() else "cpu"
launches = lambda c: [s.get("launch", "torch") for s in c.plan["steps"]]


def _compiler_runnable(path: str) -> bool:
    try:
        subprocess.run([path, "--help"], capture_output=True, text=True, timeout=15)
        return True
    except (OSError, subprocess.TimeoutExpired):
        return False


@pytest.fixture(autouse=True)
def _need_runnable_compiler(request, fusionpass_compile):
    # Frontend-only cases never invoke the compiler.
    if request.node.name == "test_unsupported_broadcast_still_raises":
        return
    if not _compiler_runnable(fusionpass_compile):
        pytest.skip("fusionpass-compile is not runnable on this host")


def _compile(*args, **kwargs):
    try:
        return compile_model(*args, **kwargs)
    except (FileNotFoundError, OSError) as e:
        pytest.skip(f"fusionpass-compile not runnable: {e}")


class Attn(nn.Module):
    def forward(self, q, k, v):
        return torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * 0.25, dim=-1), v)


@pytest.mark.parametrize("S", [1, 7, 64, 65, 130])
def test_attention_sequence_lengths(S, require_triton):
    q, k, v = (torch.randn(1, 2, S, 16, device=DEV) for _ in range(3))
    c = _compile(Attn(), (q, k, v), fuse=True)
    assert launches(c) == ["attention"]


def test_unsupported_head_dim_falls_back(require_triton):
    q, k, v = (torch.randn(1, 2, 8, 24, device=DEV) for _ in range(3))
    c = _compile(Attn(), (q, k, v), fuse=True)
    assert "attention" not in launches(c)


def test_cross_attention_shapes_fall_back():
    class Cross(nn.Module):
        def forward(self, q, k, v):
            return torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-2, -1)), dim=-1), v)

    q = torch.randn(1, 2, 8, 16, device=DEV)
    k = torch.randn(1, 2, 12, 16, device=DEV)
    v = torch.randn(1, 2, 12, 16, device=DEV)
    c = _compile(Cross(), (q, k, v), fuse=True)
    assert "attention" not in launches(c)


def test_multiuse_probabilities_not_fused():
    class Twice(nn.Module):
        def forward(self, q, k, v):
            p = torch.softmax(torch.matmul(q, k.transpose(-2, -1)), dim=-1)
            return torch.matmul(p, v) + torch.matmul(p, v)

    q, k, v = (torch.randn(1, 2, 16, 16, device=DEV) for _ in range(3))
    c = _compile(Twice(), (q, k, v), fuse=True)
    assert "attention" not in launches(c)


def test_non_last_axis_softmax_not_fused():
    class M(nn.Module):
        def forward(self, x):
            return torch.relu(torch.softmax(x * 0.5, dim=0))

    x = torch.randn(6, 32, device=DEV)
    assert "rows" not in launches(_compile(M(), (x,), fuse=True))


def test_duplicate_operand_chain(require_triton):
    class M(nn.Module):
        def forward(self, x):
            return torch.relu(x * x)

    x = torch.randn(5, 7, device=DEV)
    c = _compile(M(), (x,), fuse=True)
    assert launches(c) == ["flat"] and c.plan["steps"][0]["operands"] == ["arg0"]


def test_large_magnitude_softmax_is_stable(require_triton):
    class M(nn.Module):
        def forward(self, x):
            return torch.softmax(x * 1.0, dim=-1) * 2.0

    x = torch.randn(4, 50, device=DEV) * 50
    c = _compile(M(), (x,), fuse=True)
    torch.testing.assert_close(c(x), M()(x), atol=1e-4, rtol=1e-4)


def test_noncontiguous_input_handled(require_triton):
    class M(nn.Module):
        def forward(self, x):
            return torch.exp(torch.relu(x * 2.0) * 0.1)

    base = torch.randn(16, 8, device=DEV)
    x = base.t()
    c = _compile(M(), (x.contiguous(),), fuse=True)
    torch.testing.assert_close(c(x), M()(x), atol=1e-5, rtol=1e-5)


def test_wrong_shape_at_call_time(require_triton):
    class M(nn.Module):
        def forward(self, x):
            return torch.relu(x * 2.0)

    x = torch.randn(4, 4, device=DEV)
    c = _compile(M(), (x,), fuse=True)
    with pytest.raises(ValueError):
        c(torch.randn(4, 5, device=DEV))


def test_unsupported_broadcast_still_raises():
    class M(nn.Module):
        def forward(self, x, b):
            return x + b

    with pytest.raises(UnsupportedOp):
        compile_model(M(), (torch.randn(4, 8, device=DEV), torch.randn(8, device=DEV)), fuse=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp16 kernels need CUDA")
def test_fp16_attention_and_chain():
    class Chain(nn.Module):
        def forward(self, x):
            return torch.relu(x * 1.5) * 0.5

    q, k, v = (torch.randn(2, 4, 100, 64, device="cuda", dtype=torch.float16) for _ in range(3))
    assert launches(_compile(Attn(), (q, k, v), fuse=True)) == ["attention"]
    x = torch.randn(64, 256, device="cuda", dtype=torch.float16)
    assert launches(_compile(Chain(), (x,), fuse=True)) == ["flat"]
