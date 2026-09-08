"""Phase 2 end-to-end: fuse=True plans match eager PyTorch.

Skipped until --pipeline fuse lands and triton is importable.
"""

import pytest
import torch
from torch import nn

from fusionpass import compile_model

pytestmark = [
    pytest.mark.usefixtures("require_fuse_pipeline", "require_triton"),
]

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Attn(nn.Module):
    def forward(self, q, k, v):
        s = torch.matmul(q, k.transpose(-2, -1)) * 0.25
        return torch.matmul(torch.softmax(s, dim=-1), v)


class Chain(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.randn(8, 16))

    def forward(self, x):
        return torch.exp(torch.relu(x * 2.0 + self.b) * 0.1)


class SoftmaxChain(nn.Module):
    def forward(self, x):
        return torch.relu(torch.softmax(x * 0.5, dim=-1))


def kinds(c):
    return [s.get("launch", s["op"] if s["kind"] == "torch" else "?") for s in c.plan["steps"]]


def test_attention_fused_single_kernel():
    m = Attn().to(DEV)
    q, k, v = (torch.randn(1, 2, 20, 16, device=DEV) for _ in range(3))
    c = compile_model(m, (q, k, v), fuse=True)
    assert kinds(c) == ["attention"]
    torch.testing.assert_close(c(q, k, v), m(q, k, v), atol=5e-3, rtol=5e-3)


def test_chain_fused_single_kernel():
    m = Chain().to(DEV).eval()
    x = torch.randn(8, 16, device=DEV)
    c = compile_model(m, (x,), fuse=True)
    assert kinds(c) == ["flat"]
    torch.testing.assert_close(c(x), m(x), atol=1e-5, rtol=1e-5)


def test_softmax_chain_rows_kernel():
    m = SoftmaxChain().to(DEV)
    x = torch.randn(6, 100, device=DEV)
    c = compile_model(m, (x,), fuse=True)
    assert kinds(c) == ["rows"]
    torch.testing.assert_close(c(x), m(x), atol=1e-5, rtol=1e-5)


def test_unfused_and_fused_agree():
    m = Chain().to(DEV).eval()
    x = torch.randn(8, 16, device=DEV)
    torch.testing.assert_close(
        compile_model(m, (x,), fuse=False)(x),
        compile_model(m, (x,), fuse=True)(x),
    )
