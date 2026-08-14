import pytest
import torch
from torch import nn

from fusionpass.frontend import UnsupportedOp, trace_to_mlir


class Chain(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.randn(8, 16))

    def forward(self, x):
        return torch.exp(torch.relu(x * 2.0 + self.b))


class Attn(nn.Module):
    def forward(self, q, k, v):
        s = torch.matmul(q, k.transpose(-2, -1)) * 0.25
        return torch.matmul(torch.softmax(s, dim=-1), v)


def test_chain_emits_fp_ops():
    text, args = trace_to_mlir(Chain(), (torch.randn(8, 16),))
    for tok in ("func.func @main", "fp.scale", "fp.add", "fp.relu", "fp.exp"):
        assert tok in text
    assert len(args) == 2  # input + captured parameter


def test_broadcast_rejected():
    class B(nn.Module):
        def forward(self, x, y):
            return x + y

    with pytest.raises(UnsupportedOp):
        trace_to_mlir(B(), (torch.randn(4, 8), torch.randn(8)))


def test_attention_emits_fp_ops():
    q, k, v = (torch.randn(1, 2, 8, 16) for _ in range(3))
    text, args = trace_to_mlir(Attn(), (q, k, v))
    for tok in ("func.func @main", "fp.transpose", "fp.matmul", "fp.scale", "fp.softmax"):
        assert tok in text
    assert "axis = 3 : i64" in text
    assert len(args) == 3


def test_scale_from_mul_and_div_constant():
    class Mul(nn.Module):
        def forward(self, x):
            return x * 2.0

    class Div(nn.Module):
        def forward(self, x):
            return x / 4.0

    mul_text, _ = trace_to_mlir(Mul(), (torch.randn(4, 4),))
    div_text, _ = trace_to_mlir(Div(), (torch.randn(4, 4),))
    assert "fp.scale" in mul_text
    assert "2.000000000e+00" in mul_text
    assert "fp.scale" in div_text
    assert "2.500000000e-01" in div_text


def test_f16_dtype_emitted():
    class M(nn.Module):
        def forward(self, x):
            return torch.relu(x)

    text, _ = trace_to_mlir(M(), (torch.randn(4, 4, dtype=torch.float16),))
    assert "xf16" in text
    assert "func.func @main" in text


def test_tanh_rejected():
    class T(nn.Module):
        def forward(self, x):
            return torch.tanh(x)

    with pytest.raises(UnsupportedOp):
        trace_to_mlir(T(), (torch.randn(4, 4),))


def test_linear_module_rejected():
    class L(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)

        def forward(self, x):
            return self.lin(x)

    with pytest.raises(UnsupportedOp, match="call_module"):
        trace_to_mlir(L(), (torch.randn(4, 8),))


def test_scalar_add_rejected():
    class S(nn.Module):
        def forward(self, x):
            return x + 1.0

    with pytest.raises(UnsupportedOp):
        trace_to_mlir(S(), (torch.randn(4, 4),))


def test_unsupported_dtype_rejected():
    class M(nn.Module):
        def forward(self, x):
            return torch.relu(x)

    with pytest.raises(UnsupportedOp, match="unsupported dtype"):
        trace_to_mlir(M(), (torch.randn(4, 4, dtype=torch.float64),))


def test_multiple_outputs_rejected():
    class M(nn.Module):
        def forward(self, x):
            return x, torch.relu(x)

    with pytest.raises(UnsupportedOp, match="exactly one tensor output"):
        trace_to_mlir(M(), (torch.randn(4, 4),))


def test_non_last_transpose_rejected():
    class M(nn.Module):
        def forward(self, x):
            return x.transpose(0, 1)

    with pytest.raises(UnsupportedOp, match="last two dims"):
        trace_to_mlir(M(), (torch.randn(2, 3, 4),))


def test_tensor_div_rejected():
    class M(nn.Module):
        def forward(self, x, y):
            return x / y

    with pytest.raises(UnsupportedOp, match="tensor/tensor division"):
        trace_to_mlir(M(), (torch.randn(4, 4), torch.randn(4, 4)))
