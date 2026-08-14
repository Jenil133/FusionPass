import pytest
import torch
from torch import nn

from fusionpass import compile_model

pytestmark = pytest.mark.usefixtures("fusionpass_compile")


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


def test_chain_matches_eager():
    m, x = Chain().eval(), torch.randn(8, 16)
    c = compile_model(m, (x,))
    assert all(s["kind"] == "torch" for s in c.plan["steps"])
    torch.testing.assert_close(c(x), m(x))


def test_attention_matches_eager():
    m = Attn()
    q, k, v = (torch.randn(1, 2, 8, 16) for _ in range(3))
    c = compile_model(m, (q, k, v))
    torch.testing.assert_close(c(q, k, v), m(q, k, v))
    assert len(c.plan["steps"]) == 5


def test_chain_plan_schema():
    m, x = Chain().eval(), torch.randn(8, 16)
    c = compile_model(m, (x,))
    assert c.plan["version"] == 1
    assert [i["name"] for i in c.plan["inputs"]] == ["arg0", "arg1"]
    assert [s["op"] for s in c.plan["steps"]] == ["scale", "add", "relu", "exp"]
    assert [s["result"] for s in c.plan["steps"]] == ["v0", "v1", "v2", "v3"]
    assert c.plan["outputs"] == ["v3"]
    assert all(s["kind"] == "torch" for s in c.plan["steps"])
    assert all(s["dtype"] == "f32" for s in c.plan["steps"])


def test_attention_plan_schema():
    m = Attn()
    q, k, v = (torch.randn(1, 2, 8, 16) for _ in range(3))
    c = compile_model(m, (q, k, v))
    assert [s["op"] for s in c.plan["steps"]] == ["transpose", "matmul", "scale", "softmax", "matmul"]
    assert c.plan["outputs"] == ["v4"]
    assert c.plan["steps"][2]["attrs"]["factor"] == pytest.approx(0.25)
    assert c.plan["steps"][3]["attrs"]["axis"] == 3


def test_section11_relu_scale_smoke():
    class M(nn.Module):
        def forward(self, x):
            return torch.relu(x * 3.0)

    x = torch.randn(4, 4)
    c = compile_model(M(), (x,))
    assert len(c.plan["steps"]) == 2
    assert [s["op"] for s in c.plan["steps"]] == ["scale", "relu"]
    torch.testing.assert_close(c(x), M()(x))


def test_wrong_shape_at_call_time():
    class M(nn.Module):
        def forward(self, x):
            return torch.relu(x)

    x = torch.randn(4, 4)
    c = compile_model(M(), (x,))
    with pytest.raises(ValueError):
        c(torch.randn(4, 5))
