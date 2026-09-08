"""Phase 2 fusion-pass and fuse=True plan tests.

Skip with a reason until C++ fusion siblings land (--fp-fuse-* / --pipeline fuse).
Does not modify Phase 1 tests.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import torch
from torch import nn

from fusionpass import compile_model

REPO = Path(__file__).resolve().parents[1]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def kinds(c):
    return [s.get("launch", s["op"] if s["kind"] == "torch" else "?") for s in c.plan["steps"]]


def compile_fuse(model, xs):
    """fuse=True without the Phase 3 verify gate when that kwarg exists."""
    try:
        return compile_model(model, xs, fuse=True, verify=False)
    except TypeError:
        return compile_model(model, xs, fuse=True)


def _run_opt(opt: str, src: Path, *flags: str) -> str:
    try:
        proc = subprocess.run(
            [opt, str(src), *flags],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except OSError as exc:
        pytest.skip(f"fusionpass-opt is not runnable here: {exc}")
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


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


def test_opt_registers_fusion_passes(require_fusion_passes):
    proc = subprocess.run(
        [require_fusion_passes, "--help"], capture_output=True, text=True, timeout=30
    )
    text = (proc.stdout or "") + (proc.stderr or "")
    assert "fp-fuse-attention" in text
    assert "fp-fuse-chains" in text


def test_fuse_attention_pass_rewrites_subgraph(require_fusion_passes):
    src = REPO / "test" / "fuse_attention.mlir"
    out = _run_opt(require_fusion_passes, src, "--fp-fuse-attention")
    attn = out.split("func.func @attn_multiuse")[0]
    assert "fp.fused_attention" in attn
    assert "2.500000e-01" in attn
    assert "fp.softmax" not in attn
    multi = out.split("func.func @attn_multiuse", 1)[1]
    assert "fp.fused_attention" not in multi
    assert "fp.softmax" in multi


def test_fuse_chains_pass_rewrites_elementwise(require_fusion_passes):
    src = REPO / "test" / "fuse_chains.mlir"
    out = _run_opt(require_fusion_passes, src, "--fp-fuse-chains")
    chain = out.split("func.func @multiuse")[0]
    assert '"fp.fused_chain"' in chain
    assert "fp.add" in chain
    assert "fp.exp" in chain
    multi = out.split("func.func @multiuse", 1)[1].split("func.func @softmax_chain")[0]
    assert '"fp.fused_chain"' not in multi
    sm = out.split("func.func @softmax_chain", 1)[1].split("func.func @softmax_axis0")[0]
    assert '"fp.fused_chain"' in sm
    assert "fp.softmax" in sm
    axis0 = out.split("func.func @softmax_axis0", 1)[1].split("func.func @")[0]
    assert '"fp.fused_chain"' not in axis0


def test_compile_pipeline_fuse_writes_artifacts(require_fuse_pipeline, tmp_path):
    src = tmp_path / "model.mlir"
    src.write_text(
        "func.func @main(%a: tensor<8x16xf32>, %b: tensor<8x16xf32>) -> tensor<8x16xf32> {\n"
        "  %0 = fp.add %a, %b : tensor<8x16xf32>\n"
        "  %1 = fp.scale %0 {factor = 2.000000e+00 : f32} : tensor<8x16xf32>\n"
        "  %2 = fp.relu %1 : tensor<8x16xf32>\n"
        "  %3 = fp.exp %2 : tensor<8x16xf32>\n"
        "  return %3 : tensor<8x16xf32>\n"
        "}\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    proc = subprocess.run(
        [require_fuse_pipeline, str(src), "-o", str(out), "--pipeline", "fuse"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert (out / "plan.json").is_file()
    assert (out / "kernels.py").is_file()
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    assert plan["version"] == 1
    assert any(s["kind"] == "triton" for s in plan["steps"])
    assert any(s.get("launch") == "flat" for s in plan["steps"])
    kernels = (out / "kernels.py").read_text(encoding="utf-8")
    assert "def fp_chain_0" in kernels
    assert "import triton" in kernels


def test_attention_fused_plan_is_single_triton_step(require_fuse_pipeline, require_triton):
    m = Attn().to(DEV)
    q, k, v = (torch.randn(1, 2, 20, 16, device=DEV) for _ in range(3))
    c = compile_fuse(m, (q, k, v))
    assert kinds(c) == ["attention"]
    step = c.plan["steps"][0]
    assert step["kind"] == "triton"
    assert step["kernel"] == "fp_attention"
    assert step["attrs"]["scale"] == pytest.approx(0.25)
    assert step["attrs"]["block_m"] == 64
    assert step["attrs"]["block_n"] == 64
    assert (c.workdir / "kernels.py").is_file()
    text = (c.workdir / "kernels.py").read_text(encoding="utf-8")
    assert "def fp_attention" in text
    optimized = c.workdir / "optimized.mlir"
    if optimized.is_file():
        opt = optimized.read_text(encoding="utf-8")
        assert "fp.fused_attention" in opt
        assert "fp.softmax" not in opt


def test_chain_fused_plan_is_flat_launch(require_fuse_pipeline, require_triton):
    m = Chain().to(DEV).eval()
    x = torch.randn(8, 16, device=DEV)
    c = compile_fuse(m, (x,))
    assert kinds(c) == ["flat"]
    step = c.plan["steps"][0]
    assert step["kind"] == "triton"
    assert step["kernel"] == "fp_chain_0"
    assert step["attrs"]["block"] == 1024


def test_softmax_chain_plan_is_rows_launch(require_fuse_pipeline, require_triton):
    m = SoftmaxChain().to(DEV)
    x = torch.randn(6, 100, device=DEV)
    c = compile_fuse(m, (x,))
    assert kinds(c) == ["rows"]
    step = c.plan["steps"][0]
    assert step["kernel"] == "fp_chain_0"
    assert step["attrs"]["block_n"] == 128
    assert step["attrs"]["num_warps"] == 4


def test_unsupported_head_dim_not_fused(require_fuse_pipeline, require_triton):
    q, k, v = (torch.randn(1, 2, 8, 24, device=DEV) for _ in range(3))
    c = compile_fuse(Attn(), (q, k, v))
    assert "attention" not in kinds(c)


def test_non_last_axis_softmax_not_rows(require_fuse_pipeline):
    class M(nn.Module):
        def forward(self, x):
            return torch.relu(torch.softmax(x * 0.5, dim=0))

    x = torch.randn(6, 32, device=DEV)
    c = compile_fuse(M(), (x,))
    assert "rows" not in kinds(c)
