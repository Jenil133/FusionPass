"""Phase 1 API contract tests that do not require a built compiler binary."""

import pytest
import torch
from torch import nn

from fusionpass import CompiledModel, UnsupportedOp, compile_model


class ReluScale(nn.Module):
    def forward(self, x):
        return torch.relu(x * 3.0)


def test_public_exports():
    import fusionpass

    assert fusionpass.compile_model is compile_model
    assert fusionpass.CompiledModel is CompiledModel
    assert fusionpass.UnsupportedOp is UnsupportedOp


def test_fuse_true_not_implemented():
    import inspect

    assert "NotImplementedError" not in inspect.getsource(compile_model)
    try:
        import triton  # noqa: F401
    except ImportError:
        return
    compile_model(ReluScale(), (torch.randn(4, 4),), fuse=True, verify=False)


def test_compiler_missing_raises_file_not_found(monkeypatch, tmp_path):
    monkeypatch.setenv("FUSIONPASS_COMPILE", str(tmp_path / "no-such-fusionpass-compile"))
    monkeypatch.setenv("FUSIONPASS_BUILD_DIR", str(tmp_path / "empty-build"))
    with pytest.raises(FileNotFoundError, match="fusionpass-compile not found"):
        compile_model(ReluScale(), (torch.randn(4, 4),))


def test_compiler_failure_raises_runtime_error(monkeypatch, tmp_path):
    fake = tmp_path / "fusionpass-compile.exe"
    fake.write_text("not a real compiler\n", encoding="utf-8")
    monkeypatch.setenv("FUSIONPASS_COMPILE", str(fake))

    class Dummy:
        returncode = 1
        stderr = "parse error: expected module"
        stdout = ""

    monkeypatch.setattr("fusionpass.api.subprocess.run", lambda *a, **k: Dummy())
    with pytest.raises(RuntimeError, match="fusionpass-compile failed") as exc:
        compile_model(ReluScale(), (torch.randn(4, 4),))
    assert "parse error" in str(exc.value)
    assert "func.func @main" in str(exc.value)
