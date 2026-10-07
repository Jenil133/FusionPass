"""Phase 3 CLI and demo smoke tests.

`python -m fusionpass check` and `demo attention|chain`.
Skips when __main__.py, the compiler, or fuse=True is not ready.
"""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

try:
    from fusionpass.__main__ import main
except ImportError:
    pytest.skip("fusionpass CLI (__main__.py) not available", allow_module_level=True)


def _fuse_implemented() -> bool:
    try:
        from fusionpass.api import compile_model

        src = inspect.getsource(compile_model)
    except (ImportError, OSError):
        return False
    return "fusion arrives in Phase 2" not in src


def _run_cli(args, *, env=None, timeout=180):
    e = os.environ.copy()
    if env:
        e.update(env)
    if not torch.cuda.is_available():
        e.setdefault("TRITON_INTERPRET", "1")
    return subprocess.run(
        [sys.executable, "-m", "fusionpass", *args],
        capture_output=True,
        text=True,
        env=e,
        timeout=timeout,
        cwd=str(Path(__file__).resolve().parents[1]),
    )


def test_cli_help():
    proc = _run_cli(["--help"], timeout=30)
    text = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0
    assert "check" in text
    assert "demo" in text


def test_cli_demo_help():
    proc = _run_cli(["demo", "--help"], timeout=30)
    text = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0
    assert "attention" in text
    assert "chain" in text


def test_cli_requires_subcommand():
    proc = _run_cli([], timeout=30)
    assert proc.returncode != 0


def test_cli_check_ok(fusionpass_compile):
    code = main(["check"])
    assert code == 0


def test_cli_check_subprocess(fusionpass_compile):
    proc = _run_cli(["check"], timeout=60)
    out = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, out
    assert "compiler" in out.lower()
    assert "torch" in out.lower()


def test_cli_check_missing_compiler_exits_3(monkeypatch, tmp_path):
    monkeypatch.setenv("FUSIONPASS_COMPILE", str(tmp_path / "no-such-fusionpass-compile"))
    monkeypatch.setenv("FUSIONPASS_BUILD_DIR", str(tmp_path / "empty-build"))
    code = main(["check"])
    assert code == 3


def _compiler_runnable(path: str) -> bool:
    try:
        subprocess.run([path, "--help"], capture_output=True, text=True, timeout=15)
        return True
    except (OSError, subprocess.TimeoutExpired):
        return False


def test_cli_demo_attention_smoke(fusionpass_compile, require_triton):
    if not _fuse_implemented():
        pytest.skip("fuse=True not implemented")
    if not _compiler_runnable(fusionpass_compile):
        pytest.skip("fusionpass-compile is not runnable on this host")
    proc = _run_cli(
        [
            "demo",
            "attention",
            "--dtype",
            "f32",
            "--batch",
            "1",
            "--heads",
            "2",
            "--seq",
            "16",
            "--head-dim",
            "16",
        ],
        timeout=180,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode == 3:
        pytest.skip(f"compiler missing during demo: {out}")
    assert proc.returncode == 0, out
    assert "verified" in out.lower()
    assert "plan" in out.lower()
    if not torch.cuda.is_available():
        assert "benchmark" in out.lower()
        assert "skipped" in out.lower()


def test_cli_demo_chain_smoke(fusionpass_compile, require_triton):
    if not _fuse_implemented():
        pytest.skip("fuse=True not implemented")
    if not _compiler_runnable(fusionpass_compile):
        pytest.skip("fusionpass-compile is not runnable on this host")
    proc = _run_cli(
        [
            "demo",
            "chain",
            "--dtype",
            "f32",
            "--rows",
            "8",
            "--cols",
            "16",
        ],
        timeout=180,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode == 3:
        pytest.skip(f"compiler missing during demo: {out}")
    assert proc.returncode == 0, out
    assert "verified" in out.lower()
    assert "plan" in out.lower()
    if not torch.cuda.is_available():
        assert "benchmark" in out.lower()
        assert "skipped" in out.lower()


def test_example_scripts_wire_cli():
    root = Path(__file__).resolve().parents[1] / "examples"
    attn = root / "attention_demo.py"
    chain = root / "chain_demo.py"
    if not attn.exists() or not chain.exists():
        pytest.skip("examples/ demo scripts not present")
    assert "demo" in attn.read_text(encoding="utf-8")
    assert "attention" in attn.read_text(encoding="utf-8")
    assert "demo" in chain.read_text(encoding="utf-8")
    assert "chain" in chain.read_text(encoding="utf-8")
