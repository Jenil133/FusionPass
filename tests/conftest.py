import os
import subprocess
from pathlib import Path

# Compiler I/O and pytest tmp must live on the repo bind mount so the Docker
# fusionpass-compile wrapper can map host paths to /work. Set this BEFORE
# importing torch (tempfile.gettempdir() is cached).
_REPO = Path(__file__).resolve().parents[1]
_HOST_TMP = _REPO / "_phase1_tmp"
_HOST_TMP.mkdir(parents=True, exist_ok=True)
os.environ["TMP"] = str(_HOST_TMP)
os.environ["TEMP"] = str(_HOST_TMP)
os.environ["TMPDIR"] = str(_HOST_TMP)

# Phase 2: TRITON_INTERPRET must be set before any test imports triton.
import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest


def _is_runnable(p: Path) -> bool:
    """Skip a Linux ELF bind-mounted at build/bin/ when running on Windows."""
    if not p.is_file():
        return False
    if os.name != "nt":
        return True
    return p.suffix.lower() in {".exe", ".cmd", ".bat", ".ps1"}


def resolve_tool(name: str) -> Path | None:
    """Locate a fusionpass C++ tool, accepting a Windows .exe suffix.

    Search order: FUSIONPASS_COMPILE (for fusionpass-compile, and its sibling
    directory for fusionpass-opt), then $FUSIONPASS_BUILD_DIR/bin/<name>,
    then _phase1_verify/<name>.cmd (Docker wrapper on this Windows host).
    Non-runnable Linux ELFs are skipped.
    """
    candidates: list[Path] = []
    env_compile = os.environ.get("FUSIONPASS_COMPILE")
    if env_compile:
        env_path = Path(env_compile)
        if name.startswith("fusionpass-compile"):
            candidates.append(env_path)
        candidates.append(env_path.parent / name)
    build = Path(os.environ.get("FUSIONPASS_BUILD_DIR", "build"))
    candidates.append(build / "bin" / name)
    candidates.append(_REPO / "_phase1_verify" / name)

    seen: set[Path] = set()
    for p in candidates:
        for cand in (p, p.with_suffix(".exe"), p.with_suffix(".cmd")):
            if cand in seen:
                continue
            seen.add(cand)
            if _is_runnable(cand):
                return cand
    return None


def _tool_help(path: str) -> str:
    try:
        proc = subprocess.run([path, "--help"], capture_output=True, text=True, timeout=30)
    except OSError:
        return ""
    return (proc.stdout or "") + (proc.stderr or "")


def fusion_passes_registered(opt_path: str) -> bool:
    text = _tool_help(opt_path)
    return "fp-fuse-attention" in text and "fp-fuse-chains" in text


@pytest.fixture(scope="session")
def fusionpass_compile():
    p = resolve_tool("fusionpass-compile")
    if p is None:
        pytest.skip("fusionpass-compile not built")
    return str(p)


@pytest.fixture(scope="session")
def fusionpass_opt():
    p = resolve_tool("fusionpass-opt")
    if p is None:
        pytest.skip("fusionpass-opt not built")
    return str(p)


@pytest.fixture(scope="session")
def require_fusion_passes(fusionpass_opt):
    if not fusion_passes_registered(fusionpass_opt):
        pytest.skip(
            "fusionpass-opt does not register --fp-fuse-attention/--fp-fuse-chains yet "
            "(Phase 2 sibling)"
        )
    return fusionpass_opt


@pytest.fixture(scope="session")
def require_fuse_pipeline(fusionpass_compile, tmp_path_factory):
    """Skip until fusionpass-compile accepts --pipeline fuse."""
    tmp = tmp_path_factory.mktemp("fuse_probe")
    src = tmp / "probe.mlir"
    src.write_text(
        "func.func @main(%a: tensor<2x2xf32>) -> tensor<2x2xf32> {\n"
        "  %0 = fp.relu %a : tensor<2x2xf32>\n"
        "  return %0 : tensor<2x2xf32>\n"
        "}\n",
        encoding="utf-8",
    )
    out = tmp / "out"
    try:
        proc = subprocess.run(
            [fusionpass_compile, str(src), "-o", str(out), "--pipeline", "fuse"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except OSError as exc:
        pytest.skip(f"fusionpass-compile is not runnable here: {exc}")
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        pytest.skip(f"--pipeline fuse not available yet: {err or proc.returncode}")
    return fusionpass_compile


@pytest.fixture(scope="session")
def require_triton():
    pytest.importorskip("triton", reason="triton not importable (use the PyTorch-bundled wheel)")
