from __future__ import annotations
import json, os, subprocess, tempfile
from pathlib import Path
import torch
from .frontend import trace_to_mlir
from .runtime import execute_plan, load_kernels, validate_plan
from .verify import EquivalenceError, verify_equivalence


def _is_runnable(p: Path) -> bool:
    """True if `p` can be passed to subprocess on this OS.

    A Linux ELF bind-mounted at build/bin/ exists() on Windows but is not
    executable; prefer fusionpass-compile.exe / .cmd wrappers instead.
    """
    if not p.is_file():
        return False
    if os.name != "nt":
        return True
    return p.suffix.lower() in {".exe", ".cmd", ".bat", ".ps1"}


def _with_exe(p: Path) -> Path:
    """On Windows the CMake binary is fusionpass-compile.exe; accept .exe/.cmd too."""
    for cand in (p, p.with_suffix(".exe"), p.with_suffix(".cmd")):
        if _is_runnable(cand):
            return cand
    return p


def _find_compiler() -> str:
    env = os.environ.get("FUSIONPASS_COMPILE")
    p = env or str(Path(os.environ.get("FUSIONPASS_BUILD_DIR", "build")) / "bin" / "fusionpass-compile")
    resolved = _with_exe(Path(p))
    # When FUSIONPASS_COMPILE is unset, a Docker Linux ELF in build/bin/ is not
    # runnable on Windows; fall back to the host .cmd wrapper (do not override
    # an explicit env path — tests monkeypatch a missing compiler).
    if not _is_runnable(resolved) and not env:
        wrapper = Path(__file__).resolve().parents[2] / "_phase1_verify" / "fusionpass-compile.cmd"
        if _is_runnable(wrapper):
            return str(wrapper)
    if not _is_runnable(resolved):
        raise FileNotFoundError(
            f"fusionpass-compile not found at {p}; build the C++ project or set FUSIONPASS_COMPILE"
        )
    return str(resolved)


class CompiledModel:
    def __init__(self, plan: dict, captured: list[torch.Tensor], workdir: Path, mlir_text: str, kernels=None):
        validate_plan(plan)                      # fail early on a corrupt plan
        self.plan, self._captured, self.workdir, self.mlir_text, self.kernels = plan, captured, workdir, mlir_text, kernels
        self.report = None                       # VerifyReport once verification has run
        self.verified = False                    # True ONLY if verification ran and passed

    def __call__(self, *inputs):
        return execute_plan(self.plan, list(inputs) + self._captured, self.kernels, validate=False)


def compile_model(model, example_inputs, *, fuse: bool = False, workdir=None, verify: bool = True,
                  strict: bool = True, atol=None, rtol=None) -> CompiledModel:
    example_inputs = tuple(example_inputs)
    mlir_text, arg_tensors = trace_to_mlir(model, example_inputs)
    workdir = Path(workdir or tempfile.mkdtemp(prefix="fusionpass_"))
    workdir.mkdir(parents=True, exist_ok=True)
    src = workdir / "model.mlir"
    src.write_text(mlir_text)
    proc = subprocess.run(
        [_find_compiler(), str(src), "-o", str(workdir), "--pipeline", "fuse" if fuse else "none"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"fusionpass-compile failed:\n{proc.stderr}\nMLIR:\n{mlir_text}")
    plan = json.loads((workdir / "plan.json").read_text())
    has_triton = any(s["kind"] == "triton" for s in plan["steps"])
    kernels = load_kernels(workdir / "kernels.py") if has_triton else None
    compiled = CompiledModel(plan, arg_tensors[len(example_inputs):], workdir, mlir_text, kernels)
    if verify:
        compiled.report = verify_equivalence(model, compiled, example_inputs, atol=atol, rtol=rtol)
        compiled.verified = compiled.report.ok
        if strict and not compiled.report.ok:
            raise EquivalenceError(f"fusion changed results: {compiled.report.reason}")
    return compiled
