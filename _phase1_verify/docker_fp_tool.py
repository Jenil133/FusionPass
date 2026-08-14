"""Run Linux fusionpass-* binaries inside fusionpass-dev, mapping host paths to /work."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HOST_ROOT = Path(r"C:\Users\018224960\Desktop\FusionPass").resolve()
CONTAINER = "fusionpass-dev"
TOOLS = {
    "compile": "/work/build/bin/fusionpass-compile",
    "opt": "/work/build/bin/fusionpass-opt",
}


def to_container(arg: str) -> str:
    if arg.startswith("-") and not arg.startswith("-o"):
        return arg
    if arg in {"none", "fuse", "true", "false"}:
        return arg
    looks_path = (
        (len(arg) >= 2 and arg[1] == ":")
        or ("\\" in arg)
        or ("/" in arg)
        or Path(arg).exists()
    )
    if not looks_path:
        return arg
    p = Path(arg)
    resolved = p.resolve() if p.is_absolute() or p.exists() else (Path.cwd() / p)
    try:
        resolved = resolved.resolve()
    except OSError:
        pass
    try:
        rel = resolved.relative_to(HOST_ROOT)
    except ValueError:
        return arg
    return "/work/" + rel.as_posix()


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in TOOLS:
        print("usage: docker_fp_tool.py compile|opt [args...]", file=sys.stderr)
        return 2
    tool = TOOLS[argv[0]]
    args = [to_container(a) for a in argv[1:]]
    proc = subprocess.run(
        ["docker", "exec", "-w", "/work", CONTAINER, tool, *args],
    )
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
