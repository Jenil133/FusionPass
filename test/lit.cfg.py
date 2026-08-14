import os
import subprocess
import lit.formats

build = os.environ.get("FUSIONPASS_BUILD_DIR", "build")
config.name = "fusionpass"
# Internal shell on Windows so RUN: pipes work without bash (Phase 1 spec uses ShTest(True) on Unix).
config.test_format = lit.formats.ShTest(execute_external=(os.name != "nt"))
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(os.path.abspath(build), "test")
_exe = ".exe" if os.name == "nt" else ""
_opt_dir = os.path.join(os.path.abspath(build), "bin")
_opt = os.path.join(_opt_dir, "fusionpass-opt" + _exe)
if not os.path.exists(_opt):
    _bare = os.path.join(_opt_dir, "fusionpass-opt")
    if os.path.exists(_bare):
        _opt = _bare
config.substitutions.append(("%fusionpass-opt", _opt))
_llvm_bin = os.path.join(os.environ.get("LLVM_PREFIX", "/llvm-install"), "bin")
config.environment["PATH"] = (
    _opt_dir + os.pathsep + _llvm_bin + os.pathsep + os.environ.get("PATH", "")
)

# Skip Phase 2 pass tests until fusionpass-opt registers them (siblings land).
try:
    _help = subprocess.run([_opt, "--help"], capture_output=True, text=True, timeout=30)
    _help_text = (_help.stdout or "") + (_help.stderr or "")
except Exception:
    _help_text = ""
if "fp-fuse-attention" in _help_text:
    config.available_features.add("fp-fuse-attention")
if "fp-fuse-chains" in _help_text:
    config.available_features.add("fp-fuse-chains")
if "fp-fuse-attention" in _help_text and "fp-fuse-chains" in _help_text:
    config.available_features.add("fusion-passes")
