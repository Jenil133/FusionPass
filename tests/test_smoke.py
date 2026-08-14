"""Phase 1 smoke tests: imports, GPU visibility, CLI --help.

CLI tests skip until fusionpass-opt / fusionpass-compile exist.
There is no HTTP health endpoint in Phase 1.
"""

import json
import subprocess

import pytest
import torch


def test_import_fusionpass_package():
    import fusionpass
    from fusionpass import api, frontend, runtime

    assert callable(fusionpass.compile_model)
    assert callable(frontend.trace_to_mlir)
    assert callable(runtime.execute_plan)
    assert callable(api.compile_model)


def test_gpu_device_visibility():
    """Probe CUDA visibility. The local venv is a CPU PyTorch wheel, so
    is_available() may be False; the probe itself must still succeed."""
    available = torch.cuda.is_available()
    if not available:
        pytest.skip("CUDA not visible (CPU PyTorch wheel or no driver binding)")
    assert torch.cuda.device_count() >= 1
    name = torch.cuda.get_device_name(0)
    assert isinstance(name, str) and name


def test_fusionpass_opt_help(fusionpass_opt):
    proc = subprocess.run([fusionpass_opt, "--help"], capture_output=True, text=True)
    text = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0
    assert "fusionpass" in text.lower() or "mlir" in text.lower()


def test_fusionpass_compile_help(fusionpass_compile):
    proc = subprocess.run([fusionpass_compile, "--help"], capture_output=True, text=True)
    text = (proc.stdout or "") + (proc.stderr or "")
    assert "pipeline" in text.lower() or "output directory" in text.lower()
    assert "-o" in text


def test_fusionpass_compile_unknown_pipeline_exits_2(fusionpass_compile, tmp_path):
    src = tmp_path / "ok.mlir"
    src.write_text(
        "func.func @main(%a: tensor<2x2xf32>) -> tensor<2x2xf32> {\n"
        "  %0 = fp.relu %a : tensor<2x2xf32>\n"
        "  return %0 : tensor<2x2xf32>\n"
        "}\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [fusionpass_compile, str(src), "-o", str(tmp_path / "out"), "--pipeline", "bogus"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "unknown --pipeline" in proc.stderr


def test_fusionpass_compile_malformed_mlir_exits_1(fusionpass_compile, tmp_path):
    src = tmp_path / "bad.mlir"
    src.write_text("this is not mlir\n", encoding="utf-8")
    proc = subprocess.run(
        [fusionpass_compile, str(src), "-o", str(tmp_path / "out")],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1


def test_section7_example_emits_five_torch_steps(fusionpass_compile, tmp_path):
    src = tmp_path / "attn.mlir"
    src.write_text(
        """module {
  func.func @main(%arg0: tensor<1x2x8x16xf32>, %arg1: tensor<1x2x8x16xf32>, %arg2: tensor<1x2x8x16xf32>) -> tensor<1x2x8x16xf32> {
    %kt = fp.transpose %arg1 : tensor<1x2x8x16xf32> -> tensor<1x2x16x8xf32>
    %s = fp.matmul %arg0, %kt : (tensor<1x2x8x16xf32>, tensor<1x2x16x8xf32>) -> tensor<1x2x8x8xf32>
    %sc = fp.scale %s {factor = 2.500000000e-01 : f32} : tensor<1x2x8x8xf32>
    %p = fp.softmax %sc {axis = 3 : i64} : tensor<1x2x8x8xf32>
    %o = fp.matmul %p, %arg2 : (tensor<1x2x8x8xf32>, tensor<1x2x8x16xf32>) -> tensor<1x2x8x16xf32>
    return %o : tensor<1x2x8x16xf32>
  }
}
""",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    proc = subprocess.run(
        [fusionpass_compile, str(src), "-o", str(out)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    assert plan["version"] == 1
    assert len(plan["steps"]) == 5
    assert all(s["kind"] == "torch" for s in plan["steps"])
    assert [s["op"] for s in plan["steps"]] == ["transpose", "matmul", "scale", "softmax", "matmul"]
    assert plan["outputs"] == ["v4"]
    assert [i["name"] for i in plan["inputs"]] == ["arg0", "arg1", "arg2"]
    assert plan["steps"][2]["attrs"]["factor"] == pytest.approx(0.25)
    assert plan["steps"][3]["attrs"]["axis"] == 3
