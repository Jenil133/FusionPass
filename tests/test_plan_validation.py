import copy

import pytest

from fusionpass.runtime import validate_plan
from fusionpass.validate import validate_plan as validate_plan_mod
from fusionpass.traffic import estimate_bytes, summarize_plan

GOOD = {"version": 1, "inputs": [{"name": "arg0", "shape": [2, 2], "dtype": "f32"}],
        "steps": [{"kind": "torch", "op": "relu", "operands": ["arg0"], "result": "v0",
                   "shape": [2, 2], "dtype": "f32", "attrs": {}}],
        "outputs": ["v0"]}


def test_good_plan_ok():
    validate_plan(GOOD)
    validate_plan_mod(GOOD)


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(version=2),
    lambda p: p["steps"][0].update(operands=["nope"]),
    lambda p: p["steps"][0].update(op="tanh"),
    lambda p: p["steps"][0].update(kind="cuda"),
    lambda p: p["steps"][0].update(result="arg0"),
    lambda p: p.update(outputs=["v9"]),
])
def test_bad_plans_rejected(mutate):
    p = copy.deepcopy(GOOD)
    mutate(p)
    with pytest.raises(ValueError):
        validate_plan(p)


def test_inputs_steps_outputs_must_be_lists():
    for key in ("inputs", "steps", "outputs"):
        p = copy.deepcopy(GOOD)
        p[key] = "nope"
        with pytest.raises(ValueError, match=key):
            validate_plan(p)


def test_bad_input_spec_rejected():
    p = copy.deepcopy(GOOD)
    p["inputs"][0]["dtype"] = "bf16"
    with pytest.raises(ValueError, match="bad input spec"):
        validate_plan(p)
    p = copy.deepcopy(GOOD)
    p["inputs"][0]["shape"] = (2, 2)
    with pytest.raises(ValueError, match="bad input spec"):
        validate_plan(p)


def test_triton_step_ok_and_attention_arity():
    p = {
        "version": 1,
        "inputs": [
            {"name": "arg0", "shape": [1, 2, 8, 16], "dtype": "f32"},
            {"name": "arg1", "shape": [1, 2, 8, 16], "dtype": "f32"},
            {"name": "arg2", "shape": [1, 2, 8, 16], "dtype": "f32"},
        ],
        "steps": [{
            "kind": "triton", "kernel": "fp_attention", "launch": "attention",
            "operands": ["arg0", "arg1", "arg2"], "result": "v0",
            "shape": [1, 2, 8, 16], "dtype": "f32",
            "attrs": {"scale": 0.125, "block_m": 64, "block_n": 64, "num_warps": 4, "num_stages": 2},
        }],
        "outputs": ["v0"],
    }
    validate_plan(p)
    p["steps"][0]["operands"] = ["arg0", "arg1"]
    with pytest.raises(ValueError, match="attention needs exactly 3 operands"):
        validate_plan(p)


def test_bad_triton_step_missing_kernel_or_launch():
    p = copy.deepcopy(GOOD)
    p["steps"][0] = {
        "kind": "triton", "op": "relu", "operands": ["arg0"], "result": "v0",
        "shape": [2, 2], "dtype": "f32", "attrs": {},
    }
    with pytest.raises(ValueError, match="bad triton step"):
        validate_plan(p)


def test_summarize_and_estimate_bytes_handbuilt():
    unfused = {
        "version": 1,
        "inputs": [
            {"name": "arg0", "shape": [1, 2, 1024, 64], "dtype": "f16"},
            {"name": "arg1", "shape": [1, 2, 1024, 64], "dtype": "f16"},
            {"name": "arg2", "shape": [1, 2, 1024, 64], "dtype": "f16"},
        ],
        "steps": [
            {"kind": "torch", "op": "transpose", "operands": ["arg1"], "result": "v0",
             "shape": [1, 2, 64, 1024], "dtype": "f16", "attrs": {}},
            {"kind": "torch", "op": "matmul", "operands": ["arg0", "v0"], "result": "v1",
             "shape": [1, 2, 1024, 1024], "dtype": "f16", "attrs": {}},
            {"kind": "torch", "op": "scale", "operands": ["v1"], "result": "v2",
             "shape": [1, 2, 1024, 1024], "dtype": "f16", "attrs": {"factor": 0.125}},
            {"kind": "torch", "op": "softmax", "operands": ["v2"], "result": "v3",
             "shape": [1, 2, 1024, 1024], "dtype": "f16", "attrs": {"axis": 3}},
            {"kind": "torch", "op": "matmul", "operands": ["v3", "arg2"], "result": "v4",
             "shape": [1, 2, 1024, 64], "dtype": "f16", "attrs": {}},
        ],
        "outputs": ["v4"],
    }
    fused = {
        "version": 1,
        "inputs": unfused["inputs"],
        "steps": [{
            "kind": "triton", "kernel": "fp_attention", "launch": "attention",
            "operands": ["arg0", "arg1", "arg2"], "result": "v0",
            "shape": [1, 2, 1024, 64], "dtype": "f16",
            "attrs": {"scale": 0.125, "block_m": 64, "block_n": 64, "num_warps": 4, "num_stages": 2},
        }],
        "outputs": ["v0"],
    }
    validate_plan(unfused)
    validate_plan(fused)
    assert summarize_plan(unfused) == {"steps": 5, "triton_steps": 0, "torch_steps": 5}
    assert summarize_plan(fused) == {"steps": 1, "triton_steps": 1, "torch_steps": 0}
    assert estimate_bytes(fused) < estimate_bytes(unfused)
