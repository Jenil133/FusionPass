"""Phase 1 runtime tests: execute unfused torch plans without the C++ compiler."""

import pytest
import torch

from fusionpass.runtime import execute_plan


def _torch_step(op, operands, result, shape, dtype="f32", attrs=None):
    return {
        "kind": "torch",
        "op": op,
        "operands": operands,
        "result": result,
        "shape": shape,
        "dtype": dtype,
        "attrs": {} if attrs is None else attrs,
    }


def test_execute_elementwise_ops():
    a = torch.tensor([[-1.0, 2.0], [0.5, -3.0]])
    b = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    plan = {
        "version": 1,
        "inputs": [
            {"name": "arg0", "shape": [2, 2], "dtype": "f32"},
            {"name": "arg1", "shape": [2, 2], "dtype": "f32"},
        ],
        "steps": [
            _torch_step("add", ["arg0", "arg1"], "v0", [2, 2]),
            _torch_step("mul", ["v0", "arg1"], "v1", [2, 2]),
            _torch_step("scale", ["v1"], "v2", [2, 2], attrs={"factor": 0.5}),
            _torch_step("relu", ["v2"], "v3", [2, 2]),
            _torch_step("exp", ["v3"], "v4", [2, 2]),
        ],
        "outputs": ["v4"],
    }
    expected = torch.exp(torch.relu((a + b) * b * 0.5))
    torch.testing.assert_close(execute_plan(plan, [a, b]), expected)


def test_execute_attention_plan_matches_eager():
    q, k, v = (torch.randn(1, 2, 8, 16) for _ in range(3))
    plan = {
        "version": 1,
        "inputs": [
            {"name": "arg0", "shape": [1, 2, 8, 16], "dtype": "f32"},
            {"name": "arg1", "shape": [1, 2, 8, 16], "dtype": "f32"},
            {"name": "arg2", "shape": [1, 2, 8, 16], "dtype": "f32"},
        ],
        "steps": [
            _torch_step("transpose", ["arg1"], "v0", [1, 2, 16, 8]),
            _torch_step("matmul", ["arg0", "v0"], "v1", [1, 2, 8, 8]),
            _torch_step("scale", ["v1"], "v2", [1, 2, 8, 8], attrs={"factor": 0.25}),
            _torch_step("softmax", ["v2"], "v3", [1, 2, 8, 8], attrs={"axis": 3}),
            _torch_step("matmul", ["v3", "arg2"], "v4", [1, 2, 8, 16]),
        ],
        "outputs": ["v4"],
    }
    eager = torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * 0.25, dim=-1), v)
    torch.testing.assert_close(execute_plan(plan, [q, k, v]), eager)


def test_plan_version_mismatch():
    plan = {
        "version": 99,
        "inputs": [{"name": "arg0", "shape": [2, 2], "dtype": "f32"}],
        "steps": [_torch_step("relu", ["arg0"], "v0", [2, 2])],
        "outputs": ["v0"],
    }
    with pytest.raises(ValueError, match="unsupported plan version"):
        execute_plan(plan, [torch.randn(2, 2)])


def test_wrong_input_count():
    plan = {
        "version": 1,
        "inputs": [{"name": "arg0", "shape": [2, 2], "dtype": "f32"}],
        "steps": [_torch_step("relu", ["arg0"], "v0", [2, 2])],
        "outputs": ["v0"],
    }
    with pytest.raises(ValueError, match="expected 1 inputs"):
        execute_plan(plan, [])


def test_wrong_shape_at_call_time():
    plan = {
        "version": 1,
        "inputs": [{"name": "arg0", "shape": [2, 2], "dtype": "f32"}],
        "steps": [_torch_step("relu", ["arg0"], "v0", [2, 2])],
        "outputs": ["v0"],
    }
    with pytest.raises(ValueError, match="arg0"):
        execute_plan(plan, [torch.randn(3, 3)])


def test_wrong_dtype_at_call_time():
    plan = {
        "version": 1,
        "inputs": [{"name": "arg0", "shape": [2, 2], "dtype": "f32"}],
        "steps": [_torch_step("relu", ["arg0"], "v0", [2, 2])],
        "outputs": ["v0"],
    }
    with pytest.raises(ValueError, match="arg0"):
        execute_plan(plan, [torch.randn(2, 2, dtype=torch.float16)])


def test_unknown_torch_op():
    plan = {
        "version": 1,
        "inputs": [{"name": "arg0", "shape": [2, 2], "dtype": "f32"}],
        "steps": [_torch_step("tanh", ["arg0"], "v0", [2, 2])],
        "outputs": ["v0"],
    }
    with pytest.raises(ValueError, match="unknown torch op"):
        execute_plan(plan, [torch.randn(2, 2)])


def test_malformed_triton_step_rejected_by_validate_plan():
    plan = {
        "version": 1,
        "inputs": [{"name": "arg0", "shape": [2, 2], "dtype": "f32"}],
        "steps": [
            {
                "kind": "triton",
                "op": "relu",
                "operands": ["arg0"],
                "result": "v0",
                "shape": [2, 2],
                "dtype": "f32",
                "attrs": {},
            }
        ],
        "outputs": ["v0"],
    }
    with pytest.raises(ValueError, match="bad triton step"):
        execute_plan(plan, [torch.randn(2, 2)])
