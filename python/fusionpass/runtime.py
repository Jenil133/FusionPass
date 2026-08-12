from __future__ import annotations
import torch

from .kernels import check_device as _check_device
from .kernels import launch_step, load_kernels
from .validate import validate_plan

_DTYPES = {"f32": torch.float32, "f16": torch.float16}

__all__ = [
    "execute_plan",
    "load_kernels",
    "validate_plan",
    "_run_torch",
    "_run_triton",
    "_check_device",
    "_DTYPES",
]


def _run_torch(step: dict, ops: list[torch.Tensor]) -> torch.Tensor:
    op, attrs = step["op"], step["attrs"]
    if op == "add": return torch.add(ops[0], ops[1])
    if op == "mul": return torch.mul(ops[0], ops[1])
    if op == "scale": return ops[0] * attrs["factor"]
    if op == "relu": return torch.relu(ops[0])
    if op == "exp": return torch.exp(ops[0])
    if op == "softmax": return torch.softmax(ops[0], dim=attrs["axis"])
    if op == "matmul": return torch.matmul(ops[0], ops[1])
    if op == "transpose": return ops[0].transpose(-1, -2)
    raise ValueError(f"unknown torch op in plan: {op}")


def _run_triton(step: dict, ops: list[torch.Tensor], kernels) -> torch.Tensor:
    return launch_step(step, ops, kernels)


def execute_plan(plan: dict, args: list[torch.Tensor], kernels=None, *, validate: bool = True):
    if validate:
        validate_plan(plan)
    if plan.get("version") != 1:
        raise ValueError(f"unsupported plan version {plan.get('version')}")
    if len(args) != len(plan["inputs"]):
        raise ValueError(f"expected {len(plan['inputs'])} inputs, got {len(args)}")
    env = {}
    for spec, t in zip(plan["inputs"], args):
        if list(t.shape) != spec["shape"] or t.dtype != _DTYPES[spec["dtype"]]:
            # WHY strict: the plan (and later, kernels) are specialized to these static shapes.
            raise ValueError(f"{spec['name']}: expected {spec['dtype']}{spec['shape']}, got {t.dtype}{list(t.shape)}")
        env[spec["name"]] = t
    for step in plan["steps"]:
        ops = [env[o] for o in step["operands"]]
        if step["kind"] == "torch":
            env[step["result"]] = _run_torch(step, ops)
        elif step["kind"] == "triton":
            env[step["result"]] = _run_triton(step, ops, kernels)
        else:
            raise ValueError(f"unknown step kind {step['kind']}")
    outs = [env[o] for o in plan["outputs"]]
    return outs[0] if len(outs) == 1 else tuple(outs)
