"""Structural checks for plan.json so a corrupt/stale plan fails before kernels run."""

from __future__ import annotations

_DTYPES = {"f16", "f32"}
_TORCH_OPS = {"add", "mul", "scale", "relu", "exp", "softmax", "matmul", "transpose"}
_LAUNCHES = {"flat", "rows", "attention"}


def validate_plan(plan: dict) -> None:
    """Structural check so a corrupt/stale plan fails loudly BEFORE any kernel runs."""
    if plan.get("version") != 1:
        raise ValueError(f"unsupported plan version {plan.get('version')!r}")
    for key in ("inputs", "steps", "outputs"):
        if not isinstance(plan.get(key), list):
            raise ValueError(f"plan['{key}'] must be a list")
    defined = set()
    for spec in plan["inputs"]:
        if spec.get("dtype") not in _DTYPES or not isinstance(spec.get("shape"), list):
            raise ValueError(f"bad input spec: {spec}")
        defined.add(spec["name"])
    for i, st in enumerate(plan["steps"]):
        for o in st.get("operands", []):
            if o not in defined:
                raise ValueError(f"step {i}: operand '{o}' used before definition")
        res = st.get("result")
        if not res or res in defined:
            raise ValueError(f"step {i}: missing or duplicate result name '{res}'")
        if st.get("dtype") not in _DTYPES:
            raise ValueError(f"step {i}: bad dtype")
        if st.get("kind") == "torch":
            if st.get("op") not in _TORCH_OPS:
                raise ValueError(f"step {i}: unknown torch op '{st.get('op')}'")
        elif st.get("kind") == "triton":
            if st.get("launch") not in _LAUNCHES or not st.get("kernel"):
                raise ValueError(f"step {i}: bad triton step")
            if st["launch"] == "attention" and len(st["operands"]) != 3:
                raise ValueError(f"step {i}: attention needs exactly 3 operands")
        else:
            raise ValueError(f"step {i}: unknown kind '{st.get('kind')}'")
        defined.add(res)
    for o in plan["outputs"]:
        if o not in defined:
            raise ValueError(f"output '{o}' is never defined")
