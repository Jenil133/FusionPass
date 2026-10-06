from __future__ import annotations

from math import prod

_BYTES = {"f16": 2, "f32": 4}


def summarize_plan(plan: dict) -> dict:
    kinds = [s["kind"] for s in plan["steps"]]
    return {"steps": len(kinds), "triton_steps": kinds.count("triton"), "torch_steps": kinds.count("torch")}


def estimate_bytes(plan: dict) -> int:
    """ROUGH model of global-memory traffic: every step reads all its operands and writes its result once.
    Fused attention re-reads K and V once per query tile (ceil(S/block_m) times).
    This is an estimate for explaining WHY fusion helps, NOT a measurement (ignores caches, fusion
    inside PyTorch, in-place ops, and L2 reuse)."""
    size = {s["name"]: prod(s["shape"]) * _BYTES[s["dtype"]] for s in plan["inputs"]}
    total = 0
    for st in plan["steps"]:
        out_b = prod(st["shape"]) * _BYTES[st["dtype"]]
        size[st["result"]] = out_b
        if st.get("launch") == "attention":
            q, k, v = (size[o] for o in st["operands"])
            tiles = -(-st["shape"][-2] // st["attrs"]["block_m"])
            total += q + tiles * (k + v) + out_b
        else:
            total += sum(size[o] for o in st["operands"]) + out_b
    return total
