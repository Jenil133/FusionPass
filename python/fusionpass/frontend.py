from __future__ import annotations
import operator
import torch
import torch.fx as fx
import torch.nn.functional as F
from torch.fx.passes.shape_prop import ShapeProp


class UnsupportedOp(Exception):
    """Raised when the traced graph uses something outside the supported subset.
    WHY raise instead of falling back: a silent fallback would hide that the model was not compiled."""


_DT = {torch.float32: "f32", torch.float16: "f16"}
_MATMUL = {torch.matmul, operator.matmul}
_SOFTMAX = {torch.softmax, F.softmax}
_RELU = {torch.relu, F.relu}
_ADD = {operator.add, torch.add}
_MUL = {operator.mul, torch.mul}
_DIV = {operator.truediv, torch.div}


def _ty(shape, dtype) -> str:
    if dtype not in _DT:
        raise UnsupportedOp(f"unsupported dtype {dtype}")
    if len(shape) == 0:
        raise UnsupportedOp("rank-0 tensors are unsupported")
    return "tensor<" + "x".join(str(int(d)) for d in shape) + "x" + _DT[dtype] + ">"


def _f32(x: float) -> str:
    # WHY .9e: MLIR float literals require a '.', and repr(1e-05) has none.
    return f"{float(x):.9e}"


def _fetch_attr(root, target: str):
    obj = root
    for part in target.split("."):
        obj = getattr(obj, part)
    return obj


def trace_to_mlir(model: torch.nn.Module, example_inputs):
    """Returns (mlir_text, arg_tensors). arg_tensors = example inputs followed by captured
    parameters/buffers, in the same order as func.func @main's arguments."""
    gm = fx.symbolic_trace(model)
    ShapeProp(gm).propagate(*example_inputs)

    arg_nodes, arg_tensors = [], []
    ph = iter(example_inputs)
    for n in gm.graph.nodes:
        if n.op == "placeholder":
            arg_nodes.append(n); arg_tensors.append(next(ph))
    for n in gm.graph.nodes:
        if n.op == "get_attr":
            t = _fetch_attr(gm, n.target)
            if not isinstance(t, torch.Tensor):
                raise UnsupportedOp(f"get_attr {n.target} is not a tensor")
            arg_nodes.append(n); arg_tensors.append(t.detach())

    names, types = {}, {}
    for i, (n, t) in enumerate(zip(arg_nodes, arg_tensors)):
        names[n] = f"%arg{i}"
        types[n] = _ty(t.shape, t.dtype)

    def meta_ty(n):
        m = n.meta.get("tensor_meta")
        if m is None:
            raise UnsupportedOp(f"node {n.name} has no tensor shape info")
        return _ty(m.shape, m.dtype)

    lines, out_node = [], None
    for n in gm.graph.nodes:
        if n.op in ("placeholder", "get_attr"):
            continue
        if n.op == "output":
            out = n.args[0]
            if not isinstance(out, fx.Node):
                raise UnsupportedOp("exactly one tensor output is supported")
            out_node = out
            continue
        if n.op not in ("call_function", "call_method"):
            raise UnsupportedOp(f"node op {n.op} ({n.target}) is unsupported (call_module is not inlined)")
        names[n] = f"%{n.name}"
        types[n] = meta_ty(n)
        lines.append("    " + _emit(n, names, types))

    if out_node is None:
        raise UnsupportedOp("graph has no output")
    sig = ", ".join(f"%arg{i}: {types[n]}" for i, n in enumerate(arg_nodes))
    body = "\n".join(lines)
    text = (f"module {{\n  func.func @main({sig}) -> {types[out_node]} {{\n{body}\n"
            f"    return {names[out_node]} : {types[out_node]}\n  }}\n}}\n")
    return text, arg_tensors


def _shape(n: fx.Node):
    return tuple(n.meta["tensor_meta"].shape)


def _norm_axis(dim, rank):
    if dim is None:
        raise UnsupportedOp("softmax requires an explicit dim")
    if not -rank <= dim < rank:
        raise UnsupportedOp(f"softmax dim {dim} out of range")
    return dim % rank


def _emit(n: fx.Node, names, types) -> str:
    tgt, args, kwargs = n.target, n.args, n.kwargs
    res, rty = names[n], types[n]
    is_method = n.op == "call_method"

    def ref(x): return names[x]

    if (not is_method and tgt in _MATMUL) or (is_method and tgt == "matmul"):
        a, b = args[:2]
        if len(_shape(a)) != len(_shape(b)):
            raise UnsupportedOp("matmul with differing ranks (broadcast) is unsupported")
        return f"{res} = fp.matmul {ref(a)}, {ref(b)} : ({types[a]}, {types[b]}) -> {rty}"

    if (not is_method and tgt in _SOFTMAX) or (is_method and tgt == "softmax"):
        x = args[0]
        dim = kwargs.get("dim", args[1] if len(args) > 1 else None)
        if kwargs.get("dtype") is not None:
            raise UnsupportedOp("softmax with dtype= is unsupported")
        return f"{res} = fp.softmax {ref(x)} {{axis = {_norm_axis(dim, len(_shape(x)))} : i64}} : {rty}"

    if (not is_method and tgt in _RELU) or (is_method and tgt == "relu"):
        return f"{res} = fp.relu {ref(args[0])} : {rty}"

    if (not is_method and tgt is torch.exp) or (is_method and tgt == "exp"):
        return f"{res} = fp.exp {ref(args[0])} : {rty}"

    if is_method and tgt == "transpose":
        x, d0, d1 = args[:3]
        r = len(_shape(x))
        if {d0 % r, d1 % r} != {r - 2, r - 1}:
            raise UnsupportedOp("only transpose of the last two dims is supported")
        return f"{res} = fp.transpose {ref(x)} : {types[x]} -> {rty}"

    kind = None
    if not is_method:
        kind = "add" if tgt in _ADD else "mul" if tgt in _MUL else "div" if tgt in _DIV else None
    else:
        kind = {"add": "add", "mul": "mul", "div": "div", "__add__": "add", "__mul__": "mul", "__truediv__": "div"}.get(tgt)
    if kind is not None:
        a, b = args[:2]
        a_t, b_t = isinstance(a, fx.Node), isinstance(b, fx.Node)
        if a_t and b_t:
            if kind == "div":
                raise UnsupportedOp("tensor/tensor division is unsupported")
            if _shape(a) != _shape(b):
                raise UnsupportedOp(f"broadcasting is unsupported: {_shape(a)} vs {_shape(b)}")
            return f"{res} = fp.{kind} {ref(a)}, {ref(b)} : {rty}"
        if kind in ("mul", "div") and (a_t != b_t):
            x, c = (a, b) if a_t else (b, a)
            if kind == "div" and not a_t:
                raise UnsupportedOp("scalar / tensor is unsupported")
            if not isinstance(c, (int, float)):
                raise UnsupportedOp("scale factor must be a Python number")
            factor = float(c) if kind == "mul" else 1.0 / float(c)
            return f"{res} = fp.scale {ref(x)} {{factor = {_f32(factor)} : f32}} : {rty}"
        raise UnsupportedOp(f"{kind} with a Python scalar is unsupported (only mul/div by a constant)")

    raise UnsupportedOp(f"unsupported op: {n.op} {tgt}")
