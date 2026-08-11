#include "fusionpass/Plan/PlanEmitter.h"
#include "fusionpass/Plan/TritonEmitter.h"
#include "fusionpass/Dialect/FP/FPOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"
using namespace mlir;
using namespace fusionpass::fp;
namespace json = llvm::json;

static LogicalResult dtypeStr(Type t, std::string &out) {
  if (t.isF16()) {
    out = "f16";
    return success();
  }
  if (t.isF32()) {
    out = "f32";
    return success();
  }
  return failure();
}

static LogicalResult shapeArr(RankedTensorType t, json::Array &out) {
  if (!t.hasStaticShape())
    return failure();
  for (int64_t d : t.getShape())
    out.push_back(d);
  return success();
}

static LogicalResult namedValue(DenseMap<Value, std::string> &names, Value v,
                                std::string &out, Operation *errAt,
                                const char *what) {
  auto it = names.find(v);
  if (it == names.end())
    return errAt->emitError("plan emitter: ") << what << " is not a named SSA value";
  out = it->second;
  return success();
}

static json::Object tritonStep(const std::string &kernel, const char *launch,
                               json::Array operands, const std::string &res,
                               json::Array shape, const std::string &dt,
                               json::Object attrs) {
  return json::Object{
      {"kind", "triton"},
      {"kernel", kernel},
      {"launch", launch},
      {"operands", std::move(operands)},
      {"result", res},
      {"shape", std::move(shape)},
      {"dtype", dt},
      {"attrs", std::move(attrs)},
  };
}

LogicalResult fusionpass::emitPlan(ModuleOp module,
                                   fusionpass::TritonModuleWriter &writer,
                                   json::Value &out) {
  auto fn = module.lookupSymbol<func::FuncOp>("main");
  if (!fn)
    return module.emitError("expected func.func @main");
  if (fn.getBody().empty() || !fn.getBody().hasOneBlock())
    return fn.emitError("func.func @main must have a single basic block");

  DenseMap<Value, std::string> names;
  json::Array inputs, steps, outputs;
  int next = 0;

  for (auto [i, arg] : llvm::enumerate(fn.getArguments())) {
    auto t = dyn_cast<RankedTensorType>(arg.getType());
    if (!t)
      return fn.emitError("arguments must be ranked tensors");
    std::string dt;
    json::Array shape;
    if (failed(dtypeStr(t.getElementType(), dt)) || failed(shapeArr(t, shape)))
      return fn.emitError("arguments must be static-shape f16/f32 tensors");
    std::string n = "arg" + std::to_string(i);
    names[arg] = n;
    inputs.push_back(json::Object{
        {"name", n},
        {"shape", std::move(shape)},
        {"dtype", dt},
    });
  }

  for (Operation &op : fn.getBody().front()) {
    if (auto ret = dyn_cast<func::ReturnOp>(op)) {
      for (Value v : ret.getOperands()) {
        std::string n;
        if (failed(namedValue(names, v, n, &op, "return operand")))
          return failure();
        outputs.push_back(n);
      }
      continue;
    }
    if (op.getNumResults() != 1)
      return op.emitError("unsupported: op must have exactly one result");

    json::Array operands;
    for (Value v : op.getOperands()) {
      std::string n;
      if (failed(namedValue(names, v, n, &op, "operand")))
        return failure();
      operands.push_back(n);
    }
    // Result names vN are assigned in program order for both torch and triton steps.
    std::string res = "v" + std::to_string(next++);
    names[op.getResult(0)] = res;
    auto rt = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!rt)
      return op.emitError("result must be a ranked tensor");
    std::string dt;
    json::Array shape;
    if (failed(dtypeStr(rt.getElementType(), dt)) || failed(shapeArr(rt, shape)))
      return op.emitError("result must be a static-shape f16/f32 tensor");

    if (auto fc = dyn_cast<FusedChainOp>(&op)) {
      if (fc.getBody().empty() || !fc.getBody().hasOneBlock())
        return op.emitError("fused_chain must have a single body block");
      std::string kernel = writer.addChainKernel(fc);
      bool rows = llvm::any_of(fc.getBody().front().without_terminator(),
                               [](Operation &o) { return isa<SoftmaxOp>(o); });
      json::Object a;
      if (rows) {
        uint64_t bn = llvm::PowerOf2Ceil((uint64_t)rt.getShape().back());
        a["block_n"] = (int64_t)bn;
        a["num_warps"] = (int64_t)(bn <= 2048 ? 4 : 8);
      } else {
        a["block"] = (int64_t)1024;
      }
      steps.push_back(tritonStep(kernel, rows ? "rows" : "flat", std::move(operands),
                                 res, std::move(shape), dt, std::move(a)));
      continue;
    }
    if (auto fa = dyn_cast<FusedAttentionOp>(&op)) {
      std::string kernel = writer.addAttentionKernel();
      json::Object a;
      a["scale"] = (double)fa.getScale().convertToFloat();
      a["block_m"] = (int64_t)64;
      a["block_n"] = (int64_t)64;
      a["num_warps"] = (int64_t)4;
      a["num_stages"] = (int64_t)2;
      steps.push_back(tritonStep(kernel, "attention", std::move(operands), res,
                                 std::move(shape), dt, std::move(a)));
      continue;
    }

    json::Object attrs;
    StringRef opName = op.getName().stripDialect();
    // WHY TypeSwitch: only remaining fp ops are legal torch steps; anything else is a hard error.
    bool ok = llvm::TypeSwitch<Operation *, bool>(&op)
        .Case<AddOp, MulOp, ReluOp, ExpOp, MatmulOp, TransposeOp>([&](auto) { return true; })
        .Case<ScaleOp>([&](ScaleOp s) {
          attrs["factor"] = (double)s.getFactor().convertToFloat();
          return true;
        })
        .Case<SoftmaxOp>([&](SoftmaxOp s) {
          attrs["axis"] = (int64_t)s.getAxis();
          return true;
        })
        .Default([](Operation *) { return false; });
    if (!ok)
      return op.emitError("plan emitter: unsupported op");

    steps.push_back(json::Object{
        {"kind", "torch"},
        {"op", opName.str()},
        {"operands", std::move(operands)},
        {"result", res},
        {"shape", std::move(shape)},
        {"dtype", dt},
        {"attrs", std::move(attrs)},
    });
  }
  out = json::Object{
      {"version", int64_t(1)},
      {"inputs", std::move(inputs)},
      {"steps", std::move(steps)},
      {"outputs", std::move(outputs)},
  };
  return success();
}
