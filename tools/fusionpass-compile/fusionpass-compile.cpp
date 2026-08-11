#include "fusionpass/Dialect/FP/FPDialect.h"
#include "fusionpass/Plan/PlanEmitter.h"
#include "fusionpass/Plan/TritonEmitter.h"
#include "fusionpass/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input.mlir>"), cl::Required);
static cl::opt<std::string> outDir("o", cl::desc("output directory"), cl::Required);
static cl::opt<std::string> pipeline("pipeline", cl::desc("none | fuse"), cl::init("none"));
static cl::opt<bool> emitMlir("emit-mlir", cl::desc("also write optimized.mlir"), cl::init(true));

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "fusionpass compiler driver\n");
  MLIRContext ctx;
  ctx.loadDialect<func::FuncDialect, fusionpass::fp::FPDialect>();
  OwningOpRef<ModuleOp> module = parseSourceFile<ModuleOp>(inputFile, ParserConfig(&ctx));
  if (!module) return 1;
  if (failed(verify(*module))) return 1;

  if (pipeline == "fuse") {
    // Attention FIRST so its scale/softmax are absorbed before chain fusion.
    PassManager pm(&ctx);
    pm.addNestedPass<func::FuncOp>(fusionpass::createFuseAttentionPass());
    pm.addNestedPass<func::FuncOp>(fusionpass::createFuseChainsPass());
    if (failed(pm.run(*module))) return 1;
  } else if (pipeline != "none") {
    llvm::errs() << "error: unknown --pipeline '" << pipeline << "'\n";
    return 2;
  }

  fusionpass::TritonModuleWriter writer;
  llvm::json::Value plan(nullptr);
  if (failed(fusionpass::emitPlan(*module, writer, plan))) return 1;

  if (auto ec = llvm::sys::fs::create_directories(outDir)) {
    llvm::errs() << "cannot create " << outDir << ": " << ec.message() << "\n";
    return 1;
  }

  llvm::SmallString<256> planPath(outDir);
  llvm::sys::path::append(planPath, "plan.json");
  std::error_code ec;
  llvm::raw_fd_ostream os(planPath, ec);
  if (ec) { llvm::errs() << "cannot write plan.json: " << ec.message() << "\n"; return 1; }
  os << llvm::formatv("{0:2}\n", plan);
  os.flush();

  llvm::SmallString<256> kernelPath(outDir);
  llvm::sys::path::append(kernelPath, "kernels.py");
  std::error_code e2;
  llvm::raw_fd_ostream ks(kernelPath, e2);
  if (e2) { llvm::errs() << "cannot write kernels.py: " << e2.message() << "\n"; return 1; }
  writer.write(ks);

  if (emitMlir) {
    llvm::SmallString<256> mlirPath(outDir);
    llvm::sys::path::append(mlirPath, "optimized.mlir");
    std::error_code e3;
    llvm::raw_fd_ostream ms(mlirPath, e3);
    if (e3) { llvm::errs() << "cannot write optimized.mlir: " << e3.message() << "\n"; return 1; }
    module->print(ms);
  }
  return 0;
}
