#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "fusionpass/Dialect/FP/FPDialect.h"
#include "fusionpass/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, fusionpass::fp::FPDialect>();
  fusionpass::registerFusionPasses();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "fusionpass optimizer driver\n", registry));
}
