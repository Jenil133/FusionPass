add_mlir_library(MLIRFPPlan
  PlanEmitter.cpp
  TritonEmitter.cpp
  DEPENDS MLIRFPOpsIncGen
  LINK_LIBS PUBLIC MLIRFPDialect MLIRFuncDialect MLIRIR LLVMSupport)
