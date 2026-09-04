add_mlir_library(MLIRFPTransforms
  FuseAttention.cpp
  FuseChains.cpp
  Passes.cpp
  DEPENDS MLIRFPOpsIncGen
  LINK_LIBS PUBLIC MLIRFPDialect MLIRFuncDialect MLIRIR MLIRPass MLIRTransformUtils)
