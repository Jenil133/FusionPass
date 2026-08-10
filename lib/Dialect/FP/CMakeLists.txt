add_mlir_dialect_library(MLIRFPDialect
  FPDialect.cpp
  FPOps.cpp
  ADDITIONAL_HEADER_DIRS ${PROJECT_SOURCE_DIR}/include/fusionpass
  DEPENDS MLIRFPOpsIncGen
  LINK_LIBS PUBLIC MLIRIR MLIRSideEffectInterfaces MLIRInferTypeOpInterface)
