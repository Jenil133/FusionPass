#include "fusionpass/Dialect/FP/FPDialect.h"
#include "fusionpass/Dialect/FP/FPOps.h"
using namespace fusionpass::fp;
#include "fusionpass/Dialect/FP/FPOpsDialect.cpp.inc"

void FPDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "fusionpass/Dialect/FP/FPOps.cpp.inc"
      >();
}
