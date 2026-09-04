#include "fusionpass/Transforms/Passes.h"
#include "mlir/Pass/PassRegistry.h"

void fusionpass::registerFusionPasses() {
  // WHY registerPass(lambda): the pass's argument/description come from getArgument()/getDescription().
  mlir::registerPass([] { return createFuseAttentionPass(); });
  mlir::registerPass([] { return createFuseChainsPass(); });
}
