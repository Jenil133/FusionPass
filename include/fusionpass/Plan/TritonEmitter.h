#pragma once
#include <string>
#include "fusionpass/Dialect/FP/FPOps.h"
#include "llvm/Support/raw_ostream.h"

namespace fusionpass {
class TritonModuleWriter {
public:
  /// Appends a kernel for `op`; returns its name ("fp_chain_<N>").
  std::string addChainKernel(fp::FusedChainOp op);
  /// Ensures the shared FlashAttention-style kernel is present; returns "fp_attention".
  std::string addAttentionKernel();
  /// Writes the complete kernels.py source.
  void write(llvm::raw_ostream &os) const;
private:
  int chainCount = 0;
  bool hasAttention = false;
  std::string chains;   // concatenated chain-kernel sources
};
} // namespace fusionpass
