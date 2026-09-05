#include "fusionpass/Plan/TritonEmitter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
using namespace mlir;
using namespace fusionpass;
using namespace fusionpass::fp;

static std::string pyFloat(llvm::APFloat f) {
  llvm::SmallString<32> s;
  f.toString(s);               // e.g. "0.125" or "1.0E-1"; both are valid Python float literals
  return s.str().str();
}

std::string TritonModuleWriter::addChainKernel(FusedChainOp op) {
  std::string name = "fp_chain_" + std::to_string(chainCount++);
  Block &body = op.getBody().front();
  bool rows = llvm::any_of(body.without_terminator(), [](Operation &o) { return isa<SoftmaxOp>(o); });

  std::string src;
  llvm::raw_string_ostream os(src);
  os << "@triton.jit\ndef " << name << "(";
  for (unsigned i = 0; i < body.getNumArguments(); ++i) os << "x" << i << ", ";
  os << "out, " << (rows ? "n_cols, BLOCK_N: tl.constexpr" : "n_elements, BLOCK: tl.constexpr") << "):\n";
  if (rows) {
    // One program per row; the whole row lives in registers (BLOCK_N = next_pow2(n_cols)).
    os << "    row = tl.program_id(0)\n"
          "    cols = tl.arange(0, BLOCK_N)\n"
          "    mask = cols < n_cols\n"
          "    offs = row.to(tl.int64) * n_cols + cols\n";
  } else {
    os << "    pid = tl.program_id(0)\n"
          "    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)\n"
          "    mask = offs < n_elements\n";
  }

  DenseMap<Value, std::string> v;
  for (unsigned i = 0; i < body.getNumArguments(); ++i) {
    v[body.getArgument(i)] = "a" + std::to_string(i);
    // WHY float32 compute: matches eager accuracy for fp16 inputs; cast back on store.
    os << "    a" << i << " = tl.load(x" << i << " + offs, mask=mask, other=0.0).to(tl.float32)\n";
  }
  int k = 0;
  for (Operation &o : body.without_terminator()) {
    std::string t = "t" + std::to_string(k++);
    auto in = [&](unsigned i) { return v[o.getOperand(i)]; };
    if (isa<AddOp>(o))       os << "    " << t << " = " << in(0) << " + " << in(1) << "\n";
    else if (isa<MulOp>(o))  os << "    " << t << " = " << in(0) << " * " << in(1) << "\n";
    else if (auto s = dyn_cast<ScaleOp>(o))
                             os << "    " << t << " = " << in(0) << " * " << pyFloat(s.getFactor()) << "\n";
    else if (isa<ReluOp>(o)) os << "    " << t << " = tl.maximum(" << in(0) << ", 0.0)\n";
    else if (isa<ExpOp>(o))  os << "    " << t << " = tl.exp(" << in(0) << ")\n";
    else if (isa<SoftmaxOp>(o)) {
      // Numerically stable softmax over the row; padded lanes forced to -inf so they contribute 0.
      os << "    " << t << "_m = tl.where(mask, " << in(0) << ", float(\"-inf\"))\n"
         << "    " << t << "_e = tl.exp(" << t << "_m - tl.max(" << t << "_m, axis=0))\n"
         << "    " << t << " = " << t << "_e / tl.sum(" << t << "_e, axis=0)\n";
    }
    v[o.getResult(0)] = t;
  }
  Value y = cast<YieldOp>(body.getTerminator()).getValue();
  os << "    tl.store(out + offs, " << v[y] << ".to(out.dtype.element_ty), mask=mask)\n\n\n";
  os.flush();
  chains += src;
  return name;
}

std::string TritonModuleWriter::addAttentionKernel() { hasAttention = true; return "fp_attention"; }

// FlashAttention-style tiled attention: one program per (BLOCK_M query rows, batch*head).
// K/V are streamed in BLOCK_N tiles; softmax is computed ONLINE (running max m_i and sum l_i),
// so the S x S score matrix is never written to global memory. Triton stages tiles through
// shared memory when lowering tl.dot (num_stages controls software pipelining).
static const char *kAttentionKernel = R"PY(@triton.jit
def fp_attention(Q, K, V, O, S, scale,
                 HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    base = pid_bh.to(tl.int64) * S * HEAD_DIM          # tensors are contiguous [BH, S, D]
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
                mask=offs_m[:, None] < S, other=0.0).to(tl.float32)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    for start_n in range(0, S, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        # K tile loaded already transposed: shape [HEAD_DIM, BLOCK_N]
        k = tl.load(K + base + offs_n[None, :] * HEAD_DIM + offs_d[:, None],
                    mask=offs_n[None, :] < S, other=0.0).to(tl.float32)
        s = tl.dot(q, k, input_precision="ieee") * scale   # [BLOCK_M, BLOCK_N] fp32 IEEE
        s = tl.where(offs_n[None, :] < S, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)                    # rescale factor for previous partial results
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(V + base + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
                    mask=offs_n[:, None] < S, other=0.0).to(tl.float32)
        acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(O + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
             acc.to(O.dtype.element_ty), mask=offs_m[:, None] < S)
)PY";

void TritonModuleWriter::write(llvm::raw_ostream &os) const {
  os << "# GENERATED by fusionpass-compile. Do not edit.\n"
        "import triton\nimport triton.language as tl\n\n\n";
  os << chains;
  if (hasAttention) os << kAttentionKernel;
}
