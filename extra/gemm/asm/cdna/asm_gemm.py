import pathlib, atexit
from tinygrad import Tensor, Device, UOp, dtypes
from tinygrad.helpers import all_same, dedup
from tinygrad.engine.realize import Estimates
from tinygrad.uop.ops import Ops, KernelInfo
from tinygrad.runtime.support.compiler_amd import HIPCompiler
from extra.gemm.asm.cdna.gemm import insts

THREADS_PER_WG = 256

stats = {"used":0, "errs":[]}
def todo(msg:str="") -> bool: stats["errs"].append(msg); return False

def can_use_asm_gemm(A:Tensor, B:Tensor) -> bool:
  if A.dtype != dtypes.half or B.dtype != dtypes.half: return todo("only fp16")
  if A.shape != B.shape: return todo("matrices must be the same shape")
  if not all_same(A.shape): return todo("only supports square matrices")
  if A.shape[0] % THREADS_PER_WG != 0: return todo(f"N must be divisable by {THREADS_PER_WG}")
  return True

@atexit.register
def print_stats():
  print(f"ASM_GEMM=1: {stats['used']} used, {len(stats['errs'])} not used")
  if stats["errs"]:
    print("ASM_GEMM=1 unused reasons:")
    for e in dedup(stats["errs"]): print(f" {e}")

def asm_gemm(A:Tensor, B:Tensor) -> Tensor:
  assert can_use_asm_gemm(A, B), f"{stats['errs'][0]}"
  stats["used"] += 1

  N = A.shape[0]
  dname = A.device
  # may open the device to get arch
  arch = Device[dname].renderer.arch
  def custom_gemm_kernel(C:UOp, A:UOp, B:UOp, params:UOp) -> UOp:
    lidx = UOp.special(THREADS_PER_WG, "lidx0")
    gidx = UOp.special(N//THREADS_PER_WG * N//THREADS_PER_WG, "gidx0")

    insts_bytes = b"".join(inst.to_bytes() for inst in insts)
    insts_bytes_str = "\n".join("  .byte " + ",".join(f"0x{b:02x}" for b in insts_bytes[i:i+16]) for i in range(0, len(insts_bytes), 16)) + "\n"
    src = (pathlib.Path(__file__).parent/"template.s").read_text().replace("INSTRUCTIONS", insts_bytes_str)
    lib = HIPCompiler(arch).compile(src)

    sink = UOp.sink(C.base, A.base, B.base, params.base, lidx, gidx, arg=KernelInfo(name="gemm", estimates=Estimates(ops=N*N*N*2, mem=N*N*4*3)))
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=dname), UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src),
                                 UOp(Ops.BINARY, arg=lib)))

  params = Tensor.full((N, N), N).contiguous()
  Bt = B.T.contiguous()
  C = Tensor.empty(N, N, dtype=dtypes.half)
  C = Tensor.custom_kernel(C, A, Bt, params, fxn=custom_gemm_kernel)[0]
  return C

if __name__ == "__main__":
  import numpy as np
  from tinygrad.helpers import getenv, Context
  N = getenv("N", 4096)

  rng = np.random.default_rng(0)
  A = Tensor(rng.random((N, N), dtype=np.float32) - 0.5, dtype=dtypes.half)
  B = Tensor(rng.random((N, N), dtype=np.float32) - 0.5, dtype=dtypes.half)
  Tensor.realize(A, B)

  C_asm = asm_gemm(A, B)
  C_tiny = A @ B

  np.testing.assert_allclose(C_asm.numpy(), C_tiny.numpy())
