import pathlib
from tinygrad import Tensor, UOp, dtypes
from tinygrad.helpers import all_same
from tinygrad.engine.realize import Estimates
from tinygrad.uop.ops import Ops, KernelInfo

_stats = {"used":0, "called":0, "errs":[]}
def todo(msg:str=""): _stats["errs"].append(msg); return False

THREADS_PER_WG = 256

def can_use_asm_gemm(A:Tensor, B:Tensor) -> bool:
  _stats["called"] += 1
  if A.shape != B.shape: return todo()
  if not all_same(A.shape): return todo()
  if A.shape[0] % THREADS_PER_WG != 0: return todo()
  return True

def asm_gemm(A:Tensor, B:Tensor) -> Tensor:
  assert can_use_asm_gemm(A, B), f"{_stats['errs'][0]}"
  _stats["used"] += 1

  dname = A.device
  def custom_gemm_kernel(C:UOp, A:UOp, B:UOp, params:UOp) -> UOp:
    N = A.shape[0]
    lidx = UOp.special(THREADS_PER_WG, "lidx0")
    gidx = UOp.special(N//THREADS_PER_WG * N//THREADS_PER_WG, "gidx0")

    src = (pathlib.Path(__file__).parent/"template.s").read_text().replace("INSTRUCTIONS", (pathlib.Path(__file__).parent/"gemm.s").read_text())

    sink = UOp.sink(C.base, A.base, B.base, params.base, lidx, gidx, arg=KernelInfo(name="gemm", estimates=Estimates(ops=N*N*N*2, mem=N*N*4*3)))
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=dname), UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src)))
  
  params = Tensor.full((N, N), N).realize()
  C = Tensor.empty(N, N, dtype=dtypes.half)
  C = Tensor.custom_kernel(C, A, B.T, params, fxn=custom_gemm_kernel)[0]
  return C

if __name__ == "__main__":
  import numpy as np
  from tinygrad.helpers import getenv, Context
  N = getenv("N", 4096)

  rng = np.random.default_rng(0)
  A = Tensor(rng.random((N, N), dtype=np.float32) - 0.5, dtype=dtypes.half)
  B = Tensor(rng.random((N, N), dtype=np.float32) - 0.5, dtype=dtypes.half)
  C_asm = Tensor.empty(N, N, dtype=dtypes.half)
  Tensor.realize(A, B)
  C_asm = asm_gemm(A, B)
  C_tiny = A @ B
  C_asm.realize()

  np.testing.assert_allclose(C_asm.numpy(), C_tiny.numpy())
