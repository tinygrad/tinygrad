# Run assembly on the AMD runtime and check correctness
# VIZ=2 to profile
import pathlib
from tinygrad import Tensor, Device, dtypes, Context
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.engine.realize import Estimates
from tinygrad.helpers import getenv
import numpy as np

fp = pathlib.Path(__file__).parent/"gemm.s"

N = getenv("N", 4096)
THREADS_PER_WG = 256
NUM_WG = N//THREADS_PER_WG * N//THREADS_PER_WG

assert N % THREADS_PER_WG == 0, "N must be divisible by THREADS_PER_WG"

# ** assembly custom kernel

def custom_asm_gemm(C:UOp, A:UOp, B:UOp) -> UOp:
  lidx = UOp.special(THREADS_PER_WG, "lidx0")
  gidx = UOp.special(NUM_WG, "gidx0")

  src = (pathlib.Path(__file__).parent/"template.s").read_text().replace("INSTRUCTIONS", fp.read_text())

  sz = UOp.variable("SZ", 256, 8192)

  sink = UOp.sink(C.base, A.base, B.base, sz, lidx, gidx, arg=KernelInfo(name="gemm", estimates=Estimates(ops=N*N*N*2, mem=N*N*4*3)))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT), UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src)))

rng = np.random.default_rng(42)
A = Tensor(rng.random((N, N), dtype=np.float32) - 0.5, dtype=dtypes.half)
B = Tensor(rng.random((N, N), dtype=np.float32) - 0.5, dtype=dtypes.half)
C_asm = Tensor.empty(N, N, dtype=dtypes.half)
Tensor.realize(A, B)
C_asm = Tensor.custom_kernel(C_asm, A, B, fxn=custom_asm_gemm)[0]

sched = Tensor.schedule(C_asm)
eis = [si.lower() for si in sched]

with Context(DEBUG=2):
  for ei in eis:
    et = ei.run({"SZ":N}, wait=True)
    print(f"{(N*N*N*2 / et)*1e-12:.2f} REAL TFLOPS")

print(A.numpy())
