# Run assembly on the AMD runtime and check correctness
# VIZ=2 to profile
import pathlib

from tinygrad import dtypes, Device
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.helpers import getenv

from extra.gemm.amd_uop_matmul import test_matmul

N = getenv("N", 8192)
THREADS_PER_WG = 256
NUM_WG = N//THREADS_PER_WG * N//THREADS_PER_WG

assert N % THREADS_PER_WG == 0, "N must be divisible by THREADS_PER_WG"

fp = pathlib.Path(__file__).parent/"gemm.s"
dname = Device.DEFAULT

def asm_kernel() -> UOp:
  lidx = UOp.special(THREADS_PER_WG, "lidx0")
  gidx = UOp.special(NUM_WG, "gidx0")

  a = UOp.placeholder((N*N,), dtypes.bfloat16, slot=1)
  b = UOp.placeholder((N*N,), dtypes.bfloat16, slot=2)
  c = UOp.placeholder((N*N,), dtypes.bfloat16, slot=0)

  sz = UOp.variable("SZ", 256, 8192)
  wg = UOp.variable("WG", 1, 1024)

  sink = UOp.sink(a, b, c, sz, wg, lidx, gidx, arg=KernelInfo(name="gemm"))
  src = (pathlib.Path(__file__).parent/"template.s").read_text().replace("INSTRUCTIONS", fp.read_text())
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=dname), UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src)))

if __name__ == "__main__":
  test_matmul(asm_kernel(), dtype=dtypes.bfloat16, N=N, fixedvars={"SZ":N, "WG":NUM_WG}, transpose_b=True)
