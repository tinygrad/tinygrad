import pathlib

from tinygrad import Device, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo

from extra.gemm.amd_uop_matmul import N, test_matmul
from extra.gemm.asm.rdna4.gemm import gemm

THREADS_PER_WG = 128

# “2×2 wave-level tiling of a 128×128 workgroup C tile, where each wave computes a 64×64 sub-tile using WMMA.”
WAVE_BLOCK = 2
C_TILE = 64 * WAVE_BLOCK

TN = (N + C_TILE - 1) // C_TILE
NUM_WG =  TN * TN

assert N % THREADS_PER_WG == 0, "N must be divisible by THREADS_PER_WG"

dname:str = Device.DEFAULT
template:str = (pathlib.Path(__file__).parent/"template.s").read_text()

def asm_kernel() -> UOp:
  lidx = UOp.special(THREADS_PER_WG, "lidx0")
  gidx = UOp.special(NUM_WG, "gidx0")

  a = UOp.placeholder((N*N,), dtypes.half, slot=1)
  b = UOp.placeholder((N*N,), dtypes.half, slot=2)
  c = UOp.placeholder((N*N,), dtypes.half, slot=0)

  src = template.replace("INSTRUCTIONS", "\n".join([i if isinstance(i, str) else f"\t{i.disasm()}" for i in gemm(N)]))

  sink = UOp.sink(a, b, c, lidx, gidx, arg=KernelInfo(name="gemm"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=dname), UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src)), arg=())

if __name__ == "__main__":
  test_matmul(asm_kernel(), dtype=dtypes.half, N=N)
