import pathlib
import numpy as np
from tinygrad import Device, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo

from extra.assembly.amd.autogen.rdna4.ins import *
from extra.assembly.amd.dsl import RawImm

from extra.gemm.amd_uop_matmul import N, test_matmul

# “2×2 wave-level tiling of a 128×128 workgroup C tile, where each wave computes a 64×64 sub-tile using WMMA.”
N = 4096
WAVE_BLOCK = 2
C_TILE = 64 * WAVE_BLOCK
TN = (N + C_TILE - 1) // C_TILE
NUM_WG = TN * TN
THREADS_PER_WG = 128
assert N % THREADS_PER_WG == 0, "N must be divisible by THREADS_PER_WG"

class Kernel:
  def __init__(self): self.insts:list[str] = []
  def __iadd__(self, inst): self.insts.append(inst if isinstance(inst, str) else "  "+inst.disasm()); return self
  def __str__(self) -> str: return (pathlib.Path(__file__).parent/"template.s").read_text().replace("INSTRUCTIONS", "\n".join(self.insts))

def custom_gemm(dev:str) -> UOp:
  lidx = UOp.special(THREADS_PER_WG, "lidx0")
  gidx = UOp.special(NUM_WG, "gidx0")
  k = Kernel()
  k+= s_mov_b32(s[10], 0x1)
  k+= s_endpgm()
  a = UOp.placeholder((N*N,), dtypes.half, slot=1)
  b = UOp.placeholder((N*N,), dtypes.half, slot=2)
  c = UOp.placeholder((N*N,), dtypes.half, slot=0)
  sink = UOp.sink(a, b, c, lidx, gidx, arg=KernelInfo(name="gemm"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=dev), UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=str(k))), arg=())

if __name__ == "__main__":
  test_matmul(custom_gemm(Device.DEFAULT), dtype=dtypes.half, N=N)
