import pathlib

from tinygrad import Device, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo

from extra.gemm.amd_uop_matmul import test_matmul
from extra.gemm.asm.rdna4.gemm import insts

N = 4096
TN = 96
THREADS_PER_WG = 128
NUM_WG = 1024

dname:str = Device.DEFAULT
template:str = (pathlib.Path(__file__).parent/"template.s").read_text()

def insts_to_asm() -> str:
  lines = []
  for inst in insts:
    if isinstance(inst, str):
      lines.append(inst)
    else:
      lines.append(f"\t{inst.disasm()}")
  return "\n".join(lines)

def asm_kernel() -> UOp:
  lidx = UOp.special(THREADS_PER_WG, "lidx0")
  gidx = UOp.special(NUM_WG, "gidx0")

  a = UOp.placeholder((N*N,), dtypes.half, slot=1)
  b = UOp.placeholder((N*N,), dtypes.half, slot=2)
  c = UOp.placeholder((N*N,), dtypes.half, slot=0)

  src = template.replace("INSTRUCTIONS", insts_to_asm())

  sink = UOp.sink(a, b, c, lidx, gidx, arg=KernelInfo(name="gemm"))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=dname), UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src)), arg=())

if __name__ == "__main__":
  test_matmul(asm_kernel(), dtype=dtypes.half, N=N)
