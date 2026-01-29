import atexit
from tinygrad import Tensor, Device, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType

stats = {"used":0}
@atexit.register
def print_stats():
  print("ASM stats", stats)

def can_use_asm_gemm(a:Tensor, b:Tensor) -> bool:
  if a.dtype != b.dtype: return False
  if getattr(Device[a.device[0] if isinstance(a.device, tuple) else a.device].renderer, "arch", None) != "gfx950": return False
  if a.dtype not in {dtypes.bfloat16}: return False
  return True

def custom_gemm(C:UOp, A:UOp, B:UOp) -> UOp:
  assert A.shape[2] == B.shape[0]
  stats["used"] += 1
  M, K = A.shape[0]*A.shape[1], A.shape[2]
  N = B.shape[1]
  c2 = UOp.range(M, 1, AxisType.LOOP)
  c5 = UOp.range(N, 2, AxisType.LOOP)
  c11 = UOp.range(K, 0, AxisType.REDUCE)
  c18 = (A.index((c2*UOp.const(dtypes.index, K)+c11))*B.index((c11*UOp.const(dtypes.index, N)+c5))).cast(dtypes.float32)
  c19 = c18.reduce(c11, arg=Ops.ADD, dtype=dtypes.float32).cast(C.dtype.base)
  c21 = C.index((c2*UOp.const(dtypes.index, N)+c5), ptr=True).store(c19).end(c2, c5)
  return c21.sink(arg=KernelInfo(name=f'asm_gemm_{M}_{N}_{K}'))

def asm_gemm(a:Tensor, b:Tensor) -> Tensor:
  squeeze = a.ndim == 2
  if squeeze: a = a.unsqueeze(0)

  batch, M, K = a.shape
  N = b.shape[1]
  is_multi = isinstance(a.device, tuple)

  if is_multi:
    out = Tensor(Tensor.empty(batch//len(a.device)*M, N, dtype=a.dtype, device=a.device).uop.multi(0), device=a.device)
  else:
    out = Tensor.empty(batch*M, N, dtype=a.dtype, device=a.device)

  out = Tensor.custom_kernel(out, a, b, fxn=custom_gemm)[0]
  return out

if __name__ == "__main__":
  import numpy as np
  Tensor.manual_seed(0)
  dtype = dtypes.bfloat16

  a = Tensor.randn((8, 1024, 4096), dtype=dtypes.float).sub(0.5).cast(dtype).contiguous()
  b = Tensor.randn((4096, 1024), dtype=dtypes.float).sub(0.5).cast(dtype).contiguous()
  c = a @ b
  c.realize()

  devs = tuple(f"{Device.DEFAULT}:{i}" for i in range(8))
  a = Tensor.randn((8, 8192, 4096), dtype=dtypes.float).sub(0.5).cast(dtype).contiguous()
  b = Tensor.randn((4096, 1024), dtype=dtypes.float).sub(0.5).cast(dtype).contiguous()
  a = a.shard_(devs, axis=1)
  b = b.shard_(devs, axis=None)
  c = a @ b
  c.realize()
