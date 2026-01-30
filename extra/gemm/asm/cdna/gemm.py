import atexit
from tinygrad import Tensor, Device, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType

counters = {"used":0}
@atexit.register
def print_counters(): print("ASM gemm", counters)

def can_use_asm_gemm(a:Tensor, b:Tensor) -> bool:
  if a.dtype != b.dtype: return False
  if a.dtype not in {dtypes.bfloat16}: return False
  # only sharding on the batch is tested
  if isinstance(a.device, tuple) and not (a.ndim == 3 and a.uop.axis == 0 and b.uop.axis is None): return False
  return True

# use UOp gemm to test custom_kernel multi and backward correctness on non cdna4
def custom_uop_gemm(C:UOp, A:UOp, B:UOp) -> UOp:
  M, K = A.shape[0]*A.shape[1], A.shape[2]
  N = B.shape[2 if B.ndim == 3 else 1]
  assert K == (B.shape[1] if B.ndim == 3 else B.shape[0])
  m = UOp.range(M, 1, AxisType.LOOP)
  n = UOp.range(N, 2, AxisType.LOOP)
  k = UOp.range(K, 0, AxisType.REDUCE)
  mul = (A.index((m*UOp.const(dtypes.index, K)+k))*B.index((k*UOp.const(dtypes.index, N)+n))).cast(dtypes.float32)
  red = mul.reduce(k, arg=Ops.ADD, dtype=dtypes.float32).cast(C.dtype.base)
  store = C.index((m*UOp.const(dtypes.index, N)+n), ptr=True).store(red).end(m, n)
  return store.sink(arg=KernelInfo(name=f'uop_gemm_{M}_{N}_{K}'))

def fake_grad_fxn(_:UOp, kernel:UOp): return (None,)*len(kernel.src)

def asm_gemm(a:Tensor, b:Tensor) -> Tensor:
  assert can_use_asm_gemm(a, b)
  counters["used"] += 1
  squeeze = a.ndim == 2
  if squeeze: a = a.unsqueeze(0)

  batch, M, K = a.shape
  N = b.shape[1]
  is_multi = isinstance(a.device, tuple)

  if is_multi:
    out = Tensor(Tensor.empty(batch//len(a.device), M, N, dtype=a.dtype, device=a.device).uop.multi(0), device=a.device)
  else:
    out = Tensor.empty(batch, M, N, dtype=a.dtype, device=a.device)

  out = Tensor.custom_kernel(out, a, b, fxn=custom_uop_gemm, grad_fxn=fake_grad_fxn)[0]
  return out.squeeze(0) if squeeze else out
