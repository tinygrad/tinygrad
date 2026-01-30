import atexit, functools, pathlib
from tinygrad import Tensor, Device, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType
from tinygrad.renderer import Estimates
from tinygrad.helpers import getenv, Context

counters = {"used":0}
@atexit.register
def print_counters(): print("ASM gemm", counters)

# ** CDNA4 assembly gemm

WORKGROUP_SIZE = 256

def custom_asm_gemm(C:UOp, A:UOp, B:UOp, params:UOp, dname:str, wg:int) -> UOp:
  M, K = A.shape[0]*A.shape[1], A.shape[2]
  K2, N = B.shape[(1 if B.ndim == 3 else 0):]
  assert K == K2
  lidx = UOp.special(WORKGROUP_SIZE, "lidx0")
  gidx = UOp.special(wg, "gidx0")
  rodata = (pathlib.Path(__file__).parent/"rodata.s").read_text()
  src = rodata.replace("INSTRUCTIONS", (pathlib.Path(__file__).parent/"kernel.s").read_text())
  sink = UOp.sink(C.base, A.base, B.base, params.base, lidx, gidx,
                  arg=KernelInfo(name="gemm", estimates=Estimates(ops=2*M*N*K, mem=(M*K + K*N + M*N)*2)))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=dname), UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src)))

# gemm kernel arguments, mapped exactly to the dims, this should be removed the gemm is generic
# (batch, M, N, K) -> (numWG, iters, total)
GEMM_ARGS = {
  (1, 8192, 4096, 4096): (256, 64, 32768),
  (1, 8192, 1024, 4096): (256, 64, 32768),
  (1, 8192, 14336, 4096): (256, 64, 114688),
  (1, 8192, 4096, 14336): (256, 224, 114688),
  (1, 8192, 128256, 4096): (16032, 64, 1026048),
  (8, 8192, 1024, 4096): (256, 64, 65536),
  (1, 8192, 8192, 8192): (256, 128, 131072),
  (1, 4096, 4096, 4096): (256, 64, 16384),
}
ITERS_ARGS = {64: (67108864, 0), 128: (33554432, 0), 224: (613566757, 2147483656)}
def get_gemm_args(batch, M, N, K):
  numWG, iters, total = GEMM_ARGS[(batch, M, N, K)]
  magic, shift = ITERS_ARGS[iters]
  return [numWG, N, batch * M, 1, K, N, 0, N, 0, K, 0, iters, magic, shift, total]

def can_use_asm_gemm(a:Tensor, b:Tensor) -> bool:
  if a.dtype != b.dtype: return False
  if a.dtype not in {dtypes.bfloat16}: return False
  # only sharding on the batch is tested
  if isinstance(a.device, tuple) and not (a.ndim == 3 and a.uop.axis == 0 and b.uop.axis is None): return False
  batch, M, K = (1, *a.shape) if a.ndim == 2 else a.shape
  N = b.shape[1]
  if isinstance(a.device, tuple): batch //= len(a.device)
  if (batch, M, N, K) not in GEMM_ARGS: return False
  return True

# ** UOp gemm to test custom_kernel multi and backward correctness on non cdna4

def custom_uop_gemm(C:UOp, A:UOp, B:UOp) -> UOp:
  M, K = A.shape[0]*A.shape[1], A.shape[2]
  K2, N = B.shape[(1 if B.ndim == 3 else 0):]
  assert K == K2
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

  dname = a.device[0] if isinstance(a.device, tuple) else a.device
  arch = getattr(Device[dname].renderer, "arch", None)
  if arch.startswith("gfx950") and getenv("USE_ASM", 1):
    params = Tensor(p:=get_gemm_args(batch//len(a.device) if is_multi else batch, M, N, K))
    if is_multi: params.to_(a.device)
    # todo: remove this by converting the gemm to python dsl, these should be python constants
    with Context(DEBUG=0): params.realize()
    out = Tensor.custom_kernel(out, a, b, params, fxn=functools.partial(custom_asm_gemm, dname=dname, wg=p[0]), grad_fxn=fake_grad_fxn)[0]
  else:
    out = Tensor.custom_kernel(out, a, b, fxn=custom_uop_gemm, grad_fxn=fake_grad_fxn)[0]
  return out.squeeze(0) if squeeze else out
