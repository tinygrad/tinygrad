from __future__ import annotations
import functools, math, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates
from extra.gemm.cdna_asm_gemm import FP8_DTYPE
from extra.llama_kernels import NUM_WG, THREADS_PER_WG, alloc_like, alloc_local, compile_hip, dname_of

def rmsnorm_mul_fwd(x_in:Tensor, weight:Tensor, eps:float) -> tuple[Tensor, Tensor]:
  x = x_in.float()
  rrms = (x.square().mean(-1, keepdim=True) + eps).rsqrt()
  return ((x * rrms) * weight.float()).cast(x_in.dtype), rrms

@functools.cache
def _rmsnorm_mul_fwd_fxn(x_in_p, w_p, eps, device):
  return rmsnorm_mul_fwd(Tensor(x_in_p, device=device), Tensor(w_p, device=device), eps)

def _rmsnorm_mul_bwd(grad:UOp, call:UOp) -> tuple:
  x = Tensor(call.src[1]).float(); weight = Tensor(call.src[2]).float()
  rrms = Tensor(call.returned_outputs[1])
  x_normed = x * rrms                                  # recompute unweighted normed (x is call.src[1])
  d_y = Tensor(grad).float()
  dxn = d_y * weight                                   # d/d(x_normed)
  d_x = rrms * (dxn - x_normed * (dxn * x_normed).mean(-1, keepdim=True))
  dw = d_y * x_normed
  d_weight = dw.sum(axis=tuple(range(dw.ndim - 1)))    # reduce batch/seq -> [dim]
  return (d_x.cast(call.src[1].dtype).uop, d_weight.cast(call.src[2].dtype).uop)

def rmsnorm_mul(x_in:Tensor, weight:Tensor, eps:float) -> tuple[Tensor, Tensor]:
  fxn = _rmsnorm_mul_fwd_fxn(x_in.as_param(0).uop, weight.as_param(1).uop, eps, x_in.device)
  call = UOp.call_outputs((fxn[0].uop, fxn[1].uop), x_in.uop, weight.uop, grad_fxn=_rmsnorm_mul_bwd)
  return Tensor(call.returned_outputs[0]), Tensor(call.returned_outputs[1])

@functools.cache
def _custom_rmsnorm_mul_quantize_mxfp8_fwd(q:UOp, e8:UOp, rrms:UOp, x:UOp, weight:UOp, *, dname:str, eps:float) -> UOp:
  *lead, hidden = x.shape
  rows, padded = math.prod(lead), q.shape[-1]
  num_wg = min(NUM_WG, rows)
  threads, workgroups = UOp.special(THREADS_PER_WG, "lidx0"), UOp.special(num_wg, "gidx0")
  sink = UOp.sink(q.base, e8.base, rrms.base, x.base, weight.base, threads, workgroups,
                  arg=KernelInfo(f"rmsnorm_mul_quantize_mxfp8_{rows}_{hidden}_{padded}",
                                 estimates=Estimates(ops=8*rows*hidden, mem=rows*(hidden*2+padded+padded//32+4)+hidden*2)))
  src = (pathlib.Path(__file__).parent/"rmsnorm_mul_quantize_mxfp8.cpp").read_text()
  defines = [f"-DN_ELEMS={rows*hidden}", f"-DHIDDEN={hidden}", f"-DPADDED={padded}",
             f"-DNUM_WG={num_wg}", f"-DTHREADS_PER_WG={THREADS_PER_WG}", f"-DEPS_LITERAL={eps}f"]
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src),
                               UOp(Ops.BINARY, arg=compile_hip(src, defines))))

@functools.cache
def _custom_rmsnorm_mul_quantize_mxfp8_bwd(grad_x:UOp, grad_weight_partial:UOp, grad_q:UOp, x:UOp, weight:UOp, e8:UOp, rrms:UOp,
                *, dname:str) -> UOp:
  *lead, hidden = x.shape
  rows, padded = math.prod(lead), grad_q.shape[-1]
  num_wg = min(NUM_WG, rows)
  threads, workgroups = UOp.special(THREADS_PER_WG, "lidx0"), UOp.special(num_wg, "gidx0")
  sink = UOp.sink(grad_x.base, grad_weight_partial.base, grad_q.base, x.base, weight.base, e8.base, rrms.base,
                  threads, workgroups,
                  arg=KernelInfo(f"rmsnorm_mul_quantize_mxfp8_bwd_{rows}_{hidden}_{padded}",
                                 estimates=Estimates(ops=10*rows*hidden, mem=rows*(hidden*6+padded*2+padded//32+4)+num_wg*hidden*4)))
  src = (pathlib.Path(__file__).parent/"rmsnorm_mul_quantize_mxfp8_bwd.cpp").read_text()
  defines = [f"-DN_ELEMS={rows*hidden}", f"-DHIDDEN={hidden}", f"-DPADDED={padded}", f"-DNUM_WG={num_wg}", f"-DTHREADS_PER_WG={THREADS_PER_WG}"]
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src),
                               UOp(Ops.BINARY, arg=compile_hip(src, defines))))

def _rmsnorm_mul_quantize_mxfp8_backward(gradient:UOp, kernel:UOp) -> tuple:
  _, e8_u, rrms_u, x_u, weight_u = kernel.src[1:]
  device = x_u.device
  axis = x_u.axis if isinstance(device, tuple) else None
  *lead, hidden = x_u.shape
  num_wg = min(NUM_WG, math.prod(lead))
  grad_x = alloc_like(x_u.shape, x_u.dtype, device, axis)
  grad_weight_partial = alloc_local((num_wg, hidden), dtypes.float32, device, axis)
  grad_q = Tensor(gradient, device=device).cast(dtypes.bfloat16).contiguous()
  grad_x, grad_weight_partial, *_ = Tensor.custom_kernel(
    grad_x, grad_weight_partial, grad_q, Tensor(x_u, device=device), Tensor(weight_u, device=device),
    Tensor(e8_u.after(kernel), device=device), Tensor(rrms_u.after(kernel), device=device),
    fxn=functools.partial(_custom_rmsnorm_mul_quantize_mxfp8_bwd, dname=dname_of(device)))
  grad_weight = grad_weight_partial.sum(0).cast(weight_u.dtype)
  return None, None, None, grad_x.uop, grad_weight.uop

def rmsnorm_mul_quantize_mxfp8(x:Tensor, weight:Tensor, eps:float, padded:int|None=None) -> tuple[Tensor, Tensor, Tensor]:
  """RMSNorm(x)*weight directly to rowwise MXFP8. Returns (q, e8, rrms), without a BF16 normalized round-trip."""
  assert x.dtype == weight.dtype == dtypes.bfloat16 and x.shape[-1] == weight.shape[0], f"{x.shape=} {weight.shape=}"
  hidden = x.shape[-1]
  padded = math.ceil(hidden / 256) * 256 if padded is None else padded
  assert padded >= hidden and padded % 256 == 0 and hidden % 32 == 0
  axis = x.uop.axis if isinstance(x.device, tuple) else None
  q = alloc_like((*x.shape[:-1], padded), FP8_DTYPE, x.device, axis)
  e8 = alloc_like((*x.shape[:-1], padded // 32), dtypes.uint8, x.device, axis)
  rrms = alloc_like((*x.shape[:-1], 1), dtypes.float32, x.device, axis)
  q, e8, rrms, *_ = Tensor.custom_kernel(q, e8, rrms, x, weight,
                                         fxn=functools.partial(_custom_rmsnorm_mul_quantize_mxfp8_fwd, dname=dname_of(x.device), eps=eps),
                                         grad_fxn=_rmsnorm_mul_quantize_mxfp8_backward)
  return q, e8, rrms
