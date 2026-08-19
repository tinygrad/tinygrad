from __future__ import annotations
import functools, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates
from extra.llama_kernels import alloc_like, alloc_local, compile_hip

RMS_MUL_FWD_GROUPS, RMS_MUL_BWD_GROUPS, RMS_MUL_THREADS = 2048, 1024, 128

def _source(name:str) -> str: return (pathlib.Path(__file__).parent/name).read_text()

@functools.cache
def _rmsnorm_mul_fwd(out:UOp, rrms:UOp, x:UOp, weight:UOp, eps:float) -> UOp:
  rows, hidden = x.numel() // x.shape[-1], x.shape[-1]
  threads, groups = UOp.special(RMS_MUL_THREADS, "lidx0"), UOp.special(RMS_MUL_FWD_GROUPS, "gidx0")
  sink = UOp.sink(out.base, rrms.base, x.base, weight.base, threads, groups,
                  arg=KernelInfo(f"rmsnorm_mul_fwd_{rows}_{hidden}", estimates=Estimates(ops=6*x.numel(), mem=4*x.numel()+4*rows)))
  source = _source("rmsnorm_mul.hip")
  defines = [f"-DROWS={rows}", f"-DHIDDEN={hidden}", f"-DNUM_WG={RMS_MUL_FWD_GROUPS}", f"-DTHREADS={RMS_MUL_THREADS}", f"-DEPS={eps}f"]
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=source),
                               UOp(Ops.BINARY, arg=compile_hip(source, defines))))

@functools.cache
def _rmsnorm_mul_fwd_mxfp4_row(out:UOp, rrms:UOp, x:UOp, weight:UOp, row_fp4:UOp, row_scale:UOp, eps:float) -> UOp:
  rows, hidden = x.numel() // x.shape[-1], x.shape[-1]
  threads, groups = UOp.special(RMS_MUL_THREADS, "lidx0"), UOp.special(RMS_MUL_FWD_GROUPS, "gidx0")
  mem = 4*x.numel() + 4*rows + x.numel()//2 + x.numel()//32
  sink = UOp.sink(out.base, rrms.base, x.base, weight.base, row_fp4.base, row_scale.base, threads, groups,
                  arg=KernelInfo(f"rmsnorm_mul_fwd_mxfp4_row_{rows}_{hidden}", estimates=Estimates(ops=18*x.numel(), mem=mem)))
  source = _source("rmsnorm_mul.hip")
  inc = pathlib.Path(__file__).parent.parent/"quantize_mxfp4"
  defines = [f"-DROWS={rows}", f"-DHIDDEN={hidden}", f"-DNUM_WG={RMS_MUL_FWD_GROUPS}", f"-DTHREADS={RMS_MUL_THREADS}",
             f"-DEPS={eps}f", "-DWRITE_MXFP4_ROW=1", f"-I{inc}"]
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=source),
                               UOp(Ops.BINARY, arg=compile_hip(source, defines))))

@functools.cache
def _rmsnorm_mul_bwd(dx:UOp, dweight_partial:UOp, dout:UOp, x:UOp, rrms:UOp, weight:UOp) -> UOp:
  rows, hidden = x.numel() // x.shape[-1], x.shape[-1]
  threads, groups = UOp.special(RMS_MUL_THREADS, "lidx0"), UOp.special(RMS_MUL_BWD_GROUPS, "gidx0")
  mem = 6*x.numel() + 4*RMS_MUL_BWD_GROUPS*hidden + 4*rows + 2*hidden
  sink = UOp.sink(dx.base, dweight_partial.base, dout.base, x.base, rrms.base, weight.base, threads, groups,
                  arg=KernelInfo(f"rmsnorm_mul_bwd_{rows}_{hidden}", estimates=Estimates(ops=10*x.numel(), mem=mem)))
  source = _source("rmsnorm_mul_bwd.hip")
  defines = [f"-DROWS={rows}", f"-DHIDDEN={hidden}", f"-DNUM_WG={RMS_MUL_BWD_GROUPS}", f"-DTHREADS={RMS_MUL_THREADS}"]
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=source),
                               UOp(Ops.BINARY, arg=compile_hip(source, defines))))

def _rmsnorm_mul_gradient(gradient:UOp, call:UOp) -> tuple:
  src = call.src[1:]
  _, rrms_u, x_u, weight_u = src[:4]
  axis = x_u.axis if isinstance(x_u.device, tuple) else None
  dx = alloc_like(x_u.shape, dtypes.bfloat16, x_u.device, axis)
  partial = alloc_local((RMS_MUL_BWD_GROUPS, x_u.shape[-1]), dtypes.float32, x_u.device, axis)
  dx, partial, *_ = Tensor.custom_kernel(dx, partial, Tensor(gradient, device=x_u.device).cast(dtypes.bfloat16),
                                          Tensor(x_u, device=x_u.device), Tensor(rrms_u.after(call), device=x_u.device),
                                          Tensor(weight_u, device=x_u.device), fxn=_rmsnorm_mul_bwd)
  return (None, None, dx.uop, partial.sum(axis=0).cast(dtypes.bfloat16).uop) + (None,) * (len(src)-4)

@functools.cache
def _rmsnorm_add_mul_fwd(out:UOp, h:UOp, rrms:UOp, x:UOp, residual:UOp, weight:UOp, eps:float) -> UOp:
  rows, hidden = x.numel() // x.shape[-1], x.shape[-1]
  threads, groups = UOp.special(RMS_MUL_THREADS, "lidx0"), UOp.special(RMS_MUL_FWD_GROUPS, "gidx0")
  sink = UOp.sink(out.base, h.base, rrms.base, x.base, residual.base, weight.base, threads, groups,
                  arg=KernelInfo(f"rmsnorm_add_mul_fwd_{rows}_{hidden}",
                                 estimates=Estimates(ops=7*x.numel(), mem=8*x.numel()+4*rows)))
  source = _source("rmsnorm_add_mul.hip")
  defines = [f"-DROWS={rows}", f"-DHIDDEN={hidden}", f"-DNUM_WG={RMS_MUL_FWD_GROUPS}", f"-DTHREADS={RMS_MUL_THREADS}", f"-DEPS={eps}f"]
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=source),
                               UOp(Ops.BINARY, arg=compile_hip(source, defines))))

@functools.cache
def _rmsnorm_add_mul_fwd_mxfp4_row(out:UOp, h:UOp, rrms:UOp, x:UOp, residual:UOp, weight:UOp,
                                    row_fp4:UOp, row_scale:UOp, eps:float) -> UOp:
  rows, hidden = x.numel() // x.shape[-1], x.shape[-1]
  threads, groups = UOp.special(RMS_MUL_THREADS, "lidx0"), UOp.special(RMS_MUL_FWD_GROUPS, "gidx0")
  mem = 8*x.numel() + 4*rows + x.numel()//2 + x.numel()//32
  sink = UOp.sink(out.base, h.base, rrms.base, x.base, residual.base, weight.base, row_fp4.base, row_scale.base, threads, groups,
                  arg=KernelInfo(f"rmsnorm_add_mul_fwd_mxfp4_row_{rows}_{hidden}", estimates=Estimates(ops=19*x.numel(), mem=mem)))
  source = _source("rmsnorm_add_mul.hip")
  inc = pathlib.Path(__file__).parent.parent/"quantize_mxfp4"
  defines = [f"-DROWS={rows}", f"-DHIDDEN={hidden}", f"-DNUM_WG={RMS_MUL_FWD_GROUPS}", f"-DTHREADS={RMS_MUL_THREADS}",
             f"-DEPS={eps}f", "-DWRITE_MXFP4_ROW=1", f"-I{inc}"]
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=source),
                               UOp(Ops.BINARY, arg=compile_hip(source, defines))))

@functools.cache
def _rmsnorm_add_mul_bwd(dh:UOp, dweight_partial:UOp, dout:UOp, dh_direct:UOp, h:UOp, rrms:UOp, weight:UOp) -> UOp:
  rows, hidden = h.numel() // h.shape[-1], h.shape[-1]
  threads, groups = UOp.special(RMS_MUL_THREADS, "lidx0"), UOp.special(RMS_MUL_BWD_GROUPS, "gidx0")
  mem = 8*h.numel() + 4*RMS_MUL_BWD_GROUPS*hidden + 4*rows + 2*hidden
  sink = UOp.sink(dh.base, dweight_partial.base, dout.base, dh_direct.base, h.base, rrms.base, weight.base, threads, groups,
                  arg=KernelInfo(f"rmsnorm_add_mul_bwd_{rows}_{hidden}", estimates=Estimates(ops=11*h.numel(), mem=mem)))
  source = _source("rmsnorm_add_mul_bwd.hip")
  defines = [f"-DROWS={rows}", f"-DHIDDEN={hidden}", f"-DNUM_WG={RMS_MUL_BWD_GROUPS}", f"-DTHREADS={RMS_MUL_THREADS}"]
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=source),
                               UOp(Ops.BINARY, arg=compile_hip(source, defines))))

@functools.cache
def _rmsnorm_add_mul_bwd_mxfp4_row(dh:UOp, dweight_partial:UOp, row_fp4:UOp, row_scale:UOp,
                                    dout:UOp, dh_direct:UOp, h:UOp, rrms:UOp, weight:UOp) -> UOp:
  rows, hidden = h.numel() // h.shape[-1], h.shape[-1]
  threads, groups = UOp.special(RMS_MUL_THREADS, "lidx0"), UOp.special(RMS_MUL_BWD_GROUPS, "gidx0")
  mem = 8*h.numel() + 4*RMS_MUL_BWD_GROUPS*hidden + 4*rows + 2*hidden + h.numel()//2 + h.numel()//32
  sink = UOp.sink(dh.base, dweight_partial.base, row_fp4.base, row_scale.base, dout.base, dh_direct.base,
                  h.base, rrms.base, weight.base, threads, groups,
                  arg=KernelInfo(f"rmsnorm_add_mul_bwd_mxfp4_row_{rows}_{hidden}", estimates=Estimates(ops=23*h.numel(), mem=mem)))
  source = _source("rmsnorm_add_mul_bwd.hip")
  inc = pathlib.Path(__file__).parent.parent/"quantize_mxfp4"
  defines = [f"-DROWS={rows}", f"-DHIDDEN={hidden}", f"-DNUM_WG={RMS_MUL_BWD_GROUPS}", f"-DTHREADS={RMS_MUL_THREADS}",
             "-DWRITE_MXFP4_ROW=1", f"-I{inc}"]
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=source),
                               UOp(Ops.BINARY, arg=compile_hip(source, defines))))

def _rmsnorm_add_mul_gradient(*args, **kwargs) -> tuple:
  if "call" in kwargs: call, grads = kwargs["call"], args
  else:
    gradient, call = args
    grads = (gradient,)
  assert len(grads) == 2, "rmsnorm_add_mul requires gradients for normalized output and residual state"
  dout, dh_direct = grads
  src = call.src[1:]
  _, h_u, rrms_u, x_u, residual_u, weight_u = src[:6]
  axis = x_u.axis if isinstance(x_u.device, tuple) else None
  dh = alloc_like(x_u.shape, dtypes.bfloat16, x_u.device, axis)
  partial = alloc_local((RMS_MUL_BWD_GROUPS, x_u.shape[-1]), dtypes.float32, x_u.device, axis)
  inputs = (Tensor(dout, device=x_u.device).cast(dtypes.bfloat16), Tensor(dh_direct, device=x_u.device).cast(dtypes.bfloat16),
            Tensor(h_u.after(call), device=x_u.device), Tensor(rrms_u.after(call), device=x_u.device), Tensor(weight_u, device=x_u.device))
  if len(src) > 6:
    from extra.llama_kernels.quantize_mxfp4 import alloc_mxfp4_row_outputs, _grad_mxfp4_mailbox
    row_fp4, row_scale = alloc_mxfp4_row_outputs(dh, flatten_row=True)
    dh, partial, row_fp4, row_scale, *_ = Tensor.custom_kernel(dh, partial, row_fp4, row_scale, *inputs,
                                                               fxn=_rmsnorm_add_mul_bwd_mxfp4_row)
    _grad_mxfp4_mailbox[dh.uop] = (row_fp4.uop, row_scale.uop, None, None)
  else: dh, partial, *_ = Tensor.custom_kernel(dh, partial, *inputs, fxn=_rmsnorm_add_mul_bwd)
  return (None, None, None, dh.uop, dh.uop, partial.sum(axis=0).cast(dtypes.bfloat16).uop) + (None,) * (len(src)-6)

def rmsnorm_mul(x:Tensor, weight:Tensor, eps:float) -> tuple[Tensor, Tensor]:
  assert x.dtype == weight.dtype == dtypes.bfloat16 and x.shape[-1] == weight.shape[-1] == 4096
  axis = x.uop.axis if isinstance(x.device, tuple) else None
  out = alloc_like(x.shape, dtypes.bfloat16, x.device, axis)
  rrms_axis = axis if axis is None or axis < x.ndim-1 else None
  rrms = alloc_like(x.shape[:-1], dtypes.float32, x.device, rrms_axis)
  out, rrms, *_ = Tensor.custom_kernel(out, rrms, x, weight,
                                        fxn=functools.partial(_rmsnorm_mul_fwd, eps=eps), grad_fxn=_rmsnorm_mul_gradient)
  return out, rrms

def rmsnorm_mul_mxfp4(x:Tensor, weight:Tensor, eps:float) -> tuple[Tensor, Tensor, tuple[Tensor|None, Tensor|None, Tensor|None, Tensor|None]]:
  assert x.dtype == weight.dtype == dtypes.bfloat16 and x.shape[-1] == weight.shape[-1] == 4096
  axis = x.uop.axis if isinstance(x.device, tuple) else None
  out = alloc_like(x.shape, dtypes.bfloat16, x.device, axis)
  rrms_axis = axis if axis is None or axis < x.ndim-1 else None
  rrms = alloc_like(x.shape[:-1], dtypes.float32, x.device, rrms_axis)
  from extra.llama_kernels.quantize_mxfp4 import alloc_mxfp4_row_outputs
  row_fp4, row_scale = alloc_mxfp4_row_outputs(out)
  ret = Tensor.custom_kernel(out, rrms, x, weight, row_fp4, row_scale,
                             fxn=functools.partial(_rmsnorm_mul_fwd_mxfp4_row, eps=eps), grad_fxn=_rmsnorm_mul_gradient)
  return ret[0], ret[1], (ret[4], ret[5], None, None)

def rmsnorm_add_mul(x:Tensor, residual:Tensor, weight:Tensor, eps:float) -> tuple[Tensor, Tensor, Tensor]:
  assert x.dtype == residual.dtype == weight.dtype == dtypes.bfloat16 and x.shape == residual.shape and x.shape[-1] == weight.shape[-1] == 4096
  axis = x.uop.axis if isinstance(x.device, tuple) else None
  out = alloc_like(x.shape, dtypes.bfloat16, x.device, axis)
  h = alloc_like(x.shape, dtypes.bfloat16, x.device, axis)
  rrms_axis = axis if axis is None or axis < x.ndim-1 else None
  rrms = alloc_like(x.shape[:-1], dtypes.float32, x.device, rrms_axis)
  out, h, rrms, *_ = Tensor.custom_kernel(out, h, rrms, x, residual, weight,
                                           fxn=functools.partial(_rmsnorm_add_mul_fwd, eps=eps), grad_fxn=_rmsnorm_add_mul_gradient)
  return out, h, rrms

def rmsnorm_add_mul_mxfp4(x:Tensor, residual:Tensor, weight:Tensor, eps:float) -> \
    tuple[Tensor, Tensor, Tensor, tuple[Tensor|None, Tensor|None, Tensor|None, Tensor|None]]:
  assert x.dtype == residual.dtype == weight.dtype == dtypes.bfloat16 and x.shape == residual.shape and x.shape[-1] == weight.shape[-1] == 4096
  axis = x.uop.axis if isinstance(x.device, tuple) else None
  out = alloc_like(x.shape, dtypes.bfloat16, x.device, axis)
  h = alloc_like(x.shape, dtypes.bfloat16, x.device, axis)
  rrms_axis = axis if axis is None or axis < x.ndim-1 else None
  rrms = alloc_like(x.shape[:-1], dtypes.float32, x.device, rrms_axis)
  from extra.llama_kernels.quantize_mxfp4 import alloc_mxfp4_row_outputs
  row_fp4, row_scale = alloc_mxfp4_row_outputs(out)
  ret = Tensor.custom_kernel(out, h, rrms, x, residual, weight, row_fp4, row_scale,
                             fxn=functools.partial(_rmsnorm_add_mul_fwd_mxfp4_row, eps=eps), grad_fxn=_rmsnorm_add_mul_gradient)
  return ret[0], ret[1], ret[2], (ret[6], ret[7], None, None)

def rmsnorm_fwd(x_in:Tensor, eps:float) -> tuple[Tensor, Tensor]:
  x = x_in.float()
  rrms = (x.square().mean(-1, keepdim=True) + eps).rsqrt()
  return (x * rrms).cast(x_in.dtype), rrms

@functools.cache
def _rmsnorm_fwd_fxn(x_in_p, eps, device):
  return rmsnorm_fwd(Tensor(x_in_p, device=device), eps)

def _rmsnorm_bwd(grad:UOp, call:UOp) -> tuple:
  x_normed = Tensor(call.gettuple(0)).float()
  do_float = Tensor(grad).float()
  d_x = Tensor(call.gettuple(1)) * (do_float - x_normed * (do_float * x_normed).mean(-1, keepdim=True))
  return (d_x.cast(call.src[1].dtype).uop,)

def rmsnorm(x_in:Tensor, eps:float) -> tuple[Tensor, Tensor]:
  fxn = _rmsnorm_fwd_fxn(x_in.as_param(0).uop, eps, x_in.device)
  call = UOp.maketuple(fxn[0].uop, fxn[1].uop).call(x_in.uop, grad_fxn=_rmsnorm_bwd)
  return Tensor(call.gettuple(0)), Tensor(call.gettuple(1))
