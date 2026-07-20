from __future__ import annotations
import functools, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates
from extra.llama_kernels import NUM_WG, THREADS_PER_WG, alloc_like, alloc_local, dname_of, compile_hip

def _src() -> str: return (pathlib.Path(__file__).parent/"fused_rmsnorm_mul_quantize_fp8.cpp").read_text()
def _src_bwd() -> str: return (pathlib.Path(__file__).parent/"fused_rmsnorm_mul_quantize_fp8_bwd.cpp").read_text()

@functools.cache
def _custom_fwd(fp8_out:UOp, x_normed_out:UOp, rrms_out:UOp, amax_out:UOp,
                x:UOp, weight:UOp, amax_state:UOp, layer_num:UOp|None=None, dname:str=None, eps_val:float=0.0) -> UOp:
  MBS, SEQ, HIDDEN = x.shape
  n_elems = MBS * SEQ * HIDDEN
  threads, workgroups = UOp.special(THREADS_PER_WG, "lidx0"), UOp.special(NUM_WG, "gidx0")
  mem = n_elems * 2 + n_elems + MBS * SEQ * 4 + n_elems + HIDDEN * 2 + 4 + 4
  sink = UOp.sink(fp8_out.base, x_normed_out.base, rrms_out.base, amax_out.base,
                  x.base, weight.base, amax_state.base,
                  *((layer_num.base,) if layer_num is not None else ()), threads, workgroups,
                  arg=KernelInfo(f"fused_rmsnorm_mul_quantize_fp8_{n_elems}_h{HIDDEN}_eps{eps_val:.0e}",
                                 estimates=Estimates(ops=6*n_elems, mem=mem)))
  defines = [f"-DN_ELEMS={n_elems}", f"-DHIDDEN={HIDDEN}", f"-DNUM_WG={NUM_WG}", f"-DTHREADS_PER_WG={THREADS_PER_WG}",
             f"-DEPS_LITERAL={eps_val}f"] + (["-DLAYER_SCALE=1"] if layer_num is not None else [])
  src = _src()
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)),
                               UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=compile_hip(src, defines))))

@functools.cache
def _custom_fwd_add(fp8_out:UOp, h_out:UOp, x_normed_out:UOp, rrms_out:UOp, amax_out:UOp,
                    x:UOp, residual:UOp, weight:UOp, amax_state:UOp, layer_num:UOp|None=None, dname:str=None, eps_val:float=0.0) -> UOp:
  MBS, SEQ, HIDDEN = x.shape
  n_elems = MBS * SEQ * HIDDEN
  threads, workgroups = UOp.special(THREADS_PER_WG, "lidx0"), UOp.special(NUM_WG, "gidx0")
  mem = n_elems * 2 * 4 + MBS * SEQ * 4 + HIDDEN * 2 + 4 + 4
  sink = UOp.sink(fp8_out.base, h_out.base, x_normed_out.base, rrms_out.base, amax_out.base,
                  x.base, residual.base, weight.base, amax_state.base,
                  *((layer_num.base,) if layer_num is not None else ()), threads, workgroups,
                  arg=KernelInfo(f"fused_add_rmsnorm_mul_quantize_fp8_{n_elems}_h{HIDDEN}_eps{eps_val:.0e}",
                                 estimates=Estimates(ops=7*n_elems, mem=mem)))
  defines = [f"-DN_ELEMS={n_elems}", f"-DHIDDEN={HIDDEN}", f"-DNUM_WG={NUM_WG}", f"-DTHREADS_PER_WG={THREADS_PER_WG}",
             f"-DEPS_LITERAL={eps_val}f", "-DHAS_RESIDUAL=1"] + (["-DLAYER_SCALE=1"] if layer_num is not None else [])
  src = _src()
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)),
                               UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=compile_hip(src, defines))))

@functools.cache
def _custom_bwd(grad_x:UOp, grad_weight_partial:UOp,
                grad_fp8:UOp, x_normed:UOp, rrms:UOp, weight:UOp, amax_state:UOp, *extra:UOp,
                has_h_grad:bool=False, has_layer_num:bool=False, dname:str=None) -> UOp:
  h_grad = extra[0] if has_h_grad else None
  layer_num = extra[int(has_h_grad)] if has_layer_num else None
  MBS, SEQ, HIDDEN = x_normed.shape
  n_elems = MBS * SEQ * HIDDEN
  threads, workgroups = UOp.special(THREADS_PER_WG, "lidx0"), UOp.special(NUM_WG, "gidx0")
  mem = n_elems * 2 * (3 + has_h_grad) + NUM_WG * HIDDEN * 4 + MBS * SEQ * 4 + HIDDEN * 2 + 4
  sink = UOp.sink(grad_x.base, grad_weight_partial.base,
                  grad_fp8.base, x_normed.base, rrms.base, weight.base, amax_state.base,
                  *((h_grad.base,) if h_grad is not None else ()),
                  *((layer_num.base,) if layer_num is not None else ()), threads, workgroups,
                  arg=KernelInfo(f"fused_rmsnorm_mul_quantize_fp8_bwd{'_add' if has_h_grad else ''}_{n_elems}_h{HIDDEN}",
                                 estimates=Estimates(ops=(8+has_h_grad)*n_elems, mem=mem)))
  defines = [f"-DN_ELEMS={n_elems}", f"-DHIDDEN={HIDDEN}", f"-DNUM_WG={NUM_WG}", f"-DTHREADS_PER_WG={THREADS_PER_WG}"]
  if has_h_grad: defines.append("-DHAS_H_GRAD=1")
  if layer_num is not None: defines.append("-DLAYER_SCALE=1")
  src = _src_bwd()
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)),
                               UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=compile_hip(src, defines))))

def _bwd_common(fp8_grad_u, h_grad_u, x_u, x_normed_u, rrms_u, weight_u, amax_state_u, layer_num_u, kernel:UOp):
  device = x_u.device
  MBS, SEQ, HIDDEN = x_normed_u.shape
  axis = x_normed_u.axis if isinstance(device, tuple) else None
  grad_x = alloc_like((MBS, SEQ, HIDDEN), dtypes.bfloat16, device, axis)
  grad_weight_partial = alloc_local((NUM_WG, HIDDEN), dtypes.float32, device, axis)
  grad_h_from_fp8 = None
  grad_weight_uop = None
  if fp8_grad_u is not None:
    h_grad_t = Tensor(h_grad_u, device=device).cast(dtypes.bfloat16) if h_grad_u is not None else None
    fxn = functools.partial(_custom_bwd, dname=dname_of(device), has_h_grad=h_grad_t is not None, has_layer_num=layer_num_u is not None)
    grad_x_t, grad_weight_partial_t, *_ = Tensor.custom_kernel(
      grad_x, grad_weight_partial,
      Tensor(fp8_grad_u, device=device).cast(dtypes.bfloat16),
      Tensor(x_normed_u.after(kernel), device=device),
      Tensor(rrms_u.after(kernel), device=device),
      Tensor(weight_u, device=device),
      Tensor(amax_state_u, device=device), *((h_grad_t,) if h_grad_t is not None else ()),
      *((Tensor(layer_num_u, device=device),) if layer_num_u is not None else ()), fxn=fxn)
    grad_h_from_fp8 = grad_x_t
    grad_weight_uop = grad_weight_partial_t.sum(axis=0).cast(dtypes.bfloat16).uop
  if h_grad_u is not None and grad_h_from_fp8 is None:
    h_grad_t = Tensor(h_grad_u, device=device).cast(dtypes.bfloat16)
    grad_total = h_grad_t
  else:
    grad_total = grad_h_from_fp8
  return grad_total.uop, grad_weight_uop

def _fused_bwd(gradient:UOp, kernel:UOp):
  # NOTE: fwd inputs (fp8_out, x_normed_out, rrms_out, amax_out, x, weight, amax_state)
  src = kernel.src[1:]
  _, x_normed_u, rrms_u, _, x_u, weight_u, amax_state_u = src[:7]
  layer_num_u = src[7] if len(src) > 7 else None
  grad_x, grad_w = _bwd_common(gradient, None, x_u, x_normed_u, rrms_u, weight_u, amax_state_u, layer_num_u, kernel)
  return (None, None, None, None, grad_x, grad_w, None) + ((None,) if layer_num_u is not None else ())

def _fused_add_bwd(*args, **kwargs):
  # Two invocation modes: 1 grad => positional; >1 grads => kwarg `call=`.
  # Outputs: (fp8_out, h_out, x_normed_out, rrms_out, amax_buf). Both fp8 and h may be consumed
  # downstream — TUPLE order in gradient.py preserves kernel-output slot order.
  # Don't dispatch by dtype: matmul's bwd emits fp8 grad as bf16 (no explicit cast), so
  # dtype-detection collapses both into h_grad and silently drops the rmsnorm-bwd path.
  if 'call' in kwargs:
    kernel, all_grads = kwargs['call'], list(args)
  else:
    gradient, kernel = args
    all_grads = [gradient]
  fp8_grad_u = h_grad_u = None
  if len(all_grads) >= 2:
    fp8_grad_u, h_grad_u = all_grads[0], all_grads[1]
  elif len(all_grads) == 1:
    g = all_grads[0]
    if g.dtype == dtypes.bfloat16: h_grad_u = g
    else: fp8_grad_u = g
  src = kernel.src[1:]
  _, _, x_normed_u, rrms_u, _, x_u, _, weight_u, amax_state_u = src[:9]
  layer_num_u = src[9] if len(src) > 9 else None
  grad_h, grad_w = _bwd_common(fp8_grad_u, h_grad_u, x_u, x_normed_u, rrms_u, weight_u, amax_state_u, layer_num_u, kernel)
  return (None, None, None, None, None, grad_h, grad_h, grad_w, None) + ((None,) if layer_num_u is not None else ())

def fused_rmsnorm_mul_quantize_fp8(x:Tensor, weight:Tensor, amax_state:Tensor, eps:float, fp8_dtype,
                                   amax_out:Tensor, layer_num:Tensor|None=None) -> tuple[Tensor, Tensor, Tensor]:
  # NOTE: rmsnorm(x) * weight -> fp8 + amax. Returns (fp8, x_normed, rrms).
  # x_normed + rrms are saved for the rmsnorm backward (also recomputed here from x regs).
  assert x.dtype == dtypes.bfloat16 and weight.dtype == dtypes.bfloat16
  assert x.shape[-1] == weight.shape[-1], f"HIDDEN mismatch: x={x.shape}, weight={weight.shape}"
  MBS, SEQ, HIDDEN = x.shape
  axis = x.uop.axis if isinstance(x.device, tuple) else None
  if isinstance(x.device, tuple): assert axis in (None, 0, 1), f"unsupported sharding axis={axis}"
  fp8_out      = alloc_like((MBS, SEQ, HIDDEN), fp8_dtype,       x.device, axis)
  x_normed_out = alloc_like((MBS, SEQ, HIDDEN), dtypes.bfloat16, x.device, axis)
  rrms_out     = alloc_like((MBS, SEQ),         dtypes.float32,  x.device, axis)
  fxn = functools.partial(_custom_fwd, dname=dname_of(x.device), eps_val=eps)
  fp8_out, x_normed_out, rrms_out, amax_out, *_ = Tensor.custom_kernel(
    fp8_out, x_normed_out, rrms_out, amax_out, x, weight, amax_state, *((layer_num,) if layer_num is not None else ()), fxn=fxn, grad_fxn=_fused_bwd)
  return fp8_out, x_normed_out, rrms_out

def fused_add_rmsnorm_mul_quantize_fp8(x:Tensor, residual:Tensor, weight:Tensor, amax_state:Tensor,
                                       eps:float, fp8_dtype, amax_out:Tensor,
                                       layer_num:Tensor|None=None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
  # NOTE: h = x + residual; y_normed = rmsnorm(h); fp8 = quantize(y_normed * weight).
  # Returns (fp8, h, x_normed, rrms). h is also written so downstream can
  # reuse it without recomputing x+residual — eliminates the separate residual-add kernel.
  assert x.dtype == dtypes.bfloat16 and residual.dtype == dtypes.bfloat16 and weight.dtype == dtypes.bfloat16
  assert x.shape == residual.shape
  MBS, SEQ, HIDDEN = x.shape
  axis = x.uop.axis if isinstance(x.device, tuple) else None
  if isinstance(x.device, tuple): assert axis in (None, 0, 1), f"unsupported sharding axis={axis}"
  fp8_out      = alloc_like((MBS, SEQ, HIDDEN), fp8_dtype,       x.device, axis)
  h_out        = alloc_like((MBS, SEQ, HIDDEN), dtypes.bfloat16, x.device, axis)
  x_normed_out = alloc_like((MBS, SEQ, HIDDEN), dtypes.bfloat16, x.device, axis)
  rrms_out     = alloc_like((MBS, SEQ),         dtypes.float32,  x.device, axis)
  fxn = functools.partial(_custom_fwd_add, dname=dname_of(x.device), eps_val=eps)
  fp8_out, h_out, x_normed_out, rrms_out, amax_out, *_ = Tensor.custom_kernel(
    fp8_out, h_out, x_normed_out, rrms_out, amax_out, x, residual, weight, amax_state, *((layer_num,) if layer_num is not None else ()),
    fxn=fxn, grad_fxn=_fused_add_bwd)
  return fp8_out, h_out, x_normed_out, rrms_out
