from __future__ import annotations
import functools, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.renderer import Estimates
from extra.llama_kernels import FP8_MAX, NUM_WG, THREADS_PER_WG, alloc_like, alloc_local, scalar_amax, dname_of, compile_hip

@functools.cache
def _custom_quantize_fp8_with_amax(fp8_out:UOp, amax_partial:UOp, x:UOp, amax_state:UOp, dname:str) -> UOp:
  n_elems = 1
  for d in x.shape: n_elems *= d
  threads, workgroups = UOp.special(THREADS_PER_WG, "lidx0"), UOp.special(NUM_WG, "gidx0")
  mem = n_elems * 2 + n_elems + 4 + NUM_WG * 4
  sink = UOp.sink(fp8_out.base, amax_partial.base, x.base, amax_state.base, threads, workgroups,
                  arg=KernelInfo(f"quantize_fp8_with_amax_{n_elems}", estimates=Estimates(ops=3*n_elems, mem=mem)))
  src = (pathlib.Path(__file__).parent/"quantize_fp8_with_amax.cpp").read_text()
  defines = [f"-DN_ELEMS={n_elems}", f"-DNUM_WG={NUM_WG}", f"-DTHREADS_PER_WG={THREADS_PER_WG}"]
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=dname), UOp(Ops.LINEAR, src=(*sink.src, sink)),
                               UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=compile_hip(src, defines))))

@functools.cache
def _custom_quantize_fp8_scalar(fp8_out:UOp, x:UOp, amax_state:UOp, dname:str) -> UOp:
  n_elems = 1
  for d in x.shape: n_elems *= d
  threads, workgroups = UOp.special(THREADS_PER_WG, "lidx0"), UOp.special(NUM_WG, "gidx0")
  mem = n_elems * 2 + n_elems
  sink = UOp.sink(fp8_out.base, x.base, amax_state.base, threads, workgroups,
                  arg=KernelInfo(f"quantize_fp8_scalar_{n_elems}", estimates=Estimates(ops=2*n_elems, mem=mem)))
  src = (pathlib.Path(__file__).parent/"quantize_fp8_scalar.cpp").read_text()
  defines = [f"-DN_ELEMS={n_elems}", f"-DNUM_WG={NUM_WG}", f"-DTHREADS_PER_WG={THREADS_PER_WG}"]
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=dname), UOp(Ops.LINEAR, src=(*sink.src, sink)),
                               UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=compile_hip(src, defines))))

def _quantize_fp8_delayed_bwd(gradient:UOp, kernel:UOp):
  # NOTE: STE-equivalent backward — grad_x = grad_fp8 * scale, scale = FP8_MAX / amax_state.
  # `gradient` is bf16 grad w.r.t. fp8 output (asm_gemm bwd already applied x_scale).
  _, _, x, amax_state = kernel.src[1:]
  device = x.device
  scale = FP8_MAX / (Tensor(amax_state, device=device).float() + 1e-8)
  grad_x = (Tensor(gradient, device=device).float() * scale).cast(dtypes.bfloat16)
  return (None, None, grad_x.uop, None)

def quantize_fp8_delayed(x:Tensor, amax_state:Tensor, fp8_dtype=dtypes.fp8e4m3) -> tuple[Tensor, Tensor, Tensor, UOp]:
  # NOTE: one-pass bf16 -> fp8 quantize with delayed scaling. Returns (fp8, inv_scale, new_amax, store_effect).
  # Fused kernel reads x once and writes fp8 + per-WG |x| partials (then a small reduce produces scalar new_amax).
  # store_effect writes new_amax into amax_state's buffer — the caller must thread it into a realized
  # output via `.after(store_effect)`. Calling `amax_state.assign(new_amax)` inside a grad_fxn does
  # NOT work because .assign mutates only the temp Tensor's .uop, not the original layer-owned buffer.
  assert x.dtype == dtypes.bfloat16, f"expected bf16, got {x.dtype}"
  axis = x.uop.axis if isinstance(x.device, tuple) else None
  fp8_out      = alloc_like(x.shape,  fp8_dtype,      x.device, axis)
  amax_partial = alloc_local((NUM_WG,), dtypes.float32, x.device)
  fxn = functools.partial(_custom_quantize_fp8_with_amax, dname=dname_of(x.device))
  fp8_out, amax_partial, *_ = Tensor.custom_kernel(fp8_out, amax_partial, x, amax_state,
                                                    fxn=fxn, grad_fxn=_quantize_fp8_delayed_bwd)
  new_amax = scalar_amax(amax_partial)
  inv_scale = (amax_state.float() + 1e-8) / FP8_MAX
  store_effect = amax_state.uop.store(new_amax.uop)
  return fp8_out, inv_scale, new_amax, store_effect

def quantize_fp8_scalar(x:Tensor, amax_state:Tensor, fp8_dtype=dtypes.fp8e4m3) -> Tensor:
  # NOTE: pure one-pass bf16 -> fp8 quantize with delayed scalar scale. No amax computation.
  axis = x.uop.axis if isinstance(x.device, tuple) else None
  fp8_out = alloc_like(x.shape, fp8_dtype, x.device, axis)
  fxn = functools.partial(_custom_quantize_fp8_scalar, dname=dname_of(x.device))
  fp8_out, *_ = Tensor.custom_kernel(fp8_out, x, amax_state, fxn=fxn)
  return fp8_out
