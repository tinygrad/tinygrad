import functools
from tinygrad import Tensor, dtypes
from tinygrad.helpers import prod
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType
from extra.llama_kernels import FP8_MAX, NUM_WG, THREADS_PER_WG, alloc_like, alloc_local, scalar_amax

@functools.cache
def _custom_quantize_fp8_with_amax(fp8_out:UOp, amax_partial:UOp, x:UOp, amax_state:UOp) -> UOp:
  n_elems = prod(x.shape)
  num_wg = amax_partial.shape[0]
  elems_per_wg = n_elems // num_wg
  assert n_elems == num_wg * elems_per_wg, f"{n_elems=} must divide over {num_wg=}"
  wg = UOp.range(num_wg, 0)
  i = UOp.range(elems_per_wg, 1)
  idx = wg * elems_per_wg + i

  scale = FP8_MAX / (amax_state[0].cast(dtypes.float) + 1e-8)
  x_f = x.reshape(n_elems)[idx].cast(dtypes.float)
  abs_x = (x_f < 0).where(-x_f, x_f)

  fp8_store = fp8_out.reshape(n_elems)[idx].store((x_f * scale).cast(fp8_out.dtype.base)).end(i)
  amax_store = amax_partial.after(fp8_store)[wg].store(abs_x.reduce(i, arg=Ops.MAX))
  return amax_store.end(wg).sink(arg=KernelInfo(f"quantize_fp8_with_amax_{n_elems}"))

@functools.cache
def _custom_quantize_fp8_scalar(fp8_out:UOp, x:UOp, amax_state:UOp) -> UOp:
  n_elems = prod(x.shape)
  i = UOp.range(n_elems, 0)

  x_f = x.reshape(n_elems)[i].cast(dtypes.float)
  scale = FP8_MAX / (amax_state[0].cast(dtypes.float) + 1e-8)
  store = fp8_out.reshape(n_elems)[i].store((x_f * scale).cast(fp8_out.dtype.base))

  return store.end(i).sink(arg=KernelInfo(f"quantize_fp8_scalar_{n_elems}"))

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
  fp8_out = alloc_like(x.shape, fp8_dtype, x.device, axis)
  n_elems = prod(x.uop.shard_shape)
  num_partials = n_elems // 4
  while n_elems % num_partials != 0: num_partials -= NUM_WG
  amax_partial = alloc_local((num_partials,), dtypes.float32, x.device, axis)
  fp8_out, amax_partial, *_ = Tensor.custom_kernel(fp8_out, amax_partial, x, amax_state,
                                                    fxn=_custom_quantize_fp8_with_amax, grad_fxn=_quantize_fp8_delayed_bwd)
  new_amax = scalar_amax(amax_partial)
  inv_scale = (amax_state.float() + 1e-8) / FP8_MAX
  store_effect = amax_state.uop.store(new_amax.uop)
  return fp8_out, inv_scale, new_amax, store_effect

def quantize_fp8_scalar(x:Tensor, amax_state:Tensor, fp8_dtype=dtypes.fp8e4m3) -> Tensor:
  # NOTE: pure one-pass bf16 -> fp8 quantize with delayed scalar scale. No amax computation.
  axis = x.uop.axis if isinstance(x.device, tuple) else None
  fp8_out = alloc_like(x.shape, fp8_dtype, x.device, axis)
  fp8_out, *_ = Tensor.custom_kernel(fp8_out, x, amax_state, fxn=_custom_quantize_fp8_scalar)
  return fp8_out
