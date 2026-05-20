from __future__ import annotations
import functools
from tinygrad import Tensor, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType
from extra.llama_kernels import FP8_MAX, NUM_WG, THREADS_PER_WG, alloc_like, alloc_local, scalar_amax

@functools.cache
def _custom_quantize_fp8_with_amax(fp8_out:UOp, amax_partial:UOp, x:UOp, amax_state:UOp) -> UOp:
  VEC = 8
  n_elems = 1
  for d in x.shape: n_elems *= d
  num_wg = amax_partial.shape[0]
  threads = THREADS_PER_WG
  elems_per_thread = n_elems // (num_wg * threads * VEC)
  assert n_elems == num_wg * threads * VEC * elems_per_thread, f"{n_elems=} must divide over {num_wg=} {threads=} {VEC=}"
  scale = FP8_MAX / (amax_state[0].cast(dtypes.float) + 1e-8)

  wg = UOp.range(num_wg, 0, axis_type=AxisType.GLOBAL)
  tid = UOp.range(threads, 1, axis_type=AxisType.LOCAL)
  r = UOp.range(elems_per_thread, 2)

  local_max = UOp.placeholder((1,), dtypes.float, 0, addrspace=AddrSpace.REG)
  sdata = UOp.placeholder((threads,), dtypes.float, 0, addrspace=AddrSpace.LOCAL)
  init = local_max[0].store(0.0)
  local_max = local_max.after(init)

  base = r * (num_wg * threads * VEC) + (wg * threads + tid) * VEC
  x_vals = [x.reshape(n_elems)[base + k].cast(dtypes.float) for k in range(VEC)]
  fp8_store = UOp.group(*[fp8_out.reshape(n_elems)[base + k].store(
    (x_vals[k] * scale).maximum(-FP8_MAX).minimum(FP8_MAX).cast(fp8_out.dtype.base)) for k in range(VEC)])
  x0 = x_vals[0]
  vec_max = (x0 < 0).where(-x0, x0)
  for k in range(1, VEC):
    xk = x_vals[k]
    abs_xk = (xk < 0).where(-xk, xk)
    vec_max = vec_max.maximum(abs_xk)
  max_store = local_max[0].store(local_max.after(r)[0].maximum(vec_max))
  loop = UOp.group(fp8_store, max_store).end(r)

  local_max = local_max.after(loop)
  barrier = sdata.index(tid, ptr=True).store(local_max[0]).barrier()
  sdata_after = sdata.after(barrier)
  for s in [threads >> i for i in range(1, threads.bit_length())]:
    gate = tid < s
    red = sdata_after[tid].maximum(sdata_after[tid + s])
    barrier = sdata.index(tid.valid(gate), ptr=True).store(red).barrier()
    sdata_after = sdata.after(barrier)
  amax_store = amax_partial.index(wg.valid(tid.eq(0)), ptr=True).store(sdata_after[0])
  return UOp.sink(amax_store.end(tid, wg), arg=KernelInfo(f"quantize_fp8_with_amax_{n_elems}", opts_to_apply=()))

@functools.cache
def _custom_quantize_fp8_scalar(fp8_out:UOp, x:UOp, amax_state:UOp) -> UOp:
  n_elems = 1
  for d in x.shape: n_elems *= d
  scale = FP8_MAX / (amax_state[0].cast(dtypes.float) + 1e-8)

  i = UOp.range(n_elems, 0)
  scaled = x.reshape(n_elems)[i].cast(dtypes.float) * scale
  store = fp8_out.reshape(n_elems)[i].store(scaled.maximum(-FP8_MAX).minimum(FP8_MAX).cast(fp8_out.dtype.base)).end(i)
  return UOp.sink(store, arg=KernelInfo(f"quantize_fp8_scalar_{n_elems}"))

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
  amax_partial = alloc_local((NUM_WG,), dtypes.float32, x.device, axis)
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
