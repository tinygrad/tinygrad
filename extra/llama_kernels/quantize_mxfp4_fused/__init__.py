import functools, math
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType
from extra.llama_kernels import THREADS_PER_WG, alloc_like

BLK, PACK = 32, 4
LOG2E = 1.4426950408889634

def _e2m1_code(x:UOp) -> UOp:
  mag = x.abs()
  code = ((mag > .25).cast(dtypes.uint8) + (mag >= .75).cast(dtypes.uint8) +
          (mag > 1.25).cast(dtypes.uint8) + (mag >= 1.75).cast(dtypes.uint8) +
          (mag > 2.5).cast(dtypes.uint8) + (mag >= 3.5).cast(dtypes.uint8) +
          (mag > 5.0).cast(dtypes.uint8))
  return code | ((x < 0).cast(dtypes.uint8) << 3)

@functools.cache
def _custom_quantize_mxfp4(packed_out:UOp, e8_out:UOp, x:UOp) -> UOp:
  n_elems = math.prod(x.shape)
  n_super = n_elems // (BLK * PACK)
  threads_per_wg = min(n_super, THREADS_PER_WG)
  assert n_super % threads_per_wg == 0, f"{n_super=} must divide over {threads_per_wg=}"

  x, packed_out, e8_out = x.reshape(n_elems), packed_out.reshape(n_elems//2), e8_out.reshape(n_elems//BLK)
  wg = UOp.range(n_super // threads_per_wg, 0, AxisType.GLOBAL)
  tid = UOp.range(threads_per_wg, 1, AxisType.LOCAL)
  sb = UOp.range(PACK, 2, AxisType.UNROLL)
  lane = UOp.range(BLK//2, 3, AxisType.UNROLL)
  block = (wg * threads_per_wg + tid) * PACK + sb
  idx = block * BLK + lane * 2

  x0, x1 = x[idx].cast(dtypes.float), x[idx+1].cast(dtypes.float)
  pair_max = x0.abs().maximum(x1.abs())
  amax = pair_max.reduce(lane, arg=Ops.MAX)
  amax_rounded = ((amax.bitcast(dtypes.uint32) + 0x200000) & 0xFF800000).bitcast(dtypes.float)
  scale_exp = amax_rounded.maximum(2**-126).log2().floor().sub(2).maximum(-127).minimum(127)
  qscale = (-scale_exp).exp2()
  packed = _e2m1_code(x0 * qscale) | (_e2m1_code(x1 * qscale) << 4)

  packed_store = packed_out[block * (BLK//2) + lane].store(packed).end(lane)
  e8_store = e8_out.after(packed_store)[block].store((scale_exp + 127).cast(dtypes.uint8))
  return e8_store.end(sb, tid, wg).sink(arg=KernelInfo(f"quantize_mxfp4_{n_elems}", opts_to_apply=()))

def quantize_mxfp4_fused(x:Tensor, packed_shape:tuple[int, ...]|None=None, packed_axis:int|None=None) -> tuple[Tensor, Tensor]:
  assert x.dtype == dtypes.bfloat16 and x.ndim >= 2, f"expected bf16 with ndim >= 2, got {x.dtype} {x.shape}"
  *batch, K = x.shape
  rows = math.prod(batch)
  assert K % 256 == 0 and rows % 32 == 0, f"mxfp4 quantization needs rows%32 and K%256, got {x.shape}"
  axis = x.uop.axis if isinstance(x.device, tuple) else None
  packed = alloc_like(packed_shape or (*batch, K//2), dtypes.uint8, x.device, packed_axis if packed_shape is not None else axis)
  e8 = alloc_like((*batch, K//BLK), dtypes.uint8, x.device, axis)
  packed, e8, *_ = Tensor.custom_kernel(packed, e8, x, fxn=_custom_quantize_mxfp4)
  return packed, e8

@functools.cache
def _custom_silu_mul_quantize_mxfp4(act_out:UOp, packed_out:UOp, scale_out:UOp, x_w13:UOp) -> UOp:
  *prefix, two_k = x_w13.shape
  rows, K = math.prod(prefix), two_k//2
  n_elems, n_super = rows*K, rows*K//(BLK*PACK)
  threads_per_wg = min(n_super, THREADS_PER_WG)
  assert n_super % threads_per_wg == 0, f"{n_super=} must divide over {threads_per_wg=}"

  x_w13, act_out = x_w13.reshape(rows*two_k), act_out.reshape(n_elems)
  packed_out, scale_out = packed_out.reshape(n_elems//2), scale_out.reshape(n_elems//BLK)
  wg = UOp.range(n_super//threads_per_wg, 0, AxisType.GLOBAL)
  tid = UOp.range(threads_per_wg, 1, AxisType.LOCAL)
  sb = UOp.range(PACK, 2, AxisType.UNROLL)
  lane = UOp.range(BLK//2, 3, AxisType.UNROLL)
  block = (wg*threads_per_wg + tid)*PACK + sb
  out_idx = block*BLK + lane*2
  row, col = out_idx//K, out_idx%K

  w10, w11 = x_w13[row*two_k+col], x_w13[row*two_k+col+1]
  w30, w31 = x_w13[row*two_k+K+col], x_w13[row*two_k+K+col+1]
  act0 = w10 * (1.0 + (w10*-LOG2E).exp2()).reciprocal() * w30
  act1 = w11 * (1.0 + (w11*-LOG2E).exp2()).reciprocal() * w31
  a0, a1 = act0.cast(dtypes.float), act1.cast(dtypes.float)
  amax = a0.abs().maximum(a1.abs()).reduce(lane, arg=Ops.MAX)
  amax_rounded = ((amax.bitcast(dtypes.uint32) + 0x200000) & 0xFF800000).bitcast(dtypes.float)
  scale_exp = amax_rounded.maximum(2**-126).log2().floor().sub(2).maximum(-127).minimum(127)
  qscale = (-scale_exp).exp2()

  act0_store = act_out[out_idx].store(act0)
  act1_store = act_out.after(act0_store)[out_idx+1].store(act1)
  packed = _e2m1_code(a0*qscale) | (_e2m1_code(a1*qscale) << 4)
  packed_store = packed_out.after(act1_store)[block*(BLK//2)+lane].store(packed).end(lane)
  scale_k = K//BLK
  scale_row, scale_col = block//scale_k, block%scale_k
  row_group, row_half, row_lane = scale_row//32, scale_row%32//16, scale_row%16
  col_group, col_half, col_lane = scale_col//8, scale_col%8//4, scale_col%4
  scale_idx = (((((row_group*(scale_k//8)+col_group)*4+col_lane)*16+row_lane)*2+col_half)*2+row_half)
  scale_store = scale_out.after(packed_store)[scale_idx].store((scale_exp+127).cast(dtypes.uint8))
  return scale_store.end(sb, tid, wg).sink(arg=KernelInfo(f"silu_mul_quantize_mxfp4_{n_elems}", opts_to_apply=()))

@functools.cache
def _custom_silu_mul_bwd_mxfp4(grad_out:UOp, x_w13:UOp, grad_act:UOp) -> UOp:
  *prefix, two_k = x_w13.shape
  rows, K, VEC = math.prod(prefix), two_k//2, 8
  n_elems = rows*K
  assert n_elems % (THREADS_PER_WG*VEC) == 0
  grad_out, x_w13, grad_act = grad_out.reshape(rows*two_k), x_w13.reshape(rows*two_k), grad_act.reshape(n_elems)
  wg = UOp.range(n_elems//(THREADS_PER_WG*VEC), 0, AxisType.GLOBAL)
  tid = UOp.range(THREADS_PER_WG, 1, AxisType.LOCAL)
  lane = UOp.range(VEC, 2, AxisType.UNROLL)
  idx = (wg*THREADS_PER_WG+tid)*VEC+lane
  row, col = idx//K, idx%K

  g = grad_act[idx].cast(dtypes.float)
  w1 = x_w13[row*two_k+col].cast(dtypes.float)
  w3 = x_w13[row*two_k+K+col].cast(dtypes.float)
  sig = (1.0+(w1*-LOG2E).exp2()).reciprocal()
  grad_w1 = grad_out[row*two_k+col].store((g*sig*(1.0+w1*(1.0-sig))*w3).cast(dtypes.bfloat16))
  grad_w3 = grad_out.after(grad_w1)[row*two_k+K+col].store((g*w1*sig).cast(dtypes.bfloat16))
  return grad_w3.end(lane, tid, wg).sink(arg=KernelInfo(f"silu_mul_bwd_mxfp4_{n_elems}", opts_to_apply=()))

def _silu_mul_quantize_mxfp4_bwd(gradient:UOp, kernel:UOp):
  _, _, _, x_w13 = kernel.src[1:]
  axis = x_w13.axis if isinstance(x_w13.device, tuple) else None
  grad_out = alloc_like(x_w13.shape, dtypes.bfloat16, x_w13.device, axis)
  grad_out, *_ = Tensor.custom_kernel(grad_out, Tensor(x_w13, device=x_w13.device), Tensor(gradient, device=x_w13.device),
                                      fxn=_custom_silu_mul_bwd_mxfp4)
  return (None, None, None, grad_out.uop)

def silu_mul_quantize_mxfp4(x_w13:Tensor) -> tuple[Tensor, Tensor, Tensor]:
  assert x_w13.dtype == dtypes.bfloat16 and x_w13.ndim >= 2 and x_w13.shape[-1] % 512 == 0
  *prefix, two_k = x_w13.shape
  rows, K = math.prod(prefix), two_k//2
  axis = x_w13.uop.axis if isinstance(x_w13.device, tuple) else None
  act = alloc_like((*prefix, K), dtypes.bfloat16, x_w13.device, axis)
  packed = alloc_like((*prefix, K//2), dtypes.uint8, x_w13.device, axis)
  scale = alloc_like((rows, K//BLK), dtypes.uint8, x_w13.device, 0 if axis is not None else None)
  act, packed, scale, *_ = Tensor.custom_kernel(act, packed, scale, x_w13, fxn=_custom_silu_mul_quantize_mxfp4,
                                                 grad_fxn=_silu_mul_quantize_mxfp4_bwd)
  return act, packed, scale
