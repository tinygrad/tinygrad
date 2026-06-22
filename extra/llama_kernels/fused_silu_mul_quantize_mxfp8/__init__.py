import functools
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType
from extra.llama_kernels import FP8_MAX, THREADS_PER_WG, alloc_like

BLK = 32
PACK = 4
LOG2E = 1.4426950408889634

@functools.cache
def _custom_silu_mul_quantize_mxfp8(fp8_out:UOp, e8_out:UOp, si_out:UOp, x_w1:UOp, x_w3:UOp) -> UOp:
  rows, K = x_w1.shape
  scale_K = K // BLK
  n_elems = rows * K
  n_super = n_elems // (BLK * PACK)
  sk4 = scale_K // PACK
  assert n_super % THREADS_PER_WG == 0, f"{n_super=} must divide over {THREADS_PER_WG=}"
  nwg = n_super // THREADS_PER_WG

  x_w1, x_w3 = x_w1.reshape(n_elems), x_w3.reshape(n_elems)
  fp8_out = fp8_out.reshape(n_elems)
  e8_out = e8_out.reshape(rows * scale_K)
  si_out = si_out.reshape(sk4 * rows)

  wg = UOp.range(nwg, 0, AxisType.GLOBAL)
  tid = UOp.range(THREADS_PER_WG, 1, AxisType.LOCAL)
  sb = UOp.range(PACK, 2, AxisType.UNROLL)
  lane = UOp.range(BLK, 3, AxisType.UNROLL)

  super_idx = wg * THREADS_PER_WG + tid
  idx = super_idx * (BLK * PACK) + sb * BLK + lane

  w1 = x_w1[idx].cast(dtypes.float)
  w3 = x_w3[idx].cast(dtypes.float)
  sig = (1.0 + (w1 * -LOG2E).exp2()).reciprocal()
  act = w1 * sig * w3
  abs_a = (act < 0.0).where(-act, act)
  blk_max = abs_a.reduce(lane, arg=Ops.MAX)
  e8f = (blk_max.maximum(1e-38).log2().floor() + 127.0).maximum(0.0).minimum(254.0)
  qscale = (127.0 - e8f).exp2()
  scaled = (act * qscale).maximum(-FP8_MAX).minimum(FP8_MAX)
  e8u8 = e8f.cast(dtypes.uint8)

  fp8_store = fp8_out[idx].store(scaled.cast(fp8_out.dtype.base)).end(lane)
  e8_store = e8_out.after(fp8_store)[super_idx * PACK + sb].store(e8u8)
  packed = (e8u8.cast(dtypes.uint32) << (sb.cast(dtypes.uint32) * 8)).reduce(sb, arg=Ops.ADD)
  row, col4 = super_idx // sk4, super_idx % sk4
  si_store = si_out.after(e8_store.end(sb))[col4 * rows + row].store(packed)
  return si_store.end(tid, wg).sink(arg=KernelInfo(f"silu_mul_quantize_mxfp8_{n_elems}", opts_to_apply=()))

def fused_silu_mul_quantize_mxfp8(x_w1:Tensor, x_w3:Tensor) -> tuple[Tensor, Tensor, Tensor]:
  assert x_w1.shape == x_w3.shape, f"{x_w1.shape} != {x_w3.shape}"
  assert x_w1.dtype == dtypes.bfloat16 and x_w3.dtype == dtypes.bfloat16
  assert x_w1.ndim == 2, f"expected 2d, got {x_w1.shape}"
  from extra.gemm.cdna_asm_gemm import FP8_DTYPE
  rows, K = x_w1.shape
  scale_K = K // BLK
  axis = x_w1.uop.axis if isinstance(x_w1.device, tuple) else None
  fp8_out = alloc_like((rows, K), FP8_DTYPE, x_w1.device, axis)
  e8_out = alloc_like((rows, scale_K), dtypes.uint8, x_w1.device, axis)
  si_out = alloc_like((scale_K // PACK, rows), dtypes.uint32, x_w1.device, None if axis is None else (1 if axis == 0 else 0))
  fp8_out, e8_out, si_out, *_ = Tensor.custom_kernel(fp8_out, e8_out, si_out, x_w1, x_w3,
                                                     fxn=_custom_silu_mul_quantize_mxfp8)
  return fp8_out, e8_out, si_out
