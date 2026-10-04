import functools, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.helpers import getenv
FAST_E8M0 = getenv("FAST_E8M0", 0)  # e8m0 = biased-exponent bit-extract of amax (deletes log2/exp2), byte-exact
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType
from extra.llama_kernels import FP8_MAX, THREADS_PER_WG, alloc_like, compile_hip, owned_empty

BLK = 32
PACK = 4

@functools.cache
def _custom_quantize_mxfp8(fp8_out:UOp, e8_out:UOp, si_out:UOp, x:UOp) -> UOp:
  rows, K = x.shape
  scale_K = K // BLK
  n_elems = rows * K
  n_super = n_elems // (BLK * PACK)
  sk4 = scale_K // PACK
  assert n_super % THREADS_PER_WG == 0, f"{n_super=} must divide over {THREADS_PER_WG=}"
  nwg = n_super // THREADS_PER_WG

  x = x.reshape(n_elems)
  fp8_out = fp8_out.reshape(n_elems)
  e8_out = e8_out.reshape(rows * scale_K)
  si_out = si_out.reshape(sk4 * rows)

  wg = UOp.range(nwg, 0, AxisType.GLOBAL)
  tid = UOp.range(THREADS_PER_WG, 1, AxisType.LOCAL)
  sb = UOp.range(PACK, 2, AxisType.UNROLL)
  lane = UOp.range(BLK, 3, AxisType.UNROLL)

  super_idx = wg * THREADS_PER_WG + tid
  idx = super_idx * (BLK * PACK) + sb * BLK + lane

  x_f = x[idx].cast(dtypes.float)
  abs_x = (x_f < 0.0).where(-x_f, x_f)
  blk_max = abs_x.reduce(lane, arg=Ops.MAX)
  if FAST_E8M0:
    e8f = ((blk_max.bitcast(dtypes.uint32) >> 23) & 0xFF).minimum(254)   # e8m0 = biased exp field, no log2/exp2
    qscale = ((e8f.const_like(254) - e8f) << 23).bitcast(dtypes.float32)  # 2^(127-e8), exact power of 2
  else:
    e8f = (blk_max.maximum(1e-38).log2().floor() + 127.0).maximum(0.0).minimum(254.0)
    qscale = (127.0 - e8f).exp2()
  scaled = (x_f * qscale).maximum(-FP8_MAX).minimum(FP8_MAX)
  e8u8 = e8f.cast(dtypes.uint8)

  fp8_store = fp8_out[idx].store(scaled.cast(fp8_out.dtype)).end(lane)
  e8_store = e8_out.after(fp8_store)[super_idx * PACK + sb].store(e8u8)

  # pack the 4 e8 of this super-block into one uint32 (little-endian: byte sb), write transposed (sk4, row)
  packed = (e8u8.cast(dtypes.uint32) << (sb.cast(dtypes.uint32) * 8)).reduce(sb, arg=Ops.ADD)
  row, col4 = super_idx // sk4, super_idx % sk4
  si_store = si_out.after(e8_store.end(sb))[col4 * rows + row].store(packed)
  return si_store.end(tid, wg).sink(arg=KernelInfo(f"quantize_mxfp8_{n_elems}", opts_to_apply=()))

def _quantize_mxfp8_fused_bwd(gradient:UOp, kernel:UOp):
  _, e8_out, _, x = kernel.src[1:]
  device = x.device
  rows, K = x.shape
  scale_K = K // BLK
  e8 = Tensor(e8_out.after(kernel), device=device).reshape(rows, scale_K)
  qscale = (127.0 - e8.cast(dtypes.float32)).exp2().reshape(rows, scale_K, 1).expand(rows, scale_K, BLK).reshape(rows, K)
  grad_x = (Tensor(gradient, device=device).float() * qscale).cast(dtypes.bfloat16)
  return (None, None, None, grad_x.uop)

def quantize_mxfp8_fused(x:Tensor) -> tuple[Tensor, Tensor, Tensor]:
  assert x.dtype == dtypes.bfloat16, f"expected bf16, got {x.dtype}"
  assert x.ndim == 2, f"expected 2d (rows, K), got {x.shape}"
  from extra.gemm.cdna_asm_gemm import FP8_DTYPE
  rows, K = x.shape
  scale_K = K // BLK
  axis = x.uop.axis if isinstance(x.device, tuple) else None
  fp8_out = alloc_like((rows, K), FP8_DTYPE, x.device, axis)
  e8_out = alloc_like((rows, scale_K), dtypes.uint8, x.device, axis)
  si_out = alloc_like((scale_K // PACK, rows), dtypes.uint32, x.device, None if axis is None else (1 if axis == 0 else 0))
  fp8_out, e8_out, si_out, *_ = Tensor.custom_kernel(fp8_out, e8_out, si_out, x, fxn=_custom_quantize_mxfp8, grad_fxn=_quantize_mxfp8_fused_bwd)
  return fp8_out, e8_out, si_out

@functools.cache
def _gptoss_attention_quantize(q:UOp, e8:UOp, si:UOp, x:UOp) -> UOp:
  assert x.shard_shape == q.shard_shape == (16384, 4096) and x.dtype == dtypes.bfloat16
  assert e8.shard_shape == (16384, 128) and e8.dtype == dtypes.uint8
  assert si.shard_shape == (32, 16384) and si.dtype == dtypes.uint32
  assert FAST_E8M0, "native attention quantization requires the established exponent-bit scale rule"
  zero = UOp.const(0, dtypes.int32)
  sink = UOp.sink(q.base, e8.base, si.base, x.base,
    *(out.index(zero).store(UOp.const(0, out.dtype)) for out in (q, e8, si)), x.index(zero).load(),
    UOp.special(128, "lidx0"), UOp.special(32768, "gidx0"), arg=KernelInfo("gptoss_attention_quantize"))
  src = (pathlib.Path(__file__).parent/"gptoss_attention_quantize.cpp").read_text()
  lib = compile_hip(src, ["-fno-fast-math", "-ffp-contract=off"])
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=lib)))

def gptoss_attention_quantize(x:Tensor) -> tuple[Tensor, Tensor, Tensor]:
  """Exact attention-output quantization with the packed GEMM scale layout written directly."""
  from extra.gemm.cdna_asm_gemm import FP8_DTYPE
  rows, K = x.shape
  axis = x.uop.axis if isinstance(x.device, tuple) else None
  assert axis in (None, 0)
  q = alloc_like((rows, K), FP8_DTYPE, x.device, axis)
  e8 = owned_empty(alloc_like((rows, K//32), dtypes.uint8, x.device, axis))
  si = alloc_like((K//128, rows), dtypes.uint32, x.device, 1 if axis == 0 else None)
  q, e8, si, *_ = Tensor.custom_kernel(q, e8, si, x, fxn=_gptoss_attention_quantize, grad_fxn=_quantize_mxfp8_fused_bwd)
  return q, e8, si

from extra.gptoss_kernels.quantize_mxfp8 import quantize_mxfp8_fused_qe8 as quantize_mxfp8_fused_qe8
