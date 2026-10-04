import functools, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.helpers import getenv
FAST_E8M0 = getenv("FAST_E8M0", 0)
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType
from tinygrad.renderer import Estimates
from extra.hipcc import hip_program
from extra.llama_kernels import FP8_MAX, THREADS_PER_WG, alloc_like, owned_empty, compile_hip

BLK = 32

@functools.cache
def _custom_quantize_mxfp8_qe8_vector(fp8_out:UOp, e8_out:UOp, x:UOp, *, threads:int) -> UOp:
  """Two lanes per MX block, without a packed-scale output that dispatch would discard."""
  assert FAST_E8M0 and threads in (128, 256)
  assert x.shape == fp8_out.shape and x.shape in ((8192, 3072), (8192, 4096), (16384, 3072), (16384, 4096))
  assert x.dtype == dtypes.bfloat16 and fp8_out.dtype == dtypes.fp8e4m3 and e8_out.dtype == dtypes.uint8
  rows, k = x.shape
  assert e8_out.shape == (rows, k//BLK)
  n = rows*k
  zero = UOp.const(0, dtypes.int32)
  sink = UOp.sink(fp8_out.base, e8_out.base, x.base,
                  fp8_out.index(zero).store(UOp.const(0, fp8_out.dtype)), e8_out.index(zero).store(UOp.const(0, e8_out.dtype)),
                  x.index(zero).load(), UOp.special(threads, 'lidx0'), UOp.special(n//16//threads, 'gidx0'),
                  arg=KernelInfo(f'quantize_mxfp8_qe8_vector_{n}_t{threads}',
                                 estimates=Estimates(ops=3*n, lds=3*n+n//BLK, mem=3*n+n//BLK)))
  source = (pathlib.Path(__file__).parent/'quantize_qe8.cpp').read_text()
  lib = compile_hip(source, [f'-DTHREADS={threads}', '-fno-fast-math', '-ffp-contract=off'])
  return hip_program(sink, source, lib)

@functools.cache
def _custom_quantize_mxfp8_qe8(fp8_out:UOp, e8_out:UOp, x:UOp) -> UOp:
  """Dispatch-only q/e8 producer: avoid packing scale indices that dispatch never reads."""
  rows, K = x.shape
  scale_K = K // BLK
  n_elems = rows * K
  # One MX block per thread lowers VGPR pressure (60 vs 92 for PACK=4) and is ~8 us faster at the GPT-OSS shape.
  qe8_pack = 1
  n_super = n_elems // (BLK * qe8_pack)
  assert n_super % THREADS_PER_WG == 0, f"{n_super=} must divide over {THREADS_PER_WG=}"
  nwg = n_super // THREADS_PER_WG

  x = x.reshape(n_elems)
  fp8_out = fp8_out.reshape(n_elems)
  e8_out = e8_out.reshape(rows * scale_K)

  wg = UOp.range(nwg, 0, AxisType.GLOBAL)
  tid = UOp.range(THREADS_PER_WG, 1, AxisType.LOCAL)
  sb = UOp.range(qe8_pack, 2, AxisType.UNROLL)
  lane = UOp.range(BLK, 3, AxisType.UNROLL)

  super_idx = wg * THREADS_PER_WG + tid
  idx = super_idx * (BLK * qe8_pack) + sb * BLK + lane

  x_f = x[idx].cast(dtypes.float)
  abs_x = (x_f < 0.0).where(-x_f, x_f)
  blk_max = abs_x.reduce(lane, arg=Ops.MAX)
  if FAST_E8M0:
    e8f = ((blk_max.bitcast(dtypes.uint32) >> 23) & 0xFF).minimum(254)
    qscale = ((e8f.const_like(254) - e8f) << 23).bitcast(dtypes.float32)
  else:
    e8f = (blk_max.maximum(1e-38).log2().floor() + 127.0).maximum(0.0).minimum(254.0)
    qscale = (127.0 - e8f).exp2()
  scaled = (x_f * qscale).maximum(-FP8_MAX).minimum(FP8_MAX)
  e8u8 = e8f.cast(dtypes.uint8)

  fp8_store = fp8_out[idx].store(scaled.cast(fp8_out.dtype)).end(lane)
  e8_store = e8_out.after(fp8_store)[super_idx * qe8_pack + sb].store(e8u8)
  return e8_store.end(sb, tid, wg).sink(arg=KernelInfo(f"quantize_mxfp8_qe8_{n_elems}", opts_to_apply=()))

def _quantize_mxfp8_qe8_bwd(gradient:UOp, kernel:UOp):
  _, e8_out, x = kernel.src[1:]
  device = x.device
  rows, K = x.shape
  scale_K = K // BLK
  e8 = Tensor(e8_out.after(kernel), device=device).reshape(rows, scale_K)
  qscale = (127.0 - e8.cast(dtypes.float32)).exp2().reshape(rows, scale_K, 1).expand(rows, scale_K, BLK).reshape(rows, K)
  grad_x = (Tensor(gradient, device=device).float() * qscale).cast(dtypes.bfloat16)
  return (None, None, grad_x.uop)

def quantize_mxfp8_fused_qe8(x:Tensor) -> tuple[Tensor, Tensor]:
  assert x.dtype == dtypes.bfloat16, f"expected bf16, got {x.dtype}"
  assert x.ndim == 2, f"expected 2d (rows, K), got {x.shape}"
  from extra.gemm.cdna_asm_gemm import FP8_DTYPE
  rows, K = x.shape
  scale_K = K // BLK
  axis = x.uop.axis if isinstance(x.device, tuple) else None
  fp8_out = alloc_like((rows, K), FP8_DTYPE, x.device, axis)
  # Dispatch returns these scales as a backward checkpoint; bind its destination without a precompiled-return copy.
  e8_out = owned_empty(alloc_like((rows, scale_K), dtypes.uint8, x.device, axis))
  fxn = functools.partial(_custom_quantize_mxfp8_qe8_vector, threads=getenv('GPTOSS_QE8_VECTOR_THREADS', 256)) \
    if getenv('GPTOSS_QE8_VECTOR', 0) else _custom_quantize_mxfp8_qe8
  fp8_out = fp8_out.clone()
  fp8_out, e8_out, *_ = Tensor.custom_kernel(fp8_out, e8_out, x, fxn=fxn,
                                              grad_fxn=_quantize_mxfp8_qe8_bwd)
  return fp8_out, e8_out
