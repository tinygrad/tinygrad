import functools, os, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType
from tinygrad.renderer import Estimates
from extra.hipcc import HIPCCCompiler
from extra.llama_kernels import FP8_MAX, THREADS_PER_WG, alloc_like, compile_hip

# maps the swiglu-bwd bf16 gh uop -> its (fp8, e8) mxfp8 companion, so the gate_up dgrad can skip
# re-quantizing d_h. Keyed by the gh uop the backward returns; the consumer strips the pad/reshape
# the shrink-backward inserts before this gh reaches grouped_mx_gemm. See moe_gemm.custom_grouped_mx_gemm_bw.
_moe_dgrad_mailbox: dict = {}
FUSED_DGRAD_FP8 = os.getenv("FUSED_DGRAD_FP8", "0") == "1"

BLK = 32
PACK = 4
LOG2E = 1.4426950408889634
ALPHA = 1.702
LIMIT = 7.0

# gpt-oss swiglu, fused with the mxfp8 quantize of its output. Input h is (rows, 2*inter) INTERLEAVED:
# x_glu = h[:, 0::2], x_linear = h[:, 1::2]. y = (min(x_glu,7) * sigmoid(alpha*min(x_glu,7))) * (clamp(x_linear,-7,7)+1).
# For output flat idx, x_glu = h[2*idx], x_linear = h[2*idx+1] (since row*2K+2k == 2*(row*K+k)).
@functools.cache
def _custom_swiglu_quantize(fp8_out:UOp, e8_out:UOp, h:UOp) -> UOp:
  # outputs fp8 + e8 (1x32 block exponent) of the mxfp8-quantized swiglu(h); the packed `si` is built later
  # by the gemm pre-quant path AFTER padding K to a mult of 128 (scale_K=inter/32=90 isn't div by PACK=4).
  rows, K2 = h.shape
  K = K2 // 2
  scale_K = K // BLK
  n_elems = rows * K
  n_super = n_elems // (BLK * PACK)
  assert n_super % THREADS_PER_WG == 0, f"{n_super=} must divide {THREADS_PER_WG=}"
  nwg = n_super // THREADS_PER_WG
  h = h.reshape(rows * K2)
  fp8_out, e8_out = fp8_out.reshape(n_elems), e8_out.reshape(rows * scale_K)

  wg = UOp.range(nwg, 0, AxisType.GLOBAL)
  tid = UOp.range(THREADS_PER_WG, 1, AxisType.LOCAL)
  sb = UOp.range(PACK, 2, AxisType.UNROLL)
  lane = UOp.range(BLK, 3, AxisType.UNROLL)
  super_idx = wg * THREADS_PER_WG + tid
  idx = super_idx * (BLK * PACK) + sb * BLK + lane

  hg = h[idx * 2].cast(dtypes.float)
  hl = h[idx * 2 + 1].cast(dtypes.float)
  xg = hg.minimum(LIMIT)
  xl = hl.maximum(-LIMIT).minimum(LIMIT)
  sig = (1.0 + (xg * (ALPHA * -LOG2E)).exp2()).reciprocal()
  act = xg * sig * (xl + 1.0)
  abs_a = (act < 0.0).where(-act, act)
  blk_max = abs_a.reduce(lane, arg=Ops.MAX)
  e8f = (blk_max.maximum(1e-38).log2().floor() + 127.0).maximum(0.0).minimum(254.0)
  qscale = (127.0 - e8f).exp2()
  scaled = (act * qscale).maximum(-FP8_MAX).minimum(FP8_MAX)
  e8u8 = e8f.cast(dtypes.uint8)

  fp8_store = fp8_out[idx].store(scaled.cast(fp8_out.dtype)).end(lane)
  e8_store = e8_out.after(fp8_store)[super_idx * PACK + sb].store(e8u8)
  return e8_store.end(sb, tid, wg).sink(arg=KernelInfo(f"swiglu_quantize_{n_elems}", opts_to_apply=()))

@functools.cache
def _custom_swiglu_bwd(gh_out:UOp, h:UOp, grad_aq:UOp, e8:UOp) -> UOp:
  rows, K2 = h.shape
  K = K2 // 2
  scale_K = K // BLK
  n_elems = rows * K
  VEC = 8
  assert n_elems % (THREADS_PER_WG * VEC) == 0, f"{n_elems=} must divide {THREADS_PER_WG*VEC=}"
  nwg = n_elems // (THREADS_PER_WG * VEC)
  h, gh_out = h.reshape(rows * K2), gh_out.reshape(rows * K2)
  grad_aq, e8 = grad_aq.reshape(n_elems), e8.reshape(rows * scale_K)

  wg = UOp.range(nwg, 0, AxisType.GLOBAL)
  tid = UOp.range(THREADS_PER_WG, 1, AxisType.LOCAL)
  lane = UOp.range(VEC, 2, AxisType.UNROLL)
  idx = (wg * THREADS_PER_WG + tid) * VEC + lane

  e8v = e8[idx // BLK].cast(dtypes.float)
  qscale = (127.0 - e8v).exp2()
  ga = grad_aq[idx].cast(dtypes.float) * qscale
  hg = h[idx * 2].cast(dtypes.float)
  hl = h[idx * 2 + 1].cast(dtypes.float)
  xg = hg.minimum(LIMIT)
  xl = hl.maximum(-LIMIT).minimum(LIMIT)
  sig = (1.0 + (xg * (ALPHA * -LOG2E)).exp2()).reciprocal()
  sprime = sig * (1.0 + ALPHA * xg * (1.0 - sig))     # d(xg*sigmoid(alpha*xg))/d(xg)
  d_xg = ga * sprime * (xl + 1.0)
  d_xl = ga * (xg * sig)
  glu_ok = (hg < LIMIT).where(1.0, 0.0)                # clamp(max=7) gradient
  lin_ok = (hl > -LIMIT).where(1.0, 0.0) * (hl < LIMIT).where(1.0, 0.0)  # clamp(-7,7) gradient
  g1 = gh_out[idx * 2].store((d_xg * glu_ok).cast(gh_out.dtype))
  g3 = gh_out.after(g1)[idx * 2 + 1].store((d_xl * lin_ok).cast(gh_out.dtype))
  return g3.end(lane, tid, wg).sink(arg=KernelInfo(f"swiglu_bwd_{n_elems}", opts_to_apply=()))

@functools.cache
def _custom_swiglu_bwd_fp8(gh_out:UOp, g_fp8:UOp, g_e8:UOp, h:UOp, grad_aq:UOp, e8:UOp) -> UOp:
  # like _custom_swiglu_bwd but ALSO emits the mxfp8 (fp8,e8) of the bf16 gh it writes, so the gate_up
  # dgrad's grouped_mx_gemm skips re-quantizing d_h (the r_23 quantize kernel). VEC=16 => each thread owns
  # exactly one 32-wide output block (16 interleaved idx * 2 cols) so the block-amax is thread-local, no
  # cross-thread reduce. Block b covers gh cols [32b,32b+32) == quantize_mxfp8(gh)'s block b, bit-for-bit.
  rows, K2 = h.shape
  K = K2 // 2
  scale_K = K // BLK                  # blocks/row of the INCOMING grad companion (grad_aq is (rows,K))
  g_scale_K = K2 // BLK               # blocks/row of the gh output (rows, 2*inter)
  n_elems = rows * K
  VEC = 16
  assert n_elems % (THREADS_PER_WG * VEC) == 0, f"{n_elems=} must divide {THREADS_PER_WG*VEC=}"
  nwg = n_elems // (THREADS_PER_WG * VEC)
  h, gh_out, g_fp8 = h.reshape(rows * K2), gh_out.reshape(rows * K2), g_fp8.reshape(rows * K2)
  grad_aq, e8, g_e8 = grad_aq.reshape(n_elems), e8.reshape(rows * scale_K), g_e8.reshape(rows * g_scale_K)

  wg = UOp.range(nwg, 0, AxisType.GLOBAL)
  tid = UOp.range(THREADS_PER_WG, 1, AxisType.LOCAL)
  lane = UOp.range(VEC, 2, AxisType.UNROLL)
  base = wg * THREADS_PER_WG + tid    # output-block index in [0, rows*g_scale_K)
  idx = base * VEC + lane

  e8v = e8[idx // BLK].cast(dtypes.float)
  qscale = (127.0 - e8v).exp2()
  ga = grad_aq[idx].cast(dtypes.float) * qscale
  hg = h[idx * 2].cast(dtypes.float)
  hl = h[idx * 2 + 1].cast(dtypes.float)
  xg = hg.minimum(LIMIT)
  xl = hl.maximum(-LIMIT).minimum(LIMIT)
  sig = (1.0 + (xg * (ALPHA * -LOG2E)).exp2()).reciprocal()
  sprime = sig * (1.0 + ALPHA * xg * (1.0 - sig))
  d_xg = ga * sprime * (xl + 1.0)
  d_xl = ga * (xg * sig)
  glu_ok = (hg < LIMIT).where(1.0, 0.0)
  lin_ok = (hl > -LIMIT).where(1.0, 0.0) * (hl < LIMIT).where(1.0, 0.0)
  # quantize the SAME bf16 values that get stored to gh (so it byte-matches quantize_mxfp8(gh))
  wg_grad = (d_xg * glu_ok).cast(gh_out.dtype).cast(dtypes.float)
  wl_grad = (d_xl * lin_ok).cast(gh_out.dtype).cast(dtypes.float)
  g1 = gh_out[idx * 2].store(wg_grad.cast(gh_out.dtype))
  g3 = gh_out.after(g1)[idx * 2 + 1].store(wl_grad.cast(gh_out.dtype))
  ag = (wg_grad < 0.0).where(-wg_grad, wg_grad)
  al = (wl_grad < 0.0).where(-wl_grad, wl_grad)
  blk_max = ag.maximum(al).reduce(lane, arg=Ops.MAX)   # amax over this block's 32 gh values
  e8f = (blk_max.maximum(1e-38).log2().floor() + 127.0).maximum(0.0).minimum(254.0)
  qs = (127.0 - e8f).exp2()
  qg = (wg_grad * qs).maximum(-FP8_MAX).minimum(FP8_MAX)
  ql = (wl_grad * qs).maximum(-FP8_MAX).minimum(FP8_MAX)
  f1 = g_fp8.after(g3)[idx * 2].store(qg.cast(g_fp8.dtype))
  f2 = g_fp8.after(f1)[idx * 2 + 1].store(ql.cast(g_fp8.dtype))
  e8s = g_e8.after(f2)[base].store(e8f.cast(dtypes.uint8))
  return e8s.end(lane, tid, wg).sink(arg=KernelInfo(f"swiglu_bwd_fp8_{n_elems}", opts_to_apply=()))

@functools.cache
def _custom_swiglu_bwd_fp8_strided(gh_out:UOp, g_fp8:UOp, g_e8:UOp, h:UOp, grad_aq:UOp, *, real_inter:int) -> UOp:
  # Physical dY from a split grouped dgrad has a padded row stride. Consume it directly, apply dSwiGLU, and emit
  # bf16+rowwise-MXFP8 dH without the shrink/pad/contiguous roundtrip or the forward-Y scale cancellation. Only
  # launch derivative math for real H pairs; a few real workers also initialize the padded dH blocks to exact zero.
  rows, K2 = h.shape
  grows, grad_stride = grad_aq.shape
  K, g_scale_K = K2 // 2, K2 // BLK
  assert grows == rows and real_inter <= K <= grad_stride
  VEC = 16
  real_tail_req = int(os.getenv("DSWIGLU_REAL_ONLY_TAIL", "1"))
  gptoss_real_tail = (rows, K2, real_inter, grad_stride) == (73728, 5888, 2880, 3072)
  real_only_tail = bool(real_tail_req) and (gptoss_real_tail or real_tail_req == 2)
  # Five full waves outperform the otherwise attractive one-workgroup-per-row t180 mapping on GPT-OSS. Keep an
  # escape hatch for schedule experiments; all other shapes retain the generic geometry.
  threads_per_wg = int(os.getenv("GPTOSS_DSWIGLU_THREADS", "320")) if gptoss_real_tail and real_only_tail else THREADS_PER_WG
  assert K % VEC == 0 and real_inter % VEC == 0 and (K - real_inter) % VEC == 0
  blocks_per_row = real_inter // VEC if real_only_tail else K // VEC
  n_blocks = rows * blocks_per_row
  assert n_blocks % threads_per_wg == 0
  nwg = n_blocks // threads_per_wg
  h, gh_out, g_fp8 = h.reshape(rows * K2), gh_out.reshape(rows * K2), g_fp8.reshape(rows * K2)
  grad_aq, g_e8 = grad_aq.reshape(rows * grad_stride), g_e8.reshape(rows * g_scale_K)

  wg = UOp.range(nwg, 0, AxisType.GLOBAL)
  tid = UOp.range(threads_per_wg, 1, AxisType.LOCAL)
  lane = UOp.range(VEC, 2, AxisType.UNROLL)
  base = wg * threads_per_wg + tid
  if real_only_tail:
    row, block = base // blocks_per_row, base % blocks_per_row
    col = block * VEC + lane
    idx = row * K + col
    ga = grad_aq[row * grad_stride + col].cast(dtypes.float)
  else:
    idx = base * VEC + lane
    row, col = idx // K, idx % K
    ga = (col < real_inter).where(grad_aq[row * grad_stride + col].cast(dtypes.float), 0.0)
  hg, hl = h[idx * 2].cast(dtypes.float), h[idx * 2 + 1].cast(dtypes.float)
  xg, xl = hg.minimum(LIMIT), hl.maximum(-LIMIT).minimum(LIMIT)
  exp_arg = xg * (ALPHA * -LOG2E)
  sig = (1.0 + exp_arg.exp2()).reciprocal()
  sprime = sig * (1.0 + ALPHA * xg * (1.0 - sig))
  glu_ok = (hg < LIMIT).where(1.0, 0.0)
  lin_ok = (hl > -LIMIT).where(1.0, 0.0) * (hl < LIMIT).where(1.0, 0.0)
  wg_grad = (ga * sprime * (xl + 1.0) * glu_ok).cast(gh_out.dtype).cast(dtypes.float)
  wl_grad = (ga * (xg * sig) * lin_ok).cast(gh_out.dtype).cast(dtypes.float)
  g1 = gh_out[idx * 2].store(wg_grad.cast(gh_out.dtype))
  g3 = gh_out.after(g1)[idx * 2 + 1].store(wl_grad.cast(gh_out.dtype))
  ag, al = (wg_grad < 0.0).where(-wg_grad, wg_grad), (wl_grad < 0.0).where(-wl_grad, wl_grad)
  blk_max = ag.maximum(al).reduce(lane, arg=Ops.MAX)
  e8f = (blk_max.maximum(1e-38).log2().floor() + 127.0).maximum(0.0).minimum(254.0)
  qs = (127.0 - e8f).exp2()
  qg = (wg_grad * qs).maximum(-FP8_MAX).minimum(FP8_MAX)
  ql = (wl_grad * qs).maximum(-FP8_MAX).minimum(FP8_MAX)
  f1 = g_fp8.after(g3)[idx * 2].store(qg.cast(g_fp8.dtype))
  f2 = g_fp8.after(f1)[idx * 2 + 1].store(ql.cast(g_fp8.dtype))
  e8s = g_e8.after(f2)[row * g_scale_K + block].store(e8f.cast(dtypes.uint8)) if real_only_tail else \
    g_e8.after(f2)[base].store(e8f.cast(dtypes.uint8))
  if real_only_tail:
    # INDEX validity becomes control flow after coalescing, so only the first tail_blocks workers issue these stores.
    tail_blocks = (K - real_inter) // VEC
    tail_gate = block < tail_blocks
    tail_idx = row * K + real_inter + block * VEC + lane
    z1 = gh_out.after(e8s).index((tail_idx * 2).valid(tail_gate)).store(0.0)
    z2 = gh_out.after(z1).index((tail_idx * 2 + 1).valid(tail_gate)).store(0.0)
    z3 = g_fp8.after(z2).index((tail_idx * 2).valid(tail_gate)).store(0.0)
    z4 = g_fp8.after(z3).index((tail_idx * 2 + 1).valid(tail_gate)).store(0.0)
    e8s = g_e8.after(z4).index((row * g_scale_K + real_inter // VEC + block).valid(tail_gate)).store(0)
  suffix = ("_safehtail" if real_only_tail else "") + \
    (f"_t{threads_per_wg}" if gptoss_real_tail and threads_per_wg != 180 else "")
  ended = e8s.end(lane, tid, wg)
  info = KernelInfo(f"swiglu_bwd_fp8_strided{suffix}_{rows*K}_{grad_stride}", opts_to_apply=())
  return ended.sink(arg=info)

@functools.cache
def _custom_swiglu_bwd_fp8_fast(gh_out:UOp, g_fp8:UOp, g_e8:UOp, h:UOp, grad_aq:UOp, *, real_inter:int) -> UOp:
  rows, H = h.shape
  grows, grad_stride = grad_aq.shape
  assert grows == rows and H % 32 == 0 and real_inter <= H // 2 <= grad_stride
  blocks = rows * (H // 32)
  threads, workgroups = UOp.special(256, "lidx0"), UOp.special((blocks + 255) // 256, "gidx0")
  sink = UOp.sink(gh_out.base, g_fp8.base, g_e8.base, h.base, grad_aq.base, threads, workgroups,
                  arg=KernelInfo(f"swiglu_bwd_fp8_fast_{rows*(H//2)}_{grad_stride}",
                                 estimates=Estimates(ops=60*rows*(H//2), mem=rows*H*6)))
  src = (pathlib.Path(__file__).parent/"dswiglu_bwd_fp8_fast.cpp").read_text()
  defines = [f"-DM_DIM={rows}", f"-DH_DIM={H}", f"-DDY_STRIDE={grad_stride}", f"-DREAL_INTER={real_inter}", "-DTHREADS=256"]
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src),
                               UOp(Ops.BINARY, arg=compile_hip(src, defines))))

@functools.cache
def _custom_swiglu_bwd_fp8_dual(*args:UOp, real_inter:int) -> UOp:
  """Exact GPT-OSS dSwiGLU producer for both GEMM layouts plus 32-row dBias partials.

  Production omits the dead BF16 dH argument entirely. ``GPTOSS_DSWIGLU_DUAL_WRITE_DH=1`` adds it back as a
  byte-exact standalone oracle output.
  """
  write_dh = int(os.getenv("GPTOSS_DSWIGLU_DUAL_WRITE_DH", "0"))
  recip_nr = int(os.getenv("GPTOSS_DSWIGLU_RCP_NR", "1"))
  skip_pad_tail = int(os.getenv("GPTOSS_DSWIGLU_SKIP_PAD_TAIL", "1"))
  skip_empty_tail = int(os.getenv("GPTOSS_DSWIGLU_SKIP_EMPTY_TAIL", "1") == "1" and
                        os.getenv("MOE_SKIP_EMPTY", "0") == "1" and os.getenv("MOE_SKIP_NO_ZERO", "0") != "1")
  use_expert_counts = int(os.getenv("GPTOSS_DSWIGLU_EXPERT_COUNTS", "0"))
  xcd_map = int(os.getenv("GPTOSS_DSWIGLU_XCD_MAP", "0"))
  xcd_chunk = int(os.getenv("GPTOSS_DSWIGLU_XCD_CHUNK", "112"))
  lds_stride = int(os.getenv("GPTOSS_DSWIGLU_LDS_STRIDE", "264"))
  max_memory_clause = int(os.getenv("GPTOSS_DSWIGLU_MAX_MEMORY_CLAUSE", "1"))
  native_exp2 = int(os.getenv("GPTOSS_DSWIGLU_NATIVE_EXP2", "0"))
  tail_row_e8 = int(os.getenv("GPTOSS_DSWIGLU_TAIL_ROW_E8", "0"))
  assert native_exp2 in (0, 1), f"GPTOSS_DSWIGLU_NATIVE_EXP2 must be 0 or 1, got {native_exp2}"
  assert tail_row_e8 in (0, 1), f"GPTOSS_DSWIGLU_TAIL_ROW_E8 must be 0 or 1, got {tail_row_e8}"
  assert not (write_dh and (native_exp2 or tail_row_e8)), "native exp2 and tail row-scale remapping require the production no-dH ABI"
  assert recip_nr in (0, 1), f"GPTOSS_DSWIGLU_RCP_NR must be 0 or 1, got {recip_nr}"
  assert skip_pad_tail in (0, 1), f"GPTOSS_DSWIGLU_SKIP_PAD_TAIL must be 0 or 1, got {skip_pad_tail}"
  assert skip_empty_tail in (0, 1)
  assert use_expert_counts in (0, 1), f"GPTOSS_DSWIGLU_EXPERT_COUNTS must be 0 or 1, got {use_expert_counts}"
  assert xcd_map in (0, 1), f"GPTOSS_DSWIGLU_XCD_MAP must be 0 or 1, got {xcd_map}"
  assert xcd_chunk > 0, f"GPTOSS_DSWIGLU_XCD_CHUNK must be positive, got {xcd_chunk}"
  assert not (write_dh and use_expert_counts), "expert-count padding omission is restricted to the production no-dH ABI"
  assert lds_stride >= 256, f"GPTOSS_DSWIGLU_LDS_STRIDE must be at least 256, got {lds_stride}"
  assert max_memory_clause in (0, 1), f"GPTOSS_DSWIGLU_MAX_MEMORY_CLAUSE must be 0 or 1, got {max_memory_clause}"
  abi_args = args[:-1] if use_expert_counts else args
  expert_counts = args[-1] if use_expert_counts else None
  if write_dh:
    assert len(abi_args) == 9
    gh_out, g_fp8, g_e8, g_col, g_col_e8, bias_partial, h, grad_aq, expert_off = abi_args
    outputs = (gh_out, g_fp8, g_e8, g_col, g_col_e8, bias_partial)
  else:
    assert len(abi_args) == 8
    g_fp8, g_e8, g_col, g_col_e8, bias_partial, h, grad_aq, expert_off = abi_args
    outputs = (g_fp8, g_e8, g_col, g_col_e8, bias_partial)
  rows, H = h.shape
  grows, grad_stride = grad_aq.shape
  assert (rows, H, real_inter, grad_stride) == (73728, 5888, 2880, 3072), \
    f"GPT-OSS dual dSwiGLU only supports (73728,5888,2880,3072), got {(rows,H,real_inter,grad_stride)}"
  assert grows == rows
  assert g_fp8.shape == (rows, H) and g_e8.shape == (rows, H // 32)
  assert g_col.shape == (H, rows) and g_col_e8.shape == (H, rows // 32)
  assert bias_partial.shape == (rows // 32, H)
  assert expert_off.dtype == dtypes.int32 and expert_off.shape[-1] == 33
  if use_expert_counts:
    assert expert_counts is not None and expert_counts.dtype == dtypes.int32 and expert_counts.shape[-1] == 32
  threads = UOp.special(256, "lidx0")
  workgroups = UOp.special((rows // 32) * (H // 256), "gidx0")
  mem = rows * (H * 2 + grad_stride * 2 + H * 2 + H // 16) + bias_partial.numel() * 4 + write_dh * rows * H * 2
  # Like the reducer below, this is a precompiled binary: add metadata-only accesses so HCQ knows the real buffer
  # hazards. They do not change the supplied HIP executable or its ABI.
  zero = UOp.const(0, dtypes.int32)
  accesses = tuple(x.index(zero).store(UOp.const(0, x.dtype)) for x in outputs) + \
    (h.index(zero).load(), grad_aq.index(zero).load(), expert_off.index(zero).load()) + \
    ((expert_counts.index(zero).load(),) if expert_counts is not None else ())
  sink_args = (*outputs, h, grad_aq, expert_off) + ((expert_counts,) if expert_counts is not None else ())
  sink = UOp.sink(*(x.base for x in sink_args), *accesses, threads, workgroups,
                  arg=KernelInfo(f"swiglu_bwd_fp8_dual_gptoss_nr{recip_nr}_pt{skip_pad_tail}_et{skip_empty_tail}" +
                                 f"_cnt{use_expert_counts}" +
                                 f"_xm{xcd_map}c{xcd_chunk}" +
                                 f"_ls{lds_stride}_mc{max_memory_clause}" +
                                 ("_exp2" if native_exp2 else "") + ("_tre8" if tail_row_e8 else "") +
                                 ("_write_dh" if write_dh else ""),
                                 estimates=Estimates(ops=60*rows*(real_inter if skip_pad_tail else H//2), mem=mem)))
  src = (pathlib.Path(__file__).parent/"dswiglu_bwd_fp8_dual.cpp").read_text()
  defines = [f"-DM_DIM={rows}", f"-DH_DIM={H}", f"-DDY_STRIDE={grad_stride}", f"-DREAL_INTER={real_inter}",
             "-DTHREADS=256", f"-DNATIVE_MXFP8_CVT={int(os.getenv('NATIVE_MXFP8_CVT', '0'))}", f"-DWRITE_DH={write_dh}",
             f"-DGPTOSS_DSWIGLU_RCP_NR={recip_nr}", f"-DGPTOSS_DSWIGLU_SKIP_PAD_TAIL={skip_pad_tail}",
             f"-DGPTOSS_DSWIGLU_SKIP_EMPTY_TAIL={skip_empty_tail}",
             f"-DGPTOSS_DSWIGLU_EXPERT_COUNTS={use_expert_counts}",
             f"-DGPTOSS_DSWIGLU_XCD_MAP={xcd_map}", f"-DGPTOSS_DSWIGLU_XCD_CHUNK={xcd_chunk}",
             f"-DGPTOSS_DSWIGLU_LDS_STRIDE={lds_stride}"]
  defines += [f"-DGPTOSS_DSWIGLU_NATIVE_EXP2={native_exp2}", f"-DGPTOSS_DSWIGLU_TAIL_ROW_E8={tail_row_e8}"]
  # Do not add -ffast-math: reassociation changes a small number of BF16 boundary values and signed zeros.
  sched = ["-mllvm", "-amdgpu-sched-strategy=max-memory-clause"] if max_memory_clause else []
  lib = HIPCCCompiler("gfx950", ["-std=c++20", *defines, *sched]).compile_cached(src)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)), UOp(Ops.SOURCE, arg=src),
                               UOp(Ops.BINARY, arg=lib)))

def _swiglu_quantize_bwd(gradient:UOp, kernel:UOp):
  _, e8_out, h = kernel.src[1:]
  device = h.device
  rows, K2 = h.shape
  axis = h.axis if isinstance(device, tuple) else None
  gh = alloc_like((rows, K2), dtypes.bfloat16, device, axis)
  if FUSED_DGRAD_FP8:
    from extra.gemm.cdna_asm_gemm import FP8_DTYPE
    g_fp8 = alloc_like((rows, K2), FP8_DTYPE, device, axis)
    g_e8 = alloc_like((rows, K2 // BLK), dtypes.uint8, device, axis)
    gh, g_fp8, g_e8, *_ = Tensor.custom_kernel(gh, g_fp8, g_e8, Tensor(h, device=device),
                                               Tensor(gradient, device=device).cast(dtypes.bfloat16),
                                               Tensor(e8_out.after(kernel), device=device), fxn=_custom_swiglu_bwd_fp8)
    _moe_dgrad_mailbox[gh.uop] = (g_fp8.uop, g_e8.uop)
    return (None, None, gh.uop)
  gh, *_ = Tensor.custom_kernel(gh, Tensor(h, device=device), Tensor(gradient, device=device).cast(dtypes.bfloat16),
                                Tensor(e8_out.after(kernel), device=device), fxn=_custom_swiglu_bwd)
  return (None, None, gh.uop)

def fused_swiglu_quantize(h:Tensor) -> tuple[Tensor, Tensor]:
  # h: (rows, 2*inter) bf16 interleaved -> (fp8 y, e8) of the mxfp8-quantized swiglu output (rows, inter).
  assert h.dtype == dtypes.bfloat16 and h.ndim == 2, f"{h.shape} {h.dtype}"
  from extra.gemm.cdna_asm_gemm import FP8_DTYPE
  rows, K2 = h.shape
  K = K2 // 2
  scale_K = K // BLK
  axis = h.uop.axis if isinstance(h.device, tuple) else None
  fp8_out = alloc_like((rows, K), FP8_DTYPE, h.device, axis)
  e8_out = alloc_like((rows, scale_K), dtypes.uint8, h.device, axis)
  fp8_out, e8_out, *_ = Tensor.custom_kernel(fp8_out, e8_out, h, fxn=_custom_swiglu_quantize, grad_fxn=_swiglu_quantize_bwd)
  return fp8_out, e8_out
