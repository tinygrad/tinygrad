import functools, os, pathlib
from tinygrad import Device, Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo, AxisType
from tinygrad.renderer import Estimates
from extra.llama_kernels import FP8_MAX, THREADS_PER_WG, alloc_like, compile_hip, dname_of, owned_empty
from extra.gemm.moe_routing import _blk_for, _sharded_zeros, _sharded_invalids, GLUE_OPTS, GLUE_BEAM

BLK = 32
LOG2E = 1.4426950408889634
FUSED_COMBINE = os.getenv("FUSED_COMBINE", "0") == "1"
# FAST_E8M0: compute the e8m0 block scale as the biased-exponent bit-field of amax (== floor(log2(amax))+127 for
# normals; 0 for amax==0) instead of software log2().floor()+127 / exp2(127-e8) — deletes 2 transcendentals/block.
FAST_E8M0 = os.getenv("FAST_E8M0", "0") == "1"
COMBINE_DW_HIP = os.getenv("COMBINE_DW_HIP", "0") == "1"
COMBINE_DW_PIPE2_NTZ = os.getenv("COMBINE_DW_PIPE2_NTZ", "0") == "1"
COMBINE_DZ_HIP = os.getenv("COMBINE_DZ_HIP", "1") == "1"
COMBINE_DZ_NT_BF16 = os.getenv("COMBINE_DZ_NT_BF16", "0") == "1"
COMBINE_DZ_WRITEONLY = os.getenv("COMBINE_DZ_WRITEONLY", "0") == "1"
COMBINE_FWD_HIP = os.getenv("GPTOSS_COMBINE_FWD_HIP", "0") == "1"

# Fused MoE combine: replaces `sel = gather(z); out = (sel*weights).sum(k)` (the 754MB sel round-trip) with
# one weighted gather-sum kernel. Its backward is a CONTENTION-FREE scatter (dest_row is a dropless bijection:
# each grouped row is the target of exactly one (token,expert-slot)) that ALSO emits d_z as mxfp8 (fp8,e8,
# block-scaled along D) so the down-gemm dgrad skips re-quantizing d_z (reuses moe_gemm._dgrad_fp8_lookup via
# the shared _moe_dgrad_mailbox). d_weights (router grad) = per-slot dot(d_out, gathered z).

@functools.cache
def _custom_combine_fwd(out:UOp, z:UOp, dest_row:UOp, weights:UOp) -> UOp:
  G, T_l, D = out.shape
  k = weights.shape[2]
  blk = _blk_for(D)
  g = UOp.range(G, 0)
  t = UOp.range(T_l, 1)
  jo = UOp.range(D // blk, 2)
  ji = UOp.range(blk, 3, AxisType.LOCAL)
  d = jo * blk + ji
  acc = UOp.const(0.0, dtypes.float)
  for j in range(k):
    row = dest_row.index(g, t * k + j).cast(dtypes.weakint)
    val = z.index(g, row, d).load().cast(dtypes.float)
    w = weights.index(g, t, j).load().cast(dtypes.float)
    acc = acc + w * val
  return out.index(g, t, d).store(acc.cast(out.dtype)).end(g, t, jo, ji).sink(
    arg=KernelInfo(name=f"combine_fwd_{T_l}_{D}", opts_to_apply=GLUE_OPTS, beam=GLUE_BEAM))  # gather-sum: beam-candidate

@functools.cache
def _custom_combine_fwd_hip(out:UOp, z:UOp, dest_row:UOp, weights:UOp) -> UOp:
  G, T_l, D = out.shape
  Gz, m_l, P = z.shape
  assert (G, T_l, D, Gz, m_l, P) == (1, 16384, 2880, 1, 73728, 3072)
  assert dest_row.shape == (G, T_l * 4) and weights.shape == (G, T_l, 4)
  assert out.dtype == z.dtype == dtypes.bfloat16 and dest_row.dtype == dtypes.int32 and weights.dtype == dtypes.float32
  threads, workgroups = UOp.special(256, "lidx0"), UOp.special(T_l, "gidx0")
  zero = UOp.const(0, dtypes.int32)
  accesses = (out.index(zero).store(UOp.const(0, out.dtype)), z.index(zero).load(),
              dest_row.index(zero).load(), weights.index(zero).load())
  sink = UOp.sink(out.base, z.base, dest_row.base, weights.base, *accesses, threads, workgroups,
                  arg=KernelInfo(f"combine_fwd_token_{T_l}_{D}_{P}_k4",
                                 estimates=Estimates(ops=2*T_l*4*D, mem=10*T_l*D+8*T_l*4)))
  src = (pathlib.Path(__file__).parent/"combine_fwd.cpp").read_text()
  defines = [f"-DT_DIM={T_l}", f"-DM_DIM={m_l}", f"-DD_DIM={D}", f"-DP_DIM={P}", "-DK_DIM=4", "-DTHREADS=256"]
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)),
                               UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=compile_hip(src, defines))))

@functools.cache
def _custom_combine_bwd_dz(dz_bf16:UOp, dz_fp8:UOp, dz_e8:UOp, d_out:UOp, dest_row:UOp, weights:UOp) -> UOp:
  # SCATTER (used rows only; padding rows stay 0 from the zero-init): for each (g,t,j)
  #   vec = d_out[g,t,:] * weights[g,t,j]; quantize block-32 along the physical output width.
  G, T_l, D = d_out.shape
  Gz, m_l, P = dz_bf16.shape
  assert G == Gz and dz_fp8.shape == (G, m_l, P) and dz_e8.shape == (G, m_l, P // BLK)
  assert D <= P and D % BLK == 0 and P % BLK == 0
  k = weights.shape[2]
  scale_P = P // BLK
  g = UOp.range(G, 0)
  t = UOp.range(T_l, 1)
  jj = UOp.range(k, 2)
  db = UOp.range(scale_P, 3, AxisType.LOCAL)
  lane = UOp.range(BLK, 4, AxisType.UNROLL)
  d = db * BLK + lane
  real = d < D
  di = real.where(d, d.const_like(0))
  row = dest_row.index(g, t * k + jj).cast(dtypes.weakint)
  w = weights.index(g, t, jj).load().cast(dtypes.float)
  vec = real.where((d_out.index(g, t, di).load().cast(dtypes.float) * w).cast(dz_bf16.dtype).cast(dtypes.float),
                   UOp.const(0.0, dtypes.float))
  av = (vec < 0.0).where(-vec, vec)
  bmax = av.reduce(lane, arg=Ops.MAX)
  e8f = (bmax.maximum(1e-38).log2().floor() + 127.0).maximum(0.0).minimum(254.0)
  qs = (127.0 - e8f).exp2()
  q = (vec * qs).maximum(-FP8_MAX).minimum(FP8_MAX)
  s1 = dz_bf16.index(g, row, d).store(vec.cast(dz_bf16.dtype))
  s2 = dz_fp8.after(s1).index(g, row, d).store(q.cast(dz_fp8.dtype))
  s3 = dz_e8.after(s2).index(g, row, db).store(e8f.cast(dtypes.uint8))
  return s3.end(lane, db, jj, t, g).sink(arg=KernelInfo(name=f"combine_bwd_dz_{T_l}_{D}_{P}", opts_to_apply=()))

@functools.cache
def _custom_combine_bwd_dz_gather(dz_bf16:UOp, dz_fp8:UOp, dz_e8:UOp, d_out:UOp, src_row:UOp, weights:UOp) -> UOp:
  # GATHER form (NVIDIA-style, replaces the scatter): iterate grouped rows m IN ORDER (coalesced writes to dz),
  # gather the source (t,j) = decode(src_row[m]) and read d_out[g,t,:] (a contiguous row). src_row is the inverse
  # permutation (grouped row -> expanded source i=t*k+j; SENTINEL<0 for untargeted pad rows -> write 0 = zero-init).
  G, m_l, P = dz_bf16.shape
  Gd, T_l, D = d_out.shape
  assert G == Gd and dz_fp8.shape == (G, m_l, P) and dz_e8.shape == (G, m_l, P // BLK)
  assert D <= P and D % BLK == 0 and P % BLK == 0
  k = weights.shape[2]
  scale_P = P // BLK
  g = UOp.range(G, 0)
  m = UOp.range(m_l, 1)
  db = UOp.range(scale_P, 2, AxisType.LOCAL)
  lane = UOp.range(BLK, 3, AxisType.UNROLL)
  d = db * BLK + lane
  real = d < D
  di = real.where(d, d.const_like(0))
  src = src_row.index(g, m).load().cast(dtypes.weakint)
  valid = src >= 0
  si = valid.where(src, src.const_like(0))                 # clamp sentinel -> 0 to keep the gather index in-bounds
  t = si // k
  j = si - t * k
  w = weights.index(g, t, j).load().cast(dtypes.float)
  vec = real.where(valid.where(
    (d_out.index(g, t, di).load().cast(dtypes.float) * w).cast(dz_bf16.dtype).cast(dtypes.float),
    UOp.const(0.0, dtypes.float)), UOp.const(0.0, dtypes.float))
  av = (vec < 0.0).where(-vec, vec)
  bmax = av.reduce(lane, arg=Ops.MAX)
  if FAST_E8M0:
    # e8m0 = biased exponent field of amax (bit-exact vs floor(log2)+127 for normals, 0 for amax==0). qs = 2^(127-e8)
    # is an exact power of 2, built by placing (254-e8) into the exponent field. No log2/exp2 transcendentals.
    e8f = ((bmax.bitcast(dtypes.uint32) >> 23) & 0xFF).minimum(254)
    qs = ((e8f.const_like(254) - e8f) << 23).bitcast(dtypes.float32)
  else:
    e8f = (bmax.maximum(1e-38).log2().floor() + 127.0).maximum(0.0).minimum(254.0)
    qs = (127.0 - e8f).exp2()
  q = (vec * qs).maximum(-FP8_MAX).minimum(FP8_MAX)
  s1 = dz_bf16.index(g, m, d).store(vec.cast(dz_bf16.dtype))
  s2 = dz_fp8.after(s1).index(g, m, d).store(q.cast(dz_fp8.dtype))
  s3 = dz_e8.after(s2).index(g, m, db).store(e8f.cast(dtypes.uint8))
  return s3.end(lane, db, m, g).sink(
    arg=KernelInfo(name=f"combine_bwd_dz_{m_l}_{D}_{P}", opts_to_apply=()))  # beam-neutral reduce+quant hand-opt candidate

@functools.cache
def _custom_combine_bwd_dz_hip(dz_bf16:UOp, dz_fp8:UOp, dz_e8:UOp, d_out:UOp, src_row:UOp, weights:UOp) -> UOp:
  G, m_l, P = dz_bf16.shape
  assert dz_fp8.shape == (G, m_l, P) and dz_e8.shape == (G, m_l, P // BLK)
  assert len(d_out.shape) == 3 and d_out.shape[0] == G
  T_l, D = d_out.shape[1], d_out.shape[2]
  assert D <= P and D % BLK == 0 and P % BLK == 0
  assert src_row.shape == (G, m_l) and len(weights.shape) == 3 and weights.shape[:2] == (G, T_l)
  k = weights.shape[2]
  assert k > 0
  assert dz_bf16.dtype == dtypes.bfloat16 and dz_fp8.dtype.itemsize == 1 and dz_e8.dtype == dtypes.uint8
  assert d_out.dtype == dtypes.bfloat16 and src_row.dtype == dtypes.int32 and weights.dtype == dtypes.float32

  threads, workgroups = UOp.special(THREADS_PER_WG, "lidx0"), UOp.special(G * m_l, "gidx0")
  real_elems, physical_elems = G * m_l * D, G * m_l * P
  mem = real_elems * d_out.dtype.itemsize + physical_elems * (dz_bf16.dtype.itemsize + dz_fp8.dtype.itemsize) + \
        G * m_l * (src_row.dtype.itemsize + weights.dtype.itemsize + P // BLK)
  sink = UOp.sink(dz_bf16.base, dz_fp8.base, dz_e8.base, d_out.base, src_row.base, weights.base, threads, workgroups,
                  arg=KernelInfo(f"combine_bwd_dz_hip_{m_l}_{D}_{P}",
                                 estimates=Estimates(ops=2 * real_elems, mem=mem)))
  src = (pathlib.Path(__file__).parent/"combine_bwd_dz.cpp").read_text()
  defines = [f"-DG_DIM={G}", f"-DM_DIM={m_l}", f"-DT_DIM={T_l}", f"-DD_DIM={D}", f"-DP_DIM={P}", f"-DK_DIM={k}",
             f"-DTHREADS_PER_WG={THREADS_PER_WG}", f"-DNONTEMPORAL_BF16={int(COMBINE_DZ_NT_BF16)}"]
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)),
                               UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=compile_hip(src, defines))))

@functools.cache
def _custom_combine_bwd_dz_quant(row_q:UOp, row_e8:UOp, col_q:UOp, col_e8:UOp, bias:UOp,
                                 d_out:UOp, src_row:UOp, weights:UOp) -> UOp:
  assert row_q.shape == (73728, 3072) and row_e8.shape == (73728, 96)
  col_si = col_e8.dtype == dtypes.uint32
  assert col_q.shape == (3072, 73728) and col_e8.shape == ((576, 3072) if col_si else (3072, 2304)) and bias.shape == (2304, 3072)
  assert d_out.shape == (1, 16384, 2880) and src_row.shape == (1, 73728) and weights.shape == (1, 16384, 4)
  assert row_q.dtype == col_q.dtype and row_q.dtype.itemsize == 1
  assert row_e8.dtype == dtypes.uint8 and col_e8.dtype == (dtypes.uint32 if col_si else dtypes.uint8)
  assert bias.dtype == weights.dtype == dtypes.float32
  assert d_out.dtype == dtypes.bfloat16 and src_row.dtype == dtypes.int32
  tile256 = col_si and os.getenv("GPTOSS_COMBINE_DZ_TILE256", "0") == "1"
  threads = 256 if tile256 else 128
  args = (row_q, row_e8, col_q, col_e8, bias, d_out, src_row, weights)
  zero = UOp.const(0, dtypes.int32)
  accesses = tuple(x.index(zero).store(UOp.const(0, x.dtype)) for x in args[:5]) + tuple(x.index(zero).load() for x in args[5:])
  sink = UOp.sink(*(x.base for x in args), *accesses, UOp.special(threads, "lidx0"), UOp.special(2304 * (3072 // threads), "gidx0"),
                  arg=KernelInfo("combine_bwd_dz_quant_73728_2880_3072_" + ("t256_rowlocal1" if tile256 else "t128_dpp1") +
                                 ("_cs1" if col_si else "")))
  src = (pathlib.Path(__file__).parent/("combine_bwd_dz_quant_tile256.cpp" if tile256 else "combine_bwd_dz_quant.cpp")).read_text()
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)),
                               UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=compile_hip(src, [f"-DCOMBINE_COL_SI={int(col_si)}"]))))

@functools.cache
def _custom_combine_bwd_dz_dual(dz_bf16:UOp, dz_fp8:UOp, dz_e8:UOp, dz_col:UOp, dz_col_si:UOp,
                                d_out:UOp, src_row:UOp, weights:UOp) -> UOp:
  G, m_l, P = dz_bf16.shape
  T_l, D = d_out.shape[1:]
  k = weights.shape[2]
  assert dz_fp8.shape == (G, m_l, P) and dz_e8.shape == (G, m_l, P // BLK)
  assert dz_col.shape == (P, G * m_l) and dz_col_si.shape == (G * m_l // 128, P)
  threads = UOp.special(THREADS_PER_WG, "lidx0")
  workgroups = UOp.special((G * m_l // 128) * (P // THREADS_PER_WG), "gidx0")
  sink = UOp.sink(dz_bf16.base, dz_fp8.base, dz_e8.base, dz_col.base, dz_col_si.base,
                  d_out.base, src_row.base, weights.base, threads, workgroups,
                  arg=KernelInfo(f"combine_bwd_dz_dual_{m_l}_{D}_{P}",
                                 estimates=Estimates(ops=2 * G * m_l * D, mem=G * m_l * P * 4)))
  src = (pathlib.Path(__file__).parent/"combine_bwd_dz_dual.cpp").read_text()
  defines = [f"-DG_DIM={G}", f"-DM_DIM={m_l}", f"-DT_DIM={T_l}", f"-DD_DIM={D}", f"-DP_DIM={P}", f"-DK_DIM={k}",
             f"-DTHREADS_PER_WG={THREADS_PER_WG}", f"-DNATIVE_MXFP8_CVT={int(os.getenv('NATIVE_MXFP8_CVT', '0'))}"]
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)),
                               UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=compile_hip(src, defines))))


COMBINE_GATHER = os.getenv("COMBINE_GATHER", "0") == "1"

def _sentinel_kernel(out:UOp) -> UOp:
  i = UOp.range(out.numel(), 0)
  return out.flatten().index(i).store(UOp.const(-1, out.dtype)).end(i).sink(arg=KernelInfo(name="moe_sentinel", opts_to_apply=()))

@functools.cache
def _stream_output_touch_kernel(out:UOp) -> UOp:
  # Preserve the established JIT graph partition (three calls) while replacing
  # each full-buffer zero with one harmless element store. combine_bwd_dz then
  # overwrites the entire allocation, including this element.
  return out.flatten().index(0).store(UOp.const(0, out.dtype)).sink(
    arg=KernelInfo(name="combine_dz_output_touch", opts_to_apply=()))

def _combine_dz_output(shape, dtype, device) -> Tensor:
  out = _sharded_invalids(shape, dtype, device)
  return Tensor.custom_kernel(out, fxn=_stream_output_touch_kernel)[0]

@functools.cache
def _invmap_kernel(src_row:UOp, dest_row:UOp) -> UOp:
  # inverse permutation: for each expanded source i, src_row[dest_row[i]] = i (dest_row is an injection, no contention).
  G, Mk = dest_row.shape
  g = UOp.range(G, 0)
  i = UOp.range(Mk, 1)
  m = dest_row.index(g, i).load().cast(dtypes.weakint)
  return src_row.index(g, m).store(i.cast(src_row.dtype)).end(i, g).sink(arg=KernelInfo(name=f"moe_invmap_{Mk}", opts_to_apply=()))

@functools.cache
def _custom_combine_bwd_dw(dw:UOp, z:UOp, d_out:UOp, dest_row:UOp) -> UOp:
  G, T_l, k = dw.shape
  Gz, m_l, P = z.shape
  Gd, Td, D = d_out.shape
  assert G == Gz == Gd and T_l == Td and D <= P and dest_row.shape == (G, T_l * k)
  assert dw.dtype == dtypes.float32 and z.dtype == dtypes.bfloat16 and d_out.dtype == dtypes.bfloat16
  threads, workgroups = UOp.special(THREADS_PER_WG, "lidx0"), UOp.special(G * T_l, "gidx0")
  mem = G * T_l * D * (k + 1) * z.dtype.itemsize + dw.numel() * dw.dtype.itemsize
  sink = UOp.sink(dw.base, z.base, d_out.base, dest_row.base, threads, workgroups,
                  arg=KernelInfo(f"combine_bwd_dw_{T_l}_{D}_{P}",
                                 estimates=Estimates(ops=2 * G * T_l * k * D, mem=mem)))
  src = (pathlib.Path(__file__).parent/"combine_bwd_dw.cpp").read_text()
  exact_pipe2_ntz = COMBINE_DW_PIPE2_NTZ and (G, T_l, m_l, D, P, k, THREADS_PER_WG) == (1, 16384, 73728, 2880, 3072, 4, 256)
  defines = [f"-DG_DIM={G}", f"-DT_DIM={T_l}", f"-DM_DIM={m_l}", f"-DD_DIM={D}", f"-DP_DIM={P}", f"-DK_DIM={k}",
             f"-DTHREADS_PER_WG={THREADS_PER_WG}", f"-DPIPE2_NTZ={int(exact_pipe2_ntz)}"]
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)),
                               UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=compile_hip(src, defines))))


def _build_src_row(dest_row:Tensor, m_l:int) -> Tensor:
  # grouped_row -> expanded source i (SENTINEL -1 for untargeted pad rows), ~60us int-only. Lets the combine-bwd
  # scatter become a coalesced-write gather (NVIDIA MoE-permute style). Sentinel-init then scatter the indices.
  buf = _sharded_invalids((dest_row.shape[0], m_l), dtypes.int32, dest_row.device)
  buf = Tensor.custom_kernel(buf, fxn=_sentinel_kernel)[0]
  return Tensor.custom_kernel(buf, dest_row, fxn=_invmap_kernel)[0]

def _combine_bwd(gradient:UOp, kernel:UOp):
  from extra.llama_kernels.fused_swiglu_quantize_gptoss import _moe_dgrad_mailbox
  from extra.gemm.cdna_asm_gemm import FP8_DTYPE
  from extra.gemm.moe_routing import grouped_gather_rows
  out_u, z_u, dest_u, w_u = kernel.src[1:5]
  dev = z_u.device
  G, m_l, P = z_u.shape
  T_l, k, D = w_u.shape[1], w_u.shape[2], out_u.shape[2]
  assert D <= P and D % BLK == 0 and P % BLK == 0
  d_out = Tensor(gradient, device=dev).reshape(G, T_l, D)
  destT, wT = Tensor(dest_u, device=dev), Tensor(w_u, device=dev)
  # d_weights is a fused gather-dot on AMD: one workgroup owns all k slots for a token, reads d_out once, gathers z
  # directly, and writes k scalars. The fallback materializes only the real-width prefix.
  if COMBINE_DW_HIP and dname_of(dev) == "AMD" and w_u.dtype == dtypes.float32 and d_out.dtype == dtypes.bfloat16:
    dw = _sharded_invalids((G, T_l, k), dtypes.float32, dev)
    dw, *_ = Tensor.custom_kernel(dw, Tensor(z_u, device=dev), d_out, destT, fxn=_custom_combine_bwd_dw)
  else:
    sel = grouped_gather_rows(Tensor(z_u, device=dev), destT, G).reshape(G, T_l, k, P)[:, :, :, :D]
    dw = (sel.cast(dtypes.float) * d_out.reshape(G, T_l, 1, D)).sum(3)
  if os.getenv("GPTOSS_COMBINE_DZ_QUANT", "0") == "1":
    from extra.llama_kernels.fused_swiglu_quantize_gptoss import FUSED_DGRAD_FP8
    local_G = z_u.shard_shape[0] if isinstance(dev, tuple) and z_u.axis == 0 else G
    assert (local_G, m_l, P, T_l, D, k) == (1, 73728, 3072, 16384, 2880, 4)
    assert COMBINE_GATHER and COMBINE_DZ_HIP and dname_of(dev) == "AMD" and FUSED_DGRAD_FP8
    assert os.getenv("FUSED_DZ_BIAS_PARTIAL", "0") == "1" and os.getenv("FUSED_WGRAD_QUANT", "0") == "0", \
      "no-BF16 combine requires quantized wgrad and fused bias consumers"
    src_row = _build_src_row(destT, m_l)
    # Emit each allocation in its consumer's exact physical 2D shape. A grouped 3D output followed by
    # permute/reshape can materialize a full FP8 copy at the custom-wgrad boundary even when G_local == 1.
    # Every byte is written by this access-declaring producer, so output-touch initialization is unnecessary.
    packed_col = os.getenv("GPTOSS_COMBINE_COL_SI", "0") == "1"
    col_scale = ((G * m_l // 128, P), dtypes.uint32, 0) if packed_col else ((P, G * m_l // 32), dtypes.uint8, 1)
    outputs = [owned_empty(alloc_like(shape, dtype, dev, axis)) for shape, dtype, axis in (
      ((G * m_l, P), FP8_DTYPE, 0), ((G * m_l, P // 32), dtypes.uint8, 0),
      ((P, G * m_l), FP8_DTYPE, 1), col_scale, ((G * m_l // 32, P), dtypes.float32, 0))]
    rq, re, cq, ce, bp = Tensor.custom_kernel(*outputs, d_out, src_row, wT, fxn=_custom_combine_bwd_dz_quant)[:5]
    # This lazy cast is only a shape/gradient anchor. Down backward must consume all five companions, never cast FP8 into numeric BF16 dZ.
    anchor = rq.cast(dtypes.bfloat16).reshape(G, m_l, P)
    ent = tuple(x.uop for x in (rq, re, cq, ce, bp))
    _moe_dgrad_mailbox[anchor.uop] = _moe_dgrad_mailbox[rq.uop] = ent
    return (None, anchor.uop, None, dw.cast(w_u.dtype).uop)
  # d_z matches the producer's physical width. Padded columns are explicit zeroes in bf16/fp8/e8.
  # Both gather kernels explicitly cover every grouped row and physical column,
  # including sentinel rows and D:P padding, so their outputs need no zero-fill.
  dz_alloc = _combine_dz_output if COMBINE_GATHER and COMBINE_DZ_WRITEONLY else _sharded_zeros
  dz_bf16 = dz_alloc((G, m_l, P), dtypes.bfloat16, dev)
  dz_fp8 = dz_alloc((G, m_l, P), FP8_DTYPE, dev)
  dz_e8 = dz_alloc((G, m_l, P // BLK), dtypes.uint8, dev)
  if COMBINE_GATHER:  # coalesced-write gather via the inverse map (~2.5x the scatter); byte-exact
    src_row = _build_src_row(destT, m_l)
    dz_fxn = _custom_combine_bwd_dz_hip if COMBINE_DZ_HIP and dname_of(dev) == "AMD" else _custom_combine_bwd_dz_gather
    dz_bf16, dz_fp8, dz_e8, *_ = Tensor.custom_kernel(dz_bf16, dz_fp8, dz_e8, d_out, src_row, wT, fxn=dz_fxn)
  else:
    dz_bf16, dz_fp8, dz_e8, *_ = Tensor.custom_kernel(dz_bf16, dz_fp8, dz_e8, d_out, destT, wT, fxn=_custom_combine_bwd_dz)
  _moe_dgrad_mailbox[dz_bf16.uop] = (dz_fp8.reshape(G * m_l, P).uop, dz_e8.reshape(G * m_l, P // BLK).uop)
  return (None, dz_bf16.uop, None, dw.cast(w_u.dtype).uop)

def fused_combine(z:Tensor, r, n_tokens:int, experts_per_tok:int, real_dim:int|None=None) -> Tensor:
  # z: (G*m_l, P) grouped down-gemm output. Only the first D=real_dim columns participate; keeping physical width P
  # lets the producer allocation serve as the backward checkpoint without crop/restride copies.
  G, P = r.n_groups, z.shape[-1]
  D = P if real_dim is None else real_dim
  assert D <= P and D % BLK == 0 and P % BLK == 0
  z3 = z.reshape(G, r.m_l, P)
  out = _sharded_invalids((G, r.t_local, D), z.dtype, z3.device)
  local_G = z3.uop.shard_shape[0] if isinstance(z3.device, tuple) and z3.uop.axis == 0 else G
  exact_hip = COMBINE_FWD_HIP and dname_of(z3.device or Device.DEFAULT) == "AMD" and \
              (local_G, r.t_local, r.m_l, D, P, experts_per_tok) == (1, 16384, 73728, 2880, 3072, 4)
  out, *_ = Tensor.custom_kernel(out, z3, r.dest_row, r.weights,
                                 fxn=_custom_combine_fwd_hip if exact_hip else _custom_combine_fwd, grad_fxn=_combine_bwd)
  return out.reshape(n_tokens, D)
