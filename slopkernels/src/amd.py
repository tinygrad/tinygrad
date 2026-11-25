import functools
import math

from tinygrad import UOp
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

WARP_SIZE, WMMA_M, WMMA_N, WMMA_K = 32, 16, 16, 16
WMMA_ARG, LOG2E = ((WMMA_N, WMMA_M, WMMA_K), 32), math.log2(math.e)

def _unbind(v:UOp) -> UOp: return v.unbind_all()[0]

def _lane_xor(val:UOp, mask:int) -> UOp:
  swz = UOp.native(f"__builtin_amdgcn_ds_swizzle({{0}}, {0x1f | mask<<10})", shape=val.shape, dtype=dtypes.int32, fmt=True)
  return swz.call(val.bitcast(dtypes.int32)).bitcast(val.dtype)

def _lane_xor_safe(val:UOp, mask:int) -> UOp:
  swz = UOp.native(f"__builtin_amdgcn_ds_swizzle({{0}}, {0x1f | mask<<10})", shape=val.shape, dtype=dtypes.int32, fmt=True)
  return swz.call(val.bitcast(dtypes.int32), inline=False).bitcast(val.dtype)

def _ds_bpermute(addr:UOp, val:UOp) -> UOp:
  fn = UOp.native("__builtin_amdgcn_ds_bpermute({0}, {1})", shape=val.shape, dtype=dtypes.int32, fmt=True)
  return fn.call(addr.cast(dtypes.int32), val.bitcast(dtypes.int32), inline=True).bitcast(val.dtype)

def _ds_bpermute_safe(addr:UOp, val:UOp) -> UOp:
  fn = UOp.native("__builtin_amdgcn_ds_bpermute({0}, {1})", shape=val.shape, dtype=dtypes.int32, fmt=True)
  return fn.call(addr.cast(dtypes.int32), val.bitcast(dtypes.int32), inline=False).bitcast(val.dtype)

def _fast_exp2(val:UOp) -> UOp:
  return UOp.native("__builtin_amdgcn_exp2f({})", shape=val.shape, dtype=dtypes.float32, fmt=True).call(val, inline=True)

def _fast_log2(val:UOp) -> UOp:
  return UOp.native("__builtin_amdgcn_logf({})", shape=val.shape, dtype=dtypes.float32, fmt=True).call(val, inline=True)

def _fast_rcp(val:UOp) -> UOp:
  return UOp.native("__builtin_amdgcn_rcpf({})", shape=val.shape, dtype=dtypes.float32, fmt=True).call(val, inline=True)

def _fast_rsq(val:UOp) -> UOp:
  return UOp.native("__builtin_amdgcn_rsqf({})", shape=val.shape, dtype=dtypes.float32, fmt=True).call(val, inline=True)

def _fast_fmax(a:UOp, b:UOp) -> UOp:
  return UOp.native("__builtin_fmaxf({0}, {1})", shape=a.shape, dtype=dtypes.float32, fmt=True).call(a, b, inline=True)

def _cvt_pk_f16(a:UOp, b:UOp) -> UOp:
  return UOp.native("__builtin_bit_cast(unsigned int, __builtin_amdgcn_cvt_pkrtz({0}, {1}))",
                    shape=a.shape, dtype=dtypes.uint32, fmt=True).call(a, b, inline=True)

def _cvt_pk_f16_once(a:UOp, b:UOp) -> UOp:
  return UOp.native("__builtin_bit_cast(unsigned int, __builtin_amdgcn_cvt_pkrtz({0}, {1}))",
                    shape=a.shape, dtype=dtypes.uint32, fmt=True).call(a, b, inline=False)

def _amd_fdot2(a_u32:UOp, b_u32:UOp, c_f32:UOp) -> UOp:
  h2t = "decltype(__builtin_amdgcn_cvt_pkrtz(0.0f, 0.0f))"
  return UOp.native(f"__builtin_amdgcn_fdot2(__builtin_bit_cast({h2t}, {{0}}), __builtin_bit_cast({h2t}, {{1}}), {{2}}, false)",
                    shape=c_f32.shape, dtype=dtypes.float32, fmt=True).call(a_u32, b_u32, c_f32, inline=True)

def _amd_byte_perm(a:UOp, b:UOp, selectors:UOp) -> UOp:
  return UOp.native("__builtin_amdgcn_perm({}, {}, {})", shape=a.shape, dtype=dtypes.uint32,
                    fmt=True).call(*[x.cast(dtypes.uint32) for x in (a, b, selectors)], inline=True)

def _sb_pass(val:UOp) -> UOp:
  return UOp.native("({{ __builtin_amdgcn_sched_barrier(0); ({0}); }})", shape=val.shape, dtype=val.dtype, fmt=True).call(val, inline=False)

def warp_reduce(val:UOp, op:Ops=Ops.ADD, full_wave:bool=False, masks:tuple[int, ...]|None=None) -> UOp:
  if val.op is Ops.INDEX and val.addrspace == AddrSpace.REG: val = val.load()
  if val.op is Ops.REDUCE or any(u.op is Ops.REDUCE for u in val.src):
    tmp = UOp.alloc((1,), val.dtype, addrspace=AddrSpace.REG)
    val = tmp.after(tmp[0].store(val))[0].load()
  assert val.dtype in (dtypes.float32, dtypes.int32, dtypes.uint32)
  for mask in ((16, 8, 4, 2, 1) if full_wave else (8, 4, 2, 1)) if masks is None else masks:
    val = _fast_fmax(val, _lane_xor(val, mask)) if op is Ops.MAX and val.dtype == dtypes.float32 else val.alu(op, _lane_xor(val, mask))
  return val

def _reg(shape:tuple[int|UOp, ...], value:float, dep:UOp|None=None) -> UOp:
  ret = UOp.alloc(shape, dtypes.float, addrspace=AddrSpace.REG)
  return ret.after((ret if dep is None else ret.after(dep)).store(ret.const_like(value)))

def _amd_load(ptr:UOp, lanes:int|None=None, stream:bool=False) -> UOp:
  assert ptr.op is Ops.INDEX
  # nontemporal scalar load: streamed weights must not evict the activations/KV cache from L2
  if lanes is None: return ptr.load(arg="nontemporal")
  buf, coords = ptr.src[0], ptr.src[1:]
  idx = sum((coord*math.prod(buf.shape[i+1:]) for i,coord in enumerate(coords)), UOp.const(0))
  return UOp(Ops.SHRINK, src=(buf.flatten(), idx, UOp.const(lanes))).load(arg="nontemporal" if stream else None)

# ******** tiny dense fp16 gemv (weight-reusing across tokens) ********

@functools.cache
def _amd_f16_gemv_kernel(out:UOp, w:UOp, x:UOp, *, in_blks:UOp, out_features:UOp, tokens:UOp, val_chunk:UOp, k_unroll:UOp) -> UOp:
  lanes = WARP_SIZE
  out_row = UOp.range(out_features, 0, AxisType.GLOBAL)
  lane = UOp.range(lanes, 2, axis_type=AxisType.LOCAL)
  acc = _reg((tokens,), 0)
  i_r = UOp.range(in_blks, 4)
  ku, u = UOp.range(k_unroll, 7, AxisType.UPCAST), UOp.range(val_chunk, 6, AxisType.UPCAST)
  t_r = UOp.range(tokens, 8)
  col = ((i_r * k_unroll + ku) * lanes + lane) * val_chunk + u
  wv = w[out_row, col].load(arg="nontemporal").float()
  dot = (wv * x[t_r, col].load().float()).reduce(u, ku, arg=(Ops.ADD, 0))
  upd = acc[t_r].store(acc.after(i_r)[t_r].load() + dot).end(t_r, i_r)
  t_o = UOp.range(tokens, 9)
  total = warp_reduce(acc.after(upd)[t_o].load(), full_wave=True)
  return out[t_o, out_row.valid(lane.eq(0))].store(total.cast(out.dtype)).end(t_o, out_row, lane).sink(
    arg=KernelInfo(name="linear_f16_gemv", opts_to_apply=()))

# ******** flash attention on the KV cache ********

@functools.cache
def _amd_flash_attention_decode_partial(out:UOp, stats:UOp, q:UOp, cache_kv:UOp, *,
                                        valid_kv_len:UOp, d_blks:UOp, group:UOp, waves:UOp) -> UOp:
  # Vectorized u32 KV loads (g16=0) + prescaled half2 Q + v_dot2_f32_f16 + unmasked/tail loop split + unified epilogue.
  vlen = _unbind(valid_kv_len)
  _, B, H_KV, max_n, _ = cache_kv.shape
  D, DPL, C = d_blks * 64, d_blks * 2, out.shape[2]
  scale_l2 = D.cast(dtypes.float32).rsqrt() * LOG2E
  total_steps = (vlen + waves - 1) // waves
  live_chunks = total_steps.minimum(C)
  block_bhkv, block_chunk = UOp.range(B * H_KV, 0, AxisType.GLOBAL), UOp.range(live_chunks, 1, AxisType.GLOBAL)
  lane, wave = UOp.range(WARP_SIZE, -1, axis_type=AxisType.WARP), UOp.range(waves, 3, axis_type=AxisType.LOCAL)
  b, kv_head = block_bhkv // H_KV, block_bhkv % H_KV
  zerof, neg_inf, z32 = UOp.const(0.0, dtypes.float32), UOp.const(-1e30, dtypes.float32), UOp.const(0, dtypes.uint32)
  # Pack prescaled Q into half2 (uint32) in registers: halves Q VGPRs and feeds v_dot2_f32_f16 directly.
  qh_reg = UOp.alloc((group, d_blks), dtypes.uint32, addrspace=AddrSpace.REG)
  h_q, w_q = UOp.range(group, 15), UOp.range(d_blks, 16)
  q0 = q[b, kv_head * group + h_q, 0, lane * DPL + 2 * w_q].load().float() * scale_l2
  q1 = q[b, kv_head * group + h_q, 0, lane * DPL + 2 * w_q + 1].load().float() * scale_l2
  qh_reg = qh_reg.after(qh_reg[h_q, w_q].store(_cvt_pk_f16(q0, q1)).end(w_q, h_q))
  acc_reg, max_reg, sum_reg = _reg((group, DPL), 0.0), _reg((group,), -1e30), _reg((group,), 0.0)
  base_st, rem_st = total_steps // live_chunks, total_steps % live_chunks
  step_start = block_chunk * base_st + block_chunk.minimum(rem_st)
  step_end = step_start + base_st + (block_chunk < rem_st).where(UOp.const(1, dtypes.weakint), UOp.const(0, dtypes.weakint))
  full_end = step_end.minimum(vlen // waves)
  n_full, n_tail = (full_end - step_start).maximum(0), (step_end - full_end.maximum(step_start)).maximum(0)
  kv_u32 = cache_kv.flatten().bitcast(dtypes.uint32)
  row_u32, bh_stride_u32, v_off_u32 = d_blks * 32, max_n * (d_blks * 32), B * H_KV * max_n * (d_blks * 32)
  bh_base = (b * H_KV + kv_head) * bh_stride_u32 + lane * d_blks

  def load_kv(ks_idx:UOp, dep:tuple[UOp, ...], valid:UOp|None):
    kb = UOp.alloc((d_blks,), dtypes.uint32, addrspace=AddrSpace.REG)
    vb = UOp.alloc((d_blks,), dtypes.uint32, addrspace=AddrSpace.REG)
    ks_s = valid.where(ks_idx, UOp.const(0, dtypes.weakint)) if valid is not None else ks_idx
    w_i = UOp.range(d_blks, 25 if valid is None else 26, AxisType.UPCAST)
    k_val = kv_u32[bh_base + ks_s * row_u32 + w_i].load(arg="nontemporal")
    v_val = kv_u32[v_off_u32 + bh_base + ks_s * row_u32 + w_i].load(arg="nontemporal")
    if valid is not None: v_val = valid.where(v_val, z32)
    ks = kb.after(*dep)[w_i].store(k_val).end(w_i)
    vs = vb.after(*dep)[w_i].store(v_val).end(w_i)
    return kb.after(ks), vb.after(vs)

  def compute_step(kbuf:UOp, vbuf:UOp, valid:UOp|None, acc_in:UOp, max_in:UOp, sum_in:UOp, rid:int) -> UOp:
    h_u = UOp.range(group, rid)
    dot_reg = _reg((1,), 0.0, h_u)
    w_d = UOp.range(d_blks, rid + 1)
    d_upd = dot_reg[0].store(_amd_fdot2(qh_reg[h_u, w_d].load(), kbuf[w_d].load(), dot_reg.after(w_d)[0].load())).end(w_d)
    sc = warp_reduce(dot_reg.after(d_upd)[0].load(), full_wave=True)
    score = valid.where(sc, neg_inf) if valid is not None else sc
    old_max = max_in[h_u].load()
    new_max = _fast_fmax(old_max, score)
    alpha = _fast_exp2(old_max - new_max)
    beta = valid.where(_fast_exp2(score - new_max), zerof) if valid is not None else _fast_exp2(score - new_max)
    w_v = UOp.range(d_blks, rid + 2)
    vh2 = vbuf[w_v].load().reshape((1,)).bitcast(dtypes.half)
    a0 = acc_reg[h_u, 2 * w_v].store(alpha * acc_in[h_u, 2 * w_v].load() + beta * vh2[0].float())
    a1 = acc_reg[h_u, 2 * w_v + 1].store(alpha * acc_in[h_u, 2 * w_v + 1].load() + beta * vh2[1].float())
    return UOp.group(UOp.group(a0, a1).end(w_v), max_reg[h_u].store(new_max),
                     sum_reg[h_u].store(alpha * sum_in[h_u].load() + beta)).end(h_u)

  # 1-step register prefetch on full unmasked steps + masked tail loop
  has_full = n_full > 0
  s0 = has_full.where(step_start, UOp.const(0, dtypes.weakint))
  cur_k, cur_v = load_kv(s0 * waves + wave, (qh_reg,), None)
  it = UOp.range((n_full - 1).maximum(0), 40)
  nxt_k, nxt_v = load_kv((step_start + it + 1) * waves + wave, (it, cur_k, cur_v), None)
  comp = compute_step(cur_k.after(it, nxt_k, nxt_v), cur_v.after(it, nxt_k, nxt_v), None,
                      acc_reg.after(it), max_reg.after(it), sum_reg.after(it), 41)
  w_cp = UOp.range(d_blks, 44, AxisType.UPCAST)
  cp = UOp.group(cur_k[w_cp].store(nxt_k.after(comp)[w_cp].load()), cur_v[w_cp].store(nxt_v.after(comp)[w_cp].load())).end(w_cp)
  loop_done = UOp.group(comp, cp).end(it)
  it_l = UOp.range(has_full.cast(dtypes.weakint), 45)
  last_done = compute_step(cur_k.after(loop_done, it_l), cur_v.after(loop_done, it_l), None,
                           acc_reg.after(loop_done, it_l), max_reg.after(loop_done, it_l), sum_reg.after(loop_done, it_l), 46).end(it_l)
  acc1, max1, sum1 = acc_reg.after(last_done), max_reg.after(last_done), sum_reg.after(last_done)
  it_t = UOp.range(n_tail, 50)
  ks_t = (full_end.maximum(step_start) + it_t) * waves + wave
  kbt, vbt = load_kv(ks_t, (it_t, last_done), ks_t < vlen)
  dt = compute_step(kbt, vbt, ks_t < vlen, acc1.after(it_t), max1.after(it_t), sum1.after(it_t), 51).end(it_t)
  acc_f, max_f, sum_f = acc_reg.after(dt), max_reg.after(dt), sum_reg.after(dt)
  # Unified epilogue: zero-pad fp16 acc_lds, single wave-reduce pass writes both partial (fp16) and stats (d_f == 0).
  acc_lds = UOp.alloc((waves, group, D), dtypes.half, addrspace=AddrSpace.LOCAL)
  ml_lds = UOp.alloc((waves, group, 2), dtypes.float32, addrspace=AddrSpace.LOCAL)
  d_u, h_w = UOp.range(DPL, 11, AxisType.UPCAST), UOp.range(group, 18, AxisType.UPCAST)
  s_acc = acc_lds.reshape(waves, group, WARP_SIZE, DPL)[wave, h_w, lane, d_u].store(
    (acc_f[h_w, d_u].load() / sum_f[h_w].load().maximum(1.0)).cast(dtypes.half)).end(h_w, d_u)
  s_m = ml_lds[wave, h_w.valid(lane.eq(0)), 0].store(max_f[h_w].load()).end(h_w)
  s_l = ml_lds[wave, h_w.valid(lane.eq(0)), 1].store(sum_f[h_w].load()).end(h_w)
  acc_lds, ml_lds = acc_lds.after(s_acc, s_m, s_l), ml_lds.after(s_acc, s_m, s_l)
  tid = wave * WARP_SIZE + lane
  fi = UOp.range((group * D + waves * WARP_SIZE - 1) // (waves * WARP_SIZE), 13)
  flat = tid + fi * (waves * WARP_SIZE)
  h_f, d_f = (flat // D).minimum(group - 1), flat % D
  w_m = UOp.range(waves, 14)
  M_val = ml_lds[w_m, h_f, 0].load().reduce(w_m, arg=(Ops.MAX, 0))
  w_r = UOp.range(waves, 19)
  w_wi = _fast_exp2(ml_lds[w_r, h_f, 0].load() - M_val) * ml_lds[w_r, h_f, 1].load()
  val = (w_wi * acc_lds[w_r, h_f, d_f].load().float()).reduce(w_r, arg=(Ops.ADD, 0))
  w_r2 = UOp.range(waves, 20)
  lsum = (_fast_exp2(ml_lds[w_r2, h_f, 0].load() - M_val) * ml_lds[w_r2, h_f, 1].load()).reduce(w_r2, arg=(Ops.ADD, 0))
  vf = flat < group * D
  q_h = kv_head * group + h_f
  o_st = out[b, q_h.valid(vf), block_chunk, d_f].store((val / _fast_fmax(lsum, UOp.const(1.0, dtypes.float32))).cast(out.dtype))
  st_m = stats[b, q_h.valid(vf & d_f.eq(0)), block_chunk, 0].store(M_val)
  st_l = stats[b, q_h.valid(vf & d_f.eq(0)), block_chunk, 1].store(lsum)
  return UOp.group(o_st, st_m, st_l).end(fi, lane, wave, block_chunk, block_bhkv).sink(
    arg=KernelInfo(name="flash_decode_partial", opts_to_apply=()))

@functools.cache
def _amd_flash_decode_combine(o:UOp, partial:UOp, stats:UOp, *, valid_kv_len:UOp, step_keys:UOp) -> UOp:
  # 64-dim tile per wave: saturates 64-96 CUs for H=32, D=128; fast exp2/fmax on fp16 normalized partials.
  vlen = _unbind(valid_kv_len)
  B, H, C, D = partial.shape
  block_bh, block_dt = UOp.range(B * H, 0, AxisType.GLOBAL), UOp.range(D // 64, 1, AxisType.GLOBAL)
  lane, i_u = UOp.range(WARP_SIZE, 2, axis_type=AxisType.LOCAL), UOp.range(2, 102, AxisType.UPCAST)
  b, h = block_bh // H, block_bh % H
  live = ((vlen + step_keys - 1) // step_keys).minimum(C)
  dim = block_dt * 64 + lane * 2 + i_u
  chunk = UOp.range(live, 100)
  chunk_max = _reg((1,), -1e30)
  upd0 = chunk_max[0].store(_fast_fmax(chunk_max.after(chunk)[0].load(), stats[b, h, chunk, 0].load())).end(chunk)
  chunk_max = chunk_max.after(upd0)
  chunk2 = UOp.range(live, 101)
  acc, wsum = _reg((2,), 0.0), _reg((1,), 0.0)
  w = _fast_exp2(stats[b, h, chunk2, 0].load() - chunk_max[0].load()) * stats[b, h, chunk2, 1].load()
  upd1 = UOp.group(acc[i_u].store(acc.after(chunk2)[i_u].load() + w * partial[b, h, chunk2, dim].load().float()).end(i_u),
                   wsum[0].store(wsum.after(chunk2)[0].load() + w)).end(chunk2)
  acc, wsum = acc.after(upd1), wsum.after(upd1)
  return o[b, h, 0, dim].store((acc[i_u].load() * wsum[0].load().reciprocal()).cast(o.dtype)).end(i_u, lane, block_dt, block_bh).sink(
    arg=KernelInfo(name="flash_decode_combine", opts_to_apply=()))

@functools.cache
def _amd_flash_attention(o:UOp, q:UOp, cache:UOp, *, valid_kv_len:UOp, q_start:UOp, d_blks:UOp,
                         m_blks:UOp, waves_m:UOp, bn_tiles:UOp, gqa_w:UOp, rdna4:bool=False) -> UOp:
  # Zero-P_lds S^T = wmma(K, Q) with separate zero-pad XOR-swizzled K_lds & V_lds (2 barriers/tile),
  # register byte_perm V transpose (ds16=0), ds_bpermute alpha/l_inv (zero alpha_lds), 2-loop unmasked+tail split, GQA-fused WGs on RDNA4.
  valid_kv_len, q_start = _unbind(valid_kv_len), _unbind(q_start)
  BH, _, D = q.shape
  _, B, H_KV, physical_n, _ = cache.shape
  gqa_group, w_per_head = BH // (B * H_KV), waves_m // gqa_w
  block_m_sz, block_n_sz, threads = w_per_head * WMMA_M, bn_tiles * WMMA_N, waves_m * WARP_SIZE
  TN, TD, K_CH, V_CH = bn_tiles, d_blks * 4, d_blks * 8, bn_tiles * 2
  SCALE_L2 = (d_blks * 64).cast(dtypes.float32).rsqrt() * LOG2E
  block_hg, block_m_raw = UOp.range(BH // gqa_w, 0, AxisType.GLOBAL), UOp.range(m_blks, 1, AxisType.GLOBAL)
  block_m = m_blks - 1 - block_m_raw
  wave, lane = UOp.range(waves_m, 2, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  tid, lane_g, lane_c = wave * WARP_SIZE + lane, lane // 16, lane % 16
  block_bh, wave_m = block_hg * gqa_w + (wave // w_per_head), wave % w_per_head
  kv_head = block_hg // (gqa_group // gqa_w)
  k_u32, v_base_u32 = cache.flatten().bitcast(dtypes.uint32), B * H_KV * physical_n * (d_blks * 32)
  m_base = block_m * block_m_sz + wave_m * 16

  frag_k = 8 if rdna4 else 16
  q_reg = UOp.alloc((TD, frag_k), dtypes.half, addrspace=AddrSpace.REG)
  kq_i, qk_u = UOp.range(TD, 10), UOp.range(frag_k, 11, AxisType.UPCAST)
  q_col = kq_i * 16 + ((qk_u // 4) * 8 + lane_g * 4 + (qk_u % 4) if rdna4 else qk_u)
  q_reg = q_reg.after(q_reg[kq_i, qk_u].store((q[block_bh, m_base + lane_c, q_col].load().float() * SCALE_L2).cast(dtypes.half)).end(qk_u, kq_i))

  k_lds_raw = UOp.alloc((block_n_sz * K_CH * 8,), dtypes.half, addrspace=AddrSpace.LOCAL)
  v_lds_raw = UOp.alloc((D * V_CH * 8,), dtypes.half, addrspace=AddrSpace.LOCAL)
  acc, m_i, l_i = _reg((TD, 8), 0.0), _reg((1,), -1e30), _reg((1,), 0.0)
  S_reg_buf = UOp.alloc((TN, 8), dtypes.float32, addrspace=AddrSpace.REG)
  P_reg_buf = UOp.alloc((TN, 8), dtypes.half, addrspace=AddrSpace.REG) if rdna4 else None
  P_half_buf = UOp.alloc((TN, 4), dtypes.uint32, addrspace=AddrSpace.REG) if not rdna4 else None
  P_tmp_buf = UOp.alloc((16,), dtypes.half, addrspace=AddrSpace.REG) if not rdna4 else None

  n_end_all = ((q_start + (block_m + 1) * block_m_sz).minimum(valid_kv_len) + block_n_sz - 1) // block_n_sz
  n_full = ((q_start + block_m * block_m_sz + 1).minimum(valid_kv_len) // block_n_sz).minimum(n_end_all)
  n_tail, t_start, t_end = n_end_all - n_full, UOp.const(0, dtypes.weakint), n_end_all

  def swz_k_r4(kr:UOp, kk:UOp) -> UOp: return (kr & 7) ^ (d_blks >= 2).where((kk >> 2) & 1, UOp.const(0, dtypes.weakint))
  def swz_v_ch(d_idx:UOp, ch:UOp) -> UOp:
    if rdna4:
      return bn_tiles.eq(4).where(ch ^ (d_idx & 7) ^ ((d_idx >> 3) & 7), ch ^ ((d_idx >> 1) & 3) ^ ((d_idx >> 3) & 3))
    return bn_tiles.eq(4).where(ch ^ (d_idx & 7), ch ^ ((d_idx >> 1) & 3))

  def store_k(tile_idx:UOp, dep:tuple[UOp, ...], rid:int) -> UOp:
    if rdna4: return store_k_reg_r4(load_k_reg_r4(tile_idx, dep, rid), dep, rid + 40)
    k_lds_u32 = k_lds_raw.after(*dep).bitcast(dtypes.uint32).reshape(block_n_sz, K_CH, 4)
    k_it = (bn_tiles * d_blks * 4) // waves_m
    buf = UOp.alloc((k_it, 4), dtypes.uint32, addrspace=AddrSpace.REG)
    ki_l = UOp.range(k_it, rid)
    bid_l = tid + ki_l * threads
    kr_l, kc_l = bid_l // K_CH, bid_l % K_CH
    v0 = _amd_load(k_u32[(kv_head * physical_n + tile_idx * block_n_sz + kr_l) * (d_blks * 32) + kc_l * 4], 4)
    buf = buf.after(UOp.group(*(buf.after(*dep)[ki_l, j].store(v0[j]) for j in range(4))).end(ki_l))
    ki_s = UOp.range(k_it, rid + 40, AxisType.UPCAST)
    bid_s = tid + ki_s * threads
    kr_s, kc_s = bid_s // K_CH, bid_s % K_CH
    return UOp.group(*(k_lds_u32[kr_s, kc_s ^ (kr_s & 7), j].store(buf[ki_s, j].load()) for j in range(4))).end(ki_s)

  def load_k_reg_r4(tile_idx:UOp, dep:tuple[UOp, ...], rid:int) -> UOp:
    k_it = (bn_tiles * d_blks * 2) // waves_m
    buf = UOp.alloc((k_it, 8), dtypes.uint32, addrspace=AddrSpace.REG)
    ki = UOp.range(k_it, rid)
    bid = tid + ki * threads
    kr, kk_i = bid // TD, bid % TD
    kb = (kv_head * physical_n + tile_idx * block_n_sz + kr) * (d_blks * 32) + kk_i * 8
    v0, v1 = _amd_load(k_u32[kb], 4), _amd_load(k_u32[kb + 4], 4)
    sts = [buf.after(*dep)[ki, j].store(v0[j]) for j in range(4)] + [buf.after(*dep)[ki, 4 + j].store(v1[j]) for j in range(4)]
    return buf.after(UOp.group(*sts).end(ki))

  def store_k_reg_r4(kbuf:UOp, dep:tuple[UOp, ...], rid:int) -> UOp:
    k_lds_u32 = k_lds_raw.after(*dep).bitcast(dtypes.uint32).reshape(block_n_sz, K_CH, 4)
    ki = UOp.range((bn_tiles * d_blks * 2) // waves_m, rid, AxisType.UPCAST)
    bid = tid + ki * threads
    kr, kk_i = bid // TD, bid % TD
    xk = swz_k_r4(kr, kk_i)
    ch0, ch1 = (kk_i * 2) ^ xk, (kk_i * 2 + 1) ^ xk
    ks = ([k_lds_u32[kr, ch0, j].store(kbuf[ki, s].load()) for j, s in enumerate((0, 1, 4, 5))] +
          [k_lds_u32[kr, ch1, j].store(kbuf[ki, s].load()) for j, s in enumerate((2, 3, 6, 7))])
    return UOp.group(*ks).end(ki)

  def load_v_reg_r4(tile_idx:UOp, dep:tuple[UOp, ...], masked:bool, rid:int) -> UOp:
    v_it = (bn_tiles * d_blks * 2) // waves_m
    buf = UOp.alloc((v_it, 8), dtypes.uint32, addrspace=AddrSpace.REG)
    z32 = UOp.const(0, dtypes.uint32)
    ti = UOp.range(v_it, rid)
    tbid = tid + ti * threads
    n_ctiles = d_blks * 8
    c0, pid = (tbid % n_ctiles) * 8, tbid // n_ctiles
    ga, gb = tile_idx * block_n_sz + pid * 2, tile_idx * block_n_sz + pid * 2 + 1
    va = _amd_load(k_u32[v_base_u32 + (kv_head * physical_n + ga) * (d_blks * 32) + c0 // 2], 4)
    vb = _amd_load(k_u32[v_base_u32 + (kv_head * physical_n + gb) * (d_blks * 32) + c0 // 2], 4)
    sts = ([buf.after(*dep)[ti, j].store((ga < valid_kv_len).where(va[j], z32) if masked else va[j]) for j in range(4)] +
           [buf.after(*dep)[ti, 4 + j].store((gb < valid_kv_len).where(vb[j], z32) if masked else vb[j]) for j in range(4)])
    return buf.after(UOp.group(*sts).end(ti))

  def store_v_reg_r4(vbuf:UOp, dep:tuple[UOp, ...], rid:int) -> UOp:
    v_lds_u32 = v_lds_raw.after(*dep).bitcast(dtypes.uint32).reshape(D, V_CH, 4)
    c_lo, c_hi = UOp.const(0x05040100, dtypes.uint32), UOp.const(0x07060302, dtypes.uint32)
    ti = UOp.range((bn_tiles * d_blks * 2) // waves_m, rid)
    tbid = tid + ti * threads
    n_ctiles = d_blks * 8
    c0, pid = (tbid % n_ctiles) * 8, tbid // n_ctiles
    r_a = pid * 2
    ch_a, slot_a = r_a // 8, (r_a % 8) // 2
    vs = []
    for p in range(4):
      d0, d1 = c0 + p * 2, c0 + p * 2 + 1
      ra_p, rb_p = vbuf[ti, p].load(), vbuf[ti, 4 + p].load()
      vs.append(v_lds_u32[d0, swz_v_ch(d0, ch_a), slot_a].store(_amd_byte_perm(rb_p, ra_p, c_lo)))
      vs.append(v_lds_u32[d1, swz_v_ch(d1, ch_a), slot_a].store(_amd_byte_perm(rb_p, ra_p, c_hi)))
    return UOp.group(*vs).end(ti)

  def store_v(tile_idx:UOp, dep:tuple[UOp, ...], masked:bool, rid:int) -> UOp:
    v_lds_u32 = v_lds_raw.after(*dep).bitcast(dtypes.uint32).reshape(D, V_CH, 4)
    z32, c_lo, c_hi = UOp.const(0, dtypes.uint32), UOp.const(0x05040100, dtypes.uint32), UOp.const(0x07060302, dtypes.uint32)
    ti = UOp.range((bn_tiles * d_blks * 2) // waves_m, rid)
    tbid, n_ctiles = tid + ti * threads, d_blks * 8
    c0, pid = (tbid % n_ctiles) * 8, tbid // n_ctiles
    blk4, par = pid // 2, pid % 2
    ga, gb = tile_idx * block_n_sz + blk4 * 4 + par, tile_idx * block_n_sz + blk4 * 4 + 2 + par
    ch_a, slot_a = (blk4 // 4) * 2 + par, blk4 % 4
    va = _amd_load(k_u32[v_base_u32 + (kv_head * physical_n + ga) * (d_blks * 32) + c0 // 2], 4)
    vb = _amd_load(k_u32[v_base_u32 + (kv_head * physical_n + gb) * (d_blks * 32) + c0 // 2], 4)
    ra = [(ga < valid_kv_len).where(va[j], z32) if masked else va[j] for j in range(4)]
    rb = [(gb < valid_kv_len).where(vb[j], z32) if masked else vb[j] for j in range(4)]
    return UOp.group(*(v_lds_u32[c0 + p * 2 + m, swz_v_ch(c0 + p * 2 + m, ch_a), slot_a].store(_amd_byte_perm(rb[p], ra[p], c_hi if m else c_lo))
                       for p in range(4) for m in (0, 1))).end(ti)

  def run_stage(n_len:UOp, t_base:UOp, rid:int, masked:bool, acc_in:UOp, m_in:UOp, l_in:UOp, prev:UOp):
    it = UOp.range(n_len, rid)
    tile_idx = t_base + it
    vbuf = load_v_reg_r4(tile_idx, (prev, it), masked, rid + 1) if rdna4 else None
    K_lds = k_lds_raw.after(prev, it, *((vbuf,) if vbuf is not None else ())).reshape(block_n_sz, K_CH, 8)
    S_reg = S_reg_buf.after(S_reg_buf.after(it).store(S_reg_buf.const_like(0.0)))
    if rdna4:
      k_qk, tn1 = UOp.range(TD, rid + 2), UOp.range(TN, rid + 3)
      kr = tn1 * 16 + lane_c
      k_frag = K_lds[kr, (k_qk * 2 + lane_g) ^ swz_k_r4(kr, k_qk)]
      qk_grp = S_reg[tn1].store(UOp.wmma(k_frag, q_reg[k_qk], S_reg.after(k_qk)[tn1], *WMMA_ARG)).end(tn1)
      qk_done = UOp.group(qk_grp, S_reg[0, 0].store(_sb_pass(S_reg.after(qk_grp)[0, 0].load()))).end(k_qk)
    else:
      k_qk, tn1 = UOp.range(TD, rid + 2), UOp.range(TN, rid + 3)
      kr = tn1 * 16 + lane_c
      k_frag = K_lds[kr, (k_qk * 2) ^ (kr & 7)].cat(K_lds[kr, (k_qk * 2 + 1) ^ (kr & 7)])
      qk_done = S_reg[tn1].store(UOp.wmma(k_frag, q_reg[k_qk], S_reg.after(k_qk)[tn1], *WMMA_ARG)).end(tn1, k_qk)
    S_cur = S_reg.after(qk_done)
    v_st = store_v_reg_r4(vbuf, (qk_done, it), rid + 31) if vbuf is not None else store_v(tile_idx, (qk_done,), masked, rid + 1)
    q_idx = q_start + block_m * block_m_sz + wave_m * 16 + lane_c
    if masked:
      tn_m, e_m = UOp.range(TN, rid + 4), UOp.range(8, rid + 5, AxisType.UPCAST)
      k_idx = tile_idx * block_n_sz + tn_m * 16 + (lane_g * 8 + e_m if rdna4 else e_m * 2 + lane_g)
      causal = (k_idx <= q_idx) & (k_idx < valid_kv_len)
      S_cur = S_reg.after(S_reg[tn_m, e_m].store(causal.where(S_cur[tn_m, e_m].load(), UOp.const(-math.inf, dtypes.float32))).end(e_m, tn_m))
    tn_r1, e_r1 = UOp.range(TN, rid + 6), UOp.range(8, rid + 7, AxisType.UPCAST)
    row_max = warp_reduce(S_cur[tn_r1, e_r1].load().reduce(e_r1, tn_r1, arg=(Ops.MAX, 0)), Ops.MAX, masks=(16,))
    m_prev = m_in.after(it, qk_done)[0].load()
    m_new = _fast_fmax(m_prev, row_max)
    m_safe = _fast_fmax(m_new, UOp.const(-1e30, dtypes.float32)) if masked else m_new
    alpha = _fast_exp2(m_prev - m_safe)
    # Rescale acc in registers via ds_bpermute (zero alpha_lds; 1 live a_e VGPR at a time)
    acc_p = acc_in.after(it, qk_done)
    res_sts = []
    for e in range(8):
      a_e = _ds_bpermute((lane_g * 8 + e if rdna4 else e * 2 + lane_g) * 4, alpha)
      td_a = UOp.range(TD, rid + 20 + e)
      res_sts.append(acc[td_a, e].store(acc_p[td_a, e].load() * a_e).end(td_a))
    acc_res = UOp.group(*res_sts)
    if rdna4:
      l_sc = _reg((1,), 0.0, qk_done)
      l_sc_st = l_sc[0].store(alpha * l_in.after(it, qk_done)[0].load())
      p_dep = UOp.group(acc_res, m_i[0].store(m_new), l_sc_st)
      p_dep = UOp.group(p_dep, l_sc[0].store(_sb_pass(l_sc.after(p_dep)[0].load())))
      assert P_reg_buf is not None
      P_reg = P_reg_buf
      p_sum_r = _reg((1,), 0.0, p_dep)
      tn_p = UOp.range(TN, rid + 10)
      ps_acc = p_sum_r.after(p_dep, tn_p)[0].load()
      p_sts = []
      for j in range(4):
        p0 = _fast_exp2(S_cur.after(p_dep)[tn_p, 2 * j].load() - m_safe)
        p1 = _fast_exp2(S_cur.after(p_dep)[tn_p, 2 * j + 1].load() - m_safe)
        ps_acc = ps_acc + (p0 + p1)
        h2 = _cvt_pk_f16(p0, p1).reshape((1,)).bitcast(dtypes.half)
        p_sts += [P_reg.after(p_dep)[tn_p, 2 * j].store(h2[0]), P_reg.after(p_dep)[tn_p, 2 * j + 1].store(h2[1])]
      p_grp = UOp.group(*p_sts, p_sum_r[0].store(ps_acc))
      p_done = UOp.group(p_grp, p_sum_r[0].store(_sb_pass(p_sum_r.after(p_grp)[0].load()))).end(tn_p)
      p_sum = warp_reduce(p_sum_r.after(p_done)[0].load(), Ops.ADD, masks=(16,))
      sm_done = UOp.group(p_done, l_i[0].store(l_sc.after(p_done)[0].load() + p_sum))
      v_ready = UOp.group(sm_done, v_st).barrier()
      next_tile = (tile_idx + 1).minimum((t_end - 1).maximum(0))
      kbuf = load_k_reg_r4(next_tile, (v_ready,), rid + 12)
      V_lds = v_lds_raw.after(v_ready, kbuf).reshape(D, V_CH, 8)
      P_cur, acc_c = P_reg.after(v_ready), acc.after(v_ready)
      tn_pv, td_g, td_i = UOp.range(TN, rid + 13), UOp.range(TD // 2, rid + 14), UOp.range(2, rid + 32)
      td_pv = td_g * 2 + td_i
      d_row = td_pv * 16 + lane_c
      v_frag = V_lds[d_row, swz_v_ch(d_row, tn_pv * 2 + lane_g)]
      pv_step = acc[td_pv].store(UOp.wmma(P_cur[tn_pv], v_frag, acc_c.after(tn_pv, td_g, td_i)[td_pv], *WMMA_ARG)).end(td_i)
      pv_grp = UOp.group(pv_step, acc[0, 0].store(_sb_pass(acc.after(pv_step)[0, 0].load()))).end(td_g, tn_pv)
      k_next = store_k_reg_r4(kbuf, (pv_grp,), rid + 33)
      done = UOp.group(pv_grp, k_next).barrier().end(it)
      return acc.after(done), m_i.after(done), l_i.after(done), done
    # RDNA3: compact P_half[TN, 4] u32 -> 4x _lane_xor(16) reconstructs full 16-half fragment for P @ V
    assert P_half_buf is not None and P_tmp_buf is not None
    P_half = P_half_buf
    p_sum_r = _reg((1,), 0.0, qk_done)
    tn_p = UOp.range(TN, rid + 10)
    ps_acc = p_sum_r.after(tn_p)[0].load()
    p_sts = []
    for j in range(4):
      p0 = _fast_exp2(S_cur[tn_p, 2 * j].load() - m_safe)
      p1 = _fast_exp2(S_cur[tn_p, 2 * j + 1].load() - m_safe)
      ps_acc = ps_acc + (p0 + p1)
      p_sts.append(P_half.after(qk_done)[tn_p, j].store(_cvt_pk_f16(p0, p1)))
    p_done = UOp.group(*p_sts, p_sum_r[0].store(ps_acc)).end(tn_p)
    p_sum = warp_reduce(p_sum_r.after(p_done)[0].load(), Ops.ADD, masks=(16,))
    sm_done = UOp.group(acc_res, p_done, m_i[0].store(m_new), l_i[0].store(alpha * l_in.after(it, qk_done)[0].load() + p_sum))
    v_ready = UOp.group(v_st, sm_done).barrier()
    next_tile = (tile_idx + 1).minimum((t_end - 1).maximum(0))
    k_next = store_k(next_tile, (v_ready,), rid + 12)
    Ph_cur, acc_c = P_half.after(v_ready), acc.after(v_ready)
    V_lds = v_lds_raw.after(v_ready).reshape(D, V_CH, 8)
    is_g0 = lane_g.eq(0)
    tn_pv = UOp.range(TN, rid + 13)
    pt_sts = []
    for j in range(4):
      my_u = Ph_cur[tn_pv, j].load()
      peer_u = _lane_xor(my_u, 16)
      h2_e = is_g0.where(my_u, peer_u).reshape((1,)).bitcast(dtypes.half)
      h2_o = is_g0.where(peer_u, my_u).reshape((1,)).bitcast(dtypes.half)
      pt_sts += [P_tmp_buf.after(tn_pv)[2 * j].store(h2_e[0]), P_tmp_buf.after(tn_pv)[2 * j + 1].store(h2_e[1]),
                 P_tmp_buf.after(tn_pv)[8 + 2 * j].store(h2_o[0]), P_tmp_buf.after(tn_pv)[8 + 2 * j + 1].store(h2_o[1])]
    P_frag = P_tmp_buf.after(*pt_sts)
    td_pv = UOp.range(TD, rid + 14)
    d_row = td_pv * 16 + lane_c
    v_frag = V_lds[d_row, swz_v_ch(d_row, tn_pv * 2)].cat(V_lds[d_row, swz_v_ch(d_row, tn_pv * 2 + 1)])
    pv_grp = acc[td_pv].store(UOp.wmma(P_frag, v_frag, acc_c.after(tn_pv, td_pv)[td_pv], *WMMA_ARG)).end(td_pv, tn_pv)
    done = UOp.group(pv_grp, k_next).barrier().end(it)
    return acc.after(done), m_i.after(done), l_i.after(done), done

  k0 = store_k(t_start, (q_reg,), 20).barrier()
  acc1, m1, l1, d1 = run_stage(n_full, t_start, 100, False, acc, m_i, l_i, k0)
  acc_f, _, l_f, _ = run_stage(n_tail, n_full, 200, True, acc1, m1, l1, d1)
  l_inv = l_f[0].load().reciprocal()
  inv_es = [_ds_bpermute((lane_g * 8 + e if rdna4 else e * 2 + lane_g) * 4, l_inv) for e in range(8)]
  td_o = UOp.range(TD, 300)
  o_st = UOp.group(*(o[block_bh, m_base + (lane_g * 8 + e if rdna4 else e * 2 + lane_g), td_o * 16 + lane_c].store(
    (acc_f[td_o, e].load() * inv_es[e]).cast(o.dtype)) for e in range(8))).end(td_o)
  return o_st.end(lane, wave, block_m_raw, block_hg).sink(arg=KernelInfo(name="rdna4_flash_attn" if rdna4 else "rdna3_flash_attn", opts_to_apply=()))

# ******** gated delta net: fused recurrent scan ********

@functools.cache
def _gated_delta_prefill_kernel(core:UOp, q:UOp, k:UOp, v:UOp, beta:UOp, alpha:UOp, state:UOp, kq:UOp, *,
                                k_blks:UOp, alpha_k:UOp, start_pos:UOp) -> UOp:
  batch, heads, tokens, value_dim, row_tile = *core.shape, 4
  key_dim = k_blks * 32
  core, v = (x.reshape(batch * heads, tokens, value_dim) for x in (core, v))
  q, k = (x.reshape(batch * heads, tokens, key_dim) for x in (q, k))
  beta, kq = (x.reshape(batch * heads, tokens) for x in (beta, kq))
  alpha, state = alpha.reshape(batch * heads, tokens, alpha_k), state.reshape(batch * heads, value_dim, key_dim)
  bh_row, lane = UOp.range(batch * heads * (value_dim // row_tile), 0, AxisType.GLOBAL), UOp.range(32, 1, axis_type=AxisType.LOCAL)
  bh, row_base = bh_row // (value_dim // row_tile), (bh_row % (value_dim // row_tile)) * row_tile
  ci_u = UOp.range(k_blks, 4, AxisType.UPCAST)
  col = lane + ci_u * 32
  current = UOp.alloc((row_tile, k_blks), dtypes.float32, addrspace=AddrSpace.REG)
  r_init = UOp.range(row_tile, 3, AxisType.UPCAST)
  current = current.after(current[r_init, ci_u].store(start_pos.eq(0).where(0.0, state[bh, row_base + r_init, col].load().float())).end(ci_u, r_init))
  token = UOp.range(tokens, 2)
  kv, qv, bv, kqv = k[bh, token, col].load(), q[bh, token, col].load(), beta[bh, token].load(), kq[bh, token].load()
  av = alpha[bh, token, col % alpha_k].load()
  prev = current.after(token)
  stores: list[UOp] = []
  for r_i in range(row_tile):
    row = row_base + r_i
    decayed = prev[r_i, ci_u].load() * av
    state_k = warp_reduce((decayed * kv).reduce(ci_u, arg=(Ops.ADD, 0)), full_wave=True)
    state_q = warp_reduce((decayed * qv).reduce(ci_u, arg=(Ops.ADD, 0)), full_wave=True)
    delta = (v[bh, token, row].load() - state_k) * bv
    stores += [core[bh, token, row.valid(lane.eq(0))].store(state_q + delta * kqv),
               current[r_i, ci_u].store(decayed + delta * kv).end(ci_u)]
  step = UOp.group(*stores).end(token)
  r_out = UOp.range(row_tile, 5, AxisType.UPCAST)
  return state[bh, row_base + r_out, col].store(current.after(step)[r_out, ci_u].load().cast(state.dtype)).end(ci_u, r_out, lane, bh_row).sink(
    arg=KernelInfo(name="gated_delta_prefill", opts_to_apply=()))

@functools.cache
def _gated_delta_prefill_wmma_kernel(core:UOp, q:UOp, k:UOp, v:UOp, beta:UOp, alpha:UOp, state:UOp, *,
                                     d_blks:UOp, start_pos:UOp) -> UOp:
  # RDNA4 Chunked-WY (C=16, 64-token super-tiles) WMMA GDN prefill (D = d_blks * 64, d_blks in {1, 2}).
  batch, heads, T, D = core.shape
  BH, k_split, k_tiles, n_vsub = batch * heads, 2, d_blks * 4, 4
  core_f, q_f, k_f, v_f = (x.reshape(BH, T, D) for x in (core, q, k, v))
  beta_f, alpha_f, state_f = beta.reshape(BH, T), alpha.reshape(BH, T), state.reshape(BH, D, D)
  kt_w, v_wgs, waves, threads = d_blks * 2, d_blks, n_vsub * k_split, n_vsub * k_split * WARP_SIZE
  wg, wave, lane = UOp.range(BH * v_wgs, 0, AxisType.GLOBAL), UOp.range(waves, 1, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  bh, v_wg, tid, lane_g, lane_c = wg // v_wgs, wg % v_wgs, wave * WARP_SIZE + lane, lane // 16, lane % 16
  v_sub, kw, w_ch, w_role = wave // k_split, wave % k_split, wave % 4, wave // 4
  v_col = (v_wg * n_vsub + v_sub) * 16 + lane_c
  z0, o1 = UOp.const(0.0, dtypes.float32), UOp.const(1.0, dtypes.float32)
  c_lo, c_hi = UOp.const(0x05040100, dtypes.uint32), UOp.const(0x07060302, dtypes.uint32)
  def swz(row:UOp, ch8:UOp) -> UOp: return ch8 ^ (row & 7)
  skq, skt = UOp.alloc((2 * 64 * D,), dtypes.half, addrspace=AddrSpace.LOCAL), UOp.alloc((4 * D * 16,), dtypes.half, addrspace=AddrSpace.LOCAL)
  s_T, s_P = UOp.alloc((1024,), dtypes.half, addrspace=AddrSpace.LOCAL), UOp.alloc((1024,), dtypes.half, addrspace=AddrSpace.LOCAL)
  s_M, s_dd = UOp.alloc((1024,), dtypes.float32, addrspace=AddrSpace.LOCAL), UOp.alloc((512,), dtypes.float32, addrspace=AddrSpace.LOCAL)
  s_L, s_g, s_b = (UOp.alloc((64,), dtypes.float32, addrspace=AddrSpace.LOCAL) for _ in range(3))
  s_reg = UOp.alloc((kt_w, 8), dtypes.float32, addrspace=AddrSpace.REG)
  buf_a, buf_b = UOp.alloc((8,), dtypes.float32, addrspace=AddrSpace.REG), UOp.alloc((8,), dtypes.float32, addrspace=AddrSpace.REG)
  buf_h = UOp.alloc((8,), dtypes.half, addrspace=AddrSpace.REG)
  def pk_h8(dst:UOp, dep:tuple[UOp, ...], fn) -> UOp:
    ops = [dst.after(*dep)[2 * p + m].store(u[m]) for p in range(4) for u in (fn(p).reshape((1,)).bitcast(dtypes.half),) for m in (0, 1)]
    return dst.after(UOp.group(*ops))
  def f8_to_h8(f_slice:UOp, dep:UOp, dst:UOp=buf_h) -> UOp:
    return pk_h8(dst, (dep,), lambda p: _cvt_pk_f16_once(f_slice[2 * p].load(), f_slice[2 * p + 1].load()))
  def pk_xor1(src:UOp, dep:UOp) -> UOp:
    pk = UOp.alloc((8,), dtypes.uint32, addrspace=AddrSpace.REG)
    return pk.after(UOp.group(*(pk.after(dep)[e].store(_cvt_pk_f16_once(src[e].load(), _lane_xor_safe(src[e].load(), 1))) for e in range(8))))
  # Phase 0: zero s_reg, load state_f only when start_pos != 0
  j_z, r_ld, j_i = UOp.range(kt_w, 8), UOp.range(start_pos.ne(0).cast(dtypes.weakint), 9), UOp.range(kt_w, 10)
  s_zeroed = s_reg.after(UOp.group(*(s_reg[j_z, e].store(z0) for e in range(8))).end(j_z))
  k0_s = (kw * kt_w + j_i) * 16 + lane_g * 8
  sv0, sv1 = _amd_load(state_f[bh, v_col, k0_s], 4), _amd_load(state_f[bh, v_col, k0_s + 4], 4)
  s_ld = UOp.group(*(s_reg.after(s_zeroed, r_ld)[j_i, 4 * m + e].store((sv1 if m else sv0)[e]) for m in (0, 1) for e in range(4))).end(j_i, r_ld)
  s_cur_init = s_reg.after(s_zeroed, s_ld)
  # Super-tile loop (64 tokens/super-tile)
  sup_r = UOp.range((T + 63) // 64, 15)
  t_base, s_cur_super = sup_r * 64, s_cur_init.after(sup_r)
  n_ch = ((T - t_base + 15) // 16).minimum(4)
  # Phase 1: burst-load K, Q, K^T + prefix log-decay
  dk4_cnt, n_ch8 = D // 4, D // 8
  it_kq = UOp.range((32 * dk4_cnt) // threads, 21)
  pid = tid + it_kq * threads
  rp_all, dk4 = pid // dk4_cnt, pid % dk4_cnt
  c_loc_s, rp_loc, dk, ta_l, tb_l = rp_all // 8, rp_all % 8, dk4 * 4, rp_all * 2, rp_all * 2 + 1
  ta_g, tb_g = t_base + ta_l, t_base + tb_l
  def ld_pk(src:UOp, tg:UOp) -> tuple[UOp, UOp]:
    r, vl = _amd_load(src[bh, tg.minimum(T - 1), dk], 4), tg < T
    return _cvt_pk_f16_once(vl.where(r[0], z0), vl.where(r[1], z0)), _cvt_pk_f16_once(vl.where(r[2], z0), vl.where(r[3], z0))
  ka, kb, qa, qb = ld_pk(k_f, ta_g), ld_pk(k_f, tb_g), ld_pk(q_f, ta_g), ld_pk(q_f, tb_g)
  skq_u32, skt_u32 = skq.after(sup_r).bitcast(dtypes.uint32).reshape(2, 64, n_ch8, 4), skt.after(sup_r).bitcast(dtypes.uint32).reshape(4, D, 8)
  ch8, sub, rp_s = dk4 // 2, (dk4 % 2) * 2, rp_loc ^ ((dk4 & 1) * 4)
  ca, cb = swz(ta_l, ch8), swz(tb_l, ch8)
  kq_st = UOp.group(
    *(skq_u32[m, tl, c, sub + s].store(p[s])
      for m, tl, c, p in ((0, ta_l, ca, ka), (0, tb_l, cb, kb), (1, ta_l, ca, qa), (1, tb_l, cb, qb)) for s in (0, 1)),
    *(skt_u32[c_loc_s, dk + 2 * i + m, rp_s].store(_amd_byte_perm(kb[i], ka[i], c_hi if m else c_lo)) for i in (0, 1) for m in (0, 1)),
  ).end(it_kq)
  t_ab_l = w_ch * 16 + lane_c
  t_ab_g = t_base + t_ab_l
  v_ab = t_ab_g < T
  a_val, b_val = v_ab.where(alpha_f[bh, t_ab_g.minimum(T - 1)].load(), o1), v_ab.where(beta_f[bh, t_ab_g.minimum(T - 1)].load(), z0)
  la_reg = UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG)
  la_cur = la_reg.after(la_reg.after(sup_r)[0].store(_fast_log2(a_val)))
  for d_sh in (1, 2, 4, 8):
    lv = la_cur[0].load()
    la_cur = la_reg.after(la_reg[0].store((lane_c >= d_sh).where(lv + _ds_bpermute_safe((lane_c - d_sh).maximum(0) * 4, lv), lv)))
  la_fin, ab_en = la_cur[0].load(), lane_g.eq(0) & w_role.eq(0)
  ab_st = UOp.group(s_L.after(sup_r)[t_ab_l.valid(ab_en)].store(la_fin), s_g.after(sup_r)[t_ab_l.valid(ab_en)].store(_fast_exp2(la_fin)),
                    s_b.after(sup_r)[t_ab_l.valid(ab_en)].store(b_val))
  b1 = UOp.group(kq_st, ab_st).barrier()
  skq_8, skt_8 = skq.after(b1).reshape(2, 4, 16, n_ch8, 8), skt.after(b1).reshape(4, D, 2, 8)
  sL_r, sg_r, sb_r = s_L.after(b1).reshape(4, 16), s_g.after(b1).reshape(4, 16), s_b.after(b1).reshape(4, 16)
  # Phase 2: Parallel Gram matrices (A strictly lower triangular, P lower triangular)
  g0, jg = buf_a.after(b1).store(buf_a.const_like(0.0)), UOp.range(k_tiles, 22)
  sw_c = swz(lane_c, jg * 2 + lane_g)
  gram_done = buf_a.store(UOp.wmma(skq_8[w_role, w_ch, lane_c, sw_c], skq_8[0, w_ch, lane_c, sw_c], buf_a.after(g0, jg), *WMMA_ARG)).end(jg)
  gv, Lc, sM_1d, sP_u32 = buf_a.after(gram_done), sL_r[w_ch, lane_c].load(), s_M.after(b1), s_P.after(b1).bitcast(dtypes.uint32)
  pv_reg = UOp.alloc((8,), dtypes.float32, addrspace=AddrSpace.REG)
  a_sts, pv_sts = [], []
  for e in range(8):
    tr = lane_g * 8 + e
    g_rat = _fast_exp2(sL_r[w_ch, tr].load() - Lc)
    a_v = (lane_c < tr).where(sb_r[w_ch, tr].load() * g_rat * gv[e].load(), z0)
    a_sts.append(sM_1d[((w_ch * 16 + tr) * 16 + lane_c).valid(w_role.eq(0))].store(a_v))
    pv_sts.append(pv_reg.after(gram_done)[e].store((lane_c <= tr).where(g_rat * gv[e].load(), z0)))
  pk_r, p_en = pk_xor1(pv_reg.after(UOp.group(*pv_sts)), gram_done), (lane_c & 1).eq(0) & w_role.eq(1)
  b2 = UOp.group(*a_sts, *(sP_u32[((w_ch * 16 + lane_g * 8 + e) * 8 + (lane_c // 2)).valid(p_en)].store(pk_r[e].load()) for e in range(8))).barrier()
  # Phase 3: Parallel 8x8 forward-sub + 2-WMMA T = (I + A)^{-1}
  sA_r, sT_u32, r_fsub1 = s_M.after(b2).reshape(4, 16, 16), s_T.after(b1).bitcast(dtypes.uint32), UOp.range(w_role.eq(0).cast(dtypes.weakint), 23)
  is_hi8, c8 = lane_c >= 8, lane_c & 7
  rb8, tc8 = is_hi8.where(UOp.const(8, dtypes.weakint), UOp.const(0, dtypes.weakint)), UOp.alloc((8,), dtypes.float32, addrspace=AddrSpace.REG)
  t_c1, tc_c = [], tc8.after(b2, r_fsub1)
  for r in range(8):
    val = c8.eq(r).where(o1, z0)
    for p in range(r): val = val - sA_r[w_ch, rb8 + r, rb8 + p].load() * tc_c[p].load()
    t_c1.append(st_r := tc8[r].store(val))
    tc_c = tc8.after(st_r)
  tc8_fin, in_diag8 = tc8.after(UOp.group(*t_c1)), (lane_g.eq(0) & (lane_c < 8)) | (lane_g.eq(1) & is_hi8)
  t8_r = buf_b.after(UOp.group(*(buf_b.after(tc8_fin)[e].store(in_diag8.where(tc8_fin[e].load(), z0)) for e in range(8))))
  t8_pk_r = pk_xor1(t8_r, t8_r)
  t8_sts = [sT_u32[((w_ch * 16 + lane_g * 8 + e) * 8 + (lane_c // 2)).valid((lane_c & 1).eq(0))].store(t8_pk_r[e].load()) for e in range(8)]
  t8_h, a10_en = f8_to_h8(t8_r, t8_r), is_hi8 & lane_g.eq(0)
  a10_r = pk_h8(UOp.alloc((8,), dtypes.half, addrspace=AddrSpace.REG), (t8_h,), lambda p: _cvt_pk_f16_once(
    a10_en.where(-sA_r[w_ch, lane_c, 2 * p].load(), z0), a10_en.where(-sA_r[w_ch, lane_c, 2 * p + 1].load(), z0)))
  b_done = buf_a.store(UOp.wmma(a10_r, t8_h, buf_a.after(buf_a.after(a10_r).store(buf_a.const_like(0.0))), *WMMA_ARG))
  b_h_reg = f8_to_h8(buf_a.after(b_done), b_done, UOp.alloc((8,), dtypes.half, addrspace=AddrSpace.REG))
  b_t8 = UOp.group(*t8_sts, b_h_reg).end(r_fsub1).barrier()
  r_fsub2 = UOp.range(w_role.eq(0).cast(dtypes.weakint), 29)
  sT_mid = s_T.after(b_t8, r_fsub2).reshape(4, 16, 2, 8)
  t10_done = buf_a.store(UOp.wmma(sT_mid[w_ch, lane_c, lane_g], b_h_reg.after(b_t8, r_fsub2),
                                  buf_a.after(buf_a.after(b_t8, r_fsub2).store(buf_a.const_like(0.0))), *WMMA_ARG))
  t10_pk_r = pk_xor1(buf_a.after(t10_done), t10_done)
  t10_en, sT_u32_2 = lane_g.eq(1) & (lane_c < 8) & (lane_c & 1).eq(0), s_T.after(b_t8).bitcast(dtypes.uint32)
  b3 = UOp.group(*(sT_u32_2[((w_ch*16 + 8 + e)*8 + lane_c//2).valid(t10_en)].store(t10_pk_r[e].load()) for e in range(8))).end(r_fsub2).barrier()
  sT_8, sP_8 = s_T.after(b3).reshape(4, 16, 2, 8), s_P.after(b3).reshape(4, 16, 2, 8)
  # Phase 4: Chunk scan (C=16 chunks)
  ch = UOp.range(n_ch, 24)
  t0_g, s_cur, L_last, g_last = t_base + ch * 16, s_cur_super.after(b3, ch), sL_r[ch, 15].load(), sg_r[ch, 15].load()
  v_tmp, r_vp = UOp.alloc((8,), dtypes.float32, addrspace=AddrSpace.REG), UOp.range(kw.eq(0).cast(dtypes.weakint), 31)
  v_tmp_r = v_tmp.after(UOp.group(*(v_tmp.after(b3, ch, r_vp)[e].store(
    ((tg := t0_g + lane_g * 8 + e) < T).where(_amd_load(v_f[bh, tg.minimum(T - 1), v_col]), z0)) for e in range(8))).end(r_vp))
  x_init, qs_init = buf_a.after(b3, ch).store(buf_a.const_like(0.0)), buf_b.after(b3, ch).store(buf_b.const_like(0.0))
  j_s = UOp.range(kt_w, 25)
  sb_h, sw_s = f8_to_h8(s_cur[j_s], j_s), swz(lane_c, (kw * kt_w + j_s) * 2 + lane_g)
  xq_done = UOp.group(buf_a.store(UOp.wmma(skq_8[0, ch, lane_c, sw_s], sb_h, buf_a.after(x_init, j_s), *WMMA_ARG)),
                      buf_b.store(UOp.wmma(skq_8[1, ch, lane_c, sw_s], sb_h, buf_b.after(qs_init, j_s), *WMMA_ARG))).end(j_s)
  x_p, qs_p, s_xq_1d = buf_a.after(xq_done), buf_b.after(xq_done), s_M.after(b3, ch)
  xq_b = UOp.group(*(s_xq_1d[(((v_sub * 2 + m) * 32 + lane) * 4 + p).valid(kw.eq(1))].store(
    _cvt_pk_f16_once(src[2 * p].load(), src[2 * p + 1].load()).bitcast(dtypes.float32))
    for m, src in ((0, x_p), (1, qs_p)) for p in range(4))).barrier()
  s_xq_h, r_k0 = s_M.after(xq_b).bitcast(dtypes.half).reshape(4, 2, 32, 8), UOp.range(kw.eq(0).cast(dtypes.weakint), 27)
  vr = [v_tmp_r.after(xq_b, r_k0)[e].load() for e in range(8)]
  def y_elem(idx:int) -> UOp:
    tr = lane_g * 8 + idx
    return sb_r[ch, tr].load() * (vr[idx] - sg_r[ch, tr].load() * (x_p[idx].load() + s_xq_h[v_sub, 0, lane, idx].load().float()))
  y_ready = pk_h8(buf_h, (xq_b, r_k0), lambda p: _cvt_pk_f16_once(y_elem(2 * p), y_elem(2 * p + 1)))
  d_done = buf_a.store(UOp.wmma(sT_8[ch, lane_c, lane_g], y_ready, buf_a.after(buf_a.after(y_ready).store(buf_a.const_like(0.0))), *WMMA_ARG))
  d_v = buf_a.after(d_done)
  o_scale = UOp.group(*(buf_b.after(d_done)[e].store(
    sg_r[ch, lane_g * 8 + e].load() * (qs_p[e].load() + s_xq_h[v_sub, 1, lane, e].load().float())) for e in range(8)))
  dd_b = UOp.group(o_scale, *(s_dd.after(o_scale)[(v_sub * 32 + lane) * 4 + p].store(_cvt_pk_f16_once(
    _fast_exp2(L_last - sL_r[ch, lane_g * 8 + 2 * p].load()) * d_v[2 * p].load(),
    _fast_exp2(L_last - sL_r[ch, lane_g * 8 + 2 * p + 1].load()) * d_v[2 * p + 1].load()).bitcast(dtypes.float32))
    for p in range(4))).end(r_k0).barrier()
  dd_frag, r_k0_o = s_dd.after(dd_b).bitcast(dtypes.half).reshape(-1, 8)[v_sub * 32 + lane], UOp.range(kw.eq(0).cast(dtypes.weakint), 28)
  o_v = buf_b.after(buf_b.store(UOp.wmma(sP_8[ch, lane_c, lane_g], f8_to_h8(buf_a.after(dd_b, r_k0_o), dd_b), buf_b.after(dd_b, r_k0_o), *WMMA_ARG)))
  core_done = UOp.group(*(core_f[bh, (t_e := t0_g + lane_g * 8 + e).valid(t_e < T), v_col].store(o_v[e].load()) for e in range(8))).end(r_k0_o)
  j_u = UOp.range(kt_w, 26)
  dk_s = (kw * kt_w + j_u) * 16 + lane_c
  s_upd = s_reg[j_u].store(UOp.wmma(skt_8[ch, dk_s, lane_g ^ ((dk_s >> 2) & 1)], dd_frag,
    buf_b.after(UOp.group(*(buf_b.after(core_done, j_u)[e].store(s_cur[j_u, e].load() * g_last) for e in range(8)))), *WMMA_ARG)).end(j_u)
  s_fin, j_o = s_reg.after(s_upd.end(ch).barrier().end(sup_r)), UOp.range(kt_w, 30)
  return UOp.group(*(state_f[bh, v_col, (kw * kt_w + j_o) * 16 + lane_g * 8 + e].store(s_fin[j_o, e].load()) for e in range(8))).end(
    j_o, lane, wave, wg).sink(arg=KernelInfo(name="rdna4_gated_delta_wmma", opts_to_apply=()))

# ******** mHC coefficients (fused RMS + Sinkhorn, 1 kernel) ********

def _hc_coeffs_kernel(out:UOp, mix:UOp, flat:UOp, base:UOp, scale:UOp, count:int, waves:UOp, iters:UOp, eps:UOp, norm_eps:UOp, features:UOp) -> UOp:
  assert count == 4, "the swizzles assume 4 streams"
  token = UOp.range(out.shape[0], 0, AxisType.GLOBAL)
  wave, lane = UOp.range(waves, 1, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  stride = waves * 128
  n_steps = (features + stride - 1) // stride
  part_reg = _reg((1,), 0.0)
  st = UOp.range(n_steps, 5)
  col_base = ((st * waves + wave) * WARP_SIZE + lane) * 4
  u = UOp.range(4, 6, AxisType.UPCAST)
  c_i = col_base + u
  v_f = (c_i < features).where(flat[token, c_i.valid(c_i < features)].load(), UOp.const(0.0, dtypes.float32))
  p_upd = part_reg[0].store(part_reg.after(st)[0].load() + (v_f * v_f).reduce(u, arg=(Ops.ADD, 0))).end(st)
  w_sum = warp_reduce(part_reg.after(p_upd)[0].load(), full_wave=True)
  lds = UOp.alloc((waves,), dtypes.float32, addrspace=AddrSpace.LOCAL)
  lds_st = lds[wave.valid(lane.eq(0))].store(w_sum).barrier()
  lds_r = lds.after(lds_st)
  r_w0 = UOp.range(wave.eq(0).cast(dtypes.weakint), 7)
  w_val = (lane < waves).where(lds_r.after(lds_st, r_w0)[lane.minimum(waves - 1)].load(), UOp.const(0.0, dtypes.float32))
  total_sq = warp_reduce(w_val, full_wave=True)
  rms = _fast_rsq(total_sq * (1.0 / features.cast(dtypes.float32)) + norm_eps)
  is_comb = lane < count * count
  idx = is_comb.where(lane + 2 * count, (lane - count * count).minimum(2 * count - 1).maximum(0))
  which = is_comb.where(2, (lane < count * count + count).where(0, 1))
  s = which.eq(0).where(scale[0].load().float(), which.eq(1).where(scale[1].load().float(), scale[2].load().float()))
  v = mix[token, idx].load() * rms * s + base[idx].load().float()
  sig = _fast_rcp(UOp.const(1.0, dtypes.float32) + _fast_exp2(v * UOp.const(-LOG2E, dtypes.float32)))
  prepost = which.eq(0).where(sig + eps, sig * UOp.const(2.0, dtypes.float32))
  def rowsum(x:UOp, op=lambda a, b: a + b) -> UOp: return op(y := op(x, _lane_xor(x, 1)), _lane_xor(y, 2))
  def colsum(x:UOp) -> UOp: return (y := x + _lane_xor(x, 4)) + _lane_xor(y, 8)
  e = _fast_exp2((v - rowsum(v, _fast_fmax)) * UOp.const(LOG2E, dtypes.float32))
  c = e * _fast_rcp(rowsum(e)) + eps
  reg = UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG)
  reg = reg.after(reg[0].store(c))
  it = UOp.range(2 * iters - 1, 8)
  cv = reg.after(it)[0].load()
  c = reg.after(reg[0].store(cv * _fast_rcp((it % 2).eq(0).where(colsum(cv), rowsum(cv)) + eps)).end(it))[0].load()
  out_idx = is_comb.where(lane + 2 * count, lane - count * count)
  return out[token, out_idx.valid(lane < count * count + 2 * count)].store(is_comb.where(c, prepost)).end(r_w0, lane, wave, token).sink(
    arg=KernelInfo(name="hc_coeffs", opts_to_apply=()))
