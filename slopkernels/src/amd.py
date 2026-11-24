import functools
import math

from tinygrad import UOp
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

BLOCK_M, BLOCK_N, WARP_SIZE = 32, 32, 32
WMMA_M, WMMA_N, WMMA_K = 16, 16, 16
WAVES_M, WAVES_N, LANES_PER_WAVE_M, LANES_PER_WAVE_N = 2, 2, 2, 16
WMMA_ACC, THREADS_PER_BLOCK = WMMA_M // LANES_PER_WAVE_M, WARP_SIZE * WAVES_M * WAVES_N
LDS_PAD, WMMA_ARG, LOG2E = 4, ((WMMA_N, WMMA_M, WMMA_K), 32), math.log2(math.e)

def _unbind(v:UOp) -> UOp: return v.unbind_all()[0]

def _lane_xor(val:UOp, mask:int) -> UOp:
  swz = UOp.native(f"__builtin_amdgcn_ds_swizzle({{0}}, {0x1f | mask<<10})", shape=val.shape, dtype=dtypes.int32, fmt=True)
  return swz.call(val.bitcast(dtypes.int32)).bitcast(val.dtype)

def warp_reduce(val:UOp, op:Ops=Ops.ADD, full_wave:bool=False, masks:tuple[int, ...]|None=None) -> UOp:
  if val.op is Ops.INDEX and val.addrspace == AddrSpace.REG: val = val.load()
  if val.op is Ops.REDUCE or any(u.op is Ops.REDUCE for u in val.src):
    tmp = UOp.alloc((1,), val.dtype, addrspace=AddrSpace.REG)
    val = tmp.after(tmp[0].store(val))[0].load()
  assert val.dtype in (dtypes.float32, dtypes.int32, dtypes.uint32)
  for mask in ((16, 8, 4, 2, 1) if full_wave else (8, 4, 2, 1)) if masks is None else masks:
    swizzle = UOp.native(f"__builtin_amdgcn_ds_swizzle({{0}}, {0x1f | mask<<10})", shape=val.shape, dtype=dtypes.int32, fmt=True)
    val = val.alu(op, swizzle.call(val.bitcast(dtypes.int32)).bitcast(val.dtype))
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

# ******** tiny dense fp16 gemv ********

@functools.cache
def _amd_f16_gemv_kernel(out:UOp, w:UOp, x:UOp, *, in_blks:UOp, out_features:UOp, tokens:UOp, val_chunk:UOp) -> UOp:
  lanes = WARP_SIZE
  out_row, t = UOp.range(out_features, 0, AxisType.GLOBAL), UOp.range(tokens, 5, AxisType.GLOBAL)
  lane = UOp.range(lanes, 2, axis_type=AxisType.LOCAL)
  acc = _reg((1,), 0)
  i_r, u = UOp.range(in_blks, 4), UOp.range(val_chunk, 6, AxisType.UPCAST)
  col = (i_r * lanes + lane) * val_chunk + u
  dot = (w[out_row, col].load().float() * x[t, col].load().float()).reduce(u, arg=(Ops.ADD, 0))
  upd = acc[0].store(acc.after(i_r)[0].load() + dot).end(i_r)
  total = warp_reduce(acc.after(upd)[0].load(), full_wave=True)
  return out[t, out_row.valid(lane.eq(0))].store(total.cast(out.dtype)).end(t, out_row, lane).sink(
    arg=KernelInfo(name="linear_f16_gemv", opts_to_apply=()))

# ******** flash attention on the KV cache ********

@functools.cache
def _amd_flash_attention_decode_partial(out:UOp, stats:UOp, q:UOp, cache_kv:UOp, *,
                                        valid_kv_len:UOp, d_blks:UOp, group:UOp, waves:UOp) -> UOp:
  valid_kv_len = _unbind(valid_kv_len)
  _, B, H_KV, _, _ = cache_kv.shape
  D, DPL, CHUNK, PARTIALS, SEC = d_blks * 64, d_blks * 2, 64, out.shape[2], 64 // waves
  scale = D.cast(dtypes.float32).rsqrt()
  total_chunks = (valid_kv_len + CHUNK - 1) // CHUNK
  live_chunks = total_chunks.minimum(PARTIALS)
  block_bhkv, block_chunk = UOp.range(B * H_KV, 0, AxisType.GLOBAL), UOp.range(live_chunks, 1, AxisType.GLOBAL)
  lane, wave = UOp.range(WARP_SIZE, -1, axis_type=AxisType.WARP), UOp.range(waves, 3, axis_type=AxisType.LOCAL)
  b, kv_head = block_bhkv // H_KV, block_bhkv % H_KV
  d_u = UOp.range(DPL, 11, AxisType.UPCAST)
  zerof = UOp.const(0, dtypes.float)
  # per-lane query fragments for every GQA head, kept in registers
  qf_reg = UOp.alloc((group, DPL), dtypes.float, addrspace=AddrSpace.REG)
  h_q = UOp.range(group, 15)
  qf_reg = qf_reg.after(qf_reg[h_q, d_u].store(q[b, kv_head * group + h_q, 0, lane * DPL + d_u].load().float()).end(d_u, h_q))
  # A finite initial max keeps fully masked waves from computing exp(-inf - -inf).
  h_init = UOp.range(group, 16)
  acc_reg = UOp.alloc((group, DPL), dtypes.float, addrspace=AddrSpace.REG)
  max_reg = UOp.alloc((group,), dtypes.float, addrspace=AddrSpace.REG)
  sum_reg = UOp.alloc((group,), dtypes.float, addrspace=AddrSpace.REG)
  acc_reg = acc_reg.after(acc_reg[h_init, d_u].store(zerof).end(d_u, h_init))
  max_reg = max_reg.after(max_reg[h_init].store(UOp.const(-1e30, dtypes.float)).end(h_init))
  sum_reg = sum_reg.after(sum_reg[h_init].store(zerof).end(h_init))
  n_rounds = (block_chunk < total_chunks).where((total_chunks - 1 - block_chunk) // PARTIALS + 1, UOp.const(0, dtypes.weakint))
  chunk_round = UOp.range(n_rounds, 4)
  chunk_id = block_chunk + chunk_round * PARTIALS
  scores_reg = UOp.alloc((SEC, group), dtypes.float, addrspace=AddrSpace.REG)
  v_reg = UOp.alloc((SEC, DPL), dtypes.float, addrspace=AddrSpace.REG)
  j_s, h_s = UOp.range(SEC, 12), UOp.range(group, 10)
  key_s = chunk_id * CHUNK + wave * SEC + j_s
  valid_s = key_s < valid_kv_len
  kfrag = cache_kv[0, b, kv_head, key_s, lane * DPL + d_u].load().float()
  vfrag = valid_s.where(cache_kv[1, b, kv_head, key_s, lane * DPL + d_u].load().float(), zerof)
  dot_lane = (qf_reg[h_s, d_u].load() * kfrag).reduce(d_u, arg=(Ops.ADD, 0)) * scale
  score_s = valid_s.where(warp_reduce(dot_lane, full_wave=True), UOp.const(-1e30, dtypes.float))
  pass1 = UOp.group(scores_reg.after(chunk_round)[j_s, h_s].store(score_s).end(h_s),
                    v_reg.after(chunk_round)[j_s, d_u].store(vfrag).end(d_u)).end(j_s)
  scores_r, v_r = scores_reg.after(pass1), v_reg.after(pass1)
  prev_acc, prev_max, prev_sum = acc_reg.after(chunk_round), max_reg.after(chunk_round), sum_reg.after(chunk_round)
  h_u, j_u = UOp.range(group, 17, AxisType.UPCAST), UOp.range(SEC, 18, AxisType.UPCAST)
  key_u = chunk_id * CHUNK + wave * SEC + j_u
  valid_u = key_u < valid_kv_len
  sc_u = scores_r[j_u, h_u].load()
  row_max = prev_max[h_u].load().maximum(sc_u.reduce(j_u, arg=(Ops.MAX, 0)))
  alpha = ((prev_max[h_u].load() - row_max) * LOG2E).exp2()
  beta = valid_u.where(((sc_u - row_max) * LOG2E).exp2(), zerof)
  new_acc = alpha * prev_acc[h_u, d_u].load() + (beta * v_r[j_u, d_u].load()).reduce(j_u, arg=(Ops.ADD, 0))
  new_sum = alpha * prev_sum[h_u].load() + beta.reduce(j_u, arg=(Ops.ADD, 0))
  update = UOp.group(acc_reg[h_u, d_u].store(new_acc).end(h_u, d_u), max_reg[h_u].store(row_max).end(h_u),
                     sum_reg[h_u].store(new_sum).end(h_u)).end(chunk_round)
  acc_reg, max_reg, sum_reg = acc_reg.after(update), max_reg.after(update), sum_reg.after(update)
  # exchange across the block's waves through LDS (fp16 halves LDS so more blocks fit per CU)
  acc_lds = UOp.alloc((waves, group, D + LDS_PAD), dtypes.half, addrspace=AddrSpace.LOCAL)[:, :, :D]
  ml_lds = UOp.alloc((waves, group, 2), dtypes.float, addrspace=AddrSpace.LOCAL)
  lds_acc = acc_lds.reshape(waves, group, WARP_SIZE, DPL)
  # Normalize before fp16 to avoid overflow. Nonempty waves have sum >= 1; empty waves keep their zero accumulator.
  s_acc = lds_acc[wave, h_u, lane, d_u].store((acc_reg[h_u, d_u].load() / sum_reg[h_u].load().maximum(1.0)).cast(dtypes.half)).end(h_u, d_u)
  s_m, s_l = ml_lds[wave, h_u, 0].store(max_reg[h_u].load()).end(h_u), ml_lds[wave, h_u, 1].store(sum_reg[h_u].load()).end(h_u)
  acc_lds, ml_lds = acc_lds.after(s_acc, s_m, s_l), ml_lds.after(s_acc, s_m, s_l)
  tid, w_u = wave * WARP_SIZE + lane, UOp.range(waves, 14, AxisType.UPCAST)
  fi = UOp.range((group * D + waves * WARP_SIZE - 1) // (waves * WARP_SIZE), 13)
  flat = tid + fi * (waves * WARP_SIZE)
  h_f, d_f = (flat // D).minimum(group - 1), flat % D
  m_w = ml_lds[w_u, h_f, 0].load()
  M_val = m_w.reduce(w_u, arg=(Ops.MAX, 0))
  val = (((m_w - M_val) * LOG2E).exp2() * ml_lds[w_u, h_f, 1].load() * acc_lds[w_u, h_f, d_f].load().float()).reduce(w_u, arg=(Ops.ADD, 0))
  out_store = out[b, (kv_head * group + h_f).valid(flat < group * D), block_chunk, d_f].store(val).end(fi)
  hstat_safe = tid.minimum(group - 1)
  ms_w = ml_lds[w_u, hstat_safe, 0].load()
  Ms_val = ms_w.reduce(w_u, arg=(Ops.MAX, 0))
  Ls_val = (((ms_w - Ms_val) * LOG2E).exp2() * ml_lds[w_u, hstat_safe, 1].load()).reduce(w_u, arg=(Ops.ADD, 0))
  q_head = (kv_head * group + hstat_safe).valid(tid < group)
  return UOp.group(out_store, stats[b, q_head, block_chunk, 0].store(Ms_val), stats[b, q_head, block_chunk, 1].store(Ls_val)) \
    .end(lane, wave, block_chunk, block_bhkv).sink(arg=KernelInfo(name="flash_decode_partial", opts_to_apply=()))

@functools.cache
def _amd_flash_decode_combine(o:UOp, partial:UOp, stats:UOp, *, valid_kv_len:UOp) -> UOp:
  # one wave per (batch, head, 64-dim tile): every lane redundantly weights its chunks; no cross-lane traffic
  valid_kv_len = _unbind(valid_kv_len)
  B, H, C, D = partial.shape
  block_bh, block_dt = UOp.range(B * H, 0, AxisType.GLOBAL), UOp.range(D // 64, 1, AxisType.GLOBAL)
  lane, i_u = UOp.range(WARP_SIZE, 2, axis_type=AxisType.LOCAL), UOp.range(2, 102, AxisType.UPCAST)
  b, h = block_bh // H, block_bh % H
  live = ((valid_kv_len + 63) // 64).minimum(C)
  dim = block_dt * 64 + lane * 2 + i_u
  chunk = UOp.range(live, 100)
  chunk_max = _reg((1,), -math.inf)
  update0 = chunk_max[0].store(chunk_max.after(chunk)[0].load().maximum(stats[b, h, chunk, 0].load())).end(chunk)
  chunk_max = chunk_max.after(update0)
  chunk2 = UOp.range(live, 101)
  acc = UOp.alloc((2,), dtypes.float, addrspace=AddrSpace.REG)
  acc = acc.after(acc[i_u].store(UOp.const(0, dtypes.float)).end(i_u))
  weight_sum = _reg((1,), 0)
  w = ((stats[b, h, chunk2, 0].load() - chunk_max[0].load()) * LOG2E).exp2()
  update1 = UOp.group(acc[i_u].store(acc.after(chunk2)[i_u].load() + w * partial[b, h, chunk2, dim].load()).end(i_u),
                      weight_sum[0].store(weight_sum.after(chunk2)[0].load() + w * stats[b, h, chunk2, 1].load())).end(chunk2)
  acc, weight_sum = acc.after(update1), weight_sum.after(update1)
  return o[b, h, 0, dim].store((acc[i_u].load() / weight_sum[0].load()).cast(o.dtype)).end(i_u, lane, block_dt, block_bh).sink(
    arg=KernelInfo(name="flash_decode_combine", opts_to_apply=()))

def _wmma_fragment(fragment:UOp, lane:UOp, rdna4:bool) -> UOp:
  return fragment.reshape(2, 2, 4)[:, lane // 16, :].reshape(8) if rdna4 else fragment

@functools.cache
def _amd_flash_attention(o:UOp, q:UOp, cache:UOp, *, valid_kv_len:UOp, q_start:UOp, d_blks:UOp, rdna4:bool=False) -> UOp:
  valid_kv_len, q_start = _unbind(valid_kv_len), _unbind(q_start)
  BH, M, D = q.shape
  _, B, H_KV, physical_n, _ = cache.shape
  k, v = cache[0].reshape(B * H_KV, physical_n, D), cache[1].reshape(B * H_KV, physical_n, D)
  gqa_group = BH // k.shape[0]
  TM, TN, TD = BLOCK_M // (WAVES_M * LANES_PER_WAVE_M), BLOCK_N // LANES_PER_WAVE_N, d_blks * 2
  SCALE = (d_blks * 64).cast(dtypes.float32).rsqrt()
  block_bh, block_m = UOp.range(BH, 0, AxisType.GLOBAL), UOp.range(M // BLOCK_M, 1, AxisType.GLOBAL)
  kv_head = block_bh // gqa_group
  q_u32 = q.flatten().bitcast(dtypes.uint32)
  k_u32 = cache.flatten().bitcast(dtypes.uint32)
  o = o.reshape(BH, M // BLOCK_M, BLOCK_M, D)[block_bh, block_m]
  v = v[kv_head]
  wave_m, wave_n, lane = UOp.range(WAVES_M, 2, AxisType.LOCAL), UOp.range(WAVES_N, 3, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  tid, lane_m, lane_n = (wave_m * WAVES_N + wave_n) * WARP_SIZE + lane, lane // LANES_PER_WAVE_N, lane % LANES_PER_WAVE_N
  Q_lds_raw = UOp.alloc((BLOCK_M * (D + LDS_PAD),), dtypes.half, addrspace=AddrSpace.LOCAL)
  Q_lds = Q_lds_raw.reshape(BLOCK_M, D + LDS_PAD)[:, :D]
  qi, q4 = UOp.range(d_blks * 2, 10), UOp.range(4, 11, AxisType.UPCAST)
  q_off = tid * (d_blks * 8) + qi * 4 + q4
  q_store = Q_lds_raw.bitcast(dtypes.uint32).reshape(BLOCK_M, d_blks * 32 + 2)[:, :d_blks * 32].reshape(BLOCK_M * (d_blks * 32))[q_off].store(
    q_u32[(block_bh * (M // BLOCK_M) + block_m) * (BLOCK_M * d_blks * 32) + q_off].load()).end(q4, qi)
  Q_lds = Q_lds.after(q_store)
  KP_lds_raw = UOp.alloc((BLOCK_N * (D + LDS_PAD),), dtypes.half, addrspace=AddrSpace.LOCAL)
  KP_lds = KP_lds_raw.reshape(BLOCK_N, D + LDS_PAD)
  KV_lds = KP_lds[:, :D]
  acc, m_i, l_i = _reg((TM, TD), 0), _reg((TM,), -1e30), _reg((TM,), 0)
  n_end = ((q_start + (block_m + 1) * BLOCK_M).minimum(valid_kv_len) + BLOCK_N - 1) // BLOCK_N
  n_tile = UOp.range(n_end, 100)
  ki, k4 = UOp.range(d_blks * 2, 12), UOp.range(4, 13, AxisType.UPCAST)
  k_off = tid * (d_blks * 8) + ki * 4 + k4
  k_lds_u32 = KP_lds_raw.after(n_tile).bitcast(dtypes.uint32).reshape(BLOCK_N, d_blks * 32 + 2)[:, :d_blks * 32].reshape(BLOCK_N * (d_blks * 32))
  k_store = k_lds_u32[k_off].store(k_u32[(kv_head * physical_n + n_tile * BLOCK_N) * (d_blks * 32) + k_off].load()).end(k4, ki)
  KV_lds_k = KV_lds.after(k_store)
  S_reg = _reg((TM, TN), 0, n_tile)
  k_qk, tm1, tn1 = UOp.range(d_blks * 4, 101), UOp.range(TM // WMMA_ACC, 200), UOp.range(TN, 201)
  S_frag = S_reg.reshape(TM // WMMA_ACC, WMMA_ACC, TN).permute(0, 2, 1)[tm1, tn1]
  q_frag = Q_lds.reshape(WAVES_M, TM // WMMA_ACC, WMMA_M, d_blks * 4, WMMA_K)[wave_m, tm1, lane_n, k_qk]
  k_frag = KV_lds_k.reshape(TN, WMMA_N, d_blks * 4, WMMA_K)[tn1, lane_n, k_qk]
  # All waves must finish reading Q/K before their shared memory is reused for P/V.
  q_frag, k_frag = (_wmma_fragment(f, lane, rdna4) for f in (q_frag, k_frag))
  qk_done = S_frag.store(UOp.wmma(q_frag, k_frag, S_frag.after(k_qk), *WMMA_ARG)).end(tm1, tn1).end(k_qk).barrier()
  S_reg = S_reg.after(qk_done, S_reg.store(S_reg * SCALE))
  rm, rn = UOp.range(TM, 250), UOp.range(TN, 251)
  q_local = wave_m * WMMA_M + (lane_m * TM + rm if rdna4 else rm * LANES_PER_WAVE_M + lane_m)
  k_local = rn * LANES_PER_WAVE_N + lane_n
  q_idx, k_idx = q_start + block_m * BLOCK_M + q_local, n_tile * BLOCK_N + k_local
  causal = (k_idx <= q_idx) & (k_idx < valid_kv_len)
  S_reg = S_reg.after(S_reg[rm, rn].store(causal.where(S_reg[rm, rn], S_reg[rm, rn].const_like(-math.inf))).end(rm, rn))
  m_ij, rm2 = _reg((TM,), -math.inf, n_tile), UOp.range(TN, 261)
  m_ij = m_ij.after(m_ij.store(m_ij.after(rm2).maximum(S_reg[:, rm2])).end(rm2))
  ri_w = UOp.range(TM, 270)
  m_ij = m_ij.after(m_ij[ri_w].store(warp_reduce(m_ij[ri_w], Ops.MAX)).end(ri_w))
  m_new_reg = UOp.alloc((TM,), dtypes.float, addrspace=AddrSpace.REG)
  alpha_reg, ri_m = UOp.alloc((TM,), dtypes.float, addrspace=AddrSpace.REG), UOp.range(TM, 275)
  mn_v = m_i[ri_m].maximum(m_ij[ri_m])
  mn_safe = mn_v.maximum(-1e30)
  al_v = ((m_i[ri_m] - mn_safe) * LOG2E).exp2()
  m_upd = UOp.group(m_new_reg[ri_m].store(mn_safe), alpha_reg[ri_m].store(al_v), m_i[ri_m].store(mn_v)).end(ri_m)
  m_new_r, alpha_r, m_i = m_new_reg.after(m_upd), alpha_reg.after(m_upd), m_i.after(m_upd)
  tile_max = m_new_r.reshape(TM, 1).expand(TM, TN)
  S_reg = S_reg.after(S_reg.store(((S_reg - tile_max) * LOG2E).exp2()))
  p_local, ri_ws, rn_u = _reg((TM,), 0, n_tile), UOp.range(TM, 295), UOp.range(TN, 296, AxisType.UPCAST)
  p_sum = p_local.after(p_local[ri_ws].store(warp_reduce(S_reg[ri_ws, rn_u].load().reduce(rn_u, arg=(Ops.ADD, 0)))).end(ri_ws))
  P_lds = KP_lds.after(qk_done).flatten()[:WAVES_N * BLOCK_M * BLOCK_N].reshape(WAVES_N, BLOCK_M, BLOCK_N)
  # gfx11 distributes even/odd rows between half-waves; gfx12 distributes the low/high eight rows.
  row_shape = (LANES_PER_WAVE_M, TM) if rdna4 else (TM, LANES_PER_WAVE_M)
  row_lane, row_elem = (2, 3) if rdna4 else (3, 2)
  P_write = P_lds.reshape(WAVES_N, WAVES_M, *row_shape, TN, LANES_PER_WAVE_N).permute(1, 0, row_lane, 5, row_elem, 4) \
    .reshape(THREADS_PER_BLOCK, TM, TN)
  P_store = P_write[tid].store(S_reg.cast(dtypes.half))
  ri4, rj4 = UOp.range(TM, 330), UOp.range(TD, 331)
  correction = UOp.group(acc[ri4, rj4].store(alpha_r[ri4] * acc[ri4, rj4]).end(rj4),
                         l_i[ri4].store(alpha_r[ri4] * l_i[ri4] + p_sum[ri4])).end(ri4)
  acc, l_i = acc.after(correction), l_i.after(correction)
  V_lds = UOp.alloc((D, BLOCK_N + LDS_PAD), dtypes.half, addrspace=AddrSpace.LOCAL)[:, :BLOCK_N]
  V_copy = V_lds.after(qk_done).permute(1, 0)
  vi, v4 = UOp.range(d_blks * 4, 14), UOp.range(4, 15, AxisType.UPCAST)
  v_off = tid * (d_blks * 16) + vi * 4 + v4
  in_v = (n_tile * BLOCK_N + v_off // D) < valid_kv_len
  v_store = V_copy.reshape(BLOCK_N * D)[v_off].store(
    in_v.where(v.reshape(physical_n * D)[n_tile * (BLOCK_N * D) + v_off].load(), UOp.const(0, dtypes.half))).end(v4, vi)
  P_lds, V_lds = P_lds.after(P_store, v_store), V_lds.after(P_store, v_store)
  k_pv, tm2, tn2 = UOp.range(BLOCK_N // WMMA_K, 400), UOp.range(TM // WMMA_ACC, 401), UOp.range(TD, 402)
  acc_frag = acc.reshape(TM // WMMA_ACC, WMMA_ACC, TD).permute(0, 2, 1)[tm2, tn2]
  p_frag = P_lds[wave_n].reshape(WAVES_M, TM // WMMA_ACC, WMMA_M, BLOCK_N // WMMA_K, WMMA_K)[wave_m, tm2, lane_n, k_pv]
  v_frag = V_lds.reshape(WAVES_N, TD, WMMA_N, BLOCK_N // WMMA_K, WMMA_K)[wave_n, tn2, lane_n, k_pv]
  p_frag, v_frag = (_wmma_fragment(f, lane, rdna4) for f in (p_frag, v_frag))
  n_tile_end = acc_frag.store(UOp.wmma(p_frag, v_frag, acc_frag.after(k_pv), *WMMA_ARG)).end(tm2, tn2).end(k_pv).barrier().end(n_tile)
  acc, l_i = acc.after(n_tile_end), l_i.after(n_tile_end)
  o = o.reshape(WAVES_M, *row_shape, WAVES_N, TD, LANES_PER_WAVE_N) \
    .permute(0, 3, row_lane - 1, 5, row_elem - 1, 4).reshape(THREADS_PER_BLOCK, TM, TD)
  rm_o, rd_o = UOp.range(TM, 420), UOp.range(TD, 421)
  return o[tid, rm_o, rd_o].store(acc[rm_o, rd_o].load() / l_i[rm_o].load()).end(rm_o, rd_o).end(wave_m, wave_n, lane) \
    .end(block_m, block_bh).sink(arg=KernelInfo(opts_to_apply=()))

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

# ******** mHC coefficients (fused Sinkhorn) ********

def _hc_coeffs_kernel(out:UOp, mix:UOp, sq:UOp, base:UOp, scale:UOp, count:int, iters:UOp, eps:UOp, norm_eps:UOp, features:UOp) -> UOp:
  assert count == 4, "the swizzles assume 4 streams"
  token, lane = UOp.range(out.shape[0], 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP)
  groups = sq.shape[1]
  total = sum((sq[token, (lane + 32 * i).valid(lane + 32 * i < groups)].load() for i in range((groups + 31) // 32)), UOp.const(0.0, dtypes.float))
  rms = (warp_reduce(total, full_wave=True) * (1.0 / features.cast(dtypes.float32)) + norm_eps).rsqrt()
  is_comb = lane < count * count
  idx = is_comb.where(lane + 2 * count, (lane - count * count).minimum(2 * count - 1).maximum(0))
  which = is_comb.where(2, (lane < count * count + count).where(0, 1))
  s = which.eq(0).where(scale[0].load().float(), which.eq(1).where(scale[1].load().float(), scale[2].load().float()))
  v = mix[token, idx].load() * rms * s + base[idx].load().float()
  sig = (1.0 + (v * -LOG2E).exp2()).reciprocal()
  prepost = which.eq(0).where(sig + eps, sig * 2.0)
  def rowsum(x:UOp, op=lambda a, b: a + b) -> UOp: return op(y := op(x, _lane_xor(x, 1)), _lane_xor(y, 2))
  def colsum(x:UOp) -> UOp: return (y := x + _lane_xor(x, 4)) + _lane_xor(y, 8)
  e = ((v - rowsum(v, UOp.maximum)) * LOG2E).exp2()
  c = e * rowsum(e).reciprocal() + eps
  reg = UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG)
  reg = reg.after(reg[0].store(c))
  it = UOp.range(2 * iters - 1, 4)
  cv = reg.after(it)[0].load()
  c = reg.after(reg[0].store(cv * ((it % 2).eq(0).where(colsum(cv), rowsum(cv)) + eps).reciprocal()).end(it))[0].load()
  valid = lane < count * count + 2 * count
  out_idx = is_comb.where(lane + 2 * count, lane - count * count)
  return out[token, out_idx.valid(valid)].store(is_comb.where(c, prepost)).end(token, lane).sink(
    arg=KernelInfo(name="hc_coeffs", opts_to_apply=()))
