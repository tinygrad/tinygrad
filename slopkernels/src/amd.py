import functools
import math
from typing import cast

from tinygrad import UOp
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

BLOCK_M, BLOCK_N, WARP_SIZE = 32, 32, 32
WMMA_M, WMMA_N, WMMA_K = 16, 16, 16
WAVES_M, WAVES_N, LANES_PER_WAVE_M, LANES_PER_WAVE_N = 2, 2, 2, 16
WMMA_ACC, THREADS_PER_BLOCK = WMMA_M // LANES_PER_WAVE_M, WARP_SIZE * WAVES_M * WAVES_N
LDS_PAD, WMMA_ARG, LOG2E = 4, ((WMMA_N, WMMA_M, WMMA_K), 32), math.log2(math.e)

def _unbind(v:int|UOp) -> int|UOp: return v.unbind_all()[0] if isinstance(v, UOp) else v

def _lane_xor(val:UOp, mask:int) -> UOp:
  swz = UOp.native(f"__builtin_amdgcn_ds_swizzle({{0}}, {0x1f | mask<<10})", shape=val.shape, dtype=dtypes.int32, fmt=True)
  return swz.call(val.bitcast(dtypes.int32)).bitcast(val.dtype)

def warp_reduce(val:UOp, op:Ops=Ops.ADD, full_wave:bool=False, masks:tuple[int, ...]|None=None) -> UOp:
  if val.op is Ops.INDEX and val.addrspace == AddrSpace.REG: val = val.load()
  assert val.dtype in (dtypes.float32, dtypes.int32, dtypes.uint32)
  for mask in ((16, 8, 4, 2, 1) if full_wave else (8, 4, 2, 1)) if masks is None else masks:
    swizzle = UOp.native(f"__builtin_amdgcn_ds_swizzle({{0}}, {0x1f | mask<<10})", shape=val.shape, dtype=dtypes.int32, fmt=True)
    val = val.alu(op, swizzle.call(val.bitcast(dtypes.int32)).bitcast(val.dtype))
  return val

def _reg(shape:tuple[int, ...], value:float, dep:UOp|None=None) -> UOp:
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
def _amd_f16_gemv_kernel(out:UOp, w:UOp, x:UOp, *, in_features:UOp, out_features:UOp, tokens:UOp, val_chunk:int) -> UOp:
  lanes = WARP_SIZE
  out_row = UOp.range(out_features, 0, AxisType.GLOBAL)
  lane = UOp.range(lanes, 2, axis_type=AxisType.LOCAL)
  per = in_features // (lanes * val_chunk)
  w, x = w.reshape((out_features, per, lanes * val_chunk)), x.reshape((tokens, per, lanes * val_chunk))
  zero = UOp.const(0, dtypes.float32)
  t = UOp.range(tokens, 5, AxisType.GLOBAL)
  acc = UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG)
  acc = acc.after(acc[0].store(zero))
  i_r = UOp.range(per, 4)
  wv, xv = _vec_load(w[out_row, i_r, lane * val_chunk], val_chunk), _vec_load(x[t, i_r, lane * val_chunk], val_chunk)
  upd = acc[0].store(acc.after(i_r)[0].load() + sum((wv[j] * xv[j] for j in range(val_chunk)), zero)).end(i_r)
  total = warp_reduce(acc.after(upd)[0].load(), full_wave=True)
  return out[t, out_row.valid(lane.eq(0))].store(total.cast(out.dtype)).end(t, out_row, lane).sink(
    arg=KernelInfo(name="linear_f16_gemv", opts_to_apply=()))

# ******** flash attention on the KV cache ********

def _vec_load(ptr:UOp, lanes:int) -> tuple[UOp, ...]:
  if lanes == 1: return (ptr.load().float(),)
  vec = _amd_load(ptr, lanes)
  return tuple(vec[i].float() for i in range(lanes))

@functools.cache
def _amd_flash_attention_decode_partial(out, stats, q, cache_kv, *extra, valid_kv_len, max_kv_len, block_n, waves=4, window=0,
                                        has_rel=False, ring=False, group:int|None=None):
  valid_kv_len = _unbind(valid_kv_len)
  is_q4 = cache_kv.dtype == dtypes.uint32
  _, B, H_KV, N, _ = cast(tuple[int, int, int, int, int], cache_kv.shape)
  _, H, M, D = cast(tuple[int, int, int, int], q.shape)
  assert (1 <= M <= 16 and (not isinstance(H, int) or H % H_KV == 0) and D % WARP_SIZE == 0
          and (ring or not isinstance(N, int) or max_kv_len <= N) and (not isinstance(N, int) or N % block_n == 0))
  if is_q4: assert D == 128 and cache_kv.shape[4] == 17
  G, CHUNK, DPL, WAVES, PARTIALS = (H // H_KV if group is None else group), block_n, D // WARP_SIZE, waves, out.shape[2]
  assert CHUNK % WAVES == 0
  SEC = CHUNK // WAVES  # keys each wave scans independently
  max_span = (valid_kv_len.minimum(window) if isinstance(valid_kv_len, UOp) else min(valid_kv_len, window)) if window else valid_kv_len
  max_total_chunks = (max_span + CHUNK - 1) // CHUNK
  live_chunks = min(max_total_chunks, PARTIALS) if isinstance(max_total_chunks, int) else max_total_chunks.minimum(PARTIALS)
  block_bhkv, block_chunk = UOp.range(B*M*H_KV, 0, AxisType.GLOBAL), UOp.range(live_chunks, 1, AxisType.GLOBAL)
  lane, wave = UOp.range(WARP_SIZE, -1, axis_type=AxisType.WARP), UOp.range(WAVES, 3, axis_type=AxisType.LOCAL)
  bm, kv_head = block_bhkv // H_KV, block_bhkv % H_KV
  b, m = (bm, UOp.const(0, dtypes.weakint)) if M == 1 else (bm // M, bm % M)
  q_valid_len = valid_kv_len if M == 1 else valid_kv_len - (M - 1) + m
  q_start_k = ((q_valid_len - window).maximum(0) if isinstance(q_valid_len, UOp) else max(0, q_valid_len - window)) if window else 0
  q_span = (q_valid_len.minimum(window) if isinstance(q_valid_len, UOp) else min(q_valid_len, window)) if window else q_valid_len
  q_total_chunks = (q_span + CHUNK - 1) // CHUNK
  # per-lane query fragments for every GQA head, kept packed in registers; unpacked at use
  qf = tuple(_vec_load(q[b, kv_head*G+h, m, lane*DPL], DPL) for h in range(G))
  zerof = UOp.const(0, dtypes.float)
  if has_rel:
    r, rel_bias = extra[0], extra[1]
    E, L = cast(tuple[int, int], rel_bias.shape)
    assert L <= WARP_SIZE
    in_l, safe_l = lane < L, lane.minimum(L - 1)
    rf = tuple(in_l.where(r[b, kv_head*G+h, m, safe_l].load().float(), zerof) for h in range(G))
  # Each block scans every PARTIALS-th chunk, keeping an online softmax across rounds (exact per-query window & chunking).
  n_rounds = (block_chunk < q_total_chunks).where((q_total_chunks - 1 - block_chunk) // PARTIALS + 1, UOp.const(0, dtypes.weakint))
  chunk_round = UOp.range(n_rounds, 4)
  chunk_id = block_chunk + chunk_round*PARTIALS
  valids: list[UOp] = []
  scores: list[list[UOp]] = [[zerof]*G for _ in range(SEC)]
  vfrags: list[tuple[UOp, ...]] = [()]*SEC
  vscs: list[UOp] = [zerof]*SEC
  for j in range(SEC):
    key = q_start_k + chunk_id*CHUNK + wave*SEC + j
    valid = key < q_valid_len
    safe_key = (key % N) if ring else (key.minimum(N - 1) if (window or M > 1) else key)
    valids.append(valid)
    if is_q4:
      kw, ksc_w = cache_kv[0, b, kv_head, safe_key, lane // 2].load(), cache_kv[0, b, kv_head, safe_key, 16].load()
      ksc = ((ksc_w >> ((lane // 16) * 16).cast(dtypes.uint32)) & 0xffff).cast(dtypes.uint16).bitcast(dtypes.half).float()
      khw = kw >> ((lane & 1) * 16).cast(dtypes.uint32)
      kfrag = tuple((((khw >> (4 * i)) & 15).cast(dtypes.int32) - 8).float() for i in range(4))
      vw, vsc_w = cache_kv[1, b, kv_head, safe_key, lane // 2].load(), cache_kv[1, b, kv_head, safe_key, 16].load()
      vscs[j] = ((vsc_w >> ((lane // 16) * 16).cast(dtypes.uint32)) & 0xffff).cast(dtypes.uint16).bitcast(dtypes.half).float()
      vhw = vw >> ((lane & 1) * 16).cast(dtypes.uint32)
      vfrags[j] = tuple(valid.where((((vhw >> (4 * i)) & 15).cast(dtypes.int32) - 8).float(), zerof) for i in range(4))
    else:
      kfrag = _vec_load(cache_kv[0, b, kv_head, safe_key, lane*DPL], DPL)
      # V is prefetched in the score pass so both streams are in flight together
      vfrags[j] = tuple(valid.where(v, zerof) for v in _vec_load(cache_kv[1, b, kv_head, safe_key, lane*DPL], DPL))
    if has_rel:
      dist = (q_valid_len - 1) - key
      rb = (in_l & (dist >= 0) & (dist < E)).where(rel_bias[dist.maximum(0).minimum(E - 1), safe_l].load().float(), zerof)
    for h in range(G):
      dot_lane = sum((qf[h][i]*kfrag[i] for i in range(DPL)), zerof) * ((ksc * (1/math.sqrt(D))) if is_q4 else (1/math.sqrt(D)))
      if has_rel:
        dot_lane = UOp.native("__builtin_fmaf({}, {}, {})", shape=dot_lane.shape, dtype=dtypes.float32, fmt=True).call(
          rf[h], rb, dot_lane, inline=True)
      s = warp_reduce(dot_lane, full_wave=True)
      scores[j][h] = valid.where(s, UOp.const(-1e30, dtypes.float))
  # A finite initial max keeps fully masked waves from computing exp(-inf - -inf).
  acc_reg, max_reg, sum_reg = _reg((G, DPL), 0), _reg((G,), -1e30), _reg((G,), 0)
  prev_acc, prev_max, prev_sum = acc_reg.after(chunk_round), max_reg.after(chunk_round), sum_reg.after(chunk_round)
  row_max = [functools.reduce(UOp.maximum, (scores[j][h] for j in range(SEC)), prev_max[h].load()) for h in range(G)]
  # Rescale the previous rounds to the new max, then accumulate this round's keys.
  alpha = [((prev_max[h].load()-row_max[h])*LOG2E).exp2() for h in range(G)]
  accs = [[alpha[h]*prev_acc[h, i].load() for i in range(DPL)] for h in range(G)]
  row_sums = [alpha[h]*prev_sum[h].load() for h in range(G)]
  for j in range(SEC):
    for h in range(G):
      beta = valids[j].where(((scores[j][h]-row_max[h])*LOG2E).exp2(), zerof)
      bv = (beta * vscs[j]) if is_q4 else beta
      accs[h] = [a + bv*v for a, v in zip(accs[h], vfrags[j])]
      row_sums[h] = row_sums[h] + beta
  update = UOp.group(acc_reg.store(UOp.stack(*(x for acc in accs for x in acc)).reshape(G, DPL)),
                     max_reg.store(UOp.stack(*row_max)), sum_reg.store(UOp.stack(*row_sums))).end(chunk_round)
  acc_reg, max_reg, sum_reg = acc_reg.after(update), max_reg.after(update), sum_reg.after(update)
  # exchange across the block's waves through LDS (fp16 halves LDS so more blocks fit per CU)
  # Matching cache/LDS strides can reuse a loop-local cache index outside the loop. Pad that layout.
  acc_lds = UOp.alloc((WAVES, G, D + (LDS_PAD if G == SEC else 0)), dtypes.half, addrspace=AddrSpace.LOCAL)[:, :, :D]
  ml_lds = UOp.alloc((WAVES, G, 2), dtypes.float, addrspace=AddrSpace.LOCAL)
  lds_acc = acc_lds.reshape(WAVES, G, WARP_SIZE, DPL)
  # Normalize before fp16 to avoid overflow. Nonempty waves have sum >= 1; empty waves keep their zero accumulator.
  stores = [lds_acc[wave, h, lane].store((acc_reg[h].load() / sum_reg[h].load().maximum(1)).cast(dtypes.half)) for h in range(G)]
  # NOTE: duplicate stores of the same value from every lane are harmless here
  stores += [ml_lds[wave, h, i].store(x) for h in range(G) for i, x in enumerate((max_reg[h].load(), sum_reg[h].load()))]
  acc_lds, ml_lds = acc_lds.after(*stores), ml_lds.after(*stores)
  tid = wave*WARP_SIZE + lane
  final_stores:list[UOp] = []
  for i in range(-(-G*D//(WAVES*WARP_SIZE))):
    flat = tid + i*WAVES*WARP_SIZE
    h, d = flat // D, flat % D
    h_safe = h.minimum(G - 1) if G*D % (WAVES*WARP_SIZE) else h
    M_val = functools.reduce(UOp.maximum, (ml_lds[w, h_safe, 0].load() for w in range(WAVES)))
    # LDS holds normalized values; restore each wave's sum before combining.
    val = sum((((ml_lds[w, h_safe, 0].load()-M_val)*LOG2E).exp2() * ml_lds[w, h_safe, 1].load() * acc_lds[w, h_safe, d].load().float()
               for w in range(WAVES)), zerof)
    oidx = out[bm, kv_head*G + h_safe, block_chunk, d]
    if G*D % (WAVES*WARP_SIZE): oidx = out[bm, (kv_head*G + h_safe).valid(flat < G*D), block_chunk, d]
    final_stores.append(oidx.store(val))
  for i in range(-(-G//(WAVES*WARP_SIZE))):
    hstat = tid + i*WAVES*WARP_SIZE
    needs_mask = (i + 1)*WAVES*WARP_SIZE > G
    hstat_safe = hstat.minimum(G - 1) if needs_mask else hstat
    M_val = functools.reduce(UOp.maximum, (ml_lds[w, hstat_safe, 0].load() for w in range(WAVES)))
    L_val = sum((((ml_lds[w, hstat_safe, 0].load()-M_val)*LOG2E).exp2() * ml_lds[w, hstat_safe, 1].load() for w in range(WAVES)), zerof)
    q_head = (kv_head*G + hstat_safe).valid(hstat < G) if needs_mask else kv_head*G + hstat_safe
    final_stores += [stats[bm, q_head, block_chunk, 0].store(M_val), stats[bm, q_head, block_chunk, 1].store(L_val)]
  return UOp.group(*final_stores).end(lane, wave, block_chunk, block_bhkv).sink(arg=KernelInfo(name="flash_decode_partial", opts_to_apply=()))

@functools.cache
def _amd_flash_decode_combine(o:UOp, partial:UOp, stats:UOp, *extra:UOp, valid_kv_len:int|UOp, window:int=0,
                              has_sink:bool=False, has_gate:bool=False) -> UOp:
  # one wave per (batch, head, 64-dim tile): every lane redundantly weights its chunks; no cross-lane traffic
  valid_kv_len = _unbind(valid_kv_len)
  BM, H, C, D = partial.shape
  M = cast(int, o.shape[1] if has_gate else o.shape[2])
  B = cast(int, o.shape[0])
  DT = 64 if (not isinstance(D, int) or D % 64 == 0) else WARP_SIZE  # dims per block
  assert not isinstance(D, int) or D % DT == 0
  block_bh, block_dt = UOp.range(BM*H, 0, AxisType.GLOBAL), UOp.range(D//DT, 1, AxisType.GLOBAL)
  lane = UOp.range(WARP_SIZE, 2, axis_type=AxisType.LOCAL)
  bm, h = block_bh // H, block_bh % H
  b, m = (bm, UOp.const(0, dtypes.weakint)) if M == 1 else (bm // M, bm % M)
  q_valid_len = valid_kv_len if M == 1 else valid_kv_len - (M - 1) + m
  q_span = (q_valid_len.minimum(window) if isinstance(q_valid_len, UOp) else min(q_valid_len, window)) if window else q_valid_len
  q_live = (q_span + 63) // 64
  live = min(q_live, C) if isinstance(q_live, int) else q_live.minimum(C)
  NPD = DT // WARP_SIZE  # output dims per lane
  dims = tuple(block_dt*DT + lane*NPD + i for i in range(NPD))
  sink_m = extra[0][b, h, m].load().float() if has_sink else None
  gate_uop = extra[1 if has_sink else 0].reshape(B, M, H, D) if has_gate else None
  chunk = UOp.range(live, 100)
  def iloop(ph, val): return ph[0].store(val) if isinstance(val, UOp) else ph.store(ph.const_like(val))
  chunk_max = UOp.alloc((1,), dtypes.float, addrspace=AddrSpace.REG)
  chunk_max_i = chunk_max.after(iloop(chunk_max, sink_m if sink_m is not None else -math.inf))
  update0 = chunk_max_i.store(chunk_max_i.after(chunk).maximum(stats[bm, h, chunk, 0].load())).end(chunk)
  chunk_max = chunk_max_i.after(update0)
  chunk2 = UOp.range(live, 101)
  acc = UOp.alloc((NPD,), dtypes.float, addrspace=AddrSpace.REG)
  weight_sum = UOp.alloc((1,), dtypes.float, addrspace=AddrSpace.REG)
  init_ws: float|UOp = ((sink_m - chunk_max[0].load())*LOG2E).exp2() if sink_m is not None else 0
  acc_i, weight_sum_i = acc.after(iloop(acc, 0)), weight_sum.after(iloop(weight_sum, init_ws))
  w = ((stats[bm, h, chunk2, 0].load()-chunk_max)*LOG2E).exp2()
  update1 = UOp.group(*[acc_i[i].store(acc_i.after(chunk2)[i].load() + w*partial[bm, h, chunk2, d].load()) for i, d in enumerate(dims)],
                      weight_sum_i[0].store(weight_sum_i.after(chunk2)[0].load() + w*stats[bm, h, chunk2, 1].load())).end(chunk2)
  acc, weight_sum = acc_i.after(update1), weight_sum_i.after(update1)
  inv = 1 / weight_sum[0].load()
  if gate_uop is not None:
    og = o.reshape(B, M, H, D)
    stores = [og[b, m, h, d].store(((acc[i].load() * inv) * (1.0 / (1.0 + (gate_uop[b, m, h, d].load().float() * (-LOG2E)).exp2()))).cast(o.dtype))
              for i, d in enumerate(dims)]
  else:
    stores = [o[b, h, m, d].store((acc[i].load() * inv).cast(o.dtype)) for i, d in enumerate(dims)]
  return UOp.group(*stores).end(lane, block_dt, block_bh).sink(arg=KernelInfo(name="flash_decode_combine", opts_to_apply=()))

def _wmma_fragment(fragment:UOp, lane:UOp, rdna4:bool) -> UOp:
  return fragment.reshape(2, 2, 4)[:, lane//16, :].reshape(8) if rdna4 else fragment

@functools.cache
def _amd_flash_attention(o:UOp, q:UOp, cache:UOp, *extra:UOp, valid_kv_len:int|UOp, q_start:int|UOp|None=None,
                         valid_m:int|UOp|None=None, rdna4:bool=False, window:int=0,
                         has_rel:bool=False, has_sink:bool=False, has_gate:bool=False, ring:bool=False) -> UOp:
  valid_kv_len = _unbind(valid_kv_len)
  q_start = _unbind(q_start) if q_start is not None else None
  BH, M, D = cast(tuple[int, int, int], q.shape)
  valid_m = _unbind(valid_m) if valid_m is not None else M
  is_q4 = cache.dtype == dtypes.uint32
  _, B, H_KV, physical_n, cache_dim = cast(tuple[int, int, int, int, int], cache.shape)
  H = BH // B
  k, v = cache[0].reshape(B*H_KV, physical_n, cache_dim), cache[1].reshape(B*H_KV, physical_n, cache_dim)
  assert k.shape == v.shape and (not isinstance(BH, int) or BH % k.shape[0] == 0) and (D == 128 and cache_dim == 17 if is_q4 else k.shape[2] == D)
  gqa_group = BH // k.shape[0]
  if isinstance(M, int): assert M % BLOCK_M == 0
  assert isinstance(D, int) and D % WMMA_K == 0 and D % LANES_PER_WAVE_N == 0
  TM, TN, TD, SCALE = BLOCK_M//(WAVES_M*LANES_PER_WAVE_M), BLOCK_N//LANES_PER_WAVE_N, D//(WAVES_N*LANES_PER_WAVE_N), 1/math.sqrt(D)
  # query row 0 sits at sequence position q_base (the queries may be padded beyond valid_kv_len - q_base rows)
  q_base = valid_kv_len - M if q_start is None else q_start
  block_bh = UOp.range(BH, 0, AxisType.GLOBAL)
  block_m = UOp.range((valid_m + BLOCK_M - 1) // BLOCK_M, 1, AxisType.GLOBAL)
  kv_head = block_bh // gqa_group
  q = q.reshape(BH, M//BLOCK_M, BLOCK_M, D)[block_bh, block_m]
  if not has_gate: o = o.reshape(BH, M//BLOCK_M, BLOCK_M, D)[block_bh, block_m]
  k, v = k[kv_head], v[kv_head]
  wave_m, wave_n, lane = UOp.range(WAVES_M, 2, AxisType.LOCAL), UOp.range(WAVES_N, 3, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  tid, lane_m, lane_n = (wave_m * WAVES_N + wave_n) * WARP_SIZE + lane, lane // LANES_PER_WAVE_N, lane % LANES_PER_WAVE_N
  Q_ELEMS_PER_THREAD, KV_ELEMS_PER_THREAD = BLOCK_M * D // THREADS_PER_BLOCK, BLOCK_N * D // THREADS_PER_BLOCK
  Q_lds_alloc = UOp.alloc((BLOCK_M, D + LDS_PAD), dtypes.half, addrspace=AddrSpace.LOCAL)
  Q_lds = Q_lds_alloc[:, :D]
  q_vec = 8 if Q_ELEMS_PER_THREAD % 8 == 0 else 4 if Q_ELEMS_PER_THREAD % 4 == 0 else 1
  q_stores = []
  for qi in range(Q_ELEMS_PER_THREAD // q_vec):
    off = tid * Q_ELEMS_PER_THREAD + qi * q_vec
    qv = _amd_load(q.reshape(BLOCK_M * D)[off], q_vec) if q_vec > 1 else (q.reshape(BLOCK_M * D)[off].load(),)
    for j in range(q_vec): q_stores.append(Q_lds.reshape(BLOCK_M * D)[off + j].store(qv[j]))
  Q_lds = Q_lds.after(*q_stores)
  KP_lds = UOp.alloc((BLOCK_N, D + LDS_PAD), dtypes.half, addrspace=AddrSpace.LOCAL)
  KV_lds = KP_lds[:, :D]
  if has_rel:
    r_uop, rel_bias = extra[0], extra[1]
    E, L = cast(tuple[int, int], rel_bias.shape)
    R_ELEMS_PER_THREAD, RB_ROWS = BLOCK_M * L // THREADS_PER_BLOCK, BLOCK_M + BLOCK_N
    RB_ELEMS_PER_THREAD = RB_ROWS * L // THREADS_PER_BLOCK
    R_lds = UOp.alloc((BLOCK_M, L), dtypes.half, addrspace=AddrSpace.LOCAL)
    r_blk = r_uop.reshape(BH, M // BLOCK_M, BLOCK_M, L)[block_bh, block_m]
    if R_ELEMS_PER_THREAD % 4 == 0:
      r_st = []
      for ri_r in range(R_ELEMS_PER_THREAD // 4):
        off_r = tid * R_ELEMS_PER_THREAD + ri_r * 4
        rv = _amd_load(r_blk.reshape(BLOCK_M * L)[off_r], 4)
        for j in range(4): r_st.append(R_lds.reshape(BLOCK_M * L)[off_r + j].store(rv[j]))
      R_lds = R_lds.after(*r_st)
    else:
      R_lds = R_lds.after(R_lds.reshape(THREADS_PER_BLOCK, R_ELEMS_PER_THREAD)[tid].store(r_blk.reshape(THREADS_PER_BLOCK, R_ELEMS_PER_THREAD)[tid]))
    RB_lds = UOp.alloc((RB_ROWS, L if L % 8 == 0 else L + 2), dtypes.half, addrspace=AddrSpace.LOCAL)[:, :L]
  acc = _reg((TM, TD), 0)
  if has_sink:
    sink_uop = extra[2 if has_rel else 0]
    m_i, rm_init = UOp.alloc((TM,), dtypes.float, addrspace=AddrSpace.REG), UOp.range(TM, 80)
    q_local_init = wave_m * WMMA_M + (lane_m*TM + rm_init if rdna4 else rm_init*LANES_PER_WAVE_M + lane_m)
    m_i = m_i.after(m_i[rm_init].store(sink_uop[block_bh, block_m * BLOCK_M + q_local_init].load().float()).end(rm_init))
    l_i = _reg((TM,), 1)
  else: m_i, l_i = _reg((TM,), -1e30), _reg((TM,), 0)
  n_end = ((q_base + (block_m + 1) * BLOCK_M).minimum(valid_kv_len) + BLOCK_N - 1) // BLOCK_N
  n_start = ((q_base + block_m * BLOCK_M - window + 1).maximum(0) // BLOCK_N).minimum(n_end) if window else 0
  n_tile_i = UOp.range(n_end - n_start, 100)
  n_tile = n_start + n_tile_i
  phys_tile = (n_tile % (physical_n // BLOCK_N)) if ring else n_tile
  kv_vec = 8 if KV_ELEMS_PER_THREAD % 8 == 0 and D % 8 == 0 else 4 if KV_ELEMS_PER_THREAD % 4 == 0 and D % 4 == 0 else 1
  kv_w = KV_lds.after(n_tile_i)
  k_stores = []
  if is_q4:
    q4_r, q4_g = tid // 4, tid % 4
    k_row = (phys_tile * BLOCK_N + q4_r).minimum(physical_n - 1)
    ksc_w = k[k_row, 16].load()
    ksc_h = ((ksc_w >> ((q4_g // 2) * 16).cast(dtypes.uint32)) & 0xffff).cast(dtypes.uint16).bitcast(dtypes.half)
    k_w8 = kv_w.reshape(BLOCK_N, D // 8, 8)
    for wi in range(4):
      kw = k[k_row, q4_g * 4 + wi].load()
      vals = [(((kw >> (4 * j)) & 15).cast(dtypes.int32) - 8).cast(dtypes.half) * ksc_h for j in range(8)]
      k_stores.append(k_w8[q4_r, q4_g * 4 + wi].store(UOp.stack(*vals)))
  else:
    for ki in range(KV_ELEMS_PER_THREAD // kv_vec):
      off = tid * KV_ELEMS_PER_THREAD + ki * kv_vec
      kv_v = _amd_load(k.reshape(physical_n * D)[phys_tile * BLOCK_N * D + off], kv_vec) if kv_vec > 1 else \
        (k.reshape(physical_n * D)[phys_tile * BLOCK_N * D + off].load(),)
      for j in range(kv_vec): k_stores.append(kv_w.reshape(BLOCK_N * D)[off + j].store(kv_v[j]))
  stores_qk: list[UOp] = k_stores
  if has_rel:
    if L % 8 == 0 and RB_ELEMS_PER_THREAD % 8 == 0:
      rb_w = RB_lds.after(n_tile_i)
      for rbi in range(RB_ELEMS_PER_THREAD // 8):
        flat_rb = tid * RB_ELEMS_PER_THREAD + rbi * 8
        d_idx, l_idx = flat_rb // L, flat_rb % L
        dist_rb = (q_base + block_m * BLOCK_M - n_tile * BLOCK_N - (BLOCK_N - 1)) + d_idx
        in_rb = (dist_rb >= 0) & (dist_rb < E)
        rbv = _amd_load(rel_bias[dist_rb.maximum(0).minimum(E - 1), l_idx], 8)
        for j in range(8): stores_qk.append(rb_w[d_idx, l_idx + j].store(in_rb.where(rbv[j], UOp.const(0, dtypes.half))))
    else:
      load_rb = UOp.range(RB_ELEMS_PER_THREAD, 91)
      flat_rb = tid * RB_ELEMS_PER_THREAD + load_rb
      d_idx, l_idx = flat_rb // L, flat_rb % L
      dist_rb = (q_base + block_m * BLOCK_M - n_tile * BLOCK_N - (BLOCK_N - 1)) + d_idx
      rb_val = ((dist_rb >= 0) & (dist_rb < E)).where(rel_bias[dist_rb.maximum(0).minimum(E - 1), l_idx].load(), UOp.const(0, dtypes.half))
      stores_qk.append(RB_lds.after(n_tile_i)[d_idx, l_idx].store(rb_val).end(load_rb))
  KV_lds_k = kv_w.after(*stores_qk)
  if has_rel: RB_lds_r = RB_lds.after(*stores_qk)
  S_reg = _reg((TM, TN), 0, n_tile_i)
  k_qk, tm1, tn1 = UOp.range(D//WMMA_K, 101), UOp.range(TM//WMMA_ACC, 200), UOp.range(TN, 201)
  S_frag = S_reg.reshape(TM // WMMA_ACC, WMMA_ACC, TN).permute(0, 2, 1)[tm1, tn1]
  q_frag = Q_lds.reshape(WAVES_M, TM // WMMA_ACC, WMMA_M, D // WMMA_K, WMMA_K)[wave_m, tm1, lane_n, k_qk]
  k_frag = KV_lds_k.reshape(TN, WMMA_N, D // WMMA_K, WMMA_K)[tn1, lane_n, k_qk]
  # All waves must finish reading Q/K before their shared memory is reused for P/V.
  q_frag, k_frag = (_wmma_fragment(f, lane, rdna4) for f in (q_frag, k_frag))
  qk_done = S_frag.store(UOp.wmma(q_frag, k_frag, S_frag.after(k_qk), *WMMA_ARG)).end(tm1, tn1).end(k_qk).barrier()
  S_reg = S_reg.after(qk_done, S_reg.store(S_reg * SCALE))
  if has_rel and L % 8 == 0:
    rm = UOp.range(TM, 250)
    q_local = wave_m * WMMA_M + (lane_m*TM + rm if rdna4 else rm*LANES_PER_WAVE_M + lane_m)
    q_idx = q_base + block_m * BLOCK_M + q_local
    r_chunks = [_amd_load(R_lds[q_local, c], 8) for c in range(0, L, 8)]
    rel_stores = []
    for rn in range(TN):
      k_local = rn * LANES_PER_WAVE_N + lane_n
      k_idx = n_tile * BLOCK_N + k_local
      causal = (k_idx <= q_idx) & (k_idx < valid_kv_len)
      if window: causal = causal & (q_idx - k_idx < window)
      d_off = q_local + (BLOCK_N - 1) - k_local
      dot = S_reg[rm, rn].load()
      for ci, rc in enumerate(r_chunks):
        rbc = _amd_load(RB_lds_r[d_off, ci * 8], 8)
        for k2 in range(4):
          dot = UOp.native(
            "({{typedef _Float16 _h2 __attribute__((ext_vector_type(2))); "
            "__builtin_amdgcn_fdot2((_h2){{{}, {}}}, (_h2){{{}, {}}}, {}, false);}})",
            shape=dot.shape, dtype=dtypes.float32, fmt=True).call(
            rc[k2 * 2], rc[k2 * 2 + 1], rbc[k2 * 2], rbc[k2 * 2 + 1], dot, inline=True)
      rel_stores.append(S_reg[rm, rn].store(causal.where(dot, dot.const_like(-math.inf))))
    S_reg = S_reg.after(UOp.group(*rel_stores).end(rm))
  else:
    rm, rn = UOp.range(TM, 250), UOp.range(TN, 251)
    q_local = wave_m * WMMA_M + (lane_m*TM + rm if rdna4 else rm*LANES_PER_WAVE_M + lane_m)
    k_local = rn * LANES_PER_WAVE_N + lane_n
    q_idx, k_idx = q_base + block_m * BLOCK_M + q_local, n_tile * BLOCK_N + k_local
    causal = (k_idx <= q_idx) & (k_idx < valid_kv_len)
    if window: causal = causal & (q_idx - k_idx < window)
    s_val = S_reg[rm, rn]
    if has_rel:
      d_off = q_local + (BLOCK_N - 1) - k_local
      s_val = s_val + sum((R_lds[q_local, l].load().float() * RB_lds_r[d_off, l].load().float() for l in range(L)), UOp.const(0, dtypes.float))
    S_reg = S_reg.after(S_reg[rm, rn].store(causal.where(s_val, s_val.const_like(-math.inf))).end(rm, rn))
  m_ij, rm2 = _reg((TM,), -math.inf, n_tile_i), UOp.range(TN, 261)
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
  p_local, ri_ws = _reg((TM,), 0, n_tile_i), UOp.range(TM, 295)
  p_sum = p_local.after(p_local[ri_ws].store(sum(warp_reduce(S_reg[ri_ws, rn]) for rn in range(TN))).end(ri_ws))
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
  v_stores = []
  if is_q4:
    q4_r, q4_g = tid // 4, tid % 4
    v_pos = n_tile * BLOCK_N + q4_r
    in_v = v_pos < valid_kv_len
    v_row = (phys_tile * BLOCK_N + q4_r).minimum(physical_n - 1)
    vsc_w = v[v_row, 16].load()
    vsc_h = ((vsc_w >> ((q4_g // 2) * 16).cast(dtypes.uint32)) & 0xffff).cast(dtypes.uint16).bitcast(dtypes.half)
    for wi in range(4):
      vw = v[v_row, q4_g * 4 + wi].load()
      for j in range(8):
        val = (((vw >> (4 * j)) & 15).cast(dtypes.int32) - 8).cast(dtypes.half) * vsc_h
        v_stores.append(V_copy[q4_r, q4_g * 32 + wi * 8 + j].store(in_v.where(val, UOp.const(0, dtypes.half))))
  else:
    for vi in range(KV_ELEMS_PER_THREAD // kv_vec):
      off = tid * KV_ELEMS_PER_THREAD + vi * kv_vec
      v_pos = n_tile * BLOCK_N + off // D
      in_v = v_pos < valid_kv_len
      vv = _amd_load(v.reshape(physical_n * D)[phys_tile * BLOCK_N * D + off], kv_vec) if kv_vec > 1 else \
        (v.reshape(physical_n * D)[phys_tile * BLOCK_N * D + off].load(),)
      for j in range(kv_vec): v_stores.append(V_copy.reshape(BLOCK_N * D)[off + j].store(in_v.where(vv[j], UOp.const(0, dtypes.half))))
  P_lds, V_lds = P_lds.after(P_store, *v_stores), V_lds.after(P_store, *v_stores)
  k_pv, tm2, tn2 = UOp.range(BLOCK_N//WMMA_K, 400), UOp.range(TM//WMMA_ACC, 401), UOp.range(TD, 402)
  acc_frag = acc.reshape(TM // WMMA_ACC, WMMA_ACC, TD).permute(0, 2, 1)[tm2, tn2]
  p_frag = P_lds[wave_n].reshape(WAVES_M, TM // WMMA_ACC, WMMA_M, BLOCK_N // WMMA_K, WMMA_K)[wave_m, tm2, lane_n, k_pv]
  v_frag = V_lds.reshape(WAVES_N, TD, WMMA_N, BLOCK_N // WMMA_K, WMMA_K)[wave_n, tn2, lane_n, k_pv]
  p_frag, v_frag = (_wmma_fragment(f, lane, rdna4) for f in (p_frag, v_frag))
  n_tile_end = acc_frag.store(UOp.wmma(p_frag, v_frag, acc_frag.after(k_pv), *WMMA_ARG)).end(tm2, tn2).end(k_pv).barrier().end(n_tile_i)
  acc, l_i, m_i = acc.after(n_tile_end), l_i.after(n_tile_end), m_i.after(n_tile_end)
  if has_gate:
    gate_uop = extra[(2 if has_rel else 0) + (1 if has_sink else 0)].reshape(B, M, H, D)
    og = o.reshape(B, M, H, D)
    b_idx, h_idx = (UOp.const(0, dtypes.weakint), block_bh) if B == 1 else (block_bh // H, block_bh % H)
    rm_g, rd_g = UOp.range(TM, 420), UOp.range(TD, 421)
    t_idx = block_m * BLOCK_M + wave_m * WMMA_M + (lane_m * TM + rm_g if rdna4 else rm_g * LANES_PER_WAVE_M + lane_m)
    d_idx = (wave_n * TD + rd_g) * LANES_PER_WAVE_N + lane_n
    gv = gate_uop[b_idx, t_idx, h_idx, d_idx].load().float()
    val = (acc[rm_g, rd_g].load() / l_i[rm_g].load()) * (1.0 / (1.0 + (gv * (-LOG2E)).exp2()))
    return og[b_idx, t_idx, h_idx, d_idx].store(val.cast(o.dtype)).end(rm_g, rd_g).end(wave_m, wave_n, lane).end(block_m, block_bh).sink(
      arg=KernelInfo(opts_to_apply=()))
  acc = acc.after(acc.store(acc * (1 / l_i).reshape(TM, 1).expand(TM, TD)))
  o = o.reshape(WAVES_M, *row_shape, WAVES_N, TD, LANES_PER_WAVE_N) \
    .permute(0, 3, row_lane-1, 5, row_elem-1, 4).reshape(THREADS_PER_BLOCK, TM, TD)
  return o[tid].store(acc).end(wave_m, wave_n, lane).end(block_m, block_bh).sink(arg=KernelInfo(opts_to_apply=()))

# ******** gated delta net: fused recurrent scan ********

@functools.cache
def _gated_delta_prefill_kernel(core:UOp, q:UOp, k:UOp, v:UOp, beta:UOp, alpha:UOp, state:UOp, kq:UOp, start_pos:int|UOp|None=None) -> UOp:
  batch, heads, tokens, value_dim, row_tile = *core.shape, 4
  key_dim, alpha_dim = q.shape[-1], alpha.shape[-1] if len(alpha.shape) == 4 else 1
  assert isinstance(key_dim, int) and key_dim % 32 == 0 and (isinstance(value_dim, UOp) or value_dim % row_tile == 0)
  core, v = (x.reshape(batch*heads, tokens, value_dim) for x in (core, v))
  q, k = (x.reshape(batch*heads, tokens, key_dim) for x in (q, k))
  beta, kq = (x.reshape(batch*heads, tokens) for x in (beta, kq))
  alpha, state = alpha.reshape(batch*heads, tokens, alpha_dim), state.reshape(batch*heads, value_dim, key_dim)
  bh_row, lane = UOp.range(batch*heads*value_dim//row_tile, 0, AxisType.GLOBAL), UOp.range(32, 1, axis_type=AxisType.LOCAL)
  bh, row_base = bh_row // (value_dim//row_tile), (bh_row % (value_dim//row_tile))*row_tile
  rows, cols = tuple(row_base+i for i in range(row_tile)), tuple(lane + i*32 for i in range(key_dim//32))
  current = UOp.alloc((row_tile*key_dim//32,), dtypes.float32, addrspace=AddrSpace.REG)
  if start_pos is None or (isinstance(start_pos, int) and start_pos != 0):
    init_vals = tuple(state[bh, row, col].float() for row in rows for col in cols)
  elif isinstance(start_pos, int):
    init_vals = tuple(UOp.const(0.0, dtypes.float32) for _ in rows for _ in cols)
  else:
    initial = start_pos.eq(0)
    init_vals = tuple(initial.where(0, state[bh, row, col].float()) for row in rows for col in cols)
  current = current.after(current.store(UOp.stack(*init_vals)))
  token = UOp.range(tokens, 2)
  keys = tuple(k[bh, token, col].load() for col in cols)
  queries = tuple(q[bh, token, col].load() for col in cols)
  updates, stores = [], []
  for row_idx,row in enumerate(rows):
    previous = tuple(current.after(token)[row_idx*key_dim//32+i].load() for i in range(key_dim//32))
    decayed, bv = tuple(x * alpha[bh, token, col if alpha_dim > 1 else 0].load() for x,col in zip(previous, cols)), beta[bh, token].load()
    state_k = warp_reduce(sum((x*y for x,y in zip(decayed, keys)), UOp.const(0, dtypes.float32)), full_wave=True)
    state_q = warp_reduce(sum((x*y for x,y in zip(decayed, queries)), UOp.const(0, dtypes.float32)), full_wave=True)
    delta = (v[bh, token, row].load() - state_k) * bv
    updates += [x + delta*y for x,y in zip(decayed, keys)]
    stores.append(core[bh, token, row.valid(lane.eq(0))].store(state_q + delta*kq[bh, token]))
  step = UOp.group(*stores, current.store(UOp.stack(*updates))).end(token)
  state_stores = (state[bh, row, col].store(current.after(step)[row_idx*key_dim//32+i].load().cast(state.dtype))
                  for row_idx,row in enumerate(rows) for i,col in enumerate(cols))
  return UOp.group(*state_stores).end(lane, bh_row).sink(arg=KernelInfo(name="gated_delta_prefill", opts_to_apply=()))


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
