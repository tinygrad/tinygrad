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
def _amd_f16_gemv_kernel(out:UOp, w:UOp, x:UOp, *rest:UOp, in_features:int, out_features:int, tokens:int) -> UOp:
  bias: UOp|None = rest[0] if rest else None
  # one wave per output row; tile all tokens in registers so w is loaded once across speculative verify batches
  lanes = WARP_SIZE
  val_chunk = 4 if in_features % (lanes * 4) == 0 else 2 if in_features % (lanes * 2) == 0 else 1
  rows = math.gcd(out_features, 2 if out_features <= 128 else 4)
  out_blk = UOp.range(out_features // rows, 0, AxisType.GLOBAL)
  wave = UOp.range(rows, 3, AxisType.LOCAL)
  lane = UOp.range(lanes, 2, axis_type=AxisType.LOCAL)
  out_row = out_blk * rows + wave
  per = in_features // (lanes * val_chunk)
  assert per * lanes * val_chunk == in_features
  w = w.reshape((out_features, per, lanes*val_chunk))
  x = x.reshape((tokens, per, lanes*val_chunk))
  zero = UOp.const(0, dtypes.float32)
  if tokens > 1 and per > 4:
    accs_reg = tuple(UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG) for _ in range(tokens))
    accs_reg = tuple(a.after(a[0].store(zero)) for a in accs_reg)
    i_r = UOp.range(per, 4)
    wv = _vec_load(w[out_row, i_r, lane*val_chunk], val_chunk)
    st: list[UOp] = []
    for t in range(tokens):
      xv = _vec_load(x[t, i_r, lane*val_chunk], val_chunk)
      dot = sum((wv[j] * xv[j] for j in range(val_chunk)), zero)
      st.append(accs_reg[t][0].store(accs_reg[t].after(i_r)[0].load() + dot))
    upd = UOp.group(*st).end(i_r)
    accs = [accs_reg[t].after(upd)[0].load() for t in range(tokens)]
  else:
    accs = [zero] * tokens
    for i in range(per):
      wv = _vec_load(w[out_row, i, lane*val_chunk], val_chunk)
      for t in range(tokens):
        xv = _vec_load(x[t, i, lane*val_chunk], val_chunk)
        for j in range(val_chunk): accs[t] = accs[t] + wv[j] * xv[j]
  b_val = bias[out_row].load().float() if bias is not None else None
  stores: list[UOp] = []
  for t in range(tokens):
    total = warp_reduce(accs[t], full_wave=True)
    if b_val is not None: total = total + b_val
    stores.append(out[t, out_row.valid(lane.eq(0))].store(total.cast(out.dtype)))
  return UOp.group(*stores).end(out_blk, wave, lane).sink(
    arg=KernelInfo(name="linear_f16_gemv", opts_to_apply=()))

# ******** flash attention on the KV cache ********

def _vec_load(ptr:UOp, lanes:int) -> tuple[UOp, ...]:
  if lanes == 1: return (ptr.load().float(),)
  vec = _amd_load(ptr, lanes)
  return tuple(vec[i].float() for i in range(lanes))

@functools.cache
def _amd_flash_attention_decode_partial(out, stats, q, cache_kv, *extra, valid_kv_len, max_kv_len, block_n, waves=4, window=0,
                                        has_rel=False, ring=False):
  valid_kv_len = _unbind(valid_kv_len)
  is_q4 = cache_kv.dtype == dtypes.uint32
  _, B, H_KV, N, _ = cast(tuple[int, int, int, int, int], cache_kv.shape)
  _, H, M, D = cast(tuple[int, int, int, int], q.shape)
  assert 1 <= M <= 16 and H % H_KV == 0 and D % WARP_SIZE == 0 and (ring or max_kv_len <= N) and N % block_n == 0
  if is_q4: assert D == 128 and cache_kv.shape[4] == 17
  G, CHUNK, DPL, WAVES, PARTIALS = H // H_KV, block_n, D // WARP_SIZE, waves, out.shape[2]
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
  DT = 64 if D % 64 == 0 else WARP_SIZE  # dims per block
  assert D % DT == 0
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

@functools.cache
def _amd_flash_decode_combine_q8(o:UOp, q_out:UOp, sc_out:UOp, xs_out:UOp, partial:UOp, stats:UOp, *extra:UOp,
                                 valid_kv_len:int|UOp, window:int=0, has_sink:bool=False) -> UOp:
  # One wave per (batch*m, head, 32-dim Q8 group): computes gated attention output and quantizes in-warp to Q8 directly.
  valid_kv_len = _unbind(valid_kv_len)
  BM, H, C, D = cast(tuple[int, int, int, int], partial.shape)
  B, M = cast(int, o.shape[0]), cast(int, o.shape[1])
  ng = D // WARP_SIZE
  block_bh, block_g = UOp.range(BM * H, 0, AxisType.GLOBAL), UOp.range(ng, 1, AxisType.GLOBAL)
  lane = UOp.range(WARP_SIZE, -1, axis_type=AxisType.WARP)
  bm, h = block_bh // H, block_bh % H
  b, m = (bm, UOp.const(0, dtypes.weakint)) if M == 1 else (bm // M, bm % M)
  q_valid_len = valid_kv_len if M == 1 else valid_kv_len - (M - 1) + m
  q_span = (q_valid_len.minimum(window) if isinstance(q_valid_len, UOp) else min(q_valid_len, window)) if window else q_valid_len
  q_live = (q_span + 63) // 64
  live = min(q_live, C) if isinstance(q_live, int) else q_live.minimum(C)
  d = block_g * WARP_SIZE + lane
  sink_m = extra[0][b, h, m].load().float() if has_sink else None
  gate_uop = extra[1 if has_sink else 0].reshape(B, M, H, D)
  chunk = UOp.range(live, 100)
  def iloop(ph, val): return ph[0].store(val) if isinstance(val, UOp) else ph.store(ph.const_like(val))
  chunk_max = UOp.alloc((1,), dtypes.float, addrspace=AddrSpace.REG)
  chunk_max_i = chunk_max.after(iloop(chunk_max, sink_m if sink_m is not None else -math.inf))
  update0 = chunk_max_i.store(chunk_max_i.after(chunk).maximum(stats[bm, h, chunk, 0].load())).end(chunk)
  chunk_max = chunk_max_i.after(update0)
  chunk2 = UOp.range(live, 101)
  acc = UOp.alloc((1,), dtypes.float, addrspace=AddrSpace.REG)
  weight_sum = UOp.alloc((1,), dtypes.float, addrspace=AddrSpace.REG)
  init_ws: float|UOp = ((sink_m - chunk_max[0].load()) * LOG2E).exp2() if sink_m is not None else 0
  acc_i, weight_sum_i = acc.after(iloop(acc, 0)), weight_sum.after(iloop(weight_sum, init_ws))
  w = ((stats[bm, h, chunk2, 0].load() - chunk_max) * LOG2E).exp2()
  update1 = UOp.group(acc_i[0].store(acc_i.after(chunk2)[0].load() + w * partial[bm, h, chunk2, d].load()),
                      weight_sum_i[0].store(weight_sum_i.after(chunk2)[0].load() + w * stats[bm, h, chunk2, 1].load())).end(chunk2)
  acc, weight_sum = acc_i.after(update1), weight_sum_i.after(update1)
  inv = 1 / weight_sum[0].load()
  val = (acc[0].load() * inv) * (1.0 / (1.0 + (gate_uop[b, m, h, d].load().float() * (-LOG2E)).exp2()))
  og = o.reshape(B, M, H, D)
  g_idx = h * ng + block_g
  wmax = warp_reduce(val.abs(), Ops.MAX, full_wave=True)
  dq = UOp.native("({0}/127.0f)", shape=wmax.shape, dtype=dtypes.float, fmt=True).call(wmax).maximum(1e-8)
  quant = UOp.native("__builtin_nearbyintf({0}/{1})", shape=val.shape, dtype=dtypes.float, fmt=True).call(val, dq).clip(-127, 127).cast(dtypes.int8)
  word = warp_reduce(quant.cast(dtypes.uint8).cast(dtypes.uint32) << ((lane % 4) * 8).cast(dtypes.uint32), Ops.OR, masks=(1, 2))
  stores = [og[b, m, h, d].store(val.cast(o.dtype)),
            q_out[bm, g_idx, (lane // 4).valid((lane % 4).eq(0))].store(word),
            sc_out[bm, g_idx.valid(lane.eq(0))].store(dq),
            xs_out[bm, g_idx, (lane // 16).valid((lane % 16).eq(0))].store(warp_reduce(quant.float()))]
  return UOp.group(*stores).end(lane, block_g, block_bh).sink(arg=KernelInfo(name="flash_decode_combine_q8", opts_to_apply=()))

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
  assert k.shape == v.shape and BH % k.shape[0] == 0 and (D == 128 and cache_dim == 17 if is_q4 else k.shape[2] == D)
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
  assert all(isinstance(x, int) for x in (batch, heads, tokens, value_dim, key_dim)) and key_dim % 32 == 0 and value_dim % row_tile == 0
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

# ******** fused attention residual & relative short conv ********

@functools.cache
def _amd_attn_res_mix_kernel(out:UOp, xs:UOp, w:UOp, *, valid_t:int|UOp, N:int, Q:int, D:int, eps:float, scale:float, normed:bool) -> UOp:
  B, valid_t = cast(int, xs.shape[0]), _unbind(valid_t)
  WAVES = next(wv for wv in (8, 6, 4, 2, 1) if D % (wv * WARP_SIZE * 4) == 0)
  THREADS, PER = WAVES * WARP_SIZE, D // (WAVES * WARP_SIZE * 4)
  bt = UOp.range(B * valid_t, 0, AxisType.GLOBAL)
  b, t = (bt // valid_t, bt % valid_t) if B > 1 else (UOp.const(0, dtypes.weakint), bt)
  wave, lane = UOp.range(WAVES, 1, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, axis_type=AxisType.WARP)
  tid, zerof = wave * WARP_SIZE + lane, UOp.const(0.0, dtypes.float32)
  w_f = w.reshape(Q, D)
  raw = [[_vec_load(xs[b, t, n, (p * THREADS + tid) * 4], 4) for p in range(PER)] for n in range(N)]
  red_lds = UOp.alloc((N, WAVES), dtypes.float32, addrspace=AddrSpace.LOCAL)
  sq_stores = [red_lds[n, wave.valid(lane.eq(0))].store(warp_reduce(sum((v * v for ch in raw[n] for v in ch), zerof), full_wave=True))
               for n in range(N)]
  red_r = red_lds.after(*sq_stores)
  xn = []
  for n in range(N):
    inv_rms = (sum((red_r[n, wi].load() for wi in range(WAVES)), zerof) * (1.0 / D) + eps).rsqrt()
    xn.append([tuple(v * inv_rms for v in ch) for ch in raw[n]])
  if N > 1:
    score_lds = UOp.alloc((N, Q, WAVES), dtypes.float32, addrspace=AddrSpace.LOCAL)
    w_vals = [[_vec_load(w_f[q, (p * THREADS + tid) * 4], 4) for p in range(PER)] for q in range(Q)]
    sc_stores = [score_lds[n, q, wave.valid(lane.eq(0))].store(
      warp_reduce(sum((xn[n][p][j] * w_vals[q][p][j] for p in range(PER) for j in range(4)), zerof), full_wave=True))
      for n in range(N) for q in range(Q)]
    sc_r = score_lds.after(*sc_stores)
  src = xn if normed else raw
  out_stores: list[UOp] = []
  for q in range(Q):
    if N == 1: probs = [UOp.const(1.0, dtypes.float32)]
    else:
      sc = [sum((sc_r[n, q, wi].load() for wi in range(WAVES)), zerof) * scale for n in range(N)]
      m = functools.reduce(UOp.maximum, sc)
      exps = [((s - m) * LOG2E).exp2() for s in sc]
      inv_s = 1.0 / sum(exps[1:], exps[0])
      probs = [e * inv_s for e in exps]
    for p in range(PER):
      off = (p * THREADS + tid) * 4
      for j in range(4):
        val = sum((probs[n] * src[n][p][j] for n in range(N)), zerof)
        out_stores.append(out[b, t, q, off + j].store(val.cast(out.dtype)))
  return UOp.group(*out_stores).end(bt, wave, lane).sink(arg=KernelInfo(name="attn_res_mix", opts_to_apply=()))

@functools.cache
def _amd_attn_res_mix_streams_kernel(out:UOp, *args:UOp, valid_t:int|UOp, N:int, Q:int, D:int,
                                     eps:float, scale:float, normed:bool, pre_normed:int=0, emit_normed:bool=False) -> UOp:
  out_norm = args[0] if emit_normed else None
  w = args[1] if emit_normed else args[0]
  streams = args[2:] if emit_normed else args[1:]
  B, valid_t = cast(int, streams[0].shape[0]), _unbind(valid_t)
  WAVES = next(wv for wv in (8, 6, 4, 2, 1) if D % (wv * WARP_SIZE * 4) == 0)
  THREADS, PER = WAVES * WARP_SIZE, D // (WAVES * WARP_SIZE * 4)
  bt = UOp.range(B * valid_t, 0, AxisType.GLOBAL)
  b, t = (bt // valid_t, bt % valid_t) if B > 1 else (UOp.const(0, dtypes.weakint), bt)
  wave, lane = UOp.range(WAVES, 1, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, axis_type=AxisType.WARP)
  tid, zerof = wave * WARP_SIZE + lane, UOp.const(0.0, dtypes.float32)
  w_f = w.reshape(Q, D)
  raw = [[_vec_load(streams[n][b, t, (p * THREADS + tid) * 4], 4) for p in range(PER)] for n in range(N)]
  xn: list[list[tuple[UOp, ...]]] = list(raw[:pre_normed])
  n_unnorm = N - pre_normed
  if n_unnorm > 0:
    red_lds = UOp.alloc((n_unnorm, WAVES), dtypes.float32, addrspace=AddrSpace.LOCAL)
    sq_stores = [red_lds[i, wave.valid(lane.eq(0))].store(
      warp_reduce(sum((v * v for ch in raw[pre_normed + i] for v in ch), zerof), full_wave=True)) for i in range(n_unnorm)]
    red_r = red_lds.after(*sq_stores)
    for i in range(n_unnorm):
      inv_rms = (sum((red_r[i, wi].load() for wi in range(WAVES)), zerof) * (1.0 / D) + eps).rsqrt()
      xn.append([tuple(v * inv_rms for v in ch) for ch in raw[pre_normed + i]])
  if N > 1:
    score_lds = UOp.alloc((N, Q, WAVES), dtypes.float32, addrspace=AddrSpace.LOCAL)
    w_vals = [[_vec_load(w_f[q, (p * THREADS + tid) * 4], 4) for p in range(PER)] for q in range(Q)]
    sc_stores = [score_lds[n, q, wave.valid(lane.eq(0))].store(
      warp_reduce(sum((xn[n][p][j] * w_vals[q][p][j] for p in range(PER) for j in range(4)), zerof), full_wave=True))
      for n in range(N) for q in range(Q)]
    sc_r = score_lds.after(*sc_stores)
  src = xn if normed else raw
  out_v = out.reshape(B, cast(int, out.shape[1]), Q, D // 4, 4)
  out_stores: list[UOp] = []
  for q in range(Q):
    if N == 1: probs = [UOp.const(1.0, dtypes.float32)]
    else:
      sc = [sum((sc_r[n, q, wi].load() for wi in range(WAVES)), zerof) * scale for n in range(N)]
      m = functools.reduce(UOp.maximum, sc)
      exps = [((s - m) * LOG2E).exp2() for s in sc]
      inv_s = 1.0 / sum(exps[1:], exps[0])
      probs = [e * inv_s for e in exps]
    for p in range(PER):
      vals = [sum((probs[n] * src[n][p][j] for n in range(N)), zerof).cast(out.dtype) for j in range(4)]
      out_stores.append(out_v[b, t, q, p * THREADS + tid].store(UOp.stack(*vals)))
  if out_norm is not None:
    on_v = out_norm.reshape(B, cast(int, out_norm.shape[1]), D // 4, 4)
    for p in range(PER):
      out_stores.append(on_v[b, t, p * THREADS + tid].store(UOp.stack(*(xn[N - 1][p][j].cast(out_norm.dtype) for j in range(4)))))
  return UOp.group(*out_stores).end(bt, wave, lane).sink(arg=KernelInfo(name="attn_res_mix", opts_to_apply=()))

@functools.cache
def _amd_attn_res_update_kernel(dst:UOp, xs:UOp, add:UOp, *, valid_t:int|UOp, N:int, D:int, new_block:bool) -> UOp:
  B, valid_t = cast(int, xs.shape[0]), _unbind(valid_t)
  WAVES = next(wv for wv in (8, 6, 4, 2, 1) if D % (wv * WARP_SIZE * 4) == 0)
  THREADS, PER = WAVES * WARP_SIZE, D // (WAVES * WARP_SIZE * 4)
  bt, tid = UOp.range(B * valid_t, 0, AxisType.GLOBAL), UOp.range(THREADS, 1, AxisType.LOCAL)
  b, t = (bt // valid_t, bt % valid_t) if B > 1 else (UOp.const(0, dtypes.weakint), bt)
  stores: list[UOp] = []
  for p in range(PER):
    off = (p * THREADS + tid) * 4
    for n in range(N):
      v = _vec_load(xs[b, t, n, off], 4)
      if not new_block and n == N - 1:
        a = _vec_load(add[b, t, off], 4)
        v = tuple(vi + ai for vi, ai in zip(v, a))
      for j in range(4): stores.append(dst[b, t, n, off + j].store(v[j]))
    if new_block:
      a = _vec_load(add[b, t, off], 4)
      for j in range(4): stores.append(dst[b, t, N, off + j].store(a[j]))
  return UOp.group(*stores).end(bt, tid).sink(arg=KernelInfo(name="attn_res_update", opts_to_apply=()))

@functools.cache
def _amd_rel_short_conv_kernel(out:UOp, state:UOp, x:UOp, w:UOp, *extra_args:UOp,
                               valid_t:int|UOp, K:int, C:int, H:int, inv_scale:float, start_pos:int|UOp,
                               has_dyn:bool=False, has_res:bool=False, has_hist:bool=False) -> UOp:
  B, valid_t, start_pos = cast(int, x.shape[0]), _unbind(valid_t), _unbind(start_pos)
  THREADS = 64
  assert C % THREADS == 0 and K >= 2
  ei = 0
  dyn_uop = extra_args[ei] if has_dyn else None
  if has_dyn: ei += 1
  res_uop = extra_args[ei] if has_res else None
  if has_res: ei += 1
  hist_uop = extra_args[ei] if has_hist else None
  bc_blk, tid = UOp.range(B * (C // THREADS), 0, AxisType.GLOBAL), UOp.range(THREADS, 1, AxisType.LOCAL)
  b, c = bc_blk // (C // THREADS), (bc_blk % (C // THREADS)) * THREADS + tid
  ws = tuple(w[c, j].load().float() for j in range(K))
  hist = UOp.alloc((K - 1,), dtypes.float32, addrspace=AddrSpace.REG)
  if isinstance(start_pos, UOp):
    initial = start_pos.eq(0)
    init_vals = tuple(initial.where(UOp.const(0.0, dtypes.float32), state[b, j, c].load().float()) for j in range(K - 1))
  else:
    init_vals = tuple(UOp.const(0.0, dtypes.float32) if start_pos == 0 else state[b, j, c].load().float() for j in range(K - 1))
  hist_init_st = [hist_uop[b, j, c].store(init_vals[j].cast(hist_uop.dtype)) for j in range(K - 1)] if hist_uop is not None else []
  hist = hist.after(hist.store(UOp.stack(*init_vals)))
  t = UOp.range(valid_t, 2)
  prev = tuple(hist.after(t)[j].load() for j in range(K - 1))
  xt = x[b, t, c].load().float()
  win = prev + (xt,)
  if dyn_uop is not None:
    head = c // (C // H)
    taps = tuple(ws[j] + dyn_uop[b, t, head * K + (K - 1 - j)].load().float() for j in range(K))
  else: taps = ws
  acc = (xt + sum((win[j] * taps[j] for j in range(K)), UOp.const(0.0, dtypes.float32))) * inv_scale
  if res_uop is not None: acc = acc + res_uop[b, t, c].load().float()
  step_ops = [out[b, t, c].store(acc.cast(out.dtype)), hist.store(UOp.stack(*(prev[1:] + (xt,))))]
  if hist_uop is not None: step_ops.append(hist_uop[b, (K - 1) + t, c].store(xt.cast(hist_uop.dtype)))
  step = UOp.group(*step_ops).end(t)
  state_stores = [state[b, j, c].store(hist.after(step)[j].load().cast(state.dtype)) for j in range(K - 1)]
  return UOp.group(*hist_init_st, *state_stores).end(tid, bc_blk).sink(arg=KernelInfo(name="rel_short_conv", opts_to_apply=()))

# ******** fused loop control (down + forget/gain blend) ********

def _fast_tanh(x:UOp) -> UOp: return 2.0 / (1.0 + (x * (-2.0 * LOG2E)).exp2()) - 1.0

@functools.cache
def _amd_loop_ctrl_down_decode_kernel(out:UOp, h:UOp, m:UOp, w_down:UOp, *, BT:int=1, D:int, R:int,
                                      h_q:int=1, h_idx:int=0, m_q:int=1, m_idx:int=0) -> UOp:
  WAVES = next(wv for wv in (8, 6, 4, 2, 1) if D % (wv * WARP_SIZE * 4) == 0)
  THREADS, PER = WAVES * WARP_SIZE, D // (WAVES * WARP_SIZE * 4)
  r = UOp.range(R, 0, AxisType.GLOBAL)
  wave, lane = UOp.range(WAVES, 1, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, axis_type=AxisType.WARP)
  tid, zerof = wave * WARP_SIZE + lane, UOp.const(0.0, dtypes.float32)
  hf, mf, wf = h.reshape(BT, h_q, D), m.reshape(BT, m_q, D), w_down.reshape(R, 2 * D)
  if BT == 1:
    hv = [_vec_load(hf[0, h_idx, (p * THREADS + tid) * 4], 4) for p in range(PER)]
    mv = [_vec_load(mf[0, m_idx, (p * THREADS + tid) * 4], 4) for p in range(PER)]
    lds = UOp.alloc((2, WAVES), dtypes.float32, addrspace=AddrSpace.LOCAL)
    st = [lds[0, wave.valid(lane.eq(0))].store(warp_reduce(sum((v * v for ch in hv for v in ch), zerof), full_wave=True)),
          lds[1, wave.valid(lane.eq(0))].store(warp_reduce(sum((v * v for ch in mv for v in ch), zerof), full_wave=True))]
    lds_r = lds.after(*st)
    inv_h = (sum((lds_r[0, wi].load() for wi in range(WAVES)), zerof) * (1.0 / D) + 1e-6).rsqrt()
    inv_m = (sum((lds_r[1, wi].load() for wi in range(WAVES)), zerof) * (1.0 / D) + 1e-6).rsqrt()
    dot_h, dot_m = zerof, zerof
    for p in range(PER):
      off = (p * THREADS + tid) * 4
      wh, wm = _vec_load(wf[r, off], 4), _vec_load(wf[r, D + off], 4)
      for j in range(4):
        dot_h = dot_h + hv[p][j] * wh[j]
        dot_m = dot_m + mv[p][j] * wm[j]
    w_sum = warp_reduce(dot_h * inv_h + dot_m * inv_m, full_wave=True)
    lds2 = UOp.alloc((WAVES,), dtypes.float32, addrspace=AddrSpace.LOCAL)
    lds2_r = lds2.after(lds2[wave.valid(lane.eq(0))].store(w_sum))
    tot = sum((lds2_r[wi].load() for wi in range(WAVES)), zerof)
    return out.reshape(1, R)[0, r.valid(tid.eq(0))].store(tot).end(r, wave, lane).sink(
      arg=KernelInfo(name="loop_ctrl_down_decode", opts_to_apply=()))
  # Tile all BT tokens in one workgroup per r so w_down is read once across speculative verify batches.
  # Layout (BT, 2, WAVES) keeps h before m in the same vector load per bt, matching BT==1 floating-point contraction bit-for-bit.
  lds = UOp.alloc((BT, 2, WAVES), dtypes.float32, addrspace=AddrSpace.LOCAL)
  st = []
  for bt in range(BT):
    sh, sm = zerof, zerof
    for p in range(PER):
      off = (p * THREADS + tid) * 4
      hv_p, mv_p = _vec_load(hf[bt, h_idx, off], 4), _vec_load(mf[bt, m_idx, off], 4)
      for j in range(4):
        sh = sh + hv_p[j] * hv_p[j]
        sm = sm + mv_p[j] * mv_p[j]
    st.append(lds[bt, 0, wave.valid(lane.eq(0))].store(warp_reduce(sh, full_wave=True)))
    st.append(lds[bt, 1, wave.valid(lane.eq(0))].store(warp_reduce(sm, full_wave=True)))
  lds_r = lds.after(UOp.group(*st).barrier())
  inv_hs = [(sum((lds_r[bt, 0, wi].load() for wi in range(WAVES)), zerof) * (1.0 / D) + 1e-6).rsqrt() for bt in range(BT)]
  inv_ms = [(sum((lds_r[bt, 1, wi].load() for wi in range(WAVES)), zerof) * (1.0 / D) + 1e-6).rsqrt() for bt in range(BT)]
  dot_hs, dot_ms = [zerof] * BT, [zerof] * BT
  for p in range(PER):
    off = (p * THREADS + tid) * 4
    wh, wm = _vec_load(wf[r, off], 4), _vec_load(wf[r, D + off], 4)
    for bt in range(BT):
      hv_p, mv_p = _vec_load(hf[bt, h_idx, off], 4), _vec_load(mf[bt, m_idx, off], 4)
      for j in range(4):
        dot_hs[bt] = dot_hs[bt] + hv_p[j] * wh[j]
        dot_ms[bt] = dot_ms[bt] + mv_p[j] * wm[j]
  lds2 = UOp.alloc((BT, WAVES), dtypes.float32, addrspace=AddrSpace.LOCAL)
  st2 = [lds2[bt, wave.valid(lane.eq(0))].store(warp_reduce(dot_hs[bt] * inv_hs[bt] + dot_ms[bt] * inv_ms[bt], full_wave=True))
         for bt in range(BT)]
  lds2_r = lds2.after(UOp.group(*st2).barrier())
  out_st = [out.reshape(BT, R)[bt, r.valid(tid.eq(0))].store(sum((lds2_r[bt, wi].load() for wi in range(WAVES)), zerof)) for bt in range(BT)]
  return UOp.group(*out_st).end(r, wave, lane).sink(arg=KernelInfo(name="loop_ctrl_down_decode", opts_to_apply=()))

@functools.cache
def _amd_loop_ctrl_apply_decode_kernel(out:UOp, h:UOp, m:UOp, g:UOp, w_forget:UOp, w_gain:UOp, *, BT:int=1, D:int, R:int,
                                       h_q:int=1, h_idx:int=0, m_q:int=1, m_idx:int=0) -> UOp:
  rows, vc = math.gcd(D, 4), R // WARP_SIZE
  n_dblk = D // rows
  d_blk = UOp.range(n_dblk, 0, AxisType.GLOBAL)
  wave, lane = UOp.range(rows, 1, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, axis_type=AxisType.WARP)
  d, zerof = d_blk * rows + wave, UOp.const(0.0, dtypes.float32)
  hf, mf, gf, of = h.reshape(BT, h_q, D), m.reshape(BT, m_q, D), g.reshape(BT, R), out.reshape(BT, D)
  wf, wg = w_forget.reshape(D, R), w_gain.reshape(D, R)
  wfv, wgv = _vec_load(wf[d, lane * vc], vc), _vec_load(wg[d, lane * vc], vc)
  stores: list[UOp] = []
  for bt in range(BT):
    gv = _vec_load(gf[bt, lane * vc], vc)
    fv = warp_reduce(sum((gv[j] * wfv[j] for j in range(vc)), zerof), full_wave=True)
    gnv = warp_reduce(sum((gv[j] * wgv[j] for j in range(vc)), zerof), full_wave=True)
    val = hf[bt, h_idx, d].load().float() * (1.0 - _fast_tanh(fv)) + mf[bt, m_idx, d].load().float() * _fast_tanh(gnv)
    stores.append(of[bt, d.valid(lane.eq(0))].store(val))
  return UOp.group(*stores).end(d_blk, wave, lane).sink(arg=KernelInfo(name="loop_ctrl_apply_decode", opts_to_apply=()))

# ******** fused MoE sigmoid router top-k ********

@functools.cache
def _amd_moe_router_topk_kernel(sel_out:UOp, probs_out:UOp, logits:UOp, *extra:UOp,
                                valid_t:int|UOp, E:int, K:int, has_bias:bool, normalize:bool, scale:float) -> UOp:
  B, valid_t = cast(int, logits.shape[0]), _unbind(valid_t)
  PER = E // WARP_SIZE
  bt = UOp.range(B * valid_t, 0, AxisType.GLOBAL)
  lane = UOp.range(WARP_SIZE, -1, axis_type=AxisType.WARP)
  zerof, neg_inf = UOp.const(0.0, dtypes.float32), UOp.const(-1e30, dtypes.float32)
  lf = logits.reshape(cast(int, logits.shape[0]) * cast(int, logits.shape[1]), E)
  bias = extra[0].reshape(E) if has_bias else None
  s_reg = UOp.alloc((PER,), dtypes.float32, addrspace=AddrSpace.REG)
  r_reg = UOp.alloc((PER,), dtypes.float32, addrspace=AddrSpace.REG)
  init_st: list[UOp] = []
  for j in range(PER):
    e_idx = j * WARP_SIZE + lane
    lv = lf[bt, e_idx].load().float()
    sv = 1.0 / (1.0 + (lv * (-LOG2E)).exp2())
    rv = sv + bias[e_idx].load().float() if bias is not None else sv
    init_st += [s_reg[j].store(sv), r_reg[j].store(rv)]
  s_r, r_cur = s_reg.after(*init_st), r_reg.after(*init_st)
  picked_e: list[UOp] = []
  picked_s: list[UOp] = []
  for _ in range(K):
    best_r, best_e, best_s = r_cur[0].load(), lane.cast(dtypes.int32), s_r[0].load()
    for j in range(1, PER):
      rj, ej = r_cur[j].load(), (j * WARP_SIZE + lane).cast(dtypes.int32)
      better = rj > best_r
      best_r, best_e, best_s = better.where(rj, best_r), better.where(ej, best_e), better.where(s_r[j].load(), best_s)
    w_max = warp_reduce(best_r, Ops.MAX, full_wave=True)
    win_e = warp_reduce(best_r.eq(w_max).where(-best_e, UOp.const(-E, dtypes.int32)), Ops.MAX, full_wave=True).neg()
    win_s = warp_reduce(best_e.eq(win_e).where(best_s, zerof), Ops.ADD, full_wave=True)
    picked_e.append(win_e)
    picked_s.append(win_s)
    r_cur = r_cur.after(*(r_cur[j].store((j * WARP_SIZE + lane).cast(dtypes.int32).eq(win_e).where(neg_inf, r_cur[j].load())) for j in range(PER)))
  factor = (scale / sum(picked_s[1:], picked_s[0])) if normalize else UOp.const(scale, dtypes.float32)
  out_st: list[UOp] = []
  for k in range(K):
    out_st.append(sel_out[bt.valid(lane.eq(0)), k].store(picked_e[k]))
    out_st.append(probs_out[bt.valid(lane.eq(0)), k].store(picked_s[k] * factor))
  return UOp.group(*out_st).end(bt, lane).sink(arg=KernelInfo(name="moe_router_topk", opts_to_apply=()))

# ******** fused RMSNorm (prefill FP16 + decode RMSNorm+Q8) ********

@functools.cache
def _amd_rmsnorm_kernel(out:UOp, x:UOp, w:UOp, *, valid_t:int|UOp, D:int, eps:float, q_cnt:int=1, q_idx:int=0) -> UOp:
  B, valid_t = cast(int, x.shape[0]), _unbind(valid_t)
  WAVES = next(wv for wv in (8, 6, 4, 2, 1) if D % (wv * WARP_SIZE * 4) == 0)
  THREADS, PER = WAVES * WARP_SIZE, D // (WAVES * WARP_SIZE * 4)
  bt = UOp.range(B * valid_t, 0, AxisType.GLOBAL)
  b, t = (bt // valid_t, bt % valid_t) if B > 1 else (UOp.const(0, dtypes.weakint), bt)
  wave, lane = UOp.range(WAVES, 1, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, axis_type=AxisType.WARP)
  tid, zerof = wave * WARP_SIZE + lane, UOp.const(0.0, dtypes.float32)
  xf = x.reshape(B, cast(int, x.shape[1]), q_cnt, D)
  xv = [_vec_load(xf[b, t, q_idx, (p * THREADS + tid) * 4], 4) for p in range(PER)]
  lds = UOp.alloc((WAVES,), dtypes.float32, addrspace=AddrSpace.LOCAL)
  lds_r = lds.after(lds[wave.valid(lane.eq(0))].store(warp_reduce(sum((v * v for ch in xv for v in ch), zerof), full_wave=True)))
  inv_rms = (sum((lds_r[wi].load() for wi in range(WAVES)), zerof) * (1.0 / D) + eps).rsqrt()
  stores: list[UOp] = []
  for p in range(PER):
    off = (p * THREADS + tid) * 4
    wv = _vec_load(w[off], 4)
    for j in range(4):
      stores.append(out[b, t, off + j].store((xv[p][j] * inv_rms * wv[j]).cast(out.dtype)))
  return UOp.group(*stores).end(bt, wave, lane).sink(arg=KernelInfo(name="amd_rmsnorm", opts_to_apply=()))

@functools.cache
def _amd_rmsnorm_q8_kernel(y:UOp, q:UOp, scale:UOp, xsum:UOp, x:UOp, w:UOp, *, BT:int=1, D:int, eps:float, q_cnt:int=1, q_idx:int=0) -> UOp:
  groups = D // WARP_SIZE
  WAVES = next(wv for wv in (8, 6, 4, 2, 1) if groups % wv == 0)
  GPW = groups // WAVES
  bt = UOp.range(BT, 0, AxisType.GLOBAL) if BT > 1 else UOp.const(0, dtypes.weakint)
  wave, lane = UOp.range(WAVES, 1, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, axis_type=AxisType.WARP)
  zerof = UOp.const(0.0, dtypes.float32)
  xf, yf = x.reshape(BT, q_cnt, D), y.reshape(BT, D)
  vals = [xf[bt, q_idx, (wave * GPW + gi) * WARP_SIZE + lane].load().float() for gi in range(GPW)]
  lds = UOp.alloc((WAVES,), dtypes.float32, addrspace=AddrSpace.LOCAL)
  lds_r = lds.after(lds[wave.valid(lane.eq(0))].store(warp_reduce(sum((v * v for v in vals), zerof), full_wave=True)))
  inv_rms = (sum((lds_r[wi].load() for wi in range(WAVES)), zerof) * (1.0 / D) + eps).rsqrt()
  stores: list[UOp] = []
  for gi in range(GPW):
    g = wave * GPW + gi
    idx = g * WARP_SIZE + lane
    yv = vals[gi] * inv_rms * w[idx].load().float()
    stores.append(yf[bt, idx].store(yv))
    wmax = warp_reduce(yv.abs(), Ops.MAX, full_wave=True)
    d = UOp.native("({0}/127.0f)", shape=wmax.shape, dtype=dtypes.float, fmt=True).call(wmax).maximum(1e-8)
    quant = UOp.native("__builtin_nearbyintf({0}/{1})", shape=yv.shape, dtype=dtypes.float, fmt=True).call(yv, d).clip(-127, 127).cast(dtypes.int8)
    word = warp_reduce(quant.cast(dtypes.uint8).cast(dtypes.uint32) << ((lane % 4) * 8).cast(dtypes.uint32), Ops.OR, masks=(1, 2))
    stores.append(q[bt, g, (lane // 4).valid((lane % 4).eq(0))].store(word))
    stores.append(scale[bt, g.valid(lane.eq(0))].store(d))
    stores.append(xsum[bt, g, (lane // 16).valid((lane % 16).eq(0))].store(warp_reduce(quant.float())))
  end_axes = (bt, wave, lane) if BT > 1 else (wave, lane)
  return UOp.group(*stores).end(*end_axes).sink(arg=KernelInfo(name="rmsnorm_q8", opts_to_apply=()))

# ******** fused RelAttention Q/R/K/V prep + KV cache write ********

@functools.cache
def _amd_rel_qr_prep_kernel(q_out:UOp, r_out:UOp, q_raw:UOp, q_clip:UOp, r_raw:UOp, r_clip:UOp, *,
                            valid_t:int|UOp, start_pos:int|UOp, H:int, D:int, L:int,
                            qk_clip:float, rel_clip:float, full:bool, floor:float, alpha:float) -> UOp:
  B, valid_t, start_pos = cast(int, q_out.shape[0]), _unbind(valid_t), _unbind(start_pos)
  DPL = D // WARP_SIZE
  bth = UOp.range(B * valid_t * H, 0, AxisType.GLOBAL)
  lane = UOp.range(WARP_SIZE, -1, axis_type=AxisType.WARP)
  b, rem = (bth // (valid_t * H), bth % (valid_t * H)) if B > 1 else (UOp.const(0, dtypes.weakint), bth)
  t, h = rem // H, rem % H
  zerof = UOp.const(0.0, dtypes.float32)
  T_max = cast(int, q_out.shape[2])
  qf, rf = q_raw.reshape(B, T_max, H, D), r_raw.reshape(B, T_max, H, L)
  qcf, rcf = q_clip.reshape(H, D), r_clip.reshape(H, L)
  qv, cv = _vec_load(qf[b, t, h, lane * DPL], DPL), _vec_load(qcf[h, lane * DPL], DPL)
  qprod = tuple(qv[i] * cv[i] for i in range(DPL))
  q_sq = warp_reduce(sum((v * v for v in qprod), zerof), full_wave=True)
  q_sc = (qk_clip / (q_sq.sqrt() + 1e-6)).minimum(1.0)
  in_l, sl = lane < L, lane.minimum(L - 1)
  rprod = in_l.where(rf[b, t, h, sl].load().float() * rcf[h, sl].load().float(), zerof)
  r_sq = warp_reduce(rprod * rprod, full_wave=True)
  r_sc = (rel_clip / (r_sq.sqrt() + 1e-6)).minimum(1.0)
  if full:
    pos_f = (start_pos + t + 1).cast(dtypes.float32) if isinstance(start_pos, UOp) else (t + (start_pos + 1)).cast(dtypes.float32)
    ls = 1.0 + alpha * ((pos_f * (1.0 / floor)).maximum(1.0).log2() * (1.0 / LOG2E))
    q_sc, r_sc = q_sc * ls, r_sc * ls
  stores = [q_out[b, h, t, lane * DPL + i].store((qprod[i] * q_sc).cast(dtypes.half)) for i in range(DPL)]
  stores.append(r_out[b, h, t, sl.valid(in_l)].store((rprod * r_sc).cast(dtypes.half)))
  return UOp.group(*stores).end(bth, lane).sink(arg=KernelInfo(name="rel_qr_prep", opts_to_apply=()))

@functools.cache
def _amd_rel_kv_cache_kernel(cache_kv:UOp, k_raw:UOp, v_raw:UOp, *,
                             valid_t:int|UOp, start_pos:int|UOp, H_KV:int, D:int, qk_clip:float, eps:float, ring:bool=False) -> UOp:
  B, valid_t, start_pos = cast(int, cache_kv.shape[1]), _unbind(valid_t), _unbind(start_pos)
  physical_n = cast(int, cache_kv.shape[3])
  DPL = D // WARP_SIZE
  T_max = cast(int, k_raw.shape[1])
  bth = UOp.range(B * valid_t * H_KV, 0, AxisType.GLOBAL)
  lane = UOp.range(WARP_SIZE, -1, axis_type=AxisType.WARP)
  b, rem = (bth // (valid_t * H_KV), bth % (valid_t * H_KV)) if B > 1 else (UOp.const(0, dtypes.weakint), bth)
  t, h = rem // H_KV, rem % H_KV
  zerof = UOp.const(0.0, dtypes.float32)
  kf, vf = k_raw.reshape(B, T_max, H_KV, D), v_raw.reshape(B, T_max, H_KV, D)
  kv, vv = _vec_load(kf[b, t, h, lane * DPL], DPL), _vec_load(vf[b, t, h, lane * DPL], DPL)
  k_sc = (qk_clip / (warp_reduce(sum((v * v for v in kv), zerof), full_wave=True).sqrt() + 1e-6)).minimum(1.0)
  v_sc = (warp_reduce(sum((v * v for v in vv), zerof), full_wave=True) * (1.0 / D) + eps).rsqrt()
  dst_t = (start_pos + t) % physical_n if ring else (start_pos + t)
  stores = [cache_kv[0, b, h, dst_t, lane * DPL + i].store((kv[i] * k_sc).cast(cache_kv.dtype)) for i in range(DPL)]
  stores += [cache_kv[1, b, h, dst_t, lane * DPL + i].store((vv[i] * v_sc).cast(cache_kv.dtype)) for i in range(DPL)]
  return UOp.group(*stores).end(bth, lane).sink(arg=KernelInfo(name="rel_kv_cache", opts_to_apply=()))

@functools.cache
def _amd_rel_kv_cache_q4_kernel(cache_kv:UOp, k_raw:UOp, v_raw:UOp, *,
                                valid_t:int|UOp, start_pos:int|UOp, H_KV:int, D:int, qk_clip:float, eps:float, ring:bool=False) -> UOp:
  # Per-64 symmetric 4-bit KV cache: 16 uint32 nibble words + 1 uint32 holding 2 fp16 scales (68 bytes / 128-dim head).
  assert D == 128 and cache_kv.shape[4] == 17
  B, valid_t, start_pos = cast(int, cache_kv.shape[1]), _unbind(valid_t), _unbind(start_pos)
  physical_n = cast(int, cache_kv.shape[3])
  T_max = cast(int, k_raw.shape[1])
  bth = UOp.range(B * valid_t * H_KV, 0, AxisType.GLOBAL)
  lane = UOp.range(WARP_SIZE, -1, axis_type=AxisType.WARP)
  b, rem = (bth // (valid_t * H_KV), bth % (valid_t * H_KV)) if B > 1 else (UOp.const(0, dtypes.weakint), bth)
  t, h = rem // H_KV, rem % H_KV
  zerof = UOp.const(0.0, dtypes.float32)
  kf, vf = k_raw.reshape(B, T_max, H_KV, D), v_raw.reshape(B, T_max, H_KV, D)
  kv, vv = _vec_load(kf[b, t, h, lane * 4], 4), _vec_load(vf[b, t, h, lane * 4], 4)
  k_sc = (qk_clip / (warp_reduce(sum((v * v for v in kv), zerof), full_wave=True).sqrt() + 1e-6)).minimum(1.0)
  v_sc = (warp_reduce(sum((v * v for v in vv), zerof), full_wave=True) * (1.0 / D) + eps).rsqrt()
  dst_t = (start_pos + t) % physical_n if ring else (start_pos + t)
  stores: list[UOp] = []
  for slot, vals in ((0, tuple(kv[i] * k_sc for i in range(4))), (1, tuple(vv[i] * v_sc for i in range(4)))):
    lmax = functools.reduce(UOp.maximum, (v.abs() for v in vals))
    gmax = warp_reduce(lmax, Ops.MAX, masks=(1, 2, 4, 8))
    sc_h = (gmax * (1.0 / 7.0)).cast(dtypes.half)
    sc_f = sc_h.float()
    inv_sc = (sc_f > 0.0).where(1.0 / sc_f, zerof)
    qs = [UOp.native("__builtin_nearbyintf({0})", shape=v.shape, dtype=dtypes.float32, fmt=True).call(v * inv_sc).clip(-7.0, 7.0)
          .cast(dtypes.int32) + 8 for v in vals]
    n4 = (qs[0].cast(dtypes.uint32) & 15) | ((qs[1].cast(dtypes.uint32) & 15) << 4) | \
         ((qs[2].cast(dtypes.uint32) & 15) << 8) | ((qs[3].cast(dtypes.uint32) & 15) << 12)
    pair = n4 | (_lane_xor(n4, 1) << 16)
    stores.append(cache_kv[slot, b, h, dst_t, (lane // 2).valid((lane & 1).eq(0))].store(pair))
    sc_u = sc_h.bitcast(dtypes.uint16).cast(dtypes.uint32)
    sc_pair = sc_u | (_lane_xor(sc_u, 16) << 16)
    stores.append(cache_kv[slot, b, h, dst_t, UOp.const(16, dtypes.weakint).valid(lane.eq(0))].store(sc_pair))
  return UOp.group(*stores).end(bth, lane).sink(arg=KernelInfo(name="rel_kv_cache_q4", opts_to_apply=()))

@functools.cache
def _amd_moe_sum_slots_kernel(out:UOp, x:UOp, *extra:UOp, valid_t:int|UOp, slots:int, D:int, has_add:bool=False) -> UOp:
  B, T_max, valid_t = cast(int, out.shape[0]), cast(int, out.shape[1]), _unbind(valid_t)
  VEC = 8 if x.dtype == dtypes.half and D % (WARP_SIZE * 8) == 0 else 4
  WAVES = next(wv for wv in (8, 6, 4, 3, 2, 1) if D % (wv * WARP_SIZE * VEC) == 0)
  THREADS, PER = WAVES * WARP_SIZE, D // (WAVES * WARP_SIZE * VEC)
  bt = UOp.range(B * valid_t, 0, AxisType.GLOBAL)
  b, t = (bt // valid_t, bt % valid_t) if B > 1 else (UOp.const(0, dtypes.weakint), bt)
  tid = UOp.range(THREADS, 1, AxisType.LOCAL)
  xf = x.reshape(B * T_max, slots, D)
  add_uop = extra[0].reshape(B * T_max, D) if has_add else None
  out_v = out.reshape(B, T_max, D // VEC, VEC)
  row = b * T_max + t
  stores: list[UOp] = []
  for p in range(PER):
    off = (p * THREADS + tid) * VEC
    acc = list(_vec_load(xf[row, 0, off], VEC))
    for s in range(1, slots):
      v = _vec_load(xf[row, s, off], VEC)
      for j in range(VEC): acc[j] = acc[j] + v[j]
    if add_uop is not None:
      if VEC == 8 and add_uop.dtype == dtypes.half: av = _vec_load(add_uop[row, off], 8)
      elif VEC == 8: av = _vec_load(add_uop[row, off], 4) + _vec_load(add_uop[row, off + 4], 4)
      else: av = _vec_load(add_uop[row, off], VEC)
      for j in range(VEC): acc[j] = acc[j] + av[j]
    stores.append(out_v[b, t, p * THREADS + tid].store(UOp.stack(*(acc[j].cast(out.dtype) for j in range(VEC)))))
  return UOp.group(*stores).end(bt, tid).sink(arg=KernelInfo(name="moe_sum_slots", opts_to_apply=()))

@functools.cache
def _amd_situ_kernel(out:UOp, g:UOp, u:UOp, *, valid_t:int|UOp, D:int, b0:float, b1:float) -> UOp:
  B, valid_t = cast(int, out.shape[0]), _unbind(valid_t)
  WAVES = next(wv for wv in (8, 6, 4, 2, 1) if D % (wv * WARP_SIZE * 4) == 0)
  THREADS, PER = WAVES * WARP_SIZE, D // (WAVES * WARP_SIZE * 4)
  bt = UOp.range(B * valid_t, 0, AxisType.GLOBAL)
  b, t = (bt // valid_t, bt % valid_t) if B > 1 else (UOp.const(0, dtypes.weakint), bt)
  tid = UOp.range(THREADS, 1, AxisType.LOCAL)
  inv0, inv1 = 1.0 / b0, 1.0 / b1
  stores: list[UOp] = []
  for p in range(PER):
    off = (p * THREADS + tid) * 4
    gv, uv = _vec_load(g[b, t, off], 4), _vec_load(u[b, t, off], 4)
    for j in range(4):
      val = _fast_tanh(gv[j] * inv0) * b0 * (1.0 / (1.0 + (gv[j] * (-LOG2E)).exp2())) * _fast_tanh(uv[j] * inv1) * b1
      stores.append(out[b, t, off + j].store(val.cast(out.dtype)))
  return UOp.group(*stores).end(bt, tid).sink(arg=KernelInfo(name="amd_situ", opts_to_apply=()))

