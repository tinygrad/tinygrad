from __future__ import annotations
import functools, math
from typing import TYPE_CHECKING, Callable, cast
from tinygrad import Tensor, UOp
from tinygrad.uop.ops import AxisType, KernelInfo, Ops
from tinygrad.dtype import AddrSpace, dtypes
if TYPE_CHECKING: from tinygrad.llm.model import Linear

BLOCK_M, BLOCK_N, DECODE_HEAD_TILE, WARP_SIZE = 32, 32, 8, 32
WMMA_M, WMMA_N, WMMA_K = 16, 16, 16
WAVES_M, WAVES_N, LANES_PER_WAVE_M, LANES_PER_WAVE_N = 2, 2, 2, 16
WMMA_ACC, THREADS_PER_BLOCK = WMMA_M // LANES_PER_WAVE_M, WARP_SIZE * WAVES_M * WAVES_N
LDS_PAD, WMMA_ARG, LOG2E = 4, ((WMMA_M, WMMA_N, WMMA_K), 'AMD', 32), math.log2(math.e)
Q5_K, Q6_K, IQ4_XS, GGML_BLOCK_SIZE, Q8_GROUP_SIZE, Q5_WORDS, Q6_BYTES, IQ4_WORDS = 13, 14, 23, 256, 32, 44, 210, 34

def warp_reduce(val:UOp, lane:UOp, maximum:bool=False, full_wave:bool=False) -> UOp:
  for offset in ([16, 8, 4, 2, 1] if full_wave else [8, 4, 2, 1]):
    idx = ((lane ^ offset) * 4).int()
    if val.op is Ops.INDEX and val.addrspace == AddrSpace.REG: val = val.load()
    other = UOp(Ops.CUSTOM, dtypes.float, (idx, val),
                arg="__builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute({0}, __builtin_bit_cast(int, {1})))")
    val = val.maximum(other) if maximum else val + other
  return val

def _reg(shape:tuple[int, ...], slot:int, value:float, dep:UOp|None=None) -> UOp:
  ret = UOp.placeholder(shape, dtypes.float, slot=slot, addrspace=AddrSpace.REG)
  return ret.after((ret if dep is None else ret.after(dep)).store(ret.const_like(value)))

@functools.cache
def _amd_flash_attention_decode_partial(out, stats, q, cache_kv, cache_scale, valid_kv_len, max_kv_len, block_n):
  _, B, H_KV, N, D = cast(tuple[int, int, int, int, int], cache_kv.shape)
  _, H, M, _ = cast(tuple[int, int, int, int], q.shape)
  assert M == 1 and H % H_KV == 0 and D % WARP_SIZE == 0 and max_kv_len <= N and max_kv_len % block_n == 0
  G, CHUNK, DV, heads_per_wave = H // H_KV, block_n, D // WARP_SIZE, 2
  head_tile = min(DECODE_HEAD_TILE, G)  # share each KV stream across two GQA heads per wave
  assert G % head_tile == 0 and head_tile % heads_per_wave == 0
  decode_waves, decode_group = head_tile // heads_per_wave, 4
  block_bhkv = UOp.range(B*H_KV*(G//head_tile), 0, AxisType.GLOBAL)
  valid_chunks = (valid_kv_len+CHUNK-1)//CHUNK
  group_count = min(valid_chunks, out.shape[2]) if isinstance(valid_chunks, int) else valid_chunks.minimum(out.shape[2])
  block_n = UOp.range(group_count, 1, AxisType.GLOBAL)
  lane, wave = UOp.range(WARP_SIZE, 2, AxisType.LOCAL), UOp.range(decode_waves, 3, AxisType.LOCAL)
  head_group, bhkv = block_bhkv % (G//head_tile), block_bhkv // (G//head_tile)
  b, kv_head = bhkv // H_KV, bhkv % H_KV
  dims = tuple(lane + i*WARP_SIZE for i in range(DV))
  acc, row_max, row_sum = _reg((heads_per_wave, DV), 0, 0), _reg((heads_per_wave,), 1, -math.inf), _reg((heads_per_wave,), 2, 0)
  groups_per_chunk, offset = CHUNK // decode_group, UOp.range(((valid_chunks+group_count-1)//group_count)*(CHUNK//decode_group), 100, AxisType.REDUCE)
  chunk = block_n + (offset // groups_per_chunk) * group_count
  keys = tuple(chunk*CHUNK + (offset % groups_per_chunk)*decode_group + i for i in range(decode_group))
  valid = tuple(key < valid_kv_len for key in keys)
  kvals, vvals = (tuple(tuple(cache_kv[kv, b, kv_head, key, d].float() *
    is_valid.where(cache_scale[kv, b, kv_head, key].float(), UOp.const(0, dtypes.float)) for d in dims)
    for key,is_valid in zip(keys, valid)) for kv in range(2))
  q_heads = tuple(kv_head*G + head_group*head_tile + wave*heads_per_wave + head for head in range(heads_per_wave))
  updates:list[UOp] = []
  for head,q_head in enumerate(q_heads):
    scores = tuple(warp_reduce(sum((q[b, q_head, 0, d].float()*k for d,k in zip(dims, key_kvals)),
                                   UOp.const(0, dtypes.float)), lane + wave*WARP_SIZE, full_wave=True) / math.sqrt(D) for key_kvals in kvals)
    prev_acc, prev_max, prev_sum = acc.after(offset)[head], row_max.after(offset)[head], row_sum.after(offset)[head]
    new_max = functools.reduce(lambda a,vs:a.maximum(vs[0].where(vs[1], UOp.const(-math.inf, dtypes.float))), zip(valid, scores), prev_max)
    alpha = ((prev_max-new_max)*LOG2E).exp2()
    betas = tuple(is_valid.where(((score-new_max)*LOG2E).exp2(), UOp.const(0, dtypes.float)) for is_valid,score in zip(valid, scores))
    updates += [acc[head].store(prev_acc*alpha + sum((UOp.stack(*value)*beta for value,beta in zip(vvals, betas)), acc[head].const_like(0))),
                row_sum[head].store(prev_sum*alpha + sum(betas, UOp.const(0, dtypes.float))), row_max[head].store(new_max)]
  update = UOp.group(*updates).end(offset)
  acc, row_max, row_sum = acc.after(update), row_max.after(update), row_sum.after(update)
  stores = [out[b, q_head, block_n, d].store(acc[head, i]) for head,q_head in enumerate(q_heads) for i,d in enumerate(dims)] + \
    [stats[b, q_head.valid(lane.eq(0)), block_n, i].store(x[head])
     for head,q_head in enumerate(q_heads) for i,x in enumerate((row_max, row_sum))]
  return UOp.group(*stores).end(lane, wave, block_n, block_bhkv).sink(arg=KernelInfo(name="flash_decode_partial", opts_to_apply=()))

def amd_flash_attention_decode(q:Tensor, cache_kv:Tensor, valid_kv_len:int|UOp, cache_scale:Tensor, max_kv_len:int) -> Tensor:
  B, H, D = cache_kv.shape[1], q.shape[1], cache_kv.shape[4]
  block_n = 128
  chunks = min(64, max_kv_len // block_n)
  partial = Tensor.empty(B, H, chunks, D, dtype="float32", device=q.device)
  stats = Tensor.empty(B, H, chunks, 2, dtype="float32", device=q.device)
  decode_partial = functools.partial(_amd_flash_attention_decode_partial, valid_kv_len=valid_kv_len, max_kv_len=max_kv_len, block_n=block_n)
  partial, stats = Tensor.custom_kernel(partial, stats, q, cache_kv, cache_scale, fxn=decode_partial)[:2]
  live_chunks = (valid_kv_len+block_n-1)//block_n
  live_chunks = min(live_chunks, chunks) if isinstance(live_chunks, int) else live_chunks.minimum(chunks)
  partial, stats = partial[:, :, :live_chunks], stats[:, :, :live_chunks]
  weights = ((stats[..., 0]-stats[..., 0].max(2, keepdim=True))*LOG2E).exp2()
  return ((partial*weights.unsqueeze(-1)).sum(2) / (stats[..., 1]*weights).sum(2, keepdim=True)).unsqueeze(2)

@functools.cache
def _amd_flash_attention(o:UOp, q:UOp, cache:UOp, kv_scale:UOp, valid_kv_len:int|UOp) -> UOp:
  BH, M, D = q.shape
  _, B, H_KV, physical_n, cache_dim = cache.shape
  k, v = cache[0].reshape(B*H_KV, physical_n, cache_dim), cache[1].reshape(B*H_KV, physical_n, cache_dim)
  kv_scale = kv_scale.reshape(2, B*H_KV, physical_n)
  assert k.shape == v.shape and BH % k.shape[0] == 0 and k.shape[2] == D
  gqa_group = BH // k.shape[0]
  if isinstance(M, int) and isinstance(valid_kv_len, int):
    assert M % BLOCK_M == 0 and valid_kv_len % BLOCK_N == 0
  assert isinstance(D, int) and D % WMMA_K == 0 and D % LANES_PER_WAVE_N == 0
  TM, TN, TD, SCALE = BLOCK_M//(WAVES_M*LANES_PER_WAVE_M), BLOCK_N//LANES_PER_WAVE_N, D//(WAVES_N*LANES_PER_WAVE_N), 1/math.sqrt(D)
  block_bh, block_m = UOp.range(BH, 0, AxisType.GLOBAL), UOp.range(M // BLOCK_M, 1, AxisType.GLOBAL)
  q = q.reshape(BH, M//BLOCK_M, BLOCK_M, D)[block_bh, block_m]
  kv_head = block_bh // gqa_group
  k, v = k[kv_head], v[kv_head]
  o = o.reshape(BH, M//BLOCK_M, BLOCK_M, D)[block_bh, block_m]
  wave_m, wave_n, lane = UOp.range(WAVES_M, 2, AxisType.LOCAL), UOp.range(WAVES_N, 3, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  tid = (wave_m * WAVES_N + wave_n) * WARP_SIZE + lane
  lane_m, lane_n = lane // LANES_PER_WAVE_N, lane % LANES_PER_WAVE_N
  Q_ELEMS_PER_THREAD, KV_ELEMS_PER_THREAD = BLOCK_M * D // THREADS_PER_BLOCK, BLOCK_N * D // THREADS_PER_BLOCK
  QP_lds = UOp.placeholder((BLOCK_M, D + LDS_PAD), dtypes.half, slot=0, addrspace=AddrSpace.LOCAL)
  KV_lds = UOp.placeholder((BLOCK_N, D + LDS_PAD), dtypes.half, slot=1, addrspace=AddrSpace.LOCAL)[:, :D]
  acc, m_i, l_i = _reg((TM, TD), 2, 0), _reg((TM,), 3, -math.inf), _reg((TM,), 4, 0)
  n_tiles = (valid_kv_len - M + (block_m + 1) * BLOCK_M + BLOCK_N - 1) // BLOCK_N
  n_tile = UOp.range(n_tiles, 100, AxisType.REDUCE)
  Q_lds = QP_lds[:, :D]
  Q_store = Q_lds.after(n_tile).reshape(THREADS_PER_BLOCK, Q_ELEMS_PER_THREAD)[tid].store(q.reshape(THREADS_PER_BLOCK, Q_ELEMS_PER_THREAD)[tid])
  load_k = UOp.range(KV_ELEMS_PER_THREAD, 90, AxisType.WEAK)
  kidx = n_tile*BLOCK_N*D + tid*KV_ELEMS_PER_THREAD + load_k
  kval = k.reshape(physical_n*D)[kidx].float() * kv_scale[0, kv_head, kidx // D].float()
  K_store = KV_lds.reshape(THREADS_PER_BLOCK, KV_ELEMS_PER_THREAD)[tid, load_k].store(kval).end(load_k)
  qk_load_barrier = UOp.barrier(UOp.group(Q_store, K_store))
  Q_lds, KV_lds_k = Q_lds.after(qk_load_barrier), KV_lds.after(qk_load_barrier)
  S_reg = _reg((TM, TN), 6, 0, n_tile)
  k_qk, tm1, tn1 = UOp.range(D//WMMA_K, 101, AxisType.REDUCE), UOp.range(TM//WMMA_ACC, 200), UOp.range(TN, 201)
  S_frag = S_reg.reshape(TM // WMMA_ACC, WMMA_ACC, TN).permute(0, 2, 1)[tm1, tn1]
  q_frag = Q_lds.reshape(WAVES_M, TM // WMMA_ACC, WMMA_M, D // WMMA_K, WMMA_K)[wave_m, tm1, lane_n, k_qk]
  k_frag = KV_lds_k.reshape(TN, WMMA_N, D // WMMA_K, WMMA_K)[tn1, lane_n, k_qk]
  qk_done = S_frag.store(UOp.wmma(q_frag, k_frag, S_frag.after(k_qk), *WMMA_ARG)).end(tm1, tn1).end(k_qk)
  S_reg = S_reg.after(qk_done)
  S_reg = S_reg.after(S_reg.store(S_reg * SCALE))
  rm, rn = UOp.range(TM, 250, AxisType.WEAK), UOp.range(TN, 251, AxisType.WEAK)
  q_idx = valid_kv_len - M + block_m * BLOCK_M + wave_m * WMMA_M + rm * LANES_PER_WAVE_M + lane_m
  k_idx = n_tile * BLOCK_N + rn * LANES_PER_WAVE_N + lane_n
  valid = k_idx <= q_idx
  S_reg = S_reg.after(S_reg[rm, rn].store(valid.where(S_reg[rm, rn], S_reg[rm, rn].const_like(-math.inf))).end(rm, rn))
  m_ij, rm2 = _reg((TM,), 7, -math.inf, n_tile), UOp.range(TN, 261, AxisType.REDUCE)
  m_ij = m_ij.after(m_ij.store(m_ij.after(rm2).maximum(S_reg[:, rm2])).end(rm2))
  ri_w = UOp.range(TM, 270)
  m_ij = m_ij.after(m_ij[ri_w].store(warp_reduce(m_ij[ri_w], lane, maximum=True)).end(ri_w))
  S_reg = S_reg.after(S_reg.store(((S_reg - m_ij.reshape(TM, 1).expand(TM, TN)) * LOG2E).exp2()))
  p_local, ri_ws = _reg((TM,), 8, 0, n_tile), UOp.range(TM, 295, AxisType.WEAK)
  p_sum = p_local.after(p_local[ri_ws].store(sum((warp_reduce(S_reg[ri_ws, rn], lane) for rn in range(TN)), S_reg.const_like(0))).end(ri_ws))
  P_lds = QP_lds.flatten()[:WAVES_N * BLOCK_M * BLOCK_N].reshape(WAVES_N, BLOCK_M, BLOCK_N)
  P_write = P_lds.reshape(WAVES_N, WAVES_M, TM, LANES_PER_WAVE_M, 1, TN, LANES_PER_WAVE_N, 1)
  P_write = P_write.permute((1, 0, 3, 6, 2, 4, 5, 7)).reshape(THREADS_PER_BLOCK, TM, TN)
  P_store = P_write[tid].store(S_reg.cast(dtypes.half))
  beta_i, ri4 = UOp.placeholder((TM,), dtypes.float, slot=9, addrspace=AddrSpace.REG), UOp.range(TM, 330, AxisType.WEAK)
  m_new_val = m_i[ri4].maximum(m_ij[ri4])
  alpha_val = ((m_i[ri4] - m_new_val) * LOG2E).exp2()
  beta_val = ((m_ij[ri4] - m_new_val) * LOG2E).exp2()
  rj4 = UOp.range(TD, 331)
  correction = UOp.group(acc[ri4, rj4].store(alpha_val * acc[ri4, rj4]).end(rj4),
                         l_i[ri4].store(alpha_val * l_i[ri4] + beta_val * p_sum[ri4]),
                         m_i[ri4].store(m_new_val), beta_i[ri4].store(beta_val)).end(ri4)
  acc, l_i, m_i, beta_i = acc.after(correction), l_i.after(correction), m_i.after(correction), beta_i.after(correction)
  V_lds = UOp.placeholder((D, BLOCK_N + LDS_PAD), dtypes.half, slot=1, addrspace=AddrSpace.LOCAL)[:, :BLOCK_N]
  V_copy, load_v = V_lds.after(qk_done).permute(1, 0), UOp.range(KV_ELEMS_PER_THREAD, 390, AxisType.WEAK)
  vidx = n_tile*BLOCK_N*D + tid*KV_ELEMS_PER_THREAD + load_v
  vval = v.reshape(physical_n*D)[vidx].float() * kv_scale[1, kv_head, vidx // D].float()
  V_store = V_copy.reshape(THREADS_PER_BLOCK, KV_ELEMS_PER_THREAD)[tid, load_v].store(vval).end(load_v)
  pv_barrier = UOp.barrier(UOp.group(P_store, V_store))
  P_lds, V_lds = P_lds.after(pv_barrier), V_lds.after(pv_barrier)
  pv_acc = _reg((TM, TD), 10, 0, n_tile).after(pv_barrier)
  k_pv, tm2, tn2 = UOp.range(BLOCK_N//WMMA_K, 400, AxisType.REDUCE), UOp.range(TM//WMMA_ACC, 401, AxisType.WEAK), UOp.range(TD, 402, AxisType.WEAK)
  pv_frag = pv_acc.reshape(TM // WMMA_ACC, WMMA_ACC, TD).permute(0, 2, 1)[tm2, tn2]
  p_frag = P_lds[wave_n].reshape(WAVES_M, TM // WMMA_ACC, WMMA_M, BLOCK_N // WMMA_K, WMMA_K)[wave_m, tm2, lane_n, k_pv]
  v_frag = V_lds.reshape(WAVES_N, TD, WMMA_N, BLOCK_N // WMMA_K, WMMA_K)[wave_n, tn2, lane_n, k_pv]
  pv_done = pv_frag.store(UOp.wmma(p_frag, v_frag, pv_frag.after(k_pv), *WMMA_ARG)).end(tm2, tn2).end(k_pv)
  pv_acc = pv_acc.after(pv_done)
  ri5, rj5 = UOp.range(TM, 410, AxisType.WEAK), UOp.range(TD, 411, AxisType.WEAK)
  accumulate = acc[ri5, rj5].store(acc[ri5, rj5] + beta_i[ri5] * pv_acc[ri5, rj5]).end(ri5, rj5)
  n_tile_end = accumulate.barrier().end(n_tile)
  acc, l_i, m_i = acc.after(n_tile_end), l_i.after(n_tile_end), m_i.after(n_tile_end)
  acc = acc.after(acc.store(acc * (1 / l_i).reshape(TM, 1).expand(TM, TD)))
  o = o.reshape(WAVES_M, TM, LANES_PER_WAVE_M, 1, WAVES_N, TD, LANES_PER_WAVE_N, 1)
  o = o.permute((0, 4, 2, 6, 1, 3, 5, 7)).reshape(THREADS_PER_BLOCK, TM, TD)
  return o[tid].store(acc).end(wave_m, wave_n, lane).end(block_m, block_bh).sink(arg=KernelInfo(opts_to_apply=()))

def flash_attention_causal_cached(q:Tensor, cache_kv:Tensor, valid_kv_len:int|UOp, cache_scale:Tensor) -> Tensor:
  B, H, T, D = cast(tuple[int, int, int, int], q.shape)
  out = Tensor.empty(B*H, T, D, dtype="float32", device=q.device)
  flash_cached = functools.partial(_amd_flash_attention, valid_kv_len=valid_kv_len)
  return Tensor.custom_kernel(out, q.reshape(B*H, T, D), cache_kv, cache_scale, fxn=flash_cached)[0].reshape(B, H, T, D)

def _amd_dp4a(a:UOp, b:UOp, c:UOp) -> UOp:
  return UOp(Ops.CUSTOMI, dtypes.int32, (a.int(), b.int(), c), arg="__builtin_amdgcn_sudot4(true, {}, true, {}, {}, false)")

def _amd_byte_perm(a:UOp, b:UOp, selectors:UOp) -> UOp:
  return UOp(Ops.CUSTOMI, dtypes.uint32, tuple(x.cast(dtypes.uint32) for x in (a, b, selectors)), arg="__builtin_amdgcn_perm({}, {}, {})")

def _amd_load(ptr:UOp, lanes:int|None=None) -> UOp:
  assert ptr.op is Ops.INDEX
  if lanes is None: return UOp(Ops.CUSTOMI, ptr.dtype, (ptr,), arg="__builtin_nontemporal_load({0})")
  buf, coords = ptr.src[0], ptr.src[1:]
  idx = sum((coord*math.prod(buf.shape[i+1:]) for i,coord in enumerate(coords)), UOp.const(0, dtypes.weakint))
  return UOp(Ops.SHRINK, src=(buf.flatten(), idx, UOp.const(lanes, dtypes.weakint))).load(dtype=ptr.dtype)

def _load_byte(raw:UOp, base:UOp, offset:UOp) -> UOp: return (raw[base + offset//4] >> ((offset&3)*8).cast(dtypes.uint32)) & 255
def _half(value:UOp) -> UOp: return value.cast(dtypes.uint16).bitcast(dtypes.float16).float()

def _iq4_bytes(packed:UOp, shift:int) -> UOp:
  selectors = (packed >> shift) & 0x0f0f0f0f
  low = _amd_byte_perm(UOp.const(0xf6eaddcf, dtypes.uint32), UOp.const(0xbfad9881, dtypes.uint32), selectors)
  high = _amd_byte_perm(UOp.const(0x71594535, dtypes.uint32), UOp.const(0x26190d01, dtypes.uint32), selectors & 0x07070707)
  return _amd_byte_perm(high, low, 0x03020100 | ((selectors & 0x08080808) >> 1))

def q8_quantize(x:Tensor, tokens:int, in_features:int) -> tuple[Tensor, Tensor]:
  groups = x.float().reshape(tokens, in_features//Q8_GROUP_SIZE, Q8_GROUP_SIZE)
  scale = (groups.abs().max(-1)/127).maximum(1e-8)
  return (groups/scale.unsqueeze(-1)).round().clip(-127, 127).cast(dtypes.int8).contiguous().bitcast(dtypes.uint32), scale

@functools.cache
def _gated_delta_prefill_kernel(core:UOp, q:UOp, k:UOp, v:UOp, beta:UOp, alpha:UOp, state:UOp, kq:UOp) -> UOp:
  batch, heads, tokens, dim, row_tile = *core.shape, 4
  assert all(isinstance(x, int) for x in (batch, heads, tokens, dim)) and dim % 32 == 0 and dim % row_tile == 0
  batch, heads, tokens, dim = cast(tuple[int, int, int, int], (batch, heads, tokens, dim))
  core, q, k, v = (x.reshape(batch*heads, tokens, dim) for x in (core, q, k, v))
  beta, alpha, kq = (x.reshape(batch*heads, tokens) for x in (beta, alpha, kq))
  state = state.reshape(batch*heads, dim, dim)
  bh_row, lane = UOp.range(batch*heads*dim//row_tile, 0), UOp.range(32, 1, axis_type=AxisType.LOCAL)
  bh, row_base = bh_row // (dim//row_tile), (bh_row % (dim//row_tile))*row_tile
  rows = tuple(row_base+i for i in range(row_tile))
  cols = tuple(lane + i*32 for i in range(dim//32))
  current = UOp.placeholder((row_tile*dim//32,), dtypes.float32, slot=0, addrspace=AddrSpace.REG)
  current = current.after(current.store(UOp.stack(*(state[bh, row, col].float() for row in rows for col in cols))))
  token = UOp.range(tokens, 2, AxisType.REDUCE)
  keys = tuple(k[bh, token, col].load() for col in cols)
  queries = tuple(q[bh, token, col].load() for col in cols)
  av, bv = alpha[bh, token].load(), beta[bh, token].load()
  updates:list[UOp] = []
  stores:list[UOp] = []
  for row_idx,row in enumerate(rows):
    previous = tuple(current.after(token)[row_idx*dim//32+i].load() for i in range(dim//32))
    state_k = warp_reduce(sum((x*y for x,y in zip(previous, keys)), UOp.const(0, dtypes.float32)), lane, full_wave=True)
    state_q = warp_reduce(sum((x*y for x,y in zip(previous, queries)), UOp.const(0, dtypes.float32)), lane, full_wave=True)
    delta = (v[bh, token, row].load() - state_k*av) * bv
    updates += [x*av + delta*y for x,y in zip(previous, keys)]
    stores.append(core[bh, token, row.valid(lane.eq(0))].store(state_q*av + delta*kq[bh, token]))
  step = UOp.group(*stores, current.store(UOp.stack(*updates))).end(token)
  state_stores = (state[bh, row, col].store(current.after(step)[row_idx*dim//32+i].load().cast(state.dtype))
                  for row_idx,row in enumerate(rows) for i,col in enumerate(cols))
  return UOp.group(*state_stores).end(lane, bh_row).sink(arg=KernelInfo(name="gated_delta_prefill", opts_to_apply=()))

def gated_delta_prefill(q:Tensor, k:Tensor, v:Tensor, beta:Tensor, alpha:Tensor, state:Tensor) -> Tensor:
  batch, heads, tokens, dim = q.shape
  assert q.shape == k.shape == v.shape and beta.shape == alpha.shape == (batch, heads, tokens) and state.shape == (batch, heads, dim, dim)
  core, kq = Tensor.empty_like(q), (q*k).sum(-1).contiguous()
  return Tensor.custom_kernel(core, q.contiguous(), k.contiguous(), v.contiguous(), beta.contiguous(), alpha.contiguous(), state, kq,
                              fxn=_gated_delta_prefill_kernel)[0]

def _wmma_layout(out:UOp, out_features:int, token_tile:int, output_tiles:int):
  output_waves = 2 if out_features % (32*output_tiles) == 0 else 1
  token_block, output_block = UOp.range(out.shape[0]//token_tile, 0), UOp.range(out_features//(16*output_tiles*output_waves), 1)
  lane, wave = UOp.range(WARP_SIZE, 2, axis_type=AxisType.LOCAL), UOp.range(output_waves, 3, axis_type=AxisType.LOCAL)
  hw_lane = UOp(Ops.CUSTOM, dtypes.int32, (lane.int(),), arg="__builtin_amdgcn_mbcnt_lo(-1, 0)").cast(dtypes.weakint)
  col, half = hw_lane % 16, hw_lane // 16
  outputs = tuple((output_block*output_waves+wave)*(16*output_tiles) + tile*16 + col for tile in range(output_tiles))
  inputs = tuple(token_block*token_tile + tile*16 + col for tile in range(token_tile//16))
  tokens = tuple(tuple(token_block*token_tile + tile*16 + half*8 + i for i in range(8)) for tile in range(token_tile//16))
  return output_waves, token_block, output_block, lane, wave, half, outputs, inputs, tokens

def _wmma_stores(out, outputs, tokens, accs, update, half):
  def values(acc:UOp) -> tuple[UOp, ...]:
    vals = tuple(acc.after(update)[i].load() for i in range(8))
    swapped = tuple(UOp(Ops.CUSTOM, dtypes.float32, (value,),
      arg="__builtin_bit_cast(float, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, {0}), 50688))") for value in vals)
    low = half.eq(0)
    return tuple(low.where(vals[i], swapped[i+4]) if j == 0 else low.where(swapped[i], vals[i+4]) for i in range(4) for j in range(2))
  return [out[token, output].store(value) for output,output_accs in zip(outputs, accs)
          for tile_tokens,acc in zip(tokens, output_accs) for token,value in zip(tile_tokens, values(acc))]

def _decode_linear(out:UOp, out_features:int, group_count:int, group_dot, name:str) -> UOp:
  output, lane = UOp.range(out_features, 0), UOp.range(32, 1, axis_type=AxisType.LOCAL)
  acc = UOp.placeholder((1,), dtypes.float32, slot=0, addrspace=AddrSpace.REG)[0].set(0)
  chunk = UOp.range((group_count+31)//32, 2, AxisType.REDUCE)
  group = (lane+chunk*32).valid(lane+chunk*32 < group_count)
  acc = acc[0].set(acc.after(chunk)[0] + group_dot(output, group), end=chunk)
  total = warp_reduce(acc[0].load(), lane, full_wave=True)
  return out[0, output.valid(lane.eq(0))].store(total.cast(out.dtype)).end(output, lane).sink(arg=KernelInfo(name=name, opts_to_apply=()))

def _q5_scales(raw:UOp, base:UOp, subgroup:UOp) -> tuple[UOp, UOp, UOp, UOp]:
  scale = (subgroup < 4).where(_load_byte(raw, base, 4 + subgroup) & 63,
    (_load_byte(raw, base, 8 + subgroup) & 15) | ((_load_byte(raw, base, subgroup) >> 6) << 4))
  minimum = (subgroup < 4).where(_load_byte(raw, base, 8 + subgroup) & 63,
    (_load_byte(raw, base, 8 + subgroup) >> 4) | ((_load_byte(raw, base, 4 + subgroup) >> 6) << 4))
  d, dmin = (raw[base] & 0xffff).cast(dtypes.uint16), (raw[base] >> 16).cast(dtypes.uint16)
  return _half(d), _half(dmin), scale.float(), minimum.float()

@functools.cache
def _quant_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, raw_offset:UOp, out_features:int, in_features:int, ggml_type:int) -> UOp:
  group_count = in_features // Q8_GROUP_SIZE
  def group_dot(output:UOp, group:UOp) -> UOp:
    block, subgroup = group // 8, group % 8
    xwords = _amd_load(xq[0, group, 0], 8)
    if ggml_type == Q5_K:
      base = raw_offset + (output * in_features//GGML_BLOCK_SIZE + block) * Q5_WORDS
      qs_base, dot, qsum = base + 12 + (subgroup//2)*8, UOp.const(0, dtypes.int32), UOp.const(0, dtypes.int32)
      for word_idx in range(8):
        word = (raw[qs_base+word_idx] >> ((subgroup&1)*4).cast(dtypes.uint32)) & 0x0f0f0f0f
        word |= ((raw[base+4+word_idx] >> subgroup.cast(dtypes.uint32)) & 0x01010101) << 4
        dot, qsum = _amd_dp4a(word, xwords[word_idx], dot), _amd_dp4a(UOp.const(0x01010101, dtypes.uint32), xwords[word_idx], qsum)
      d, dmin, scale, minimum = _q5_scales(raw, base, subgroup)
      return (dot.float()*d*scale - qsum.float()*dmin*minimum) * xd[0, group]
    if ggml_type == IQ4_XS:
      base = raw_offset + (output * in_features//GGML_BLOCK_SIZE + block) * IQ4_WORDS
      dot = UOp.const(0, dtypes.int32)
      for word_idx in range(8):
        packed = _amd_load(raw[base + 2 + subgroup*4 + word_idx%4])
        dot = _amd_dp4a(_iq4_bytes(packed, 4*(word_idx//4)), xwords[word_idx], dot)
      d, scale = _iq4_scales(raw, base, subgroup)
      return dot.float() * xd[0, group] * d * scale
    base = raw_offset*4 + (output*in_features//GGML_BLOCK_SIZE+block)*Q6_BYTES
    dots = [UOp.const(0, dtypes.int32), UOp.const(0, dtypes.int32)]
    for word_idx in range(8):
      pos, within = subgroup*32 + word_idx*4, (subgroup*32 + word_idx*4)%128
      low = _amd_load(raw[base + (pos//128)*64 + within%64], 4) >> ((within//64)*4).cast(dtypes.uint8)
      high = _amd_load(raw[base + 128 + (pos//128)*32 + within%32], 4) >> ((within//32)*2).cast(dtypes.uint8)
      quant = ((low & 15) | ((high & 3) << 4)).bitcast(dtypes.int8) - 32
      word = sum((quant[i].cast(dtypes.uint8).cast(dtypes.uint32) << (i*8) for i in range(4)), UOp.const(0, dtypes.uint32))
      dots[word_idx//4] = _amd_dp4a(word, xwords[word_idx], dots[word_idx//4])
    scales = [raw[base + 192 + subgroup*2+i].cast(dtypes.uint8).bitcast(dtypes.int8).float() for i in range(2)]
    dbits = raw[base+208].cast(dtypes.uint16) | (raw[base+209].cast(dtypes.uint16) << 8)
    return (dots[0].float()*scales[0] + dots[1].float()*scales[1]) * xd[0, group] * _half(dbits)
  return _decode_linear(out, out_features, group_count, group_dot, {Q5_K:"linear_q5_k", IQ4_XS:"linear_iq4_xs", Q6_K:"linear_q6"}[ggml_type])

def _quant_linear_wmma(out, x, out_features, in_features, type_words, layout, dequant, name):
  x = x.reshape(out.shape[0], in_features)
  _, token_block, output_block, lane, wave, physical_half, outputs, input_tokens, tokens = layout
  token_tile, output_tiles = len(tokens)*16, len(outputs)
  output_words = in_features // GGML_BLOCK_SIZE * type_words
  accs = tuple(tuple(UOp.placeholder((8,), dtypes.float32, slot=ot*(token_tile//16)+tile, addrspace=AddrSpace.REG)
                     for tile in range(token_tile // 16)) for ot in range(output_tiles))
  accs = tuple(tuple(acc.after(acc.store(acc.const_like(0))) for acc in output_accs) for output_accs in accs)
  group = UOp.range(in_features // Q8_GROUP_SIZE, 4, AxisType.REDUCE)
  block, subgroup = group // 8, group % 8
  wmma_accs = [list(output_accs) for output_accs in accs]
  for half in range(2):
    afrags = tuple(UOp.stack(*(x[input_token, group*32 + half*16 + i].cast(dtypes.float16) for i in range(16)))
                   for input_token in input_tokens)
    for output_tile,output in enumerate(outputs):
      bfrag = UOp.stack(*dequant(output*output_words + block*type_words, subgroup, half))
      for tile,afrag in enumerate(afrags):
        previous = accs[output_tile][tile].after(group) if half == 0 else wmma_accs[output_tile][tile]
        wmma_accs[output_tile][tile] = UOp.wmma(afrag, bfrag, previous, *WMMA_ARG)
  update = UOp.group(*(acc.store(value) for output_accs,output_values in zip(accs, wmma_accs)
                       for acc,value in zip(output_accs, output_values))).end(group)
  return UOp.group(*_wmma_stores(out, outputs, tokens, accs, update, physical_half)).end(token_block, output_block, lane, wave).sink(
    arg=KernelInfo(name=name, opts_to_apply=()))

@functools.cache
def _q5_linear_f16_wmma_kernel(out:UOp, raw:UOp, x:UOp, raw_offset:UOp, out_features:int, in_features:int) -> UOp:
  token_tile, output_tiles = (64, 1) if out_features <= 1024 and out.shape[0] % 64 == 0 else \
    (64, 2) if out.shape[0] % 64 == 0 else (32 if out.shape[0] % 32 == 0 else 16, 2)
  def dequant(base:UOp, subgroup:UOp, half:int) -> tuple[UOp, ...]:
    base = raw_offset + base
    d, dmin, scale, minimum = _q5_scales(raw, base, subgroup)
    qs_base = base + 12 + (subgroup // 2)*8 + half*4
    words = tuple((raw[qs_base+i] >> ((subgroup&1)*4).cast(dtypes.uint32) & 0x0f0f0f0f) |
      ((raw[base+4+half*4+i] >> subgroup.cast(dtypes.uint32) & 0x01010101) << 4) for i in range(4))
    return tuple(((word >> (byte*8) & 255).float()*d*scale-dmin*minimum).cast(dtypes.float16) for word in words for byte in range(4))
  return _quant_linear_wmma(out, x, out_features, in_features, Q5_WORDS,
                            _wmma_layout(out, out_features, token_tile, output_tiles), dequant, "linear_q5_k_f16_wmma")

def _iq4_scales(raw:UOp, base:UOp, subgroup:UOp) -> tuple[UOp, UOp]:
  low = _load_byte(raw, base, 4 + subgroup//2)
  scale = ((low >> (4*(subgroup%2)).cast(dtypes.uint32)) & 15) | ((((raw[base] >> 16) >> (2*subgroup).cast(dtypes.uint32)) & 3) << 4)
  return _half(raw[base] & 0xffff), (scale.cast(dtypes.uint8).bitcast(dtypes.int8)-32).float()

@functools.cache
def _iq4_linear_f16_wmma_kernel(out:UOp, raw:UOp, x:UOp, lut:UOp, raw_offset:UOp, out_features:int, in_features:int) -> UOp:
  token_tile = 32 if out_features <= 1024 and out.shape[0] % 32 == 0 else 64 if out.shape[0] % 64 == 0 and \
    (out_features <= 6144 or out_features == 5120 and in_features > 8192) else 128 if out.shape[0] % 128 == 0 else \
    32 if out.shape[0] % 32 == 0 else 16
  output_tiles = 1 if out_features <= 1024 else 2 if out_features <= 6144 else 1 if out_features < 8192 else 2
  layout = _wmma_layout(out, out_features, token_tile, output_tiles)
  output_waves, _, _, lane, wave, _, _, _, _ = layout
  local_lut = UOp.placeholder((256,), dtypes.uint32, slot=32, addrspace=AddrSpace.LOCAL)
  tid, lut_items = wave*32+lane, 256//(32*output_waves)
  lut = local_lut.after(UOp.group(*(local_lut[tid*lut_items+i].store(lut[tid*lut_items+i]) for i in range(lut_items))).barrier())
  def dequant(base:UOp, subgroup:UOp, half:int) -> tuple[UOp, ...]:
    base = raw_offset + base
    d, scale = _iq4_scales(raw, base, subgroup)
    scale = scale * d
    if out_features <= 6144:
      pairs = tuple(lut[((raw[base + 2 + subgroup*4 + word] >> (byte*8)) & 255).cast(dtypes.weakint)]
                    for word in range(4) for byte in range(4))
      return tuple((_half((pair >> (half*16)) & 0xffff)*scale).cast(dtypes.float16) for pair in pairs)
    def nibble(packed:UOp, index:int): return (packed >> (8*index+4*half)) & 15
    lut_pairs = (lut[(nibble(packed, i) | nibble(packed, i+1)<<4).cast(dtypes.weakint)]
                 for packed in (raw[base+2+subgroup*4+i] for i in range(4)) for i in (0, 2))
    return tuple((_half((pair >> (i*16)) & 0xffff)*scale).cast(dtypes.float16) for pair in lut_pairs for i in range(2))
  return _quant_linear_wmma(out, x, out_features, in_features, IQ4_WORDS, layout, dequant, "linear_iq4_xs_f16_wmma")

def q8_linear(layer:Linear, x:Tensor) -> Tensor:
  assert layer.ggml_type in (Q5_K, Q6_K, IQ4_XS) and layer._raw_offset_uop is not None
  tokens = int(x.numel()) // layer.in_features
  out = Tensor.empty(tokens, layer.out_features, dtype=dtypes.float32, device=x.device).uop
  raw, offset = layer.weight.uop.buf_uop, layer._raw_offset_uop
  out_features, in_features = layer.out_features, layer.in_features
  use_wmma = tokens % 16 == 0 and layer.out_features % 16 == 0
  def run(fxn:Callable[..., UOp], *srcs:UOp) -> Tensor:
    params = tuple(UOp.placeholder_like(src, slot=i) for i,src in enumerate(srcs))
    kernel = fxn(*params[:-1], params[-1][0], out_features=out_features, in_features=in_features).call(*srcs)
    result = Tensor(srcs[0].after(kernel)).reshape(*x.shape[:-1], layer.out_features)
    return result if layer.bias is None else result + layer.bias

  if layer.ggml_type == Q5_K and use_wmma:
    return run(_q5_linear_f16_wmma_kernel, out, raw.bitcast(dtypes.uint32), x.cast(dtypes.float16).contiguous().uop, offset)
  if layer.ggml_type == IQ4_XS and use_wmma:
    return run(_iq4_linear_f16_wmma_kernel, out, raw.bitcast(dtypes.uint32), x.cast(dtypes.float16).contiguous().uop,
               iq4_half_lut(str(x.device)).uop, offset)
  xq, xd = q8_quantize(x, tokens, layer.in_features)
  decode = functools.partial(_quant_decode_kernel, ggml_type=layer.ggml_type)
  return run(decode, out, raw if layer.ggml_type == Q6_K else raw.bitcast(dtypes.uint32), xq.uop, xd.uop, offset)

@functools.cache
def iq4_half_lut(device:str) -> Tensor:
  from tinygrad.runtime.autogen.ggml_common import kvalues_iq4nl
  return Tensor([x for j in range(16) for i in range(16) for x in (kvalues_iq4nl[i], kvalues_iq4nl[j])],
                dtype=dtypes.float16, device=device).bitcast(dtypes.uint32).contiguous().realize()
