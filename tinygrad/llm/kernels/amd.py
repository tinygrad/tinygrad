from __future__ import annotations
import functools, math
from typing import TYPE_CHECKING, cast
from tinygrad import Tensor, UOp
from tinygrad.uop.ops import AxisType, KernelInfo, Ops
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.llm.gguf import _GGML_QUANT
if TYPE_CHECKING:
  from tinygrad.llm.model import Embedding, Linear

BLOCK_M, BLOCK_N = 32, 32
DECODE_HEAD_TILE = 8
WARP_SIZE = 32
WMMA_M, WMMA_N, WMMA_K = 16, 16, 16
WAVES_M, WAVES_N = 2, 2
LANES_PER_WAVE_M, LANES_PER_WAVE_N = 2, 16
WMMA_ACC = WMMA_M // LANES_PER_WAVE_M
THREADS_PER_BLOCK = WARP_SIZE * WAVES_M * WAVES_N
LDS_PAD = 4  # pad LDS rows to reduce bank conflicts

WMMA_ARG = (WMMA_M, WMMA_N, WMMA_K), 'AMD', 32
LOG2E = math.log2(math.e)

def warp_shfl_xor(val, offset, lane):
  """Read val from lane ^ offset using ds_bpermute."""
  idx = ((lane ^ offset) * 4).int()
  if val.op is Ops.INDEX and val.addrspace == AddrSpace.REG: val = val.load()
  return UOp(Ops.CUSTOM, dtypes.float, (idx, val),
             arg="__builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute({0}, __builtin_bit_cast(int, {1})))")

def warp_reduce_max(val, lane):
  """Tree reduce MAX across LANES_PER_WAVE_N=16 lanes."""
  for offset in [8, 4, 2, 1]:
    val = val.maximum(warp_shfl_xor(val, offset, lane))
  return val

def warp_reduce_sum(val, lane):
  """Tree reduce SUM across LANES_PER_WAVE_N=16 lanes."""
  for offset in [8, 4, 2, 1]:
    val = val + warp_shfl_xor(val, offset, lane)
  return val

def wave_reduce_sum(val, lane):
  for offset in [16, 8, 4, 2, 1]: val = val + warp_shfl_xor(val, offset, lane)
  return val

@functools.cache
def _amd_flash_attention_decode_partial(out:UOp, stats:UOp, q:UOp, cache_kv:UOp, cache_scale:UOp|None,
                                        valid_kv_len:int|UOp, max_kv_len:int, block_n:int) -> UOp:
  _, B, H_KV, N, D = cast(tuple[int, int, int, int, int], cache_kv.shape)
  _, H, M, _ = cast(tuple[int, int, int, int], q.shape)
  assert M == 1 and H % H_KV == 0 and D % WARP_SIZE == 0 and max_kv_len <= N and max_kv_len % block_n == 0
  G, CHUNK, DV = H // H_KV, block_n, D // WARP_SIZE
  # Each wave owns two GQA query heads while the workgroup shares one KV head. This keeps per-wave register pressure low
  # and lets the cache coalesce the identical KV stream instead of launching a second workgroup for the same KV head.
  heads_per_wave = 2
  head_tile = min(DECODE_HEAD_TILE, G)
  assert G % head_tile == 0 and head_tile % heads_per_wave == 0
  decode_waves = head_tile // heads_per_wave
  decode_group = 4
  block_bhkv = UOp.range(B*H_KV*(G//head_tile), 0, AxisType.GLOBAL)
  valid_chunks = (valid_kv_len+CHUNK-1)//CHUNK
  group_count = min(valid_chunks, out.shape[2]) if isinstance(valid_chunks, int) else valid_chunks.minimum(out.shape[2])
  block_n = UOp.range(group_count, 1, AxisType.GLOBAL)
  lane, wave = UOp.range(WARP_SIZE, 2, AxisType.LOCAL), UOp.range(decode_waves, 3, AxisType.LOCAL)
  head_group = block_bhkv % (G//head_tile)
  bhkv = block_bhkv // (G//head_tile)
  b, kv_head = bhkv // H_KV, bhkv % H_KV
  dims = tuple(lane + i*WARP_SIZE for i in range(DV))

  acc = UOp.placeholder((heads_per_wave, DV), dtypes.float, slot=0, addrspace=AddrSpace.REG)
  row_max = UOp.placeholder((heads_per_wave,), dtypes.float, slot=1, addrspace=AddrSpace.REG)
  row_sum = UOp.placeholder((heads_per_wave,), dtypes.float, slot=2, addrspace=AddrSpace.REG)
  init = UOp.group(acc.store(acc.const_like(0)), row_max.store(row_max.const_like(-math.inf)), row_sum.store(row_sum.const_like(0)))
  acc, row_max, row_sum = acc.after(init), row_max.after(init), row_sum.after(init)

  groups_per_chunk = CHUNK // decode_group
  offset = UOp.range(((valid_chunks+group_count-1)//group_count)*groups_per_chunk, 100, AxisType.REDUCE)
  chunk = block_n + (offset // groups_per_chunk) * group_count
  keys = tuple(chunk*CHUNK + (offset % groups_per_chunk)*decode_group + i for i in range(decode_group))
  valid = tuple(key < valid_kv_len for key in keys)
  kscales = tuple(cache_scale[0, b, kv_head, key].float() if cache_scale is not None else UOp.const(1, dtypes.float) for key in keys)
  vscales = tuple(cache_scale[1, b, kv_head, key].float() if cache_scale is not None else UOp.const(1, dtypes.float) for key in keys)
  kvals = tuple(tuple(cache_kv[0, b, kv_head, key, d].float() * scale for d in dims) for key,scale in zip(keys, kscales))
  vvals = tuple(tuple(cache_kv[1, b, kv_head, key, d].float() * scale for d in dims) for key,scale in zip(keys, vscales))
  updates = []
  for head in range(heads_per_wave):
    q_head = kv_head*G + head_group*head_tile + wave*heads_per_wave + head
    scores = tuple(wave_reduce_sum(sum((q[b, q_head, 0, d].float()*k for d,k in zip(dims, key_kvals)),
                                       UOp.const(0, dtypes.float)), lane + wave*WARP_SIZE) / math.sqrt(D) for key_kvals in kvals)
    prev_acc, prev_max, prev_sum = acc.after(offset)[head], row_max.after(offset)[head], row_sum.after(offset)[head]
    new_max = prev_max
    for is_valid, score in zip(valid, scores): new_max = new_max.maximum(is_valid.where(score, UOp.const(-math.inf, dtypes.float)))
    alpha = ((prev_max-new_max)*LOG2E).exp2()
    betas = tuple(is_valid.where(((score-new_max)*LOG2E).exp2(), UOp.const(0, dtypes.float)) for is_valid,score in zip(valid, scores))
    updates += [acc[head].store(prev_acc*alpha + sum((UOp.stack(*value)*beta for value,beta in zip(vvals, betas)), acc[head].const_like(0))),
                row_sum[head].store(prev_sum*alpha + sum(betas, UOp.const(0, dtypes.float))), row_max[head].store(new_max)]
  update = UOp.group(*updates).end(offset)
  acc, row_max, row_sum = acc.after(update), row_max.after(update), row_sum.after(update)

  stores = []
  for head in range(heads_per_wave):
    q_head = kv_head*G + head_group*head_tile + wave*heads_per_wave + head
    stores += [out[b, q_head, block_n, d].store(acc[head, i]) for i,d in enumerate(dims)]
    stores += [stats[b, q_head.valid(lane.eq(0)), block_n, 0].store(row_max[head]),
               stats[b, q_head.valid(lane.eq(0)), block_n, 1].store(row_sum[head])]
  return UOp.group(*stores).end(lane, wave, block_n, block_bhkv).sink(arg=KernelInfo(name="flash_decode_partial", opts_to_apply=()))

@functools.cache
def _amd_flash_attention_decode_reduce(out:UOp, partial:UOp, stats:UOp, valid_chunks:int|UOp) -> UOp:
  B, H, _, D = cast(tuple[int, int, int, int], out.shape)
  assert D % WARP_SIZE == 0
  DV = D // WARP_SIZE
  block_bh, lane = UOp.range(B*H, 0, AxisType.GLOBAL), UOp.range(WARP_SIZE, 1, AxisType.LOCAL)
  b, head = block_bh // H, block_bh % H
  dims = tuple(lane + i*WARP_SIZE for i in range(DV))

  row_max = UOp.placeholder((1,), dtypes.float, slot=0, addrspace=AddrSpace.REG)
  row_max = row_max.after(row_max.store(row_max.const_like(-math.inf)))
  chunk_max = UOp.range(valid_chunks, 100, AxisType.REDUCE)
  max_done = row_max.store(row_max.after(chunk_max).maximum(stats[b, head, chunk_max, 0])).end(chunk_max)
  row_max = row_max.after(max_done)

  numerator = UOp.placeholder((DV,), dtypes.float, slot=1, addrspace=AddrSpace.REG)
  denominator = UOp.placeholder((1,), dtypes.float, slot=2, addrspace=AddrSpace.REG)
  init = UOp.group(numerator.store(numerator.const_like(0)), denominator.store(denominator.const_like(0)))
  numerator, denominator = numerator.after(init), denominator.after(init)
  chunk = UOp.range(valid_chunks, 101, AxisType.REDUCE)
  scale = ((stats[b, head, chunk, 0]-row_max[0])*LOG2E).exp2()
  update = UOp.group(numerator.store(numerator.after(chunk) + UOp.stack(*(partial[b, head, chunk, d] for d in dims))*scale),
                     denominator.store(denominator.after(chunk) + stats[b, head, chunk, 1]*scale)).end(chunk)
  numerator, denominator = numerator.after(update), denominator.after(update)
  stores = [out[b, head, 0, d].store(numerator[i]/denominator[0]) for i,d in enumerate(dims)]
  return UOp.group(*stores).end(lane, block_bh).sink(arg=KernelInfo(name="flash_decode_reduce", opts_to_apply=()))

def amd_flash_attention_decode(q:Tensor, cache_kv:Tensor, valid_kv_len:int|UOp, max_kv_len:int|None=None,
                               cache_scale:Tensor|None=None) -> Tensor:
  _, B, H_KV, N, D = cast(tuple[int, int, int, int, int], cache_kv.shape)
  _, H, M, _ = cast(tuple[int, int, int, int], q.shape)
  max_kv_len = N if max_kv_len is None else max_kv_len
  block_n = 128
  assert M == 1 and max_kv_len <= N and max_kv_len % block_n == 0
  chunks = min(64, max_kv_len // block_n)
  partial = Tensor.empty(B, H, chunks, D, dtype="float32", device=q.device)
  stats = Tensor.empty(B, H, chunks, 2, dtype="float32", device=q.device)
  srcs = (partial, stats, q, cache_kv) if cache_scale is None else (partial, stats, q, cache_kv, cache_scale)
  def decode_partial(*uops:UOp) -> UOp:
    out, stat, query, cache, *scale = uops
    return _amd_flash_attention_decode_partial(out, stat, query, cache, scale[0] if scale else None,
      valid_kv_len=valid_kv_len, max_kv_len=max_kv_len, block_n=block_n)
  partial, stats = Tensor.custom_kernel(*srcs, fxn=decode_partial)[:2]
  live_chunks = (valid_kv_len+block_n-1)//block_n
  live_chunks = min(live_chunks, chunks) if isinstance(live_chunks, int) else live_chunks.minimum(chunks)
  out = Tensor.empty(B, H, 1, D, dtype="float32", device=q.device)
  return Tensor.custom_kernel(out, partial, stats,
    fxn=functools.partial(_amd_flash_attention_decode_reduce, valid_chunks=live_chunks))[0]

@functools.cache
def _amd_flash_attention(o:UOp, q:UOp, k:UOp, v:UOp, kv_scale:UOp, valid_kv_len:int|UOp,
                         key_limit:int|UOp|None=None) -> UOp:
  # inputs are q=(B*H, M, D), k/v=(B*H, N, D). For causal attention q is the final M tokens of k/v.
  BH, M, D = q.shape
  physical_n = k.shape[1]
  N = valid_kv_len
  assert k.shape == v.shape and BH % k.shape[0] == 0 and k.shape[2] == D
  gqa_group = BH // k.shape[0]
  if isinstance(M, int) and isinstance(N, int):
    assert M % BLOCK_M == 0 and N % BLOCK_N == 0, \
      f"M={M} and N={N} must be divisible by BLOCK_M={BLOCK_M} and BLOCK_N={BLOCK_N}"
  assert isinstance(D, int) and D % WMMA_K == 0 and D % LANES_PER_WAVE_N == 0, \
    f"D={D} must be divisible by WMMA_K={WMMA_K} and LANES_PER_WAVE_N={LANES_PER_WAVE_N}"
  assert BLOCK_M % (WAVES_M * WMMA_M) == 0 and BLOCK_N % LANES_PER_WAVE_N == 0
  TM = BLOCK_M // (WAVES_M * LANES_PER_WAVE_M)
  # Each N wave computes the same score tile, then owns a disjoint slice of D for P@V.
  TN = BLOCK_N // LANES_PER_WAVE_N
  TD = D // (WAVES_N * LANES_PER_WAVE_N)
  SCALE = 1.0 / math.sqrt(D)

  block_bh = UOp.range(BH, 0, AxisType.GLOBAL)
  block_m = UOp.range(M // BLOCK_M, 1, AxisType.GLOBAL)

  q = q.reshape(BH, M//BLOCK_M, BLOCK_M, D)[block_bh, block_m]
  kv_head = block_bh // gqa_group
  k, v = k[kv_head], v[kv_head]
  o = o.reshape(BH, M//BLOCK_M, BLOCK_M, D)[block_bh, block_m]

  wave_m = UOp.range(WAVES_M, 2, AxisType.LOCAL)
  wave_n = UOp.range(WAVES_N, 3, AxisType.LOCAL)
  lane = UOp.range(WARP_SIZE, -1, AxisType.WARP)
  tid = (wave_m * WAVES_N + wave_n) * WARP_SIZE + lane
  lane_m = lane // LANES_PER_WAVE_N
  lane_n = lane % LANES_PER_WAVE_N

  # LDS allocation: slot 0 = Q then P (shared), slot 1 = K then V
  # TODO: the memory planner should be able to find this reuse
  Q_ELEMS_PER_THREAD = BLOCK_M * D // THREADS_PER_BLOCK
  KV_ELEMS_PER_THREAD = BLOCK_N * D // THREADS_PER_BLOCK
  QP_lds = UOp.placeholder((BLOCK_M, D + LDS_PAD), dtypes.half, slot=0, addrspace=AddrSpace.LOCAL)
  KV_lds = UOp.placeholder((BLOCK_N, D + LDS_PAD), dtypes.half, slot=1, addrspace=AddrSpace.LOCAL)[:, :D]

  # register state
  acc = UOp.placeholder((TM, TD), dtypes.float, slot=2, addrspace=AddrSpace.REG)
  m_i = UOp.placeholder((TM,), dtypes.float, slot=3, addrspace=AddrSpace.REG)
  l_i = UOp.placeholder((TM,), dtypes.float, slot=4, addrspace=AddrSpace.REG)
  acc = acc.after(acc.store(acc.const_like(0)))
  m_i = m_i.after(m_i.store(m_i.const_like(-math.inf)))
  l_i = l_i.after(l_i.store(l_i.const_like(0)))

  # ====== KV tile loop ======
  # Causal blocks never need KV tiles strictly to their right. Besides saving work, this avoids an all
  # -inf tile, whose online-softmax update would otherwise contain -inf - -inf.
  n_tiles = (N - M + (block_m + 1) * BLOCK_M + BLOCK_N - 1) // BLOCK_N
  n_tile = UOp.range(n_tiles, 100, AxisType.REDUCE)

  # load Q + K into LDS (Q reloaded each iteration since P overwrites slot 0)
  Q_lds = QP_lds[:, :D]
  Q_store = Q_lds.after(n_tile).reshape(THREADS_PER_BLOCK, Q_ELEMS_PER_THREAD)[tid].store(
    q.reshape(THREADS_PER_BLOCK, Q_ELEMS_PER_THREAD)[tid])
  load_k = UOp.range(KV_ELEMS_PER_THREAD, 90, AxisType.WEAK)
  kidx = n_tile*BLOCK_N*D + tid*KV_ELEMS_PER_THREAD + load_k
  kval = k.reshape(physical_n*D)[kidx]
  kval = kval.float() * kv_scale[0, kv_head, kidx // D].float()
  K_store = KV_lds.reshape(THREADS_PER_BLOCK, KV_ELEMS_PER_THREAD)[tid, load_k].store(kval).end(load_k)
  qk_load_barrier = UOp.barrier(UOp.group(Q_store, K_store))
  Q_lds = Q_lds.after(qk_load_barrier)
  KV_lds_k = KV_lds.after(qk_load_barrier)

  # -- S = Q @ K^T via WMMA (re-init each n_tile) --
  S_reg = UOp.placeholder((TM, TN), dtypes.float, slot=6, addrspace=AddrSpace.REG)
  S_reg = S_reg.after(S_reg.after(n_tile).store(S_reg.const_like(0)))
  k_qk = UOp.range(D // WMMA_K, 101, AxisType.REDUCE)
  tm1 = UOp.range(TM // WMMA_ACC, 200)
  tn1 = UOp.range(TN, 201)
  S_frag = S_reg.reshape(TM // WMMA_ACC, WMMA_ACC, TN).permute(0, 2, 1)[tm1, tn1]
  q_frag = Q_lds.reshape(WAVES_M, TM // WMMA_ACC, WMMA_M, D // WMMA_K, WMMA_K)[wave_m, tm1, lane_n, k_qk]
  k_frag = KV_lds_k.reshape(TN, WMMA_N, D // WMMA_K, WMMA_K)[tn1, lane_n, k_qk]
  qk = UOp.wmma(q_frag, k_frag, S_frag.after(k_qk), *WMMA_ARG)
  qk_done = S_frag.store(qk).end(tm1, tn1).end(k_qk)
  S_reg = S_reg.after(qk_done)

  # -- softmax in registers with warp shuffles --
  S_reg = S_reg.after(S_reg.store(S_reg * SCALE))

  # WMMA accumulator ownership: each lane owns an 8x4 fragment of the 64x64 score tile.
  # q is aligned to the right of k, matching PyTorch's causal_lower_right mask.
  rm, rn = UOp.range(TM, 250, AxisType.WEAK), UOp.range(TN, 251, AxisType.WEAK)
  q_idx = N - M + block_m * BLOCK_M + wave_m * WMMA_M + rm * LANES_PER_WAVE_M + lane_m
  k_idx = n_tile * BLOCK_N + rn * LANES_PER_WAVE_N + lane_n
  valid = k_idx <= q_idx
  if key_limit is not None: valid = valid & (k_idx < key_limit)
  S_reg = S_reg.after(S_reg[rm, rn].store(valid.where(S_reg[rm, rn], S_reg[rm, rn].const_like(-math.inf))).end(rm, rn))

  # per-thread local row max over TN=4 elements, then warp reduce across 16 lanes
  m_ij = UOp.placeholder((TM,), dtypes.float, slot=7, addrspace=AddrSpace.REG)
  m_ij = m_ij.after(m_ij.after(n_tile).store(m_ij.const_like(-math.inf)))
  rm2 = UOp.range(TN, 261, AxisType.REDUCE)
  m_ij = m_ij.after(m_ij.store(m_ij.after(rm2).maximum(S_reg[:, rm2])).end(rm2))
  # warp reduce max (in-place)
  ri_w = UOp.range(TM, 270)
  m_ij = m_ij.after(m_ij[ri_w].store(warp_reduce_max(m_ij[ri_w], lane)).end(ri_w))

  # compute P = exp(S - m_ij) in S_reg
  S_reg = S_reg.after(S_reg.store(((S_reg - m_ij.reshape(TM, 1).expand(TM, TN)) * LOG2E).exp2()))

  p_local = UOp.placeholder((TM,), dtypes.float, slot=8, addrspace=AddrSpace.REG)
  p_local = p_local.after(p_local.after(n_tile).store(p_local.const_like(0)))
  ri_ws = UOp.range(TM, 295, AxisType.WEAK)
  # Reduce contiguous 16-key groups independently, matching the ordinary softmax reduction tree.
  p_sum = p_local.after(p_local[ri_ws].store(
    sum((warp_reduce_sum(S_reg[ri_ws, rn], lane) for rn in range(TN)), S_reg.const_like(0))).end(ri_ws))

  # Store softmax weights in half for the WMMA P@V product; accumulation remains float.
  P_lds = QP_lds.flatten()[:WAVES_N * BLOCK_M * BLOCK_N].reshape(WAVES_N, BLOCK_M, BLOCK_N)
  P_write = P_lds.reshape(WAVES_N, WAVES_M, TM, LANES_PER_WAVE_M, 1, TN, LANES_PER_WAVE_N, 1)
  P_write = P_write.permute((1, 0, 3, 6, 2, 4, 5, 7)).reshape(THREADS_PER_BLOCK, TM, TN)
  P_store = P_write[tid].store(S_reg.cast(dtypes.half))

  # -- online softmax correction --
  beta_i = UOp.placeholder((TM,), dtypes.float, slot=9, addrspace=AddrSpace.REG)
  ri4 = UOp.range(TM, 330, AxisType.WEAK)
  m_new_val = m_i[ri4].maximum(m_ij[ri4])
  alpha_val = ((m_i[ri4] - m_new_val) * LOG2E).exp2()
  beta_val = ((m_ij[ri4] - m_new_val) * LOG2E).exp2()
  rj4 = UOp.range(TD, 331)
  correction = UOp.group(
    acc[ri4, rj4].store(alpha_val * acc[ri4, rj4]).end(rj4),
    l_i[ri4].store(alpha_val * l_i[ri4] + beta_val * p_sum[ri4]),
    m_i[ri4].store(m_new_val),
    beta_i[ri4].store(beta_val),
  ).end(ri4)
  acc = acc.after(correction)
  l_i = l_i.after(correction)
  m_i = m_i.after(correction)
  beta_i = beta_i.after(correction)

  # Load V transposed into LDS: PV's B operand is logically (D, BLOCK_N), while global V is (BLOCK_N, D).
  # It reuses K's slot and must wait for QK WMMA to finish reading that slot.
  V_lds = UOp.placeholder((D, BLOCK_N + LDS_PAD), dtypes.half, slot=1, addrspace=AddrSpace.LOCAL)[:, :BLOCK_N]
  V_copy = V_lds.after(qk_done).permute(1, 0)
  load_v = UOp.range(KV_ELEMS_PER_THREAD, 390, AxisType.WEAK)
  vidx = n_tile*BLOCK_N*D + tid*KV_ELEMS_PER_THREAD + load_v
  vval = v.reshape(physical_n*D)[vidx]
  vval = vval.float() * kv_scale[1, kv_head, vidx // D].float()
  V_store = V_copy.reshape(THREADS_PER_BLOCK, KV_ELEMS_PER_THREAD)[tid, load_v].store(vval).end(load_v)
  pv_barrier = UOp.barrier(UOp.group(P_store, V_store))
  P_lds = P_lds.after(pv_barrier)
  V_lds = V_lds.after(pv_barrier)

  # -- acc += beta * (P @ V) via WMMA --
  pv_acc = UOp.placeholder((TM, TD), dtypes.float, slot=10, addrspace=AddrSpace.REG)
  pv_acc = pv_acc.after(pv_acc.after(n_tile).store(pv_acc.const_like(0))).after(pv_barrier)
  k_pv = UOp.range(BLOCK_N // WMMA_K, 400, AxisType.REDUCE)
  tm2 = UOp.range(TM // WMMA_ACC, 401, AxisType.WEAK)
  tn2 = UOp.range(TD, 402, AxisType.WEAK)
  pv_frag = pv_acc.reshape(TM // WMMA_ACC, WMMA_ACC, TD).permute(0, 2, 1)[tm2, tn2]
  p_frag = P_lds[wave_n].reshape(WAVES_M, TM // WMMA_ACC, WMMA_M, BLOCK_N // WMMA_K, WMMA_K)[wave_m, tm2, lane_n, k_pv]
  v_frag = V_lds.reshape(WAVES_N, TD, WMMA_N, BLOCK_N // WMMA_K, WMMA_K)[wave_n, tn2, lane_n, k_pv]
  pv = UOp.wmma(p_frag, v_frag, pv_frag.after(k_pv), *WMMA_ARG)
  pv_done = pv_frag.store(pv).end(tm2, tn2).end(k_pv)
  pv_acc = pv_acc.after(pv_done)

  ri5 = UOp.range(TM, 410, AxisType.WEAK)
  rj5 = UOp.range(TD, 411, AxisType.WEAK)
  accumulate = acc[ri5, rj5].store(acc[ri5, rj5] + beta_i[ri5] * pv_acc[ri5, rj5]).end(ri5, rj5)

  # end KV tile loop
  n_tile_end = accumulate.barrier().end(n_tile)
  acc = acc.after(n_tile_end)
  l_i = l_i.after(n_tile_end)
  m_i = m_i.after(n_tile_end)

  # normalize: acc /= l_i
  acc = acc.after(acc.store(acc * (1 / l_i).reshape(TM, 1).expand(TM, TD)))

  # store output
  o = o.reshape(WAVES_M, TM, LANES_PER_WAVE_M, 1, WAVES_N, TD, LANES_PER_WAVE_N, 1)
  o = o.permute((0, 4, 2, 6, 1, 3, 5, 7)).reshape(THREADS_PER_BLOCK, TM, TD)
  return o[tid].store(acc).end(wave_m, wave_n, lane).end(block_m, block_bh).sink(arg=KernelInfo(opts_to_apply=()))

def flash_attention_causal_cached(q:Tensor, cache_kv:Tensor, valid_kv_len:int|UOp, key_limit:int|UOp|None=None,
                                  cache_scale:Tensor|None=None) -> Tensor:
  assert cache_scale is not None
  B, H, T, D = cast(tuple[int, int, int, int], q.shape)
  q_flat = q.reshape(B*H, T, D)
  out = Tensor.empty(B*H, T, D, dtype="float32", device=q.device)
  srcs = (out, q_flat, cache_kv, cache_scale)
  def flash_cached(*uops:UOp) -> UOp:
    output, query, cache, scale = uops
    _, b, h_kv, n, d = cast(tuple[int, int, int, int, int], cache.shape)
    return _amd_flash_attention(output, query, cache[0].reshape(b*h_kv, n, d), cache[1].reshape(b*h_kv, n, d),
                                scale.reshape(2, b*h_kv, n), valid_kv_len, key_limit)
  return Tensor.custom_kernel(*srcs, fxn=flash_cached)[0].reshape(B, H, T, D)

def _amd_dp4a(a:UOp, b:UOp, c:UOp) -> UOp:
  return UOp(Ops.CUSTOMI, dtypes.int32, (a.int(), b.int(), c),
             arg="__builtin_amdgcn_sudot4(true, {}, true, {}, {}, false)")

def _amd_byte_perm(a:UOp, b:UOp, selectors:UOp) -> UOp:
  return UOp(Ops.CUSTOMI, dtypes.uint32, (a.cast(dtypes.uint32), b.cast(dtypes.uint32), selectors.cast(dtypes.uint32)),
             arg="__builtin_amdgcn_perm({}, {}, {})")

def _amd_vector_load(ptr:UOp, lanes:int) -> UOp:
  assert ptr.op is Ops.INDEX
  buf, coords = ptr.src[0], ptr.src[1:]
  index = sum((coord*math.prod(buf.shape[i+1:]) for i,coord in enumerate(coords)), UOp.const(0, dtypes.weakint))
  return UOp(Ops.SHRINK, src=(buf.flatten(), index, UOp.const(lanes, dtypes.weakint))).load(dtype=ptr.dtype)

def _amd_stream_load(ptr:UOp) -> UOp:
  assert ptr.op is Ops.INDEX
  return UOp(Ops.CUSTOMI, ptr.dtype, (ptr,), arg="__builtin_nontemporal_load({0})")

def _iq4_bytes(packed:UOp, shift:int) -> UOp:
  selectors = (packed >> shift) & 0x0f0f0f0f
  low = _amd_byte_perm(UOp.const(0xf6eaddcf, dtypes.uint32), UOp.const(0xbfad9881, dtypes.uint32), selectors)
  high = _amd_byte_perm(UOp.const(0x71594535, dtypes.uint32), UOp.const(0x26190d01, dtypes.uint32), selectors & 0x07070707)
  return _amd_byte_perm(high, low, 0x03020100 | ((selectors & 0x08080808) >> 1))

def _amd_wave_sum(value:UOp, lane:UOp, lane_count:int, wave:UOp|None=None) -> UOp:
  assert lane_count in (8, 16, 32)
  for offset in (16, 8, 4, 2, 1)[{32:0, 16:1, 8:2}[lane_count]:]:
    source_lane = (lane ^ offset) + (wave * lane_count if wave is not None else 0)
    value = value + UOp(Ops.CUSTOM, dtypes.float32, ((source_lane * 4).int(), value),
      arg="__builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute({0}, __builtin_bit_cast(int, {1})))")
  return value

def _amd_wave_max(value:UOp, lane:UOp) -> UOp:
  for offset in (16, 8, 4, 2, 1):
    value = value.maximum(UOp(Ops.CUSTOM, dtypes.float32, (((lane ^ offset) * 4).int(), value),
      arg="__builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute({0}, __builtin_bit_cast(int, {1})))"))
  return value

def _q8_pack(raw_value:UOp, lane:UOp) -> tuple[UOp, UOp]:
  d = (_amd_wave_max(raw_value.abs(), lane) / 127).maximum(1e-8)
  quantized = (raw_value / d).round().maximum(-127).minimum(127).cast(dtypes.int8).bitcast(dtypes.uint8).cast(dtypes.uint32)
  word = UOp.const(0, dtypes.uint32)
  for byte_idx in range(4):
    byte = UOp(Ops.CUSTOM, dtypes.uint32, (((lane * 4 + byte_idx) * 4).int(), quantized),
               arg="__builtin_amdgcn_ds_bpermute({0}, {1})")
    word = word | (byte << (8 * byte_idx))
  return d, word

@functools.cache
def _q8_kernel(quant:UOp, scale:UOp, x:UOp, in_features:int, group_sum:UOp|None=None) -> UOp:
  x = x.flatten()
  token, group = UOp.range(quant.shape[0], 0), UOp.range(in_features // 32, 1)
  lane = UOp.range(32, 2, axis_type=AxisType.LOCAL)
  raw_value = x[token * in_features + group * 32 + lane].load().float()
  d, word = _q8_pack(raw_value, lane)
  stores = [scale[token.valid(lane.eq(0)), group].store(d), quant[token, group, lane.valid(lane < 8)].store(word)]
  if group_sum is not None:
    qsum = _amd_dp4a(UOp.const(0x01010101, dtypes.uint32), word, UOp.const(0, dtypes.int32))
    for offset in (4, 2, 1):
      qsum = qsum + UOp(Ops.CUSTOM, dtypes.int32, (((lane ^ offset) * 4).int(), qsum),
                       arg="__builtin_amdgcn_ds_bpermute({0}, {1})")
    stores.append(group_sum[token.valid(lane.eq(0)), group].store(qsum))
  return UOp.group(*stores).end(token, group, lane).sink(arg=KernelInfo(name="q8_quantize", opts_to_apply=()))

def q8_quantize(x:Tensor, tokens:int, in_features:int) -> tuple[Tensor, Tensor]:
  quant = Tensor.empty(tokens, in_features // 32, 8, dtype=dtypes.uint32, device=x.device)
  scale = Tensor.empty(tokens, in_features // 32, dtype=dtypes.float32, device=x.device)
  return tuple(Tensor.custom_kernel(quant, scale, x,
    fxn=lambda quant,scale,x:_q8_kernel(quant, scale, x, in_features))[:2])  # type: ignore[return-value]

def q8_quantize_sum(x:Tensor, tokens:int, in_features:int) -> tuple[Tensor, Tensor, Tensor]:
  quant = Tensor.empty(tokens, in_features // 32, 8, dtype=dtypes.uint32, device=x.device)
  scale = Tensor.empty(tokens, in_features // 32, dtype=dtypes.float32, device=x.device)
  group_sum = Tensor.empty(tokens, in_features // 32, dtype=dtypes.int32, device=x.device)
  return tuple(Tensor.custom_kernel(quant, scale, group_sum, x,
    fxn=lambda quant,scale,group_sum,x:_q8_kernel(quant, scale, x, in_features, group_sum))[:3])  # type: ignore[return-value]

@functools.cache
def _gated_delta_prefill_kernel(core:UOp, next_state:UOp, q:UOp, k:UOp, v:UOp, beta:UOp, alpha:UOp, state:UOp,
                                kq:UOp) -> UOp:
  batch, heads, tokens, dim = core.shape
  row_tile = 4
  assert all(isinstance(x, int) for x in (batch, heads, tokens, dim)) and dim % 32 == 0 and dim % row_tile == 0
  batch, heads, tokens, dim = cast(tuple[int, int, int, int], (batch, heads, tokens, dim))
  core, q, k, v = (x.reshape(batch*heads, tokens, dim) for x in (core, q, k, v))
  beta, alpha, kq = (x.reshape(batch*heads, tokens) for x in (beta, alpha, kq))
  state, next_state = (x.reshape(batch*heads, dim, dim) for x in (state, next_state))
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
  updates, stores = [], []
  for row_idx,row in enumerate(rows):
    previous = tuple(current.after(token)[row_idx*dim//32+i].load() for i in range(dim//32))
    state_k = _amd_wave_sum(sum((x*y for x,y in zip(previous, keys)), UOp.const(0, dtypes.float32)), lane, 32)
    state_q = _amd_wave_sum(sum((x*y for x,y in zip(previous, queries)), UOp.const(0, dtypes.float32)), lane, 32)
    delta = (v[bh, token, row].load() - state_k*av) * bv
    updates += [x*av + delta*y for x,y in zip(previous, keys)]
    stores.append(core[bh, token, row.valid(lane.eq(0))].store(state_q*av + delta*kq[bh, token]))
  step = UOp.group(*stores, current.store(UOp.stack(*updates))).end(token)
  return UOp.group(*(next_state[bh, row, col].store(current.after(step)[row_idx*dim//32+i].load().cast(next_state.dtype))
                     for row_idx,row in enumerate(rows) for i,col in enumerate(cols))).end(lane, bh_row).sink(
    arg=KernelInfo(name="gated_delta_prefill", opts_to_apply=()))

def gated_delta_prefill(q:Tensor, k:Tensor, v:Tensor, beta:Tensor, alpha:Tensor, state:Tensor) -> tuple[Tensor, Tensor]:
  batch, heads, tokens, dim = q.shape
  assert q.shape == k.shape == v.shape and beta.shape == alpha.shape == (batch, heads, tokens) and \
         state.shape == (batch, heads, dim, dim)
  core, next_state = Tensor.empty_like(q), Tensor.empty_like(state)
  kq = (q*k).sum(-1).contiguous()
  srcs = (core.uop, next_state.uop, q.contiguous().uop, k.contiguous().uop, v.contiguous().uop,
          beta.contiguous().uop, alpha.contiguous().uop, state.uop, kq.uop)
  params = [UOp.placeholder_like(src, slot=i) for i,src in enumerate(srcs)]
  out = _gated_delta_prefill_kernel(*params).call(*srcs)
  return Tensor(srcs[0].after(out)), Tensor(srcs[1].after(out))

@functools.cache
def _q4_embedding_kernel(out:UOp, raw:UOp, idx:UOp, raw_offset:UOp, embed_size:int) -> UOp:
  token_count = math.prod(idx.shape)
  out, idx, raw_offset = out.reshape(token_count, embed_size), idx.flatten(), raw_offset.cast(dtypes.uint64)
  token, group = UOp.range(token_count, 0), UOp.range(embed_size // 32, 1)
  lane = UOp.range(32, 2, axis_type=AxisType.LOCAL)
  block, subgroup = group // 8, group % 8
  type_words, row_words = _GGML_QUANT[12][1] // 4, embed_size // 256 * (_GGML_QUANT[12][1] // 4)
  base = raw_offset + idx[token].cast(dtypes.uint64) * row_words + block * type_words
  def load_byte(byte_offset:UOp) -> UOp:
    return (raw[base + byte_offset // 4] >> ((byte_offset & 3) * 8).cast(dtypes.uint32)) & 255
  scale = (subgroup < 4).where(load_byte(4 + subgroup) & 63,
    (load_byte(8 + subgroup) & 15) | ((load_byte(subgroup) >> 6) << 4)).float()
  minimum = (subgroup < 4).where(load_byte(8 + subgroup) & 63,
    (load_byte(8 + subgroup) >> 4) | ((load_byte(4 + subgroup) >> 6) << 4)).float()
  packed = raw[base + 4 + (subgroup // 2) * 8 + lane // 4]
  q = ((packed >> ((subgroup & 1) * 4).cast(dtypes.uint32)) >> ((lane % 4) * 8).cast(dtypes.uint32)) & 15
  scales = raw[base]
  d = (scales & 0xffff).cast(dtypes.uint16).bitcast(dtypes.float16).float()
  dmin = (scales >> 16).cast(dtypes.uint16).bitcast(dtypes.float16).float()
  value = (q.float() * d * scale - dmin * minimum).cast(dtypes.float16)
  return out[token, group*32+lane].store(value).end(token, group, lane).sink(
    arg=KernelInfo(name="embedding_q4_k", opts_to_apply=()))

def q4_embedding(layer:Embedding, idx:Tensor) -> Tensor:
  if layer._raw_uop is None: layer._prepare_packed()
  assert layer._raw_uop is not None and layer._raw_offset_uop is not None
  out = Tensor.empty(*idx.shape, layer.embed_size, dtype=dtypes.float16, device=idx.device)
  raw = layer._raw_uop.bitcast(dtypes.uint32)
  srcs = (out.uop, raw, idx.contiguous().uop, layer._raw_offset_uop)
  params = [UOp.placeholder_like(src, slot=i) for i,src in enumerate(srcs)]
  kernel = _q4_embedding_kernel(params[0], params[1], params[2], params[3], layer.embed_size).call(*srcs)
  return Tensor(srcs[0].after(kernel))

@functools.cache
def _qk_linear_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, xsum:UOp, out_features:int, in_features:int,
                      raw_offset:int|UOp=0) -> UOp:
  if isinstance(raw_offset, UOp): raw_offset = raw_offset.cast(dtypes.uint64)
  def load_byte(base:UOp, byte_offset:UOp) -> UOp:
    return (raw[base + byte_offset // 4] >> ((byte_offset & 3) * 8).cast(dtypes.uint32)) & 255
  output_tile = 1
  output_block, lane = UOp.range(out_features // output_tile, 0), UOp.range(32, 1, axis_type=AxisType.LOCAL)
  outputs, group_count = tuple(output_block * output_tile + i for i in range(output_tile)), in_features // 32
  type_words, output_words = _GGML_QUANT[13][1] // 4, in_features // 256 * _GGML_QUANT[13][1] // 4

  def group_dot(group:UOp, output:UOp) -> UOp:
    block, subgroup = group // 8, group % 8
    base = raw_offset + output * output_words + block * type_words
    qs_base = base + 12 + (subgroup // 2) * 8
    xwords = _amd_vector_load(xq[0, group, 0], 8)
    dot = UOp.const(0, dtypes.int32)
    for word_idx in range(8):
      word = (raw[qs_base + word_idx] >> ((subgroup & 1) * 4).cast(dtypes.uint32)) & 0x0f0f0f0f
      word = word | (((raw[base + 4 + word_idx] >> subgroup.cast(dtypes.uint32)) & 0x01010101) << 4)
      dot = _amd_dp4a(word, xwords[word_idx], dot)
    scale = (subgroup < 4).where(load_byte(base, 4 + subgroup) & 63,
      (load_byte(base, 8 + subgroup) & 15) | ((load_byte(base, subgroup) >> 6) << 4))
    minimum = (subgroup < 4).where(load_byte(base, 8 + subgroup) & 63,
      (load_byte(base, 8 + subgroup) >> 4) | ((load_byte(base, 4 + subgroup) >> 6) << 4))
    scales = raw[base]
    dbits, dminbits = (scales & 0xffff).cast(dtypes.uint16), (scales >> 16).cast(dtypes.uint16)
    return (dot.float() * dbits.bitcast(dtypes.float16).float() * scale.float() -
            xsum[0, group].float() * dminbits.bitcast(dtypes.float16).float() * minimum.float()) * xd[0, group]

  accs = tuple(UOp.placeholder((1,), dtypes.float32, slot=i, addrspace=AddrSpace.REG) for i in range(output_tile))
  accs = tuple(acc.after(acc.store(acc.const_like(0))) for acc in accs)
  chunk = UOp.range((group_count + 31) // 32, 2, AxisType.REDUCE)
  update = UOp.group(*(acc.store(acc.after(chunk) + group_dot((lane + chunk*32).valid(lane + chunk*32 < group_count), output))
                       for acc,output in zip(accs, outputs))).end(chunk)
  totals = [_amd_wave_sum(acc.after(update)[0].load(), lane, 32) for acc in accs]
  stores = [out[0, output.valid(lane.eq(0))].store(total.cast(out.dtype)) for output,total in zip(outputs, totals)]
  return UOp.group(*stores).end(output_block, lane).sink(arg=KernelInfo(name="linear_q5_k", opts_to_apply=()))

@functools.cache
def _qk_linear_f16_wmma_kernel(out:UOp, raw:UOp, x:UOp, out_features:int, in_features:int,
                               raw_offset:UOp) -> UOp:
  x = x.reshape(out.shape[0], in_features)
  raw_offset = raw_offset.cast(dtypes.uint64)
  def load_byte(base:UOp, byte_offset:UOp) -> UOp:
    return (raw[base + byte_offset // 4] >> ((byte_offset & 3)*8).cast(dtypes.uint32)) & 255
  token_tile, output_tiles = (64, 1) if out_features <= 1024 and out.shape[0] % 64 == 0 else \
    (64, 2) if out.shape[0] % 64 == 0 else (32 if out.shape[0] % 32 == 0 else 16, 2)
  output_waves = 2 if out_features % (32*output_tiles) == 0 else 1
  token_block = UOp.range(out.shape[0] // token_tile, 0)
  output_block = UOp.range(out_features // (16*output_tiles*output_waves), 1)
  lane, wave = UOp.range(32, 2, axis_type=AxisType.LOCAL), UOp.range(output_waves, 3, axis_type=AxisType.LOCAL)
  hw_lane = UOp(Ops.CUSTOM, dtypes.int32, (lane.int(),), arg="__builtin_amdgcn_mbcnt_lo(-1, 0)").cast(dtypes.weakint)
  physical_col, physical_half = hw_lane % 16, hw_lane // 16
  outputs = tuple((output_block*output_waves+wave)*(16*output_tiles) + output_tile*16 + physical_col for output_tile in range(output_tiles))
  input_tokens = tuple(token_block*token_tile + tile*16 + physical_col for tile in range(token_tile // 16))
  tokens = tuple(tuple(token_block*token_tile + tile*16 + physical_half*8 + i for i in range(8))
                 for tile in range(token_tile // 16))
  group_count, type_words = in_features // 32, _GGML_QUANT[13][1] // 4
  output_words = in_features // 256 * type_words
  accs = tuple(tuple(UOp.placeholder((8,), dtypes.float32, slot=output_tile*(token_tile//16)+tile, addrspace=AddrSpace.REG)
                     for tile in range(token_tile // 16)) for output_tile in range(output_tiles))
  accs = tuple(tuple(acc.after(acc.store(acc.const_like(0))) for acc in output_accs) for output_accs in accs)
  group = UOp.range(group_count, 4, AxisType.REDUCE)
  block, subgroup = group // 8, group % 8
  wmma_accs = [list(output_accs) for output_accs in accs]
  for half in range(2):
    afrags = tuple(UOp.stack(*(x[input_token, group*32 + half*16 + i].cast(dtypes.float16) for i in range(16)))
                   for input_token in input_tokens)
    for output_tile,output in enumerate(outputs):
      base = raw_offset + output*output_words + block*type_words
      scale = (subgroup < 4).where(load_byte(base, 4 + subgroup) & 63,
        (load_byte(base, 8 + subgroup) & 15) | ((load_byte(base, subgroup) >> 6) << 4)).float()
      minimum = (subgroup < 4).where(load_byte(base, 8 + subgroup) & 63,
        (load_byte(base, 8 + subgroup) >> 4) | ((load_byte(base, 4 + subgroup) >> 6) << 4)).float()
      scales = raw[base]
      d = (scales & 0xffff).cast(dtypes.uint16).bitcast(dtypes.float16).float()
      dmin = (scales >> 16).cast(dtypes.uint16).bitcast(dtypes.float16).float()
      weight_scale, weight_min = d*scale, dmin*minimum
      qs_base = base + 12 + (subgroup // 2)*8 + half*4
      qwords = [((raw[qs_base + i] >> ((subgroup & 1)*4).cast(dtypes.uint32)) & 0x0f0f0f0f) for i in range(4)]
      qwords = [word | (((raw[base + 4 + half*4 + i] >> subgroup.cast(dtypes.uint32)) & 0x01010101) << 4)
                for i,word in enumerate(qwords)]
      bfrag = UOp.stack(*(((word >> (byte*8) & 255).float()*weight_scale-weight_min).cast(dtypes.float16)
                          for word in qwords for byte in range(4)))
      for tile,afrag in enumerate(afrags):
        previous = accs[output_tile][tile].after(group) if half == 0 else wmma_accs[output_tile][tile]
        wmma_accs[output_tile][tile] = UOp.wmma(afrag, bfrag, previous, (16, 16, 16), 'AMD', 32)
  update = UOp.group(*(acc.store(value) for output_accs,output_values in zip(accs, wmma_accs)
                       for acc,value in zip(output_accs, output_values))).end(group)
  logical_values = []
  for output_accs in accs:
    output_values = []
    for acc in output_accs:
      vals = tuple(acc.after(update)[i].load() for i in range(8))
      swapped = tuple(UOp(Ops.CUSTOM, dtypes.float32, (value,),
        arg="__builtin_bit_cast(float, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, {0}), 50688))") for value in vals)
      low = physical_half.eq(0)
      output_values.append((low.where(vals[0], swapped[4]), low.where(swapped[0], vals[4]),
                            low.where(vals[1], swapped[5]), low.where(swapped[1], vals[5]),
                            low.where(vals[2], swapped[6]), low.where(swapped[2], vals[6]),
                            low.where(vals[3], swapped[7]), low.where(swapped[3], vals[7])))
    logical_values.append(output_values)
  stores = [out[token, output].store(value) for output,output_values in zip(outputs, logical_values)
            for tile_tokens,logical in zip(tokens, output_values) for token,value in zip(tile_tokens, logical)]
  return UOp.group(*stores).end(token_block, output_block, lane, wave).sink(
    arg=KernelInfo(name="linear_q5_k_f16_wmma", opts_to_apply=()))

@functools.cache
def _iq4_linear_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, out_features:int, in_features:int,
                       raw_offset:int|UOp=0) -> UOp:
  if isinstance(raw_offset, UOp): raw_offset = raw_offset.cast(dtypes.uint64)
  def load_byte(base:UOp, byte_offset:UOp) -> UOp:
    return (raw[base + byte_offset // 4] >> ((byte_offset & 3) * 8).cast(dtypes.uint32)) & 255
  output_tile = 1
  output_block, lane = UOp.range(out_features // output_tile, 0), UOp.range(32, 1, axis_type=AxisType.LOCAL)
  outputs, group_count = tuple(output_block * output_tile + i for i in range(output_tile)), in_features // 32
  type_words, output_words = _GGML_QUANT[23][1] // 4, in_features // 256 * _GGML_QUANT[23][1] // 4

  def group_dot(group:UOp, output:UOp) -> UOp:
    block, subgroup = group // 8, group % 8
    base = raw_offset + output * output_words + block * type_words
    xwords = _amd_vector_load(xq[0, group, 0], 8)
    dot = UOp.const(0, dtypes.int32)
    for word_idx in range(8):
      packed, shift = _amd_stream_load(raw[base + 2 + subgroup*4 + word_idx % 4]), 4 * (word_idx // 4)
      dot = _amd_dp4a(_iq4_bytes(packed, shift), xwords[word_idx], dot)
    low_byte = load_byte(base, 4 + subgroup // 2)
    scale = ((low_byte >> (4*(subgroup % 2)).cast(dtypes.uint32)) & 15) | \
            ((((raw[base] >> 16) >> (2*subgroup).cast(dtypes.uint32)) & 3) << 4)
    d = (raw[base] & 0xffff).cast(dtypes.uint16).bitcast(dtypes.float16).float()
    return dot.float() * xd[0, group] * d * (scale.cast(dtypes.uint8).bitcast(dtypes.int8)-32).float()

  accs = tuple(UOp.placeholder((1,), dtypes.float32, slot=i, addrspace=AddrSpace.REG) for i in range(output_tile))
  accs = tuple(acc.after(acc.store(acc.const_like(0))) for acc in accs)
  chunk = UOp.range((group_count + 31) // 32, 2, AxisType.REDUCE)
  update = UOp.group(*(acc.store(acc.after(chunk) + group_dot((lane + chunk*32).valid(lane + chunk*32 < group_count), output))
                       for acc,output in zip(accs, outputs))).end(chunk)
  totals = [_amd_wave_sum(acc.after(update)[0].load(), lane, 32) for acc in accs]
  stores = [out[0, output.valid(lane.eq(0))].store(total.cast(out.dtype)) for output,total in zip(outputs, totals)]
  return UOp.group(*stores).end(output_block, lane).sink(arg=KernelInfo(name="linear_iq4_xs", opts_to_apply=()))

@functools.cache
def _iq4_linear_f16_wmma_kernel(out:UOp, raw:UOp, x:UOp, lut:UOp, out_features:int, in_features:int,
                                raw_offset:UOp) -> UOp:
  x = x.reshape(out.shape[0], in_features)
  raw_offset = raw_offset.cast(dtypes.uint64)
  def load_byte(base:UOp, byte_offset:UOp) -> UOp:
    return (raw[base + byte_offset // 4] >> ((byte_offset & 3) * 8).cast(dtypes.uint32)) & 255

  def dequant(base:UOp, subgroup:UOp) -> tuple[tuple[UOp, ...], tuple[UOp, ...]]:
    low_byte = load_byte(base, 4 + subgroup // 2)
    scale_bits = ((low_byte >> (4*(subgroup % 2)).cast(dtypes.uint32)) & 15) | \
                 ((((raw[base] >> 16) >> (2*subgroup).cast(dtypes.uint32)) & 3) << 4)
    scale = (scale_bits.cast(dtypes.uint8).bitcast(dtypes.int8)-32).float() * \
            (raw[base] & 0xffff).cast(dtypes.uint16).bitcast(dtypes.float16).float()
    if out_features <= 6144:
      pairs = tuple(lut[((raw[base + 2 + subgroup*4 + word] >> (byte*8)) & 255).cast(dtypes.weakint)]
                    for word in range(4) for byte in range(4))
      return tuple(tuple((((pair >> (half*16)) & 0xffff).cast(dtypes.uint16).bitcast(dtypes.float16).float()*scale).cast(dtypes.float16)
                         for pair in pairs) for half in range(2))  # type: ignore[return-value]
    halves = []
    for half in range(2):
      values = []
      for packed in (raw[base + 2 + subgroup*4 + i] for i in range(4)):
        nibbles = tuple((packed >> (8*i + 4*half)) & 15 for i in range(4))
        pairs = tuple(lut[(nibbles[i] | (nibbles[i+1] << 4)).cast(dtypes.weakint)] for i in (0, 2))
        values += [(((pair >> (i*16)) & 0xffff).cast(dtypes.uint16).bitcast(dtypes.float16).float()*scale).cast(dtypes.float16)
                   for pair in pairs for i in range(2)]
      halves.append(tuple(values))
    return tuple(halves)  # type: ignore[return-value]

  token_tile = 32 if out_features <= 1024 and out.shape[0] % 32 == 0 else \
    128 if out_features == 5120 and in_features > 8192 and out.shape[0] % 128 == 0 else \
    64 if out_features <= 6144 and out.shape[0] % 64 == 0 else \
    128 if out.shape[0] % 128 == 0 else \
    64 if out.shape[0] % 64 == 0 else 32 if out.shape[0] % 32 == 0 else 16
  output_tiles = 1 if out_features <= 1024 else 2 if out_features <= 6144 else 1 if out_features < 8192 else 2
  output_waves = 2 if out_features % (32*output_tiles) == 0 else 1
  assert out_features % (16*output_tiles*output_waves) == 0
  token_block = UOp.range(out.shape[0] // token_tile, 0)
  output_block = UOp.range(out_features // (16*output_tiles*output_waves), 1)
  lane, wave = UOp.range(32, 2, axis_type=AxisType.LOCAL), UOp.range(output_waves, 3, axis_type=AxisType.LOCAL)
  hw_lane = UOp(Ops.CUSTOM, dtypes.int32, (lane.int(),), arg="__builtin_amdgcn_mbcnt_lo(-1, 0)").cast(dtypes.weakint)
  local_lut = UOp.placeholder((256,), dtypes.uint32, slot=32, addrspace=AddrSpace.LOCAL)
  tid = wave*32+lane
  lut_items = 256 // (32*output_waves)
  lut_ready = UOp.group(*(local_lut[tid*lut_items+i].store(lut[tid*lut_items+i]) for i in range(lut_items))).barrier()
  lut = local_lut.after(lut_ready)
  physical_col, physical_half = hw_lane % 16, hw_lane // 16
  outputs = tuple((output_block*output_waves+wave)*(16*output_tiles) + output_tile*16 + physical_col for output_tile in range(output_tiles))
  input_tokens = tuple(token_block*token_tile + tile*16 + physical_col for tile in range(token_tile // 16))
  tokens = tuple(tuple(token_block*token_tile + tile*16 + physical_half*8 + i for i in range(8)) for tile in range(token_tile // 16))
  type_words = _GGML_QUANT[23][1] // 4
  output_words = in_features // 256 * type_words
  accs = tuple(tuple(UOp.placeholder((8,), dtypes.float32, slot=output_tile*(token_tile//16)+tile, addrspace=AddrSpace.REG)
                     for tile in range(token_tile // 16)) for output_tile in range(output_tiles))
  accs = tuple(tuple(acc.after(acc.store(acc.const_like(0))) for acc in output_accs) for output_accs in accs)
  group = UOp.range(in_features // 32, 4, AxisType.REDUCE)
  block, subgroup = group // 8, group % 8
  weights = tuple(dequant(raw_offset + output*output_words + block*type_words, subgroup) for output in outputs)
  wmma_accs = [list(output_accs) for output_accs in accs]
  for half in range(2):
    afrags = tuple(UOp.stack(*(x[input_token, group*32 + half*16 + i].cast(dtypes.float16) for i in range(16)))
                   for input_token in input_tokens)
    for output_tile,weight in enumerate(weights):
      bfrag = UOp.stack(*weight[half])
      for tile,afrag in enumerate(afrags):
        previous = accs[output_tile][tile].after(group) if half == 0 else wmma_accs[output_tile][tile]
        wmma_accs[output_tile][tile] = UOp.wmma(afrag, bfrag, previous, (16, 16, 16), 'AMD', 32)
  update = UOp.group(*(acc.store(value) for output_accs,output_values in zip(accs, wmma_accs)
                       for acc,value in zip(output_accs, output_values))).end(group)

  def logical_values(acc:UOp) -> tuple[UOp, ...]:
    vals = tuple(acc.after(update)[i].load() for i in range(8))
    swapped = tuple(UOp(Ops.CUSTOM, dtypes.float32, (value,),
      arg="__builtin_bit_cast(float, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, {0}), 50688))") for value in vals)
    low = physical_half.eq(0)
    return (low.where(vals[0], swapped[4]), low.where(swapped[0], vals[4]),
            low.where(vals[1], swapped[5]), low.where(swapped[1], vals[5]),
            low.where(vals[2], swapped[6]), low.where(swapped[2], vals[6]),
            low.where(vals[3], swapped[7]), low.where(swapped[3], vals[7]))
  stores = [out[token, output].store(value) for output,output_accs in zip(outputs, accs)
            for tile_tokens,acc in zip(tokens, output_accs) for token,value in zip(tile_tokens, logical_values(acc))]
  return UOp.group(*stores).end(token_block, output_block, lane, wave).sink(
    arg=KernelInfo(name="linear_iq4_xs_f16_wmma", opts_to_apply=()))

@functools.cache
def _q6_linear_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, out_features:int, in_features:int, raw_offset:int|UOp=0) -> UOp:
  if isinstance(raw_offset, UOp): raw_offset = raw_offset.cast(dtypes.uint64)
  output_tile = 1
  output_block, lane = UOp.range(out_features // output_tile, 0), UOp.range(32, 1, axis_type=AxisType.LOCAL)
  outputs = tuple(output_block * output_tile + i for i in range(output_tile))
  group_count, type_size = in_features // 32, _GGML_QUANT[14][1]
  output_size = in_features // 256 * type_size

  accs = tuple(UOp.placeholder((1,), dtypes.float32, slot=i, addrspace=AddrSpace.REG) for i in range(output_tile))
  accs = tuple(acc.after(acc.store(acc.const_like(0))) for acc in accs)
  chunk = UOp.range((group_count + 31) // 32, 2, AxisType.REDUCE)
  group = (lane + chunk * 32).valid(lane + chunk * 32 < group_count)
  block, subgroup = group // 8, group % 8
  xwords = _amd_vector_load(xq[0, group, 0], 8)
  updates = []
  for acc,output in zip(accs, outputs):
    base = raw_offset + output * output_size + block * type_size
    dots = [UOp.const(0, dtypes.int32), UOp.const(0, dtypes.int32)]
    for word_idx in range(8):
      word = UOp.const(0, dtypes.uint32)
      for byte_idx in range(4):
        pos = subgroup * 32 + word_idx * 4 + byte_idx
        within = pos % 128
        low_byte = raw[base + (pos // 128) * 64 + within % 64]
        low = (low_byte >> ((within // 64) * 4).cast(dtypes.uint8)) & 15
        high_byte = raw[base + 128 + (pos // 128) * 32 + within % 32]
        high = (high_byte >> ((within // 32) * 2).cast(dtypes.uint8)) & 3
        q = (low | (high << 4)).cast(dtypes.uint8).bitcast(dtypes.int8) - 32
        word = word | (q.cast(dtypes.int8).bitcast(dtypes.uint8).cast(dtypes.uint32) << (8 * byte_idx))
      dots[word_idx // 4] = _amd_dp4a(word, xwords[word_idx], dots[word_idx // 4])
    scales = [raw[base + 192 + subgroup * 2 + i].cast(dtypes.uint8).bitcast(dtypes.int8).float() for i in range(2)]
    dbits = raw[base + 208].cast(dtypes.uint16) | (raw[base + 209].cast(dtypes.uint16) << 8)
    value = (dots[0].float() * scales[0] + dots[1].float() * scales[1]) * xd[0, group] * dbits.bitcast(dtypes.float16).float()
    updates.append(acc.store(acc.after(chunk) + value))
  update = UOp.group(*updates).end(chunk)
  totals = [_amd_wave_sum(acc.after(update)[0].load(), lane, 32) for acc in accs]
  stores = [out[0, output.valid(lane.eq(0))].store(total.cast(out.dtype)) for output,total in zip(outputs, totals)]
  return UOp.group(*stores).end(output_block, lane).sink(arg=KernelInfo(name="linear_q6", opts_to_apply=()))

def q8_linear(layer:Linear, x:Tensor, prepared:tuple[Tensor, ...]|None=None) -> Tensor:
  tokens = int(x.numel()) // layer.in_features
  if layer.ggml_type == 13:
    xq, xd, xsum = prepared if prepared is not None and len(prepared) == 3 else q8_quantize_sum(x, tokens, layer.in_features)
  else:
    xq, xd = prepared[:2] if prepared is not None else q8_quantize(x, tokens, layer.in_features)
  out = Tensor.empty(tokens, layer.out_features, dtype=dtypes.float32, device=x.device)
  if layer._raw_uop is None: layer._prepare_packed()
  assert layer._raw_uop is not None and layer._raw_offset_uop is not None
  if layer.ggml_type == 13:
    raw_words = layer._raw_uop.bitcast(dtypes.uint32)
    if tokens % 16 == 0 and layer.out_features % 16 == 0:
      qk_srcs = (out.uop, raw_words, x.cast(dtypes.float16).contiguous().uop, layer._raw_offset_uop)
      params = [UOp.placeholder_like(src, slot=i) for i,src in enumerate(qk_srcs)]
      kernel = _qk_linear_f16_wmma_kernel(params[0], params[1], params[2], layer.out_features,
                                          layer.in_features, params[3][0]).call(*qk_srcs)
      out = Tensor(qk_srcs[0].after(kernel)).reshape(*x.shape[:-1], layer.out_features)
      return out if layer.bias is None else out + layer.bias
    qk_decode_srcs = (out.uop, raw_words, xq.uop, xd.uop, xsum.uop, layer._raw_offset_uop)
    params = [UOp.placeholder_like(src, slot=i) for i,src in enumerate(qk_decode_srcs)]
    kernel = _qk_linear_kernel(params[0], params[1], params[2], params[3], params[4], layer.out_features,
                               layer.in_features, params[5][0]).call(*qk_decode_srcs)
    out = Tensor(qk_decode_srcs[0].after(kernel)).reshape(*x.shape[:-1], layer.out_features)
    return out if layer.bias is None else out + layer.bias
  if layer.ggml_type == 23:
    use_wmma = tokens % 16 == 0 and layer.out_features % 16 == 0
    raw_words = layer._raw_uop.bitcast(dtypes.uint32)
    if use_wmma:
      lut = iq4_half_lut(str(x.device))
      iq4_srcs = (out.uop, raw_words, x.cast(dtypes.float16).contiguous().uop, lut.uop, layer._raw_offset_uop)
    else: iq4_srcs = (out.uop, raw_words, xq.uop, xd.uop, layer._raw_offset_uop)
    params = [UOp.placeholder_like(src, slot=i) for i,src in enumerate(iq4_srcs)]
    kernel = (_iq4_linear_f16_wmma_kernel(params[0], params[1], params[2], params[3], layer.out_features,
                                          layer.in_features, params[4][0])
              if use_wmma else _iq4_linear_kernel(params[0], params[1], params[2], params[3], layer.out_features,
                                                   layer.in_features, params[4][0])).call(*iq4_srcs)
    out = Tensor(iq4_srcs[0].after(kernel)).reshape(*x.shape[:-1], layer.out_features)
    return out if layer.bias is None else out + layer.bias
  assert layer.ggml_type == 14
  srcs = (out.uop, layer._raw_uop, xq.uop, xd.uop, layer._raw_offset_uop)
  params = [UOp.placeholder_like(src, slot=i) for i,src in enumerate(srcs)]
  kernel = _q6_linear_kernel(params[0], params[1], params[2], params[3], layer.out_features,
                             layer.in_features, params[4][0] * 4).call(*srcs)
  out = Tensor(srcs[0].after(kernel)).reshape(*x.shape[:-1], layer.out_features)
  return out if layer.bias is None else out + layer.bias

@functools.cache
def iq4_half_lut(device:str) -> Tensor:
  from tinygrad.runtime.autogen.ggml_common import kvalues_iq4nl
  values = [x for j in range(16) for i in range(16) for x in (kvalues_iq4nl[i], kvalues_iq4nl[j])]
  return Tensor(values, dtype=dtypes.float16, device=device).bitcast(dtypes.uint32).contiguous().realize()
