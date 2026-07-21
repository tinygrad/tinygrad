from tinygrad import Tensor, UOp, getenv
from tinygrad.uop.ops import AxisType, KernelInfo, Ops
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import GlobalCounters, Context
import functools, math

BLOCK_M, BLOCK_N = 32, 32
DECODE_HEAD_TILE = 4
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
  idx = ((lane ^ offset) * 4).cast(dtypes.int)
  if val.op is Ops.INDEX and val.addrspace == AddrSpace.REG: val = val.load()
  return UOp(Ops.CUSTOM, dtypes.float, (idx, val),
             arg="__builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute({0}, __builtin_bit_cast(int, {1})))")

def warp_reduce_max(val, lane):
  """Tree reduce MAX across LANES_PER_WAVE_N=16 lanes."""
  for offset in [8, 4, 2, 1]:
    val = UOp(Ops.MAX, dtypes.float, (val, warp_shfl_xor(val, offset, lane)))
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
def _amd_flash_attention_decode_partial(out:UOp, stats:UOp, q:UOp, cache_kv:UOp, valid_kv_len:int|UOp, max_kv_len:int,
                                        block_n:int) -> UOp:
  _, B, H_KV, N, D = cache_kv.shape
  _, H, M, _ = q.shape
  assert M == 1 and H % H_KV == 0 and D % WARP_SIZE == 0 and max_kv_len <= N and max_kv_len % block_n == 0
  G, CHUNK, DV = H // H_KV, block_n, D // WARP_SIZE
  # One wave avoids an unsafe LDS merge and gives each workgroup a contiguous KV range to stream. At long context, updating
  # two tokens at once halves the accumulator rescale work without increasing register pressure as much as larger groups.
  decode_waves = 1
  decode_group = 1 if max_kv_len <= 8192 else 2
  assert G % DECODE_HEAD_TILE == 0
  block_bhkv = UOp.range(B*H_KV*(G//DECODE_HEAD_TILE), 0, AxisType.GLOBAL)
  block_n = UOp.range((valid_kv_len+CHUNK-1)//CHUNK, 1, AxisType.GLOBAL)
  lane, wave = UOp.range(WARP_SIZE, 2, AxisType.LOCAL), UOp.range(decode_waves, 3, AxisType.LOCAL)
  head_group = block_bhkv % (G//DECODE_HEAD_TILE)
  bhkv = block_bhkv // (G//DECODE_HEAD_TILE)
  b, kv_head = bhkv // H_KV, bhkv % H_KV
  dims = tuple(lane + i*WARP_SIZE for i in range(DV))

  acc = UOp.placeholder((DECODE_HEAD_TILE, DV), dtypes.float, slot=0, addrspace=AddrSpace.REG)
  row_max = UOp.placeholder((DECODE_HEAD_TILE,), dtypes.float, slot=1, addrspace=AddrSpace.REG)
  row_sum = UOp.placeholder((DECODE_HEAD_TILE,), dtypes.float, slot=2, addrspace=AddrSpace.REG)
  init = UOp.group(acc.store(acc.const_like(0)), row_max.store(row_max.const_like(-math.inf)), row_sum.store(row_sum.const_like(0)))
  acc, row_max, row_sum = acc.after(init), row_max.after(init), row_sum.after(init)

  offset = UOp.range(CHUNK//decode_waves//decode_group, 100, AxisType.REDUCE)
  keys = tuple(block_n*CHUNK + wave*(CHUNK//decode_waves) + offset*decode_group + i for i in range(decode_group))
  valid = tuple(key < valid_kv_len for key in keys)
  kvals = tuple(tuple(cache_kv[0, b, kv_head, key, d].float() for d in dims) for key in keys)
  vvals = tuple(tuple(cache_kv[1, b, kv_head, key, d].float() for d in dims) for key in keys)
  updates = []
  for head in range(DECODE_HEAD_TILE):
    q_head = kv_head*G + head_group*DECODE_HEAD_TILE + head
    scores = tuple(wave_reduce_sum(sum((q[b, q_head, 0, d].float()*k for d,k in zip(dims, key_kvals)),
                                       UOp.const(dtypes.float, 0)), lane) / math.sqrt(D) for key_kvals in kvals)
    new_max = row_max[head]
    for is_valid, score in zip(valid, scores): new_max = new_max.maximum(is_valid.where(score, UOp.const(dtypes.float, -math.inf)))
    alpha = ((row_max[head]-new_max)*LOG2E).exp2()
    betas = tuple(is_valid.where(((score-new_max)*LOG2E).exp2(), UOp.const(dtypes.float, 0)) for is_valid,score in zip(valid, scores))
    updates += [acc[head].store(acc[head]*alpha + sum((UOp.stack(*value)*beta for value,beta in zip(vvals, betas)), acc[head].const_like(0))),
                row_sum[head].store(row_sum[head]*alpha + sum(betas, UOp.const(dtypes.float, 0))), row_max[head].store(new_max)]
  update = UOp.group(*updates).end(offset)
  acc, row_max, row_sum = acc.after(update), row_max.after(update), row_sum.after(update)

  partial_acc = UOp.placeholder((DECODE_HEAD_TILE, decode_waves, D), dtypes.float, slot=3, addrspace=AddrSpace.LOCAL)
  partial_stats = UOp.placeholder((DECODE_HEAD_TILE, decode_waves, 2), dtypes.float, slot=4, addrspace=AddrSpace.LOCAL)
  partial_stores = []
  for head in range(DECODE_HEAD_TILE):
    partial_stores += [partial_acc[head, wave, d].store(acc[head, i]) for i,d in enumerate(dims)]
    partial_stores += [partial_stats[head, wave.valid(lane.eq(0)), 0].store(row_max[head]),
                       partial_stats[head, wave.valid(lane.eq(0)), 1].store(row_sum[head])]
  merged = UOp.group(*partial_stores).barrier()
  stores = []
  for head in range(DECODE_HEAD_TILE):
    q_head = kv_head*G + head_group*DECODE_HEAD_TILE + head
    wave_max = tuple(partial_stats.after(merged)[head, w, 0] for w in range(decode_waves))
    maximum = wave_max[0]
    for wave_value in wave_max[1:]: maximum = maximum.maximum(wave_value)
    scales = tuple(((partial_stats.after(merged)[head, w, 0]-maximum)*LOG2E).exp2() for w in range(decode_waves))
    denominator = sum((partial_stats.after(merged)[head, w, 1]*scales[w] for w in range(decode_waves)), UOp.const(dtypes.float, 0))
    stores += [out[b, q_head, block_n, d.valid(wave.eq(0))].store(
      sum((partial_acc.after(merged)[head, w, d]*scales[w] for w in range(decode_waves)), UOp.const(dtypes.float, 0))) for d in dims]
    stores += [stats[b, q_head.valid(lane.eq(0) & wave.eq(0)), block_n, 0].store(maximum),
               stats[b, q_head.valid(lane.eq(0) & wave.eq(0)), block_n, 1].store(denominator)]
  return UOp.group(*stores).end(lane, wave, block_n, block_bhkv).sink(arg=KernelInfo(name="flash_decode_partial", opts_to_apply=()))

@functools.cache
def _amd_flash_attention_decode_reduce(out:UOp, partial:UOp, stats:UOp, valid_chunks:int|UOp) -> UOp:
  B, H, _, D = out.shape
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

@functools.cache
def _amd_flash_attention_decode_reduce_partial(out:UOp, out_stats:UOp, partial:UOp, stats:UOp,
                                               valid_chunks:int|UOp, group_size:int) -> UOp:
  B, H, _, D = partial.shape
  DV = D // WARP_SIZE
  block = UOp.range(B*H*((valid_chunks+group_size-1)//group_size), 0, AxisType.GLOBAL)
  lane = UOp.range(WARP_SIZE, 1, AxisType.LOCAL)
  group, bh = block % ((valid_chunks+group_size-1)//group_size), block // ((valid_chunks+group_size-1)//group_size)
  b, head = bh // H, bh % H
  dims = tuple(lane + i*WARP_SIZE for i in range(DV))
  start, count = group*group_size, (valid_chunks-group*group_size).minimum(group_size)

  row_max = UOp.placeholder((1,), dtypes.float, slot=0, addrspace=AddrSpace.REG)
  row_max = row_max.after(row_max.store(row_max.const_like(-math.inf)))
  max_chunk = UOp.range(count, 100, AxisType.REDUCE)
  max_done = row_max.store(row_max.after(max_chunk).maximum(stats[b, head, start+max_chunk, 0])).end(max_chunk)
  row_max = row_max.after(max_done)

  numerator = UOp.placeholder((DV,), dtypes.float, slot=1, addrspace=AddrSpace.REG)
  denominator = UOp.placeholder((1,), dtypes.float, slot=2, addrspace=AddrSpace.REG)
  init = UOp.group(numerator.store(numerator.const_like(0)), denominator.store(denominator.const_like(0)))
  numerator, denominator = numerator.after(init), denominator.after(init)
  chunk = UOp.range(count, 101, AxisType.REDUCE)
  scale = ((stats[b, head, start+chunk, 0]-row_max[0])*LOG2E).exp2()
  update = UOp.group(numerator.store(numerator.after(chunk) + UOp.stack(*(partial[b, head, start+chunk, d] for d in dims))*scale),
                     denominator.store(denominator.after(chunk) + stats[b, head, start+chunk, 1]*scale)).end(chunk)
  numerator, denominator = numerator.after(update), denominator.after(update)
  stores = [out[b, head, group, d].store(numerator[i]) for i,d in enumerate(dims)] + \
           [out_stats[b, head.valid(lane.eq(0)), group, 0].store(row_max[0]),
            out_stats[b, head.valid(lane.eq(0)), group, 1].store(denominator[0])]
  return UOp.group(*stores).end(lane, block).sink(arg=KernelInfo(name="flash_decode_reduce_partial", opts_to_apply=()))

def amd_flash_attention_decode(q:Tensor, cache_kv:Tensor, valid_kv_len:int|UOp, max_kv_len:int|None=None) -> Tensor:
  _, B, H_KV, N, D = cache_kv.shape
  _, H, M, _ = q.shape
  max_kv_len = N if max_kv_len is None else max_kv_len
  block_n = 128 if max_kv_len <= 8192 else 1024
  assert M == 1 and max_kv_len <= N and max_kv_len % block_n == 0
  chunks = max_kv_len // block_n
  partial = Tensor.empty(B, H, chunks, D, dtype="float32", device=q.device)
  stats = Tensor.empty(B, H, chunks, 2, dtype="float32", device=q.device)
  partial, stats = Tensor.custom_kernel(partial, stats, q, cache_kv,
    fxn=functools.partial(_amd_flash_attention_decode_partial, valid_kv_len=valid_kv_len, max_kv_len=max_kv_len, block_n=block_n))[:2]
  live_chunks = (valid_kv_len+block_n-1)//block_n
  if max_kv_len > 8192:
    reduce_group = 8
    reduced_chunks = (chunks+reduce_group-1)//reduce_group
    reduced = Tensor.empty(B, H, reduced_chunks, D, dtype="float32", device=q.device)
    reduced_stats = Tensor.empty(B, H, reduced_chunks, 2, dtype="float32", device=q.device)
    partial, stats = Tensor.custom_kernel(reduced, reduced_stats, partial, stats,
      fxn=functools.partial(_amd_flash_attention_decode_reduce_partial, valid_chunks=live_chunks, group_size=reduce_group))[:2]
    live_chunks = (live_chunks+reduce_group-1)//reduce_group
  out = Tensor.empty(B, H, 1, D, dtype="float32", device=q.device)
  return Tensor.custom_kernel(out, partial, stats,
    fxn=functools.partial(_amd_flash_attention_decode_reduce, valid_chunks=live_chunks))[0]

@functools.cache
def _amd_flash_attention(o:UOp, q:UOp, k:UOp, v:UOp, causal:bool, valid_kv_len:int|UOp|None=None,
                         key_limit:int|UOp|None=None) -> UOp:
  # inputs are q=(B*H, M, D), k/v=(B*H, N, D). For causal attention q is the final M tokens of k/v.
  BH, M, D = q.shape
  physical_n = k.shape[1]
  N = physical_n if valid_kv_len is None else valid_kv_len
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
  k, v = k[block_bh // gqa_group], v[block_bh // gqa_group]
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
  n_tiles = (N - M + (block_m + 1) * BLOCK_M + BLOCK_N - 1) // BLOCK_N if causal else N // BLOCK_N
  n_tile = UOp.range(n_tiles, 100, AxisType.REDUCE)

  # load Q + K into LDS (Q reloaded each iteration since P overwrites slot 0)
  Q_lds = QP_lds[:, :D]
  Q_store = Q_lds.after(n_tile).reshape(THREADS_PER_BLOCK, Q_ELEMS_PER_THREAD)[tid].store(
    q.reshape(THREADS_PER_BLOCK, Q_ELEMS_PER_THREAD)[tid])
  load_k = UOp.range(KV_ELEMS_PER_THREAD, 90, AxisType.LOOP)
  K_store = KV_lds.reshape(THREADS_PER_BLOCK, KV_ELEMS_PER_THREAD)[tid, load_k].store(
    k.reshape(physical_n*D)[n_tile*BLOCK_N*D + tid*KV_ELEMS_PER_THREAD + load_k]).end(load_k)
  qk_load_barrier = UOp.barrier(UOp.group(Q_store, K_store))
  Q_lds = Q_lds.after(qk_load_barrier)
  KV_lds_k = KV_lds.after(qk_load_barrier)

  # -- S = Q @ K^T via WMMA (re-init each n_tile) --
  S_reg = UOp.placeholder((TM, TN), dtypes.float, slot=6, addrspace=AddrSpace.REG)
  S_reg = S_reg.after(S_reg.after(n_tile).store(S_reg.const_like(0)))
  k_qk = UOp.range(D // WMMA_K, 101, AxisType.REDUCE)
  tm1 = UOp.range(TM // WMMA_ACC, 200, AxisType.LOOP)
  tn1 = UOp.range(TN, 201, AxisType.LOOP)
  S_frag = S_reg.reshape(TM // WMMA_ACC, WMMA_ACC, TN).permute(0, 2, 1)[tm1, tn1]
  q_frag = Q_lds.reshape(WAVES_M, TM // WMMA_ACC, WMMA_M, D // WMMA_K, WMMA_K)[wave_m, tm1, lane_n, k_qk]
  k_frag = KV_lds_k.reshape(TN, WMMA_N, D // WMMA_K, WMMA_K)[tn1, lane_n, k_qk]
  qk = UOp.wmma(q_frag, k_frag, S_frag.after(k_qk), *WMMA_ARG)
  qk_done = S_frag.store(qk).end(tm1, tn1).end(k_qk)
  S_reg = S_reg.after(qk_done)

  # -- softmax in registers with warp shuffles --
  S_reg = S_reg.after(S_reg.store(S_reg * SCALE))

  if causal:
    # WMMA accumulator ownership: each lane owns an 8x4 fragment of the 64x64 score tile.
    # q is aligned to the right of k, matching PyTorch's causal_lower_right mask.
    rm = UOp.range(TM, 250, AxisType.LOOP)
    rn = UOp.range(TN, 251, AxisType.LOOP)
    q_idx = N - M + block_m * BLOCK_M + wave_m * WMMA_M + rm * LANES_PER_WAVE_M + lane_m
    k_idx = n_tile * BLOCK_N + rn * LANES_PER_WAVE_N + lane_n
    valid = k_idx <= q_idx
    if key_limit is not None: valid = valid & (k_idx < key_limit)
    masked = valid.where(S_reg[rm, rn], S_reg[rm, rn].const_like(-math.inf))
    S_reg = S_reg.after(S_reg[rm, rn].store(masked).end(rm, rn))

  # per-thread local row max over TN=4 elements, then warp reduce across 16 lanes
  m_ij = UOp.placeholder((TM,), dtypes.float, slot=7, addrspace=AddrSpace.REG)
  m_ij = m_ij.after(m_ij.after(n_tile).store(m_ij.const_like(-math.inf)))
  rm2 = UOp.range(TN, 261, AxisType.REDUCE)
  m_ij = m_ij.after(m_ij.store(m_ij.after(rm2).maximum(S_reg[:, rm2])).end(rm2))
  # warp reduce max (in-place)
  ri_w = UOp.range(TM, 270, AxisType.LOOP)
  m_ij = m_ij.after(m_ij[ri_w].store(warp_reduce_max(m_ij[ri_w], lane)).end(ri_w))

  # compute P = exp(S - m_ij) in S_reg
  S_reg = S_reg.after(S_reg.store(((S_reg - m_ij.reshape(TM, 1).expand(TM, TN)) * LOG2E).exp2()))

  p_local = UOp.placeholder((TM,), dtypes.float, slot=8, addrspace=AddrSpace.REG)
  p_local = p_local.after(p_local.after(n_tile).store(p_local.const_like(0)))
  ri_ws = UOp.range(TM, 295, AxisType.LOOP)
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
  ri4 = UOp.range(TM, 330, AxisType.LOOP)
  m_new_val = m_i[ri4].maximum(m_ij[ri4])
  alpha_val = ((m_i[ri4] - m_new_val) * LOG2E).exp2()
  beta_val = ((m_ij[ri4] - m_new_val) * LOG2E).exp2()
  rj4 = UOp.range(TD, 331, AxisType.LOOP)
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
  load_v = UOp.range(KV_ELEMS_PER_THREAD, 390, AxisType.LOOP)
  V_store = V_copy.reshape(THREADS_PER_BLOCK, KV_ELEMS_PER_THREAD)[tid, load_v].store(
    v.reshape(physical_n*D)[n_tile*BLOCK_N*D + tid*KV_ELEMS_PER_THREAD + load_v]).end(load_v)
  pv_barrier = UOp.barrier(UOp.group(P_store, V_store))
  P_lds = P_lds.after(pv_barrier)
  V_lds = V_lds.after(pv_barrier)

  # -- acc += beta * (P @ V) via WMMA --
  pv_acc = UOp.placeholder((TM, TD), dtypes.float, slot=10, addrspace=AddrSpace.REG)
  pv_acc = pv_acc.after(pv_acc.after(n_tile).store(pv_acc.const_like(0))).after(pv_barrier)
  k_pv = UOp.range(BLOCK_N // WMMA_K, 400, AxisType.REDUCE)
  tm2 = UOp.range(TM // WMMA_ACC, 401, AxisType.LOOP)
  tn2 = UOp.range(TD, 402, AxisType.LOOP)
  pv_frag = pv_acc.reshape(TM // WMMA_ACC, WMMA_ACC, TD).permute(0, 2, 1)[tm2, tn2]
  p_frag = P_lds[wave_n].reshape(WAVES_M, TM // WMMA_ACC, WMMA_M, BLOCK_N // WMMA_K, WMMA_K)[wave_m, tm2, lane_n, k_pv]
  v_frag = V_lds.reshape(WAVES_N, TD, WMMA_N, BLOCK_N // WMMA_K, WMMA_K)[wave_n, tn2, lane_n, k_pv]
  pv = UOp.wmma(p_frag, v_frag, pv_frag.after(k_pv), *WMMA_ARG)
  pv_done = pv_frag.store(pv).end(tm2, tn2).end(k_pv)
  pv_acc = pv_acc.after(pv_done)

  ri5 = UOp.range(TM, 410, AxisType.LOOP)
  rj5 = UOp.range(TD, 411, AxisType.LOOP)
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

def amd_flash_attention(o:UOp, q:UOp, k:UOp, v:UOp) -> UOp:
  return _amd_flash_attention(o, q, k, v, causal=False)

def amd_flash_attention_causal(o:UOp, q:UOp, k:UOp, v:UOp) -> UOp:
  return _amd_flash_attention(o, q, k, v, causal=True)

def amd_flash_attention_causal_cached(o:UOp, q:UOp, cache_kv:UOp, *, valid_kv_len:int|UOp, key_limit:int|UOp|None=None) -> UOp:
  _, B, H_KV, N, D = cache_kv.shape
  k = cache_kv[0].reshape(B*H_KV, N, D)
  v = cache_kv[1].reshape(B*H_KV, N, D)
  return _amd_flash_attention(o, q, k, v, causal=True, valid_kv_len=valid_kv_len, key_limit=key_limit)

if __name__ == "__main__":
  B, H, N, D = getenv("B", 1), getenv("H", 32), getenv("N", 1024), getenv("D", 64)
  M, causal = getenv("M", N), getenv("CAUSAL", 0)
  q = Tensor.rand(B, H, M, D).cast(dtypes.half)
  k = Tensor.rand(B, H, N, D).cast(dtypes.half)
  v = Tensor.rand(B, H, N, D).cast(dtypes.half)
  o = Tensor.empty(B, H, M, D, dtype=dtypes.float)
  with Context(DEBUG=0): Tensor.realize(q, k, v)

  q_flat, k_flat, v_flat, o_flat = q.reshape(B*H, M, D), k.reshape(B*H, N, D), v.reshape(B*H, N, D), o.reshape(B*H, M, D)
  NUM_RUNS = getenv("CNT", 5)
  ets = []
  with Context(DEBUG=2):
    for _ in range(NUM_RUNS):
      GlobalCounters.reset()
      tst = Tensor.custom_kernel(o_flat, q_flat, k_flat, v_flat,
                                 fxn=amd_flash_attention_causal if causal else amd_flash_attention)[0].realize()
      ets.append(GlobalCounters.time_sum_s)
  print(f"best time: {min(ets)*1e3:.2f}ms")

  if getenv("VERIFY", 1):
    with Context(DEBUG=0):
      mask = Tensor.full((1, 1, M, N), float("-inf"), buffer=False).triu(N-M+1) if causal else None
      ref = q.float().scaled_dot_product_attention(k.float(), v.float(), attn_mask=mask).reshape(B*H, M, D).realize()
      diff = (ref - tst).abs()
      err, max_err = diff.square().mean().item(), diff.max().item()
    print(f"mean squared error {err}, max error {max_err}")
    if err > 1e-2:
      raise RuntimeError("flash attention is wrong!")
    else:
      print("flash attention is correct!")
