from tinygrad import Tensor, UOp, getenv
from tinygrad.uop.ops import AxisType, KernelInfo, Ops
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import DEBUG, GlobalCounters, Context
import math

B = getenv("B", 1)
H = getenv("H", 32)
N = getenv("N", 1024)
D = getenv("D", 64)
assert D % 16 == 0 and N % 16 == 0

BLOCK_M, BLOCK_N = 64, 64
WARP_SIZE = 32
WMMA_M, WMMA_N, WMMA_K = 16, 16, 16
WAVES_M = 4
LANES_PER_WAVE_M, LANES_PER_WAVE_N = 2, 16
WMMA_ACC = WMMA_M // LANES_PER_WAVE_M
THREADS_PER_BLOCK = WARP_SIZE * WAVES_M

TM = BLOCK_M // (WAVES_M * LANES_PER_WAVE_M)
TN = BLOCK_N // LANES_PER_WAVE_N
TD = D // LANES_PER_WAVE_N
LDS_PAD = 4  # pad LDS rows to reduce bank conflicts

WMMA_ARG = ((WMMA_M, WMMA_N, WMMA_K), 'AMD', 32)
SCALE = 1.0 / math.sqrt(D)
LOG2E = math.log2(math.e)

def warp_shfl_xor(val:UOp, offset:int, lane:UOp) -> UOp:
  """Read val from lane ^ offset using ds_bpermute."""
  return UOp(Ops.CUSTOM, dtypes.float, (((lane ^ offset) * 4).cast(dtypes.int), val),
             arg="__builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute({0}, __builtin_bit_cast(int, {1})))")

def warp_reduce_max(val:UOp, lane:UOp) -> UOp:
  for offset in [8, 4, 2, 1]: val = val.maximum(warp_shfl_xor(val, offset, lane))
  return val

def warp_reduce_sum(val:UOp, lane:UOp) -> UOp:
  for offset in [8, 4, 2, 1]: val = val + warp_shfl_xor(val, offset, lane)
  return val

def amd_flash_attention(o:UOp, q:UOp, k:UOp, v:UOp) -> UOp:
  block_bh = UOp.range(B * H, 0, AxisType.GLOBAL)
  block_m = UOp.range(N // BLOCK_M, 1, AxisType.GLOBAL)

  q = q.reshape(B*H, N//BLOCK_M, BLOCK_M, D)[block_bh, block_m]
  k = k.reshape(B*H, N//BLOCK_N, BLOCK_N, D)[block_bh]
  v = v.reshape(B*H, N//BLOCK_N, BLOCK_N, D)[block_bh]
  o = o.reshape(B*H, N//BLOCK_M, BLOCK_M, D)[block_bh, block_m]

  wave_m = UOp.range(WAVES_M, 2, AxisType.LOCAL)
  lane = UOp.range(WARP_SIZE, -1, AxisType.WARP)
  tid = wave_m * WARP_SIZE + lane
  lane_n = lane % LANES_PER_WAVE_N

  # LDS: slot 0 = Q/P (Q reloaded each iter, P overwrites after QK), slot 1 = K/V
  # TODO: memory planner should hande this aliasing
  ELEMS_PER_THREAD = BLOCK_M * D // THREADS_PER_BLOCK
  QP_lds = UOp.placeholder((BLOCK_M, max(D, BLOCK_N) + LDS_PAD), dtypes.half, slot=0, addrspace=AddrSpace.LOCAL)
  Q_lds = QP_lds[:, :D]
  P_lds = QP_lds[:, :BLOCK_N]
  KV_lds = UOp.placeholder((BLOCK_N, D + LDS_PAD), dtypes.half, slot=1, addrspace=AddrSpace.LOCAL)[:, :D]

  # register state
  acc = UOp.placeholder((TM, TD), dtypes.float, slot=2, addrspace=AddrSpace.REG)
  m_i = UOp.placeholder((TM,), dtypes.float, slot=3, addrspace=AddrSpace.REG)
  l_i = UOp.placeholder((TM,), dtypes.float, slot=4, addrspace=AddrSpace.REG)
  acc = acc.after(acc.store(acc.const_like(0)))
  m_i = m_i.after(m_i.store(m_i.const_like(-math.inf)))
  l_i = l_i.after(l_i.store(l_i.const_like(0)))

  # ====== KV tile loop ======
  n_tile = UOp.range(N // BLOCK_N, 100, AxisType.REDUCE)

  # load Q + K into LDS (Q reloaded each iteration since P overwrites slot 0)
  Q_store = Q_lds.after(n_tile).reshape(THREADS_PER_BLOCK, ELEMS_PER_THREAD)[tid].store(
    q.reshape(THREADS_PER_BLOCK, ELEMS_PER_THREAD)[tid])
  K_store = KV_lds.reshape(THREADS_PER_BLOCK, ELEMS_PER_THREAD)[tid].store(
    k[n_tile].reshape(THREADS_PER_BLOCK, ELEMS_PER_THREAD)[tid])
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
  qk = UOp(Ops.SHAPED_WMMA, dtypes.float, (q_frag, k_frag, S_frag.after(k_qk)), arg=WMMA_ARG)
  qk_done = S_frag.store(qk).end(tm1, tn1).end(k_qk)
  S_reg = S_reg.after(qk_done)

  # -- softmax in registers with warp shuffles --
  S_reg = S_reg.after(S_reg.store(S_reg * SCALE))

  # per-thread local row max over TN=4 elements, then warp reduce across 16 lanes
  m_ij = UOp.placeholder((TM,), dtypes.float, slot=7, addrspace=AddrSpace.REG)
  m_ij = m_ij.after(m_ij.after(n_tile).store(m_ij.const_like(-math.inf)))
  rm1 = UOp.range(TM, 260, AxisType.LOOP)
  rm2 = UOp.range(TN, 261, AxisType.REDUCE)
  m_ij = m_ij.after(m_ij[rm1].store(m_ij.after(rm2)[rm1].maximum(S_reg[rm1, rm2])).end(rm2, rm1))
  # warp reduce max (in-place)
  ri_w = UOp.range(TM, 270, AxisType.LOOP)
  m_ij = m_ij.after(m_ij[ri_w].store(warp_reduce_max(m_ij[ri_w], lane)).end(ri_w))

  # compute P = exp(S - m_ij) in S_reg
  S_reg = S_reg.after(S_reg.store(((S_reg - m_ij.reshape(TM, 1).expand(TM, TN)) * LOG2E).exp2()))

  p_local = UOp.placeholder((TM,), dtypes.float, slot=8, addrspace=AddrSpace.REG)
  p_local = p_local.after(p_local.after(n_tile).store(p_local.const_like(0)))
  rp1 = UOp.range(TM, 290, AxisType.LOOP)
  rp2 = UOp.range(TN, 291, AxisType.REDUCE)
  p_local = p_local.after(p_local[rp1].store(p_local.after(rp2)[rp1] + S_reg[rp1, rp2]).end(rp2, rp1))
  ri_ws = UOp.range(TM, 295, AxisType.LOOP)
  p_sum = p_local.after(p_local[ri_ws].store(warp_reduce_sum(p_local[ri_ws], lane)).end(ri_ws))

  # write P = exp(S - m_ij) to P_lds (reuses slot 0, Q no longer needed)
  P_write = P_lds.reshape(WAVES_M, TM // WMMA_ACC, WMMA_ACC, LANES_PER_WAVE_M, TN, LANES_PER_WAVE_N)
  P_write = P_write.permute((0, 3, 5, 1, 2, 4)).reshape(THREADS_PER_BLOCK, TM, TN)
  rw1 = UOp.range(TM, 296, AxisType.LOOP)
  rw2 = UOp.range(TN, 297, AxisType.LOOP)
  P_store = P_write[tid, rw1, rw2].store(S_reg[rw1, rw2].cast(dtypes.half)).end(rw1, rw2)

  # -- online softmax correction --
  ri4 = UOp.range(TM, 330, AxisType.LOOP)
  m_new_val = m_i[ri4].maximum(m_ij[ri4])
  alpha_val = ((m_i[ri4] - m_new_val) * LOG2E).exp2()
  beta_val = ((m_ij[ri4] - m_new_val) * LOG2E).exp2()
  rj4 = UOp.range(TD, 331, AxisType.LOOP)
  correction = UOp.group(
    acc[ri4, rj4].store(alpha_val * acc[ri4, rj4]).end(rj4),
    l_i[ri4].store(alpha_val * l_i[ri4] + beta_val * p_sum[ri4]),
    m_i[ri4].store(m_new_val),
  ).end(ri4)
  acc = acc.after(correction)
  l_i = l_i.after(correction)
  m_i = m_i.after(correction)

  # load V into KV_lds (must wait for QK WMMA to finish reading K from KV_lds)
  V_store = KV_lds.after(qk_done).reshape(THREADS_PER_BLOCK, ELEMS_PER_THREAD)[tid].store(
    v[n_tile].reshape(THREADS_PER_BLOCK, ELEMS_PER_THREAD)[tid])
  pv_barrier = UOp.barrier(UOp.group(P_store, V_store))
  P_lds = P_lds.after(pv_barrier)
  KV_lds_v = KV_lds.after(pv_barrier)

  # -- acc += P @ V via WMMA --
  k_pv = UOp.range(BLOCK_N // WMMA_K, 400, AxisType.REDUCE)
  tm2 = UOp.range(TM // WMMA_ACC, 401, AxisType.LOOP)
  tn2 = UOp.range(TD, 402, AxisType.LOOP)
  acc_frag = acc.reshape(TM // WMMA_ACC, WMMA_ACC, TD).permute(0, 2, 1)[tm2, tn2]
  p_frag = P_lds.reshape(WAVES_M, TM // WMMA_ACC, WMMA_M, BLOCK_N // WMMA_K, WMMA_K)[wave_m, tm2, lane_n, k_pv]
  v_frag = KV_lds_v.reshape(TD, WMMA_N, BLOCK_N // WMMA_K, WMMA_K)[tn2, lane_n, k_pv]
  pv = UOp(Ops.SHAPED_WMMA, dtypes.float, (p_frag, v_frag, acc_frag.after(k_pv)), arg=WMMA_ARG)

  # end KV tile loop
  n_tile_end = acc_frag.store(pv).end(tm2, tn2).end(k_pv).barrier().end(n_tile)
  acc = acc.after(n_tile_end)
  l_i = l_i.after(n_tile_end)
  m_i = m_i.after(n_tile_end)

  # normalize: acc /= l_i
  acc = acc.after(acc.store(acc * (1 / l_i).reshape(TM, 1).expand(TM, TD)))

  # store output
  o = o.reshape(WAVES_M, TM // WMMA_ACC, WMMA_ACC, LANES_PER_WAVE_M, TD, LANES_PER_WAVE_N)
  o = o.permute((0, 3, 5, 1, 2, 4)).reshape(THREADS_PER_BLOCK, TM, TD)
  return o[tid].store(acc).end(wave_m, lane).end(block_m, block_bh).sink(arg=KernelInfo(opts_to_apply=()))

if __name__ == "__main__":
  q = Tensor.rand(B, H, N, D).cast(dtypes.half)
  k = Tensor.rand(B, H, N, D).cast(dtypes.half)
  v = Tensor.rand(B, H, N, D).cast(dtypes.half)
  o = Tensor.empty(B, H, N, D, dtype=dtypes.float)
  with Context(DEBUG=0): Tensor.realize(q, k, v)

  q_flat, k_flat, v_flat, o_flat = q.reshape(B*H, N, D), k.reshape(B*H, N, D), v.reshape(B*H, N, D), o.reshape(B*H, N, D)
  NUM_RUNS = getenv("CNT", 5)
  ets = []
  with Context(DEBUG=2):
    for _ in range(NUM_RUNS):
      GlobalCounters.reset()
      tst = Tensor.custom_kernel(o_flat, q_flat, k_flat, v_flat, fxn=amd_flash_attention)[0].realize()
      ets.append(GlobalCounters.time_sum_s)
  print(f"best time: {min(ets)*1e3:.2f}ms")

  if getenv("VERIFY", 1):
    with Context(DEBUG=0):
      ref = q.float().scaled_dot_product_attention(k.float(), v.float()).reshape(B*H, N, D).realize()
      err = (ref - tst).square().mean().item()
    print(f"mean squared error {err}")
    if err > 1e-2:
      raise RuntimeError("flash attention is wrong!")
    else:
      print("flash attention is correct!")
