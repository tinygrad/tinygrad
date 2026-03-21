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
WAVES_M, WAVES_N = 2, 2
LANES_PER_WAVE_M, LANES_PER_WAVE_N = 2, 16
WMMA_ACC = WMMA_M // LANES_PER_WAVE_M
THREADS_PER_BLOCK = WARP_SIZE * WAVES_M * WAVES_N

TM = BLOCK_M // (WAVES_M * LANES_PER_WAVE_M)
TN = BLOCK_N // (WAVES_N * LANES_PER_WAVE_N)
TD = D // (WAVES_N * LANES_PER_WAVE_N)

N_THREADS_PER_ROW = WAVES_N * LANES_PER_WAVE_N

WMMA_ARG = ((WMMA_M, WMMA_N, WMMA_K), 'AMD', 32)
SCALE = 1.0 / math.sqrt(D)
LOG2E = math.log2(math.e)

def row_index(wave_m, ri, re, lane_m):
  return wave_m * (BLOCK_M // WAVES_M) + ri * WMMA_M + re * LANES_PER_WAVE_M + lane_m

def amd_flash_attention(o:UOp, q:UOp, k:UOp, v:UOp) -> UOp:
  block_bh = UOp.range(B * H, 0, AxisType.GLOBAL)
  block_m = UOp.range(N // BLOCK_M, 1, AxisType.GLOBAL)

  q = q.reshape(B*H, N//BLOCK_M, BLOCK_M, D)[block_bh, block_m]
  k = k.reshape(B*H, N//BLOCK_N, BLOCK_N, D)[block_bh]
  v = v.reshape(B*H, N//BLOCK_N, BLOCK_N, D)[block_bh]
  o = o.reshape(B*H, N//BLOCK_M, BLOCK_M, D)[block_bh, block_m]

  wave_m = UOp.range(WAVES_M, 2, AxisType.LOCAL)
  wave_n = UOp.range(WAVES_N, 3, AxisType.LOCAL)
  lane = UOp.range(WARP_SIZE, -1, AxisType.WARP)
  tid = (wave_m * WAVES_N + wave_n) * WARP_SIZE + lane
  lane_m = lane // LANES_PER_WAVE_N
  lane_n = lane % LANES_PER_WAVE_N
  n_thread_idx = wave_n * LANES_PER_WAVE_N + lane_n

  # load Q into LDS
  Q_lds = UOp.placeholder((BLOCK_M, D), dtypes.half, slot=0, addrspace=AddrSpace.LOCAL)
  Q_lds = Q_lds.after(UOp.barrier(Q_lds.reshape(-1, THREADS_PER_BLOCK)[:, tid].store(
    q.reshape(-1, THREADS_PER_BLOCK)[:, tid])))

  KV_lds = UOp.placeholder((BLOCK_N, D), dtypes.half, slot=1, addrspace=AddrSpace.LOCAL)

  # register state
  acc = UOp.placeholder((TM, TD), dtypes.float, slot=2, addrspace=AddrSpace.REG)
  m_i = UOp.placeholder((TM,), dtypes.float, slot=3, addrspace=AddrSpace.REG)
  l_i = UOp.placeholder((TM,), dtypes.float, slot=4, addrspace=AddrSpace.REG)
  acc = acc.after(acc.store(acc.const_like(0)))
  m_i = m_i.after(m_i.store(m_i.const_like(-math.inf)))
  l_i = l_i.after(l_i.store(l_i.const_like(0)))

  # LDS for cross-thread reduction
  red_lds = UOp.placeholder((BLOCK_M, N_THREADS_PER_ROW), dtypes.float, slot=5, addrspace=AddrSpace.LOCAL)
  P_lds = UOp.placeholder((BLOCK_M, BLOCK_N), dtypes.half, slot=6, addrspace=AddrSpace.LOCAL)

  # ====== KV tile loop ======
  n_tile = UOp.range(N // BLOCK_N, 100, AxisType.REDUCE)

  # load K into LDS
  KV_lds_k = KV_lds.after(UOp.barrier(KV_lds.reshape(-1, THREADS_PER_BLOCK)[:, tid].store(
    k[n_tile].reshape(-1, THREADS_PER_BLOCK)[:, tid])))

  # -- S = Q @ K^T via WMMA (re-init each n_tile) --
  S_reg = UOp.placeholder((TM, TN), dtypes.float, slot=7, addrspace=AddrSpace.REG)
  S_reg = S_reg.after(S_reg.after(n_tile).store(S_reg.const_like(0)))
  k_qk = UOp.range(D // WMMA_K, 101, AxisType.REDUCE)
  tm1 = UOp.range(TM // WMMA_ACC, 200, AxisType.LOOP)
  tn1 = UOp.range(TN, 201, AxisType.LOOP)
  S_frag = S_reg.reshape(TM // WMMA_ACC, WMMA_ACC, TN).permute(0, 2, 1)[tm1, tn1]
  q_frag = Q_lds.reshape(WAVES_M, TM // WMMA_ACC, WMMA_M, D // WMMA_K, WMMA_K)[wave_m, tm1, lane_n, k_qk]
  k_frag = KV_lds_k.reshape(WAVES_N, TN, WMMA_N, D // WMMA_K, WMMA_K)[wave_n, tn1, lane_n, k_qk]
  qk = UOp(Ops.SHAPED_WMMA, dtypes.float, (q_frag, k_frag, S_frag.after(k_qk)), arg=WMMA_ARG)
  S_reg = S_reg.after(S_frag.store(qk).end(tm1, tn1).end(k_qk).barrier())

  # -- softmax in registers with LDS reduction --
  S_reg = S_reg.after(S_reg.store(S_reg * SCALE))

  # per-thread local row max over TN elements
  m_local = UOp.placeholder((TM,), dtypes.float, slot=8, addrspace=AddrSpace.REG)
  m_local = m_local.after(m_local.after(n_tile).store(m_local.const_like(-math.inf)))
  rm1 = UOp.range(TM, 260, AxisType.LOOP)
  rm2 = UOp.range(TN, 261, AxisType.REDUCE)
  m_local = m_local.after(m_local[rm1].store(UOp(Ops.MAX, dtypes.float, (m_local.after(rm1, rm2)[rm1], S_reg[rm1, rm2]))).end(rm2, rm1))

  # write local max to LDS, barrier, read global max
  ri_r = UOp.range(TM, 270, AxisType.LOOP)
  red_lds = red_lds.after(UOp.barrier(red_lds[row_index(wave_m, ri_r // WMMA_ACC, ri_r % WMMA_ACC, lane_m), n_thread_idx]
    .store(m_local[ri_r]).end(ri_r)))

  m_ij = UOp.placeholder((TM,), dtypes.float, slot=9, addrspace=AddrSpace.REG)
  m_ij = m_ij.after(m_ij.after(n_tile).store(m_ij.const_like(-math.inf)))
  ri_g = UOp.range(TM, 280, AxisType.LOOP)
  rn_g = UOp.range(N_THREADS_PER_ROW, 281, AxisType.REDUCE)
  m_ij = m_ij.after(m_ij[ri_g].store(UOp(Ops.MAX, dtypes.float, (m_ij.after(ri_g, rn_g)[ri_g],
    red_lds[row_index(wave_m, ri_g // WMMA_ACC, ri_g % WMMA_ACC, lane_m), rn_g]))).end(rn_g, ri_g))

  # per-thread local exp and sum
  p_local = UOp.placeholder((TM,), dtypes.float, slot=10, addrspace=AddrSpace.REG)
  p_local = p_local.after(p_local.after(n_tile).store(p_local.const_like(0)))
  rp1 = UOp.range(TM, 290, AxisType.LOOP)
  rp2 = UOp.range(TN, 291, AxisType.REDUCE)
  p_local = p_local.after(p_local[rp1].store(p_local.after(rp1, rp2)[rp1] + ((S_reg[rp1, rp2] - m_ij[rp1]) * LOG2E).exp2()).end(rp2, rp1))

  # write P to P_lds (shaped)
  P_write = P_lds.reshape(WAVES_M, TM // WMMA_ACC, WMMA_ACC, LANES_PER_WAVE_M, WAVES_N, TN, LANES_PER_WAVE_N)
  P_write = P_write.permute((0, 4, 3, 6, 1, 2, 5)).reshape(THREADS_PER_BLOCK, TM, TN)
  P_store = P_write[tid].store(((S_reg - m_ij.reshape(TM, 1).expand(TM, TN)) * LOG2E).exp2().cast(dtypes.half))

  # write local sum to LDS, barrier, read global sum
  ri_s = UOp.range(TM, 300, AxisType.LOOP)
  sum_barrier = UOp.barrier(UOp.group(
    red_lds[row_index(wave_m, ri_s // WMMA_ACC, ri_s % WMMA_ACC, lane_m), n_thread_idx].store(p_local[ri_s]).end(ri_s),
    P_store))
  red_lds = red_lds.after(sum_barrier)
  P_lds = P_lds.after(sum_barrier)

  p_sum = UOp.placeholder((TM,), dtypes.float, slot=11, addrspace=AddrSpace.REG)
  p_sum = p_sum.after(p_sum.after(n_tile).store(p_sum.const_like(0)))
  ri_gs = UOp.range(TM, 310, AxisType.LOOP)
  rn_gs = UOp.range(N_THREADS_PER_ROW, 311, AxisType.REDUCE)
  p_sum = p_sum.after(p_sum[ri_gs].store(p_sum.after(ri_gs, rn_gs)[ri_gs] +
    red_lds[row_index(wave_m, ri_gs // WMMA_ACC, ri_gs % WMMA_ACC, lane_m), rn_gs]).end(rn_gs, ri_gs))

  # -- online softmax correction (shaped) --
  m_new = UOp(Ops.MAX, dtypes.float, (m_i, m_ij))
  alpha = ((m_i - m_new) * LOG2E).exp2()
  beta = ((m_ij - m_new) * LOG2E).exp2()
  correction = UOp.group(
    acc.store(alpha.reshape(TM, 1).expand(TM, TD) * acc),
    l_i.store(alpha * l_i + beta * p_sum),
    m_i.store(m_new),
  )
  acc = acc.after(correction)
  l_i = l_i.after(correction)
  m_i = m_i.after(correction)

  # load V into LDS
  KV_lds_v = KV_lds.after(UOp.barrier(KV_lds.reshape(-1, THREADS_PER_BLOCK)[:, tid].store(
    v[n_tile].reshape(-1, THREADS_PER_BLOCK)[:, tid])))

  # -- acc += P @ V via WMMA --
  k_pv = UOp.range(BLOCK_N // WMMA_K, 400, AxisType.REDUCE)
  tm2 = UOp.range(TM // WMMA_ACC, 401, AxisType.LOOP)
  tn2 = UOp.range(TD, 402, AxisType.LOOP)
  acc_frag = acc.reshape(TM // WMMA_ACC, WMMA_ACC, TD).permute(0, 2, 1)[tm2, tn2]
  p_frag = P_lds.reshape(WAVES_M, TM // WMMA_ACC, WMMA_M, BLOCK_N // WMMA_K, WMMA_K)[wave_m, tm2, lane_n, k_pv]
  v_frag = KV_lds_v.reshape(WAVES_N, TD, WMMA_N, BLOCK_N // WMMA_K, WMMA_K)[wave_n, tn2, lane_n, k_pv]
  pv = UOp(Ops.SHAPED_WMMA, dtypes.float, (p_frag, v_frag, acc_frag.after(k_pv)), arg=WMMA_ARG)

  # end KV tile loop
  n_tile_end = acc_frag.store(pv).end(tm2, tn2).end(k_pv).barrier().end(n_tile)
  acc = acc.after(n_tile_end)
  l_i = l_i.after(n_tile_end)
  m_i = m_i.after(n_tile_end)

  # normalize: acc /= l_i
  acc = acc.after(acc.store(acc * (1 / l_i).reshape(TM, 1).expand(TM, TD)))

  # store output
  o = o.reshape(WAVES_M, TM // WMMA_ACC, WMMA_ACC, LANES_PER_WAVE_M, WAVES_N, TD, LANES_PER_WAVE_N)
  o = o.permute((0, 4, 3, 6, 1, 2, 5)).reshape(THREADS_PER_BLOCK, TM, TD)
  return o[tid].store(acc).end(wave_m, wave_n, lane).end(block_m, block_bh).sink(arg=KernelInfo(opts_to_apply=()))

if __name__ == "__main__":
  q = Tensor.rand(B, H, N, D).cast(dtypes.half)
  k = Tensor.rand(B, H, N, D).cast(dtypes.half)
  v = Tensor.rand(B, H, N, D).cast(dtypes.half)
  o = Tensor.empty(B, H, N, D, dtype=dtypes.float)
  with Context(DEBUG=0): Tensor.realize(q, k, v)

  q_flat, k_flat, v_flat, o_flat = q.reshape(B*H, N, D), k.reshape(B*H, N, D), v.reshape(B*H, N, D), o.reshape(B*H, N, D)
  NUM_RUNS = getenv("CNT", 5)
  ets = []
  with Context(DEBUG=getenv("KDBG", 2)):
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
