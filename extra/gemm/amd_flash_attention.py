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

WMMA_ARG = ((WMMA_M, WMMA_N, WMMA_K), 'AMD', 32)
SCALE = 1.0 / math.sqrt(D)
LOG2E = math.log2(math.e)

def row_index(wave_m, ri, re, lane_m):
  """Map per-thread (tile, acc_element) indices to global row within BLOCK_M"""
  return wave_m * UOp.const(dtypes.weakint, BLOCK_M // WAVES_M) + ri * UOp.const(dtypes.weakint, WMMA_M) + \
         re * UOp.const(dtypes.weakint, LANES_PER_WAVE_M) + lane_m

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

  # load Q into LDS
  Q_lds = UOp.placeholder((BLOCK_M, D), dtypes.half, slot=0, addrspace=AddrSpace.LOCAL)
  Q_lds = Q_lds.after(UOp.barrier(Q_lds.reshape(-1, THREADS_PER_BLOCK)[:, tid].store(
    q.reshape(-1, THREADS_PER_BLOCK)[:, tid])))

  KV_lds = UOp.placeholder((BLOCK_N, D), dtypes.half, slot=1, addrspace=AddrSpace.LOCAL)

  # register state: output accumulator, row max, row sum
  acc = UOp.placeholder((TM, TD), dtypes.float, slot=2, addrspace=AddrSpace.REG)
  m_i = UOp.placeholder((TM,), dtypes.float, slot=3, addrspace=AddrSpace.REG)
  l_i = UOp.placeholder((TM,), dtypes.float, slot=4, addrspace=AddrSpace.REG)
  acc = acc.after(acc.store(UOp.const(dtypes.float, 0).reshape(1, 1).expand(TM, TD)))
  m_i = m_i.after(m_i.store(UOp.const(dtypes.float, -math.inf).reshape(1).expand(TM)))
  l_i = l_i.after(l_i.store(UOp.const(dtypes.float, 0).reshape(1).expand(TM)))

  # shared LDS for S scores and P softmax output
  S_lds = UOp.placeholder((BLOCK_M, BLOCK_N), dtypes.float, slot=5, addrspace=AddrSpace.LOCAL)
  P_lds = UOp.placeholder((BLOCK_M, BLOCK_N), dtypes.half, slot=6, addrspace=AddrSpace.LOCAL)

  # ====== KV tile loop ======
  n_tile = UOp.range(N // BLOCK_N, 100, AxisType.REDUCE)

  # load K into LDS
  KV_lds_k = KV_lds.after(UOp.barrier(KV_lds.reshape(-1, THREADS_PER_BLOCK)[:, tid].store(
    k[n_tile].reshape(-1, THREADS_PER_BLOCK)[:, tid])))

  # -- S = Q @ K^T via WMMA into registers (re-init each n_tile) --
  S_reg = UOp.placeholder((TM, TN), dtypes.float, slot=7, addrspace=AddrSpace.REG)
  S_reg = S_reg.after(S_reg.after(n_tile).store(UOp.const(dtypes.float, 0).reshape(1, 1).expand(TM, TN)))
  k_qk = UOp.range(D // WMMA_K, 101, AxisType.REDUCE)
  tm1 = UOp.range(TM // WMMA_ACC, 200, AxisType.LOOP)
  tn1 = UOp.range(TN, 201, AxisType.LOOP)
  S_frag = S_reg.reshape(TM // WMMA_ACC, WMMA_ACC, TN).permute(0, 2, 1)[tm1, tn1]
  q_frag = Q_lds.reshape(WAVES_M, TM // WMMA_ACC, WMMA_M, D // WMMA_K, WMMA_K)[wave_m, tm1, lane_n, k_qk]
  k_frag = KV_lds_k.reshape(WAVES_N, TN, WMMA_N, D // WMMA_K, WMMA_K)[wave_n, tn1, lane_n, k_qk]
  qk = UOp(Ops.SHAPED_WMMA, dtypes.float, (q_frag, k_frag, S_frag.after(k_qk)), arg=WMMA_ARG)
  S_reg = S_reg.after(S_frag.store(qk).end(tm1, tn1).end(k_qk).barrier())

  # write S*scale from registers to S_lds
  S_write = S_lds.reshape(WAVES_M, TM // WMMA_ACC, WMMA_ACC, LANES_PER_WAVE_M, WAVES_N, TN, LANES_PER_WAVE_N)
  S_write = S_write.permute((0, 4, 3, 6, 1, 2, 5)).reshape(THREADS_PER_BLOCK, TM, TN)
  rs1 = UOp.range(TM, 250, AxisType.LOOP)
  rs2 = UOp.range(TN, 251, AxisType.LOOP)
  S_lds = S_lds.after(UOp.barrier(S_write[tid, rs1, rs2].store(S_reg[rs1, rs2] * UOp.const(dtypes.float, SCALE)).end(rs1, rs2)))

  # -- softmax from LDS --
  # row max (re-init each n_tile iteration)
  m_ij = UOp.placeholder((TM,), dtypes.float, slot=8, addrspace=AddrSpace.REG)
  m_ij = m_ij.after(m_ij.after(n_tile).store(UOp.const(dtypes.float, -math.inf).reshape(1).expand(TM)))
  ri = UOp.range(TM // WMMA_ACC, 300, AxisType.LOOP)
  re = UOp.range(WMMA_ACC, 301, AxisType.LOOP)
  rj = UOp.range(BLOCK_N, 302, AxisType.REDUCE)
  flat_i = ri * UOp.const(dtypes.weakint, WMMA_ACC) + re
  m_ij = m_ij.after(m_ij[flat_i].store(UOp(Ops.MAX, dtypes.float, (m_ij.after(ri, re, rj)[flat_i], S_lds[row_index(wave_m, ri, re, lane_m), rj])))
    .end(rj).end(re, ri))

  # rowsum of exp(S - m_ij) (re-init each n_tile)
  p_sum = UOp.placeholder((TM,), dtypes.float, slot=9, addrspace=AddrSpace.REG)
  p_sum = p_sum.after(p_sum.after(n_tile).store(UOp.const(dtypes.float, 0).reshape(1).expand(TM)))
  ri2 = UOp.range(TM // WMMA_ACC, 310, AxisType.LOOP)
  re2 = UOp.range(WMMA_ACC, 311, AxisType.LOOP)
  rj2 = UOp.range(BLOCK_N, 312, AxisType.REDUCE)
  flat_i2 = ri2 * UOp.const(dtypes.weakint, WMMA_ACC) + re2
  exp_val = ((S_lds[row_index(wave_m, ri2, re2, lane_m), rj2] - m_ij[flat_i2]) * UOp.const(dtypes.float, LOG2E)).exp2()
  p_sum = p_sum.after(p_sum[flat_i2].store(p_sum.after(ri2, re2, rj2)[flat_i2] + exp_val).end(rj2).end(re2, ri2))

  # -- online softmax correction: compute m_new, alpha, beta BEFORE writing P --
  ri4 = UOp.range(TM, 330, AxisType.LOOP)
  m_new_val = UOp(Ops.MAX, dtypes.float, (m_i[ri4], m_ij[ri4]))
  alpha_val = ((m_i[ri4] - m_new_val) * UOp.const(dtypes.float, LOG2E)).exp2()
  beta_val = ((m_ij[ri4] - m_new_val) * UOp.const(dtypes.float, LOG2E)).exp2()
  rj4 = UOp.range(TD, 331, AxisType.LOOP)
  correction = UOp.group(
    acc[ri4, rj4].store(alpha_val * acc[ri4, rj4]).end(rj4),
    l_i[ri4].store(alpha_val * l_i[ri4] + beta_val * p_sum[ri4]),
    m_i[ri4].store(m_new_val),
  ).end(ri4)
  acc = acc.after(correction)
  l_i = l_i.after(correction)
  m_i = m_i.after(correction)

  # write P = beta * exp(S - m_ij) = exp(S - m_new) to P_lds as fp16
  ri3 = UOp.range(TM // WMMA_ACC, 320, AxisType.LOOP)
  re3 = UOp.range(WMMA_ACC, 321, AxisType.LOOP)
  rj3 = UOp.range(BLOCK_N, 322, AxisType.LOOP)
  flat_i3 = ri3 * UOp.const(dtypes.weakint, WMMA_ACC) + re3
  # exp(S*scale - m_new) = exp2((S*scale - m_new) * log2e)
  exp_val3 = ((S_lds[row_index(wave_m, ri3, re3, lane_m), rj3] - m_i[flat_i3]) * UOp.const(dtypes.float, LOG2E)).exp2()
  P_lds = P_lds.after(UOp.barrier(P_lds[row_index(wave_m, ri3, re3, lane_m), rj3].store(exp_val3.cast(dtypes.half)).end(rj3).end(re3, ri3)))

  # load V into LDS (reuse KV_lds)
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
  rn1 = UOp.range(TM, 500, AxisType.LOOP)
  rn2 = UOp.range(TD, 501, AxisType.LOOP)
  acc = acc.after(acc[rn1, rn2].store(acc[rn1, rn2] / l_i[rn1]).end(rn1, rn2))

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
