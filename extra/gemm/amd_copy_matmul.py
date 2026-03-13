from tinygrad import UOp, getenv
from tinygrad.uop.ops import AxisType, KernelInfo, Ops
from tinygrad.dtype import AddrSpace, dtypes

N = getenv("N", 4096)
M = getenv("M", N)
K = getenv("K", N)

WARP_SIZE = 32
BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 8
TM, TN = 4, 4
LANES_PER_WAVE_M, LANES_PER_WAVE_N = 4, 8
assert N % BLOCK_N == 0 and M % BLOCK_M == 0 and K % BLOCK_K == 0

is_kernel5 = getenv("K5", 0)
THREADS_PER_BLOCK = 128 if is_kernel5 else 256
WAVES_PER_BLOCK_N = 1 if is_kernel5 else 2
WAVES_PER_BLOCK_M = THREADS_PER_BLOCK // WARP_SIZE // WAVES_PER_BLOCK_N
REG_TILES_PER_WAVE_N = BLOCK_N // (WAVES_PER_BLOCK_N * LANES_PER_WAVE_N * TN)
REG_TILES_PER_WAVE_M = BLOCK_M // (WAVES_PER_BLOCK_M * LANES_PER_WAVE_M * TM)

assert WAVES_PER_BLOCK_M*REG_TILES_PER_WAVE_M*LANES_PER_WAVE_M*TM == BLOCK_M, "M reshape is wrong"
assert WAVES_PER_BLOCK_N*REG_TILES_PER_WAVE_N*LANES_PER_WAVE_N*TN == BLOCK_N, "N reshape is wrong"

consts = {"wpb_m":WAVES_PER_BLOCK_M, "lpw_m":LANES_PER_WAVE_M, "rt_m":REG_TILES_PER_WAVE_M, "t_m": TM,
          "wpb_n":WAVES_PER_BLOCK_N, "lpw_n":LANES_PER_WAVE_N, "rt_n":REG_TILES_PER_WAVE_N, "t_n": TN}

# 128x128 out, kx128, kx128 in
def block_128x128_gemm(c:UOp, a:UOp, b:UOp) -> UOp:
  tid = UOp.range(THREADS_PER_BLOCK, 2, AxisType.LOCAL)

  #tid = UOp.special(THREADS_PER_BLOCK, "lidx0")
  warp, lane = tid // WARP_SIZE, tid % WARP_SIZE
  wave_n, wave_m = warp % WAVES_PER_BLOCK_N, warp // WAVES_PER_BLOCK_N
  lane_n, lane_m = lane % LANES_PER_WAVE_N, lane // LANES_PER_WAVE_N

  # define locals
  A_local = UOp.placeholder((BLOCK_K, BLOCK_M), dtypes.float, slot=0, addrspace=AddrSpace.LOCAL)
  B_local = UOp.placeholder((BLOCK_K, BLOCK_N), dtypes.float, slot=1, addrspace=AddrSpace.LOCAL)

  # open the main reduction range and copy in GLOBAL -> LOCAL
  a = a.reshape(K // BLOCK_K, BLOCK_K, BLOCK_M)
  b = b.reshape(K // BLOCK_K, BLOCK_K, BLOCK_N)
  k_tile_range = UOp.range(K // BLOCK_K, 3, AxisType.REDUCE)
  barrier = UOp.barrier(A_local.store(a[k_tile_range]), B_local.store(b[k_tile_range]))  # TODO: allow the : to be implicit
  A_local, B_local = A_local.after(barrier), B_local.after(barrier)

  # define accumulator (128x128), but broadcast across tid
  c_regs = UOp.placeholder((REG_TILES_PER_WAVE_M*TM, REG_TILES_PER_WAVE_N*TN), dtypes.float, slot=2, addrspace=AddrSpace.REG)
  c_regs = c_regs.after(c_regs.store(UOp.const(dtypes.float, 0).reshape((1,)*len(c_regs.shape)).expand(c_regs.shape)))

  # define registers (NOTE: the thread count is the device count for this multi, it's sharded across the THREADS_PER_BLOCK)
  A_col = UOp.placeholder((REG_TILES_PER_WAVE_M, TM), dtypes.float, slot=0, addrspace=AddrSpace.REG)
  B_row = UOp.placeholder((REG_TILES_PER_WAVE_N, TN), dtypes.float, slot=1, addrspace=AddrSpace.REG)

  # LOCAL -> REGS
  k = UOp.range(BLOCK_K, 4, AxisType.REDUCE)
  A_col = A_col.after(A_col.store(A_local[k].reshape(WAVES_PER_BLOCK_M, REG_TILES_PER_WAVE_M, LANES_PER_WAVE_M, TM)[wave_m, :, lane_m, :]))
  B_row = B_row.after(B_row.store(B_local[k].reshape(WAVES_PER_BLOCK_N, REG_TILES_PER_WAVE_N, LANES_PER_WAVE_N, TN)[wave_n, :, lane_n, :]))

  # do FMA
  A_col = A_col.reshape(REG_TILES_PER_WAVE_M*TM, 1).expand(REG_TILES_PER_WAVE_M*TM, REG_TILES_PER_WAVE_N*TN)
  B_row = B_row.reshape(1, REG_TILES_PER_WAVE_N*TN).expand(REG_TILES_PER_WAVE_M*TM, REG_TILES_PER_WAVE_N*TN)
  c_regs = c_regs.after(c_regs.store(c_regs.after(k) + (A_col * B_row)).end(k).barrier().end(k_tile_range))

  # store back to c
  c_store = c.rearrange("(wpb_m rt_m lpw_m t_m) (wpb_n rt_n lpw_n t_n) -> (wpb_m wpb_n lpw_m lpw_n) (rt_m t_m) (rt_n t_n)", **consts)
  return c_store[tid].store(c_regs).end(tid)

def amd_copy_matmul(c:UOp, a:UOp, b:UOp) -> UOp:
  block_id_n = UOp.range(N // BLOCK_N, 0, AxisType.GLOBAL)
  block_id_m = UOp.range(M // BLOCK_M, 1, AxisType.GLOBAL)

  # index the output with the globals
  c = c.reshape(M // BLOCK_M, BLOCK_M, N // BLOCK_N, BLOCK_N)[block_id_m, :, block_id_n, :]
  a = a.T.reshape(K, M // BLOCK_M, BLOCK_M)[:, block_id_m, :]
  b = b.reshape(K, N // BLOCK_N, BLOCK_N)[:, block_id_n, :]
  return block_128x128_gemm(c, a, b).end(block_id_n, block_id_m).sink(arg=KernelInfo(opts_to_apply=()))

if __name__ == "__main__":
  from amd_uop_matmul import eval_custom_matmul
  eval_custom_matmul(amd_copy_matmul)
