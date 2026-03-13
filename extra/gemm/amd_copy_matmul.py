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

# 128x128 out, kx128, kx128 in
def block_128x128_gemm(a:UOp, b:UOp) -> UOp:
  tid = UOp.special(THREADS_PER_BLOCK, "lidx0")

  # define locals
  A_local = UOp.placeholder((BLOCK_K, BLOCK_M), dtypes.float, slot=0, addrspace=AddrSpace.LOCAL)
  B_local = UOp.placeholder((BLOCK_K, BLOCK_N), dtypes.float, slot=1, addrspace=AddrSpace.LOCAL)

  # open the main reduction range and copy in GLOBAL -> LOCAL
  a = a.reshape(K // BLOCK_K, BLOCK_K, BLOCK_M)
  b = b.reshape(K // BLOCK_K, BLOCK_K, BLOCK_N)
  k_tile_range = UOp.range(K // BLOCK_K, 0, AxisType.REDUCE)
  barrier = UOp.barrier(A_local.store(a[k_tile_range, :, :]), B_local.store(b[k_tile_range, :, :]))  # TODO: allow the : to be implicit
  A_local, B_local = A_local.after(barrier), B_local.after(barrier)

  consts = {"wpb_m":WAVES_PER_BLOCK_M, "lpw_m":LANES_PER_WAVE_M, "wpb_n":WAVES_PER_BLOCK_N, "lpw_n":LANES_PER_WAVE_N}

  # define accumulator (128x128) #, shard=tid)
  c_regs = UOp.placeholder((THREADS_PER_BLOCK, REG_TILES_PER_WAVE_M, TM, REG_TILES_PER_WAVE_N, TN), dtypes.float, slot=2, addrspace=AddrSpace.REG)
  c_regs = c_regs.rearrange("(wpb_m lpw_m wpb_n lpw_n) rt_m t_m rt_n t_n -> (wpb_m rt_m lpw_m t_m) (wpb_n rt_n lpw_n t_n)", **consts)

  # define registers (NOTE: the thread count is the device count for this multi, it's sharded across the THREADS_PER_BLOCK)
  A_col = UOp.placeholder((THREADS_PER_BLOCK, REG_TILES_PER_WAVE_M, TM), dtypes.float, slot=0, addrspace=AddrSpace.REG) #, shard=tid)
  B_row = UOp.placeholder((THREADS_PER_BLOCK, REG_TILES_PER_WAVE_N, TN), dtypes.float, slot=1, addrspace=AddrSpace.REG) #, shard=tid)

  # broadcast into registers from locals
  A_col = A_col.rearrange("(wpb_m lpw_m wpb_n lpw_n) rt t -> (wpb_n lpw_n) (wpb_m rt lpw_m t)", **consts).r(Ops.NOOP, (0,)).reshape(128)
  B_row = B_row.rearrange("(wpb_m lpw_m wpb_n lpw_n) rt t -> (wpb_m lpw_m) (wpb_n rt lpw_n t)", **consts).r(Ops.NOOP, (0,)).reshape(128)

  # LOCAL -> REGS + do FMA
  k = UOp.range(BLOCK_K, 3, AxisType.REDUCE)
  A_col = A_col.after(A_col.store(A_local[k, :]))
  B_row = B_row.after(B_row.store(B_local[k, :]))
  sink = c_regs.store(A_col.reshape(128, 1).expand(128, 128) * B_row.reshape(1, 128).expand(128, 128))  # TODO: allow broadcast

  # end the loops, return c_regs
  return c_regs.after(sink.end(k).barrier().end(k_tile_range))

def amd_copy_matmul(c:UOp, a:UOp, b:UOp) -> UOp:
  block_id_n = UOp.special(N // BLOCK_N, "gidx0")
  block_id_m = UOp.special(M // BLOCK_M, "gidx1")

  # index the output with the globals
  c = c.reshape(M // BLOCK_M, BLOCK_M, N // BLOCK_N, BLOCK_N)[block_id_m, :, block_id_n, :]
  a = a.T.reshape(K, M // BLOCK_M, BLOCK_M)[:, block_id_m, :]
  b = b.reshape(K, N // BLOCK_N, BLOCK_N)[:, block_id_n, :]

  # return 128x128 chunk of regs
  c_regs = block_128x128_gemm(a, b)
  return c.store(c_regs).sink(arg=KernelInfo(opts_to_apply=()))

if __name__ == "__main__":
  from amd_uop_matmul import eval_custom_matmul
  eval_custom_matmul(amd_copy_matmul)
