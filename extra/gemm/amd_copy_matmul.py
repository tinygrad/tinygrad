from tinygrad import UOp, getenv
from tinygrad.uop.ops import AxisType, KernelInfo, Ops
from tinygrad.dtype import AddrSpace, dtypes

N = getenv("N", 4096)
M = getenv("M", N)
K = getenv("K", N)

WARP_SIZE = 32
BLOCK_M, BLOCK_N = 128, 128

use_wmma = getenv("WMMA", 0)

if use_wmma:
  # RDNA3 WMMA: v_wmma_f32_16x16x16_f16 (fp16 in, fp32 acc)
  WMMA_M, WMMA_N, WMMA_K = 16, 16, 16
  BLOCK_K = getenv("BK", 16)
  THREADS_PER_BLOCK = 128
  WAVES_M, WAVES_N = 2, 2
  TILES_PER_WAVE_M = BLOCK_M // (WAVES_M * WMMA_M)  # 4
  TILES_PER_WAVE_N = BLOCK_N // (WAVES_N * WMMA_N)   # 4
  assert WAVES_M * TILES_PER_WAVE_M * WMMA_M == BLOCK_M
  assert WAVES_N * TILES_PER_WAVE_N * WMMA_N == BLOCK_N
else:
  BLOCK_K = 8
  TM, TN = 4, 4
  LANES_PER_WAVE_M, LANES_PER_WAVE_N = 4, 8
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

assert N % BLOCK_N == 0 and M % BLOCK_M == 0 and K % BLOCK_K == 0

def block_128x128_wmma(c:UOp, a:UOp, b:UOp) -> UOp:
  tid = UOp.range(THREADS_PER_BLOCK, 2, AxisType.WARP)
  warp, lane = tid // WARP_SIZE, tid % WARP_SIZE
  wave_m, wave_n = warp // WAVES_N, warp % WAVES_N
  lane_row, lane_half = lane % WMMA_M, lane // WMMA_M

  # LDS: spatial outer, k inner -- k is contiguous so loads from LDS vectorize
  A_local = UOp.placeholder((BLOCK_M, BLOCK_K), dtypes.half, slot=0, addrspace=AddrSpace.LOCAL)
  B_local = UOp.placeholder((BLOCK_N, BLOCK_K), dtypes.half, slot=1, addrspace=AddrSpace.LOCAL)

  a = a.reshape(K // BLOCK_K, BLOCK_K, BLOCK_M)
  b = b.reshape(K // BLOCK_K, BLOCK_K, BLOCK_N)
  k_tile = UOp.range(K // BLOCK_K, 3, AxisType.REDUCE)
  copy_k = UOp.range(BLOCK_K, 500, AxisType.UPCAST)
  A_store = A_local[tid, copy_k].store(a[k_tile][copy_k, tid]).end(copy_k)
  B_store = B_local[tid, copy_k].store(b[k_tile][copy_k, tid]).end(copy_k)
  barrier = UOp.barrier(A_store, B_store)
  A_local, B_local = A_local.after(barrier), B_local.after(barrier)

  # accumulator
  acc = UOp.placeholder((TILES_PER_WAVE_M, TILES_PER_WAVE_N), dtypes.float.vec(8), slot=2, addrspace=AddrSpace.REG)
  zi = UOp.range(TILES_PER_WAVE_M, 200); zj = UOp.range(TILES_PER_WAVE_N, 201)
  acc = acc[zi, zj].set(UOp.const(dtypes.float.vec(8), 0.0), end=(zi, zj))

  A_tiles = A_local.reshape(WAVES_M, TILES_PER_WAVE_M, WMMA_M, BLOCK_K // WMMA_K, WMMA_K)
  B_tiles = B_local.reshape(WAVES_N, TILES_PER_WAVE_N, WMMA_N, BLOCK_K // WMMA_K, WMMA_K)

  k = UOp.range(BLOCK_K // WMMA_K, 4, AxisType.REDUCE)
  tile_m = UOp.range(TILES_PER_WAVE_M, 5, AxisType.LOOP)
  tile_n = UOp.range(TILES_PER_WAVE_N, 6, AxisType.LOOP)

  k_upcast_a = UOp.range(WMMA_K, 301, axis_type=AxisType.UPCAST)
  a_frag = A_tiles[wave_m, tile_m, lane_row, k, k_upcast_a].contract(k_upcast_a)
  k_upcast_b = UOp.range(WMMA_K, 311, axis_type=AxisType.UPCAST)
  b_frag = B_tiles[wave_n, tile_n, lane_row, k, k_upcast_b].contract(k_upcast_b)

  acc_load = acc.after(k_tile, k, tile_m, tile_n)[tile_m, tile_n]
  wmma_arg = ('WMMA_16_16_16_half_float', (16, 16, 16), dtypes.half, dtypes.float, 'AMD', 32,
              (((301, 16),), ((311, 16),), ()), ())
  out = UOp(Ops.WMMA, dtypes.float.vec(8), (a_frag, b_frag, acc_load), arg=wmma_arg)
  acc_store = acc[tile_m, tile_n].store(out)
  acc = acc.after(acc_store.end(tile_m, tile_n, k).barrier().end(k_tile))

  # store accumulator to output
  c = c.reshape(WAVES_M, TILES_PER_WAVE_M, WMMA_M, WAVES_N, TILES_PER_WAVE_N, WMMA_N)
  st_m = UOp.range(TILES_PER_WAVE_M, 9, AxisType.LOOP)
  st_n = UOp.range(TILES_PER_WAVE_N, 10, AxisType.LOOP)
  stores = [c[wave_m, st_m, e*2 + lane_half, wave_n, st_n, lane_row].store(acc[st_m, st_n].gep(e)) for e in range(8)]
  return UOp.group(*stores).end(st_m, st_n, tid)

# 128x128 out, kx128, kx128 in
def block_128x128_gemm(c:UOp, a:UOp, b:UOp) -> UOp:
  tid = UOp.range(THREADS_PER_BLOCK, 2, AxisType.LOCAL)
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
  A_store = A_local.reshape(-1, THREADS_PER_BLOCK)[:, tid].store(a[k_tile_range].reshape(-1, THREADS_PER_BLOCK)[:, tid])
  B_store = B_local.reshape(-1, THREADS_PER_BLOCK)[:, tid].store(b[k_tile_range].reshape(-1, THREADS_PER_BLOCK)[:, tid])
  barrier = UOp.barrier(A_store, B_store)
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
  block_id_m = UOp.range(M // BLOCK_M, 0, AxisType.GLOBAL)
  block_id_n = UOp.range(N // BLOCK_N, 1, AxisType.GLOBAL)
  c = c.reshape(M // BLOCK_M, BLOCK_M, N // BLOCK_N, BLOCK_N)[block_id_m, :, block_id_n, :]
  a = a.T.reshape(K, M // BLOCK_M, BLOCK_M)[:, block_id_m, :]
  b = b.reshape(K, N // BLOCK_N, BLOCK_N)[:, block_id_n, :]
  block_fn = block_128x128_wmma if use_wmma else block_128x128_gemm
  return block_fn(c, a, b).end(block_id_n, block_id_m).sink(arg=KernelInfo(opts_to_apply=()))

if __name__ == "__main__":
  from amd_uop_matmul import eval_custom_matmul
  eval_custom_matmul(amd_copy_matmul, dtypes.half if use_wmma else dtypes.float)
