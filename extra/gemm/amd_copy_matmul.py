from tinygrad import UOp, getenv
from tinygrad.uop.ops import AxisType, KernelInfo, Ops
from tinygrad.dtype import AddrSpace, dtypes

N = getenv("N", 4096)
M = getenv("M", N)
K = getenv("K", N)

WARP_SIZE = 32
BLOCK_M, BLOCK_N = 128, 128
BLOCK_K = getenv("BK", 16)
assert N % BLOCK_N == 0 and M % BLOCK_M == 0 and K % BLOCK_K == 0

use_wmma = getenv("WMMA")

if use_wmma:
  UNROLL_M, UNROLL_N = 1, 1
  WMMA_M, WMMA_N, WMMA_K = 16, 16, 16
  WMMA_ACC = 8  # accumulator elements per thread in the M dimension
  WAVES_M, WAVES_N = 2, 2
  LANES_PER_WAVE_M, LANES_PER_WAVE_N = 2, 16
  TM = BLOCK_M // (WAVES_M * WMMA_M) * WMMA_ACC  # 32
  TN = BLOCK_N // (WAVES_N * WMMA_N)             # 4
else:
  UNROLL_M, UNROLL_N = 4, 4
  WAVES_M, WAVES_N = 4, 1
  LANES_PER_WAVE_M, LANES_PER_WAVE_N = 4, 8
  TM = BLOCK_M // (WAVES_M * LANES_PER_WAVE_M)  # 8
  TN = BLOCK_N // (WAVES_N * LANES_PER_WAVE_N)  # 16

# WARP_SIZE * total waves
THREADS_PER_BLOCK = WARP_SIZE * WAVES_M * WAVES_N

def block_128x128_gemm(c:UOp, a:UOp, b:UOp) -> UOp:
  wave_m = UOp.range(WAVES_M, 100, AxisType.LOCAL)
  wave_n = UOp.range(WAVES_N, 101, AxisType.LOCAL)
  lane = UOp.range(WARP_SIZE, -1, AxisType.WARP)
  tid = (wave_m * WAVES_N + wave_n) * WARP_SIZE + lane

  # -- GLOBAL -> LOCAL --
  # wmma: spatial outer, k inner (k contiguous for vectorized WMMA tile loads)
  # gemm: k outer, spatial inner
  A_local = UOp.placeholder((BLOCK_M, BLOCK_K) if use_wmma else (BLOCK_K, BLOCK_M), a.dtype.base, slot=0, addrspace=AddrSpace.LOCAL)
  B_local = UOp.placeholder((BLOCK_N, BLOCK_K) if use_wmma else (BLOCK_K, BLOCK_N), b.dtype.base, slot=1, addrspace=AddrSpace.LOCAL)

  a = a.reshape(K // BLOCK_K, BLOCK_K, BLOCK_M)
  b = b.reshape(K // BLOCK_K, BLOCK_K, BLOCK_N)
  k_tile = UOp.range(K // BLOCK_K, 3, AxisType.REDUCE)

  # copy with transpose for wmma (input is k×spatial, LDS is spatial×k)
  A_copy = A_local.permute((1,0)) if use_wmma else A_local
  B_copy = B_local.permute((1,0)) if use_wmma else B_local
  A_store = A_copy.reshape(-1, THREADS_PER_BLOCK)[:, tid].store(a[k_tile].reshape(-1, THREADS_PER_BLOCK)[:, tid])
  B_store = B_copy.reshape(-1, THREADS_PER_BLOCK)[:, tid].store(b[k_tile].reshape(-1, THREADS_PER_BLOCK)[:, tid])
  barrier = UOp.barrier(A_store, B_store)
  A_local, B_local = A_local.after(barrier), B_local.after(barrier)

  # -- COMPUTE --
  lane_m, lane_n = lane // LANES_PER_WAVE_N, lane % LANES_PER_WAVE_N

  # accumulator (unified: both paths use (TM, TN) with scalar dtypes.float)
  acc = UOp.placeholder((TM, TN), dtypes.float, slot=2, addrspace=AddrSpace.REG)
  acc = acc.after(acc.store(UOp.const(dtypes.float, 0).reshape((1,)*len(acc.shape)).expand(acc.shape)))

  if use_wmma:
    A_tiles = A_local.reshape(WAVES_M, TM // WMMA_ACC, WMMA_M, BLOCK_K // WMMA_K, WMMA_K)
    B_tiles = B_local.reshape(WAVES_N, TN, WMMA_N, BLOCK_K // WMMA_K, WMMA_K)

    k = UOp.range(BLOCK_K // WMMA_K, 4, AxisType.REDUCE)
    tile_m = UOp.range(TM // WMMA_ACC, 5, AxisType.LOOP)
    tile_n = UOp.range(TN, 6, AxisType.LOOP)

    k_upcast_a = UOp.range(WMMA_K, 301, axis_type=AxisType.UPCAST)
    a_frag = A_tiles[wave_m, tile_m, lane_n, k, k_upcast_a].contract(k_upcast_a)
    k_upcast_b = UOp.range(WMMA_K, 311, axis_type=AxisType.UPCAST)
    b_frag = B_tiles[wave_n, tile_n, lane_n, k, k_upcast_b].contract(k_upcast_b)

    acc_ref = acc.reshape(TM // WMMA_ACC, WMMA_ACC, TN).after(k_tile, k, tile_m, tile_n)
    acc_load = UOp(Ops.VECTORIZE, dtypes.float.vec(WMMA_ACC), tuple(acc_ref[tile_m, e, tile_n] for e in range(WMMA_ACC)))
    wmma_arg = ('WMMA_16_16_16_half_float', (16, 16, 16), dtypes.half, dtypes.float, 'AMD', 32,
                (((301, 16),), ((311, 16),), ((302, WMMA_ACC),)), ())
    out = UOp(Ops.WMMA, dtypes.float.vec(WMMA_ACC), (a_frag, b_frag, acc_load), arg=wmma_arg)

    acc_wr = acc.reshape(TM // WMMA_ACC, WMMA_ACC, TN)
    acc_store = UOp.group(*[acc_wr[tile_m, e, tile_n].store(out.gep(e)) for e in range(WMMA_ACC)]).end(tile_m, tile_n)
  else:
    # registers for LOCAL -> REG
    a_frag = UOp.placeholder((TM//UNROLL_M, UNROLL_M), dtypes.float, slot=0, addrspace=AddrSpace.REG)
    b_frag = UOp.placeholder((TN//UNROLL_N, UNROLL_N), dtypes.float, slot=1, addrspace=AddrSpace.REG)

    k = UOp.range(BLOCK_K, 4, AxisType.REDUCE)
    a_frag = a_frag.after(a_frag.store(A_local[k].reshape(WAVES_M, TM//UNROLL_M, LANES_PER_WAVE_M, UNROLL_M)[wave_m, :, lane_m, :]))
    b_frag = b_frag.after(b_frag.store(B_local[k].reshape(WAVES_N, TN//UNROLL_N, LANES_PER_WAVE_N, UNROLL_N)[wave_n, :, lane_n, :]))

    # FMA
    a_frag = a_frag.reshape(TM, 1).expand(TM, TN)
    b_frag = b_frag.reshape(1, TN).expand(TM, TN)
    acc_store = acc.store(acc.after(k) + (a_frag * b_frag))

  # store accumulator and loop
  acc = acc.after(acc_store.end(k).barrier().end(k_tile))

  # store accumulator to output (unified)
  c = c.reshape(WAVES_M, TM//UNROLL_M, LANES_PER_WAVE_M, UNROLL_M,
                WAVES_N, TN//UNROLL_N, LANES_PER_WAVE_N, UNROLL_N)
  c = c.permute((0,4,2,6, 1,3,5,7))
  return c.reshape(THREADS_PER_BLOCK, TM, TN)[tid].store(acc).end(wave_m, wave_n, lane)

def amd_copy_matmul(c:UOp, a:UOp, b:UOp) -> UOp:
  block_id_m = UOp.range(M // BLOCK_M, 0, AxisType.GLOBAL)
  block_id_n = UOp.range(N // BLOCK_N, 1, AxisType.GLOBAL)
  c = c.reshape(M // BLOCK_M, BLOCK_M, N // BLOCK_N, BLOCK_N)[block_id_m, :, block_id_n, :]
  a = a.T.reshape(K, M // BLOCK_M, BLOCK_M)[:, block_id_m, :]
  b = b.reshape(K, N // BLOCK_N, BLOCK_N)[:, block_id_n, :]
  return block_128x128_gemm(c, a, b).end(block_id_n, block_id_m).sink(arg=KernelInfo(opts_to_apply=()))

if __name__ == "__main__":
  from amd_uop_matmul import eval_custom_matmul
  eval_custom_matmul(amd_copy_matmul, dtypes.half if use_wmma else dtypes.float)
