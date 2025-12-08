import numpy as np
from tinygrad import Tensor, Device, Context, GlobalCounters, dtypes
from tinygrad.uop.ops import UOp, KernelInfo, AxisType
from tinygrad.engine.realize import ExecItem, get_runner
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import getenv
from extra.gemm.amd_uop_matmul import copy, rngs_for_shape

N = getenv("N", 4096)
M = K = N
run_count = getenv("CNT", 5)

WARP_SIZE = 32

BLOCK_N = 256   # columns of C per block
BLOCK_M = 128   # rows of C per block
BLOCK_K = 64    # K-slice per block iteration

TN = 4  # columns per thread
TM = 4  # rows per thread

THREADS_PER_BLOCK = 256  # 8 warps
assert THREADS_PER_BLOCK % BLOCK_N == 0
assert THREADS_PER_BLOCK % BLOCK_K == 0
assert (BLOCK_N * BLOCK_K) % THREADS_PER_BLOCK == 0
assert (BLOCK_M * BLOCK_K) % THREADS_PER_BLOCK == 0

WARPS_PER_BLOCK = THREADS_PER_BLOCK // WARP_SIZE
WARP_TILE_N = 64
WARP_TILE_M = BLOCK_N * BLOCK_M // WARPS_PER_BLOCK // WARP_TILE_N
assert BLOCK_N % WARP_TILE_N == 0
assert BLOCK_M % WARP_TILE_M == 0
WARPS_IN_BLOCK_X = BLOCK_N // WARP_TILE_N
WARPS_IN_BLOCK_Y = BLOCK_M // WARP_TILE_M
assert WARPS_IN_BLOCK_X * WARPS_IN_BLOCK_Y == WARPS_PER_BLOCK

LANES_PER_WARP_X = 8
LANES_PER_WARP_Y = 4
ITERS_PER_WARP_N = WARP_TILE_N // (LANES_PER_WARP_X * TN)
ITERS_PER_WARP_M = WARP_TILE_M // (LANES_PER_WARP_Y * TM)
assert WARP_TILE_N % (LANES_PER_WARP_X * TN) == 0
assert WARP_TILE_M % (LANES_PER_WARP_Y * TM) == 0

def hand_spec():
  # ---------------------------
  # block indices & placeholders
  # ---------------------------
  blockIdx_x = UOp.special(N // BLOCK_N, "gidx0")
  blockIdx_y = UOp.special(M // BLOCK_M, "gidx1")

  a = UOp.placeholder((N, N), dtypes.half, slot=1)
  b = UOp.placeholder((N, N), dtypes.half, slot=2)
  c = UOp.placeholder((N, N), dtypes.half, slot=0)

  c = c.reshape(M // BLOCK_M, BLOCK_M, N // BLOCK_N, BLOCK_N)[blockIdx_y, :, blockIdx_x, :]

  k_tile_range = UOp.range(N // BLOCK_K, 0, AxisType.REDUCE)
  a = a.reshape(M // BLOCK_M, BLOCK_M, N // BLOCK_K, BLOCK_K)[blockIdx_y, :, k_tile_range, :]
  b = b.reshape(N // BLOCK_K, BLOCK_K, N // BLOCK_N, BLOCK_N)[k_tile_range, :, blockIdx_x, :]

  del blockIdx_y, blockIdx_x

  # ---------------------------
  # GLOBAL -> LOCAL (As, Bs)
  # ---------------------------
  tid = UOp.special(THREADS_PER_BLOCK, "lidx0")

  As = UOp.placeholder((BLOCK_K, BLOCK_M), dtypes.half, slot=0, addrspace=AddrSpace.LOCAL)
  Bs = UOp.placeholder((BLOCK_K, BLOCK_N), dtypes.half, slot=1, addrspace=AddrSpace.LOCAL)

  As_store = copy(As.permute((1, 0)).reshape(-1, THREADS_PER_BLOCK)[:, tid],
                  a.reshape(-1, THREADS_PER_BLOCK)[:, tid], rng=100)
  Bs_store = copy(Bs.reshape(-1, THREADS_PER_BLOCK)[:, tid],
                  b.reshape(-1, THREADS_PER_BLOCK)[:, tid], rng=200)

  barrier = UOp.barrier(As_store, Bs_store)
  As, Bs = As.after(barrier), Bs.after(barrier)

  k = UOp.range(BLOCK_K, 3, AxisType.REDUCE)

  # ---------------------------
  # LOCAL -> REG (per-warp tiles)
  # ---------------------------
  warpIdx = (tid // WARP_SIZE) % WARPS_IN_BLOCK_X
  warpIdy = (tid // WARP_SIZE) // WARPS_IN_BLOCK_X
  assert warpIdy.vmax+1 == WARPS_IN_BLOCK_Y

  laneIdx = (tid % WARP_SIZE) % LANES_PER_WARP_X
  laneIdy = (tid % WARP_SIZE) // LANES_PER_WARP_X
  assert laneIdy.vmax+1 == LANES_PER_WARP_Y

  A_frag = UOp.placeholder((ITERS_PER_WARP_M, TM), dtypes.half, slot=0, addrspace=AddrSpace.REG)
  A_frag = copy(A_frag, As[k, :].reshape(WARPS_IN_BLOCK_Y, ITERS_PER_WARP_M, LANES_PER_WARP_Y, TM)[warpIdy, :, laneIdy, :],
               300, set=True, upcast=True)

  B_frag = UOp.placeholder((ITERS_PER_WARP_N, TN), dtypes.half, slot=1, addrspace=AddrSpace.REG)
  B_frag = copy(B_frag, Bs[k, :].reshape(WARPS_IN_BLOCK_X, ITERS_PER_WARP_N, LANES_PER_WARP_X, TN)[warpIdx, :, laneIdx, :],
               400, set=True, upcast=True)

  # ---------------------------
  # FMA: c_regs (fp32) += A_frag (half) * B_frag (half)
  # ---------------------------
  c_regs = UOp.placeholder((ITERS_PER_WARP_M, TM, ITERS_PER_WARP_N, TN), dtypes.float, slot=2, addrspace=AddrSpace.REG)
  i = UOp.range(c_regs.size, 16)
  c_regs = c_regs.after(c_regs.flatten()[i].store(0.0).end(i))

  iterWarpM, yt, iterWarpN, xt = rngs = rngs_for_shape(c_regs.shape, 500)
  sink = c_regs[*rngs].store(c_regs.after(k)[*rngs] + A_frag[iterWarpM, yt].cast(dtypes.float) * B_frag[iterWarpN, xt].cast(dtypes.float)).end(iterWarpM, iterWarpN, yt, xt)

  sink = sink.end(k).barrier().end(k_tile_range)

  # ---------------------------
  # REG -> GLOBAL (epilogue)
  # ---------------------------
  c = c.reshape(WARPS_IN_BLOCK_Y, ITERS_PER_WARP_M, LANES_PER_WARP_Y, TM,
                WARPS_IN_BLOCK_X, ITERS_PER_WARP_N, LANES_PER_WARP_X, TN)
  c = c[warpIdy, :, laneIdy, :,
        warpIdx, :, laneIdx, :]
  iterWarpM2, yt2, iterWarpN2, xt2 = rngs2 = rngs_for_shape(c_regs.shape, 600)
  sink = c[*rngs2].store(c_regs.after(sink)[*rngs2].cast(dtypes.half)).end(*rngs2)

  return sink.sink(arg=KernelInfo(opts_to_apply=())).simplify()

def test_matmul_fp16(sink:UOp, N=N):
  rng = np.random.default_rng()
  a = Tensor((rng.random((N, N), dtype=np.float32) - 0.5).astype(np.float16))
  b = Tensor((rng.random((N, N), dtype=np.float32) - 0.5).astype(np.float16))
  hc = Tensor.empty(N, N, dtype=dtypes.half)
  Tensor.realize(a, b, hc)

  ei = ExecItem(get_runner(Device.DEFAULT, sink), [t.uop.buffer for t in [hc, a, b]])

  ets = []
  with Context(DEBUG=2):
    for _ in range(run_count):
      ets.append(ei.run(wait=True))
  print(f"REAL TFLOPS {N * N * N * 2 / min(ets) * 1e-12:.2f}")

  if getenv("VERIFY", 1):
    GlobalCounters.reset()
    with Context(DEBUG=2):
      tc = (a.cast(dtypes.float) @ b.cast(dtypes.float)).realize()
    with Context(DEBUG=0):
      err = (hc.cast(dtypes.float) - tc).square().mean().item()
    print(f"mean squared error {err}")
    if err > 1e-02:
      raise RuntimeError("matmul is wrong!")

if __name__ == "__main__":
  test_matmul_fp16(hand_spec(), N=N)
