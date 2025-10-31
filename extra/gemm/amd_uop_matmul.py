from tinygrad import Tensor, Device, Context, GlobalCounters, dtypes
from tinygrad.uop.ops import UOp, KernelInfo
from tinygrad.engine.realize import ExecItem, get_runner
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import getenv

N = 4096
run_count = 5

# ---------------------------
# launch/config constants
# ---------------------------

WARP_SIZE = 32

# Threadblock tile sizes (block-level tile of C that a block computes)
BLOCK_N = 128   # columns of C (N-dim) per block
BLOCK_M = 128   # rows of C (M-dim) per block
BLOCK_K = 8     # K-slice per block iteration

# Register tile sizes (per-thread accumulator tile of C)
TN = 4     # columns per thread
TM = 4     # rows per thread

is_kernel5 = getenv("K5", 0)
THREADS_PER_BLOCK = 128 if is_kernel5 else 256
assert THREADS_PER_BLOCK % BLOCK_N == 0, "THREADS_PER_BLOCK must be divisible by BLOCK_N"
assert THREADS_PER_BLOCK % BLOCK_K == 0, "THREADS_PER_BLOCK must be divisible by BLOCK_K"
assert (BLOCK_N * BLOCK_K) % THREADS_PER_BLOCK == 0
assert (BLOCK_M * BLOCK_K) % THREADS_PER_BLOCK == 0

WARPS_PER_BLOCK = THREADS_PER_BLOCK // WARP_SIZE
WAVE_TILE_N = 128 if is_kernel5 else 64
WAVE_TILE_M = BLOCK_N * BLOCK_M // WARPS_PER_BLOCK // WAVE_TILE_N
assert BLOCK_N % WAVE_TILE_N == 0, "BN must be a multiple of WN"
assert BLOCK_M % WAVE_TILE_M == 0, "BM must be a multiple of WM"
WAVES_IN_BLOCK_X = BLOCK_N // WAVE_TILE_N
WAVES_IN_BLOCK_Y = BLOCK_M // WAVE_TILE_M
assert WAVES_IN_BLOCK_X * WAVES_IN_BLOCK_Y == WARPS_PER_BLOCK, "wave grid must match warps/block"

LANES_PER_WAVE_X = 8
LANES_PER_WAVE_Y = 4
ITERS_PER_WAVE_N = WAVE_TILE_N // (LANES_PER_WAVE_X * TN)
ITERS_PER_WAVE_M = WAVE_TILE_M // (LANES_PER_WAVE_Y * TM)
N_PER_ITER = WAVE_TILE_N // ITERS_PER_WAVE_N
M_PER_ITER = WAVE_TILE_M // ITERS_PER_WAVE_M
assert WAVE_TILE_N % (LANES_PER_WAVE_X * TN) == 0, "WAVE_TILE_N must be divisible by LANES_PER_WAVE_X*TN"
assert WAVE_TILE_M % (LANES_PER_WAVE_Y * TM) == 0, "WAVE_TILE_M must be divisible by LANES_PER_WAVE_Y*TM"

def hand_spec_kernel3():
  # ---------------------------
  # per-thread read mapping
  # ---------------------------
  # A: read BK x BN tiles; B: read BN x BK tiles
  tid = UOp.special(THREADS_PER_BLOCK, "lidx0")

  waveIdx = (tid // WARP_SIZE) % WAVES_IN_BLOCK_X
  waveIdy = (tid // WARP_SIZE) // WAVES_IN_BLOCK_X
  assert waveIdy.vmax+1 == WAVES_IN_BLOCK_Y

  idxInWave = (tid % WARP_SIZE) % LANES_PER_WAVE_X
  idyInWave = (tid % WARP_SIZE) // LANES_PER_WAVE_X
  assert idyInWave.vmax+1 == LANES_PER_WAVE_Y

  # ---------------------------
  # block indices & placeholders
  # ---------------------------
  blockIdx_x = UOp.special(N // BLOCK_N, "gidx0")
  blockIdx_y = UOp.special(N // BLOCK_M, "gidx1")

  a = UOp.placeholder((N, N), dtypes.float, slot=1)
  b = UOp.placeholder((N, N), dtypes.float, slot=2)
  c = UOp.placeholder((N, N), dtypes.float, slot=0)

  BM_As_stride = (BLOCK_M + 4) if is_kernel5 else BLOCK_M
  As = UOp.placeholder((BLOCK_K, BM_As_stride), dtypes.float, slot=0, addrspace=AddrSpace.LOCAL).shrink_to((BLOCK_K, BLOCK_M))
  Bs = UOp.placeholder((BLOCK_K, BLOCK_N), dtypes.float, slot=1, addrspace=AddrSpace.LOCAL)

  A_col = UOp.placeholder((ITERS_PER_WAVE_M, TM), dtypes.float, slot=0, addrspace=AddrSpace.REG)
  B_row = UOp.placeholder((ITERS_PER_WAVE_N, TN), dtypes.float, slot=1, addrspace=AddrSpace.REG)
  c_regs = UOp.placeholder((ITERS_PER_WAVE_M, TM, ITERS_PER_WAVE_N, TN), dtypes.float, slot=2, addrspace=AddrSpace.REG)

  i = UOp.range(c_regs.size, 16)
  c_regs = c_regs[i].set(0.0, end=i)

  k_tile_range = UOp.range(N // BLOCK_K, 0)

  # ---------------------------
  # GLOBAL -> LOCAL (As, Bs)
  # ---------------------------
  b = b.reshape((N // BLOCK_K, BLOCK_K,
                 N // BLOCK_N, BLOCK_N))
  i = UOp.range(BLOCK_N * BLOCK_K // THREADS_PER_BLOCK, 1)
  index_x = tid % BLOCK_N
  index_y = (tid // BLOCK_N) + (THREADS_PER_BLOCK // BLOCK_N) * i
  Bs_store = Bs[index_y, index_x].store(b[k_tile_range, index_y, blockIdx_x, index_x]).end(i)

  a = a.reshape((N // BLOCK_M, BLOCK_M,
                 N // BLOCK_K, BLOCK_K))
  i = UOp.range(BLOCK_M * BLOCK_K // THREADS_PER_BLOCK, 2)
  index_x = tid % BLOCK_K
  index_y = (tid // BLOCK_K) + (THREADS_PER_BLOCK // BLOCK_K) * i
  As_store = As[index_x, index_y].store(a[blockIdx_y, index_y, k_tile_range, index_x]).end(i)

  # TODO: can we automate barrier?
  barrier = UOp.barrier(As_store, Bs_store)
  Bs = Bs.after(barrier)
  As = As.after(barrier)

  # open inner k range
  k = UOp.range(BLOCK_K, 3)

  # ---------------------------
  # LOCAL -> REG (per-wave tiles)
  # ---------------------------
  Bs_view = Bs.reshape((BLOCK_K, WAVES_IN_BLOCK_X, ITERS_PER_WAVE_N, LANES_PER_WAVE_X, TN))
  iterWaveN = UOp.range(ITERS_PER_WAVE_N, 4)
  i = UOp.range(TN, 5)
  B_row = B_row[iterWaveN, i].set(Bs_view[k, waveIdx, iterWaveN, idxInWave, i], end=(iterWaveN, i))

  As_view = As.reshape((BLOCK_K, WAVES_IN_BLOCK_Y, ITERS_PER_WAVE_M, LANES_PER_WAVE_Y, TM))
  iterWaveM = UOp.range(ITERS_PER_WAVE_M, 6)
  i = UOp.range(TM, 7)
  A_col = A_col[iterWaveM, i].set(As_view[k, waveIdy, iterWaveM, idyInWave, i], end=(iterWaveM, i))

  # ---------------------------
  # FMA: c_regs += A_col * B_row
  # ---------------------------
  iterWaveM = UOp.range(ITERS_PER_WAVE_M, 8)
  yt = UOp.range(TM, 9)
  iterWaveN = UOp.range(ITERS_PER_WAVE_N, 10)
  xt = UOp.range(TN, 12)
  c_idx = c_regs.after(k, k_tile_range)[iterWaveM, yt, iterWaveN, xt]
  sink = c_idx.store(c_idx + A_col[iterWaveM, yt] * B_row[iterWaveN, xt]).end(iterWaveM, iterWaveN, yt, xt)

  # Close k, sync, and close K tiles
  sink = sink.end(k).barrier().end(k_tile_range)

  # ---------------------------
  # REG -> GLOBAL (epilogue)
  # ---------------------------
  c = c.reshape((N//BLOCK_M, WAVES_IN_BLOCK_Y, ITERS_PER_WAVE_M, LANES_PER_WAVE_Y, TM,
                 N//BLOCK_N, WAVES_IN_BLOCK_X, ITERS_PER_WAVE_N, LANES_PER_WAVE_X, TN))
  iterWaveM = UOp.range(ITERS_PER_WAVE_M, 1000)
  yt = UOp.range(TM, 1001)
  iterWaveN = UOp.range(ITERS_PER_WAVE_N, 1002)
  xt = UOp.range(TN, 1003)
  c_glbl_idx = c[blockIdx_y, waveIdy, iterWaveM, idyInWave, yt, blockIdx_x, waveIdx, iterWaveN, idxInWave, xt]
  sink = c_glbl_idx.store(c_regs.after(sink)[iterWaveM, yt, iterWaveN, xt])
  sink = sink.end(iterWaveM, iterWaveN, yt, xt)

  return sink.sink(arg=KernelInfo(opts_to_apply=())).simplify()

def test_matmul(sink:UOp, N=N):
  with Context(DEBUG=0):
    a = Tensor.randn(N, N)
    b = Tensor.randn(N, N)
    hc = Tensor.empty(N, N)
    Tensor.realize(a, b, hc)

  ei = ExecItem(get_runner(Device.DEFAULT, sink), [t.uop.buffer for t in [hc, a, b]])

  GlobalCounters.reset()
  ets = []
  with Context(DEBUG=2):
    for _ in range(run_count):
      ets.append(ei.run(wait=True))
  print(f"REAL TFLOPS {N * N * N * 2 / min(ets) * 1e-12:.2f}")

  GlobalCounters.reset()
  with Context(DEBUG=2):
    tc = (a @ b).realize()
  with Context(DEBUG=0):
    err = (hc - tc).square().mean().item()
  print(f"mean squared error {err}")
  if err > 1e-06:
    raise RuntimeError("matmul is wrong!")

if __name__ == "__main__":
  test_matmul(hand_spec_kernel3(), N=N)
