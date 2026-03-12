import numpy as np
from tinygrad import Tensor, Device, Context, GlobalCounters, dtypes
from tinygrad.uop.ops import UOp, KernelInfo, sint, AxisType
from tinygrad.engine.realize import ExecItem, get_runner
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import getenv

N = getenv("N", 4096)
M = getenv("M", N)
K = getenv("K", N)
NUM_RUNS = getenv("CNT", 5)

# ---------------------------
# launch/config constants
# ---------------------------

WARP_SIZE = 32

# Threadblock tile sizes (block-level tile of C that a block computes)
BLOCK_N = 128   # columns of C (N-dim) per block
BLOCK_M = 128   # rows of C (M-dim) per block
BLOCK_K = 8     # K-slice per block iteration
assert N % BLOCK_N == 0, f"N ({N}) must be a multiple of BLOCK_N ({BLOCK_N})"
assert M % BLOCK_M == 0, f"M ({M}) must be a multiple of BLOCK_M ({BLOCK_M})"
assert K % BLOCK_K == 0, f"K ({K}) must be a multiple of BLOCK_K ({BLOCK_K})"

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
WAVES_PER_BLOCK_N = BLOCK_N // WAVE_TILE_N
WAVES_PER_BLOCK_M = BLOCK_M // WAVE_TILE_M
assert WAVES_PER_BLOCK_N * WAVES_PER_BLOCK_M == WARPS_PER_BLOCK, "wave grid must match warps/block"

LANES_PER_WAVE_N = 8
LANES_PER_WAVE_M = 4
REG_TILES_PER_WAVE_N = WAVE_TILE_N // (LANES_PER_WAVE_N * TN)
REG_TILES_PER_WAVE_M = WAVE_TILE_M // (LANES_PER_WAVE_M * TM)
assert WAVE_TILE_N % (LANES_PER_WAVE_N * TN) == 0, "WAVE_TILE_N must be divisible by LANES_PER_WAVE_N*TN"
assert WAVE_TILE_M % (LANES_PER_WAVE_M * TM) == 0, "WAVE_TILE_M must be divisible by LANES_PER_WAVE_M*TM"

def rngs_for_shape(shape:tuple[sint, ...], rng:int, axis_type=AxisType.LOOP): return [UOp.range(s, rng+i, axis_type) for i,s in enumerate(shape)]
def copy(dest:UOp, src:UOp, rng:int, set=False, upcast=False):
  assert dest.shape == src.shape
  rngs = rngs_for_shape(src.shape, rng, AxisType.UPCAST if upcast else AxisType.LOOP)
  copy = dest[*rngs].store(src[*rngs]).end(*rngs)
  return dest.after(copy) if set else copy

def hand_spec_kernel3():
  # ---------------------------
  # block indices & placeholders
  # ---------------------------
  block_id_n = UOp.special(N // BLOCK_N, "gidx0")
  block_id_m = UOp.special(M // BLOCK_M, "gidx1")

  a = UOp.placeholder((M, K), dtypes.float, slot=1)
  b = UOp.placeholder((K, N), dtypes.float, slot=2)
  c = UOp.placeholder((M, N), dtypes.float, slot=0)

  # index the output with the globals
  c = c.reshape(M // BLOCK_M, BLOCK_M, N // BLOCK_N, BLOCK_N)[block_id_m, :, block_id_n, :]

  # open the main reduction range
  k_tile_range = UOp.range(K // BLOCK_K, 0, AxisType.REDUCE)
  a = a.reshape(M // BLOCK_M, BLOCK_M, K // BLOCK_K, BLOCK_K)[block_id_m, :, k_tile_range, :]
  b = b.reshape(K // BLOCK_K, BLOCK_K, N // BLOCK_N, BLOCK_N)[k_tile_range, :, block_id_n, :]

  # globals are no longer used, they are already in the indexes
  del block_id_m, block_id_n

  # ---------------------------
  # GLOBAL -> LOCAL (A_local, B_local)
  # ---------------------------
  tid = UOp.special(THREADS_PER_BLOCK, "lidx0")

  # A: read BM x BK tiles (permute on store into locals)
  BM_A_local_stride = (BLOCK_M + 4) if is_kernel5 else BLOCK_M
  A_local = UOp.placeholder((BLOCK_K, BM_A_local_stride), dtypes.float, slot=0, addrspace=AddrSpace.LOCAL).shrink_to((BLOCK_K, BLOCK_M))
  A_local_store = copy(A_local.permute((1,0)).reshape(-1, THREADS_PER_BLOCK)[:, tid], a.reshape(-1, THREADS_PER_BLOCK)[:, tid], rng=100)

  # B: read BK x BN tiles
  B_local = UOp.placeholder((BLOCK_K, BLOCK_N), dtypes.float, slot=1, addrspace=AddrSpace.LOCAL)
  B_local_store = copy(B_local.reshape(-1, THREADS_PER_BLOCK)[:, tid], b.reshape(-1, THREADS_PER_BLOCK)[:, tid], rng=200)

  # TODO: can we automate barrier?
  barrier = UOp.barrier(A_local_store, B_local_store)
  A_local, B_local = A_local.after(barrier), B_local.after(barrier)

  # open inner k range
  k = UOp.range(BLOCK_K, 3, AxisType.REDUCE)

  # ---------------------------
  # LOCAL -> REG (per-wave tiles)
  # ---------------------------
  waveIdx = (tid // WARP_SIZE) % WAVES_PER_BLOCK_N
  waveIdy = (tid // WARP_SIZE) // WAVES_PER_BLOCK_N
  assert waveIdy.vmax+1 == WAVES_PER_BLOCK_M

  laneIdx = (tid % WARP_SIZE) % LANES_PER_WAVE_N
  laneIdy = (tid % WARP_SIZE) // LANES_PER_WAVE_N
  assert laneIdy.vmax+1 == LANES_PER_WAVE_M

  A_col = UOp.placeholder((REG_TILES_PER_WAVE_M, TM), dtypes.float, slot=0, addrspace=AddrSpace.REG)
  A_local_slice = A_local[k, :].reshape(WAVES_PER_BLOCK_M, REG_TILES_PER_WAVE_M, LANES_PER_WAVE_M, TM)[waveIdy, :, laneIdy, :]
  A_col = copy(A_col, A_local_slice , 300, set=True, upcast=True)

  B_row = UOp.placeholder((REG_TILES_PER_WAVE_N, TN), dtypes.float, slot=1, addrspace=AddrSpace.REG)
  B_local_slice = B_local[k, :].reshape(WAVES_PER_BLOCK_N, REG_TILES_PER_WAVE_N, LANES_PER_WAVE_N, TN)[waveIdx, :, laneIdx, :]
  B_row = copy(B_row, B_local_slice, 400, set=True, upcast=True)

  # ---------------------------
  # FMA: c_regs += A_col * B_row
  # ---------------------------
  c_regs = UOp.placeholder((REG_TILES_PER_WAVE_M, TM, REG_TILES_PER_WAVE_N, TN), dtypes.float, slot=2, addrspace=AddrSpace.REG)
  i = UOp.range(c_regs.size, 16)
  c_regs = c_regs.after(c_regs.flatten()[i].store(0.0).end(i))

  # TODO: why don't these work as upcast?
  # why if the ranges merge is it slow?!? (if you change the order on end, they will merge. big slowdown on METAL)
  iter_m, t_m, iter_n, t_n = rngs = rngs_for_shape(c_regs.shape, 500)
  sink = c_regs[*rngs].store(c_regs.after(k)[*rngs] + A_col[iter_m, t_m] * B_row[iter_n, t_n]).end(iter_m, iter_n, t_m, t_n)

  # Close k, sync, and close K tiles
  sink = sink.end(k).barrier().end(k_tile_range)

  # ---------------------------
  # REG -> GLOBAL (epilogue)
  # ---------------------------
  c = c.reshape(WAVES_PER_BLOCK_M, REG_TILES_PER_WAVE_M, LANES_PER_WAVE_M, TM,
                WAVES_PER_BLOCK_N, REG_TILES_PER_WAVE_N, LANES_PER_WAVE_N, TN)
  c = c[waveIdy, :, laneIdy, :,
        waveIdx, :, laneIdx, :]
  sink = copy(c, c_regs.after(sink), rng=600)

  return sink.sink(arg=KernelInfo(opts_to_apply=())).simplify()

def test_matmul(sink:UOp, dtype=dtypes.float32, M=M, N=N, K=K):
  rng = np.random.default_rng()
  a = Tensor(rng.random((M, K), dtype=np.float32)-0.5, dtype=dtype)
  b = Tensor(rng.random((K, N), dtype=np.float32)-0.5, dtype=dtype)
  hc = Tensor.empty(M, N, dtype=dtype)
  Tensor.realize(a, b, hc)

  ei = ExecItem(sink, [t.uop.buffer for t in [hc, a, b]], prg=get_runner(Device.DEFAULT, sink))

  ets = []
  with Context(DEBUG=2):
    for _ in range(NUM_RUNS):
      ets.append(ei.run(wait=True))
  print(f"REAL TFLOPS {M * N * K * 2 / min(ets) * 1e-12:.2f}")

  if getenv("VERIFY", 1):
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
