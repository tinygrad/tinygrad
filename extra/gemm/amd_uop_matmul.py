from tinygrad import Tensor, Device, Context, GlobalCounters, dtypes
from tinygrad.uop.ops import UOp, KernelInfo
from tinygrad.engine.realize import ExecItem, get_runner
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import getenv

N = 4096
run_count = 5

# block for locals
BN = 128
BM = 128
BK = 8

# t for registers
TN = 4
TM = 4


def hand_spec_kernel3(kernel5=getenv("K5", 0)):
  # ---------------------------
  # launch/config constants
  # ---------------------------

  BLOCK_SIZE = 128 if kernel5 else 256

  nbWaves = BLOCK_SIZE // 32
  WN = 128 if kernel5 else 64
  WM = BN * BM // nbWaves // WN

  # Sanity checks (fail fast if shapes/tiles misalign)
  assert BN % WN == 0, "BN must be a multiple of WN"
  assert BM % WM == 0, "BM must be a multiple of WM"
  nbWaveX = BN // WN
  nbWaveY = BM // WM

  assert BLOCK_SIZE % BN == 0, "BLOCK_SIZE must be divisible by BN"
  assert BLOCK_SIZE % BK == 0, "BLOCK_SIZE must be divisible by BK"

  assert (BN * BK) % BLOCK_SIZE == 0
  assert (BM * BK) % BLOCK_SIZE == 0

  # ---------------------------
  # per-thread read mapping
  # ---------------------------
  # A: read BK x BN tiles; B: read BN x BK tiles

  threadIdx_x = UOp.special(BLOCK_SIZE, "lidx0")
  waveIndex = threadIdx_x // 32
  waveIdx = waveIndex % nbWaveX
  waveIdy = waveIndex // nbWaveX
  indexInWave = threadIdx_x % 32

  nbThreadXPerWave = 8
  nbThreadYPerWave = 4

  idxInWave = indexInWave % nbThreadXPerWave
  idyInWave = indexInWave // nbThreadXPerWave

  nbIterWaveN = WN // (nbThreadXPerWave * TN)
  nbIterWaveM = WM // (nbThreadYPerWave * TM)

  SUBWN = WN // nbIterWaveN
  SUBWM = WM // nbIterWaveM

  # ---------------------------
  # block indices & placeholders
  # ---------------------------
  blockIdx_x = UOp.special(N // BN, "gidx0")
  blockIdx_y = UOp.special(N // BM, "gidx1")

  a = UOp.placeholder(dtypes.float, (N, N), slot=1)
  b = UOp.placeholder(dtypes.float, (N, N), slot=2)
  c = UOp.placeholder(dtypes.float, (N, N), slot=0)

  BM_As_stride = (BM + 4) if kernel5 else BM
  As = UOp.placeholder(dtypes.float, (BK, BM_As_stride), slot=0, addrspace=AddrSpace.LOCAL)
  Bs = UOp.placeholder(dtypes.float, (BK, BN), slot=1, addrspace=AddrSpace.LOCAL)

  A_col = UOp.placeholder(dtypes.float, (nbIterWaveM, TM), slot=0, addrspace=AddrSpace.REG)
  B_row = UOp.placeholder(dtypes.float, (nbIterWaveN, TN), slot=1, addrspace=AddrSpace.REG)
  c_regs = UOp.placeholder(dtypes.float, (nbIterWaveM, TM, nbIterWaveN, TN), slot=2, addrspace=AddrSpace.REG)

  i = UOp.range(c_regs.dtype.size, 16)
  c_regs = c_regs[i].set(0.0, end=i)

  kId_range = UOp.range(N // BK, 0)
  kId = kId_range * BK

  # ---------------------------
  # GLOBAL -> LOCAL (As, Bs)
  # ---------------------------
  nbReadsB = BN * BK // BLOCK_SIZE
  i = UOp.range(nbReadsB, 1)
  rBIdx = threadIdx_x % BN
  rBIdy = threadIdx_x // BN
  strideReadB = BLOCK_SIZE // BN
  index_x = BN * blockIdx_x + rBIdx
  index_y = rBIdy + i * strideReadB + kId
  Bs_store = Bs[index_y % BK, index_x % BN].store(b[index_y, index_x]).end(i)

  nbReadsA = BM * BK // BLOCK_SIZE
  i = UOp.range(nbReadsA, 2)
  rAIdx = threadIdx_x % BK
  rAIdy = threadIdx_x // BK
  strideReadA = BLOCK_SIZE // BK
  index_x = rAIdx + kId
  index_y = BM * blockIdx_y + rAIdy + i * strideReadA
  As_store = As[index_x % BK, index_y % BM].store(a[index_y, index_x]).end(i)

  # TODO: can we automate barrier?
  barrier = UOp.barrier(As_store, Bs_store)
  Bs = Bs.after(barrier)
  As = As.after(barrier)

  # open inner k range
  k = UOp.range(BK, 3)

  # ---------------------------
  # LOCAL -> REG (per-wave tiles)
  # ---------------------------
  iterWave = UOp.range(nbIterWaveN, 4)
  i = UOp.range(TN, 5)
  index = waveIdx * WN + iterWave * SUBWN + TN * idxInWave + i
  B_row = B_row[iterWave, i].set(Bs[k, index], end=(iterWave, i))

  iterWave = UOp.range(nbIterWaveM, 6)
  i = UOp.range(TM, 7)
  index = waveIdy * WM + iterWave * SUBWM + TM * idyInWave + i
  A_col = A_col[iterWave, i].set(As[k, index], end=(iterWave, i))

  # ---------------------------
  # FMA: c_regs += A_col * B_row
  # ---------------------------
  iterWaveM = UOp.range(nbIterWaveM, 8)
  yt = UOp.range(TM, 9)
  iterWaveN = UOp.range(nbIterWaveN, 10)
  xt = UOp.range(TN, 12)
  c_idx = c_regs.after(k, kId_range)[iterWaveM, yt, iterWaveN, xt]
  sink = c_idx.store(c_idx + A_col[iterWaveM, yt] * B_row[iterWaveN, xt]).end(iterWaveM, iterWaveN, yt, xt)

  # Close k, sync, and close K tiles
  sink = sink.end(k).barrier().end(kId_range)

  # ---------------------------
  # REG -> GLOBAL (epilogue)
  # ---------------------------
  iterWaveM = UOp.range(nbIterWaveM, 1000)
  yt = UOp.range(TM, 1001)
  iterWaveN = UOp.range(nbIterWaveN, 1002)
  xt = UOp.range(TN, 1003)
  xOut = blockIdx_x * BN + waveIdx * WN + iterWaveN * SUBWN + TN * idxInWave
  yOut = blockIdx_y * BM + waveIdy * WM + iterWaveM * SUBWM + TM * idyInWave
  sink = c[yOut + yt, xOut + xt].store(c_regs.after(sink)[iterWaveM, yt, iterWaveN, xt])
  sink = sink.end(iterWaveM, iterWaveN, yt, xt)

  return sink.sink(arg=KernelInfo(opts_to_apply=()))


if __name__ == "__main__":
  with Context(DEBUG=0):
    a = Tensor.randn(N, N)
    b = Tensor.randn(N, N)
    hc = Tensor.empty(N, N)
    Tensor.realize(a, b, hc)

  sink = hand_spec_kernel3()
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
