from tinygrad import Tensor, Device, Context, GlobalCounters, dtypes
from tinygrad.uop.ops import UOp, KernelInfo
from tinygrad.engine.realize import CompiledRunner, ExecItem, get_program
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import getenv

N = 4096
run_count = 5

BN = 128
BM = 128
BK = 8

TN = 4
TM = 4

def hand_spec_kernel3(kernel5=getenv("K5", 0)):
  BLOCK_SIZE = 128 if kernel5 else 256

  # ---------------------------
  # launch/config constants
  # ---------------------------
  nbWaves = BLOCK_SIZE // 32
  WN = 128 if kernel5 else 64
  WM = BN * BM // nbWaves // WN

  # Sanity checks (fail fast if shapes/tiles misalign)
  assert BN % WN == 0, "BN must be a multiple of WN"
  assert BM % WM == 0, "BM must be a multiple of WM"
  nbWaveX = BN // WN
  nbWaveY = BM // WM

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
  # per-thread read mapping
  # ---------------------------
  # A: read BK x BN tiles; B: read BN x BK tiles
  rAIdx = threadIdx_x % BK
  rAIdy = threadIdx_x // BK
  rBIdx = threadIdx_x % BN
  rBIdy = threadIdx_x // BN

  strideReadB = BLOCK_SIZE // BN
  strideReadA = BLOCK_SIZE // BK
  assert BLOCK_SIZE % BN == 0, "BLOCK_SIZE must be divisible by BN"
  assert BLOCK_SIZE % BK == 0, "BLOCK_SIZE must be divisible by BK"

  nbReadsB = BN * BK // BLOCK_SIZE
  nbReadsA = BM * BK // BLOCK_SIZE
  assert (BN * BK) % BLOCK_SIZE == 0
  assert (BM * BK) % BLOCK_SIZE == 0

  # ---------------------------
  # block indices & placeholders
  # ---------------------------
  blockIdx_x = UOp.special(N//BN, "gidx0")
  blockIdx_y = UOp.special(N//BM, "gidx1")

  a = UOp.placeholder(dtypes.float, (N, N), slot=1)
  b = UOp.placeholder(dtypes.float, (N, N), slot=2)
  c = UOp.placeholder(dtypes.float, (N, N), slot=0)

  BM_As_stride = (BM+4) if kernel5 else BM
  As = UOp.placeholder(dtypes.float, (BK, BM_As_stride), slot=0, addrspace=AddrSpace.LOCAL)
  Bs = UOp.placeholder(dtypes.float, (BK, BN), slot=1, addrspace=AddrSpace.LOCAL)

  A_col = UOp.placeholder(dtypes.float, (nbIterWaveM, TM), slot=0, addrspace=AddrSpace.REG)
  B_row = UOp.placeholder(dtypes.float, (nbIterWaveN, TN), slot=1, addrspace=AddrSpace.REG)
  c_regs = UOp.placeholder(dtypes.float, (nbIterWaveM, TM, nbIterWaveN, TN), slot=2, addrspace=AddrSpace.REG)

  i = UOp.range(c_regs.dtype.size, 16)
  c_regs = c_regs[i].set(0.0, end=i)

  kId_range = UOp.range(N//BK, 0)
  kId = kId_range*BK

  # ---------------------------
  # GLOBAL -> LOCAL (As, Bs)
  # ---------------------------
  i = UOp.range(nbReadsB, 1)
  index_x = BN * blockIdx_x + rBIdx
  index_y = rBIdy + i * strideReadB + kId
  Bs_store = Bs[index_y % BK, index_x % BN].store(b[index_y, index_x]).end(i)

  i = UOp.range(nbReadsA, 2)
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

  return sink.sink(arg=KernelInfo(name="tinygemm", opts_to_apply=()))

if __name__ == "__main__":
  HL = getenv("HL")
  hprg = hand_spec_kernel3()
  prg = get_program(hprg, Device.default.renderer)
  print(prg.src)
  if getenv("SRC"): exit(0)
  hrunner = CompiledRunner(prg)

  a = Tensor.randn(N, N).realize()
  b = Tensor.randn(N, N).realize()
  hc = Tensor.zeros(N, N).contiguous().realize()

  GlobalCounters.reset()
  with Context(DEBUG=2):
    for _ in range(run_count): tc = (a@b).realize()

  GlobalCounters.reset()
  buffers = [hc.uop.buffer, a.uop.buffer, b.uop.buffer]
  ei = ExecItem(hrunner, buffers)
  ets = []
  with Context(DEBUG=2):
    for _ in range(run_count):
      ets.append(ei.run(wait=True))
  err = (hc-tc).square().mean().item()
  print(f"hrunner {err}")
  print(f"TFLOPS {N*N*N*2/min(ets)*1e-12:.2f}")
  if err > 1e-06: raise RuntimeError("matmul is wrong!")
