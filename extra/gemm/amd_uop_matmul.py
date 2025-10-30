from tinygrad import Tensor, Device, Context, GlobalCounters, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo, graph_rewrite, AxisType, PatternMatcher, UPat, axis_colors
from tinygrad.engine.realize import CompiledRunner, ExecItem, get_program
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import getenv, colored, prod, unwrap
from tinygrad.codegen.opt import Opt, OptOps

def to_colored(full_shape, axis_types): return '_'.join([colored(str(s), axis_colors[at]) for s,at in zip(full_shape, axis_types)])

N = 4096
run_count = 5

BN = 128
BM = 128
BK = 8

TN = 4
TM = 4

def hand_spec_kernel3(kernel4=getenv("K4", 0), kernel5=getenv("K5", 0)):
  BLOCK_SIZE = 128 if kernel5 else 256

  nbWaves = BLOCK_SIZE // 32
  WN = 128 if kernel5 else 64
  WM = BN * BM // nbWaves // WN

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

  # Thread mapping to read BKxBN block from A
  rAIdx = threadIdx_x % BK
  rAIdy = threadIdx_x // BK
  # Thread mapping to read BNxBK block from B
  rBIdx = threadIdx_x % BN
  rBIdy = threadIdx_x // BN

  strideReadB = BLOCK_SIZE // BN
  strideReadA = BLOCK_SIZE // BK
  nbReadsB = BN * BK // BLOCK_SIZE
  nbReadsA = BM * BK // BLOCK_SIZE

  blockIdx_x = UOp.special(N//BN, "gidx0")
  blockIdx_y = UOp.special(N//BM, "gidx1")

  a = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(N*N), arg=1)
  b = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(N*N), arg=2)
  c = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(N*N), arg=0)

  A_col = UOp(Ops.DEFINE_REG, dtypes.float.ptr(nbIterWaveM * TM, AddrSpace.REG), arg=0)
  B_row = UOp(Ops.DEFINE_REG, dtypes.float.ptr(nbIterWaveN * TN, AddrSpace.REG), arg=1)

  BM_As_stride = (BM+4) if kernel5 else BM
  As = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(BK*BM_As_stride, AddrSpace.LOCAL), arg=0)
  Bs = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(BK*BN, AddrSpace.LOCAL), arg=1)

  c_regs = UOp(Ops.DEFINE_REG, dtypes.float.ptr(TM * nbIterWaveM * TN * nbIterWaveN), arg=2)

  i = UOp.range(c_regs.dtype.size, 16)
  init_store = c_regs[i].store(UOp.const(dtypes.float, 0.0)).end(i)

  if kernel4:
    regA = UOp(Ops.DEFINE_REG, dtypes.float.ptr(nbReadsA, AddrSpace.REG), arg=3)
    regB = UOp(Ops.DEFINE_REG, dtypes.float.ptr(nbReadsB, AddrSpace.REG), arg=4)

    # initial load from globals into locals (0)
    kId = 0

    # load from globals into locals
    i = UOp.range(nbReadsB, 0)
    index_x = BN * blockIdx_x + rBIdx
    index_y = rBIdy + i * strideReadB + kId
    Bs_store = Bs[(index_y % BK) * BN + index_x % BN].store(b[N * index_y + index_x], i)

    i = UOp.range(nbReadsA, 1)
    index_x = rAIdx + kId
    index_y = BM * blockIdx_y + rAIdy + i * strideReadA
    As_store = As[(index_x % BK) * BM_As_stride + index_y % BM].store(a[N * index_y + index_x], i)

    # iterate over the middle chunk
    kId_range = UOp.range(N//BK-1, 2)
    kId = kId_range*BK

    barrier = UOp.barrier(As_store, Bs_store)

    # load from globals into registers (next round)
    i = UOp.range(nbReadsB, 3)
    index_x = BN * blockIdx_x + rBIdx
    index_y = rBIdy + i * strideReadB + kId + BK
    regB_store = regB[i].store(b[N * index_y + index_x], i)

    i = UOp.range(nbReadsA, 4)
    index_x = rAIdx + kId + BK
    index_y = BM * blockIdx_y + rAIdy + i * strideReadA
    regA_store = regA[i].store(a[N * index_y + index_x], i)

    def inner_loop(first_range, inp_dep=()):
      # inner unroll
      k = UOp.range(BK, first_range+0)

      # load from locals into registers
      iterWave = UOp.range(nbIterWaveN, first_range+1)
      i = UOp.range(TN, first_range+2)
      index = waveIdx * WN + iterWave * SUBWN + TN * idxInWave + i
      B_row_store = B_row[iterWave*TN + i].store(Bs[k*BN + index].after(*inp_dep), iterWave, i)

      iterWave = UOp.range(nbIterWaveM, first_range+3)
      i = UOp.range(TM, first_range+4)
      index = waveIdy * WM + iterWave * SUBWM + TM * idyInWave + i
      A_col_store = A_col[iterWave*TM + i].store(As[k*BM_As_stride + index].after(*inp_dep), iterWave, i)

      # do the GEMM math
      iterWaveM = UOp.range(nbIterWaveM, first_range+5)
      yt = UOp.range(TM, first_range+6)
      iterWaveN = UOp.range(nbIterWaveN, first_range+7)
      xt = UOp.range(TN, first_range+8)
      x = iterWaveN * TN + xt
      y = iterWaveM * TM + yt
      c_regs_idx = c_regs[y * TN * nbIterWaveN + x]
      # sketchy, this should end the kId_range but it doesn't
      sink = c_regs_idx.store(c_regs_idx.after(init_store) + A_col[y].after(A_col_store) * B_row[x].after(B_row_store)).end(iterWaveM, iterWaveN, yt, xt, k)
      return sink

    # TODO: kId_range should endrange after a barrier
    sink = inner_loop(5, (barrier, regB_store, regA_store)).barrier()

    # load from registers into locals
    i = UOp.range(nbReadsB, 14)
    index_x = BN * blockIdx_x + rBIdx
    index_y = rBIdy + i * strideReadB + kId + BK
    Bs_store = Bs[(index_y % BK) * BN + index_x % BN].store(regB[i].load(sink), i, kId_range)

    i = UOp.range(nbReadsA, 15)
    index_x = rAIdx + kId + BK
    index_y = BM * blockIdx_y + rAIdy + i * strideReadA
    As_store = As[(index_x % BK) * BM_As_stride + index_y % BM].store(regA[i].load(sink), i, kId_range)

    # final iteration without the copy
    sink = inner_loop(16, (UOp.barrier(Bs_store, As_store),))
  else:
    kId_range = UOp.range(N//BK, 0)
    kId = kId_range*BK

    # load from globals into locals
    i = UOp.range(nbReadsB, 1)
    index_x = BN * blockIdx_x + rBIdx
    index_y = rBIdy + i * strideReadB + kId
    Bs_store = Bs[(index_y % BK) * BN + index_x % BN].store(b[N * index_y + index_x]).end(i)

    i = UOp.range(nbReadsA, 2)
    index_x = rAIdx + kId
    index_y = BM * blockIdx_y + rAIdy + i * strideReadA
    As_store = As[(index_x % BK) * BM_As_stride + index_y % BM].store(a[N * index_y + index_x]).end(i)

    barrier = UOp.barrier(As_store, Bs_store)

    k = UOp.range(BK, 3)

    # load from locals into registers
    iterWave = UOp.range(nbIterWaveN, 4)
    i = UOp.range(TN, 5)
    index = waveIdx * WN + iterWave * SUBWN + TN * idxInWave + i
    B_row_store = B_row[iterWave*TN + i].store(Bs.after(barrier)[k*BN + index]).end(iterWave, i)

    iterWave = UOp.range(nbIterWaveM, 6)
    i = UOp.range(TM, 7)
    index = waveIdy * WM + iterWave * SUBWM + TM * idyInWave + i
    A_col_store = A_col[iterWave*TM + i].store(As.after(barrier)[k*BM_As_stride + index]).end(iterWave, i)

    # do the GEMM math
    iterWaveM = UOp.range(nbIterWaveM, 8)
    yt = UOp.range(TM, 9)
    iterWaveN = UOp.range(nbIterWaveN, 10)
    xt = UOp.range(TN, 12)
    x = iterWaveN * TN + xt
    y = iterWaveM * TM + yt

    gemm = c_regs.after(init_store)[y * TN * nbIterWaveN + x] + A_col.after(A_col_store)[y] * B_row.after(B_row_store)[x]
    sink = c_regs[y * TN * nbIterWaveN + x].store(gemm).end(iterWaveM, iterWaveN, yt, xt, k).barrier().end(kId_range)

  # store c_regs into c
  iterWaveM = UOp.range(nbIterWaveM, 1000)
  yt = UOp.range(TM, 1001)
  iterWaveN = UOp.range(nbIterWaveN, 1002)
  xt = UOp.range(TN, 1003)
  xOut = blockIdx_x * BN + waveIdx * WN + iterWaveN * SUBWN + TN * idxInWave
  yOut = blockIdx_y * BM + waveIdy * WM + iterWaveM * SUBWM + TM * idyInWave
  indexC = N * (yOut + yt) + xOut + xt
  sink = c[indexC].store(c_regs.after(sink)[TN * nbIterWaveN * (iterWaveM * TM + yt) + (iterWaveN * TN + xt)]).end(iterWaveM, iterWaveN, yt, xt)

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
