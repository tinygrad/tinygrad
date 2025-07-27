from tinygrad import Tensor, Device, Context, GlobalCounters, dtypes
from tinygrad.uop.ops import UOp, Ops, KernelInfo, graph_rewrite, AxisType
from tinygrad.engine.realize import CompiledRunner, ExecItem, get_program
from tinygrad.dtype import AddrSpace
from tinygrad.schedule.kernelize import merge_views
from tinygrad.helpers import getenv, colored
from tinygrad.shape.shapetracker import ShapeTracker

N = 4096
run_count = 5

BN = 128
BM = 128
BK = 8

TN = 4
TM = 4

def hl_spec_kernel3():
  nbIterWaveM = 2
  nbIterWaveN = 2

  # define buffers
  # TODO: remove these views once the defines have a shape
  a = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(N*N), arg=0).view(ShapeTracker.from_shape((N,N)))
  b = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(N*N), arg=1).view(ShapeTracker.from_shape((N,N))).permute((1,0))
  c = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(N*N), arg=2).view(ShapeTracker.from_shape((N,N)))
  As = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(BK*BM, AddrSpace.LOCAL), arg=0).view(ShapeTracker.from_shape((BK*BM,)))
  Bs = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(BK*BN, AddrSpace.LOCAL), arg=1).view(ShapeTracker.from_shape((BK*BN,)))
  A_col = UOp(Ops.DEFINE_REG, dtypes.float.ptr(nbIterWaveM * TM, AddrSpace.REG), arg=0).view(ShapeTracker.from_shape((nbIterWaveM * TM,)))
  B_row = UOp(Ops.DEFINE_REG, dtypes.float.ptr(nbIterWaveN * TN, AddrSpace.REG), arg=1).view(ShapeTracker.from_shape((nbIterWaveN * TN,)))

  # shape buffers. TODO: permutes
  full_shape = (N//BM, nbIterWaveM, BM//(nbIterWaveM * TM), TM, N//BN, nbIterWaveN, BN//(nbIterWaveN * TN), TN, N//BK, BK)
  a = a.reshape((N//BM, nbIterWaveM, BM//(nbIterWaveM * TM), TM, 1, 1, 1, 1, N//BK, BK)).expand(full_shape)
  b = b.reshape((1, 1, 1, 1, N//BN, nbIterWaveN, BN//(nbIterWaveN * TN), TN, N//BK, BK)).expand(full_shape)
  c = c.reshape((N//BM, nbIterWaveM, BM//(nbIterWaveM * TM), TM, N//BN, nbIterWaveN, BN//(nbIterWaveN * TN), TN, 1, 1))
  As = As.reshape((1, nbIterWaveM, BM//(nbIterWaveM * TM), TM, 1, 1, 1, 1, 1, BK)).expand(full_shape)
  Bs = Bs.reshape((1, 1, 1, 1, 1, nbIterWaveN, BN//(nbIterWaveN * TN), TN, 1, BK)).expand(full_shape)
  A_col = A_col.reshape((1, nbIterWaveM, 1, TM, 1, 1, 1, 1, 1, 1)).expand(full_shape)
  B_row = B_row.reshape((1, 1, 1, 1, 1, nbIterWaveN, 1, TN, 1, 1)).expand(full_shape)

  #out = (a.load() * b.load()).r(Ops.ADD, (8, 9))
  out = (As.load(As.store(a.load())) * Bs.load(Bs.store(b.load()))).r(Ops.ADD, (8, 9))
  #out = (A_col.load(A_col.store(As.load(As.store(a.load())))) * B_row.load(B_row.store(Bs.load(Bs.store(b.load()))))).r(Ops.ADD, (8, 9))

  axis_types = (
    AxisType.GLOBAL, AxisType.UPCAST, AxisType.LOCAL, AxisType.UPCAST,
    AxisType.GLOBAL, AxisType.UPCAST, AxisType.LOCAL, AxisType.UPCAST,
    AxisType.REDUCE, AxisType.UNROLL)

  from tinygrad.opt.kernel import axis_colors
  shape = '_'.join([colored(str(s), axis_colors[at]) for s,at in zip(full_shape, axis_types)])
  sink = c.store(out).sink(arg=KernelInfo(name="tg_"+shape, axis_types=axis_types))
  sink = graph_rewrite(sink, merge_views)
  return sink

def hand_spec_kernel3():
  BLOCK_SIZE = 256

  nbWaves = BLOCK_SIZE // 32
  WN = 64
  WM = BN * BM // nbWaves // WN

  nbWaveX = BN // WN
  nbWaveY = BM // WM

  threadIdx_x = UOp(Ops.SPECIAL, dtypes.int, arg=("lidx0", BLOCK_SIZE))
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

  blockIdx_x = UOp(Ops.SPECIAL, dtypes.int, arg=("gidx0", N//BN))
  blockIdx_y = UOp(Ops.SPECIAL, dtypes.int, arg=("gidx1", N//BM))

  a = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(N*N), arg=0)
  b = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(N*N), arg=1)
  c = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(N*N), arg=2)

  A_col = UOp(Ops.DEFINE_REG, dtypes.float.ptr(nbIterWaveM * TM, AddrSpace.REG), arg=0)
  B_row = UOp(Ops.DEFINE_REG, dtypes.float.ptr(nbIterWaveN * TN, AddrSpace.REG), arg=1)

  As = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(BK*BM, AddrSpace.LOCAL), arg=0)
  Bs = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(BK*BN, AddrSpace.LOCAL), arg=1)

  c_regs = UOp(Ops.DEFINE_REG, dtypes.float.ptr(TM * nbIterWaveM * TN * nbIterWaveN), arg=2)

  i = UOp.range(dtypes.int, c_regs.dtype.size, 16)
  init_store = c_regs[i].store(UOp.const(dtypes.float, 0.0), i)

  kId_range = UOp.range(dtypes.int, N//BK, 0)
  kId = kId_range*BK

  # load from globals into locals
  i = UOp.range(dtypes.int, nbReadsB, 1)
  index_x = BN * blockIdx_x + rBIdx
  index_y = rBIdy + i * strideReadB + kId
  Bs_store = Bs[(index_y % BK) * BN + index_x % BN].store(b[N * index_y + index_x].load(), i)

  i = UOp.range(dtypes.int, nbReadsA, 2)
  index_x = rAIdx + kId
  index_y = BM * blockIdx_y + rAIdy + i * strideReadA
  As_store = As[(index_x % BK) * BM + index_y % BM].store(a[N * index_y + index_x].load(), i)

  barrier = UOp(Ops.BARRIER, src=(As_store, Bs_store))

  k = UOp.range(dtypes.int, BK, 3)

  # load from locals into registers
  iterWave = UOp.range(dtypes.int, nbIterWaveN, 4)
  i = UOp.range(dtypes.int, TN, 5)
  index = waveIdx * WN + iterWave * SUBWN + TN * idxInWave + i
  B_row_store = B_row[iterWave*TN + i].store(Bs[k*BN + index].load(barrier), iterWave, i)

  iterWave = UOp.range(dtypes.int, nbIterWaveM, 6)
  i = UOp.range(dtypes.int, TM, 7)
  index = waveIdy * WM + iterWave * SUBWM + TM * idyInWave + i
  A_col_store = A_col[iterWave*TM + i].store(As[k*BM + index].load(barrier), iterWave, i)

  # do the GEMM math
  iterWaveM = UOp.range(dtypes.int, nbIterWaveM, 8)
  iterWaveN = UOp.range(dtypes.int, nbIterWaveN, 9)
  yt = UOp.range(dtypes.int, TM, 10)
  xt = UOp.range(dtypes.int, TN, 11)
  x = iterWaveN * TN + xt
  y = iterWaveM * TM + yt
  c_regs_idx = c_regs[y * TN * nbIterWaveN + x]
  sink = c_regs_idx.store(c_regs_idx.load(init_store) + A_col[y].load(A_col_store) * B_row[x].load(B_row_store),
                          iterWaveM, iterWaveN, yt, xt, k, kId_range)

  # store c_regs into c
  iterWaveM = UOp.range(dtypes.int, nbIterWaveM, 12)
  iterWaveN = UOp.range(dtypes.int, nbIterWaveN, 13)
  yt = UOp.range(dtypes.int, TM, 14)
  xt = UOp.range(dtypes.int, TN, 15)
  xOut = blockIdx_x * BN + waveIdx * WN + iterWaveN * SUBWN + TN * idxInWave
  yOut = blockIdx_y * BM + waveIdy * WM + iterWaveM * SUBWM + TM * idyInWave
  indexC = N * (yOut + yt) + xOut + xt
  sink = c[indexC].store(c_regs[TN * nbIterWaveN * (iterWaveM * TM + yt) + (iterWaveN * TN + xt)].load(sink),
                         iterWaveM, iterWaveN, yt, xt)

  return sink.sink(arg=KernelInfo(name="tinygemm"))

if __name__ == "__main__":
  hprg = hl_spec_kernel3() if getenv("HL") else hand_spec_kernel3()
  prg = get_program(hprg, Device.default.renderer)
  print(prg.src)
  hrunner = CompiledRunner(prg)

  a = Tensor.randn(N, N).realize()
  b = Tensor.randn(N, N).realize()
  hc = Tensor.zeros(N, N).contiguous().realize()

  GlobalCounters.reset()
  with Context(DEBUG=2):
    for _ in range(run_count): tc = (a@b).realize()

  GlobalCounters.reset()
  ei = ExecItem(hrunner, [a.uop.buffer, b.uop.buffer, hc.uop.buffer])
  with Context(DEBUG=2):
    for _ in range(run_count): ei.run(wait=True)
  err = (hc-tc).square().mean().item()
  print(f"hrunner {err}")
  if err > 1e-06: raise RuntimeError("matmul is wrong!")
