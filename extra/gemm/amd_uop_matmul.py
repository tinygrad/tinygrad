from tinygrad import Tensor, Device, Context, GlobalCounters, dtypes
from tinygrad.helpers import prod, unwrap
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.opt.kernel import AxisType
from tinygrad.engine.realize import CompiledRunner, ExecItem, get_program
from tinygrad.uop.ops import graph_rewrite, PatternMatcher, UPat, Ops, UOp, GroupOp
from tinygrad.shape.shapetracker import ShapeTracker, strides_for_shape
from tinygrad.kernelize.kernelize import merge_views
from tinygrad.shape.view import View

N = 4096
run_count = 5

# change reduceop axes and input ShapeTrackers, view gets replaced with a reshape.
# src->r->view  -->   src->view->r
def swizzle_reduceop(src:UOp, r:UOp, view:UOp):
  if r.tag is not None: return None
  # confirm the input is in order
  # TODO: replace this with a UOp that allows for nothing else then remove this
  permute = tuple(i for i in range(len(src.shape)) if i not in r.axis_arg)+r.axis_arg
  assert permute == tuple(range(len(permute))), f"reduce axis must already be in order, {permute} isn't"

  # append the reduce shape to each of the views
  reduce_count = len(r.axis_arg)
  prshape = prod(rshape:=src.shape[-reduce_count:])
  rstrides = strides_for_shape(rshape)
  nv = [View.create(v.shape[:-reduce_count]+rshape, tuple(x*prshape for x in v.strides[:-reduce_count])+rstrides, v.offset*prshape,
                    v.mask[:-reduce_count]+tuple((0,s) for s in rshape) if v.mask is not None else None) for v in unwrap(view.st).views]

  # no reshape required with shrinking REDUCE_AXIS
  return UOp(Ops.REDUCE_AXIS, r.dtype, (src.view(ShapeTracker(tuple(nv))),),
             (r.arg[0], tuple(range(len(view.shape)-reduce_count, len(view.shape)))))

early_view_left = merge_views+PatternMatcher([
  # view before elementwise and buffer ops
  (UPat(Ops.VIEW, src=(UPat({*GroupOp.ALU, Ops.CAST, Ops.BITCAST, Ops.BIND, Ops.VALID, Ops.STORE, Ops.LOAD}, name="e"),), name="view"),
   lambda e,view: e.replace(src=tuple(s.view(view.st) for s in e.src)) if e.tag is None else None),
  # push a non contiguous ShapeTracker through reduceop
  (UPat(Ops.VIEW, src=(UPat(Ops.REDUCE_AXIS, src=(UPat.var("src"),), name="r"),), name="view"), swizzle_reduceop),
])

def hand_spec():
  # Block Tile size  . 128x128
  # Thread Tile size . 4x4
  # Wave Tile size   . 128x32
  # A wave is        . 8x4
  # ────── problem size and tiling params (mirror the C kernel) ───────────────────
  BK = 8             # depth of K-tile
  BN = BM = 128      # block-tile (output) sizes
  # the real thread is 16x8 = 128 regs
  TM = 4
  nbIterWaveM = 2
  TN = 4
  nbIterWaveN = 4

  # ────── shared-memory tile sizes (unchanged) ───────────────────────────────────
  LDS_A_SZ = BK * BM          # 1024 floats
  LDS_B_SZ = BK * BN          # 1024 floats

  bC = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(N*N), arg=0)   # output C
  bA = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(N*N), arg=1)   # input A
  bB = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(N*N), arg=2)   # input B

  # TODO: this should not be a string, just a number
  lAs = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(LDS_A_SZ, local=True), arg="As")
  lBs = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(LDS_B_SZ, local=True), arg="Bs")

  s0 = ShapeTracker.from_shape((N, N, N), (N, 0, 1))
  s1 = ShapeTracker.from_shape((N, N, N), (0, 1, N))
  s2 = ShapeTracker.from_shape((N, N, 1), (N, 1, 0))

  ls0 = ShapeTracker.from_shape((BM, BK))
  ls1 = ShapeTracker.from_shape((BN, BK))

  axis_types = [AxisType.GLOBAL, AxisType.UPCAST, AxisType.LOCAL, AxisType.UPCAST,
                AxisType.GLOBAL, AxisType.UPCAST, AxisType.LOCAL, AxisType.UPCAST,
                AxisType.REDUCE, AxisType.UNROLL]
  #axis_types = [AxisType.LOOP]*8 + [AxisType.REDUCE]*2
  s0 = s0.reshape((N//BM, nbIterWaveM, (BM//nbIterWaveM)//TM, TM, N//BN, nbIterWaveN, (BN//nbIterWaveN)//TN, TN, N//BK, BK))
  s1 = s1.reshape((N//BM, nbIterWaveM, (BM//nbIterWaveM)//TM, TM, N//BN, nbIterWaveN, (BN//nbIterWaveN)//TN, TN, N//BK, BK))
  s2 = s2.reshape((N//BM, nbIterWaveM, (BM//nbIterWaveM)//TM, TM, N//BN, nbIterWaveN, (BN//nbIterWaveN)//TN, TN, 1, 1))

  ls0 = ls0.reshape((1, nbIterWaveM, (BM//nbIterWaveM)//TM, TM, 1, 1, 1, 1, 1, BK)).expand(s0.shape)
  ls1 = ls1.reshape((1, 1, 1, 1, 1, nbIterWaveN, (BN//nbIterWaveN)//TN, TN, 1, BK)).expand(s1.shape)
  assert ls0.real_size() == LDS_A_SZ
  assert ls1.real_size() == LDS_B_SZ

  permaxis = []
  for axis_order in [AxisType.GLOBAL, AxisType.LOCAL, AxisType.LOOP, AxisType.GROUP_REDUCE, AxisType.REDUCE, AxisType.UNROLL, AxisType.UPCAST]:
    permaxis += [i for i,a in enumerate(axis_types) if a == axis_order]
  axis_types = [axis_types[x] for x in permaxis]
  s0, s1, s2, ls0, ls1 = [x.permute(tuple(permaxis)) for x in [s0, s1, s2, ls0, ls1]]
  print(axis_types)

  lw0, lr0 = ls0, ls0
  lw1, lr1 = ls1, ls1

  print(ls0)
  print(ls1)

  # global_load-local_store optimizations. you have to apply the same permutation to both sides of the load/store
  #s0 = s0.permute((0,1,2,5,4,3,6,7))
  #lw0 = lw0.permute((0,1,2,5,4,3,6,7))
  #s1 = s1.permute((0,1,6,3,4,5,2,7))
  #lw1 = lw1.permute((0,1,6,3,4,5,2,7))
  #print(lw0)
  #print(lw1)

  # loads and stores
  bs0 = bA.view(s0).load()
  bs1 = bB.view(s1).load()
  bs0 = lAs.view(lr0).load(lAs.view(lw0).store(bs0))
  bs1 = lBs.view(lr1).load(lBs.view(lw1).store(bs1))

  mat = (bs0 * bs1).r(Ops.ADD, tuple([i for i,a in enumerate(axis_types) if a in (AxisType.REDUCE, AxisType.UNROLL)]), permute=False)
  st = bC.view(s2).store(mat)

  global_dims = sum([1 for a in axis_types if a == AxisType.GLOBAL])
  local_dims = sum([1 for a in axis_types if a == AxisType.LOCAL])
  upcasted_dims = sum([1 for a in axis_types if a in (AxisType.UPCAST, AxisType.UNROLL)])
  ast = st.sink(arg=KernelInfo(global_dims=global_dims, local_dims=local_dims, upcasted=upcasted_dims, name="tinygemm"))
  ast = graph_rewrite(ast, merge_views)
  prg = get_program(ast, Device.default.renderer)
  print(prg.src)
  return prg


if __name__ == "__main__":
  hprg = hand_spec()
  hrunner = CompiledRunner(hprg)

  a = Tensor.randn(N, N).realize()
  b = Tensor.randn(N, N).realize()
  hc = Tensor.zeros(N, N).contiguous().realize()

  GlobalCounters.reset()
  with Context(DEBUG=2, BEAM=4):
    for _ in range(run_count): tc = (a@b).realize()

  GlobalCounters.reset()
  ei = ExecItem(hrunner, [hc.uop.buffer, a.uop.buffer, b.uop.buffer])
  with Context(DEBUG=2):
    for _ in range(run_count): ei.run(wait=True)
  print(f"hrunner {(hc-tc).square().mean().item()}")
