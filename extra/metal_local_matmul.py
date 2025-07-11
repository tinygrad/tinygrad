import numpy as np
from tinygrad import Tensor, Context, dtypes, Device
from tinygrad.opt.kernel import Opt, OptOps
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.engine.realize import get_program, CompiledRunner, ExecItem
from tinygrad.uop.ops import graph_rewrite
from tinygrad.kernelize.kernelize import merge_views

N = 1024
L = 8

# direct UOp syntax

if __name__ == "__main__":
  b0 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(N*N), arg=0)
  b1 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(N*N), arg=1)
  b2 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(N*N), arg=2)

  # TODO: this should not be a string, just a number
  l1 = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(L*L, local=True), arg=f"temp0")
  l2 = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(L*L, local=True), arg=f"temp1")

  # global
  gv1 = ShapeTracker.from_shape((N//L, N//L, L, L, N//L, L), (N*L, 0, N, 0, L,   1))
  gv2 = ShapeTracker.from_shape((N//L, N//L, L, L, N//L, L), (0,   L, 0, 1, N*L, N))
  # local-write
  lwv1 = l1.view(ShapeTracker.from_shape((N//L, N//L, L, L, N//L, L), (0, 0, L, 0, 0, 1)))
  lwv2 = l2.view(ShapeTracker.from_shape((N//L, N//L, L, L, N//L, L), (0, 0, 0, 1, 0, L)))
  # local-read
  lrv1 = l1.view(ShapeTracker.from_shape((N//L, N//L, L, L, N//L, L), (0, 0, L, 0, 0, 1)))
  lrv2 = l2.view(ShapeTracker.from_shape((N//L, N//L, L, L, N//L, L), (0, 0, 0, 1, 0, L)))

  # global_load-local_store optimizations. you have to apply the same permutation to both sides of the load/store
  gv1 = gv1.permute((0,1,2,5,4,3))
  lwv1 = lwv1.permute((0,1,2,5,4,3))
  gv2 = gv2.permute((0,1,2,5,4,3))
  lwv2 = lwv2.permute((0,1,2,5,4,3))

  # local_store-local_load optimizations. you have to apply the same permutation to both sides of the load/store
  lwv2 = lwv2.permute((0,1,2,5,4,3))
  lrv2 = lrv2.permute((0,1,2,5,4,3))

  bs1 = b1.view(gv1).load()
  bs2 = b2.view(gv2).load()
  lbs1 = lrv1.load(lwv1.store(bs1))
  lbs2 = lrv2.load(lwv2.store(bs2))
  #lbs1, lbs2 = bs1, bs2

  mat = (lbs1*lbs2).r(Ops.ADD, (4, 5))
  st =  b0.view(ShapeTracker.from_shape((N//L, N//L, L, L, 1, 1), (N*L, L, N, 1, 0, 0))).store(mat)

  ast = st.sink(arg=KernelInfo(global_dims=2, local_dims=2, upcasted=1))
  ast = graph_rewrite(ast, merge_views)
  prg = get_program(ast, Device.default.renderer)
  print(prg.src)

  rng = np.random.default_rng()
  t0 = Tensor.empty(N, N, dtype=dtypes.float).realize()
  t1 = Tensor(na:=rng.random((N, N), dtype=np.float32)).realize()
  t2 = Tensor(nb:=rng.random((N, N), dtype=np.float32)).realize()

  # run the program
  ExecItem(CompiledRunner(prg), [x.uop.buffer.ensure_allocated() for x in [t0, t1, t2]]).run()
  np.testing.assert_allclose(t0.numpy(), na@nb)
