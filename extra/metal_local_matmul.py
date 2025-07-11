import numpy as np
from tinygrad import Tensor, Context, dtypes, Device
from tinygrad.opt.kernel import Opt, OptOps
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.engine.realize import get_program, CompiledRunner, ExecItem

N = 1024
L = 8

# direct UOp syntax

if __name__ == "__main__":
  b0 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(N*N), arg=0)
  b1 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(N*N), arg=1)
  b2 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(N*N), arg=2)

  # TODO: this should not be a string, just a number
  l1 = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(L, local=True), arg=f"temp0")
  l2 = UOp(Ops.DEFINE_LOCAL, dtypes.float.ptr(L, local=True), arg=f"temp1")

  bs1 = b1.view(ShapeTracker.from_shape((N//L, N//L, L, L, N), (N*L, 0, N, 0, 1))).load()
  bs2 = b2.view(ShapeTracker.from_shape((N//L, N//L, L, L, N), (0,   L, 0, 1, N))).load()

  lv1 = l1.view(ShapeTracker.from_shape((N//L, N//L, L, L, N), (0, 0, 1, 0, 0)))
  lv2 = l2.view(ShapeTracker.from_shape((N//L, N//L, L, L, N), (0, 0, 0, 1, 0)))

  lbs1 = lv1.load(lv1.store(bs1))
  lbs2 = lv2.load(lv2.store(bs2))

  mat = (lbs1*lbs2).r(Ops.ADD, (4,))
  st =  b0.view(ShapeTracker.from_shape((N//L, N//L, L, L, 1), (N*L, L, N, 1, 0))).store(mat)

  ast = st.sink(arg=KernelInfo(global_dims=2, local_dims=2))

  prg = get_program(ast, Device.default.renderer)
  print(prg.src)

  rng = np.random.default_rng()
  t0 = Tensor.empty(N, N, dtype=dtypes.float).realize()
  t1 = Tensor(na:=rng.random((N, N), dtype=np.float32)).realize()
  t2 = Tensor(nb:=rng.random((N, N), dtype=np.float32)).realize()

  # run the program
  ExecItem(CompiledRunner(prg), [x.uop.buffer.ensure_allocated() for x in [t0, t1, t2]]).run()
  np.testing.assert_allclose(t0.numpy(), na@nb)
