# ruff: noqa: E501
import os
os.environ["VIZ"] = "1"
os.environ["DEBUG"] = "2"
from tinygrad.uop.ops import UOp, Ops
from tinygrad.shape.shapetracker import ShapeTracker, View
from tinygrad.engine.realize import get_runner, ExecItem
from tinygrad import Tensor, Device, dtypes, Context

DEVICE = Device.DEFAULT

ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
  UOp(Ops.STORE, dtypes.void, arg=None, src=(
    UOp(Ops.VIEW, dtypes.float.ptr(6553600), arg=ShapeTracker(views=(View(shape=(512, 1, 32, 20, 20, 1, 1, 1), strides=(12800, 0, 400, 20, 1, 0, 0, 0), offset=0, mask=None, contiguous=True),)), src=(
      UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(6553600), arg=0, src=()),)),
    UOp(Ops.ADD, dtypes.float, arg=None, src=(
      UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (5, 6, 7)), src=(
        UOp(Ops.MUL, dtypes.float, arg=None, src=(
          UOp(Ops.LOAD, dtypes.float, arg=None, src=(
            UOp(Ops.VIEW, dtypes.float.ptr(9437184), arg=ShapeTracker(views=(View(shape=(512, 1, 32, 20, 20, 32, 5, 5), strides=(18432, 0, 0, 24, 1, 576, 24, 1), offset=0, mask=None, contiguous=False),)), src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(9437184), arg=1, src=()),)),)),
          UOp(Ops.LOAD, dtypes.float, arg=None, src=(
            UOp(Ops.VIEW, dtypes.float.ptr(25600), arg=ShapeTracker(views=(View(shape=(512, 1, 32, 20, 20, 32, 5, 5), strides=(0, 0, 800, 0, 0, 25, 5, 1), offset=0, mask=None, contiguous=False),)), src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(25600), arg=2, src=()),)),)),)),)),
      UOp(Ops.LOAD, dtypes.float, arg=None, src=(
        UOp(Ops.VIEW, dtypes.float.ptr(32), arg=ShapeTracker(views=(View(shape=(512, 1, 32, 20, 20, 1, 1, 1), strides=(0, 0, 1, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=(
          UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(32), arg=3, src=()),)),)),)),)),))

with Context(NOOPT=1):
  p1 = get_runner(DEVICE, ast)
p2 = get_runner(DEVICE, ast)

bufs = []
for b in ast.toposort():
  if b.op is Ops.DEFINE_GLOBAL:
    t = Tensor.empty(b.dtype.size, dtype=b.dtype.base)
    bufs.append(t.uop.buffer.allocate())

ExecItem(p1, bufs).run()
ExecItem(p2, bufs).run()
