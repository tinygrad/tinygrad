# ruff: noqa: E501
from tinygrad import dtypes
from tinygrad.ops import UOp, Ops
from tinygrad.codegen.kernel import Kernel#, Opt, OptOps
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
from tinygrad.engine.realize import CompiledRunner
from tinygrad.engine.search import bufs_from_lin

ast = UOp(Ops.SINK, dtypes.void, arg=None, src=(
  UOp(Ops.STORE, dtypes.void, arg=None, src=(
    UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(11), arg=0, src=()),
    UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 1, 1, 11), strides=(0, 0, 0, 1), offset=0, mask=None, contiguous=True),)), src=()),
    UOp(Ops.REDUCE_AXIS, dtypes.float, arg=(Ops.ADD, (1, 2)), src=(
      UOp(Ops.MUL, dtypes.float, arg=None, src=(
        UOp(Ops.ADD, dtypes.float, arg=None, src=(
          UOp(Ops.ADD, dtypes.float, arg=None, src=(
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(25516392), arg=1, src=()),
              UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 76, 30522, 11), strides=(0, 30522, 1, 2319672), offset=0, mask=None, contiguous=False),)), src=()),)),
            UOp(Ops.MUL, dtypes.float, arg=None, src=(
              UOp(Ops.LOAD, dtypes.float, arg=None, src=(
                UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(836), arg=2, src=()),
                x13:=UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 76, 30522, 11), strides=(0, 1, 0, 76), offset=0, mask=None, contiguous=False),)), src=()),)),
              x14:=UOp(Ops.CONST, dtypes.float, arg=-1.0, src=(
                x15:=UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 76, 30522, 11), strides=(0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),
          UOp(Ops.MUL, dtypes.float, arg=None, src=(
            UOp(Ops.LOAD, dtypes.float, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(836), arg=3, src=()),
               x13,)),
             x14,)),)),
        UOp(Ops.CAST, dtypes.float, arg=None, src=(
          UOp(Ops.AND, dtypes.bool, arg=None, src=(
            UOp(Ops.CMPNE, dtypes.bool, arg=None, src=(
              UOp(Ops.CMPNE, dtypes.bool, arg=None, src=(
                UOp(Ops.LOAD, dtypes.int, arg=None, src=(
                  UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(30522), arg=4, src=()),
                  UOp(Ops.VIEW, dtypes.void, arg=ShapeTracker(views=(View(shape=(1, 76, 30522, 11), strides=(0, 0, 1, 0), offset=0, mask=None, contiguous=False),)), src=()),)),
                UOp(Ops.LOAD, dtypes.int, arg=None, src=(
                  UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(836), arg=5, src=()),
                   x13,)),)),
              UOp(Ops.CONST, dtypes.bool, arg=True, src=(
                 x15,)),)),
            UOp(Ops.LOAD, dtypes.bool, arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.bool.ptr(836), arg=6, src=()),
               x13,)),)),)),)),)),)),))

# opts = [Opt(op=OptOps.UNROLL, axis=0, arg=2)]
opts = []

k = Kernel(ast)
for opt in opts: k.apply_opt(opt)
bufs = bufs_from_lin(k)

prg = CompiledRunner(k.to_program())

for i in range(10):
  speed = prg(bufs, var_vals={}, wait=True)
  print(f"kernel time: {speed*1e3:.2f} ms")