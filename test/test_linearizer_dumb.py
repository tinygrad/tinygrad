# ruff: noqa: E501
# tests where the Linearizer is doing something dumb
# like test_linearizer_failures, but they don't have to fail

import unittest
from tinygrad import Device, dtypes
from tinygrad.ops import LazyOp, BinaryOps, UnaryOps, ReduceOps, TernaryOps, BufferOps, MemBuffer, ConstBuffer, MetaOps # noqa: F401 # pylint: disable=unused-import
from tinygrad.shape.shapetracker import ShapeTracker, View
from tinygrad.engine.search import Opt, OptOps
from tinygrad.codegen.kernel import Kernel

class TestLinearizerDumb(unittest.TestCase):
  @unittest.skipUnless(Device.DEFAULT == "METAL", "only tested on METAL")
  def test_unmerged_ifs(self):
    ast = LazyOp(MetaOps.KERNEL, arg=None, src=(
      LazyOp(BufferOps.STORE, arg=MemBuffer(idx=0, dtype=dtypes.half, st=ShapeTracker(views=(View(shape=(64, 1, 512, 7, 7, 1, 1, 1), strides=(25088, 0, 49, 7, 1, 0, 0, 0), offset=0, mask=None, contiguous=True),))), src=(
        LazyOp(BinaryOps.MAX, arg=None, src=(
          LazyOp(BinaryOps.MUL, arg=None, src=(
            LazyOp(UnaryOps.CAST, arg=dtypes.half, src=(
              LazyOp(ReduceOps.SUM, arg=(5, 6, 7), src=(
                LazyOp(UnaryOps.CAST, arg=dtypes.float, src=(
                  LazyOp(BinaryOps.MUL, arg=None, src=(
                    LazyOp(BufferOps.LOAD, arg=MemBuffer(idx=1, dtype=dtypes.half, st=ShapeTracker(views=(View(shape=(1, 64, 1, 512, 4, 9, 4, 9), strides=(0, 25088, 0, 49, 0, 7, 0, 1), offset=-8, mask=((0, 1), (0, 64), (0, 1), (0, 512), (0, 4), (1, 8), (0, 4), (1, 8)), contiguous=False), View(shape=(64, 1, 512, 7, 7, 512, 3, 3), strides=(663552, 0, 0, 36, 1, 1296, 360, 10), offset=0, mask=None, contiguous=False)))), src=()),
                    LazyOp(BufferOps.LOAD, arg=MemBuffer(idx=2, dtype=dtypes.half, st=ShapeTracker(views=(View(shape=(64, 1, 512, 7, 7, 512, 3, 3), strides=(0, 0, 4608, 0, 0, 9, 3, 1), offset=0, mask=None, contiguous=False),))), src=()),)),)),)),)),
            LazyOp(BufferOps.CONST, arg=ConstBuffer(val=0.9999950000374996, dtype=dtypes.half, st=ShapeTracker(views=(View(shape=(64, 1, 512, 7, 7, 1, 1, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),))), src=()),)),
          LazyOp(BufferOps.CONST, arg=ConstBuffer(val=0.0, dtype=dtypes.half, st=ShapeTracker(views=(View(shape=(64, 1, 512, 7, 7, 1, 1, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),))), src=()),)),)),))
    opts = [Opt(op=OptOps.TC, axis=2, amt=2), Opt(op=OptOps.UPCAST, axis=2, amt=0), Opt(op=OptOps.UNROLL, axis=1, amt=0)]
    k = Kernel(ast, opts=Device["METAL"].renderer)
    k.required_optimizations()
    for opt in opts: k.apply_opt(opt)
    prg = k.to_program()
    prg.uops.print()
    print(prg.src)

if __name__ == '__main__':
  unittest.main()
