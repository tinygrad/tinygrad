import unittest
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.features.search import Opt, OptOps
from tinygrad.ops import Device
from test.external.fuzz_linearizer import run_linearizer

# stuff needed to unpack a kernel
from tinygrad.ops import LazyOp, TernaryOps, BinaryOps, UnaryOps, ReduceOps, BufferOps, MemBuffer, ConstBuffer
from tinygrad.helpers import dtypes
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
from tinygrad.shape.symbolic import Variable
inf, nan = float('inf'), float('nan')

def helper_test_lin(lin: Linearizer, opts, fixed_platforms):
  for opt in opts:
    try:
      lin.apply_opt(opt)
    except AssertionError:
      # it's considered fixed if we invalidated the opts
      return Device.DEFAULT in fixed_platforms
  if Device.DEFAULT in fixed_platforms:
    return run_linearizer(lin) == "PASS"
  else:
    return run_linearizer(lin) != "PASS"

class TestLinearizerFailures(unittest.TestCase):
  @unittest.skipUnless(Device.DEFAULT in ["CPU", "TORCH", "CLANG", "METAL", "GPU", "LLVM"], "fails on these backends")
  def test_failure_1(self):
    ast = LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BufferOps.MEM, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(32, 16, 16), strides=(16, 1, 0), offset=0, mask=None, contiguous=False),)))),), arg=(32, 16, 1)), LazyOp(op=BufferOps.MEM, src=(), arg=MemBuffer(idx=2, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(32, 16, 1), strides=(0, 1, 0), offset=0, mask=None, contiguous=False),))))), arg=None), LazyOp(op=BufferOps.MEM, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(32, 16, 1), strides=(16, 1, 0), offset=0, mask=None, contiguous=True),))))), arg=None)
    assert helper_test_lin(Linearizer(ast), [], fixed_platforms=["CPU", "TORCH", "METAL", "GPU", "LLVM"])

  # NOTE: test cases from fuzzer run. if you fixed something and it no longer fails, add platform to fixed_platforms list in helper_test_lin().

  @unittest.skipUnless(Device.DEFAULT in ["CPU", "TORCH"], "fails on these backends")
  def test_failure_2(self):
    ast = LazyOp(op=ReduceOps.MAX, src=(LazyOp(op=BufferOps.MEM, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(32, 2, 111, 27), strides=(6160, 3080, 28, 1), offset=0, mask=((0, 32), (0, 2), (0, 110), (0, 27)), contiguous=False), View(shape=(32, 2, 37, 9, 2, 2), strides=(5994, 2997, 81, 3, 27, 1), offset=0, mask=None, contiguous=False))))),), arg=(32, 2, 37, 9, 1, 1))
    opts = [Opt(op=OptOps.LOCAL, axis=0, amt=32)]
    assert helper_test_lin(Linearizer(ast), opts, fixed_platforms=[])

  @unittest.skipUnless(Device.DEFAULT in ["METAL", "GPU", "LLVM"], "fails on these backends")
  def test_failure_3(self):
    ast = LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BufferOps.MEM, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(32, 8, 16, 16), strides=(2048, 256, 16, 1), offset=0, mask=None, contiguous=True),)))),), arg=(32, 8, 16, 1))
    opts = [Opt(op=OptOps.GROUP, axis=0, amt=4), Opt(op=OptOps.UPCAST, axis=0, amt=4), Opt(op=OptOps.UPCAST, axis=0, amt=2), Opt(op=OptOps.UNROLL, axis=1, amt=0), Opt(op=OptOps.UPCAST, axis=0, amt=4), Opt(op=OptOps.LOCAL, axis=0, amt=2), Opt(op=OptOps.LOCAL, axis=0, amt=2), Opt(op=OptOps.UPCAST, axis=1, amt=0), Opt(op=OptOps.LOCAL, axis=0, amt=32)]
    # METAL: AssertionError: Error Domain=AGXMetalG13X Code=3 "Threadgroup memory size (65536) exceeds the maximum threadgroup memory allowed (32768)" UserInfo={NSLocalizedDescription=Threadgroup memory size (65536) exceeds the maximum threadgroup memory allowed (32768)}
    assert helper_test_lin(Linearizer(ast), opts, fixed_platforms=["LLVM"])

  @unittest.skipUnless(Device.DEFAULT in ["METAL", "LLVM"], "fails on these backends")
  def test_failure_4(self):
    ast = LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BufferOps.MEM, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(1, 1, 1, 4, 1, 12, 2, 29), strides=(0, 0, 0, 2, 0, 216, 1, 8), offset=0, mask=((0, 1), (0, 1), (0, 1), (0, 4), (0, 1), (0, 11), (0, 2), (0, 27)), contiguous=False), View(shape=(1, 1, 1, 4, 22, 84), strides=(0, 0, 0, 696, 58, 1), offset=0, mask=((0, 1), (0, 1), (0, 1), (0, 4), (0, 12), (0, 58)), contiguous=False), View(shape=(1, 1, 1, 4, 2, 11, 3, 28), strides=(0, 0, 0, 1848, 924, 84, 28, 1), offset=0, mask=None, contiguous=True))))),), arg=(1, 1, 1, 4, 1, 11, 1, 28))
    opts = [Opt(op=OptOps.LOCAL, axis=2, amt=4), Opt(op=OptOps.UPCAST, axis=0, amt=2), Opt(op=OptOps.UPCAST, axis=0, amt=0), Opt(op=OptOps.LOCAL, axis=2, amt=2), Opt(op=OptOps.UPCAST, axis=3, amt=0), Opt(op=OptOps.UPCAST, axis=2, amt=0), Opt(op=OptOps.UNROLL, axis=0, amt=0), Opt(op=OptOps.UPCAST, axis=1, amt=0), Opt(op=OptOps.NOLOCALS, axis=None, amt=None)]
    # related to OptOps.NOLOCALS
    # IndexError: list index out of range
    assert helper_test_lin(Linearizer(ast), opts, fixed_platforms=["LLVM"])

  @unittest.skipUnless(Device.DEFAULT in ["CLANG", "LLVM"], "fails on these backends")
  def test_failure_5(self):
    ast = LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=0.1464405059814453, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(2, 1, 4, 1, 3, 1, 4, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)))), LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=1.0, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(2, 1, 4, 1, 3, 1, 4, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None), LazyOp(op=BufferOps.MEM, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(2, 1, 4, 1, 3, 1, 4, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None), LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=0.1464405059814453, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(2, 1, 4, 1, 3, 1, 4, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)))), LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=1.0, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(2, 1, 4, 1, 3, 1, 4, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None), LazyOp(op=BufferOps.MEM, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(2, 1, 4, 1, 3, 1, 4, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None)), arg=None),), arg=(1, 1, 1, 1, 1, 1, 1, 1))
    opts = [Opt(op=OptOps.UNROLL, axis=0, amt=4), Opt(op=OptOps.UNROLL, axis=0, amt=0)]
    # EXEC_ERROR, it has no global_size
    assert helper_test_lin(Linearizer(ast), opts, fixed_platforms=["CLANG", "LLVM"])


if __name__ == '__main__':
  unittest.main()
