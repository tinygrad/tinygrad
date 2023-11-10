import unittest
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.ops import Device, Interpreted
from test.external.fuzz_linearizer import fuzz_linearizer

# stuff needed to unpack a kernel
from tinygrad.ops import LazyOp, TernaryOps, BinaryOps, UnaryOps, ReduceOps, BufferOps, MemBuffer, ConstBuffer
from tinygrad.helpers import dtypes
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
from tinygrad.shape.symbolic import Variable
inf, nan = float('inf'), float('nan')

class TestLinearizerFailures(unittest.TestCase):
  @unittest.skip("this is currently failing")
  def test_failure_1(self):
    ast = LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BufferOps.MEM, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(32, 16, 16), strides=(16, 1, 0), offset=0, mask=None, contiguous=False),)))),), arg=(32, 16, 1)), LazyOp(op=BufferOps.MEM, src=(), arg=MemBuffer(idx=2, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(32, 16, 1), strides=(0, 1, 0), offset=0, mask=None, contiguous=False),))))), arg=None), LazyOp(op=BufferOps.MEM, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(32, 16, 1), strides=(16, 1, 0), offset=0, mask=None, contiguous=True),))))), arg=None)
    lin = Linearizer(ast)
    prg = Device[Device.DEFAULT].to_program(lin)

  # NOTE: test cases from fuzzer run. if you fixed something and it no longer fails, remove the test case / backend.

  @unittest.skipUnless(isinstance(Device[Device.DEFAULT], Interpreted), "fails on Interpreted")
  def test_failure_2(self):
    ast = LazyOp(op=ReduceOps.MAX, src=(LazyOp(op=BufferOps.MEM, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(32, 2, 111, 27), strides=(6160, 3080, 28, 1), offset=0, mask=((0, 32), (0, 2), (0, 110), (0, 27)), contiguous=False), View(shape=(32, 2, 37, 9, 2, 2), strides=(5994, 2997, 81, 3, 27, 1), offset=0, mask=None, contiguous=False))))),), arg=(32, 2, 37, 9, 1, 1))
    lin = Linearizer(ast)
    assert fuzz_linearizer(lin) != "PASS"

  @unittest.skipUnless(Device.DEFAULT in ["METAL", "GPU", "LLVM"], "fails on these backends")
  def test_failure_3(self):
    ast = LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BufferOps.MEM, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(32, 8, 16, 16), strides=(2048, 256, 16, 1), offset=0, mask=None, contiguous=True),)))),), arg=(32, 8, 16, 1))
    lin = Linearizer(ast)
    # METAL: AssertionError: Error Domain=AGXMetalG13X Code=3 "Threadgroup memory size (65536) exceeds the maximum threadgroup memory allowed (32768)" UserInfo={NSLocalizedDescription=Threadgroup memory size (65536) exceeds the maximum threadgroup memory allowed (32768)}
    assert fuzz_linearizer(lin) != "PASS"

  @unittest.skipUnless(Device.DEFAULT in ["METAL", "LLVM"], "fails on these backends")
  def test_failure_4(self):
    ast = LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BufferOps.MEM, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(1, 1, 1, 4, 1, 12, 2, 29), strides=(0, 0, 0, 2, 0, 216, 1, 8), offset=0, mask=((0, 1), (0, 1), (0, 1), (0, 4), (0, 1), (0, 11), (0, 2), (0, 27)), contiguous=False), View(shape=(1, 1, 1, 4, 22, 84), strides=(0, 0, 0, 696, 58, 1), offset=0, mask=((0, 1), (0, 1), (0, 1), (0, 4), (0, 12), (0, 58)), contiguous=False), View(shape=(1, 1, 1, 4, 2, 11, 3, 28), strides=(0, 0, 0, 1848, 924, 84, 28, 1), offset=0, mask=None, contiguous=True))))),), arg=(1, 1, 1, 4, 1, 11, 1, 28))
    lin = Linearizer(ast)
    # related to OptOps.NOLOCALS
    # IndexError: list index out of range
    assert fuzz_linearizer(lin) != "PASS"

  @unittest.skipUnless(Device.DEFAULT in ["CLANG", "LLVM"], "fails on these backends")
  def test_failure_5(self):
    ast = LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=0.1464405059814453, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(2, 1, 4, 1, 3, 1, 4, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)))), LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=1.0, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(2, 1, 4, 1, 3, 1, 4, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None), LazyOp(op=BufferOps.MEM, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(2, 1, 4, 1, 3, 1, 4, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None), LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=0.1464405059814453, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(2, 1, 4, 1, 3, 1, 4, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)))), LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=1.0, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(2, 1, 4, 1, 3, 1, 4, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None), LazyOp(op=BufferOps.MEM, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(2, 1, 4, 1, 3, 1, 4, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None)), arg=None),), arg=(1, 1, 1, 1, 1, 1, 1, 1))
    # EXEC_ERROR, it has no global_size
    lin = Linearizer(ast)
    assert fuzz_linearizer(lin) != "PASS"


if __name__ == '__main__':
  unittest.main()
