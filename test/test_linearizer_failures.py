import unittest
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.features.search import Opt, OptOps
from tinygrad import Device
from tinygrad.helpers import OSX, CI
from test.external.fuzz_linearizer import run_linearizer

# stuff needed to unpack a kernel
from tinygrad.ops import LazyOp, BinaryOps, UnaryOps, ReduceOps, BufferOps, MemBuffer, ConstBuffer, get_lazyop_info
from tinygrad.helpers import dtypes
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View
inf, nan = float('inf'), float('nan')

def helper_test_lin(lin: Linearizer, opts, failed_platforms):
  for opt in opts:
    try:
      lin.apply_opt(opt)
    except AssertionError:
      # it's considered fixed if we invalidated the opts
      assert Device.DEFAULT not in failed_platforms
  if Device.DEFAULT not in failed_platforms:
    assert run_linearizer(lin) == "PASS"
  else:
    assert run_linearizer(lin) != "PASS"

def helper_add_store(op):
  info = get_lazyop_info(op)
  return LazyOp(BufferOps.STORE, (op, ), MemBuffer(0, info.dtype, ShapeTracker.from_shape(info.shape)))

@unittest.skipIf(CI and Device.DEFAULT=="CUDA", "failed on CUDA CI")
class TestLinearizerFailures(unittest.TestCase):
  def test_failure_1(self):
    ast = LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(32, 16, 16), strides=(16, 1, 0), offset=0, mask=None, contiguous=False),)))),), arg=(32, 16, 1)), LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=2, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(32, 16, 1), strides=(0, 1, 0), offset=0, mask=None, contiguous=False),))))), arg=None), LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(32, 16, 1), strides=(16, 1, 0), offset=0, mask=None, contiguous=True),))))), arg=None)
    ast = helper_add_store(ast)
    helper_test_lin(Linearizer(ast), [], failed_platforms=["CLANG"])

  def test_failure_2(self):
    ast = LazyOp(op=ReduceOps.MAX, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(32, 2, 111, 27), strides=(6160, 3080, 28, 1), offset=0, mask=((0, 32), (0, 2), (0, 110), (0, 27)), contiguous=False), View(shape=(32, 2, 37, 9, 2, 2), strides=(5994, 2997, 81, 3, 27, 1), offset=0, mask=None, contiguous=False))))),), arg=(32, 2, 37, 9, 1, 1))
    opts = [Opt(op=OptOps.LOCAL, axis=0, amt=32)]
    ast = helper_add_store(ast)
    helper_test_lin(Linearizer(ast), opts, failed_platforms=["CPU", "TORCH"])

  @unittest.skipIf(CI and Device.DEFAULT=="METAL", "behaves differently on METAL CI")
  def test_failure_3(self):
    ast = LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(32, 8, 16, 16), strides=(2048, 256, 16, 1), offset=0, mask=None, contiguous=True),)))),), arg=(32, 8, 16, 1))
    opts = [Opt(op=OptOps.GROUP, axis=0, amt=4), Opt(op=OptOps.UPCAST, axis=0, amt=4), Opt(op=OptOps.UPCAST, axis=0, amt=2), Opt(op=OptOps.UNROLL, axis=1, amt=0), Opt(op=OptOps.UPCAST, axis=0, amt=4), Opt(op=OptOps.LOCAL, axis=0, amt=2), Opt(op=OptOps.LOCAL, axis=0, amt=2), Opt(op=OptOps.UPCAST, axis=1, amt=0), Opt(op=OptOps.LOCAL, axis=0, amt=32)]
    # METAL: AssertionError: Error Domain=AGXMetalG13X Code=3 "Threadgroup memory size (65536) exceeds the maximum threadgroup memory allowed (32768)" UserInfo={NSLocalizedDescription=Threadgroup memory size (65536) exceeds the maximum threadgroup memory allowed (32768)}
    ast = helper_add_store(ast)
    helper_test_lin(Linearizer(ast), opts, failed_platforms=["METAL", "GPU", "CUDA"])

  def test_failure_4(self):
    ast = LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(1, 1, 1, 4, 1, 12, 2, 29), strides=(0, 0, 0, 2, 0, 216, 1, 8), offset=0, mask=((0, 1), (0, 1), (0, 1), (0, 4), (0, 1), (0, 11), (0, 2), (0, 27)), contiguous=False), View(shape=(1, 1, 1, 4, 22, 84), strides=(0, 0, 0, 696, 58, 1), offset=0, mask=((0, 1), (0, 1), (0, 1), (0, 4), (0, 12), (0, 58)), contiguous=False), View(shape=(1, 1, 1, 4, 2, 11, 3, 28), strides=(0, 0, 0, 1848, 924, 84, 28, 1), offset=0, mask=None, contiguous=True))))),), arg=(1, 1, 1, 4, 1, 11, 1, 28))
    opts = [Opt(op=OptOps.LOCAL, axis=2, amt=4), Opt(op=OptOps.UPCAST, axis=0, amt=2), Opt(op=OptOps.UPCAST, axis=0, amt=0), Opt(op=OptOps.LOCAL, axis=2, amt=2), Opt(op=OptOps.UPCAST, axis=3, amt=0), Opt(op=OptOps.UPCAST, axis=2, amt=0), Opt(op=OptOps.UNROLL, axis=0, amt=0), Opt(op=OptOps.UPCAST, axis=1, amt=0), Opt(op=OptOps.NOLOCALS, axis=None, amt=None)]
    # related to OptOps.NOLOCALS
    # IndexError: list index out of range
    ast = helper_add_store(ast)
    helper_test_lin(Linearizer(ast), opts, failed_platforms=["METAL"])

  def test_failure_5(self):
    ast = LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=0.1464405059814453, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(2, 1, 4, 1, 3, 1, 4, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)))), LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=1.0, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(2, 1, 4, 1, 3, 1, 4, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None), LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(2, 1, 4, 1, 3, 1, 4, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None), LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=0.1464405059814453, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(2, 1, 4, 1, 3, 1, 4, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)))), LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=1.0, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(2, 1, 4, 1, 3, 1, 4, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None), LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(2, 1, 4, 1, 3, 1, 4, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),))))), arg=None)), arg=None),), arg=(1, 1, 1, 1, 1, 1, 1, 1))
    opts = [Opt(op=OptOps.UNROLL, axis=0, amt=4), Opt(op=OptOps.UNROLL, axis=0, amt=0)]
    # EXEC_ERROR, it has no global_size
    ast = helper_add_store(ast)
    helper_test_lin(Linearizer(ast), opts, failed_platforms=[])

  def test_failure_6(self):
    ast = LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=-1.0, dtype=dtypes.int32, st=ShapeTracker(views=(View(shape=(11, 19), strides=(0, 0), offset=0, mask=((0, 11), (9, 19)), contiguous=False), View(shape=(10, 10), strides=(1, 20), offset=0, mask=None, contiguous=False))))),), arg=(10, 1)), LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=10.0, dtype=dtypes.int32, st=ShapeTracker(views=(View(shape=(10, 1), strides=(0, 0), offset=0, mask=None, contiguous=False),))))), arg=None)
    opts = [Opt(op=OptOps.UPCAST, axis=0, amt=2), Opt(op=OptOps.UPCAST, axis=0, amt=0)]
    # COMPILE FAILED, KeyError: UOps.CONST
    ast = helper_add_store(ast)
    helper_test_lin(Linearizer(ast), opts, failed_platforms=[])

  @unittest.skipIf(Device.DEFAULT=="LLVM", "Segmentation fault")
  def test_failure_7(self):
    ast = LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(512, 32, 6, 8, 4, 6, 8, 4), strides=(2048, 64, 6291456, 8, 0, 1048576, 1, 0), offset=0, mask=((0, 512), (0, 32), (0, 6), (0, 8), (0, 1), (0, 6), (0, 8), (0, 1)), contiguous=False), View(shape=(512, 32, 6, 35, 6, 35), strides=(1179648, 36864, 6144, 192, 32, 1), offset=0, mask=((0, 512), (0, 32), (0, 6), (0, 32), (0, 6), (0, 32)), contiguous=False), View(shape=(512, 32, 238, 238), strides=(1411200, 44100, 210, 1), offset=0, mask=((0, 512), (0, 32), (0, 210), (0, 210)), contiguous=False), View(shape=(512, 32, 7, 34, 7, 34), strides=(1812608, 56644, 8092, 238, 34, 1), offset=0, mask=None, contiguous=True))))),), arg=(512, 32, 1, 34, 1, 34))
    opts = [Opt(op=OptOps.UPCAST, axis=0, amt=4)]
    # test/test_linearizer_failures.py Fatal Python error: Segmentation fault
    ast = helper_add_store(ast)
    helper_test_lin(Linearizer(ast), opts, failed_platforms=["LLVM"])

  @unittest.skipIf(Device.DEFAULT=="LLVM" and not OSX, "Segmentation fault on ubuntu")
  def test_failure_8(self):
    ast = LazyOp(op=UnaryOps.SQRT, src=(LazyOp(op=BinaryOps.DIV, src=(LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=1.0, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(1, 1, 1), strides=(0, 0, 0), offset=0, mask=None, contiguous=True),)))), LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=ReduceOps.SUM, src=(LazyOp(op=BinaryOps.MUL, src=(LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.half, st=ShapeTracker(views=(View(shape=(1, 1, 4096), strides=(0, 0, 1), offset=0, mask=None, contiguous=True),)))), LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=2, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(1, 1, 4096), strides=(0, 0, 1), offset=0, mask=None, contiguous=True),))))), arg=None), LazyOp(op=BinaryOps.ADD, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.half, st=ShapeTracker(views=(View(shape=(1, 1, 4096), strides=(0, 0, 1), offset=0, mask=None, contiguous=True),)))), LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=2, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(1, 1, 4096), strides=(0, 0, 1), offset=0, mask=None, contiguous=True),))))), arg=None)), arg=None),), arg=(1, 1, 1)), LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=0.000244140625, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(1, 1, 1), strides=(0, 0, 0), offset=0, mask=None, contiguous=True),))))), arg=None), LazyOp(op=BufferOps.CONST, src=(), arg=ConstBuffer(val=1e-06, dtype=dtypes.float, st=ShapeTracker(views=(View(shape=(1, 1, 1), strides=(0, 0, 0), offset=0, mask=None, contiguous=True),))))), arg=None)), arg=None),), arg=None)
    opts = [Opt(op=OptOps.UNROLL, axis=0, amt=4), Opt(op=OptOps.UNROLL, axis=0, amt=4), Opt(op=OptOps.UNROLL, axis=0, amt=4), Opt(op=OptOps.UNROLL, axis=0, amt=4)]
    # fatal error: bracket nesting level exceeded maximum of 256
    # note: use -fbracket-depth=N to increase maximum nesting level
    ast = helper_add_store(ast)
    helper_test_lin(Linearizer(ast), opts, failed_platforms=["CLANG", "METAL"])

if __name__ == '__main__':
  unittest.main()
