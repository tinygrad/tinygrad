# ruff: noqa: E501
import unittest
from test.helpers import ast_const
from tinygrad import dtypes, Device
from tinygrad.helpers import CI
from tinygrad.codegen.kernel import Kernel
from tinygrad.engine.search import Opt, OptOps
from tinygrad.engine.search import time_linearizer, bufs_from_lin

# stuff needed to unpack a kernel
from tinygrad.ops import UOp, UOps, BinaryOps, UnaryOps
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View

def _test_overflow(ast, opts):
  lin = Kernel(ast)
  for opt in opts: lin.apply_opt(opt)
  lin.linearize()
  bufs = bufs_from_lin(lin)
  print(bufs)
  time_linearizer(lin, bufs)

# NOTE: if you want these to trigger, set launch bounds on HIP kernels
@unittest.skip("unneeded without launch bounds")
class TestLinearizerOverflow(unittest.TestCase):
  def test_overflow_1(self):
    ast = UOp(UOps.SINK, None, arg=None, src=(
      UOp(UOps.STORE, None, arg=None, src=(
        UOp(UOps.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(UOps.VIEW, None, arg=ShapeTracker(views=(View(shape=(64, 1, 64, 112, 112, 1, 1, 1), strides=(802816, 0, 12544, 112, 1, 0, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(UOps.ALU, dtypes.float, arg=BinaryOps.MAX, src=(
          UOp(UOps.ALU, dtypes.float, arg=BinaryOps.ADD, src=(
            UOp(UOps.ALU, dtypes.float, arg=BinaryOps.MUL, src=(
              UOp(UOps.ALU, dtypes.float, arg=BinaryOps.MUL, src=(
                UOp(UOps.ALU, dtypes.float, arg=BinaryOps.ADD, src=(
                  UOp(UOps.REDUCE_AXIS, dtypes.float, arg=(BinaryOps.ADD, (7, 6, 5)), src=(
                    UOp(UOps.ALU, dtypes.float, arg=BinaryOps.MUL, src=(
                      UOp(UOps.LOAD, dtypes.float, arg=None, src=(
                        UOp(UOps.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
                        UOp(UOps.VIEW, None, arg=ShapeTracker(views=(View(shape=(1, 64, 1, 3, 8, 230, 8, 230), strides=(0, 150528, 0, 50176, 0, 224, 0, 1), offset=-675, mask=((0, 1), (0, 64), (0, 1), (0, 3), (0, 8), (3, 227), (0, 8), (3, 227)), contiguous=False), View(shape=(64, 1, 64, 112, 112, 3, 7, 7), strides=(10156800, 0, 0, 3680, 2, 3385600, 425040, 231), offset=0, mask=None, contiguous=False))), src=()),)),
                      UOp(UOps.LOAD, dtypes.float, arg=None, src=(
                        UOp(UOps.DEFINE_GLOBAL, dtypes.float.ptr(), arg=2, src=()),
                        UOp(UOps.VIEW, None, arg=ShapeTracker(views=(View(shape=(64, 1, 64, 112, 112, 3, 7, 7), strides=(0, 0, 147, 0, 0, 49, 7, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),
                  x16:=ast_const(dtypes.float, 0.0, st_src=(
                    UOp(UOps.VIEW, None, arg=ShapeTracker(views=(View(shape=(64, 1, 64, 112, 112, 1, 1, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
                UOp(UOps.LOAD, dtypes.float, arg=None, src=(
                  UOp(UOps.DEFINE_GLOBAL, dtypes.float.ptr(), arg=3, src=()),
                  UOp(UOps.VIEW, None, arg=ShapeTracker(views=(View(shape=(64, 1, 64, 112, 112, 1, 1, 1), strides=(0, 0, 1, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
              UOp(UOps.ALU, dtypes.float, arg=UnaryOps.SQRT, src=(
                UOp(UOps.ALU, dtypes.float, arg=BinaryOps.MUL, src=(
                  x23:=ast_const(dtypes.float, 1.0, st_src=(
                    UOp(UOps.VIEW, None, arg=ShapeTracker(views=(View(shape=(64, 1, 64, 112, 112, 1, 1, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),
                  UOp(UOps.ALU, dtypes.float, arg=UnaryOps.RECIP, src=(
                    UOp(UOps.ALU, dtypes.float, arg=BinaryOps.ADD, src=(
                       x23,
                      ast_const(dtypes.float, 1e-05, st_src=(
                        UOp(UOps.VIEW, None, arg=ShapeTracker(views=(View(shape=(64, 1, 64, 112, 112, 1, 1, 1), strides=(0, 0, 0, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),)),)),
            UOp(UOps.LOAD, dtypes.float, arg=None, src=(
              UOp(UOps.DEFINE_GLOBAL, dtypes.float.ptr(), arg=4, src=()),
              UOp(UOps.VIEW, None, arg=ShapeTracker(views=(View(shape=(64, 1, 64, 112, 112, 1, 1, 1), strides=(0, 0, 1, 0, 0, 0, 0, 0), offset=0, mask=None, contiguous=False),)), src=()),)),)),
       x16,)),)),))
    opts = [Opt(op=OptOps.LOCAL, axis=3, amt=16), Opt(op=OptOps.LOCAL, axis=2, amt=16), Opt(op=OptOps.UPCAST, axis=1, amt=4), Opt(op=OptOps.UPCAST, axis=0, amt=4), Opt(op=OptOps.UPCAST, axis=2, amt=0)]
    _test_overflow(ast, opts)

  # From BEAM on hlb_cifar.py
  def test_overflow_2(self):
    ast = UOp(UOps.SINK, None, arg=None, src=(
      UOp(UOps.STORE, None, arg=None, src=(
        UOp(UOps.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(UOps.VIEW, None, arg=ShapeTracker(views=(View(shape=(512, 1, 64, 32, 32, 1, 1, 1), strides=(65536, 0, 1024, 32, 1, 0, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(UOps.REDUCE_AXIS, dtypes.float, arg=(BinaryOps.ADD, (7, 6, 5)), src=(
          UOp(UOps.ALU, dtypes.float, arg=BinaryOps.MUL, src=(
            UOp(UOps.LOAD, dtypes.float, arg=None, src=(
              UOp(UOps.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
              UOp(UOps.VIEW, None, arg=ShapeTracker(views=(View(shape=(1, 512, 1, 32, 4, 34, 4, 34), strides=(0, 32768, 0, 1024, 0, 32, 0, 1), offset=-33, mask=((0, 1), (0, 512), (0, 1), (0, 32), (0, 4), (1, 33), (0, 4), (1, 33)), contiguous=False), View(shape=(512, 1, 64, 32, 32, 32, 3, 3), strides=(591872, 0, 0, 136, 1, 18496, 4760, 35), offset=0, mask=None, contiguous=False))), src=()),)),
            UOp(UOps.LOAD, dtypes.float, arg=None, src=(
              UOp(UOps.DEFINE_GLOBAL, dtypes.float.ptr(), arg=2, src=()),
              UOp(UOps.VIEW, None, arg=ShapeTracker(views=(View(shape=(512, 1, 64, 32, 32, 32, 3, 3), strides=(0, 0, 288, 0, 0, 9, 3, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),))
    opts = [Opt(op=OptOps.LOCAL, axis=3, amt=16), Opt(op=OptOps.LOCAL, axis=2, amt=4), Opt(op=OptOps.UPCAST, axis=1, amt=4), Opt(op=OptOps.UPCAST, axis=2, amt=0), Opt(op=OptOps.UPCAST, axis=1, amt=4), Opt(op=OptOps.UNROLL, axis=0, amt=0)]
    _test_overflow(ast, opts)

  # from BEAM on default simple_conv.py (which is quite large):
  def test_overflow_3(self):
    ast = UOp(UOps.SINK, None, arg=None, src=(
      UOp(UOps.STORE, None, arg=None, src=(
        UOp(UOps.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(UOps.VIEW, None, arg=ShapeTracker(views=(View(shape=(16, 1, 128, 128, 128, 1, 1, 1), strides=(2097152, 0, 16384, 128, 1, 0, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(UOps.REDUCE_AXIS, dtypes.float, arg=(BinaryOps.ADD, (7, 6, 5)), src=(
          UOp(UOps.ALU, dtypes.float, arg=BinaryOps.MUL, src=(
            UOp(UOps.LOAD, dtypes.float, arg=None, src=(
              UOp(UOps.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
              UOp(UOps.VIEW, None, arg=ShapeTracker(views=(View(shape=(1, 16, 1, 128, 4, 130, 4, 130), strides=(0, 2097152, 0, 16384, 0, 128, 0, 1), offset=-129, mask=((0, 1), (0, 16), (0, 1), (0, 128), (0, 4), (1, 129), (0, 4), (1, 129)), contiguous=False), View(shape=(16, 1, 128, 128, 128, 128, 3, 3), strides=(34611200, 0, 0, 520, 1, 270400, 68120, 131), offset=0, mask=None, contiguous=False))), src=()),)),
            UOp(UOps.LOAD, dtypes.float, arg=None, src=(
              UOp(UOps.DEFINE_GLOBAL, dtypes.float.ptr(), arg=2, src=()),
              UOp(UOps.VIEW, None, arg=ShapeTracker(views=(View(shape=(16, 1, 128, 128, 128, 128, 3, 3), strides=(0, 0, 1152, 0, 0, 9, 3, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),))
    opts = [Opt(op=OptOps.LOCAL, axis=3, amt=16), Opt(op=OptOps.LOCAL, axis=2, amt=8), Opt(op=OptOps.UPCAST, axis=1, amt=4), Opt(op=OptOps.UPCAST, axis=3, amt=0), Opt(op=OptOps.UPCAST, axis=1, amt=2), Opt(op=OptOps.UPCAST, axis=2, amt=2)]
    _test_overflow(ast, opts)

  # from BEAM on BS=4 simple_conv.py:
  def test_overflow_4(self):
    ast = UOp(UOps.SINK, None, arg=None, src=(
      UOp(UOps.STORE, None, arg=None, src=(
        UOp(UOps.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(UOps.VIEW, None, arg=ShapeTracker(views=(View(shape=(4, 1, 128, 128, 128, 1, 1, 1), strides=(2097152, 0, 16384, 128, 1, 0, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(UOps.REDUCE_AXIS, dtypes.float, arg=(BinaryOps.ADD, (7, 6, 5)), src=(
          UOp(UOps.ALU, dtypes.float, arg=BinaryOps.MUL, src=(
            UOp(UOps.LOAD, dtypes.float, arg=None, src=(
              UOp(UOps.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
              UOp(UOps.VIEW, None, arg=ShapeTracker(views=(View(shape=(1, 4, 1, 128, 4, 130, 4, 130), strides=(0, 2097152, 0, 16384, 0, 128, 0, 1), offset=-129, mask=((0, 1), (0, 4), (0, 1), (0, 128), (0, 4), (1, 129), (0, 4), (1, 129)), contiguous=False), View(shape=(4, 1, 128, 128, 128, 128, 3, 3), strides=(34611200, 0, 0, 520, 1, 270400, 68120, 131), offset=0, mask=None, contiguous=False))), src=()),)),
            UOp(UOps.LOAD, dtypes.float, arg=None, src=(
              UOp(UOps.DEFINE_GLOBAL, dtypes.float.ptr(), arg=2, src=()),
              UOp(UOps.VIEW, None, arg=ShapeTracker(views=(View(shape=(4, 1, 128, 128, 128, 128, 3, 3), strides=(0, 0, 1152, 0, 0, 9, 3, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),))
    opts = [Opt(op=OptOps.UPCAST, axis=3, amt=4), Opt(op=OptOps.LOCAL, axis=3, amt=16), Opt(op=OptOps.UPCAST, axis=1, amt=4), Opt(op=OptOps.LOCAL, axis=2, amt=4), Opt(op=OptOps.UPCAST, axis=1, amt=2), Opt(op=OptOps.UPCAST, axis=2, amt=4)]
    _test_overflow(ast, opts)

  # from BEAM on BS=2 simple_conv.py:
  def test_overflow_5(self):
    ast = UOp(UOps.SINK, None, arg=None, src=(
      UOp(UOps.STORE, None, arg=None, src=(
        UOp(UOps.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(UOps.VIEW, None, arg=ShapeTracker(views=(View(shape=(2, 1, 128, 128, 128, 1, 1, 1), strides=(2097152, 0, 16384, 128, 1, 0, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(UOps.REDUCE_AXIS, dtypes.float, arg=(BinaryOps.ADD, (7, 6, 5)), src=(
          UOp(UOps.ALU, dtypes.float, arg=BinaryOps.MUL, src=(
            UOp(UOps.LOAD, dtypes.float, arg=None, src=(
              UOp(UOps.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
              UOp(UOps.VIEW, None, arg=ShapeTracker(views=(View(shape=(1, 2, 1, 128, 4, 130, 4, 130), strides=(0, 2097152, 0, 16384, 0, 128, 0, 1), offset=-129, mask=((0, 1), (0, 2), (0, 1), (0, 128), (0, 4), (1, 129), (0, 4), (1, 129)), contiguous=False), View(shape=(2, 1, 128, 128, 128, 128, 3, 3), strides=(34611200, 0, 0, 520, 1, 270400, 68120, 131), offset=0, mask=None, contiguous=False))), src=()),)),
            UOp(UOps.LOAD, dtypes.float, arg=None, src=(
              UOp(UOps.DEFINE_GLOBAL, dtypes.float.ptr(), arg=2, src=()),
              UOp(UOps.VIEW, None, arg=ShapeTracker(views=(View(shape=(2, 1, 128, 128, 128, 128, 3, 3), strides=(0, 0, 1152, 0, 0, 9, 3, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),))
    opts = [Opt(op=OptOps.LOCAL, axis=3, amt=16), Opt(op=OptOps.UPCAST, axis=1, amt=4), Opt(op=OptOps.UPCAST, axis=3, amt=0), Opt(op=OptOps.LOCAL, axis=2, amt=2), Opt(op=OptOps.UPCAST, axis=1, amt=2), Opt(op=OptOps.UPCAST, axis=2, amt=2)]
    _test_overflow(ast, opts)

  # from BEAM on BS=3 simple_conv.py:
  def test_overflow_6(self):
    ast = UOp(UOps.SINK, None, arg=None, src=(
      UOp(UOps.STORE, None, arg=None, src=(
        UOp(UOps.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(UOps.VIEW, None, arg=ShapeTracker(views=(View(shape=(3, 1, 128, 128, 128, 1, 1, 1), strides=(2097152, 0, 16384, 128, 1, 0, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(UOps.REDUCE_AXIS, dtypes.float, arg=(BinaryOps.ADD, (7, 6, 5)), src=(
          UOp(UOps.ALU, dtypes.float, arg=BinaryOps.MUL, src=(
            UOp(UOps.LOAD, dtypes.float, arg=None, src=(
              UOp(UOps.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
              UOp(UOps.VIEW, None, arg=ShapeTracker(views=(View(shape=(1, 3, 1, 128, 4, 130, 4, 130), strides=(0, 2097152, 0, 16384, 0, 128, 0, 1), offset=-129, mask=((0, 1), (0, 3), (0, 1), (0, 128), (0, 4), (1, 129), (0, 4), (1, 129)), contiguous=False), View(shape=(3, 1, 128, 128, 128, 128, 3, 3), strides=(34611200, 0, 0, 520, 1, 270400, 68120, 131), offset=0, mask=None, contiguous=False))), src=()),)),
            UOp(UOps.LOAD, dtypes.float, arg=None, src=(
              UOp(UOps.DEFINE_GLOBAL, dtypes.float.ptr(), arg=2, src=()),
              UOp(UOps.VIEW, None, arg=ShapeTracker(views=(View(shape=(3, 1, 128, 128, 128, 128, 3, 3), strides=(0, 0, 1152, 0, 0, 9, 3, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),))
    opts = [Opt(op=OptOps.LOCAL, axis=3, amt=16), Opt(op=OptOps.UPCAST, axis=3, amt=0), Opt(op=OptOps.UPCAST, axis=1, amt=4), Opt(op=OptOps.LOCAL, axis=2, amt=8), Opt(op=OptOps.UPCAST, axis=1, amt=2), Opt(op=OptOps.UPCAST, axis=3, amt=2)]
    _test_overflow(ast, opts)

  # from BEAM on BS=3 simple_conv.py: (alt)
  def test_overflow_7(self):
    ast = UOp(UOps.SINK, None, arg=None, src=(
      UOp(UOps.STORE, None, arg=None, src=(
        UOp(UOps.DEFINE_GLOBAL, dtypes.float.ptr(), arg=0, src=()),
        UOp(UOps.VIEW, None, arg=ShapeTracker(views=(View(shape=(3, 1, 128, 128, 128, 1, 1, 1), strides=(2097152, 0, 16384, 128, 1, 0, 0, 0), offset=0, mask=None, contiguous=True),)), src=()),
        UOp(UOps.REDUCE_AXIS, dtypes.float, arg=(BinaryOps.ADD, (7, 6, 5)), src=(
          UOp(UOps.ALU, dtypes.float, arg=BinaryOps.MUL, src=(
            UOp(UOps.LOAD, dtypes.float, arg=None, src=(
              UOp(UOps.DEFINE_GLOBAL, dtypes.float.ptr(), arg=1, src=()),
              UOp(UOps.VIEW, None, arg=ShapeTracker(views=(View(shape=(1, 3, 1, 128, 4, 130, 4, 130), strides=(0, 2097152, 0, 16384, 0, 128, 0, 1), offset=-129, mask=((0, 1), (0, 3), (0, 1), (0, 128), (0, 4), (1, 129), (0, 4), (1, 129)), contiguous=False), View(shape=(3, 1, 128, 128, 128, 128, 3, 3), strides=(34611200, 0, 0, 520, 1, 270400, 68120, 131), offset=0, mask=None, contiguous=False))), src=()),)),
            UOp(UOps.LOAD, dtypes.float, arg=None, src=(
              UOp(UOps.DEFINE_GLOBAL, dtypes.float.ptr(), arg=2, src=()),
              UOp(UOps.VIEW, None, arg=ShapeTracker(views=(View(shape=(3, 1, 128, 128, 128, 128, 3, 3), strides=(0, 0, 1152, 0, 0, 9, 3, 1), offset=0, mask=None, contiguous=False),)), src=()),)),)),)),)),))
    opts = [Opt(op=OptOps.UPCAST, axis=3, amt=4), Opt(op=OptOps.LOCAL, axis=3, amt=16), Opt(op=OptOps.UPCAST, axis=1, amt=4), Opt(op=OptOps.LOCAL, axis=2, amt=8), Opt(op=OptOps.UPCAST, axis=1, amt=2), Opt(op=OptOps.UPCAST, axis=2, amt=4)]
    _test_overflow(ast, opts)

@unittest.skipIf(Device.DEFAULT not in {"GPU", "HSA", "CUDA", "METAL"}, "only backends with locals")
@unittest.skipIf(CI, "slow")
class TestLinearizerOverflowAlt(unittest.TestCase):
  def test_overflow_1(self):
    BS = 2
    g0, g1, g2 = [UOp(UOps.DEFINE_GLOBAL, dtypes.float.ptr(), arg=i) for i in range(3)]
    in_st_1 = ShapeTracker(views=(View(shape=(1, BS, 1, 3, 8, 230, 8, 230), strides=(0, 150528, 0, 50176, 0, 224, 0, 1), offset=-675, mask=((0, 1), (0, BS), (0, 1), (0, 3), (0, 8), (3, 227), (0, 8), (3, 227)), contiguous=False),
                                  View(shape=(BS, 1, 64, 112, 112, 3, 7, 7), strides=(10156800, 0, 0, 3680, 2, 3385600, 425040, 231), offset=0, mask=None, contiguous=False))).to_uop()
    in_st_2 = ShapeTracker(views=(View(shape=(BS, 1, 64, 112, 112, 3, 7, 7), strides=(0, 0, 147, 0, 0, 49, 7, 1), offset=0, mask=None, contiguous=False),)).to_uop()
    ot_st = ShapeTracker(views=(View(shape=(BS, 1, 64, 112, 112, 1, 1, 1), strides=(802816, 0, 12544, 112, 1, 0, 0, 0), offset=0, mask=None, contiguous=True),)).to_uop()
    prod = UOp(UOps.LOAD, dtypes.float, (g1, in_st_1)) * UOp(UOps.LOAD, dtypes.float, (g2, in_st_2))
    store = UOp(UOps.STORE, src=(g0, ot_st, UOp(UOps.REDUCE_AXIS, dtypes.float, (prod,), (BinaryOps.ADD, (7, 6, 5)))))
    ast = UOp(UOps.SINK, src=(store,))
    opts = [Opt(op=OptOps.LOCAL, axis=3, amt=16), Opt(op=OptOps.LOCAL, axis=2, amt=2), Opt(op=OptOps.UPCAST, axis=0, amt=2)]
    _test_overflow(ast, opts)
  def test_overflow_2(self):
    BS = 2
    g0, g1, g2 = [UOp(UOps.DEFINE_GLOBAL, dtypes.float.ptr(), arg=i) for i in range(3)]
    in_st_1 = ShapeTracker(views=(View(shape=(1, BS, 1, 3, 8, 230, 8, 230), strides=(0, 150528, 0, 50176, 0, 224, 0, 1), offset=-675, mask=((0, 1), (0, BS), (0, 1), (0, 3), (0, 8), (3, 227), (0, 8), (3, 227)), contiguous=False),
                                  View(shape=(BS, 1, 64, 112, 112, 3, 7, 7), strides=(10156800, 0, 0, 3680, 2, 3385600, 425040, 231), offset=0, mask=None, contiguous=False))).to_uop()
    in_st_2 = ShapeTracker(views=(View(shape=(BS, 1, 64, 112, 112, 3, 7, 7), strides=(0, 0, 147, 0, 0, 49, 7, 1), offset=0, mask=None, contiguous=False),)).to_uop()
    ot_st = ShapeTracker(views=(View(shape=(BS, 1, 64, 112, 112, 1, 1, 1), strides=(802816, 0, 12544, 112, 1, 0, 0, 0), offset=0, mask=None, contiguous=True),)).to_uop()
    prod = UOp(UOps.LOAD, dtypes.float, (g1, in_st_1)) * UOp(UOps.LOAD, dtypes.float, (g2, in_st_2))
    store = UOp(UOps.STORE, src=(g0, ot_st, UOp(UOps.REDUCE_AXIS, dtypes.float, (prod,), (BinaryOps.ADD, (7, 6, 5)))))
    ast = UOp(UOps.SINK, src=(store,))
    opts = [Opt(op=OptOps.LOCAL, axis=3, amt=16), Opt(op=OptOps.UPCAST, axis=1, amt=4), Opt(op=OptOps.LOCAL, axis=2, amt=16), Opt(op=OptOps.UPCAST, axis=4, amt=4), Opt(op=OptOps.UPCAST, axis=1, amt=2), Opt(op=OptOps.UPCAST, axis=5, amt=2)]
    _test_overflow(ast, opts)

if __name__ == '__main__':
  unittest.main()
