# ruff: noqa: E501
#!/usr/bin/env python
import unittest
from test.helpers import assert_equiv_st
from tinygrad.dtype import PtrDType, dtypes
from tinygrad.ops import BinaryOps, UOp, UOps, graph_rewrite
from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d
from tinygrad.engine.schedule import create_schedule, reduceop_fusor
from tinygrad.shape.shapetracker import ShapeTracker, View
from tinygrad.helpers import prod
from test.unit.test_shapetracker import shapetracker_getitem

class TestConvShapetracker(unittest.TestCase):
  def test_conv_3x3_one_view(self):
    conv = Conv2d(16, 32, (3, 3))
    seen = set()

    # first run to init the weights, they are saved in seen
    create_schedule([conv(Tensor.empty(1, 16, 10, 10)).lazydata], seen)
    # run it again to get the kernels
    sched = [si for si in create_schedule([conv(Tensor.empty(1, 16, 10, 10)).lazydata], seen) if si.ast.op is UOps.SINK]
    assert len(sched) == 1, f"conv should only have one kernel, getting {len(sched)}"
    for st in [x.st_arg for x in sched[0].ast.parents if x.op is UOps.LOAD]:
      assert len(st.views) == 1

  def test_conv_2x2_backward_one_view(self):
    X = Tensor.rand(1, 1, 3, 3, requires_grad=True)
    conv = Conv2d(1, 1, (2, 2), bias=False)
    conv(X).mean().backward()
    si = X.grad.schedule()[-1]
    print(si)
    ldb = [x for x in si.ast.parents if x.op is UOps.LOAD][0]
    st: ShapeTracker = ldb.st_arg.simplify()
    # NOTE: st.real_size() is broken
    print(si.inputs[0].size)
    #self.assertEqual(si.inputs[0].size, st.real_size())
    for v in st.views: print(v)

    # same st
    test_st = ShapeTracker((
      View(shape=(1, 1, 2, 4, 2, 4), strides=(0, 0, 2, 8, 1, 4), offset=0, mask=((0, 1), (0, 1), (0, 2), (0, 2), (0, 2), (0, 2)), contiguous=False),
      View(shape=(1, 1, 1, 1, 3, 3, 3, 3), strides=(0, 0, 0, 0, 24, 8, 3, 1), offset=0,
           mask=((0, 1), (0, 1), (0, 1), (0, 1), (0, 2), (0, 3), (0, 2), (0, 3)), contiguous=False)))
    #test_st = ShapeTracker((
    #  View(shape=(2,4), strides=(1,4), offset=0, mask=None, contiguous=False),
    #)).simplify()
      #View(shape=(1, 1, 2, 4, 2, 4), strides=(0, 0, 2, 8, 1, 4), offset=0, mask=((0, 1), (0, 1), (0, 2), (0, 2), (0, 2), (0, 2)), contiguous=False),
      #View(shape=(1, 1, 1, 1, 3, 3, 3, 3), strides=(0, 0, 0, 0, 24, 8, 3, 1), offset=0,
      #     mask=((0, 1), (0, 1), (0, 1), (0, 1), (0, 2), (0, 3), (0, 2), (0, 3)), contiguous=False))).simplify()
    print("*** new ***")
    for v in test_st.views: print(v)
    for i in range(prod(st.shape)):
      i1, i2 = shapetracker_getitem(st, i), shapetracker_getitem(test_st, i)
      print(i, i1, i2, si.inputs[0].size, i1==i2)
      #self.assertEqual(i1, i2)

    for stt in [st, test_st]:
      s,va = stt.expr_idxs()
      print(s)
      print(va)
    with self.assertRaises(AssertionError):
      assert len(st.views) <= 2

  def test_swizzle_conv(self):
    swizzle_st = ShapeTracker(views=(View(shape=(2, 3, 3, 65, 3, 65), strides=(103788, 34596, 3, 558, 1, 9), offset=0, mask=((0, 2), (0, 3), (0, 3), (0, 62), (0, 3), (0, 62)), contiguous=False), View(shape=(2, 3, 256, 256), strides=(114075, 38025, 195, 1), offset=0, mask=((0, 2), (0, 3), (0, 195), (0, 195)), contiguous=False), View(shape=(1, 2, 1, 3, 4, 64, 4, 64), strides=(0, 196608, 0, 65536, 16384, 256, 64, 1), offset=0, mask=None, contiguous=True)))
    first_ld = ShapeTracker(views=(View(shape=(2, 1, 3, 16, 62, 62, 3, 3), strides=(0, 0, 9, 27, 0, 0, 3, 1), offset=0, mask=None, contiguous=False),))
    second_ld = ShapeTracker(views=(View(shape=(2, 1, 3, 16, 62, 62, 3, 3), strides=(61504, 0, 0, 3844, 62, 1, 0, 0), offset=0, mask=None, contiguous=False),))
    sink = UOp(UOps.SWIZZLE, dtypes.float, arg=swizzle_st, src=(
      UOp(UOps.REDUCE_AXIS, dtypes.float, arg=(BinaryOps.ADD, (3,)), src=(
        UOp(UOps.ALU, dtypes.float, arg=BinaryOps.MUL, src=(
          UOp(UOps.LOAD, dtypes.float, arg=None, src=(
            UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), arg=1, src=()), UOp(UOps.SHAPETRACKER, None, arg=first_ld, src=()),)),
          UOp(UOps.LOAD, dtypes.float, arg=None, src=(
            UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.float), arg=2, src=()), UOp(UOps.SHAPETRACKER, None, arg=second_ld, src=()),)),)),)),))
    sink = graph_rewrite(sink, reduceop_fusor)
    first_ld, second_ld = [x.src[1].arg for x in sink.parents if x.op is UOps.LOAD]
    assert_equiv_st(first_ld, ShapeTracker(views=(View(shape=(2, 3, 3, 65, 3, 65, 16), strides=(0, 9, 3, 0, 1, 0, 27), offset=0, mask=((0, 2), (0, 3), (0, 3), (0, 62), (0, 3), (0, 62), (0, 16)), contiguous=False), View(shape=(2, 3, 256, 256, 16), strides=(1825200, 608400, 3120, 16, 1), offset=0, mask=((0, 2), (0, 3), (0, 195), (0, 195), (0, 16)), contiguous=False), View(shape=(1, 2, 1, 3, 4, 64, 4, 64, 16), strides=(0, 3145728, 0, 1048576, 262144, 4096, 1024, 16, 1), offset=0, mask=None, contiguous=True))))
    with self.assertRaises(AssertionError):
      assert_equiv_st(second_ld, ShapeTracker(views=(View(shape=(2, 3, 3, 65, 3, 65, 16), strides=(61504, 0, 0, 62, 0, 1, 3844), offset=0, mask=((0, 2), (0, 3), (0, 3), (0, 62), (0, 3), (0, 62), (0, 16)), contiguous=False), View(shape=(2, 3, 256, 256, 16), strides=(1825200, 608400, 3120, 16, 1), offset=0, mask=((0, 2), (0, 3), (0, 195), (0, 195), (0, 16)), contiguous=False), View(shape=(1, 2, 1, 3, 4, 64, 4, 64, 16), strides=(0, 3145728, 0, 1048576, 262144, 4096, 1024, 16, 1), offset=0, mask=None, contiguous=True))))

if __name__ == '__main__':
  unittest.main()
