import unittest
from tinygrad import Tensor
from tinygrad.codegen import apply_rewrites, get_rewrites_for_renderer
from tinygrad.device import Device
from tinygrad.dtype import dtypes
from tinygrad.helpers import CI
from tinygrad.uop.ops import Ops, UOp

class TestLoopSplittingGood(unittest.TestCase):
  def setUp(self) -> None:
    rewrites = list(get_rewrites_for_renderer(Device.default.renderer))
    while rewrites[0].name != "lowerer": rewrites.pop(0)
    rewrites.pop(0)
    rewrites = list(reversed(rewrites))
    while rewrites[0].name != "split_loop": rewrites.pop(0)
    self.rewrites = list(reversed(rewrites))

  def test_basic_cat(self):
    sink = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.INDEX, dtypes.float.ptr(128), arg=None, src=(
          UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(128), arg=0, src=()),
          x3:=UOp(Ops.RANGE, dtypes.int, arg=0, src=(
            UOp(Ops.CONST, dtypes.int, arg=128, src=()),)),)),
        UOp(Ops.ADD, dtypes.float, arg=None, src=(
          UOp(Ops.LOAD, dtypes.float, arg=None, src=(
            UOp(Ops.INDEX, dtypes.float.ptr(64), arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(64), arg=1, src=()),
               x3,
              x9:=UOp(Ops.CMPLT, dtypes.bool, arg=None, src=(
                 x3,
                UOp(Ops.CONST, dtypes.int, arg=64, src=()),)),)),)),
          UOp(Ops.LOAD, dtypes.float, arg=None, src=(
            UOp(Ops.INDEX, dtypes.float.ptr(64), arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(64), arg=2, src=()),
              UOp(Ops.ADD, dtypes.int, arg=None, src=(
                 x3,
                UOp(Ops.CONST, dtypes.int, arg=-64, src=()),)),
              UOp(Ops.CMPNE, dtypes.bool, arg=None, src=(
                 x9,
                UOp(Ops.CONST, dtypes.bool, arg=True, src=()),)),)),)),)),)),))

    self.check_range_splits(sink, 1, 2)

  def test_cat_2d(self):
    sink = UOp(Ops.SINK, dtypes.void, arg=None, src=(
      UOp(Ops.STORE, dtypes.void, arg=None, src=(
        UOp(Ops.INDEX, dtypes.float.ptr(8192), arg=None, src=(
          UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(8192), arg=0, src=()),
          x3:=UOp(Ops.ADD, dtypes.int, arg=None, src=(
            UOp(Ops.MUL, dtypes.int, arg=None, src=(
              x5:=UOp(Ops.RANGE, dtypes.int, arg=0, src=(
                UOp(Ops.CONST, dtypes.int, arg=128, src=()),)),
              x7:=UOp(Ops.CONST, dtypes.int, arg=64, src=()),)),
            UOp(Ops.RANGE, dtypes.int, arg=1, src=(
               x7,)),)),)),
        UOp(Ops.ADD, dtypes.float, arg=None, src=(
          UOp(Ops.LOAD, dtypes.float, arg=None, src=(
            UOp(Ops.INDEX, dtypes.float.ptr(4096), arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(4096), arg=1, src=()),
               x3,
              x13:=UOp(Ops.CMPLT, dtypes.bool, arg=None, src=(
                 x5,
                 x7,)),)),)),
          UOp(Ops.LOAD, dtypes.float, arg=None, src=(
            UOp(Ops.INDEX, dtypes.float.ptr(4096), arg=None, src=(
              UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(4096), arg=2, src=()),
              UOp(Ops.ADD, dtypes.int, arg=None, src=(
                 x3,
                UOp(Ops.CONST, dtypes.int, arg=-4096, src=()),)),
              UOp(Ops.CMPNE, dtypes.bool, arg=None, src=(
                 x13,
                UOp(Ops.CONST, dtypes.bool, arg=True, src=()),)),)),)),)),)),))

    sink = self.check_range_splits(sink, 1, 2)
    self.assertEqual(len(sink.src), 2)

  def check_range_splits(self, sink: UOp, n_ranges_removed: int, n_ranges_added: int):
    ranges_before = set(uop for uop in sink.parents if uop.op is Ops.RANGE)
    sink = apply_rewrites(sink, self.rewrites)
    ranges_after = set(uop for uop in sink.parents if uop.op is Ops.RANGE)

    self.assertEqual(len(ranges_before - ranges_after), n_ranges_removed)
    self.assertEqual(len(ranges_after - ranges_before), n_ranges_added)

    return sink

@unittest.skipIf(CI, "bad tests, need to clean up")
class TestLoopSplitting(unittest.TestCase):
  def test_basic_cat(self):
    a = Tensor.empty(64)
    b = Tensor.empty(64)

    self.verify_schedule(a.cat(b))

  def test_cat_2d(self):
    a = Tensor.empty(64, 64)
    b = Tensor.empty(64, 64)

    self.verify_schedule(a.cat(b))

  def test_cat_reduce1(self): # TODO: this is a shitty test
    a = Tensor.empty(64, 64)
    b = Tensor.empty(64, 64)

    # TODO: make this nice!
    with self.assertRaises(AssertionError):
      self.verify_schedule(a.cat(b, dim=0).sum(1)) # this can not be split!

  def test_cat_reduce2(self):
    a = Tensor.empty(64, 64)
    b = Tensor.empty(64, 64)

    self.verify_schedule(a.cat(b, dim=1).sum(1))

  def test_add_pad(self):
    a = Tensor.empty(64, 64)
    b = Tensor.empty(128, 128)

    self.verify_schedule((a.pad([(0, 64), (0, 64)]) + b))

  def verify_schedule(self, t: Tensor):
    for si in t.schedule():
      rewrites = list(get_rewrites_for_renderer(Device.default.renderer))
      sink = si.ast
      while rewrites[0].name != "split_loop": sink = rewrites.pop(0)(sink)
      initial_ranges = set(uop for uop in sink.toposort() if uop.op is Ops.RANGE)

      if len(initial_ranges) > 0:
        sink = rewrites.pop(0)(sink)
        new_ranges = set(uop for uop in sink.toposort() if uop.op is Ops.RANGE)
        self.assertNotEqual(initial_ranges, new_ranges) # if loop splitting worked, we expect new ranges to exist

        while rewrites and rewrites[0].name != "split_loop": sink = rewrites.pop(0)(sink)

if __name__ == '__main__':
  unittest.main()
