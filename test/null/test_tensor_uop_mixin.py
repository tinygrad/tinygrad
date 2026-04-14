import math, unittest
from tinygrad import Tensor

def _t(*shape):
  return Tensor.arange(math.prod(shape)).reshape(*shape)

# Tensor().func().uop should be the same as UOp.func()
def _check(tc: unittest.TestCase, t: Tensor, fn):
  tc.assertIs(fn(t).uop, fn(t.uop), f"\ntensor.uop = {fn(t).uop}\nuop = {fn(t.uop)}")

class TestTensorUOpGetitem(unittest.TestCase):
  # ---- pure slice patterns ----
  def test_slice_full(self):           _check(self, _t(4), lambda x: x[slice(None)])
  def test_slice_positive(self):       _check(self, _t(8), lambda x: x[1:5])
  def test_slice_open_start(self):     _check(self, _t(8), lambda x: x[:5])
  def test_slice_open_stop(self):      _check(self, _t(8), lambda x: x[3:])
  def test_slice_negative_start(self): _check(self, _t(8), lambda x: x[-3:])
  def test_slice_negative_stop(self):  _check(self, _t(8), lambda x: x[:-2])
  def test_slice_both_negative(self):  _check(self, _t(8), lambda x: x[-5:-1])

  # ---- slice with stride ----
  def test_slice_stride(self):                  _check(self, _t(6), lambda x: x[::2])
  def test_slice_start_stop_stride(self):       _check(self, _t(6), lambda x: x[1:5:2])
  def test_slice_reverse(self):                 _check(self, _t(6), lambda x: x[::-1])
  def test_slice_singleton_negative_step(self): _check(self, _t(8), lambda x: x[3:2:-1])

  # ---- empty / out-of-bounds slice ----
  def test_slice_empty(self):    _check(self, _t(6), lambda x: x[3:1])
  def test_slice_oob_stop(self): _check(self, _t(6), lambda x: x[0:100])

  # ---- single int (reduces a dim) ----
  def test_int_positive(self): _check(self, _t(8), lambda x: x[3])
  def test_int_negative(self): _check(self, _t(8), lambda x: x[-1])

  # ---- ellipsis ----
  def test_ellipsis_only(self):       _check(self, _t(2, 3, 4), lambda x: x[...])
  def test_ellipsis_then_int(self):   _check(self, _t(2, 3, 4), lambda x: x[..., -1])
  def test_ellipsis_then_slice(self): _check(self, _t(2, 3, 4), lambda x: x[..., 1:3])
  def test_ellipsis_then_none(self):  _check(self, _t(2, 3), lambda x: x[..., None])

  # ---- None (unsqueeze) ----
  def test_none_front(self):    _check(self, _t(4), lambda x: x[None])
  def test_none_back(self):     _check(self, _t(4), lambda x: x[:, None])
  def test_none_middle(self):   _check(self, _t(2, 3), lambda x: x[:, None, :])
  def test_multiple_none(self): _check(self, _t(2, 3), lambda x: x[None, :, None])

  # ---- mixed multi-dim ----
  def test_int_then_slice(self):    _check(self, _t(2, 3), lambda x: x[1, :])
  def test_multi_int(self):         _check(self, _t(2, 3, 4), lambda x: x[1, 2])
  def test_mixed_slice_int(self):   _check(self, _t(2, 3, 4), lambda x: x[0:2, -1, 1:3])
  def test_mixed_slice_slice(self): _check(self, _t(3, 4, 5), lambda x: x[1:3, :, 0:2])
  def test_high_rank_combo(self):   _check(self, _t(4, 5, 6), lambda x: x[1:3, :, -1, None])

class TestTensorUOpCumalu(unittest.TestCase):
  def test_cumsum_1d(self):       _check(self, _t(5), lambda x: x.cumsum())
  def test_cumsum_2d(self):       _check(self, _t(3, 4), lambda x: x.cumsum(1))
  def test_cumsum_non_last(self): _check(self, _t(3, 4), lambda x: x.cumsum(0))
  def test_cumsum_large(self):    _check(self, _t(600), lambda x: x.cumsum())  # exercises _split_cumalu
  def test_cumprod(self):         _check(self, _t(4), lambda x: x.cumprod(0))

class TestTensorUOpCat(unittest.TestCase):
  def test_cat_dim0(self):     _check(self, _t(2, 3), lambda x: x.cat(x, dim=0))
  def test_cat_dim1(self):     _check(self, _t(2, 3), lambda x: x.cat(x, dim=1))
  def test_cat_3tensors(self): _check(self, _t(2, 3), lambda x: x.cat(x, x, dim=0))
  def test_cat_neg_dim(self):  _check(self, _t(2, 3, 4), lambda x: x.cat(x, dim=-1))

class TestTensorUOpStack(unittest.TestCase):
  def test_stack_dim0(self):     _check(self, _t(2, 3), lambda x: x.stack(x, dim=0))
  def test_stack_dim1(self):     _check(self, _t(2, 3), lambda x: x.stack(x, dim=1))
  def test_stack_3tensors(self): _check(self, _t(2, 3), lambda x: x.stack(x, x, dim=0))
  def test_stack_new_last(self): _check(self, _t(2, 3), lambda x: x.stack(x, dim=-1))

if __name__ == "__main__":
  unittest.main()
