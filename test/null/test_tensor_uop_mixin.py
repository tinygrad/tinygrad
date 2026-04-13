import math, unittest
from tinygrad import Tensor

# TODO: make all the expectedFailure cases pass — i.e. UOp.__getitem__ should produce the same UOp graph as
# Tensor.__getitem__ for every view-returning index pattern.

def _t(*shape):
  return Tensor.arange(math.prod(shape)).reshape(*shape)

class TestTensorUOpGetitem(unittest.TestCase):
  """For each pattern, check that `Tensor(x)[idx].uop` equals `x.uop[idx]`."""

  def _check(self, t: Tensor, idx):
    via_tensor = t[idx].uop
    via_uop = t.uop[idx]
    self.assertIs(via_tensor, via_uop, f"\nidx={idx!r}\ntensor.uop = {via_tensor}\nuop[idx] = {via_uop}")

  # ---- pure slice patterns ----
  def test_slice_full(self):       self._check(_t(4), slice(None))
  def test_slice_positive(self):   self._check(_t(8), slice(1, 5))
  def test_slice_open_start(self): self._check(_t(8), slice(None, 5))
  def test_slice_open_stop(self):  self._check(_t(8), slice(3, None))
  def test_slice_negative_start(self): self._check(_t(8), slice(-3, None))
  def test_slice_negative_stop(self):  self._check(_t(8), slice(None, -2))
  def test_slice_both_negative(self):  self._check(_t(8), slice(-5, -1))

  # ---- slice with stride ----
  def test_slice_stride(self):              self._check(_t(6), slice(None, None, 2))
  def test_slice_start_stop_stride(self):   self._check(_t(6), slice(1, 5, 2))
  def test_slice_reverse(self):             self._check(_t(6), slice(None, None, -1))
  def test_slice_singleton_negative_step(self): self._check(_t(8), slice(3, 2, -1))

  # ---- empty / out-of-bounds slice ----
  def test_slice_empty(self):    self._check(_t(6), slice(3, 1))
  def test_slice_oob_stop(self): self._check(_t(6), slice(0, 100))

  # ---- single int (reduces a dim) ----
  def test_int_positive(self): self._check(_t(8), 3)
  def test_int_negative(self): self._check(_t(8), -1)

  # ---- ellipsis ----
  def test_ellipsis_only(self):       self._check(_t(2, 3, 4), (Ellipsis,))
  def test_ellipsis_then_int(self):   self._check(_t(2, 3, 4), (Ellipsis, -1))
  def test_ellipsis_then_slice(self): self._check(_t(2, 3, 4), (Ellipsis, slice(1, 3)))
  def test_ellipsis_then_none(self):  self._check(_t(2, 3), (Ellipsis, None))

  # ---- None (unsqueeze) ----
  def test_none_front(self):    self._check(_t(4), (None,))
  def test_none_back(self):     self._check(_t(4), (slice(None), None))
  def test_none_middle(self):   self._check(_t(2, 3), (slice(None), None, slice(None)))
  def test_multiple_none(self): self._check(_t(2, 3), (None, slice(None), None))

  # ---- mixed multi-dim ----
  def test_int_then_slice(self):    self._check(_t(2, 3), (1, slice(None)))
  def test_multi_int(self):         self._check(_t(2, 3, 4), (1, 2))
  def test_mixed_slice_int(self):   self._check(_t(2, 3, 4), (slice(0, 2), -1, slice(1, 3)))
  def test_mixed_slice_slice(self): self._check(_t(3, 4, 5), (slice(1, 3), slice(None), slice(0, 2)))
  def test_high_rank_combo(self):   self._check(_t(4, 5, 6), (slice(1, 3), slice(None), -1, None))

if __name__ == "__main__":
  unittest.main()
