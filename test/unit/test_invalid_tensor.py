import math, unittest
from unittest.mock import patch
from tinygrad import Tensor
from tinygrad.device import Device
from tinygrad.dtype import Invalid, dtypes
from tinygrad.helpers import unwrap_class_type

class TestInvalidTensor(unittest.TestCase):
  def _invalid_test_helper(self, out, expected):
    before = None
    original_call = (runtime_cls:=unwrap_class_type(Device[Device.DEFAULT].runtime)).__call__

    def patched_call(self_prg, *bufs, **kwargs):
      nonlocal before
      before = Device[Device.DEFAULT].allocator._as_buffer(bufs[0]).cast(out.dtype.fmt).tolist()
      return original_call(self_prg, *bufs, **kwargs)

    with patch.object(runtime_cls, '__call__', patched_call): ret = out.tolist()

    for i,v in enumerate(expected):
      if v is None: assert before[i] == ret[i] or (math.isnan(before[i]) and math.isnan(ret[i]))
      else: assert ret[i] == v

    return before, ret

  def test_where_x_invalid(self):
    mask = Tensor.arange(4) < 2
    out = mask.where(Tensor([1.0, 2.0, 3.0, 4.0]), Invalid)
    self._invalid_test_helper(out, [1.0, 2.0, None, None])

  def test_where_invalid_x(self):
    mask = Tensor.arange(4) < 2
    out = mask.where(Invalid, Tensor([1.0, 2.0, 3.0, 4.0]))
    self._invalid_test_helper(out, [None, None, 3.0, 4.0])

  def test_where_invalid_2d(self):
    mask = Tensor.arange(6).reshape(2, 3) < 3
    vals = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = mask.where(vals, Invalid)
    before, ret = self._invalid_test_helper(out, [])
    assert ret[0] == [1.0, 2.0, 3.0]
    assert before[3] == ret[1][0] or (math.isnan(before[3]) and math.isnan(ret[1][0]))
    assert before[4] == ret[1][1] or (math.isnan(before[4]) and math.isnan(ret[1][1]))
    assert before[5] == ret[1][2] or (math.isnan(before[5]) and math.isnan(ret[1][2]))

  def test_where_invalid_int(self):
    mask = Tensor.arange(3) < 2
    out = mask.where(Tensor([10, 20, 30]), Invalid)
    self._invalid_test_helper(out, [10, 20, None])

  def test_where_invalid_add(self):
    mask = Tensor.arange(3) < 2
    mixed = mask.where(Tensor([10.0, 20.0, 30.0]), Invalid)
    out = mixed + Tensor([1.0, 2.0, 3.0])
    self._invalid_test_helper(out, [11.0, 22.0, None])

  def test_where_invalid_add_left(self):
    mask = Tensor.arange(3) < 2
    mixed = mask.where(Tensor([10.0, 20.0, 30.0]), Invalid)
    out = Tensor([1.0, 2.0, 3.0]) + mixed
    self._invalid_test_helper(out, [11.0, 22.0, None])

  def test_where_always_true(self):
    mask = Tensor.arange(3) < 10
    out = mask.where(Tensor([10.0, 20.0, 30.0]), Invalid)
    self._invalid_test_helper(out, [10.0, 20.0, 30.0])

  def test_where_cast(self):
    mask = Tensor.arange(4) < 2
    out = mask.where(Tensor([1.0, 2.0, 3.0, 4.0]), Invalid).cast(dtypes.int)
    self._invalid_test_helper(out, [1, 2, None, None])

  def test_where_compare(self):
    mask = Tensor.arange(4) < 2
    out = mask.where(Tensor([1.0, 2.0, 3.0, 4.0]), Invalid) > 1
    self._invalid_test_helper(out, [False, True, None, None])

  def test_where_unary(self):
    mask = Tensor.arange(4) < 2
    out = mask.where(Tensor([1.0, 4.0, 9.0, 16.0]), Invalid).sqrt()
    self._invalid_test_helper(out, [1.0, 2.0, None, None])

  def test_where_where(self):
    mask1 = Tensor.arange(4) < 2
    mask2 = Tensor.arange(4) > 0
    out = mask2.where(mask1.where(Tensor([1.0, 2.0, 3.0, 4.0]), Invalid), Invalid)
    self._invalid_test_helper(out, [None, 2.0, None, None])

  def test_where_reduce_always_true(self):
    mask = Tensor.arange(4) < 9
    out = mask.where(Tensor([1.0, 2.0, 3.0, 4.0]), Invalid).sum()
    before, ret = self._invalid_test_helper(out, [])
    assert ret == 10.0

  def test_invalid_unary(self):
    mask = Tensor.arange(4) < 2
    out = mask.where(Tensor([1.0, 2.0, 3.0, 4.0]), Tensor.full((4,), Invalid, dtype=dtypes.float).sqrt())
    self._invalid_test_helper(out, [1.0, 2.0, None, None])

  def test_invalid_binary(self):
    mask = Tensor.arange(4) < 2
    out = mask.where(Tensor([1.0, 2.0, 3.0, 4.0]), Tensor.full((4,), Invalid, dtype=dtypes.float) + 2)
    self._invalid_test_helper(out, [1.0, 2.0, None, None])

  def test_invalid_binary_left(self):
    mask = Tensor.arange(4) < 2
    out = mask.where(Tensor([1.0, 2.0, 3.0, 4.0]), 2 + Tensor.full((4,), Invalid, dtype=dtypes.float))
    self._invalid_test_helper(out, [1.0, 2.0, None, None])

  def test_invalid_reshape(self):
    mask = Tensor.arange(4) < 2
    out = mask.where(Tensor([1.0, 2.0, 3.0, 4.0]), Invalid).reshape(2,2)
    before, ret = self._invalid_test_helper(out, [])
    assert ret[0] == [1.0, 2.0]
    assert ret[1][0] == before[2] or (math.isnan(ret[1][0]) and math.isnan(before[2]))
    assert ret[1][1] == before[3] or (math.isnan(ret[1][1]) and math.isnan(before[3]))

  def test_invalid_cast(self):
    mask = Tensor.arange(4) < 2
    out = mask.where(Tensor([1.0, 2.0, 3.0, 4.0]), Tensor.full((4,), Invalid, dtype=dtypes.int).cast(dtypes.float))
    self._invalid_test_helper(out, [1.0, 2.0, None, None])

  def test_invalid_bitcast(self):
    mask = Tensor.arange(4) < 2
    out = mask.where(Tensor([1.0, 2.0, 3.0, 4.0]), Tensor.full((4,), Invalid, dtype=dtypes.int).bitcast(dtypes.float))
    self._invalid_test_helper(out, [1.0, 2.0, None, None])

  def test_where_bitcast(self):
    mask = Tensor.arange(4) < 2
    out = mask.where(Tensor([1.0, 2.0, 3.0, 4.0]), Tensor.full((4,), Invalid, dtype=dtypes.int)).bitcast(dtypes.int)
    self._invalid_test_helper(out, [0x3f800000, 0x40000000, None, None])

  # tensor indexing uses reduce, so the entire result becomes invalid
  @unittest.expectedFailure
  def test_tensor_index(self):
    idx = (Tensor.arange(4) < 2).where(Tensor([0, 1, 2, 3]), Invalid)
    out = Tensor([1.0, 2.0, 3.0, 4.0])[idx]
    self._invalid_test_helper(out, [1.0, 2.0, None, None])

if __name__ == '__main__':
  unittest.main()
