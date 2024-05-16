import unittest
from tinygrad import Tensor, dtypes

# format types: https://docs.python.org/3/library/struct.html

class TestTensorData(unittest.TestCase):
  def test_data(self):
    a = Tensor([1,2,3,4], dtype=dtypes.int32)
    dat = a.data()
    assert dat.itemsize == 4
    assert list(dat) == [1,2,3,4]
    assert dat.shape == (4,)
    assert dat[0] == 1
    assert dat[1] == 2

  def test_data_uint8(self):
    a = Tensor([1,2,3,4], dtype=dtypes.uint8)
    dat = a.data()
    assert dat.format == "B"
    assert dat.itemsize == 1
    assert dat[0] == 1
    assert dat[1] == 2

  def test_data_nested(self):
    a = Tensor([[1,2],[3,4]], dtype=dtypes.int32)
    dat = a.data()
    assert dat.format == "i"
    assert dat.itemsize == 4
    assert dat.tolist() == [[1, 2], [3, 4]]
    assert dat.shape == (2,2)
    assert dat[0, 0] == 1
    assert dat[1, 1] == 4

  def test_data_const(self):
    a = Tensor(3, dtype=dtypes.int32)
    dat = a.data()
    assert dat.format == "i"
    assert dat.itemsize == 4
    assert dat.tolist() == 3
    assert dat.shape == ()

  def test_data_float32(self):
    a = Tensor([[1,2.5],[3,4]], dtype=dtypes.float32)
    dat = a.data()
    assert dat.format == "f"
    assert dat[0, 1] == 2.5

  @unittest.skip("requires python 3.12")
  def test_data_float16(self):
    a = Tensor([[1,2.5],[3,4]], dtype=dtypes.float16)
    dat = a.data()
    assert dat.format == "e"
    assert dat.shape == (2,2)
    # NOTE: python can't deref float16

if __name__ == '__main__':
  unittest.main()