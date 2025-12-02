import unittest
from tinygrad import Tensor

class TestScheduleCache(unittest.TestCase):
  def test_simple(self):
    a = Tensor.ones(10).contiguous()
    b = Tensor.ones(10).contiguous()
    Tensor.realize(a, b)
    for _ in range(5):
      num = (a.sum().contiguous()+b.sum().contiguous()).item()
      print(num)

if __name__ == "__main__":
  unittest.main()
