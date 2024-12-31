import unittest
from tinygrad import Tensor

class TestSchedule(unittest.TestCase):
  def test_simple_copy(self):
    a = Tensor([1])
    a.realize()
    print(a.lazydata.realized)

if __name__ == "__main__":
  unittest.main()
