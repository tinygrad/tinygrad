import unittest
from tinygrad.function import function
from tinygrad import Tensor

class TestFunction(unittest.TestCase):
  def test_simple(x):
    @function
    def f(a:Tensor, b:Tensor) -> Tensor: return a+b

    a = Tensor([1,2,3])
    b = Tensor([4,5,6])
    c = f(a,b)
    c.realize()

if __name__ == '__main__':
  unittest.main()
