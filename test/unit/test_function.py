import unittest
from tinygrad.function import function
from tinygrad import Tensor

class TestFunction(unittest.TestCase):
  def test_simple(self):
    @function
    def f(a:Tensor, b:Tensor) -> Tensor: return a+b

    a = Tensor([1,2,3])
    b = Tensor([4,5,6])
    c = f(a,b)
    c.realize()

  def test_implicit(self):
    inp = Tensor([7,8,9])
    @function
    def f(a:Tensor, b:Tensor) -> Tensor: return a+b+inp

    a = Tensor([1,2,3])
    b = Tensor([4,5,6])
    c = f(a,b)
    c.realize()

  def test_implicit_2(self):
    inp = Tensor([7,8,9])
    @function
    def f(a:Tensor, b:Tensor) -> Tensor:
      return a+b+inp
    inp2 = Tensor([7,8,10])
    @function
    def g(a:Tensor, b:Tensor) -> Tensor:
      return a+b+inp2

    a = Tensor([1,2,3])
    b = Tensor([4,5,6])
    c = f(a,b)
    d = g(a,b)
    c.realize(d)

if __name__ == '__main__':
  unittest.main()
