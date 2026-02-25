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
    def f(a:Tensor, b:Tensor) -> Tensor: return a+b+inp
    @function
    def g(a:Tensor, b:Tensor) -> Tensor: return a+b+inp

    in0 = Tensor([1,2,3])
    in1 = Tensor([4,5,6])
    out0 = f(in0, in1)
    in2 = Tensor([0,2,3])
    in3 = Tensor([0,5,6])
    out1 = g(in2, in3)
    out0.realize(out1)

if __name__ == '__main__':
  unittest.main()
