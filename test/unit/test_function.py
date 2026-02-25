import numpy as np
import unittest
from tinygrad.function import function
from tinygrad import Tensor

class TestFunction(unittest.TestCase):
  def test_simple(self):
    @function
    def f(a:Tensor, b:Tensor) -> Tensor: return a+b

    a = Tensor([1,2,3])
    b = Tensor([4,5,6])
    np.testing.assert_equal(f(a,b).numpy(), [5,7,9])

  def test_simple_same(self):
    @function
    def f(a:Tensor, b:Tensor) -> Tensor: return a+b

    a = Tensor([1,2,3])
    np.testing.assert_equal(f(a,a).numpy(), [2,4,6])

  def test_implicit(self):
    inp = Tensor([7,8,9])
    @function
    def f(a:Tensor, b:Tensor) -> Tensor: return a+b+inp

    a = Tensor([1,2,3])
    b = Tensor([4,5,6])
    np.testing.assert_equal(f(a,b).numpy(), [12,15,18])

  def test_implicit_same_as_input(self):
    inp = Tensor([7,8,9])
    @function
    def f(a:Tensor, b:Tensor) -> Tensor: return a+b+inp

    a = Tensor([1,2,3])
    np.testing.assert_equal(f(a, inp).numpy(), [15,18,21])

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
    np.testing.assert_equal(c.numpy(), [12,15,18])
    np.testing.assert_equal(d.numpy(), [12,15,19])

  def test_implicit_unrealized(self):
    inp = Tensor([1,2,3]) + Tensor([4,5,6])
    @function
    def f(a:Tensor) -> Tensor: return a + inp

    np.testing.assert_equal(f(Tensor([10,20,30])).numpy(), [15,27,39])

  def test_detach(self):
    @function
    def f(a:Tensor, b:Tensor) -> Tensor: return a.detach() + b

    a = Tensor([1,2,3])
    b = Tensor([4,5,6])
    np.testing.assert_equal(f(a, b).numpy(), [5,7,9])

  def test_method(self):
    class Foo:
      def __init__(self): self.w = Tensor([10,20,30])
      @function
      def __call__(self, x:Tensor) -> Tensor: return x + self.w

    foo = Foo()
    np.testing.assert_equal(foo(Tensor([1,2,3])).numpy(), [11,22,33])

if __name__ == '__main__':
  unittest.main()
