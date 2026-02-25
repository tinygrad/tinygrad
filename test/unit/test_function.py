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

  def test_grad_gemm(self):
    @function
    def f(a:Tensor, b:Tensor) -> Tensor: return a @ b

    a = Tensor([[1.,2.],[3.,4.]], requires_grad=True)
    b = Tensor([[5.,6.],[7.,8.]], requires_grad=True)
    na, nb = a.numpy(), b.numpy()
    (f(a, b).contiguous() * b).sum().backward()
    # L = sum((a@b) * b), dL/d(a@b) = b, dL/da = b @ b^T, dL/db = a^T @ b + (a@b)
    np.testing.assert_allclose(a.grad.numpy(), nb @ nb.T)
    np.testing.assert_allclose(b.grad.numpy(), na.T @ nb + na @ nb)

  def test_grad_implicit(self):
    w = Tensor([1., 2., 3.], requires_grad=True)
    @function
    def f(x:Tensor) -> Tensor: return x * w

    x = Tensor([4., 5., 6.])
    f(x).sum().backward()
    np.testing.assert_allclose(w.grad.numpy(), [4., 5., 6.])

  def test_symbolic_index(self):
    from tinygrad.uop.ops import UOp
    table = Tensor([10,20,30,40]).contiguous().realize()
    @function
    def f(x:Tensor, start_pos:int|UOp) -> Tensor:
      return x + table[start_pos]

    v = UOp.variable("start_pos", 0, 3)
    np.testing.assert_equal(f(Tensor([1,2,3]), v.bind(0)).numpy(), [11,12,13])

  def test_nested_calls(self):
    w = Tensor([10., 20., 30.])
    @function
    def f(a:Tensor) -> Tensor: return a + w
    @function
    def g(a:Tensor) -> Tensor: return a * w

    a = Tensor([1., 2., 3.])
    np.testing.assert_allclose(g(f(a)).numpy(), [110., 440., 990.])

if __name__ == '__main__':
  unittest.main()
