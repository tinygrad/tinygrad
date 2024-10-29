# basic self-contained tests of the external functionality of tinygrad
import unittest
from tinygrad import Tensor, Context, Variable, TinyJit

class TestTiny(unittest.TestCase):

  # *** basic functionality ***

  def test_plus(self):
    out = Tensor([1.,2,3]) + Tensor([4.,5,6])
    self.assertListEqual(out.tolist(), [5.0, 7.0, 9.0])

  def test_plus_big(self):
    out = Tensor.ones(16).contiguous() + Tensor.ones(16).contiguous()
    self.assertListEqual(out.tolist(), [2]*16)

  def test_cat(self):
    out = Tensor.cat(Tensor.ones(8).contiguous(), Tensor.ones(8).contiguous())
    self.assertListEqual(out.tolist(), [1]*16)

  def test_gemm(self):
    N = 4
    a = Tensor.ones(N,N).contiguous()
    b = Tensor.eye(N).contiguous()
    self.assertListEqual((a@b).flatten().tolist(), [1.0]*(N*N))

  # *** JIT (for Python speed) ***

  def test_jit(self):
    cnt = 0
    @TinyJit
    def fxn(a,b):
      nonlocal cnt
      cnt += 1
      return a+b
    fa,fb = Tensor([1.,2,3]), Tensor([4.,5,6])
    for _ in range(3): fxn(fa, fb)
    # function is only called twice
    self.assertEqual(cnt, 2)

  # *** BEAM (for Kernel speed) ***

  def test_beam(self):
    with Context(BEAM=1): self.test_plus()

  # *** symbolic (to allow less recompilation) ***

  def test_symbolic(self):
    i = Variable('i', 1, 10)
    for s in [2,5]:
      ret = Tensor.ones(s).contiguous().reshape(i.bind(s)) + 1
      self.assertListEqual(ret.reshape(s).tolist(), [2.0]*s)

  def test_symbolic_reduce(self):
    i = Variable('i', 1, 10)
    for s in [2,5]:
      ret = Tensor.ones(s).contiguous().reshape(i.bind(s)).sum()
      self.assertEqual(ret.item(), s)

if __name__ == '__main__':
  unittest.main()

