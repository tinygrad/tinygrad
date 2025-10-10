import unittest
from tinygrad import Tensor, UOp, GlobalCounters, Context

class TestOuterworld(unittest.TestCase):
  def test_range_plus_1(self):
    t = Tensor.arange(100).reshape(10,10).realize()

    # passthrough ranges
    a = UOp.range(10, -1)
    sel = t[a] + 1
    assert sel.shape == (10,)
    cpy = sel.reshape(1, 10).expand(a, 10).contiguous().realize()

    self.assertTrue((t+1==cpy).all().item())

  def test_flip_range(self):
    t = Tensor.rand(10, 10).realize()

    # passthrough ranges
    a = UOp.range(10, -1)
    sel = t[9-a]
    cpy = sel.reshape(1, 10).expand(a, 10).contiguous().realize()

    self.assertTrue((t.flip(0)==cpy).all().item())

  def test_vmap(self):
    def f(x): return x.sum(axis=0)*2

    x = Tensor.ones(3, 10, 2).contiguous()

    # vmap across axis 0
    a = UOp.range(3, -1)
    out = f(x[a])
    out = out.reshape(1, 2).expand(a, 2).contiguous()

    # 3x2 grid of 20
    out.realize()
    self.assertTrue((out==20).all().item())

  @unittest.skip("opts don't work")
  def test_triple_gemm(self):
    x = Tensor.rand(1, 16).realize()
    W = Tensor.rand(3, 16, 16).realize()

    manual = (x @ W[0] @ W[1] @ W[2]).contiguous().realize()

    a = UOp.range(3, -1)
    x = x.assign(x @ W[a])
    out = x.contiguous(a)[-1].contiguous().realize()

    self.assertTrue((manual==out).all().item())

  def test_setitem_pyrange(self):
    with Context(DEBUG=0):
      t = Tensor.rand(10).realize()
      o = Tensor.empty(10)
    GlobalCounters.reset()
    for i in range(10):
      o[i] = t[i]
    o.realize()
    self.assertTrue((t==o).all().item())

  @unittest.skip("TODO: fix this")
  def test_setitem(self):
    with Context(DEBUG=0):
      t = Tensor.rand(10).realize()
      o = Tensor.empty(10)
    GlobalCounters.reset()
    i = UOp.range(10, -1)
    o[i] = t[i]
    o.contiguous(i).realize()
    self.assertTrue((t==o).all().item())

if __name__ == '__main__':
  unittest.main()