import unittest
from tinygrad import Tensor, UOp, GlobalCounters, Context
from tinygrad.uop.ops import Ops, AxisType

class TestOuterworldReduce(unittest.TestCase):
  def test_reduce(self):
    x = Tensor.ones(10, 10).contiguous()
    a = UOp.range(10, -1, AxisType.REDUCE)
    out = x[a]
    # TODO: syntax for this
    t = Tensor(UOp(Ops.REDUCE, dtype=out.uop.dtype, src=(out.uop, a), arg=Ops.ADD))
    print(t.numpy())

  def test_triple_gemm(self):
    Tensor.manual_seed(1337)
    x0 = Tensor.rand(1, 16).realize()
    Tensor.manual_seed(1337)
    x1 = Tensor.rand(1, 16).realize()
    W = Tensor.rand(3, 16, 16).realize()

    for i in range(3): x0 = x0.assign(x0 @ W[i])
    print(x0.numpy())

    # does ASSIGN always terminate the range?
    a = UOp.range(3, -1, AxisType.REDUCE)
    x1 = x1.assign(x1 @ W[a])
    out = Tensor(UOp(Ops.ENDRANGE, dtype=x1.uop.dtype, src=(x1.uop, a))).contiguous()
    out.realize()
    print(out)

    #cpy = x.reshape(1, 1, 16).expand(a, 1, 16).contiguous().realize()
    #print(x.numpy())

    #x = x @ W[a]
    #out = Tensor(UOp(Ops.REDUCE, dtype=x.uop.dtype, src=(x.uop, a), arg=Ops.MAX))
    #print(out.numpy())

    #self.assertTrue((manual==out).all().item())

class TestOuterworld(unittest.TestCase):
  def test_range_plus_1(self):
    t = Tensor.arange(100).reshape(10,10).realize()

    # passthrough ranges
    a = UOp.range(10, -1)
    sel = t[a] + 1
    assert sel.shape == (10,)
    cpy = sel.reshape(1, 10).expand(a, 10).contiguous().realize()

    self.assertTrue((t+1==cpy).all().item())

  def test_range_plus_1_transpose(self):
    t = Tensor.arange(100).reshape(10,10).realize()

    # passthrough ranges
    a = UOp.range(10, -1)
    sel = t[a] + 1
    assert sel.shape == (10,)
    cpy = sel.reshape(10, 1).expand(10, a).contiguous().realize()

    self.assertTrue(((t+1).T==cpy).all().item())

  def test_flip_range(self):
    t = Tensor.rand(10, 10).realize()

    # passthrough ranges
    a = UOp.range(10, -1)
    sel = t[9-a]
    assert sel.shape == (10,)
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

  def test_fancy_vmap(self):
    def f(x,y): return x+y

    x = Tensor.arange(9).reshape(3,3).contiguous()
    y = Tensor.arange(9).reshape(3,3).contiguous()

    a = UOp.range(3, -1)
    out = f(x[:,a], y[a,:])
    # TODO: this should support flatten
    out = out.reshape(1, 3).expand(a, 3).contiguous().realize()
    self.assertListEqual([[0,4,8],[4,8,12],[8,12,16]], out.tolist())

@unittest.skip("TODO: clarify this")
class TestOuterworldUnclear(unittest.TestCase):
  def test_triple_gemm(self):
    x = Tensor.rand(1, 16).realize()
    W = Tensor.rand(3, 16, 16).realize()

    manual = (x @ W[0] @ W[1] @ W[2]).contiguous().realize()

    a = UOp.range(3, -1)
    x = x.assign(x @ W[a])
    out = x.reshape(1, 16, 1).expand(1, 16, a).contiguous().realize()

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

  def test_setitem(self):
    with Context(DEBUG=0):
      t = Tensor.rand(10).realize()
      o = Tensor.empty(10)
    GlobalCounters.reset()
    i = UOp.range(10, -1)
    o[i] = t[i]
    o.contiguous().realize()
    self.assertTrue((t==o).all().item())

if __name__ == '__main__':
  unittest.main()