import unittest
from tinygrad import Tensor, UOp
from tinygrad.uop.ops import AxisType, Ops

class TestOuterworldReduce(unittest.TestCase):
  def test_reduce(self):
    x = Tensor.ones(5, 5).contiguous()
    a = UOp.range(5, -1, AxisType.REDUCE)
    out = x[a]
    # TODO: syntax for this
    t = Tensor(UOp(Ops.REDUCE, dtype=out.uop.dtype, src=(out.uop, a), arg=Ops.ADD))
    self.assertListEqual(t.tolist(), [5.,5.,5.,5.,5.])

# TODO: delete test_outerworld_range?
class TestOuterRange(unittest.TestCase):
  def test_simple_range(self):
    a = Tensor.ones(10).contiguous()
    acc = Tensor.zeros().contiguous()
    Tensor.realize(a, acc)

    # this is fold
    i = UOp.range(10, -100, AxisType.OUTER)
    acc_i = acc.uop.after(i)
    vi = UOp.variable("i", i.vmin, i.vmax).bind(i)
    out = Tensor(acc.uop.after(acc_i.store(acc_i + a[vi].uop).end(i)))
    out.realize()
    assert out.item() == 10.0

  def test_inner_range(self):
    a = Tensor.ones(10, 10).contiguous()
    acc = Tensor.zeros(10).contiguous()
    Tensor.realize(a, acc)

    # this is fold
    i = UOp.range(10, -100, AxisType.OUTER)
    acc_i = acc.uop.after(i)
    vi = UOp.variable("i", i.vmin, i.vmax).bind(i)
    out = Tensor(acc.uop.after(acc_i.store(acc_i + a[:, vi].uop).end(i)))
    out.realize()
    assert all(x == 10.0 for x in out.tolist())

  def test_range_matmul(self):
    vec = Tensor.randn(1, 10).realize()
    mats = Tensor.randn(3, 10, 10).realize()

    # 3 matmuls in "scan"
    ref = ((vec @ mats[0]) @ mats[1]) @ mats[2]
    ref.realize()

    # 3 matmuls with outer world range
    i = UOp.range(3, -100, AxisType.OUTER)
    vec_i = Tensor(vec.uop.after(i))
    vi = UOp.variable("i", i.vmin, i.vmax).bind(i)
    out = Tensor(vec.uop.after(vec_i.uop.store((vec_i.contiguous() @ mats[vi]).uop).end(i)))
    out.realize()

    # TODO: testing allclose
    assert Tensor.allclose(ref, out, atol=1e-6), f"{ref.numpy()=}, {out.numpy()=}"

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

if __name__ == '__main__':
  unittest.main()