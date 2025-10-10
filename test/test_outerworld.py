import unittest
from tinygrad import Tensor, UOp, Variable, nn
from tinygrad.uop.ops import AxisType, Ops

class TestOuterworldTrain(unittest.TestCase):
  @Tensor.train()
  def test_train(self):
    # same example over and over
    X = Tensor.rand(1, 32).expand(16,32).contiguous()
    Y = Tensor.rand(1, 1).expand(16,1).contiguous()

    layer = nn.Linear(32, 1, bias=False)
    opt = nn.optim.SGD(nn.state.get_parameters(layer))
    Tensor.realize(X, Y, *nn.state.get_parameters(layer))

    print("train")

    # if everything is correct, this should be a 16 step training loop
    steps = UOp.range(16, -1)
    opt.zero_grad()
    loss = (layer(X[steps]) - Y[steps]).square().mean().backward()
    opt.schedule_step() # TODO: does this need to know anything about steps?
    # NOTE: this can't work. the inputs to layer are not the assign, need to run twice for the fixed point?
    all_losses = loss.reshape(1).expand(steps).contiguous()
    all_losses.realize()
    print(all_losses.numpy())

#@unittest.skip("TODO: understand assign")
class TestOuterworldAssign(unittest.TestCase):
  def test_triple_add_inner(self):
    t = Tensor.zeros(5).contiguous().realize()
    t2 = Tensor.ones(3).contiguous().realize()
    a = UOp.range(3, -1)
    t = t.reshape(1,5).expand(a+1,5)[a].assign(t+t2[a])
    self.assertListEqual(t.tolist(), [3,3,3,3,3])

  def test_triple_add_outer(self):
    t = Tensor.zeros(5).contiguous().realize()
    t2 = Tensor.ones(3).contiguous().realize()

    # OUTER is a loop at the schedule level
    a = UOp.range(3, -1, AxisType.OUTER)
    va = Variable("loop", 0, 2).bind(a)
    t = t.assign(t+t2[va])
    t = Tensor(UOp(Ops.ENDRANGE, dtype=t.uop.dtype, src=(a, t.uop)))

    self.assertListEqual(t.tolist(), [3,3,3,3,3])

  def test_triple_gemm(self):
    x = Tensor.rand(1, 16).realize()
    W = Tensor.rand(3, 16, 16).realize()

    #manual = (x @ W[0] @ W[1] @ W[2]).contiguous().realize()

    a = UOp.range(3, -1)

    out = (x @ W[a]).contiguous()
    t = Tensor(UOp(Ops.ASSIGN, dtype=out.uop.dtype, src=(x.uop, out.uop, a)))
    #t = Tensor(UOp(Ops.REDUCE, dtype=out.uop.dtype, src=(out.uop, x.uop, a), arg=Ops.NOOP))
    t.realize()

class TestOuterworldReduce(unittest.TestCase):
  def test_reduce(self):
    x = Tensor.ones(5, 5).contiguous()
    a = UOp.range(5, -1, AxisType.REDUCE)
    out = x[a]
    # TODO: syntax for this
    t = Tensor(UOp(Ops.REDUCE, dtype=out.uop.dtype, src=(out.uop, a), arg=Ops.ADD))
    self.assertListEqual(t.tolist(), [5.,5.,5.,5.,5.])

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

if __name__ == '__main__':
  unittest.main()