import numpy as np
from typing import Callable
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

class TestAfterVmap(unittest.TestCase):
  @staticmethod
  def _vmap(fn:Callable[[Tensor], Tensor], n: int, m: int)->Callable[[Tensor], Tensor]:
    r = UOp.range(n, -1)
    return lambda x: fn(x[r]).reshape(1,m).expand(r,m)

  n,m = 3,6

  @staticmethod
  def _fn(x: Tensor)->Tensor:
    return Tensor.arange(TestAfterVmap.m) * x

  _vfn = _vmap(_fn, n, m)

  expected_vmap_res = np.tile(np.arange(m, dtype=np.float64), (n, 1))

  def test_vmap(self):
    x = Tensor.ones(self.n, self.m)
    self.assertListEqual(TestAfterVmap._vfn(x).tolist(), self.expected_vmap_res.tolist())

  def test_flatten(self):
    x = Tensor.ones(self.n, self.m)
    self.assertListEqual(TestAfterVmap._vfn(x).flatten().tolist(), self.expected_vmap_res.flatten().tolist())

  def test_reshape(self):
    if self.m % 2 != 0:
      self.skipTest(f"self.m = {self.m} % 2 != 0")
    x = Tensor.ones(self.n, self.m)
    self.assertListEqual(TestAfterVmap._vfn(x).reshape(-1,2).tolist(), self.expected_vmap_res.reshape(-1,2).tolist())

  def test_transpose(self):
    x = Tensor.ones(self.n, self.m)
    self.assertListEqual(TestAfterVmap._vfn(x).transpose().realize().tolist(), self.expected_vmap_res.transpose().tolist())

  def test_pad(self):
    x = Tensor.ones(self.n, self.m)
    # numpy convention
    for pad_width in [((1,0), (0,0)),((0,1),(0,0)), ((0,0), (1,0)), ((0,0), (0,1))]:
      self.assertListEqual(TestAfterVmap._vfn(x).pad(pad_width).tolist(),np.pad(self.expected_vmap_res, pad_width).tolist())

  def test_flip(self):
    x = Tensor.ones(self.n, self.m)
    self.assertListEqual(TestAfterVmap._vfn(x).flip(0).tolist(), np.flip(self.expected_vmap_res, 0).tolist())
    self.assertListEqual(TestAfterVmap._vfn(x).flip(1).tolist(), np.flip(self.expected_vmap_res, 1).tolist())
    self.assertListEqual(TestAfterVmap._vfn(x).flip((0,1)).tolist(), np.flip(self.expected_vmap_res, (0,1)).tolist())

  def test_indexing(self):
    x = Tensor.ones(self.n, self.m)
    self.assertEqual(TestAfterVmap._vfn(x)[2,4].item(), self.expected_vmap_res[2,4])
    i,j = [0,1,2], [0,2,4]
    self.assertListEqual(TestAfterVmap._vfn(x)[i,j].tolist(), self.expected_vmap_res[i,j].tolist())
    # self.assertListEqual(TestAfterVmap._vfn(x)[np.array(i), np.array(j)].tolist(), self.expected_vmap_res[i,j].tolist())
    self.assertListEqual(TestAfterVmap._vfn(x)[Tensor(i), Tensor(j)].tolist(), self.expected_vmap_res[i,j].tolist())
    i = [0, 8, 16]
    self.assertListEqual(TestAfterVmap._vfn(x).flatten()[i].tolist(), self.expected_vmap_res.flatten()[i].tolist())
    self.assertListEqual(TestAfterVmap._vfn(x).flatten()[Tensor(i)].tolist(), self.expected_vmap_res.flatten()[i].tolist())

  def test_slicing(self):
    x = Tensor.ones(self.n, self.m)
    self.assertListEqual(TestAfterVmap._vfn(x)[0].tolist(), self.expected_vmap_res[0].tolist())
    self.assertListEqual(TestAfterVmap._vfn(x)[:, 1].tolist(), self.expected_vmap_res[:, 1].tolist())
    if self.n>=3 and self.m >= 4:
      self.assertListEqual(TestAfterVmap._vfn(x)[1:3, 2:4].tolist(), self.expected_vmap_res[1:3, 2:4].tolist())

  def test_sum(self):
    x = Tensor.ones(self.n, self.m)
    self.assertEqual(TestAfterVmap._vfn(x).sum().item(), self.expected_vmap_res.sum())
    self.assertListEqual(TestAfterVmap._vfn(x).sum(0).tolist(), self.expected_vmap_res.sum(0).tolist())
    self.assertListEqual(TestAfterVmap._vfn(x).sum(1).tolist(), self.expected_vmap_res.sum(1).tolist())

  def test_cmp(self):
    x = Tensor.ones(self.n, self.m)
    self.assertTrue((TestAfterVmap._vfn(x)[:,0] == 0.0).all().item())
    self.assertTrue((TestAfterVmap._vfn(x)[:,1] > 0.0).all().item())
    self.assertTrue((TestAfterVmap._vfn(x) < self.m).all().item())

  def test_linalg(self):
    x = Tensor.ones(self.n, self.m)
    self.assertListEqual((TestAfterVmap._vfn(x) @ Tensor.ones(self.m)).tolist(), (self.expected_vmap_res @ np.ones(self.m)).tolist())
    self.assertTrue((TestAfterVmap._vfn(x) @ Tensor.ones(self.m) == sum(range(self.m))).all().item())
    self.assertListEqual((TestAfterVmap._vfn(x) @ Tensor.arange(self.m*10).reshape(self.m, 10)).tolist(),
                         (self.expected_vmap_res @ np.arange(10*self.m).reshape(self.m, 10)).tolist())

  def test_squeeze(self):
    x = Tensor.ones(self.n, self.m)
    self.assertListEqual(TestAfterVmap._vfn(x).unsqueeze(0).tolist(), np.expand_dims(self.expected_vmap_res,0).tolist())
    self.assertListEqual(TestAfterVmap._vfn(x).unsqueeze(-1).tolist(), np.expand_dims(self.expected_vmap_res,-1).tolist())
    self.assertListEqual(TestAfterVmap._vfn(x).unsqueeze(-1).squeeze(-1).tolist(), self.expected_vmap_res.tolist())
    if self.n != 1: self.assertTrue((ret:=TestAfterVmap._vfn(x)).squeeze(0) is ret)
    if self.m != 1: self.assertTrue((ret:=TestAfterVmap._vfn(x)).squeeze(0) is ret)

  def test_cat(self):
    x = Tensor.ones(self.n, self.m)
    self.assertListEqual(Tensor.cat(TestAfterVmap._vfn(x), x, dim=0).tolist(),
                        np.concatenate([self.expected_vmap_res, np.ones((self.n, self.m))], axis=0).tolist())
    self.assertListEqual(Tensor.cat(x, TestAfterVmap._vfn(x), dim=0).tolist(),
                        np.concatenate([np.ones((self.n, self.m)), self.expected_vmap_res], axis=0).tolist())

  def test_broadcast_to(self):
    x = Tensor.ones(self.n, self.m)
    # same shape but with symbolic ones
    self.assertListEqual(TestAfterVmap._vfn(x)._broadcast_to((self.n, self.m)).tolist(), self.expected_vmap_res.tolist())
    self.assertListEqual(TestAfterVmap._vfn(x)._broadcast_to((UOp.range(self.n, -2), self.m)).tolist(), self.expected_vmap_res.tolist())
    self.assertListEqual(TestAfterVmap._vfn(x)._broadcast_to((self.n, UOp.range(self.m, -2))).tolist(), self.expected_vmap_res.tolist())

if __name__ == '__main__':
  unittest.main()
