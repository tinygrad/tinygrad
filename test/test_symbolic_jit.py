import math, unittest
from tinygrad.jit import TinyJit
from tinygrad.shape.symbolic import Variable
from tinygrad.tensor import Tensor, Device
from tinygrad.helpers import getenv
import numpy as np
import torch

@unittest.skipUnless(Device.DEFAULT in ("CLANG", "METAL"), "only CLANG or METAL for now")
class TestSymbolicJit(unittest.TestCase):
  def test_add(self):
    @TinyJit
    def f(a, b): return (a+b).realize()
    for i in range(1, 5):
      ii = Variable("i", 1, 10)
      a = Tensor.rand(i, 10).reshape(ii, 10)
      b = Tensor.rand(i, 10).reshape(ii, 10)
      c = f(a, b)
      np.testing.assert_equal(c.cpu().numpy(), a.numpy()+b.numpy())
    assert len(f.jit_cache) == 1

  def test_2d_matmul(self):
    @TinyJit
    def matmul(a, b): return (a@b).realize()
    for i in range(1, 5):
      ii = Variable("i", 1, 10)
      a = Tensor.rand(3, i).reshape(3, ii)
      b = Tensor.rand(i, 5).reshape(ii, 5)
      c = matmul(a, b).cpu().numpy()
      np.testing.assert_equal(c, a.cpu().numpy()@b.cpu().numpy())
    assert len(matmul.jit_cache) == 1

  def test_4d_matmul(self):
    @TinyJit
    def matmul(a, b): return (a@b).realize()
    for i in range(1, 5):
      ii = Variable("i", 1, 10)
      a = Tensor.rand(1, 4, 3, i).reshape(1, 4, 3, ii)
      b = Tensor.rand(1, 4, i, 5).reshape(1, 4, ii, 5)
      c = matmul(a, b).cpu().numpy()
      np.testing.assert_equal(c, a.cpu().numpy()@b.cpu().numpy())
    assert len(matmul.jit_cache) == 1

  def test_4d_matmul_add(self):
    @TinyJit
    def matmul(a, b): return (a@b+a@b).realize()
    for i in range(1, 5):
      ii = Variable("i", 1, 10)
      a = Tensor.rand(1, 4, 3, i).reshape(1, 4, 3, ii)
      b = Tensor.rand(1, 4, i, 5).reshape(1, 4, ii, 5)
      c = matmul(a, b).cpu().numpy()
      np.testing.assert_equal(c, a.cpu().numpy()@b.cpu().numpy()+a.cpu().numpy()@b.cpu().numpy())
    assert len(matmul.jit_cache) == 1

  def test_multiple_kernels(self):
    @TinyJit
    def matmul(a, b):
      aa = (a+a).realize()
      bb = (b+b).realize()
      return (aa@bb).realize()
    for i in range(1, 5):
      ii = Variable("i", 1, 10)
      a = Tensor.rand(1, 4, 3, i).reshape(1, 4, 3, ii)
      b = Tensor.rand(1, 4, i, 5).reshape(1, 4, ii, 5)
      c = matmul(a, b).cpu().numpy()
      np.testing.assert_equal(c, (a.cpu().numpy()+a.cpu().numpy())@(b.cpu().numpy()+b.cpu().numpy()))
    assert len(matmul.jit_cache) == 3

  def test_reshape_updates_symbols(self):
    for i in range(1, 6):
      x = Variable("x", 1, 100)
      t = Tensor.rand(i, 5)
      t = t.reshape(x, 5)
      assert len(t.lazydata.symbols) == 1
      assert t.lazydata.symbols[x] == i

  def test_attention_like(self):
    head_dim = 128
    @TinyJit
    def attn(q,k,v):
      q = q.transpose(1, 2)
      k = k.transpose(1, 2)
      v = v.transpose(1, 2)
      s = q.matmul(k.transpose(2, 3)) / math.sqrt(head_dim)
      return s.realize().softmax().matmul(v).realize()
    for pos in range(3, 10):
      symbol = Variable("pos", 1, 10)
      q = Tensor.rand(1, 1, 32, 128)
      k = Tensor.rand(1, pos, 32, 128).reshape(1, symbol, 32, 128)
      v = Tensor.rand(1, pos, 32, 128).reshape(1, symbol, 32, 128)
      a = attn(q,k,v)

      tq = torch.tensor(q.cpu().numpy()).transpose(1, 2)
      tk = torch.tensor(k.cpu().numpy()).transpose(1, 2)
      tv = torch.tensor(v.cpu().numpy()).transpose(1, 2)
      s = tq@(tk.transpose(2, 3)) / math.sqrt(head_dim)
      s = s.softmax(-1).matmul(tv)
      np.testing.assert_allclose(a.cpu().numpy(), s, atol=1e-6, rtol=1e-7)
    assert len(attn.jit_cache) == 6

  @unittest.expectedFailure
  def test_cat_dim0(self):
    @TinyJit
    def cat(a, b): return a.cat(b, dim=0).realize()
    for i in range(1, 5):
      for j in range(1, 5):
        ii = Variable("i", 1, 10)
        jj = Variable("j", 1, 10)
        a = Tensor.rand(i, 3).reshape(ii, 3)
        b = Tensor.rand(j, 3).reshape(jj, 3)
        c = cat(a, b)
        np.testing.assert_equal(c, np.concatenate([a.cpu().numpy(), b.cpu().numpy()], axis=0))
    assert len(cat.jit_cache) == 1

  @unittest.expectedFailure
  def test_cat_dim1(self):
    @TinyJit
    def cat(a, b): return a.cat(b, dim=1).realize()
    for i in range(1, 5):
      for j in range(1, 5):
        ii = Variable("i", 1, 10)
        jj = Variable("j", 1, 10)
        a = Tensor.rand(3, i).reshape(3, ii)
        b = Tensor.rand(3, j).reshape(3, jj)
        c = cat(a, b)
        np.testing.assert_equal(c, np.concatenate([a.cpu().numpy(), b.cpu().numpy()], axis=1))
    assert len(cat.jit_cache) == 1