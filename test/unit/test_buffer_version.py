#!/usr/bin/env python
import unittest
import numpy as np
from tinygrad import Tensor, TinyJit

class TestBufferVersion(unittest.TestCase):
  def test_rand(self):
    r1 = Tensor.rand(4)
    r2 = Tensor.rand(4)
    v2 = r2.tolist()
    v1 = r1.tolist()
    self.assertNotEqual(v1, v2)

  def test_assign_buffer(self):
    t1 = Tensor([0,0,0,0])
    t2 = Tensor([0,0,0,0])
    buff = Tensor([0,0,0,0])
    t1 = t1+buff
    buff.assign([1,1,1,1])
    t2 = t2+buff
    t2.realize()
    t1.realize()
    self.assertNotEqual(t1.tolist(), t2.tolist())
    np.testing.assert_equal(t1.numpy(), [0,0,0,0])
    np.testing.assert_equal(t2.numpy(), [1,1,1,1])

  def test_jit_inplace_add(self):
    @TinyJit
    def add(a):
      a += 1
      a.realize()
    a = Tensor.zeros(1).contiguous().realize()
    for _ in range(5): add(a)
    self.assertEqual(a.item(), 5)

  def test_assign_two_buffers(self):
    t1 = Tensor([0,0,0,0])
    t2 = Tensor([0,0,0,0])
    t3 = Tensor([0,0,0,0])
    buff = Tensor([0,0,0,0])
    t1 = t1+buff
    buff.assign([1,1,1,1])
    buff2 = Tensor([1,1,1,1])
    t1 = t1+buff2
    t2 = t2+buff
    buff2.assign([1,1,1,1])
    t2 = t2+buff
    t3 = t3+buff
    buff.assign([3,3,3,3])
    t2.realize()
    t1.realize()
    self.assertNotEqual(t1.tolist(), t2.tolist())
    np.testing.assert_equal(t1.numpy(), [1,1,1,1])
    np.testing.assert_equal(t2.numpy(), [2,2,2,2])

  def test_assign_mul(self):
    buf = Tensor([2,2,2,2])
    t1 = Tensor([3,3,3,3]) * buf
    buf.assign([4,4,4,4])
    t2 = Tensor([3,3,3,3]) * buf
    self.assertEqual(t2.tolist(), [12,12,12,12])
    self.assertEqual(t1.tolist(), [6,6,6,6])

  def test_assign_iadd(self):
    buf = Tensor([0,0,0,0])
    t1 = Tensor([1,1,1,1]) + buf
    buf += 1
    t2 = Tensor([1,1,1,1]) + buf
    self.assertEqual(t2.tolist(), [2,2,2,2])
    self.assertEqual(t1.tolist(), [1,1,1,1])

  def test_assign_slice(self):
    buf = Tensor([0,1,2,3])
    t1 = buf[:2] + Tensor([0,0])
    buf.assign([9,9,9,9])
    t2 = buf[:2] + Tensor([0,0])
    self.assertEqual(t2.tolist(), [9,9])
    self.assertEqual(t1.tolist(), [0,1])

  def test_assign_reshape(self):
    buf = Tensor([[0,0],[0,0]])
    t1 = buf.reshape(4) + Tensor([1,1,1,1])
    buf.assign([[2,2],[2,2]])
    t2 = buf.reshape(4) + Tensor([1,1,1,1])
    self.assertEqual(t2.tolist(), [3,3,3,3])
    self.assertEqual(t1.tolist(), [1,1,1,1])

  def test_assign_cat(self):
    buf = Tensor([0,0])
    t1 = buf.cat(buf)
    buf.assign([1,1])
    t2 = buf.cat(buf)
    self.assertEqual(t2.tolist(), [1,1,1,1])
    self.assertEqual(t1.tolist(), [0,0,0,0])

  def test_assign_sum(self):
    buf = Tensor([1,1,1,1])
    t1 = buf.sum()
    buf.assign([2,2,2,2])
    t2 = buf.sum()
    self.assertEqual(t2.tolist(), 8)
    self.assertEqual(t1.tolist(), 4)

  def test_assign_where(self):
    buf = Tensor([0,1,0,1])
    five, zero = Tensor([5,5,5,5]), Tensor([0,0,0,0])
    t1 = (buf > 0).where(five, zero)
    buf.assign([1,0,1,0])
    t2 = (buf > 0).where(five, zero)
    self.assertEqual(t2.tolist(), [5,0,5,0])
    self.assertEqual(t1.tolist(), [0,5,0,5])

  def test_assign_earlier_first(self):
    t1 = Tensor([0,0,0,0])
    t2 = Tensor([0,0,0,0])
    buff = Tensor([0,0,0,0])
    t1 = t1+buff
    buff.assign([1,1,1,1])
    t2 = t2+buff
    self.assertEqual(t1.tolist(), [0,0,0,0])
    self.assertEqual(t2.tolist(), [1,1,1,1])

  def test_assign_corealize(self):
    t1 = Tensor([0,0,0,0])
    t2 = Tensor([0,0,0,0])
    buff = Tensor([0,0,0,0])
    t1 = t1+buff
    buff.assign([1,1,1,1])
    t2 = t2+buff
    Tensor.realize(t1, t2)
    self.assertEqual(t1.tolist(), [0,0,0,0])
    self.assertEqual(t2.tolist(), [1,1,1,1])

  def test_assign_corealize_later_first(self):
    t1 = Tensor([0,0,0,0])
    t2 = Tensor([0,0,0,0])
    buff = Tensor([0,0,0,0])
    t1 = t1+buff
    buff.assign([1,1,1,1])
    t2 = t2+buff
    Tensor.realize(t2, t1)
    self.assertEqual(t1.tolist(), [0,0,0,0])
    self.assertEqual(t2.tolist(), [1,1,1,1])

  def test_assign_corealize_three(self):
    buf = Tensor([0,0,0,0])
    t1 = Tensor([0,0,0,0]) + buf
    buf.assign([1,1,1,1])
    t2 = Tensor([0,0,0,0]) + buf
    buf.assign([2,2,2,2])
    t3 = Tensor([0,0,0,0]) + buf
    Tensor.realize(t1, t2, t3)
    self.assertEqual(t1.tolist(), [0,0,0,0])
    self.assertEqual(t2.tolist(), [1,1,1,1])
    self.assertEqual(t3.tolist(), [2,2,2,2])

  def test_realize_buf_with_old(self):
    t1 = Tensor([0,0,0,0])
    buff = Tensor([0,0,0,0])
    t1 = t1+buff
    buff.assign([1,1,1,1])
    Tensor.realize(buff, t1)
    self.assertEqual(buff.tolist(), [1,1,1,1])
    self.assertEqual(t1.tolist(), [0,0,0,0])

  def test_two_bufs_corealize(self):
    a, b = Tensor([1,1,1,1]), Tensor([2,2,2,2])
    t1 = a + b
    a.assign([10,10,10,10])
    b.assign([20,20,20,20])
    t2 = a + b
    Tensor.realize(t1, t2)
    self.assertEqual(t1.tolist(), [3,3,3,3])
    self.assertEqual(t2.tolist(), [30,30,30,30])

  def test_rand_corealize(self):
    Tensor.manual_seed(0)
    r1, r2 = Tensor.rand(4), Tensor.rand(4)
    Tensor.realize(r1, r2)
    self.assertNotEqual(r1.tolist(), r2.tolist())

  def test_rand_stack(self):
    Tensor.manual_seed(0)
    r1, r2 = Tensor.rand(4), Tensor.rand(4)
    st = r1.stack(r2)
    self.assertNotEqual(st[0].tolist(), st[1].tolist())

  def test_realize_then_assign(self):
    buf = Tensor([0,0,0,0])
    t1 = Tensor([0,0,0,0]) + buf
    self.assertEqual(t1.tolist(), [0,0,0,0])
    buf.assign([1,1,1,1])
    t2 = Tensor([0,0,0,0]) + buf
    self.assertEqual(t2.tolist(), [1,1,1,1])
    self.assertEqual(t1.tolist(), [0,0,0,0])

  def test_realize_then_assign_twice(self):
    buf = Tensor([0,0,0,0])
    t1 = Tensor([0,0,0,0]) + buf
    t1.realize()
    buf.assign([1,1,1,1])
    t2 = Tensor([0,0,0,0]) + buf
    t2.realize()
    buf.assign([2,2,2,2])
    t3 = Tensor([0,0,0,0]) + buf
    self.assertEqual(t1.tolist(), [0,0,0,0])
    self.assertEqual(t2.tolist(), [1,1,1,1])
    self.assertEqual(t3.tolist(), [2,2,2,2])

  def test_jit_assign(self):
    @TinyJit
    def f():
      buf = Tensor([0,0,0,0])
      t1 = Tensor([0,0,0,0]) + buf
      buf.assign([1,1,1,1])
      t2 = Tensor([0,0,0,0]) + buf
      return t1.realize(), t2.realize()
    for _ in range(3):
      a, b = f()
      self.assertEqual(a.tolist(), [0,0,0,0])
      self.assertEqual(b.tolist(), [1,1,1,1])

  def test_jit_assign_corealize(self):
    @TinyJit
    def f():
      buf = Tensor([0,0,0,0])
      t1 = Tensor([0,0,0,0]) + buf
      buf.assign([1,1,1,1])
      t2 = Tensor([0,0,0,0]) + buf
      Tensor.realize(t1, t2)
      return t1, t2
    for _ in range(3):
      a, b = f()
      self.assertEqual(a.tolist(), [0,0,0,0])
      self.assertEqual(b.tolist(), [1,1,1,1])

  def test_assign_backward(self):
    buf = Tensor([1., 1., 1., 1.])
    x = Tensor([2., 2., 2., 2.])
    t1 = (x * buf).sum()
    buf.assign([10., 10., 10., 10.])
    t2 = (x * buf).sum()
    t1.backward()
    self.assertEqual(x.grad.tolist(), [1., 1., 1., 1.])
    self.assertEqual(t1.tolist(), 8.0)
    self.assertEqual(t2.tolist(), 80.0)

if __name__ == "__main__":
  unittest.main()
