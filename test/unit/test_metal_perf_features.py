import unittest
import numpy as np
from tinygrad import Tensor, TinyJit, Device
from tinygrad.helpers import Context

# *** beam search heuristic seeding ***

class TestBeamHeuristicSeed(unittest.TestCase):
  def test_beam_sum_correctness(self):
    """beam search seeded with heuristic produces correct sum results."""
    with Context(BEAM=2, IGNORE_BEAM_CACHE=1):
      a = Tensor.ones(4096).contiguous()
      self.assertEqual(a.sum().item(), 4096)

  def test_beam_large_reduction_correctness(self):
    """beam search seeded with heuristic produces correct results for large 2D reduction."""
    with Context(BEAM=2, IGNORE_BEAM_CACHE=1):
      a = Tensor.ones(64, 64).contiguous()
      np.testing.assert_allclose(a.sum().numpy(), 4096.0, rtol=1e-5)

  def test_beam_sum_axis_correctness(self):
    """beam search seeded with heuristic produces correct partial sum results."""
    with Context(BEAM=2, IGNORE_BEAM_CACHE=1):
      a = Tensor([[1.,2.],[3.,4.]])
      np.testing.assert_allclose(a.sum(axis=1).numpy(), [3.0, 7.0], rtol=1e-5)

# *** fast JIT replay ***

class TestFastJitReplay(unittest.TestCase):
  def test_fast_replay_correctness(self):
    """fast JIT replay path (cnt >= 3) produces correct results."""
    @TinyJit
    def add(a, b): return (a+b).realize()
    for i in range(6):
      a = Tensor([float(i), float(i+1)])
      b = Tensor([float(i+2), float(i+3)])
      c = add(a, b)
      np.testing.assert_allclose(c.numpy(), [i+i+2, i+1+i+3], rtol=1e-5)
    # cnt should be 6 (fast replay used for iterations 3-5)
    self.assertEqual(add.cnt, 6)

  def test_fast_replay_setup(self):
    """_fast_replay attribute is set after first successful replay (cnt == 2)."""
    @TinyJit
    def mul(a): return (a*2).realize()
    for i in range(4):
      a = Tensor([float(i)])
      mul(a)
    self.assertTrue(hasattr(mul, '_fast_replay'))

  def test_fast_replay_sum(self):
    """fast JIT replay works for sum reduction."""
    @TinyJit
    def f(a): return a.sum().realize()
    for i in range(5):
      a = Tensor.ones(1024).contiguous() * (i + 1)
      a.realize()
      out = f(a)
      np.testing.assert_allclose(out.numpy(), 1024.0 * (i + 1), rtol=1e-5)

  def test_fast_replay_not_set_with_kwargs(self):
    """fast replay is not enabled when kwargs are used."""
    @TinyJit
    def add(a, b): return (a+b).realize()
    for i in range(4):
      a = Tensor([float(i)])
      b = Tensor([float(i)])
      add(a, b=b)
    self.assertFalse(hasattr(add, '_fast_replay'))

# *** Metal direct dispatch ***

@unittest.skipIf(Device.DEFAULT != "METAL", "Metal support required")
class TestMetalDirectDispatch(unittest.TestCase):
  def test_direct_dispatch_correctness(self):
    """MetalGraph direct dispatch path produces correct results via JIT."""
    @TinyJit
    def f(a):
      b = (a + 1).realize()
      c = (b * 2).realize()
      return c
    for i in range(5):
      a = Tensor([float(i)]).contiguous().realize()
      c = f(a)
      np.testing.assert_allclose(c.numpy(), [(i + 1) * 2], rtol=1e-5)

  def test_direct_dispatch_sum(self):
    """MetalGraph direct dispatch produces correct results for sum operations."""
    @TinyJit
    def f(a): return a.sum().realize()
    for i in range(5):
      a = Tensor.ones(2048, 2048).contiguous().realize()
      out = f(a)
      np.testing.assert_allclose(out.numpy(), 2048*2048, rtol=1e-3)

# *** Metal GPU wake kernel and spin-wait ***

@unittest.skipIf(Device.DEFAULT != "METAL", "Metal support required")
class TestMetalWakeAndSync(unittest.TestCase):
  def test_wake_kernel_infrastructure(self):
    """MetalDevice has wake kernel infrastructure initialized."""
    dev = Device[Device.DEFAULT]
    self.assertTrue(hasattr(dev, '_wake_pipeline'))
    self.assertTrue(hasattr(dev, '_wake_buf'))
    self.assertTrue(hasattr(dev, '_wake_queue'))
    self.assertTrue(hasattr(dev, '_gpu_needs_wake'))

  def test_synchronize_correctness(self):
    """spin-wait synchronize produces correct results."""
    a = Tensor.ones(1024).contiguous().realize()
    b = (a + 1).realize()
    Device[Device.DEFAULT].synchronize()
    np.testing.assert_allclose(b.numpy(), np.full(1024, 2.0), rtol=1e-5)

  def test_synchronize_empty(self):
    """synchronize with no buffers in flight does not error."""
    Device[Device.DEFAULT].synchronize()
    Device[Device.DEFAULT].synchronize()

if __name__ == '__main__':
  unittest.main()
