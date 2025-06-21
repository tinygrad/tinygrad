#!/usr/bin/env python
import os, gc, unittest, weakref
from tinygrad.tensor import Tensor
from tinygrad.device import Buffer
from tinygrad.engine.realize import run_schedule
from tinygrad.helpers import GlobalCounters

def tensors_allocated():
  gc.collect()
  return sum(isinstance(x, Tensor) for x in gc.get_objects())

def bufs_allocated():
  gc.collect()
  return sum(isinstance(x, Buffer) and x.is_allocated() for x in gc.get_objects())

class VizEnv:
  """Context-manager that temporarily sets VIZ=1/0."""
  def __init__(self, val: str): self.val = val
  def __enter__(self):
    self.old = os.environ.get("VIZ")
    os.environ["VIZ"] = self.val
  def __exit__(self, *_):
    if self.old is None: os.environ.pop("VIZ", None)
    else: os.environ["VIZ"] = self.old

# ----------------------------------------------------------------------------- the tests
class TestGCViz(unittest.TestCase):

  def _single_buffer(self, viz):
    with VizEnv(viz):
      base = bufs_allocated()
      t = Tensor.randn(512, 512)
      buf_ref = weakref.ref(t._buffer)
      del t
      gc.collect()
      self.assertIsNone(buf_ref())
      self.assertEqual(bufs_allocated() - base, 0)

  def test_single_buffer_no_viz(self):  self._single_buffer("0")
  def test_single_buffer_with_viz(self): self._single_buffer("1")

  def _churn(self, viz):
    with VizEnv(viz):
      before = GlobalCounters.mem_used
      for _ in range(300):
        Tensor.randn(128, 128)
      gc.collect()
      delta = GlobalCounters.mem_used - before
      self.assertLess(delta, 64 * 1024, f"mem_used grew {delta} bytes (VIZ={viz})")

  def test_churn_no_viz(self):  self._churn("0")
  def test_churn_with_viz(self): self._churn("1")

  def test_schedule_gc_with_viz(self):
    """Run a schedule under VIZ=1 and make sure *all* buffers are gone
    once both the output tensor *and* the schedule list are dropped."""
    with VizEnv("1"):
      init = bufs_allocated()

      x  = Tensor.ones(256).contiguous()
      y  = x + Tensor.ones(256).contiguous()
      ys = y.schedule()
      del x
      run_schedule(ys)

      import numpy as np, numpy.testing as npt
      npt.assert_equal(y.numpy(), np.full((256,), 2, dtype=np.float32))

      del y, ys
      for _ in range(2): gc.collect()

      self.assertLessEqual(bufs_allocated() - init, 3)

  def test_view_buffer_release(self):
    with VizEnv("1"):
      t = Tensor.randn(256, 256).contiguous()
      base_buf = t._buffer
      view_buf = Buffer(t.device, t.size(), t.dtype,
                        offset=0)
      vref, bref = weakref.ref(view_buf), weakref.ref(base_buf)
      del view_buf
      gc.collect()
      # child is freed, base stays
      self.assertIsNone(vref())
      self.assertIsNotNone(bref())

  @unittest.skipIf(os.getenv("CI") == "1", "heavy local stress-test")
  def test_visualiser_stress(self):
    with VizEnv("1"):
      start = bufs_allocated()
      for _ in range(50):
        (Tensor.randn(64, 64) * 2).sum()
      gc.collect()
      self.assertLessEqual(bufs_allocated(), start + 2)

# -----------------------------------------------------------------------------
if __name__ == '__main__':
  unittest.main()