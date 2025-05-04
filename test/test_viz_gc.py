#!/usr/bin/env python
import gc, unittest, os
from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv
from tinygrad.device import Buffer

def count_buffers():
  gc.collect()
  return sum(1 for obj in gc.get_objects() if isinstance(obj, Buffer))

class TestVizGC(unittest.TestCase):
  def test_viz_buffer_gc(self):
    """Test buffer GC with VIZ enabled"""
    original_viz = os.environ.get("VIZ", "0")

    try:
      # Test with VIZ=0 (baseline)
      os.environ["VIZ"] = "0"
      init_bufs = count_buffers()
      x = Tensor.ones(256).contiguous().realize()
      y = Tensor.ones(5, 5).contiguous().schedule()
      del x, y
      gc.collect()
      self.assertEqual(count_buffers(), init_bufs, "Buffers not GC'd with VIZ=0")

      # Test with VIZ=1
      os.environ["VIZ"] = "1"
      x = Tensor.ones(256).contiguous().realize()
      y = Tensor.ones(5, 5).contiguous().schedule()
      del x, y
      gc.collect()
      self.assertEqual(count_buffers(), init_bufs, "Buffers not GC'd with VIZ=1")

    finally:
      if original_viz != "0":
        os.environ["VIZ"] = original_viz
      elif "VIZ" in os.environ:
        del os.environ["VIZ"]

  def test_cpu_buffer_gc(self):
    """Test CPU buffer GC with VIZ enabled"""
    original_device = getenv("DEVICE", "")
    original_viz = os.environ.get("VIZ", "0")

    try:
      os.environ["DEVICE"] = "CPU"
      os.environ["VIZ"] = "1"

      init_bufs = count_buffers()
      x = Tensor.ones(256).contiguous().realize()
      del x
      gc.collect()

      self.assertEqual(count_buffers(), init_bufs, "CPU buffers not GC'd with VIZ=1")

    finally:
      if original_device:
        os.environ["DEVICE"] = original_device
      else:
        del os.environ["DEVICE"]

      if original_viz != "0":
        os.environ["VIZ"] = original_viz
      elif "VIZ" in os.environ:
        del os.environ["VIZ"]

if __name__ == '__main__':
  unittest.main()
