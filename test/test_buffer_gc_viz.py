#!/usr/bin/env python3
import unittest, gc, os
from tinygrad import Tensor

class TestBufferGCViz(unittest.TestCase):
  def test_buffer_gc_with_viz(self):
    """Test that buffers are properly garbage collected when VIZ=1"""
    os.environ["VIZ"] = "1"
    try:
      a = Tensor([1,2,3,4])
      initial_refs = len(gc.get_referrers(a._buffer))
      b = a + 1
      c = b * 2
      del b, c
      gc.collect()
      final_refs = len(gc.get_referrers(a._buffer))
      self.assertLessEqual(final_refs, initial_refs + 1)
    finally:
      os.environ.pop("VIZ", None)

if __name__ == "__main__":
  unittest.main()