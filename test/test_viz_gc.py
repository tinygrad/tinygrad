import os
os.environ["VIZ"] = "1"
import gc, unittest, weakref
from tinygrad.tensor import Tensor
from tinygrad.device import Buffer
from tinygrad.ops import tracked_ctxs, TrackedGraphRewrite

def count_buffers():
  gc.collect()
  return sum(1 for obj in gc.get_objects() if isinstance(obj, Buffer))

def count_uops():
  gc.collect()
  from tinygrad.ops import UOp
  return sum(1 for obj in gc.get_objects() if isinstance(obj, UOp))

class TestVizGC(unittest.TestCase):
  def setUp(self):
    self.original_viz = os.environ.get("VIZ", "0")
    os.environ["VIZ"] = "1"
    gc.collect()

  def tearDown(self):
    if self.original_viz != "0":
      os.environ["VIZ"] = self.original_viz
    elif "VIZ" in os.environ:
      del os.environ["VIZ"]
    gc.collect()

  def test_tracked_graph_rewrite_weakrefs(self):
    # Run a tracked rewrite and keep weakrefs to the UOp and TrackedGraphRewrite
    x = Tensor.ones(8).contiguous().realize()
    uop_ref = weakref.ref(x.lazydata)
    # Find the last tracked context and its sink
    self.assertTrue(tracked_ctxs, "No tracked contexts found")
    last_ctx = tracked_ctxs[-1][-1]
    self.assertIsInstance(last_ctx, TrackedGraphRewrite)
    sink_ref = last_ctx.sink if isinstance(last_ctx.sink, weakref.ReferenceType) else weakref.ref(last_ctx.sink)
    # Delete strong references and force GC
    del x
    gc.collect()
    # The UOp and sink should be collected (weakref returns None)
    self.assertIsNone(uop_ref(), "UOp not GC'd after rewrite")
    self.assertIsNone(sink_ref(), "TrackedGraphRewrite.sink not GC'd after rewrite")

  def test_no_buffer_leak_with_viz(self):
    # Baseline buffer count
    init_bufs = count_buffers()
    # Create and realize a tensor
    x = Tensor.ones(128).contiguous().realize()
    del x
    gc.collect()
    self.assertEqual(count_buffers(), init_bufs, "Buffer not GC'd with VIZ=1")

  def test_no_uop_leak_with_viz(self):
    # Baseline UOp count
    init_uops = count_uops()
    x = Tensor.ones(32).contiguous().realize()
    del x
    gc.collect()
    self.assertLessEqual(count_uops(), init_uops + 2, "UOp not GC'd with VIZ=1")  # allow a small margin

if __name__ == '__main__':
  unittest.main()