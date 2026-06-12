import unittest
from unittest.mock import MagicMock
from tinygrad import Device
from tinygrad.uop.ops import Ops, UOp
from tinygrad.dtype import dtypes

@unittest.skipUnless(Device.DEFAULT == "METAL", "Metal device required to run")
class TestMetalGraph(unittest.TestCase):
  def setUp(self):
    from tinygrad.runtime.graph.metal import MetalGraph
    self.MetalGraph = MetalGraph
    self.dev = Device[Device.DEFAULT]

  def metal_buf(self, offset):
    buf = MagicMock()
    if offset > 0:
      buf.op = Ops.SLICE
      src = MagicMock()
      src.dtype = dtypes.uint8
      buf.src = (src, UOp.const(dtypes.weakint, offset))
      buf.dtype = dtypes.uint8
    else:
      buf.op = Ops.BUFFER
    buf.device = Device.DEFAULT
    return buf

  def call(self, *bufs):
    c = MagicMock()
    c.src = (MagicMock(op=Ops.PROGRAM),) + tuple(bufs)
    return c

  def test_supports_uop_normal_offset(self):
    assert self.MetalGraph.supports_uop([self.dev], self.call(self.metal_buf(0), self.metal_buf(100), self.metal_buf(0xFFFFFFFF))) is True

  def test_supports_uop_overflow_offset(self):
    assert self.MetalGraph.supports_uop([self.dev], self.call(self.metal_buf(0), self.metal_buf(0x100000000))) is False

  def test_supports_uop_nonmetal_buf(self):
    # non-SLICE ops should not be checked for offset
    buf = MagicMock()
    buf.op = Ops.BUFFER
    buf.device = Device.DEFAULT
    self.MetalGraph.supports_uop([self.dev], self.call(buf))

  def test_track_inflight(self):
    # Partially fixes #16595
    # TODO: Another unit test to make sure per-step time no longer linearly grows?
    # Need to make sure auto-cleanup of in-flight command buffers works correctly. This test checks
    # that the completion handler is called and the command buffer is removed from the in-flight set.

    # track_inflight must: (1) register a block that discards cbuf from the set when fired,
    # (2) stash the handler on cbuf so the C function pointer stays alive, and
    # (3) call addCompletedHandler before commit (Metal asserts otherwise)
    cbuf = self.dev.mtl_queue.commandBuffer().retained()
    self.dev.track_inflight(cbuf)
    # (2) CFUNCTYPE stashed on cbuf keeps the C function pointer alive
    self.assertTrue(callable(cbuf._h))
    cbuf.commit()
    cbuf.waitUntilCompleted()
    # (1) the completion handler discarded cbuf from the in-flight set
    self.assertNotIn(cbuf, self.dev.mtl_buffers_in_flight)
    # (3) addCompletedHandler before commit is proven by the fact that the set was cleaned
    # (Metal should assert in track_inflight if called after commit)

if __name__ == "__main__":
  unittest.main()
