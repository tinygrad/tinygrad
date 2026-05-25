import unittest
from unittest.mock import MagicMock, patch
from tinygrad import Device
from tinygrad.helpers import Context
from tinygrad.uop.ops import Ops
from tinygrad.dtype import dtypes
from test.helpers import call_is_graph

@unittest.skipUnless(Device.DEFAULT == "METAL", "Metal device required to run")
class TestMetalGraph(unittest.TestCase):
  def setUp(self):
    from tinygrad.runtime.graph.metal import MetalGraph
    self.MetalGraph = MetalGraph
    self.dev = Device[Device.DEFAULT]

  def metal_buf(self, offset):
    buf = MagicMock()
    if offset > 0:
      buf.op = Ops.BUFFER_VIEW
      buf.arg = (None, offset)
      buf.dtype = dtypes.uint8
    else:
      buf.op = Ops.BUFFER
    buf.device = Device.DEFAULT
    return buf

  def call(self, *bufs):
    c = MagicMock()
    c.src = (MagicMock(op=Ops.PROGRAM),) + tuple(bufs)
    return c

  def graph_runtime(self, fxn):
    from tinygrad.engine.realize import get_graph_runtime

    assert fxn.captured is not None
    assert len(fxn.captured.linear.src) == 1
    call = fxn.captured.linear.src[0]
    assert call_is_graph(call)
    rt = get_graph_runtime(call.src[0])
    assert isinstance(rt, self.MetalGraph)
    return rt

  def test_supports_uop_normal_offset(self):
    assert self.MetalGraph.supports_uop([self.dev], self.call(self.metal_buf(0), self.metal_buf(100), self.metal_buf(0xFFFFFFFF))) is True

  def test_supports_uop_overflow_offset(self):
    assert self.MetalGraph.supports_uop([self.dev], self.call(self.metal_buf(0), self.metal_buf(0x100000000))) is False

  def test_supports_uop_nonmetal_buf(self):
    # non-BUFFER_VIEW ops should not be checked for offset
    buf = MagicMock()
    buf.op = Ops.BUFFER
    buf.device = Device.DEFAULT
    self.MetalGraph.supports_uop([self.dev], self.call(buf))

  def test_static_graph_replay_does_not_wait_for_previous_command_buffer(self):
    from tinygrad import Tensor, TinyJit
    from tinygrad.runtime.graph import metal as metal_graph

    if self.dev.graph is None: self.skipTest("Metal graph not supported")

    with Context(DEBUG=0, JIT=1, JIT_BATCH_SIZE=32, PROFILE=0):
      w = Tensor.zeros(16, device=Device.DEFAULT).contiguous().realize()
      g = Tensor.ones(16, device=Device.DEFAULT).contiguous().realize()

      @TinyJit
      def step():
        w.assign(w + g).realize()
        w.assign(w + g).realize()

      for _ in range(3): step()
      self.dev.synchronize()
      rt = self.graph_runtime(step)
      assert rt.updatable == []

      with patch.object(metal_graph, "wait_check", wraps=metal_graph.wait_check) as wait_check:
        step()
        step()
        self.dev.synchronize()

      wait_check.assert_not_called()
      assert w[0].item() == 10

  def test_updatable_graph_replay_waits_before_replacing_icb_buffers(self):
    from tinygrad import Tensor, TinyJit
    from tinygrad.runtime.graph import metal as metal_graph

    if self.dev.graph is None: self.skipTest("Metal graph not supported")

    with Context(DEBUG=0, JIT=1, JIT_BATCH_SIZE=32, PROFILE=0):
      @TinyJit
      def step(a, b):
        c = (a + b).realize()
        return (c * b).realize()

      a = Tensor([2.0], device=Device.DEFAULT).contiguous().realize()
      b = Tensor([3.0], device=Device.DEFAULT).contiguous().realize()
      c = Tensor([4.0], device=Device.DEFAULT).contiguous().realize()
      d = Tensor([5.0], device=Device.DEFAULT).contiguous().realize()

      for _ in range(3): step(a, b)
      self.dev.synchronize()
      rt = self.graph_runtime(step)
      assert rt.updatable

      with patch.object(metal_graph, "wait_check", wraps=metal_graph.wait_check) as wait_check:
        step(c, d)
        out = step(a, b)
        self.dev.synchronize()

      assert wait_check.call_count == 1
      assert out[0].item() == 15

  def test_static_graph_replay_waits_when_profiling(self):
    from tinygrad import Tensor, TinyJit
    from tinygrad.runtime.graph import metal as metal_graph

    if self.dev.graph is None: self.skipTest("Metal graph not supported")

    with Context(DEBUG=0, JIT=1, JIT_BATCH_SIZE=32, PROFILE=1):
      w = Tensor.zeros(16, device=Device.DEFAULT).contiguous().realize()
      g = Tensor.ones(16, device=Device.DEFAULT).contiguous().realize()

      @TinyJit
      def step():
        w.assign(w + g).realize()
        w.assign(w + g).realize()

      for _ in range(3): step()
      self.dev.synchronize()
      rt = self.graph_runtime(step)
      assert rt.updatable == []

      with patch.object(metal_graph, "wait_check", wraps=metal_graph.wait_check) as wait_check:
        step()
        step()
        self.dev.synchronize()

      assert wait_check.call_count == 1
      assert w[0].item() == 10

if __name__ == "__main__":
  unittest.main()
