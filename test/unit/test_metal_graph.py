import unittest
from unittest.mock import MagicMock
from tinygrad import Device
from tinygrad.engine.realize import CompiledRunner

@unittest.skipUnless(Device.DEFAULT == "METAL", "Metal device required to run")
class TestMetalGraph(unittest.TestCase):
  def setUp(self):
    from tinygrad.runtime.graph.metal import MetalGraph
    from tinygrad.runtime.ops_metal import MetalBuffer
    self.MetalGraph = MetalGraph
    self.MetalBuffer = MetalBuffer
    self.dev = Device[Device.DEFAULT]

  def metal_buf(self, offset): return MagicMock(_buf=self.MetalBuffer(MagicMock(), 4, offset))

  def ei(self, *bufs):
    ei = MagicMock()
    ei.prg = MagicMock(spec=CompiledRunner)
    ei.bufs = list(bufs)
    return ei

  def test_supports_exec_item_normal_offset(self):
    assert self.MetalGraph.supports_exec_item([self.dev], self.ei(self.metal_buf(0), self.metal_buf(100), self.metal_buf(0xFFFFFFFF))) is True

  def test_supports_exec_item_overflow_offset(self):
    assert self.MetalGraph.supports_exec_item([self.dev], self.ei(self.metal_buf(0), self.metal_buf(0x100000000))) is False

  def test_supports_exec_item_nonmetal_buf(self):
    # HCQBuffer.offset is a method, not an int — must not crash
    self.MetalGraph.supports_exec_item([self.dev], self.ei(MagicMock(**{"_buf.offset": lambda: 0})))

if __name__ == "__main__":
  unittest.main()
