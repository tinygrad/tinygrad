import unittest
from unittest.mock import MagicMock
from tinygrad import Device
from tinygrad.engine.realize import CompiledRunner
from tinygrad.runtime.ops_metal import MetalBuffer

def metal_buf(offset): return MagicMock(_buf=MetalBuffer(MagicMock(), 4, offset))

def ei(*bufs):
  ei = MagicMock()
  ei.prg = MagicMock(spec=CompiledRunner)
  ei.bufs = list(bufs)
  return ei

@unittest.skipUnless(Device.DEFAULT == "METAL", "Metal device required to run")
class TestMetalGraph(unittest.TestCase):
  def setUp(self):
    from tinygrad.runtime.graph.metal import MetalGraph
    self.MetalGraph = MetalGraph
    self.dev = Device[Device.DEFAULT]

  def test_supports_exec_item_normal_offset(self):
    assert self.MetalGraph.supports_exec_item([self.dev], ei(metal_buf(0), metal_buf(100), metal_buf(0xFFFFFFFF))) is True

  def test_supports_exec_item_overflow_offset(self):
    assert self.MetalGraph.supports_exec_item([self.dev], ei(metal_buf(0), metal_buf(0x100000000))) is False

  def test_supports_exec_item_nonmetal_buf(self):
    # HCQBuffer.offset is a method, not an int — must not crash
    self.MetalGraph.supports_exec_item([self.dev], ei(MagicMock(**{"_buf.offset": lambda: 0})))

if __name__ == "__main__":
  unittest.main()
