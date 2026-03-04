import unittest
from unittest.mock import MagicMock
from tinygrad import Device
from tinygrad.engine.realize import CompiledRunner
from tinygrad.runtime.graph.metal import MetalGraph

def _ei(*offsets):
  ei = MagicMock()
  ei.prg = MagicMock(spec=CompiledRunner)
  ei.bufs = [None if o is None else MagicMock(**{"_buf.offset": o}) for o in offsets]
  return ei

@unittest.skipUnless(Device.DEFAULT == "METAL", "Metal device required to run")
class TestMetalGraph(unittest.TestCase):
  def setUp(self): self.dev = Device[Device.DEFAULT]

  def test_supports_exec_item_normal_offset(self):
    assert MetalGraph.supports_exec_item([self.dev], _ei(0, 100, 0xFFFFFFFF)) is True

  def test_supports_exec_item_overflow_offset(self):
    # overflow is rejected before reaching GraphRunner.supports_exec_item, so devs doesn't matter
    assert MetalGraph.supports_exec_item([], _ei(0, 0x100000000)) is False

if __name__ == "__main__":
  unittest.main()
