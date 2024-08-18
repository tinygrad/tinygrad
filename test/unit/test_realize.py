import unittest
from unittest.mock import patch
from tinygrad.tensor import Tensor
from tinygrad.ops import UOps, UOp
from tinygrad.helpers import Context
from tinygrad.engine.realize import get_kernel
from tinygrad.device import Device


class TestGetKernel(unittest.TestCase):
  @patch("builtins.print")
  def test_debug_print_trigger(self, mock_print):
    t = Tensor.arange(10)
    t = t + t * Tensor.rand(10)
    with Context(DEBUG=1, BEAM=2):
      schedule_item = t.schedule()[-1]
      kernel = get_kernel(Device[Device.DEFAULT].renderer, schedule_item.ast)
      uops = kernel.linearize().uops
      if uops:
        sink = UOp(UOps.SINK, None, (uops[-1],))

    for call in mock_print.ase:
      self.assertNotIn("us", call[0][0])


if __name__ == "__main__":
  unittest.main(verbosity=2)
