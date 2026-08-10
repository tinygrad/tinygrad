import unittest, numpy as np
from unittest.mock import patch
from tinygrad import Device, Tensor
from tinygrad.uop.ops import UOp, Ops
from tinygrad.dtype import dtypes
from tinygrad.runtime.support import hcq2

class TestKernelCopy(unittest.TestCase):
  def test_copy_lowers_to_kernel(self):
    dst, src = (UOp.new_buffer("CPU", 16, dtypes.float) for _ in range(2))
    call = src.copy_to_device("CPU").call(dst, src)

    with patch.object(hcq2, "HCQ_DEVS", frozenset(("CPU",))), patch.object(type(Device["CPU"]), "has_copy_queue", False):
      lowered = hcq2.pm_insert_copy_staging.rewrite(call)
    self.assertIs(lowered.src[0].op, Ops.PROGRAM)
    self.assertEqual(lowered.src[1:], call.src[1:]) # same dst and src, the copy is just a kernel now

  def test_copy_keeps_copy_queue(self):
    dst, src = (UOp.new_buffer("CPU", 16, dtypes.float) for _ in range(2))
    call = src.copy_to_device("CPU").call(dst, src)

    with patch.object(hcq2, "HCQ_DEVS", frozenset(("CPU",))), patch.object(type(Device["CPU"]), "has_copy_queue", True):
      self.assertIsNone(hcq2.pm_insert_copy_staging.rewrite(call))

@unittest.skipIf(Device[Device.DEFAULT].has_copy_queue, "device has a copy queue, run with AMD_DISABLE_SDMA=1")
class TestKernelCopyExec(unittest.TestCase):
  def test_copy_across_devices(self):
    a = (Tensor.ones(64) * 2).to(Device.DEFAULT).contiguous().realize()
    np.testing.assert_equal(a.to(f"{Device.DEFAULT.split(':')[0]}:1").contiguous().realize().numpy(), np.full(64, 2))

  def test_copy_from_host(self):
    a = Tensor(np.arange(64, dtype=np.float32)).to(Device.DEFAULT).contiguous().realize()
    np.testing.assert_equal(a.numpy(), np.arange(64))

if __name__ == "__main__":
  unittest.main()
