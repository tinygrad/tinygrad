import unittest
from tinygrad import Device
from tinygrad.helpers import CI
from tinygrad.runtime.ops_gpu import CLDevice, CLAllocator

@unittest.skipUnless(Device.DEFAULT in ["GPU"] and not CI, "Runs only on OpenCL (GPU)")
class TestOCLOOM(unittest.TestCase):
  def test_opencl_oom(self):
    with self.assertRaises(RuntimeError) as err:
      allocator = CLAllocator(CLDevice())
      for i in range(1_000_000):
        allocator.alloc(1_000_000_000)
    assert str(err.exception) == "OpenCL Error -6: CL_OUT_OF_HOST_MEMORY"
