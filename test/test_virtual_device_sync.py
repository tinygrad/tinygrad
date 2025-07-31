import unittest
from tinygrad.tensor import Tensor
from tinygrad.device import Device

@unittest.skipUnless(Device.DEFAULT in ["METAL", "CUDA", "HIP"], "virtual device support required")
class TestVirtualDeviceSync(unittest.TestCase):
  def setUp(self):
    self.d0, self.d1 = f"{Device.DEFAULT}:0", f"{Device.DEFAULT}:1"
    # Skip if virtual devices aren't available
    try:
      Tensor([1], device=self.d0).realize()
      Tensor([1], device=self.d1).realize()  
    except Exception:
      self.skipTest("need at least two virtual devices")

  def test_cross_virtual_device_d0_to_d1(self):
    """Test cross-virtual device transfers d0->d1 with race amplification"""
    for i in range(200):  # High iteration count to surface races
      # Create distinctive data on device 0
      x = Tensor.arange(64, device=self.d0).contiguous().realize()
      # Transfer to device 1 and immediately read (race window)
      y = x.to(self.d1).realize()
      # Verify data integrity - would fail if reader ran before copy completed
      self.assertEqual(x.tolist(), y.tolist(), f"iteration {i}: d0->d1 transfer race")

  def test_cross_virtual_device_d1_to_d0(self):
    """Test cross-virtual device transfers d1->d0 with race amplification"""
    for i in range(200):
      # Create data on device 1  
      x = Tensor.arange(64, device=self.d1).contiguous().realize()
      # Transfer to device 0 and immediately read
      y = x.to(self.d0).realize()
      self.assertEqual(x.tolist(), y.tolist(), f"iteration {i}: d1->d0 transfer race")

  def test_cross_virtual_with_computation(self):
    """Test that computation after cross-device transfer is consistent"""
    for i in range(100):
      # Create and compute on device 0
      a = Tensor.arange(32, device=self.d0).realize()
      b = (a * 2 + 1).realize()
      # Transfer to device 1 and compute immediately
      c = b.to(self.d1).realize()
      d = (c + 10).realize()
      # Verify computation consistency
      expected = ((Tensor.arange(32) * 2 + 1) + 10).tolist()
      self.assertEqual(d.tolist(), expected, f"iteration {i}: computation after transfer")

if __name__ == '__main__':
  unittest.main()