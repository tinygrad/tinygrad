#!/usr/bin/env python3
import pytest
from tinygrad import Tensor, Device

class TestVirtualDeviceSync:
  def test_cross_virtual_device_transfers(self):
    """Test that cross-virtual device transfers are properly synchronized to prevent race conditions."""

    d0, d1 = f"{Device.DEFAULT}:0", f"{Device.DEFAULT}:1"

    # Skip if virtual devices don't exist (try to create tensors)
    try:
      Tensor([1], device=d0).realize()
      Tensor([1], device=d1).realize()
    except Exception:
      pytest.skip(f"virtual devices {d0} and {d1} not available")

    # Test d0 -> d1 transfers (amplify race with multiple iterations)
    for i in range(200):
      x = Tensor.randn(64, device=d0)
      y = x.to(d1)  # cross-virtual transfer that needs sync
      z = y + 1     # computation on d1 that could race with the copy
      result = z.numpy()  # force execution
      assert result.shape == (64,), f"iteration {i}: wrong shape {result.shape}"

    # Test d1 -> d0 transfers
    for i in range(200):
      x = Tensor.randn(64, device=d1)
      y = x.to(d0)  # cross-virtual transfer that needs sync
      z = y + 1     # computation on d0 that could race with the copy
      result = z.numpy()  # force execution
      assert result.shape == (64,), f"iteration {i}: wrong shape {result.shape}"