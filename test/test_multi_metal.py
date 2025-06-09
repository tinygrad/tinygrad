import unittest
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.device import Device

# Helper to get available Metal device IDs.
def get_metal_device_ids():
  return [d for d in Device._devices if d.startswith("METAL")]

@unittest.skipIf(len(get_metal_device_ids()) < 2, "Requires at least two Metal devices")
class TestMetalMultiDeviceSync(unittest.TestCase):

  def test_cross_device_transfers_are_exact(self):
    """
    This test validates that memory is copied correctly between all available
    Metal devices. It uses a deterministic pattern to ensure transfers are bit-for-bit accurate.
    """
    device_ids = get_metal_device_ids()
    source_tensors = []
    for i, device_id in enumerate(device_ids):
      # The value is i + 1 to avoid using zero, making verification unambiguous.
      fill_value = float(i + 1)
      t = Tensor.full((128, 128), fill_value=fill_value, device=device_id)
      source_tensors.append(t)

    for src_idx, src_device_id in enumerate(device_ids):
      for dst_idx, dst_device_id in enumerate(device_ids):
        if src_idx == dst_idx:
          continue
        with self.subTest(f"Transfer from {src_device_id} to {dst_device_id}"):
          # Get the original tensor from the source device.
          original_tensor = source_tensors[src_idx]
          expected_data = np.full(original_tensor.shape, fill_value=float(src_idx + 1), dtype=np.float32)
          transferred_tensor = original_tensor.to(dst_device_id)
          transferred_tensor.realize()

          np.testing.assert_array_equal(transferred_tensor.numpy(), expected_data,
                                        err_msg=f"Data mismatch after transfer from {src_device_id} to {dst_device_id}")

if __name__ == '__main__':
  unittest.main()
