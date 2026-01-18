from dataclasses import dataclass
import unittest
from typing import Any
from tinygrad import Tensor, dtypes, Device
from tinygrad.device import is_dtype_supported
from tinygrad.runtime.autogen import dlpack as c_dlpack

def is_dlpack_supported(device: str | None = None) -> bool:
  """Check if device supports DLPack export (has _device_ptr method)."""
  if device is None: device = Device.DEFAULT
  return hasattr(Device[device].allocator, '_device_ptr')

class TestDLPackDevice(unittest.TestCase):
  def test_dlpack_device_cpu(self):
    t = Tensor([1, 2, 3], device="CPU")
    device_type, device_id = t.__dlpack_device__()
    self.assertEqual(device_type, c_dlpack.kDLCPU)
    self.assertEqual(device_id, 0)

@unittest.skipUnless(is_dlpack_supported(), f"DLPack not supported on {Device.DEFAULT}")
class TestDLPackExport(unittest.TestCase):
  def test_basic_export_1d(self):
    t = Tensor([1, 2, 3, 4], dtype=dtypes.float32)
    capsule = t.__dlpack__()
    self.assertIsNotNone(capsule)

  def test_basic_export_2d(self):
    t = Tensor([[1, 2], [3, 4]], dtype=dtypes.float32)
    capsule = t.__dlpack__()
    self.assertIsNotNone(capsule)

  def test_basic_export_3d(self):
    t = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=dtypes.float32)
    capsule = t.__dlpack__()
    self.assertIsNotNone(capsule)

  def test_basic_export_scalar(self):
    t = Tensor(42, dtype=dtypes.float32)
    capsule = t.__dlpack__()
    self.assertIsNotNone(capsule)

  def test_dtypes(self):
    for dt in [dtypes.float16, dtypes.float32, dtypes.float64, dtypes.bfloat16,
               dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64,
               dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64]:
      if not is_dtype_supported(dt): continue
      with self.subTest(dtype=dt):
        t = Tensor([1, 2, 3], dtype=dt)
        capsule = t.__dlpack__()
        self.assertIsNotNone(capsule, f"__dlpack__ returned None for {dt}")

  def test_dtype_bool(self):
    t = Tensor([True, False, True], dtype=dtypes.bool)
    capsule = t.__dlpack__()
    self.assertIsNotNone(capsule)

@unittest.skipUnless(is_dlpack_supported(), f"DLPack not supported on {Device.DEFAULT}")
class TestDLPackCopy(unittest.TestCase):
  def test_non_contiguous_with_copy_none(self):
    # Non-contiguous tensor should work with copy=None (default)
    t = Tensor([[1, 2, 3], [4, 5, 6]], dtype=dtypes.float32).T
    capsule = t.__dlpack__()  # Should succeed by making a copy
    self.assertIsNotNone(capsule)

  def test_non_contiguous_with_copy_true(self):
    t = Tensor([[1, 2, 3], [4, 5, 6]], dtype=dtypes.float32).T
    capsule = t.__dlpack__(copy=True)
    self.assertIsNotNone(capsule)

  def test_non_contiguous_with_copy_false(self):
    t = Tensor([[1, 2, 3], [4, 5, 6]], dtype=dtypes.float32).T
    with self.assertRaises(BufferError):
      t.__dlpack__(copy=False)

  def test_contiguous_with_copy_false(self):
    t = Tensor([1, 2, 3, 4], dtype=dtypes.float32).realize()
    capsule = t.__dlpack__(copy=False)
    self.assertIsNotNone(capsule)

@unittest.skipUnless(is_dlpack_supported(), f"DLPack not supported on {Device.DEFAULT}")
class TestDLPackVersion(unittest.TestCase):
  def test_max_version_compatible(self):
    t = Tensor([1, 2, 3], dtype=dtypes.float32)
    capsule = t.__dlpack__(max_version=(1, 0))
    self.assertIsNotNone(capsule)

  def test_max_version_newer(self):
    t = Tensor([1, 2, 3], dtype=dtypes.float32)
    capsule = t.__dlpack__(max_version=(2, 0))
    self.assertIsNotNone(capsule)

  def test_max_version_incompatible(self):
    t = Tensor([1, 2, 3], dtype=dtypes.float32)
    with self.assertRaises(RuntimeError):
      t.__dlpack__(max_version=(0, 5))

@unittest.skipUnless(is_dlpack_supported(), f"DLPack not supported on {Device.DEFAULT}")
class TestDLPackCrossDevice(unittest.TestCase):
  def test_dl_device_same_device(self):
    t = Tensor([1, 2, 3], dtype=dtypes.float32, device="CPU")
    capsule = t.__dlpack__(dl_device=(c_dlpack.kDLCPU, 0))
    self.assertIsNotNone(capsule)

  def test_dl_device_cross_device_copy_false(self):
    # Should raise BufferError if cross-device and copy=False
    t = Tensor([1, 2, 3], dtype=dtypes.float32, device="CPU")
    # Request a different device type
    with self.assertRaises(BufferError):
      t.__dlpack__(dl_device=(c_dlpack.kDLCUDA, 0), copy=False)

class TestDLPackNumpyInterop(unittest.TestCase):
  """Test DLPack interop with NumPy. Uses CPU tensors since NumPy only supports CPU DLPack."""
  @classmethod
  def setUpClass(cls):
    try:
      import numpy as np
      cls.np = np
    except ImportError:
      raise unittest.SkipTest("NumPy not installed")

  def test_1d(self):
    t = Tensor([1.0, 2.0, 3.0, 4.0], dtype=dtypes.float32, device="CPU")
    arr = self.np.from_dlpack(t)
    self.assertEqual(arr.tolist(), [1.0, 2.0, 3.0, 4.0])

  def test_2d(self):
    t = Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtypes.float32, device="CPU")
    arr = self.np.from_dlpack(t)
    self.assertEqual(arr.tolist(), [[1.0, 2.0], [3.0, 4.0]])

  def test_3d(self):
    t = Tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=dtypes.float32, device="CPU")
    arr = self.np.from_dlpack(t)
    self.assertEqual(arr.tolist(), [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])

  def test_capsule_keeps_tensor_alive(self):
    t = Tensor([1.0, 2.0, 3.0, 4.0], dtype=dtypes.float32, device="CPU")
    capsule = t.__dlpack__()
    del t  # Delete original tensor reference

    @dataclass
    class DLPackProxy:
      capsule: Any
      def __dlpack__(self, *args, **kwargs):
        return capsule

    proxy = DLPackProxy(capsule=capsule)
    arr = self.np.from_dlpack(proxy)
    self.assertEqual(arr.tolist(), [1.0, 2.0, 3.0, 4.0])

class TestDLPackPytorchInterop(unittest.TestCase):
  """Test DLPack interop with PyTorch. Uses CPU tensors to avoid CUDA initialization issues."""
  @classmethod
  def setUpClass(cls):
    try:
      import torch
      cls.torch = torch
    except ImportError:
      raise unittest.SkipTest("PyTorch not installed")

  def test_1d(self):
    t = Tensor([1.0, 2.0, 3.0, 4.0], dtype=dtypes.float32, device="CPU")
    pt = self.torch.from_dlpack(t)
    self.assertEqual(pt.tolist(), [1.0, 2.0, 3.0, 4.0])

@unittest.skipUnless(is_dlpack_supported(), f"DLPack not supported on {Device.DEFAULT}")
class TestDLPackMemoryManagement(unittest.TestCase):
  def test_multiple_capsules_from_same_tensor(self):
    t = Tensor([1, 2, 3, 4], dtype=dtypes.float32)
    capsule1 = t.__dlpack__()
    capsule2 = t.__dlpack__()
    self.assertIsNotNone(capsule1)
    self.assertIsNotNone(capsule2)

class TestDLPackImport(unittest.TestCase):
  """Test importing tensors via Tensor.from_dlpack()."""
  @classmethod
  def setUpClass(cls):
    try:
      import numpy as np
      cls.np = np
    except ImportError:
      raise unittest.SkipTest("NumPy not installed")

  def test_import_1d(self):
    arr = self.np.array([1.0, 2.0, 3.0, 4.0], dtype=self.np.float32)
    t = Tensor.from_dlpack(arr)
    self.assertEqual(t.tolist(), [1.0, 2.0, 3.0, 4.0])

  def test_import_2d(self):
    arr = self.np.array([[1.0, 2.0], [3.0, 4.0]], dtype=self.np.float32)
    t = Tensor.from_dlpack(arr)
    self.assertEqual(t.tolist(), [[1.0, 2.0], [3.0, 4.0]])

  def test_import_3d(self):
    arr = self.np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=self.np.float32)
    t = Tensor.from_dlpack(arr)
    self.assertEqual(t.tolist(), [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])

  def test_import_dtypes(self):
    for np_dt, tg_dt in [(self.np.int32, dtypes.int32), (self.np.int64, dtypes.int64),
                         (self.np.float32, dtypes.float32), (self.np.float64, dtypes.float64)]:
      with self.subTest(dtype=np_dt):
        arr = self.np.array([1, 2, 3], dtype=np_dt)
        t = Tensor.from_dlpack(arr)
        self.assertEqual(t.dtype, tg_dt)

  def test_import_preserves_device(self):
    arr = self.np.array([1.0, 2.0, 3.0], dtype=self.np.float32)
    t = Tensor.from_dlpack(arr)
    self.assertEqual(t.device, "CPU")

class TestDLPackRoundTrip(unittest.TestCase):
  """Test round-trip: tinygrad -> numpy -> tinygrad."""
  @classmethod
  def setUpClass(cls):
    try:
      import numpy as np
      cls.np = np
    except ImportError:
      raise unittest.SkipTest("NumPy not installed")

  def test_roundtrip_1d(self):
    t1 = Tensor([1.0, 2.0, 3.0, 4.0], dtype=dtypes.float32, device="CPU")
    arr = self.np.from_dlpack(t1)
    t2 = Tensor.from_dlpack(arr)
    self.assertEqual(t1.tolist(), t2.tolist())

  def test_roundtrip_2d(self):
    t1 = Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtypes.float32, device="CPU")
    arr = self.np.from_dlpack(t1)
    t2 = Tensor.from_dlpack(arr)
    self.assertEqual(t1.tolist(), t2.tolist())

  def test_roundtrip_multiple_dtypes(self):
    for dt, np_dt in [(dtypes.float32, self.np.float32), (dtypes.int32, self.np.int32)]:
      with self.subTest(dtype=dt):
        t1 = Tensor([1, 2, 3, 4], dtype=dt, device="CPU")
        arr = self.np.from_dlpack(t1)
        t2 = Tensor.from_dlpack(arr)
        self.assertEqual(t1.tolist(), t2.tolist())
        self.assertEqual(t2.dtype, dt)

  def test_tinygrad_to_tinygrad(self):
    t1 = Tensor([1.0, 2.0, 3.0], dtype=dtypes.float32, device="CPU")
    t2 = Tensor.from_dlpack(t1)
    self.assertEqual(t1.tolist(), t2.tolist())

if __name__ == "__main__":
  unittest.main()
