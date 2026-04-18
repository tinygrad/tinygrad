import unittest

from tinygrad.runtime.support.compiler_cuda import _cuda_toolchain_cache_tag


class TestCUDAToolchainCacheTag(unittest.TestCase):
  def test_stable_for_identical_inputs(self):
    parts = {
      "cuda_path": "/opt/cuda-13.0",
      "nvrtc_path": "/opt/cuda-13.0/lib/libnvrtc.so.13",
      "nvjitlink_path": "/opt/cuda-13.0/lib/libnvJitLink.so.13",
      "nvrtc_version": "13.0",
      "ptx_mode": "1",
    }
    self.assertEqual(_cuda_toolchain_cache_tag(parts), _cuda_toolchain_cache_tag(parts))

  def test_order_independent(self):
    parts_a = {"b": "2", "a": "1", "c": "3"}
    parts_b = {"c": "3", "a": "1", "b": "2"}
    self.assertEqual(_cuda_toolchain_cache_tag(parts_a), _cuda_toolchain_cache_tag(parts_b))

  def test_changes_when_toolchain_path_changes(self):
    base = {
      "cuda_path": "/opt/cuda-13.0",
      "nvrtc_path": "/opt/cuda-13.0/lib/libnvrtc.so.13",
      "nvjitlink_path": "/opt/cuda-13.0/lib/libnvJitLink.so.13",
      "nvrtc_version": "13.0",
      "ptx_mode": "1",
    }
    changed = dict(base)
    changed["nvrtc_path"] = "/opt/cuda-13.2/lib/libnvrtc.so.13"
    self.assertNotEqual(_cuda_toolchain_cache_tag(base), _cuda_toolchain_cache_tag(changed))


if __name__ == "__main__":
  unittest.main()
