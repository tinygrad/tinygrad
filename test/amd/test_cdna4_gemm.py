# GEMM tests for the CDNA4 emulator
import os, unittest
import numpy as np
from tinygrad import Tensor

def is_cdna4_mock(): return os.environ.get("MOCKGPU_ARCH", "") == "cdna4"

@unittest.skipUnless(is_cdna4_mock(), "MOCKGPU_ARCH=cdna4 required")
class TestCDNA4GEMM(unittest.TestCase):
  def _matmul_check(self, M, K, N, dtype=np.float32, atol=1e-4, rtol=1e-4):
    rng = np.random.default_rng(42)
    a_np = rng.standard_normal((M, K)).astype(dtype)
    b_np = rng.standard_normal((K, N)).astype(dtype)
    expected = a_np @ b_np

    a = Tensor(a_np, device="AMD")
    b = Tensor(b_np, device="AMD")
    result = (a @ b).numpy()

    np.testing.assert_allclose(result, expected, atol=atol, rtol=rtol,
      err_msg=f"GEMM({M},{K},{N}) dtype={dtype.__name__} failed")

  def test_gemm_16x4x16(self): self._matmul_check(16,4,16)
  def test_gemm_32x2x32(self): self._matmul_check(32,2,32)
  def test_gemm_16x16x16(self): self._matmul_check(16,16,16)
  def test_gemm_32x32x32(self): self._matmul_check(32,32,32)
  def test_gemm_17x5x13(self): self._matmul_check(17,5,13, atol=1e-3)
  def test_gemm_48x7x33(self): self._matmul_check(48,7,33, atol=1e-3)
  def test_gemm_128x64x128(self): self._matmul_check(128,64,128, atol=1e-3, rtol=1e-3)
  def test_gemm_256x256x256(self): self._matmul_check(256,256,256, atol=1e-3, rtol=1e-3)
  def test_fp16_16x16x16(self): self._matmul_check(16,16,16, np.float16, 1e-3, 1e-2)
  def test_fp16_64x64x64(self): self._matmul_check(64,64,64, np.float16, 5e-3, 1e-2)

  def test_batched(self):
    rng = np.random.default_rng(7)
    B,M,K,N = 4,32,32,32
    a_np = rng.standard_normal((B,M,K)).astype(np.float32)
    b_np = rng.standard_normal((B,K,N)).astype(np.float32)

    a = Tensor(a_np, device="AMD")
    b = Tensor(b_np, device="AMD")

    np.testing.assert_allclose((a @ b).numpy(), a_np @ b_np, atol=1e-3, rtol=1e-3 )

if __name__ == "__main__":
  unittest.main()