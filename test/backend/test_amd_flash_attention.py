import unittest
import numpy as np

from tinygrad import Device, Tensor, TinyJit
from extra.gemm.amd_flash_attention import amd_flash_attention_decode


@unittest.skipUnless(Device.DEFAULT.startswith("AMD"), "AMD flash attention required")
class TestAMDFlashAttention(unittest.TestCase):
  def test_short_decode_is_finite_and_matches_reference(self):
    rng = np.random.default_rng(1)
    q_np = rng.standard_normal((1, 16, 1, 256)).astype(np.float16)
    kv_np = rng.standard_normal((2, 1, 2, 8192, 256)).astype(np.float16)
    q, kv = Tensor(q_np).realize(), Tensor(kv_np).realize()

    @TinyJit
    def decode(q:Tensor, kv:Tensor): return amd_flash_attention_decode(q, kv, 25, 8192).realize()

    out = None
    for _ in range(3): out = decode(q, kv).numpy()
    assert out is not None
    q_ref = q_np[0, :, 0].astype(np.float32)
    k_ref, v_ref = kv_np[:, 0, :, :25].astype(np.float32)
    expected = np.empty((16, 256), dtype=np.float32)
    for head in range(16):
      scores = q_ref[head] @ k_ref[head // 8].T / np.sqrt(256)
      probs = np.exp(scores - scores.max())
      expected[head] = probs @ v_ref[head // 8] / probs.sum()

    self.assertTrue(np.isfinite(out).all())
    np.testing.assert_allclose(out[0, :, 0], expected, rtol=2e-3, atol=2e-3)


if __name__ == "__main__": unittest.main()
