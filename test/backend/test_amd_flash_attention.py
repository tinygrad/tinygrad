import unittest
import numpy as np

from tinygrad import Device, Tensor, TinyJit
from tinygrad.llm.kernels.amd import amd_flash_attention_decode


@unittest.skipUnless(Device.DEFAULT.startswith("AMD"), "AMD flash attention required")
class TestAMDFlashAttention(unittest.TestCase):
  def _test_decode(self, max_kv_len:int, valid_kv_len:int, n_heads:int=16, n_kv_heads:int=2):
    rng = np.random.default_rng(1)
    q_np = rng.standard_normal((1, n_heads, 1, 256)).astype(np.float16)
    kv_np = rng.standard_normal((2, 1, n_kv_heads, max_kv_len, 256)).astype(np.float16)
    q, kv = Tensor(q_np).realize(), Tensor(kv_np).realize()

    @TinyJit
    def decode(q:Tensor, kv:Tensor): return amd_flash_attention_decode(q, kv, valid_kv_len, max_kv_len).realize()

    out = None
    for _ in range(3): out = decode(q, kv).numpy()
    assert out is not None
    q_ref = q_np[0, :, 0].astype(np.float32)
    k_ref, v_ref = kv_np[:, 0, :, :valid_kv_len].astype(np.float32)
    expected = np.empty((n_heads, 256), dtype=np.float32)
    for head in range(n_heads):
      scores = q_ref[head] @ k_ref[head // (n_heads // n_kv_heads)].T / np.sqrt(256)
      probs = np.exp(scores - scores.max())
      expected[head] = probs @ v_ref[head // (n_heads // n_kv_heads)] / probs.sum()

    self.assertTrue(np.isfinite(out).all())
    np.testing.assert_allclose(out[0, :, 0], expected, rtol=2e-3, atol=2e-3)

  def test_short_decode_is_finite_and_matches_reference(self): self._test_decode(8192, 25)

  def test_six_query_heads_per_kv_head(self): self._test_decode(8192, 25, n_heads=12, n_kv_heads=2)

  def test_hierarchical_decode_matches_reference(self): self._test_decode(16384, 4097)


if __name__ == "__main__": unittest.main()
