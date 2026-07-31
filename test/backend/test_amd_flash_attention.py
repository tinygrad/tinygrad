import unittest
import numpy as np

from tinygrad import Device, Tensor, TinyJit
from tinygrad.llm.kernels.amd import amd_flash_attention_decode, flash_attention_causal_cached


@unittest.skipUnless(Device.DEFAULT.startswith("AMD"), "AMD flash attention required")
class TestAMDFlashAttention(unittest.TestCase):
  def _test_decode(self, max_kv_len:int, valid_kv_len:int, n_heads:int=16, n_kv_heads:int=2, quantized:bool=False):
    rng = np.random.default_rng(1)
    q_np = rng.standard_normal((1, n_heads, 1, 256)).astype(np.float16)
    kv_np = rng.standard_normal((2, 1, n_kv_heads, max_kv_len, 256)).astype(np.float16)
    scale_np = np.maximum(np.max(np.abs(kv_np.astype(np.float32)), axis=-1), 1e-8) / 127
    if quantized:
      kv_np = np.clip(np.rint(kv_np.astype(np.float32) / scale_np[..., None]), -127, 127).astype(np.int8)
    q, kv = Tensor(q_np).realize(), Tensor(kv_np).realize()
    scale = Tensor(scale_np.astype(np.float16)).realize() if quantized else None

    @TinyJit
    def decode(q:Tensor, kv:Tensor): return amd_flash_attention_decode(q, kv, valid_kv_len, max_kv_len, scale).realize()

    out = None
    for _ in range(3): out = decode(q, kv).numpy()
    assert out is not None
    q_ref = q_np[0, :, 0].astype(np.float32)
    kv_ref = kv_np.astype(np.float32) * scale_np[..., None] if quantized else kv_np.astype(np.float32)
    k_ref, v_ref = kv_ref[:, 0, :, :valid_kv_len]
    expected = np.empty((n_heads, 256), dtype=np.float32)
    for head in range(n_heads):
      scores = q_ref[head] @ k_ref[head // (n_heads // n_kv_heads)].T / np.sqrt(256)
      probs = np.exp(scores - scores.max())
      expected[head] = probs @ v_ref[head // (n_heads // n_kv_heads)] / probs.sum()

    self.assertTrue(np.isfinite(out).all())
    np.testing.assert_allclose(out[0, :, 0], expected, rtol=2e-3, atol=2e-3)

  def test_short_decode_is_finite_and_matches_reference(self): self._test_decode(8192, 25)

  def test_q8_cache_matches_dequantized_reference(self): self._test_decode(8192, 25, quantized=True)

  def test_q8_cached_prefill_matches_dequantized_reference(self):
    rng = np.random.default_rng(2)
    heads, kv_heads, tokens, dim = 16, 2, 32, 256
    q = rng.standard_normal((1, heads, tokens, dim)).astype(np.float16)
    kv = rng.standard_normal((2, 1, kv_heads, tokens, dim)).astype(np.float16)
    scale = np.maximum(np.max(np.abs(kv.astype(np.float32)), axis=-1), 1e-8) / 127
    packed = np.clip(np.rint(kv.astype(np.float32) / scale[..., None]), -127, 127).astype(np.int8)
    got = flash_attention_causal_cached(Tensor(q).realize(), Tensor(packed).realize(), tokens, tokens,
                                        Tensor(scale.astype(np.float16)).realize()).numpy()
    dequant = packed.astype(np.float32) * scale.astype(np.float16).astype(np.float32)[..., None]
    expected = np.empty_like(got)
    for head in range(heads):
      scores = q[0, head].astype(np.float32) @ dequant[0, 0, head // (heads // kv_heads)].T / np.sqrt(dim)
      scores[np.triu_indices(tokens, 1)] = -np.inf
      probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
      expected[0, head] = probs @ dequant[1, 0, head // (heads // kv_heads)] / probs.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(got, expected, rtol=2e-3, atol=2e-3)

  def test_six_query_heads_per_kv_head(self): self._test_decode(8192, 25, n_heads=12, n_kv_heads=2)

  def test_hierarchical_decode_matches_reference(self): self._test_decode(16384, 4097)


if __name__ == "__main__": unittest.main()
