# Flash Attention tests for the CDNA4 emulator.
import unittest
import numpy as np
from typing import Any
from tinygrad import Tensor
from tinygrad.helpers import getenv

def is_cdna4_mock(): return getenv("MOCKGPU_ARCH", "") == "cdna4"

def _ref_attention(q, k, v, mask=None):
  scale = q.shape[-1] ** -0.5
  scores = (q @ k.swapaxes(-1, -2)) * scale
  if mask is not None: scores = np.where(mask, scores, -1e9)
  scores -= scores.max(axis=-1, keepdims=True)
  weights = np.exp(scores)
  weights /= weights.sum(axis=-1, keepdims=True)
  return weights @ v

def _test_attn_impl(B, H, S, D, dtype: Any = np.float32, atol=1e-4, rtol=1e-4):
  rng = np.random.default_rng(42)
  q_np = rng.standard_normal((B, H, S, D)).astype(dtype)
  k_np = rng.standard_normal((B, H, S, D)).astype(dtype)
  v_np = rng.standard_normal((B, H, S, D)).astype(dtype)

  expected = _ref_attention(q_np.astype(np.float32), k_np.astype(np.float32), v_np.astype(np.float32)).astype(dtype)

  q = Tensor(q_np, device="AMD")
  k = Tensor(k_np, device="AMD")
  v = Tensor(v_np, device="AMD")

  result = q.scaled_dot_product_attention(k, v).numpy()
  np.testing.assert_allclose(result, expected, atol=atol, rtol=rtol, err_msg=f"Attention(B={B},H={H},S={S},D={D}) dtype={dtype.__name__} failed")

def _test_causal_attn_impl(B, H, S, D, atol=1e-3, rtol=1e-3):
  rng = np.random.default_rng(0)
  q_np = rng.standard_normal((B, H, S, D)).astype(np.float32)
  k_np = rng.standard_normal((B, H, S, D)).astype(np.float32)
  v_np = rng.standard_normal((B, H, S, D)).astype(np.float32)

  causal = np.tril(np.ones((S, S), dtype=bool))
  expected = _ref_attention(q_np, k_np, v_np, mask=causal)

  q = Tensor(q_np, device="AMD")
  k = Tensor(k_np, device="AMD")
  v = Tensor(v_np, device="AMD")
  mask_t = Tensor(causal, device="AMD")

  result = q.scaled_dot_product_attention(k, v, attn_mask=mask_t).numpy()
  np.testing.assert_allclose(result, expected, atol=atol, rtol=rtol, err_msg="Causal attention failed")

@unittest.skipUnless(is_cdna4_mock(), "MOCKGPU_ARCH=cdna4 required")
class TestCDNA4Attention(unittest.TestCase):
  def test_attn_1h_16s_16d(self): _test_attn_impl(1, 1, 16, 16)
  def test_attn_1h_32s_32d(self): _test_attn_impl(1, 1, 32, 32)
  def test_attn_2h_32s_32d(self): _test_attn_impl(1, 2, 32, 32)
  def test_attn_8h_64s_64d(self): _test_attn_impl(2, 8, 64, 64, atol=1e-3, rtol=1e-3)
  def test_attn_8h_128s_64d(self): _test_attn_impl(2, 8, 128, 64, atol=1e-3, rtol=1e-3)
  def test_attn_8h_256s_64d(self): _test_attn_impl(2, 8, 256, 64, atol=1e-3, rtol=1e-3)
  def test_attn_16h_512s_64d(self): _test_attn_impl(1, 16, 512, 64, atol=2e-3, rtol=1e-3)
  def test_fp16_1h_32s_32d(self): _test_attn_impl(1, 1, 32, 32, np.float16, atol=1e-2, rtol=1e-2)
  def test_fp16_8h_64s_64d(self): _test_attn_impl(2, 8, 64, 64, np.float16, atol=2e-2, rtol=1e-2)
  def test_causal_8h_64s_64d(self): _test_causal_attn_impl(2, 8, 64, 64)

if __name__ == "__main__":
  unittest.main()