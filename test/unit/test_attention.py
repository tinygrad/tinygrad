import unittest
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.apps.llm import apply_rope as apply_rope_new, precompute_freqs_cis, pairwise_topk

def apply_rope(x:Tensor, start_pos:int):
  B, H, T, Hd = x.shape
  precompute_freqs_cis.cache_clear()
  freqs_cis = precompute_freqs_cis(Hd, start_pos+T)[start_pos:start_pos+T]
  return apply_rope_new(x, freqs_cis)

class TestAttention(unittest.TestCase):
  def test_apply_rope(self):
    x = Tensor.randn(1, 2, 4, 8, dtype=dtypes.float32)
    result = apply_rope(x, 0)
    self.assertEqual(result.shape, x.shape)
    self.assertEqual(result.dtype, x.dtype)
    self.assertGreater((result - apply_rope(x, 5)).abs().max().item(), 1e-6)
    with self.assertRaises(AssertionError): apply_rope(Tensor.randn(1, 1, 4, 7, dtype=dtypes.float32), 0)

class TestPairwiseTopk(unittest.TestCase):
  def test_basic_topk(self):
    x = Tensor([[[1.0, 3.0, 2.0, 5.0, 4.0]]])
    vals, sel = pairwise_topk(x, 3)
    np.testing.assert_allclose(vals.numpy(), [[[3.0, 4.0, 5.0]]])
    np.testing.assert_equal(sel.numpy(), [[[1, 4, 3]]])

  def test_duplicates(self):
    x = Tensor([[[5.0, 5.0, 3.0, 5.0]]])
    vals, sel = pairwise_topk(x, 2)
    np.testing.assert_allclose(vals.numpy(), [[[5.0, 5.0]]])
    np.testing.assert_equal(sel.numpy(), [[[1, 0]]])

  def test_matches_numpy(self):
    np.random.seed(42)
    data = np.random.randn(4, 2, 16).astype(np.float32)
    vals, sel = pairwise_topk(Tensor(data), 5)
    for b in range(4):
      for t in range(2):
        expected = set(np.argsort(-data[b, t])[:5].tolist())
        self.assertEqual(set(sel.numpy()[b, t].tolist()), expected)
        np.testing.assert_allclose(vals.numpy()[b, t], data[b, t][sel.numpy()[b, t]])

if __name__ == '__main__':
  unittest.main()
