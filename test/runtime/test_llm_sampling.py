import unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor
from tinygrad.llm.kernels.amd_qwen import nucleus
from tinygrad.llm.qwen4exp import Transformer, TransformerConfig

def nucleus_ref(z:np.ndarray, p:float) -> np.ndarray:
  # the smallest set of the most likely tokens with a probability of at least p
  pr = np.exp(z - z.max())
  pr /= pr.sum()
  order = np.argsort(-pr, kind="stable")
  keep = np.zeros(len(z), dtype=bool)
  keep[order[:int(np.searchsorted(np.cumsum(pr[order]), p)) + 1]] = True
  return keep

class TestNucleus(unittest.TestCase):
  def test_matches_sorted(self):
    rng = np.random.default_rng(0)
    for scale, p, V, n in [(3.0, 0.95, 1000, 1000), (8.0, 0.9, 2000, 2000), (2.0, 0.95, 1024, 1000)]:
      z = (rng.standard_normal((3, V)) * scale).astype(np.float32)
      got = nucleus(Tensor(z), n, Tensor([[p]] * 3)).numpy()
      for b in range(3):
        ref = nucleus_ref(z[b, :n], p)
        # (every token of the nucleus, past it only ones as likely as its last to the search's resolution)
        self.assertFalse((ref & ~got[b, :n]).any())
        pr = np.exp(z[b, :n] - z[b, :n].max())
        self.assertTrue(np.all(np.log(pr[got[b, :n] & ~ref]) >= np.log(pr[ref].min()) - 1e-3))
        self.assertFalse(got[b, n:].any(), "a padded column")

  def test_one(self):
    # top_p 1: every token, a peaked distribution: its top one
    z = np.array([[5.0, 1.0, 0.0, -1.0]], dtype=np.float32)
    np.testing.assert_equal(nucleus(Tensor(z), 4, Tensor([[1.0]])).numpy(), [[True] * 4])
    np.testing.assert_equal(nucleus(Tensor(z), 4, Tensor([[0.9]])).numpy(), [[True, False, False, False]])

class TestSamplingModes(unittest.TestCase):
  @staticmethod
  def config():
    return TransformerConfig(num_blocks=1, dim=8, hidden_dim=16, n_heads=2, n_kv_heads=2, norm_eps=1e-6,
                             vocab_size=32, head_dim=4, rope_theta=10000, rope_dim=4, v_head_dim=4, max_context=32)

  def test_greedy_and_stochastic_jits_are_separate(self):
    Tensor.manual_seed(81)
    model = Transformer(self.config())
    expected = model.forward(Tensor([[1, 2]]), 0, None).item()
    for temperature in (0., .7, 0., .4, 0.):
      model._cached_tokens = []
      gen = model.generate([1, 2], chunk_size=8, temperature=temperature, top_p=.9)
      if temperature == 0:
        with patch("tinygrad.llm.qwen4exp.nucleus", side_effect=AssertionError("greedy sampling must not compute the nucleus")), \
             patch.object(Tensor, "rand_like", side_effect=AssertionError("greedy sampling must not draw noise")):
          self.assertEqual(next(gen), expected)
      else: self.assertIn(next(gen), range(32))
      gen.close()
    self.assertIsNot(model.jits[8], model.jits_greedy[8])


if __name__ == "__main__":
  unittest.main()
