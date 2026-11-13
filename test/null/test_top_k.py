import unittest, math
import numpy as np
from tinygrad import Tensor
from tinygrad.llm.kernels import amd_qwen, amd_glm
from tinygrad.llm.serve import sampling_top_k, Request
from tinygrad.llm import glm5next, qwen4exp

def _expected(z:np.ndarray, n:int, k:int) -> np.ndarray:
  V = z.shape[-1]
  if k < 1 or k >= n: return np.ones(V, bool)
  kth = np.sort(z[:n])[::-1][k-1]
  out = np.zeros(V, bool)
  out[:n] = z[:n] >= kth
  return out

class TestTopK(unittest.TestCase):
  def test_mask_exact(self):
    # the k-th largest found on every bit of the floats' keys: ties at the k-th all in, negatives, -inf, k none / past the vocabulary
    rng = np.random.default_rng(0)
    B, V, n = 7, 1000, 990
    z = rng.standard_normal((B, V)).astype(np.float32) * 5
    z[1, :50] = 3.0
    z[2] = -np.abs(z[2])
    z[3, 10:900] = -math.inf
    z[4] = np.float32(1.0) + np.arange(V, dtype=np.float32) * np.float32(1e-7)  # (neighbours an ulp or so apart)
    ks = np.array([[20], [10], [1], [5], [300], [0], [5000]], dtype=np.float32)
    for mod in (amd_qwen, amd_glm):
      m = mod.top_k_mask(Tensor(z, device="CPU"), n, Tensor(ks, device="CPU")).numpy()
      for b in range(B): np.testing.assert_array_equal(m[b], _expected(z[b], n, int(ks[b, 0])), err_msg=f"{mod.__name__} row {b}")

  def test_mask_then_nucleus(self):
    # top_k first (as HF's warpers), then the top_p nucleus of what is left: the k most likely of a flat distribution, top_p 1: all k of them
    z = np.zeros((1, 64), np.float32)
    z[0, :8] = 2.0
    zt = Tensor(z, device="CPU")
    keep = amd_qwen.top_k_mask(zt, 64, Tensor([[8.0]], device="CPU"))
    zk = keep.where(zt, -math.inf)
    nuc = amd_qwen.nucleus(zk, 64, Tensor([[1.0]], device="CPU")).numpy()
    np.testing.assert_array_equal((keep.numpy() & nuc)[0], z[0] == 2.0)
    nuc = amd_qwen.nucleus(zk, 64, Tensor([[0.5]], device="CPU")).numpy() & keep.numpy()
    self.assertTrue(4 <= nuc.sum() <= 8 and (nuc[0] <= (z[0] == 2.0)).all())

  def test_request_param(self):
    self.assertIsNone(sampling_top_k({}))
    self.assertEqual(sampling_top_k({"top_k": 20}), 20)
    self.assertEqual(sampling_top_k({"top_k": -1}), 0)
    self.assertEqual(sampling_top_k({"top_k": 0}), 0)
    for bad in (-2, 1.5, True, "20"):
      with self.assertRaises(ValueError): sampling_top_k({"top_k": bad})
    self.assertEqual(Request([1], "", None, 0.6, 0.95).sampling, (0.6, 0.95))
    self.assertEqual(Request([1], "", None, 0.6, 0.95, top_k=20).sampling, (0.6, 0.95, 20.0))

  def test_rows(self):
    # the seq steps' rows: temperature, top_p, top_k (0: none); the batch steps' pairs drop a top_k
    for mod in (glm5next, qwen4exp):
      self.assertEqual(list(mod._temp_row(0.0)), [0.0, 1.0, 0.0])
      self.assertEqual(list(mod._temp_row((0.6, 0.95))), [0.6000000238418579, 0.949999988079071, 0.0])
      self.assertEqual(list(mod._temp_row((1.0, 1.0, 20.0))), [1.0, 1.0, 20.0])
    self.assertEqual(list(glm5next._temp_pair((1.0, 0.5, 20.0))), [1.0, 0.5])

if __name__ == "__main__":
  unittest.main()
