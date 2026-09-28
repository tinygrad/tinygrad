import unittest
import numpy as np
from tinygrad import Tensor
from tinygrad.llm.model import precompute_freqs_cis
from tinygrad.llm.vision import imrope_freqs_cis

class TestImropeFreqs(unittest.TestCase):
  def test_text_positions_match_plain_rope(self):
    # when t == h == w, interleaved mrope is the same as plain rope
    seq = np.stack([np.arange(7)] * 3, axis=1).astype(np.int32)
    got = imrope_freqs_cis(Tensor(seq), 64, 10000000.0, (11, 11, 10, 0))
    want = precompute_freqs_cis(64, 7, 10000000.0)
    np.testing.assert_allclose(got.numpy(), want.numpy(), atol=1e-5)
  def test_sections_pick_channels(self):
    # sections (1, 1, 1, 0): pair 0 -> t, pair 1 -> h, pair 2 -> w, pair 3 -> t, ...
    pos = Tensor([[[10], [100], [1000]]]).reshape(1, 3)  # t=10, h=100, w=1000
    freqs = imrope_freqs_cis(pos, 8, 10000.0, (1, 1, 1, 0)).numpy()[0]
    n = 4
    base = 10000.0 ** (-np.arange(n) / n)
    cos = freqs[:n]
    np.testing.assert_allclose(cos, np.cos(base * [10, 100, 1000, 10]), atol=1e-5)

if __name__ == "__main__": unittest.main()
