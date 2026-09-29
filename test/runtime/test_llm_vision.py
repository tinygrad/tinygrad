import unittest
from collections import OrderedDict
from unittest.mock import Mock
import numpy as np
from tinygrad import Tensor
from tinygrad.llm.model import precompute_freqs_cis
from tinygrad.llm.vision import Qwen3VLTower, imrope_freqs_cis

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

class TestVisionCache(unittest.TestCase):
  def tower(self, budget):
    tower = Qwen3VLTower.__new__(Qwen3VLTower)
    tower.device, tower.patch_size, tower.merge_size = 'CPU', 1, 2
    tower.min_pixels, tower.max_pixels, tower.max_patches = 4, 16, 16
    tower.cache_tokens, tower._cache_tokens, tower._cache = budget, 0, OrderedDict()
    tower._buf_img = Tensor.zeros(48, dtype='uint8', device='CPU').realize()
    tower._buf_geom = Tensor.zeros(2, dtype='int32', device='CPU').realize()
    tower._buf_mask = Tensor.zeros(16, device='CPU').realize()
    output = Tensor.zeros(4, 8, dtype='float16', device='CPU').realize()
    # Simulate a JIT whose output buffer is overwritten by each image.
    tower._vit = Mock(side_effect=lambda img, *_: output.assign(img[0].half().reshape(1, 1).expand(4, 8)).realize())
    return tower

  def image(self, color, size=(2, 2)):
    from PIL import Image
    return Image.new('RGB', size, (color, color, color))

  def test_more_than_128_images(self):
    tower = self.tower(129)
    for _ in range(2):
      for i in range(129):
        embeds, gh, gw = tower.encode(self.image(i))
        self.assertEqual((embeds.shape, gh, gw), ((1, 8), 1, 1))
        np.testing.assert_array_equal(embeds.numpy(), i)
    self.assertEqual(tower._vit.call_count, 129)
    self.assertEqual(tower._cache_tokens, 129)

  def test_lru_token_budget(self):
    tower = self.tower(5)
    small, big, medium = self.image(1), self.image(2, (4, 4)), self.image(3, (2, 4))
    first = tower.encode(small)
    tower.encode(big)
    self.assertIs(tower.encode(small), first)  # touch small; big is now least recently used
    tower.encode(medium)
    self.assertEqual(tower._cache_tokens, 3)
    self.assertEqual(tower._vit.call_count, 3)
    self.assertIs(tower.encode(small), first)
    tower.encode(big)
    self.assertEqual(tower._cache_tokens, 5)  # medium was evicted, small still fits
    self.assertIs(tower.encode(small), first)
    self.assertEqual(tower._vit.call_count, 4)

  def test_oversized_image_not_cached(self):
    tower = self.tower(1)
    first = tower.encode(self.image(1))
    for _ in range(2): tower.encode(self.image(2, (4, 4)))
    self.assertIs(tower.encode(self.image(1)), first)
    self.assertEqual(tower._cache_tokens, 1)
    self.assertEqual(tower._vit.call_count, 3)

  def test_geometry_is_part_of_key(self):
    tower = self.tower(4)
    self.assertEqual(tower.encode(self.image(1, (2, 4)))[1:], (2, 1))
    self.assertEqual(tower.encode(self.image(1, (4, 2)))[1:], (1, 2))
    self.assertEqual(tower._vit.call_count, 2)

if __name__ == "__main__": unittest.main()
