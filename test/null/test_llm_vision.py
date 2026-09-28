import unittest
import numpy as np
from tinygrad import Tensor
from tinygrad.llm.vision import ImageEmbed, smart_resize, mrope_positions, expand_image_tokens, prepare_prompt

class TestSmartResize(unittest.TestCase):
  def test_aligns_and_keeps_ratio(self):
    w, h = smart_resize(320, 240, 32, 8192, 4194304)
    self.assertEqual((w, h), (320, 256))  # matches llama.cpp: 240 rounds to 256, not 224
    self.assertEqual((w % 32, h % 32), (0, 0))
  def test_max_pixels(self):
    w, h = smart_resize(4000, 3000, 32, 8192, 1_000_000)
    self.assertLessEqual(w * h, 1_000_000)
    self.assertAlmostEqual(w / h, 4000 / 3000, delta=0.1)
  def test_min_pixels(self):
    w, h = smart_resize(33, 30, 32, 8192, 4194304)
    self.assertGreaterEqual(w * h, 8192)

class TestMRopePositions(unittest.TestCase):
  def test_text_only(self):
    pos, cursor = mrope_positions(10, [])
    np.testing.assert_array_equal(np.array(pos)[:, 0], np.arange(10))
    self.assertEqual(cursor, 10)
  def test_image_block(self):
    # 3 text tokens, 2x3 image grid (6 tokens), 2 text tokens
    pos, cursor = mrope_positions(11, [ImageEmbed(3, Tensor.zeros(6, 8), 2, 3)])
    pos = np.array(pos)
    np.testing.assert_array_equal(pos[:3, 0], [0, 1, 2])
    # image tokens: t constant, h = row, w = col
    np.testing.assert_array_equal(pos[3:9, 0], [3] * 6)
    np.testing.assert_array_equal(pos[3:9, 1], [3, 3, 3, 4, 4, 4])
    np.testing.assert_array_equal(pos[3:9, 2], [3, 4, 5, 3, 4, 5])
    # text resumes at max(gh, gw) past the image start
    np.testing.assert_array_equal(pos[9:, 0], [6, 7])
    self.assertEqual(cursor, 8)

class TestExpandImageTokens(unittest.TestCase):
  def test_expand(self):
    class FakeTower:
      def __init__(self): self.calls = 0
      def encode(self, img):
        self.calls += 1
        return Tensor.zeros(6, 8), 2, 3
    tower = FakeTower()
    ids, embeds = expand_image_tokens([1, 2, 99, 3], ["img"], tower, 99)
    self.assertEqual(ids, [1, 2] + [99] * 6 + [3])
    self.assertEqual(len(embeds), 1)
    self.assertEqual((embeds[0].start, embeds[0].grid_h, embeds[0].grid_w), (2, 2, 3))
    self.assertEqual(tower.calls, 1)
  def test_no_tower_raises(self):
    with self.assertRaises(RuntimeError): prepare_prompt([1, 99, 2], ["img"], None, 99)
    with self.assertRaises(RuntimeError): prepare_prompt([1, 99, 2], ["img"], object(), None)
    with self.assertRaises(RuntimeError): prepare_prompt([1, 99, 2], ["a", "b"], object(), 99)

if __name__ == "__main__": unittest.main()
