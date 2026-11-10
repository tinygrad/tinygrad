import unittest
from unittest.mock import patch
from tinygrad.llm import qwen4exp

class TestMoeTile16(unittest.TestCase):
  def test_padding_and_cap(self):
    with patch.object(qwen4exp, "MOE_TILE16", 40):
      # C4's k=10, E=512: 576 rows favor 16, 1152 ties at 32 padded rows/expert and favors 32, 2048 favors 16 again.
      for tokens, expected in ((192, True), (576, True), (1152, False), (2048, True), (2304, False), (4096, False)):
        self.assertEqual(qwen4exp._moe_use_tile16(tokens * 10, 512), expected, tokens)
      self.assertTrue(qwen4exp._moe_use_tile16(16 * 512, 512))
      self.assertFalse(qwen4exp._moe_use_tile16(16 * 512 + 1, 512))
      self.assertFalse(qwen4exp._moe_use_tile16(32 * 512, 512))
      self.assertTrue(qwen4exp._moe_use_tile16(32 * 512 + 1, 512))
      self.assertFalse(qwen4exp._moe_use_tile16(40 * 512 + 1, 512))
    with patch.object(qwen4exp, "MOE_TILE16", 0):
      self.assertFalse(qwen4exp._moe_use_tile16(576 * 10, 512))

if __name__ == "__main__":
  unittest.main()
