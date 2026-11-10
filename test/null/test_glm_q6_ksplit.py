import unittest

from tinygrad.llm.kernels.amd_glm import _q6_input_splits, _q6_rdna4_pad_rows, _q6_rdna4_padded_rows, _q6_rdna4_token_tile


class TestGLMQ6InputSplits(unittest.TestCase):
  def test_rdna4_requested_shapes(self):
    # C6 GLM_SEQ G=32 has 96 rows (three rows per sequence), token tile 96.
    self.assertEqual(_q6_input_splits(4096, 1536, 96, 96, 192, True), 16)
    self.assertEqual(_q6_input_splits(1536, 4096, 96, 96, 192, True), 2)
    self.assertEqual(_q6_input_splits(4096, 384, 96, 96, 192, True), 16)

  def test_rdna4_tile_128_and_ceiling_output_tiles(self):
    self.assertEqual(_q6_input_splits(4096, 1536, 128, 128, 192, True), 16)
    self.assertEqual(_q6_input_splits(4096, 129, 128, 128, 192, True), 16)

  def test_rdna3_retains_64_wide_output_tile_count(self):
    # Three 32-row tiles x 24 64-wide output tiles = 72 workgroups before splitting.
    self.assertEqual(_q6_input_splits(4096, 1536, 96, 32, 192, False), 4)

  def test_rdna3_matches_prior_inline_selector(self):
    # Rebase guard: the shared helper must preserve the merged RDNA3 selector, not only its common 4096x1536 case.
    for inc in (256, 512, 1536, 4096, 8192):
      for out in (64, 128, 384, 512, 1536, 4096):
        for rows in (32, 64, 96, 128):
          for target in (1, 32, 192, 384):
            with self.subTest(inc=inc, out=out, rows=rows, target=target):
              wgs = (rows // 32) * -(-out // 64)
              expected = next((s for s in (1, 2, 4, 8, 16) if (inc // 32) % (8*s) == 0 and wgs*s >= target),
                              next(s for s in (16, 8, 4, 2, 1) if (inc // 32) % (8*s) == 0))
              self.assertEqual(_q6_input_splits(inc, out, rows, 32, target, False), expected)

  def test_rdna4_padding_and_tiles_preserve_default_and_enable_small_rows(self):
    self.assertEqual([_q6_rdna4_padded_rows(r, True) for r in (24, 48, 72, 96)], [32, 64, 96, 96])
    # Recursive q8 padding takes the target, not a delta; aligned inputs must not pad again.
    self.assertEqual([_q6_rdna4_pad_rows(r, True) for r in (24, 32, 48, 64, 72, 96, 128)], [32, 0, 64, 0, 96, 0, 0])
    for rows in (24, 48, 72, 96):
      self.assertEqual(_q6_rdna4_padded_rows(rows, False), rows)
    self.assertEqual([_q6_rdna4_token_tile(r, True) for r in (32, 64, 96)], [32, 64, 96])
    self.assertEqual(_q6_rdna4_token_tile(48, True), 0)  # padded to 64 before selection
    for rows in (32, 48):
      self.assertEqual(_q6_rdna4_token_tile(rows, False), 0)
    self.assertEqual(_q6_rdna4_token_tile(96, False), 96)
    self.assertEqual(_q6_rdna4_token_tile(72, True), 0)  # padded to 96 before selection

  def test_every_choice_has_whole_q6_blocks_and_power_of_two_splits(self):
    for rdna4 in (False, True):
      for in_features, local_out in ((4096, 1536), (1536, 4096), (4096, 384)):
        splits = _q6_input_splits(in_features, local_out, 96, 96 if rdna4 else 32, 192, rdna4)
        self.assertIn(splits, (1, 2, 4, 8, 16))
        self.assertEqual((in_features // (32*splits)) % 8, 0)


if __name__ == "__main__":
  unittest.main()
