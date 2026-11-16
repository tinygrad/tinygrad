import inspect
import os
import unittest
from unittest.mock import patch

from tinygrad.helpers import getenv
from tinygrad.llm.kernels.amd_glm import _q6_split_target, q8_linear


class TestGLMQ6Default(unittest.TestCase):
  def setUp(self):
    getenv.cache_clear()

  def tearDown(self):
    getenv.cache_clear()
  def test_all_q6_gates_share_row_gated_target(self):
    source = inspect.getsource(q8_linear)
    self.assertIn("q6_split_target = _q6_split_target(tokens, _wmma_rdna4(x.device))", source)
    self.assertIn("tgt:=q6_split_target", source)
    self.assertNotIn('getenv("Q6_KSPLIT", 0)', source)

  def test_default_rdna4_crossover_and_rdna3_compatibility(self):
    with patch.dict(os.environ, {"Q6_KSPLIT": "192", "Q6_KSPLIT_MIN": "24"}, clear=True):
      self.assertEqual([_q6_split_target(rows, True) for rows in (3, 6, 9, 12, 18, 24, 48, 96)],
                       [0, 0, 0, 0, 0, 192, 192, 192])
      self.assertEqual([_q6_split_target(rows, False) for rows in (3, 24, 96)], [192, 192, 192])

if __name__ == "__main__":
  unittest.main()
