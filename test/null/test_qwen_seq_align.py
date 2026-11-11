import os
import subprocess
import sys
import unittest

from tinygrad.llm import qwen4exp
from tinygrad.llm.qwen4exp import Transformer


class TestQwenSeqAlign(unittest.TestCase):
  def test_region_base_alignment_defaults_to_move_alignment(self):
    self.assertEqual(qwen4exp.SEQ_BASE_ALIGN, qwen4exp.REGION_ALIGN)
    self.assertEqual(Transformer.SEQ_BASE_ALIGN, qwen4exp.REGION_ALIGN)
    self.assertEqual(Transformer.SEQ_MOVE, qwen4exp.REGION_ALIGN)

  def test_explicit_alignment_does_not_change_move_or_stash_alignment(self):
    env = os.environ.copy()
    env["SEQ_BASE_ALIGN"] = "24"
    subprocess.run([sys.executable, "-c", """
from tinygrad.llm import qwen4exp
from tinygrad.llm.qwen4exp import Transformer
assert qwen4exp.SEQ_BASE_ALIGN == 24
assert Transformer.SEQ_BASE_ALIGN == 24
assert qwen4exp.REGION_ALIGN == Transformer.SEQ_MOVE == 3072
assert Transformer.STASH_CALL == (16, 4, 1)
"""], check=True, env=env)


if __name__ == "__main__": unittest.main()
