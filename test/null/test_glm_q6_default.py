import inspect
import unittest

from tinygrad.llm.kernels.amd_glm import q8_linear


class TestGLMQ6Default(unittest.TestCase):
  def test_all_q6_gates_share_default_on_target(self):
    source = inspect.getsource(q8_linear)
    self.assertIn('q6_split_target = getenv("Q6_KSPLIT", 192)', source)
    self.assertIn("tgt:=q6_split_target", source)
    self.assertNotIn('getenv("Q6_KSPLIT", 0)', source)


if __name__ == "__main__":
  unittest.main()
