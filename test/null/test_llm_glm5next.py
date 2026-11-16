import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from tinygrad import Tensor, Context
from tinygrad.llm.glm5next import Transformer


class TestRecurrentCheckpointReuse(unittest.TestCase):
  def setUp(self): self.enterContext(Context(DEV="CPU"))

  def model(self) -> Transformer:
    model = Transformer.__new__(Transformer)
    model._cached_tokens = [1, 2, 3, 99, 98]
    model._ckpt_tokens = [1, 2, 3]
    model.has_recurrent_block = True
    model._checkpoint = Mock()
    return model

  def test_reuse_prompt_checkpoint(self):
    model, tokens = self.model(), [1, 2, 3]
    self.assertEqual(model._reuse(tokens), (3, True))
    model._checkpoint(tokens, save=False)
    model._checkpoint.assert_called_once_with(tokens, save=False)

  def test_generate_samples_exact_checkpoint(self):
    model, tokens = self.model(), [1, 2, 3]
    model.max_context = 8
    model.config = SimpleNamespace(spec_drafts=0, ple=None)
    model.token_embd = SimpleNamespace(weight=SimpleNamespace(device="CPU"))
    model._finish_load = Mock()
    model.prefill_chunks = Mock(return_value=(4,))
    model._spec_h = Tensor.zeros((1, 1))
    model._sample = Mock(return_value=Tensor([[42]]))
    checkpoints = []
    model._checkpoint = Mock(side_effect=lambda toks, save: checkpoints.append((list(toks), save)))
    with patch.object(Transformer, "__call__", side_effect=AssertionError("exact checkpoint was prefilled again")):
      self.assertEqual(next(model.generate(list(tokens))), 42)
    self.assertEqual(checkpoints, [(tokens, False), (tokens, True)])
    model._sample.assert_called_once()
    self.assertEqual(model._cached_tokens, tokens)

  def test_prompt_after_checkpoint(self):
    model = self.model()
    self.assertEqual(model._reuse([1, 2, 3, 4]), (3, True))

  def test_divergent_prompt_restarts(self):
    model = self.model()
    self.assertEqual(model._reuse([1, 2, 4]), (0, False))


class TestGLMQ8GatherShapes(unittest.TestCase):
  def test_pack_needs_whole_warps_of_groups(self):
    # MLA's 576 dims are 18 groups of 32. Eight groups are packed per warp, so the last two groups would be omitted.
    from tinygrad.llm.kernels.amd_glm import _q8_pack_kernel, _q8_unpack_kernel
    self.assertRaises(AssertionError, _q8_pack_kernel, Tensor.empty(4, 18, 9, dtype="uint32").uop, Tensor.empty(4, 576, dtype="half").uop)
    self.assertRaises(AssertionError, _q8_unpack_kernel, Tensor.empty(4, 576, dtype="half").uop, Tensor.empty(4, 18, 9, dtype="uint32").uop)


if __name__ == '__main__':
  unittest.main()
