import unittest
from types import SimpleNamespace
from tinygrad.llm import glm5next

class _Fake:
  """the parts of the GLM_SEQ driver the draft choice of a step reads"""
  SEQ_SIZES = (1, 2, 4, 8, 16)
  drafts_of, _step_drafts = glm5next.Transformer.drafts_of, glm5next.Transformer._step_drafts
  def __init__(self, base:int, by:str):
    by_size = glm5next.spec_by_size(base, self.SEQ_SIZES, by)
    self.config = SimpleNamespace(spec_drafts=max([base, *(d for _, d in by_size)]), spec_by_size=by_size)

class TestGLMSpecBySize(unittest.TestCase):
  """SPEC_DRAFTS_BY_SIZE: the drafts of a step of m sequences, the layouts of the states the most of them"""
  def test_unset(self):
    self.assertEqual(glm5next.spec_by_size(2, (1, 4), ""), ())
    f = _Fake(2, "")
    self.assertEqual([f.drafts_of(m) for m in f.SEQ_SIZES], [2]*5)
    self.assertEqual(f._step_drafts(4), [2])

  def test_by_size(self):
    f = _Fake(2, "1:2,4:3,8:3,32:4")
    # (sizes not listed: SPEC_DRAFTS; listed sizes not in SEQ_SIZES ignored)
    self.assertEqual(f.config.spec_by_size, ((1, 2), (2, 2), (4, 3), (8, 3), (16, 2)))
    self.assertEqual(f.config.spec_drafts, 3)
    self.assertEqual([f.drafts_of(m) for m in f.SEQ_SIZES], [2, 2, 3, 3, 2])
    # (a sequence from a step of 3 drafts into a size of 2: the step of 3 warmed too)
    self.assertEqual(f._step_drafts(1), [2, 3])
    self.assertEqual(f._step_drafts(4), [3])

  def test_bad(self):
    with self.assertRaises(AssertionError): glm5next.spec_by_size(2, (1,), "1:0")

if __name__ == "__main__":
  unittest.main()
