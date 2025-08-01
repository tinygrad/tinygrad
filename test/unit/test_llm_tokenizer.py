import unittest
from tinygrad.apps.llm import SimpleTokenizer

class TestLLMTokenizer(unittest.TestCase):
  def _test_basic(self, text: str, expected_tokens: list[int]):
    tok = SimpleTokenizer(".*", { b"a": 0, b"b": 1, b"c": 2, b"ab": 3, b"bc": 4 }, { "<x>": 5, "<y>": 6, "<z>": 7 })
    self.assertEqual(tok.encode(text), expected_tokens)
    self.assertEqual(tok.decode(expected_tokens), text)

  def test_abc(self): self._test_basic("abc", [ 3, 2 ])
  def test_abbc(self): self._test_basic("abbc", [ 3, 4 ])
  def test_aabbbcc(self): self._test_basic("aabbbcc", [ 0, 3, 1, 4, 2 ])

  def test_specials1(self): self._test_basic("a<x>a<y>a<z>a", [ 0, 5, 0, 6, 0, 7, 0 ])
  def test_specials2(self): self._test_basic("<x>a<y>a<z>", [ 5, 0, 6, 0, 7 ])

  def test_invalid_token(self):
    with self.assertRaises(RuntimeError): self._test_basic("L", [])

if __name__ == '__main__':
  unittest.main()
