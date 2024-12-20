from typing import Literal
import unittest
import extra.bpetokenizer as bpetokenizer
try:
  import tiktoken
except ModuleNotFoundError:
  raise unittest.SkipTest("tiktoken not installed, skipping test")

class TestBPETokenizerGPT2(unittest.TestCase):
  def setUp(self) -> None:
    self.tt = tiktoken.get_encoding("gpt2")
    self.bpe = bpetokenizer.get_encoding("gpt2")

  def _test_encode_decode(self, text: str, allowed_special: set[str] | Literal["all"] = set()):
    self.assertDictEqual(self.tt._special_tokens, self.bpe._special_tokens)

    tt_tokens = self.tt.encode(text, allowed_special=allowed_special, disallowed_special=set())
    bpe_tokens = self.bpe.encode(text, allowed_special=allowed_special, disallowed_special=set())

    self.assertListEqual(tt_tokens, bpe_tokens)

    tt_text = self.tt.decode(tt_tokens)
    bpe_text = self.bpe.decode(bpe_tokens)
    self.assertEqual(tt_text, bpe_text)

  def test_basic(self): self._test_encode_decode("Phrase to start with")
  def test_special_suppress(self): self._test_encode_decode("Hello<|endoftext|> World! ")
  def test_special_allow(self): self._test_encode_decode("Hello<|endoftext|> World! ", allowed_special={ "<|endoftext|>" })

  def test_disallow_special(self):
    with self.assertRaises(ValueError): self.tt.encode("<|endoftext|>")
    with self.assertRaises(ValueError): self.bpe.encode("<|endoftext|>")

if __name__ == '__main__':
  unittest.main()