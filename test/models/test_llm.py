import unittest
import numpy as np
from tinygrad import Tensor, TinyJit, UOp, dtypes
from tinygrad.helpers import fetch
from tinygrad.llm.cli import models
from tinygrad.llm.model import Transformer

class TestLLMBlocks(unittest.TestCase):
  def test_llama3_2_1b_decode(self):
    # The default `python -m tinygrad.llm --benchmark 5`, restricted to its first two transformer blocks.
    model, kv = Transformer.from_gguf(fetch(models["llama3.2:1b"]), max_context=4096, realize=False)
    blocks = model.blk[:2]
    tokens = Tensor([[kv.get("tokenizer.ggml.bos_token_id", 0)]], dtype=dtypes.int32)
    x = model.token_embd(tokens).float().contiguous().realize()
    del model

    def run(x:Tensor, start_pos:UOp):
      for block in blocks: x = block(x, start_pos)
      return x.realize()
    prefill, decode = TinyJit(run), TinyJit(run)
    start_pos = UOp.variable("start_pos", 0, 4095)
    toks = UOp.variable("toks", 1, 32)
    # generate() starts with a symbolic-length prompt, then uses fixed one-token decode inputs.
    prompt = x.pad_to((1, 32, blocks[0].config.dim)).contiguous().realize()[:, :toks.bind(1)]
    for i in range(5):
      with self.subTest(step=i):
        out = prefill(prompt, start_pos.bind(i)) if i == 0 else decode(x, start_pos.bind(i))
        values = out[:, :1].numpy()
        self.assertEqual(values.shape, (1, 1, blocks[0].config.dim))
        self.assertTrue(np.isfinite(values).all())
    # Every step writes its own KV-cache slot; the remaining context must stay untouched.
    for block in blocks:
      cache = block.cache_kv.numpy()
      for i in range(5): self.assertTrue(np.any(cache[:, :, :, i, :] != 0))
      np.testing.assert_array_equal(cache[:, :, :, 5:, :], 0)

if __name__ == "__main__": unittest.main()
