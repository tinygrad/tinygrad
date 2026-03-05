import unittest
from unittest.mock import patch
from tinygrad import Tensor, UOp
from tinygrad.engine.schedule import schedule_cache

class TestTransformerGenerate(unittest.TestCase):
  def test_start_pos_parameter_is_used(self):
    """Test that start_pos parameter is not ignored (regression test for always resetting to 0)."""
    from tinygrad.apps.llm import Transformer
    # Create a minimal transformer
    model = Transformer(num_blocks=1, dim=64, hidden_dim=128, n_heads=2, n_kv_heads=2,
                        norm_eps=1e-5, vocab_size=100, head_dim=32, rope_theta=10000.0, max_context=32)

    captured_inputs = []
    def mock_call(self, tokens, start_pos):
      captured_inputs.append((tokens.shape, start_pos if isinstance(start_pos, int) else start_pos.val))
      return Tensor([[42]])  # return a fake next token

    with patch.object(Transformer, '__call__', mock_call):
      tokens = [1, 2, 3, 4, 5]
      gen = model.generate(tokens, start_pos=3)
      next(gen)  # get first token

    # With start_pos=3, the initial tensor should only have tokens[3:] = [4, 5] (length 2)
    # If the bug existed (start_pos always reset to 0), it would have all 5 tokens
    toks_shape = captured_inputs[0][0][-1]
    self.assertEqual(toks_shape.val if isinstance(toks_shape, UOp) else toks_shape, 2)  # toks bound to 2
    self.assertEqual(captured_inputs[0][1], 3)  # start_pos should be 3, not 0

  def test_two_prompts_schedule_cache(self):
    """Second prompt prefill should hit the schedule cache, not miss."""
    from tinygrad.apps.llm import Transformer
    model = Transformer(num_blocks=1, dim=64, hidden_dim=128, n_heads=2, n_kv_heads=2,
                        norm_eps=1e-5, vocab_size=100, head_dim=32, rope_theta=10000.0, max_context=64)

    # first prompt: prefill + a few decode steps
    ids = list(range(1, 6))
    gen = model.generate(ids, start_pos=0)
    for _ in range(3): next(gen)
    cache_size_after_first = len(schedule_cache)

    # second prompt: simulates multi-turn chat
    start_pos = len(ids) - 1
    ids += list(range(10, 15))
    gen = model.generate(ids, start_pos)
    for _ in range(3): next(gen)

    # the second prompt should reuse the same schedule cache entries, not create new ones
    self.assertEqual(cache_size_after_first, len(schedule_cache),
      f"second prompt added {len(schedule_cache) - cache_size_after_first} new schedule cache entries (expected 0)")

if __name__ == '__main__':
  unittest.main()
