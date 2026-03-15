import unittest
from unittest.mock import patch
from tinygrad import Tensor, UOp
from tinygrad.engine.schedule import schedule_cache

class TestTransformerGenerate(unittest.TestCase):
  def test_kv_cache_reuse(self):
    """Test that generate reuses the KV cache when tokens extend the cached prefix."""
    from tinygrad.apps.llm import Transformer
    model = Transformer(num_blocks=1, dim=64, hidden_dim=128, n_heads=2, n_kv_heads=2,
                        norm_eps=1e-5, vocab_size=100, head_dim=32, rope_theta=10000.0, max_context=32)

    captured_inputs = []
    def mock_call(self, tokens, start_pos):
      captured_inputs.append((tokens.shape, start_pos if isinstance(start_pos, int) else start_pos.val))
      return Tensor([[42]])

    with patch.object(Transformer, '__call__', mock_call):
      # first conversation: prefill 5 tokens + 1 decode
      tokens = [1, 2, 3, 4, 5]
      gen = model.generate(tokens)
      next(gen)  # prefill
      next(gen)  # decode

      # second call extends the conversation — cached prefix should be reused
      captured_inputs.clear()
      tokens = [1, 2, 3, 4, 5, 42, 42, 10, 11, 12]
      gen = model.generate(tokens)
      next(gen)

    # should only process tokens[7:] = [10, 11, 12] since first 7 are cached
    toks_shape = captured_inputs[0][0][-1]
    self.assertEqual(toks_shape.val if isinstance(toks_shape, UOp) else toks_shape, 3)
    self.assertEqual(captured_inputs[0][1], 7)

  def test_kv_cache_invalidation(self):
    """Test that generate invalidates the KV cache when tokens diverge from the cached prefix."""
    from tinygrad.apps.llm import Transformer
    model = Transformer(num_blocks=1, dim=64, hidden_dim=128, n_heads=2, n_kv_heads=2,
                        norm_eps=1e-5, vocab_size=100, head_dim=32, rope_theta=10000.0, max_context=32)

    captured_inputs = []
    def mock_call(self, tokens, start_pos):
      captured_inputs.append((tokens.shape, start_pos if isinstance(start_pos, int) else start_pos.val))
      return Tensor([[42]])

    with patch.object(Transformer, '__call__', mock_call):
      # first conversation
      gen = model.generate([1, 2, 3, 4, 5])
      next(gen)

      # completely different prompt — KV cache should be invalidated
      captured_inputs.clear()
      gen = model.generate([10, 20, 30])
      next(gen)

    # should process all 3 tokens from start
    toks_shape = captured_inputs[0][0][-1]
    self.assertEqual(toks_shape.val if isinstance(toks_shape, UOp) else toks_shape, 3)
    self.assertEqual(captured_inputs[0][1], 0)

  def test_two_prompts_schedule_cache(self):
    """Third prompt should hit the schedule cache, not miss (first two warm up both jits: prefill + decode)."""
    from tinygrad.apps.llm import Transformer
    model = Transformer(num_blocks=1, dim=64, hidden_dim=128, n_heads=2, n_kv_heads=2,
                        norm_eps=1e-5, vocab_size=100, head_dim=32, rope_theta=10000.0, max_context=64)

    # first two prompts warm up both jits (prefill + decode)
    ids = list(range(1, 6))
    gen = model.generate(ids)
    for _ in range(3): next(gen)

    ids += list(range(10, 15))
    gen = model.generate(ids)
    for _ in range(3): next(gen)
    cache_size_after_warmup = len(schedule_cache)

    # third prompt should reuse the same schedule cache entries, not create new ones
    ids += list(range(20, 25))
    gen = model.generate(ids)
    for _ in range(3): next(gen)

    self.assertEqual(cache_size_after_warmup, len(schedule_cache),
      f"third prompt added {len(schedule_cache) - cache_size_after_warmup} new schedule cache entries (expected 0)")

  def test_chunked_prefill(self):
    """When prompt > chunk_size, all chunks should be prefill"""
    from tinygrad.apps.llm import Transformer
    from tinygrad.uop.ops import resolve
    model = Transformer(num_blocks=1, dim=64, hidden_dim=128, n_heads=2, n_kv_heads=2,
                        norm_eps=1e-5, vocab_size=100, head_dim=32, rope_theta=10000.0, max_context=64)

    def get_prefill_flags(tokens, chunk_size):
      is_prefill = []
      def mock_call(self, tokens, start_pos):
        is_prefill.append(resolve(tokens.shape[1] != 1))
        return Tensor([[42]])
      with patch.object(Transformer, '__call__', mock_call):
        gen = model.generate(tokens, chunk_size=chunk_size)
        for _ in range(3): next(gen)
      model._cached_tokens = []
      return is_prefill

    # 8 tokens, chunk_size=4 -> 2 prefill chunks
    self.assertEqual(get_prefill_flags(list(range(8)), 4), [True, True, False, False])
    # 9 tokens, chunk_size=4 -> 3 prefill chunks (4+4+1)
    self.assertEqual(get_prefill_flags(list(range(9)), 4), [True, True, True, False, False])
    # 4 tokens, chunk_size=4 -> 1 prefill chunk
    self.assertEqual(get_prefill_flags(list(range(4)), 4), [True, False, False])

if __name__ == '__main__':
  unittest.main()
