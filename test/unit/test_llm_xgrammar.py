import unittest
from tinygrad import Tensor
from tinygrad.llm.model import Transformer, TransformerConfig

TEST_CONFIG = TransformerConfig(num_blocks=1, dim=64, hidden_dim=128, n_heads=2, n_kv_heads=2,
                           norm_eps=1e-5, vocab_size=16, head_dim=32, rope_theta=10000.0, rope_dim=32, v_head_dim=32, max_context=32)

class TestTransformerConstrainedGeneration(unittest.TestCase):
  def test_token_selector_can_override_next_token(self):
    model = Transformer(TEST_CONFIG)
    captured = []

    def fake_logits(tokens, start_pos):
      captured.append((tokens.shape, start_pos if isinstance(start_pos, int) else start_pos.val))
      logits = Tensor.full((1, TEST_CONFIG.vocab_size), -10.0)
      return logits.cat(Tensor.zeros(1,0), dim=1) if TEST_CONFIG.vocab_size == 0 else logits.scatter(1, Tensor([[5]]), Tensor([[10.0]]))

    model.get_next_logits = fake_logits

    chosen = []
    def selector(logits, tokens):
      chosen.append((tuple(tokens), tuple(int(x) for x in logits.argmax(axis=-1).numpy().tolist())))
      return 7

    gen = model.generate([1, 2, 3], token_selector=selector)
    self.assertEqual(next(gen), 7)
    self.assertEqual(chosen[0][0], (1, 2, 3))
    self.assertEqual(chosen[0][1], (5,))
    toks_shape = captured[0][0][-1]
    self.assertEqual(toks_shape.val if hasattr(toks_shape, 'val') else toks_shape, 3)

  def test_constrained_generation_keeps_cache_reuse(self):
    model = Transformer(TEST_CONFIG)
    seen = []
    def fake_logits(tokens, start_pos):
      seen.append(start_pos if isinstance(start_pos, int) else start_pos.val)
      logits = Tensor.zeros(1, TEST_CONFIG.vocab_size)
      return logits
    model.get_next_logits = fake_logits

    selector = lambda logits, tokens: 4
    gen = model.generate([1, 2, 3], token_selector=selector)
    next(gen)
    next(gen)
    seen.clear()
    gen = model.generate([1, 2, 3, 4, 4, 9], token_selector=selector)
    next(gen)
    self.assertEqual(seen[0], 4)

if __name__ == '__main__':
  unittest.main()
