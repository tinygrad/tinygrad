import unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Tensor
from tinygrad.llm.model import Transformer, TransformerConfig, HCConfig, IndexerConfig, QwenMTPBlock

class TestMTPCache(unittest.TestCase):
  @staticmethod
  def config():
    return TransformerConfig(num_blocks=1, dim=64, hidden_dim=128, n_heads=2, n_kv_heads=2, norm_eps=1e-6,
                            vocab_size=128, head_dim=32, rope_theta=10000, rope_dim=32, v_head_dim=32, max_context=64,
                            qk_norm=32, attn_output_gate=True, hc=HCConfig(4, 1, 1e-6, low_rank=16),
                            indexer=IndexerConfig(2, 32, 16, kpool=4, rope_dim=32), qsa=True)

  def test_cache_only_matches_full_prediction(self):
    Tensor.manual_seed(37)
    block = QwenMTPBlock(self.config())
    block.hnorm["weight"] = Tensor.ones(256)
    for hc in (block.hc_attn, block.hc_ffn, block.hc_head): hc.norm["weight"] = Tensor.ones(256)
    emb, hidden = Tensor.randn(1, 12, 64).realize(), Tensor.randn(1, 12, 256).realize()
    block._init_state(hidden.reshape(1, 12, 4, 64))
    caches = [block.cache_k, block.indexer.cache_k, block.indexer.cache_pk]
    reference = []
    for cache_only in (False, True):
      Tensor.realize(*[c.assign(Tensor.zeros_like(c)) for c in caches])
      for i, (start, count) in enumerate(((0, 7), (7, 5))):
        e, h = emb[:, start:start+count], hidden[:, start:start+count]
        if cache_only:
          with patch.object(block, "attn_q", side_effect=AssertionError("cache priming must not compute Q")), \
               patch.object(block, "_feed_forward", side_effect=AssertionError("cache priming must not run the FFN")):
            block.prefill_cache(e, h, start, count)[:, :, :1, :1].contiguous().realize()
        else:
          full = block.predict(e, h, start, count).realize()
          selected = block.predict_selected(e, h, start, count-1).realize()
          np.testing.assert_allclose(selected.numpy(), full[:, -1:].numpy(), rtol=1e-6, atol=1e-6)
        actual = [c.numpy().copy() for c in caches]
        if not cache_only: reference.append(actual)
        else:
          for a, b in zip(actual, reference[i]): np.testing.assert_array_equal(a, b)
    self.assertTrue(np.any(reference[-1][0] != 0))
    self.assertTrue(np.any(reference[-1][1] != 0))

  def test_final_mix_only_last_token(self):
    Tensor.manual_seed(38)
    model = Transformer(self.config())
    for hc in (model.blk[0].hc_attn, model.blk[0].hc_ffn, model.output_hc): hc.norm["weight"] = Tensor.ones(256)
    tokens = Tensor([[1, 2, 3]])
    x = model.token_embd(tokens).float().unsqueeze(2).expand(1, 3, 4, 64).contiguous()
    for block in model.blk: x = block(x, 0)
    logits = model.output(model._hc_out(x)[:, -1, :]).numpy()
    with patch.object(model, "_hc_out", wraps=model._hc_out) as mix:
      actual = model.forward(tokens, 0, None).numpy()
      self.assertEqual(mix.call_count, 1)
      self.assertEqual(mix.call_args.args[0].shape[1], 1)
    np.testing.assert_array_equal(actual.reshape(-1), logits.argmax(-1))

if __name__ == "__main__": unittest.main()
