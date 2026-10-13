import unittest
import numpy as np
from tinygrad import Tensor
from tinygrad.llm.model import TransformerConfig, IndexerConfig, MLATransformerBlock

def _config(**kwargs):
  return TransformerConfig(**{"num_blocks": 1, "dim": 16, "hidden_dim": 16, "n_heads": 2, "n_kv_heads": 1, "norm_eps": 1e-5, "vocab_size": 10,
    "head_dim": 8, "rope_theta": 1e4, "rope_dim": 0, "v_head_dim": 8, "kv_lora_rank": 8, "q_lora_rank": 8, "max_context": 16} | kwargs)

class TestGLM5(unittest.TestCase):
  def test_dsa_mask(self):
    # each query sees its best complete pool (top_k = kpool) and the tokens of its own pool while that's incomplete
    kpool, max_context = 4, 14  # (not a multiple of kpool)
    indexer = IndexerConfig(n_heads=2, head_dim=8, top_k=kpool, kpool=kpool, norm_eps=1e-6)
    block = MLATransformerBlock(_config(max_context=max_context, indexer=indexer))
    block.indexer.compressor_ape["weight"] = Tensor.randn(kpool, 8)
    x = Tensor.randn(1, max_context, 16).realize()
    block._init_state(x)
    ik = block.indexer.k_norm(block.indexer.attn_k(x)).numpy()[0]
    iq = block.indexer.attn_q_b(block.attn_q_a_norm(block.attn_q_a(x))).numpy()[0].reshape(max_context, 2, 8)
    head_w = block.indexer.proj(x).numpy()[0] / 4  # (n_heads * head_dim) ** -0.5
    # a complete pool's key: its keys weighted by a softmax over its tokens of the gate (+ the positional ape)
    pools = max_context // kpool
    gate = np.exp(block.indexer.compressor_gate(x).numpy()[0, :pools*kpool].reshape(pools, kpool, 8) + block.indexer.compressor_ape["weight"].numpy())
    pool_k = (gate / gate.sum(1, keepdims=True) * ik[:pools*kpool].reshape(pools, kpool, 8)).sum(1)
    for start_pos, T in ((0, 7), (7, 1), (8, 1), (9, 5)):  # decode, pool boundaries, then prefill chunks
      mask = block._attn_mask(x[:, start_pos:start_pos+T], start_pos).numpy()[0, 0] == 0
      for t, p in enumerate(range(start_pos, start_pos+T)):
        expected, complete = np.zeros(start_pos+T, bool), (p + 1) // kpool
        if complete:
          best = (np.maximum(iq[p] @ pool_k[:complete].T, 0) * head_w[p][:, None]).sum(0).argmax()
          expected[best*kpool:(best+1)*kpool] = True
        expected[complete*kpool:p+1] = True
        np.testing.assert_equal(mask[t], expected, err_msg=f"token {p}")

if __name__ == '__main__':
  unittest.main()
