import unittest
import numpy as np
from tinygrad import Tensor
from tinygrad.llm.model import TransformerConfig, SSMConfig, IndexerConfig, GatedDeltaNetBlock, DSAMLATransformerBlock

def _config(**kwargs):
  return TransformerConfig(**{"num_blocks": 1, "dim": 16, "hidden_dim": 16, "n_heads": 2, "n_kv_heads": 1, "norm_eps": 1e-5, "vocab_size": 10,
    "head_dim": 8, "rope_theta": 1e4, "rope_dim": 0, "v_head_dim": 8, "kv_lora_rank": 8, "q_lora_rank": 8, "max_context": 16} | kwargs)

class TestGLM5(unittest.TestCase):
  def test_kda_qk_l2norm(self):
    # GLM normalizes q/k as x/sqrt(sum(x²)+1e-6): tiny vectors aren't scaled up to unit length
    ssm = SSMConfig(conv_kernel=4, state_size=8, group_count=2, time_step_rank=2, inner_size=16, kda=True, qk_l2norm_eps=1e-6)
    block = GatedDeltaNetBlock(_config(ssm=ssm), ssm)
    block.attn_qkv.weight = Tensor([1e-4]*32 + [1.0]*16).reshape(48, 1).expand(48, 16) / 16  # x of ones: q = k = 1e-4, v = 1
    block.ssm_conv1d["weight"], block.ssm_beta.weight = Tensor.ones(48, 4), Tensor.zeros(2, 16)
    x = Tensor.ones(1, 1, 16)
    block._init_state(x)
    block._attention(x, 0).realize()
    # the first token's state is beta * v (x) l2norm(k), with beta = sigmoid(0) and v, k through the conv and silu
    def silu(z): return z / (1 + np.exp(-z))
    np.testing.assert_allclose(block.recurrent_state.numpy(), 0.5 * silu(1.0) * silu(1e-4) / np.sqrt(8 * silu(1e-4)**2 + 1e-6), rtol=1e-3)

  def test_dsa_mask(self):
    # each query sees its best complete pool (top_k = kpool) and the tokens of its own pool while that's incomplete
    kpool, max_context = 4, 30  # (not a multiple of kpool)
    indexer = IndexerConfig(n_heads=2, head_dim=8, top_k=kpool, kpool=kpool, norm_eps=1e-6)
    block = DSAMLATransformerBlock(_config(max_context=max_context, indexer=indexer))
    block.indexer_compressor_ape["weight"] = Tensor.randn(kpool, 8)
    x = Tensor.randn(1, max_context, 16).realize()
    block._init_state(x)
    ik = block.indexer["k_norm"](block.indexer["attn_k"](x)).numpy()[0]
    iq = block.indexer["attn_q_b"](block.attn_q_a_norm(block.attn_q_a(x))).numpy()[0].reshape(max_context, 2, 8)
    head_w = block.indexer["proj"](x).numpy()[0] / 4  # (n_heads * head_dim) ** -0.5
    # a complete pool's key: its keys weighted by a softmax over its tokens of the gate (+ the positional ape)
    pools = max_context // kpool
    gate = np.exp(block.indexer_compressor_gate(x).numpy()[0, :pools*kpool].reshape(pools, kpool, 8) + block.indexer_compressor_ape["weight"].numpy())
    pool_k = (gate / gate.sum(1, keepdims=True) * ik[:pools*kpool].reshape(pools, kpool, 8)).sum(1)
    for start_pos, T in ((0, 9), (9, 21)):  # chunks, in order: the indexer caches every token
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
