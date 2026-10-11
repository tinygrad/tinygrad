import unittest
import numpy as np
from tinygrad import Tensor, nn
from tinygrad.llm.model import TransformerConfig, SSMConfig, IndexerConfig, GatedDeltaNetBlock, DSAMLATransformerBlock

def _config(**kwargs) -> TransformerConfig:
  return TransformerConfig(**{"num_blocks": 1, "dim": 16, "hidden_dim": 16, "n_heads": 2, "n_kv_heads": 1, "norm_eps": 1e-5, "vocab_size": 10,
    "head_dim": 8, "rope_theta": 1e4, "rope_dim": 0, "v_head_dim": 8, "max_context": 32, "kv_lora_rank": 8, "q_lora_rank": 8} | kwargs)

def _randomize(block, seed:int) -> dict[str, np.ndarray]:
  rng = np.random.default_rng(seed)
  for v in nn.state.get_state_dict(block).values(): v.replace(Tensor(rng.standard_normal(v.shape).astype(np.float32)).realize())
  return {k: v.numpy() for k, v in nn.state.get_state_dict(block).items()}

class TestGLM5KDA(unittest.TestCase):
  def test_qk_l2norm_small_vectors(self):
    # the first token's state update is beta * v (x) l2norm(k): with q/k projections tiny, FLA's x/sqrt(sum(x²)+eps) isn't a unit vector
    ssm = SSMConfig(conv_kernel=4, state_size=8, group_count=2, time_step_rank=2, inner_size=16, kda=True, gate_lower_bound=-5.0,
                    qk_l2norm_eps=1e-6)
    block = GatedDeltaNetBlock(_config(ssm=ssm), ssm)
    w = _randomize(block, 0)
    qkv_names = [n for n in ("attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_qkv.weight") if n in w]
    qkv_w = np.concatenate([w[n] for n in qkv_names], axis=0)
    qkv_w[:32] *= 1e-4  # tiny q and k
    for n, part in zip(qkv_names, np.split(qkv_w, np.cumsum([w[n].shape[0] for n in qkv_names])[:-1])):
      nn.state.get_state_dict(block)[n].replace(Tensor(part))
    x = np.random.default_rng(1).standard_normal((1, 1, 16)).astype(np.float32)
    block._init_state(Tensor(x))
    block._attention(Tensor(x), 0).realize()

    qkv = x.astype(np.float16).astype(np.float32)[0, 0] @ qkv_w.T
    conv = qkv * w["ssm_conv1d.weight"][:, -1]  # the conv window is [0, 0, 0, qkv]
    conv = conv / (1 + np.exp(-conv))
    k, v = conv[16:32].reshape(2, 8), conv[32:].reshape(2, 8)
    k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
    beta = 1 / (1 + np.exp(-(x.astype(np.float16).astype(np.float32)[0, 0] @ w["ssm_beta.weight"].T)))
    np.testing.assert_allclose(block.recurrent_state.numpy()[0], beta[:, None, None] * v[:, :, None] * k[:, None, :], rtol=2e-2, atol=1e-6)

class TestGLM5DSA(unittest.TestCase):
  def _check_mask(self, kpool:int, top_k:int, max_context:int, chunks:list[tuple[int, int]]):
    idx = IndexerConfig(n_heads=2, head_dim=8, top_k=top_k, kpool=kpool, norm_eps=1e-6)
    cfg = _config(max_context=max_context, indexer=idx, ssm=SSMConfig(4, 8, 2, 2, 16, kda=True))
    block = DSAMLATransformerBlock(cfg)
    W = _randomize(block, kpool * 100 + top_k)
    xs = np.random.default_rng(2).standard_normal((1, max_context, 16)).astype(np.float32)
    block._init_state(Tensor(xs[:, :1]))
    def layernorm(z, w, b): return (z - z.mean(-1, keepdims=True)) / np.sqrt(z.var(-1, keepdims=True) + idx.norm_eps) * w + b
    def rmsnorm(z, w): return z / np.sqrt((z * z).mean(-1, keepdims=True) + cfg.norm_eps) * w
    def reference(start_pos, T):  # a query sees the top_k/kpool best complete pools and its incomplete pool (the tail)
      ik = layernorm(xs[0, :start_pos+T] @ W["indexer.attn_k.weight"].T, W["indexer.k_norm.weight"], W["indexer.k_norm.bias"])
      ig = xs[0, :start_pos+T] @ W["indexer_compressor_gate.weight"].T
      xq = xs[0, start_pos:start_pos+T]
      iq = (rmsnorm(xq @ W["attn_q_a.weight"].T, W["attn_q_a_norm.weight"]) @ W["indexer.attn_q_b.weight"].T).reshape(T, idx.n_heads, idx.head_dim)
      head_w = (xq @ W["indexer.proj.weight"].T) * (idx.n_heads * idx.head_dim) ** -0.5
      visible = np.zeros((T, start_pos+T), bool)
      for t in range(T):
        complete = (start_pos + t + 1) // kpool
        scores = []
        for p in range(complete):
          g = ig[p*kpool:(p+1)*kpool] + W["indexer_compressor_ape.weight"]
          g = np.exp(g - g.max(0)) / np.exp(g - g.max(0)).sum(0)
          scores.append((np.maximum(iq[t] @ (g * ik[p*kpool:(p+1)*kpool]).sum(0), 0) * head_w[t]).sum())
        for p in np.argsort(-np.array(scores), kind="stable")[:top_k // kpool]: visible[t, p*kpool:(p+1)*kpool] = True
        visible[t, complete*kpool:start_pos+t+1] = True
      return visible
    for start_pos, T in chunks:  # (contiguous: every position's indexer key gets cached)
      mask = block._attn_mask(Tensor(xs[:, start_pos:start_pos+T]), start_pos).numpy()[0, 0] == 0
      np.testing.assert_equal(mask, reference(start_pos, T), err_msg=f"start_pos={start_pos} T={T}")

  def test_complete_pool_needs_selection(self): self._check_mask(4, 4, 32, [(0, 8), (8, 5), (13, 6), (19, 5), (24, 8)])
  def test_unaligned_max_context(self): self._check_mask(2, 8, 33, [(0, 4), (4, 4), (8, 8), (16, 1), (17, 3), (20, 4), (24, 9)])

if __name__ == '__main__':
  unittest.main()
