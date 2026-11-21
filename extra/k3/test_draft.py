import json, tempfile, unittest
from unittest.mock import patch
import numpy as np
from tinygrad import Device, Tensor, TinyJit, UOp, dtypes, nn
from tinygrad.llm.kernels.amd import amd_custom_kernels_supported
from extra.k3.draft import Eagle3, DSpark, rows16
from extra.k3.test_engine import eagle, DIM, VOCAB

class TestDraftFeedback(unittest.TestCase):
  def test_norm_output(self): # norm_output: a chain's next step gets the normed hidden state (the lm head's input), else the raw one
    rng, devs = np.random.default_rng(11), (Device.DEFAULT,)
    embed = Tensor(rng.normal(0, 0.3, (VOCAB, DIM)).astype(np.float16)).shard(devs).realize()
    tokens, hidden = Tensor([[3, 5]], dtype=dtypes.int32).shard(devs), Tensor(rng.normal(0, 0.3, (1, 2, 3, DIM)).astype(np.float32)).shard(devs)
    with tempfile.TemporaryDirectory() as folder, patch("extra.k3.draft.amd_custom_kernels_supported", return_value=False):
      cfg = json.load(open(f"{(path:=eagle(folder, rng))}/config.json"))
      drafts = []
      for norm in (False, True):
        json.dump({**cfg, "norm_output": norm}, open(f"{path}/config.json", "w"))
        drafts.append(Eagle3(path, lambda t: embed[t], devs, 16))
      (l0, raw, _), (l1, normed, _) = (d(tokens, hidden, 0) for d in drafts)
      np.testing.assert_array_equal(l0.numpy(), l1.numpy())
      np.testing.assert_allclose(normed.numpy(), drafts[1].norm(raw, "norm").numpy(), rtol=1e-5, atol=1e-5)

def dspark(folder:str, rng:np.random.Generator, taps:int=3, window:int=6) -> str:
  """a tiny random DSpark (Inferact/Kimi-K3.1-Pro-DSpark's layout) on the target's streams after blocks 0.."""
  H, KV, hd, L, ff, r = 4, 2, 8, 2, 32, 4
  json.dump({"aux_hidden_state_layer_ids": list(range(1, taps + 1)), "mask_token_id": VOCAB - 1, "markov_head_type": "vanilla",
             "sample_from_anchor": True, "transformer_layer_config": {"num_attention_heads": H, "num_key_value_heads": KV, "head_dim": hd,
             "num_hidden_layers": L, "rms_norm_eps": 1e-5, "sliding_window": window,
             "rope_parameters": {"rope_type": "default", "rope_theta": 100.0}}},
            open(f"{folder}/config.json", "w"))
  shapes = {"fc.weight": (DIM, taps * DIM), "hidden_norm.weight": (DIM,), "norm.weight": (DIM,), "markov_head.markov_w1.weight": (VOCAB, r),
            "markov_head.markov_w2.weight": (VOCAB, r), "confidence_head.proj.weight": (1, DIM + r), "confidence_head.proj.bias": (1,)}
  for l in range(L):
    a, p = f"layers.{l}.self_attn.", f"layers.{l}."
    shapes.update({a+"q_proj.weight": (H*hd, DIM), a+"k_proj.weight": (KV*hd, DIM), a+"v_proj.weight": (KV*hd, DIM),
                   a+"o_proj.weight": (DIM, H*hd),
                   a+"q_norm.weight": (hd,), a+"k_norm.weight": (hd,), p+"input_layernorm.weight": (DIM,),
                   p+"post_attention_layernorm.weight": (DIM,),
                   p+"mlp.gate_proj.weight": (ff, DIM), p+"mlp.up_proj.weight": (ff, DIM), p+"mlp.down_proj.weight": (DIM, ff)})
  nn.state.safe_save({k: Tensor((1 + rng.normal(0, 0.1, s) if "norm" in k else rng.normal(0, 0.3, s)).astype(np.float32))
                      for k, s in shapes.items()}, f"{folder}/model.safetensors")
  return folder

def dspark_ref(path:str, emb:np.ndarray, out:np.ndarray, toks:list[int], hidden:np.ndarray, a:int, k:int) -> tuple[list[int], np.ndarray]:
  """the block after anchor toks[a] as speculators' DFlash/DSpark reference has it (its mask_mod, the markov chain greedy): tokens, base logits"""
  cfg = json.load(open(f"{path}/config.json"))
  w = {k: v.numpy().astype(np.float64) for k, v in nn.state.safe_load(f"{path}/model.safetensors").items()}
  tl = cfg["transformer_layer_config"]
  H, KV, hd, win = tl["num_attention_heads"], tl["num_key_value_heads"], tl["head_dim"], tl["sliding_window"]
  def norm(x, name): return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-5) * w[name + ".weight"]
  def rope(x, pos):  # (x (T, heads, hd), rotate_half)
    f = np.outer(pos, 1.0 / 100.0 ** (np.arange(0, hd, 2) / hd))
    cos, sin = np.cos(f)[:, None], np.sin(f)[:, None]
    x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)
  ctx = norm(hidden.reshape(len(toks), -1) @ w["fc.weight"].T, "hidden_norm")
  x, qpos, kpos = emb[[toks[a]] + [cfg["mask_token_id"]] * (k - 1)].astype(np.float64), np.arange(a, a + k), np.arange(len(toks))
  for l in range(tl["num_hidden_layers"]):
    p = f"layers.{l}.self_attn."
    h = norm(x, f"layers.{l}.input_layernorm")
    q = rope(norm((h @ w[p+"q_proj.weight"].T).reshape(k, H, hd), p+"q_norm"), qpos)
    kk = np.concatenate([rope(norm((y @ w[p+"k_proj.weight"].T).reshape(-1, KV, hd), p+"k_norm"), ps) for y, ps in ((ctx, kpos), (h, qpos))])
    vv = np.concatenate([(y @ w[p+"v_proj.weight"].T).reshape(-1, KV, hd) for y in (ctx, h)])
    allowed = np.concatenate([(kpos[None] < a) & (kpos[None] > qpos[:, None] - win), np.tril(np.ones((k, k), bool))], 1)
    o = np.zeros((k, H, hd))
    for i in range(H):
      s = np.where(allowed, q[:, i] @ kk[:, i // (H // KV)].T / np.sqrt(hd), -np.inf)
      s = np.exp(s - s.max(-1, keepdims=True))
      o[:, i] = (s / s.sum(-1, keepdims=True)) @ vv[:, i // (H // KV)]
    x = x + o.reshape(k, -1) @ w[p+"o_proj.weight"].T
    m, p = norm(x, f"layers.{l}.post_attention_layernorm"), f"layers.{l}.mlp."
    g = m @ w[p+"gate_proj.weight"].T
    x = x + (g / (1 + np.exp(-g)) * (m @ w[p+"up_proj.weight"].T)) @ w[p+"down_proj.weight"].T
  logits, prev, got = norm(x, "norm") @ out.T, toks[a], []
  for j in range(k):
    got.append(prev:=int((logits[j] + w["markov_head.markov_w1.weight"][prev] @ w["markov_head.markov_w2.weight"].T).argmax()))
  return got, logits

class TestDSpark(unittest.TestCase):
  def test_rows16(self):  # a linear's rows past 8 padded to 16s (its rows, then the result's, cut back): as unpadded
    w, seen = Tensor(np.random.default_rng(2).normal(0, 1, (6, 4)).astype(np.float32)), []
    def f(x:Tensor) -> Tensor:
      seen.append(x.shape)
      return x @ w.T
    for shape, rows in (((2, 3, 4), 6), ((3, 5, 4), 16), ((4, 4, 4), 16), ((5, 7, 4), 48)):
      x = Tensor(np.random.default_rng(3).normal(0, 1, shape).astype(np.float32))
      np.testing.assert_allclose(rows16(f, x).numpy(), (x @ w.T).numpy(), rtol=1e-5)
      self.assertEqual(int(np.prod(seen[-1][:-1])), rows)

  def test_head_not_a_parameter(self):  # the target's head isn't the draft's (an engine realizes a draft's parameters: a quantized head decoded)
    devs, head = (Device.DEFAULT,), nn.Linear(DIM, VOCAB, bias=False)
    with tempfile.TemporaryDirectory() as folder, patch("extra.k3.draft.amd_custom_kernels_supported", return_value=False):
      d = DSpark(dspark(folder, np.random.default_rng(1)), lambda t: t, head, devs, 16)
    self.assertNotIn(id(head.weight), {id(t) for t in nn.state.get_parameters(d)})

  def test_block(self):  # a block's guesses and logits as the reference's, on all the cache's positions or the window's slice of them
    rng, devs, N, k = np.random.default_rng(3), (Device.DEFAULT,), 12, 4
    emb, out = rng.normal(0, 0.3, (VOCAB, DIM)).astype(np.float16), rng.normal(0, 0.3, (VOCAB, DIM)).astype(np.float16)
    E, O = Tensor(emb).shard(devs).realize(), Tensor(out).shard(devs).realize()
    with tempfile.TemporaryDirectory() as folder, patch("extra.k3.draft.amd_custom_kernels_supported", return_value=False):
      draft = DSpark(path:=dspark(folder, rng), lambda t: E[t], lambda x: x.half() @ O.T, devs, 32)
      toks, hidden = rng.integers(0, VOCAB - 1, N).tolist(), rng.normal(0, 1, (N, 3, DIM)).astype(np.float32)
      draft.fill(None, Tensor(hidden[None]).shard(devs), 0).realize()
      anchors = [1, 5, 9, 11]
      at, ap = Tensor([toks[a] for a in anchors], dtype=dtypes.int32).shard(devs), Tensor(anchors, dtype=dtypes.int32).shard(devs)
      for keys in ((0, 32), (11 - 6, 8)):
        got, _, logits = draft(at, ap, k, keys)
        if keys[0] == 0: first = got.numpy()[:, 0]
        for i, a in enumerate(anchors):
          if a - 6 < keys[0]: continue  # (the slice doesn't hold its window)
          want, want_logits = dspark_ref(path, emb, out, toks, hidden, a, k)
          np.testing.assert_allclose(logits.numpy()[i], want_logits, rtol=2e-2, atol=1e-2 * np.abs(want_logits).max())  # (f16 weights)
          self.assertEqual(got.numpy()[i].tolist(), want)
      prev = Tensor([[toks[a]] + [3] * (k - 1) for a in anchors], dtype=dtypes.int32).shard(devs)  # (given: the markov terms on them)
      got_given, _, _ = draft(at, ap, k, (0, 32), prev)
      self.assertEqual(got_given.numpy()[:, 0].tolist(), first.tolist())

@unittest.skipUnless(amd_custom_kernels_supported(Device.DEFAULT), "AMD attention kernel")
class TestDraftAttention(unittest.TestCase):
  def test_attention(self):
    rng, N = np.random.default_rng(7), 64
    devs = (Device.DEFAULT,)
    embed = Tensor(rng.normal(0, 0.3, (VOCAB, DIM)).astype(np.float16)).shard(devs).realize()
    with tempfile.TemporaryDirectory() as folder, patch("extra.k3.draft.amd_custom_kernels_supported", return_value=False):
      draft = Eagle3(eagle(folder, rng, lora=32), lambda t: embed[t], devs, N)
    for T, start in ((1, 0), (2, 31), (5, 50)):
      tokens = Tensor(rng.integers(0, VOCAB, (1, T)), dtype=dtypes.int32).shard(devs).realize()
      hidden = Tensor(rng.normal(0, 0.3, (1, T, 3, DIM)).astype(np.float32)).shard(devs).realize()
      cache = rng.standard_normal(draft.cache.shape).astype(np.float16)
      cache[..., start + T:, :] = np.nan  # unwritten cache rows must never affect a guess
      def reset(): draft.cache.assign(Tensor(cache).shard(devs)).realize()
      reset()
      with patch("extra.k3.draft.amd_custom_kernels_supported", return_value=False):
        logits, h, c = draft(tokens, hidden, start)
        want = logits.numpy(), h.numpy(), Tensor(c).numpy()[..., :start + T, :]
      reset()
      @TinyJit
      def run(pos):
        logits, h, c = draft(tokens, hidden, pos)
        out = logits, h, Tensor(c)
        Tensor.realize(*out)
        return out
      for _ in range(3):
        logits, h, c = run(UOp.variable("start_pos", 0, N - T).bind(start))
        for got, ref in zip((logits.numpy(), h.numpy(), c.numpy()[..., :start + T, :]), want):
          np.testing.assert_allclose(got, ref, rtol=2e-3, atol=2e-3)

if __name__ == "__main__": unittest.main()
