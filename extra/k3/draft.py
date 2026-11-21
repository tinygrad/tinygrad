# the EAGLE-3 draft of K3 (kimi-k3-eagle3.1-mla): one MLA decoder layer on the target's hidden states after three of its blocks. a position
# j takes the next token t[j+1] and the target's hidden states after t[j], and guesses t[j+2]. its MLA keys (the normed latent, the roped key)
# in a cache. its linears split over the target's gpus like the trunk's (outputs or inputs), in q8_0 (in f16 where there's no kernel)
import json, math
from typing import Callable, cast
from tinygrad import Tensor, UOp, dtypes, nn
from tinygrad.uop.ops import smax
from tinygrad.helpers import prod, round_up
from tinygrad.llm.model import Transformer, FFNBlock, precompute_freqs_cis, apply_rope, YaRNConfig
from tinygrad.llm.kernels.amd import Linear, amd_custom_kernels_supported, mla_topk_attention, mla_topk_supported, allgather_last
from tinygrad.llm.kernels.amd_gguf import q8_0_linear

def weight(t:Tensor, devs:tuple[str, ...], axis:int|None=None, q8:bool=True) -> Tensor|Linear:
  """a checkpoint's weight (read once) split over devs on axis (None: a copy on each): a matrix in q8_0 where there's a kernel, else f16"""
  if not q8 or len(t.shape) == 1 or t.shape[1] % (32 * len(devs)) or not amd_custom_kernels_supported(devs):  # (vectors, rows not of blocks)
    return t.to(devs[0]).half().shard(devs, axis).realize()
  (lin:=q8_0_linear(t.to(devs[0]), devs, axis if len(devs) > 1 else None)).weight.realize()  # (one gpu: whole)
  return lin

class Eagle3:
  def __init__(self, path:str, embed:Callable[[Tensor], Tensor], devs:tuple[str, ...], max_context:int):
    cfg = json.load(open(f"{path}/config.json"))
    self.H, self.nope, self.rope, self.vd, self.lora = (cfg[k] for k in ("num_attention_heads", "qk_nope_head_dim", "qk_rope_head_dim",
                                                                          "v_head_dim", "kv_lora_rank"))
    self.eps, self.embed, self.norm_output = cfg["rms_norm_eps"], embed, cfg.get("norm_output", False)
    self.taps = [(i - 1, False) for i in cfg["eagle_config"]["eagle_aux_hidden_state_layer_ids"]]  # (the target's blocks it reads after, unmixed)
    rs = cfg["rope_scaling"]
    yarn = YaRNConfig(factor=rs["factor"], orig_ctx_len=rs["original_max_position_embeddings"], beta_fast=rs["beta_fast"], beta_slow=rs["beta_slow"])
    # deepseek's yarn: plain cos/sin (precompute_freqs_cis scales them by the concentration), the scores scaled by mscale^2 instead
    self.mscale = 0.1 * rs["mscale_all_dim"] * math.log(rs["factor"]) + 1.0
    self.freqs_cis = (precompute_freqs_cis(self.rope, max_context, cfg["rope_theta"], yarn=yarn) / self.mscale).to(devs).realize()
    st = nn.state.safe_load(f"{path}/model.safetensors")
    self.w = {k[:-len(".weight")]: weight(st[k], devs, axis) for k, axis in (
      *((k, None) for k in st if k.endswith("norm.weight") or k.startswith("fc_norm") or "_a_proj" in k or k == "fc.weight"),
      ("layers.0.self_attn.o_proj.weight", 1), ("layers.0.mlp.down_proj.weight", 1), ("layers.0.self_attn.q_b_proj.weight", 0),
      ("layers.0.mlp.gate_proj.weight", 0), ("layers.0.mlp.up_proj.weight", 0), ("lm_head.weight", 0))}
    # kv_b by head: the up projections of the latent to the nope keys (absorbed into the queries) and to the values
    kv_b = st["layers.0.self_attn.kv_b_proj.weight"].to(devs[0]).half().reshape(self.H, self.nope + self.vd, self.lora)
    self.k_b, self.v_b = (x.contiguous().shard(devs, 0).realize() for x in (kv_b[:, :self.nope], kv_b[:, self.nope:]))
    self.cache = Tensor.empty(1, 1, max_context, self.lora + self.rope, dtype=dtypes.half, device=devs).realize()
    self.positions = Tensor.arange(max_context).to(devs).realize()  # (the causal mask's, on the gpus: a symbolic arange isn't)

  width, scored = 1, False  # (a row's guesses: the next ones a chain's; no scores of them)
  blk:list[FFNBlock] = []
  def norm(self, x:Tensor, name:str) -> Tensor: return x.float().rmsnorm(eps=self.eps) * self.w[name]
  def lin(self, x:Tensor, name:str) -> Tensor: return w(x) if isinstance(w:=self.w[name], Linear) else x.half() @ w.T
  def guess(self, tokens:Tensor, hidden:Tensor, pos:int|UOp, row:Tensor|int) -> tuple[Tensor, None, Tensor]:
    """a step's guess (1,) after row row's next token (of its rows' (1, T)) on the target's taps, every row's feedback state (a chain's)"""
    logits, x, _ = self(tokens, hidden, pos)
    return pick(logits[0].argmax(-1), row).reshape(1).cast(dtypes.int32), None, x
  def fc(self, hidden:Tensor) -> Tensor:  # (the target's hidden states (1, T, 3, dim) to the layer's)
    return self.lin(Tensor.cat(*[self.norm(hidden[:, :, i], f"fc_norm.{i}") for i in range(hidden.shape[2])], dim=-1), "fc")

  def keys(self, tokens:Tensor, hid:Tensor, pos:int|UOp, cache:UOp) -> tuple[Tensor, UOp]:
    """the layer's input of (1, T) tokens at positions pos.. on hid (1, T, dim), the cache after their keys (the normed latent, the roped key)"""
    T, p, a = tokens.shape[1], "layers.0.", "layers.0.self_attn."
    h = self.norm(self.embed(tokens), p + "input_layernorm").cat(self.norm(hid, p + "hidden_norm"), dim=-1)
    kv = self.lin(h, a + "kv_a_proj_with_mqa")
    k_pe = apply_rope(kv[..., self.lora:].reshape(1, 1, T, self.rope), self.freqs_cis[pos:pos + T], interleaved=True)
    store = self.norm(kv[..., :self.lora], a + "kv_a_layernorm").reshape(1, 1, T, self.lora).cat(k_pe, dim=-1)
    return h, cache.after(cache[:, :, pos:pos + T].store(store.half().uop))

  def fill(self, tokens:Tensor, hidden:Tensor, pos:int|UOp) -> Tensor:
    """the cache after the keys of a prompt's chunk (tokens at positions pos.. on the target's hidden states), its guesses not needed"""
    return Tensor(self.keys(tokens, self.fc(hidden), pos, self.cache.uop)[1])

  def __call__(self, tokens:Tensor, hidden:Tensor|None, pos:int|UOp, hid:Tensor|None=None, cache:UOp|None=None) -> tuple[Tensor, Tensor, UOp]:
    """(1, T) tokens at positions pos.., (1, T, 3, dim) the target's hidden states (or hid, its own of the step before: a chain of guesses)
    -> (1, T, vocab) logits, (1, T, dim) its feedback states (output-normed if configured), its cache after their keys (a chain's next step's)"""
    T, p, a = tokens.shape[1], "layers.0.", "layers.0.self_attn."
    if hid is None: hid = self.fc(cast(Tensor, hidden))
    h, cache = self.keys(tokens, hid, pos, self.cache.uop if cache is None else cache)
    q = self.lin(self.norm(self.lin(h, a + "q_a_proj"), a + "q_a_layernorm"), a + "q_b_proj").reshape(1, T, self.H, -1).transpose(1, 2)
    q_pe = apply_rope(q[..., self.nope:], self.freqs_cis[pos:pos + T], interleaved=True)
    # absorbed: the nope scores (q_nope k_b) . latent, causal over the cache, the values v_b (attention . latent)
    qa = (q[..., :self.nope].half() @ self.k_b).float().cat(q_pe.float(), dim=-1)
    scale = self.mscale ** 2 / math.sqrt(self.nope + self.rope)
    if mla_topk_supported(qa.device, self.lora):
      attn = mla_topk_attention(qa, Tensor(cache), None, pos, self.lora, scale)
    else:
      k = Tensor(cache)[:, :, :pos + T].float()
      mask = (self.positions[:pos + T].reshape(1, 1, 1, -1) > self.positions[pos:pos + T].reshape(1, 1, T, 1)).where(float("-inf"), 0)
      attn = ((qa @ k.transpose(-1, -2)) * scale + mask).softmax(-1) @ k[..., :self.lora]
    x = self.lin((attn.half() @ self.v_b.transpose(-1, -2)).transpose(1, 2).reshape(1, T, -1), a + "o_proj") + hid
    m = self.norm(x, p + "post_attention_layernorm")
    x = x + self.lin(self.lin(m, p + "mlp.gate_proj").silu() * self.lin(m, p + "mlp.up_proj"), p + "mlp.down_proj")
    normed = self.norm(x, "norm")
    return self.lin(normed, "lm_head"), normed if self.norm_output else x, cache

# the DSpark draft of K3.1 (Inferact/Kimi-K3.1-Pro-DSpark): a block of k guesses in one pass. its five GQA layers' keys and values of a position
# are projected from one feature, the target's streams after five blocks (raw, concatenated, a linear and a norm); a block's queries are the
# anchor (a known token at position a) and k-1 masks at a.., each attending causally to the block and to the context's positions before a in a
# window. the j-th guesses t[a+j+1], its logits biased by a low-rank markov term of the token before it (the guess before, the anchor first).
# shares the target's embedding and head. its context's keys and values in a cache of every position (layer, k/v, 1, kv heads, position, dim)
def pick(x:Tensor, row:Tensor|int) -> Tensor:  # (x's row row (a tensor's: in the graph), x (T, ...))
  return (Tensor.arange(x.shape[0]).to(x.device).reshape(-1, *[1] * (x.ndim - 1)) == row).where(x, 0).sum(0)

def rows16(f:Callable[[Tensor], Tensor], x:Tensor) -> Tensor:
  """f(x) of a q8 linear (rows (..., in) -> (..., out)), its rows past the gemv kernel's 8 padded to the wmma kernel's 16s"""
  if (n:=prod(x.shape[:-1])) <= 8 or n % 16 == 0: return f(x)
  return f(x.reshape(n, x.shape[-1]).pad(((0, round_up(n, 16) - n), (0, 0))))[:n].reshape(*x.shape[:-1], -1)

class DSpark:
  blk:list[FFNBlock] = []
  def __init__(self, path:str, embed:Callable[[Tensor], Tensor], head:Callable[[Tensor], Tensor], devs:tuple[str, ...], max_context:int,
               width:int=1, q8:bool=True):
    cfg = json.load(open(f"{path}/config.json"))
    tl = cfg["transformer_layer_config"]
    self.H, self.KV, self.hd, self.L = tl["num_attention_heads"], tl["num_key_value_heads"], tl["head_dim"], tl["num_hidden_layers"]
    self.eps, self.window, self.mask_id, self.embed = tl["rms_norm_eps"], tl["sliding_window"], cfg["mask_token_id"], embed
    # (the target's head in a closure: not the draft's parameters, an engine realizes those (a quantized head's decoded weights, its kernel lost))
    self.head:Callable[[Tensor], Tensor] = lambda x: head(x)
    self.width, self.max_context, self.scored = width, max_context, True  # (a step's block of guesses after each row, scored: the confidence)
    assert cfg["sample_from_anchor"] and cfg["markov_head_type"] == "vanilla" and tl["rope_parameters"]["rope_type"] == "default"
    self.taps = [(i - 1, False) for i in cfg["aux_hidden_state_layer_ids"]]  # (the ids count the blocks done)
    self.freqs_cis = precompute_freqs_cis(self.hd, max_context, tl["rope_parameters"]["rope_theta"]).to(devs).realize()
    st = nn.state.safe_load(f"{path}/model.safetensors")
    # split by heads, by the mlp's rows, the markov term by the vocabulary (a guess at a time reads all of it)
    split = {"fc": 0, "q_proj": 0, "k_proj": 0, "v_proj": 0, "o_proj": 1, "gate_proj": 0, "up_proj": 0, "down_proj": 1, "markov_w2": 0}
    self.w = {k[:-len(".weight")]: weight(v, devs, split.get(k.split(".")[-2]), q8 and k.startswith(("layers.", "fc.", "markov_head.markov_w2")))
              for k, v in st.items() if k.endswith(".weight")}
    self.markov = nn.Embedding(*st["markov_head.markov_w1.weight"].shape)  # (the r-dim embedding of the token before a guess)
    self.markov.weight = cast(Tensor, self.w["markov_head.markov_w1"])
    self.conf_b = st["confidence_head.proj.bias"].to(devs[0]).float().to(devs).realize()
    self.cache = Tensor.zeros(self.L, 2, 1, self.KV, max_context, self.hd, dtype=dtypes.half).contiguous().shard(devs, 3).realize()

  def norm(self, x:Tensor, name:str) -> Tensor: return x.float().rmsnorm(eps=self.eps) * self.w[name]
  def lin(self, x:Tensor, name:str) -> Tensor: return rows16(w, x) if isinstance(w:=self.w[name], Linear) else x.half() @ w.T
  def kv(self, x:Tensor, a:str, freqs:Tensor) -> tuple[Tensor, Tensor]:  # (x (B, T, dim) -> keys, values (B, kv, T, hd), the keys roped)
    B, T = x.shape[:2]
    k = self.norm(self.lin(x, a + "k_proj").reshape(B, T, self.KV, self.hd), a + "k_norm").transpose(1, 2)
    return apply_rope(k, freqs), self.lin(x, a + "v_proj").reshape(B, T, self.KV, self.hd).transpose(1, 2).float()

  def fill(self, tokens:Tensor, hidden:Tensor, pos:int|UOp) -> Tensor:
    """the cache after the context's keys and values of positions pos.. on the target's taps hidden (1, T, taps, dim)"""
    T = hidden.shape[1]
    c = self.norm(allgather_last(self.lin(hidden.reshape(1, T, -1), "fc"), exact=True), "hidden_norm")
    stores = []
    for l in range(self.L):
      for i, x in enumerate(self.kv(c, f"layers.{l}.self_attn.", self.freqs_cis[pos:pos + T])):
        stores.append(self.cache[l, i, :, :, pos:pos + T].uop.store(x.cast(dtypes.half).uop))
    return Tensor(self.cache.uop.after(*stores))

  def guess(self, tokens:Tensor, hidden:Tensor, pos:int|UOp, row:Tensor|int) -> tuple[Tensor, Tensor, None]:
    """a step's block of guesses (width,) after row row's next token (of its rows' (1, T): the anchor, at position pos+row+1) and the confidence
    each is right (given the ones before right), the cache after the rows' keys on the target's taps hidden (1, T, taps, dim) first"""
    T = tokens.shape[1]
    n = min(self.window + T, self.max_context)  # (the keys a block sees: from pos-window on)
    cache = self.fill(tokens, hidden, pos)
    apos = pick((Tensor.arange(T) + (pos + 1)).to(hidden.device), row).reshape(1)
    guesses, conf, _ = self(pick(tokens[0], row).reshape(1), apos, self.width, (smax(pos + T - n, 0), n), cache=cache)
    return guesses[0], conf[0], None

  def __call__(self, anchors:Tensor, apos:Tensor, k:int, keys:tuple[int|UOp, int], prev:Tensor|None=None,
               cache:Tensor|None=None) -> tuple[Tensor, Tensor, Tensor]:
    """A blocks of k guesses after anchors (A,) at positions apos (A,) on the cache's positions keys (start, n): their tokens (A, k), the
    confidence each is right (A, k), its logits (A, k, vocab). prev (A, k): the tokens before the guesses taken as given (an evaluation's)"""
    A, (k0, n), cache = anchors.shape[0], keys, self.cache if cache is None else cache
    qpos = apos.reshape(A, 1) + Tensor.arange(k).to(anchors.device).reshape(1, k)  # (A, k)
    freqs = self.freqs_cis[qpos.flatten()].reshape(A, k, -1)
    # a query sees the context before its anchor in the window, its block causally
    kp = (Tensor.arange(n).to(anchors.device) + k0).reshape(1, 1, 1, n)
    ctx = (kp < apos.reshape(A, 1, 1, 1)) & (kp > qpos.reshape(A, 1, k, 1) - self.window)
    blk = Tensor.ones(k, k, dtype=dtypes.bool, device=anchors.device).tril().reshape(1, 1, k, k)
    mask = ctx.expand(A, 1, k, n).cat(blk.expand(A, 1, k, k), dim=-1).where(0.0, -math.inf).reshape(A, 1, 1, k, n + k)
    x = self.embed(anchors.reshape(A, 1).pad(((0, 0), (0, k - 1)), value=self.mask_id)).float()  # (A, k, dim)
    G = self.H // self.KV
    for l in range(self.L):
      a = f"layers.{l}.self_attn."
      h = self.norm(x, f"layers.{l}.input_layernorm")
      q = apply_rope(self.norm(self.lin(h, a + "q_proj").reshape(A, k, self.H, self.hd), a + "q_norm").transpose(1, 2), freqs)
      kb, vb = self.kv(h, a, freqs)
      # every block's queries on one copy of the context's keys (kv, n, hd), a block's own keys (A, kv, k, hd) apart
      kc, vc = (cache[l, i, 0, :, k0:k0 + n].float() for i in range(2))
      q = q.reshape(A, self.KV, G, k, self.hd)
      def flat(x:Tensor) -> Tensor: return x.permute(1, 0, 2, 3, 4).reshape(self.KV, A * G * k, -1)  # ((A, kv, G, k, d) -> (kv, A*G*k, d))
      def unflat(x:Tensor) -> Tensor: return x.reshape(self.KV, A, G, k, -1).permute(1, 0, 2, 3, 4)
      C, N = 32, round_up(n, 32)
      sc = (unflat(flat(q) @ kc.transpose(-1, -2)) / math.sqrt(self.hd) + mask[..., :n]).pad((None,) * 4 + ((0, N - n),), value=-math.inf)
      sc, sb = sc.reshape(A, self.KV, G, k, C, N // C), q @ kb.unsqueeze(2).transpose(-1, -2) / math.sqrt(self.hd) + mask[..., n:]
      cm = sc.max(-1).maximum(-1e30).contiguous()
      top = cm.max(-1).maximum(sb.max(-1)).unsqueeze(-1)
      e, w, eb = (sc - cm.unsqueeze(-1)).exp(), (cm - top).exp(), (sb - top).exp()
      ev = e.permute(1, 4, 0, 2, 3, 5).reshape(self.KV, C, A * G * k, N // C) @ vc.pad((None, (0, N - n), None)).reshape(self.KV, C, N // C, -1)
      ev = ev.contiguous().reshape(self.KV, C, A, G, k, self.hd).permute(2, 0, 3, 4, 1, 5)
      o = ((ev * w.unsqueeze(-1)).sum(-2) + eb @ vb.unsqueeze(2)) / ((e.sum(-1).contiguous() * w).sum(-1) + eb.sum(-1)).unsqueeze(-1)
      o = o.reshape(A, self.H, k, self.hd).transpose(1, 2).reshape(A, k, -1)
      x = x + self.lin(o, a + "o_proj")
      m = self.norm(x, f"layers.{l}.post_attention_layernorm")
      p = f"layers.{l}.mlp."
      x = x + self.lin(self.lin(m, p + "gate_proj").silu() * self.lin(m, p + "up_proj"), p + "down_proj")
    h = self.norm(x, "norm")
    logits = rows16(self.head, h).float().contiguous()  # (A, k, vocab) (a buffer: the guesses' argmaxes read it, not recompute it)
    def markov(t:Tensor) -> Tensor: return self.markov(t).float()
    if prev is None:  # (greedy, a guess at a time: the next one's markov term on it)
      t, outs, embs = anchors, [], []
      for j in range(k):
        embs.append(e:=markov(t))
        outs.append(t:=(logits[:, j] + self.lin(e, "markov_head.markov_w2").float()).argmax(-1))
      out, e = Tensor.stack(*outs, dim=1), Tensor.stack(*embs, dim=1)
    else:
      e = markov(prev)
      logits = logits + self.lin(e, "markov_head.markov_w2").float()
      out = logits.argmax(-1)
    conf = (h.cat(e, dim=-1) @ self.w["confidence_head.proj"].float().T + self.conf_b).sigmoid()[..., 0]
    return out.cast(dtypes.int32), conf, logits

class MTP:
  width, scored = 1, True
  def __init__(self, model:Transformer):
    self.blk, self.taps = [model.mtp], [(len(model.blk) - 1, False)]
    self.embed:Callable[[Tensor], Tensor] = lambda t: model.token_embd(t).float()
    self.output_norm:Callable[[Tensor], Tensor] = lambda x: model.output_norm(x)
    self.head:Callable[[Tensor], Tensor] = lambda x: model.output(x)

  def guess(self, tokens:Tensor, hidden:Tensor, pos:int|UOp, row:Tensor|int) -> tuple[Tensor, Tensor, Tensor]:
    logits, x, _ = self(tokens, hidden, pos)
    return pick(logits[0].argmax(-1), row).reshape(1).cast(dtypes.int32), pick(logits[0].softmax(-1).max(-1), row).reshape(1), x

  def fill(self, tokens:Tensor, hidden:Tensor, pos:int|UOp) -> Tensor:
    b = self.blk[0]
    h = b.attn_norm(b.project(self.embed(tokens), self.output_norm(hidden[:, :, 0])))
    return b.fill_cache(h, pos)

  def __call__(self, tokens:Tensor, hidden:Tensor|None, pos:int|UOp, hid:Tensor|None=None, cache:None=None) -> tuple[Tensor, Tensor, None]:
    b = self.blk[0]
    x, _ = b(b.project(self.embed(tokens), self.output_norm(cast(Tensor, hidden)[:, :, 0]) if hid is None else hid), (pos,))
    h = b.shared_head_norm(x)
    return self.head(h).float(), h, None

def load(path:str, embed:Callable[[Tensor], Tensor], head:Callable[[Tensor], Tensor], devs:tuple[str, ...], max_context:int,
         width:int) -> Eagle3|DSpark:
  """the draft at path (its config's kind), its blocks of width guesses (a block draft's)"""
  if json.load(open(f"{path}/config.json")).get("speculators_model_type") == "dspark": return DSpark(path, embed, head, devs, max_context, width)
  return Eagle3(path, embed, devs, max_context)
