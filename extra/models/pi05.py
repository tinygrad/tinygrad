# pi0.5 (Physical Intelligence): PaliGemma VLM + flow matching action expert. https://arxiv.org/abs/2504.16054
# reference: https://github.com/Physical-Intelligence/openpi (src/openpi/models/gemma.py, siglip.py, pi0.py)
import math
from tinygrad import Tensor, nn, dtypes, TinyJit

# gemma variants used by pi0.5: the PaliGemma 2B backbone and the 300M action expert (openpi models/gemma.py)
GEMMA_2B   = dict(width=2048, depth=18, mlp_dim=16384, num_heads=8, num_kv_heads=1, head_dim=256)
GEMMA_300M = dict(width=1024, depth=18, mlp_dim=4096,  num_heads=8, num_kv_heads=1, head_dim=256)
PI05_EXPERT = GEMMA_300M | {"adarms": True}   # in pi05 the expert is conditioned on the flow timestep via adaRMSNorm
PALIGEMMA_VOCAB_SIZE = 257_152

def apply_rope(x:Tensor, positions:Tensor, max_wavelength=10_000) -> Tensor:
  # split-half rope (gemma/big_vision style), NOT interleaved like llama. x is (B, L, heads, head_dim), positions is (B, L)
  d = x.shape[-1]
  freqs = positions.float()[..., None] / (max_wavelength ** (Tensor.arange(d//2, dtype=dtypes.float32) * 2 / d))
  sin, cos = freqs.sin()[..., None, :], freqs.cos()[..., None, :]
  x1, x2 = x.float().chunk(2, dim=-1)
  return (x1*cos - x2*sin).cat(x2*cos + x1*sin, dim=-1).cast(x.dtype)

def posemb_sincos(t:Tensor, dim:int, min_period=4e-3, max_period=4.0) -> Tensor:
  # sine-cosine embedding of the flow timestep t in [0, 1]. t is (B,), output is (B, dim)
  fraction = Tensor.arange(dim//2, dtype=dtypes.float32) / (dim//2 - 1)
  period = min_period * (max_period / min_period) ** fraction
  x = t.float().reshape(-1, 1) * (2 * math.pi / period).reshape(1, -1)
  return x.sin().cat(x.cos(), dim=-1)

class RMSNorm:
  # gemma style: computed in float32, scale is (1 + weight) with weight zero-init
  def __init__(self, dim:int, eps:float=1e-6): self.weight, self.eps = Tensor.zeros(dim), eps
  def __call__(self, x:Tensor) -> Tensor:
    normed = x.float() * (x.float().square().mean(-1, keepdim=True) + self.eps).rsqrt()
    return (normed * (1 + self.weight.float())).cast(x.dtype)

class AdaRMSNorm:
  # pi05 action expert norm: scale/shift/gate come from a dense layer on the flow time embedding. the dense is
  # zero-init so at init scale=shift=gate=0 and the expert starts as an identity network (gated residuals add nothing).
  # returns (normed, gate) for the gated residual x + y*gate.
  def __init__(self, dim:int, eps:float=1e-6):
    self.dense, self.eps = nn.Linear(dim, 3*dim), eps
    self.dense.weight.replace(self.dense.weight.zeros_like())
    self.dense.bias.replace(self.dense.bias.zeros_like())
  def __call__(self, x:Tensor, cond:Tensor) -> tuple[Tensor, Tensor]:
    normed = x.float() * (x.float().square().mean(-1, keepdim=True) + self.eps).rsqrt()
    scale, shift, gate = self.dense(cond).reshape(x.shape[0], 1, -1).chunk(3, dim=-1)
    return (normed * (1 + scale.float()) + shift.float()).cast(x.dtype), gate

def _norm(norms, i, x, conds): return norms[i](x, conds[i]) if isinstance(norms[i], AdaRMSNorm) else (norms[i](x), None)

class FeedForward:
  def __init__(self, dim:int, hidden_dim:int):
    self.gate = nn.Linear(dim, hidden_dim, bias=False)
    self.up = nn.Linear(dim, hidden_dim, bias=False)
    self.down = nn.Linear(hidden_dim, dim, bias=False)
  def __call__(self, x:Tensor) -> Tensor: return self.down(self.gate(x).gelu() * self.up(x))  # tanh gelu, matches jax.nn.gelu

class Attention:
  # joint attention over a mixture of experts: each expert projects its own tokens with its own weights,
  # the sequences are concatenated, attention runs once over everything, and outputs are split back per expert.
  # this is how the action expert reads the VLM: same attention op, different weights per token group.
  def __init__(self, configs:tuple[dict, ...]):
    c0 = configs[0]
    assert all(c[k] == c0[k] for c in configs for k in ["num_heads", "num_kv_heads", "head_dim"])
    self.num_heads, self.num_kv_heads, self.head_dim = c0["num_heads"], c0["num_kv_heads"], c0["head_dim"]
    self.wq = [nn.Linear(c["width"], c["num_heads"]*c["head_dim"], bias=False) for c in configs]
    self.wk = [nn.Linear(c["width"], c["num_kv_heads"]*c["head_dim"], bias=False) for c in configs]
    self.wv = [nn.Linear(c["width"], c["num_kv_heads"]*c["head_dim"], bias=False) for c in configs]
    self.wo = [nn.Linear(c["num_heads"]*c["head_dim"], c["width"], bias=False) for c in configs]

  def __call__(self, xs:list[Tensor|None], positions:Tensor, mask:Tensor, kv_cache=None):
    active = [(i, x) for i, x in enumerate(xs) if x is not None]
    B, dtype = active[0][1].shape[0], active[0][1].dtype
    q = Tensor.cat(*[self.wq[i](x).reshape(B, x.shape[1], self.num_heads, self.head_dim) for i, x in active], dim=1)
    k = Tensor.cat(*[self.wk[i](x).reshape(B, x.shape[1], self.num_kv_heads, self.head_dim) for i, x in active], dim=1)
    v = Tensor.cat(*[self.wv[i](x).reshape(B, x.shape[1], self.num_kv_heads, self.head_dim) for i, x in active], dim=1)

    q, k = apply_rope(q, positions), apply_rope(k, positions)
    q = q * self.head_dim**-0.5
    if kv_cache is not None: k, v = kv_cache[0].cat(k, dim=1), kv_cache[1].cat(v, dim=1)

    # grouped-query attention in float32: every group of q heads shares one kv head
    T, S = q.shape[1], k.shape[1]
    q = q.reshape(B, T, self.num_kv_heads, self.num_heads // self.num_kv_heads, self.head_dim)
    logits = Tensor.einsum("btkgh,bskh->bkgts", q.float(), k.float())
    logits = mask.reshape(B, 1, 1, T, S).where(logits, -2.3819763e38)  # big_neg, see gemma/modules.py
    out = Tensor.einsum("bkgts,bskh->btkgh", logits.softmax(-1).cast(dtype), v).reshape(B, T, self.num_heads*self.head_dim)

    outs, start = [None]*len(xs), 0
    for i, x in active:
      outs[i] = self.wo[i](out[:, start:start+x.shape[1]])
      start = start + x.shape[1]
    return outs, (k, v)

class TransformerBlock:
  def __init__(self, configs:tuple[dict, ...]):
    self.attention = Attention(configs)
    def norm(c): return AdaRMSNorm(c["width"]) if c.get("adarms") else RMSNorm(c["width"])
    self.attention_norm = [norm(c) for c in configs]
    self.ffn_norm = [norm(c) for c in configs]
    self.feed_forward = [FeedForward(c["width"], c["mlp_dim"]) for c in configs]

  def __call__(self, xs:list[Tensor|None], positions:Tensor, mask:Tensor, conds:list, kv_cache=None):
    normed, gates = zip(*[_norm(self.attention_norm, i, x, conds) if x is not None else (None, None) for i, x in enumerate(xs)])
    attn_out, kv_cache = self.attention(list(normed), positions, mask, kv_cache)
    hs = [x if x is None else x + (y if g is None else y*g) for x, y, g in zip(xs, attn_out, gates)]
    out = []
    for i, h in enumerate(hs):
      if h is None:
        out.append(None)
        continue
      y, g = _norm(self.ffn_norm, i, h, conds)
      y = self.feed_forward[i](y)
      out.append(h + (y if g is None else y*g))
    return out, kv_cache

class Gemma:
  # a stack of TransformerBlocks over a mixture of experts. xs is a list of token streams, one per expert
  # (None to skip an expert). expert 0 is the PaliGemma language model and owns the vocabulary.
  def __init__(self, configs:tuple[dict, ...]=(GEMMA_2B, PI05_EXPERT), vocab_size:int=PALIGEMMA_VOCAB_SIZE):
    assert all(c["depth"] == configs[0]["depth"] for c in configs)
    self.configs = configs
    self.embedder = nn.Embedding(vocab_size, configs[0]["width"])
    self.layers = [TransformerBlock(configs) for _ in range(configs[0]["depth"])]
    self.final_norm = [AdaRMSNorm(c["width"]) if c.get("adarms") else RMSNorm(c["width"]) for c in configs]

  # embedder may live on another device (e.g. CPU to save vram), route through it transparently
  def embed(self, tokens:Tensor) -> Tensor:
    return self.embedder(tokens.to(self.embedder.weight.device)).to(tokens.device) * self.configs[0]["width"]**0.5
  def decode(self, x:Tensor) -> Tensor:  # logits share the embedding table, f32: the 2048-wide dot overflows half
    return (x.to(self.embedder.weight.device).float() @ self.embedder.weight.float().T).to(x.device)

  def __call__(self, xs:list[Tensor|None], positions:Tensor, mask:Tensor, conds:list|None=None, kv_cache=None):
    if conds is None: conds = [None]*len(self.configs)
    new_cache = []
    for i, layer in enumerate(self.layers):
      xs, kv = layer(xs, positions, mask, conds, kv_cache[i] if kv_cache is not None else None)
      new_cache.append(kv)
    return [_norm(self.final_norm, i, x, conds)[0] if x is not None else None for i, x in enumerate(xs)], new_cache

# *** siglip vision tower: So400m/14, the eyes. standard pre-norm ViT, full bidirectional attention ***

class VisionAttention:
  def __init__(self, dim:int, n_heads:int):
    self.n_heads = n_heads
    self.q_proj, self.k_proj, self.v_proj, self.out_proj = [nn.Linear(dim, dim) for _ in range(4)]
  def __call__(self, x:Tensor) -> Tensor:
    B, T, C = x.shape
    q, k, v = [p(x).reshape(B, T, self.n_heads, C//self.n_heads).transpose(1, 2) for p in (self.q_proj, self.k_proj, self.v_proj)]
    return self.out_proj(q.scaled_dot_product_attention(k, v).transpose(1, 2).reshape(B, T, C))

class VisionMLP:
  def __init__(self, dim:int, hidden_dim:int): self.fc1, self.fc2 = nn.Linear(dim, hidden_dim), nn.Linear(hidden_dim, dim)
  def __call__(self, x:Tensor) -> Tensor: return self.fc2(self.fc1(x).gelu())

class VisionBlock:
  def __init__(self, dim:int, n_heads:int, mlp_dim:int):
    self.layer_norm1, self.layer_norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
    self.self_attn, self.mlp = VisionAttention(dim, n_heads), VisionMLP(dim, mlp_dim)
  def __call__(self, x:Tensor) -> Tensor:
    x = x + self.self_attn(self.layer_norm1(x))
    return x + self.mlp(self.layer_norm2(x))

class SigLIP:
  # 224x224 image in [-1, 1] -> 16x16 patches of 14x14 pixels -> 256 tokens of dim 1152
  def __init__(self, dim:int=1152, depth:int=27, n_heads:int=16, mlp_dim:int=4304, patch_size:int=14, image_size:int=224):
    self.patch_embedding = nn.Conv2d(3, dim, patch_size, stride=patch_size)
    self.position_embedding = nn.Embedding((image_size//patch_size)**2, dim)
    self.layers = [VisionBlock(dim, n_heads, mlp_dim) for _ in range(depth)]
    self.post_layernorm = nn.LayerNorm(dim)
  def __call__(self, x:Tensor) -> Tensor:
    x = self.patch_embedding(x).flatten(2).transpose(1, 2) + self.position_embedding.weight
    for layer in self.layers: x = layer(x)
    return self.post_layernorm(x)

# *** pi0.5: prefix (images + language + discretized state as text) into the VLM, flow matching in the expert ***

class Pi05:
  def __init__(self, configs:tuple[dict, ...]=(GEMMA_2B, PI05_EXPERT), action_dim:int=32, action_horizon:int=50, vision:SigLIP|None=None):
    self.vision_tower = vision if vision is not None else SigLIP()
    self.multi_modal_projector = nn.Linear(self.vision_tower.patch_embedding.weight.shape[0], configs[0]["width"])
    self.llm = Gemma(configs)
    ew = configs[1]["width"]
    self.action_in_proj, self.action_out_proj = nn.Linear(action_dim, ew), nn.Linear(ew, action_dim)
    self.time_mlp_in, self.time_mlp_out = nn.Linear(ew, ew), nn.Linear(ew, ew)
    self.action_dim, self.action_horizon = action_dim, action_horizon
    self._prefill_jits, self._step_jits = {}, {}  # keyed by (batch, prefix len): kernel shapes depend on both

  def embed_image(self, image:Tensor) -> Tensor:
    return self.multi_modal_projector(self.vision_tower(image))  # image tokens are NOT sqrt-scaled like text

  def embed_prefix(self, images:list[Tensor], tokens:Tensor) -> Tensor:
    text = self.llm.embed(tokens).cast(self.multi_modal_projector.weight.dtype)  # f32 if the embedder lives on CPU
    return Tensor.cat(*[self.embed_image(img) for img in images], text, dim=1)

  def embed_suffix(self, noisy_actions:Tensor, time:Tensor) -> tuple[Tensor, Tensor]:
    dtype = self.action_in_proj.weight.dtype
    time_emb = posemb_sincos(time, self.time_mlp_in.weight.shape[1]).cast(dtype)
    cond = self.time_mlp_out(self.time_mlp_in(time_emb).silu()).silu()  # swish mlp -> adaRMSNorm conditioning
    return self.action_in_proj(noisy_actions.cast(dtype)), cond

  def _prefill(self, prefix:Tensor, positions:Tensor, mask:Tensor) -> list[Tensor]:
    _, kv_cache = self.llm([prefix, None], positions, mask)
    return [t for kv in kv_cache for t in kv]  # flat list: TinyJit finds tensors one container level deep

  def _step(self, x_t:Tensor, t:Tensor, positions:Tensor, mask:Tensor, flat_cache:list[Tensor], dt:float) -> Tensor:
    suffix, cond = self.embed_suffix(x_t, t)
    (_, out), _ = self.llm([None, suffix], positions, mask, conds=[None, cond], kv_cache=list(zip(flat_cache[0::2], flat_cache[1::2])))
    return x_t + dt * self.action_out_proj(out).float()  # euler update inside the jit: no python scheduling between steps

  def sample_actions(self, prefix:Tensor, noise:Tensor|None=None, num_steps:int=10, prefix_mask:Tensor|None=None) -> Tensor:
    # flow matching: start from noise at t=1, integrate x += dt*v toward the data at t=0 (openpi time convention).
    # prefill and step are TinyJit'd: first calls record the kernel stream, later calls replay it without scheduling.
    # prefix_mask (B, P) bool marks valid tokens: pad the prompt to a fixed length and jit shapes stay stable.
    B, P, H = prefix.shape[0], prefix.shape[1], self.action_horizon
    valid = prefix_mask if prefix_mask is not None else Tensor.ones(B, P, dtype=dtypes.bool)
    dt = -1.0 / num_steps
    if (key := (B, P, num_steps)) not in self._step_jits:
      self._prefill_jits[key], self._step_jits[key] = TinyJit(self._prefill), TinyJit(self._step)
      self._t_cache = getattr(self, "_t_cache", {})
      self._t_cache[key] = [Tensor([1.0 + i*dt]*B).realize() for i in range(num_steps)]
    flat_cache = self._prefill_jits[key](prefix, (valid.int().cumsum(1) - 1).clone().realize(),
                                         (valid.reshape(B, 1, P) * valid.reshape(B, P, 1)).clone().realize())
    x_t = (noise if noise is not None else Tensor.randn(B, H, self.action_dim)).contiguous().realize()
    # action tokens attend to the valid prefix and to each other bidirectionally; the prefix cache never sees them
    mask = valid.reshape(B, 1, P).expand(B, H, P).cat(Tensor.ones(B, H, H, dtype=dtypes.bool), dim=2).clone().realize()
    positions = (valid.int().sum(1, keepdim=True) + Tensor.arange(H).reshape(1, H)).clone().realize()
    for i in range(num_steps):
      x_t = self._step_jits[key](x_t, self._t_cache[key][i], positions, mask, flat_cache, dt)
    return x_t.clone().realize()  # the jit returns its fixed output buffer, the next call would overwrite it

# *** helpers ***

def convert_from_lerobot(weights:dict[str, Tensor], configs:tuple[dict, ...]=(GEMMA_2B, PI05_EXPERT)) -> dict[str, Tensor]:
  # key mapping for the huggingface lerobot/pi05_base checkpoint. the embedding table is stored tied to lm_head.
  LM, EX = "paligemma_with_expert.paligemma.model.language_model", "paligemma_with_expert.gemma_expert.model"
  sd = {"embedder.weight": weights["paligemma_with_expert.paligemma.lm_head.weight"]}
  for l in range(configs[0]["depth"]):
    for i, p in enumerate((f"{LM}.layers.{l}", f"{EX}.layers.{l}")[:len(configs)]):
      for x in "qkvo": sd[f"layers.{l}.attention.w{x}.{i}.weight"] = weights[f"{p}.self_attn.{x}_proj.weight"]
      for x in ("gate", "up", "down"): sd[f"layers.{l}.feed_forward.{i}.{x}.weight"] = weights[f"{p}.mlp.{x}_proj.weight"]
      if configs[i].get("adarms"):  # adaptive norms are a dense layer instead of a weight vector
        for a, b in (("attention_norm", "input_layernorm"), ("ffn_norm", "post_attention_layernorm")):
          for x in ("weight", "bias"): sd[f"layers.{l}.{a}.{i}.dense.{x}"] = weights[f"{p}.{b}.dense.{x}"]
      else:
        sd[f"layers.{l}.attention_norm.{i}.weight"] = weights[f"{p}.input_layernorm.weight"]
        sd[f"layers.{l}.ffn_norm.{i}.weight"] = weights[f"{p}.post_attention_layernorm.weight"]
  for i, p in enumerate((LM, EX)[:len(configs)]):
    if configs[i].get("adarms"):
      for x in ("weight", "bias"): sd[f"final_norm.{i}.dense.{x}"] = weights[f"{p}.norm.dense.{x}"]
    else: sd[f"final_norm.{i}.weight"] = weights[f"{p}.norm.weight"]
  return sd

def load_from_pretrained(model:Gemma, ckpt_path:str, dtype=dtypes.bfloat16):
  # bfloat16, not float16: gemma activations overflow the float16 range. cast on CPU so the gpu never holds the float32 copy
  sd = {k: v.to("CPU").cast(dtype) for k, v in convert_from_lerobot(nn.state.safe_load(ckpt_path), model.configs).items()}
  nn.state.load_state_dict(model, sd, strict=False)

def load_siglip(model:SigLIP, ckpt_path:str, dtype=dtypes.bfloat16):
  V = "paligemma_with_expert.paligemma.model.vision_tower.vision_model."
  sd = {k[len(V):].replace("encoder.layers.", "layers.").replace("embeddings.", ""): v.to("CPU").cast(dtype)
        for k, v in nn.state.safe_load(ckpt_path).items() if k.startswith(V)}
  nn.state.load_state_dict(model, sd)

def load_projector(proj:nn.Linear, ckpt_path:str, dtype=dtypes.bfloat16):
  # 1152 -> 2048: lifts siglip tokens into gemma space. image tokens are NOT sqrt-scaled like text (openpi convention)
  P = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear."
  w = nn.state.safe_load(ckpt_path)
  nn.state.load_state_dict(proj, {x: w[P+x].to("CPU").cast(dtype) for x in ("weight", "bias")})

def _normalize_lerobot_keys(weights:dict[str, Tensor]) -> dict[str, Tensor]:
  # newer lerobot saves prefix everything with "model." and drop the vision_model. path segment
  out = {}
  for k, v in weights.items():
    k = k.removeprefix("model.")
    if ".vision_tower." in k and ".vision_tower.vision_model." not in k:
      k = k.replace(".vision_tower.", ".vision_tower.vision_model.")
    out[k] = v
  return out

def convert_pi05_from_lerobot(weights:dict[str, Tensor], configs:tuple[dict, ...]=(GEMMA_2B, PI05_EXPERT)) -> dict[str, Tensor]:
  # full Pi05 state dict (llm./vision_tower./multi_modal_projector./action heads) from a lerobot checkpoint
  weights = _normalize_lerobot_keys(weights)
  sd = {f"llm.{k}": v for k, v in convert_from_lerobot(weights, configs).items()}
  V = "paligemma_with_expert.paligemma.model.vision_tower.vision_model."
  sd |= {"vision_tower." + k[len(V):].replace("encoder.layers.", "layers.").replace("embeddings.", ""): v
         for k, v in weights.items() if k.startswith(V)}
  P = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear."
  sd |= {f"multi_modal_projector.{x}": weights[P+x] for x in ("weight", "bias")}
  for name in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
    for x in ("weight", "bias"): sd[f"{name}.{x}"] = weights[f"{name}.{x}"]
  return sd

def load_pi05(model:Pi05, ckpt_path:str, dtype=dtypes.bfloat16, embedder_device:str|None=None):
  # embedder_device="CPU" keeps the 1GB embedding table off the gpu (embed/decode route through it transparently)
  if embedder_device is not None: model.llm.embedder.weight.replace(model.llm.embedder.weight.to(embedder_device))
  weights = nn.state.safe_load(ckpt_path)
  if "llm.embedder.weight" not in weights:  # raw lerobot checkpoint: remap keys and cast
    weights = {k: v.to("CPU").cast(dtype) for k, v in convert_pi05_from_lerobot(weights, model.llm.configs).items()}
  # a CPU-resident embedder stays float32: the clang backend can't do bf16 arithmetic (no __truncsfbf2)
  if embedder_device == "CPU": weights["llm.embedder.weight"] = weights["llm.embedder.weight"].to("CPU").float()
  nn.state.load_state_dict(model, weights)  # strict: every model tensor must come from the checkpoint

if __name__ == "__main__":
  # tiny two-expert model: expert 0 plays the VLM (prefix), expert 1 plays the action expert (suffix)
  Tensor.manual_seed(42)
  c0 = dict(width=64, depth=4, mlp_dim=128, num_heads=8, num_kv_heads=1, head_dim=16)
  c1 = dict(width=32, depth=4, mlp_dim=64,  num_heads=8, num_kv_heads=1, head_dim=16)
  model = Gemma((c0, c1), vocab_size=1000)

  B, P, S = 2, 6, 3
  prefix = model.embed(Tensor.randint(B, P, high=1000))
  suffix = Tensor.randn(B, S, 32)
  positions = Tensor.arange(P+S).reshape(1, P+S).expand(B, P+S)

  # pi0-style block mask: prefix is bidirectional within itself, suffix sees prefix + itself causally, prefix never sees suffix
  causal = Tensor.ones(P+S, P+S).tril()
  block = (Tensor.arange(P+S) < P).reshape(1, P+S) * (Tensor.arange(P+S) < P).reshape(P+S, 1)
  mask = ((causal + block) > 0).reshape(1, P+S, P+S).expand(B, P+S, P+S)

  out, _ = model([prefix, suffix], positions, mask)
  assert out[0].shape == (B, P, 64) and out[1].shape == (B, S, 32)

  # one-way rule: changing the suffix must not change the prefix output
  out2, _ = model([prefix, Tensor.randn(B, S, 32)], positions, mask)
  assert (out[0] - out2[0]).abs().max().item() == 0.0
  print("one-way rule: prefix untouched by suffix ✓")

  # causality within suffix: perturbing the last suffix token must not change earlier suffix outputs
  suffix2 = suffix.clone()
  suffix2[:, -1, :] = Tensor.randn(B, 32)
  out3, _ = model([prefix, suffix2], positions, mask)
  assert (out[1][:, :-1] - out3[1][:, :-1]).abs().max().item() < 1e-5   # only float rounding, no information leak
  assert (out[1][:, -1] - out3[1][:, -1]).abs().max().item() > 1e-2    # the perturbed token itself must change
  print("causal mask within suffix ✓")

  # kv cache: prefix once + cached suffix pass must equal the joint pass (this is the inference pattern)
  prefix_mask = mask[:, :P, :P]
  _, cache = model([prefix, None], positions[:, :P], prefix_mask)
  suffix_out, _ = model([None, suffix], positions[:, P:], mask[:, P:, :], kv_cache=cache)
  err = (out[1] - suffix_out[1]).abs().max().item()
  assert err < 1e-5, f"cache mismatch {err}"
  print(f"kv cache == joint pass (err {err:.2e}) ✓")

  # full-size param counts vs openpi (gemma_2b + gemma_300m, the comment in their gemma.py says 311M for the expert)
  n_total = sum(v.numel() for v in nn.state.get_state_dict(Gemma((GEMMA_2B, GEMMA_300M))).values())
  n_expert = n_total - sum(v.numel() for v in nn.state.get_state_dict(Gemma((GEMMA_2B,))).values())
  print(f"gemma_2b + expert: {n_total/1e9:.3f}B params total, action expert alone: {n_expert/1e6:.1f}M")
  assert abs(n_expert - 311.5e6) < 1e6 and abs(n_total - 2.82e9) < 0.01e9
  print("param counts match openpi ✓")

  # siglip: tiny config shape test, full-size param count (~412M, the "400M" in So400m)
  tiny = SigLIP(dim=32, depth=2, n_heads=4, mlp_dim=64, patch_size=14, image_size=56)
  assert tiny(Tensor.randn(2, 3, 56, 56)).shape == (2, 16, 32)
  n_vit = sum(v.numel() for v in nn.state.get_state_dict(SigLIP()).values())
  print(f"siglip So400m/14: {n_vit/1e6:.1f}M params")
  assert abs(n_vit - 412.4e6) < 1e6
  print("siglip shapes and param count ✓")

  # pi05 flow matching, tiny: the adarms dense layers are zero-init, so the expert starts as an identity network
  # and the sampled actions must not depend on the prefix at all. randomize them and the prefix must matter.
  tc1 = c1 | {"adarms": True}
  pi = Pi05((c0, tc1), action_dim=7, action_horizon=5, vision=SigLIP(dim=32, depth=2, n_heads=4, mlp_dim=64, patch_size=14, image_size=56))
  toks, noise = Tensor.randint(B, 4, high=1000), Tensor.randn(B, 5, 7)
  pre1 = pi.embed_prefix([Tensor.randn(B, 3, 56, 56)], toks)
  pre2 = pi.embed_prefix([Tensor.randn(B, 3, 56, 56)], toks)
  assert pre1.shape == (B, 16+4, 64)
  a1, a2 = pi.sample_actions(pre1, noise, num_steps=3), pi.sample_actions(pre2, noise, num_steps=3)
  assert a1.shape == (B, 5, 7)
  assert (a1 - a2).abs().max().item() == 0.0, "zero-init gates must block the prefix"
  print("flow sampling runs, zero-init expert ignores prefix ✓")
  for k, v in nn.state.get_state_dict(pi.llm).items():
    if ".dense." in k: v.assign(Tensor.randn(*v.shape) * 0.02).realize()  # assign, not replace: the jit captured these buffers
  a1, a2 = pi.sample_actions(pre1, noise, num_steps=3), pi.sample_actions(pre2, noise, num_steps=3)
  assert (a1 - a2).abs().max().item() > 1e-3, "with live gates the prefix must matter"
  print("adarms conditioning wired: prefix now steers the actions ✓")

  # padding: junk tokens behind a False mask must not change the actions (fixed prompt lengths keep jit shapes stable)
  padded = pre1.cat(Tensor.randn(B, 3, 64), dim=1)
  pmask = Tensor.ones(B, 20, dtype=dtypes.bool).cat(Tensor.zeros(B, 3, dtype=dtypes.bool), dim=1)
  a3 = pi.sample_actions(padded, noise, num_steps=3, prefix_mask=pmask)
  assert (a1 - a3).abs().max().item() < 1e-4, f"padded prefix diverged: {(a1 - a3).abs().max().item()}"
  print("padded prefix == unpadded prefix ✓")
