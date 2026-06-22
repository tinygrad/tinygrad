"""FLUX.2-klein Qwen3 text encoder in tinygrad.

The FLUX.2 prompt encoder is a Qwen3 causal LM run as a prefill encoder. The DiT's
`encoder_hidden_states` (dim 7680 = 3 * 2560) is the residual-stream hidden states
captured after layers 9, 18 and 27, concatenated on the feature axis. So only the
first 27 transformer layers need to run.

Mirrors mflux (MLX) `Flux2PromptEncoder` / `Qwen3TextEncoder`:
  - HF-style rotate_half RoPE (cos/sin from concat([freqs, freqs])), theta 1e6
  - per-head q/k RMSNorm over head_dim (Qwen3 specific)
  - GQA (32 query heads, 8 kv heads, head_dim 128)
  - causal attention + a padding mask (pad tokens get -inf), attention in float32
  - SwiGLU MLP, RMSNorm
  - hidden-state list = [embeddings] + [output of each layer]; capture indices 9/18/27

  PYTHONPATH=. DEV=METAL python -m extra.models.flux2.text_encoder
"""
from __future__ import annotations
import os
from tinygrad import Tensor, dtypes, nn
from tinygrad.helpers import getenv, fetch
from tinygrad.nn.state import safe_load, load_state_dict
from extra.models.flux2 import HF_BASE


# --- config (text_encoder/config.json) ---
VOCAB_SIZE = 151936
HIDDEN_SIZE = 2560
NUM_LAYERS = 36
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE = 9728
ROPE_THETA = 1000000.0
RMS_EPS = 1e-6
OUT_LAYERS = (9, 18, 27)
PAD_TOKEN_ID = 151643  # Qwen pad == bos; mflux pads input_ids to max_length with this


def qwen3_rotary_emb(seq_len: int, head_dim: int, theta: float = ROPE_THETA) -> tuple[Tensor, Tensor]:
  # HF-style: inv_freq over even dims, freqs = pos * inv_freq, emb = concat([freqs, freqs]).
  inv_freq = 1.0 / (theta ** (Tensor.arange(0, head_dim, 2, dtype=dtypes.float32) / head_dim))
  pos = Tensor.arange(seq_len, dtype=dtypes.float32).reshape(seq_len, 1)
  freqs = pos * inv_freq.reshape(1, -1)           # (seq_len, head_dim/2)
  emb = freqs.cat(freqs, dim=-1)                   # (seq_len, head_dim)
  return emb.cos(), emb.sin()


def rotate_half(x: Tensor) -> Tensor:
  half = x.shape[-1] // 2
  x1, x2 = x[..., :half], x[..., half:]
  return (-x2).cat(x1, dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
  # q,k: (b, n_heads, seq, head_dim); cos,sin: (seq, head_dim) -> broadcast over (b, heads)
  cos = cos.reshape(1, 1, *cos.shape)
  sin = sin.reshape(1, 1, *sin.shape)
  q_embed = (q * cos) + (rotate_half(q) * sin)
  k_embed = (k * cos) + (rotate_half(k) * sin)
  return q_embed, k_embed


class Qwen3Attention:
  def __init__(self):
    self.q_proj = nn.Linear(HIDDEN_SIZE, NUM_HEADS * HEAD_DIM, bias=False)
    self.k_proj = nn.Linear(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False)
    self.v_proj = nn.Linear(HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False)
    self.o_proj = nn.Linear(NUM_HEADS * HEAD_DIM, HIDDEN_SIZE, bias=False)
    self.q_norm = nn.RMSNorm(HEAD_DIM, RMS_EPS)  # per-head RMSNorm over head_dim
    self.k_norm = nn.RMSNorm(HEAD_DIM, RMS_EPS)

  def __call__(self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor) -> Tensor:
    bsz, q_len, _ = x.shape
    q = self.q_proj(x).reshape(bsz, q_len, NUM_HEADS, HEAD_DIM)
    k = self.k_proj(x).reshape(bsz, q_len, NUM_KV_HEADS, HEAD_DIM)
    v = self.v_proj(x).reshape(bsz, q_len, NUM_KV_HEADS, HEAD_DIM)

    q = self.q_norm(q)  # normalize over the last (head_dim) axis, before transpose
    k = self.k_norm(k)

    q = q.transpose(1, 2)  # (b, n_heads, seq, head_dim)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    # mflux runs attention in float32; mask is an additive float32 (causal + padding) bias.
    attn = q.float().scaled_dot_product_attention(k.float(), v.float(), attn_mask=mask, enable_gqa=True)
    attn = attn.cast(x.dtype).transpose(1, 2).reshape(bsz, q_len, NUM_HEADS * HEAD_DIM)
    return self.o_proj(attn)


class Qwen3MLP:
  def __init__(self):
    self.gate_proj = nn.Linear(HIDDEN_SIZE, INTERMEDIATE, bias=False)
    self.up_proj = nn.Linear(HIDDEN_SIZE, INTERMEDIATE, bias=False)
    self.down_proj = nn.Linear(INTERMEDIATE, HIDDEN_SIZE, bias=False)

  def __call__(self, x: Tensor) -> Tensor:
    return self.down_proj(self.gate_proj(x).silu() * self.up_proj(x))


class Qwen3DecoderLayer:
  def __init__(self):
    self.input_layernorm = nn.RMSNorm(HIDDEN_SIZE, RMS_EPS)
    self.self_attn = Qwen3Attention()
    self.post_attention_layernorm = nn.RMSNorm(HIDDEN_SIZE, RMS_EPS)
    self.mlp = Qwen3MLP()

  def __call__(self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor) -> Tensor:
    x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask)
    x = x + self.mlp(self.post_attention_layernorm(x))
    return x


class Qwen3TextEncoder:
  def __init__(self, n_layers: int = NUM_LAYERS, out_layers: tuple[int, ...] = OUT_LAYERS):
    # only run as deep as the deepest captured layer (default 27 of 36)
    self.n_run = max(out_layers)
    self.out_layers = out_layers
    self.embed_tokens = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
    self.layers = [Qwen3DecoderLayer() for _ in range(n_layers)]
    self.norm = nn.RMSNorm(HIDDEN_SIZE, RMS_EPS)  # final norm (unused for captured states)

  def get_prompt_embeds(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
    bsz, seq_len = input_ids.shape
    h = self.embed_tokens(input_ids)

    cos, sin = qwen3_rotary_emb(seq_len, HEAD_DIM)
    cos, sin = cos.cast(h.dtype), sin.cast(h.dtype)

    # additive float32 mask: causal (upper triangle -inf) + padding (pad cols -inf)
    neg = -float("inf")
    causal = Tensor.full((seq_len, seq_len), neg, dtype=dtypes.float32).triu(1)
    mask = causal.reshape(1, 1, seq_len, seq_len)
    if attention_mask is not None:
      pad = (attention_mask.reshape(bsz, 1, 1, seq_len) == 0).where(neg, 0.0).cast(dtypes.float32)
      mask = mask + pad

    captured = [h]  # mflux: hidden_states_list[0] is the embedding output
    for i, layer in enumerate(self.layers):
      h = layer(h, cos, sin, mask)
      captured.append(h)
      if i + 1 == self.n_run: break

    # stack captured layer outputs on a new axis, reorder to (b, seq, n_layers, hidden), flatten feature
    picks = [captured[i] for i in self.out_layers]
    stacked = Tensor.stack(*picks, dim=1)                       # (b, n_layers, seq, hidden)
    prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(bsz, seq_len, len(self.out_layers) * HIDDEN_SIZE)
    return prompt_embeds


def prepare_text_ids(seq_len: int, bsz: int = 1) -> Tensor:
  # mflux prepare_text_ids: coords (t,h,w,token_id) with t=h=w=0, token_id=arange(seq_len)
  z = Tensor.zeros(seq_len, dtype=dtypes.int32)
  tid = Tensor.arange(seq_len, dtype=dtypes.int32)
  coords = Tensor.stack(z, z, z, tid, dim=1)  # (seq_len, 4)
  return coords.reshape(1, seq_len, 4).expand(bsz, seq_len, 4)


# 2-shard text encoder; fetch downloads on first use (symlink the local cache into the fetch dir to reuse it).
TEXT_ENCODER_SHARDS = tuple(HF_BASE + f"text_encoder/model-{i:05d}-of-00002.safetensors" for i in (1, 2))

def load_text_encoder(n_layers: int = NUM_LAYERS) -> Qwen3TextEncoder:
  model = Qwen3TextEncoder(n_layers=n_layers)
  sd = {}
  for url in TEXT_ENCODER_SHARDS:
    sd.update(safe_load(fetch(url, url.rsplit("/", 1)[1], subdir="flux2-klein-4b/text_encoder")))
  # HF key -> module path: drop "model." prefix; lm_head not needed (tied, and we stop at layer 27)
  remap = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
  load_state_dict(model, remap, strict=False, consume=True)
  return model


def encode_prompt_from_ids(model: Qwen3TextEncoder, input_ids: Tensor, attention_mask: Tensor | None = None):
  """input_ids: (1, N) int32. Returns (prompt_embeds (1,N,7680), text_ids (1,N,4))."""
  prompt_embeds = model.get_prompt_embeds(input_ids, attention_mask)
  text_ids = prepare_text_ids(input_ids.shape[1], input_ids.shape[0])
  return prompt_embeds, text_ids


if __name__ == "__main__":
  from tinygrad import Device, GlobalCounters
  ref_path = getenv("FLUX2_QWEN_REF", "/tmp/flux2_qwen_ref.safetensors")
  ref = {k: v.to(Device.DEFAULT) for k, v in safe_load(ref_path).items()}
  input_ids = ref["input_ids"].cast(dtypes.int32)
  attention_mask = ref["attention_mask"].cast(dtypes.int32)

  model = load_text_encoder()
  GlobalCounters.reset()
  embeds, text_ids = encode_prompt_from_ids(model, input_ids, attention_mask)
  embeds = embeds.realize()
  print(f"device {Device.DEFAULT}  prompt_embeds {embeds.shape}  kernels {GlobalCounters.kernel_count}")

  out = embeds.float()
  refp = ref["prompt_embeds"].float()
  diff = (out - refp).abs()
  a, b = out.reshape(-1), refp.reshape(-1)
  cos = (a * b).sum() / (a.square().sum().sqrt() * b.square().sum().sqrt())
  print(f"vs mflux: cosine {cos.item():.6f}  max {diff.max().item():.5f}  mean {diff.mean().item():.6f}")
