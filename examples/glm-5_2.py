# Using GLM-5.2 weights from https://huggingface.co/zai-org/GLM-5.2-FP8
# Using the https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py as source of truth

from math import ceil
from typing import Literal
from dataclasses import dataclass
import functools, argparse, jinja2, json
from tinygrad import fetch, Tensor, nn, dtypes, Device

def device_to_load(device: tuple[str, ...], num: int) -> str:
  return device[num % len(device)]

# Copying this over from llama cause weight_scale is a little different
class FP8Linear:
  def __init__(self, in_features, out_features, bias=True):
    self.block_size, self.out_features, self.in_features = 128, out_features, in_features
    self.weight = Tensor.empty(out_features, in_features, dtype=dtypes.fp8e4m3)
    self.bias = Tensor.empty(out_features, dtype=dtypes.float16) if bias else None
    self.weight_scale_inv = Tensor.empty((ceil(out_features/self.block_size), ceil(in_features/self.block_size)), dtype=dtypes.float16)

  def __call__(self, x:Tensor) -> Tensor:
    weight_scale_inv = self.weight_scale_inv.unsqueeze(1).unsqueeze(-1)
    out_blocks, in_blocks = ceil(self.out_features/self.block_size), ceil(self.in_features/self.block_size)
    weight_scale_inv = weight_scale_inv.expand(out_blocks, self.block_size, in_blocks, self.block_size)
    weight_scale_inv = weight_scale_inv.reshape(out_blocks * self.block_size, in_blocks * self.block_size)[:self.out_features, :self.in_features]
    weight = self.weight.cast(dtypes.float32) * weight_scale_inv.cast(dtypes.float32)

    y = x.dot(weight.T)
    if self.bias is not None: y = y + self.bias.cast(y.dtype)
    return y.cast(x.dtype)

class GLMTokenizer:
  def __init__(self, token_config: str):
    from tokenizers import Tokenizer
    self.model = Tokenizer.from_file(token_config)

  @property
  def bos_id(self): return self.model.encode("[gMASK]<bos>").ids[0]
  @property
  def stop_tokens(self): return [154820, 154827, 154829] # got the values from the config
  @property
  def pad_token(self): return 154820 # pad_id got from config on hf
  def decode(self, toks): return self.model.decode(toks)
  def encode(self, text, allow_special=False):
    return self.model.encode(text, add_special_tokens=allow_special)

LayerType = Literal["dense", "sparse"]
IndexerType = Literal["full", "shared"]
@dataclass(frozen=True)
class GLMConfig:
  max_batch: int
  dim: int
  layers: list[LayerType]
  vocab_size: int
  max_context: int
  intermediate_size: int
  num_exp: int
  num_exp_per_tok: int
  routed_scaling_fact: float
  moe_intermediate_size: int
  attn_dim: int
  attn_heads: int
  q_latent_dim: int
  kv_latent_dim: int
  norm_eps: float
  idx_attn_dim: int
  idx_heads: int
  idx_topk: int
  indexers: list[IndexerType]
  attn_rope_dim: int
  rope_theta: float

# KV cache Sparse MLA(DSA)
class DSAKVCache:
  def __init__(self, layers: int, indexers: list[IndexerType], max_context: int, latent_dim: int, idx_latent_dim: int, max_batch: int):
    self.layers, self.max_context, self.attn_latent_dim = layers, max_context, latent_dim
    self.idx_k_cache_dim, self.max_batch, self.indexer_type = idx_latent_dim, max_batch, indexers
    self.full_to_slot = {i: s  for s, i in enumerate([i for i, v in enumerate(indexers) if v == "full"])}
    self.store_attn = Tensor.empty(layers, max_batch, max_context, latent_dim, dtype=dtypes.bfloat16)
    self.store_attn_idx = Tensor.empty(indexers.count("full"), max_batch, max_context, idx_latent_dim, dtype=dtypes.bfloat16)

  CacheSegment = Literal["attention", "indexer"]
  # This expects the shape (B, T, d)
  def update_cache(self, layer: int, positional_ids: Tensor, value: Tensor, segment: CacheSegment) -> Tensor:
    B, T, d = value.shape
    assert 1 <= B <= self.max_batch, "batch size is greater than max batch"
    assert 1 <= T <= self.max_context, "tokens more than max context"
    assert layer < self.layers, "layer is greater than max_layer"
    assert (segment == "attention" and d == self.attn_latent_dim) or \
           (segment == "indexer" and d == self.idx_k_cache_dim), "shape of dim is incorrect"
    to_write = (positional_ids >= 0).sum(-1) # (B)
    padded_values = (positional_ids < 0).sum(-1) # (B)
    for b in range(B):
      writes = int(to_write[b].item())
      start_point = int(positional_ids[b, int(padded_values[b].item())].item())
      if segment == "attention":
        self.store_attn[layer, b, start_point:start_point+writes].assign(value[b, -writes:])
      else:
        self.store_attn_idx[self.full_to_slot[layer], b, start_point:start_point+writes].assign(value[b, -writes:])
    if segment == "attention": return self.store_attn[layer, :B, :]
    else: return self.store_attn_idx[self.full_to_slot[layer], :B, :]

@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float, device:str|None=None) -> Tensor:
  assert dim % 2 == 0, "dim must be even for ROPE"
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2) / dim))
  freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0) # (end, 1) * (1, dim//2) -> (end, dim//2)
  return freqs.cos().cat(freqs.sin(), dim=-1).clone(device) # (end, dim)

# This does interleaved ROPE, which is slightly different from the half-split(llama) version
# x: (B, S, H, HD)
def apply_rope(x:Tensor, freqs_cis:Tensor, positional_ids: Tensor) -> Tensor:
  assert x.shape[-1] % 2 == 0
  valid = positional_ids >= 0 # (B, S)
  idx = valid.where(positional_ids, 0)
  batched_freqs = freqs_cis[idx].cast(dtypes.bfloat16) # Taking slice for each token from freqs
  cos, sin = batched_freqs.reshape(x.shape[0], x.shape[1], 1, -1).chunk(2, dim=-1) # (B, S, 1, dim//2)
  cos, sin = valid.unsqueeze(-2).unsqueeze(-1).where(cos, 1), valid.unsqueeze(-2).unsqueeze(-1).where(sin, 0)
  x1, x2 = x[..., 0::2], x[..., 1::2]
  return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)

class GLMDSAIndexer():
  def __init__(self, layer:int, config:GLMConfig):
    self.idx_num_heads, self.idx_head_dim, self.attn_rope_dim = config.idx_heads, config.idx_attn_dim, config.attn_rope_dim
    self.layer, self.idx_topk = layer, config.idx_topk
    self.q_up_proj_idx = FP8Linear(config.q_latent_dim, self.idx_num_heads * self.idx_head_dim, bias=False)
    self.wk = FP8Linear(config.dim, self.idx_head_dim, bias=False)
    self.k_norm = nn.LayerNorm(self.idx_head_dim, eps=1e-6)
    self.weights_proj = nn.Linear(config.dim, self.idx_num_heads, bias=False)
    self.softmax_scale = self.idx_head_dim ** -0.5

  def __call__(self, x: Tensor, q_latent: Tensor, position_embeddings: Tensor, kv_cache: DSAKVCache, positional_ids: Tensor, attn_mask: Tensor) -> Tensor:
    assert x.shape[0] == q_latent.shape[0], "error in batch dim"
    q_idx = self.q_up_proj_idx(q_latent) # (B, S, H*HD)
    q_idx = q_idx.reshape(*q_idx.shape[:-1], self.idx_num_heads, self.idx_head_dim) # (B, S, H, HD)
    q_rot_idx, q_pass_idx = q_idx[..., :self.attn_rope_dim], q_idx[..., self.attn_rope_dim:]

    k_idx = self.k_norm(self.wk(x)).unsqueeze(2) # (B, S, HD) -> (B, S, 1, HD)
    k_rot_idx, k_pass_idx = k_idx[..., :self.attn_rope_dim], k_idx[...,self.attn_rope_dim:]

    q_rot_idx, k_rot_idx = apply_rope(q_rot_idx, position_embeddings, positional_ids), apply_rope(k_rot_idx, position_embeddings, positional_ids)
    q_idx, k_idx = q_rot_idx.cat(q_pass_idx, dim=-1), k_rot_idx.cat(k_pass_idx, dim=-1).squeeze(2) # q_idx: (B, S, H, HD) k_idx: (B, S, HD)

    # (B, S_longest, HD)
    k_idx = kv_cache.update_cache(self.layer, positional_ids, k_idx, "indexer") # This will return values that are padded to the max sequnce of all batches

    # (B, S, H, HD) @ (B, 1, HD, T) -> (B, S, H, T)
    scores = q_idx.float() @ k_idx.transpose(-1, -2).float().unsqueeze(1)
    scores: Tensor = scores * (self.idx_head_dim ** -0.5)
    scores = scores.relu()

    # weights per head (B, S, idx_num_heads)
    weights = self.weights_proj(x.cast(self.weights_proj.weight.dtype)).float() * (self.idx_num_heads ** -0.5)
    index_scores = weights.unsqueeze(-2) @ scores # (B, S, 1, H) @ (B, S, H, T) => (B, S, 1, T)
    index_scores = index_scores.squeeze(-2) # (B, S, T)

    index_scores = index_scores + attn_mask

    topk = int(min(self.idx_topk, index_scores.shape[-1]))
    return index_scores.topk(topk, dim=-1)[1].cast(dtypes.int32)

def create_causal_attn_mask(positional_ids: Tensor, longest_seq: int) -> Tensor:
  valid_queries = positional_ids >= 0
  mask = Tensor.arange(longest_seq).unsqueeze(0).unsqueeze(1).expand(*positional_ids.shape, longest_seq) # (B, S, longest_seq)
  key_ok = mask <= positional_ids.unsqueeze(-1) # (B, S, longest_seq) < (B, S, 1) => (B, S, longest_deq)
  keep = key_ok & valid_queries.unsqueeze(-1) # (B, S, longest_seq)
  return keep.where(0, dtypes.float.min)

def basic_attn(queries: Tensor, keys: Tensor, values: Tensor, attn_mask: Tensor, attn_heads: int, attn_dim: int, batch: int):
  attn_weights = ((queries @ keys.transpose(-1, -2)) * attn_dim ** -0.5) + attn_mask.unsqueeze(1) # (B, attn_head, S_inp, S_longest)
  attn_weights = attn_weights.softmax(-1) # (B, attn_head, S_inp, S_longest)
  attn_output = (attn_weights @ values).permute(0, 2, 1, 3) # (B, S_inp, attn_head, attn_dim)
  attn_output = attn_output.reshape(batch, -1, attn_heads * attn_dim).contiguous() # (B, S_inp, attn_head * attn_dim)
  return attn_weights, attn_output

def perform_attn(queries: Tensor, keys: Tensor, values: Tensor, attn_mask: Tensor, topk_idx: Tensor, attn_heads: int, attn_dim: int) -> tuple[Tensor, Tensor]:
  batch, queries, keys, values = queries.shape[0], queries.permute(0, 2, 1, 3), keys.permute(0, 2, 1, 3), values.permute(0, 2, 1, 3)
  topk_idx_inv = Tensor.ones_like(attn_mask).scatter(-1, topk_idx, 0)
  attn_mask = attn_mask.masked_fill(topk_idx_inv.bool(), dtypes.float.min) # (B, S_inp, S_longest)
  return basic_attn(queries, keys, values, attn_mask, attn_heads, attn_dim, int(batch))

class GLMAttention():
  def __init__(self, config: GLMConfig, layer: int):
    self.attn_dim, self.attn_rope_dim, self.kv_latent_dim, self.attn_heads, self.layer = config.attn_dim, config.attn_rope_dim, config.kv_latent_dim, config.attn_heads, layer
    self.q_down_proj = FP8Linear(config.dim, config.q_latent_dim, bias=False)
    self.q_norm = nn.RMSNorm(config.q_latent_dim)
    self.q_up_proj = FP8Linear(config.q_latent_dim, config.attn_heads * config.attn_dim, bias=False)

    self.kv_down_proj = FP8Linear(config.dim, config.kv_latent_dim + config.attn_rope_dim, bias=False)
    self.kv_norm = nn.RMSNorm(config.kv_latent_dim)
    self.kv_up_proj = FP8Linear(config.kv_latent_dim, config.attn_heads * ((config.attn_dim - config.attn_rope_dim) + config.attn_dim), bias=False)

    self.o_proj = FP8Linear(config.attn_heads * config.attn_dim, config.dim, bias=False)

    self.indexer = GLMDSAIndexer(layer, config) if config.indexers[layer] == "full" else None

  # x: (B, S, dim)
  # TODO: precompute the up proj single matrix for k, q, v for MLA inference optimization
  def __call__(self, x: Tensor, position_embeddings: Tensor, positional_ids: Tensor, kv_cache: DSAKVCache, prev_idx_topk: Tensor|None):
    batch, orig_seq = x.shape[:-1]
    q_latent = self.q_norm(self.q_down_proj(x)) # (B, S, q_lat)
    q_pass, q_rot = self.q_up_proj(q_latent).reshape(batch, orig_seq, -1, self.attn_dim).split([self.attn_dim - self.attn_rope_dim, self.attn_rope_dim], dim=-1)

    cached_kv = self.kv_down_proj(x)
    kv_common, k_pre_rot= cached_kv.split([self.kv_latent_dim, self.attn_rope_dim], dim=-1) # kv_common: (B, S_inp, kv_latent_dim) k_pre_rot: (B, S_inp, attn_rope_dom)
    k_pre_rot = k_pre_rot.reshape(batch, orig_seq, 1, self.attn_rope_dim)
    q_rot, k_rot = apply_rope(q_rot, position_embeddings, positional_ids), apply_rope(k_pre_rot, position_embeddings, positional_ids)

    kv_common, k_rot = kv_cache.update_cache(self.layer, positional_ids, kv_common.cat(k_rot.squeeze(2), dim=-1), "attention").split([self.kv_latent_dim, self.attn_rope_dim], dim=-1)
    longest_seq = kv_common.shape[1] # Now this becomes the longest seq in this set of batches that came from kv_cache
    k_rot = k_rot.unsqueeze(2)

    # k_pass: (B, S_longest, attn_dim - attn_rope_dim) values: (B, S_longest, attn_dim)
    k_pass, values = self.kv_up_proj(self.kv_norm(kv_common)).reshape(batch, longest_seq, self.attn_heads, -1).split([self.attn_dim - self.attn_rope_dim, self.attn_dim], dim=-1)
    k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

    queries = q_pass.cat(q_rot, dim=-1) # (B, S_inp, attn_head, attn_dim)
    keys = k_pass.cat(k_rot, dim=-1) # (B, S_longest, attn_head, attn_dim)

    # creating the attention mask after kv cache fills in old keys as well
    attn_mask = create_causal_attn_mask(positional_ids, int(longest_seq)) # (B, S_inp, S_longest)

    if self.indexer is not None:
      topk_idx = self.indexer(x, q_latent, position_embeddings, kv_cache, positional_ids, attn_mask) # (B, S_inp, K)
    else:
      assert prev_idx_topk is not None, "previous idx topk should come for shared idx layers"
      topk_idx = prev_idx_topk

    attn_weights, attn_output = perform_attn(queries, keys, values, attn_mask, topk_idx, self.attn_heads, self.attn_dim)
    attn_ouput_up_proj = self.o_proj(attn_output) # (B, S_inp, dim)
    return attn_ouput_up_proj, attn_weights, topk_idx

class GLMMoeGateTopK():
  def __init__(self, config: GLMConfig):
    # NOTE: we have only one group in GLM so not taking that into consideration
    self.num_exp, self.num_exp_per_token, self.dim = config.num_exp ,config.num_exp_per_tok, config.dim
    self.routed_scaling_fact = config.routed_scaling_fact
    self.weight = Tensor.empty(self.num_exp, self.dim)
    self.e_score_correction_bias = Tensor.empty(self.num_exp)

  # x: (B, S, dim)
  def __call__(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    router_logits = x @ self.weight.transpose() # (B, S, dim) @ (dim, E) => (B, S, E)
    scores = router_logits.sigmoid() # (B, S, E)
    scores_for_choice = scores + self.e_score_correction_bias # (B, S, E)
    topk_idx = scores_for_choice.topk(self.num_exp_per_token, -1)[1] # (B, S, K)
    topk_weights = scores.gather(-1,topk_idx)

    # Normalizing the weights
    denominator = topk_weights.sum(-1, keepdim=True)
    topk_weights = topk_weights / (denominator + 1e-20)
    topk_weights = topk_weights * self.routed_scaling_fact

    return router_logits, topk_weights, topk_idx

class GLMMoeExpert():
  def __init__(self, config: GLMConfig):
    self.num_experts = config.num_exp
    self.up_proj = FP8Linear(config.dim, config.moe_intermediate_size, bias=False)
    self.gate_proj = FP8Linear(config.dim, config.moe_intermediate_size, bias=False)
    self.down_proj =  FP8Linear(config.moe_intermediate_size, config.dim, bias=False)

  # This call expects that the inputs and weights are already on the right device shard
  # x : (N, dim) w: (N)
  def __call__(self, x: Tensor, w: Tensor) -> Tensor:
    gate = self.gate_proj(x) # (N, I)
    up = self.up_proj(x) # (N, I)
    x = (gate.silu() * up).contiguous() # (N, I)
    x = self.down_proj(x) # (N, I) @ (I, dim) => (N, dim)
    return (x * w.unsqueeze(-1)) # (N, dim)

class GLMMoeLayer():
  def __init__(self, config: GLMConfig, devices: tuple[str, ...]):
    self.num_exp, self.devices = config.num_exp, devices
    self.experts = [GLMMoeExpert(config) for _ in range(self.num_exp)]
    self.gate = GLMMoeGateTopK(config)
    self.shared_experts = GLMMLPLayer(config, config.moe_intermediate_size)

  # x: (B, S, dim)
  def __call__(self, x: Tensor) -> Tensor:
    residual = x
    _, topk_weights, topk_idx = self.gate(x)

    default_device = self.devices[0]
    exp_arange = Tensor.zeros(*topk_idx.shape[:-1], self.num_exp) # (B, S, num_exp)
    exp_map = exp_arange.scatter(-1, topk_idx, 1).bool().permute(2, 0, 1) # (num_exp, B, S)
    output = Tensor.zeros_like(x)

    for exp in range(self.num_exp):
      if not exp_map[exp].any().item(): continue
      shard_device = device_to_load(self.devices, exp)
      coords = exp_map[exp].nonzero()
      b, s = coords[:, 0], coords[:, 1]
      x_exp = x[b, s].to(shard_device) # (num_true, dim)
      w = ((topk_idx[b, s] == exp) * topk_weights[b,s]).sum(-1).to(shard_device) # (num_true)
      output[b,s] += self.experts[exp](x_exp, w).to(default_device)

    x = output + self.shared_experts(residual)
    return x # (B, S, dim)

class GLMMLPLayer():
  def __init__(self, config: GLMConfig, intermediate_size: int|None):
    intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
    self.gate_proj = FP8Linear(config.dim, intermediate_size, bias=False)
    self.up_proj = FP8Linear(config.dim, intermediate_size, bias=False)
    self.down_proj = FP8Linear(intermediate_size, config.dim, bias=False)

  def __call__(self, x: Tensor) -> Tensor:
    return self.down_proj(self.gate_proj(x).silu() * self.up_proj(x))

class GLMBlock():
  def __init__(self, config: GLMConfig, layer: int, devices: tuple[str, ...]):
    self.input_layernorm = nn.RMSNorm(config.dim, config.norm_eps)
    self.self_attn = GLMAttention(config, layer)
    if config.layers[layer] == "dense":
      self.mlp = GLMMLPLayer(config, None)
    else:
      self.mlp = GLMMoeLayer(config, devices)
    self.post_attention_layernorm = nn.RMSNorm(config.dim, config.norm_eps)

  def __call__(self, x: Tensor, position_embeddings: Tensor, positional_ids: Tensor, kv_cache: DSAKVCache, prev_idx_topk: Tensor|None) -> tuple[Tensor, Tensor]:
    residual = x
    x = self.input_layernorm(x)
    x, _, topk_idx = self.self_attn(x, position_embeddings, positional_ids, kv_cache, prev_idx_topk)
    x = x + residual

    residual = x
    x = self.post_attention_layernorm(x)
    x = self.mlp(x)
    x = x + residual
    return x, topk_idx

class GLMModel():
  def __init__(self, config: GLMConfig, devices: tuple[str, ...]):
    self.max_context, self.rope_theta, self.attn_rope_dim, self.max_batch = config.max_context, config.rope_theta, config.attn_rope_dim, config.max_batch
    self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)
    self.layers = [GLMBlock(config, i, devices) for i in range(len(config.layers))]
    self.norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

  def __call__(self, input_toks: Tensor, positional_ids: Tensor, kv_cache: DSAKVCache) -> Tensor:
    x = self.embed_tokens(input_toks)
    batch = input_toks.shape[0]
    assert batch <= self.max_batch, "batch size is bigger than max_batch"
    position_embeddings = precompute_freqs_cis(self.attn_rope_dim, self.max_context, self.rope_theta)

    topk_idx = None
    for block in self.layers: x, topk_idx = block(x, position_embeddings, positional_ids, kv_cache, topk_idx)

    x = self.norm(x)
    # TODO: sample with temperature
    return self.lm_head(x[:, -1, :]).argmax(axis=-1) # (B)

def prompt_jinja_render(prompts: list) -> list[str]:
  return [template.render(p) for p in prompts]

def prefill_encode_and_pad(tokenizer: GLMTokenizer, prompts: list) -> tuple[Tensor, Tensor]:
  rendered_prompts = prompt_jinja_render(prompts)
  encoded = [tokenizer.encode(r).ids for r in rendered_prompts]
  longest_prompt = max([len(e) for e in encoded])
  # Padding is done on the left according to the config
  input_tensor = Tensor.stack(*[Tensor(e).pad((longest_prompt - len(e), 0), value=tokenizer.pad_token) for e in encoded])
  positional_ids = Tensor.stack(*[Tensor.arange(len(e)).pad((longest_prompt - len(e), 0), value=-1) for e in encoded])
  return input_tensor, positional_ids

def decode_tokens(tokenizer: GLMTokenizer, outputs: Tensor) -> tuple[list[int], list[str]]:
  ids = [int(o.item()) for o in outputs]
  return ids, [tokenizer.decode([i]) for i in ids]

# NOTE: just to test on smaller gpu clusters
testing_start_layer = 0
testing_end_layer = 15

indexer_types:list[IndexerType] = ["full", "full", "full", "shared", "shared", "shared", "full", "shared", "shared", "shared", "full", "shared", "shared", "shared", "full", "shared", "shared", "shared", "full", "shared", "shared", "shared", "full", "shared", "shared", "shared", "full", "shared", "shared", "shared", "full", "shared", "shared", "shared", "full", "shared", "shared", "shared", "full", "shared", "shared", "shared", "full", "shared", "shared", "shared", "full", "shared", "shared", "shared", "full", "shared", "shared", "shared", "full", "shared", "shared", "shared", "full", "shared", "shared", "shared", "full", "shared", "shared", "shared", "full", "shared", "shared", "shared", "full", "shared", "shared", "shared", "full", "shared", "shared", "shared"]
layer_types: list[LayerType] =  ["dense", "dense", "dense", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse"]
config = GLMConfig (
  max_batch = 10,
  dim = 6_144,
  layers = layer_types[testing_start_layer:testing_end_layer],
  norm_eps = 1e-05,
  vocab_size = 154_880,
  max_context = 1_000,
  intermediate_size = 12_288,
  num_exp = 256,
  num_exp_per_tok = 8,
  routed_scaling_fact = 2.5,
  moe_intermediate_size = 2_048,
  attn_dim = 256,
  attn_heads = 64,
  q_latent_dim = 2_048,
  kv_latent_dim = 512,
  idx_attn_dim = 128,
  idx_heads = 32,
  idx_topk = 2_048,
  indexers = indexer_types[testing_start_layer: testing_end_layer],
  attn_rope_dim = 64,
  rope_theta = 8_000_000
)

# This has total 141 files each approx 5.3 GB ~= 750 GB
def load_model(model: GLMModel,devices: tuple[str, ...]):

  def replace_with_pat(k: str) -> str:
    replace_map: dict = { "model.": "", "wq_b.": "q_up_proj_idx.", "kv_a_proj_with_mqa.": "kv_down_proj.", "kv_b_proj.": "kv_up_proj.", "q_a_proj.": "q_down_proj.", "q_b_proj.": "q_up_proj.", "kv_a_layernorm.": "kv_norm.", "q_a_layernorm.": "q_norm." }
    for pat, rep in replace_map.items(): k = k.replace(pat, rep)
    return k

  model_state_dict = nn.state.get_state_dict(model)
  model_index = json.loads(fetch("https://huggingface.co/zai-org/GLM-5.2-FP8/resolve/main/model.safetensors.index.json").read_text())['weight_map']
  wanted_files = {model_index[k] for k in model_index.keys() if replace_with_pat(k) in model_state_dict}

  model_tensors: dict[str, Tensor] = {}
  for file in wanted_files: model_tensors.update(nn.state.safe_load(fetch(f"https://huggingface.co/zai-org/GLM-5.2-FP8/resolve/main/{file}")))

  for k,v in model_tensors.items():
    k = replace_with_pat(k)
    if k not in model_state_dict: continue
    if "layers.78" in k: continue # layer 78 is the MTP layer
    k_comps = k.split('.')
    if len(k_comps) >= 5 and k_comps[3] == "experts":
      exp_num = int(k_comps[4])
      device = device_to_load(devices, exp_num)
    else: device = device_to_load(devices, 0)
    model_state_dict[k].replace(v.to(device)).realize()
  return model

def prefill(tokenizer: GLMTokenizer, prompts: list, model: GLMModel, kv_cache: DSAKVCache) -> tuple[Tensor, Tensor, Tensor, list[bool]]:
  input_tokens, positional_ids = prefill_encode_and_pad(tokenizer, prompts)
  batches = [True for _ in range(input_tokens.shape[0])]
  outputs = model(input_tokens, positional_ids, kv_cache)
  return input_tokens, outputs, (positional_ids[:, -1] + 1), batches

def decode(input_tokens: Tensor, positional_ids: Tensor, kv_cache: DSAKVCache) -> tuple[Tensor, Tensor, Tensor]:
  outputs = model(input_tokens, positional_ids, kv_cache)
  positional_ids = positional_ids + 1
  positional_ids = positional_ids.reshape(len(batches), 1)
  input_tokens = outputs.reshape(len(batches), 1)
  return input_tokens, positional_ids, outputs

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--shard", type=int, default=1, help="Shard the model across multiple devices")
  args = parser.parse_args()

  devices = tuple(f"{Device.DEFAULT}:{i}" for i in range(args.shard)) if args.shard > 1 else (Device.DEFAULT, )

  tokenizer = GLMTokenizer(str(fetch("https://huggingface.co/zai-org/GLM-5.2-FP8/resolve/main/tokenizer.json")))
  kv_cache = DSAKVCache(len(config.layers), config.indexers, config.max_context, config.kv_latent_dim + config.attn_rope_dim, config.idx_attn_dim, config.max_batch)
  model = GLMModel(config, devices)
  loaded_model = load_model(model, devices)

  template = jinja2.Template(str(fetch("https://huggingface.co/zai-org/GLM-5.2-FP8/resolve/main/chat_template.jinja").read_text()))
  prompt_one = {
    "reasoning_effort": "high",
    "tools": None,
    "messages": [
        {
          "role": "system", 
          "content": "You are an helpful assistant.",
        },
        {
          "role": "user",
          "content": "What is the capital of India?"
        }
    ],
    "add_generation_prompt": True
  }

  prompt_two = {
    "reasoning_effort": "high",
    "tools": None,
    "messages": [
        {
          "role": "system", 
          "content": "You are an helpful assistant.",
        },
        {
          "role": "user",
          "content": "What is the sum of 2 + 2?"
        }
    ],
    "add_generation_prompt": True
  }

  prompt_three = {
    "reasoning_effort": "high",
    "tools": None,
    "messages": [
        {
          "role": "system", 
          "content": "You are an helpful assistant.",
        },
        {
          "role": "user",
          "content": "Can you give me a short essay on United States of America?"
        }
    ],
    "add_generation_prompt": True
  }

  input_tokens, outputs, positional_ids, batches = prefill(tokenizer, [prompt_one, prompt_two, prompt_three], model, kv_cache)

  while True:
    input_tokens, positional_ids, outputs = decode(input_tokens, positional_ids, kv_cache)
    decoded_ids, decoded_strs = decode_tokens(tokenizer, outputs)
    for b, (id, tok) in enumerate(zip(decoded_ids, decoded_strs)):
      if not batches[b]: continue
      if id in tokenizer.stop_tokens:
        batches[b] = False
      print(f"batch: {b} -> {tok}")

    if batches.count(True) == 0: print("All batches done"); break
    elif (positional_ids[:, -1].max().item()) >= config.max_context: raise RuntimeError("max context reached")
