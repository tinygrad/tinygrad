# Using GLM-5.2 weights from https://huggingface.co/RedHatAI/GLM-5.2-NVFP4 model
# Using the https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py as source of truth

from tinygrad import fetch, Tensor, nn, dtypes
from dataclasses import dataclass
import functools
from typing import Literal

LayerType = Literal["dense", "sparse"]
IndexerType = Literal["full", "shared"]
@dataclass(frozen=True)
class GLMConfig:
  dim: int
  norm_eps: float
  vocab_size: int
  max_context: int
  intermediate_size: int
  num_experts: int
  num_token_experts: int
  moe_intermediate_size: int
  hidden_act: str
  head_dim: int
  num_heads: int
  qk_rope_head_dim: int
  q_lora_rank: int
  norm_eps: float
  idx_head_dim: int
  idx_num_heads: int
  idx_topk: int
  idx_topk: int
  layers: list[LayerType]
  indexer_type: list[IndexerType]
  rope_dim: int
  attn_bias: bool = False
  attn_gate: bool = True

# KV cache Sparse MLA(DSA)
class DSAKVCache:
  def __init__(self, layers: int, indexers: list[IndexerType], max_context: int, latent_dim: int, idx_k_cache_dim: int, max_batch: int):
    self.layers, self.max_context, self.attn_latent_dim = layers, max_context, latent_dim
    self.idx_k_cache_dim, self.max_batch, self.indexer_type = idx_k_cache_dim, max_batch, indexers
    self.full_to_slot = {i: s  for s, i in enumerate([i for i, v in enumerate(indexers) if v == "full"])}
    self.store_attn = Tensor.empty(layers, max_batch, max_context, latent_dim)
    self.store_attn_idx = Tensor.empty(indexers.count("full"), max_batch, max_context, idx_k_cache_dim)

  CacheSegment = Literal["attention", "indexer"]
  # This expects the shape (B, T, d)
  def update_cache(self, layer: int, start_pos: list[int], to_write: list[int], value: Tensor, segment: CacheSegment) -> Tensor:
    B, T, d = value.shape
    assert 1 <= B <= self.max_batch, "batch size is greater than max batch"
    assert 1 <= T <= self.max_context, "tokens more than max context"
    assert 1 <= len(start_pos) == len(to_write) == B, "list values are wrong"
    assert layer < self.layers, "layer is greater than max_layer"
    assert all([s + v <= self.max_context for (s, v) in zip(start_pos, to_write)]), "sequence is greater than max_context"
    max_end_point = max([a+b for a, b in zip(start_pos, to_write)])
    if segment == "attention":
      assert d == self.attn_latent_dim, "shape of dim is incorrect"
      [self.store_attn[layer, b, s:s + v].assign(value[b, :v]) for b,(s, v) in enumerate(zip(start_pos, to_write))]
      return self.store_attn[layer, :B, :max_end_point]
    else:
      assert d == self.idx_k_cache_dim, "shape of dim is incorrect"
      assert self.indexer_type[layer] == "full", "there should be no k value for indexers that are shared"
      slot = self.full_to_slot[layer]
      [self.store_attn_idx[slot, b, s:s + v].assign(value[b, :v]) for b,(s, v) in enumerate(zip(start_pos, to_write))]
      return self.store_attn_idx[slot, :B, :max_end_point]

@functools.cache
def precompute_freqs_cis(batch:int, dim: int, end: int, theta: float, device:str|None=None) -> Tensor:
  assert dim % 2 == 0, "dim must be even for ROPE"
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2) / dim))
  freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0) # (end, 1) * (1, dim//2) -> (end, dim//2)
  return freqs.cos().cat(freqs.sin(), dim=-1).clone(device) # (end, dim)
 
# This does interleaved ROPE, which is slightly different from the 
# half-split(llama) version
# Exptets x to be of shape (B, S, H, HD)
def apply_rope(x:Tensor, freqs_cis:Tensor, start_pos: list[int], to_write: list[int]) -> Tensor:
  assert max(to_write) == x.shape[1]
  assert x.shape[-1] % 2 == 0
  batched_freqs = Tensor.empty(x.shape[0], max(to_write), freqs_cis.shape[-1])
  for i, (s, tw) in enumerate(zip(start_pos, to_write)): batched_freqs[i].assign(freqs_cis[s:s+tw].pad_to(max(to_write), None)) # Taking slice for each token from freqs
  cos, sin = batched_freqs.reshape(x.shape[0], max(to_write), 1, -1).chunk(2, dim=-1) # (B, max(to_write), 1, dim//2)
  x1, x2 = x[..., 0::2], x[..., 1::2]
  return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)

class GLMDSAIndexer():
  def __init__(self, layer:int, config:GLMConfig):
    self.idx_num_heads, self.idx_head_dim, self.qk_rope_head_dim = config.idx_num_heads, config.idx_head_dim, config.qk_rope_head_dim
    self.layer, self.idx_topk = layer, config.idx_topk
    self.q_up_proj_idx = nn.Linear(config.q_lora_rank, self.idx_num_heads * self.idx_head_dim, bias=False)
    self.wk_idx = nn.Linear(config.dim, self.idx_head_dim, bias=False)
    self.k_norm_idx = nn.LayerNorm(self.idx_head_dim, eps=1e-6)
    self.weights_proj_idx = nn.Linear(config.dim, self.idx_num_heads, bias=False)
    self.softmax_scale = self.idx_head_dim ** -0.5

  def forward(self, x: Tensor, q_resid: Tensor, position_embeddings: Tensor, kv_cache: DSAKVCache, step_t: list[int], to_write: list[int], attention_mask: Tensor):
    assert x.shape[0] == len(step_t) == len(to_write) == q_resid.shape[0], "error in batch dim"
    q_idx = self.q_up_proj_idx(q_resid) # (B, S, H*HD)
    q_idx = q_idx.reshape(*q_idx.shape[:-1], self.idx_num_heads, self.idx_head_dim) # (B, S, H, HD)
    q_rot_idx, q_pass_idx = q_idx[..., :self.qk_rope_head_dim], q_idx[..., self.qk_rope_head_dim:]

    k_idx = self.k_norm_idx(self.wk_idx(x)).unsqueeze(2) # (B, S, HD) -> (B, S, 1, HD)
    k_rot_idx, k_pass_idx = k_idx[..., :self.qk_rope_head_dim], k_idx[...,self.qk_rope_head_dim:]

    q_rot_idx, k_rot_idx = apply_rope(q_rot_idx, position_embeddings, step_t, to_write), apply_rope(k_rot_idx, position_embeddings, step_t, to_write)
    q_idx, k_idx = q_rot_idx.cat(q_pass_idx), k_rot_idx.cat(k_pass_idx).squeeze(2) # q_idx: (B, S, H, HD) k_idx: (B, S, HD)

    k_idx = kv_cache.update_cache(self.layer, step_t, to_write, k_idx, "indexer") # This will return values that are padded to the max sequnce of all batches

    # (B, S, H, HD) @ (B, 1, HD, S) -> (B, S, H, S)
    scores = q_idx.float() @ k_idx.transpose(-1, -2).float().unsqueeze(1)
    scores: Tensor = scores * (self.idx_head_dim ** -0.5)
    scores = scores.relu()

    # weights per head (B, T, idx_num_heads)
    weights = self.weights_proj_idx(x.cast(self.weights_proj_idx.weight.dtype)).float() * (self.idx_num_heads ** -0.5)
    index_scores = weights.unsqueeze(-2) @ scores
    index_scores = index_scores.squeeze(-2)

    index_scores = index_scores + attention_mask

    topk = int(min(self.idx_topk, index_scores.shape[-1]))
    return index_scores.topk(topk, dim=-1)[1].cast(dtypes.int32)



class GLMAttention():
  def __init__(self, config: GLMConfig):
    proj_out = config.head_dim * config.num_heads
    self.attn_q = nn.Linear(config.dim, proj_out, bias=config.attn_bias)
    self.attn_k = nn.Linear(config.dim, proj_out, bias=config.attn_bias)
    self.attn_v = nn.Linear(config.dim, proj_out, bias=config.attn_bias)
    self.attn_gate = nn.Linear(config.dim, proj_out, bias=config.attn_bias) if config.attn_gate else None

    self.attn_output = nn.Linear(proj_out, config.dim, bias=config.attn_bias)

class GLMMoeGateTopK():
  def __init__(self, config: GLMConfig):
    pass

class GLMMoeExperts():
  def __init__(self, config: GLMConfig):
    self.gate_proj = Tensor.empty(config.num_experts, 2 * config.moe_intermediate_size, config.dim)
    self.down_proj = Tensor.empty(config.num_experts, config.dim,)
  pass

class GLMMoeLayer():
  def __init__(self, config: GLMConfig):
    self.experts = GLMMoeExperts(config)
    self.gate = GLMMoeGateTopK(config)
    pass

# Done
class GLMMLPLayer():
  def __init__(self, config: GLMConfig):
    self.gate_proj = nn.Linear(config.dim, config.intermediate_size, bias=False)
    self.up_proj = nn.Linear(config.dim, config.intermediate_size, bias=False)
    self.down_proj = nn.Linear(config.intermediate_size, config.dim, bias=False)

  def forward(self, x: Tensor) -> Tensor:
    return self.down_proj(self.gate_proj(x).silu() * self.up_proj(x))


class GLMBlock():
  def __init__(self, config: GLMConfig, layer: int):
    self.input_layernorm = nn.RMSNorm(config.dim, config.norm_eps)
    self.self_attn = GLMAttention(config)
    if config.layers[layer] == "dense":
      self.block_layer = GLMMLPLayer(config)
    else:
      self.block_layer = GLMMoeLayer()
    self.post_attn_layer_norm = nn.RMSNorm(config.dim, config.norm_eps)

class GLMModel():
  def __init__(self, config: GLMConfig):
    self.embedding = nn.Embedding(config.vocab_size, config.dim)
    self.layers = [GLMBlock(config, i) for i in range(len(config.layers))]
    self.norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.rotary_embedding = GLMRope()
    pass


#TODO: try to get rid of this import
from tokenizers import Tokenizer
def load_tokenizer()-> Tokenizer: 
  #TODO: try to get rid of this and make own tokenizer with the json
  return Tokenizer.from_file(str(fetch("https://huggingface.co/RedHatAI/GLM-5.2-NVFP4/resolve/main/tokenizer.json")))

# This has a total of 99)
  model_tensors: dict[str, Tensor] = {}
  for i in range(to_load): model_tensors.update(nn.state.safe_load(fetch(f"https://huggingface.co/RedHatAI/GLM-5.2-NVFP4/resolve/main/model-0000{i+1}-of-00009.safetensors")))
  return model_tensors

if __name__ == "__main__":
  #TODO: look into making a fast tokenizer
  tokenizer = load_tokenizer()
  model_part_1 = load_model(1)

  # encoded = tokenizer.encode("Hello, my name is risahbh")
  # print("ids", encoded.ids)
  # print("tokens", encoded.tokens)
  #
  # decoded = tokenizer.decode(encoded.ids)
  # print(decoded)

  print([x for x in model_part_1.keys() if "layers.0." in x])
