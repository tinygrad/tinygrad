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
  layers: list[LayerType]
  norm_eps: float
  vocab_size: int
  max_context: int
  intermediate_size: int
  num_exp: int
  num_exp_per_tok: int
  routed_scaling_fact: float
  moe_intermediate_size: int
  norm_topk_prob: bool
  hidden_act: str
  attn_dim: int
  attn_heads: int
  attn_rope_dim: int
  q_latent_dim: int
  kv_latent_dim: int
  attn_rot_dim: int
  norm_eps: float
  idx_head_dim: int
  idx_num_heads: int
  idx_topk: int
  indexer_type: list[IndexerType]
  rope_dim: int

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
    self.idx_num_heads, self.idx_head_dim, self.qk_rope_head_dim = config.idx_num_heads, config.idx_head_dim, config.attn_rope_dim
    self.layer, self.idx_topk = layer, config.idx_topk
    self.q_up_proj_idx = nn.Linear(config.q_latent_dim, self.idx_num_heads * self.idx_head_dim, bias=False)
    self.wk_idx = nn.Linear(config.dim, self.idx_head_dim, bias=False)
    self.k_norm_idx = nn.LayerNorm(self.idx_head_dim, eps=1e-6)
    self.weights_proj_idx = nn.Linear(config.dim, self.idx_num_heads, bias=False)
    self.softmax_scale = self.idx_head_dim ** -0.5

  def __call__(self, x: Tensor, q_latent: Tensor, position_embeddings: Tensor, kv_cache: DSAKVCache, step_t: list[int], to_write: list[int], attn_mask: Tensor) -> Tensor:
    assert x.shape[0] == len(step_t) == len(to_write) == q_latent.shape[0], "error in batch dim"
    q_idx = self.q_up_proj_idx(q_latent) # (B, S, H*HD)
    q_idx = q_idx.reshape(*q_idx.shape[:-1], self.idx_num_heads, self.idx_head_dim) # (B, S, H, HD)
    q_rot_idx, q_pass_idx = q_idx[..., :self.qk_rope_head_dim], q_idx[..., self.qk_rope_head_dim:]

    k_idx = self.k_norm_idx(self.wk_idx(x)).unsqueeze(2) # (B, S, HD) -> (B, S, 1, HD)
    k_rot_idx, k_pass_idx = k_idx[..., :self.qk_rope_head_dim], k_idx[...,self.qk_rope_head_dim:]

    q_rot_idx, k_rot_idx = apply_rope(q_rot_idx, position_embeddings, step_t, to_write), apply_rope(k_rot_idx, position_embeddings, step_t, to_write)
    q_idx, k_idx = q_rot_idx.cat(q_pass_idx), k_rot_idx.cat(k_pass_idx).squeeze(2) # q_idx: (B, S, H, HD) k_idx: (B, S, HD)

    # (B, S_longest, HD)
    # TODO: is this the correct place to store cache
    k_idx = kv_cache.update_cache(self.layer, step_t, to_write, k_idx, "indexer") # This will return values that are padded to the max sequnce of all batches 

    # (B, S, H, HD) @ (B, 1, HD, T) -> (B, S, H, T)
    scores = q_idx.float() @ k_idx.transpose(-1, -2).float().unsqueeze(1)
    scores: Tensor = scores * (self.idx_head_dim ** -0.5)
    scores = scores.relu()

    # weights per head (B, S, idx_num_heads)
    weights = self.weights_proj_idx(x.cast(self.weights_proj_idx.weight.dtype)).float() * (self.idx_num_heads ** -0.5)
    index_scores = weights.unsqueeze(-2) @ scores # (B, S, 1, H) @ (B, S, H, T) => (B, S, 1, T)
    index_scores = index_scores.squeeze(-2) # (B, S, T)

    index_scores = index_scores + attn_mask

    topk = int(min(self.idx_topk, index_scores.shape[-1]))
    return index_scores.topk(topk, dim=-1)[1].cast(dtypes.int32)

# TODO: make this cleaner, maybe use abs positions in a Tensor instead of step_t and to_write
def create_attention_mask(step_t: list[int], to_write: list[int], longest_seq: int) -> Tensor:
    longest_input_seqs = max(to_write)
    seq_pos = Tensor.stack(*[Tensor([s + s_i for s_i in range(w)]).pad((None, (0, longest_input_seqs - w)), value=dtypes.float.max) for s, w in zip(step_t, to_write)]) # (B, S_inp)
    query_ok = seq_pos < Tensor([s + w for s, w in zip(step_t, to_write)]).unsqueeze(-1) # (B, S_inp)
    keys_arange = Tensor.arange(longest_seq).unsqueeze(0).unsqueeze(1) # (1, 1, T)
    key_ok = keys_arange < Tensor([s + w for s, w in zip(step_t, to_write)]).unsqueeze(1).unsqueeze(2) # (1, 1, T) (B, 1, 1) => (B, 1, T)
    valid = ((keys_arange <= seq_pos.unsqueeze(-1)) & key_ok) & query_ok.unsqueeze(-1) # (B, S_inp, longest_seq)
    return valid.where(0, dtypes.float.min)

def basic_attention(queries: Tensor, keys: Tensor, values: Tensor, attn_mask: Tensor, topk_idx: Tensor, attn_heads: int, attn_dim: int) -> tuple[Tensor, Tensor]:
    batch, queries, keys, values = queries.shape[0], queries.permute(0, 2, 1, 3), keys.permute(0, 2, 1, 3), values.permute(0, 2, 1, 3)
    topk_idx_inv = Tensor.ones_like(attn_mask).scatter(-1, topk_idx, 0)
    attn_mask = attn_mask.masked_fill(topk_idx_inv.bool(), dtypes.float.min) # (B, S_inp, S_longest)

    attn_weights = ((queries @ keys.transpose(-1, -2)) + attn_mask.unsqueeze(1)) * (attn_dim ** -0.5) # (B, attn_head, S_inp, S_longest)
    attn_weights = attn_weights.softmax(-1) # (B, attn_head, S_inp, S_longest)
    attn_output = attn_weights @ values # (B, attn_head, S_inp, attn_dim)
    attn_output = attn_output.permute(0, 2, 1, 3) # (B, S_inp, attn_head, attn_dim)
    attn_output = attn_output.reshape(batch, -1, attn_heads * attn_dim).contigous() # (B, S_inp, attn_head * attn_dim)
    return attn_weights, attn_output


class GLMAttention():
  def __init__(self, config: GLMConfig, layer: int):
    self.attn_dim, self.attn_rot_dim, self.kv_latent_dim, self.attn_heads, self.layer = config.attn_dim, config.attn_rot_dim, config.kv_latent_dim, config.attn_heads, layer
    self.q_down_proj = nn.Linear(config.dim, config.q_latent_dim, bias=False)
    self.q_norm = nn.RMSNorm(config.q_latent_dim)
    self.q_up_proj = nn.Linear(config.q_latent_dim, config.attn_heads * config.attn_dim, bias=False)

    self.kv_down_proj = nn.Linear(config.dim, config.kv_latent_dim + config.attn_rot_dim, bias=False)
    self.kv_norm = nn.RMSNorm(config.kv_latent_dim)
    self.kv_up_proj = nn.Linear(config.kv_latent_dim, config.attn_heads * ((config.attn_dim - config.attn_rot_dim) + config.attn_dim))

    self.o_proj = nn.Linear(config.attn_heads * config.attn_dim, config.dim, bias=False)

    self.indexer = GLMDSAIndexer(layer, config) if config.indexer_type[layer] == "full" else None

  # x: (B, S, dim)
  def __call__(self, x: Tensor, position_embeddings: Tensor, step_t: list[int], to_write: list[int], kv_cache: DSAKVCache, prev_idx_topk: Tensor|None):
    batch, orig_seq = x.shape[:-1]
    q_latent = self.q_norm(self.q_down_proj(x)) # (B, S, q_lat)
    q_pass, q_rot = self.q_up_proj(q_latent).reshape(batch, orig_seq, -1, self.attn_dim).split([self.attn_dim - self.attn_rot_dim, self.attn_rot_dim], dim=-1)

    cached_kv = self.kv_down_proj(x)
    kv_common, k_pre_rot= cached_kv.split([self.kv_latent_dim, self.attn_rot_dim], dim=-1)
    k_pre_rot = k_pre_rot.reshape(batch, orig_seq, 1, self.attn_rot_dim)
    q_rot, k_rot = apply_rope(q_rot, position_embeddings, step_t, to_write), apply_rope(k_pre_rot, position_embeddings, step_t, to_write)

    kv_common, k_rot = kv_cache.update_cache(self.layer, step_t, to_write, kv_common.cat(k_rot.squeeze(2)), "attention").split([self.kv_latent_dim, self.attn_rot_dim], dim=-1)
    longest_seq = kv_common.shape[1] # Now this becomes the longest seq in this set of batches that came from kv_cache
    k_rot = k_rot.unsqueeze(2)

    # k_pass: (B, S_longest, attn_dim - attn_rot_dim) values: (B, S_longest, attn_dim)
    k_pass, values = self.kv_up_proj(self.kv_norm(kv_common)).reshape(batch, longest_seq, self.attn_heads, -1).split([self.attn_dim - self.attn_rot_dim, self.attn_dim], dim=-1)
    
    k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

    queries = q_pass.cat(q_rot, dim=-1) # (B, S_inp, attn_head, attn_dim)
    keys = k_pass.cat(k_rot, dim=-1) # (B, S_longest, attn_head, attn_dim)
    
    # creating the attention mask after kv cache fills in old keys as well
    attn_mask = create_attention_mask(step_t, to_write, int(longest_seq)) # (B, S_inp, S_longest)

    if self.indexer is not None:
      #TODO:: Double check is this the right way to put mask, it will have the shape of the biggest sequence
      topk_idx = self.indexer(x, q_latent, position_embeddings, kv_cache, step_t, to_write, attn_mask) # (B, S_inp, K)
    else:
      assert prev_idx_topk is not None, "previous idx topk should come for shared idx layers"
      topk_idx = prev_idx_topk

    topk_idx_inv = Tensor.ones_like(attn_mask).scatter(-1, topk_idx, 0)
    attn_mask = attn_mask.masked_fill(topk_idx_inv.bool(), dtypes.float.min) # (B, S_inp, S_longest)
    attn_weights, attn_output = basic_attention(queries, keys, values, attn_mask, topk_idx, self.attn_heads, self.attn_dim)
    attn_ouput_up_proj = self.o_proj(attn_output) # (B, dim)
    return attn_ouput_up_proj, attn_weights, topk_idx

class GLMMoeGateTopK():
  def __init__(self, config: GLMConfig):
    # NOTE: we have only one group in GLM so not taking that into consideration
    self.num_exp, self.num_exp_per_token, self.dim = config.num_exp ,config.num_exp_per_tok, config.dim
    self.routed_scaling_fact = config.routed_scaling_fact
    self.weight = Tensor.empty(self.num_exp, self.dim)
    self.exp_score_corretion_bias = Tensor.empty(self.num_exp)

  # x: (B, S, dim)
  def __call__(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    router_logits = x @ self.weight.transpose() # (B, S, dim) @ (dim, E) => (B, S, E)
    scores = router_logits.sigmoid() # (B, S, E)
    scores_for_choice = scores + self.exp_score_corretion_bias # (B, S, E)
    topk_idx = scores_for_choice.topk(self.num_exp_per_token, -1)[1] # (B, S, K)
    topk_weights = scores.gather(-1,topk_idx)

    # Normalizing the weights
    denominator = topk_weights.sum(-1, keepdim=True)
    topk_weights = topk_weights / (denominator + 1e-20)
    topk_weights = topk_weights * self.routed_scaling_fact

    return router_logits, topk_weights, topk_idx

class GLMMoeExperts():
  def __init__(self, config: GLMConfig):
    self.num_experts = config.num_exp
    self.gate_up_proj = Tensor.empty(config.num_exp, 2 * config.moe_intermediate_size, config.dim)
    self.down_proj = Tensor.empty(config.num_exp, config.dim, config.moe_intermediate_size)

  # x : (B, S, dim) topk_idx: (B, S, k) topk_weigths: (B, S, k)
  def __call__(self, x: Tensor, topk_idx: Tensor, topk_weights: Tensor): 
    # gate, up: (B, S, K, I)
    gate, up = (x.unsqueeze(2).unsqueeze(3) @ self.gate_up_proj[topk_idx].transpose(-1, -2)).squeeze(-2).chunk(2, dim=-1) # (B, S, 1, 1, dim) @ (B, S, k, dim, 2I) => (B,S, K, 1, 2I)
    x = (gate.silu() * up).contiguous() # (B, S, K, I)
    x = (x.unsqueeze(-2) @ self.down_proj[topk_idx].transpose(-1, -2)).squeeze(-2) # (B, S, K, 1, I) @ (B, S, K, I, dim) => (B, S, K, 1, dim)
    x = x * topk_weights.unsqueeze(-1) # (B, S, K, dim)
    return x.sum(axis=-2) # (B, S, dim)

class GLMMoeLayer():
  def __init__(self, config: GLMConfig):
    self.experts = GLMMoeExperts(config)
    self.gate = GLMMoeGateTopK(config)

  def __call__(self):
    pass

# Done
class GLMMLPLayer():
  def __init__(self, config: GLMConfig):
    self.gate_proj = nn.Linear(config.dim, config.intermediate_size, bias=False)
    self.up_proj = nn.Linear(config.dim, config.intermediate_size, bias=False)
    self.down_proj = nn.Linear(config.intermediate_size, config.dim, bias=False)

  def __call__(self, x: Tensor) -> Tensor:
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

  def __call__(self):
    pass

class GLMModel():
  def __init__(self, config: GLMConfig):
    self.embedding = nn.Embedding(config.vocab_size, config.dim)
    self.layers = [GLMBlock(config, i) for i in range(len(config.layers))]
    self.norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.rotary_embedding = GLMRope()

  def __call__(self):
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
