from __future__ import annotations
import enum, functools, itertools, json, math, pathlib, re
from dataclasses import dataclass, replace
from typing import Callable
from tinygrad import Tensor, nn, UOp, TinyJit, getenv, function, dtypes, Device
from tinygrad.llm.kernels.amd import Linear, ExpertWeights, fp8_matmul, gated_delta_prefill, flash_attention, amd_custom_kernels_supported, \
  indexer_scores, topk_indices, mla_topk_attention, mla_topk_supported, res_mix_norm, kda_decode, RES_MAX_STREAMS, allgather_last, allgather_lasts
from tinygrad.llm.gguf import gguf_parse, gguf_shard, GGUFPackedTable, _GGML_QUANT, _GGML_NATIVE
from tinygrad.llm.kernels.amd_gguf import q8_linear, head_linear, QUANT_SIZES
from tinygrad.llm.hf import hf_load
from tinygrad.uop.ops import resolve, smax, ssimplify
from tinygrad.helpers import unwrap, round_up, get_child

StartPos = tuple[int|UOp, ...]

class ExpertGating(enum.IntEnum):
  SOFTMAX = 1
  SIGMOID = 2
  SOFTMAX_WEIGHT = 3  # softmax over the top-k selected logits
  SQRT_SOFTPLUS = 4

@dataclass(frozen=True)
class YaRNConfig:
  factor: float
  orig_ctx_len: int
  beta_fast: float
  beta_slow: float
  mscale: float|None = None

@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device:str|tuple[str, ...]|None=None, yarn:YaRNConfig|None=None) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2) / dim))
  concentration = 1.0
  if yarn is not None and yarn.factor > 1.0:
    concentration = 0.1 * math.log(yarn.factor) + 1.0 if yarn.mscale is None else 1.0
    d_half = dim // 2
    low = max(0, math.floor(d_half * math.log(yarn.orig_ctx_len / (yarn.beta_fast * 2 * math.pi)) / math.log(theta)))
    high = min(dim-1, math.ceil(d_half * math.log(yarn.orig_ctx_len / (yarn.beta_slow * 2 * math.pi)) / math.log(theta)))
    interp = 1 - ((Tensor.arange(d_half).float() - low) / max(0.001, high-low)).clamp(0, 1)
    freqs = freqs * interp + (freqs / yarn.factor) * (1 - interp)
  freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  return (freqs.cos() * concentration).cat(freqs.sin() * concentration, dim=-1).clone(device)

def gated_activation(gate:Tensor, up:Tensor, config:TransformerConfig) -> Tensor:
  alpha, limit, up_bias, situ = config.swiglu_alpha, config.swiglu_limit, config.swiglu_up_bias, config.situ
  if situ is not None: return (gate / situ[0]).tanh() * situ[0] * gate.sigmoid() * (up / situ[1]).tanh() * situ[1]
  if limit is not None and limit > 0: gate, up = gate.clamp(max_=limit), up.clamp(-limit, limit)
  # TODO: remove the need for this contiguous
  return gate * (gate * alpha).sigmoid().contiguous() * (up + up_bias)

def apply_rope(x:Tensor, freqs_cis:Tensor, interleaved:bool=False) -> Tensor:
  assert x.shape[-1] % 2 == 0
  if interleaved: x = x.rearrange("b h t (d two) -> b h t (two d)", two=2)
  cos, sin = freqs_cis.unsqueeze(-3).chunk(2, dim=-1)
  x1, x2 = x.chunk(2, dim=-1)
  return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)

def _update_cache(cache:Tensor, value:Tensor, start_pos:StartPos, dim:int) -> Tensor:
  def store(c:Tensor, v:Tensor, pos:int|UOp) -> UOp:
    return c.shrink(tuple((pos, pos+v.shape[dim]) if i == dim else None for i in range(c.ndim))).uop.store(v.cast(c.dtype).uop)
  stores = ([store(cache, value, start_pos[0])] if len(start_pos) == 1 else [store(cache[b:b+1], value[b:b+1], p) for b, p in enumerate(start_pos)])
  return Tensor(cache.uop.after(*stores))

def _cache_prefix(cache:Tensor, start_pos:StartPos, n_tokens:int|UOp, dim:int) -> Tensor:
  end = ssimplify(smax(start_pos) + n_tokens)
  cache = cache.shrink(tuple((0, end) if i == dim else None for i in range(cache.ndim)))
  if len(start_pos) > 1:
    keys = Tensor.arange(end).reshape(tuple(end if i == dim else 1 for i in range(cache.ndim)))
    ends = (Tensor.stack(*(Tensor(p) for p in start_pos)) + Tensor(n_tokens)).reshape(len(start_pos), *((1,) * (cache.ndim-1)))
    cache = (keys < ends).where(cache, 0)  # future entries can be uninitialized, including NaNs
  return cache

def _causal_mask(x:Tensor, start_pos:StartPos, window:int=0) -> Tensor|None:
  T = x.shape[1]
  if len(start_pos) == 1 and not resolve(T != 1) and not window: return None
  q_pos = Tensor.stack(*(Tensor(p) for p in start_pos)).reshape(-1, 1, 1, 1) + Tensor.arange(T).reshape(1, 1, T, 1)
  k_pos = Tensor.arange(ssimplify(smax(start_pos) + T)).reshape(1, 1, 1, -1)
  valid = k_pos <= q_pos
  if window: valid = valid & (k_pos > q_pos-window)
  return valid.where(0.0, -math.inf).cast(x.dtype)

def permute_rope(w:Tensor, n_heads:int, rope_dim:int) -> Tensor:
  x = w.reshape(n_heads, w.shape[0]//n_heads, -1)
  prefix = x.shape[1]-rope_dim
  return x[:, :prefix].cat(x[:, prefix:].rearrange("n (h two) d -> n (two h) d", two=2), dim=1).reshape(w.shape)

def pairwise_topk(x: Tensor, k: int) -> tuple[Tensor, Tensor]:
  n = x.shape[-1]
  vals = Tensor.arange(n).reshape(1,1,n).cast(x.dtype).expand(x.shape)
  cmp = (x.unsqueeze(-1) > x.unsqueeze(-2)) | ((x.unsqueeze(-1) == x.unsqueeze(-2)) & \
    (Tensor.arange(n).reshape(1,1,n,1) < Tensor.arange(n).reshape(1,1,1,n)))
  sel = x.const_like(0).scatter(-1, cmp.sum(axis=-1).cast('int32'), vals)[:,:,n-k:].cast('int32')
  return x.gather(-1, sel), sel

def short_conv(state:Tensor, x:Tensor, weight:Tensor, initial:Tensor, dilation:int=1, live:UOp|None=None) -> tuple[Tensor, UOp]:
  assert state.shape[1] == (weight.shape[1]-1)*dilation, f"state has {state.shape[1]} rows, expected {(weight.shape[1]-1)*dilation}"
  # assemble the conv window in a static-size buffer: [state | x rows | zero-pad], state is zeroed when initial
  T, T_pad = x.shape[1], x.max_shape[1]
  window = initial.where(0, state).cat(x.cast(state.dtype), dim=1).pad_to((x.shape[0], state.shape[1] + T_pad, x.shape[-1])).contiguous()
  # the last state.shape[1] rows before the padding become the next state
  end = T if live is None else live
  store = state.uop.store(window[:, end:end+state.shape[1]].uop)
  return functools.reduce(lambda a,b: a+b, (window[:, i*dilation:i*dilation+T_pad] * weight[:, i] for i in range(weight.shape[1]))).silu(), store

@dataclass(frozen=True)
class SSMConfig:
  conv_kernel: int
  state_size: int
  group_count: int
  time_step_rank: int
  inner_size: int
  kda: bool = False
  gate_lower_bound: float|None = None  # kda: bounded decay gate
  full_rank: tuple[bool, bool] = (False, False)  # kda: the output (g) and decay (f) gates are full rank or low rank
  silu_gate: bool = False

@dataclass(frozen=True)
class HCConfig:  # manifold-constrained hyper-connections: count residual streams, mixed by a doubly stochastic matrix
  count: int
  sinkhorn_iters: int
  eps: float
  low_rank: int = 0

@dataclass(frozen=True)
class PLEConfig:
  layer: int
  head_dim: int
  multipliers: tuple[int, ...]
  vocab_sizes: tuple[int, ...]
  eos: int
  conv_kernel: int

@dataclass(frozen=True)
class IndexerConfig:  # select top_k tokens by scoring individual keys or pools of kpool keys
  n_heads: int
  head_dim: int
  top_k: int
  kpool: int = 1
  norm_eps: float = 1e-6
  rope_dim: int = 0
  rope_interleave: bool = True
  msa: bool = False
  qsa: bool = False

@dataclass(frozen=True)
class TransformerConfig:
  num_blocks: int
  dim: int
  hidden_dim: int
  n_heads: int
  n_kv_heads: int
  norm_eps: float
  vocab_size: int
  head_dim: int
  rope_theta: float
  rope_dim: int
  v_head_dim: int
  yarn: YaRNConfig|None = None
  max_context: int = 0
  qk_norm: int = 0
  num_experts: int = 0
  num_experts_per_tok: int = 0
  norm_topk_prob: bool = False
  expert_gating_func: ExpertGating = ExpertGating.SOFTMAX
  expert_group_count: int = 0
  expert_group_used: int = 0
  q_lora_rank: int = 0
  kv_lora_rank: int = 0
  shared_expert_dim: int = 0
  ssm_layers: tuple[bool, ...] = ()
  attn_output_gate: bool = False
  attn_output_bias: bool = False
  attn_sinks: bool = False
  ssm: SSMConfig|None = None
  shared_expert_gate: bool = True
  leading_dense_blocks: int = 0
  dense_hidden_dim: int = 0
  routed_scaling_factor: float = 1.0
  qkv_bias: bool = False
  expert_bias: bool = False
  expert_proj_bias: bool = False
  swiglu_alpha: float = 1.0
  swiglu_limit: float|None = None
  swiglu_up_bias: float = 0.0
  sliding_window: int = 0
  sliding_window_pattern: int = 0
  hc: HCConfig|None = None
  indexer: IndexerConfig|None = None
  ple: PLEConfig|None = None
  indexer_shared: tuple[bool, ...] = ()
  rope_interleave: bool = False
  expert_latent_dim: int = 0
  situ: tuple[float, float]|None = None
  attn_res_block_size: int = 0
  attn_res_new_block: bool = False
  attn_head_gate: bool = False
  mla_nope: bool = False  # NoPE (No Positional Embeddings)
  streamed_experts: bool = False  # no routed expert weights: _routed is zero unless replaced (by an engine that streams the experts in)
  mtp: bool = False

class AttentionResidual:
  def __init__(self, config:TransformerConfig, new_block:bool=False):
    self.weight, self.norm_eps, self.new_block = Tensor.zeros(config.dim), config.norm_eps, new_block

  def mix(self, xs:Tensor, norm:Callable[[Tensor], Tensor]|None=None, count:UOp|None=None, rows:tuple[int|UOp, int]|None=None) -> Tensor:
    # xs: (B, T, N, D) streams: (B, T, D), then the norm. count: the streams mixed, the rest (zero) left out. rows: (start, n) xs's rows mixed
    # (xs split on its rows over the devices: each device's own rows start..)
    if isinstance(norm, nn.RMSNorm) and isinstance(xs.shape[1], int) and xs.shape[2] <= RES_MAX_STREAMS and \
       xs.shape[-1] % 256 == 0 and xs.shape[-1] <= 8192 and amd_custom_kernels_supported(xs.device):
      return res_mix_norm(xs, self.weight, unwrap(norm.weight), self.norm_eps, norm.eps, count, rows)
    if rows is not None and xs.uop.axis == 1 and isinstance(xs.device, tuple):
      xs = xs.reshape(xs.shape[0], len(xs.device), -1, *xs.shape[2:])[:, :, rows[0]:rows[0] + rows[1]].flatten(1, 2)
    elif rows is not None: xs = xs[:, rows[0]:rows[0] + rows[1]]
    xs = xs.float()
    scores = (xs.rmsnorm(eps=self.norm_eps) * self.weight).sum(-1, keepdim=True)
    if count is not None: scores = (Tensor.arange(xs.shape[2]).to(xs.device).reshape(-1, 1) < Tensor(count)).where(scores, -math.inf)
    out = (scores.softmax(2) * xs).sum(2)
    return out if norm is None else norm(out)

  def __call__(self, xs:Tensor, sublayer:Callable[[Tensor], tuple[Tensor, Tensor|None]],
               norm:Callable[[Tensor], Tensor]|None=None) -> tuple[Tensor, Tensor|None]:
    out, aux = sublayer(self.mix(xs, norm))
    # (B, T, N+1, D) if new_block else (B, T, N, D)
    return (xs.cat(out.unsqueeze(2), dim=2) if self.new_block else xs + out.unsqueeze(2).pad((None, None, (xs.shape[2] - 1, 0), None))), aux

class HyperConnection:
  def __init__(self, config:TransformerConfig):
    self.count, self.eps, self.iters, self.norm_eps = (hc:=unwrap(config.hc)).count, hc.eps, hc.sinkhorn_iters, config.norm_eps
    self.fn = Linear(self.count * config.dim, (2 + self.count) * self.count, bias=False)
    self.base, self.scale = {"weight": Tensor.zeros((2 + self.count) * self.count)}, {"weight": Tensor.zeros(3)}

  def __call__(self, xs:Tensor, sublayer:Callable[[Tensor], tuple[Tensor, Tensor|None]]) -> tuple[Tensor, Tensor|None]:
    """mHC: the sublayer sees a pre-weighted sum of the (B, T, hc, D) streams, its output is added back by post onto the streams mixed by comb"""
    base, scale, mix = self.base["weight"], self.scale["weight"], self.fn(xs.flatten(2).rmsnorm(eps=self.norm_eps))
    pre  = (mix[..., :self.count] * scale[0] + base[:self.count]).sigmoid() + self.eps
    post = (mix[..., self.count:2*self.count] * scale[1] + base[self.count:2*self.count]).sigmoid() * 2
    comb = (mix[..., 2*self.count:] * scale[2] + base[2*self.count:]).reshape(*mix.shape[:-1], self.count, self.count).softmax(-1) + self.eps
    # sinkhorn: normalize the columns then the rows, alternating, to make comb doubly stochastic
    for i in range(2 * self.iters - 1): comb = comb / (comb.sum(-1 if i % 2 else -2, keepdim=True) + self.eps)
    out, aux = sublayer((xs * pre.unsqueeze(-1)).sum(2))
    return out.unsqueeze(2) * post.unsqueeze(-1) + (comb.transpose(-1, -2) @ xs), aux

class GroupedRMSNorm:  # x: (B, T, count, dim), w: (count*dim,)
  def __init__(self, count:int, dim:int, eps:float): self.weight, self.eps = Tensor.zeros(count*dim), eps
  def __call__(self, x:Tensor) -> Tensor: return x.rmsnorm(eps=self.eps) * self.weight.reshape(x.shape[-2:])

class GatedResidual(HyperConnection):
  def __init__(self, config:TransformerConfig, inject:bool=True):
    self.count, self.dim, low_rank = (hc:=unwrap(config.hc)).count, config.dim, hc.low_rank
    self.norm = GroupedRMSNorm(self.count, self.dim, config.norm_eps)
    self.down, self.up = Linear(self.count*self.dim, low_rank, bias=False), Linear(low_rank, self.count*self.dim, bias=False)
    if inject: self.inject = Linear(self.count*self.dim, self.count, bias=False)

  def mix(self, xs:Tensor) -> tuple[Tensor, Tensor]:
    xn = self.norm(xs).flatten(2)
    return (self.up((self.down(xn) / self.count).silu()).sigmoid() * xn).reshape(xs.shape).mean(2), xn

  def __call__(self, xs:Tensor, sublayer:Callable[[Tensor], tuple[Tensor, Tensor|None]]) -> tuple[Tensor, Tensor|None]:
    mixed, xn = self.mix(xs)
    out, aux = sublayer(mixed)
    return xs + out.unsqueeze(2) * (self.inject(xn) / self.count).sigmoid().unsqueeze(-1) * 2, aux

class FFNBlock:
  def __init__(self, config:TransformerConfig):
    self.config = config

    # --- RMSNorms --------------------------------------------------------
    self.attn_norm: Callable[[Tensor], Tensor] = nn.RMSNorm(config.dim, config.norm_eps)
    self.ffn_norm:  Callable[[Tensor], Tensor] = nn.RMSNorm(config.dim, config.norm_eps)
    if config.hc is not None:
      hc = GatedResidual if config.hc.low_rank else HyperConnection
      self.hc_attn, self.hc_ffn = hc(config), hc(config)
      if config.hc.low_rank: self.attn_norm = self.ffn_norm = lambda x: x
    if config.ple is not None: self.ple = PLELayer(config)
    if config.attn_res_block_size:
      self.attn_res_score, self.ffn_res_score = AttentionResidual(config, config.attn_res_new_block), AttentionResidual(config)

    # --- feed-forward (MoE or dense) -------------------------------------
    if config.num_experts > 0:
      self.ffn_gate_inp = Linear(config.dim, config.num_experts, bias=config.expert_proj_bias)  # router
      if config.expert_bias: self.exp_probs_b = {"bias": Tensor.zeros(config.num_experts)}
      expert_dim = config.expert_latent_dim or config.dim
      if not config.streamed_experts:
        self.ffn_gate_exps = ExpertWeights(config.num_experts, expert_dim, config.hidden_dim, bias=config.expert_proj_bias)
        self.ffn_up_exps = ExpertWeights(config.num_experts, expert_dim, config.hidden_dim, bias=config.expert_proj_bias)
        self.ffn_down_exps = ExpertWeights(config.num_experts, config.hidden_dim, expert_dim, bias=config.expert_proj_bias)
      if config.shared_expert_dim > 0:
        self.ffn_gate_shexp = Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_up_shexp = Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_down_shexp = Linear(config.shared_expert_dim, config.dim, bias=False)
        if config.shared_expert_gate: self.ffn_gate_inp_shexp = {"weight": Tensor.zeros(config.dim)}
      if config.expert_latent_dim:
        self.ffn_routed_down, self.ffn_routed_up = Linear(config.dim, expert_dim, bias=False), Linear(expert_dim, config.dim, bias=False)
        self.ffn_routed_norm = nn.RMSNorm(expert_dim, config.norm_eps)
    else:
      self.ffn_gate    = Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_up      = Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_down    = Linear(config.hidden_dim, config.dim, bias=False)

  def _routed(self, x:Tensor, h:Tensor) -> Tensor:
    """the routed experts of the tokens x on their expert inputs h (B, T, D): their weighted sum (B, T, D)"""
    assert not self.config.streamed_experts, "the gguf has no routed experts: an engine streams them in (in place of _routed)"
    h = h.unsqueeze(2)  # (B, T, 1, D) - add expert dim for broadcasting
    logits = self.ffn_gate_inp(x)
    bias = self.exp_probs_b["bias"] if hasattr(self, 'exp_probs_b') else None
    gating, normalize_topk = self.config.expert_gating_func, self.config.norm_topk_prob
    # fast path: without selection bias, normalized SOFTMAX is equivalent to SOFTMAX_WEIGHT
    if gating == ExpertGating.SOFTMAX and bias is None and normalize_topk:
      gating, normalize_topk = ExpertGating.SOFTMAX_WEIGHT, False
    if   gating == ExpertGating.SOFTMAX_WEIGHT: scores = logits
    elif gating == ExpertGating.SOFTMAX:        scores = logits.softmax(-1)
    elif gating == ExpertGating.SIGMOID:        scores = logits.sigmoid()
    elif gating == ExpertGating.SQRT_SOFTPLUS:  scores = logits.softplus().sqrt()

    route_scores = scores if bias is None else scores + bias
    if self.config.expert_group_count > 1:
      groups = route_scores.unflatten(-1, (self.config.expert_group_count, -1))
      gscore = groups.topk(1 if bias is None else 2)[0].sum(-1)
      keep = gscore.zeros_like(dtype=dtypes.bool).scatter(-1, gscore.topk(self.config.expert_group_used)[1], True)
      route_scores = keep.unsqueeze(-1).where(groups, -math.inf).flatten(-2)
    _, sel = pairwise_topk(route_scores, self.config.num_experts_per_tok)
    probs = scores.gather(-1, sel)
    # SOFTMAX_WEIGHT applies softmax after top-k selection
    if gating == ExpertGating.SOFTMAX_WEIGHT: probs = probs.softmax(-1)
    if normalize_topk: probs = probs / probs.sum(axis=-1, keepdim=True)
    probs = probs * self.config.routed_scaling_factor
    gate, up = self.ffn_gate_exps(sel, h), self.ffn_up_exps(sel, h)
    x_down = self.ffn_down_exps(sel, gated_activation(gate, up, self.config).contiguous())  # ([ndev,] B, T, k, D)
    out = (x_down * probs.unsqueeze(-1)).sum(axis=-2)  # ([ndev,] B, T, D)
    return out.sum(0) if out.ndim == x.ndim + 1 else out

  def _feed_forward(self, x:Tensor) -> Tensor:
    if hasattr(self, 'ffn_gate_inp'): return self._expert_output(self._routed(x, self._expert_input(x)), self._shared(x))
    return self.ffn_down(gated_activation(self.ffn_gate(x), self.ffn_up(x), self.config))

  # a moe's feed forward in parts (a prompt's window routes its tokens' experts between them): the routed experts' input, the shared experts'
  # output and the routed experts' output out (the same rows) to the block's. the latent projections sharded on their outputs: gathered
  def _expert_input(self, x:Tensor) -> Tensor: return allgather_last(self.ffn_routed_down(x)) if self.config.expert_latent_dim else x
  def _shared(self, x:Tensor) -> Tensor|None:
    if not hasattr(self, 'ffn_gate_shexp'): return None
    shexp = self.ffn_down_shexp(gated_activation(self.ffn_gate_shexp(x), self.ffn_up_shexp(x), self.config).contiguous())
    return shexp * (x * self.ffn_gate_inp_shexp["weight"]).sum(axis=-1, keepdim=True).sigmoid() if hasattr(self, 'ffn_gate_inp_shexp') else shexp
  def _expert_output(self, out:Tensor, shexp:Tensor|None) -> Tensor:
    if self.config.expert_latent_dim: out = allgather_last(self.ffn_routed_up(self.ffn_routed_norm(out)))
    return out if shexp is None else out + shexp

  # given the token-prefix match, return how much cached state this block can still reuse
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return prefix_len
  def _init_state(self, x:Tensor): raise NotImplementedError
  def _attention(self, x:Tensor, start_pos:StartPos, idx_mask:Tensor|None=None, live:UOp|None=None) -> Tensor: raise NotImplementedError
  def _indexer_mask(self, x:Tensor, start_pos:StartPos, idx_mask:Tensor|None) -> Tensor|None:
    # a full indexer layer selects the top-k tokens for itself and the shared layers after it (MLATransformerBlock)
    return idx_mask

  def _attend(self, x:Tensor, start_pos:StartPos, idx_mask:Tensor|None, live:UOp|None) -> tuple[Tensor, Tensor|None]:
    mask = self._indexer_mask(x, start_pos, idx_mask)
    return self._attention(x, start_pos, mask, live), mask

  def __call__(self, x: Tensor, start_pos: StartPos, idx_mask: Tensor|None=None, live:UOp|None=None):
    self._init_state(x)
    if hasattr(self, 'ple'): self.ple._init_state(x)
    # we pass in the weights implicitly so we unpack the GGUF on the fly
    @function(precompile=True, allow_implicit=True)
    def _run(x:Tensor, start_pos:StartPos, idx_mask:Tensor|None, live:UOp|None):
      if hasattr(self, 'ple'): x = x + self.ple(x, start_pos)  # (B, T, hc, D)
      res_attn = res_ffn = lambda xs, sublayer: (xs + (ret:=sublayer(xs))[0], ret[1])
      attn_norm, ffn_norm = self.attn_norm, self.ffn_norm
      if self.config.hc is not None: res_attn, res_ffn = self.hc_attn, self.hc_ffn
      elif self.config.attn_res_block_size:  # (the mixes do the norms)
        res_attn = functools.partial(self.attn_res_score, norm=attn_norm)
        res_ffn = functools.partial(self.ffn_res_score, norm=ffn_norm)
        attn_norm = ffn_norm = lambda h: h
      def attend(h:Tensor) -> tuple[Tensor, Tensor|None]:
        return self._attend(attn_norm(h), start_pos, idx_mask, live)
      x, mask = res_attn(x, attend)
      out = res_ffn(x, lambda h: (self._feed_forward(ffn_norm(h)), None))[0].contiguous()
      return (out, mask) if mask is not None and mask is not idx_mask else out
    out = _run(x, start_pos, idx_mask, live)
    return out if isinstance(out, tuple) else (out, idx_mask)

class TransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    assert config.v_head_dim == config.head_dim, "TransformerBlock requires v_head_dim == head_dim"

    # --- attention projections (all linear, bias-free) ------------------
    q_proj_out       = config.head_dim * config.n_heads * (2 if config.attn_output_gate else 1)
    kv_proj_out      = config.head_dim * config.n_kv_heads
    self.attn_q      = Linear(config.dim, q_proj_out,  bias=config.qkv_bias)
    self.attn_k      = Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    self.attn_v      = Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    self.attn_output = Linear(config.head_dim * config.n_heads, config.dim, bias=config.attn_output_bias)
    if config.qk_norm: self.attn_q_norm, self.attn_k_norm = nn.RMSNorm(config.qk_norm, config.norm_eps), nn.RMSNorm(config.qk_norm, config.norm_eps)
    if config.attn_sinks: self.attn_sinks = {"weight": Tensor.zeros(config.n_heads)}
    self.indexer = Indexer(config) if config.indexer else None

  def _attention(self, x:Tensor, start_pos:StartPos, idx_mask:Tensor|None=None, live:UOp|None=None) -> Tensor:
    q, k, v = self.attn_q(x), self.attn_k(x), self.attn_v(x)
    if self.config.qk_norm and self.config.qk_norm != self.config.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    B, T, _ = x.shape
    if self.config.attn_output_gate:
      qg = q.reshape(B, T, self.config.n_heads, 2, self.config.head_dim)
      q, gate = qg[:, :, :, 0, :], qg[:, :, :, 1, :].reshape(B, T, self.config.n_heads * self.config.head_dim)
    q = q.reshape(B, T, self.config.n_heads,    self.config.head_dim).transpose(1, 2)  # (B,H,T,Hd)
    k = k.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    v = v.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    if self.config.qk_norm == self.config.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    freqs = Tensor.stack(*(self.freqs_cis[p:p+T] for p in start_pos))
    q = apply_rope(q[..., :self.config.rope_dim], freqs).cat(q[..., self.config.rope_dim:], dim=-1)
    k = apply_rope(k[..., :self.config.rope_dim], freqs).cat(k[..., self.config.rope_dim:], dim=-1)

    # NOTE: we don't want to change self.cache_kv, the function API doesn't support this well
    assigned_kv = _update_cache(self.cache_kv.transpose(0, 1), Tensor.stack(k, v).transpose(0, 1), start_pos, 3)
    # on RDNA3/4, hybrid models use custom flash attention kernels on the KV cache
    if  idx_mask is None and amd_custom_kernels_supported(x.device) and self.config.ssm is not None and len(start_pos) == 1:
      attn = flash_attention(q, assigned_kv.transpose(0, 1), start_pos[0]+T)
      attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
      return self.attn_output(attn if not self.config.attn_output_gate else (attn * gate.sigmoid()))
    assigned_kv = _cache_prefix(assigned_kv, start_pos, T, 3).transpose(0, 1)
    k, v = assigned_kv[0], assigned_kv[1]
    mask = (idx_mask.repeat_interleave(self.config.n_heads // int(idx_mask.shape[1]), dim=1) if idx_mask is not None else
            _causal_mask(x, start_pos, self.config.sliding_window))
    if hasattr(self, 'attn_sinks'):
      S = k.shape[-2]
      k, v = k.cat(k[..., :1, :].const_like(0), dim=-2), v.cat(v[..., :1, :].const_like(0), dim=-2)
      sink_col = self.attn_sinks["weight"].reshape(1, -1, 1, 1).expand(1, self.config.n_heads, T, 1)
      if mask is None: mask = Tensor.zeros(1, 1, T, S, dtype=x.dtype, buffer=False)
      mask = mask.expand(B, self.config.n_heads, T, S).cat(sink_col.expand(B, -1, -1, -1), dim=-1)
    attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)     # (B,H,T,Hd)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
    return self.attn_output(attn if not self.config.attn_output_gate else (attn * gate.sigmoid()))

  def _indexer_mask(self, x:Tensor, start_pos:StartPos, idx_mask:Tensor|None) -> Tensor|None:
    if self.indexer is None or self.config.max_context <= self.indexer.idx.top_k: return idx_mask
    return self.indexer(x, x, start_pos, self.freqs_cis)

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_kv"):
      # zeroed so the flash kernels can safely read whole tiles past the valid region (masked lanes multiply by 0)
      self.cache_kv = Tensor.zeros(2, x.shape[0], self.config.n_kv_heads, self.config.max_context, self.config.head_dim,
                                   dtype=dtypes.half, device=x.device if isinstance(x.device, str) else None)
      if isinstance(x.device, tuple): self.cache_kv = self.cache_kv.shard(x.device, 2).realize()
      # start_pos and T have independent symbolic bounds, the slice can overrun the context by all but one of T
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context + x.max_shape[1]-1, self.config.rope_theta,
                                            device=x.device, yarn=self.config.yarn)
    if self.indexer: self.indexer._init_state(x)

class PLELayer:
  def __init__(self, config:TransformerConfig):
    self.table: GGUFPackedTable
    self.ple, self.count, self.dim = unwrap(config.ple), unwrap(config.hc).count, config.dim
    self.max_context, self.hist, width = config.max_context, (self.ple.conv_kernel - 1) * len(self.ple.multipliers), self.count * config.dim
    self.key, self.value = (Linear(len(self.ple.vocab_sizes) * self.ple.head_dim, out, bias=False) for out in (width, config.dim))
    self.norm_key, self.norm_query, self.norm_conv = (GroupedRMSNorm(self.count, self.dim, config.norm_eps) for _ in range(3))
    self.conv1d = {"weight": Tensor.zeros(width, self.ple.conv_kernel)}

  def _init_state(self, x:Tensor):
    if not hasattr(self, "conv_state"): self.conv_state = Tensor.zeros(x.shape[0], self.hist, self.count*self.dim, device=x.device).clone()

  def write_rows(self, tokens:list[int], start_pos:int, n_toks:int, chunk_size:int):
    # hash the n-grams ending at each token (t0*m0 ^ t1*m1 ^ ..., EOS before the start or an EOS)
    ple, n, ids, offsets = self.ple, len(self.ple.multipliers), [], list(itertools.accumulate(self.ple.vocab_sizes, initial=0))
    for p in range(start_pos, start_pos+n_toks):
      ctx = [ple.eos if p < i or ple.eos in tokens[p-i:p] else tokens[p-i] for i in range(n)]
      hashes = list(itertools.accumulate((c*m for c,m in zip(ctx, ple.multipliers)), lambda a,b: a^b))[1:]
      ids.append([hashes[j*(n-1)//len(ple.vocab_sizes)] % v + o for j,(v,o) in enumerate(zip(ple.vocab_sizes, offsets))])
    self.table.write(start_pos, ids, self.max_context+chunk_size)

  def __call__(self, x:Tensor, start_pos:StartPos) -> Tensor:
    assert len(start_pos) == 1, "independent positions are not supported for per-layer embeddings"
    pos = start_pos[0]
    (B, T), T_pad, n = x.shape[:2], x.max_shape[1], len(self.ple.multipliers)
    emb = self.table.read(pos, T_pad, x.device).reshape(B, T_pad, -1)[:, :T].float()
    s = (self.norm_key(self.key(emb).reshape(x.shape)) * self.norm_query(x)).sum(-1, keepdim=True) * self.dim**-0.5
    gated = self.value(emb).unsqueeze(2) * (s.sign() * s.abs().clamp(min_=1e-6).sqrt()).sigmoid()
    conv, store = short_conv(self.conv_state, self.norm_conv(gated).flatten(2), self.conv1d["weight"], Tensor(pos).eq(0), n)
    return Tensor((gated + conv[:, :T].reshape(gated.shape)).contiguous().uop.after(store))

def topk_mask(x:Tensor, k:int) -> Tensor:
  """x >= its k-th largest on the last axis (all of it if shorter, ties all in): a radix select of the floats' bits in order, a byte a pass"""
  u = x.float().bitcast(dtypes.uint32)
  key = (u >> 31).eq(1).where(u ^ 0xFFFFFFFF, u | 0x80000000).unsqueeze(-1)
  kth, need, byte = key[..., :1, :].zeros_like(), x[..., :1].full_like(k, dtype=dtypes.int32), Tensor.arange(256, dtype=dtypes.uint32)
  for shift in (24, 16, 8, 0):  # (ge: the keys under the bytes chosen so far with this byte >= each one)
    ge = ((key & (0xFFFFFFFF << (shift + 8) & 0xFFFFFFFF)).eq(kth) & ((key >> shift) & 255 >= byte)).sum(-2)
    b = ((ge >= need).sum(-1, keepdim=True) - 1).maximum(0)
    need = need - (byte.cast(b.dtype) == b + 1).where(ge, 0).sum(-1, keepdim=True)
    kth = kth | (b.cast(dtypes.uint32) << shift).unsqueeze(-1)
  return key[..., 0] >= kth[..., 0]

class Indexer:
  def __init__(self, config:TransformerConfig):
    self.idx, self.max_context = (idx:=unwrap(config.indexer)), config.max_context
    self.attn_q_b = Linear(config.q_lora_rank or config.dim, idx.n_heads * idx.head_dim, bias=False)
    self.attn_k = Linear(config.dim, idx.head_dim, bias=False)
    self.k_norm = (nn.RMSNorm if idx.msa or idx.qsa else nn.LayerNorm)(idx.head_dim, eps=idx.norm_eps)
    if idx.msa or idx.qsa: self.q_norm = nn.RMSNorm(idx.head_dim, eps=idx.norm_eps)
    else: self.proj = Linear(config.dim, idx.n_heads, bias=False)
    if idx.kpool > 1 and not (idx.msa or idx.qsa):
      self.compressor_gate, self.compressor_ape = Linear(config.dim, idx.head_dim, bias=False), {"weight": Tensor.zeros(idx.kpool, idx.head_dim)}

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_kg"):
      self.cache_kg = Tensor.zeros(x.shape[0], round_up(self.max_context, self.idx.kpool),
                                  *((2, self.idx.head_dim) if hasattr(self, "compressor_gate") else (self.idx.head_dim,)),
                                  dtype=dtypes.half, device=x.device)

  def store_keys(self, x:Tensor, start_pos:StartPos, rot:Tensor|None) -> Tensor:
    idx = self.idx
    k = self.attn_k(x) if idx.qsa else self.k_norm(self.attn_k(x))
    if rot is not None and not idx.qsa:
      k = apply_rope(k[..., :idx.rope_dim].unsqueeze(1), rot, idx.rope_interleave).squeeze(1).cat(k[..., idx.rope_dim:], dim=-1)
    if hasattr(self, "compressor_gate"): k = k.stack(self.compressor_gate(x), dim=2)
    return _update_cache(self.cache_kg, k, start_pos, 1)

  def __call__(self, x:Tensor, q:Tensor, start_pos:StartPos, freqs:Tensor|None=None) -> Tensor:
    (B, T, _), idx, S = x.shape, self.idx, ssimplify(smax(start_pos) + x.shape[1])
    P = round_up(S, idx.kpool)  # (the pools so far)
    rot = Tensor.stack(*(unwrap(freqs)[p:p+T] for p in start_pos)) if idx.rope_dim else None
    keys = self.store_keys(x, start_pos, rot)
    iq = self.attn_q_b(q).reshape(B, T, idx.n_heads, idx.head_dim)
    if idx.msa or idx.qsa: iq = self.q_norm(iq)
    if rot is not None:
      iq = apply_rope(iq[..., :idx.rope_dim].transpose(1, 2), rot, idx.rope_interleave).transpose(1, 2).cat(iq[..., idx.rope_dim:], dim=-1)
    if idx.kpool == 1 and not (idx.msa or idx.qsa) and idx.head_dim % 16 == 0 and len(start_pos) == 1 and amd_custom_kernels_supported(x.device):
      # (a key each: the positions (B, T, top_k) themselves, the attention gathers their keys)
      w = self.proj(x) * (idx.n_heads * idx.head_dim)**-0.5
      # (a symbolic T: the kernels on its max rows, the rest sliced off)
      iq, w = iq.pad_to(iq.max_shape), w.pad_to(w.max_shape)
      return topk_indices(indexer_scores(iq, keys, w, start_pos[0], S), idx.top_k, S).reshape(B, iq.shape[1], -1)[:, :T]
    if hasattr(self, "compressor_gate"):
      # cache an indexer key and a pooling gate per token, a pool's key is the gate softmax weighted sum of its keys
      kg = keys[:, :P].float().reshape(B, -1, idx.kpool, 2, idx.head_dim)
      k = ((kg[..., 1, :] + self.compressor_ape["weight"]).softmax(2) * kg[..., 0, :]).sum(2)  # (B, S, D)
    else:
      k = keys[:, :P].float()  # (B, S, D)
    if idx.qsa:
      k, rot = self.k_norm(k.reshape(B, -1, idx.kpool, idx.head_dim).mean(2)), unwrap(freqs)[:self.max_context:idx.kpool][:P // idx.kpool]
      k = apply_rope(k[..., :idx.rope_dim].unsqueeze(1), rot, idx.rope_interleave).squeeze(1).cat(k[..., idx.rope_dim:], dim=-1)
    scores = iq @ k.transpose(-1, -2).unsqueeze(1)
    if idx.msa:   scores = scores.reshape(B, T, idx.n_heads, scores.shape[-1] // idx.kpool, idx.kpool).max(-1).transpose(1, 2)
    elif idx.qsa: scores = (scores.relu() * idx.head_dim**-0.5).sum(2).unsqueeze(1)
    else:         scores = (scores.relu() * (self.proj(x) * (idx.n_heads * idx.head_dim)**-0.5).unsqueeze(-1)).sum(2).unsqueeze(1)
    # the query sees the top_k/kpool best complete pools and the tokens of its own pool while incomplete (the tail)
    q_pos = Tensor.stack(*(Tensor(p) for p in start_pos)).reshape(-1, 1, 1, 1) + Tensor.arange(T).reshape(1, 1, T, 1)
    k_pos = Tensor.arange(S)
    scores = ((Tensor.arange(scores.shape[-1]) + 1) * idx.kpool <= q_pos + 1).where(scores, -math.inf)
    if idx.msa: scores = (Tensor.arange(scores.shape[-1]) == q_pos // idx.kpool).where(math.inf, scores)
    # selected incomplete pools add only tail/future tokens, handled by the final tail and causal mask
    picked = topk_mask(scores, idx.top_k // idx.kpool).repeat_interleave(idx.kpool, dim=-1)[..., :S]
    return ((picked | (k_pos >= (q_pos + 1) // idx.kpool * idx.kpool)) & (k_pos <= q_pos)).where(0.0, -math.inf).cast(x.dtype)


def _heads(x:Tensor, w:Tensor) -> Tensor:  # (x @ w.T of each head's own weights w (H, out, in): the tensor cores' kernel where there's one)
  if (y:=head_linear(x, w)) is not None: return y
  return (x.pad_to(x.max_shape) @ w.transpose(-1, -2))[:, :, :x.shape[2]]  # (a symbolic T: the matmul on its max rows, static)

class MLATransformerBlock(FFNBlock):
  kv_b_block_size:tuple[int, int] = (128, 128)
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    qk_nope_head_dim = config.head_dim - config.rope_dim
    if config.q_lora_rank > 0:
      self.attn_q_a = Linear(config.dim, config.q_lora_rank, bias=False)
      self.attn_q_a_norm = nn.RMSNorm(config.q_lora_rank, config.norm_eps)
      self.attn_q_b = Linear(config.q_lora_rank, config.n_heads * config.head_dim, bias=False)
    else:
      self.attn_q = Linear(config.dim, config.n_heads * config.head_dim, bias=False)
    self.attn_kv_a_mqa = Linear(config.dim, config.kv_lora_rank + config.rope_dim, bias=False)
    self.attn_kv_a_norm = nn.RMSNorm(config.kv_lora_rank, config.norm_eps)
    self.attn_k_b = {"weight": Tensor.zeros(config.n_heads, config.kv_lora_rank, qk_nope_head_dim)}
    self.attn_v_b = {"weight": Tensor.zeros(config.n_heads, config.v_head_dim, config.kv_lora_rank)}
    self.attn_output = Linear(config.n_heads * config.v_head_dim, config.dim, bias=False)
    self.indexer = Indexer(config) if config.indexer else None
    if config.attn_output_gate: self.attn_gate = Linear(config.dim, config.n_heads * config.v_head_dim, bias=False)
    if config.attn_head_gate: self.attn_gate = Linear(config.dim, config.n_heads, bias=False)

  def _latents(self, x:Tensor) -> tuple[Tensor|None, Tensor]:
    if self.config.q_lora_rank == 0: return None, allgather_last(self.attn_kv_a_mqa(x), exact=True)
    q_a, kv_a = allgather_lasts(self.attn_q_a(x), self.attn_kv_a_mqa(x), exact=True)
    return self.attn_q_a_norm(q_a), kv_a

  def store_latents(self, x:Tensor, start_pos:StartPos, freqs:Tensor|None, kv_a:Tensor|None=None) -> Tensor:
    B, T, _ = x.shape
    if kv_a is None: kv_a = allgather_last(self.attn_kv_a_mqa(x), exact=True)
    c_kv = self.attn_kv_a_norm(kv_a[..., :self.config.kv_lora_rank])
    k_rope = kv_a[..., self.config.kv_lora_rank:].reshape(B, T, 1, self.config.rope_dim).transpose(1, 2)
    if freqs is not None: k_rope = apply_rope(k_rope, freqs, self.config.rope_interleave)

    k_store = c_kv.reshape(B, 1, T, self.config.kv_lora_rank).cat(k_rope.reshape(B, 1, T, self.config.rope_dim), dim=-1)
    return _update_cache(self.cache_k, k_store, start_pos, 2)

  def _attend(self, x:Tensor, start_pos:StartPos, idx_mask:Tensor|None, live:UOp|None) -> tuple[Tensor, Tensor|None]:
    latents = self._latents(x)
    mask = self._indexer_mask(x, start_pos, idx_mask, latents[0])
    return self._attention(x, start_pos, mask, live, latents), mask

  def _attention(self, x:Tensor, start_pos:StartPos, idx_mask:Tensor|None=None, live:UOp|None=None,
                 latents:tuple[Tensor|None, Tensor]|None=None) -> Tensor:
    B, T, _ = x.shape
    q_nope_head_dim = self.config.head_dim - self.config.rope_dim
    q_a, kv_a = self._latents(x) if latents is None else latents
    q_proj = self.attn_q(x) if q_a is None else self.attn_q_b(q_a)
    q = q_proj.reshape(B, T, self.config.n_heads, self.config.head_dim).transpose(1, 2)
    q_nope, q_rope = q[..., :q_nope_head_dim], q[..., q_nope_head_dim:]
    freqs = Tensor.stack(*(unwrap(self.freqs_cis)[p:p+T] for p in start_pos)) if not self.config.mla_nope else None
    if freqs is not None: q_rope = apply_rope(q_rope, freqs, self.config.rope_interleave)
    if 'weight_scale_inv' in self.attn_k_b:
      q_nope = fp8_matmul(q_nope, self.attn_k_b['weight'], self.attn_k_b['weight_scale_inv'], self.kv_b_block_size[::-1])
    else: q_nope = _heads(q_nope, self.attn_k_b['weight'])
    q = q_nope.cat(q_rope, dim=-1)

    cache = self.store_latents(x, start_pos, freqs, kv_a)
    scale = (self.config.yarn and self.config.yarn.mscale or 1.0) ** 2 / self.config.head_dim ** 0.5
    if (idx_mask is None or dtypes.is_int(idx_mask.dtype)) and len(start_pos) == 1 and mla_topk_supported(x.device, self.config.kv_lora_rank):
      attn = mla_topk_attention(q, cache, idx_mask, start_pos[0], self.config.kv_lora_rank, scale)  # (no indexer: every key there is)
    elif idx_mask is not None and dtypes.is_int(idx_mask.dtype):
      # the indexer's positions (B, T, top_k): each token's keys gathered (the slots past the keys there are the first key's, masked as its future)
      S, K = ssimplify(smax(start_pos) + T), idx_mask.shape[-1]
      pos = (Tensor.arange(K) < S).where(idx_mask, 0)
      keys = cache[:, 0].gather(1, pos.reshape(B, -1, 1).expand(B, T*K, cache.shape[-1])).reshape(B, 1, T, K, -1).contiguous()
      q_pos = Tensor.stack(*(Tensor(p) for p in start_pos)).reshape(-1, 1, 1) + Tensor.arange(T).reshape(1, T, 1)
      mask = ((Tensor.arange(K) < S) & (pos <= q_pos)).where(0.0, -math.inf).reshape(B, 1, T, 1, K)
      attn = ((q.unsqueeze(-2) @ keys.transpose(-1, -2) * scale + mask).softmax(-1) @ keys[..., :self.config.kv_lora_rank]).squeeze(-2).contiguous()
    else:
      k = _cache_prefix(cache, start_pos, T, 2)
      attn = q @ k.transpose(-1, -2) * scale
      mask = idx_mask if idx_mask is not None else _causal_mask(x, start_pos)
      if mask is not None: attn = attn + mask
      attn = attn.softmax(-1) @ k[..., :self.config.kv_lora_rank]
    if 'weight_scale_inv' in self.attn_v_b:
      attn = fp8_matmul(attn, self.attn_v_b['weight'], self.attn_v_b['weight_scale_inv'], self.kv_b_block_size)
    else: attn = _heads(attn, self.attn_v_b['weight'])
    attn = attn.transpose(1, 2).reshape(B, T, -1)
    if self.config.attn_head_gate: return self.attn_output(attn * self.attn_gate(x).sigmoid().repeat_interleave(self.config.v_head_dim, -1))
    return self.attn_output(attn * self.attn_gate(x).sigmoid() if self.config.attn_output_gate else attn)

  def _indexer_mask(self, x:Tensor, start_pos:StartPos, idx_mask:Tensor|None, q_a:Tensor|None=None) -> Tensor|None:
    # a full indexer layer selects the top-k tokens for itself and the shared layers after it
    if self.indexer is not None and self.config.max_context > self.indexer.idx.top_k:
      if q_a is None: q_a = self.attn_q_a_norm(allgather_last(self.attn_q_a(x), exact=True))
      return self.indexer(x, q_a, start_pos, self.freqs_cis)
    return idx_mask

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_k"):
      self.cache_k = Tensor.empty(x.shape[0], 1, self.config.max_context, self.config.kv_lora_rank + self.config.rope_dim, dtype=dtypes.half,
                                  device=x.device)
      needs_rope = not self.config.mla_nope or (self.indexer is not None and self.indexer.idx.rope_dim > 0)
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta,
                                          device=x.device, yarn=self.config.yarn) if needs_rope else None
    if self.indexer: self.indexer._init_state(x)

class MTPBlock(MLATransformerBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    self.eh_proj = Linear(2 * config.dim, config.dim, bias=False)
    self.enorm, self.hnorm, self.shared_head_norm = (nn.RMSNorm(config.dim, config.norm_eps) for _ in range(3))

  def project(self, emb:Tensor, hidden:Tensor) -> Tensor: return self.eh_proj(self.enorm(emb).cat(self.hnorm(hidden), dim=-1))

  def fill_cache(self, x:Tensor, pos:int|UOp) -> Tensor:
    rot = self.freqs_cis[pos:pos+x.shape[1]].unsqueeze(0) if self.freqs_cis is not None else None
    cache = self.store_latents(x, (pos,), None if self.config.mla_nope else rot)
    if self.indexer is not None and self.config.max_context > self.indexer.idx.top_k:
      return cache.realize(self.indexer.store_keys(x, (pos,), rot if self.indexer.idx.rope_dim else None))
    return cache

class GatedDeltaNetBlock(FFNBlock):
  def __init__(self, config:TransformerConfig, ssm:SSMConfig):
    super().__init__(config)
    self.head_k_dim, self.num_k_heads, self.num_v_heads = ssm.state_size, ssm.group_count, ssm.time_step_rank
    assert self.num_v_heads % self.num_k_heads == 0
    self.head_v_dim, self.ssm_conv_kernel = ssm.inner_size // ssm.time_step_rank, ssm.conv_kernel
    self.conv_channels, self.q_dim = ssm.inner_size + 2*ssm.group_count*ssm.state_size, ssm.state_size*ssm.group_count
    if ssm.kda:
      self.qkv_dims = (self.q_dim, self.q_dim, ssm.inner_size)
      self.attn_q, self.attn_k, self.attn_v = (Linear(config.dim, d, bias=False) for d in self.qkv_dims)
      self.ssm_conv1d_q, self.ssm_conv1d_k, self.ssm_conv1d_v = ({"weight": Tensor.zeros(d, 1, ssm.conv_kernel)} for d in self.qkv_dims)
      if ssm.full_rank[0]: self.ssm_g = Linear(config.dim, ssm.inner_size, bias=False)
      else: self.ssm_g_a, self.ssm_g_b = Linear(config.dim, self.head_v_dim, bias=False), Linear(self.head_v_dim, ssm.inner_size, bias=False)
      if ssm.full_rank[1]: self.ssm_f = Linear(config.dim, ssm.inner_size, bias=False)
      else: self.ssm_f_a, self.ssm_f_b = Linear(config.dim, self.head_k_dim, bias=False), Linear(self.head_k_dim, ssm.inner_size, bias=False)
    else:
      self.attn_qkv = Linear(config.dim, self.conv_channels, bias=False)
      self.ssm_conv1d = {"weight": Tensor.zeros(self.conv_channels, self.ssm_conv_kernel)}
      self.attn_gate = Linear(config.dim, ssm.inner_size, bias=False)
      self.ssm_alpha = Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_beta = Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_dt = {"bias": Tensor.zeros(ssm.inner_size if ssm.kda else self.num_v_heads)}
    self.ssm_a = Tensor.zeros(self.num_v_heads, 1) if ssm.kda else Tensor.zeros(self.num_v_heads)
    self.ssm_norm, self.ssm_out = nn.RMSNorm(self.head_v_dim, config.norm_eps), Linear(ssm.inner_size, config.dim, bias=False)
    self.kept:list[list[Tensor]] = []
    self.keep = 0  # (a speculative decoder's short steps: the states kept after up to keep of their tokens)

  # a short step (as a speculative decoding verify) of a block asked to (keep) keeps the states after its first keep tokens, for Transformer.rollback
  def _kept(self, T:int|UOp) -> int: return min(self.keep, T-1) if isinstance(T, int) else 0
  def _states(self) -> list[Tensor]: return [*(self.conv_states if unwrap(self.config.ssm).kda else [self.conv_state]), self.recurrent_state]

  def _attention(self, x:Tensor, start_pos:StartPos, idx_mask:Tensor|None=None, live:UOp|None=None) -> Tensor:
    B, T, _ = x.shape
    assert len(start_pos) == 1, "independent positions are not supported for recurrent blocks"
    # bind ints to a variable so the reset flag stays a runtime value (it toggles when generation restarts at position 0)
    pos = start_pos[0]
    pos = pos if isinstance(pos, UOp) else UOp.variable("start_pos", 0, self.config.max_context-1).bind(pos)
    initial = Tensor(pos).eq(0)
    is_kda = unwrap(self.config.ssm).kda
    symbolic = isinstance(T, UOp)
    T_pad = x.max_shape[1]  # symbolic chunks are padded to their max size: one graph serves every size

    # input processing
    x = x.half()
    if is_kda and B == 1 and not symbolic and live is None and self.num_k_heads == self.num_v_heads and \
       (lb:=unwrap(self.config.ssm).gate_lower_bound) is not None and not unwrap(self.config.ssm).silu_gate and \
       hasattr(self, "ssm_g") and self.head_k_dim == self.head_v_dim and self.head_k_dim % 32 == 0 and T - 1 <= len(self.kept) and \
       amd_custom_kernels_supported(x.device):
      # a token at a time, the rest of the block in one kernel: token t reads the states after t-1 and writes those after t (kept, the
      # last token's the block's)
      q, k, v, g, b = (f(x) for f in (self.attn_q, self.attn_k, self.attn_v, self.ssm_g, self.ssm_beta))
      f = self.ssm_f(x) if hasattr(self, "ssm_f") else self.ssm_f_b(self.ssm_f_a(x))
      # (a kept recurrent state the kept buffer's slot t)
      sts = [[self.recurrent_state, *self.conv_states], *[[self.kept_states, *s[:-1]] for s in self.kept[:T-1]]]
      conv_w = [w["weight"].flatten(1) for w in (self.ssm_conv1d_q, self.ssm_conv1d_k, self.ssm_conv1d_v)]
      zq:tuple[Tensor, ...] = ()
      src = sts[0] if T > 1 else None
      for t in range(T):
        zq, src = kda_decode(q, k, v, g, b, f, conv_w, self.ssm_dt["bias"], self.ssm_a, unwrap(self.ssm_norm.weight), pos + t, lb,
                             self.ssm_norm.eps, sts[(t + 1) % T], src, t, zq, t if t < T - 1 else 0, max(t - 1, 0))
      # (the out projection on the kernel's q8 of z)
      return q8_linear(self.ssm_out, zq[0], zq[1:]) if self.ssm_out.ggml_type in QUANT_SIZES else self.ssm_out(zq[0])
    out_gate = (self.ssm_g(x) if hasattr(self, "ssm_g") else self.ssm_g_b(self.ssm_g_a(x))) if is_kda else self.attn_gate(x)
    out_gate = out_gate.reshape(B, T, self.num_v_heads, self.head_v_dim)
    beta = self.ssm_beta(x).float().sigmoid().reshape(B, T, self.num_v_heads)  # (in float: a symbolic chunk's isn't fused, kept in half)
    alpha = (self.ssm_f(x) if hasattr(self, "ssm_f") else self.ssm_f_b(self.ssm_f_a(x))) if is_kda else self.ssm_alpha(x)
    g, a = (alpha.float() + self.ssm_dt["bias"]).reshape(B, T, self.num_v_heads, -1), self.ssm_a.reshape(self.num_v_heads, -1)
    # the decay: log_alpha = softplus(g) * A, or bounded by gate_lower_bound: sigmoid(g * |A|) * lb
    log_alpha = g.softplus() * a if (lb:=unwrap(self.config.ssm).gate_lower_bound) is None else (g * a.abs()).sigmoid() * lb

    # qkv conv, conv_state is reset when starting from position 0
    projs = (self.attn_q, self.attn_k, self.attn_v) if is_kda else (self.attn_qkv,)
    convs = (self.ssm_conv1d_q, self.ssm_conv1d_k, self.ssm_conv1d_v) if is_kda else (self.ssm_conv1d,)
    states, kept = self._states()[:-1], self.kept[:self._kept(T)]
    conv_ins = [proj(x) for proj in projs]
    conv_outs, conv_state_stores = zip(*(short_conv(s, p, w["weight"].flatten(1), initial, live=live) for s, p, w in zip(states, conv_ins, convs)))
    # the kept conv states: the rows of [conv_state | x rows] after each token
    kept_stores = [ks.uop.store(initial.where(0, s).cat(p.cast(s.dtype), dim=1)[:, t+1:t+1+s.shape[1]].uop)
                   for t, kt in enumerate(kept) for ks, s, p in zip(kt, states, conv_ins)]
    # padded steps are exact no-ops: beta=0 (delta rule off), log_alpha=0 (decay 1 after exp)
    if symbolic:
      out_gate = out_gate.pad_to((B, T_pad, self.num_v_heads, self.head_v_dim))
      beta, log_alpha = beta.pad_to((B, T_pad, self.num_v_heads)), log_alpha.pad_to((B, T_pad, *log_alpha.shape[2:]))
    q, k, v = conv_outs if is_kda else conv_outs[0].split([self.q_dim, self.q_dim, self.conv_channels - 2*self.q_dim], dim=-1)
    q, k = (z.reshape(B, T_pad, self.num_k_heads, self.head_k_dim) for z in (q, k))
    # l2 norm x/sqrt(sum(x²)+1e-6) as in the FLA kernels, then the v heads share the k heads
    q, k = ((z.rmsnorm(eps=1e-6/self.head_k_dim) / self.head_k_dim**0.5).repeat(1, 1, self.num_v_heads//self.num_k_heads, 1) for z in (q, k))
    v = v.reshape(B, T_pad, self.num_v_heads, self.head_v_dim)
    # layout the per-step operands to broadcast against the (B, H, V, K) state
    q, k, v, beta = (z.transpose(1, 2).float() for z in (q, k, v, beta))
    q = q * self.head_k_dim**-0.5
    alpha = log_alpha.transpose(1, 2).exp()  # per-channel decay for kda, per-head otherwise (B, H, T, K|1)
    # mask the fp32 coefficients to preserve the projections' fp16 rounding
    if live is not None:
      valid = (Tensor.arange(T_pad).to(x.device) < Tensor(live, device=x.device)).reshape(1, 1, T_pad)
      beta, alpha = valid.where(beta, 0), valid.unsqueeze(-1).where(alpha, 1)

    # recurrent: scan over the (padded) tokens, updating the recurrent state. collect the per-step outputs
    if self.head_k_dim % 32 == 0 and self.head_v_dim % 4 == 0 and amd_custom_kernels_supported(x.device):
      # one fused kernel for the whole scan; it resets and updates the recurrent state in place (RDNA3/4), writes the kept ones
      state = Tensor(self.recurrent_state.uop.after(*conv_state_stores, *kept_stores))  # carry the conv writes into this graph
      # Pass the whole kept buffer: shrinking it would redirect the kernel's writes into a temporary.
      core = gated_delta_prefill(q, k, v, beta, alpha, state, Tensor(pos), self.kept_states if kept else None,
                                 T if symbolic else None).transpose(1, 2)
    else:
      q, k, v, beta = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)
      alpha = alpha.unsqueeze(-2)
      state = initial.where(0, self.recurrent_state.float())
      outs = []
      for t in range(T_pad):
        s1 = state * alpha[:, :, t]  # decay the state
        delta = (v[:, :, t] - (s1*k[:, :, t]).sum(-1, keepdim=True)) * beta[:, :, t]  # the delta rule update
        state = s1 + delta * k[:, :, t]
        outs.append((state * q[:, :, t]).sum(-1))
        if t < len(kept): kept_stores.append(kept[t][-1].uop.store(state.cast(kept[t][-1].dtype).uop))

      # store the updated recurrent state in place, then read the stacked outputs after the state and conv writes
      state_store = self.recurrent_state.uop.after(self.recurrent_state.uop.store(state.cast(self.recurrent_state.dtype).uop))
      conv_writes = [c.uop.after(st) for c, st in zip(states, conv_state_stores)]
      core = Tensor(outs[0].stack(*outs[1:], dim=1).contiguous().uop.after(state_store, *conv_writes, *kept_stores))

    # output; undo the padding before the output projection
    z = (self.ssm_norm(core) * (out_gate.silu() if unwrap(self.config.ssm).silu_gate else out_gate.sigmoid())).cast(x.dtype).contiguous()
    if symbolic: z = z[:, :T]
    return self.ssm_out(z.reshape(B, T, -1))

  def _init_state(self, x):
    if not hasattr(self, "recurrent_state"):
      if kda:=unwrap(self.config.ssm).kda:
        self.conv_states = [Tensor.zeros(x.shape[0], self.ssm_conv_kernel-1, d, device=x.device, axis=-1).clone() for d in self.qkv_dims]
      else: self.conv_state = Tensor.zeros(x.shape[0], self.ssm_conv_kernel-1, self.conv_channels, device=x.device).clone()
      self.recurrent_state = Tensor.zeros(x.shape[0], self.num_v_heads, self.head_v_dim, self.head_k_dim, device=x.device,
                                          axis=1 if kda else None).clone()
    if self.keep and not self.kept:  # (the kept recurrent states in one buffer: the fused scan writes them)
      shape, axis = self.recurrent_state.shape, self.recurrent_state.uop.axis
      self.kept_states = Tensor.empty(self.keep, *shape, device=x.device, axis=None if axis is None else axis+1).realize()
      self.kept = [[*(s.empty_like() for s in self._states()[:-1]), self.kept_states[t]] for t in range(self.keep)]

class Transformer:
  def __init__(self, config:TransformerConfig):
    dense_config = replace(config, num_experts=0, num_experts_per_tok=0, shared_expert_dim=0, hidden_dim=config.dense_hidden_dim or config.hidden_dim)
    if config.ssm: config = replace(config, qk_norm=config.head_dim)
    self.config = config
    block_cls = MLATransformerBlock if config.kv_lora_rank > 0 else TransformerBlock
    self.blk:list[FFNBlock] = []
    for i in range(config.num_blocks):
      c = dense_config if i < config.leading_dense_blocks else config
      if config.sliding_window_pattern != 0 and (i+1) % config.sliding_window_pattern == 0: c = replace(c, sliding_window=0)
      if config.indexer_shared and config.indexer_shared[i]: c = replace(c, indexer=None)
      if config.attn_res_block_size and i % config.attn_res_block_size == 0: c = replace(c, attn_res_new_block=True)
      if config.ple and i != config.ple.layer: c = replace(c, ple=None)
      self.blk.append(GatedDeltaNetBlock(c, config.ssm) if config.ssm and config.ssm_layers[i] else block_cls(c))
    self.token_embd  = nn.Embedding(config.vocab_size, config.dim)
    self.output_norm: Callable[[Tensor], Tensor] = nn.RMSNorm(config.dim, config.norm_eps)
    if config.hc and config.hc.low_rank: self.output_hc, self.output_norm = GatedResidual(config, inject=False), lambda x: x
    if config.attn_res_block_size: self.output_res_score = AttentionResidual(config)
    self.output = Linear(config.dim, config.vocab_size, bias=False)
    if config.mtp: self.mtp = MTPBlock(config)
    self.max_context = config.max_context
    self.has_recurrent_block = any(isinstance(b, GatedDeltaNetBlock) for b in self.blk)
    self._cached_tokens: list[int] = []
    # (prompts fed, or where they diverged from the tokens fed before, and which of the two state sets has the recurrent states after them)
    self._saved: list[tuple[list[int], int]] = []
    self._sets: list[list[Tensor]] = []
    self._position_count: int|None = None
    # we specialize the JIT for prefill and rollout
    self.prefill_jit = TinyJit(self.forward)
    self.rollout_jit = TinyJit(self.forward)
    self._rollbacks: dict[int, Callable] = {}
    self._copies: dict[tuple[int, bool], Callable] = {}

  def forward(self, tokens:Tensor, temperature:Tensor, *start_pos:int|UOp) -> Tensor:
    x = self.token_embd(tokens.to(self.token_embd.weight.device)).float()  # (B, T, D)
    if (hc:=self.config.hc) is not None: x = x.unsqueeze(2).expand(*x.shape[:2], hc.count, x.shape[-1]).contiguous()
    elif self.config.attn_res_block_size: x = x.unsqueeze(2)
    idx_mask = None
    for block in self.blk: x, idx_mask = block(x, start_pos, idx_mask)
    if hc is not None: x = self.output_hc.mix(x)[0] if hc.low_rank else x.mean(2)
    elif self.config.attn_res_block_size: x = self.output_res_score.mix(x)
    # only run the output projection on the last token
    logits = self.output(self.output_norm(x[:, -1:]))[:, -1, :].to(tokens.device)
    # Gumbel-max trick: argmax(logits/temp - log(-log(uniform))) is equivalent to sampling from softmax(logits/temp)
    return (logits / temperature.maximum(1e-12) - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)

  def rollback(self, n:int):
    """after a short step, the recurrent states go back to after its first n tokens (attention caches are rewritten from start_pos)"""
    if not (blocks:=[b for b in self.blk if isinstance(b, GatedDeltaNetBlock)]): return
    assert all(1 <= n <= len(b.kept) for b in blocks), f"rollback({n}) after a step that kept {[len(b.kept) for b in blocks]}"
    def copy(): Tensor.realize(*[s.assign(k) for b in blocks for s, k in zip(b._states(), b.kept[n-1])])
    if n not in self._rollbacks: self._rollbacks[n] = TinyJit(copy, capture_first=True)  # (a copy a state, scheduled once)
    self._rollbacks[n]()

  def _states(self) -> list[Tensor]: return [s for b in self.blk if isinstance(b, GatedDeltaNetBlock) for s in b._states()]
  def save(self, tokens:list[int]):
    """the recurrent states after tokens (a prompt fed, or its part before it diverged from the tokens fed before: fork) kept, with the deepest
    kept ones tokens extend: get_start_pos restores the deepest a prompt extends"""
    if not (states:=self._states()): return
    if not self._sets:
      self._sets = [[s.empty_like() for s in states] for _ in range(2)]
      Tensor.realize(*[s for st in self._sets for s in st])
    keep = [(t, i) for t, i in self._saved if len(t) < len(tokens) and tokens[:len(t)] == t][-1:]
    i = next(j for j in range(2) if all(j != k for _, k in keep))
    self._copy(i, save=True)
    self._saved = keep + [(tokens[:], i)]

  def _copy(self, i:int, save:bool):
    """the recurrent states into saved set i (save) or back from it"""
    def copy(): Tensor.realize(*[k.assign(s) if save else s.assign(k) for k, s in zip(self._sets[i], self._states())])
    if (i, save) not in self._copies: self._copies[i, save] = TinyJit(copy, capture_first=True)  # (a copy a state, scheduled once)
    self._copies[i, save]()

  def fork(self, tokens:list[int], start:int, step:int) -> int:
    """where (start and whole steps of a feed) tokens diverge from the tokens fed before, its states to be saved there for a later prompt that
    shares only that part (start: nowhere)"""
    n = sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], zip(tokens, self._cached_tokens)))
    return f if self.has_recurrent_block and start < (f:=start + (n - start) // step * step) < len(tokens) else start

  def __call__(self, tokens:Tensor, start_pos:StartPos, temperature:Tensor) -> Tensor:
    assert len(start_pos) in (1, tokens.shape[0]), "expected one shared position or one position per batch row"
    if self._position_count is None: self._position_count = len(start_pos)
    assert len(start_pos) == self._position_count, "position count is fixed for this model"
    positions = tuple(UOp.variable(f"start_pos_{b}", 0, self.max_context-1).bind(p) if isinstance(p, int) else p for b, p in enumerate(start_pos))
    return (self.prefill_jit if resolve(tokens.shape[1] != 1) else self.rollout_jit)(tokens.contiguous(), temperature, *positions)

  @staticmethod
  def from_gguf(gguf:Tensor|str|pathlib.Path, max_context:int|None=None,
                realize=bool(getenv("REALIZE", 0)), shard:int=1) -> tuple[Transformer, dict]:
    # TODO: remove the need for copy to default device
    kv, entries = gguf_parse(gguf.to(None).realize() if isinstance(gguf, Tensor) else gguf)
    arch = kv['general.architecture']
    # kimi-k31 is kimi-k3 plus dsa indexers
    if arch == 'kimi-k31': arch, kv = 'kimi-k3', {k.replace('kimi-k31.', 'kimi-k3.', 1): v for k, v in kv.items()}
    kda, mla_nope = arch in ('kimi-linear', 'glm5-next', 'kimi-k3', 'bailingmoe3'), arch in ('kimi-linear', 'glm5-next', 'kimi-k3')
    ple_entry = entries.pop('per_layer_token_embd.weight', None)
    if arch == 'bailingmoe3': entries = {k.replace('ssm_f_a', 'ssm_f').replace('ssm_g_a', 'ssm_g'):v for k,v in entries.items()}
    n_heads, n_kv_heads = kv[f'{arch}.attention.head_count'], kv[f'{arch}.attention.head_count_kv']
    assert shard >= 1, f"shard must be at least 1, got {shard}"
    rope_permuted = not kda and arch != "glm-dsa"
    shard_map, repeats = {}, {}
    if shard > 1:
      if not kv.get(f'{arch}.attention.kv_lora_rank') and n_kv_heads < shard and shard % n_kv_heads == 0:
        # too few KV heads to split: repeat each one so every device holds one
        repeats = {k: shard // n_kv_heads for k in entries if k.split('.', 2)[-1] in ('attn_k.weight', 'attn_v.weight', 'attn_k.bias', 'attn_v.bias')}
        n_kv_heads = shard
      heads = n_heads if kv.get(f'{arch}.attention.kv_lora_rank') else n_kv_heads
      assert heads % shard == 0, f"tensor parallel needs the attention heads to split over {shard} devices"
      # shard MLA heads, routed experts and their latent projections (gathered after) while replicating the other latent projections, KV
      # cache and shared experts
      rules = {**{w: 0 for w in ('token_embd.weight', 'output.weight', 'attn_q.weight', 'attn_k.weight', 'attn_v.weight', 'ffn_gate.weight',
        'ffn_up.weight', 'attn_q_a.weight', 'attn_q_b.weight', 'attn_k_b.weight', 'attn_v_b.weight', 'ffn_routed_down.weight',
        'ffn_routed_up.weight')},
        **{w: 1 for w in ('attn_output.weight', 'ffn_down.weight', 'ffn_gate_exps.weight', 'ffn_up_exps.weight')}, 'ffn_down_exps.weight':2}
      if not rope_permuted: rules['attn_kv_a_mqa.weight'] = 0
      if kda: rules.update({w: 0 for w in ('ssm_conv1d_q.weight', 'ssm_conv1d_k.weight', 'ssm_f.weight',
        'ssm_conv1d_v.weight', 'ssm_dt.bias', 'ssm_a', 'ssm_beta.weight', 'ssm_f_b.weight', 'ssm_g.weight', 'ssm_g_b.weight', 'attn_gate.weight')})
      if all(shape[-1] % (shard*_GGML_QUANT.get(typ, (1,))[0]) == 0 for k,(_,shape,typ) in entries.items() if 'ffn_down_shexp.' in k):
        rules.update({'ffn_gate_shexp.weight': 0, 'ffn_up_shexp.weight': 0, 'ffn_down_shexp.weight': 1})
      shard_map = {name: rules[k] for name in entries if (k:=re.sub(r"^blk\.\d+\.", "", name)) in rules}
      for name in entries:
        if name.endswith('.nextn.eh_proj.weight'): shard_map.pop(name.removesuffix('nextn.eh_proj.weight') + 'attn_kv_a_mqa.weight', None)
    devices = tuple(Device.canonicalize(f'{Device.DEFAULT}:{i}') for i in range(shard))
    ep = any(shape[-1] % (shard*_GGML_QUANT.get(typ, (1,))[0]) for k,(_,shape,typ) in entries.items() if 'ffn_down_exps.' in k)
    state_dict = gguf_shard(entries, devices, {k: 0 if ep and '_exps.' in k else v for k,v in shard_map.items()}, repeats)

    # all state items should be float16, not float32
    state_dict = {k:v.cast('float16') if getenv("HALF", 1) else v for k,v in state_dict.items()}
    for _, names in itertools.groupby(state_dict, lambda k: k.split(".")[1] if k.startswith("blk.") else k):
      if native:=[state_dict[k] for k in names if entries[k][2] in _GGML_NATIVE]: Tensor.realize(*native)

    # some models like Llama 3.2 don't have an output.weight, they just tie to the token_embd.weight
    if 'output.weight' not in state_dict: state_dict['output.weight'] = state_dict['token_embd.weight']

    hc = HCConfig(kv[f'{arch}.hyper_connection.count'], kv.get(f'{arch}.hyper_connection.sinkhorn_iterations', 0),
                  kv.get(f'{arch}.hyper_connection.epsilon', 0.0),
                  kv.get(f'{arch}.hyper_connection.low_rank', 0)) if f'{arch}.hyper_connection.count' in kv else None
    if hc is not None: state_dict = {re.sub(r'(hc_attn|hc_ffn|output_hc|\.ple)_', r'\1.', k):v for k,v in state_dict.items()}

    kpool = kv.get(f'{arch}.attention.indexer.block_size' if arch == 'minimax-m3' else f'{arch}.attention.indexer.kpool',
                   1 if f'{arch}.attention.indexer.top_k' in kv else None)
    if arch == 'qwen4exp': kpool = max(kv[f'{arch}.attention.compress_ratios'])
    indexer = IndexerConfig(kv[f'{arch}.attention.indexer.head_count'], kv[f'{arch}.attention.indexer.key_length'],
                            kv[f'{arch}.attention.indexer.top_k'], kpool,
                            kv.get(f'{arch}.attention.layer_norm_epsilon') or kv[f'{arch}.attention.layer_norm_rms_epsilon']) if kpool else None
    if indexer is not None: state_dict = {k.replace('.indexer_compressor_', '.indexer.compressor_'):v for k,v in state_dict.items()}
    if arch == 'glm-dsa': indexer = replace(unwrap(indexer), rope_dim=kv[f'{arch}.rope.dimension_count'], norm_eps=1e-6)
    if arch in ('minimax-m3', 'qwen4exp'):
      indexer = replace(unwrap(indexer), rope_dim=kv[f'{arch}.rope.dimension_count'], rope_interleave=False, msa=arch == 'minimax-m3',
                        qsa=arch == 'qwen4exp', top_k=kv[f'{arch}.attention.indexer.top_k'] * kv.get(f'{arch}.attention.indexer.block_size', 1))
      state_dict = {k.replace('.indexer.q_proj', '.indexer.attn_q_b').replace('.indexer.k_proj', '.indexer.attn_k'):v for k,v in state_dict.items()}

    max_context = min(max_context, kv[f'{arch}.context_length']) if max_context is not None else kv[f'{arch}.context_length']

    num_blocks = kv[f'{arch}.block_count'] - kv.get(f'{arch}.nextn_predict_layers', 0)
    ssm = None
    ssm_layers: tuple[bool, ...] = ()
    if arch in ('qwen35', 'qwen35moe', 'qwen4exp'):
      ssm = SSMConfig(**{k: kv[f'{arch}.ssm.{k}'] for k in ('conv_kernel','state_size','group_count','time_step_rank','inner_size')},
                      silu_gate=arch != 'qwen4exp')
      ssm_layers = tuple((i+1) % kv[f'{arch}.full_attention_interval'] != 0 for i in range(num_blocks))
    elif kda:
      ssm_layers = tuple(x == 0 for x in n_kv_heads)[:num_blocks]
      n_kv_heads = max(n_kv_heads)
      ssm = SSMConfig(kv[f'{arch}.ssm.conv_kernel'], kv[f'{arch}.kda.head_dim'], n_heads, n_heads, n_heads*kv[f'{arch}.kda.head_dim'], kda=True,
                      gate_lower_bound=kv.get(f'{arch}.kda.gate_lower_bound'), full_rank=(arch in ('kimi-k3', 'bailingmoe3'), arch == 'bailingmoe3'))
      for i, is_ssm in enumerate(ssm_layers):
        if not is_ssm: continue
        state_dict[f"blk.{i}.ssm_a"] = state_dict[f"blk.{i}.ssm_a"].reshape(n_heads, 1)
        state_dict[f"blk.{i}.ssm_out.weight"] = state_dict.pop(f"blk.{i}.attn_output.weight")
    if arch in ('qwen35', 'qwen35moe', 'glm4moe', 'gpt-oss'):
      state_dict = {k.replace('post_attention_norm', 'ffn_norm'):v for k,v in state_dict.items()}

    kv_lora_rank = kv.get(f'{arch}.attention.kv_lora_rank', 0)
    if (mtp:=kv_lora_rank > 0 and f'blk.{num_blocks}.nextn.eh_proj.weight' in state_dict):
      state_dict = {re.sub(rf'^blk\.{num_blocks}\.(nextn\.)?', 'mtp.', k):v for k,v in state_dict.items()}
    head_dim = kv.get(f'{arch}.attention.key_length_mla', kv.get(f'{arch}.attention.key_length', kv[f'{arch}.embedding_length'] // n_heads))
    rope_dim, log_mul = kv.get(f'{arch}.rope.dimension_count', head_dim), kv.get(f'{arch}.rope.scaling.yarn_log_multiplier', 0.0)
    mscale = log_mul * math.log(kv.get(f'{arch}.rope.scaling.factor', 1.0)) + 1.0 if kv_lora_rank or arch.startswith('deepseek') else None
    yarn = YaRNConfig(factor=kv[f'{arch}.rope.scaling.factor'],
                      orig_ctx_len=kv.get(f'{arch}.rope.scaling.original_context_length', kv[f'{arch}.context_length']),
                      beta_fast=kv.get(f'{arch}.rope.scaling.yarn_beta_fast', 32.0), mscale=mscale,
                      beta_slow=kv.get(f'{arch}.rope.scaling.yarn_beta_slow', 1.0)) if kv.get(f'{arch}.rope.scaling.type') == 'yarn' else None

    # Permute RoPE weights from interleaved to half-split layout.
    for name in state_dict:
      if not rope_permuted or '.indexer.' in name: continue
      if ('attn_q.weight' in name or 'attn_q_b.weight' in name) and (arch == 'llama' or kv_lora_rank):
        state_dict[name] = permute_rope(state_dict[name], n_heads, rope_dim)
      elif arch == 'llama' and 'attn_k.weight' in name:
        state_dict[name] = permute_rope(state_dict[name], n_kv_heads, state_dict[name].shape[0]//n_kv_heads)
      elif kv_lora_rank and 'attn_kv_a_mqa.weight' in name:
        state_dict[name] = permute_rope(state_dict[name], 1, state_dict[name].shape[0]-kv_lora_rank)
    swiglu_oai = arch in ('gpt-oss', 'minimax-m3')
    config = TransformerConfig(
      num_blocks=num_blocks, dim=kv[f'{arch}.embedding_length'],
      hidden_dim=kv.get(f'{arch}.expert_feed_forward_length', kv.get(f'{arch}.feed_forward_length', 0)),
      n_heads=n_heads, n_kv_heads=n_kv_heads, norm_eps=kv[f'{arch}.attention.layer_norm_rms_epsilon'],
      vocab_size=len(kv['tokenizer.ggml.tokens']),
      head_dim=head_dim,
      rope_theta=kv.get(f'{arch}.rope.freq_base', 10000.0), rope_dim=rope_dim, yarn=yarn,
      v_head_dim=kv.get(f'{arch}.attention.value_length_mla', kv.get(f'{arch}.attention.value_length', head_dim)),
      max_context=max_context,
      qk_norm=int(state_dict['blk.0.attn_q_norm.weight'].shape[0]) if 'blk.0.attn_q_norm.weight' in state_dict else 0,
      num_experts=kv.get(f'{arch}.expert_count', 0), num_experts_per_tok=kv.get(f'{arch}.expert_used_count', 0),
      norm_topk_prob=kv.get(f'{arch}.expert_weights_norm', arch in ('qwen3moe', 'qwen35moe', 'kimi-linear', 'qwen4exp')),
      expert_gating_func=ExpertGating(kv.get(f'{arch}.expert_gating_func',
        ExpertGating.SOFTMAX_WEIGHT if arch == 'gpt-oss' else ExpertGating.SOFTMAX)),
      expert_group_count=kv.get(f'{arch}.expert_group_count', 0), expert_group_used=kv.get(f'{arch}.expert_group_used_count', 0),
      kv_lora_rank=kv_lora_rank, q_lora_rank=kv.get(f'{arch}.attention.q_lora_rank', 0),
      leading_dense_blocks=kv.get(f'{arch}.leading_dense_block_count', 0),
      shared_expert_dim=kv.get(
        f'{arch}.expert_shared_feed_forward_length',
        kv.get(f'{arch}.expert_shared_count', 0) * kv.get(f'{arch}.expert_feed_forward_length', 0)),
      shared_expert_gate=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.ffn_gate_inp_shexp.weight" in state_dict,
      dense_hidden_dim=kv.get(f'{arch}.feed_forward_length', 0) if kv.get(f'{arch}.leading_dense_block_count', 0) else 0,
      routed_scaling_factor=kv.get(f'{arch}.expert_weights_scale', 1.0), attn_output_gate=arch in ('qwen35', 'qwen35moe', 'kimi-k3', 'qwen4exp'),
      ssm=ssm, ssm_layers=ssm_layers, attn_head_gate=arch == 'bailingmoe3', rope_interleave=arch in ('bailingmoe3', 'glm-dsa'), mla_nope=mla_nope,
      qkv_bias='blk.0.attn_q.bias' in state_dict,
      expert_bias=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.exp_probs_b.bias" in state_dict,
      expert_proj_bias='blk.0.ffn_gate_exps.bias' in state_dict, attn_output_bias='blk.0.attn_output.bias' in state_dict,
      swiglu_alpha=1.702 if swiglu_oai else 1.0, swiglu_limit=7.0 if swiglu_oai else kv.get(f'{arch}.swiglu_clamp_exp', [None])[0],
      swiglu_up_bias=1.0 if swiglu_oai else 0.0, attn_sinks='blk.0.attn_sinks.weight' in state_dict,
      sliding_window=kv.get(f'{arch}.attention.sliding_window', 0),
      sliding_window_pattern=kv.get(f'{arch}.attention.sliding_window_pattern', 2 if arch == 'gpt-oss' else 0), hc=hc, indexer=indexer,
      # without indexer.types, GLM-5.2+ (1M context) runs the indexer on blocks 0-2 and every 4th after (6, 10, ..), GLM-5/5.1 on all
      indexer_shared=tuple(not full for full in kv[f'{arch}.attention.indexer.types']) if f'{arch}.attention.indexer.types' in kv else
        tuple(i > 2 and i % 4 != 2 for i in range(num_blocks)) if arch == 'glm-dsa' and kv[f'{arch}.context_length'] >= 1048576 else (),
      ple=PLEConfig(kv[f'{arch}.ple.layers'][0], ple_entry[1][1], tuple(kv[f'{arch}.ple.layer_multipliers']),
                    tuple(kv[f'{arch}.ple.head_vocab_sizes']), kv[f'{arch}.ple.eos_token_id'], kv[f'{arch}.ple.conv_kernel']) if ple_entry else None,
      expert_latent_dim=kv.get(f'{arch}.expert_latent_length', 0), attn_res_block_size=kv.get(f'{arch}.attn_res.block_size', 0),
      situ=(kv[f'{arch}.activation.situ_beta'], kv[f'{arch}.activation.situ_linear_beta']) if f'{arch}.activation.situ_beta' in kv else None,
      streamed_experts=bool(kv.get(f'{arch}.expert_count')) and not any(k.endswith('.ffn_gate_exps.weight') for k in state_dict), mtp=mtp)
    model = Transformer(config)
    for p in (nn.state.get_parameters(model) if shard > 1 else []): p.to_(devices)
    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)  # NOTE: rope_freqs.weight (32,) is unused
    # NOTE: without this contiguous, it unpacks the weights from the model every time. we shouldn't need this, but for now it's faster
    if realize:
      for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
      Tensor.realize(*params)
    if config.ple is not None: model.blk[config.ple.layer].ple.table = GGUFPackedTable(unwrap(ple_entry))
    return model, kv

  @staticmethod
  def from_hf(path:str|pathlib.Path, max_context:int|None=None,
              realize=bool(getenv("REALIZE", 0)), shard:int=1) -> tuple[Transformer, dict]:
    path = pathlib.Path(path)
    cfg = json.loads((path/'config.json').read_text())
    assert cfg['architectures'][0] == 'GlmMoeDsaForCausalLM', "from_hf only supports glm-dsa checkpoints"
    assert shard >= 1 and cfg['num_attention_heads'] % shard == 0, "tensor parallel must divide the attention heads"
    devices = tuple(Device.canonicalize(f'{Device.DEFAULT}:{i}') for i in range(shard))
    hf_names = {'token_embd.weight': 'model.embed_tokens.weight', 'output_norm.weight': 'model.norm.weight', 'output.weight': 'lm_head.weight',
                'attn_norm.weight': 'input_layernorm.weight', 'ffn_norm.weight': 'post_attention_layernorm.weight',
                'attn_q_a.weight': 'self_attn.q_a_proj.weight', 'attn_q_a_norm.weight': 'self_attn.q_a_layernorm.weight',
                'attn_q_b.weight': 'self_attn.q_b_proj.weight', 'attn_kv_a_mqa.weight': 'self_attn.kv_a_proj_with_mqa.weight',
                'attn_kv_a_norm.weight': 'self_attn.kv_a_layernorm.weight', 'attn_k_b.weight': 'self_attn.kv_b_proj.weight',
                'attn_output.weight': 'self_attn.o_proj.weight', 'indexer.attn_q_b.weight': 'self_attn.indexer.wq_b.weight',
                'indexer.attn_k.weight': 'self_attn.indexer.wk.weight', 'indexer.proj.weight': 'self_attn.indexer.weights_proj.weight',
                'indexer.k_norm.weight': 'self_attn.indexer.k_norm.weight', 'indexer.k_norm.bias': 'self_attn.indexer.k_norm.bias',
                'ffn_gate_inp.weight': 'mlp.gate.weight', 'exp_probs_b.bias': 'mlp.gate.e_score_correction_bias',
                **{f'ffn_{p}.weight': f'mlp.{p}_proj.weight' for p in ('gate', 'up', 'down')},
                **{f'ffn_{p}_shexp.weight': f'mlp.shared_experts.{p}_proj.weight' for p in ('gate', 'up', 'down')},
                **{f'ffn_{p}_exps.weight': f'mlp.experts.{{}}.{p}_proj.weight' for p in ('gate', 'up', 'down')}}
    shard_axes = {**{w: 0 for w in ('token_embd.weight', 'output.weight', 'attn_q_b.weight', 'attn_k_b.weight', 'ffn_gate.weight', 'ffn_up.weight')},
                  **{w: 1 for w in ('attn_output.weight', 'ffn_down.weight', 'ffn_gate_exps.weight', 'ffn_up_exps.weight')},
                  **{w: 2 for w in ('ffn_down_exps.weight',)}}

    global_names = ('token_embd.weight', 'output_norm.weight', 'output.weight')
    sources:dict[str, str|list[str]] = {k: hf_names[k] for k in global_names}
    for layer in range(cfg['num_hidden_layers']):
      for key, source in hf_names.items():
        if key in global_names or (key.startswith('indexer.') and cfg['indexer_types'][layer] != 'full'): continue
        source = f'model.layers.{layer}.' + source
        sources[f'blk.{layer}.{key}'] = [source.format(i) for i in range(cfg['n_routed_experts'])] if '{}' in source else source
    shard_map = {name: shard_axes[key] for name in sources if (key:=re.sub(r'^blk\.\d+\.', '', name)) in shard_axes}
    state_dict = hf_load(path, devices, sources, shard_map)
    block_size = tuple(cfg['quantization_config']['weight_block_size'])
    nope, value = cfg['qk_nope_head_dim'], cfg['v_head_dim']
    step = math.gcd(block_size[0], nope, value)

    config = TransformerConfig(
      num_blocks=cfg['num_hidden_layers'], dim=cfg['hidden_size'], hidden_dim=cfg['moe_intermediate_size'],
      n_heads=cfg['num_attention_heads'], n_kv_heads=1, norm_eps=cfg['rms_norm_eps'], vocab_size=cfg['vocab_size'],
      head_dim=cfg['qk_nope_head_dim'] + cfg['qk_rope_head_dim'], v_head_dim=cfg['v_head_dim'],
      rope_theta=cfg['rope_parameters']['rope_theta'], rope_dim=cfg['qk_rope_head_dim'],
      max_context=min(max_context, cfg['max_position_embeddings']) if max_context is not None else cfg['max_position_embeddings'],
      num_experts=cfg['n_routed_experts'], num_experts_per_tok=cfg['num_experts_per_tok'], norm_topk_prob=cfg['norm_topk_prob'],
      expert_gating_func=ExpertGating.SIGMOID if cfg['scoring_func'] == 'sigmoid' else ExpertGating.SOFTMAX,
      kv_lora_rank=cfg['kv_lora_rank'], q_lora_rank=cfg['q_lora_rank'], leading_dense_blocks=cfg['first_k_dense_replace'],
      shared_expert_dim=cfg['n_shared_experts'] * cfg['moe_intermediate_size'], shared_expert_gate=False, expert_bias=True,
      dense_hidden_dim=cfg['intermediate_size'], routed_scaling_factor=cfg['routed_scaling_factor'],
      indexer=IndexerConfig(cfg['index_n_heads'], cfg['index_head_dim'], cfg['index_topk'], rope_dim=cfg['qk_rope_head_dim'],
                            rope_interleave=cfg['indexer_rope_interleave']),
      indexer_shared=tuple(t != 'full' for t in cfg['indexer_types']), rope_interleave=cfg['rope_interleave'])
    model = Transformer(config)
    for p in (nn.state.get_parameters(model) if shard > 1 else []): p.to_(devices)
    for i, block in enumerate(model.blk):
      assert isinstance(block, MLATransformerBlock)
      prefix = f'blk.{i}.'
      w = state_dict[prefix+'attn_k_b.weight']
      kv = w.reshape(config.n_heads, nope + value, config.kv_lora_rank)
      state_dict[prefix+'attn_k_b.weight'] = kv[:, :nope].transpose(1, 2)
      state_dict[prefix+'attn_v_b.weight'] = kv[:, nope:]
      if (scale:=state_dict.get(prefix+'attn_k_b.weight_scale_inv')) is not None:
        # Repeat scales to align their blocks with each head's K/V slices.
        scales = scale.repeat_interleave(block_size[0] // step, -2)[:w.shape[0] // step].reshape(config.n_heads, -1, scale.shape[-1])
        state_dict[prefix+'attn_k_b.weight_scale_inv'] = scales[:, :nope // step].transpose(-1, -2)
        state_dict[prefix+'attn_v_b.weight_scale_inv'] = scales[:, nope // step:]
        block.kv_b_block_size = (step, block_size[1])
    for name, scale in state_dict.items():
      if not name.endswith('.weight_scale_inv'): continue
      projection = get_child(model, name.removesuffix('.weight_scale_inv'))
      if isinstance(projection, dict): projection['weight_scale_inv'] = scale
      else: projection.weight_scale_inv, projection.weight_block_size = scale, block_size
    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)
    if realize:
      params = nn.state.get_parameters(model)
      packed = [s.bitcast(dtypes.uint8).contiguous() for s in params]
      Tensor.realize(*packed)
      for s, data in zip(params, packed): s.replace(data.bitcast(s.dtype))
    return model, cfg

  def warmup(self):
    for _ in range(2): list(zip(range(2), self.generate([0])))

  def get_start_pos(self, tokens:list[int]) -> int:
    # recurrent state can't be partially reused after divergence: reuse it only when tokens extend the cached prefix, else the deepest saved
    # one (restored, the attention caches are still valid up to it). the saved ones tokens don't extend are dropped: they'll be overwritten
    if self.has_recurrent_block:
      def extends(prefix:list[int]) -> bool: return 0 < len(prefix) < len(tokens) and tokens[:len(prefix)] == prefix
      if extends(self._cached_tokens): return len(self._cached_tokens)
      self._saved = [(t, i) for t, i in self._saved if extends(t)]
      if not self._saved: return 0
      prompt, i = self._saved[-1]
      self._copy(i, save=False)
      self._cached_tokens = prompt[:]
      return len(prompt)
    prefix_len = sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], zip(tokens[:-1], self._cached_tokens)))
    return min(block._reusable_prefix_len(prefix_len, len(self._cached_tokens)) for block in self.blk)

  def generate(self, tokens:list[int], chunk_size:int=32, temperature:float=0.0):
    if self.has_recurrent_block and not amd_custom_kernels_supported(self.token_embd.weight.device): chunk_size = 1
    v_start_pos = UOp.variable("start_pos", 0, self.max_context-1)
    v_toks = UOp.variable("toks", 1, chunk_size)
    # TODO: use UOp.variable for temperature once float variables are supported
    temp = Tensor([temperature])
    # assign all input tokens once, then slice from start_pos for the model call
    t = Tensor(tokens + [0] * (self.max_context - len(tokens)), dtype="int32").reshape(1, self.max_context)
    # recompute start_pos from what's currently valid in the caches
    start_pos = self.get_start_pos(tokens)
    fork, out, prompt_len = self.fork(tokens, start_pos, chunk_size), None, len(tokens)
    while len(tokens) < self.max_context:
      n_toks = min(chunk_size, len(tokens) - start_pos, *([fork - start_pos] if start_pos < fork else []))
      if self.config.ple: self.blk[self.config.ple.layer].ple.write_rows(tokens, start_pos, n_toks, chunk_size)
      sp, nt = v_start_pos.bind(start_pos), v_toks.bind(n_toks)
      out = self(t[:, sp:sp+nt] if start_pos < prompt_len or out is None else out, (sp,), temp).realize()
      start_pos += n_toks
      if start_pos == fork: self.save(tokens[:fork])
      # chunked prefill: keep processing until all prompt tokens are consumed
      if start_pos < len(tokens): continue
      if start_pos == prompt_len: self.save(tokens)
      tokens.append(int(out.item()))
      self._cached_tokens = tokens[:-1]
      yield tokens[-1]
