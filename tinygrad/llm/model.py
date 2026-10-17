from __future__ import annotations
import array, enum, functools, itertools, math, pathlib, re
from dataclasses import dataclass, replace
from typing import Any, Callable, cast
from tinygrad import Tensor, nn, UOp, TinyJit, getenv, function, dtypes, Device
from tinygrad.llm.kernels.amd import Linear, gated_delta_prefill, flash_attention, amd_custom_kernels_supported, find_packed, moe_linear, \
  hc_coeffs, hc_pre, q8_quantize, q8_linear, QUANT_SIZES, q8_quantize_cache_clear, amd_flash_attention_decode as flash_attention_decode, moe_route, \
  moe_grouped_linear, amd_flash_attention_mla_prefill as flash_attention_mla_prefill, partial_sum, fuse_gate_up, kda_decode, moe_router, \
  indexer_scores, _wmma_rdna4, pool_keys, pool_index_rows, \
  indexer_decode_index, amd_flash_attention_mla_sparse as flash_attention_mla_sparse, moe_combine, token_sharded, shard_rows, reduce_scatter, all_gather
from tinygrad.llm.gguf import gguf_parse, gguf_shard, _GGML_QUANT
from tinygrad.uop.ops import resolve
from tinygrad.helpers import unwrap, round_up, to_tuple, prod
from tinygrad.device import Buffer, BufferSpec

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

@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device:str|tuple[str, ...]|None=None, yarn:YaRNConfig|None=None) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2) / dim))
  concentration = 1.0
  if yarn is not None and yarn.factor > 1.0:
    concentration = 0.1 * math.log(yarn.factor) + 1.0
    d_half = dim // 2
    low = max(0, math.floor(d_half * math.log(yarn.orig_ctx_len / (yarn.beta_fast * 2 * math.pi)) / math.log(theta)))
    high = min(dim-1, math.ceil(d_half * math.log(yarn.orig_ctx_len / (yarn.beta_slow * 2 * math.pi)) / math.log(theta)))
    interp = 1 - ((Tensor.arange(d_half).float() - low) / max(0.001, high-low)).clamp(0, 1)
    freqs = freqs * interp + (freqs / yarn.factor) * (1 - interp)
  freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  return (freqs.cos() * concentration).cat(freqs.sin() * concentration, dim=-1).clone(device)

class ExpertWeights:
  """Like Linear but with num_experts dimension. Weight shape: (num_experts, out_features, in_features)."""
  ggml_type:int|None = None
  fused = False  # the packed rows hold the gate and up weights interleaved by 16 (fuse_gate_up)
  slices = 0  # sliced experts (expert_slicing): the weights are the slices of every device, num_experts = experts*slices
  experts = 0
  def __init__(self, num_experts:int, in_features:int, out_features:int, bias:bool=False):
    self.num_experts, self.in_features = num_experts, in_features
    self.weight = Tensor.zeros(num_experts, out_features, in_features)
    if bias: self.bias = Tensor.zeros(num_experts, out_features)
  @property
  def expert_parallel(self) -> bool: return isinstance(self.weight.device, tuple) and self.weight.uop.axis == 0
  def set_quantized(self):
    # packed ggml experts run in a kernel that reads only the selected experts. with expert parallel weights, a device reads its experts
    if self.ggml_type is not None or hasattr(self, 'bias') or not amd_custom_kernels_supported(self.weight.device): return
    devices = self.weight.device if isinstance(self.weight.device, tuple) else (str(self.weight.device),)
    if self.weight.uop.axis not in (None, 0) or (found:=find_packed(self.weight, self.in_features)) is None: return
    n = len(devices) if self.expert_parallel else 1
    self.ggml_type, self.weight, self.local_experts = found[0], found[1], self.num_experts // n
    # sliced: the selections are mapped to the local slices of every device (slice_table), the ids are local
    first = Tensor([0 if self.slices else i*self.local_experts for i in range(n)], dtype=dtypes.int32)
    self.first = (first.shard(devices, 0) if n > 1 else first.to(devices[0] if len(devices) == 1 else devices)).realize()
    if self.slices:
      tables = [expert_slice_table(units, self.experts, self.slices) for units in unwrap(expert_slicing(self.experts, self.slices, n))]
      self.slice_table = Tensor(tables, dtype=dtypes.int32).shard(devices, 0).realize()
  def select_slices(self, sel:Tensor, probs:Tensor) -> tuple[Tensor, Tensor]:
    """sliced: the selections (B, T, k) of experts to the local slices of every device (devices, B, T, k*w) (-1: none), their probabilities"""
    E, w = self.slice_table.shape[-2:]
    onehot = (sel.unsqueeze(-1) == Tensor.arange(E, dtype=dtypes.int32)).unsqueeze(-1)
    lsel = onehot.where(self.slice_table.reshape(-1, 1, 1, 1, E, w), 0).sum(-2, dtype=dtypes.int32)
    return lsel.reshape(*lsel.shape[:3], -1), probs.unsqueeze(-1).expand(*probs.shape, w).reshape(*probs.shape[:2], -1)
  def grouped(self, x:Tensor, entries:Tensor, tiles:Tensor, slots:int=0, selections:int=0, act:tuple|None=None, out_dtype=dtypes.float32,
              sorted_out:bool=False) -> Tensor:
    """the selections sorted by expert (moe_route), see moe_grouped_linear. expert parallel: a leading device axis"""
    return moe_grouped_linear(self.weight, unwrap(self.ggml_type), self.local_experts, x, entries, tiles, slots, selections, act, out_dtype,
                              sorted_out)
  def fuse_up(self, up:ExpertWeights, rows:int):
    """take the up weights: the packed rows interleaved by 16 with this gate's, one kernel for both"""
    assert self.ggml_type is not None and self.ggml_type == up.ggml_type and self.weight.device == up.weight.device
    self.weight = fuse_gate_up(self.weight, up.weight, self.num_experts, rows).contiguous().realize()
    self.fused = True
  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
    # sel: (B, T, k), x: (B, T, 1, in) or (B, T, k, in) -> output: (B, T, k, out)
    # expert parallel: x and the output have a leading device axis, (devices, B, T, k, out) partial outputs from the experts of each device
    self.set_quantized()
    if self.ggml_type is not None:
      if isinstance(sel.numel(), int): return moe_linear(self.weight, self.ggml_type, self.local_experts, x, sel, self.first)
      # symbolic token count: pad to the max chunk size so the kernels see static shapes, garbage rows are sliced off
      out = moe_linear(self.weight, self.ggml_type, self.local_experts, x.pad_to(x.max_shape), sel.pad_to(sel.max_shape), self.first)
      return out.shrink(tuple((0, s) for s in out.shape[:-4])+tuple((0, s) for s in sel.shape)+((0, out.shape[-1]),))
    ret = (x.unsqueeze(-2) @ self.weight[sel].transpose(-1, -2)).contiguous().squeeze(-2)
    return ret + self.bias[sel] if hasattr(self, 'bias') else ret

def expert_slicing(experts:int, slices:int, n:int) -> list[list[int]]|None:
  """
  the units (expert*slices + slice) of every device: every device holds slices//n slices of every expert, the slices%n left go to a class of
  devices (by expert, cycling): the work of a token's experts is spread over all the devices. None if the devices don't get the same count
  """
  base, r = divmod(slices, n)
  if r and (n % r or experts % (n // r)): return None
  per:list[list[int]] = [[] for _ in range(n)]
  for e in range(experts):
    extra = set(range((e % (n // r))*r, (e % (n // r))*r + r)) if r else set()
    devs = [d for d in range(n) if d in extra for _ in range(base+1)] + [d for d in range(n) if d not in extra for _ in range(base)]
    for j, d in enumerate(devs): per[d].append(e*slices + j)
  return per

def expert_slice_table(units:list[int], experts:int, slices:int) -> list[list[int]]:
  """(experts, w): the local index of every slice a device holds of an expert, -1 padded"""
  table:list[list[int]] = [[] for _ in range(experts)]
  for i, u in enumerate(units): table[u // slices].append(i)
  w = max(len(t) for t in table)
  return [t + [-1]*(w-len(t)) for t in table]

def gated_activation(gate:Tensor, up:Tensor, *, alpha:float=1.0, limit:float|None=None, up_bias:float=0.0) -> Tensor:
  if limit is not None and limit > 0: gate, up = gate.clamp(max_=limit), up.clamp(-limit, limit)
  # TODO: remove the need for this contiguous
  return gate * (gate * alpha).sigmoid().contiguous() * (up + up_bias)

def apply_rope(x:Tensor, freqs_cis:Tensor) -> Tensor:
  assert x.shape[-1] % 2 == 0
  cos, sin = freqs_cis.reshape(1, 1, x.shape[2], -1).chunk(2, dim=-1)
  x1, x2 = x.chunk(2, dim=-1)
  return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)

def pairwise_topk(x: Tensor, k: int) -> tuple[Tensor, Tensor]:
  n = x.shape[-1]
  vals = Tensor.arange(n).reshape(1,1,n).cast(x.dtype).expand(x.shape)
  cmp = (x.unsqueeze(-1) > x.unsqueeze(-2)) | ((x.unsqueeze(-1) == x.unsqueeze(-2)) & \
    (Tensor.arange(n).reshape(1,1,n,1) < Tensor.arange(n).reshape(1,1,1,n)))
  # the element of rank r (the count of the elements below it) is the r-th smallest: the top k are the ranks n-k..n-1, in that order
  picked = cmp.sum(axis=-1).unsqueeze(-2) == Tensor.arange(n-k, n).reshape(k, 1)
  return picked.where(x.unsqueeze(-2), 0).sum(-1), picked.where(vals.unsqueeze(-2), 0).sum(-1).cast('int32')

def pairwise_topk_mask(x:Tensor, k:int) -> Tensor:
  """the top k of the last axis: those beaten by fewer than k (a higher value, or the same and a lower index). one reduce instead of a sort"""
  n = x.shape[-1]
  first = Tensor.arange(n).reshape(1, n) < Tensor.arange(n).reshape(n, 1) if isinstance(n, int) else None
  if first is None:  # a symbolic length: slice the comparison of a static arange
    ar = Tensor.arange(cast(UOp, n).vmax)[:n]
    first = ar.reshape(1, n) < ar.reshape(n, 1)
  return ((x.unsqueeze(-2) > x.unsqueeze(-1)) | ((x.unsqueeze(-2) == x.unsqueeze(-1)) & first)).sum(-1) < k

def topk_mask(x:Tensor, k:int) -> Tensor:
  """
  the top k of the last axis (and the values tied with the k-th): the k-th largest is found bit by bit on order preserving integer keys.
  32 reduces over the axis instead of one over its square
  """
  bits = x.float().bitcast(dtypes.uint32)
  key = (bits >= 0x80000000).where(bits ^ 0xFFFFFFFF, bits | 0x80000000)
  thr = key.sum(-1, keepdim=True).const_like(0).cast(dtypes.uint32)
  for bit in range(31, -1, -1):
    cand = thr | (1 << bit)
    thr = ((key >= cand).sum(-1, keepdim=True) >= k).where(cand, thr)
  return key >= thr

@dataclass(frozen=True)
class SSMConfig:
  conv_kernel: int
  state_size: int
  group_count: int
  time_step_rank: int
  inner_size: int
  kda: bool = False
  gate_lower_bound: float|None = None  # kda: bounded decay gate

@dataclass(frozen=True)
class HCConfig:  # manifold-constrained hyper-connections: count residual streams, mixed by a doubly stochastic matrix
  count: int
  sinkhorn_iters: int
  eps: float

@dataclass(frozen=True)
class IndexerConfig:  # a query attends to the top_k keys picked by an indexer scoring pools of kpool keys
  n_heads: int
  head_dim: int
  top_k: int
  kpool: int
  norm_eps: float

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
  expert_slices: int = 0  # the experts are split in slices of their hidden dim, spread over the devices (see expert_slicing)
  swiglu_alpha: float = 1.0
  swiglu_limit: float|None = None
  swiglu_up_bias: float = 0.0
  sliding_window: int = 0
  sliding_window_pattern: int = 0
  hc: HCConfig|None = None
  indexer: IndexerConfig|None = None
  padded_vocab_size: int = 0  # the output rows, zero padded to split over devices

# set while a block's rows are split over the devices (sequence parallel prefill): the sublayers return their partial sums (a leading device
# axis), the hyper-connection sums each device's rows of them (reduce_scatter) instead of all of them on every device
_partial_out = [False]
# and the sublayer's input rows of this device (before the gather): per token work (the router) runs on them, its results are gathered
_sp_rows:list[Tensor|None] = [None]
def _out(layer:Linear, x:Tensor) -> Tensor:
  """a sublayer's output projection: its partial sums if they are kept (_partial_out)"""
  return layer.partial(x) if _partial_out[0] else layer(x)

def _identity(x:Tensor) -> Tensor: return x

def _argmax_rows(x:Tensor, cols:int=256) -> Tensor:
  """
  x (B, V) -> (B, 1) argmax over a long last axis (the vocab) in two parallel steps: the max of every row of `cols`, then the first row with the
  max and the first index of it there (the first index of the max, as argmax)
  """
  B, V = x.shape
  rows = (V + cols - 1) // cols
  g = x.pad(((0, 0), (0, rows*cols - V)), value=float("-inf")).reshape(B, rows, cols)
  row = g.max(-1).argmax(-1, keepdim=True)  # (B, 1)
  col = g.gather(1, row.unsqueeze(-1).expand(B, 1, cols)).argmax(-1)  # (B, 1)
  return row * cols + col

class HyperConnection:
  def __init__(self, config:TransformerConfig):
    self.count, self.eps, self.iters, self.norm_eps = (hc:=unwrap(config.hc)).count, hc.eps, hc.sinkhorn_iters, config.norm_eps
    self.fn = Linear(self.count * config.dim, (2 + self.count) * self.count, bias=False)
    self.base, self.scale = {"weight": Tensor.zeros((2 + self.count) * self.count)}, {"weight": Tensor.zeros(3)}

  def __call__(self, xs:Tensor, sublayer:Callable[[Tensor], Tensor], norm:Callable[[Tensor], Tensor]|None=None, rows:bool=False) -> Tensor:
    """
    mHC: the sublayer sees the norm of a pre-weighted sum of the (B, T, hc, D) streams, its output is added back by post onto the streams mixed
    by comb. rows: the streams have static padded rows, so do the sublayer's input and output.
    rows split over the devices (sequence parallel): every device computes the coefficients and mixes of its rows, the sublayer sees all
    of them (all_gather) and returns its partial sums, every device sums its rows of them (reduce_scatter)
    """
    base, scale = self.base["weight"], self.scale["weight"]
    if norm is None: norm = _identity
    if amd_custom_kernels_supported(xs.device) and self.fn.ggml_type in QUANT_SIZES:
      # a symbolic token count runs the streams padded to the max size: static shapes, the sublayer sees the real tokens
      T = xs.shape[1]
      if not isinstance(xs.numel(), int): xs = xs.pad_to(xs.max_shape).contiguous()
      # fn(rmsnorm(x)) is fn(x) scaled by the rms: the quantization of x has its sums of squares, one kernel for the coefficients and sinkhorn
      flat = xs.flatten(2)
      n, features = int(math.prod(flat.shape[:-1])), int(flat.shape[-1])
      split = token_sharded(xs)  # sequence parallel: the rows of the streams are split over the devices
      *xq, sq = q8_quantize(flat, n, features, sumsq=True, split=split)
      mix = q8_linear(self.fn, flat, xq=tuple(xq), split=split)
      if isinstance(norm, nn.RMSNorm) and norm.weight is not None and getenv("HC_PRE", 1):
        # one kernel for the coefficients, the mix of the streams and its norm
        pre, post, comb, hn = hc_pre(mix, sq, base, scale, xs, norm.weight, self.count, self.iters, self.eps, self.norm_eps, norm.eps, features,
                                     split)
      else:
        pre, post, comb = hc_coeffs(mix, sq, base, scale, self.count, self.iters, self.eps, self.norm_eps, features, split)
        hn = norm((xs * pre.unsqueeze(-1)).sum(2))
      if split:
        _partial_out[0], _sp_rows[0] = True, hn
        try: part = sublayer(all_gather(hn, 1, dtypes.half).float())
        finally: _partial_out[0], _sp_rows[0] = False, None
        out = reduce_scatter(part, 1) if part.ndim == hn.ndim + 1 else shard_rows(part, 1)
      else: out = sublayer(hn if rows else hn[:, :T])
      if out.shape[1] is not xs.shape[1]: out = out.pad_to((out.shape[0], xs.shape[1], out.shape[2]))
      ret = out.unsqueeze(2) * post.unsqueeze(-1) + (comb.transpose(-1, -2) @ xs)
      return ret if rows else ret[:, :T]
    else:
      flat = xs.flatten(2)
      mix = self.fn(flat.rmsnorm(eps=self.norm_eps))
      pre  = (mix[..., :self.count] * scale[0] + base[:self.count]).sigmoid() + self.eps
      post = (mix[..., self.count:2*self.count] * scale[1] + base[self.count:2*self.count]).sigmoid() * 2
      comb = (mix[..., 2*self.count:] * scale[2] + base[2*self.count:]).reshape(*mix.shape[:-1], self.count, self.count).softmax(-1) + self.eps
      # sinkhorn: normalize the columns then the rows, alternating, to make comb doubly stochastic
      for i in range(2 * self.iters - 1): comb = comb / (comb.sum(-1 if i % 2 else -2, keepdim=True) + self.eps)
    out = sublayer(norm((xs * pre.unsqueeze(-1)).sum(2)))
    return out.unsqueeze(2) * post.unsqueeze(-1) + (comb.transpose(-1, -2) @ xs)

class FFNBlock:
  def __init__(self, config:TransformerConfig):
    self.config = config

    # --- RMSNorms --------------------------------------------------------
    self.attn_norm   = nn.RMSNorm(config.dim, config.norm_eps)
    self.ffn_norm    = nn.RMSNorm(config.dim, config.norm_eps)
    if config.hc is not None: self.hc_attn, self.hc_ffn = HyperConnection(config), HyperConnection(config)

    # --- feed-forward (MoE or dense) -------------------------------------
    if config.num_experts > 0:
      self.ffn_gate_inp = Linear(config.dim, config.num_experts, bias=config.expert_proj_bias)  # router
      if config.expert_bias: self.exp_probs_b = {"bias": Tensor.zeros(config.num_experts)}
      # sliced experts: num_experts*slices experts of hidden_dim/slices
      S = config.expert_slices or 1
      self.ffn_gate_exps = ExpertWeights(config.num_experts*S, config.dim, config.hidden_dim//S, bias=config.expert_proj_bias)
      self.ffn_up_exps = ExpertWeights(config.num_experts*S, config.dim, config.hidden_dim//S, bias=config.expert_proj_bias)
      self.ffn_down_exps = ExpertWeights(config.num_experts*S, config.hidden_dim//S, config.dim, bias=config.expert_proj_bias)
      if config.expert_slices:
        for e in (self.ffn_gate_exps, self.ffn_up_exps, self.ffn_down_exps): e.slices, e.experts = S, config.num_experts
      if config.shared_expert_dim > 0:
        self.ffn_gate_shexp = Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_up_shexp = Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_down_shexp = Linear(config.shared_expert_dim, config.dim, bias=False)
        if config.shared_expert_gate: self.ffn_gate_inp_shexp = {"weight": Tensor.zeros(config.dim)}
    else:
      self.ffn_gate    = Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_up      = Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_down    = Linear(config.hidden_dim, config.dim, bias=False)

  def _feed_forward(self, x:Tensor) -> Tensor:
    # a symbolic token count runs the MoE padded to the max size: static shapes, the garbage tokens are sliced off
    if hasattr(self, 'ffn_gate_exps') and not isinstance(x.shape[1], int) and amd_custom_kernels_supported(x.device):
      return self._feed_forward(x.pad_to(x.max_shape))[:, :x.shape[1]]
    if hasattr(self, 'ffn_gate_exps'):
      h = x.unsqueeze(2)  # (B, T, 1, D) - add expert dim for broadcasting
      router = self.ffn_gate_inp
      # sequence parallel: every device routes its rows, the selections are gathered
      xr = rows if (rows:=_sp_rows[0]) is not None else x
      if router.ggml_type is None and router.weight.dtype == dtypes.half and router.bias is None and isinstance(xr.shape[1], int) and \
         xr.shape[1] > 32:
        # many tokens: half copies of the input and the output, a float accumulator: the tensor cores
        logits = xr.half().contiguous().matmul(router.weight.T, dtype=dtypes.float).contiguous()
      else: logits = router(xr)
      bias = self.exp_probs_b["bias"] if hasattr(self, 'exp_probs_b') else None
      gating, normalize_topk = self.config.expert_gating_func, self.config.norm_topk_prob
      if gating == ExpertGating.SIGMOID and normalize_topk and bias is not None and isinstance(xr.shape[1], int) and \
         amd_custom_kernels_supported(x.device) and self.config.num_experts <= 1024 and getenv("MOE_ROUTER", 1):
        sel, probs = moe_router(logits, bias, self.config.num_experts_per_tok, self.config.routed_scaling_factor)
        if rows is not None: sel, probs = all_gather(sel, 1), all_gather(probs, 1)
        return self._experts(x, h, sel, probs)
      assert rows is None, "sequence parallel routing needs the router kernel"
      # fast path: without selection bias, normalized SOFTMAX is equivalent to SOFTMAX_WEIGHT
      if gating == ExpertGating.SOFTMAX and bias is None and normalize_topk:
        gating, normalize_topk = ExpertGating.SOFTMAX_WEIGHT, False
      if   gating == ExpertGating.SOFTMAX_WEIGHT: scores = logits
      elif gating == ExpertGating.SOFTMAX:        scores = logits.softmax(-1)
      elif gating == ExpertGating.SIGMOID:        scores = logits.sigmoid()
      elif gating == ExpertGating.SQRT_SOFTPLUS:  scores = logits.softplus().sqrt()

      _, sel = pairwise_topk(scores if bias is None else scores + bias, self.config.num_experts_per_tok)
      probs = scores.gather(-1, sel)
      # SOFTMAX_WEIGHT applies softmax after top-k selection
      if gating == ExpertGating.SOFTMAX_WEIGHT: probs = probs.softmax(-1)
      if normalize_topk: probs = probs / probs.sum(axis=-1, keepdim=True)
      probs = probs * self.config.routed_scaling_factor
      return self._experts(x, h, sel, probs)
    return _out(self.ffn_down, gated_activation(self.ffn_gate(x), self.ffn_up(x), limit=self.config.swiglu_limit))

  def _experts(self, x:Tensor, h:Tensor, sel:Tensor, probs:Tensor) -> Tensor:
    """the selected experts (and the shared expert) of every token, summed over the devices"""
    exps = (self.ffn_gate_exps, self.ffn_down_exps) + (() if self.ffn_gate_exps.fused else (self.ffn_up_exps,))
    for e in exps: e.set_quantized()
    # sliced experts: every device runs its slices of the selected experts
    if self.ffn_gate_exps.slices: sel, probs = self.ffn_gate_exps.select_slices(sel, probs)
    grouped = resolve(x.shape[1] != 1) and all(e.ggml_type is not None for e in exps)
    shexp:Tensor|None = None
    if grouped and hasattr(self, 'ffn_gate_shexp') and not hasattr(self, 'ffn_gate_inp_shexp') and self.ffn_gate_exps.expert_parallel and \
       isinstance(x.shape[1], int) and self.ffn_down_shexp.shard_axis == 1:
      # the shared expert's partial sums are added by the kernel summing the experts' rows of every token
      shexp_act = gated_activation(self.ffn_gate_shexp(x), self.ffn_up_shexp(x), limit=self.config.swiglu_limit).contiguous()
      shexp = self.ffn_down_shexp.partial(shexp_act)
    if grouped:
      out = self._grouped_experts(x, sel, probs, add=shexp)
    else:
      if self.ffn_gate_exps.fused:  # one kernel for the gate and up rows, interleaved by 16
        gu = self.ffn_gate_exps(sel, h)
        gu = gu.reshape(*gu.shape[:-1], -1, 2, 16)
        gate, up = gu[..., 0, :].flatten(-2), gu[..., 1, :].flatten(-2)
      else: gate, up = self.ffn_gate_exps(sel, h), self.ffn_up_exps(sel, h)
      act = gated_activation(gate, up, alpha=self.config.swiglu_alpha, limit=self.config.swiglu_limit, up_bias=self.config.swiglu_up_bias)
      x_down = self.ffn_down_exps(sel, act.contiguous())  # (B, T, k, D), expert parallel: (devices, B, T, k, D) partials
      out = (x_down * probs.unsqueeze(-1)).sum(axis=-2)  # (B, T, D), expert parallel: (devices, B, T, D)
    partial = out.ndim == x.ndim + 1
    keep = partial and _partial_out[0]  # the partial sums are returned (sequence parallel)
    if shexp is not None:
      if partial and not keep: out = partial_sum(out)
    elif hasattr(self, 'ffn_gate_shexp'):
      shexp_act = gated_activation(self.ffn_gate_shexp(x), self.ffn_up_shexp(x), limit=self.config.swiglu_limit).contiguous()
      # a row parallel shared expert adds its partial sums to the ones of the experts: one allreduce for both
      shexp = self.ffn_down_shexp.partial(shexp_act) if partial else self.ffn_down_shexp(shexp_act)
      if hasattr(self, 'ffn_gate_inp_shexp'): shexp = shexp * (x * self.ffn_gate_inp_shexp["weight"]).sum(axis=-1, keepdim=True).sigmoid()
      if shexp.ndim == out.ndim: out = out + shexp
      elif keep: out = out + shexp / int(out.shape[0])  # a whole shared expert output: a share of it in every partial
      if partial and not keep: out = partial_sum(out)
      if shexp.ndim != out.ndim + partial and not keep: out = out + shexp
    elif partial and not keep: out = partial_sum(out)
    return out

  def _grouped_experts(self, x:Tensor, sel:Tensor, probs:Tensor, add:Tensor|None=None) -> Tensor:
    """prefill: the selections sorted by expert, every expert reads its weights once for all its tokens. (devices, B, T, D) partials if split"""
    B, T, D = x.shape
    # a symbolic token count runs padded to the max size, the garbage tokens are sliced off
    if not isinstance(T, int): x, sel, probs = x.pad_to(x.max_shape), sel.pad_to(sel.max_shape), probs.pad_to(probs.max_shape)
    gate_exps, down_exps = self.ffn_gate_exps, self.ffn_down_exps
    _, entries, tiles, pos = moe_route(sel, gate_exps.first, gate_exps.local_experts)
    k, sel_shape = int(sel.shape[-1]), sel.shape[-3:]
    # gate and up read the token of each sorted row, down writes the row of each selection
    if gate_exps.fused:  # the activation is computed in the kernel of the interleaved gate and up rows
      act = gate_exps.grouped(x.reshape(-1, D), entries, tiles, slots=k,
                              act=(self.config.swiglu_alpha, self.config.swiglu_limit or 0.0, self.config.swiglu_up_bias))
    else:
      gate, up = (e.grouped(x.reshape(-1, D), entries, tiles, slots=k) for e in (gate_exps, self.ffn_up_exps))
      act = gated_activation(gate, up, alpha=self.config.swiglu_alpha, limit=self.config.swiglu_limit, up_bias=self.config.swiglu_up_bias)
    # down has a row per sorted row (in half if sliced: a token sums many), every token sums its rows weighed by their probabilities
    down = down_exps.grouped(act.contiguous(), entries, tiles, sorted_out=True, out_dtype=dtypes.half if down_exps.slices else dtypes.float32)
    y = moe_combine(down, pos, probs, int(prod(sel_shape[:-1])), int(sel_shape[-1]), add)
    y = y.reshape(-1, *sel_shape[:-1], D)[:, :, :x.shape[1] if isinstance(T, int) else T]
    return y if gate_exps.expert_parallel else y[0]

  # given the token-prefix match, return how much cached state this block can still reuse
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return prefix_len
  def _init_state(self, x:Tensor): raise NotImplementedError
  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor: raise NotImplementedError

  def __call__(self, x: Tensor, start_pos: int|UOp, T:int|UOp|None=None):
    # T (hyper-connections): x has static padded rows of which the first T are the tokens, the output too
    self._init_state(x)
    # we pass in the weights implicitly so we unpack the GGUF on the fly
    @function(precompile=True, allow_implicit=True)
    def _run(x:Tensor, start_pos:int|UOp, T:int|UOp|None):
      if self.config.hc is not None:
        xs = self.hc_attn(x, lambda h: self._attention(h, start_pos, *(() if T is None else (T,))), self.attn_norm, rows=T is not None)
        return self.hc_ffn(xs, self._feed_forward, self.ffn_norm, rows=T is not None).contiguous()
      h =     x + self._attention(self.attn_norm(x), start_pos)
      return (h + self._feed_forward(self.ffn_norm(h))).contiguous()
    return _run(x, start_pos, T)

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

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
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

    q = apply_rope(q[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos+T]).cat(q[..., self.config.rope_dim:], dim=-1)
    k = apply_rope(k[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos+T]).cat(k[..., self.config.rope_dim:], dim=-1)

    # NOTE: we don't want to change self.cache_kv, the function API doesn't support this well
    store = self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(Tensor.stack(k, v).cast(self.cache_kv.dtype).uop)
    assigned_kv = Tensor(self.cache_kv.uop.after(store))
    # on RDNA3/4, hybrid models use custom flash attention kernels on the KV cache
    if amd_custom_kernels_supported(x.device) and self.config.ssm is not None:
      attn = flash_attention(q, assigned_kv, start_pos+T)
      attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
      return _out(self.attn_output, attn if not self.config.attn_output_gate else (attn * gate.sigmoid()))
    k = assigned_kv[0, :, :, 0:start_pos+T, :]
    v = assigned_kv[1, :, :, 0:start_pos+T, :]

    #self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v))
    #k = self.cache_kv[0, :, :, 0:start_pos+T, :]
    #v = self.cache_kv[1, :, :, 0:start_pos+T, :]

    # NOTE: this mask is causal_lower_right, not the causal_upper_left generated by is_casual = True
    # TODO: this if statement should be removed and it shouldn't generate extra kernels
    mask, window = None, self.config.sliding_window
    if resolve(T != 1) or window:
      mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, buffer=False)
      mask = mask.triu(start_pos+1) + mask.tril(start_pos-window) if window else mask.triu(start_pos+1)
    if hasattr(self, 'attn_sinks'):
      k, v = k.cat(k[..., :1, :].const_like(0), dim=-2), v.cat(v[..., :1, :].const_like(0), dim=-2)
      sink_col = self.attn_sinks["weight"].reshape(1, -1, 1, 1).expand(1, self.config.n_heads, T, 1)
      if mask is None: mask = Tensor.zeros(1, 1, T, start_pos+T, dtype=x.dtype, buffer=False)
      mask = mask.expand(1, self.config.n_heads, T, start_pos+T).cat(sink_col, dim=-1)
    attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)     # (B,H,T,Hd)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
    return _out(self.attn_output, attn if not self.config.attn_output_gate else (attn * gate.sigmoid()))

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_kv"):
      # zeroed so the flash kernels can safely read whole tiles past the valid region (masked lanes multiply by 0)
      self.cache_kv = Tensor.zeros(2, x.shape[0], self.config.n_kv_heads, self.config.max_context, self.config.head_dim,
                                   dtype=dtypes.half, device=x.device if isinstance(x.device, str) else None)
      if isinstance(x.device, tuple): self.cache_kv = self.cache_kv.shard(x.device, 2).realize()
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta,
                                            device=x.device, yarn=self.config.yarn)

class Indexer:
  def __init__(self, config:TransformerConfig):
    self.idx, self.max_context = (idx:=unwrap(config.indexer)), config.max_context
    self.attn_q_b, self.attn_k = Linear(config.q_lora_rank, idx.n_heads * idx.head_dim, bias=False), Linear(config.dim, idx.head_dim, bias=False)
    self.proj, self.k_norm = Linear(config.dim, idx.n_heads, bias=False), nn.LayerNorm(idx.head_dim, eps=idx.norm_eps)
    self.compressor_gate, self.compressor_ape = Linear(config.dim, idx.head_dim, bias=False), {"weight": Tensor.zeros(idx.kpool, idx.head_dim)}

  def _init_state(self, x:Tensor):
    if hasattr(self, "cache_pk"): return
    idx, max_pools = self.idx, round_up(self.max_context, self.idx.kpool) // self.idx.kpool
    # the sparse path (RDNA4): no mask, the pools a query picks are lists (decode) or per 16 queries (prefill, see pool_union)
    self.sparse = x.shape[0] == 1 and amd_custom_kernels_supported(x.device) and _wmma_rdna4(x.device) and 16 % idx.kpool == 0 and \
      idx.head_dim % 16 == 0 and max_pools % 64 == 0 and bool(getenv("INDEXER_SPARSE", 1))
    # the pooled keys, a call pools its tokens' pools only. the sparse path keeps them in half and the tokens of the last incomplete pool in a
    # ring, the other one every token's key and gate
    self.cache_pk = Tensor.zeros(*(() if self.sparse else (x.shape[0],)), max_pools, idx.head_dim,
                                 dtype=dtypes.half if self.sparse else dtypes.float32, device=x.device).contiguous()
    if self.sparse: self.ring = Tensor.zeros(idx.kpool, 2, idx.head_dim, dtype=dtypes.half, device=x.device).contiguous()
    else: self.cache_kg = Tensor.zeros(x.shape[0], max_pools*idx.kpool, 2, idx.head_dim, dtype=dtypes.half, device=x.device).contiguous()

  def picks(self, x:Tensor, q:Tensor, start_pos:UOp, T:int|UOp|None=None) -> Tensor:
    """the sparse path: for every query (Tq, n+1) the list of the n keys it sees at most (its top pools and the tail of its own), the count last"""
    Tq, idx = int(x.shape[1]), self.idx
    S = start_pos + (T := Tq if T is None else T)
    kg, iq = self.k_norm(self.attn_k(x)).stack(self.compressor_gate(x), dim=2), self.attn_q_b(q).reshape(Tq, idx.n_heads, idx.head_dim)
    pk = pool_keys(self.cache_pk, self.ring, kg[0].half(), self.compressor_ape["weight"], start_pos, T)
    weights, pools, k = (self.proj(x) * (idx.n_heads * idx.head_dim)**-0.5)[0], (S + idx.kpool - 1) // idx.kpool, idx.top_k // idx.kpool
    # one token: the heads are the rows of the scores' matmul, a kernel picks its pools
    if Tq == 1 and idx.n_heads % 16 == 0: return indexer_decode_index(iq[0], pk, weights[0], start_pos, pools, k, idx.kpool).reshape(1, -1)
    # the scores on the tensor cores (rows of 16 queries), a kernel per query picks its pools
    rows = round_up(Tq, 16)
    scores = indexer_scores(iq.pad(((0, rows-Tq), None, None)), pk, weights.pad(((0, rows-Tq), None)), pools)
    return pool_index_rows(scores, start_pos, pools, k, idx.kpool)[:Tq]

  def __call__(self, x:Tensor, q:Tensor, start_pos:int|UOp, T:int|UOp|None=None) -> Tensor:
    # x has Tq rows, the first T (default all) are the tokens at start_pos: the mask (B, 1, Tq, S) of the keys every query sees
    (B, Tq, _), idx = x.shape, self.idx
    S = start_pos + (T := Tq if T is None else T)
    # cache an indexer key and a pooling gate per token, a pool's key is the gate softmax weighted sum of its keys
    kg, iq = self.k_norm(self.attn_k(x)).stack(self.compressor_gate(x), dim=2), self.attn_q_b(q).reshape(B, Tq, idx.n_heads, idx.head_dim)
    kg = Tensor(self.cache_kg.uop.after(self.cache_kg[:, start_pos:S].uop.store(kg[:, :T].half().uop))).float()
    kg = kg.reshape(B, -1, idx.kpool, 2, idx.head_dim)
    # only the pools with tokens: the cost doesn't grow with the max context. the pools before the first token's are cached
    # (the count of new pools in a form that is visibly not negative)
    first, pools = start_pos // idx.kpool, (S + idx.kpool - 1) // idx.kpool
    new_pools = (start_pos % idx.kpool + T + idx.kpool - 1) // idx.kpool
    kg = kg.shrink((None, (first, first + new_pools), None, None, None))
    key, gate = (kg.shrink((None, None, None, (i, i+1), None)).squeeze(3) for i in range(2))
    new = ((gate + self.compressor_ape["weight"]).softmax(2) * key).sum(2)  # (B, new pools, D)
    store = self.cache_pk.shrink((None, (first, first + new_pools), None)).uop.store(new.uop)
    pooled = Tensor(self.cache_pk.uop.after(store)).shrink((None, (0, pools), None))  # (B, pools, D)
    weights = self.proj(x) * (idx.n_heads * idx.head_dim)**-0.5
    pools, max_pools = pooled.shape[1], self.cache_kg.shape[1] // idx.kpool
    scores = ((iq @ pooled.transpose(-1, -2).unsqueeze(1)).relu() * weights.unsqueeze(-1)).sum(2)
    # the query sees the top_k/kpool best complete pools and the tokens of its own pool while incomplete (the tail)
    # NOTE: a symbolic arange is a reduce over the positions, slice a static one
    q_pos, k_pos = (Tensor(start_pos) + Tensor.arange(Tq)).reshape(Tq, 1), Tensor.arange(self.cache_kg.shape[1])[:S]
    arange = Tensor.arange(max_pools)[:pools]
    masked = ((arange + 1) * idx.kpool <= q_pos + 1).where(scores, -math.inf)
    # selected incomplete pools add only tail/future tokens, handled by the final tail and causal mask
    picked = (topk_mask(masked, idx.top_k // idx.kpool) if Tq != 1 else pairwise_topk_mask(masked, idx.top_k // idx.kpool))
    picked = picked.repeat_interleave(idx.kpool, dim=-1)[..., :S]
    return ((picked | (k_pos >= (q_pos + 1) // idx.kpool * idx.kpool)) & (k_pos <= q_pos)).where(0.0, -math.inf).reshape(B, 1, Tq, S).cast(x.dtype)


class MLATransformerBlock(FFNBlock):
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

  def _attention(self, x:Tensor, start_pos:int|UOp, T:int|UOp|None=None) -> Tensor:
    if T is not None: return self._attention_rows(x, start_pos, T)  # rows mode: x is padded, the first T rows are the tokens
    N, D = self.cache_k.shape[2:]
    flash = amd_custom_kernels_supported(x.device) and N % 64 == 0 and D & (D-1) == 0
    # a symbolic chunk computes padded to its max size (static shapes), it stores and attends with its real tokens
    if flash and not isinstance(T:=x.shape[1], int): return self._attention_rows(x.pad_to(x.max_shape), start_pos, T)[:, :T]
    return self._attention_rows(x, start_pos, x.shape[1])

  def _attention_rows(self, x:Tensor, start_pos:int|UOp, T:int|UOp) -> Tensor:
    # x has Tq rows, the first T are the tokens at start_pos
    B, Tq, _ = x.shape
    q_nope_head_dim = self.config.head_dim - self.config.rope_dim
    q_proj = self.attn_q_b(self.attn_q_a_norm(self.attn_q_a(x))) if self.config.q_lora_rank > 0 else self.attn_q(x)
    q = q_proj.reshape(B, Tq, self.config.n_heads, self.config.head_dim).transpose(1, 2)
    q_nope, q_rope = q[..., :q_nope_head_dim], q[..., q_nope_head_dim:]
    freqs = self.freqs_cis[start_pos:start_pos+T]
    if Tq is not T: freqs = freqs.pad_to((Tq, freqs.shape[1]))
    if not self.config.ssm or not self.config.ssm.kda: q_rope = apply_rope(q_rope, freqs)
    # half weights: half copies of the input and the output and a float accumulator use the tensor cores (casts or permutes fused into the
    # matmul don't)
    k_b, half = self.attn_k_b["weight"].transpose(-1, -2), self.attn_k_b["weight"].dtype == dtypes.half and Tq > 1
    q_nope = q_nope.half().contiguous().matmul(k_b, dtype=dtypes.float).contiguous() if half else q_nope @ k_b
    q = q_nope.cat(q_rope.cast(q_nope.dtype), dim=-1)

    kv_a = self.attn_kv_a_mqa(x)
    c_kv = self.attn_kv_a_norm(kv_a[..., :self.config.kv_lora_rank])
    k_rope = kv_a[..., self.config.kv_lora_rank:].reshape(B, Tq, 1, self.config.rope_dim).transpose(1, 2)
    if not self.config.ssm or not self.config.ssm.kda: k_rope = apply_rope(k_rope, freqs)

    k_store = c_kv.reshape(B, 1, Tq, self.config.kv_lora_rank).cat(k_rope.reshape(B, 1, Tq, self.config.rope_dim), dim=-1)[:, :, :T]
    cache = Tensor(self.cache_k.uop.after(self.cache_k[:, :, start_pos:start_pos+T, :].uop.store(k_store.cast(self.cache_k.dtype).uop)))
    (N, D), fast = self.cache_k.shape[2:], amd_custom_kernels_supported(x.device) and self.cache_k.shape[2] % 64 == 0
    fast = fast and D & (D-1) == 0
    # an indexer on the sparse path: no mask, the attention of a token gathers the keys it sees (a list)
    picks = None
    if fast and self.indexer is not None and self.indexer.sparse and isinstance(start_pos, UOp) and self.config.max_context > self.indexer.idx.top_k:
      picks, mask = self.indexer.picks(x, self.attn_q_a_norm(self.attn_q_a(x)), start_pos, T), None
    else: mask = self._attn_mask(x, start_pos, T)
    if fast:
      # flash attention on the latent cache, the keys and values of every head
      scale, latent = 1.0/self.config.head_dim**0.5, cache.reshape(1, *cache.shape)
      if Tq == 1:
        if mask is not None: mask = mask.reshape(B, -1).pad_to((B, N)).float().contiguous()
        attn = flash_attention_decode(q.half(), latent, start_pos+T, int(N), shared=True, scale=scale, mask=mask,
                                      index=picks.reshape(-1) if picks is not None else None)
      elif picks is not None: attn = flash_attention_mla_sparse(q, latent, picks, scale, start_pos+T)
      else:
        if mask is not None: mask = mask.reshape(B, Tq, -1).pad_to((B, Tq, N)).float().contiguous()
        attn = flash_attention_mla_prefill(q, latent, start_pos, start_pos+T, int(N), scale, mask)
      v_b, attn = self.attn_v_b["weight"].transpose(-1, -2), attn[..., :self.config.kv_lora_rank]
      attn = attn.half().contiguous().matmul(v_b, dtype=dtypes.float).contiguous() if half else attn @ v_b
      return _out(self.attn_output, attn.transpose(1, 2).reshape(B, Tq, -1))
    k = cache[:, :, 0:start_pos+T, :].float()
    v = k[..., :self.config.kv_lora_rank]

    attn = q @ k.transpose(-1, -2) * (1.0 / self.config.head_dim ** 0.5)
    if mask is not None: attn = attn + mask
    attn = attn.softmax(-1)
    attn = ((attn @ v) @ self.attn_v_b["weight"].transpose(-1, -2)).transpose(1, 2).reshape(B, T, -1)
    return _out(self.attn_output, attn)

  def _attn_mask(self, x:Tensor, start_pos:int|UOp, T:int|UOp|None=None) -> Tensor|None:  # (1, 1, Tq, start_pos+T), None: all visible
    if self.indexer and self.config.max_context > self.indexer.idx.top_k:
      return self.indexer(x, self.attn_q_a_norm(self.attn_q_a(x)), start_pos, T)
    T = x.shape[1]
    return Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, buffer=False).triu(start_pos+1) if resolve(T != 1) else None

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_k"):
      self.cache_k = Tensor.empty(x.shape[0], 1, self.config.max_context, self.config.kv_lora_rank + self.config.rope_dim, dtype=dtypes.half,
                                  device=x.device)
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta,
                                            device=x.device, yarn=self.config.yarn)
    if self.indexer: self.indexer._init_state(x)

class GatedDeltaNetBlock(FFNBlock):
  def __init__(self, config:TransformerConfig, ssm:SSMConfig):
    super().__init__(config)
    self.head_k_dim, self.num_k_heads, self.num_v_heads = ssm.state_size, ssm.group_count, ssm.time_step_rank
    assert self.num_v_heads % self.num_k_heads == 0
    self.head_v_dim, self.ssm_conv_kernel = ssm.inner_size // ssm.time_step_rank, ssm.conv_kernel
    self.conv_channels, self.q_dim = ssm.inner_size + 2*ssm.group_count*ssm.state_size, ssm.state_size*ssm.group_count
    # kda keeps the q/k/v projections and convolutions apart, so they can be split by heads over devices
    self.qkv_dims = (self.q_dim, self.q_dim, self.conv_channels - 2*self.q_dim)
    if ssm.kda:
      self.attn_q, self.attn_k, self.attn_v = (Linear(config.dim, d, bias=False) for d in self.qkv_dims)
      self.ssm_conv1d_q, self.ssm_conv1d_k, self.ssm_conv1d_v = ({"weight": Tensor.zeros(d, self.ssm_conv_kernel)} for d in self.qkv_dims)
    else:
      self.attn_qkv = Linear(config.dim, self.conv_channels, bias=False)
      self.ssm_conv1d = {"weight": Tensor.zeros(self.conv_channels, self.ssm_conv_kernel)}
    if ssm.kda:
      self.ssm_g_a, self.ssm_g_b = Linear(config.dim, self.head_v_dim, bias=False), Linear(self.head_v_dim, ssm.inner_size, bias=False)
      self.ssm_f_a, self.ssm_f_b = Linear(config.dim, self.head_k_dim, bias=False), Linear(self.head_k_dim, ssm.inner_size, bias=False)
    else:
      self.attn_gate = Linear(config.dim, ssm.inner_size, bias=False)
      self.ssm_alpha = Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_beta = Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_dt = {"bias": Tensor.zeros(ssm.inner_size if ssm.kda else self.num_v_heads)}
    self.ssm_a = Tensor.zeros(self.num_v_heads, 1) if ssm.kda else Tensor.zeros(self.num_v_heads)
    self.ssm_norm, self.ssm_out = nn.RMSNorm(self.head_v_dim, config.norm_eps), Linear(ssm.inner_size, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp, T:int|UOp|None=None) -> Tensor:
    # T: x has static padded rows of which the first T are the tokens (rows mode), the padded rows are computed but don't change the state
    rows = T is not None
    B, Tq, _ = x.shape
    if T is None: T = Tq
    # bind ints to a variable so the reset flag stays a runtime value (it toggles when generation restarts at position 0)
    start_pos = start_pos if isinstance(start_pos, UOp) else UOp.variable("start_pos", 0, self.config.max_context-1).bind(start_pos)
    initial = Tensor(start_pos).eq(0)
    is_kda = hasattr(self, "ssm_g_a")
    symbolic = isinstance(T, UOp) and not rows
    T_pad = x.max_shape[1]  # symbolic chunks are padded to their max size: one graph serves every size

    # input processing
    x = x.half()
    if is_kda and Tq == 1 and B == 1 and not rows and self.head_k_dim == self.head_v_dim == 128 and self.ssm_conv_kernel == 4 and \
       amd_custom_kernels_supported(x.device) and getenv("KDA_DECODE", 1):
      # one token: a kernel for the convolutions, the delta rule, the norm and the gate
      z = kda_decode(self.recurrent_state, self.conv_states, [self.attn_q(x), self.attn_k(x), self.attn_v(x)],
                     [w["weight"] for w in (self.ssm_conv1d_q, self.ssm_conv1d_k, self.ssm_conv1d_v)], self.ssm_g_b(self.ssm_g_a(x)),
                     self.ssm_f_b(self.ssm_f_a(x)), self.ssm_beta(x), self.ssm_dt["bias"], self.ssm_a, unwrap(self.ssm_norm.weight),
                     Tensor(start_pos), unwrap(self.config.ssm).gate_lower_bound, self.ssm_norm.eps)
      return self.ssm_out(z)
    out_gate = self.ssm_g_b(self.ssm_g_a(x)) if is_kda else self.attn_gate(x)
    out_gate = out_gate.reshape(B, Tq, self.num_v_heads, self.head_v_dim)
    beta = self.ssm_beta(x).sigmoid().reshape(B, Tq, self.num_v_heads)
    alpha = self.ssm_f_b(self.ssm_f_a(x)) if is_kda else self.ssm_alpha(x)
    g, a = (alpha.float() + self.ssm_dt["bias"]).reshape(B, Tq, self.num_v_heads, -1), self.ssm_a.reshape(self.num_v_heads, -1)
    # the decay: log_alpha = softplus(g) * A, or bounded by gate_lower_bound: sigmoid(g * -A) * lb
    log_alpha = g.softplus() * a if (lb:=unwrap(self.config.ssm).gate_lower_bound) is None else (g * -a).sigmoid() * lb
    if rows and Tq is not T:  # the padded rows are exact no-ops: beta=0 (delta rule off), log_alpha=0 (decay 1)
      real = Tensor.arange(Tq).reshape(1, Tq, 1) < cast(int, T)  # T can be a bound variable
      beta, log_alpha = real.where(beta, 0), real.unsqueeze(-1).where(log_alpha, 0)

    # the states (2, ...) at a position p are p%2: the ones of start_pos are read, the ones after the tokens written. the other ones keep the
    # state before: the step of a token can be taken back (a generated token that the next prompt doesn't have)
    rp, wp = start_pos % 2, (start_pos + T) % 2
    def parity(s:Tensor, p:UOp) -> Tensor: return s.shrink(((p, p+1),) + (None,)*(s.ndim-1))
    # qkv conv, conv_state is reset when starting from position 0
    def conv(states:Tensor, proj:Tensor, weight:Tensor) -> tuple[Tensor, UOp]:
      # assemble the conv window in a static-size buffer: [conv_state | qkv rows | zero-pad].
      # padded steps are exact no-ops: beta=0 (delta rule off), log_alpha=0 (decay 1 after exp)
      state = parity(states, rp).reshape(states.shape[1:])
      window = initial.where(0, state).cat(proj.cast(state.dtype), dim=1)
      window = window.pad_to((B, self.ssm_conv_kernel-1 + T_pad, proj.shape[-1])).contiguous()
      # the last conv_kernel-1 columns of the window become the next conv state
      store = parity(states, wp).uop.store(window[:, T:T+self.ssm_conv_kernel-1].cast(state.dtype).unsqueeze(0).uop)
      return functools.reduce(lambda a,b: a+b, (window[:, i:i+T_pad] * weight[:, i] for i in range(self.ssm_conv_kernel))).silu(), store
    if is_kda:
      convs = [conv(state, proj(x), w["weight"]) for state, proj, w in
               zip(self.conv_states, (self.attn_q, self.attn_k, self.attn_v), (self.ssm_conv1d_q, self.ssm_conv1d_k, self.ssm_conv1d_v))]
      (q, k, v), conv_state_stores = [o for o,_ in convs], tuple(s for _,s in convs)
    else:
      conv_out, conv_state_store = conv(self.conv_state, self.attn_qkv(x), self.ssm_conv1d["weight"])
      conv_state_stores = (conv_state_store,)
      q, k, v = conv_out.split(list(self.qkv_dims), dim=-1)
    if symbolic:
      out_gate = out_gate.pad_to((B, T_pad, self.num_v_heads, self.head_v_dim))
      beta, log_alpha = beta.pad_to((B, T_pad, self.num_v_heads)), log_alpha.pad_to((B, T_pad, *log_alpha.shape[2:]))
    q, k = (z.reshape(B, T_pad, self.num_k_heads, self.head_k_dim) for z in (q, k))
    # l2 norm x/sqrt(sum(x²)+1e-6) as in the FLA kernels, then the v heads share the k heads
    q, k = ((z.rmsnorm(eps=1e-6/self.head_k_dim) / self.head_k_dim**0.5).repeat(1, 1, self.num_v_heads//self.num_k_heads, 1) for z in (q, k))
    v = v.reshape(B, T_pad, self.num_v_heads, self.head_v_dim)
    # layout the per-step operands to broadcast against the (B, H, V, K) state
    q, k, v, beta = (z.transpose(1, 2).float() for z in (q, k, v, beta))
    q = q * self.head_k_dim**-0.5
    alpha = log_alpha.transpose(1, 2).exp()  # per-channel decay for kda, per-head otherwise (B, H, T, K|1)

    # recurrent: scan over the (padded) tokens, updating the recurrent state. collect the per-step outputs
    state = Tensor(self.recurrent_state.uop.after(*conv_state_stores))  # carry the conv write into this graph
    if self.head_k_dim % 32 == 0 and self.head_v_dim % 4 == 0 and amd_custom_kernels_supported(x.device):
      # one fused kernel for the whole scan; it resets and updates the recurrent state in place (RDNA3/4)
      core = gated_delta_prefill(q, k, v, beta, alpha, state, Tensor(start_pos), end_pos=start_pos + T).transpose(1, 2)
    else:
      q, k, v, beta = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1), beta.unsqueeze(-1).unsqueeze(-1)
      alpha = alpha.unsqueeze(-2)
      state = initial.where(0, parity(state, rp).reshape(state.shape[1:]).float())
      outs = []
      for t in range(T_pad):
        s1 = state * alpha[:, :, t]  # decay the state
        delta = (v[:, :, t] - (s1*k[:, :, t]).sum(-1, keepdim=True)) * beta[:, :, t]  # the delta rule update
        state = s1 + delta * k[:, :, t]
        outs.append((state * q[:, :, t]).sum(-1))

      # store the updated recurrent state in place, then read the stacked outputs after the write
      state_store = parity(self.recurrent_state, wp).uop.store(state.cast(self.recurrent_state.dtype).reshape(self.recurrent_state.shape[1:])
                                                                .unsqueeze(0).uop)
      core = Tensor(outs[0].stack(*outs[1:], dim=1).contiguous().uop.after(state_store))

    # output; undo the padding before the output projection
    z = (self.ssm_norm(core) * (out_gate.sigmoid() if is_kda else out_gate.silu())).cast(x.dtype).contiguous()
    if symbolic: z = z[:, :T]
    return _out(self.ssm_out, z.reshape(B, Tq, -1))

  def _init_state(self, x):
    if not hasattr(self, "recurrent_state"):
      # with heads split over devices, the states are too
      split = isinstance(x.device, tuple) and isinstance(self.ssm_a.device, tuple) and self.ssm_a.uop.axis == 0
      # two of each state: the state at a position p is p%2 (see _attention)
      def state(*shape:int, axis:int) -> Tensor:
        return (Tensor.zeros(2, *shape).shard(x.device, axis+1) if split else Tensor.zeros(2, *shape, device=x.device)).clone()
      if hasattr(self, "attn_qkv"): self.conv_state = state(x.shape[0], self.ssm_conv_kernel-1, self.conv_channels, axis=2)
      else: self.conv_states = [state(x.shape[0], self.ssm_conv_kernel-1, d, axis=2) for d in self.qkv_dims]
      self.recurrent_state = state(x.shape[0], self.num_v_heads, self.head_v_dim, self.head_k_dim, axis=1)

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
      self.blk.append(GatedDeltaNetBlock(c, config.ssm) if config.ssm and config.ssm_layers[i] else block_cls(c))
    self.token_embd  = nn.Embedding(config.vocab_size, config.dim)
    self.output_norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.output = Linear(config.dim, config.padded_vocab_size or config.vocab_size, bias=False)
    self.max_context = config.max_context
    self.has_recurrent_block = any(isinstance(b, GatedDeltaNetBlock) for b in self.blk)
    self._cached_tokens: list[int] = []
    self._ckpt_tokens: list[int] = []  # the prompt of the recurrent state's checkpoint (_checkpoint)
    # a jit for every token count of a call (the max size of a symbolic chunk, 1 for decoding): static padded shapes of a small chunk don't pay
    # for a big one
    self.jits:dict[int, Any] = {}

  def forward(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    x = self.token_embd(tokens.to(self.token_embd.weight.device)).float()  # (B, T, D)
    if (hc:=self.config.hc) is not None: x = x.unsqueeze(2).expand(*x.shape[:2], hc.count, x.shape[-1]).contiguous()
    # a symbolic chunk with hyper-connections runs every block on static padded rows (their real count is T), sliced at the end
    if hc is not None and not isinstance(T:=x.shape[1], int) and amd_custom_kernels_supported(x.device) and \
       all(isinstance(b, (MLATransformerBlock, GatedDeltaNetBlock)) for b in self.blk):
      x = x.pad_to(x.max_shape).contiguous()
      # sequence parallel: every device keeps the streams of its rows, the blocks gather and scatter the sublayers' inputs and outputs
      if sp:=self.sequence_parallel(int(x.shape[1])): x = shard_rows(x, 1)
      for block in self.blk: x = block(x, start_pos, T)
      if sp: x = all_gather(x.mean(2), 1)
      x = x[:, :T]
    else:
      for block in self.blk: x = block(x, start_pos)
    if hc is not None and x.ndim == 4: x = x.mean(2)
    # only run the output projection on the last token
    logits = self.output(self.output_norm(x[:, -1:]))[:, -1, :].to(tokens.device)[:, :self.config.vocab_size]
    q8_quantize_cache_clear()  # don't keep the graph alive
    # Gumbel-max trick: argmax(logits/temp - log(-log(uniform))) is equivalent to sampling from softmax(logits/temp)
    tok = _argmax_rows((logits / temperature.maximum(1e-12) - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()))
    # the tokens are a view of generate's token buffer: the sampled token goes after them unless that is a prompt token (chunked prefill)
    if (buf:=getattr(self, "_token_buf", None)) is not None and tokens.uop.base is buf.uop.base:
      tok = (dst:=buf[:, (nxt:=start_pos + tokens.shape[1]):nxt+1]).assign((dst < 0).where(tok.cast(buf.dtype), dst))
    return tok

  def __call__(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    # a view of the token buffer is already realized: no schedule for the jit input
    if (buf:=getattr(self, "_token_buf", None)) is None or tokens.uop.base is not buf.uop.base: tokens = tokens.contiguous()
    key = int(T.vmax) if isinstance(T:=tokens.shape[1], UOp) else T
    if (jit:=self.jits.get(key)) is None:
      # once a jit ran the model (its states and caches are set up), a new one captures on its first call
      jit = self.jits[key] = TinyJit(self.forward, capture_first=any(j.cnt > 0 for j in self.jits.values()))
    return jit(tokens, start_pos, temperature)

  @staticmethod
  def from_gguf(gguf:Tensor|str|pathlib.Path, max_context:int|None=None,
                realize=bool(getenv("REALIZE", 0)), shard:int=1) -> tuple[Transformer, dict]:
    # TODO: remove the need for copy to default device
    kv, entries = gguf_parse(gguf.to(None).realize() if isinstance(gguf, Tensor) else gguf)
    arch = kv['general.architecture']
    n_heads, n_kv_heads = kv[f'{arch}.attention.head_count'], kv[f'{arch}.attention.head_count_kv']
    assert shard >= 1, f"shard must be at least 1, got {shard}"
    # the next token prediction (MTP) layers are not used
    num_blocks = kv[f'{arch}.block_count'] - kv.get(f'{arch}.nextn_predict_layers', 0)
    entries = {k:v for k,v in entries.items() if not ((m:=re.match(r"blk\.(\d+)\.", k)) and int(m.group(1)) >= num_blocks)}
    shard_map:dict[str, int] = {}
    units_map:dict[str, tuple[int, list[list[int]]]] = {}
    pads:dict[str, int] = {}
    padded_heads, sizes, expert_slices = n_heads, {}, 0
    if shard > 1:
      # MLA and kda attention split all their tensors by the n_heads heads: those are zero padded to a multiple of the devices
      pad_heads = bool(kv.get(f'{arch}.attention.kv_lora_rank')) or arch in ('kimi-linear', 'glm5-next')
      if not pad_heads: assert n_kv_heads % shard == 0, f"tensor parallel needs the attention heads to split over {shard} devices"
      padded_heads = round_up(n_heads, shard) if pad_heads else n_heads
      # shard heads, FFNs and routed experts while replicating latent projections, KV cache and the small kda projections
      head_rules = {**{w: 0 for w in ('attn_q.weight', 'attn_k.weight', 'attn_v.weight', 'attn_q_b.weight', 'attn_k_b.weight', 'attn_v_b.weight')},
                    'attn_output.weight': 1}
      # the kda layers split by heads too (the gated deltanet of qwen3.5 is copied to every device)
      if arch in ('kimi-linear', 'glm5-next'): head_rules.update({w: 0 for w in ('ssm_conv1d_q.weight', 'ssm_conv1d_k.weight', 'ssm_conv1d_v.weight',
        'ssm_dt.bias', 'ssm_a', 'ssm_beta.weight', 'ssm_f_b.weight', 'ssm_g_b.weight')})
      ffn_rules = {**{w: 0 for w in ('ffn_gate.weight', 'ffn_up.weight', 'ffn_gate_shexp.weight', 'ffn_up_shexp.weight', 'output.weight')},
                   **{w: 1 for w in ('ffn_down.weight', 'ffn_down_shexp.weight')}}
      if len(kv['tokenizer.ggml.tokens']) % shard == 0: ffn_rules['token_embd.weight'] = 0
      # experts are split by their hidden dim, or by experts when it doesn't split (expert parallel)
      exp_hidden, exp_count = kv.get(f'{arch}.expert_feed_forward_length', 0), kv.get(f'{arch}.expert_count', 0)
      down_block = max([_GGML_QUANT.get(t, (1,))[0] for k,(_,_,t) in entries.items() if k.endswith('ffn_down_exps.weight')], default=1)
      exp_names = ('ffn_gate_exps.weight', 'ffn_up_exps.weight', 'ffn_down_exps.weight')
      if exp_hidden % shard == 0 and (exp_hidden // shard) % down_block == 0:
        ffn_rules.update({'ffn_gate_exps.weight': 1, 'ffn_up_exps.weight': 1, 'ffn_down_exps.weight': 2})
      elif exp_count and exp_hidden % down_block == 0 and getenv("EXPERT_SLICES", 1) and \
           (slicing:=expert_slicing(exp_count, exp_hidden // down_block, shard)) is not None:
        # slices of the experts' hidden dim (whole quant blocks of the down projection) spread over the devices: the work of a token is balanced
        expert_slices = exp_hidden // down_block
        units_map = {name: (expert_slices, slicing) for name in entries if re.sub(r"^blk\.\d+\.", "", name) in exp_names}
      elif exp_count % shard == 0: ffn_rules.update({w: 0 for w in exp_names})
      for name, (_, shape, typ) in entries.items():
        if (k:=re.sub(r"^blk\.\d+\.", "", name)) in head_rules:
          axis = shard_map[name] = head_rules[k]
          sizes[name] = shape[axis] // n_heads * padded_heads
        elif k in ffn_rules:
          axis = shard_map[name] = ffn_rules[k]
          # FFN hidden dims pad to whole quant blocks of the down projection on every device
          down = entries.get(re.sub(r"ffn_(gate|up)(_shexp)?\.", r"ffn_down\2.", name), entries[name])
          unit = 1 if k == 'output.weight' or '_exps' in k else _GGML_QUANT.get(down[2], (1,))[0]
          sizes[name] = round_up(shape[axis], shard*unit)
      pads = {k: sizes[k] for k in sizes if sizes[k] != entries[k][1][shard_map[k]]}
    devices = tuple(Device.canonicalize(f'{Device.DEFAULT}:{i}') for i in range(shard))
    # gate and up experts of the same format load as one (fuse_gate_up)
    fuse = {g: u for g in units_map if g.endswith("ffn_gate_exps.weight") and (u:=g.replace("gate", "up")) in units_map and
            entries[g][2] == entries[u][2] and getenv("FUSE_GATE_UP", 1)}
    state_dict = gguf_shard(entries, devices, shard_map, units_map, pads if shard > 1 else {}, fuse)

    # all state items should be float16, not float32
    state_dict = {k:v.cast('float16') if getenv("HALF", 1) else v for k,v in state_dict.items()}

    # some models like Llama 3.2 don't have an output.weight, they just tie to the token_embd.weight
    if 'output.weight' not in state_dict: state_dict['output.weight'] = state_dict['token_embd.weight']

    hc = HCConfig(kv[f'{arch}.hyper_connection.count'], kv[f'{arch}.hyper_connection.sinkhorn_iterations'],
                  kv[f'{arch}.hyper_connection.epsilon']) if f'{arch}.hyper_connection.count' in kv else None
    if hc is not None: state_dict = {k.replace('.hc_attn_', '.hc_attn.').replace('.hc_ffn_', '.hc_ffn.'):v for k,v in state_dict.items()}

    indexer = IndexerConfig(kv[f'{arch}.attention.indexer.head_count'], kv[f'{arch}.attention.indexer.key_length'],
                            kv[f'{arch}.attention.indexer.top_k'], kv[f'{arch}.attention.indexer.kpool'],
                            kv[f'{arch}.attention.layer_norm_epsilon']) if f'{arch}.attention.indexer.kpool' in kv else None
    if indexer is not None: state_dict = {k.replace('.indexer_compressor_', '.indexer.compressor_'):v for k,v in state_dict.items()}

    max_context = min(max_context, kv[f'{arch}.context_length']) if max_context is not None else kv[f'{arch}.context_length']

    ssm = None
    ssm_layers: tuple[bool, ...] = ()
    if arch in ('qwen35', 'qwen35moe'):
      ssm = SSMConfig(**{k: kv[f'{arch}.ssm.{k}'] for k in ('conv_kernel','state_size','group_count','time_step_rank','inner_size')})
      ssm_layers = tuple((i+1) % kv[f'{arch}.full_attention_interval'] != 0 for i in range(num_blocks))
    elif arch in ('kimi-linear', 'glm5-next'):
      ssm_layers = tuple(x == 0 for x in n_kv_heads)[:num_blocks]
      n_kv_heads = max(n_kv_heads)
      ssm = SSMConfig(kv[f'{arch}.ssm.conv_kernel'], kv[f'{arch}.kda.head_dim'], padded_heads, padded_heads,
                      padded_heads*kv[f'{arch}.kda.head_dim'], kda=True, gate_lower_bound=kv.get(f'{arch}.kda.gate_lower_bound'))
      for i, is_ssm in enumerate(ssm_layers):
        if not is_ssm: continue
        state_dict[f"blk.{i}.ssm_a"] = state_dict[f"blk.{i}.ssm_a"].reshape(padded_heads, 1)
        for c in "qkv": state_dict[f"blk.{i}.ssm_conv1d_{c}.weight"] = state_dict[f"blk.{i}.ssm_conv1d_{c}.weight"].squeeze(1)
        state_dict[f"blk.{i}.ssm_out.weight"] = state_dict.pop(f"blk.{i}.attn_output.weight")
    if arch in ('qwen35', 'qwen35moe', 'glm4moe', 'gpt-oss'):
      state_dict = {k.replace('post_attention_norm', 'ffn_norm'):v for k,v in state_dict.items()}

    kv_lora_rank = kv.get(f'{arch}.attention.kv_lora_rank', 0)
    head_dim = kv.get(f'{arch}.attention.key_length_mla', kv.get(f'{arch}.attention.key_length', kv[f'{arch}.embedding_length'] // n_heads))
    rope_dim = kv.get(f'{arch}.rope.dimension_count', head_dim)
    yarn = YaRNConfig(factor=kv[f'{arch}.rope.scaling.factor'],
                      orig_ctx_len=kv.get(f'{arch}.rope.scaling.original_context_length', kv[f'{arch}.context_length']),
                      beta_fast=kv.get(f'{arch}.rope.scaling.yarn_beta_fast', 32.0),
                      beta_slow=kv.get(f'{arch}.rope.scaling.yarn_beta_slow', 1.0)) if kv.get(f'{arch}.rope.scaling.type') == 'yarn' else None

    # Permute RoPE weights from interleaved to half-split layout.
    for name in state_dict:
      if arch in ('kimi-linear', 'glm5-next'): continue
      if ('attn_q.weight' in name or 'attn_q_b.weight' in name) and (arch == 'llama' or kv_lora_rank):
        w = state_dict[name].reshape(n_heads, state_dict[name].shape[0]//n_heads, -1)
        prefix = head_dim-rope_dim
        state_dict[name] = w[:, :prefix].cat(w[:, prefix:].rearrange("n (h two) d -> n (two h) d", two=2), dim=1).reshape(-1, w.shape[-1])
      elif arch == 'llama' and 'attn_k.weight' in name:
        w = state_dict[name].reshape(n_kv_heads, state_dict[name].shape[0]//n_kv_heads, -1)
        state_dict[name] = w.rearrange("n (h two) d -> n (two h) d", two=2).reshape(-1, w.shape[-1])
      elif kv_lora_rank and 'attn_kv_a_mqa.weight' in name:
        state_dict[name] = state_dict[name][:kv_lora_rank].cat(state_dict[name][kv_lora_rank:].rearrange("(h two) d -> (two h) d", two=2), dim=0)
    config = TransformerConfig(
      num_blocks=num_blocks, dim=kv[f'{arch}.embedding_length'],
      hidden_dim=kv.get(f'{arch}.expert_feed_forward_length', sizes.get('blk.0.ffn_gate.weight', kv.get(f'{arch}.feed_forward_length', 0))),
      n_heads=padded_heads, n_kv_heads=n_kv_heads, norm_eps=kv[f'{arch}.attention.layer_norm_rms_epsilon'],
      vocab_size=len(kv['tokenizer.ggml.tokens']), padded_vocab_size=int(state_dict['output.weight'].shape[0]),
      head_dim=head_dim,
      rope_theta=kv.get(f'{arch}.rope.freq_base', 10000.0), rope_dim=rope_dim, yarn=yarn,
      v_head_dim=kv.get(f'{arch}.attention.value_length_mla', kv.get(f'{arch}.attention.value_length', head_dim)),
      max_context=max_context,
      qk_norm=int(state_dict['blk.0.attn_q_norm.weight'].shape[0]) if 'blk.0.attn_q_norm.weight' in state_dict else 0,
      num_experts=kv.get(f'{arch}.expert_count', 0), num_experts_per_tok=kv.get(f'{arch}.expert_used_count', 0),
      norm_topk_prob=kv.get(f'{arch}.expert_weights_norm', arch in ('qwen3moe', 'qwen35moe', 'kimi-linear')),
      expert_gating_func=ExpertGating(kv.get(f'{arch}.expert_gating_func',
        ExpertGating.SOFTMAX_WEIGHT if arch == 'gpt-oss' else ExpertGating.SOFTMAX)),
      kv_lora_rank=kv_lora_rank, q_lora_rank=kv.get(f'{arch}.attention.q_lora_rank', 0),
      leading_dense_blocks=kv.get(f'{arch}.leading_dense_block_count', 0),
      shared_expert_dim=sizes.get(f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.ffn_gate_shexp.weight", kv.get(
        f'{arch}.expert_shared_feed_forward_length',
        kv.get(f'{arch}.expert_shared_count', 0) * kv.get(f'{arch}.expert_feed_forward_length', 0))),
      shared_expert_gate=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.ffn_gate_inp_shexp.weight" in state_dict,
      dense_hidden_dim=sizes.get('blk.0.ffn_gate.weight', kv.get(f'{arch}.feed_forward_length', 0))
        if kv.get(f'{arch}.leading_dense_block_count', 0) else 0,
      routed_scaling_factor=kv.get(f'{arch}.expert_weights_scale', 1.0), attn_output_gate=arch in ('qwen35', 'qwen35moe'), ssm=ssm,
      ssm_layers=ssm_layers,
      qkv_bias='blk.0.attn_q.bias' in state_dict,
      expert_bias=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.exp_probs_b.bias" in state_dict, expert_slices=expert_slices,
      expert_proj_bias='blk.0.ffn_gate_exps.bias' in state_dict, attn_output_bias='blk.0.attn_output.bias' in state_dict,
      swiglu_alpha=1.702 if arch == 'gpt-oss' else 1.0, swiglu_limit=7.0 if arch == 'gpt-oss' else kv.get(f'{arch}.swiglu_clamp_exp', [None])[0],
      swiglu_up_bias=1.0 if arch == 'gpt-oss' else 0.0, attn_sinks='blk.0.attn_sinks.weight' in state_dict,
      sliding_window=kv.get(f'{arch}.attention.sliding_window', 0),
      sliding_window_pattern=kv.get(f'{arch}.attention.sliding_window_pattern', 2 if arch == 'gpt-oss' else 0), hc=hc, indexer=indexer)
    model = Transformer(config)
    # gate experts loaded with the up rows (fused): one weight of twice the rows
    for i, b in enumerate(model.blk):
      if (g:=state_dict.get(f"blk.{i}.ffn_gate_exps.weight")) is not None and f"blk.{i}.ffn_up_exps.weight" not in state_dict and \
         hasattr(b, "ffn_up_exps"):
        e = ExpertWeights(int(g.shape[0]), int(g.shape[2]), int(g.shape[1]))
        e.fused, e.slices, e.experts = True, b.ffn_gate_exps.slices, b.ffn_gate_exps.experts
        b.ffn_gate_exps = e
        del b.ffn_up_exps
    for p in (nn.state.get_parameters(model) if shard > 1 else []): p.to_(devices)
    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)  # NOTE: rope_freqs.weight (32,) is unused
    # NOTE: without this contiguous, it unpacks the weights from the model every time. we shouldn't need this, but for now it's faster
    if realize:
      for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
      Tensor.realize(*params)
    else: model.prepare()
    return model, kv

  def prepare(self):
    """use the packed ggml weights in the quantized kernels, and realize the small weights that aren't: they aren't decoded at every call"""
    def modules(obj, seen:set[int]):
      if id(obj) in seen or isinstance(obj, (Tensor, str, int, float, bool)) or obj is None: return
      seen.add(id(obj))
      if isinstance(obj, (Linear, ExpertWeights)): yield obj
      for v in (obj if isinstance(obj, (list, tuple)) else obj.values() if isinstance(obj, dict) else getattr(obj, '__dict__', {}).values()):
        yield from modules(v, seen)
    packed:set[int] = set()
    for m in modules([self.blk, self.output], set()):
      if isinstance(m, Linear) and m.ggml_type is None and m.use_custom_quant and amd_custom_kernels_supported(m.weight.device):
        m.set_quantized(m.weight)
      if isinstance(m, ExpertWeights): m.set_quantized()
      if m.ggml_type is not None: packed.add(id(m.weight))
    # packed gate and up experts of the same format are one weight: their rows interleaved
    for b in self.blk:
      if (g:=getattr(b, 'ffn_gate_exps', None)) is not None and hasattr(b, 'ffn_up_exps') and g.ggml_type is not None and \
         g.ggml_type == b.ffn_up_exps.ggml_type and \
         amd_custom_kernels_supported(g.weight.device) and getenv("FUSE_GATE_UP", 1):
        packed.discard(id(b.ffn_up_exps.weight))
        packed.discard(id(g.weight))
        g.fuse_up(b.ffn_up_exps, b.config.hidden_dim // (b.config.expert_slices or 1))
        del b.ffn_up_exps
        packed.add(id(g.weight))
    small = [p for p in nn.state.get_parameters(self) if id(p) not in packed and p is not self.token_embd.weight and not p.uop.is_realized]
    for p in small: p.replace(p.contiguous())
    Tensor.realize(*small)

  def sequence_parallel(self, rows:int) -> bool:
    """the hyper-connection streams of a prefill chunk of `rows` are split over the devices: tiles of 16 rows on every device"""
    devs = self.token_embd.weight.device
    return isinstance(devs, tuple) and self.config.hc is not None and rows % (16*len(devs)) == 0 and bool(getenv("SEQ_PARALLEL", 1))

  def prefill_chunks(self, chunk_size:int|None=None) -> tuple[int, int]:
    """the (max, small) prefill chunk sizes: the custom kernels compute chunks padded to a static size, a prompt that fits the small size
    runs it (its own jit), large chunks are efficient. sequence parallel: whole tiles of rows on every device"""
    fast = amd_custom_kernels_supported(self.token_embd.weight.device)
    chunk_size = chunk_size or min(getenv("PREFILL_CHUNK", 2048 if fast else 32), self.max_context)
    if self.has_recurrent_block and not fast: chunk_size = 1
    small = min(chunk_size, getenv("PREFILL_SMALL", 512 if fast else chunk_size))
    if fast and (devs:=self.token_embd.weight.device) and isinstance(devs, tuple) and self.config.hc is not None and getenv("SEQ_PARALLEL", 1):
      # whole tiles of 16 rows on every device, the kernels of the whole chunk on tiles of 64
      tile = math.lcm(16*len(devs), 64)
      def tiled(n:int) -> int: return min(max(tile, round(n / tile) * tile), self.max_context // tile * tile or tile)
      chunk_size, small = tiled(chunk_size), tiled(small)
    return chunk_size, small

  def warmup(self):
    """compile and capture every jit a request uses: a prompt of each prefill chunk size (twice: a jit captures on its second call), decoding"""
    chunk, small = self.prefill_chunks()
    for n in sorted({1, small + 1 if chunk > small else 1}):
      for _ in range(2):
        self._cached_tokens, self._ckpt_tokens = [], []
        list(zip(range(4), self.generate([0] * min(n, self.max_context - 8))))
    self._cached_tokens, self._ckpt_tokens = [], []

  def _recurrent_states(self) -> list[Tensor]:
    # the state that isn't indexed by position: the recurrent and conv states, the indexers' rings of the last pool
    states:list[Tensor] = []
    for b in self.blk:
      if isinstance(b, GatedDeltaNetBlock) and hasattr(b, "recurrent_state"):
        states += [b.recurrent_state, *getattr(b, "conv_states", [getattr(b, "conv_state", None)])]
      if (idx:=getattr(b, "indexer", None)) is not None and hasattr(idx, "ring"): states.append(idx.ring)
    return [s for s in states if s is not None]

  def _checkpoint(self, tokens:list[int], save:bool):
    """
    a copy of the recurrent state at the end of a prompt (save) or back from it: the next prompt has this one, but the model's reply can come
    back different (re-rendered by a chat template, re-tokenized): the reply and what follows are prefilled again, not everything
    """
    if not (states:=self._recurrent_states()): return
    if not hasattr(self, "_ckpt"): self._ckpt = [s.clone().realize() for s in states]
    pairs = zip(self._ckpt, states) if save else zip(states, self._ckpt)
    Tensor.realize(*[dst.assign(src) for dst, src in pairs])
    if save: self._ckpt_tokens = tokens[:]

  def _reuse(self, tokens:list[int]) -> tuple[int, bool]:
    # recurrent state can't be partially reused after divergence: the cached prefix if the tokens extend it, else the checkpoint's (restored)
    cached, ckpt = self._cached_tokens, getattr(self, "_ckpt_tokens", [])
    if cached and len(cached) < len(tokens) and tokens[:len(cached)] == cached: return len(cached), False
    if ckpt and len(ckpt) < len(tokens) and tokens[:len(ckpt)] == ckpt: return len(ckpt), True
    return 0, False

  def get_start_pos(self, tokens:list[int]) -> int:
    if self.has_recurrent_block: return self._reuse(tokens)[0]
    prefix_len = sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], zip(tokens[:-1], self._cached_tokens)))
    return min(block._reusable_prefix_len(prefix_len, len(self._cached_tokens)) for block in self.blk)

  def generate(self, tokens:list[int], chunk_size:int|None=None, temperature:float=0.0):
    # the prefill jits are specialized for one chunk size each
    chunk_size, small = self.prefill_chunks(chunk_size)
    v_start_pos = UOp.variable("start_pos", 0, self.max_context-1)
    v_toks, v_small = UOp.variable("toks", 1, chunk_size), UOp.variable("toks", 1, small)
    # TODO: use UOp.variable for temperature once float variables are supported
    temp = Tensor([temperature])
    devs = to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device))
    if fast:=amd_custom_kernels_supported(self.token_embd.weight.device):
      # in host memory: every step writes its sampled token after its input tokens (see forward), the host polls it
      if not hasattr(self, "_token_buf"):
        buf = Buffer(devs[0], 4*(self.max_context+1), options=BufferSpec(host=True, cpu_access=True)).ensure_allocated()
        self._token_host, self._token_buf = buf.as_memoryview(allow_zero_copy=True).cast('i'), Tensor(UOp.from_buffer(buf, dtypes.int32))[None]
      for d in devs: Device[d].synchronize()  # a step run ahead by the last generate may still use it
      self._token_host[:] = array.array('i', tokens + [-1] * (self.max_context + 1 - len(tokens)))
      t = self._token_buf
    else:
      # assign all input tokens once, then slice from start_pos for the model call
      t = Tensor(tokens + [0] * (self.max_context - len(tokens)), dtype="int32").reshape(1, self.max_context)
    # recompute start_pos from what's currently valid in the caches
    start_pos = self.get_start_pos(tokens)
    if self.has_recurrent_block:
      start_pos, restore = self._reuse(tokens)
      if restore: self._checkpoint(tokens, save=False)
      # a prefill from before the checkpoint overwrites the caches it relies on
      elif start_pos < len(getattr(self, "_ckpt_tokens", [])): self._ckpt_tokens = []
    out, prompt_len, ahead = None, len(tokens), False
    while len(tokens) < self.max_context:
      n_toks = min(chunk_size, len(tokens) - start_pos)
      # the prompt in chunks (symbolic sizes), then one token at a time (the decode jit)
      sp, nt = v_start_pos.bind(start_pos), 1 if start_pos >= prompt_len else (v_small if n_toks <= small else v_toks).bind(n_toks)
      # with run ahead, the previous step already submitted this one
      if not ahead: out = self(t[:, sp:sp+nt] if start_pos < prompt_len or out is None or fast else out, sp, temp).realize()
      # the token is polled only if the call wrote it to the token buffer (a replaced __call__ doesn't)
      if fast and (out is None or out.uop.base is not self._token_buf.uop.base): fast = False
      start_pos += n_toks
      # chunked prefill: keep processing until all prompt tokens are consumed
      if start_pos < len(tokens): continue
      # the end of the prompt: a checkpoint of the recurrent state (before any step of the reply)
      if self.has_recurrent_block and start_pos == prompt_len and getenv("CHECKPOINT", 1): self._checkpoint(tokens, save=True)
      if fast:
        # run ahead: submit the next step (it reads this token on the device) before waiting for this one. it writes the recurrent states of
        # the next position, the ones before it are kept (see GatedDeltaNetBlock._attention): the caches still hold every token but the last
        if ahead:=len(tokens) + 1 < self.max_context: self(t[:, (sp:=v_start_pos.bind(start_pos)):sp+1], sp, temp)
        while (tok:=self._token_host[start_pos]) < 0: pass
      tokens.append(tok if fast else int(cast(Tensor, out).item()))
      self._cached_tokens = tokens[:-1]
      yield tokens[-1]
