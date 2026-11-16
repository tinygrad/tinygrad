from __future__ import annotations
import array, enum, functools, itertools, math, pathlib, re, time
from dataclasses import dataclass, replace, field
from typing import Any, Callable, Iterator, cast
from tinygrad import Tensor, nn, UOp, TinyJit, getenv, function, dtypes, Device
from tinygrad.engine.jit import capture_parallel, warmup_wanted, _TinyJit
from tinygrad.engine import jit as jit_mod
from tinygrad.engine.realize import warm_worker_pool
from tinygrad.llm.kernels.amd_glm_dsa import Linear, amd_custom_kernels_supported, find_packed, moe_linear, \
  q8_linear, q8_quantize_cache_clear, amd_flash_attention_decode as flash_attention_decode, moe_route, \
  moe_grouped_linear, amd_flash_attention_mla_prefill as flash_attention_mla_prefill, partial_sum, fuse_gate_up, moe_router, \
  indexer_scores, _wmma_rdna4, pool_keys, pool_index_rows, indexer_decode_index, amd_flash_attention_mla_sparse as flash_attention_mla_sparse, \
  moe_combine, token_sharded, shard_rows, all_gather, all_gather_q8, sharded_argmax, skinny_matmul, moe_combine_scatter, tables_ready, \
  moe_router_local, indexer_decode_index_rows, Q8_ROWS_I8, MOE_TILE, grouped_rows, mla_bmm, bmm_wmma, weighted_sum, rows_gather, \
  nucleus, seq_store, mla_sparse_seq, seq_views, seq_rows, SEQ_BLK, seq_gather, seq_store_grouped, FP8_ROW, FP8_ROPE_ROW, \
  IDX_FP8_ROW, pool_index_shard, Q4_ROW, Q4_ROPE_ROW, reduced_rows
from tinygrad.llm.kernels.amd_glm_dsa import p2p_check, host_allgather, Q8_0, lists_pad, GEMM_ROWS
from tinygrad.llm.kernels.amd_glm_dsa import sp_reduce_scatter, all_to_all, mla_sparse_tokens
from tinygrad.llm.glm_dsa_gguf import gguf_parse, gguf_shard, ggml_data_to_tensor, _GGML_QUANT
from tinygrad.llm import glm5next as glm_seq_base
from tinygrad.uop.ops import resolve, Ops
from tinygrad.dtype import DType
from tinygrad.helpers import unwrap, round_up, to_tuple, prod, dedup
from tinygrad.device import Buffer, BufferSpec, BufferStorage, MultiBuffer
from tinygrad.llm.kernels.amd_glm_dsa import P2PRunOut

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
    # sliced: the selections are mapped to the local slices of every device (slice_table), the ids are local. the same for every layer (one
    # tensor), realized with the model's small weights (prepare)
    self.first = _expert_first(tuple(0 if self.slices else i*self.local_experts for i in range(n)), devices)
    if self.slices: self.slice_table = _slice_table(self.experts, self.slices, devices)
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
  def __call__(self, sel:Tensor, x:Tensor, out_dtype:DType=dtypes.float32) -> Tensor:
    # sel: (B, T, k), x: (B, T, 1, in) or (B, T, k, in) -> output: (B, T, k, out) (out_dtype: the bundles of batched decoding's)
    # expert parallel: x and the output have a leading device axis, (devices, B, T, k, out) partial outputs from the experts of each device
    self.set_quantized()
    if self.ggml_type is not None:
      if isinstance(sel.numel(), int): return moe_linear(self.weight, self.ggml_type, self.local_experts, x, sel, self.first, out_dtype)
      # symbolic token count: pad to the max chunk size so the kernels see static shapes, garbage rows are sliced off
      out = moe_linear(self.weight, self.ggml_type, self.local_experts, x.pad_to(x.max_shape), sel.pad_to(sel.max_shape), self.first)
      return out.shrink(tuple((0, s) for s in out.shape[:-4])+tuple((0, s) for s in sel.shape)+((0, out.shape[-1]),))
    ret = (x.unsqueeze(-2) @ self.weight[sel].transpose(-1, -2)).contiguous().squeeze(-2)
    return ret + self.bias[sel] if hasattr(self, 'bias') else ret

@functools.cache
def _expert_first(first:tuple[int, ...], devices:tuple[str, ...]) -> Tensor:
  # (from host memory: no peer copies)
  t = Tensor(list(first), dtype=dtypes.int32, device="CPU")
  return t.shard(devices, 0) if len(first) > 1 else t.to(devices[0] if len(devices) == 1 else devices)

@functools.cache
def _slice_table(experts:int, slices:int, devices:tuple[str, ...]) -> Tensor:
  tables = [expert_slice_table(units, experts, slices) for units in unwrap(expert_slicing(experts, slices, len(devices)))]
  return Tensor(tables, dtype=dtypes.int32, device="CPU").shard(devices, 0)

def expert_slicing(experts:int, slices:int, n:int) -> list[list[int]]|None:
  """
  the units (expert*slices + slice) of every device: every device holds slices//n slices of every expert, the slices%n left go to a class of
  devices (by expert, cycling): the work of a token's experts is spread over all the devices. None if the devices don't get the same count
  """
  base, r = divmod(slices, n)
  if r and (n % r or experts % (n // r)):
    # (EXPERT_PAD: the units in turn over the devices (an expert's slices on different ones), the devices padded to the same count with -1)
    if not (1) or slices > n: return None
    units = [[u for u in range(experts*slices) if u % n == d] for d in range(n)]
    w = max(len(p) for p in units)
    return [p + [-1]*(w-len(p)) for p in units]
  per:list[list[int]] = [[] for _ in range(n)]
  for e in range(experts):
    extra = set(range((e % (n // r))*r, (e % (n // r))*r + r)) if r else set()
    devs = [d for d in range(n) if d in extra for _ in range(base+1)] + [d for d in range(n) if d not in extra for _ in range(base)]
    for j, d in enumerate(devs): per[d].append(e*slices + j)
  return per

def expert_slice_table(units:list[int], experts:int, slices:int) -> list[list[int]]:
  """(experts, w): the local index of every slice a device holds of an expert, -1 padded"""
  table:list[list[int]] = [[] for _ in range(experts)]
  for i, u in enumerate(units):
    if u >= 0: table[u // slices].append(i)
  w = max(len(t) for t in table)
  return [t + [-1]*(w-len(t)) for t in table]

# DSA: the mask of the indexer of the last full layer (the input of the shared layers after it), and the one a block's indexer made
_idx_mask:list[Tensor|None] = [None, None]
INLINE_ACT = (1)  # a decoding activation computed in the kernel that quantizes it (not a kernel of its own)

def gated_activation(gate:Tensor, up:Tensor, *, alpha:float=1.0, limit:float|None=None, up_bias:float=0.0) -> Tensor:
  if limit is not None and limit > 0: gate, up = gate.clamp(max_=limit), up.clamp(-limit, limit)
  # TODO: remove the need for this contiguous (gate and up from the custom kernels: one kernel)
  sig = (gate * alpha).sigmoid()
  return gate * (sig if amd_custom_kernels_supported(gate.device) else sig.contiguous()) * (up + up_bias)

def apply_rope(x:Tensor, freqs_cis:Tensor, interleaved:bool=False) -> Tensor:
  assert x.shape[-1] % 2 == 0
  if interleaved: x = x.rearrange("b h t (d two) -> b h t (two d)", two=2)
  cos, sin = freqs_cis.reshape(1, 1, x.shape[2], -1).chunk(2, dim=-1)
  x1, x2 = x.chunk(2, dim=-1)
  return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)

def mla_q_cat(q_nope:Tensor, q_rope:Tensor) -> Tensor:
  """
  q_nope.cat(q_rope, dim=-1) with q_nope (a matmul's output, lazy) and q_rope stored into the two slices of one buffer of q_nope's dtype
  (MLA_QCAT, default on: the same values; not the matmul's buffer and a copy of both into the cat's, 0.3 ms a layer of a 2688-token chunk)
  """
  lora = q_nope.shape[-1]
  qb = Tensor.empty(*q_nope.shape[:-1], lora + q_rope.shape[-1], dtype=q_nope.dtype, device=q_nope.device,
                    axis=q_nope.uop.axis if isinstance(q_nope.device, tuple) else None)
  return Tensor(qb.uop.after(qb[..., :lora].uop.store(q_nope.uop), qb[..., lora:].uop.store(q_rope.cast(q_nope.dtype).uop)))

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
class IndexerConfig:  # select top_k tokens by scoring individual keys or pools of kpool keys
  n_heads: int
  head_dim: int
  top_k: int
  kpool: int = 1
  norm_eps: float = 1e-6
  rope_dim: int = 0
  rope_interleave: bool = True

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
  ssm_layers: tuple[bool, ...] = ()  # (glm-dsa: no recurrent blocks; glm5next's shared methods read it)
  attn_output_gate: bool = False
  attn_output_bias: bool = False
  attn_sinks: bool = False
  ssm: None = None  # (glm-dsa: no recurrent blocks)
  shared_expert_gate: bool = True
  leading_dense_blocks: int = 0
  dense_hidden_dim: int = 0
  routed_scaling_factor: float = 1.0
  qkv_bias: bool = False
  expert_bias: bool = False
  expert_proj_bias: bool = False
  expert_slices: int = 0  # the experts are split in slices of their hidden dim, spread over the devices (see expert_slicing)
  expert_units: int = 0  # (sliced) the slices of all the devices, padded to the same count on every one
  swiglu_alpha: float = 1.0
  swiglu_limit: float|None = None
  swiglu_up_bias: float = 0.0
  sliding_window: int = 0
  sliding_window_pattern: int = 0
  hc: None = None  # (no hyper-connections; glm5next's shared methods read it)
  indexer: IndexerConfig|None = None
  indexer_shared: tuple[bool, ...] = ()
  mtp: bool = False  # the next token prediction layer (MTPBlock)
  rope_interleave: bool = False  # the rope dims rotate adjacent pairs (the gguf's layout of a deepseek style model, not permuted)
  spec_drafts: int = 0  # speculative decoding: the tokens the MTP layer drafts for every step of the model
  # (SPEC_DRAFTS_BY_SIZE: the drafts of a step of m sequences (GLM_SEQ), (m, drafts) a size; spec_drafts the most of them: the layouts')
  spec_by_size: tuple[tuple[int, int], ...] = ()
  @property
  def rewind(self) -> int:
    """
    how far before the last position the recurrent states write the next call can start: a token (a generated token that the next prompt doesn't
    have), with speculative decoding the tokens of a step (one more than the drafts) and the drafts of the one launched after it"""
    return 2*self.spec_drafts + 1 if self.spec_drafts else 1
  padded_vocab_size: int = 0  # the output rows, zero padded to split over devices
  real_heads: int = 0  # (the heads before they were padded to split over the devices, 0: n_heads)
  cache_rows: int = 0  # (GLM_SEQ: the regions of the sequences sharing the caches)
  qsa: bool = False  # (the generic SeqEngine driver distinguishes Qwen sparse attention)
  ple: Any = None  # (GLM has no prefix-learning embeddings)

# Share the generic GLM SeqEngine driver's traced state, not its model block types.
_seq_base = glm_seq_base._seq_base
_seq_batch = glm_seq_base._seq_batch
def _seq_alone() -> bool: return glm_seq_base._seq_alone()
# A whole SEQ_BLK block on every device of the 14-GPU target, and a multiple of every supported indexer pool size.
REGION_ALIGN = 3584

def _f8_pack(x:Tensor) -> Tensor:
  """x's rows (last axis) as e4m3 bytes of a scale a row (its absolute max over 448) and the scale's 4 bytes after them: (..., D+4) uint8"""
  sc = x.float().abs().max(-1, keepdim=True).maximum(1e-20) / 448
  return (x.float() / sc).cast(dtypes.fp8e4m3).bitcast(dtypes.uint8).cat(sc.bitcast(dtypes.uint8), dim=-1)

def _f8_unpack(p:Tensor, D:int) -> Tensor:
  """the half rows of _f8_pack's"""
  return (p[..., :D].bitcast(dtypes.fp8e4m3).float() * p[..., D:D+4].contiguous().bitcast(dtypes.float32)).half()

_partial_out = [False]
_partial_half = [True]  # (and in half: sent so. GLM's sequence parallel sends them as half rows with a scale, from float)
# (two batch overlap: the selections and probabilities of every row, routed before)
_sp_route:list[tuple[Tensor, Tensor]|None] = [None]
# and the sublayer's input rows of this device (before the gather): per token work (the router) runs on them, its results are gathered
_sp_rows:list[Tensor|None] = [None]
# (SP_ROWPROJ: a sequence parallel attention sublayer's row projections made on the rows' own device, gathered: see _sp_rowproj)
_sp_pre:list[dict|None] = [None]
# the tokens of a prefill chunk (its rows past them are padding): the experts don't take the padding rows
_ffn_rows:list[UOp|int|None] = [None]
def _out(layer:Linear, x:Tensor) -> Tensor:
  """a sublayer's output projection: its partial sums if they are kept (_partial_out), in half (they're sent so)"""
  return layer.partial(x, half=_partial_half[0]) if _partial_out[0] else layer(x)

# the experts of a few tokens are the ones of every token (a decode kernel each), more are grouped by expert (prefill)
MOE_DECODE_ROWS = (96)
P2P_CHECK_S = (1.0)  # (a wait this long after the last: the p2p polls' notes read, see _wait_host)
# batched decoding: rows of experts decoded (in bundles) up to this many, grouped (the prefill kernels) past it
MOE_BATCH_ROWS = (512)
def _skinny(lin:Linear, x:Tensor) -> Tensor|None:
  """lin(x) by skinny_matmul (few outputs, few rows) if its weight is stored for it (see Transformer._finish_load), else None"""
  if (wt:=getattr(lin, "weight_t", None)) is None or not isinstance(x.shape[-2], int) or token_sharded(x): return None
  return skinny_matmul(x, wt)

def _decode_rows() -> int: return MOE_BATCH_ROWS if _batch_rows[0] is not None else MOE_DECODE_ROWS

# batched decoding (Transformer.forward_batch): the rows are groups of a sequence each, group g (its rows the tokens at positions pos.. of the
# sequence) in the state slot g: (pos, rows) of every group
# Shared with the generic speculative sequence driver (glm5next._forward_seqs_glm).
_batch_rows = glm_seq_base._batch_rows

@dataclass(frozen=True)
class BatchPrefill:
  """a step of prefills of G prompts into slots (Transformer.forward_batch_prefill): P rows each, their token counts and slots (tensors and the
  variables they're of), the effects of the blocks (the states into the slots) realized with the step"""
  P:int; counts:Tensor; slots:Tensor; count_vars:tuple[UOp, ...]; slot_vars:tuple[UOp, ...]; effects:list = field(default_factory=list) # noqa: E702
  # (the positions the prompts start at: a continuation of the conversation in the slot, from its states at the position. None: all at 0)
  starts:Tensor|None = None
  # (a mixed step (Transformer.forward_batch_mixed): the rows of batched decoding before the prompts' rows, their groups the first of _batch_rows)
  off:int = 0
  @property
  def real(self) -> Tensor:
    """(off+G*P,) the rows that are tokens (the others pad a prompt to P rows)"""
    real = (Tensor.arange(self.P).reshape(1, self.P).to(self.counts.device) < self.counts.reshape(-1, 1)).reshape(-1)
    return Tensor.ones(self.off, dtype=dtypes.bool, device=self.counts.device).cat(real) if self.off else real
_batch_prefill:list[BatchPrefill|None] = [None]

# (MIX_SHARED: a mixed step's attention splits its rows after the projections, see _attention_batch)
MIX_SHARED = (1)
def _batch_attention(x:Tensor, groups:list[tuple[UOp, int]], fn:Callable[[Tensor, list[tuple[UOp, int]]], Tensor]) -> Tensor:
  """
  the attention of batched rows: fn(x, groups). a mixed step's (_batch_prefill's off rows of decoding, then its prompts): fn of the decoding rows
  (no prefill) and fn of the prompts' rows, one after the other (MIX_SHARED: fn splits them)
  """
  if (pf:=_batch_prefill[0]) is None or not pf.off or MIX_SHARED: return fn(x, groups)
  nd = len(groups) - len(pf.count_vars)
  try:
    _batch_prefill[0] = None
    dec = fn(x[:, :pf.off], groups[:nd])
    _batch_prefill[0] = replace(pf, off=0)
    pre = fn(x[:, pf.off:], groups[nd:])
  finally: _batch_prefill[0] = pf
  return dec.cat(pre, dim=1)

def _group_positions(groups:list[tuple[UOp, int]], device:str|tuple[str, ...]|None) -> Tensor:
  """the start positions of the groups of batched decoding (their variables) as an int32 tensor on device"""
  # (all at 0, the prefills of batch_prefill: zeros. a stack of constants as a custom kernel's input fails the kernel graph's spec)
  if all(p.op is Ops.CONST and p.arg == 0 for p, _ in groups): return Tensor.zeros(len(groups), dtype=dtypes.int32, device=device).contiguous()
  return Tensor.stack(*[Tensor(p) for p, _ in groups]).cast(dtypes.int32).contiguous().to(device).contiguous()

def _slot_pool(n:int, like:Tensor, shape:tuple[int|UOp, ...]|None=None, dtype:DType|None=None) -> tuple[Tensor, list[Tensor]]:
  """
  n slots of a state like `like` (of shape, or like's, of dtype or like's) in one buffer of every device, made but not allocated (a jit
  captured before its memory is holds it): the pool (n*shape[0], ...) and the n slots, views of it (buffers of their own: a kernel writes one
  in place)
  """
  shape, devs, axis = tuple(int(x) for x in (shape or like.shape)), cast(str|tuple[str, ...], like.device), like.uop.axis
  local = tuple(x//len(devs) if isinstance(devs, tuple) and i == axis else x for i, x in enumerate(shape))
  dtype = dtype or like.dtype
  size, item = prod(local), dtype.itemsize
  base = UOp.new_buffer(devs, n * size, dtype)
  def tensor(u:UOp, rows:int) -> Tensor:
    u = u.reshape((rows * local[0], *local[1:]))
    return Tensor(u.unshard(axis) if isinstance(devs, tuple) and axis is not None else u)
  def view(i:int) -> Buffer|MultiBuffer:
    if not isinstance(b:=base.buffer, MultiBuffer): return b.view(size * item, i * size * item)
    (mb:=MultiBuffer.__new__(MultiBuffer)).bufs = [x.view(size * item, i * size * item) for x in b.bufs]
    return mb
  return tensor(base, n), [tensor(UOp.from_buffer(view(i), dtype, devs), 1) for i in range(n)]

def _host_buffer(devs:str|tuple[str, ...], nbytes:int, dtype=dtypes.uint8) -> tuple[memoryview, Tensor]:
  """nbytes of host memory that every device reads and writes directly (no copies): its memoryview and a tensor (replicated on devs)"""
  devs = to_tuple(devs)
  host = Buffer(devs[0], nbytes, options=BufferSpec(host=True, cpu_access=True)).ensure_allocated()
  if len(devs) == 1: return host.as_memoryview(allow_zero_copy=True), Tensor(UOp.from_buffer(host, dtype))
  # the other devices' buffers are the host's mappings on them (external: not freed with them)
  aliases = [Buffer(d, nbytes, opaque=BufferStorage(a:=host.get_buf(d)), options=BufferSpec(external_ptr=a)) for d in devs[1:]]
  (mb:=MultiBuffer.__new__(MultiBuffer)).bufs = [host] + aliases
  return host.as_memoryview(allow_zero_copy=True), Tensor(UOp.from_buffer(mb, dtype, devs))

def _open_devices(devices:tuple[str, ...]):
  """the devices opened at the same time (most of opening one is waiting for it): the first alone (what they share is made once)"""
  import threading
  if len(devices) < 2: return
  Device[devices[0]]
  threads = [threading.Thread(target=Device.__getitem__, args=(d,), daemon=True) for d in devices[1:]]
  for t in threads: t.start()
  for t in threads: t.join()

def _read_ahead(entries:dict):
  """
  the files of the entries read into memory in the background: a thread a file, in order (the disk's fastest: 6.8 GB/s on the 990 pro,
  scattered reads 4.9), the loader finds them there
  """
  import os, threading
  paths = dedup(b.device[5:] for d, _, _ in entries.values() if isinstance(b:=d.uop.base.buffer, Buffer) and b.device.startswith("DISK:"))
  def run(path:str):
    fd, mv = os.open(path, os.O_RDONLY), memoryview(bytearray(64 << 20))
    try:
      size, off = os.fstat(fd).st_size, 0
      while off < size and (got:=os.preadv(fd, [mv], off)) > 0: off += got
    finally: os.close(fd)
  for path in paths: threading.Thread(target=run, args=(path,), daemon=True, name="read_ahead").start()

def _realize_later(ts:list[Tensor], pending:list[Callable[[], None]], scratch:dict|None=None):
  """the tensors are buffers now (allocated, their devices and split kept), their values are computed by the function added to pending
  scratch: the jits' shared memory of intermediates (LOAD_DEFER: the values computed after the caches are made, no room for their own)"""
  pairs, ts = [], list({id(t): t for t in ts}.values())  # (a tensor of two modules once)
  # (an empty tensor is just its memory: allocated now)
  if now:=[t for t in ts if t.numel() == 0 or t.uop.base.op is Ops.ALLOC]: Tensor.realize(*now)
  # one allocation of the devices of a group (one each is slow: a device's page tables are flushed after every one), the tensors' views of it
  groups:dict[str|tuple[str, ...], list[Tensor]] = {}
  for t in ts:
    if t.numel() > 0 and not any(t is x for x in now): groups.setdefault(cast(str|tuple[str, ...], t.device), []).append(t)
  for devs, grp in groups.items():
    def shard(t:Tensor) -> tuple[int, ...]:
      return cast(tuple[int, ...], t.uop.shard_shape if isinstance(devs, tuple) and t.uop.axis is not None else t.shape)
    sizes = [round_up(prod(shard(t)) * t.dtype.itemsize, 256) for t in grp]
    big = [Buffer(d, sum(sizes)).ensure_allocated() for d in to_tuple(devs)]
    for t, off in zip(grp, itertools.accumulate([0] + sizes[:-1])):
      views = [b.view(prod(shape:=shard(t)) * t.dtype.itemsize, off).ensure_allocated() for b in big]
      if isinstance(devs, tuple): (mb:=MultiBuffer.__new__(MultiBuffer)).bufs = views
      u = UOp.from_buffer(mb if isinstance(devs, tuple) else views[0], t.dtype, devs).reshape(shape)
      lazy, u = t.uop, u.unshard(t.uop.axis) if isinstance(devs, tuple) and t.uop.axis is not None else u
      t.replace(Tensor(u))
      pairs.append((u, lazy))
  # (a jit: compiled with the model's jits, see _capture_parallel)
  def compute(): Tensor.realize(*[Tensor(u).assign(Tensor(lazy)) for u, lazy in pairs])
  if pairs: pending.append(TinyJit(compute, capture_first=True, scratch=scratch))

def _host_ints(devs:str|tuple[str, ...], n:int) -> tuple[memoryview, Tensor]:
  """n int32 in host memory that the host polls: a (1, n) tensor every device reads and writes directly"""
  mv, t = _host_buffer(devs, 4*n, dtypes.int32)
  return mv.cast('i'), t[None]

def _hash_uniform(seed:Tensor, keys:Tensor, like:Tensor) -> Tensor:
  """
  uniform noise in (0, 1) like `like` (B, V): a hash of seed (a step's, uint32 (1,)), the key of every row (B,) and the column. the same for the
  same key: (SPEC_SHARED) a draft of speculative decoding and the verification of its position use the same noise, the draft is the token
  sampled there whenever the draft distribution is close to the model's (not only when the most likely is). the sampled token is the model's
  sample all the same: its noise is fresh at every position of every step
  """
  col = Tensor.arange(V:=int(like.shape[-1])).reshape(1, V).cast(dtypes.uint32).to(like.device)
  h = seed.reshape(1, 1).to(like.device) * 0x9E3779B1 + keys.reshape(-1, 1).cast(dtypes.uint32).to(like.device) * 0x85EBCA77 + col * 0xC2B2AE3D
  # (murmur3's finalizer: every bit of the sum mixed into every bit)
  for sh, mul in ((16, 0x85EBCA6B), (13, 0xC2B2AE35)): h = (h ^ (h >> sh)) * mul
  h = h ^ (h >> 16)
  # (23 bits, as glm5next's: (k+0.5)/2**24 rounds to 1.0 in float32 at the top (one token in 2**24, ~1% of the steps over the vocabulary): its
  # noise an infinity, a token of the whole vocabulary sampled whatever its probability (ts5's stray tokens at top_p 1, Nov 16))
  return ((h >> 9).cast(dtypes.float32) + 0.5) * (1.0 / (1 << 23))

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

def seq_sharded(device:str|tuple[str, ...]|None) -> bool:
  """MLA_SEQ_SHARD: the MLA latent caches split on their positions over the devices (memory for a context 1/devices of a copy on every one)"""
  return isinstance(device, tuple) and len(device) > 1 and amd_custom_kernels_supported(device) and bool((1))

class FFNBlock:
  def __init__(self, config:TransformerConfig):
    self.config = config

    # --- RMSNorms --------------------------------------------------------
    self.attn_norm   = nn.RMSNorm(config.dim, config.norm_eps)
    self.ffn_norm    = nn.RMSNorm(config.dim, config.norm_eps)

    # --- feed-forward (MoE or dense) -------------------------------------
    if config.num_experts > 0:
      self.ffn_gate_inp = Linear(config.dim, config.num_experts, bias=config.expert_proj_bias)  # router
      if config.expert_bias: self.exp_probs_b = {"bias": Tensor.zeros(config.num_experts)}
      # sliced experts: num_experts*slices experts of hidden_dim/slices
      S = config.expert_slices or 1
      U = config.expert_units or config.num_experts*S
      self.ffn_gate_exps = ExpertWeights(U, config.dim, config.hidden_dim//S, bias=config.expert_proj_bias)
      self.ffn_up_exps = ExpertWeights(U, config.dim, config.hidden_dim//S, bias=config.expert_proj_bias)
      self.ffn_down_exps = ExpertWeights(U, config.hidden_dim//S, config.dim, bias=config.expert_proj_bias)
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

  def _shexp_gate_up(self, x:Tensor) -> tuple[Tensor, Tensor]:
    return self.ffn_gate_shexp(x), self.ffn_up_shexp(x)
  def _feed_forward(self, x:Tensor) -> Tensor:
    # a symbolic token count runs the MoE padded to the max size: static shapes, the garbage tokens are sliced off
    if hasattr(self, 'ffn_gate_exps') and not isinstance(x.shape[1], int) and amd_custom_kernels_supported(x.device):
      return self._feed_forward(x.pad_to(x.max_shape))[:, :x.shape[1]]
    if hasattr(self, 'ffn_gate_exps'):
      h = x.unsqueeze(2)  # (B, T, 1, D) - add expert dim for broadcasting
      if (pre:=_sp_route[0]) is not None: return self._experts(x, h, *pre)
      router = self.ffn_gate_inp
      # sequence parallel: every device routes its rows, the selections are gathered
      xr = rows if (rows:=_sp_rows[0]) is not None else x
      bias = self.exp_probs_b["bias"] if hasattr(self, 'exp_probs_b') else None
      gating, normalize_topk = self.config.expert_gating_func, self.config.norm_topk_prob
      exps = self.ffn_gate_exps
      exps.set_quantized()
      parts:Tensor|None = None
      if rows is None and (lg:=_skinny(router, xr)) is not None: logits = lg
      elif router.ggml_type is None and router.weight.dtype == dtypes.half and router.bias is None and isinstance(xr.shape[1], int) and \
         xr.shape[1] > 32:
        # many tokens: half copies of the input and the output, a float accumulator: the tensor cores
        logits = xr.half().contiguous().matmul(router.weight.T, dtype=dtypes.float).contiguous()
      elif isinstance(T:=xr.shape[1], int) and 16 < T < 128 and T % 16 and (1):
        # (the rows of batched decoding (3 a sequence): padded to whole tiles of 16, the matmul's kernel of 60 rows is several times slower)
        logits = router(xr.pad((None, (0, round_up(T, 16) - T), None)))[:, :T]
      else: logits = router(xr)
      if gating == ExpertGating.SIGMOID and normalize_topk and bias is not None and isinstance(xr.shape[1], int) and \
         amd_custom_kernels_supported(x.device) and self.config.num_experts <= 1024 and (1):
        k, scale, exps = self.config.num_experts_per_tok, self.config.routed_scaling_factor, self.ffn_gate_exps
        exps.set_quantized()
        if rows is None and xr.shape[1] <= _decode_rows() and exps.slices and exps.ggml_type is not None and (1):
          # decoding: the device's slices of the selections in the router's kernel
          return self._experts(x, h, *moe_router_local(logits, bias, k, scale, exps.slice_table, parts=parts), local=True)
        sel, probs = moe_router(logits, bias, k, scale)
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

  def _experts(self, x:Tensor, h:Tensor, sel:Tensor, probs:Tensor, local:bool=False) -> Tensor:
    """the selected experts (and the shared expert) of every token, summed over the devices. local: sel and probs are the slices already"""
    exps = (self.ffn_gate_exps, self.ffn_down_exps) + (() if self.ffn_gate_exps.fused else (self.ffn_up_exps,))
    for e in exps: e.set_quantized()
    # sliced experts: every device runs its slices of the selected experts
    if self.ffn_gate_exps.slices and not local: sel, probs = self.ffn_gate_exps.select_slices(sel, probs)
    # (the padding rows of a prefill chunk select nothing: a short prompt's experts are the ones of its tokens)
    if (pf:=_batch_prefill[0]) is not None: sel = pf.real.reshape(-1, 1).to(sel.device).where(sel, -1)
    if (T:=_ffn_rows[0]) is not None and sel.ndim >= 3 and isinstance(R:=sel.shape[-2], int) and (1):
      sel = (Tensor.arange(R).reshape(R, 1) < (Tensor(T) if isinstance(T, UOp) else T)).to(sel.device).where(sel, -1)
    # many tokens: grouped by expert (prefill), a few (the drafts of speculative decoding) as decoding
    grouped = (not isinstance(x.shape[1], int) or x.shape[1] > _decode_rows()) and all(e.ggml_type is not None for e in exps)
    shexp:Tensor|None = None
    if grouped and hasattr(self, 'ffn_gate_shexp') and not hasattr(self, 'ffn_gate_inp_shexp') and self.ffn_gate_exps.expert_parallel and \
       isinstance(x.shape[1], int) and self.ffn_down_shexp.shard_axis == 1:
      # the shared expert's partial sums are added by the kernel summing the experts' rows of every token
      shexp_act = gated_activation(self.ffn_gate_shexp(x), self.ffn_up_shexp(x), limit=self.config.swiglu_limit).contiguous()
      shexp = self.ffn_down_shexp.partial(shexp_act)
    if grouped:
      # the partial sums kept (sequence parallel) are sent in half: stored so if nothing is added after
      out = self._grouped_experts(x, sel, probs, add=shexp, half=_partial_out[0] and _partial_half[0] and
                                  (shexp is not None or not hasattr(self, 'ffn_gate_shexp')))
    else:
      if self.ffn_gate_exps.fused:  # one kernel for the gate and up rows, interleaved by 16
        gu = self.ffn_gate_exps(sel, h)
        gu = gu.reshape(*gu.shape[:-1], -1, 2, 16)
        gate, up = gu[..., 0, :].flatten(-2), gu[..., 1, :].flatten(-2)
      else: gate, up = self.ffn_gate_exps(sel, h), self.ffn_up_exps(sel, h)
      act = gated_activation(gate, up, alpha=self.config.swiglu_alpha, limit=self.config.swiglu_limit, up_bias=self.config.swiglu_up_bias)
      # (batched decoding: a row a selection of a slice, in half as the grouped experts' (their weighted sum in float))
      # (the activation computed by the kernel that quantizes it: custom kernel inputs are inlined)
      x_down = self.ffn_down_exps(sel, act if INLINE_ACT else act.contiguous(),
                                  dtypes.half if _batch_rows[0] is not None and self.ffn_down_exps.slices else dtypes.float32)
      ws = weighted_sum(x_down, probs) if _batch_rows[0] is not None and not _seq_alone() else None
      out = ws if ws is not None else (x_down * probs.unsqueeze(-1)).sum(axis=-2)  # (B, T, D), expert parallel: (devices, B, T, D)
    partial = out.ndim == x.ndim + 1
    keep = partial and _partial_out[0]  # the partial sums are returned (sequence parallel)
    if shexp is not None:
      if partial and not keep: out = partial_sum(out)
    elif hasattr(self, 'ffn_gate_shexp'):
      shexp_act = gated_activation(*self._shexp_gate_up(x), limit=self.config.swiglu_limit)
      if not INLINE_ACT: shexp_act = shexp_act.contiguous()
      # a row parallel shared expert adds its partial sums to the ones of the experts: one allreduce for both
      shexp = self.ffn_down_shexp.partial(shexp_act) if partial else self.ffn_down_shexp(shexp_act)
      if hasattr(self, 'ffn_gate_inp_shexp'): shexp = shexp * (x * self.ffn_gate_inp_shexp["weight"]).sum(axis=-1, keepdim=True).sigmoid()
      if shexp.ndim == out.ndim: out = out + shexp
      elif keep: out = out + shexp / int(out.shape[0])  # a whole shared expert output: a share of it in every partial
      if partial and not keep: out = partial_sum(out)
      if shexp.ndim != out.ndim + partial and not keep: out = out + shexp
    elif partial and not keep: out = partial_sum(out)
    return out

  def _grouped_experts(self, x:Tensor, sel:Tensor, probs:Tensor, add:Tensor|None=None, half:bool=False) -> Tensor:
    """prefill: the selections sorted by expert, every expert reads its weights once for all its tokens. (devices, B, T, D) partials if split"""
    B, T, D = x.shape
    # a symbolic token count runs padded to the max size, the garbage tokens are sliced off
    if not isinstance(T, int): x, sel, probs = x.pad_to(x.max_shape), sel.pad_to(sel.max_shape), probs.pad_to(probs.max_shape)
    gate_exps, down_exps = self.ffn_gate_exps, self.ffn_down_exps
    # (batched decoding: a few rows an expert, small tiles)
    tile = (16) if _batch_rows[0] is not None and _batch_prefill[0] is None else MOE_TILE
    _, entries, tiles, pos = moe_route(sel, gate_exps.first, gate_exps.local_experts, tile)
    k, sel_shape = int(sel.shape[-1]), sel.shape[-3:]
    # gate and up read the token of each sorted row, down writes the row of each selection
    if gate_exps.fused:  # the activation is computed in the kernel of the interleaved gate and up rows
      act = gate_exps.grouped(x.reshape(-1, D), entries, tiles, slots=k,
                              act=(self.config.swiglu_alpha, self.config.swiglu_limit or 0.0, self.config.swiglu_up_bias))
    else:
      gate, up = (e.grouped(x.reshape(-1, D), entries, tiles, slots=k) for e in (gate_exps, self.ffn_up_exps))
      act = gated_activation(gate, up, alpha=self.config.swiglu_alpha, limit=self.config.swiglu_limit, up_bias=self.config.swiglu_up_bias)
    # down has a row per selection (in half if sliced: a token sums many; the selections of other devices' experts aren't written: fewer rows
    # than the sorted ones, padded to tiles), every token sums its rows weighed by their probabilities
    tokens, k = int(prod(sel_shape[:-1])), int(sel_shape[-1])
    # (MOE_SORTED_DOWN: a row per sorted row, the selections read through pos: the rows written are together, not spread over every selection's)
    sorted_down = bool((1))
    down = down_exps.grouped(act.contiguous(), entries, tiles, selections=tokens*k,
                             out_dtype=
                             dtypes.bfloat16, sorted_out=sorted_down)
    # sequence parallel: summed over the devices and split on the tokens around the ring as they're combined
    if half and gate_exps.expert_parallel and _sp_rows[0] is not None and isinstance(T, int) and tokens % len(cast(tuple, x.device)) == 0 and \
       tokens == T and (1):
      return moe_combine_scatter(down, pos, probs, tokens, k, add, by_selection=not sorted_down)
    y = moe_combine(down, pos, probs, tokens, k, add, out_dtype=dtypes.half if half and gate_exps.expert_parallel else dtypes.float32,
                    by_selection=not sorted_down)
    y = y.reshape(-1, *sel_shape[:-1], D)[:, :, :x.shape[1] if isinstance(T, int) else T]
    return y if gate_exps.expert_parallel else y[0]

  # given the token-prefix match, return how much cached state this block can still reuse
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return prefix_len
  def _init_slots(self, n:int, ctx:int): pass  # the states of batched decoding (a block with states of its own)
  def _init_state(self, x:Tensor): raise NotImplementedError
  def _attention(self, x:Tensor, start_pos:int|UOp, T:int|UOp|None=None) -> Tensor: raise NotImplementedError

  def __call__(self, x: Tensor, start_pos: int|UOp, T:int|UOp|None=None):
    self._init_state(x)
    # we pass in the weights implicitly so we unpack the GGUF on the fly
    # (DSA: the indexer's mask of the last full layer, the shared layers' after it)
    mask_in = _idx_mask[0] if self.config.indexer_shared and getattr(self, "indexer", None) is None else None
    @function(precompile=True, allow_implicit=True)
    def _run(x:Tensor, start_pos:int|UOp, T:int|UOp|None, mask_in:Tensor|None):
      _idx_mask[0], _idx_mask[1] = mask_in, None
      h =     x + self._attention(self.attn_norm(x), start_pos, *(() if T is None else (T,)))
      out = (h + self._feed_forward(self.ffn_norm(h))).contiguous()
      return (out, m.contiguous()) if (m:=_idx_mask[1]) is not None else out
    ret = _run(x, start_pos, T, mask_in)
    if isinstance(ret, tuple): ret, _idx_mask[0] = ret
    return ret

  def _sp_sub(self, sub:str, hn:Tensor, start_pos:int|UOp, T:int|UOp) -> Tensor:
    """sequence parallel sublayer: hn (1, R, D) split on its rows, gathered as q8 groups, the sublayer's partial sums reduce scattered to the rows"""
    if sub == "attn" and (o:=self._sp_attn_tok(hn, start_pos, T)) is not None: return sp_reduce_scatter(o.contiguous())
    if sub == "attn" and (1) and (pre:=self._sp_rowproj(hn, start_pos, T)) is not None: x, _sp_pre[0] = pre.pop("x"), pre
    else: x = all_gather_q8(hn.contiguous(), 1, ring=bool((1)))
    _partial_out[0], _sp_rows[0], _partial_half[0] = True, hn, False
    if sub != "attn": _ffn_rows[0] = T
    if isinstance(R:=x.shape[1], int): GEMM_ROWS[0] = (R, T)
    try: out = self._attention(x, start_pos, T) if sub == "attn" else self._feed_forward(x)
    finally: _partial_out[0], _sp_rows[0], _ffn_rows[0], _partial_half[0], GEMM_ROWS[0], _sp_pre[0] = False, None, None, True, None, None
    if out.ndim == hn.ndim and token_sharded(out): return out
    # (the partial sums in float, sent as half rows with a power of two of each: the range of GLM-5's activations past half's)
    return sp_reduce_scatter(out.contiguous()) if out.ndim == hn.ndim + 1 else shard_rows(out, 1)

  def _sp_attn_tok(self, hn:Tensor, start_pos:int|UOp, T:int|UOp) -> Tensor|None:
    """the attention of sequence parallel blocks with the heads of its rows on every device (MLATransformerBlock), None if it hasn't one"""
    return None

  def _sp_rowproj(self, hn:Tensor, start_pos:int|UOp, T:int|UOp) -> dict|None:
    """the row projections of a sequence parallel attention made on the rows' devices (MLATransformerBlock), None if it hasn't them"""
    return None

  def sp_block(self, xs:Tensor, start_pos:int|UOp, T:int|UOp) -> Tensor:
    """sequence parallel (prefill, no hyper-connections): xs (1, R, D) the residual stream split on its rows over the devices, T of them real"""
    self._init_state(xs)
    mask_in = _idx_mask[0] if self.config.indexer_shared and getattr(self, "indexer", None) is None else None
    @function(precompile=True, allow_implicit=True)
    def _run(xs:Tensor, start_pos:int|UOp, T:int|UOp, mask_in:Tensor|None):
      _idx_mask[0], _idx_mask[1] = mask_in, None
      h = xs + self._sp_sub("attn", self.attn_norm(xs), start_pos, T)
      out = (h + self._sp_sub("ffn", self.ffn_norm(h), start_pos, T)).contiguous()
      return (out, m.contiguous()) if (m:=_idx_mask[1]) is not None else out
    ret = _run(xs, start_pos, T, mask_in)
    outs = ret if isinstance(ret, tuple) else (ret,)
    if len(outs) > 1: _idx_mask[0] = outs[1]
    return outs[0]

class Indexer:
  shard = 0  # (SEQ_IDX_SHARD: the devices the pooled keys are split over)
  def pk_rows(self, t:Tensor) -> Tensor:
    """the pooled keys (or a pool of the slots') as rows: the local ones of every device if split (SEQ_IDX_SHARD, see slot_rows_view)"""
    return MLATransformerBlock.slot_rows_view(t) if self.shard else t
  def pk_slot(self) -> int:
    """the rows of a slot's pooled keys (of a device if split)"""
    return int(self.slot_pk[0].shape[0]) // (self.shard or 1)
  pool_pk:Tensor  # batched decoding: the caches of every slot, and of all of them (MLATransformerBlock._init_slots)
  pool_ring:Tensor
  slot_pk:list[Tensor]
  slot_ring:list[Tensor]
  def __init__(self, config:TransformerConfig):
    self.idx, self.max_context, self.rewind = (idx:=unwrap(config.indexer)), config.max_context, config.rewind
    self.cache_rows = config.cache_rows
    self.attn_q_b, self.attn_k = Linear(config.q_lora_rank, idx.n_heads * idx.head_dim, bias=False), Linear(config.dim, idx.head_dim, bias=False)
    self.proj, self.k_norm = Linear(config.dim, idx.n_heads, bias=False), nn.LayerNorm(idx.head_dim, eps=idx.norm_eps)
    if idx.kpool > 1:
      self.compressor_gate, self.compressor_ape = Linear(config.dim, idx.head_dim, bias=False), {"weight": Tensor.zeros(idx.kpool, idx.head_dim)}

  def _init_state(self, x:Tensor):
    if hasattr(self, "cache_pk") or hasattr(self, "cache_dk"): return
    if self.idx.kpool == 1:
      # (DSA: a key per token, rope on its first rope_dim dims, every query picks its top_k keys. the sparse path (RDNA4): the lists of the keys
      # picked by the indexer kernels, the keys of one sequence (no batch axis))
      self.sparse = x.shape[0] == 1 and amd_custom_kernels_supported(x.device) and _wmma_rdna4(x.device) and self.idx.head_dim % 16 == 0 and \
        self.max_context % 64 == 0 and bool((1))
      # (GLM_SEQ: the regions of the sequences sharing the keys, and the padding sequences')
      N = max(self.max_context, self.cache_rows) + (REGION_ALIGN if self.cache_rows else 0)
      shape = (N, self.idx.head_dim) if self.sparse else (x.shape[0], N, self.idx.head_dim)
      # (SEQ_IDX_DSA_REGIONS=0: GLM_SEQ's keys replicated on every device (~14x their memory))
      if self.sparse and seq_sharded(x.device) and (1) and (not self.cache_rows or (1)):
        # (a sequence sharded latent cache (MLA_SEQ_SHARD): the keys split on their positions as it is, a device scores its own, see
        # pool_index_shard. a sequence's region from base: a device's rows from base//devices (base a multiple of SEQ_BLK*devices))
        self.shard = len(devs:=cast(tuple[str, ...], x.device))
        self.cache_dk = Tensor.empty(self.shard*seq_rows(N, self.shard), self.idx.head_dim, dtype=dtypes.half, device=devs, axis=0)
        self.dk_width = seq_rows(self.max_context, self.shard)  # (a device's keys of a sequence: the scores' columns)
        return
      self.cache_dk = Tensor.zeros(*shape, dtype=dtypes.half, device=x.device).contiguous()
      return
    idx, max_pools = self.idx, round_up(self.max_context, self.idx.kpool) // self.idx.kpool
    self.max_pools = max_pools
    # the sparse path (RDNA4): no mask, the pools a query picks are lists (decode) or per 16 queries (prefill, see pool_union)
    self.sparse = x.shape[0] == 1 and amd_custom_kernels_supported(x.device) and _wmma_rdna4(x.device) and 16 % idx.kpool == 0 and \
      idx.head_dim % 16 == 0 and max_pools % 64 == 0 and bool((1))
    # the pooled keys, a call pools its tokens' pools only. the sparse path keeps them in half and the tokens of the last incomplete pool in a
    # ring (the last tokens: the next call can start before the end, see pool_keys), the other one every token's key and gate
    # (SEQ_FP8 with a sequence sharded cache: fp8 pooled keys, a row of 32 words of e4m3 values and the scale. SEQ_FP8_IDX=0: the latent
    # cache's rows fp8, the indexer's not)
    fp8 = self.sparse and bool(getenv("SEQ_FP8_IDX", getenv("SEQ_FP8", 0))) and seq_sharded(x.device) and idx.head_dim == 128
    row, dt = ((IDX_FP8_ROW,), dtypes.uint32) if fp8 else ((idx.head_dim,), dtypes.half if self.sparse else dtypes.float32)
    region_pools = (max(self.max_context, self.cache_rows) + REGION_ALIGN) // idx.kpool if self.cache_rows else max_pools
    self.cache_pk = Tensor.zeros(*(() if self.sparse else (x.shape[0],)), region_pools, *row, dtype=dt, device=x.device).contiguous()
    ring = round_up(max(2*idx.kpool, self.rewind + idx.kpool - 1), idx.kpool)
    # (SEQ_IDX_SHARD with a sequence sharded cache: the pooled keys split on their pools as the cache on its positions, a device stores and
    # scores its own, see pool_index_shard)
    self.shard = len(cast(tuple, x.device)) if self.sparse and bool((0)) and seq_sharded(x.device) else 0
    if self.shard:
      devs, local = cast(tuple[str, ...], x.device), seq_rows(self.max_context, self.shard) // idx.kpool
      self.cache_pk = Tensor.empty(self.shard*local, *row, dtype=dt, device=devs, axis=0)
    if self.sparse: self.ring = Tensor.zeros(ring, 2, idx.head_dim, dtype=dtypes.half, device=x.device).contiguous()
    else: self.cache_kg = Tensor.zeros(x.shape[0], max_pools*idx.kpool, 2, idx.head_dim, dtype=dtypes.half, device=x.device).contiguous()

  def picks(self, x:Tensor, q:Tensor, start_pos:UOp, T:int|UOp|None=None) -> Tensor:
    """the sparse path: for every query (Tq, n+1) the list of the n keys it sees at most (its top pools and the tail of its own), the count last"""
    Tq, idx = int(x.shape[1]), self.idx
    S = start_pos + (T := Tq if T is None else T)
    kg, iq = self.k_norm(self.attn_k(x)).stack(self.compressor_gate(x), dim=2), self.attn_q_b(q).reshape(Tq, idx.n_heads, idx.head_dim)
    b = _seq_base[0] // idx.kpool
    assert not self.shard or (isinstance(b, int) and b == 0), "SEQ_IDX_SHARD: no regions"
    pk = pool_keys(self.cache_pk, self.ring, kg[0].half(), self.compressor_ape["weight"], start_pos, T, pbase=b)
    weights, pools, k = (self.proj(x) * (idx.n_heads * idx.head_dim)**-0.5)[0], (S + idx.kpool - 1) // idx.kpool, idx.top_k // idx.kpool
    # one token: the heads are the rows of the scores' matmul, a kernel picks its pools
    if Tq == 1 and idx.n_heads % 16 == 0:
      return indexer_decode_index(iq[0], pk, weights[0], start_pos, pools, k, idx.kpool, pbase=b, width=self.max_pools).reshape(1, -1)
    # a few (the drafts of speculative decoding): the same, a row a query
    if Tq <= 8 and idx.n_heads % 16 == 0 and (1):
      return indexer_decode_index_rows(iq, pk, weights, start_pos, pools, k, idx.kpool, pbase=b, width=self.max_pools)
    # the scores on the tensor cores (rows of 16 queries), a kernel per query picks its pools
    rows = round_up(Tq, 16)
    iq, weights = iq.pad(((0, rows-Tq), None, None)), weights.pad(((0, rows-Tq), None))
    # blocks of rows one after the other: the scores of a block (rows x the max pools) are its memory, not the ones of the chunk
    block, picks = round_up(min(rows, (1056)), 16), None
    for r0 in range(0, rows, block):
      qb = iq[r0:r0+block].contiguous()
      if picks is not None: qb = Tensor(qb.uop.after(*picks.uop.src[1:]))  # after the last block's lists: its scores are done
      if picks is None: picks = Tensor.empty(rows, k*idx.kpool + idx.kpool + 1, dtype=dtypes.int32, device=x.device)
      sc = indexer_scores(qb, pk, weights[r0:r0+block], pools, idx.kpool, pbase=b, width=self.max_pools)
      picks = pool_index_shard(sc, start_pos, pools, k, idx.kpool, row0=r0, out=picks) if self.shard else \
              pool_index_rows(sc, start_pos, pools, k, idx.kpool, row0=r0, out=picks)
    return unwrap(picks)[:Tq]

  def dsa_picks(self, x:Tensor, q:Tensor, start_pos:UOp, freqs:Tensor, T:int|UOp|None=None, k_raw:Tensor|None=None,
                w_raw:Tensor|None=None) -> Tensor:
    """
    DSA, the sparse path: x has Tq rows, the first T the tokens at start_pos. the lists (Tq, top_k+2) of the keys every query sees, count last.
    k_raw: attn_k(x) made already
    """
    Tq, idx = int(x.shape[1]), self.idx
    S = start_pos + (T := Tq if T is None else T)
    R = idx.rope_dim
    k = self.k_norm((k_raw if k_raw is not None else self.attn_k(x)).float())[0]
    k = apply_rope(k[:, :R].reshape(1, 1, Tq, R), freqs, idx.rope_interleave).reshape(Tq, R).cat(k[:, R:], dim=-1)
    iq_raw = self.attn_q_b(q).float().reshape(Tq, idx.n_heads, idx.head_dim)
    def roped(raw:Tensor, fr:Tensor) -> Tensor:
      x = raw.transpose(0, 1).unsqueeze(0)
      return apply_rope(x[..., :R], fr, idx.rope_interleave).cat(x[..., R:], dim=-1)[0].transpose(0, 1)  # (rows, H, D)
    iq = roped(iq_raw, freqs)
    b = _seq_base[0]
    # (the region's keys: split, a device's rows from b//devices (its scores' columns a sequence's), else rows from b)
    if self.shard: pk, kb, width = seq_store(self.cache_dk, k.half(), start_pos + b, T), b // self.shard, self.dk_width
    else: pk, kb, width = Tensor(self.cache_dk.uop.after(self.cache_dk[start_pos+b:S+b].uop.store(k[:T].half().uop))), b, self.max_context
    w = w_raw if w_raw is not None else p if (p:=_skinny(self.proj, x)) is not None else self.proj(x)
    weights = (w.float() * (idx.n_heads * idx.head_dim)**-0.5)[0]
    if Tq == 1 and idx.n_heads % 16 == 0:
      return indexer_decode_index(iq[0], pk, weights[0], start_pos, S, idx.top_k, 1, pbase=kb, width=width).reshape(1, -1)
    if Tq <= 8 and idx.n_heads % 16 == 0:
      return indexer_decode_index_rows(iq, pk, weights, start_pos, S, idx.top_k, 1, pbase=kb, width=width)
    # the scores on the tensor cores in blocks of rows (their memory: rows x the max context), a kernel a query picks its keys
    rows = round_up(Tq, 16)
    iq, weights = iq.pad(((0, rows-Tq), None, None)), weights.pad(((0, rows-Tq), None))
    block = round_up(min(rows, (1056)), 16)
    picks:Tensor|None = None
    # (IDX_QBLK, default on, the same values: a block's queries roped from the projection's rows, not a copy of the block from all of them roped
    # into a buffer (a float copy of the chunk's queries in every layer); the padding rows 0 either way)
    qblk = rows // block > 1
    if qblk: raw_p, fr_p = iq_raw.pad(((0, rows-Tq), None, None)), freqs.reshape(Tq, -1).pad(((0, rows-Tq), None))
    for r0 in range(0, rows, block):
      qb = (roped(raw_p[r0:r0+block], fr_p[r0:r0+block]) if qblk else iq[r0:r0+block]).contiguous()
      if picks is not None: qb = Tensor(qb.uop.after(*picks.uop.src[1:]))  # (after the last block's lists: its scores are done)
      if picks is None: picks = Tensor.empty(rows, idx.top_k + 2, dtype=dtypes.int32, device=x.device)
      # (IDX_PAD (with LISTS_PAD): a block's scores and candidates of the chunk's tokens only, its padding rows' lists are emptied after)
      rv = None
      sc = indexer_scores(qb, pk, weights[r0:r0+block], S, 1, pbase=kb, width=width, rows=rv)
      picks = pool_index_shard(sc, start_pos, S, idx.top_k, 1, row0=r0, out=picks, valid=rv) if self.shard else \
              pool_index_rows(sc, start_pos, S, idx.top_k, 1, row0=r0, out=picks)
    return unwrap(picks)[:Tq]

  def dsa_mask(self, x:Tensor, q:Tensor, start_pos:int|UOp, freqs:Tensor, T:int|UOp|None=None) -> Tensor:
    """DSA (kpool 1): x has Tq rows, the first T the tokens at start_pos (freqs: the rope of the Tq rows). the mask (B, 1, Tq, S)"""
    (B, Tq, _), idx = x.shape, self.idx
    S = start_pos + (T := Tq if T is None else T)
    R = idx.rope_dim
    k = self.k_norm(self.attn_k(x).float())
    k = apply_rope(k[..., :R].unsqueeze(1), freqs, idx.rope_interleave).squeeze(1).cat(k[..., R:], dim=-1)
    iq = self.attn_q_b(q).float().reshape(B, Tq, idx.n_heads, idx.head_dim).transpose(1, 2)
    iq = apply_rope(iq[..., :R], freqs, idx.rope_interleave).cat(iq[..., R:], dim=-1)  # (B, H, Tq, D)
    keys = Tensor(self.cache_dk.uop.after(self.cache_dk[:, start_pos:S].uop.store(k[:, :T].half().uop)))[:, :S].float()  # (B, S, D)
    weights = self.proj(x).float() * (idx.n_heads * idx.head_dim)**-0.5  # (B, Tq, H)
    scores = ((iq @ keys.unsqueeze(1).transpose(-1, -2)).relu() * weights.transpose(1, 2).unsqueeze(-1)).sum(1)  # (B, Tq, S)
    q_pos, k_pos = (Tensor(start_pos) + Tensor.arange(Tq)).reshape(Tq, 1), Tensor.arange(self.max_context)[:S]
    causal = k_pos <= q_pos
    picked = topk_mask(causal.where(scores, -math.inf), idx.top_k)
    return (picked & causal).where(0.0, -math.inf).reshape(B, 1, Tq, S).cast(x.dtype)

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
  pool_k:Tensor
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
    if (groups:=_batch_rows[0]) is not None: return _batch_attention(x, groups, self._attention_batch)
    if T is not None: return self._attention_rows(x, start_pos, T)  # rows mode: x is padded, the first T rows are the tokens
    N, D = (self.config.max_context, self.config.kv_lora_rank + self.config.rope_dim) if self.seq_sharded else self.cache_k.shape[2:]
    flash = amd_custom_kernels_supported(x.device) and N % 64 == 0 and D % 64 == 0
    # a symbolic chunk computes padded to its max size (static shapes), it stores and attends with its real tokens
    if flash and not isinstance(T:=x.shape[1], int): return self._attention_rows(x.pad_to(x.max_shape), start_pos, T)[:, :T]
    return self._attention_rows(x, start_pos, x.shape[1])

  def _sp_rowproj(self, hn:Tensor, start_pos:int|UOp, T:int|UOp) -> dict|None:
    """
    SP_ROWPROJ (sequence sharded cache, sequence parallel prefill): the attention's row projections (attn_q_a, attn_kv_a_mqa, the indexer's
    attn_k and proj) of a device's OWN rows, not of all the rows on every device (14x the work), their outputs gathered, not hn (6144 wide).
    The same values: the projections' input is what every device unpacked of the row before (hn packed and unpacked as all_gather_q8 does),
    q_b's int8 groups of the normed q_a those the row's device made (registered for the gathered rows), the rest gathered in float.
    returns the gathered tensors and x: a placeholder of the gathered hn's shape (never read)
    """
    cfg, ind = self.config, self.indexer
    devs = hn.device
    if not (self.seq_sharded and cfg.q_lora_rank > 0 and isinstance(devs, tuple) and isinstance(R:=hn.shape[1], int) and hn.ndim == 3 and
            hn.uop.axis == 1 and (ind is None or ind.idx.kpool == 1)): return None
    D = int(hn.shape[-1])
    from tinygrad.llm.kernels.amd_glm_dsa import _q8_pack_kernel, _q8_unpack_kernel
    packed = Tensor.custom_kernel(Tensor.empty(R, D//32, 9, dtype=dtypes.uint32, device=devs, axis=0),
                                  hn.contiguous().reshape(R, D), fxn=_q8_pack_kernel)[0]
    xl = Tensor.custom_kernel(Tensor.empty(R, D, dtype=dtypes.half, device=devs, axis=0), packed, fxn=_q8_unpack_kernel)[0].reshape(1, R, D)
    qn = self.attn_q_a_norm(self.attn_q_a(xl)).contiguous()
    # (the gathered rows' normed q_a itself (q_b quantizes it as before, the indexer's attn_q_b reads its values): a constant placeholder with
    # only its q8 registered gave the indexer 3.0s (wrong lists))
    if (1):
      # (SP_QN_HALF: its readers (the q8 gemms of attn_q_b and of the indexer's attn_q_b) read it as _half_rows does: a row's halves of qn/s,
      # s a power of two (1 unless the row is past HALF_ROWS_MAX), the outputs times s. sent so, half the bytes of the float rows: the gathered
      # rows h*s give the same halves and scale (s exact, h*s/s = h; s=1: |h| <= HALF_ROWS_MAX), the same values)
      from tinygrad.llm.kernels.amd_glm_dsa import HALF_ROWS_MAX
      amax = qn.abs().max(-1, keepdim=True)
      s = (amax > HALF_ROWS_MAX).where((amax / HALF_ROWS_MAX).log2().ceil().exp2(), 1.0).float()
      g = all_gather((qn / s).cast(dtypes.half).cat(s.bitcast(dtypes.half), dim=-1).contiguous(), 1)
      L = int(qn.shape[-1])
      qn_all = g[..., :L].float() * g[..., L:].bitcast(dtypes.float32)

    # (x: hn as it is (its shape and devices; the attention reads none of it))
    pre:dict = {"x": hn, "qn": qn_all}
    kv_a = self.attn_kv_a_mqa(xl)
    if self.cache_s.dtype == dtypes.half or int(self.cache_s.shape[-1]) in (FP8_ROPE_ROW, Q4_ROPE_ROW):
      # (the latent and rope key of the device's rows (the same norm and rope a row), sent in half as the cache and the gather keep them)
      lora, rope = cfg.kv_lora_rank, cfg.rope_dim
      freqs_own = shard_rows(self.freqs_cis[start_pos:start_pos+T].pad_to((R, *self.freqs_cis.shape[1:])).contiguous(), 0)
      c_kv = self.attn_kv_a_norm(kv_a[..., :lora])
      k_rope = apply_rope(kv_a[..., lora:].reshape(1, R, 1, rope).transpose(1, 2), freqs_own, cfg.rope_interleave)
      pre["k_store"] = all_gather(c_kv.reshape(1, R, lora).cat(k_rope.reshape(1, R, rope), dim=-1).half().contiguous(), 1)
    else: pre["kv_a"] = all_gather(kv_a.contiguous(), 1)
    if ind is not None:
      pre["k"] = all_gather(ind.attn_k(xl).contiguous(), 1)
      pre["w"] = all_gather((p if (p:=_skinny(ind.proj, xl)) is not None else ind.proj(xl)).contiguous(), 1)
    return pre

  def _sp_attn_tok(self, hn:Tensor, start_pos:int|UOp, T:int|UOp) -> Tensor|None:
    """
    sequence parallel attention (DSA, see Block._sp_sub), every device the heads of its rows: the row projections of hn (1, R, dim, split on
    its rows) are a device's own rows' (the latent and the indexer's keys gathered, the queries' latent q8 gathered); the queries of the
    devices' heads of all rows sent to the devices of the rows (all_to_all), a device attends its rows with every head (their lists: the
    indexer on its rows), the outputs sent back to the devices of their heads: (devices, 1, R, dim) the partial sums of the output projection
    """
    cfg = self.config
    if not self.tok_attn_ok(hn, start_pos): return None
    if self.indexer is None and not ((prev:=_idx_mask[0]) is not None and prev.dtype == dtypes.int32 and prev.uop.axis == 0): return None
    (B, R, _), lora, rope = hn.shape, cfg.kv_lora_rank, cfg.rope_dim
    freqs = self.freqs_cis[start_pos:start_pos+T].pad_to((R, *self.freqs_cis.shape[1:]))
    freqs_own = shard_rows(freqs.contiguous(), 0)
    # the row projections of the device's rows
    qa = self.attn_q_a_norm(self.attn_q_a(hn))
    kv_a = self.attn_kv_a_mqa(hn)
    c_kv = self.attn_kv_a_norm(kv_a[..., :lora])
    k_rope = apply_rope(kv_a[..., lora:].reshape(B, R, 1, rope).transpose(1, 2), freqs_own, cfg.rope_interleave)
    rows = [c_kv.reshape(R, lora).cat(k_rope.reshape(R, rope).cast(c_kv.dtype), dim=-1).half()]
    if (ind:=self.indexer) is not None:
      idx, ir = ind.idx, ind.idx.rope_dim
      ik = ind.k_norm(ind.attn_k(hn).float())[0]
      ik = apply_rope(ik[:, :ir].reshape(1, 1, R, ir), freqs_own, idx.rope_interleave).reshape(R, ir).cat(ik[:, ir:], dim=-1)
      rows.append(ik.half())
    whole = all_gather(Tensor.cat(*rows, dim=-1).contiguous(), 0)
    qa_all = all_gather_q8(qa.contiguous(), 1, ring=bool((1))) if (1) else \
      all_gather(qa.half(), 1)
    # the caches of the chunk's tokens on every device
    cache = Tensor(self.cache_k.uop.after(self.cache_k[:, :, start_pos:start_pos+T, :].uop.store(
      whole[:T, :lora+rope].reshape(1, 1, T, lora+rope).cast(self.cache_k.dtype).uop)))
    if ind is not None:
      S = start_pos + T
      pk = Tensor(ind.cache_dk.uop.after(ind.cache_dk[start_pos:S].uop.store(whole[:T, lora+rope:].uop)))
      iq = ind.attn_q_b(qa).float().reshape(R, idx.n_heads, idx.head_dim).transpose(0, 1).unsqueeze(0)
      iq = apply_rope(iq[..., :ir], freqs_own, idx.rope_interleave).cat(iq[..., ir:], dim=-1)[0].transpose(0, 1)
      weights = (ind.proj(hn).float() * (idx.n_heads * idx.head_dim)**-0.5)[0]
      picks = _idx_mask[1] = pool_index_rows(indexer_scores(iq, pk, weights, S, 1), cast(UOp, start_pos), S, idx.top_k, 1)
    else: picks = unwrap(_idx_mask[0])
    # the queries of the device's heads (all rows), to the devices of the rows
    q = self.attn_q_b(qa_all).reshape(B, R, -1, cfg.head_dim).transpose(1, 2)
    nope = cfg.head_dim - rope
    q_nope, q_rope = q[..., :nope], apply_rope(q[..., nope:], freqs, cfg.rope_interleave)
    k_b = self.attn_k_b["weight"].transpose(-1, -2)
    q_nope = q_nope.half().contiguous().matmul(k_b, dtype=dtypes.float) if k_b.dtype == dtypes.half else q_nope @ k_b
    q = q_nope.cat(q_rope.cast(q_nope.dtype), dim=-1)
    # (SP_A2A_F8: the queries and the outputs sent as e4m3 rows with a scale: half the bytes)
    if (1): q = _f8_unpack(all_to_all(_f8_pack(q).contiguous(), 1, 2), int(q.shape[-1]))

    attn = mla_sparse_tokens(q, cache.reshape(1, 1, *cache.shape[2:]), picks, 1.0/cfg.head_dim**0.5, cast(UOp, start_pos+T), lora,
                             cfg.real_heads or cfg.n_heads)
    if (1): attn = _f8_unpack(all_to_all(_f8_pack(attn).contiguous(), 2, 1), int(attn.shape[-1]))

    v_b = self.attn_v_b["weight"].transpose(-1, -2)
    attn = attn.contiguous().matmul(v_b, dtype=dtypes.float).contiguous() if v_b.dtype == dtypes.half else attn.float() @ v_b
    return self.attn_output.partial(attn.transpose(1, 2).reshape(B, R, -1))

  def tok_attn_ok(self, x:Tensor, start_pos:int|UOp) -> bool:
    """the attention of sequence parallel blocks can be a device's rows with every head (_sp_attn_tok)"""
    cfg, idc, devs = self.config, self.config.indexer, x.device
    self._init_state(x)
    # (a shared layer's config has no indexer: the lists of the full layer before it)
    full = idc is not None and idc.kpool == 1 and cfg.max_context > idc.top_k
    if not ((1) and isinstance(devs, tuple) and (full or cfg.indexer_shared) and isinstance(start_pos, UOp) and
            not self.seq_sharded and cfg.q_lora_rank > 0 and self.cache_k.dtype == dtypes.half and
            (cfg.real_heads or cfg.n_heads) % 16 == 0 and self.cache_k.shape[2] % 64 == 0 and amd_custom_kernels_supported(devs)): return False
    return self.indexer is None or (full and self.indexer.sparse)

  def _dec_proj(self, x:Tensor) -> tuple[Tensor, ...]|None:
    """
    decoding (a few rows, DEC_SPLIT): attn_q_a, attn_kv_a_mqa (and the indexer's attn_k) of x by every device's part of their rows (a copy of
    them split over the devices, _dproj), the parts gathered: (q_a, kv_a[, k]). the replicated weights are read by every device otherwise
    """
    if (dp:=getattr(self, "_dproj", None)) is None or not isinstance(Tq:=x.shape[1], int) or Tq > 8 or token_sharded(x) or x.shape[0] != 1:
      return None
    lin, sizes = dp
    N, per = len(cast(tuple, x.device)), lin.out_features // len(cast(tuple, x.device))
    out = q8_linear(lin, x.reshape(Tq, -1))
    g = host_allgather(out.reshape(Tq, N, per).permute(1, 0, 2).reshape(N, Tq*per))
    if g is None: return None
    full = g.reshape(N, Tq, per).permute(1, 0, 2).reshape(1, Tq, N*per)
    offs = list(itertools.accumulate(sizes, initial=0))
    return tuple(full[..., a:b] for a, b in zip(offs, offs[1:]))

  def _attention_rows(self, x:Tensor, start_pos:int|UOp, T:int|UOp) -> Tensor:
    # x has Tq rows, the first T are the tokens at start_pos
    B, Tq, _ = x.shape
    q_nope_head_dim = self.config.head_dim - self.config.rope_dim
    pre = _sp_pre[0]
    dec = self._dec_proj(x) if pre is None else None
    qa = (dec[0] if dec is not None else self.attn_q_a(x)) if self.config.q_lora_rank > 0 and pre is None else None
    q_proj = self.attn_q_b(pre["qn"]) if pre is not None else self.attn_q_b(self.attn_q_a_norm(qa)) if qa is not None else self.attn_q(x)
    q = q_proj.reshape(B, Tq, self.config.n_heads, self.config.head_dim).transpose(1, 2)
    q_nope, q_rope = q[..., :q_nope_head_dim], q[..., q_nope_head_dim:]
    freqs = self.freqs_cis[start_pos:start_pos+T]
    if Tq is not T: freqs = freqs.pad_to((Tq, freqs.shape[1]))
    q_rope = apply_rope(q_rope, freqs, self.config.rope_interleave)
    # half weights: half copies of the input and the output and a float accumulator use the tensor cores (casts or permutes fused into the
    # matmul don't)
    k_b, half = self.attn_k_b["weight"].transpose(-1, -2), self.attn_k_b["weight"].dtype == dtypes.half and (not isinstance(Tq, int) or Tq > 1)
    # a few tokens: a thread an output (the same values)
    bmm = half and B == 1 and isinstance(Tq, int) and Tq <= 8 and amd_custom_kernels_supported(x.device) and bool((1))
    if bmm: q_nope = mla_bmm(q_nope, self.attn_k_b["weight"])
    else: q_nope = q_nope.half().contiguous().matmul(k_b, dtype=dtypes.float) if half else q_nope @ k_b
    q = mla_q_cat(q_nope, q_rope) if half else \
      (q_nope.contiguous() if half else q_nope).cat(q_rope.cast(q_nope.dtype), dim=-1)

    self._freqs_rows = freqs  # (the indexer's rope: the same frequencies)
    if pre is not None and (ks:=pre.get("k_store")) is not None: k_store = ks.reshape(B, 1, Tq, -1)  # (SP_ROWPROJ: made on the rows' devices)
    else:
      kv_a = pre["kv_a"] if pre is not None else dec[1] if dec is not None else self.attn_kv_a_mqa(x)
      c_kv = self.attn_kv_a_norm(kv_a[..., :self.config.kv_lora_rank])
      k_rope = kv_a[..., self.config.kv_lora_rank:].reshape(B, Tq, 1, self.config.rope_dim).transpose(1, 2)
      k_rope = apply_rope(k_rope, freqs, self.config.rope_interleave)
      k_store = c_kv.reshape(B, 1, Tq, self.config.kv_lora_rank).cat(k_rope.reshape(B, 1, Tq, self.config.rope_dim), dim=-1)
    if self.seq_sharded:
      # the cache split on its positions over the devices (MLA_SEQ_SHARD): every device stores its rows of the tokens, attends all the heads
      # over its keys (the indexer's lists of every token: the pooled keys are on every device), the outputs combined on the devices of the heads
      assert isinstance(start_pos, UOp), "a sequence sharded cache needs the sparse indexer"
      b = _seq_base[0]
      cache = seq_store(self.cache_s, self._seq_rows(k_store[0, 0]), start_pos + b, T)
      if self.indexer is None:
        # (DSA, a shared layer: the lists of the full layer before it)
        lists = unwrap(_idx_mask[0])
      elif self.indexer.idx.kpool == 1:
        lists = _idx_mask[1] = self.indexer.dsa_picks(x, pre["qn"] if pre is not None else self.attn_q_a_norm(unwrap(qa)), start_pos, freqs, T,
                                                      pre["k"] if pre is not None else dec[2] if dec is not None and len(dec) > 2 else None,
                                                      pre["w"] if pre is not None else None)
        # (LISTS_PAD, default on: a chunk's padding rows (past its T tokens) with empty lists, their attention reads no keys (the shared layers
        # after it take these lists too). the tokens' rows are the same: the same bits. 4k: 2.74 -> 2.55 s)
        if (1) and isinstance(Tq, int) and Tq > (16): lists = _idx_mask[1] = lists_pad(lists, T)
      else: lists = self.indexer.picks(x, self.attn_q_a_norm(self.attn_q_a(x)), start_pos, T)
      scale = 1.0/self.config.head_dim**0.5
      if isinstance(Tq, int) and Tq > (16) and (1):
        # a chunk of a prompt: the cache before it read from the devices that have it (peer to peer), the chunk's own rows (every device
        # computes them), the attention of the device's heads on the whole of it
        whole = seq_gather(cache, k_store[0, 0], start_pos, T, round_up(self.config.max_context, 64), base=b, lists=lists)
        # (SEQ_ATTN_TOK: a device's rows with every head, see mla_sparse_seq_tokens)

        # (SEQ_ATTN_LDS: the values' dims only, the kernel that reads a tile of keys into LDS in whole rows: the same values)
        attn = flash_attention_mla_sparse(q, whole.reshape(1, 1, 1, -1, whole.shape[-1]), lists, scale, start_pos+T,
                                                vdims=self.config.kv_lora_rank)
      else: attn = mla_sparse_seq(q, cache, lists, scale, start_pos+T, dims=self.config.kv_lora_rank, base=b)
      v_b, attn = self.attn_v_b["weight"].transpose(-1, -2), attn[..., :self.config.kv_lora_rank]
      if bmm: attn = mla_bmm(attn, self.attn_v_b["weight"])
      else: attn = attn.half().contiguous().matmul(v_b, dtype=dtypes.float).contiguous() if half else attn @ v_b
      return _out(self.attn_output, attn.transpose(1, 2).reshape(B, Tq, -1))
    k_store = k_store[:, :, :T]
    cache = Tensor(self.cache_k.uop.after(self.cache_k[:, :, start_pos:start_pos+T, :].uop.store(k_store.cast(self.cache_k.dtype).uop)))
    (N, D), fast = self.cache_k.shape[2:], amd_custom_kernels_supported(x.device) and self.cache_k.shape[2] % 64 == 0
    fast = fast and D % 64 == 0
    # an indexer on the sparse path: no mask, the attention of a token gathers the keys it sees (a list)
    picks = None
    sparse_ctx = isinstance(start_pos, UOp) and (idc:=self.config.indexer) is not None and self.config.max_context > idc.top_k
    if fast and self.indexer is not None and self.indexer.sparse and sparse_ctx:
      if self.indexer.idx.kpool == 1:
        # (DSA: the lists are the next shared layers' too)
        picks = _idx_mask[1] = self.indexer.dsa_picks(x, self.attn_q_a_norm(unwrap(qa)), cast(UOp, start_pos), freqs, T,
                                                       dec[2] if dec is not None and len(dec) > 2 else None)
      else: picks = self.indexer.picks(x, self.attn_q_a_norm(self.attn_q_a(x)), cast(UOp, start_pos), T)
      mask = None
    elif fast and self.indexer is None and self.config.indexer_shared and (prev:=_idx_mask[0]) is not None and prev.dtype == dtypes.int32:
      # (a shared layer: the lists of the full layer before it, a shared layer's config has no indexer)
      picks, mask = prev, None
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
      if bmm: attn = mla_bmm(attn, self.attn_v_b["weight"])
      else: attn = attn.half().contiguous().matmul(v_b, dtype=dtypes.float).contiguous() if half else attn @ v_b
      return _out(self.attn_output, attn.transpose(1, 2).reshape(B, Tq, -1))
    k = cache[:, :, 0:start_pos+T, :].float()
    v = k[..., :self.config.kv_lora_rank]

    attn = q @ k.transpose(-1, -2) * (1.0 / self.config.head_dim ** 0.5)
    if mask is not None: attn = attn + mask
    attn = attn.softmax(-1)
    attn = ((attn @ v) @ self.attn_v_b["weight"].transpose(-1, -2)).transpose(1, 2).reshape(B, T, -1)
    return _out(self.attn_output, attn)

  def _attention_batch(self, x:Tensor, groups:list[tuple[UOp, int]]) -> Tensor:
    """batched decoding: a group of rows a sequence at its positions, its caches in its slot (the projections of all of them in one kernel each)"""
    B, Tq, _ = x.shape
    sb = _seq_batch[0]
    assert self.config.rope_dim == 0 or sb is not None, \
      "grouped MLA rope needs per-sequence positions"
    q_nope_head_dim = self.config.head_dim - self.config.rope_dim
    freqs = rows_gather(self.freqs_cis, sb.row_positions) if sb is not None and self.config.rope_dim else None
    q_lora = self.attn_q_a_norm(self.attn_q_a(x))
    q = self.attn_q_b(q_lora).reshape(B, Tq, self.config.n_heads, self.config.head_dim).transpose(1, 2)
    q_nope, q_rope = q[..., :q_nope_head_dim], q[..., q_nope_head_dim:]
    if freqs is not None: q_rope = apply_rope(q_rope, freqs, self.config.rope_interleave)
    # (many rows: the tensor cores)
    tc = Tq > (32) and self.attn_k_b["weight"].dtype == dtypes.half
    if (qb:=bmm_wmma(q_nope, self.attn_k_b["weight"])) is not None: q = qb
    elif tc: q = q_nope.half().contiguous().matmul(self.attn_k_b["weight"].transpose(-1, -2), dtype=dtypes.float).contiguous()
    else: q = mla_bmm(q_nope, self.attn_k_b["weight"])
    q = q.cat(q_rope.cast(q.dtype), dim=-1)
    kv_a = self.attn_kv_a_mqa(x)
    c_kv = self.attn_kv_a_norm(kv_a[..., :self.config.kv_lora_rank])
    k_rope = kv_a[..., self.config.kv_lora_rank:].reshape(B, Tq, 1, self.config.rope_dim).transpose(1, 2)
    if freqs is not None: k_rope = apply_rope(k_rope, freqs, self.config.rope_interleave)
    k_store = c_kv.reshape(B, 1, Tq, self.config.kv_lora_rank).cat(k_rope.reshape(B, 1, Tq, self.config.rope_dim), dim=-1)
    idx = self.indexer
    if idx is not None:
      ic = idx.idx
      if ic.kpool == 1:
        ir = ic.rope_dim
        assert freqs is not None, "GLM DSA grouped steps need their rows' RoPE"
        kg = idx.k_norm(idx.attn_k(x).float())[0]
        kg = apply_rope(kg[:, :ir].reshape(1, 1, Tq, ir), freqs, ic.rope_interleave).reshape(Tq, ir).cat(kg[:, ir:], dim=-1)
        iq = idx.attn_q_b(q_lora).float().reshape(Tq, ic.n_heads, ic.head_dim).transpose(0, 1).unsqueeze(0)
        iq = apply_rope(iq[..., :ir], freqs, ic.rope_interleave).cat(iq[..., ir:], dim=-1)[0].transpose(0, 1)
      else:
        kg = idx.k_norm(idx.attn_k(x)).stack(idx.compressor_gate(x), dim=2)[0]
        iq = idx.attn_q_b(q_lora).reshape(Tq, ic.n_heads, ic.head_dim)
      iw = (p if not _seq_alone() and (p:=_skinny(idx.proj, x)) is not None else idx.proj(x)) * (ic.n_heads * ic.head_dim)**-0.5
      iw = iw[0]
    # (sequences sharing the caches: a layer sharing the indexer of the full layer before it (indexer_shared) takes that layer's lists)
    shared = idx is None and sb is not None and bool(self.config.indexer_shared) and _idx_mask[0] is not None
    assert idx is not None or shared, "batched decoding: the sparse attention of the indexer"
    if idx is None: kg = iq = iw = cast(Tensor, None)  # (a shared layer: no keys or queries of its own)
    def core(q:Tensor, k_store:Tensor, kg:Tensor, iq:Tensor, iw:Tensor, groups:list[tuple[UOp, int]], pf:BatchPrefill|None,
             pools:tuple[Tensor, ...]) -> tuple[Tensor, tuple[Tensor, ...]]:
      # one kernel for all the sequences (the same rows each, the caches of slot g the g-th of the pools): their positions in a tensor. returns
      # the attention and the pools (pooled keys, ring, cache) after it (another part's writes after its reads)
      assert len({T for _, T in groups}) == 1, "batched decoding: groups of the same rows"
      Tq, scale = int(q.shape[2]), 1.0/self.config.head_dim**0.5
      if sb is not None:
        assert pf is None and self.seq_sharded, "sequences sharing the caches: decoding steps, a sequence sharded cache"
        U, nd = REGION_ALIGN, len(cast(tuple, x.device))
        assert U % (SEQ_BLK*nd) == 0 and (idx is None or U % idx.idx.kpool == 0), f"REGION_ALIGN {U}: blocks of {SEQ_BLK} on {nd} devices"
        span, rslots = -(-self.config.max_context // U), (sb.bases // U).contiguous()
        if idx is None:
          picks, pk = unwrap(_idx_mask[0]), None
          cache = seq_store_grouped(pools[0], self._seq_rows(k_store[0, 0]), sb.positions, rslots, U // nd, span=span)
        elif ic.kpool == 1:
          # (split keys: a region's rows on every device as the latents', U//devices a region slot; else U rows on every device)
          ku = U // nd if idx.shard else U
          pk = seq_store_grouped(pools[0], kg.half(), sb.positions, rslots, ku, span=span, split=bool(idx.shard))
          # (the lists are the next shared layers' too, see FFNBlock.__call__)
          picks = _idx_mask[1] = indexer_decode_index_rows(iq, pk, iw, None, None, ic.top_k, 1,
                                                           grouped=(sb.positions, ku, rslots, span))
          cache = seq_store_grouped(pools[1], self._seq_rows(k_store[0, 0]), sb.positions, rslots, U // nd, span=span)
        else:
          R = int(idx.slot_ring[0].shape[0])
          pk = pool_keys(pools[0], pools[1], kg.half(), idx.compressor_ape["weight"], None, Tq,
                         grouped=(sb.positions, U // ic.kpool, R, rslots, None, span, sb.slots))
          picks = indexer_decode_index_rows(iq, pk, iw, None, None, ic.top_k // ic.kpool, ic.kpool,
                                            grouped=(sb.positions, U // ic.kpool, rslots, span))
          cache = seq_store_grouped(pools[2], self._seq_rows(k_store[0, 0]), sb.positions, rslots, U // nd, span=span)
        attn = mla_sparse_seq(q, cache, picks, scale, None, grouped=(sb.positions, rslots, U // nd, span),
                              dims=self.config.kv_lora_rank)
        return attn, ((cache,) if pk is None else (pk, cache))
      assert idx is not None
      positions = _group_positions(groups, x.device)
      P, R = idx.pk_slot(), int(idx.slot_ring[0].shape[0])
      # (prefills: the slots of the prompts, their token counts)
      slots, counts = (pf.slots.to(x.device), pf.counts.to(x.device)) if pf is not None else (None, None)
      pk = pool_keys(pools[0], pools[1], kg.half(), idx.compressor_ape["weight"], None, Tq, grouped=(positions, P, R, slots, counts))
      picks = indexer_decode_index_rows(iq, pk, iw, None, None, ic.top_k // ic.kpool, ic.kpool, grouped=(positions, P, slots))
      if self.seq_sharded:
        # (the slots' caches split on their positions as the model's: a device stores its rows of every sequence's tokens, attends all the
        # heads of every row over its keys of the row's slot)
        stride = int(self.slot_s[0].shape[0]) // len(cast(tuple, x.device))
        cache = seq_store_grouped(pools[2], self._seq_rows(k_store[0, 0]), positions, slots, stride)
        attn = mla_sparse_seq(q, cache, picks, scale, None, grouped=(positions, slots, stride), dims=self.config.kv_lora_rank)
      else:
        ctx, D = int(self.pool_k.shape[2]), int(self.pool_k.shape[-1])
        cache = grouped_rows(pools[2].reshape(-1, D), k_store[0, 0], positions, ctx, slots)
        attn = flash_attention_mla_sparse(q, cache, picks, scale, None, grouped=(positions, ctx, slots))[..., :self.config.kv_lora_rank]
      return attn, (pk, Tensor(pools[1].uop.after(pk.uop)), cache)
    pools:tuple[Tensor, ...]
    if sb is not None:
      pools = (self.cache_s,) if idx is None else (idx.cache_dk, self.cache_s) if idx.idx.kpool == 1 else (idx.cache_pk, idx.pool_ring, self.cache_s)
    else: pools = (unwrap(idx).pool_pk, unwrap(idx).pool_ring, self.pool_s if self.seq_sharded else self.pool_k)
    if (pf:=_batch_prefill[0]) is not None and pf.off:
      # (a mixed step: the rows of decoding, then the prompts' (their writes after the decoding's reads))
      o, nd = pf.off, len(groups) - len(pf.count_vars)
      dec, pools = core(q[:, :, :o], k_store[:, :, :o], kg[:o], iq[:o], iw[:o], groups[:nd], None, pools)
      attn = dec.cat(core(q[:, :, o:], k_store[:, :, o:], kg[o:], iq[o:], iw[o:], groups[nd:], replace(pf, off=0), pools)[0], dim=2)
    else: attn = core(q, k_store, kg, iq, iw, groups, pf, pools)[0]
    if (vb:=bmm_wmma(attn, self.attn_v_b["weight"])) is not None: attn = vb
    elif tc: attn = attn.half().contiguous().matmul(self.attn_v_b["weight"].transpose(-1, -2), dtype=dtypes.float).contiguous()
    else: attn = mla_bmm(attn, self.attn_v_b["weight"])
    return _out(self.attn_output, attn.transpose(1, 2).reshape(B, Tq, -1))

  def _init_slots(self, n:int, ctx:int):
    """the caches of n sequences of batched decoding of at most ctx tokens (one slot each)"""
    if len(getattr(self, "slot_k", getattr(self, "slot_s", []))) >= n and (self.indexer is None or len(getattr(self.indexer, "slot_ring", [])) >= n):
      return
    # (aliased: the caches of the slots made with the model's, see _init_state)
    if len(getattr(self, "slot_k", getattr(self, "slot_s", []))) < n:
      if self.seq_sharded:
        # (every device its rows of every slot, as the model's cache: seq_rows of the slot's positions)
        nd = len(cast(tuple, self.cache_s.device))
        self.pool_s, self.slot_s = _slot_pool(n, self.cache_s, (nd*seq_rows(ctx, nd), self.cache_s.shape[-1]))
      else: self.pool_k, self.slot_k = _slot_pool(n, self.cache_k, (1, 1, ctx, self.cache_k.shape[-1]))
    if (idx:=self.indexer) is not None:
      pools = round_up(ctx, idx.idx.kpool) // idx.idx.kpool
      if idx.shard: pools = idx.shard * (seq_rows(ctx, idx.shard) // idx.idx.kpool)  # (every device its pools of every slot)
      if len(getattr(idx, "slot_pk", [])) < n: idx.pool_pk, idx.slot_pk = _slot_pool(n, idx.cache_pk, (pools, idx.cache_pk.shape[-1]))
      idx.pool_ring, idx.slot_ring = _slot_pool(n, idx.ring)

  def _init_seq_slots(self, n:int):
    """Pooled-indexer rings are per-slot recurrent state; kpool=1 DSA keys live wholly in each sequence's region."""
    idx = self.indexer
    if idx is not None and idx.idx.kpool > 1 and len(getattr(idx, "slot_ring", [])) < n:
      idx.pool_ring, idx.slot_ring = _slot_pool(n, idx.ring)

  def _attn_mask(self, x:Tensor, start_pos:int|UOp, T:int|UOp|None=None) -> Tensor|None:  # (1, 1, Tq, start_pos+T), None: all visible
    if self.indexer and self.config.max_context > self.indexer.idx.top_k:
      if self.indexer.idx.kpool == 1:
        # (DSA: the mask is the next shared layers' too)
        m = self.indexer.dsa_mask(x, self.attn_q_a_norm(self.attn_q_a(x)), start_pos, self._freqs_rows, T)
        _idx_mask[1] = m
        return m
      return self.indexer(x, self.attn_q_a_norm(self.attn_q_a(x)), start_pos, T)
    if self.config.indexer_shared and _idx_mask[0] is not None:
      assert _idx_mask[0].dtype != dtypes.int32, "the lists of the sparse path aren't a mask"
      return _idx_mask[0]
    T = x.shape[1]
    return Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, buffer=False).triu(start_pos+1) if resolve(T != 1) else None

  @property
  def seq_sharded(self) -> bool: return hasattr(self, "cache_s")
  def _seq_rows(self, k:Tensor) -> Tensor:
    """the rows of the tokens as the sequence sharded cache's: half, or fp8 / 4-bit (SEQ_FP8 / SEQ_KV4, see fp8_rows / q4_rows)"""
    return reduced_rows(k, int(self.cache_s.shape[-1])) if self.cache_s.dtype == dtypes.uint32 else k
  def slot_rows(self) -> int:
    """(sequence sharded) the rows of a device of a slot"""
    return int(self.slot_s[0].shape[0]) // len(cast(tuple, self.cache_s.device))
  @staticmethod
  def slot_rows_view(t:Tensor) -> Tensor:
    """(sequence sharded) a (n*rows, D) cache or pool as (rows, n, D): its rows on every device (axis 0 isn't the split one)"""
    n = len(to_tuple(t.device))
    return t.reshape(n, -1, t.shape[-1]).permute(1, 0, 2)
  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_k") and not hasattr(self, "cache_s"):
      D = self.config.kv_lora_rank + self.config.rope_dim
      if seq_sharded(x.device):
        # (MLA_SEQ_SHARD: every device the rows of its positions, see seq_store)
        devs = cast(tuple[str, ...], x.device)
        # (SEQ_FP8: a row of fp8 values and their scales, see fp8_rows: half the memory. 576 dims (C9): the rope dims in half, 656 bytes)
        # (SEQ_KV4 with it: 4-bit rows, see q4_rows: 416 bytes)
        fp8, q4 = getenv("SEQ_FP8", 0) and D in (512, 576), getenv("SEQ_KV4", 0)
        W = (Q4_ROW if q4 else FP8_ROW) if D == 512 else (Q4_ROPE_ROW if q4 else FP8_ROPE_ROW)
        row, dt = ((W,), dtypes.uint32) if fp8 else ((D,), dtypes.half)
        N = max(self.config.max_context, self.config.cache_rows) + (REGION_ALIGN if self.config.cache_rows else 0)
        self.cache_s = Tensor.empty(len(devs)*seq_rows(N, len(devs)), *row, dtype=dt, device=devs, axis=0)
      else: self.cache_k = Tensor.empty(x.shape[0], 1, self.config.max_context, D, dtype=dtypes.half, device=x.device)
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta,
                                            device=x.device, yarn=self.config.yarn)
    if self.indexer: self.indexer._init_state(x)

class MTPBlock(MLATransformerBlock):
  """
  the next token prediction layer: from the hidden state of a position (the last block's, before the output norm) and the embedding of the
  next token, a transformer block (without hyper-connections) predicts the token after it (with the output head)
  """
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    self.eh_proj = Linear(2 * config.dim, config.dim, bias=False)
    self.enorm, self.hnorm = nn.RMSNorm(config.dim, config.norm_eps), nn.RMSNorm(config.dim, config.norm_eps)
    self.shared_head_norm = nn.RMSNorm(config.dim, config.norm_eps)
  def predict(self, emb:Tensor, h:Tensor, start_pos:int|UOp) -> Tensor:
    """emb (B, T, dim) the embeddings of the next tokens, h (B, T, dim) the hidden states: the normed hidden states of the predictions"""
    _idx_mask[0] = None
    return self.shared_head_norm(self(self.eh_proj(self.enorm(emb).cat(self.hnorm(h), dim=-1)), start_pos))

# The runtime driver is shared; its model-specific block/kernel types deliberately are not (mypy sees the boundary as Any).
GLMSeqDriver: Any = glm_seq_base.Transformer
class Transformer(GLMSeqDriver):
  # (no top_k in this sampler (_sample): a server with SAMPLE_TOP_K / TOP_K ignores a request's, said once, see serve.LLMServer)
  SAMPLE_TOP_K = False
  def __init__(self, config:TransformerConfig):
    dense_config = replace(config, num_experts=0, num_experts_per_tok=0, shared_expert_dim=0, hidden_dim=config.dense_hidden_dim or config.hidden_dim)
    self.config = config
    assert config.kv_lora_rank > 0, "glm_dsa runs GLM-5.3 (glm-dsa): MLA blocks only, no recurrent or plain attention blocks"
    self.blk:list[FFNBlock] = []
    for i in range(config.num_blocks):
      c = dense_config if i < config.leading_dense_blocks else config
      if config.sliding_window_pattern != 0 and (i+1) % config.sliding_window_pattern == 0: c = replace(c, sliding_window=0)
      if config.indexer_shared and config.indexer_shared[i]: c = replace(c, indexer=None)
      self.blk.append(MLATransformerBlock(c))
    self.token_embd  = nn.Embedding(config.vocab_size, config.dim)
    self.output_norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.output = Linear(config.dim, config.padded_vocab_size or config.vocab_size, bias=False)
    if config.mtp: self.mtp = MTPBlock(config)
    self.max_context = config.max_context
    self.has_recurrent_block = False  # (glm5next's methods ask)
    self._cached_tokens: list[int] = []
    self._effects: list[Tensor]|None = None  # tensors that forward realizes with its output (see _forward_effects)
    self._ckpt_tokens: list[int] = []  # the prompt of the recurrent state's checkpoint (_checkpoint)
    # a jit for every token count of a call (the max size of a symbolic chunk, 1 for decoding): static padded shapes of a small chunk don't pay
    # for a big one
    self.jits:dict[Any, Any] = {}
    self._pending_load:list[Callable[[], None]] = []  # the work left of loading (from_gguf defer_load)
    self._loaded = False  # the weights are loaded (from_gguf)
    self._scratch:dict = {}  # the memory of the jits' intermediates: they run one at a time

  def _main_token_rows(self) -> int:
    # (GLM_SEQ: the token buffer is the sequences' regions' too, whichever path makes it first)
    return self._token_rows() if self.config.cache_rows else self.max_context+1

  def _embed(self, tokens:Tensor) -> Tensor:
    # (GLM_SEQ: the tokens are a view of the token buffer at seq_base+start_pos, two variables: embedded from a copy. on the view the embedding
    # isn't a gather, the one-hot of the vocab is made and reduced with the whole table (2688 rows: minutes, the devices' wait times out))
    tokens = tokens.to(self.token_embd.weight.device)
    return self.token_embd(tokens.contiguous() if isinstance(_seq_base[0], UOp) else tokens).float()

  def forward(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    _idx_mask[0] = None
    x = emb = self._embed(tokens)  # (B, T, D)
    if self.glm_sp(x):
      # (sequence parallel prefill: every device the residual of its rows, the sublayers' inputs gathered, their outputs reduce scattered)
      T, _idx_mask[0] = x.shape[1], None
      rs = shard_rows(x.pad_to(x.max_shape).contiguous(), 1)
      for block in self.blk: rs = block.sp_block(rs, start_pos, T)
      x = all_gather(rs, 1)
      if self.config.spec_drafts:
        _idx_mask[0] = None
        self._mtp_prefill(emb, x, cast(UOp, start_pos), cast(UOp, T))
      x = x[:, :T]
    else:
      for block in self.blk: x = block(x, start_pos)
      if self.config.spec_drafts and not isinstance(T:=x.shape[1], int) and amd_custom_kernels_supported(x.device):
        # (a prompt chunk: the MTP layer's caches of it, on the model's output padded to the chunk's rows (static), T of them real)
        _idx_mask[0] = None
        self._mtp_prefill(emb, x.pad_to(x.max_shape).contiguous(), cast(UOp, start_pos), T)
    return self._tail(x, tokens, start_pos, temperature)

  def _tail(self, x:Tensor, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    # only run the output projection on the last token
    tok = self._sample(self.output_norm(x[:, -1:])[:, -1, :], temperature, tokens.device)
    q8_quantize_cache_clear()  # don't keep the graph alive
    # the tokens are a view of generate's token buffer: the sampled token goes after them unless that is a prompt token (chunked prefill)
    if (buf:=getattr(self, "_token_buf", None)) is not None and tokens.uop.base is buf.uop.base:
      # GLM_SEQ: tokens are a region view, while start_pos and RoPE remain sequence-relative.
      tok = (dst:=buf[:, (nxt:=_seq_base[0] + start_pos + tokens.shape[1]):nxt+1]).assign((dst < 0).where(tok.cast(buf.dtype), dst))
    return tok

  # ** sequence parallel prefill in parts: a jit of a range of the blocks each (captured at once, in processes of their own), the residual
  # stream's rows of every device and the indexer's lists of the last full layer from one to the next
  def _part_ranges(self) -> list[tuple[int, int]]:
    P = max(1, min(getenv("PREFILL_PARTS", 4), len(self.blk)))
    b = [round(i*len(self.blk)/P) for i in range(P+1)]
    return [(b[i], b[i+1]) for i in range(P)]
  def _sp_tokens(self, tokens:Tensor) -> bool:
    devs = self.token_embd.weight.device
    return isinstance(devs, tuple) and not isinstance(T:=tokens.shape[1], int) and \
      int(T.vmax) % (16*len(devs)) == 0 and amd_custom_kernels_supported(devs) and bool((1)) and \
      len(self._part_ranges()) > 1
  def _has_picks(self, b:int) -> bool: return any(getattr(blk, "indexer", None) is not None for blk in self.blk[:b])
  def _sp_part(self, k:int, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> tuple:
    # (the parts pass the rows and the lists in buffers of their own, two of each: a part reads one and writes the other)
    self._effects = []
    ranges, T, key = self._part_ranges(), tokens.shape[1], int(cast(UOp, tokens.shape[1]).vmax)
    (a, b), bufs = ranges[k], self._sp_bufs[key]
    emb = self._embed(tokens)
    if k == 0: x, pick = shard_rows(emb.pad_to(emb.max_shape).contiguous(), 1), None
    else: x, pick = bufs[k % 2][0], bufs[k % 2][1] if self._has_picks(a) else None
    for block in self.blk[a:b]:
      _idx_mask[0] = pick
      x = block.sp_block(x, start_pos, T)
      pick = _idx_mask[0]
    if k < len(ranges) - 1:
      self._effects = None
      q8_quantize_cache_clear()
      nxt = bufs[(k + 1) % 2]
      return (nxt[0].assign(x), *((nxt[1].assign(unwrap(pick).cast(dtypes.int32)),) if self._has_picks(b) else ()))
    x = all_gather(x, 1)
    if self.config.spec_drafts:
      _idx_mask[0] = None
      self._mtp_prefill(emb, x, cast(UOp, start_pos), cast(UOp, T))
    out = self._tail(x[:, :T], tokens, start_pos, temperature)
    effects, self._effects = self._effects, None
    return (out, *effects)
  def _part_jits(self, key:int) -> list:
    if (jits:=self.jits.get(("parts", key))) is None:
      self._init_states()
      if not hasattr(self, "_sp_bufs"): self._sp_bufs:dict[int, list[tuple[Tensor, Tensor]]] = {}
      devs = cast(tuple[str, ...], self.token_embd.weight.device)
      top = unwrap(self.config.indexer).top_k + 2
      # (the lists: the device's rows' (token parallel attention), else every row's on every device (a sequence sharded cache))
      pax = 0 if (1) and not seq_sharded(devs) else None
      self._sp_bufs[key] = [(Tensor.empty(1, key, self.config.dim, dtype=dtypes.float32, device=devs, axis=1).contiguous().realize(),
                             Tensor.empty(key, top, dtype=dtypes.int32, device=devs, axis=pax).contiguous().realize()) for _ in range(2)]
      jits = self.jits[("parts", key)] = [TinyJit(functools.partial(self._sp_part, k), capture_first=True,
                                                  scratch=self._scratch if (1) else None)
                                           for k in range(len(self._part_ranges()))]
      for k, j in enumerate(jits): setattr(j, "warmup_name", f"prefill{key}_p{k}")
    return jits

  def _sample(self, h:Tensor, temperature:Tensor|None, device:str|tuple[str, ...]|None, noise:tuple[Tensor, Tensor]|None=None) -> Tensor:
    """
    the token (B, 1) after every normed hidden state h (B, D), sampled (greedy without a temperature). on device unless the output is split.
    temperature: (rows, 1) the temperature of every row (or one), or (rows, 2) and its top_p (only the tokens of the top_p nucleus are sampled),
    or (rows, 3) and a top_k (ignored: no top_k in this sampler, see SAMPLE_TOP_K).
    noise: (seed, keys (B,)): the uniform noise of a row a hash of the seed, its key and the token (see _hash_uniform), not drawn
    """
    logits = self.output(h)
    # (LOGPROBS: the logits of the last row in a buffer split as they are, realized with the step: comparisons with a reference)
    # Gumbel-max trick: argmax(logits/temp - log(-log(uniform))) is equivalent to sampling from softmax(logits/temp)
    if temperature is not None:
      # (a row of temperature 0: the most likely, not scaled or noised. it was the logits / 1e-12 plus the noise: a tie a random one of them,
      # and a u that the device's log takes to 0 an infinite noise, a random token: batched greedy decoding (GLM_SEQ: temperatures always
      # given) differed run to run, in its drafts (~4% of the steps of 4 sequences) and its tokens)
      tp = temperature.to(logits.device).reshape(-1, temperature.shape[-1])
      hot = tp[:, :1] > 0
      z = hot.where(logits.float() / tp[:, :1].maximum(1e-12), logits.float())
      u = _hash_uniform(noise[0], noise[1], logits) if noise is not None else Tensor.rand_like(logits)
      # (-log(u) at least 1-u (it is): the device's log of a u a few ulps below 1 is 0, its noise an infinity)
      logits = z - hot.where(u.maximum(1e-12).log().neg().maximum(1 - u).log(), 0)
      # (the rows: (temperature,), (temperature, top_p), or the seq steps' (temperature, top_p, top_k) (glm5next TEMP_W). a test of exactly two
      # columns dropped the top_p of every seq step from a6dd1485d on: tokens from the whole vocabulary at temperature 1, see
      # results/incidents/ts5-stray-tokens)
      if tp.shape[1] >= 2: logits = (nucleus(z, self.config.vocab_size, tp[:, 1:2]) | hot.logical_not()).where(logits, -math.inf)
    # a split output: every device takes the max of its part
    if (tok:=sharded_argmax(logits, self.config.vocab_size)) is not None: return tok
    return _argmax_rows(logits.to(device)[:, :self.config.vocab_size])

  # ******** speculative decoding: the next token prediction (MTP) layer drafts k tokens, the model verifies them in one step ********
  # the MTP layer at a position p predicts the token after it from its token and the (normed) hidden state of the model at p-1

  def _spec_init(self):
    if hasattr(self, "_spec_h"): return
    k, devs = self.config.spec_drafts, self.token_embd.weight.device
    # the hidden states of the last verified tokens (row 0: the last prompt token after a prefill), the row of the last accepted token
    # two of them: a step reads the last one's while it writes its own (the step after it can be launched before the host gets the tokens)
    self._spec_h, self._spec_row = Tensor.zeros(2*(k+1), self.config.dim, device=devs).contiguous().realize(), 0
    self.spec_stats = [0, 0]  # the steps and the tokens they gave
    # in host memory: the step's input tokens (the last step's outputs), its sampled tokens and its drafts (the host polls them)
    self._spec_host, self._spec_in = _host_ints(cast(str|tuple[str, ...], devs), k+1)
    # the outputs are written by one device: the host takes them, resets them and launches the next step once one device wrote them, another
    # one writing them later would overwrite the reset with the last step's
    self._spec_host_out, self._spec_out = _host_ints(devs[0] if isinstance(devs, tuple) else cast(str, devs), 2*k+1)

  def _spec_row0(self, row:int):
    """the hidden state of the position before a prompt (row 0 of _spec_h): the one of row `row` (the last accepted token), 0: none (zero)"""
    if not hasattr(self, "_spec_row0_jit"):
      self._spec_row0_jit = TinyJit(self._spec_row0_copy, capture_first=True)
      setattr(self._spec_row0_jit, "warmup_name", "copy/spec_row0")
    k = self.config.spec_drafts
    self._spec_row0_jit(UOp.variable("spec_row", 0, 2*k+1).bind(row), UOp.variable("spec_keep", 0, 1).bind(int(row != 0)))

  def _mtp_prefill(self, emb:Tensor, x:Tensor, start_pos:UOp, T:UOp) -> Tensor:
    """
    the MTP layer on a prompt chunk (for its caches): emb (1, T, D) the embeddings of its tokens, x (1, rows, D) the model's (padded) output.
    the hidden state of the position before the chunk is row 0 of _spec_h, the one of its last token is written there. returns x
    """
    hn = self.output_norm(x).float()
    emb, carry = emb.pad_to((1, x.shape[1], emb.shape[-1])), self._spec_h[0:1].unsqueeze(0).contiguous()
    # the MTP layer reads the carry before it's written (the schedule orders an assign after the reads). realized with the call's output
    inp = self.mtp.enorm(emb).cat(self.mtp.hnorm(carry.cat(hn[:, :-1], dim=1)), dim=-1)
    m = self.mtp(self.mtp.eh_proj(inp), start_pos, T)
    # (a column of every row: a row split over the devices isn't sliced)
    effects = [m[:, :, :1].contiguous(), self._spec_h[0:1].assign(hn[0].shrink(((T-1, T), None)))]
    if self._effects is None: Tensor.realize(*effects)
    else: self._effects += effects
    return x

  def forward_spec(self, mtp_pos:UOp, j:UOp, start_pos:UOp, par:UOp, temperature:Tensor, *chain:UOp) -> Tensor:
    """
    a step of speculative decoding. the last one verified the tokens at mtp_pos-1 (_spec_in: the sampled token after every one, _spec_h: their
    hidden states, the rows of the other parity), j were accepted: the token at start_pos=mtp_pos+j is the sampled one after the last accepted.
    the MTP layer catches up on the positions from mtp_pos (with the real tokens and hidden states, the rows after j are overwritten later) and
    drafts k tokens after start_pos (at the positions of chain), the model verifies them (their hidden states in the rows of parity par).
    returns (1, 2k+1): the sampled token after every one of the k+1, the drafts
    """
    k, devs, y = self.config.spec_drafts, self.token_embd.weight.device, self._spec_in
    last = self._spec_h.shrink((((1-par)*(k+1), (2-par)*(k+1)), None)).unsqueeze(0).contiguous()
    m = self.mtp.predict(self.token_embd(y.to(devs)).float(), last, mtp_pos).shrink((None, (j, j+1), None))
    drafts:list[Tensor] = []
    for i in range(k):
      drafts.append(self._sample(m[:, 0, :], None, y.device))
      if i < k-1: m = self.mtp.predict(self.token_embd(drafts[-1].to(devs)).float(), m, chain[i])
    # verify: the model on the token at start_pos and the drafts, the hidden states of all of them are kept
    x = self.token_embd(y.shrink((None, (j, j+1))).cat(*drafts, dim=1).contiguous()).float()
    _idx_mask[0] = None
    for block in self.blk: x = block(x, start_pos)
    tok = self._sample(self._spec_h.shrink(((par*(k+1), (par+1)*(k+1)), None)).assign(self.output_norm(x)[0].float()), temperature, y.device)
    q8_quantize_cache_clear()  # don't keep the graph alive
    out = tok.reshape(1, k+1).cat(*drafts, dim=1).cast(dtypes.int32)
    return self._spec_out.assign(Tensor(out.uop.mselect(0)) if isinstance(out.device, tuple) else out)

  def _spec_call(self, mtp_pos:int, j:int, par:int, temperature:Tensor):
    """a step of speculative decoding (forward_spec, its jit)"""
    k, pos = self.config.spec_drafts, mtp_pos + j
    if not hasattr(self, "_spec_jit"):
      self._spec_jit = TinyJit(self.forward_spec, capture_first=getattr(self, "_states_ready", False), scratch=self._scratch)
      setattr(self._spec_jit, "warmup_name", "step/spec")
    if not hasattr(self, "_spec_vars"):
      # (made once: a step binds them)
      pos_var = functools.partial(UOp.variable, min_val=0, max_val=self.max_context-1)
      self._spec_vars = (pos_var("mtp_pos"), UOp.variable("spec_j", 0, k), pos_var("start_pos"), UOp.variable("spec_par", 0, 1),
                         tuple(pos_var(f"spec_pos{i}") for i in range(k-1)))
    v_mtp, v_j, v_pos, v_par, v_chain = self._spec_vars
    self._spec_jit(v_mtp.bind(mtp_pos), v_j.bind(j), v_pos.bind(pos), v_par.bind(par), temperature, *(v.bind(pos+1+i) for i,v in enumerate(v_chain)))

  def seq_prefill(self, slot:int, base:int, tokens:list[int], start:int, hrow:int, temp:float|tuple[float, ...]=0.0,
                  upto:int|None=None) -> int|None:
    """
    the sequence engine's prefill (glm5next), its p2p run-out retried (P2P_RETRY, default 2): a p2p poll that ran out in the prompt's chunks
    (P2P_AR_TIMEOUT_MS: a stalled exchange; the sums of its step wrong, P2PRunOut at the wait for the token) runs the prompt from start again:
    every cache row from start written again, the token buffer reset, the same values (the epochs and mailboxes are consistent after a
    run-out). its slot's states read at start (hrow, keep) are the ones the failed run wrote back: only the first token's draft (MTP) can differ.
    slices (upto: not waited for) are not retried here: a run-out there is seen by a later wait, which runs them again first (_p2p_slices).
    P2P_RETRY_TEST=1: the first prefill's token taken as a run-out (the retry's path without a stalled card)
    """
    retries, slices = 0, list(self._p2p_slices)
    while True:
      try:
        out = super().seq_prefill(slot, base, tokens, start, hrow, temp, upto)
        if upto is not None: self._p2p_unwaited, self._p2p_slices = True, [*self._p2p_slices, (slot, base, list(tokens), start, hrow, temp, upto)]
        if upto is None and getenv("P2P_RETRY_TEST", 0) and retries == 0: raise P2PRunOut("P2P_RETRY_TEST: the first prefill taken as a run-out")
        return out
      except P2PRunOut as e:
        if upto is not None or (retries:=retries+1) > getenv("P2P_RETRY", 2): raise
        print(f"seq_prefill: {e}: the prompt again from {start} ({retries}), {len(slices)} unwaited slices first", flush=True)
        self._p2p_redo(slices)

  def seq_first(self, *args, **kwargs) -> int:
    """the token after a prompt that is its checkpoint (glm5next), its p2p run-out retried as seq_step's (only reads the slot's states)"""
    retries, slices = 0, list(self._p2p_slices)
    while True:
      try: return super().seq_first(*args, **kwargs)
      except P2PRunOut as e:
        if (retries:=retries+1) > getenv("P2P_RETRY", 2): raise
        print(f"seq_first: {e}: again ({retries}), {len(slices)} unwaited slices first", flush=True)
        self._p2p_redo(slices)

  def _p2p_redo(self, slices:list[tuple]) -> None:
    """after a run-out: the devices finished, then the prompt slices since the last wait run again in their order (the run-out can be one
    of theirs: their rows written again at the same positions, the same values); the caller runs its own work again after this"""
    for d in to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device)): Device[d].synchronize()
    self._p2p_slices = []
    for a in slices:
      super().seq_prefill(*a)
      self._p2p_unwaited, self._p2p_slices = True, [*self._p2p_slices, a]
    self._p2p_mark = time.perf_counter()

  def _wait_host(self, get:Callable[[], Any], what:str) -> Any:
    """the value a step writes to host memory (polled: -1 until then). a step that doesn't come (the devices' errors): an error, not a hang"""
    st = time.perf_counter()
    while isinstance(v:=get(), int) and v < 0:
      if time.perf_counter() - st > 10:
        for d in to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device)): Device[d].synchronize()
        if isinstance(v:=get(), int) and v < 0: raise RuntimeError(f"{what}: none after the devices finished")
        break
    # (a p2p poll that runs out (P2P_AR_TIMEOUT_MS) ends with wrong sums: the time since the last wait is at least that long then. after a long
    # one the devices' notes are read (p2p_check): an error, not the tokens of wrong sums. the steps of a graph are far shorter: no reads)
    if (now:=time.perf_counter()) - self._p2p_mark > P2P_CHECK_S:
      p2p_check(cast(tuple[str, ...], to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device))))
      now = time.perf_counter()
    self._p2p_mark, self._p2p_unwaited, self._p2p_slices = now, False, []
    return v
  _p2p_mark, _p2p_unwaited, _p2p_slices = 0.0, False, cast(list[tuple], [])

  def seq_step(self, seqs, *args, **kwargs):
    """
    a step of the sequences (glm5next), its p2p run-out retried (P2P_RETRY, default 2): the same step again, its cache rows written again at the
    same positions (the epochs and mailboxes are consistent after a run-out), the replayed states' positions as before it. the prompt slices
    since the last wait (not waited for: the run-out can be one of theirs) run again first (ts5 Nov 15: such a run-out ended 4 streams)
    """
    retries, kbase, slices = 0, dict(getattr(self, "_seq_kbase", {})), list(self._p2p_slices)
    while True:
      try: return super().seq_step(seqs, *args, **kwargs)
      except P2PRunOut as e:
        if (retries:=retries+1) > getenv("P2P_RETRY", 2): raise
        print(f"seq_step: {e}: the step of {len(seqs)} sequences again ({retries}), {len(slices)} unwaited slices first", flush=True)
        self._p2p_redo(slices)
        if hasattr(self, "_seq_kbase"):
          self._seq_kbase.clear()
          self._seq_kbase.update(kbase)

  def _spec_steps(self, tokens:list[int], temperature:Tensor) -> Iterator[int]:
    """speculative decoding after a prefill: tokens[-1] is the sampled token after the prompt"""
    k, host, host_out = self.config.spec_drafts, self._spec_host, self._spec_host_out
    def launch(mtp_pos:int, j:int, par:int) -> bool:
      if mtp_pos + j + k + 1 >= self.max_context: return False
      host_out[:] = array.array('i', [-1] * (2*k+1))
      self._spec_call(mtp_pos, j, par, temperature)
      return True
    # the first step catches up on the last prompt token's position only (j=0), its hidden state is row 0 (parity 0)
    host[0], mtp_pos, j, par = tokens[-1], len(tokens) - 1, 0, 1
    launched = launch(mtp_pos, j, par)
    while launched:
      out = self._wait_host(lambda: min(o:=host_out.tolist()) >= 0 and o or -1, "a step of speculative decoding")
      sampled, drafts = out[:k+1], out[k+1:]
      j, pos = next((i for i in range(k) if drafts[i] != sampled[i]), k), mtp_pos + j
      host[0:k+1] = array.array('i', sampled)
      self.spec_stats[0] += 1
      self.spec_stats[1] += j+1
      # run ahead: the next step is launched before the tokens are taken (the states it writes don't overwrite the ones of their positions)
      launched, row, mtp_pos, par = launch(pos + 1, j, 1-par), par*(k+1), pos + 1, 1-par
      for i in range(j+1):
        tokens.append(sampled[i])
        # the caches hold every token but the last: its hidden state is the verified row i
        self._cached_tokens, self._spec_row = tokens[:-1], row + i
        yield sampled[i]

  # ******** batched decoding in slots (glm5next's schedule() path, for models with recurrent blocks): none here ********
  def batch_slots(self) -> int:
    """no slots: GLM-5.3 (MLA only) decodes many sequences on the sequences' path (Transformer._seq_init, serve.schedule_seqs)"""
    return 0

  # ** GLM_SEQ on the 14-GPU DSA target: reuse the generic driver; override only DSA cache/state ownership.
  REGION_ALIGN = SEQ_MOVE = REGION_ALIGN
  @property
  def _spec_dim(self) -> int: return self.config.dim
  def _derive_weights(self):
    """DSA's weight derivations are already scheduled by its own loader/_finish_load."""
  def seqs_supported(self) -> bool:
    devs = self.token_embd.weight.device
    return super().seqs_supported() and isinstance(devs, tuple) and \
      self.REGION_ALIGN % (SEQ_BLK*len(devs)) == 0

  def _seq_init(self):
    if hasattr(self, "seq_spec_h"): return
    for d in to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device)): Device[d].allocator.free_cache()
    self._init_states()
    k, n, devs, m = self.config.spec_drafts, self.seq_slots() + 1, cast(tuple[str, ...], self.token_embd.weight.device), max(self.SEQ_SIZES)
    for b in self.blk + [self.mtp]:
      if isinstance(b, MLATransformerBlock): b._init_seq_slots(n)
    self.seq_spec_h = glm_seq_base._zeros(n*2*(k+1), self.config.dim, device=devs)
    self.seq_ckpt_h = glm_seq_base._zeros(n, self.config.dim, device=devs)
    self.seq_ckpt_states:list[list[Tensor]] = []
    self.seq_ckpt_rings = [(b.indexer, _slot_pool(n, b.indexer.ring)[0]) for b in self.blk + [self.mtp]
                           if isinstance(b, MLATransformerBlock) and b.indexer is not None and b.indexer.idx.kpool > 1]
    self._spec_init()
    if not hasattr(self, "_token_buf"): self._token_host, self._token_buf = _host_ints(devs, self._token_rows())
    self._seq_host_in, self._seq_in = _host_ints(devs, m*(k+1))
    self._seq_m = m
    # (the rows of the shared seq steps, glm5next's: TEMP_W floats a sequence (temperature, top_p, top_k), a mixed step's prompt's, the seed)
    tmv, self._seq_temp = _host_buffer(devs, 4*glm_seq_base.TEMP_W*(m + 1) + 4, dtypes.float32)
    self._seq_host_temp = tmv.cast('f')
    tmv, self._seq_pf_temp = _host_buffer(devs, 4*glm_seq_base.TEMP_W, dtypes.float32)
    self._seq_pf_host_temp = tmv.cast('f')
    self._seq_host_out, self._seq_out = _host_ints(devs[0], m*(2*k+1) + 1)
    # (SEQ_COPY_SCRATCH=1: the slot copy / checkpoint / region move jits' intermediates in the jits' shared arenas, not their own (one runs at a
    # time): a move's cache rows go through a contiguous copy, ~520 MB a card of the moves' own arenas at C9's 268,800 rows)
    sc:dict|None = self._scratch if getenv("SEQ_COPY_SCRATCH", 0) else None
    self._seq_copy_jits = {sv: TinyJit(functools.partial(self._seq_copy, sv), capture_first=True, scratch=sc) for sv in (True, False)}
    self._seq_ckpt_jits = {sv: TinyJit(functools.partial(self._seq_ckpt_copy, sv), capture_first=True, scratch=sc) for sv in (True, False)}
    self._seq_mat_jit = None  # no KDA recurrent state or replay on the DSA target
    self._seq_kbase:dict[int, int] = {}
    self._move_calls = self._chunk_calls((self.seq_rows() + self.REGION_ALIGN) // self.SEQ_MOVE - 1)
    self._seq_move_jits = {chunks: TinyJit(functools.partial(self._seq_move_copy, chunks), capture_first=True, scratch=sc)
                           for chunks in self._move_calls}
    self._seq_stash_init()
    todo = [self.seq_spec_h, self.seq_ckpt_h] + [b.indexer.pool_ring for b in self.blk + [self.mtp]
             if isinstance(b, MLATransformerBlock) and b.indexer is not None and b.indexer.idx.kpool > 1] + \
           [ring for _, ring in self.seq_ckpt_rings]
    glm_seq_base._fill_zero(*todo)
    # (the MLA caches' peer views made (allocated) before jits are captured in forked processes, see seq_gather)
    for b in self.blk + [self.mtp]:
      if isinstance(b, MLATransformerBlock) and b.seq_sharded: seq_views(b.cache_s)

  def _seq_copy(self, save:bool, slot:UOp, pm:UOp, ps:UOp, pc:UOp, hrow:UOp, keep:UOp):
    """The DSA sequence slot has hidden MTP rows; pooled-indexer rings too. Direct DSA keys stay in the region."""
    def rows(t:Tensor, at:UOp|int, n:UOp|int=1) -> Tensor:
      return t.shrink(((at, at+n),) + (None,)*(t.ndim-1))
    H = 2*(self.config.spec_drafts+1)
    pairs = [(rows(self.seq_spec_h, slot*H + hrow), rows(self._spec_h, 0))]
    for b in self.blk + [self.mtp]:
      if isinstance(b, MLATransformerBlock) and (idx:=b.indexer) is not None and idx.idx.kpool > 1:
        pairs.append((rows(idx.pool_ring, slot*int(idx.ring.shape[0]), int(idx.ring.shape[0])), idx.ring))
    if save: Tensor.realize(*[d.assign(s.cast(d.dtype)) for d, s in pairs])
    else:
      (h, hs), rest = pairs[0], pairs[1:]
      Tensor.realize(hs.assign(h.contiguous() * Tensor(keep).cast(dtypes.float)), *[s.assign(d.cast(s.dtype)) for d, s in rest])

  def _seq_move_copy(self, chunks:int, src:UOp, dst:UOp):
    n, copies = self.SEQ_MOVE * chunks, []
    def mv(t:Tensor, s:UOp, d:UOp, m:int, axis:int=0) -> Tensor:
      return t.shrink(tuple((d, d+m) if i == axis else None for i in range(t.ndim))).assign(
        t.shrink(tuple((s, s+m) if i == axis else None for i in range(t.ndim))).contiguous())
    for b in self.blk + [self.mtp]:
      if not isinstance(b, MLATransformerBlock) or not b.seq_sharded: continue
      nd = len(to_tuple(b.cache_s.device))
      copies.append(mv(b.slot_rows_view(b.cache_s), src // nd, dst // nd, n // nd))
      if (idx:=b.indexer) is not None:
        div = 1 if idx.idx.kpool == 1 else idx.idx.kpool
        if idx.idx.kpool == 1 and idx.shard: copies.append(mv(b.slot_rows_view(idx.cache_dk), src // nd, dst // nd, n // nd))
        else: copies.append(mv(idx.cache_dk if div == 1 else idx.cache_pk, src // div, dst // div, n // div))
    Tensor.realize(*copies)

  def _seq_stash_init(self):
    C, gb = self.SEQ_MOVE, getenv("SEQ_STASH_GB", 4.0)
    devs = to_tuple(self.token_embd.weight.device)
    nd = len(devs)
    assert nd and C % nd == 0, f"SEQ_MOVE={C} must be a multiple of the {nd} devices"
    def per_row(t:Tensor) -> int:
      return t.nbytes() // int(t.shape[0]) // (len(t.device) if isinstance(t.device, tuple) and t.uop.axis is not None else 1)
    kb = [b for b in self.blk + [self.mtp] if isinstance(b, MLATransformerBlock)]
    def index(b:MLATransformerBlock) -> Indexer:
      assert b.indexer is not None and (not b.indexer.shard or b.indexer.idx.kpool == 1), "GLM_SEQ: split pooled keys (SEQ_IDX_SHARD)"
      return b.indexer
    # (the keys as rows: split DSA keys a device's rows (as the latents), else the rows of every device)
    def keys(b:MLATransformerBlock) -> Tensor:
      return (b.slot_rows_view(index(b).cache_dk) if index(b).shard else index(b).cache_dk) if index(b).idx.kpool == 1 else index(b).cache_pk
    def kdiv(b:MLATransformerBlock) -> int: return nd if index(b).idx.kpool == 1 and index(b).shard else index(b).idx.kpool
    # (a layer sharing another's indexer (indexer_shared) has no keys of its own: its latents only)
    chunk = C * sum(per_row(b.cache_s) + (per_row(index(b).cache_dk if index(b).idx.kpool == 1 else index(b).cache_pk) //
                    (1 if index(b).idx.kpool == 1 and index(b).shard else index(b).idx.kpool) if b.indexer is not None else 0) for b in kb)
    ring = [r for _, r in self.seq_ckpt_rings]
    entry = per_row(self.seq_spec_h) + per_row(self.seq_ckpt_h) + sum(per_row(index(b).pool_ring) + per_row(r)
              for b, r in zip([b for b in kb if b.indexer is not None and b.indexer.idx.kpool > 1], ring))
    self.stash_entries = getenv("SEQ_STASH_ENTRIES", 32) if gb > 0 and chunk > 0 else 0
    self.stash_chunks = max(0, int(gb * 1e9 - self.stash_entries * entry) // chunk) if self.stash_entries else 0
    if not self.stash_chunks: self.stash_entries = 0
    self.stash_chunk_nbytes, self.stash_entry_nbytes = chunk, entry
    if not self.stash_entries: return
    self._stash_kv = []
    for b in kb:
      latent = glm_seq_base._host_like([(b.slot_rows_view(b.cache_s), 0, self.stash_chunks * C // nd)])[0]
      pools = glm_seq_base._host_like([(keys(b), 0, self.stash_chunks * C // kdiv(b))])[0] if b.indexer is not None else None
      self._stash_kv.append((latent, pools))
    states = [self.seq_spec_h, self.seq_ckpt_h]
    self._stash_rings = list(self.seq_ckpt_rings)
    extra:list[Tensor] = []
    for idx, ck in self._stash_rings:
      for t in (idx.pool_ring, ck): extra.append(glm_seq_base._host_like([(t, 0, self.stash_entries * int(idx.ring.shape[0]))])[0])
    self._stash_state = glm_seq_base._host_like([(t, 0, self.stash_entries) for t in states]) + extra
    self._stash_state_count = len(states)
    self._stash_calls = self._chunk_calls(min((self.seq_rows() + self.REGION_ALIGN) // C - 1, self.stash_chunks))
    self._stash_kv_jits = {(sv, n): TinyJit(functools.partial(self._stash_kv_copy, sv, n), capture_first=True)
                           for sv in (True, False) for n in self._stash_calls}
    self._stash_state_jits = {sv: TinyJit(functools.partial(self._stash_state_copy, sv), capture_first=True) for sv in (True, False)}

  def _stash_kv_copy(self, save:bool, chunks:int, row:UOp, chunk:UOp):
    C, pairs, saves = self.SEQ_MOVE, [], []
    nd = len(to_tuple(self.token_embd.weight.device))
    def rows(t:Tensor, at:UOp, n:int) -> Tensor: return t.shrink(((at, at+n),) + (None,)*(t.ndim-1))
    for b, st in zip([b for b in self.blk + [self.mtp] if isinstance(b, MLATransformerBlock)], self._stash_kv):
      # Host rows come from the stash chunk; device rows come from the independently allocated region.
      items = [(rows(st[0], chunk*(C//nd), C*chunks//nd), rows(b.slot_rows_view(b.cache_s), row//nd, C*chunks//nd))]
      if (idx:=b.indexer) is not None:
        kp = idx.idx.kpool
        # (split DSA keys: a device's rows, as the latents')
        split = kp == 1 and bool(idx.shard)
        kd, kt = (nd, b.slot_rows_view(idx.cache_dk)) if split else (kp, idx.cache_dk if kp == 1 else idx.cache_pk)
        items.append((rows(cast(Tensor, st[1]), chunk*(C//kd), C*chunks//kd), rows(kt, row//kd, C*chunks//kd)))
      for dst, src in items:
        if save: saves.append(dst.assign(src))
        else: pairs.append((src, dst))
    Tensor.realize(*saves, *[dst.assign(src.contiguous()) for dst, src in pairs])

  def _stash_state_copy(self, save:bool, slot:UOp, entry:UOp, ps:UOp, pc:UOp, hrow:UOp):
    def rows(t:Tensor, at:UOp, n:UOp|int=1) -> Tensor: return t.shrink(((at, at+n),) + (None,)*(t.ndim-1))
    H = 2*(self.config.spec_drafts+1)
    slots = [rows(self.seq_spec_h, slot*H + hrow), rows(self.seq_ckpt_h, slot)]
    pairs = [(rows(st, entry), x) for st, x in zip(self._stash_state[:self._stash_state_count], slots)]
    for gi, (idx, ck) in enumerate(self._stash_rings):
      curr, R = self._stash_state_count + 2*gi, int(idx.ring.shape[0])
      pairs += [(rows(self._stash_state[curr], entry*R, R), rows(idx.pool_ring, slot*R, R)),
                (rows(self._stash_state[curr+1], entry*R, R), rows(ck, slot*R, R))]
    Tensor.realize(*[(d.assign(x) if save else x.assign(d)) for d, x in pairs])

  def _forward_seqs_glm(self, vs:tuple[UOp, ...]) -> Tensor:
    _idx_mask[0] = None  # DSA shared attention layers must never see another step's lists
    # (SEQ_I8: the rows' activations of the dense layers int8 (as the decode kernels' of a sequence alone), the int8 tensor cores. glm5next's
    # sets amd_glm's flag, this model's kernels read amd_glm_dsa's. the f16 rows of the row parallel layers are scaled by every row's max of
    # all the devices: a ring of copies a layer)
    prev, Q8_ROWS_I8[0] = Q8_ROWS_I8[0], bool((0))
    try: return super()._forward_seqs_glm(vs)
    finally:
      q8_quantize_cache_clear()
      Q8_ROWS_I8[0] = prev

  def _freqs_host(self, devs:tuple[str, ...]):
    """
    FREQS_HOST (default off): the rope table of the attention blocks (max_context x rope_dim float32: 268 MB a device at 1M, the same on
    every device) in ONE host buffer every device reads directly, not a copy in every device's memory. its values as before: computed on
    the first device by the same kernels, copied out (a chunk reads its T rows over the bus: ~0.7 MB a layer at 2688 rows). the table is
    made by the function under the cache (__wrapped__): the cached one stayed realized on the first device for the life of the process
    """
    blocks:list[Any] = [b for b in self.blk + ([self.mtp] if hasattr(self, "mtp") else []) if isinstance(getattr(b, "freqs_cis", None), Tensor)]
    if not blocks or blocks[0].freqs_cis.uop.is_realized: return
    c = self.config
    table = precompute_freqs_cis.__wrapped__(c.rope_dim, c.max_context, c.rope_theta, device=devs[0], yarn=c.yarn).realize()
    mv, host = _host_buffer(devs, table.nbytes(), table.dtype)
    mv[:] = cast(Buffer, table.uop.buffer).as_memoryview()
    del table
    host = host.reshape(c.max_context, -1)
    for b in blocks: b.freqs_cis = host

  def _init_states(self):
    """the states and caches of the blocks, realized (the jits capture on their first call: not the one time setup)"""
    if getattr(self, "_states_ready", False): return
    devs = self.token_embd.weight.device
    if devs is not None and amd_custom_kernels_supported(devs): tables_ready(devs)
    x = Tensor.empty(1, 1, self.config.dim, device=devs)
    holders:list = []
    for b in self.blk + ([self.mtp] if hasattr(self, "mtp") else []):
      b._init_state(x)
      holders += [b] + ([idx] if (idx:=getattr(b, "indexer", None)) is not None else [])
    if getenv("FREQS_HOST", 0) and isinstance(devs, tuple): self._freqs_host(devs)
    def lazy(v) -> list[Tensor]:
      if isinstance(v, Tensor): return [v] if not v.uop.is_realized else []
      return [t for x in v for t in lazy(x)] if isinstance(v, (list, tuple)) else []
    # (and the parameters of a model initialized but not loaded. a loaded one's are: the quantized weights' are their decoding, not used)
    todo = [t for h in holders for v in vars(h).values() for t in lazy(v)] + ([] if self._loaded else lazy(nn.state.get_parameters(self)))
    # with the weights still to read: their memory now, their values after
    if self._pending_load: _realize_later(todo, self._pending_load, self._scratch)
    else: Tensor.realize(*todo)
    # (a sequence sharded cache: the views of the other devices' parts (peer mappings) made here, not in the processes that capture the jits)
    for b in self.blk + ([self.mtp] if hasattr(self, "mtp") else []):
      if hasattr(b, "cache_s"): seq_views(b.cache_s)
    self._states_ready = True

  def __call__(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor, base:int|UOp=0, part:int|None=None) -> Tensor:
    # a view of the token buffer is already realized: no schedule for the jit input
    if (buf:=getattr(self, "_token_buf", None)) is None or tokens.uop.base is not buf.uop.base: tokens = tokens.contiguous()
    key = int(T.vmax) if isinstance(T:=tokens.shape[1], UOp) else T
    if isinstance(base, UOp):
      jits = self.__dict__.setdefault("jits_seq", {})
      if (jit:=jits.get(key)) is None:
        self._init_states()
        jit = jits[key] = TinyJit(self._forward_effects, capture_first=True, scratch=self._scratch)
        setattr(jit, "warmup_name", f"prefill{key}/seq")
      return jit(tokens, start_pos, temperature, base)[0]
    if self._sp_tokens(tokens):
      jits = self._part_jits(key)
      for k, j in enumerate(jits[:-1]):
        j(tokens, start_pos, temperature)

      return jits[-1](tokens, start_pos, temperature)[0]
    if (jit:=self.jits.get(key)) is None:
      # the states and caches are set up before: a jit captures on its first call
      self._init_states()
      jit = self.jits[key] = TinyJit(self._forward_effects, capture_first=True, scratch=self._scratch)
      setattr(jit, "warmup_name", "step/decode" if key == 1 else f"prefill{key}")
    return jit(tokens, start_pos, temperature)[0]

  def _new_jit(self, rows:int) -> Any:
    return TinyJit(self._forward_effects, capture_first=True, scratch=self._scratch)
  def _forward_effects(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor, base:int|UOp=0, part:int|None=None) -> tuple[Tensor, ...]:
    # forward and the tensors realized with its output for their effects (e.g. the caches of the MTP layer)
    self._effects = []
    old_base, _seq_base[0] = _seq_base[0], base
    try: out = self.forward(tokens, start_pos, temperature)
    finally: _seq_base[0] = old_base
    effects, self._effects = self._effects, None
    return (out, *effects)

  @staticmethod
  def from_gguf(gguf:Tensor|str|pathlib.Path, max_context:int|None=None,
                realize=False, shard:int=1, defer_load:bool=bool(getenv("LOAD_DEFER", 0))) -> tuple[Transformer, dict]:
    """defer_load (LOAD_DEFER, default off): the weights are read later (warmup compiles meanwhile), before the first call"""
    # TODO: remove the need for copy to default device
    st = time.perf_counter()
    def lap(what:str):
      if getenv("DEBUG_LOAD", 1): print(f"from_gguf: {what} at {time.perf_counter()-st:.2f}s", flush=True)
    kv, entries = gguf_parse(gguf.to(None).realize() if isinstance(gguf, Tensor) else gguf)
    lap("parsed")
    # (the jit cache: of this model's files and settings)
    if not isinstance(gguf, Tensor) and hasattr(jit_mod, 'jit_cache_key'):
      import os
      def disk(t:Tensor) -> str:
        try: return str(t.uop.buffer.device)
        except Exception: return ""
      files = sorted({dev for d, _, _ in entries.values() if (dev:=disk(d)).startswith("DISK:")})
      jit_mod.jit_cache_key[0] = repr((files, [(os.path.getsize(f[5:]), int(os.path.getmtime(f[5:]))) for f in files], max_context, shard,
                                       defer_load))
    # deferred: the files are read ahead from now (the disk works while the model is made, the loader finds them in memory)
    if defer_load: _read_ahead(entries)
    arch = kv['general.architecture']
    assert arch == 'glm-dsa', f"glm_dsa loads GLM-5.3 (glm-dsa) only, got {arch}"
    n_heads, n_kv_heads = kv[f'{arch}.attention.head_count'], kv[f'{arch}.attention.head_count_kv']
    assert shard >= 1, f"shard must be at least 1, got {shard}"
    # the next token prediction (MTP) layers: the first is loaded with MTP=1 (model.mtp), the others aren't used
    num_blocks = kv[f'{arch}.block_count'] - kv.get(f'{arch}.nextn_predict_layers', 0)
    mtp = kv.get(f"{arch}.nextn_predict_layers", 0) > 0
    # (NUM_BLOCKS: only the first blocks (debugging: a small model of the same kernels), no MTP layer)
    entries = {k:v for k,v in entries.items() if not ((m:=re.match(r"blk\.(\d+)\.", k)) and int(m.group(1)) >= num_blocks + mtp)}
    # (GLM-5: the indexers of the first 3 layers and every 4th one after, the layers between use the last one's picks (the MTP layer has its own))
    freq, skip = (4), (3)
    indexer_shared:tuple[bool, ...] = tuple(not (i < skip or (i - skip + 1) % freq == 0) for i in range(num_blocks))
    entries = {k:v for k,v in entries.items() if not ((m:=re.match(r"blk\.(\d+)\.indexer\.", k)) and int(m.group(1)) < num_blocks and
                                                       indexer_shared[int(m.group(1))])}
    shard_map:dict[str, int] = {}
    units_map:dict[str, tuple[int, list[list[int]]]] = {}
    pads:dict[str, int] = {}
    padded_heads, sizes, expert_slices, expert_units = n_heads, {}, 0, 0
    if shard > 1:
      # MLA and kda attention split all their tensors by the n_heads heads: those are zero padded to a multiple of the devices
      pad_heads = bool(kv.get(f'{arch}.attention.kv_lora_rank'))
      if not pad_heads: assert n_kv_heads % shard == 0, f"tensor parallel needs the attention heads to split over {shard} devices"
      padded_heads = round_up(n_heads, shard) if pad_heads else n_heads
      # shard heads, FFNs and routed experts while replicating latent projections, KV cache and the small kda projections
      head_rules = {**{w: 0 for w in ('attn_q.weight', 'attn_k.weight', 'attn_v.weight', 'attn_q_b.weight', 'attn_k_b.weight', 'attn_v_b.weight')},
                    'attn_output.weight': 1}
      ffn_rules = {**{w: 0 for w in ('ffn_gate.weight', 'ffn_up.weight', 'ffn_gate_shexp.weight', 'ffn_up_shexp.weight', 'output.weight')},
                   **{w: 1 for w in ('ffn_down.weight', 'ffn_down_shexp.weight')}}
      if len(kv['tokenizer.ggml.tokens']) % shard == 0: ffn_rules['token_embd.weight'] = 0
      # experts are split by their hidden dim, or by experts when it doesn't split (expert parallel)
      exp_hidden, exp_count = kv.get(f'{arch}.expert_feed_forward_length', 0), kv.get(f'{arch}.expert_count', 0)
      down_block = max([_GGML_QUANT.get(t, (1,))[0] for k,(_,_,t) in entries.items() if k.endswith('ffn_down_exps.weight')], default=1)
      # (EXPERT_SLICE_MULT: an expert's slices of that many quant blocks of the down projection: fewer, bigger slices)
      down_block *= (1)
      exp_names = ('ffn_gate_exps.weight', 'ffn_up_exps.weight', 'ffn_down_exps.weight')
      if exp_hidden % shard == 0 and (exp_hidden // shard) % down_block == 0:
        ffn_rules.update({'ffn_gate_exps.weight': 1, 'ffn_up_exps.weight': 1, 'ffn_down_exps.weight': 2})
      elif exp_count and exp_hidden % down_block == 0 and (1) and \
           (slicing:=expert_slicing(exp_count, exp_hidden // down_block, shard)) is not None:
        # slices of the experts' hidden dim (whole quant blocks of the down projection) spread over the devices: the work of a token is balanced
        expert_slices, expert_units = exp_hidden // down_block, shard * len(slicing[0])
        # (the padding units (-1) are loaded as the first one: no selection maps to them)
        units_map = {name: (expert_slices, [[max(u, 0) for u in us] for us in slicing]) for name in entries
                     if re.sub(r"^blk\.\d+\.", "", name) in exp_names}
      elif exp_count % shard == 0: ffn_rules.update({w: 0 for w in exp_names})
      # (a hidden dim's quant block: the largest of every layer's down projection of that kind (UD quants mix types over the layers), the
      # layers' hidden dims padded alike: one size in the config)
      kind_block: dict[str, int] = {}
      for name, (_, _, typ) in entries.items():
        if (k:=re.sub(r"^blk\.\d+\.", "", name)) in ('ffn_down.weight', 'ffn_down_shexp.weight'):
          kind_block[k] = max(kind_block.get(k, 1), _GGML_QUANT.get(typ, (1,))[0])
      for name, (_, shape, typ) in entries.items():
        if (k:=re.sub(r"^blk\.\d+\.", "", name)) in head_rules:
          axis = shard_map[name] = head_rules[k]
          sizes[name] = shape[axis] // n_heads * padded_heads
        elif k in ffn_rules:
          axis = shard_map[name] = ffn_rules[k]
          # FFN hidden dims pad to whole quant blocks of the down projection on every device
          down = entries.get(re.sub(r"ffn_(gate|up)(_shexp)?\.", r"ffn_down\2.", name), entries[name])
          # (the output rows: whole tiles of 32 on every device, the tensor cores take a few tokens' logits)
          unit = 32 if k == 'output.weight' else 1 if '_exps' in k else \
            kind_block.get(re.sub(r"ffn_(gate|up)(_shexp)?\.", r"ffn_down\2.", k), _GGML_QUANT.get(down[2], (1,))[0])
          sizes[name] = round_up(shape[axis], shard*unit)
      pads = {k: sizes[k] for k in sizes if sizes[k] != entries[k][1][shard_map[k]]}
    devices = tuple(Device.canonicalize(f'{Device.DEFAULT}:{i}') for i in range(shard))
    lap("planned")
    _open_devices(devices)
    # (the compile workers started and warmed up while the weights are read)
    warm_worker_pool(devices[0])
    lap("devices open")
    # gate and up experts of the same format load as one (fuse_gate_up)
    fuse = {g: u for g in units_map if g.endswith("ffn_gate_exps.weight") and (u:=g.replace("gate", "up")) in units_map and
            entries[g][2] == entries[u][2] and (1)}
    # a token embedding copied to every device (a vocab that doesn't split) stays in host memory: a call looks up a few of its rows
    host_embd = None
    if shard > 1 and 'token_embd.weight' not in shard_map and 'output.weight' in entries and amd_custom_kernels_supported(devices) and \
       (1):
      data, shape, typ = entries.pop('token_embd.weight')
      mv, raw = _host_buffer(devices, int(data.nbytes()))
      mv[:] = data.to("CPU").data()
      host_embd = ggml_data_to_tensor(raw, prod(shape), typ).reshape(shape)
    pending:list[Callable[[], None]] = []
    state_dict = gguf_shard(entries, devices, shard_map, units_map, pads if shard > 1 else {}, fuse, pending if defer_load else None)
    lap("sharded")
    if host_embd is not None: state_dict['token_embd.weight'] = host_embd

    # all state items should be float16, not float32
    state_dict = {k:v.cast('float16') if (1) else v for k,v in state_dict.items()}

    # some models like Llama 3.2 don't have an output.weight, they just tie to the token_embd.weight
    if 'output.weight' not in state_dict: state_dict['output.weight'] = state_dict['token_embd.weight']

    assert f'{arch}.hyper_connection.count' not in kv, "hyper-connections: not supported (GLM-5.3 has none)"

    if f'{arch}.attention.indexer.kpool' in kv:
      state_dict = {k.replace('.indexer_compressor_', '.indexer.compressor_'):v for k,v in state_dict.items()}
    indexer = IndexerConfig(kv[f'{arch}.attention.indexer.head_count'], kv[f'{arch}.attention.indexer.key_length'],
                            kv[f'{arch}.attention.indexer.top_k'], 1, 1e-6, rope_dim=kv[f'{arch}.rope.dimension_count'], rope_interleave=True)
    if mtp: state_dict = {re.sub(rf"^blk\.{num_blocks}\.(nextn\.)?", "mtp.", k):v for k,v in state_dict.items()}

    max_context = min(max_context, kv[f'{arch}.context_length']) if max_context is not None else kv[f'{arch}.context_length']

    kv_lora_rank = kv.get(f'{arch}.attention.kv_lora_rank', 0)
    head_dim = kv.get(f'{arch}.attention.key_length_mla', kv.get(f'{arch}.attention.key_length', kv[f'{arch}.embedding_length'] // n_heads))
    rope_dim = kv.get(f'{arch}.rope.dimension_count', head_dim)
    yarn = YaRNConfig(factor=kv[f'{arch}.rope.scaling.factor'],
                      orig_ctx_len=kv.get(f'{arch}.rope.scaling.original_context_length', kv[f'{arch}.context_length']),
                      beta_fast=kv.get(f'{arch}.rope.scaling.yarn_beta_fast', 32.0),
                      beta_slow=kv.get(f'{arch}.rope.scaling.yarn_beta_slow', 1.0)) if kv.get(f'{arch}.rope.scaling.type') == 'yarn' else None

    # (SPEC_DRAFTS_BY_SIZE=1:2,4:3,...: a step of m sequences drafts that many, the sizes not listed SPEC_DRAFTS)
    spec = getenv("SPEC_DRAFTS", 3)
    by_size = glm_seq_base.spec_by_size(spec, glm_seq_base.Transformer.SEQ_SIZES)
    config = TransformerConfig(
      num_blocks=num_blocks, dim=kv[f'{arch}.embedding_length'],
      hidden_dim=kv.get(f'{arch}.expert_feed_forward_length', sizes.get('blk.0.ffn_gate.weight', kv.get(f'{arch}.feed_forward_length', 0))),
      n_heads=padded_heads, real_heads=n_heads, n_kv_heads=n_kv_heads, norm_eps=kv[f'{arch}.attention.layer_norm_rms_epsilon'],
      vocab_size=len(kv['tokenizer.ggml.tokens']), padded_vocab_size=int(state_dict['output.weight'].shape[0]),
      head_dim=head_dim,
      rope_theta=kv.get(f'{arch}.rope.freq_base', 10000.0), rope_dim=rope_dim, yarn=yarn,
      v_head_dim=kv.get(f'{arch}.attention.value_length_mla', kv.get(f'{arch}.attention.value_length', head_dim)),
      max_context=max_context,
      qk_norm=int(state_dict['blk.0.attn_q_norm.weight'].shape[0]) if 'blk.0.attn_q_norm.weight' in state_dict else 0,
      num_experts=kv.get(f'{arch}.expert_count', 0), num_experts_per_tok=kv.get(f'{arch}.expert_used_count', 0),
      norm_topk_prob=kv.get(f'{arch}.expert_weights_norm', False),
      expert_gating_func=ExpertGating(kv.get(f'{arch}.expert_gating_func', ExpertGating.SOFTMAX)),
      kv_lora_rank=kv_lora_rank, q_lora_rank=kv.get(f'{arch}.attention.q_lora_rank', 0),
      leading_dense_blocks=kv.get(f'{arch}.leading_dense_block_count', 0),
      shared_expert_dim=sizes.get(f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.ffn_gate_shexp.weight", kv.get(
        f'{arch}.expert_shared_feed_forward_length',
        kv.get(f'{arch}.expert_shared_count', 0) * kv.get(f'{arch}.expert_feed_forward_length', 0))),
      shared_expert_gate=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.ffn_gate_inp_shexp.weight" in state_dict,
      dense_hidden_dim=sizes.get('blk.0.ffn_gate.weight', kv.get(f'{arch}.feed_forward_length', 0))
        if kv.get(f'{arch}.leading_dense_block_count', 0) else 0,
      routed_scaling_factor=kv.get(f'{arch}.expert_weights_scale', 1.0),
      qkv_bias='blk.0.attn_q.bias' in state_dict,
      expert_bias=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.exp_probs_b.bias" in state_dict, expert_slices=expert_slices,
      expert_proj_bias='blk.0.ffn_gate_exps.bias' in state_dict, attn_output_bias='blk.0.attn_output.bias' in state_dict,
      swiglu_limit=kv.get(f'{arch}.swiglu_clamp_exp', [None])[0],
      attn_sinks='blk.0.attn_sinks.weight' in state_dict,
      sliding_window=kv.get(f'{arch}.attention.sliding_window', 0),
      sliding_window_pattern=kv.get(f'{arch}.attention.sliding_window_pattern', 0), indexer=indexer, mtp=mtp,
      spec_drafts=max([spec, *(d for _, d in by_size)]) if mtp else 0, spec_by_size=by_size if mtp else (), rope_interleave=True,
      indexer_shared=indexer_shared,
      expert_units=expert_units,
      cache_rows=round_up(getenv("SEQ_ROWS", 0), REGION_ALIGN) if (1) else 0)
    lap("state dict")
    with Linear.empty_init(): model = Transformer(config)  # (the weights are loaded)
    lap("model made")
    # gate experts loaded with the up rows (fused): one weight of twice the rows
    for name, b in [(f"blk.{i}", b) for i, b in enumerate(model.blk)] + ([("mtp", model.mtp)] if mtp else []):
      if (g:=state_dict.get(f"{name}.ffn_gate_exps.weight")) is not None and f"{name}.ffn_up_exps.weight" not in state_dict and \
         hasattr(b, "ffn_up_exps"):
        e = ExpertWeights(int(g.shape[0]), int(g.shape[2]), int(g.shape[1]))
        e.fused, e.slices, e.experts = True, b.ffn_gate_exps.slices, b.ffn_gate_exps.experts
        b.ffn_gate_exps = e
        del b.ffn_up_exps
    for p in (nn.state.get_parameters(model) if shard > 1 else []): p.to_(devices)
    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)  # NOTE: rope_freqs.weight (32,) is unused
    model._pending_load, model._loaded = pending, True
    lap("state dict loaded")
    # NOTE: without this contiguous, it unpacks the weights from the model every time. we shouldn't need this, but for now it's faster
    if realize:
      for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
      Tensor.realize(*params)
    else: model.prepare()
    lap("prepared")
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
    for m in modules([self.blk, self.output] + ([self.mtp] if hasattr(self, "mtp") else []), set()):
      if isinstance(m, Linear) and m.ggml_type is None and m.use_custom_quant and amd_custom_kernels_supported(m.weight.device):
        m.set_quantized(m.weight)
      if isinstance(m, ExpertWeights): m.set_quantized()
      if m.ggml_type is not None: packed.add(id(m.weight))
    # packed gate and up experts of the same format are one weight: their rows interleaved
    for b in self.blk + ([self.mtp] if hasattr(self, "mtp") else []):
      if (g:=getattr(b, 'ffn_gate_exps', None)) is not None and hasattr(b, 'ffn_up_exps') and g.ggml_type is not None and \
         g.ggml_type == b.ffn_up_exps.ggml_type and \
         amd_custom_kernels_supported(g.weight.device) and (1):
        packed.discard(id(b.ffn_up_exps.weight))
        packed.discard(id(g.weight))
        g.fuse_up(b.ffn_up_exps, b.config.hidden_dim // (b.config.expert_slices or 1))
        del b.ffn_up_exps
        packed.add(id(g.weight))
    small = [p for p in nn.state.get_parameters(self) if id(p) not in packed and p is not self.token_embd.weight and not p.uop.is_realized]
    for p in small: p.replace(p.contiguous())
    tables = dedup([t for m in modules([self.blk] + ([self.mtp] if hasattr(self, "mtp") else []), set()) if isinstance(m, ExpertWeights)
                    for t in (getattr(m, "first", None), getattr(m, "slice_table", None)) if t is not None and not t.uop.is_realized])
    # with the weights still to read: the small ones' memory now, their values after
    if self._pending_load: _realize_later(small, self._pending_load, self._scratch)
    if (todo:=[*tables, *([] if self._pending_load else small)]): Tensor.realize(*todo)
    self._prepare_dproj()

  def _prepare_dproj(self):
    """the split copies of the row projections of decoding (MLATransformerBlock._dec_proj): their memory now, their values after the load"""
    self._dproj_fill:list[tuple[Tensor, Tensor]] = []
    devs = self.token_embd.weight.device
    if not isinstance(devs, tuple) or not (1): return
    N = len(devs)
    for b in self.blk + ([self.mtp] if hasattr(self, "mtp") else []):
      maybe = [getattr(b, "attn_q_a", None), getattr(b, "attn_kv_a_mqa", None)] + \
              ([ind.attn_k] if (ind:=getattr(b, "indexer", None)) is not None and (1) else [])
      lins = [l for l in maybe if isinstance(l, Linear) and l.ggml_type == Q8_0 and l.shard_axis is None]
      if len(lins) != len(maybe) or any(l.in_features != lins[0].in_features for l in lins): continue
      hw, sizes = lins[0].in_features // 32 * 17, [l.out_features for l in lins]
      per = round_up(-(-sum(sizes) // N), 32)
      w = Tensor.empty(N*per*hw, dtype=dtypes.uint16, device=devs, axis=0).realize()
      with Linear.empty_init(): lin = Linear(lins[0].in_features, N*per, bias=False)
      lin.weight, lin.ggml_type, lin.shard_axis = w, Q8_0, 0
      setattr(b, "_dproj", (lin, sizes))
      src = Tensor.cat(*[l.weight.reshape(l.out_features, hw) for l in lins], Tensor.zeros(N*per - sum(sizes), hw, dtype=dtypes.uint16,
                       device=devs)).reshape(-1)
      self._dproj_fill.append((w, src))

  def _finish_load(self):
    """the work left of loading (deferred): the weights read, the values computed from them"""
    while self._pending_load: self._pending_load.pop(0)()
    # (the split copies of the row projections of decoding, _prepare_dproj)
    if (fill:=getattr(self, "_dproj_fill", [])):
      # (a block at a time: the copy kernel reads the replicated weights, no copy of them is made)
      for w, src in fill: w.assign(shard_rows(src, 0) if w.uop.axis == 0 else src).realize()
      self._dproj_fill = []
    # (SKINNY) the float weights of few outputs that decoding multiplies by (the routers, the indexers' head weights) stored transposed (the
    # outputs the fast axis), the Linear's a view of it: skinny_matmul reads every input's weights of all the outputs at once
    blocks = self.blk + ([self.mtp] if hasattr(self, "mtp") else [])
    lins = [l for b in blocks for l in (getattr(b, "ffn_gate_inp", None), getattr(getattr(b, "indexer", None), "proj", None))
            if isinstance(l, Linear) and l.ggml_type is None and l.weight.dtype in (dtypes.float32, dtypes.half) and l.weight.ndim == 2 and
            getattr(l, "bias", None) is None and not hasattr(l, "weight_t") and amd_custom_kernels_supported(l.weight.device) and (1)]
    if lins:
      Tensor.realize(*(wts:=[l.weight.T.contiguous() for l in dedup(lins)]))
      for l, wt in zip(dedup(lins), wts): l.weight_t, l.weight = wt, wt.T

  def glm_sp(self, x:Tensor) -> bool:
    """sequence parallel prefill of a model without hyper-connections (GLM_SP): a chunk of a symbolic size, whole tiles of 16 rows a device"""
    devs = x.device
    return isinstance(devs, tuple) and not isinstance(x.shape[1], int) and \
      int(x.max_shape[1]) % (16*len(devs)) == 0 and amd_custom_kernels_supported(devs) and bool((1))

  def prefill_chunks(self, chunk_size:int|None=None) -> tuple[int, ...]:
    """the prefill chunk sizes, ascending: the custom kernels compute chunks padded to a static size, a piece of a prompt runs the smallest
    one it fits (its own jit), large chunks are efficient. sequence parallel: whole tiles of rows on every device"""
    fast = amd_custom_kernels_supported(self.token_embd.weight.device)
    # (a context of 64K or less (many sequences batched): chunks of 2048, their memory the slots')
    # (MLA with DSA without hyper-connections (GLM-5): chunks of 512, as fast a token as bigger ones (the allreduces' bytes are the time))
    # (GLM_SP: 2688, whole tiles of 96 rows of the int8 gemm on 14 devices)
    glm = self.config.indexer is not None and self.config.indexer.kpool == 1
    default = (2688 if (1) else 512) if glm else 4096 if self.max_context > 65536 else 2048
    chunk_size = chunk_size or min(getenv("PREFILL_CHUNK", default if fast else 32), self.max_context)
    sizes = [chunk_size] + [min(chunk_size, n) for n in (((192), (512),
                                                          (chunk_size if glm else 2048),
                                                          4096) if fast else ())]
    # (PREFILL_SIZES: more sizes, less padding of a prompt's last piece (the turns of agents: a few hundred tokens), a jit each)
    if fast and (more:=getenv("PREFILL_SIZES", "")): sizes += [min(chunk_size, int(n)) for n in str(more).split(",")]
    if fast and (devs:=self.token_embd.weight.device) and isinstance(devs, tuple):
      # whole tiles of 16 rows on every device, the kernels of the whole chunk on tiles of 64
      tile = math.lcm(16*len(devs), 64)
      def tiled(n:int) -> int: return min(max(tile, round(n / tile) * tile), self.max_context // tile * tile or tile)
      sizes = [tiled(n) for n in sizes]
    return tuple(sorted(set(sizes)))

  def _seq_shared(self, read:bool=True):
    """the generic driver's shared resources, and this kernel module's collectives (its own mailboxes, made before the forked captures)"""
    super()._seq_shared(read)
    from tinygrad.llm.kernels.amd_glm_dsa import host_allreduce, p2p_allreduce, ring_mailbox, _device_index as dsa_device_index
    devs = cast(tuple[str, ...], self.token_embd.weight.device)
    if isinstance(devs, tuple):
      p2p_allreduce(devs), host_allreduce(devs), ring_mailbox(devs), dsa_device_index(devs)

  def warmup(self):
    """compile and capture every jit a request uses (they capture on their first call): a prompt of each prefill chunk size, decoding"""
    if self.config.cache_rows and self.seqs_supported(): return super().seq_warmup()
    sizes, st = self.prefill_chunks(), time.perf_counter()
    def lap(what:str):
      if getenv("DEBUG_LOAD", 1): print(f"warmup: {what} at {time.perf_counter()-st:.2f}s", flush=True)
    def run(what:str, call:Callable[[], Any]) -> Any:
      if not getenv("DEBUG_LOAD", 1): return call()
      began = time.perf_counter()
      lap(f"{what} begin")
      try: return call()
      finally: lap(f"{what} end (+{time.perf_counter()-began:.2f}s)")
    def warm_generate(n:int):
      return list(zip(range(4), self.generate([0] * min(n, self.max_context - 8))))
    # (a sequence sharded cache (a long context): one after the other, the weights read before the caches are made (their memory is the device's))
    parallel = (1)
    if parallel: self._capture_parallel(sizes)
    lap("captured")
    self._finish_load()
    lap("load finished")
    # The big chunk first: its scratch memory is the most, the other jits share it. A prompt one past a size runs the next one (1: smallest).
    # Parallel capture covers every size; without it, execute the big chunk too or the first real prefill pays its JIT/allocation cost.
    warm_chunks = sizes[:-1] if parallel else sizes
    for n in sorted({1, *(s + 1 for s in warm_chunks)}, reverse=True):
      # (WARMUP_JITS names: prefill<chunk>, step<m>, bpf<P>_<G>, slots and host; other graphs are captured on their first call)
      if not warmup_wanted(f"prefill{next((s for s in sizes if s >= n), sizes[-1])}"): continue
      self._cached_tokens, self._ckpt_tokens = [], []
      run(f"generate {n}", functools.partial(warm_generate, n))
    self._cached_tokens, self._ckpt_tokens = [], []

  def _capture_parallel(self, sizes:tuple[int, ...]):
    """the jits of the prefill chunk sizes and of speculative decoding compiled at the same time (in forked processes), called as generate does"""
    if not (devs:=self.token_embd.weight.device) or not amd_custom_kernels_supported(devs): return
    self._init_states()
    from tinygrad.llm.kernels.amd_glm_dsa import host_allreduce, p2p_allreduce
    # (the device memory the jits use is made here: not in the processes)
    if isinstance(devs, tuple) and p2p_allreduce(devs) is None: host_allreduce(devs)
    if not hasattr(self, "_token_buf"): self._token_host, self._token_buf = _host_ints(devs, self._main_token_rows())
    self._token_prev:list[int]|None = None  # (the captures write the buffer: the next generate writes all of it)
    # the random state of every device (sampling's noise) made here: one made in a capture's process isn't the one the jits run with, its
    # counter wouldn't advance (every step and request the same noise)
    for d in to_tuple(devs):
      key, ctr = Tensor._next_counter(d, 0)
      Tensor.realize(key, ctr, Tensor._device_rng_counters[d])
    temp = Tensor([0.0, 1.0]).to(devs).realize()  # (temperature, top_p)
    v_start_pos, calls = UOp.variable("start_pos", 0, self.max_context-1), cast(list[tuple[Any, Callable[[], Any]]], [])
    for n in sizes:
      sp, nt = v_start_pos.bind(0), UOp.variable("toks", 1, n).bind(n)
      if self._sp_tokens(self._token_buf[:, sp:sp+nt]):
        # (sequence parallel prefill in parts: every part's jit its own capture, the inputs of the ones after the first made here)
        for j in self._part_jits(n): calls.append((j, functools.partial(j, self._token_buf[:, sp:sp+nt], sp, temp)))
        continue
      if n not in self.jits: self.jits[n] = TinyJit(self._forward_effects, capture_first=True, scratch=self._scratch)
      setattr(self.jits[n], "warmup_name", f"prefill{n}")
      calls.append((self.jits[n], functools.partial(self, self._token_buf[:, sp:sp+nt], sp, temp)))
    if self.config.spec_drafts == 0 and (1):
      # (decoding without speculative decoding: its jit too)
      if 1 not in self.jits: self.jits[1] = TinyJit(self._forward_effects, capture_first=True, scratch=self._scratch)
      setattr(self.jits[1], "warmup_name", "decode")
      sp = v_start_pos.bind(0)
      calls.append((self.jits[1], functools.partial(self, self._token_buf[:, sp:sp+1], sp, temp)))
    # the computations left of loading (their jits) too: run after the weights are read
    for i, j in enumerate(j for j in self._pending_load if isinstance(j, _TinyJit)):
      setattr(j, "warmup_name", f"load{i}")
      calls.append((j, j))
    if self.config.spec_drafts > 0:
      self._spec_init()
      if not hasattr(self, "_spec_jit"): self._spec_jit = TinyJit(self.forward_spec, capture_first=True, scratch=self._scratch)
      setattr(self._spec_jit, "warmup_name", "spec")
      calls.append((self._spec_jit, functools.partial(self._spec_call, 0, 0, 1, temp)))
    # the weights are read (and what's computed from them) while the jits compile, and a checkpoint is saved and restored (its kernels)
    def meanwhile(take):
      # (the jits of the computations left run after: they're compiled meanwhile)
      while self._pending_load and not isinstance(self._pending_load[0], _TinyJit): self._pending_load.pop(0)()
    load_jits = [j for j in self._pending_load if isinstance(j, _TinyJit)]
    # (WARMUP_JITS: the jits of the names wanted (see warmup), the others captured on their first call; the loads' always)
    calls = [(j, c) for j, c in calls if j in load_jits or warmup_wanted(getattr(j, "warmup_group", getattr(j, "warmup_name", "")))]
    capture_parallel(calls, meanwhile)

  def generate(self, tokens:list[int], chunk_size:int|None=None, temperature:float=0.0, top_p:float=1.0):
    self._finish_load()
    # the prefill jits are specialized for one chunk size each
    sizes = self.prefill_chunks(chunk_size)
    chunk_size = sizes[-1]
    v_start_pos = UOp.variable("start_pos", 0, self.max_context-1)
    v_toks = {n: UOp.variable("toks", 1, n) for n in sizes}
    # TODO: use UOp.variable for temperature once float variables are supported
    devs = to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device))
    # (made once a temperature and top_p: a realize outside a jit walks every tensor of the model)
    if not hasattr(self, "_temps"): self._temps:dict[tuple[float, float], Tensor] = {}
    if (temp:=self._temps.get((temperature, top_p))) is None:
      temp = self._temps[(temperature, top_p)] = Tensor([temperature, top_p]).to(self.token_embd.weight.device).realize()
    if fast:=amd_custom_kernels_supported(self.token_embd.weight.device):
      # in host memory: every step writes its sampled token after its input tokens (see forward), the host polls it
      if not hasattr(self, "_token_buf"):
        self._token_host, self._token_buf = _host_ints(cast(str|tuple[str, ...], self.token_embd.weight.device), self._main_token_rows())
      for d in devs: Device[d].synchronize()  # a step run ahead by the last generate may still use it
      # the prompt, then -1 (not sampled yet) up to the tokens the last generate wrote (its list grew with them, steps run ahead write a few
      # more): the whole buffer only the first time (a write of max_context ints takes ~20 ms)
      n, prev = self.max_context + 1, getattr(self, "_token_prev", None)
      end = n if prev is None else min(n, max(len(prev), len(tokens)) + 2*self.config.spec_drafts + 64)
      self._token_host[:end] = array.array('i', tokens + [-1] * (end - len(tokens)))
      self._token_prev = tokens
      t = self._token_buf
    else:
      # assign all input tokens once, then slice from start_pos for the model call
      t = Tensor(tokens + [0] * (self.max_context - len(tokens)), dtype="int32").reshape(1, self.max_context)
    # recompute start_pos from what's currently valid in the caches
    start_pos = self.get_start_pos(tokens)
    if spec:=self.config.spec_drafts > 0 and fast:
      self._spec_init()
      # the MTP layer's input of the first prompt token: the hidden state of the one before (a zero one at the start)
      if start_pos == 0: self._spec_row = 0
      # (in a jit: a realize outside one walks every tensor of the model)
      if start_pos == 0 or self._spec_row: self._spec_row0(0 if start_pos == 0 else self._spec_row)
    out, prompt_len, ahead = None, len(tokens), False
    start0, row0, retries = start_pos, getattr(self, "_spec_row", 0), 0
    while len(tokens) < self.max_context:
      n_toks = min(chunk_size, len(tokens) - start_pos)
      # the prompt in chunks (symbolic sizes), then one token at a time (the decode jit)
      sp, nt = v_start_pos.bind(start_pos), 1 if start_pos >= prompt_len else v_toks[next(n for n in sizes if n_toks <= n)].bind(n_toks)
      # with run ahead, the previous step already submitted this one
      if not ahead: out = self(t[:, sp:sp+nt] if start_pos < prompt_len or out is None or fast else out, sp, temp).realize()
      # the token is polled only if the call wrote it to the token buffer (a replaced __call__ doesn't)
      if fast and (out is None or out.uop.base is not self._token_buf.uop.base): fast = False
      start_pos += n_toks
      # chunked prefill: keep processing until all prompt tokens are consumed
      if start_pos < len(tokens): continue
      # the end of the prompt: a checkpoint of the recurrent state (before any step of the reply)
      if spec:
        try: tok = self._wait_host(lambda: self._token_host[start_pos], "a token of generate")
        except P2PRunOut as e:
          # (P2P_RETRY, default 2: a p2p poll that ran out in the prompt's chunks (a long prompt's first check, the stall of the deadline):
          # the prompt's chunks again from the same place, every cache row they wrote written again; the epochs and mailboxes are consistent)
          if start_pos != prompt_len or (retries:=retries+1) > (2): raise
          print(f"generate: {e}: the prompt again ({retries})", flush=True)
          for d in devs: Device[d].synchronize()
          self._token_host[prompt_len:end] = array.array('i', [-1] * (end - prompt_len))
          start_pos, self._spec_row = start0, row0
          if start0 == 0 or row0: self._spec_row0(row0)
          continue
        tokens.append(tok)
        self._cached_tokens, self._spec_row = tokens[:-1], 0
        yield tok
        yield from self._spec_steps(tokens, temp)
        return
      if fast:
        # run ahead: submit the next step (it reads this token on the device) before waiting for this one. it writes the recurrent states of
        # the next position, the ones before it are kept (see GatedDeltaNetBlock._attention): the caches still hold every token but the last
        if ahead:=len(tokens) + 1 < self.max_context: self(t[:, (sp:=v_start_pos.bind(start_pos)):sp+1], sp, temp)
        tok = self._wait_host(lambda: self._token_host[start_pos], "a token of generate")
      tokens.append(tok if fast else int(cast(Tensor, out).item()))
      self._cached_tokens = tokens[:-1]
      yield tokens[-1]
