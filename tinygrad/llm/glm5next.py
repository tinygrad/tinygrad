from __future__ import annotations
import array, contextlib, enum, functools, gc, itertools, math, os, pathlib, random, re, time
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterator, cast
from tinygrad import Tensor, nn, UOp, TinyJit, getenv, function, dtypes, Device
from tinygrad.engine.realize import warm_worker_pool
from tinygrad.engine.jit import capture_parallel, warmup_select, warmup_wanted, _TinyJit, run_as_they_come
from tinygrad.llm.kernels.amd_glm import pool_expand
from tinygrad.llm.kernels.amd_glm import Linear, gated_delta_prefill, flash_attention, amd_custom_kernels_supported, find_packed, moe_linear, \
  hc_coeffs, hc_pre, q8_quantize, q8_linear, QUANT_SIZES, q8_quantize_cache_clear, amd_flash_attention_decode as flash_attention_decode, moe_route, \
  moe_grouped_linear, amd_flash_attention_mla_prefill as flash_attention_mla_prefill, partial_sum, fuse_gate_up, kda_decode, moe_router, \
  indexer_scores, _wmma_rdna4, pool_keys, pool_index_rows, slot_scatter, slot_gather, \
  indexer_decode_index, amd_flash_attention_mla_sparse as flash_attention_mla_sparse, moe_combine, token_sharded, shard_rows, reduce_scatter, \
  all_gather, all_gather_q8, sharded_argmax, skinny_matmul, moe_combine_scatter, kda_conv, kda_replay, tables_ready, moe_router_local, \
  hc_post_quant, \
  indexer_decode_index_rows, Q8_ROWS_I8, MOE_TILE, I8_WMMA_QUANTS, grouped_rows, q8_quantize_known, SLOT_SPAN, ple_gather, \
  ghc_pre, ghc_post, ghc_prefill_norm, ghc_prefill_mix, moe_softmax_router, Q8_0, IQ3_S, \
  mla_bmm, bmm_wmma, weighted_sum, ghc_decode_mix, FLAG_LANE, DEC_LANE, IDX_LANE, \
  flag_allreduce, ring_mailbox, ring_queue_of, RING_QUEUE_ON, mb_queue, f32t_gemm, few_out_rows, \
  host_allreduce, p2p_allreduce, _argmax_parts, _device_index, nucleus, seq_store, mla_sparse_seq, seq_rows, SEQ_BLK, \
  seq_gather, seq_views, seq_store_grouped, fp8_rows, FP8_ROW, IDX_FP8_ROW, q4_rows, Q4_ROW, \
  pool_index_shard, top_k_mask
from tinygrad.llm.gguf import gguf_parse, ggml_data_to_tensor, _GGML_QUANT
from tinygrad.llm.gguf_load import gguf_shard_units, _disk_bytes, _disk_location
from tinygrad.uop.ops import resolve, Ops
from tinygrad.dtype import DType
from tinygrad.helpers import unwrap, round_up, to_tuple, prod, dedup, DEBUG
from tinygrad.device import Buffer, BufferSpec, BufferStorage, MultiBuffer

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
  devices (by expert, cycling): the work of a token's experts is spread over all the devices. None if the devices don't get the same count.
  The remainder rotates around the device ring, including when it doesn't divide the device count.
  """
  base, r = divmod(slices, n)
  if experts*slices % n: return None
  per:list[list[int]] = [[] for _ in range(n)]
  for e in range(experts):
    extra = {(e*r+j) % n for j in range(r)}
    devs = [d for d in range(n) if d in extra for _ in range(base+1)] + [d for d in range(n) if d not in extra for _ in range(base)]
    for j, d in enumerate(devs): per[d].append(e*slices + j)
  return per

def expert_slice_table(units:list[int], experts:int, slices:int) -> list[list[int]]:
  """(experts, w): the local index of every slice a device holds of an expert, -1 padded"""
  table:list[list[int]] = [[] for _ in range(experts)]
  for i, u in enumerate(units): table[u // slices].append(i)
  w = max(len(t) for t in table)
  return [t + [-1]*(w-len(t)) for t in table]

_block_q8:list[tuple[UOp, tuple[Tensor, ...]]|None] = [None]  # a decoding block's output and its quantization (the next block's input's)
INLINE_ACT = (1)  # a decoding activation computed in the kernel that quantizes it (not a kernel of its own)

def gated_activation(gate:Tensor, up:Tensor, config:TransformerConfig|None=None, *, alpha:float=1.0, limit:float|None=None,
                     up_bias:float=0.0) -> Tensor:
  # (config: its swiglu alpha, limit and up bias, else the keywords')
  if config is not None: alpha, limit, up_bias = config.swiglu_alpha, config.swiglu_limit, config.swiglu_up_bias
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

def permute_rope(w:Tensor, n_heads:int, rope_dim:int) -> Tensor:
  x = w.reshape(n_heads, w.shape[0]//n_heads, -1)
  prefix = x.shape[1]-rope_dim
  return x[:, :prefix].cat(x[:, prefix:].rearrange("n (h two) d -> n (two h) d", two=2), dim=1).reshape(w.shape)

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
  # kda layout of a gated deltanet with a decay per head (qwen4exp: its q and k heads repeated for the v heads at load): the decay of a head
  # from ssm_alpha (every channel of it), the output gate attn_gate
  head_decay: bool = False
  full_rank_gate: bool = False
  silu_gate: bool = False

@dataclass(frozen=True)
class HCConfig:  # manifold-constrained hyper-connections: count residual streams, mixed by a doubly stochastic matrix
  count: int
  sinkhorn_iters: int
  eps: float
  low_rank: int = 0  # qwen4exp: the streams are mixed by a sigmoid gated low-rank projection instead (GatedResidual)

@dataclass(frozen=True)
class PLEConfig:  # qwen4exp n-gram embeddings: hashed rows of a host-side IQ4_NL table, mixed into the streams of one layer
  layer: int
  head_dim: int
  multipliers: tuple[int, ...]  # per n-gram position
  vocab_sizes: tuple[int, ...]  # per head: the bigram heads, then the trigram heads
  offsets: tuple[int, ...]
  eos: int
  conv_kernel: int
  ggml_type: int = 20  # the table's (IQ4_NL, Q8_0)

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
  # rows of the regions of sequences sharing caches (see Transformer.regions), at least max_context
  cache_rows: int = 0
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
  indexer_shared: tuple[bool, ...] = ()
  mtp: bool = False  # the next token prediction layer (MTPBlock)
  ple: PLEConfig|None = None
  expert_pad: int = 0  # the experts padded to this many (never selected) so their slices spread evenly over the devices (expert_slicing)
  spec_drafts: int = 0  # speculative decoding: the tokens the MTP layer drafts for every step of the model
  @property
  def rewind(self) -> int:
    """
    how far before the last position the recurrent states write the next call can start: a token (a generated token that the next prompt doesn't
    have), with speculative decoding the tokens of a step (one more than the drafts) and the drafts of the one launched after it"""
    return 2*self.spec_drafts + 1 if self.spec_drafts else 1
  padded_vocab_size: int = 0  # the output rows, zero padded to split over devices

# set while a block's rows are split over the devices (sequence parallel prefill): the sublayers return their partial sums (a leading device
# axis), the hyper-connection sums each device's rows of them (reduce_scatter) instead of all of them on every device
_partial_out = [False]
# and the sublayer's input rows of this device (before the gather): per token work (the router) runs on them, its results are gathered
_sp_rows:list[Tensor|None] = [None]
# the tokens of a prefill chunk (its rows past them are padding): the experts don't take the padding rows
_ffn_rows:list[UOp|int|None] = [None]
# (MB_DEP=state: the states and caches a micro batch's sublayer writes (the tensors after their writes), the next batch's sublayer waits for
# a kernel reading a value of each: not for the sublayer's output (its collectives))
_state_writes:list[list[Tensor]|None] = [None]
_mb_active = [False]
def _state_written(*ts:Tensor):
  if _state_writes[0] is not None: _state_writes[0].extend(ts)
# (a mixed step's parts: their sublayers' outputs before the projection, projected at once (one allreduce of all the rows))
_out_defer:list[bool] = [False]
def _out(layer:Linear, x:Tensor) -> Tensor:
  """a sublayer's output projection: its partial sums if they are kept (_partial_out), in half (they're sent so)"""
  if _out_defer[0]: return x
  return layer.partial(x, half=True) if _partial_out[0] else layer(x)

def _identity(x:Tensor) -> Tensor: return x

# the experts of a few tokens are the ones of every token (a decode kernel each), more are grouped by expert (prefill)
MOE_DECODE_ROWS = (96)
# batched decoding: rows of experts decoded (in bundles) up to this many, grouped (the prefill kernels) past it
MOE_BATCH_ROWS = (512)
def _skinny(lin:Linear, x:Tensor) -> Tensor|None:
  """lin(x) by skinny_matmul (few outputs, few rows) if its weight is stored for it (see Transformer._finish_load), else None"""
  if (wt:=getattr(lin, "weight_t", None)) is None or not isinstance(x.shape[-2], int): return None
  # (rows split over the devices (a prefill's): a few a device)
  if token_sharded(x) and int(x.uop.shard_shape[-2]) > (512): return None
  return skinny_matmul(x, wt)

def _decode_rows() -> int: return MOE_BATCH_ROWS if _batch_rows[0] is not None else MOE_DECODE_ROWS

# batched decoding (Transformer.forward_batch): the rows are groups of a sequence each, group g (its rows the tokens at positions pos.. of the
# sequence) in the state slot g: (pos, rows) of every group
_batch_rows:list[list[tuple[UOp, int]]|None] = [None]

def _batch_attention(x:Tensor, groups:list[tuple[UOp, int]], fn:Callable[[Tensor, list[tuple[UOp, int]]], Tensor]) -> Tensor:
  return fn(x, groups)

# SEQ_REPLAY: a sequence slot's recurrent state is one (after the first token of its last step) and the inputs of the tokens after it, replayed
# by the next step (see kda_decode): a step writes one state, not k+1 (their memory and their writes)
SEQ_REPLAY = (1)
def replay_q(k:int) -> int: return 2*(k+1)
def spec_by_size(base:int, sizes:tuple[int, ...], v:str|None=None) -> tuple[tuple[int, int], ...]:
  """SPEC_DRAFTS_BY_SIZE=m:d,... (or v): the drafts of a step of every size of sizes (the ones not listed: base), () if not set"""
  if not (v:=str(getenv("SPEC_DRAFTS_BY_SIZE", "")) if v is None else v): return ()
  by = {int(m): int(d) for m, d in (x.split(":") for x in v.split(",") if x)}
  assert all(d >= 1 for d in by.values()), f"SPEC_DRAFTS_BY_SIZE={v}: a step drafts one token at least"
  return tuple((m, by.get(m, base)) for m in sizes)

def slot_states(k:int) -> tuple[int, int]:
  """
  the S of a slot's recurrent states and of its conv states (the state of a position p is p%S, see kda_decode): a step of k+1 tokens reads
  one and writes k+1, the recurrent ones can overwrite the one read (SLOT_STATES_TIGHT: a quarter less memory, more slots)
  """
  return (k + 1 if k and (1) else k + 2), k + 2

def _group_positions(groups:list[tuple[UOp, int]], device:str|tuple[str, ...]|None) -> Tensor:
  """the start positions of the groups of batched decoding (their variables) as an int32 tensor on device"""
  # (all at 0, the prefills of batch_prefill: zeros. a stack of constants as a custom kernel's input fails the kernel graph's spec)
  if all(p.op is Ops.CONST and p.arg == 0 for p, _ in groups): return Tensor.zeros(len(groups), dtype=dtypes.int32, device=device).contiguous()
  return Tensor.stack(*[Tensor(p) for p, _ in groups]).cast(dtypes.int32).contiguous().to(device).contiguous()

def _slot_pool(n:int, like:Tensor, shape:tuple[int|UOp, ...]|None=None, dtype:DType|None=None, host:bool=False) -> tuple[Tensor, list[Tensor]]:
  """
  n slots of a state like `like` (of shape, or like's, of dtype or like's) in one buffer of every device, made but not allocated (a jit
  captured before its memory is holds it): the pool (n*shape[0], ...) and the n slots, views of it (buffers of their own: a kernel writes one
  in place). host: in host memory (every device its part), the devices read and write it directly
  """
  shape, devs, axis = tuple(int(x) for x in (shape or like.shape)), cast(str|tuple[str, ...], like.device), like.uop.axis
  local = tuple(x//len(devs) if isinstance(devs, tuple) and i == axis else x for i, x in enumerate(shape))
  dtype = dtype or like.dtype
  size, item = prod(local), dtype.itemsize
  if host:
    hb = [Buffer(d, n * size * item, options=BufferSpec(host=True)) for d in to_tuple(devs)]
    if isinstance(devs, tuple): (hm:=MultiBuffer.__new__(MultiBuffer)).bufs = hb
    base = UOp.from_buffer(hm if isinstance(devs, tuple) else hb[0], dtype, devs)
  else: base = UOp.new_buffer(devs, n * size, dtype)
  def tensor(u:UOp, rows:int) -> Tensor:
    u = u.reshape((rows * local[0], *local[1:]))
    return Tensor(u.unshard(axis) if isinstance(devs, tuple) and axis is not None else u)
  def view(i:int) -> Buffer|MultiBuffer:
    if not isinstance(b:=base.buffer, MultiBuffer): return b.view(size * item, i * size * item)
    (mb:=MultiBuffer.__new__(MultiBuffer)).bufs = [x.view(size * item, i * size * item) for x in b.bufs]
    return mb
  return tensor(base, n), [tensor(UOp.from_buffer(view(i), dtype, devs), 1) for i in range(n)]

def _region_views(devs:tuple[str, ...], axis:int|None, row:tuple[int, ...], dtype:DType, rows:int, spans:list[tuple[int, int]]) -> list[Tensor]:
  """
  views of one buffer of `rows` rows (of shape row) on every device, allocated: a tensor of every span (its first row, its rows). axis 0: a
  split one, the rows of a span are every device's part ((n*rows, *row)), None: whole on every device ((rows, *row))
  """
  item = prod(row) * dtype.itemsize
  base = cast(MultiBuffer, UOp.new_buffer(devs, rows * prod(row), dtype).buffer)
  out = []
  for r0, nr in spans:
    assert 0 <= r0 and r0 + nr <= rows, f"rows {r0}..{r0+nr} of {rows}"
    (mb:=MultiBuffer.__new__(MultiBuffer)).bufs = [x.view(nr * item, r0 * item) for x in base.bufs]
    for b in mb.bufs: b.ensure_allocated()
    u = UOp.from_buffer(mb, dtype, devs).reshape((nr, *row))
    out.append(Tensor(u.unshard(axis) if axis is not None else u))
  return out


def _host_buffer(devs:str|tuple[str, ...], nbytes:int, dtype=dtypes.uint8) -> tuple[memoryview, Tensor]:
  """nbytes of host memory that every device reads and writes directly (no copies): its memoryview and a tensor (replicated on devs)"""
  devs = to_tuple(devs)
  host = Buffer(devs[0], nbytes, options=BufferSpec(host=True, cpu_access=True)).ensure_allocated()
  if len(devs) == 1: return host.as_memoryview(allow_zero_copy=True), Tensor(UOp.from_buffer(host, dtype))
  # the other devices' buffers are the host's mappings on them (external: not freed with them)
  aliases = [Buffer(d, nbytes, opaque=BufferStorage(a:=host.get_buf(d)), options=BufferSpec(external_ptr=a)) for d in devs[1:]]
  (mb:=MultiBuffer.__new__(MultiBuffer)).bufs = [host] + aliases
  return host.as_memoryview(allow_zero_copy=True), Tensor(UOp.from_buffer(mb, dtype, devs))

_zero_jits:dict[Any, Any] = {}
def _fill_zero(*ts:Tensor):
  """
  the buffers of ts zeroed (allocated if not): chunks of a few sizes by jits of a kernel (one a size for the devices of a buffer, its input the
  chunk's view of every device's: a buffer's tail the chunk ending at its end), not a kernel scheduled for every tensor
  """
  for t in ts:
    for u in t.uop.toposort():
      if u.op is not Ops.BUFFER or (b:=u.buffer) is None: continue
      for x in (parts:=b.bufs if isinstance(b, MultiBuffer) else [b]): x.ensure_allocated()
      if _zero_later: _zero_later[-1].extend(parts)
      else:
        for x in parts: _zero_buffer(x)

# (while not empty: _fill_zero allocates, the buffers are zeroed later, see _zero_deferred)
_zero_later:list[list[Buffer]] = []
@contextlib.contextmanager
def _zero_deferred():
  """the buffers _fill_zero allocates in the context: not zeroed, the list it yields (see _zero_jit)"""
  _zero_later.append(bufs:=cast(list[Buffer], []))
  try: yield bufs
  finally: _zero_later.pop()

def _zero_jit(bufs:list[Buffer]) -> tuple[Any, Callable[[], Any]]:
  """a jit that zeroes the buffers (a kernel each, in place), and its call"""
  # (not the views of one zeroed whole: its inputs don't alias)
  bases = {id(x) for x in bufs if x._base is None}
  bufs = [x for x in bufs if x._base is None or id(x.base) not in bases]
  ts = [Tensor(UOp.from_buffer(x, dtypes.uint32 if x.nbytes % 4 == 0 else dtypes.uint8, x.device)) for x in bufs]
  # (t & 0: in place, an assign of zeros is made in a buffer of its own and copied)
  jit = TinyJit(lambda: Tensor.realize(*[t.assign(t.bitwise_and(0)) for t in ts]), capture_first=True)
  return jit, jit

def _zero_buffer(x:Buffer):
  if (n:=x.nbytes) < 4096: return x.copy_from(Buffer("NPY", n, opaque=memoryview(bytearray(n))))
  at = 0
  while at < n:
    C = next(c for c in (64 << 20, 2 << 20, 64 << 10, 4 << 10) if c <= n)
    if (jit:=_zero_jits.get((x.device, C))) is None: jit = _zero_jits[(x.device, C)] = TinyJit(lambda t: t.assign(t.bitwise_and(0)).realize())
    off = min(at, n - C)
    jit(Tensor(UOp.from_buffer(x.view(C, off).ensure_allocated(), dtypes.uint32, x.device)))
    at = off + C if n - at >= C else n

def _buffers(*shape:int, dtype:DType, device:str|tuple[str, ...]|None, axis:int|None=None, options:BufferSpec|None=None) -> Tensor:
  """
  a tensor of buffers (made, allocated when used) as a realize of an empty tensor makes them (not a realize: it walks every tensor): split on
  axis over the devices, or on every one
  """
  devs = to_tuple(device) if device is not None else (Device.canonicalize(Device.DEFAULT),)
  if not isinstance(device, tuple): axis = None
  if axis is not None and shape[axis] % len(devs): raise RuntimeError(f"can't split {shape} on axis {axis} over {devs}")
  shard = tuple(n // len(devs) if i == axis else n for i, n in enumerate(shape))
  bufs = [Buffer(d, max(prod(shard), 1) * dtype.itemsize, options=options) for d in devs]
  if isinstance(device, tuple): (mb:=MultiBuffer.__new__(MultiBuffer)).bufs = bufs
  u = UOp.from_buffer(mb if isinstance(device, tuple) else bufs[0], dtype, device if device is not None else devs[0]).reshape(shard)
  return Tensor(u.unshard(axis) if axis is not None else u)

def _zeros(*shape:int, dtype:DType=dtypes.float32, device:str|tuple[str, ...]|None, axis:int|None=None) -> Tensor:
  """a tensor of zeros, realized (see _fill_zero): split on axis over the devices, or on every one"""
  _fill_zero(t:=_buffers(*shape, dtype=dtype, device=device, axis=axis))
  return t


def _host_like(specs:list[tuple[Tensor, int, int]]) -> list[Tensor]:
  """
  tensors like t (its devices, split and dtype) of `rows` on axis, for every (t, axis, rows): views of one buffer in the host memory of every
  device (one allocation a device, made when used)
  """
  devs = to_tuple(cast(str|tuple[str, ...], specs[0][0].device))
  locs, at = [], 0
  for t, axis, rows in specs:
    shape = tuple(rows if i == axis else int(x) for i, x in enumerate(t.shape))
    local = tuple(x // len(devs) if isinstance(t.device, tuple) and i == t.uop.axis else x for i, x in enumerate(shape))
    locs.append((t, shape, local, at))
    at += round_up(prod(local) * t.dtype.itemsize, 4096)
  bases = [Buffer(d, max(at, 1), options=BufferSpec(host=True)) for d in devs]
  ret = []
  for t, shape, local, off in locs:
    views = [b.view(prod(local) * t.dtype.itemsize, off) for b in bases]
    if isinstance(t.device, tuple): (mb:=MultiBuffer.__new__(MultiBuffer)).bufs = views
    u = UOp.from_buffer(mb if isinstance(t.device, tuple) else views[0], t.dtype, t.device).reshape(local)
    ret.append(Tensor(u.unshard(t.uop.axis) if isinstance(t.device, tuple) and t.uop.axis is not None else u))
  return ret

@functools.cache
def _freqs(dim:int, end:int, theta:float, device:str|tuple[str, ...]|None, yarn:YaRNConfig|None) -> Tensor:
  """the rope table (one for every block of the same)"""
  return precompute_freqs_cis(dim, end, theta, device=device, yarn=yarn).realize()

_t_import = time.perf_counter()
def _lap(what:str):
  """DEBUG_LOAD=1: a line when loading gets past `what` (seconds since the import, the epoch: the launcher's start is known in epoch)"""
  if getenv("DEBUG_LOAD"):
    from tinygrad.device import alloc_stats
    print(f"load: {what} at {time.perf_counter()-_t_import:.2f}s epoch {time.time():.2f} ({alloc_stats[0]} allocations in {alloc_stats[1]:.2f}s)",
          flush=True)

def _open_devices(devices:tuple[str, ...]):
  """the devices opened at the same time (most of opening one is waiting for it): the first alone (what they share is made once)"""
  import threading
  if len(devices) < 2: return
  Device[devices[0]]
  threads = [threading.Thread(target=Device.__getitem__, args=(d,), daemon=True) for d in devices[1:]]
  for t in threads: t.start()
  for t in threads: t.join()

def _read_ahead(entries:dict, skip:tuple[str, ...]=()):
  """
  the bytes of the entries (not skip's: read otherwise) read into memory in the background: a thread a file, in order (the disk's fastest:
  6.8 GB/s on the 990 pro, scattered reads 4.9), the loader finds them there
  """
  import threading
  ranges:dict[str, list[tuple[int, int]]] = {}
  for name, (d, _, _) in entries.items():
    if name not in skip and (loc:=_disk_location(d)) is not None: ranges.setdefault(loc[0], []).append((loc[1], loc[1] + loc[2]))
  def run(path:str, rs:list[tuple[int, int]]):
    fd, mv = os.open(path, os.O_RDONLY), memoryview(bytearray(64 << 20))
    try:
      for lo, hi in merge_ranges(rs):
        while lo < hi and (got:=os.preadv(fd, [mv[:min(len(mv), hi - lo)]], lo)) > 0: lo += got
    finally: os.close(fd)
  for path, rs in ranges.items(): threading.Thread(target=run, args=(path, rs), daemon=True, name="read_ahead").start()

def merge_ranges(rs:list[tuple[int, int]]) -> list[tuple[int, int]]:
  """the ranges (lo, hi) sorted, the ones that touch or overlap one"""
  out:list[tuple[int, int]] = []
  for lo, hi in sorted(rs):
    if out and lo <= out[-1][1]: out[-1] = (out[-1][0], max(out[-1][1], hi))
    else: out.append((lo, hi))
  return out

def _is_buffer(u:UOp) -> bool:
  """the uop is a buffer's (realized: its reshapes, split over devices or not), not a computation"""
  while u.op in (Ops.RESHAPE, Ops.UNSHARD): u = u.src[0]
  return u.op is Ops.BUFFER

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


def _temp_row(t:float|tuple[float, ...]) -> array.array:
  """a row's temperature, top_p and top_k (a temperature alone: top_p 1; no top_k: 0, none) for the seq steps' host buffers (TEMP_W a row)"""
  t = t if isinstance(t, tuple) else (t, 1.0)
  return array.array('f', (*t, 0.0)[:TEMP_W] if len(t) < TEMP_W else t[:TEMP_W])
TEMP_W = 3  # (the floats of a row of the seq steps' temperatures: temperature, top_p, top_k)
# (SAMPLE_TOP_K: the sampled steps' graphs take their rows' top_k (8 rounds of counts over the vocabulary, exchanged when it's split): on
# with a server's TOP_K, off (no cost, a request's top_k ignored) without)
SAMPLE_TOP_K = getenv("SAMPLE_TOP_K", int(getenv("TOP_K", 0) > 0))

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
  # (23 bits: (k+0.5)/2**24 rounds to 1.0 in float32 at the top, its noise an infinity)
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

class HyperConnection:
  def __init__(self, config:TransformerConfig):
    self.count, self.eps, self.iters, self.norm_eps = (hc:=unwrap(config.hc)).count, hc.eps, hc.sinkhorn_iters, config.norm_eps
    self.fn = Linear(self.count * config.dim, (2 + self.count) * self.count, bias=False)
    self.base, self.scale = {"weight": Tensor.zeros((2 + self.count) * self.count)}, {"weight": Tensor.zeros(3)}

  def pre_split(self, xs:Tensor, norm:nn.RMSNorm) -> tuple[Tensor, Tensor, Tensor]:
    """the streams' rows split over the devices (static rows): the coefficients post, comb and the normed sublayer input of every device's rows"""
    flat = xs.flatten(2)
    n, features = int(math.prod(flat.shape[:-1])), int(flat.shape[-1])
    *xq, sq = q8_quantize(flat, n, features, sumsq=True, split=True)
    mix = q8_linear(self.fn, flat, xq=tuple(xq), split=True)
    _, post, comb, hn = hc_pre(mix, sq, self.base["weight"], self.scale["weight"], xs, unwrap(norm.weight), self.count, self.iters, self.eps,
                               self.norm_eps, norm.eps, features, True)
    return hn, post, comb

  def post_split(self, out:Tensor, post:Tensor, comb:Tensor, xs:Tensor) -> Tensor:
    """the sublayer output (every device's rows) added to the streams mixed by comb"""
    return out.unsqueeze(2) * post.unsqueeze(-1) + (comb.transpose(-1, -2) @ xs)

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
      # a few tokens: the streams after the sublayer and their quantization (the next one's) in a kernel
      if isinstance(norm, nn.RMSNorm) and norm.weight is not None and not split and not rows and xs.shape[0] == 1 and (1):
        coef, hn = hc_pre(mix, sq, base, scale, xs, norm.weight, self.count, self.iters, self.eps, self.norm_eps, norm.eps, features, raw=True)
        return hc_post_quant(sublayer(hn[:, :T]), coef, xs, self.count)
      if isinstance(norm, nn.RMSNorm) and norm.weight is not None and (1):
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
      # (a few tokens: materialized, the next sublayer's kernels read it, one buffer for them)
      return ret if rows else ret[:, :T].contiguous()
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

class GatedResidual:
  """
  qwen4exp hyper-connections: the sublayer input is a sigmoid gated low-rank mix of the normed streams (their mean), its output is added to every
  stream with a 2*sigmoid weight of its own
  """
  _winj:Tensor|None = None  # (the inject weights in float for ghc_pre, see Transformer._finish_load)
  def __init__(self, config:TransformerConfig, inject:bool=True):
    self.count, self.dim, self.eps, low_rank = (hc:=unwrap(config.hc)).count, config.dim, config.norm_eps, hc.low_rank
    self.norm = {"weight": Tensor.zeros(self.count*self.dim)}
    self.down, self.up = Linear(self.count*self.dim, low_rank, bias=False), Linear(low_rank, self.count*self.dim, bias=False)
    if inject: self.inject = Linear(self.count*self.dim, self.count, bias=False)

  def _fused(self, xs:Tensor) -> tuple[Tensor, Tensor]|None:
    """
    a few rows (decoding): the mix (B, T, dim) and the inject (B, T, count) (divided by count) in three kernels (ghc_pre, the down projection,
    ghc_post) instead of a dozen. None: not those rows or weights
    """
    B, T = xs.shape[:2]
    if isinstance(B, int) and isinstance(T, int) and (8) < B*T <= (1024) and \
       amd_custom_kernels_supported(xs.device) and \
       not _wmma_rdna4(xs.device) and self.down.ggml_type == self.up.ggml_type == Q8_0 and self.count == 4 and self.dim % 256 == 0 and \
       getattr(self, "_winj", None) is not None and not token_sharded(xs) and (1):
      # (the rows of batched decoding: the prefill's kernels on whole tiles of 32 rows (the rows past them padding), not a dozen of generic ones)
      n = B*T
      flat = xs.reshape(n, self.count*self.dim).float().pad(((0, round_up(n, 32) - n), None))
      xn, xhalf, part = ghc_prefill_norm(flat, self.norm["weight"], unwrap(self._winj), self.count, self.eps)
      lo = q8_linear(self.down, xhalf) * (1 / self.count)
      mixed = ghc_prefill_mix((lo * lo.sigmoid()).contiguous(), self.up.weight, xn)
      return mixed[:n].reshape(B, T, self.dim), (part.sum(1) * (1 / self.count))[:n].reshape(B, T, -1)[..., :self.count]
    if not (isinstance(B, int) and isinstance(T, int) and B*T <= (8) and (1) and
            amd_custom_kernels_supported(xs.device)): return None
    for lin in (self.down, self.up):
      if lin.ggml_type is None and lin.use_custom_quant: lin.set_quantized(lin.weight)
    if self.down.ggml_type != Q8_0 or self.up.ggml_type != Q8_0 or getattr(self, "_winj", None) is None: return None
    xn, xq, part = ghc_pre(xs.reshape(B*T, self.count*self.dim), self.norm["weight"], unwrap(self._winj), self.count, self.eps)
    down = q8_linear(self.down, xn, xq=xq)
    if 1 < B*T <= 8 and self.count == 4 and self.dim % 32 == 0 and self.up.in_features % 32 == 0 and 0 < self.up.in_features <= 512 and \
       not _wmma_rdna4(xs.device):
      mixed, inj = ghc_decode_mix(down, self.up.weight, xn, part, self.count)
      return mixed.reshape(B, T, self.dim), inj.reshape(B, T, -1)[..., :self.count]
    if B*T <= (1):
      mixed, inj = ghc_post(down, self.up.weight, xn, part, self.count)
      return mixed.reshape(B, T, self.dim), inj.reshape(B, T, -1)[..., :self.count]
    lo = down * (1 / self.count)
    # (more rows: the up projection is the decode linear, its weights read once for the rows. ghc_post's every row reads them)
    g = self.up((lo * lo.sigmoid()).contiguous()).reshape(B*T, self.count, self.dim)
    # (materialized: the sublayer's kernels read it, inlined into them it's computed again by every one)
    mixed = (g.sigmoid() * xn.reshape(B*T, self.count, self.dim).float()).mean(1).contiguous()
    return mixed.reshape(B, T, self.dim), (part.sum(1) * (1 / self.count)).reshape(B, T, -1)[..., :self.count]

  def mix(self, xs:Tensor) -> tuple[Tensor, Tensor]:
    """the (B, T, count, dim) streams: the sublayer's input (B, T, dim), the normed streams (B, T, count*dim)"""
    xn = (xs.float().rmsnorm(eps=self.eps) * self.norm["weight"].reshape(self.count, self.dim)).flatten(2)
    gate = self.up((self.down(xn) / self.count).silu()).sigmoid()
    return (gate * xn).unflatten(-1, (self.count, self.dim)).mean(2), xn

  def __call__(self, xs:Tensor, sublayer:Callable[[Tensor], Tensor], norm:Callable[[Tensor], Tensor]|None=None, rows:bool=False) -> Tensor:
    # (norm: the streams are normed by the mix. rows: static padded rows, the sublayer's input and output too)
    if (fused:=self._fused(xs)) is not None: mixed, inj = fused
    else:
      mixed, xn = self.mix(xs)
      inj = (s if (s:=_skinny(self.inject, xn)) is not None else self.inject(xn)) / self.count
    out = sublayer(mixed)
    if out.shape[1] is not xs.shape[1]: out = out.pad_to((out.shape[0], xs.shape[1], out.shape[2]))
    return xs + out.unsqueeze(2) * (inj.sigmoid() * 2).unsqueeze(-1)

  def pre_split(self, xs:Tensor, norm:Callable[[Tensor], Tensor]|None=None) -> tuple[Tensor, Tensor, Tensor]:
    """
    the streams' rows split over the devices (sequence parallel, static rows): the sublayer's input of every device's rows and their output
    weights (2*sigmoid(inject)), twice (the phases' coefficients are three, see post_split)
    """
    B, R = xs.shape[:2]
    rows = int(B * R)
    if self._winj is not None and self.dim % 256 == 0 and amd_custom_kernels_supported(xs.device):
      xn, xhalf, part = ghc_prefill_norm(xs.reshape(rows, -1), self.norm["weight"], self._winj, self.count, self.eps)
      inj = part.sum(1).reshape(B, R, -1)
    else:
      xn = (xs.float().rmsnorm(eps=self.eps) * self.norm["weight"].reshape(self.count, self.dim)).flatten(2).contiguous()
      xhalf, inj = xn.reshape(rows, -1).half().contiguous(), self.inject(xn)
    lo = q8_linear(self.down, xhalf, split=True) * (1 / self.count)
    act = (lo * lo.sigmoid()).contiguous()
    if self.up.ggml_type == Q8_0 and self.count == 4 and self.dim % 32 == 0 and not _wmma_rdna4(xs.device) and \
       int(act.uop.shard_shape[0]) % 32 == 0:
      mixed = ghc_prefill_mix(act, self.up.weight, xn.reshape(rows, -1), (int(B), int(R), self.dim))
    else:
      g = q8_linear(self.up, act, split=True).reshape(B, R, self.count, self.dim)
      mixed = (g.sigmoid() * xn.reshape(B, R, self.count, self.dim)).mean(2)
    post = (inj / self.count).sigmoid() * 2
    return mixed, post, post

  def post_split(self, out:Tensor, post:Tensor, comb:Tensor, xs:Tensor) -> Tensor:
    """the sublayer output (every device's rows) added to the streams with their weights"""
    return xs.float() + out.unsqueeze(2) * post.unsqueeze(-1)

  def mixed(self, xs:Tensor) -> Tensor:
    """the mix of the streams alone (the output's, the MTP layer's head)"""
    return fused[0] if (fused:=self._fused(xs)) is not None else self.mix(xs)[0]

class PLELayer:
  """
  qwen4exp n-gram embeddings: the hashed rows of the token's n-grams (emb), every stream gates their values by their normed key-query score, plus a
  causal depthwise conv over the gated values dilated by the n-gram size. its conv state is a state of the gated deltanet block it's in (one
  of its conv_states: slotted as them, see GatedDeltaNetBlock)
  """
  def __init__(self, config:TransformerConfig):
    self.ple, self.count, self.dim, self.eps = unwrap(config.ple), unwrap(config.hc).count, config.dim, config.norm_eps
    self.n = len(self.ple.multipliers)
    self.hist, width = (self.ple.conv_kernel - 1) * self.n, self.count * config.dim
    self.key, self.value = Linear(config.dim, width, bias=False), Linear(config.dim, config.dim, bias=False)
    self.norm_key, self.norm_query, self.norm_conv = ({"weight": Tensor.zeros(width)} for _ in range(3))
    self.conv1d = {"weight": Tensor.zeros(width, self.ple.conv_kernel)}

  def _norm(self, x:Tensor, norm:dict[str, Tensor]) -> Tensor:  # rmsnorm per stream: (..., count*dim) -> (..., count, dim)
    return x.unflatten(-1, (self.count, self.dim)).float().rmsnorm(eps=self.eps) * norm["weight"].reshape(self.count, self.dim)

  def gated(self, x:Tensor, emb:Tensor) -> tuple[Tensor, Tensor]:
    """the streams x (B, T, count, dim) and the rows emb (B, T, dim): the gated values (B, T, count, dim) and the conv's inputs (B, T, width)"""
    s = (self._norm(self.key(emb), self.norm_key) * self._norm(x.flatten(2), self.norm_query)).sum(-1, keepdim=True) * self.dim**-0.5
    g = self.value(emb).float().unsqueeze(2) * (s.sign() * s.abs().maximum(1e-6).sqrt()).sigmoid()
    return g, self._norm(g.flatten(2), self.norm_conv).flatten(2)

  def conv(self, window:Tensor, T:int) -> Tensor:
    """the conv of the T tokens after the hist rows of window (B, hist+T, width): (B, T, width)"""
    w = self.conv1d["weight"]
    return functools.reduce(lambda a,b: a+b, (w[:, i] * window[:, i*self.n:i*self.n+T] for i in range(self.ple.conv_kernel))).silu()

class FFNBlock:
  def __init__(self, config:TransformerConfig):
    self.config = config

    # --- RMSNorms (gated hyper-connections norm the streams themselves) --
    self.gated = config.hc is not None and config.hc.low_rank > 0
    self.attn_norm:Callable[[Tensor], Tensor] = _identity if self.gated else nn.RMSNorm(config.dim, config.norm_eps)
    self.ffn_norm:Callable[[Tensor], Tensor] = _identity if self.gated else nn.RMSNorm(config.dim, config.norm_eps)
    self.hc_attn:HyperConnection|GatedResidual
    self.hc_ffn:HyperConnection|GatedResidual
    if config.hc is not None:
      self.hc_attn, self.hc_ffn = (GatedResidual(config), GatedResidual(config)) if self.gated else (HyperConnection(config), HyperConnection(config))
    if config.ple is not None: self.ple = PLELayer(config)

    # --- feed-forward (MoE or dense) -------------------------------------
    if config.num_experts > 0:
      self.ffn_gate_inp = Linear(config.dim, config.num_experts, bias=config.expert_proj_bias)  # router
      if config.expert_bias: self.exp_probs_b = {"bias": Tensor.zeros(config.num_experts)}
      # sliced experts: num_experts*slices experts of hidden_dim/slices (padded: experts that are never selected, see expert_pad)
      S, E = config.expert_slices or 1, config.expert_pad or config.num_experts
      self.ffn_gate_exps = ExpertWeights(E*S, config.dim, config.hidden_dim//S, bias=config.expert_proj_bias)
      self.ffn_up_exps = ExpertWeights(E*S, config.dim, config.hidden_dim//S, bias=config.expert_proj_bias)
      self.ffn_down_exps = ExpertWeights(E*S, config.hidden_dim//S, config.dim, bias=config.expert_proj_bias)
      if config.expert_slices:
        for e in (self.ffn_gate_exps, self.ffn_up_exps, self.ffn_down_exps): e.slices, e.experts = S, E
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
      bias = self.exp_probs_b["bias"] if hasattr(self, 'exp_probs_b') else None
      gating, normalize_topk = self.config.expert_gating_func, self.config.norm_topk_prob
      exps = self.ffn_gate_exps
      exps.set_quantized()
      if (lg:=_skinny(router, xr)) is not None: logits = lg
      elif router.ggml_type is None and router.weight.dtype in (dtypes.half, dtypes.float) and router.bias is None and \
         isinstance(xr.shape[1], int) and int(xr.uop.shard_shape[1]) >= 32 and (router.weight.dtype == dtypes.half or rows is not None) and \
         amd_custom_kernels_supported(x.device):
        # many tokens: half copies of the input and the output, a float accumulator: the tensor cores (a float weight's half copy made)
        # (f32t_gemm: the transposed float weight (SKINNY's) staged in half by a kernel of whole tiles)
        wt = getattr(router, "weight_t", None)
        if wt is None or (lt:=f32t_gemm(xr.reshape(-1, xr.shape[-1]), wt)) is None:
          logits = xr.half().contiguous().matmul(router.weight.T.half().contiguous(), dtype=dtypes.float).contiguous()
        else: logits = lt.reshape(*xr.shape[:-1], -1)
      elif isinstance(T:=xr.shape[1], int) and 16 < T < 128 and T % 16 and (1):
        # (the rows of batched decoding (3 a sequence): padded to whole tiles of 16, the matmul's kernel of 60 rows is several times slower)
        logits = router(xr.pad((None, (0, round_up(T, 16) - T), None)))[:, :T]
      else: logits = router(xr)
      if bias is None and (gating == ExpertGating.SOFTMAX_WEIGHT or (gating == ExpertGating.SOFTMAX and normalize_topk)) and \
         self.config.num_experts % 32 == 0 and self.config.num_experts_per_tok <= 32 and amd_custom_kernels_supported(x.device):
        k, scale = self.config.num_experts_per_tok, self.config.routed_scaling_factor
        if rows is None and xr.shape[1] <= _decode_rows() and exps.slices and exps.ggml_type is not None and exps.slice_table.shape[-1]*k <= 32:
          return self._experts(x, h, *moe_softmax_router(logits, k, scale, exps.slice_table), local=True)
        sel, probs = moe_softmax_router(logits, k, scale)
        if rows is not None:
          # One exchange for both outputs: separate independent collectives share the same mutable flag counter.
          gathered = _on_lane(FLAG_LANE[0] + SEL_LANE, all_gather, sel.bitcast(dtypes.float32).stack(probs, dim=-1).contiguous(), 1)
          sel, probs = gathered[..., 0].bitcast(dtypes.int32), gathered[..., 1]
        return self._experts(x, h, sel, probs)
      if gating == ExpertGating.SIGMOID and normalize_topk and bias is not None and isinstance(xr.shape[1], int) and \
         amd_custom_kernels_supported(x.device) and self.config.num_experts <= 1024 and (1):
        k, scale, exps = self.config.num_experts_per_tok, self.config.routed_scaling_factor, self.ffn_gate_exps
        exps.set_quantized()
        if rows is None and xr.shape[1] <= _decode_rows() and exps.slices and exps.ggml_type is not None and (1):
          # decoding: the device's slices of the selections in the router's kernel
          return self._experts(x, h, *moe_router_local(logits, bias, k, scale, exps.slice_table), local=True)
        sel, probs = moe_router(logits, bias, k, scale)
        if rows is not None:
          # (one exchange for both, as the softmax router's: two independent collectives would share the flag lane's state)
          gathered = _on_lane(FLAG_LANE[0] + SEL_LANE, all_gather, sel.bitcast(dtypes.float32).stack(probs, dim=-1).contiguous(), 1)
          sel, probs = gathered[..., 0].bitcast(dtypes.int32), gathered[..., 1]
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
    if (T:=_ffn_rows[0]) is not None and sel.ndim >= 3 and isinstance(R:=sel.shape[-2], int) and (1):
      sel = (Tensor.arange(R).reshape(R, 1) < (Tensor(T) if isinstance(T, UOp) else T)).to(sel.device).where(sel, -1)
    # many tokens: grouped by expert (prefill), a few (the drafts of speculative decoding) as decoding
    grouped = (not isinstance(x.shape[1], int) or x.shape[1] > _decode_rows()) and all(e.ggml_type is not None for e in exps)
    shexp:Tensor|None = None
    shexp_gate:Tensor|None = None
    if grouped and hasattr(self, 'ffn_gate_shexp') and self.ffn_gate_exps.expert_parallel and \
       isinstance(x.shape[1], int) and self.ffn_down_shexp.shard_axis == 1:
      # the shared expert's partial sums are added by the kernel summing the experts' rows of every token (its gate: a scale of a row)
      shexp_act = gated_activation(self.ffn_gate_shexp(x), self.ffn_up_shexp(x), limit=self.config.swiglu_limit).contiguous()
      shexp = self.ffn_down_shexp.partial(shexp_act)
      if hasattr(self, 'ffn_gate_inp_shexp'): shexp_gate = (x * self.ffn_gate_inp_shexp["weight"]).sum(axis=-1).sigmoid().float()
    if grouped:
      # the partial sums kept (sequence parallel) are sent in half: stored so if nothing is added after
      out = self._grouped_experts(x, sel, probs, add=shexp, half=_partial_out[0] and (shexp is not None or not hasattr(self, 'ffn_gate_shexp')),
                                  add_scale=shexp_gate)
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
      shexp_act = gated_activation(self.ffn_gate_shexp(x), self.ffn_up_shexp(x), limit=self.config.swiglu_limit)
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

  def _grouped_experts(self, x:Tensor, sel:Tensor, probs:Tensor, add:Tensor|None=None, half:bool=False, add_scale:Tensor|None=None) -> Tensor:
    """prefill: the selections sorted by expert, every expert reads its weights once for all its tokens. (devices, B, T, D) partials if split"""
    B, T, D = x.shape
    # a symbolic token count runs padded to the max size, the garbage tokens are sliced off
    if not isinstance(T, int): x, sel, probs = x.pad_to(x.max_shape), sel.pad_to(sel.max_shape), probs.pad_to(probs.max_shape)
    gate_exps, down_exps = self.ffn_gate_exps, self.ffn_down_exps
    # (batched decoding: a few rows an expert, small tiles)
    # RDNA3: 32 rows for f16 and IQ3_S K256 kernels; the latter needs fewer padded rows and fits two LDS tiles per WGP.
    i8 = gate_exps.ggml_type in I8_WMMA_QUANTS and gate_exps.ggml_type != IQ3_S and (1)
    tile = (16) if _batch_rows[0] is not None else MOE_TILE if _wmma_rdna4(x.device) or (0) or i8 else 32
    # (MOE_TILE32_ROWS: RDNA4 tiles of 32 rows while the experts have at most this many rows each on average (a 64-row tile mostly padding))
    # (MOE_TILE32_TOKENS: RDNA3 tiles of 32 rows for a chunk of at most this many tokens (192: ~5 rows an expert, a 64-row tile mostly
    #  padding: the MoE layer 1.8x faster on a 7900 XTX at 192, 1.3x at 384, a chunk 4% at 768 and 1% at 1152, the same bits). 0: off)
    Tm = T if isinstance(T, int) else int(T.vmax)
    if tile == 64 and not _wmma_rdna4(x.device) and B * Tm <= (1152): tile = 32
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
    down = down_exps.grouped(act.contiguous(), entries, tiles, selections=tokens*k, out_dtype=dtypes.half if down_exps.slices else dtypes.float32,
                             sorted_out=sorted_down)
    # sequence parallel: summed over the devices and split on the tokens around the ring as they're combined
    if not self.gated and half and gate_exps.expert_parallel and _sp_rows[0] is not None and isinstance(T, int) and \
       tokens % len(cast(tuple, x.device)) == 0 and \
       tokens == T and (1):
      if add_scale is None: return moe_combine_scatter(down, pos, probs, tokens, k, add, by_selection=not sorted_down)
    y = moe_combine(down, pos, probs, tokens, k, add, add_scale=add_scale,
                    out_dtype=dtypes.half if half and gate_exps.expert_parallel else dtypes.float32,
                    by_selection=not sorted_down)
    y = y.reshape(-1, *sel_shape[:-1], D)[:, :, :x.shape[1] if isinstance(T, int) else T]
    return y if gate_exps.expert_parallel else y[0]

  # given the token-prefix match, return how much cached state this block can still reuse
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return prefix_len
  def _init_state(self, x:Tensor): raise NotImplementedError
  def _attention(self, x:Tensor, start_pos:int|UOp, T:int|UOp|None=None) -> Tensor: raise NotImplementedError

  def _ple(self, x:Tensor, emb:Tensor, start_pos:int|UOp, T:int|UOp|None, dep:UOp|None=None) -> Tensor: raise NotImplementedError

  def __call__(self, x: Tensor, start_pos: int|UOp, T:int|UOp|None=None, ple_emb:Tensor|None=None):
    # T (hyper-connections): x has static padded rows of which the first T are the tokens, the output too
    # ple_emb: the n-gram embedding rows of the tokens (the block with the PLE layer)
    self._init_state(x)
    # we pass in the weights implicitly so we unpack the GGUF on the fly
    # (decoding: the output's quantization made by its last kernel is returned too, the next block's input's: BLOCK_Q8_OUT)
    q8_out = T is None and self.config.hc is not None and not self.gated and bool((1))
    # (the block before's output's quantization, if x is that output: its function's arguments, the quantization of x in it)
    xq = _block_q8[0][1] if q8_out and _block_q8[0] is not None and _block_q8[0][0] is x.uop else ()
    if ple_emb is not None and not hasattr(self, "ple"): ple_emb = None
    @function(precompile=True, allow_implicit=True)
    def _run(x:Tensor, start_pos:int|UOp, T:int|UOp|None, ple_emb:Tensor|None, *xq:Tensor):
      if ple_emb is not None: x = x + self._ple(x, ple_emb, start_pos, T)
      if xq:
        fx = x.flatten(2)
        q8_quantize_known(fx, int(prod(fx.shape[:-1])), int(fx.shape[-1]), True, tuple(xq))
      if self.config.hc is not None:
        xs = self.hc_attn(x, lambda h: self._attention(h, start_pos, *(() if T is None else (T,))), self.attn_norm, rows=T is not None)
        _ffn_rows[0] = T
        try:
          out = self.hc_ffn(xs, self._feed_forward, self.ffn_norm, rows=T is not None)
          if not q8_out: return out.contiguous()
          flat = out.flatten(2)
          return (out.contiguous(), *q8_quantize(flat, int(prod(flat.shape[:-1])), int(flat.shape[-1]), sumsq=True))
        finally: _ffn_rows[0] = None
      h =     x + self._attention(self.attn_norm(x), start_pos, *(() if T is None else (T,)))
      return (h + self._feed_forward(self.ffn_norm(h))).contiguous()
    ret = _run(x, start_pos, T, ple_emb, *xq)
    if not q8_out: return ret
    out, *q = ret
    _block_q8[0] = (out.uop, tuple(q))
    return out

  # the phases of a sublayer on a micro batch (see micro_batch_blocks): functions, scheduled once
  def _phase_pre(self, hc:HyperConnection, norm:nn.RMSNorm, xs:Tensor) -> tuple[Tensor, Tensor, Tensor]:
    @function(precompile=True, allow_implicit=True)
    def _pre(xs:Tensor) -> tuple[Tensor, Tensor, Tensor]:
      hn, post, comb = hc.pre_split(xs, norm)
      return _fn_out(hn), _fn_out(post), _fn_out(comb)
    return _pre(xs)
  def _phase_sub(self, sub:str, hn:Tensor, start_pos:int|UOp, T:int|UOp) -> Tensor:
    @function(precompile=True, allow_implicit=True)
    def _sub(hn:Tensor, start_pos:int|UOp, T:int|UOp) -> Any:
      # (AG_Q8: sent as int8 groups of 32, as the quantized layers would make them)
      x = all_gather_q8(hn, 1) if (1) and hn.shape[-1] % 256 == 0 else all_gather(hn, 1, dtypes.half)
      # Route each device's own rows; gather the small selections/probabilities rather than repeat the router projection everywhere.
      _partial_out[0], _sp_rows[0] = True, hn
      # (in half: the layers' kernels read it so, a cast to float and back is two copies)
      xin = x
      if sub != "attn": _ffn_rows[0] = T

      try: out = self._attention(xin, start_pos, T) if sub == "attn" else self._feed_forward(xin)
      finally: _partial_out[0], _sp_rows[0], _ffn_rows[0] = False, None, None

      # partial sums, or the whole output
      # (summed and split already: moe_combine_scatter)
      done = out.ndim == hn.ndim and token_sharded(out)
      # (MB_DEP=compute: the token after the sublayer's compute (a value of its output before its collective))

      out = _fn_out(out if done else reduce_scatter(out, 1) if out.ndim == hn.ndim + 1 else shard_rows(out, 1))
      return out
    return _sub(hn, start_pos, T)
  def _phase_ple(self, x:Tensor, emb:Tensor, start_pos:int|UOp, T:int|UOp) -> Tensor:
    """the n-gram embedding of all the rows x of a micro batch (a function: its conv state's write is the call's, the next batch's after it)"""
    @function(precompile=True, allow_implicit=True)
    def _ple(x:Tensor, emb:Tensor, start_pos:int|UOp, T:int|UOp) -> Tensor: return self._ple(x, emb, start_pos, T).contiguous()
    return _ple(x, emb, start_pos, T)
  def _phase_post(self, hc:HyperConnection, out:Tensor, post:Tensor, comb:Tensor, xs:Tensor) -> Tensor:
    @function(precompile=True, allow_implicit=True)
    def _post(out:Tensor, post:Tensor, comb:Tensor, xs:Tensor) -> Tensor:
      # (HC_STREAMS_HALF: the streams kept in half, their kernels read half the bytes)
      ret = hc.post_split(out, post, comb, xs)
      return ret.cast(dtypes.half).contiguous()
    return _post(out, post, comb, xs)

def _fn_out(t:Tensor) -> Tensor:
  """
  a function's output: a kernel's output as it is (its buffer: contiguous() of it in a function is a copy kernel, ~0.1 ms of 1152 rows of
  2560 floats, 500 a 13824 chunk), else made contiguous
  """
  return t if (1) and t.uop.has_buffer_identity(after_ok=True) else t.contiguous()

def _on_queue(ts:tuple[Tensor, ...], queue:str) -> tuple[Tensor, ...]:
  """the outputs of a function call, its kernels on the compute queue `queue` (its copies on the copy queue of that index)"""
  def call_of(u:UOp) -> UOp: return u.src[1] if u.op is Ops.AFTER and u.src[1].op is Ops.CALL else call_of(u.src[0])
  call = call_of(ts[0].uop)
  def requeue(u:UOp) -> UOp:
    if u.op is Ops.AFTER and u.src[1] is call: return u.replace(src=(u.src[0], call.replace(arg=replace(call.arg, queue=queue)), *u.src[2:]))
    return u.replace(src=(requeue(u.src[0]), *u.src[1:]))
  return tuple(Tensor(requeue(t.uop)) for t in ts)

def _on_lane(lane:int, fn:Callable, *args):
  """fn's collectives on a lane of their own (see FLAG_LANE)"""
  prev, FLAG_LANE[0] = FLAG_LANE[0], lane
  try: return fn(*args)
  finally: FLAG_LANE[0] = prev
SEL_LANE = 8  # (the selections' gathers of the experts on lanes of their own: independent of the input's gather, a pool of the same size)

def micro_batch_blocks(blocks:list, xs:list[Tensor], starts:list, ts:list, ple:Tensor|None=None) -> list[Tensor]:
  """
  the blocks on micro batches of rows (sequence parallel streams, a batch starts at the end of the tokens before it). a sublayer is three
  phases: the coefficients and its normed input (then gathered), its compute (its partial sums then scattered), its output added to the streams.
  a sublayer of a batch runs after the one of the batch before it: it reads the state and the cache it writes (implicit, not graph inputs)
  """
  # (the order of the batches as a read of the one before's output (a value of it, zero): a call on another queue waits for the calls whose
  # buffers it reads, the state and cache they write are implicit inputs)
  def queued(i:int, *outs:Tensor) -> tuple[Tensor, ...]:
    return _on_queue(outs, f"COMPUTE:{q}") if (q:=mb_queue(i)) and (1) else outs
  def after(i:int, x:Tensor, dep:Tensor|None, rows:bool=True) -> Tensor:
    if dep is None: return x
    z = (dep[..., :1] if rows else dep.flatten()[:1]).cast(x.dtype) * 0
    while rows and z.ndim < x.ndim: z = z.unsqueeze(-1)
    return x + (z != z).where(0, z)
  subs = [(block, hc, norm, sub) for block in blocks for hc, norm, sub in ((block.hc_attn, block.attn_norm, "attn"),
                                                                           (block.hc_ffn, block.ffn_norm, "ffn"))]
  for block in blocks: block._init_state(xs[0])
  ple_done:list[Tensor] = []
  def pre(i:int, k:int) -> tuple[Tensor, ...]:
    block, hc, norm, sub = subs[k]
    if sub == "attn" and ple is not None and hasattr(block, "ple"):
      # (the n-gram embedding's conv runs over the rows in order: on all of them of the batch, every device keeps its rows. a batch's conv
      # state is the one the batch before it wrote: after it)
      n = int(ple.shape[1]) // len(xs)
      xi = after(i, xs[i], ple_done[-1] if ple_done else None, rows=False)
      p = block._phase_ple(all_gather(xi, 1), ple[:, i*n:(i+1)*n].contiguous(), starts[i], ts[i])
      ple_done.append(p)
      xs[i] = xs[i] + shard_rows(p, 1)
    return queued(i, *block._phase_pre(hc, norm, xs[i]))
  # the batches' sublayers in a skewed order: batch i's sublayer k in step k + i*MB_SKEW (the first batch's next sublayer before the last's
  # this one: with the collectives of the batches on one queue (RING_QUEUE_LANES=0) in that order, a batch's exchanges run while the other
  # computes and none runs ahead of the others: the queue's order keeps them together. a batch only waits for the batch before it)
  RING_QUEUE_ON[0] = _mb_active[0] = True
  done:list[list[Tensor|None]] = [[None]*len(subs) for _ in xs]
  deps:list[list[Tensor|None]] = [[None]*len(subs) for _ in xs]
  phases:list[tuple[Tensor, ...]] = [() for _ in xs]
  skew = (-1)
  if skew < 0: skew = len(subs)  # (a batch's sublayers all before the next batch's)
  for step in range(len(subs) + skew*(len(xs)-1)):
    for i in range(len(xs)):
      if not 0 <= (k:=step - i*skew) < len(subs): continue
      block, hc, norm, sub = subs[k]
      FLAG_LANE[0] = i
      if k == 0: phases[i] = pre(i, 0)
      hn, post, comb = phases[i]
      # (the batch before it ran this sublayer: its output's call)
      # (MB_DEP: what of the batch before a batch's sublayer waits for: its sublayer's output (sub), its streams after it (post), its next
      # sublayer's coefficients (pre))
      dep = deps[i-1][k] if i else None
      block._token = None
      out = queued(i, block._phase_sub(sub, after(i, hn, dep), starts[i], ts[i]))[0]
      done[i][k] = deps[i][k] = out
      # (MB_DEP=state: the next batch waits for the states this one wrote (none: a feed forward's), not its output)

      xs[i] = queued(i, block._phase_post(hc, out, post, comb, xs[i]))[0]

      if k + 1 < len(subs):
        phases[i] = pre(i, k+1)

  FLAG_LANE[0], RING_QUEUE_ON[0], _mb_active[0] = 0, False, False
  return xs

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
    self.indexer:Indexer|None = Indexer(config) if config.indexer else None

  def _attention(self, x:Tensor, start_pos:int|UOp, T:int|UOp|None=None) -> Tensor:
    assert T is None, "rows mode is for MLA and gated deltanet blocks"
    idx_mask = self._indexer_mask(x, start_pos, None)
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
    if idx_mask is None and amd_custom_kernels_supported(x.device) and self.config.ssm is not None:
      attn = flash_attention(q, assigned_kv, start_pos+T)
      attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
      return _out(self.attn_output, attn if not self.config.attn_output_gate else (attn * gate.sigmoid()))
    k = assigned_kv[0, :, :, 0:start_pos+T, :]
    v = assigned_kv[1, :, :, 0:start_pos+T, :]

    # NOTE: this mask is causal_lower_right, not the causal_upper_left generated by is_casual = True
    # TODO: this if statement should be removed and it shouldn't generate extra kernels
    mask, window = None, self.config.sliding_window
    if idx_mask is not None: mask = idx_mask.repeat_interleave(self.config.n_heads // int(idx_mask.shape[1]), dim=1)
    elif resolve(T != 1) or window:
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

  def _indexer_mask(self, x:Tensor, start_pos:int|UOp, idx_mask:Tensor|None) -> Tensor|None:
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

# VISION (GLM 5.3 Flash's pictures, see Transformer._vis_embed): set while the vision variant of a prefill chunk (a chunk with a picture's rows)
# is traced
_vis_on:list[bool] = [False]
# the first row of a sequence's region in the shared attention caches (see Transformer.regions) while its call is
# traced: a bound variable (or 0), a multiple of REGION_ALIGN
_seq_base:list[UOp|int] = [0]
# (a multiple of SEQ_BLK*6 (a block on every device of the MLA cache split on positions, see seq_store) and of the pools)
REGION_ALIGN = 3072

@dataclass(frozen=True)
class SeqBatch:
  """
  a step of G sequences sharing the caches (Transformer.forward_seqs): R rows of every one, its state slot, the first row of its region of the
  caches, the position of its first row (int32 tensors (G,) of the devices)
  """
  G:int; R:int; slots:Tensor; bases:Tensor; positions:Tensor # noqa: E702
  kbases:Tensor|None = None  # (SEQ_REPLAY: the positions of the slots' recurrent states, see kda_decode)
  def _rows(self, t:Tensor) -> Tensor: return t.reshape(self.G, 1).expand(self.G, self.R).reshape(-1)
  @property
  def row_positions(self) -> Tensor:
    """(G*R,) the position of every row"""
    ar = Tensor.arange(self.R, dtype=dtypes.int32).reshape(1, self.R).expand(self.G, self.R).reshape(-1)
    return (self._rows(self.positions) + ar.to(self.positions.device)).contiguous()
  @property
  def row_bases(self) -> Tensor: return self._rows(self.bases).contiguous()
  @property
  def row_phys(self) -> Tensor:
    """(G*R,) the row of the caches of every row"""
    return (self.row_positions + self.row_bases).contiguous()
# (set while a step of sequences sharing the caches is traced)
_seq_batch:list[SeqBatch|None] = [None]
def _seq_alone() -> bool:
  """a step of one sequence sharing the caches (GLM_SEQ): the kernels of the model's own step (forward_spec), its greedy tokens the same"""
  return (sb:=_seq_batch[0]) is not None and sb.G == 1 and bool((1))

def _rope(x:Tensor, cos:Tensor, sin:Tensor) -> Tensor:
  """half-split rope of x (..., d) with cos and sin (..., d/2) broadcast to it"""
  x1, x2 = x.chunk(2, dim=-1)
  return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)

def _pool_select(scores:Tensor, q_pos:Tensor, k_pos:Tensor, idx:IndexerConfig) -> Tensor:
  """
  scores (R, pools) of R queries at q_pos (R, 1) on the keys at k_pos (1, S): the mask (R, S) of the keys a query sees, the top_k/kpool best
  complete pools and the tokens of its own pool (the tail), causal
  """
  pools = scores.shape[-1]
  ar = Tensor.arange(pools.vmax)[:pools] if isinstance(pools, UOp) else Tensor.arange(pools)
  masked = ((ar + 1) * idx.kpool <= q_pos + 1).where(scores, -math.inf)
  picked = topk_mask(masked, idx.top_k // idx.kpool).repeat_interleave(idx.kpool, dim=-1)[..., :k_pos.shape[-1]]
  return ((picked | (k_pos >= (q_pos + 1) // idx.kpool * idx.kpool)) & (k_pos <= q_pos)).where(0.0, -math.inf)


class Indexer:
  _regions:tuple[int, int, int]|None = None  # (SLOT_ALIAS: the slots, their positions, the regions of the cache, see _region_views)
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
    self.cache_rows = config.cache_rows  # (GLM_SEQ: the regions of the sequences sharing the caches, see Transformer.seq_rows)
    self.attn_q_b = Linear(config.q_lora_rank or config.dim, idx.n_heads * idx.head_dim, bias=False)
    self.attn_k = Linear(config.dim, idx.head_dim, bias=False)
    self.k_norm = (nn.RMSNorm if idx.msa else nn.LayerNorm)(idx.head_dim, eps=idx.norm_eps)
    if idx.msa: self.q_norm = nn.RMSNorm(idx.head_dim, eps=idx.norm_eps)
    else: self.proj = Linear(config.dim, idx.n_heads, bias=False)
    if idx.kpool > 1 and not idx.msa:
      self.compressor_gate, self.compressor_ape = Linear(config.dim, idx.head_dim, bias=False), {"weight": Tensor.zeros(idx.kpool, idx.head_dim)}

  @property
  def dense(self) -> bool: return self.idx.msa or self.idx.rope_dim > 0  # (the plain path of every token's mask: minimax-m3, roped keys)

  def _init_state(self, x:Tensor):
    if self.dense:
      if not hasattr(self, "cache_kg"):
        self.cache_kg = Tensor.zeros(x.shape[0], round_up(self.max_context, self.idx.kpool),
                                     *((2, self.idx.head_dim) if hasattr(self, "compressor_gate") else (self.idx.head_dim,)),
                                     dtype=dtypes.half, device=x.device)
      return
    if hasattr(self, "cache_pk"): return
    idx, max_pools = self.idx, round_up(self.max_context, self.idx.kpool) // self.idx.kpool
    self.max_pools = max_pools  # (a sequence's: the scores' columns)
    # the sparse path (RDNA3/4): no mask, the pools a query picks are lists (decode) or per 16 queries (prefill, see pool_union)
    self.sparse = x.shape[0] == 1 and amd_custom_kernels_supported(x.device) and 16 % idx.kpool == 0 and \
      idx.head_dim % 16 == 0 and max_pools % 64 == 0 and bool((1))
    # the pooled keys, a call pools its tokens' pools only. the sparse path keeps them in half and the tokens of the last incomplete pool in a
    # ring (the last tokens: the next call can start before the end, see pool_keys), the other one every token's key and gate
    # (SEQ_FP8 with a sequence sharded cache: fp8 pooled keys, a row of 32 words of e4m3 values and the scale)
    fp8 = self.sparse and bool(getenv("SEQ_FP8", 1)) and seq_sharded(x.device) and idx.head_dim == 128
    row, dt = ((IDX_FP8_ROW,), dtypes.uint32) if fp8 else ((idx.head_dim,), dtypes.half if self.sparse else dtypes.float32)
    # (GLM_SEQ: the pools of every sequence's region, and the padding sequences')
    region_pools = (max(self.max_context, self.cache_rows) + REGION_ALIGN) // idx.kpool if self.cache_rows else max_pools
    self.cache_pk = Tensor.zeros(*(() if self.sparse else (x.shape[0],)), region_pools, *row, dtype=dt, device=x.device).contiguous()
    ring = round_up(max(2*idx.kpool, self.rewind + idx.kpool - 1), idx.kpool)
    # (SEQ_IDX_SHARD with a sequence sharded cache: the pooled keys split on their pools as the cache on its positions, a device stores and
    # scores its own, see pool_index_shard)
    self.shard = len(cast(tuple, x.device)) if self.sparse and bool(getenv("SEQ_IDX_SHARD", 0)) and seq_sharded(x.device) else 0
    if self.shard:
      # (GLM_SEQ: every region's, as the cache's rows (a region's pools a device's local rows from base//kpool//n, see core's grouped))
      rows = max(self.max_context, self.cache_rows) + REGION_ALIGN if self.cache_rows else self.max_context
      devs, local = cast(tuple[str, ...], x.device), seq_rows(rows, self.shard) // idx.kpool
      self.cache_pk = Tensor.empty(self.shard*local, *row, dtype=dt, device=devs, axis=0)
      if (reg:=self._regions) is not None:
        n, S, R = reg
        Pl = seq_rows(S, self.shard) // idx.kpool
        W = SLOT_SPAN
        views = _region_views(devs, 0, row, dt, R*Pl, [(0, local), (W*Pl, n*Pl)] + [((W+i)*Pl, Pl) for i in range(n)])
        self.cache_pk, self.pool_pk, self.slot_pk = views[0], views[1], views[2:]
    elif self.sparse and (reg:=self._regions) is not None:
      # (the slots' pooled keys in the same buffer, past the model's first slot's worth: see MLATransformerBlock._init_state)
      n, S, R = reg
      P = S // idx.kpool
      views = _region_views(cast(tuple[str, ...], x.device), None, row, dt, R*P,
                            [(0, R*P), (0, max_pools), (SLOT_SPAN*P, n*P)] + [((SLOT_SPAN+i)*P, P) for i in range(n)])
      views[0].assign(Tensor.zeros(R*P, *row, dtype=dt, device=x.device)).realize()
      self.cache_pk, self.pool_pk, self.slot_pk = views[1], views[2], views[3:]
    if self.sparse: self.ring = Tensor.zeros(ring, 2, idx.head_dim, dtype=dtypes.half, device=x.device).contiguous()
    else: self.cache_kg = Tensor.zeros(x.shape[0], max_pools*idx.kpool, 2, idx.head_dim, dtype=dtypes.half, device=x.device).contiguous()

  def picks(self, x:Tensor, q:Tensor, start_pos:UOp, T:int|UOp|None=None) -> Tensor:
    """the sparse path: for every query (Tq, n+1) the list of the n keys it sees at most (its top pools and the tail of its own), the count last"""
    Tq, idx = int(x.shape[1]), self.idx
    S = start_pos + (T := Tq if T is None else T)
    kg, iq = self.k_norm(self.attn_k(x)).stack(self.compressor_gate(x), dim=2), self.attn_q_b(q).reshape(Tq, idx.n_heads, idx.head_dim)
    # (_seq_base: the sequence's region of the caches, its pooled keys from b//kpool. the ring is the model's (a slot's copied, see _seq_copy))
    b = _seq_base[0]
    # (SEQ_IDX_SHARD: a region's pools from a device's local row b//kpool//n, b a multiple of the devices' blocks (REGION_ALIGN))
    pb = b // idx.kpool // (self.shard or 1)
    pk = pool_keys(self.cache_pk, self.ring, kg[0].half(), self.compressor_ape["weight"], start_pos, T, pbase=pb)
    weights, pools, k = (self.proj(x) * (idx.n_heads * idx.head_dim)**-0.5)[0], (S + idx.kpool - 1) // idx.kpool, idx.top_k // idx.kpool
    # one token: the heads are the rows of the scores' matmul, a kernel picks its pools
    if Tq == 1 and idx.n_heads % 16 == 0:
      return indexer_decode_index(iq[0], pk, weights[0], start_pos, pools, k, idx.kpool, pbase=pb, width=self.max_pools).reshape(1, -1)
    # a few (the drafts of speculative decoding): the same, a row a query
    if Tq <= 8 and idx.n_heads % 16 == 0 and (1):
      return indexer_decode_index_rows(iq, pk, weights, start_pos, pools, k, idx.kpool, pbase=pb, width=self.max_pools)
    # (index sharding: a prefill chunk's queries split on their rows over the devices (every device had all of them: the same lists six
    # times), each device the scores and the lists of its rows, the lists gathered. the same lists: exact)
    n = len(x.device) if isinstance(x.device, tuple) else 1
    if n > 1 and not self.shard and Tq % (16*n) == 0 and Tq // n <= (1056) and \
       idx.n_heads % 16 == 0 and amd_custom_kernels_supported(x.device):
      iq_s = shard_rows(iq.contiguous(), 0)  # (the q projection of every row as before, then split: the same queries)
      w_s = shard_rows(weights.contiguous(), 0)
      sc = indexer_scores(iq_s, pk, w_s, pools, idx.kpool, pbase=pb, width=self.max_pools)
      # (compact: the pools gathered, a quarter of the bytes, then expanded to the keys on every device (pool_expand))
      return pool_expand(all_gather(pool_index_rows(sc, start_pos, pools, k, idx.kpool, compact=True), 0), k, idx.kpool)[:Tq]
    # the scores on the tensor cores (rows of 16 queries), a kernel per query picks its pools
    rows = round_up(Tq, 16)
    iq, weights = iq.pad(((0, rows-Tq), None, None)), weights.pad(((0, rows-Tq), None))
    # blocks of rows one after the other: the scores of a block (rows x the max pools) are its memory, not the ones of the chunk
    block, picks = round_up(min(rows, (1056)), 16), None
    for r0 in range(0, rows, block):
      qb = iq[r0:r0+block].contiguous()
      if picks is not None: qb = Tensor(qb.uop.after(*picks.uop.src[1:]))  # after the last block's lists: its scores are done
      if picks is None: picks = Tensor.empty(rows, k*idx.kpool + idx.kpool + 1, dtype=dtypes.int32, device=x.device)
      sc = indexer_scores(qb, pk, weights[r0:r0+block], pools, idx.kpool, pbase=pb, width=self.max_pools)
      picks = pool_index_shard(sc, start_pos, pools, k, idx.kpool, row0=r0, out=picks, width=self.max_pools) if self.shard else \
              pool_index_rows(sc, start_pos, pools, k, idx.kpool, row0=r0, out=picks)
    return unwrap(picks)[:Tq]

  def _dense(self, x:Tensor, q:Tensor, start_pos:int|UOp, freqs:Tensor|None=None) -> Tensor:
    (B, T, _), idx, S = x.shape, self.idx, start_pos + x.shape[1]
    k, iq = self.k_norm(self.attn_k(x)), self.attn_q_b(q).reshape(B, T, idx.n_heads, idx.head_dim)
    if idx.msa: iq = self.q_norm(iq)
    if idx.rope_dim:
      rot = unwrap(freqs)[start_pos:start_pos+T]
      k = apply_rope(k[..., :idx.rope_dim].unsqueeze(1), rot, idx.rope_interleave).squeeze(1).cat(k[..., idx.rope_dim:], dim=-1)
      iq = apply_rope(iq[..., :idx.rope_dim].transpose(1, 2), rot, idx.rope_interleave).transpose(1, 2).cat(iq[..., idx.rope_dim:], dim=-1)
    if hasattr(self, "compressor_gate"):
      # cache an indexer key and a pooling gate per token, a pool's key is the gate softmax weighted sum of its keys
      kg = k.stack(self.compressor_gate(x), dim=2)  # (B, T, kpool, 2, D)
      kg = Tensor(self.cache_kg.uop.after(self.cache_kg[:, start_pos:S].uop.store(kg.half().uop))).float().reshape(B, -1, idx.kpool, 2, idx.head_dim)
      k = ((kg[..., 1, :] + self.compressor_ape["weight"]).softmax(2) * kg[..., 0, :]).sum(2)  # (B, S, D)
    else:
      k = Tensor(self.cache_kg.uop.after(self.cache_kg[:, start_pos:S].uop.store(k.half().uop))).float()  # (B, S, D)
    scores = iq @ k.transpose(-1, -2).unsqueeze(1)
    if idx.msa: scores = scores.reshape(B, T, idx.n_heads, scores.shape[-1] // idx.kpool, idx.kpool).max(-1).transpose(1, 2)
    else:       scores = (scores.relu() * (self.proj(x) * (idx.n_heads * idx.head_dim)**-0.5).unsqueeze(-1)).sum(2).unsqueeze(1)
    # the query sees the top_k/kpool best complete pools and the tokens of its own pool while incomplete (the tail)
    q_pos, k_pos = (Tensor(start_pos) + Tensor.arange(T)).reshape(T, 1), Tensor.arange(S)
    scores = ((Tensor.arange(scores.shape[-1]) + 1) * idx.kpool <= q_pos + 1).where(scores, -math.inf)
    if idx.msa: scores = (Tensor.arange(scores.shape[-1]) == q_pos // idx.kpool).where(math.inf, scores)
    _, sel = scores.topk(idx.top_k // idx.kpool)
    # selected incomplete pools add only tail/future tokens, handled by the final tail and causal mask
    picked = scores.zeros_like(dtype=dtypes.bool).scatter(-1, sel, True).repeat_interleave(idx.kpool, dim=-1)[..., :S]
    return ((picked | (k_pos >= (q_pos + 1) // idx.kpool * idx.kpool)) & (k_pos <= q_pos)).where(0.0, -math.inf).cast(x.dtype)

  def __call__(self, x:Tensor, q:Tensor, start_pos:int|UOp, freqs:Tensor|None=None, T:int|UOp|None=None) -> Tensor:
    # x has Tq rows, the first T (default all) are the tokens at start_pos: the mask (B, 1, Tq, S) of the keys every query sees
    if self.dense:
      assert T is None, "the dense indexer takes whole calls"
      return self._dense(x, q, start_pos, freqs)
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
  _regions:tuple[int, int, int]|None = None  # sequence regions for shared caches (see SeqEngine)
  pool_k:Tensor  # (batched decoding: the slots' caches, see _init_slots)
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
    # a few tokens: a thread an output (the same values)
    bmm = half and B == 1 and isinstance(Tq, int) and Tq <= 8 and amd_custom_kernels_supported(x.device) and bool((1))
    if bmm: q_nope = mla_bmm(q_nope, self.attn_k_b["weight"])
    else: q_nope = q_nope.half().contiguous().matmul(k_b, dtype=dtypes.float).contiguous() if half else q_nope @ k_b
    q = q_nope.cat(q_rope.cast(q_nope.dtype), dim=-1)

    kv_a = self.attn_kv_a_mqa(x)
    c_kv = self.attn_kv_a_norm(kv_a[..., :self.config.kv_lora_rank])
    k_rope = kv_a[..., self.config.kv_lora_rank:].reshape(B, Tq, 1, self.config.rope_dim).transpose(1, 2)
    if not self.config.ssm or not self.config.ssm.kda: k_rope = apply_rope(k_rope, freqs)

    k_store = c_kv.reshape(B, 1, Tq, self.config.kv_lora_rank).cat(k_rope.reshape(B, 1, Tq, self.config.rope_dim), dim=-1)
    if self.seq_sharded:
      # the cache split on its positions over the devices (MLA_SEQ_SHARD): every device stores its rows of the tokens, attends all the heads
      # over its keys (the indexer's lists of every token: the pooled keys are on every device), the outputs combined on the devices of the heads
      assert self.indexer is not None and self.indexer.sparse and isinstance(start_pos, UOp), "a sequence sharded cache needs the sparse indexer"
      # (_seq_base: the sequence's region of the caches from that position (sequences sharing them, GLM_SEQ))
      b = _seq_base[0]
      cache = seq_store(self.cache_s, self._seq_rows(k_store[0, 0]), start_pos + b, T)
      lists = self.indexer.picks(x, self.attn_q_a_norm(self.attn_q_a(x)), start_pos, T)
      scale = 1.0/self.config.head_dim**0.5
      if isinstance(Tq, int) and Tq > (16) and (1):
        # a chunk of a prompt: the cache before it read from the devices that have it (peer to peer), the chunk's own rows (every device
        # computes them), the attention of the device's heads on the whole of it
        whole = seq_gather(cache, k_store[0, 0], start_pos, T, round_up(self.config.max_context, 64), base=b)
        attn = flash_attention_mla_sparse(q, whole.reshape(1, 1, 1, -1, whole.shape[-1]), lists, scale, start_pos+T)
      else: attn = mla_sparse_seq(q, cache, lists, scale, start_pos+T, dims=self.config.kv_lora_rank, base=b)
      v_b, attn = self.attn_v_b["weight"].transpose(-1, -2), attn[..., :self.config.kv_lora_rank]
      if bmm: attn = mla_bmm(attn, self.attn_v_b["weight"])
      else: attn = attn.half().contiguous().matmul(v_b, dtype=dtypes.float).contiguous() if half else attn @ v_b
      return _out(self.attn_output, attn.transpose(1, 2).reshape(B, Tq, -1))
    k_store = k_store[:, :, :T]
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
    assert self.config.rope_dim == 0 or (self.config.ssm is not None and self.config.ssm.kda), "batched decoding of MLA without rope only"
    q_nope_head_dim = self.config.head_dim - self.config.rope_dim
    q_lora = self.attn_q_a_norm(self.attn_q_a(x))
    q = self.attn_q_b(q_lora).reshape(B, Tq, self.config.n_heads, self.config.head_dim).transpose(1, 2)
    q_nope, q_rope = q[..., :q_nope_head_dim], q[..., q_nope_head_dim:]
    # (many rows: the tensor cores)
    tc = Tq > (32) and self.attn_k_b["weight"].dtype == dtypes.half
    if (qb:=bmm_wmma(q_nope, self.attn_k_b["weight"])) is not None: q = qb
    elif tc: q = q_nope.half().contiguous().matmul(self.attn_k_b["weight"].transpose(-1, -2), dtype=dtypes.float).contiguous()
    else: q = mla_bmm(q_nope, self.attn_k_b["weight"])
    q = q.cat(q_rope.cast(q.dtype), dim=-1)
    kv_a = self.attn_kv_a_mqa(x)
    c_kv = self.attn_kv_a_norm(kv_a[..., :self.config.kv_lora_rank])
    k_rope = kv_a[..., self.config.kv_lora_rank:].reshape(B, Tq, 1, self.config.rope_dim).transpose(1, 2)
    k_store = c_kv.reshape(B, 1, Tq, self.config.kv_lora_rank).cat(k_rope.reshape(B, 1, Tq, self.config.rope_dim), dim=-1)
    idx = self.indexer
    if idx is not None:
      ic = idx.idx
      kg = idx.k_norm(idx.attn_k(x)).stack(idx.compressor_gate(x), dim=2)[0]
      iq = idx.attn_q_b(q_lora).reshape(Tq, ic.n_heads, ic.head_dim)
      iw = (p if not _seq_alone() and (p:=_skinny(idx.proj, x)) is not None else idx.proj(x)) * (ic.n_heads * ic.head_dim)**-0.5
      iw = iw[0]
    assert idx is not None, "batched decoding: the sparse attention of the indexer"
    def core(q:Tensor, k_store:Tensor, kg:Tensor, iq:Tensor, iw:Tensor, groups:list[tuple[UOp, int]],
             pools:tuple[Tensor, ...]) -> tuple[Tensor, tuple[Tensor, ...]]:
      # one kernel for all the sequences (the same rows each, the caches of slot g the g-th of the pools): their positions in a tensor. returns
      # the attention and the pools (pooled keys, ring, cache) after it (another part's writes after its reads)
      assert len({T for _, T in groups}) == 1, "batched decoding: groups of the same rows"
      Tq, scale = int(q.shape[2]), 1.0/self.config.head_dim**0.5
      if (sb:=_seq_batch[0]) is not None:
        # sequences sharing the caches (GLM_SEQ): every one's region of the caches from its base, its ring in its state slot. the regions
        # as slots of REGION_ALIGN rows (a block on every device, see seq_store): slot base//REGION_ALIGN, a sequence's span of them
        assert self.seq_sharded, "sequences sharing the caches: decoding steps, a sequence sharded cache"
        U, nd = REGION_ALIGN, len(cast(tuple, x.device))
        assert U % (SEQ_BLK*nd) == 0 and U % ic.kpool == 0, f"REGION_ALIGN {U}: blocks of {SEQ_BLK} on {nd} devices"
        span, R, rslots = -(-self.config.max_context // U), int(idx.slot_ring[0].shape[0]), (sb.bases // U).contiguous()
        # (SEQ_IDX_SHARD: a region slot's pools are a device's U//kpool//nd local rows (its blocks'), not U//kpool)
        stride = U // ic.kpool // (idx.shard or 1)
        pk = pool_keys(pools[0], pools[1], kg.half(), idx.compressor_ape["weight"], None, Tq,
                       grouped=(sb.positions, stride, R, rslots, None, span, sb.slots))
        picks = indexer_decode_index_rows(iq, pk, iw, None, None, ic.top_k // ic.kpool, ic.kpool, grouped=(sb.positions, stride, rslots, span))
        cache = seq_store_grouped(pools[2], self._seq_rows(k_store[0, 0]), sb.positions, rslots, U // nd, span=span)
        attn = mla_sparse_seq(q, cache, picks, scale, None, grouped=(sb.positions, rslots, U // nd, span), dims=self.config.kv_lora_rank)
        return attn, (pk, Tensor(pools[1].uop.after(pk.uop)), cache)
      positions = _group_positions(groups, x.device)
      P, R = idx.pk_slot(), int(idx.slot_ring[0].shape[0])
      slots, counts = None, None
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
    pools:tuple[Tensor, ...] = (idx.cache_pk, idx.pool_ring, self.cache_s) if _seq_batch[0] is not None else \
      (idx.pool_pk, idx.pool_ring, self.pool_s if self.seq_sharded else self.pool_k)
    attn = core(q, k_store, kg, iq, iw, groups, pools)[0]
    if (vb:=bmm_wmma(attn, self.attn_v_b["weight"])) is not None: attn = vb
    elif tc: attn = attn.half().contiguous().matmul(self.attn_v_b["weight"].transpose(-1, -2), dtype=dtypes.float).contiguous()
    else: attn = mla_bmm(attn, self.attn_v_b["weight"])
    return _out(self.attn_output, attn.transpose(1, 2).reshape(B, Tq, -1))

  def _init_seq_slots(self, n:int):
    """sequences sharing the caches (GLM_SEQ): the indexer's ring of n slots (the caches are the model's, a region a sequence)"""
    if (idx:=self.indexer) is not None and len(getattr(idx, "slot_ring", [])) < n: idx.pool_ring, idx.slot_ring = _slot_pool(n, idx.ring)

  def _attn_mask(self, x:Tensor, start_pos:int|UOp, T:int|UOp|None=None) -> Tensor|None:  # (1, 1, Tq, start_pos+T), None: all visible
    if self.indexer and self.config.max_context > self.indexer.idx.top_k:
      return self.indexer(x, self.attn_q_a_norm(self.attn_q_a(x)), start_pos, T=T)
    T = x.shape[1]
    return Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, buffer=False).triu(start_pos+1) if resolve(T != 1) else None

  @property
  def seq_sharded(self) -> bool: return hasattr(self, "cache_s")
  def _seq_rows(self, k:Tensor) -> Tensor:
    """the rows of the tokens as the sequence sharded cache's: half, or fp8 (SEQ_FP8, see fp8_rows)"""
    if self.cache_s.dtype != dtypes.uint32: return k
    return q4_rows(k) if int(self.cache_s.shape[-1]) == Q4_ROW else fp8_rows(k)
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
        # (SEQ_FP8: a row of fp8 values and their scales, see fp8_rows: half the memory)
        # (SEQ_KV4 (default off, a numerics change): 4-bit rows, see q4_rows: 288 bytes instead of 528)
        row, dt = ((Q4_ROW if getenv("SEQ_KV4", 0) else FP8_ROW,), dtypes.uint32) if getenv("SEQ_FP8", 1) and D == 512 else ((D,), dtypes.half)
        if (reg:=self._regions) is not None:
          # (SLOT_ALIAS: the slots of batched decoding in the same buffer, regions of S positions: the model's first (SLOT_SPAN of them), slot
          # i's the SLOT_SPAN+i-th. the model's cache past them is the slots': a long sequence alone overwrites them, see Transformer.main_overlaps)
          n, S, R = reg
          stride, W = seq_rows(S, len(devs)), SLOT_SPAN
          views = _region_views(devs, 0, row, dt, R*stride, [(0, seq_rows(self.config.max_context, len(devs))), (W*stride, n*stride)] +
                                [((W+i)*stride, stride) for i in range(n)])
          self.cache_s, self.pool_s, self.slot_s = views[0], views[1], views[2:]
        else:
          # (GLM_SEQ: the regions of the sequences sharing the caches, and the padding sequences' one after them, see Transformer.seq_rows)
          N = max(self.config.max_context, self.config.cache_rows) + (REGION_ALIGN if self.config.cache_rows else 0)
          self.cache_s = Tensor.empty(len(devs)*seq_rows(N, len(devs)), *row, dtype=dt, device=devs, axis=0)
      else: self.cache_k = Tensor.empty(x.shape[0], 1, self.config.max_context, D, dtype=dtypes.half, device=x.device)
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
    self.is_kda = ssm.kda
    if ssm.kda and not ssm.head_decay:
      self.ssm_g_a, self.ssm_g_b = Linear(config.dim, self.head_v_dim, bias=False), Linear(self.head_v_dim, ssm.inner_size, bias=False)
      self.ssm_f_a, self.ssm_f_b = Linear(config.dim, self.head_k_dim, bias=False), Linear(self.head_k_dim, ssm.inner_size, bias=False)
    else:
      self.attn_gate = Linear(config.dim, ssm.inner_size, bias=False)
      self.ssm_alpha = Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_beta = Linear(config.dim, self.num_v_heads, bias=False)
    # (head_decay: a head's bias, added to its decay's input before every channel of it: the kernels' bias of every channel is _dt_ch, zeros)
    self.ssm_dt = {"bias": Tensor.zeros(ssm.inner_size if ssm.kda and not ssm.head_decay else self.num_v_heads)}
    self.ssm_a = Tensor.zeros(self.num_v_heads, 1) if ssm.kda else Tensor.zeros(self.num_v_heads)
    self.ssm_norm, self.ssm_out = nn.RMSNorm(self.head_v_dim, config.norm_eps), Linear(ssm.inner_size, config.dim, bias=False)

  def _gate(self, x:Tensor) -> Tensor:
    """the output gate's input of every channel (a sigmoid gate in the kda layout)"""
    return self.attn_gate(x) if hasattr(self, "attn_gate") else self.ssm_g_b(self.ssm_g_a(x))
  def _few_rows(self, layer:Linear, x:Tensor) -> Tensor:
    """layer(x) of a float weight of few outputs (alpha, beta): the rows of batched decoding (32 or more) by few_out_rows (padded to its tiles)"""
    R = int(prod(x.shape[:-1]))
    if layer.bias is None and layer.ggml_type is None and R >= 32 and (1):
      xr = x.reshape(R, x.shape[-1])
      if R % 32: xr = xr.pad(((0, round_up(R, 32) - R), None))
      if (r:=few_out_rows(xr, layer.weight)) is not None: return r[:R].reshape(*x.shape[:-1], -1)
    return layer(x)
  def _alpha(self, x:Tensor, rows:bool=False) -> Tensor:
    """the decay's input of every channel (kda): a head's (head_decay) for each of its channels"""
    if not hasattr(self, "ssm_alpha"): return self.ssm_f_b(self.ssm_f_a(x))
    a = (self._few_rows(self.ssm_alpha, x) if rows else self.ssm_alpha(x)).float() + self.ssm_dt["bias"]
    return a.unsqueeze(-1).expand(*a.shape, self.head_k_dim).flatten(-2)

  def _ple(self, x:Tensor, emb:Tensor, start_pos:int|UOp, T:int|UOp|None, dep:UOp|None=None) -> Tensor:
    """
    the n-gram embedding added to the streams x (B, Tq, count, dim): its conv state is conv_states[3] (the states (S, B, hist, width) at a
    position p are p%S, as the other conv states'), slot g's in pool_conv[3] (batched decoding)
    dep: a store its conv state is read after (its own is _ple_store)
    """
    ple, (B, Tq) = self.ple, x.shape[:2]
    g, inp = ple.gated(x, emb)
    if (sb:=_seq_batch[0]) is not None:
      # sequences sharing the caches: R rows of every one from its position, its state of the position read from its slot, the ones after
      # every row written (a step's rows past the ones accepted are taken back)
      G, R, SC = sb.G, sb.R, int(self.slot_conv[0][3].shape[0])
      assert R < SC, "the states of the rows and the one before"
      pool = self.pool_conv[3].reshape(self.pool_conv[3].shape[0], ple.hist, -1)
      before = slot_gather(pool, sb.slots, sb.positions, SC, dtypes.float)  # (G, hist, width)
      window = (sb.positions.reshape(G, 1, 1) > 0).where(before, 0).cat(inp.reshape(G, R, -1).float(), dim=1).contiguous()
      effects = [slot_scatter(pool, window, sb.slots, Tensor.full((G,), r+1, dtype=dtypes.int32).to(x.device).contiguous(), SC, shift=1,
                              starts=sb.positions) for r in range(R)]
      conv = Tensor(ple.conv(window, R).contiguous().uop.after(*[e.uop for e in effects])).reshape(B, Tq, -1)
      return g + conv.unflatten(-1, (ple.count, ple.dim))
    if (groups:=_batch_rows[0]) is not None:
      # batched decoding: a row a sequence (its slot g the group g), its state of its position read from the slot, the one after written
      assert all(t == 1 for _, t in groups), "batched n-gram embeddings: a token a sequence"
      G, SC = len(groups), int(self.slot_conv[0][3].shape[0])
      pool = self.pool_conv[3].reshape(self.pool_conv[3].shape[0], ple.hist, -1)
      positions, slots = _group_positions(groups, x.device), Tensor.arange(G, dtype=dtypes.int32).to(x.device).contiguous()
      before = slot_gather(pool, slots, positions, SC, dtypes.float)  # (G, hist, width)
      window = (positions.reshape(G, 1, 1) > 0).where(before, 0).cat(inp.reshape(G, 1, -1), dim=1).contiguous()
      ones = Tensor.ones(G, dtype=dtypes.int32).to(x.device).contiguous()
      self._ple_effects = [slot_scatter(pool, window, slots, ones, SC, shift=1, starts=positions)]
      conv = Tensor(ple.conv(window, 1).contiguous().uop.after(self._ple_effects[0].uop)).reshape(B, Tq, -1)
      return g + conv.unflatten(-1, (ple.count, ple.dim))
    start_pos = start_pos if isinstance(start_pos, UOp) else UOp.variable("start_pos", 0, self.config.max_context-1).bind(start_pos)
    if T is None: T = Tq
    states, S = self.conv_states[3], int(self.conv_states[3].shape[0])
    if dep is not None: states = Tensor(states.uop.after(dep))
    rp, wp = start_pos % S, (start_pos + T) % S
    state = states.shrink(((rp, rp+1),) + (None,)*(states.ndim-1)).reshape(states.shape[1:])
    window = Tensor(start_pos).eq(0).where(0, state).cat(inp.cast(state.dtype), dim=1)
    window = window.pad_to((B, ple.hist + x.max_shape[1], window.shape[-1])).contiguous()
    store = states.shrink(((wp, wp+1),) + (None,)*(states.ndim-1)).uop.store(window[:, T:T+ple.hist].unsqueeze(0).uop)
    self._ple_store = store
    conv = Tensor(ple.conv(window, int(x.max_shape[1])).contiguous().uop.after(store))[:, :Tq]
    return g + conv.unflatten(-1, (ple.count, ple.dim))

  def _attention_batch(self, x:Tensor, groups:list[tuple[UOp, int]]) -> Tensor:
    """batched decoding: a group of rows a sequence, its states in its slot (the projections of all of them in one kernel each)"""
    x = x.half()
    projs, gate = [self.attn_q(x), self.attn_k(x), self.attn_v(x)], self._gate(x)
    alpha, beta = self._alpha(x, rows=True), self._few_rows(self.ssm_beta, x)
    weights = [w["weight"] for w in (self.ssm_conv1d_q, self.ssm_conv1d_k, self.ssm_conv1d_v)]
    # groups of the same size: a kernel for all of them (their states in the pools, their positions in a tensor)
    if len({T for _, T in groups}) == 1 and (1):
      S, SC, T = int(self.slot_rec[0].shape[0]), int(self.slot_conv[0][0].shape[0]), groups[0][1]
      # (sequences sharing the caches: their slots)
      sb = _seq_batch[0]
      rp = getattr(self, "pool_rp", None) if sb is not None and sb.kbases is not None else None
      z = kda_decode(self.pool_rec, self.pool_conv[:3], projs, weights, gate, alpha, beta, self._dt_ch, self.ssm_a,
                     unwrap(self.ssm_norm.weight), Tensor(groups[0][0]), unwrap(self.config.ssm).gate_lower_bound, self.ssm_norm.eps, row=0, rows=T,
                     positions=sb.positions if sb is not None else _group_positions(groups, self.pool_rec.device), groups=len(groups), S=S, SC=SC,
                     slots=sb.slots if sb is not None else None, replay=rp, kbases=cast(SeqBatch, sb).kbases if rp is not None else None,
                     replay_q=replay_q(self.config.spec_drafts))
      return _out(self.ssm_out, z)
    # (every row's kernel reads its row of the inputs)
    starts = list(itertools.accumulate((t for _, t in groups), initial=0))
    zs = [kda_decode(self.slot_rec[g], self.slot_conv[g][:3], projs, weights, gate, alpha, beta, self._dt_ch, self.ssm_a,
                     unwrap(self.ssm_norm.weight), Tensor(pos), unwrap(self.config.ssm).gate_lower_bound, self.ssm_norm.eps, row=starts[g], rows=T,
                     SC=int(self.slot_conv[g][0].shape[0]))
          for g, (pos, T) in enumerate(groups)]
    return _out(self.ssm_out, zs[0].cat(*zs[1:], dim=1))

  def _init_slots(self, n:int, ctx:int, half:bool|None=None, replay:bool=False):
    """
    the states of n sequences of batched decoding (one slot each): S of each state, the one of a position p is p%S (see slot_states). replay:
    one recurrent state a slot and the replayed inputs (see SEQ_REPLAY)
    """
    if len(getattr(self, "slot_rec", [])) >= n: return
    S, SC = slot_states(self.config.spec_drafts)
    if replay:
      S = 1
      K, V = self.head_k_dim, self.head_v_dim
      # (a row of a head: its k, decay, v and beta)
      self.pool_rp = _slot_pool(n, self.recurrent_state, (replay_q(self.config.spec_drafts), *self.recurrent_state.shape[1:3], 2*K + V + 1),
                                dtypes.float32)[0]
    # (in one buffer of every device: slot_store writes what a step reads)
    # (in float as the model's: a step reads and writes it, in half its rounding adds up over a long reply. in half for many slots
    # (SLOT_STATE_HALF, default past 8): their memory, a long context's model and 32 slots of it don't fit)
    if half is None: half = n > 8
    self.pool_rec, self.slot_rec = _slot_pool(n, self.recurrent_state, (S, *self.recurrent_state.shape[1:]), dtypes.half if half else None)
    pools = [_slot_pool(n, c, (SC, *c.shape[1:])) for c in self.conv_states]
    self.pool_conv, self.slot_conv = [p for p, _ in pools], [[slots[i] for _, slots in pools] for i in range(n)]

  def _attention(self, x:Tensor, start_pos:int|UOp, T:int|UOp|None=None) -> Tensor:
    if (groups:=_batch_rows[0]) is not None: return _batch_attention(x, groups, self._attention_batch)
    # T: x has static padded rows of which the first T are the tokens (rows mode), the padded rows are computed but don't change the state
    rows = T is not None
    B, Tq, _ = x.shape
    if T is None: T = Tq
    # bind ints to a variable so the reset flag stays a runtime value (it toggles when generation restarts at position 0)
    start_pos = start_pos if isinstance(start_pos, UOp) else UOp.variable("start_pos", 0, self.config.max_context-1).bind(start_pos)
    initial = Tensor(start_pos).eq(0)
    is_kda = self.is_kda
    symbolic = isinstance(T, UOp) and not rows
    T_pad = x.max_shape[1]  # symbolic chunks are padded to their max size: one graph serves every size

    # input processing
    x = x.half()
    if is_kda and isinstance(Tq, int) and Tq <= self.config.spec_drafts + 1 and B == 1 and not rows and self.ssm_conv_kernel == 4 and \
       self.head_k_dim == self.head_v_dim == 128 and amd_custom_kernels_supported(x.device) and (1):
      # a few tokens (one, or the drafts of speculative decoding): a kernel for the convolutions, the delta rule, the norm and the gate
      z = kda_decode(self.recurrent_state, self.conv_states[:3], [self.attn_q(x), self.attn_k(x), self.attn_v(x)],
                     [w["weight"] for w in (self.ssm_conv1d_q, self.ssm_conv1d_k, self.ssm_conv1d_v)], self._gate(x),
                     self._alpha(x), self.ssm_beta(x), self._dt_ch, self.ssm_a, unwrap(self.ssm_norm.weight),
                     Tensor(start_pos), unwrap(self.config.ssm).gate_lower_bound, self.ssm_norm.eps)
      return _out(self.ssm_out, z)
    out_gate = self._gate(x)
    out_gate = out_gate.reshape(B, Tq, self.num_v_heads, self.head_v_dim)
    # (a prefill's many rows: few outputs a device, their kernel)
    def few(layer) -> Tensor:
      return r if layer.bias is None and layer.ggml_type is None and (r:=few_out_rows(x, layer.weight)) is not None else layer(x)
    beta = few(self.ssm_beta).sigmoid().reshape(B, Tq, self.num_v_heads)
    if is_kda and hasattr(self, "ssm_alpha"):
      alpha = (few(self.ssm_alpha).float() + self.ssm_dt["bias"]).unsqueeze(-1).expand(B, Tq, self.num_v_heads, self.head_k_dim).flatten(-2)
    else: alpha = self._alpha(x) if is_kda else few(self.ssm_alpha)
    g, a = (alpha.float() + (self._dt_ch if is_kda else self.ssm_dt["bias"])).reshape(B, Tq, self.num_v_heads, -1), \
      self.ssm_a.reshape(self.num_v_heads, -1)
    # the decay: log_alpha = softplus(g) * A, or bounded by gate_lower_bound: sigmoid(g * -A) * lb
    log_alpha = g.softplus() * a if (lb:=unwrap(self.config.ssm).gate_lower_bound) is None else (g * -a).sigmoid() * lb
    if rows and Tq is not T:  # the padded rows are exact no-ops: beta=0 (delta rule off), log_alpha=0 (decay 1)
      real = Tensor.arange(Tq).reshape(1, Tq, 1) < cast(int, T)  # T can be a bound variable
      beta, log_alpha = real.where(beta, 0), real.unsqueeze(-1).where(log_alpha, 0)

    # the states (S, ...) at a position p are p%S: the ones of start_pos are read, the ones after the tokens written. the other ones keep the
    # states before: steps can be taken back (a generated token that the next prompt doesn't have, drafts that weren't accepted)
    S = int(self.recurrent_state.shape[0])
    rp, wp = start_pos % S, (start_pos + T) % S
    def parity(s:Tensor, p:UOp) -> Tensor: return s.shrink(((p, p+1),) + (None,)*(s.ndim-1))
    # qkv conv, conv_state is reset when starting from position 0
    def conv(states:Tensor, proj:Tensor, weight:Tensor) -> tuple[Tensor, UOp, tuple[Tensor, Tensor]]:
      # assemble the conv window in a static-size buffer: [conv_state | qkv rows | zero-pad].
      # padded steps are exact no-ops: beta=0 (delta rule off), log_alpha=0 (decay 1 after exp)
      state = parity(states, rp).reshape(states.shape[1:])
      window = (masked:=initial.where(0, state)).cat(proj.cast(state.dtype), dim=1)
      # (the fused kernel makes the conv itself: the window is the next state's rows only, not a buffer)
      window = window.pad_to((B, self.ssm_conv_kernel-1 + T_pad, proj.shape[-1]))
      if not fused: window = window.contiguous()
      # the last conv_kernel-1 columns of the window become the next conv state
      store = parity(states, wp).uop.store(window[:, T:T+self.ssm_conv_kernel-1].cast(state.dtype).unsqueeze(0).uop)
      return functools.reduce(lambda a,b: a+b, (window[:, i:i+T_pad] * weight[:, i] for i in range(self.ssm_conv_kernel))).silu(), store, \
        (masked, proj.pad_to((B, T_pad, proj.shape[-1])))
    # the convolutions, norms and layouts of q, k and v in a kernel (and the dot of q and k)
    fused = is_kda and B == 1 and self.num_v_heads == self.num_k_heads and self.head_k_dim == self.head_v_dim and self.head_k_dim % 32 == 0 and \
      not symbolic and amd_custom_kernels_supported(x.device) and bool((1))
    kq = None
    if is_kda:
      weights = [w["weight"] for w in (self.ssm_conv1d_q, self.ssm_conv1d_k, self.ssm_conv1d_v)]
      convs = [conv(state, proj(x), w) for state, proj, w in zip(self.conv_states, (self.attn_q, self.attn_k, self.attn_v), weights)]
      (q, k, v), conv_state_stores = [o for o,_,_ in convs], tuple(s for _,s,_ in convs)
      if fused: q, k, v, kq = kda_conv([w[0] for _,_,w in convs], [w[1] for _,_,w in convs], weights, self.head_k_dim, self.head_k_dim**-0.5)
    else:
      conv_out, conv_state_store, _ = conv(self.conv_state, self.attn_qkv(x), self.ssm_conv1d["weight"])
      conv_state_stores = (conv_state_store,)
      q, k, v = conv_out.split(list(self.qkv_dims), dim=-1)
    if symbolic:
      out_gate = out_gate.pad_to((B, T_pad, self.num_v_heads, self.head_v_dim))
      beta, log_alpha = beta.pad_to((B, T_pad, self.num_v_heads)), log_alpha.pad_to((B, T_pad, *log_alpha.shape[2:]))
    if not fused:
      q, k = (z.reshape(B, T_pad, self.num_k_heads, self.head_k_dim) for z in (q, k))
      # l2 norm x/sqrt(sum(x²)+1e-6) as in the FLA kernels, then the v heads share the k heads
      q, k = ((z.rmsnorm(eps=1e-6/self.head_k_dim) / self.head_k_dim**0.5).repeat(1, 1, self.num_v_heads//self.num_k_heads, 1) for z in (q, k))
      v = v.reshape(B, T_pad, self.num_v_heads, self.head_v_dim)
      # layout the per-step operands to broadcast against the (B, H, V, K) state
      q, k, v = (z.transpose(1, 2).float() for z in (q, k, v))
      q = q * self.head_k_dim**-0.5
    beta = beta.transpose(1, 2).float()
    alpha = log_alpha.transpose(1, 2).exp()  # per-channel decay for kda, per-head otherwise (B, H, T, K|1)

    # recurrent: scan over the (padded) tokens, updating the recurrent state. collect the per-step outputs
    state = Tensor(self.recurrent_state.uop.after(*conv_state_stores))  # carry the conv write into this graph
    if self.head_k_dim % 32 == 0 and self.head_v_dim % 4 == 0 and amd_custom_kernels_supported(x.device):
      # one fused kernel for the whole scan; it resets and updates the recurrent state in place (RDNA3/4)
      core = gated_delta_prefill(q, k, v, beta, alpha, state, Tensor(start_pos), end_pos=start_pos + T, kq=kq)
      # (the conv states' writes are before it on its queue: its signal is after them)
      _state_written(core)
      core = core.transpose(1, 2)
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
    z = (self.ssm_norm(core) * (out_gate.silu() if unwrap(self.config.ssm).silu_gate else out_gate.sigmoid())).cast(x.dtype).contiguous()
    if symbolic: z = z[:, :T]
    return _out(self.ssm_out, z.reshape(B, Tq, -1))

  def _init_state(self, x):
    if not hasattr(self, "recurrent_state"):
      # with heads split over devices, the states are too
      split = isinstance(x.device, tuple) and isinstance(self.ssm_a.device, tuple) and self.ssm_a.uop.axis == 0
      # S of each state: the state at a position p is p%S (see _attention). S-1 steps can be taken back (config.rewind)
      S = self.config.rewind + 1
      def state(*shape:int, axis:int) -> Tensor: return _zeros(S, *shape, device=x.device, axis=axis+1 if split else None)
      if hasattr(self, "attn_qkv"): self.conv_state = state(x.shape[0], self.ssm_conv_kernel-1, self.conv_channels, axis=2)
      else: self.conv_states = [state(x.shape[0], self.ssm_conv_kernel-1, d, axis=2) for d in self.qkv_dims]
      # (the n-gram embedding's conv state: on every device, slotted as the conv states)
      if hasattr(self, "ple"):
        self.conv_states.append(_zeros(S, x.shape[0], self.ple.hist, self.ple.count*self.ple.dim, device=x.device))
      self.recurrent_state = state(x.shape[0], self.num_v_heads, self.head_v_dim, self.head_k_dim, axis=1)
      # (the kernels' decay bias of every channel: a head's is added to its input already (head_decay), else the weight's)
      ch = self.num_v_heads * self.head_k_dim
      if hasattr(self, "ssm_alpha") and self.is_kda:
        self._dt_ch = _zeros(ch, device=x.device, axis=0 if split else None)
      else: self._dt_ch = self.ssm_dt["bias"]

class MTPBlock(MLATransformerBlock):
  """
  the next token prediction layer: from the hidden state of a position (the last block's, before the output norm) and the embedding of the
  next token, a transformer block (without hyper-connections) predicts the token after it (with the output head)
  """
  def __init__(self, config:TransformerConfig):
    super().__init__(replace(config, hc=None))
    self.eh_proj = Linear(2 * config.dim, config.dim, bias=False)
    self.enorm, self.hnorm = nn.RMSNorm(config.dim, config.norm_eps), nn.RMSNorm(config.dim, config.norm_eps)
    self.shared_head_norm = nn.RMSNorm(config.dim, config.norm_eps)
  def predict(self, emb:Tensor, h:Tensor, start_pos:int|UOp) -> Tensor:
    """emb (B, T, dim) the embeddings of the next tokens, h (B, T, dim) the hidden states: the normed hidden states of the predictions"""
    return self.shared_head_norm(self(self.eh_proj(self.enorm(emb).cat(self.hnorm(h), dim=-1)), start_pos))


class Transformer:
  # (VISION: the server's VISION_MMPROJ sets vision (glm_vision's tower makes the pictures' rows); image_pad_id the template's <|image|>)
  vision:bool = False
  image_pad_id:int = -1
  vision_mrope = False  # (a picture's rows at plain positions: the text model has no rope, no M-RoPE delta (engine's _delta))
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
      if config.ple is not None and i != config.ple.layer: c = replace(c, ple=None)
      self.blk.append(GatedDeltaNetBlock(c, config.ssm) if config.ssm and config.ssm_layers[i] else block_cls(c))
    self.token_embd  = nn.Embedding(config.vocab_size, config.dim)
    # (gated hyper-connections: the last mix of the streams norms them)
    self.gated = config.hc is not None and config.hc.low_rank > 0
    self.output_norm:Callable[[Tensor], Tensor] = _identity if self.gated else nn.RMSNorm(config.dim, config.norm_eps)
    if self.gated: self.output_hc = GatedResidual(config, inject=False)
    self.output = Linear(config.dim, config.padded_vocab_size or config.vocab_size, bias=False)
    if config.mtp: self.mtp = MTPBlock(config)
    self.max_context = config.max_context
    self.has_recurrent_block = any(isinstance(b, GatedDeltaNetBlock) for b in self.blk)
    self._cached_tokens: list[int] = []
    self.ple_table:memoryview|None = None  # (the n-gram table's bytes, see from_gguf)
    self._kv_real = config.n_kv_heads  # (the file's KV heads, see from_gguf)
    self._ple_file:tuple[str, int, int]|None = None  # (its file, offset and size, if they're a gguf file's)
    self._effects: list[Tensor]|None = None  # tensors that forward realizes with its output (see _forward_effects)
    self._ckpt_tokens: list[int] = []  # the prompt of the recurrent state's checkpoint (_checkpoint)
    # a jit for every token count of a call (the max size of a symbolic chunk, 1 for decoding): static padded shapes of a small chunk don't pay
    # for a big one
    self.jits:dict[int, Any] = {}
    self._pending_load:list[Callable[[], None]] = []  # the work left of loading (from_gguf defer_load)
    self._loaded = False  # the weights are loaded (from_gguf)
    self._scratch:dict = {}  # the memory of the jits' intermediates: they run one at a time

  @property
  def _spec_dim(self) -> int:
    """the width of a position's hidden state the MTP layer reads: the streams (gated hyper-connections), else the normed hidden state"""
    return unwrap(self.config.hc).count * self.config.dim if self.gated else self.config.dim

  def _hc_out(self, x:Tensor) -> Tensor:
    """the streams (B, T, count, dim) into the output's hidden state"""
    return self.output_hc.mixed(x) if self.gated else x.mean(2)

  # ** the n-gram embeddings (qwen4exp): the rows of every position's n-grams in host memory the devices read (written by the host: hashed
  # from the tokens before it, see _ple_ids), the table they're from in the gguf file (mmapped)
  PLE_PAD = 8192  # (rows past the max context: a prefill chunk's padded rows)
  def _ple_init(self):
    if hasattr(self, "_ple_host") or (ple:=self.config.ple) is None: return
    devs = cast(str|tuple[str, ...], self.token_embd.weight.device)
    self._ple_rb = ple.head_dim // _GGML_QUANT[ple.ggml_type][0] * _GGML_QUANT[ple.ggml_type][1]
    width = len(ple.vocab_sizes) * self._ple_rb
    mv, self._ple_buf = _host_buffer(devs, (self.max_context + self.PLE_PAD) * width)
    self._ple_host = memoryview(mv).cast('B')

  def _ple_ids(self, ctxs) -> Any:
    """the table rows (P, heads) of every position from its n-gram context (P, n) [t, t-1, t-2..] (EOS before a start or an EOS)"""
    import numpy as np
    ple = unwrap(self.config.ple)
    c = np.asarray(ctxs, dtype=np.int64).reshape(-1, len(ple.multipliers))
    hashes = np.bitwise_xor.accumulate(c * np.asarray(ple.multipliers, dtype=np.int64), axis=1)[:, 1:]  # (P, n-1): bigram, trigram..
    n, h = len(ple.multipliers), len(ple.vocab_sizes)
    which = np.asarray([j*(n-1)//h for j in range(h)])
    return hashes[:, which] % np.asarray(ple.vocab_sizes, dtype=np.int64) + np.asarray(ple.offsets, dtype=np.int64)

  def _ple_rows(self, ids) -> Any:
    """the packed rows (P, heads, rb) of the table rows ids (P, heads), a numpy gather of the (mmapped) table"""
    import numpy as np
    if not hasattr(self, "_ple_np"): self._ple_np = np.frombuffer(unwrap(self.ple_table), dtype=np.uint8).reshape(-1, self._ple_rb)
    return self._ple_np[ids]

  def _write_ple_rows(self, tokens:list[int], start:int, end:int|None=None):
    """the rows of the positions start..end (default all the tokens) into the host buffer"""
    import numpy as np
    ple, end = unwrap(self.config.ple), len(tokens) if end is None else end
    self._ple_init()
    n, width = len(ple.multipliers), len(ple.vocab_sizes) * self._ple_rb
    if end <= start: return
    # the n-gram context of every position: the tokens before it, EOS before the start and at and after an EOS (looking back)
    t = np.concatenate([np.full(max(0, (n-1)-start), ple.eos, dtype=np.int64), np.asarray(tokens[max(0, start-(n-1)):end], dtype=np.int64)])
    P = end - start
    cols = [t[n-1-i:n-1-i+P] for i in range(n)]  # t[p-i] of every position p
    ctx = np.stack(cols, axis=1)
    eos = np.stack([c == ple.eos for c in cols], axis=1)
    # ctx i is EOS if any of t[p-i..p-1] is
    seen = np.logical_or.accumulate(eos[:, 1:], axis=1) if n > 1 else eos[:, :0]
    if n > 1: ctx[:, 1:] = np.where(seen, ple.eos, ctx[:, 1:])
    np.frombuffer(self._ple_host, dtype=np.uint8)[start*width:end*width] = self._ple_rows(self._ple_ids(ctx)).reshape(-1)

  def _ple_dev(self) -> bool:
    """the n-gram rows hashed and gathered on the devices (the table in host memory they read): spec decoding's drafts' rows, run ahead"""
    return self.config.ple is not None and self.ple_table is not None and amd_custom_kernels_supported(self.token_embd.weight.device) and \
      bool((1))

  def _ple_dev_init(self, read:bool=True):
    """
    the table in host memory of the devices (the heads' tables views of it: their rows' bytes are 32 bit offsets), read from the file (direct:
    not kept in the page cache besides, the table is 29GB) or copied from its bytes. read=False: its buffers only, the memory made and read by
    _ple_dev_read (jits that read it can capture before)
    """
    if hasattr(self, "_ple_heads"): return
    ple, devs = unwrap(self.config.ple), to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device))
    self._ple_init()
    rb, table, loc = self._ple_rb, unwrap(self.ple_table), self._ple_file
    # (direct reads: from a block of the file, to an aligned address)
    pre = loc[1] % 4096 if loc is not None else 0
    host = Buffer(devs[0], size:=round_up(pre + table.nbytes, 4096), options=BufferSpec(host=True, cpu_access=True))
    # (the other devices': the host's mappings on them, made with it)
    self._ple_parts, heads, views = [host] + [Buffer(d, size) for d in devs[1:]], [], []
    for o, v in zip(ple.offsets, ple.vocab_sizes):
      views += (vs:=[x.view(v*rb, pre + o*rb) for x in self._ple_parts])
      if len(devs) > 1: (mb:=MultiBuffer.__new__(MultiBuffer)).bufs = vs
      heads.append(Tensor(UOp.from_buffer(mb if len(devs) > 1 else vs[0], dtypes.uint8, devs if len(devs) > 1 else devs[0])))
    self._ple_heads, self._ple_views, self._ple_pre = heads, views, pre
    if read: self._ple_dev_read()

  def _ple_dev_alloc(self):
    """the memory of the table (see _ple_dev_init)"""
    if hasattr(self, "_ple_mv"): return
    import numpy as np
    host, *aliases = self._ple_parts
    host.ensure_allocated()
    for x in aliases: x.allocate(opaque=BufferStorage(a:=host.get_buf(x.device)), external_ptr=a)
    for v in self._ple_views: v.ensure_allocated()
    self._ple_mv, n = host.as_memoryview(allow_zero_copy=True), unwrap(self.ple_table).nbytes
    # (the host's reads of the table: the copy, not the file's pages)
    self._ple_np = np.frombuffer(self._ple_mv, dtype=np.uint8)[self._ple_pre:self._ple_pre + n].reshape(-1, self._ple_rb)

  def _ple_dev_read(self, background:bool=False):
    """the table read into its host memory (see _ple_dev_init). background: by threads (joined by _ple_dev_wait)"""
    if not hasattr(self, "_ple_heads"): self._ple_dev_init(read=False)
    if getattr(self, "_ple_read", False): return
    self._ple_dev_alloc()
    import concurrent.futures
    self._ple_read, table, loc, mv, pre, C = True, unwrap(self.ple_table), self._ple_file, self._ple_mv, self._ple_pre, 64 << 20
    fd = os.open(loc[0], os.O_RDONLY | getattr(os, "O_DIRECT", 0)) if loc is not None else None
    def cp(a:int):
      # (direct: a read is a short one only at the file's end)
      if fd is None: mv[a:a+C] = table[a:a+C]
      else: os.preadv(fd, [mv[a:min(len(mv), a + C)]], unwrap(loc)[1] - pre + a)
    ex = concurrent.futures.ThreadPoolExecutor(16)
    self._ple_copies, self._ple_fd = [ex.submit(cp, a) for a in range(0, len(mv) if fd is not None else table.nbytes, C)], fd
    ex.shutdown(wait=False)
    if not background: self._ple_dev_wait()
  def _ple_dev_wait(self):
    for f in getattr(self, "_ple_copies", []): f.result()
    self._ple_copies = []
    if (fd:=getattr(self, "_ple_fd", None)) is not None:
      # (the table's pages of the file a process before read: the page cache is the weights')
      loc = cast(tuple[str, int, int], self._ple_file)
      if hasattr(os, "posix_fadvise"): os.posix_fadvise(fd, loc[1], loc[2], os.POSIX_FADV_DONTNEED)
      os.close(fd)
      self._ple_fd = None

  def _ple_rows_dev(self, window:Tensor) -> Tensor:
    """
    window (W,) the tokens of positions p0-(n-1).. (EOS before the start): the packed rows (W-(n-1), heads*rb) of the n-grams ending at p0..,
    hashed and gathered on the devices. (G, W): the windows of G sequences, their rows one after another
    """
    ple = unwrap(self.config.ple)
    self._ple_dev_init()
    if window.ndim == 1: window = window.unsqueeze(0)
    n, H, W = len(ple.multipliers), len(ple.vocab_sizes), int(window.shape[1])
    T = W - (n-1)
    cols = [window[:, n-1-i:n-1-i+T].reshape(-1).cast(dtypes.int64) for i in range(n)]  # t[p-i]
    ctx, seen = [cols[0]], None
    for i in range(1, n):
      seen = (cols[i] == ple.eos) if seen is None else seen | (cols[i] == ple.eos)
      ctx.append(seen.where(ple.eos, cols[i]))
    h, hashes = ctx[0] * ple.multipliers[0], []
    for i in range(1, n):
      h = h ^ (ctx[i] * ple.multipliers[i])
      hashes.append(h)
    # (rows of no token (-1, not sampled yet) read row 0: their outputs aren't used)
    ids = Tensor.stack(*[(((hashes[j*(n-1)//H] % v) + v) % v) for j, v in enumerate(ple.vocab_sizes)], dim=1)
    return ple_gather(ids.cast(dtypes.int32), self._ple_heads, self._ple_rb)

  def _ple_emb(self, rows:Tensor) -> Tensor:
    """the packed rows (B, T, width) dequantized on the devices: (B, T, heads*head_dim)"""
    ple = unwrap(self.config.ple)
    B, T = rows.shape[:2]
    rows = rows.to(self.token_embd.weight.device)
    q, nb = _GGML_QUANT[ple.ggml_type]
    return ggml_data_to_tensor(rows.flatten(), int(prod(rows.shape)) // nb * q, ple.ggml_type).reshape(B, T, -1).float()

  def forward(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor|None, base:int|UOp=0, part:int|None=None) -> Tensor:
    """base: the first row of the sequence's region of the caches (and of the token buffer), see regions. part: of a prefill in parts"""
    _seq_base[0] = base
    try: return self._forward(tokens, start_pos, temperature, base, part)
    finally: _seq_base[0] = 0

  # ******** PREFILL_PARTS: a prefill chunk's jit in parts of the blocks (jits captured at once, in processes of their own: a jit's capture
  # is ~80 s of python for all the blocks of two micro batches). the streams of the micro batches between parts in buffers of the model ********

  def prefill_parts(self, rows:int) -> list[tuple[int, int]]:
    """the blocks of the parts of a prefill chunk of rows (one part: not split)"""
    k, n = getenv("PREFILL_PARTS", 1), len(self.blk)
    mb = (1)
    if k <= 1 or self.config.hc is None or not amd_custom_kernels_supported(self.token_embd.weight.device) or \
       not (self.sequence_parallel(rows) and (mb > 1 or (1)) and self.sequence_parallel(rows // mb) and rows % mb == 0):
      return [(0, n)]
    return [(i*n//k, (i+1)*n//k) for i in range(k)]

  def _carry(self, rows:int) -> list[Tensor]:
    """the streams of every micro batch of a prefill chunk of rows between its parts (made before the jits are captured)"""
    carry = self.__dict__.setdefault("_pf_carry", {})
    if rows not in carry:
      hc, mb, devs = unwrap(self.config.hc), (1), cast(tuple[str, ...], self.token_embd.weight.device)
      dt = dtypes.half
      carry[rows] = [Tensor.zeros(1, rows // mb, hc.count, self.config.dim, dtype=dt).shard(devs, axis=1).contiguous().realize() for _ in range(mb)]
    return carry[rows]

  def _forward(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor|None, base:int|UOp, part:int|None=None) -> Tensor:
    # (a view of the token buffer from a variable base: copied, the lookup of the embedding's rows of a view at two variables is a reduce over
    # the vocabulary)
    x = emb = self.token_embd((tokens.contiguous() if isinstance(base, UOp) else tokens).to(self.token_embd.weight.device)).float()  # (B, T, D)
    if _vis_on[0]: x = emb = self._vis_embed(emb)
    if (hc:=self.config.hc) is not None: x = x.unsqueeze(2).expand(*x.shape[:2], hc.count, x.shape[-1]).contiguous()
    ple = None
    if (pc:=self.config.ple) is not None:
      self._ple_init()
      width, R, n = len(pc.vocab_sizes) * self._ple_rb, int(x.max_shape[1]), len(pc.multipliers)
      if self._ple_dev() and (buf:=getattr(self, "_token_buf", None)) is not None and tokens.uop.base is buf.uop.base:
        # (the tokens of the rows and the n-1 before them from the token buffer, EOS before the start)
        win = self._token_window(buf, base, start_pos, R + n-1)
        ple = self._ple_emb(self._ple_rows_dev(win).unsqueeze(0))
      else: ple = self._ple_emb(self._ple_buf.reshape(-1, width).shrink(((start_pos, start_pos + R), None)).unsqueeze(0))
    # a symbolic chunk with hyper-connections runs every block on static padded rows (their real count is T), sliced at the end
    if hc is not None and not isinstance(T:=x.shape[1], int) and amd_custom_kernels_supported(x.device) and \
       all(isinstance(b, (MLATransformerBlock, GatedDeltaNetBlock)) for b in self.blk):
      x = x.pad_to(x.max_shape).contiguous()
      R = int(x.shape[1])
      # sequence parallel: every device keeps the streams of its rows, the blocks gather and scatter the sublayers' inputs and outputs
      # micro batches (on queues of their own, AM_MEC_PIPES: the copies of one run with the kernels of the other)
      mb = (1)
      if (sp:=self.sequence_parallel(R)) and (mb > 1 or (1)) and self.sequence_parallel(R // mb) and R % mb == 0:
        # the blocks in phases (micro_batch_blocks), on micro batches of rows if MICRO_BATCHES > 1 (the copies of one overlap the kernels of
        # the next). a batch starts at the end of the tokens before it (all padding if there are none): its states follow theirs
        n, xs, ts, starts = R // mb, [], [], []
        parts = self.prefill_parts(R)
        a, b = parts[part or 0]
        for i in range(mb):
          if not a: xs.append(shard_rows(x[:, i*n:(i+1)*n].contiguous(), 1))
          ts.append((T - i*n).maximum(0).minimum(n))
          starts.append(start_pos + (T.minimum(i*n) if i else 0))
        if a: xs = list(self._carry(R))
        xs = micro_batch_blocks(self.blk[a:b], xs, starts, ts, ple)
        if b < len(self.blk):
          # (not the last part: the streams to the model's buffers, the output one of them)
          carry = self._carry(R)
          outs = [Tensor(c.uop.after(c.uop.store(xi.cast(c.dtype).uop))) for c, xi in zip(carry, xs)]
          return functools.reduce(lambda a, o: a + o[:, :, :1, :1].float(), outs[1:], outs[0][:, :, :1, :1].float()).contiguous()
        if self.gated:
          # (the streams of all the rows: the MTP layer's input, their last mix the output's)
          if self.config.spec_drafts and (1):
            raise AssertionError("gated speculative prefill: use tinygrad.llm.qwen4exp")

        else: x = Tensor.cat(*[_on_lane(i, all_gather, xi.mean(2), 1) for i, xi in enumerate(xs)], dim=1)
      else:
        # (gated hyper-connections split their rows in the phases only: without them every device has all the rows)
        if sp and self.gated: sp = False
        if sp: x = shard_rows(x, 1)
        for block in self.blk: x = block(x, start_pos, T, ple_emb=ple)
        if self.gated:
          if self.config.spec_drafts: raise AssertionError("gated speculative prefill: use tinygrad.llm.qwen4exp")
        else: x = all_gather(x.mean(2), 1) if sp else self._hc_out(x)
      if self.config.spec_drafts and not self.gated: x = self._mtp_prefill(emb, x, cast(UOp, start_pos), T, sp=bool(sp))
      if x.shape[1] != 1: x = x[:, :T]
    else:
      for block in self.blk: x = block(x, start_pos, ple_emb=ple)
    if hc is not None and x.ndim == 4: x = self._hc_out(x[:, -1:])
    # Only the last token needs the final stream mix and vocabulary projection.
    tok = self._sample(self.output_norm(x[:, -1:])[:, -1, :], temperature, tokens.device)
    q8_quantize_cache_clear()  # don't keep the graph alive
    # the tokens are a view of generate's token buffer: the sampled token goes after them unless that is a prompt token (chunked prefill)
    if (buf:=getattr(self, "_token_buf", None)) is not None and tokens.uop.base is buf.uop.base:
      tok = (dst:=buf[:, (nxt:=base + start_pos + tokens.shape[1]):nxt+1]).assign((dst < 0).where(tok.cast(buf.dtype), dst))
    return tok

  def _token_window(self, buf:Tensor, base:int|UOp, start:int|UOp, n:int) -> Tensor:
    """
    the n tokens (n,) of the sequence of the region at base from position start-(n_gram-1) in the token buffer buf (1, rows): EOS before its
    first (a region's rows before it are another's)
    """
    ple = unwrap(self.config.ple)
    k = len(ple.multipliers) - 1
    win = buf.pad(((0, 0), (k, 0)), value=ple.eos)[0].shrink(((base + start, base + start + n),))
    if isinstance(base, int) and base == 0: return win
    return ((Tensor.arange(n) + Tensor(start - k)).to(buf.device) >= 0).where(win, ple.eos)

  def _sample(self, h:Tensor, temperature:Tensor|None, device:str|tuple[str, ...]|None, noise:tuple[Tensor, Tensor]|None=None) -> Tensor:
    """
    the token (B, 1) after every normed hidden state h (B, D), sampled (greedy without a temperature). on device unless the output is split.
    temperature: (rows, 1) the temperature of every row (or one), or (rows, 2) and its top_p (only the tokens of the top_p nucleus are sampled).
    noise: (seed, keys (B,)): the uniform noise of a row a hash of the seed, its key and the token (see _hash_uniform), not drawn
    """
    logits = self.output(h)
    # Gumbel-max trick: argmax(logits/temp - log(-log(uniform))) is equivalent to sampling from softmax(logits/temp)
    if temperature is not None:
      # (a row of temperature 0: the most likely, not scaled or noised)
      tp = temperature.to(logits.device).reshape(-1, temperature.shape[-1])
      hot = tp[:, :1] > 0
      z = hot.where(logits.float() / tp[:, :1].maximum(1e-6), logits.float())
      u = _hash_uniform(noise[0], noise[1], logits) if noise is not None else Tensor.rand_like(logits)
      # -log(u) at least 1-u (it is): the device's log of a u a few ulps below 1 is 0, its noise an infinity (a random token sampled)
      logits = z - hot.where(u.maximum(1e-12).log().neg().maximum(1 - u).log(), 0)
      if tp.shape[1] >= 3 and SAMPLE_TOP_K:
        # (a row's top_k (0: none) first, as HF's warpers: the nucleus is the top_p of the top k's distribution)
        keep = top_k_mask(z, self.config.vocab_size, tp[:, 2:3]) | hot.logical_not()
        z, logits = keep.where(z, -math.inf), keep.where(logits, -math.inf)
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
    self._spec_h, self._spec_row = Tensor.zeros(2*(k+1), self._spec_dim, device=devs).contiguous().realize(), 0
    self.spec_stats = [0, 0]  # the steps and the tokens they gave
    # in host memory: the step's input tokens (the last step's outputs), its sampled tokens and its drafts (the host polls them)
    self._spec_host, self._spec_in = _host_ints(cast(str|tuple[str, ...], devs), k+1)
    # the outputs are written by one device: the host takes them, resets them and launches the next step once one device wrote them, another
    # one writing them later would overwrite the reset with the last step's
    self._spec_host_out, self._spec_out = _host_ints(devs[0] if isinstance(devs, tuple) else cast(str, devs), 2*k+1)

  def _spec_row0_copy(self, row:UOp, keep:UOp):
    """row 0 of _spec_h: the hidden state of row `row` (keep 1), or zero (keep 0)"""
    self._spec_h[0:1].assign(self._spec_h.shrink(((row, row+1), None)).contiguous() * Tensor(keep).cast(dtypes.float)).realize()
  def _spec_row0(self, row:int):
    """the hidden state of the position before a prompt (row 0 of _spec_h): the one of row `row` (the last accepted token), 0: none (zero)"""
    if not hasattr(self, "_spec_row0_jit"): self._spec_row0_jit = TinyJit(self._spec_row0_copy, capture_first=True)
    k = self.config.spec_drafts
    self._spec_row0_jit(UOp.variable("spec_row", 0, 2*k+1).bind(row), UOp.variable("spec_keep", 0, 1).bind(int(row != 0)))

  def _mtp_prefill(self, emb:Tensor, x:Tensor, start_pos:UOp, T:UOp, sp:bool=False) -> Tensor:
    """
    the MTP layer on a prompt chunk (for its caches): emb (1, T, D) the embeddings of its tokens, x (1, rows, D) the model's (padded) output.
    the hidden state of the position before the chunk is row 0 of _spec_h, the one of its last token is written there. returns x
    """
    assert not self.gated, "gated speculative prefill: use tinygrad.llm.qwen4exp"
    hn = self.output_norm(x).float()
    emb, carry = emb.pad_to((1, x.shape[1], emb.shape[-1])), self._spec_h[0:1].unsqueeze(0).contiguous()
    # the MTP layer reads the carry before it's written (the schedule orders an assign after the reads). realized with the call's output
    inp = self.mtp.enorm(emb).cat(self.mtp.hnorm(carry.cat(hn[:, :-1], dim=1)), dim=-1)
    if sp and (1):
      # sequence parallel (as the blocks): every device its rows (the projection of the input too), the sublayers on all of them
      mtp, h = self.mtp, q8_linear(self.mtp.eh_proj, shard_rows(inp.contiguous(), 1), split=True)
      h = h + mtp._phase_sub("attn", mtp.attn_norm(h), start_pos, T)
      m = h + mtp._phase_sub("ffn", mtp.ffn_norm(h), start_pos, T)
    else: m = self.mtp(self.mtp.eh_proj(inp), start_pos, T)
    # (a column of every row: a row split over the devices isn't sliced)
    effects = [m[:, :, :1].contiguous(), self._spec_h[0:1].assign(hn[0].shrink(((T-1, T), None)))]
    if self._effects is None: Tensor.realize(*effects)
    else: self._effects += effects
    return x

  def forward_spec(self, mtp_pos:UOp, j:UOp, start_pos:UOp, par:UOp, temperature:Tensor|None, *chain:UOp) -> Tensor:
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
    if (hc:=self.config.hc) is not None: x = x.unsqueeze(2).expand(*x.shape[:2], hc.count, x.shape[-1]).contiguous()
    for block in self.blk: x = block(x, start_pos)
    if hc is not None: x = x.mean(2)
    tok = self._sample(self._spec_h.shrink(((par*(k+1), (par+1)*(k+1)), None)).assign(self.output_norm(x)[0].float()), temperature, y.device)
    q8_quantize_cache_clear()  # don't keep the graph alive
    out = tok.reshape(1, k+1).cat(*drafts, dim=1).cast(dtypes.int32)
    return self._spec_out.assign(Tensor(out.uop.mselect(0)) if isinstance(out.device, tuple) else out)

  def _spec_call(self, mtp_pos:int, j:int, par:int, temperature:Tensor|None):
    """a step of speculative decoding (forward_spec, its jit)"""
    k, pos = self.config.spec_drafts, mtp_pos + j
    # select the ordinary/dense-indexer JIT variant for this context
    dense = self._dense_ctx(pos, k + 1)
    name = ("_spec_jit_dense" if dense else "_spec_jit") + ("_greedy" if temperature is None else "")
    if not hasattr(self, name): setattr(self, name, TinyJit(self.forward_spec, capture_first=getattr(self, "_states_ready", False),
                                                            scratch=self._scratch))
    jit = getattr(self, name)
    if not hasattr(self, "_spec_vars"):
      # (made once: a step binds them)
      pos_var = functools.partial(UOp.variable, min_val=0, max_val=self.max_context-1)
      self._spec_vars = (pos_var("mtp_pos"), UOp.variable("spec_j", 0, k), pos_var("start_pos"), UOp.variable("spec_par", 0, 1),
                         tuple(pos_var(f"spec_pos{i}") for i in range(k-1)))
    v_mtp, v_j, v_pos, v_par, v_chain = self._spec_vars
    jit(v_mtp.bind(mtp_pos), v_j.bind(j), v_pos.bind(pos), v_par.bind(par), temperature, *(v.bind(pos+1+i) for i,v in enumerate(v_chain)))

  def _wait_host(self, get:Callable[[], Any], what:str) -> Any:
    """the value a step writes to host memory (polled: -1 until then). a step that doesn't come (the devices' errors): an error, not a hang"""
    st = last = time.perf_counter()
    while isinstance(v:=get(), int) and v < 0:
      last = time.perf_counter()
      if last - st > 10:
        for d in to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device)): Device[d].synchronize()
        if isinstance(v:=get(), int) and v < 0: raise RuntimeError(f"{what}: none after the devices finished")
        break
    self.step_times[2] += time.perf_counter() - last  # (the last poll's: how late it's seen at most)
    return v

  def _spec_steps(self, tokens:list[int], temperature:Tensor|None) -> Iterator[int]:
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
      # (n-gram rows on the devices: the token buffer has every accepted token before the next step reads it)
      if self.config.ple is not None:
        for i in range(j+1): self._token_host[len(tokens) + i] = sampled[i]
      self.spec_stats[0] += 1
      self.spec_stats[1] += j+1
      # run ahead: the next step is launched before the tokens are taken (the states it writes don't overwrite the ones of their positions)
      launched, row, mtp_pos, par = launch(pos + 1, j, 1-par), par*(k+1), pos + 1, 1-par
      for i in range(j+1):
        tokens.append(sampled[i])
        # the caches hold every token but the last: its hidden state is the verified row i
        self._cached_tokens, self._spec_row = tokens[:-1], row + i
        yield sampled[i]

  # ******** sequences sharing caches: each has a region of attention-cache rows (and the token buffer) plus a slot for
  # the recurrent states. a sequence is prefilled by the model's calls on its region (its states loaded from its slot before, stored after),
  # a step of speculative decoding takes every sequence decoding at once (forward_seqs). the regions are the caller's (see SeqEngine) ********

  # (GLM: the steps of 3 and 6 sequences their own sizes: a step of n sequences runs the jit of the smallest size n fits (its padding
  # sequences are rows computed for nothing): 3 sequences 22 ms, not 26 (the jit of 4))
  SEQ_SIZES = tuple(int(x) for x in str(getenv("SEQ_SIZES", "1,2,3,4,6,8")).split(","))
  REGION_ALIGN = REGION_ALIGN  # (the alignment of the regions, see SeqEngine)
  def drafts_of(self, m:int) -> int:
    """the drafts of a step of m sequences (SPEC_DRAFTS_BY_SIZE, see spec_by_size): the layouts of the states are spec_drafts' (the most)"""
    return dict(getattr(self.config, "spec_by_size", ())).get(m, self.config.spec_drafts)
  def seqs_supported(self) -> bool:
    devs = self.token_embd.weight.device
    # (GLM (GLM_SEQ): MLA with the sparse indexer on a sequence sharded cache, speculative decoding)
    glm = bool(self.config.cache_rows and self.config.kv_lora_rank and seq_sharded(devs))
    return bool(glm and self.config.spec_drafts and self.config.indexer is not None and
                isinstance(devs, tuple) and amd_custom_kernels_supported(devs))
  def seq_drop_prefill(self) -> dict:
    """
    the prefill chunks' jits dropped and the jits' shared arenas (_scratch) shrunk to what the other jits use: a process that prefills no more
    (the board harness after its donors: the arenas a 2304-row chunk grew stay with the decode jits otherwise). a later prefill captures its
    jit again. no numerics change: a jit's buffers keep their offsets in a smaller arena. returns {arena key: (bytes before, after)}
    """
    from tinygrad.engine.jit import shrink_scratch
    for k in [k for k in self.__dict__ if k.startswith("jits")] + ["_pf_carry"]: self.__dict__.pop(k, None)
    gc.collect()
    return shrink_scratch(self._scratch)
  def seq_slots(self) -> int:
    """the slots of the sequences (the one after them is the padding sequences' of a step)"""
    # (GLM: 16, the steps' sizes up to 8: more conversations resident than decoded at once)
    return getenv("SEQ_SLOTS", 16)
  def _token_rows(self) -> int:
    """the token buffer: a token of every row of the caches (the regions of the sequences sharing them), and one past"""
    return max(self.max_context, self.config.cache_rows) + self.REGION_ALIGN + 1
  def seq_rows(self) -> int:
    """the rows of the caches the regions of the sequences are of (the REGION_ALIGN after them: the padding sequences' region)"""
    return max(self.max_context, self.config.cache_rows)

  def _seq_init(self):
    if hasattr(self, "seq_spec_h"): return
    # (the buffers freed by loading are cached for reuse: their memory is the caches')
    for d in to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device)): Device[d].allocator.free_cache()
    self._init_states()
    k, n, devs, m = self.config.spec_drafts, self.seq_slots() + 1, cast(tuple[str, ...], self.token_embd.weight.device), max(self.SEQ_SIZES)
    # (the slots' recurrent states in float, SEQ_STATE_HALF: in half)
    for b in self.blk:
      if isinstance(b, GatedDeltaNetBlock): b._init_slots(n, 0, half=bool((0)), replay=bool(SEQ_REPLAY))
    # (GLM: the indexer's ring of every slot, the MLA caches the regions')
    for b in self.blk + [self.mtp]:
      if isinstance(b, MLATransformerBlock): b._init_seq_slots(n)
    # every slot's hidden states (the streams) of the last step's tokens, two parities of k+1
    self.seq_spec_h = _zeros(n*2*(k+1), self._spec_dim, device=devs)
    # every slot's checkpoint: its states of a position (the end of its last prompt), the hidden state of the last token
    self.seq_ckpt_h = _zeros(n, self._spec_dim, device=devs)
    # (the checkpoints' recurrent states in host memory: a copy a turn, the device memory more slots')
    self.seq_ckpt_states = [[_slot_pool(n, b.pool_rec, (1, *b.pool_rec.shape[1:]), host=True)[0]] +
                            [_slot_pool(n, p, (1, *p.shape[1:]))[0] for p in b.pool_conv] for b in self.blk if isinstance(b, GatedDeltaNetBlock)]
    # (GLM: the checkpoints' indexer rings (the last tokens of the prompt, its last pool's), see _seq_ckpt_copy)
    self.seq_ckpt_rings = [(b.indexer, _slot_pool(n, b.indexer.ring)[0]) for b in self.blk + [self.mtp]
                           if isinstance(b, MLATransformerBlock) and b.indexer is not None]
    self._spec_init()
    if not hasattr(self, "_token_buf"): self._token_host, self._token_buf = _host_ints(devs, self._token_rows())
    # in host memory: a step's input tokens, the temperature and top_p of every sequence, its sampled tokens and drafts (written by one device)
    self._seq_host_in, self._seq_in = _host_ints(devs, m*(k+1))
    if self.vision: self._vis_init(devs)
    # (a mixed step's prompt's at 2m, the step's seed last)
    self._seq_m = m
    tmv, self._seq_temp = _host_buffer(devs, 4*TEMP_W*(m + 1) + 4, dtypes.float32)
    self._seq_host_temp = tmv.cast('f')
    # (a prefill's temperature and top_p: in host memory, a new value isn't a new tensor (its realize walks every tensor of the model))
    tmv, self._seq_pf_temp = _host_buffer(devs, 4*TEMP_W, dtypes.float32)
    self._seq_pf_host_temp = tmv.cast('f')
    self._seq_host_out, self._seq_out = _host_ints(devs[0] if isinstance(devs, tuple) else devs, m*(2*k+1) + 1)
    # (SEQ_COPY_SCRATCH=1: their intermediates in the jits' shared arenas, not their own (~140 MB a card at 30 slots): one runs at a time)
    sc:dict|None = self._scratch if getenv("SEQ_COPY_SCRATCH", 0) else None
    self._seq_copy_jits = {sv: TinyJit(functools.partial(self._seq_copy, sv), capture_first=True, scratch=sc) for sv in (True, False)}
    self._seq_ckpt_jits = {sv: TinyJit(functools.partial(self._seq_ckpt_copy, sv), capture_first=True, scratch=sc) for sv in (True, False)}
    # (SEQ_REPLAY: a slot's recurrent states advanced to a position, see _seq_materialize; the position of every slot's state)
    self._seq_mat_jit = TinyJit(self._seq_mat, capture_first=True, scratch=sc) if SEQ_REPLAY else None
    self._seq_kbase:dict[int, int] = {}
    # (the calls' chunk counts that fit the caches: a short max context's are a few chunks)
    self._move_calls = self._chunk_calls((self.seq_rows() + self.REGION_ALIGN) // self.SEQ_MOVE - 1)
    self._seq_move_jits = {m: TinyJit(functools.partial(self._seq_move_copy, m), capture_first=True) for m in self._move_calls}
    self._seq_stash_init()
    todo = [self.seq_spec_h, self.seq_ckpt_h] + [p for b in self.blk if isinstance(b, GatedDeltaNetBlock) for p in (b.pool_rec, *b.pool_conv)] + \
      [b.indexer.pool_ring for b in self.blk + [self.mtp] if isinstance(b, MLATransformerBlock) and b.indexer is not None] + \
      [b.pool_rp for b in self.blk if isinstance(b, GatedDeltaNetBlock) and hasattr(b, "pool_rp")] + \
      [t for c in self.seq_ckpt_states for t in c] + [r for _, r in self.seq_ckpt_rings]
    _fill_zero(*todo)
    # (GLM: the MLA caches' peer views made (allocated) before jits are captured in forked processes, see seq_gather)
    for b in self.blk + [self.mtp]:
      if isinstance(b, MLATransformerBlock) and b.seq_sharded: seq_views(b.cache_s)

  def _seq_copy(self, save:bool, slot:UOp, pm:UOp, ps:UOp, pc:UOp, hrow:UOp, keep:UOp):
    """
    the recurrent states of the model of a position (its rows pm) into slot `slot` (its rows ps of the recurrent states, pc of the conv ones) and
    the hidden state of the last token (the model's row 0) into the slot's row hrow (save), or back (the hidden state times keep: 0 none)
    """
    def rows(t:Tensor, start:UOp|int) -> Tensor: return t.shrink(((start, start+1),) + (None,)*(t.ndim-1))
    H = 2*(self.config.spec_drafts+1)
    pairs:list[tuple[Tensor, Tensor]] = [(rows(self.seq_spec_h, slot*H + hrow), rows(self._spec_h, 0))]
    # (save, a prompt's (hrow 0): the rest of the rows of its first step zeroed, as a fresh model's: the MTP layer's catch-up reads them (its
    # cache rows after the prompt's, the next steps' drafts), an earlier sequence's leftovers moved C6's greedy tokens after the warmup)
    if save and (k:=self.config.spec_drafts) and (1):
      pairs.append(((d:=self.seq_spec_h.shrink(((slot*H + 1, slot*H + k + 1), None))), d.zeros_like()))
    for b in self.blk:
      if isinstance(b, GatedDeltaNetBlock):
        S, SC = int(b.slot_rec[0].shape[0]), int(b.slot_conv[0][0].shape[0])
        pairs.append((rows(b.pool_rec, slot*S + ps), rows(b.recurrent_state, pm)))
        pairs += [(rows(d, slot*SC + pc), rows(c, pm)) for d, c in zip(b.pool_conv, b.conv_states)]
    # (GLM: the indexer's ring of every MLA block (its last tokens, the pool a prompt ends in), all its rows)
    for b in self.blk + [self.mtp]:
      if isinstance(b, MLATransformerBlock) and (idx:=b.indexer) is not None:
        R = int(idx.ring.shape[0])
        pairs.append((idx.pool_ring.shrink(((slot*R, slot*R + R),) + (None,)*(idx.ring.ndim-1)), idx.ring))
    if save: Tensor.realize(*[d.assign(s.cast(d.dtype)) for d, s in pairs])
    else:
      (h, hs), rest = pairs[0], pairs[1:]
      Tensor.realize(hs.assign(h.contiguous() * Tensor(keep).cast(dtypes.float)), *[s.assign(d.cast(s.dtype)) for d, s in rest])

  def _seq_mat(self, slot:UOp, base:UOp, n:UOp):
    """the recurrent states of slot `slot` (SEQ_REPLAY: of position base) advanced n tokens (the replayed inputs of its last step)"""
    T = self.config.spec_drafts + 1
    outs = [kda_replay(b.pool_rec, b.pool_rp, slot, base, n, T, replay_q(T-1)) for b in self.blk if isinstance(b, GatedDeltaNetBlock)]
    Tensor.realize(*outs)
  def _mat_vars(self, slot:int, base:int, n:int) -> list[UOp]:
    return [UOp.variable("mat_slot", 0, self.seq_slots()).bind(slot), UOp.variable("mat_base", 0, self.max_context).bind(base),
            UOp.variable("mat_n", 0, self.config.spec_drafts).bind(n)]
  def _seq_materialize(self, slot:int, pos:int):
    """SEQ_REPLAY: slot `slot`'s recurrent states of position pos (its state's and the replayed tokens after it) made its state"""
    if self._seq_mat_jit is None or (base:=self._seq_kbase.get(slot)) is None or base == pos: return
    # (a position its state isn't a few tokens before (a slot not decoded to it): its state as it is, as the states of other positions are)
    if 0 < pos - base <= self.config.spec_drafts: self._seq_mat_jit(*self._mat_vars(slot, base, pos - base))
    self._seq_kbase[slot] = pos

  def _seq_copy_call(self, save:bool, slot:int, pos:int, hrow:int, keep:bool=True):
    k = self.config.spec_drafts
    S, SC = self._seq_S()
    if not save and keep: self._seq_materialize(slot, pos)
    Sm = self.config.rewind + 1
    if SEQ_REPLAY and save: self._seq_kbase[slot] = pos
    self._seq_copy_jits[save](UOp.variable("seq_slot", 0, self.seq_slots()).bind(slot), UOp.variable("seq_pm", 0, Sm-1).bind(pos % Sm),
                              UOp.variable("seq_ps", 0, S-1).bind(pos % S), UOp.variable("seq_pc", 0, SC-1).bind(pos % SC),
                              UOp.variable("seq_hrow", 0, 2*k+1).bind(hrow), UOp.variable("seq_keep", 0, 1).bind(int(keep)))

  def _seq_move_copy(self, chunks:int, src:UOp, dst:UOp):
    """chunks*SEQ_MOVE rows of every attention cache from row src to row dst (the indexer's pooled keys: their pools)"""
    n, copies = self.SEQ_MOVE * chunks, []
    def mv(t:Tensor, s:UOp, d:UOp, m:int, axis:int) -> Tensor:
      def rng(x:UOp) -> tuple: return tuple((x, x+m) if i == axis else None for i in range(t.ndim))
      return t.shrink(rng(d)).assign(t.shrink(rng(s)).contiguous())
    for b in self.blk + [self.mtp]:
      if isinstance(b, MLATransformerBlock) and b.seq_sharded:
        # (GLM: a region's rows of the MLA cache on every device (its rows from base/devices), the indexer's pools)
        nd = len(cast(tuple, b.cache_s.device))
        copies.append(mv(b.slot_rows_view(b.cache_s), src // nd, dst // nd, n // nd, 0))
        if (gi:=b.indexer) is not None:
          # (SEQ_IDX_SHARD: the pools split as the cache's rows, a device's local ones)
          pk, kd = (b.slot_rows_view(gi.cache_pk), gi.idx.kpool * nd) if gi.shard else (gi.cache_pk, gi.idx.kpool)
          copies.append(mv(pk, src // kd, dst // kd, n // kd, 0))
    Tensor.realize(*copies)
  SEQ_MOVE = REGION_ALIGN
  def seq_move(self, src:int, dst:int, n:int):
    """the first n positions of the sequence of the region at src moved to the region at dst (not overlapping)"""
    v = functools.partial(UOp.variable, min_val=0, max_val=self.seq_rows())
    # (calls of 16, 4 or 1 chunks. up: the last rows first, the source rows a call writes over are moved already)
    calls, at, left = [], 0, -(-n // self.SEQ_MOVE)
    while left:
      m = next(m for m in self._move_calls if m <= left)
      calls.append((at, m))
      at, left = at + m * self.SEQ_MOVE, left - m
    for o, m in (reversed(calls) if dst > src else calls):
      self._seq_move_jits[m](v("move_src").bind(src + o), v("move_dst").bind(dst + o))
    self._token_host[dst:dst+n+1] = array.array('i', self._token_host[src:src+n+1])

  # ** the stash: sequences evicted from the caches kept in host memory, back when a prompt continues one
  def _seq_stash_init(self):
    """
    SEQ_STASH_GB (on every device): host memory of chunks of SEQ_MOVE rows of every attention cache and of entries of a position's states
    (made, allocated when used). none: an evicted sequence is prefilled again
    """
    C, gb = self.SEQ_MOVE, getenv("SEQ_STASH_GB", 4.0)
    if self.config.cache_rows and self.config.kv_lora_rank > 0:
      return self._seq_stash_init_glm(C, gb)
    # Other model families have no stash: an evicted sequence is prefilled again.
    self.stash_entries = self.stash_chunks = self.stash_chunk_nbytes = self.stash_entry_nbytes = 0
  def _seq_stash_init_glm(self, C:int, gb:float):
    """GLM's region stash (GLM_SEQ): position-sharded latent rows, replicated index pools, all recurrent/checkpoint/replay state."""
    devs = cast(tuple[str, ...], self.token_embd.weight.device)
    nd = len(devs)
    assert nd and C % nd == 0, f"SEQ_MOVE={C} must be a multiple of the {nd} devices"
    def per_row(t:Tensor, axis:int=0) -> int:
      return t.nbytes() // int(t.shape[axis]) // (len(t.device) if isinstance(t.device, tuple) and t.uop.axis is not None else 1)
    kb = [b for b in self.blk + [self.mtp] if isinstance(b, MLATransformerBlock)]
    def index(b:MLATransformerBlock) -> Indexer:
      assert b.indexer is not None, "GLM stash requires indexed pools"
      return b.indexer
    gdn = [b for b in self.blk if isinstance(b, GatedDeltaNetBlock)]
    ring = [r for _, r in self.seq_ckpt_rings]
    replay = [b.pool_rp for b in gdn if hasattr(b, "pool_rp")]
    # (a card's bytes of C positions: its latent rows, its pools (split over the cards (SEQ_IDX_SHARD): a card's part of every pool's))
    chunk = sum(C * per_row(b.cache_s) + C * per_row(index(b).cache_pk) // index(b).idx.kpool for b in kb)
    entry = sum(per_row(t) for b in gdn for t in (b.pool_rec, *b.pool_conv)) + per_row(self.seq_spec_h) + \
      sum(per_row(t) for c in self.seq_ckpt_states for t in c) + per_row(self.seq_ckpt_h) + \
      sum(per_row(index(b).pool_ring) + per_row(r) for b, r in zip(kb, ring)) + sum(per_row(t) for t in replay)
    self.stash_entries = getenv("SEQ_STASH_ENTRIES", 32) if gb > 0 and chunk > 0 else 0
    self.stash_chunks = max(0, int(gb * 1e9 - self.stash_entries * entry) // chunk) if self.stash_entries else 0
    if not self.stash_chunks: self.stash_entries = 0
    self.stash_chunk_nbytes, self.stash_entry_nbytes = chunk, entry
    if not self.stash_entries: return
    # Latents and split indices use seq_move's (local rows, devices, features) view on every card.
    # A device saves/restores only its own index strip, exactly the ownership used by the attention kernels.
    self._stash_kv = []
    for b in kb:
      gi = index(b)
      init_latent = b.slot_rows_view(b.cache_s)
      init_pk = b.slot_rows_view(gi.cache_pk) if gi.shard else gi.cache_pk
      pk_div = gi.idx.kpool * (nd if gi.shard else 1)
      latent = _host_like([(init_latent, 0, self.stash_chunks * C // nd)])[0]
      pools = _host_like([(init_pk, 0, self.stash_chunks * C // pk_div)])[0]
      self._stash_kv.append((latent, pools))
    # Whole ring/replay windows, not one position: every row after a checkpoint restore must have the same state as before eviction.
    self._stash_rings = list(self.seq_ckpt_rings)
    self._stash_replays = replay
    states = [t for b in gdn for t in (b.pool_rec, *b.pool_conv)] + [self.seq_spec_h] + \
             [t for c in self.seq_ckpt_states + [[self.seq_ckpt_h]] for t in c]
    extra:list[Tensor] = []
    for idx, ck in self._stash_rings:
      for t in (idx.pool_ring, ck): extra.append(_host_like([(t, 0, self.stash_entries * int(idx.ring.shape[0]))])[0])
    for t in replay: extra.append(_host_like([(t, 0, self.stash_entries * replay_q(self.config.spec_drafts))])[0])
    self._stash_state = _host_like([(t, 0, self.stash_entries) for t in states]) + extra
    self._stash_state_count = len(states)
    self._stash_calls = self._chunk_calls(min((self.seq_rows() + self.REGION_ALIGN) // C - 1, self.stash_chunks))
    self._stash_kv_jits = {(sv, n): TinyJit(functools.partial(self._stash_kv_copy, sv, n), capture_first=True)
                           for sv in (True, False) for n in self._stash_calls}
    self._stash_state_jits = {sv: TinyJit(functools.partial(self._stash_state_copy, sv), capture_first=True) for sv in (True, False)}

  STASH_CALL = (16, 4, 1)  # (the chunks a call of the stash copies: a run is copied by calls of these, the most first)
  def _chunk_calls(self, most:int) -> tuple[int, ...]:
    """the STASH_CALL counts of chunks of at most `most` (1 always)"""
    return tuple(m for m in self.STASH_CALL if m <= max(1, most))
  def _stash_kv_copy(self, save:bool, chunks:int, row:UOp, chunk:UOp):
    """chunks*SEQ_MOVE rows from row of every attention cache into the stash from chunk (save), or back (static sizes: wide loads and stores)"""
    C, pairs, saves = self.SEQ_MOVE, [], []
    devs = cast(tuple[str, ...], self.token_embd.weight.device)
    def rows(t:Tensor, at:UOp, m:int, axis:int) -> Tensor: return t.shrink(tuple((at, at+m) if i == axis else None for i in range(t.ndim)))
    if self.config.cache_rows and self.config.kv_lora_rank > 0:
      nd = len(devs)
      for gb, st in zip([b for b in self.blk + [self.mtp] if isinstance(b, MLATransformerBlock)], self._stash_kv):
        assert gb.indexer is not None
        glm_idx, kd = gb.indexer, gb.indexer.idx.kpool
        pk = gb.slot_rows_view(glm_idx.cache_pk) if glm_idx.shard else glm_idx.cache_pk
        if glm_idx.shard: kd *= nd
        # (the stash's rows from its chunk, the caches' from the region's row: a sequence comes back to another region)
        items = [(rows(st[0], chunk*(C//nd), C*chunks//nd, 0), rows(gb.slot_rows_view(gb.cache_s), row//nd, C*chunks//nd, 0)),
                 (rows(st[1], chunk*(C//kd), C*chunks//kd, 0), rows(pk, row//kd, C*chunks//kd, 0))]
        for dst, src in items:
          # The host and device have identical region-relative row layout; the grouped kernels' absolute base stays out of this copy.
          if save: saves.append(dst.assign(src))
          else: pairs.append((src, dst))
      # (a restore's copy kernels read the host rows: no device copy of them first (a stash call's rows of every cache: ~126 MB on C5))
      Tensor.realize(*saves, *[x.assign(d) for x, d in pairs])
      return
  def _stash_state_copy(self, save:bool, slot:UOp, entry:UOp, ps:UOp, pc:UOp, hrow:UOp):
    """the states of slot of a position (its rows ps of the recurrent states, pc of the conv ones, hrow of the hidden states) and its checkpoint
    into entry of the stash (save), or back"""
    def rows(t:Tensor, at:UOp) -> Tensor: return t.shrink(((at, at+1),) + (None,)*(t.ndim-1))
    H, slots = 2*(self.config.spec_drafts+1), []
    for b in self.blk:
      if isinstance(b, GatedDeltaNetBlock):
        S, SC = int(b.slot_rec[0].shape[0]), int(b.slot_conv[0][0].shape[0])
        slots += [rows(b.pool_rec, slot*S + ps)] + [rows(p, slot*SC + pc) for p in b.pool_conv]
    slots.append(rows(self.seq_spec_h, slot*H + hrow))
    slots += [rows(t, slot) for c in self.seq_ckpt_states + [[self.seq_ckpt_h]] for t in c]
    pairs = [(rows(st, entry), x) for st, x in zip(self._stash_state[:self._stash_state_count], slots)]
    for gi, (idx, ck) in enumerate(self._stash_rings):
      curr = self._stash_state_count + 2*gi
      R = int(idx.ring.shape[0])
      full = lambda t, start: t.shrink(((start, start+R),) + (None,)*(t.ndim-1))  # noqa: E731
      pairs += [(full(self._stash_state[curr], entry*R), full(idx.pool_ring, slot*R)),
                (full(self._stash_state[curr+1], entry*R), full(ck, slot*R))]
    for i, replay in enumerate(self._stash_replays):
      pos, R = self._stash_state_count + 2*len(self._stash_rings) + i, replay_q(self.config.spec_drafts)
      full = lambda t, start: t.shrink(((start, start+R),) + (None,)*(t.ndim-1))  # noqa: E731
      pairs.append((full(self._stash_state[pos], entry*R), full(replay, slot*R)))
    Tensor.realize(*[(d.assign(x) if save else x.assign(d)) for d, x in pairs])
  def _stash_state_vars(self, slot:int, entry:int, pos:int, hrow:int) -> list[UOp]:
    k = self.config.spec_drafts
    S, SC = self._seq_S()
    return [UOp.variable("stash_slot", 0, self.seq_slots()).bind(slot), UOp.variable("stash_entry", 0, self.stash_entries - 1).bind(entry),
            UOp.variable("stash_ps", 0, S-1).bind(pos % S), UOp.variable("stash_pc", 0, SC-1).bind(pos % SC),
            UOp.variable("stash_hrow", 0, 2*k+1).bind(hrow)]
  def _stash_kv_vars(self, row:int, chunk:int) -> list[UOp]:
    return [UOp.variable("stash_row", 0, self.seq_rows()).bind(row), UOp.variable("stash_chunk", 0, self.stash_chunks - 1).bind(chunk)]
  def seq_stash(self, save:bool, slot:int, base:int, segs:list[tuple[int, int, int]], entry:int, pos:int, hrow:int):
    """
    the sequence of slot (its region at base) of pos tokens into the stash: its segments of chunks (a chunk SEQ_MOVE rows) (its chunk, the
    stash's, count) and entry (its states of pos, its hidden state row hrow), or back
    """
    for i, c, n in segs:
      while n:
        m = next(m for m in self._stash_calls if m <= n)
        self._stash_kv_jits[(save, m)](*self._stash_kv_vars(base + i * self.SEQ_MOVE, c))
        i, c, n = i + m, c + m, n - m
    if save: self._seq_materialize(slot, pos)
    elif SEQ_REPLAY: self._seq_kbase[slot] = pos
    self._stash_state_jits[save](*self._stash_state_vars(slot, entry, pos, hrow))

  def _seq_ckpt_copy(self, save:bool, slot:UOp, ps:UOp, pc:UOp):
    """the checkpoint of slot `slot`: its recurrent states (rows ps, the conv ones pc) and its hidden state of row 0 (save), or back"""
    def rows(t:Tensor, start:UOp|int) -> Tensor: return t.shrink(((start, start+1),) + (None,)*(t.ndim-1))
    H = 2*(self.config.spec_drafts+1)
    pairs:list[tuple[Tensor, Tensor]] = [(rows(self.seq_ckpt_h, slot), rows(self.seq_spec_h, slot*H))]
    for b, c in zip([b for b in self.blk if isinstance(b, GatedDeltaNetBlock)], self.seq_ckpt_states):
      S, SC = int(b.slot_rec[0].shape[0]), int(b.slot_conv[0][0].shape[0])
      pairs += [(rows(c[0], slot), rows(b.pool_rec, slot*S + ps))] + [(rows(d, slot), rows(p, slot*SC + pc)) for d, p in zip(c[1:], b.pool_conv)]
    for gi, ck in self.seq_ckpt_rings:
      R = int(gi.ring.shape[0])
      def ring(t:Tensor) -> Tensor: return t.shrink(((slot*R, slot*R + R),) + (None,)*(t.ndim-1))
      pairs.append((ring(ck), ring(gi.pool_ring)))
    Tensor.realize(*[(d.assign(s) if save else s.assign(d)) for d, s in pairs])
  def _seq_S(self) -> tuple[int, int]:
    """the S of a sequence slot's recurrent states (one with SEQ_REPLAY) and of its conv states"""
    S, SC = slot_states(self.config.spec_drafts)
    return (1 if SEQ_REPLAY else S), SC
  def seq_ckpt(self, save:bool, slot:int, pos:int):
    """the checkpoint of slot `slot`: its states of position pos (and the hidden state of row 0) saved, or restored"""
    S, SC = self._seq_S()
    if save: self._seq_materialize(slot, pos)
    elif SEQ_REPLAY: self._seq_kbase[slot] = pos
    self._seq_ckpt_jits[save](UOp.variable("seq_slot", 0, self.seq_slots()).bind(slot), UOp.variable("seq_ps", 0, S-1).bind(pos % S),
                              UOp.variable("seq_pc", 0, SC-1).bind(pos % SC))

  def _prefill_jit(self, n:int, dense:bool, vis:bool=False) -> Any:
    """the jit of a prefill chunk of n tokens on a sequence's region (dense: a context the indexer's top_k covers; vis: VISION's variant), see
    __call__"""
    jits = self.__dict__.setdefault("jits" + ("_dense" if dense else "") + "_seq" + ("_vis" if vis else ""), {})
    if n not in jits: jits[n] = self._new_jit(n)
    return jits[n]

  def _seq_shared(self, read:bool=True):
    """the resources the jits share, made before they're captured (or loaded): the random state of the devices, the n-gram tables, the
    collectives' mailboxes, the argmax's counts"""
    devs = cast(tuple[str, ...], self.token_embd.weight.device)
    # (one realize: one walks every tensor)
    Tensor.realize(*[x for d in devs for x in (*Tensor._next_counter(d, 0), Tensor._device_rng_counters[d])])
    if self.config.ple is not None: self._ple_dev_init(read)
    # (a lane of the collectives a micro batch, and its compute queue (MB_QUEUES), see micro_batch_blocks)
    for lane in range((1)):
      if mb_queue(lane) and (1):
        for d in devs: getattr(Device[d], "compute_queue_n", lambda i: None)(mb_queue(lane))
      # (and the decoding steps' lanes, see DEC_LANE)
      for ln in (lane, lane + SEL_LANE) + ((DEC_LANE, DEC_LANE + SEL_LANE) if lane == 0 and (1) else ()) + \
                ((IDX_LANE,) if lane == 0 and getenv("SEQ_IDX_SHARD", 0) else ()):
        FLAG_LANE[0] = ln
        for g in (4, 16, 64): flag_allreduce(devs, g)
        # (the ring's mailbox: made before the jits are captured (in a fork, it'd be the child's))
        ring_mailbox(devs)
    FLAG_LANE[0] = 0
    # (the ring collectives' queue of their own, see ring_queue)
    for lane in range((1)):
      if ring_queue_of(lane, any_time=True):
        for d in devs: getattr(Device[d], "compute_queue_n", lambda i: None)(ring_queue_of(lane, any_time=True))
    p2p_allreduce(devs), host_allreduce(devs), _device_index(devs)
    _argmax_parts(devs, (self.config.padded_vocab_size or self.config.vocab_size) // len(devs), self.config.vocab_size)

  def seq_warmup(self):
    """
    every jit of the sequences sharing the caches (a prefill of every chunk size, a context the indexer's top_k covers and longer, a step of
    every size, the copies of the slots' states and of a region), captured and run once (linked)
    """
    # the memory of everything the jits read made first (the derived weights', the states'), the jits compiled while the weights are read and
    # what's computed from them, the states zeroed, the n-gram table copied
    st = time.perf_counter()
    _lap("seq_warmup")
    self._derive_weights()
    with _zero_deferred() as zero_bufs: self._seq_init()
    self._seq_shared(read=False)
    # (the computations from the weights and the zeroing of the states: jits compiled with the others, run when the weights are read)
    loads = [(j, j) for j in self._pending_load if isinstance(j, _TinyJit)] + [_zero_jit(dedup(zero_bufs))]
    if DEBUG >= 1: print(f"seq_warmup: memory made at {time.perf_counter()-st:.2f}s")
    ran:set[str] = set()
    def meanwhile(take:Callable[..., bool|None]):
      # the files' reads (the weights', then the n-gram table's while the values are computed from the weights: not at once, the reads of
      # both are slower than one after the other)
      while self._pending_load and not isinstance(self._pending_load[0], _TinyJit): self._pending_load.pop(0)()
      if self.config.ple is not None: self._ple_dev_read(background=True)
      if DEBUG >= 1: print(f"seq_warmup: read at {time.perf_counter()-st:.2f}s")
      for j, call in loads:
        if j in self._pending_load: self._pending_load.remove(j)
        take(j)
        call()
      self._finish_load()
      if DEBUG >= 1: print(f"seq_warmup: weights and zeros at {time.perf_counter()-st:.2f}s")
      self._ple_dev_wait()
      if DEBUG >= 1: print(f"seq_warmup: loaded at {time.perf_counter()-st:.2f}s")
      # (the warm-up's runs as their jits come: while the others compile)
      run_as_they_come(take, calls, ran)
    top, k = unwrap(self.config.indexer).top_k, self.config.spec_drafts
    calls:dict[str, tuple[Any, Callable[[], Any]]] = {}
    def add(name:str, jit:Any, call:Callable[[], Any]):
      if all(j is not jit for j, _ in calls.values()): calls[name] = (jit, call)
    # every chunk size's jits: of a context top_k covers (its fewest tokens from the start: a piece of a prompt it's the size for) and longer
    chunks = [(c, min(c, top)) for c in sorted(self.prefill_chunks(), reverse=True)]
    for c, n in chunks:
      if self._dense_ctx(0, n) or top + c > self.seq_rows():
        add(f"prefill{c}", self._prefill_jit(c, self._dense_ctx(0, n)), functools.partial(self.seq_prefill, 0, 0, [0]*n, 0, 0))
      if top + c <= self.seq_rows():
        add(f"prefill{c}_sparse", self._prefill_jit(c, False), functools.partial(self.seq_prefill, 0, 0, [0]*(top+c), top, 0))
    if self.vision:
      # (VISION: every prefill jit's vision variant too, a chunk of text rows run through it)
      novis:tuple[list, dict] = ([], {})
      for c, n in chunks:
        if self._dense_ctx(0, n) or top + c > self.seq_rows():
          add(f"vis_prefill{c}", self._prefill_jit(c, self._dense_ctx(0, n), vis=True), functools.partial(self._seq_chunk, 0, 0, n, vis=novis))
        if top + c <= self.seq_rows():
          add(f"vis_prefill{c}_sparse", self._prefill_jit(c, False, vis=True), functools.partial(self._seq_chunk, 0, top, c, vis=novis))
    for m in sorted(self.SEQ_SIZES, reverse=True):
      add(f"step{m}", self._seq_jit(m), functools.partial(self.seq_step, [(0, 0, 1, 0, 1, [0]*(k+1), 0.0)] * m))
      for d in self._step_drafts(m)[1:]:
        add(f"step{m}k{d}", self._seq_jit(m, k=d), functools.partial(self.seq_step, [(0, 0, 1, 0, 1, [0]*(k+1), 0.0)] * m, drafts=d))
    for sv in (True, False):
      add(f"copy{int(sv)}", self._seq_copy_jits[sv], functools.partial(self._seq_copy_call, sv, 0, 1, 0))
      add(f"ckpt{int(sv)}", self._seq_ckpt_jits[sv], functools.partial(self.seq_ckpt, sv, 0, 1))
    add("move", self._seq_move_jits[1], functools.partial(self.seq_move, 0, self.REGION_ALIGN, 1))
    if self._seq_mat_jit is not None and self.config.spec_drafts:
      add("mat", self._seq_mat_jit, functools.partial(lambda: cast(Any, self._seq_mat_jit)(*self._mat_vars(0, 1, 1))))
    # the prefills' and steps' jits captured and compiled at once (in forked processes, see capture_parallel): their jits called alone (no
    # device work but theirs)
    par:list[tuple[Any, Callable[[], Any]]] = []
    def pf(jit:Any, call:Callable[..., Any]):
      # (a prefill in parts: every part's jit captured on its own)
      par.extend([(j, functools.partial(call, part=p)) for p, j in enumerate(jit)] if isinstance(jit, list) else [(jit, call)])
    for c, n in chunks:
      if self._dense_ctx(0, n) or top + c > self.seq_rows():
        pf(self._prefill_jit(c, self._dense_ctx(0, n)), functools.partial(self._seq_chunk, 0, 0, n))
      if top + c <= self.seq_rows(): pf(self._prefill_jit(c, False), functools.partial(self._seq_chunk, 0, top, c))
    for m in self.SEQ_SIZES:
      for d in self._step_drafts(m):
        par.append((jit:=self._seq_jit(m, k=d), functools.partial(lambda j, m, d: j(*self._seq_vars([(0, 0, 1, 0, 1)]*m, d)), jit, m, d)))
    # (and the copies of the states: their jits alone, not seq_move's write of the token buffer)
    for sv in (True, False):
      par += [(self._seq_copy_jits[sv], functools.partial(self._seq_copy_call, sv, 0, 1, 0)),
              (self._seq_ckpt_jits[sv], functools.partial(self.seq_ckpt, sv, 0, 1))]
    v = functools.partial(UOp.variable, min_val=0, max_val=self.seq_rows())
    par += [(j, functools.partial(lambda j: j(v("move_src").bind(0), v("move_dst").bind(self.REGION_ALIGN)), j))
            for j in self._seq_move_jits.values()]
    if self._seq_mat_jit is not None and self.config.spec_drafts:
      par.append((self._seq_mat_jit, functools.partial(lambda j: j(*self._mat_vars(0, 1, 1)), self._seq_mat_jit)))
    if self.stash_entries:
      for sv in (True, False):
        par += [(j:=self._stash_kv_jits[(sv, n)], functools.partial(lambda j: j(*self._stash_kv_vars(0, 0)), j)) for n in self._stash_calls]
        par.append((self._stash_state_jits[sv], functools.partial(lambda j: j(*self._stash_state_vars(0, 0, 0, 0)), self._stash_state_jits[sv])))
    # (WARMUP_JITS: the ones of the names wanted, the others captured on their first call)
    calls, par = warmup_select(calls, par)
    par += loads
    if DEBUG >= 1:
      for i, (_, call) in enumerate(par): print(f"seq_warmup: jit {i}: {getattr(call, 'func', call)} {getattr(call, 'args', '')}"[:160])
    _lap("capture_parallel")
    capture_parallel(par, meanwhile)
    _lap("captured")
    if DEBUG >= 1: print(f"seq_warmup: {len(par)} jits compiled at {time.perf_counter()-st:.2f}s")
    for name, (jit, call) in calls.items():
      if name in ran: continue
      st = time.perf_counter()
      call()
      if DEBUG >= 1: print(f"seq_warmup: {name} run in {time.perf_counter()-st:.2f}s")

  def seq_prefill(self, slot:int, base:int, tokens:list[int], start:int, hrow:int, temp:float|tuple[float, ...]=0.0,
                  upto:int|None=None, pictures:dict[int, Callable[[], Any]]|None=None) -> int|None:
    """
    the tokens from start of the sequence of slot `slot` (its region of the caches from base) prefilled, from its states of position start (the
    slot's, and its row hrow the hidden state of the token before it; start 0: none): its states of position len(tokens) into the slot (the
    hidden state of the last token its row 0). returns the token sampled after the last. upto: only the tokens up to upto (a slice of the
    prompt, the rest from there): None returned.
    VISION: a picture's rows are its ids (qwen_vision.picture_ids, negative), pictures the rows (n, dim) float32 of every picture of the
    prompt by its first row's id (a function: computed when a chunk needs them); a chunk with a picture's rows runs the vision variant
    """
    self._finish_load()
    self._seq_init()
    runs = []
    if self.vision and min(tokens, default=0) < 0:
      from tinygrad.llm.qwen_vision import picture_runs
      runs = picture_runs(tokens)
    def chunk(pos:int, n:int):
      vis = [r for r in runs if r[0] < pos + n and r[0] + r[1] > pos]
      self._seq_chunk(base, pos, n, vis=(vis, pictures or {}) if vis else None)
    if upto is not None:
      assert 0 <= start < upto < len(tokens) and base % self.REGION_ALIGN == 0 and base + len(tokens) <= self.seq_rows()
      for d in to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device)): Device[d].synchronize()
      # (the token after the slice the prompt's: the slice's sampled token isn't written over it, the next slice's tokens are its)
      self._token_host[base+start:base+upto+1] = self._dev_ids(tokens[start:upto+1])
      self._seq_copy_call(False, slot, start, hrow, keep=start > 0)
      pos = start
      for n in self._seq_chunks(upto - start):
        chunk(pos, n)
        pos += n
      self._seq_copy_call(True, slot, upto, 0)
      return None
    assert 0 <= start < len(tokens) and base % self.REGION_ALIGN == 0 and base + len(tokens) <= self.seq_rows()
    L, devs = len(tokens), to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device))
    for d in devs: Device[d].synchronize()  # (a step may still read the token buffer)
    self._token_host[base+start:base+L+1] = self._dev_ids(tokens[start:]) + array.array('i', [-1])
    self._seq_copy_call(False, slot, start, hrow, keep=start > 0)
    self._seq_pf_host_temp[:TEMP_W] = _temp_row(temp)
    pos = start
    for n in self._seq_chunks(L - start):
      chunk(pos, n)
      pos += n
    self._seq_copy_call(True, slot, L, 0)
    return self._wait_host(lambda: self._token_host[base + L], "the token after a prefill")

  def _seq_first(self, slot:UOp, at:UOp):
    """the token after the hidden state of row 0 of slot `slot` (a prompt's last token's, normed: GLM) sampled into the token buffer at `at`"""
    H = 2*(self.config.spec_drafts+1)
    tok = self._sample(self.seq_spec_h.shrink(((slot*H, slot*H+1), None)), self._seq_pf_temp, self._token_buf.device)
    self._token_buf[:, at:at+1].assign(tok.reshape(1, 1).cast(self._token_buf.dtype)).realize()
  def _first_vars(self, slot:int, at:int) -> list[UOp]:
    return [UOp.variable("first_slot", 0, self.seq_slots()).bind(slot), UOp.variable("first_at", 0, self._token_rows()-1).bind(at)]
  def seq_first(self, slot:int, base:int, pos:int, temp:float|tuple[float, ...]=0.0) -> int:
    """
    the token after the sequence of slot `slot` (its region from base) of pos tokens, from its states there (its hidden state of the last token
    its row 0: a prompt's checkpoint restored, see SeqEngine.prefill): a prompt that is all in the caches has no token to prefill
    """
    self._seq_init()
    if not hasattr(self, "_seq_first_jit"): self._seq_first_jit = TinyJit(self._seq_first, capture_first=True)
    for d in to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device)): Device[d].synchronize()  # (a step may read the token buffer)
    self._token_host[base+pos] = -1
    self._seq_pf_host_temp[:TEMP_W] = _temp_row(temp)
    self._seq_first_jit(*self._first_vars(slot, base+pos))
    return self._wait_host(lambda: self._token_host[base + pos], "the token after a prompt")

  def seq_slice(self, n:int) -> int:
    """a prefill slice of about n tokens (see the server's PREFILL_SLICE): GLM's whole chunks of generate's (its prefill the same chunks)"""
    return round_up(n, self.prefill_chunks()[-1])
  def _seq_chunks(self, n:int) -> list[int]:
    """
    the chunks of a prefill of n tokens: GLM as generate's (the biggest size, the rest: the same chunks, its greedy tokens the main path's)
    """
    big, ret = self.prefill_chunks()[-1], []
    # (PREFILL_BIG_ALONE=1, default off: the biggest chunk size only while no other sequence decodes (the server sets _pf_alone before a
    # slice), else the next size: a long chunk is a long wait for the decoding ones. the chunks then depend on the load: a reorder)
    if getenv("PREFILL_BIG_ALONE", 0) and not getattr(self, "_pf_alone", True) and len(sizes:=self.prefill_chunks()) > 1: big = sizes[-2]
    while n > 0:
      ret.append(min(big, n))
      n -= ret[-1]
    return ret
  def _seq_chunk(self, base:int, pos:int, n:int, part:int|None=None,
                 vis:tuple[list[tuple[int, int, int, int]], dict[int, Callable[[], Any]]]|None=None):
    """
    a prefill chunk: the n tokens from pos of the region from base in the token buffer (the jit of the smallest chunk size of n or more).
    vis (VISION): the pictures with rows in the chunk (runs, their rows by first id): the vision variant
    """
    sizes = self.prefill_chunks()
    vb, sp = UOp.variable("seq_base", 0, self.seq_rows()).bind(base), UOp.variable("start_pos", 0, self.max_context-1).bind(pos)
    nt = UOp.variable("toks", 1, next(s for s in sizes if n <= s)).bind(n)
    if vis is not None: self._vis_fill(*vis, pos, n)
    _vis_on[0] = vis is not None
    try: self(self._token_buf[:, vb+sp:vb+sp+nt], sp, self._seq_pf_temp, vb, part)
    finally: _vis_on[0] = False

  # ******** VISION: a prefill chunk with a picture's rows runs the vision variant of the chunk's jit (__call__ keys it apart; text-only chunks
  # run the jits of before, unchanged): the chunk's embedding with the tower's rows (host memory, written before the chunk: _vis_fill) where
  # the row mask is set, the MTP layer's input too. no positions to change: GLM's text model has no rope ********
  def _vis_init(self, devs:tuple[str, ...]):
    assert self.image_pad_id >= 0, "VISION: the tokenizer's <|image|> token"
    self._vis_rows_max = self.prefill_chunks()[-1]
    mv, self._vis_rows = _host_buffer(devs, 4*self._vis_rows_max*self.config.dim, dtypes.float32)
    self._vis_host_rows = mv.cast('f')
    self._vis_host_mask, self._vis_mask = _host_ints(devs, self._vis_rows_max)

  def _vis_embed(self, emb:Tensor) -> Tensor:
    """the vision variant: emb (1, T, dim) with the picture rows of the chunk's rows"""
    T, D, M = emb.shape[1], self.config.dim, self._vis_rows_max
    rows = self._vis_rows.reshape(M, D).shrink(((0, T), None)).reshape(1, T, D)
    return (self._vis_mask[0].shrink(((0, T),)).reshape(1, T, 1) != 0).where(rows, emb)

  def _vis_fill(self, runs:list[tuple[int, int, int, int]], pictures:dict[int, Callable[[], Any]], pos:int, n:int):
    """VISION: the vision variant's host inputs of the chunk of n rows from pos: the picture rows there and their mask"""
    # (the chunk before may still read them)
    for d in to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device)): Device[d].synchronize()
    M, D = self._vis_rows_max, self.config.dim
    assert n <= M, f"a vision chunk of {n} rows (at most {M})"
    mask = array.array('i', bytes(4*M))
    for start, cnt, _, first in runs:
      a, b = max(start, pos), min(start + cnt, pos + n)
      if a >= b: continue
      assert first in pictures, f"the rows of the picture at row {start} (id {first}): not given"
      pic = pictures[first]()
      assert pic.shape == (cnt, D) and pic.dtype.itemsize == 4, f"the picture at row {start}: rows {pic.shape} {pic.dtype}, its ids {cnt} x {D}"
      self._vis_host_rows[(a-pos)*D:(b-pos)*D] = memoryview(pic[a-start:b-start].astype('float32').reshape(-1))
      mask[a-pos:b-pos] = array.array('i', [1]*(b-a))
    self._vis_host_mask[:M] = mask

  def _dev_ids(self, ids:list[int]) -> array.array:
    """the token buffer's ids of a token list: a picture's rows (VISION: negative ids) the image pad id"""
    if not self.vision: return array.array('i', ids)
    return array.array('i', [t if t >= 0 else self.image_pad_id for t in ids])

  def forward_seqs(self, *vs:UOp) -> Tensor: return self._forward_seqs(vs)
  def _ints(self, xs:tuple[UOp, ...]) -> Tensor:
    """variables as an int32 tensor of the devices (a stack of variables has no device: onto zeros of the devices)"""
    devs = self.token_embd.weight.device
    return (Tensor.zeros(len(xs), dtype=dtypes.int32, device=devs) + Tensor.stack(*[Tensor(x) for x in xs]).cast(dtypes.int32)).contiguous()

  def _forward_seqs_glm(self, vs:tuple[UOp, ...]) -> Tensor:
    """
    forward_seqs of GLM (MLA with the indexer, KDA blocks, the MTP layer; no hyper-connections' streams, no n-gram rows): as forward_batch_spec,
    a sequence's caches its region (its base), its states its slot's. the hidden states kept are the normed ones (as forward_spec's)
    """
    k, G, devs = self.config.spec_drafts if self._step_k is None else self._step_k, len(vs) // 6, self.token_embd.weight.device
    sl, bs, mp, js, ps, kbs = (list(vs[i::6]) for i in range(6))
    # (the hidden states' layout: K+1 rows a parity (spec_drafts, the most drafts of a step), a step of k drafts its first k+1)
    K = self.config.spec_drafts
    slots, bases, H = self._ints(tuple(sl)), self._ints(tuple(bs)), 2*(K+1)
    y = self._seq_in[:, :G*(k+1)]
    temps = self._seq_temp[:TEMP_W*G].reshape(G, TEMP_W)
    def batch(pos:list[UOp], R:int) -> SeqBatch: return SeqBatch(G, R, slots, bases, self._ints(tuple(pos)))
    # (k == K: the expressions as before, the same graph)
    last = Tensor.cat(*[self.seq_spec_h.shrink(((s*H + (1-p)*(K+1), s*H + ((2-p)*(K+1) if k == K else (1-p)*(K+1) + k+1)), None))
                        for s, p in zip(sl, ps)])
    seed = self._seq_temp[-1:].bitcast(dtypes.uint32)
    Q8_ROWS_I8[0] = bool((0))
    try:
      # the MTP layer catches up on the k+1 positions from mtp_pos, row j of every sequence drafts
      _seq_batch[0], _batch_rows[0] = batch(mp, k+1), [(p, k+1) for p in mp]
      m = self.mtp.predict(self.token_embd(y.to(devs)).float(), last.unsqueeze(0), 0)
      m = Tensor.cat(*[m.shrink((None, (g*(k+1)+j, g*(k+1)+j+1), None)) for g, j in enumerate(js)], dim=1)
      drafts:list[Tensor] = []
      for i in range(k):
        # (draft i of sequence g: verified at its row i, sampled with that row's noise at its temperature, not its top_p)
        drafts.append(self._sample(m[0], temps[:, :1], y.device, (seed, Tensor.arange(G) * (k+1) + i)).reshape(1, G))
        if i < k-1:
          pos = [p + j + 1 + i for p, j in zip(mp, js)]
          _seq_batch[0], _batch_rows[0] = batch(pos, 1), [(p, 1) for p in pos]
          m = self.mtp.predict(self.token_embd(drafts[-1].to(devs)).float(), m, 0)
      # verify: the model on every sequence's token at start_pos (mtp_pos+j) and its drafts
      vt = Tensor.cat(*[y.shrink((None, (g*(k+1)+j, g*(k+1)+j+1))).cat(*[d[:, g:g+1] for d in drafts], dim=1) for g, j in enumerate(js)],
                      dim=1).contiguous()
      starts = [p + j for p, j in zip(mp, js)]
      x = self.token_embd(vt.to(devs)).float()
      _seq_batch[0], _batch_rows[0] = batch(starts, k+1), [(s, k+1) for s in starts]
      # (SEQ_REPLAY: the positions of the slots' recurrent states, their replayed tokens up to a sequence's start)
      if SEQ_REPLAY: _seq_batch[0] = replace(cast(SeqBatch, _seq_batch[0]), kbases=self._ints(tuple(kbs)))
      if (hc:=self.config.hc) is not None: x = x.unsqueeze(2).expand(*x.shape[:2], hc.count, x.shape[-1]).contiguous()
      for block in self.blk: x = block(x, 0)
      if hc is not None: x = x.mean(2)
      hn_all = self.output_norm(x)[0].float()
      hs = Tensor.cat(*[self.seq_spec_h.shrink(((s*H + p*(K+1), s*H + ((p+1)*(K+1) if k == K else p*(K+1) + k+1)), None))
                        .assign(hn_all[g*(k+1):(g+1)*(k+1)])
                        for g, (s, p) in enumerate(zip(sl, ps))])
      tt = temps.reshape(G, 1, TEMP_W).expand(G, k+1, TEMP_W).reshape(G*(k+1), TEMP_W)
      tok = self._sample(hs, tt, y.device, (seed, Tensor.arange(G*(k+1))))
    finally: _seq_batch[0], _batch_rows[0], Q8_ROWS_I8[0] = None, None, False
    q8_quantize_cache_clear()  # don't keep the graph alive
    out = tok.reshape(G, k+1).cat(*[d.reshape(G, 1) for d in drafts], dim=1).reshape(1, G*(2*k+1)).cast(dtypes.int32)
    return self._seq_out[:, :G*(2*k+1)].assign(Tensor(out.uop.mselect(0)) if isinstance(out.device, tuple) else out)

  def _forward_seqs(self, vs:tuple[UOp, ...]) -> Tensor:
    """
    a step of speculative decoding of G sequences sharing the caches: vs the slot, the base of the region, mtp_pos, j and par of every one (as
    forward_spec's), its k+1 tokens (the last step's sampled ones) in _seq_in, its temperature and top_p in _seq_temp. returns (1, G*(2k+1)):
    every one's k+1 sampled tokens and k drafts. (the drafts are sampled with the noise of the row they're verified at: an exact sample of the
    model, more of them accepted at a temperature)
    """
    return self._forward_seqs_glm(vs)

  _step_k:int|None = None  # (the drafts of the step being captured, see _seq_jit)
  def _step_drafts(self, m:int) -> list[int]:
    """the drafts of the steps of m sequences: the size's, then the ones of a sequence whose last step (of another size) drafted more"""
    d = self.drafts_of(m)
    return [d] + [x for x in sorted({self.drafts_of(s) for s in self.SEQ_SIZES}) if x > d]
  def _seq_jit(self, m:int, k:int|None=None):
    """the jit of a step of m sequences and k drafts (default the size's, see drafts_of)"""
    if not hasattr(self, "_seq_jits"): self._seq_jits:dict[tuple, Any] = {}
    k = self.drafts_of(m) if k is None else k
    # (a step of fewer drafts than the layouts': captured with _step_k set)
    key = (m,) if k == self.config.spec_drafts else (m, k)
    if key not in self._seq_jits:
      fn = self.forward_seqs
      if k != self.config.spec_drafts:
        assert not self.has_recurrent_block, "SPEC_DRAFTS_BY_SIZE: GLM_SEQ steps of MLA models only (no replay)"
        def forward_seqs_k(*vs:UOp, fn=fn, k=k) -> Tensor:
          self._step_k = k
          try: return fn(*vs)
          finally: self._step_k = None
        fn = forward_seqs_k
      self._seq_jits[key] = TinyJit(fn, capture_first=True, scratch=self._scratch)
    return self._seq_jits[key]
  def _seq_vars(self, seqs:list[tuple[int, int, int, int, int]], k:int|None=None) -> list[UOp]:
    k, n = self.config.spec_drafts if k is None else k, self.seq_slots()
    return [v for g, (sl, b, mp, j, par) in enumerate(seqs) for v in (
      UOp.variable(f"seq_slot{g}", 0, n).bind(sl), UOp.variable(f"seq_base{g}", 0, self.seq_rows()).bind(b),
      UOp.variable(f"seq_mtp{g}", 0, self.max_context-1).bind(mp), UOp.variable(f"seq_j{g}", 0, k).bind(j),
      UOp.variable(f"seq_par{g}", 0, 1).bind(par), UOp.variable(f"seq_kb{g}", 0, self.max_context).bind(self._kbase_of(sl, mp + j)))]
  def _kbase_of(self, slot:int, start:int) -> int:
    """the position of slot `slot`'s recurrent state for a step at start (SEQ_REPLAY: the replayed tokens' before it), else start"""
    if not SEQ_REPLAY or (b:=getattr(self, "_seq_kbase", {}).get(slot)) is None or not 0 <= start - b <= self.config.spec_drafts: return start
    return b

  step_times = [0.0, 0.0, 0.0]  # (the launches' and the waits' time of the steps, see the server's status line)
  def seq_step(self, seqs:list[tuple[int, int, int, int, int, list[int], float|tuple[float, ...]]],
               drafts:int|None=None) -> list[tuple[list[int], list[int]]]:
    """
    a step of speculative decoding of sequences sharing the caches: seqs (slot, base, mtp_pos, j, par, the k+1 tokens of the last step,
    temperature) of every one (see _spec_steps), their k+1 sampled tokens and k drafts.
    drafts: the step's (default the size's, see drafts_of); a sequence's j past them (its last step drafted more): the step drafts j or more
    (its MTP catches up on j+1 tokens, see _step_drafts), y as long as the step's k+1 (the tokens past j not read). the layouts of the
    states are spec_drafts'
    """
    self._seq_init()
    n = len(seqs)
    m = next(s for s in self.SEQ_SIZES if s >= n)
    k = max([self.drafts_of(m) if drafts is None else drafts, *(j for _, _, _, j, *_ in seqs)])
    # (more than the size's: the fewest of the warmed steps' that are enough)
    if drafts is None: k = next((d for d in self._step_drafts(m) if d >= k), k)
    assert k <= self.config.spec_drafts, f"a step of {k} drafts, the layouts have {self.config.spec_drafts}"
    # (the sequences past n: the padding slot's at position 0 in the padding region)
    seqs = seqs + [(self.seq_slots(), self.seq_rows(), 0, 0, 1, [0]*(k+1), 0.0)] * (m - n)
    self._seq_host_in[:m*(k+1)] = array.array('i', [t for *_, y, _ in seqs for t in (list(y) + [0]*k)[:k+1]])
    self._seq_host_temp[:TEMP_W*m] = array.array('f', [v for *_, temp in seqs for v in _temp_row(temp)])
    # (the step's seed, see forward_seqs)
    self._seq_host_temp[-1] = array.array('f', bytes(random.getrandbits(8) for _ in range(4)))[0]
    W = m*(2*k+1)
    self._seq_host_out[:W] = array.array('i', [-1] * W)
    jit, st = self._seq_jit(m, k=k), time.perf_counter()
    if (cap:=getattr(jit, "captured", None)) is not None:
      # (captured: the variables' values as they are, no bound variables made a step)
      cap([], {name: v for g, (sl, b, mp, j, par, _, _) in enumerate(seqs)
               for name, v in ((f"seq_slot{g}", sl), (f"seq_base{g}", b), (f"seq_mtp{g}", mp), (f"seq_j{g}", j), (f"seq_par{g}", par),
                               (f"seq_kb{g}", self._kbase_of(sl, mp + j)))})
    else: jit(*self._seq_vars([(sl, b, mp, j, par) for sl, b, mp, j, par, _, _ in seqs], k))
    # (SEQ_REPLAY: every sequence's recurrent states now the ones after its step's first token)
    if SEQ_REPLAY:
      for sl, _, mp, j, _, _, _ in seqs[:n]: self._seq_kbase[sl] = mp + j + 1
    launched = time.perf_counter()
    out = self._wait_host(lambda: min(o:=self._seq_host_out[:W].tolist()) >= 0 and o or -1, f"a step of {m} sequences")
    self.step_times[0] += launched - st
    self.step_times[1] += time.perf_counter() - launched
    return [(out[g*(2*k+1):g*(2*k+1)+k+1], out[g*(2*k+1)+k+1:(g+1)*(2*k+1)]) for g in range(n)]
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
    def lazy(v) -> list[Tensor]:
      # (realized: a buffer, split over the devices or not. not a copy made of one)
      if isinstance(v, Tensor): return [v] if not _is_buffer(v.uop) else []
      return [t for x in v for t in lazy(x)] if isinstance(v, (list, tuple)) else []
    # (and the parameters of a model initialized but not loaded. a loaded one's are: the quantized weights' are their decoding, not used)
    todo = [t for h in holders for v in vars(h).values() for t in lazy(v)] + ([] if self._loaded else lazy(nn.state.get_parameters(self)))
    # with the weights still to read: their memory now, their values after
    if self._pending_load: _realize_later(todo, self._pending_load, self._scratch)
    elif todo: Tensor.realize(*todo)
    self._states_ready = True

  def __call__(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor|None, base:int|UOp=0, part:int|None=None) -> Tensor:
    # a view of the token buffer is already realized: no schedule for the jit input
    if (buf:=getattr(self, "_token_buf", None)) is None or tokens.uop.base is not buf.uop.base: tokens = tokens.contiguous()
    key = int(T.vmax) if isinstance(T:=tokens.shape[1], UOp) else T
    # select the ordinary/dense-indexer JIT variant for this context
    dense = self._dense_ctx(start_pos, T)
    # (a sequence's region from a variable base: jits of their own, see seq_prefill)
    # (VISION: the vision variant of a chunk, see _vis_embed)
    jits = self.__dict__.setdefault("jits" + ("_dense" if dense else "") + ("_greedy" if temperature is None else "") +
                                    ("_seq" if isinstance(base, UOp) else "") + ("_vis" if _vis_on[0] else ""), {})
    if (jit:=jits.get(key)) is None:
      # the states and caches are set up before: a jit captures on its first call
      self._init_states()
      jit = jits[key] = self._new_jit(key)
    if not isinstance(jit, list): return jit(tokens, start_pos, temperature, base)[0]
    # (a prefill in parts: one of them (part, see seq_warmup), else all in order)
    for p, j in enumerate(jit):
      if part is None or p == part: out = j(tokens, start_pos, temperature, base)[0]
    return out

  def _new_jit(self, rows:int) -> Any:
    """the jit of a call of rows (a list of the parts' jits: PREFILL_PARTS)"""
    if len(parts:=self.prefill_parts(rows)) == 1: return TinyJit(self._forward_effects, capture_first=True, scratch=self._scratch)
    self._carry(rows)
    return [TinyJit(functools.partial(self._forward_effects, part=p), capture_first=True, scratch=self._scratch) for p in range(len(parts))]

  def _dense_ctx(self, start_pos:int|UOp, T:int|UOp) -> bool:
    return False

  def _forward_effects(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor|None, base:int|UOp=0, part:int|None=None) -> tuple[Tensor, ...]:
    # forward and the tensors realized with its output for their effects (e.g. the caches of the MTP layer)
    self._effects = []
    out = self.forward(tokens, start_pos, temperature, base, part)
    effects, self._effects = self._effects, None
    return (out, *effects)

  @staticmethod
  def from_gguf(gguf:Tensor|str|pathlib.Path, max_context:int|None=None,
                realize=False, shard:int=1, defer_load:bool=False) -> tuple[Transformer, dict]:
    """defer_load: ignored, the weights are read before the warmup (no loading in the background)"""
    defer_load = False
    # TODO: remove the need for copy to default device
    kv, entries = gguf_parse(gguf.to(None).realize() if isinstance(gguf, Tensor) else gguf)
    _lap("parsed")
    # deferred: the files are read ahead from now (the disk works while the model is made, the loader finds them in memory)
    # (not the n-gram table: read to host memory by its loader, see _ple_dev_init)
    if defer_load: _read_ahead(entries, skip=("per_layer_token_embd.weight",))
    arch = kv['general.architecture']
    assert arch != 'qwen4exp', "qwen4exp: use tinygrad.llm.qwen4exp"
    n_heads, n_kv_heads = kv[f'{arch}.attention.head_count'], kv[f'{arch}.attention.head_count_kv']
    assert shard >= 1, f"shard must be at least 1, got {shard}"
    # the next token prediction (MTP) layers: the first is loaded with MTP=1 (model.mtp), the others aren't used
    num_blocks = kv[f'{arch}.block_count'] - kv.get(f'{arch}.nextn_predict_layers', 0)
    mtp = bool(int(arch == "glm5-next")) and kv.get(f"{arch}.nextn_predict_layers", 0) > 0
    entries = {k:v for k,v in entries.items() if not ((m:=re.match(r"blk\.(\d+)\.", k)) and int(m.group(1)) >= num_blocks + mtp)}
    # (qwen4exp: the n-gram table stays in host memory, the gated deltanets in the kda layout, a KV head a device)
    ple_entry = entries.pop('per_layer_token_embd.weight', None)
    rows_map:dict[str, tuple[int, list[tuple[int, int]]]] = {}
    shard_map:dict[str, int] = {}
    units_map:dict[str, tuple[int, list[list[int]]]] = {}
    repeats:dict[str, int] = {}
    pads:dict[str, int] = {}
    padded_heads, sizes, expert_slices, expert_pad = n_heads, {}, 0, 0
    if shard > 1:
      # MLA and kda attention split all their tensors by the n_heads heads: those are zero padded to a multiple of the devices
      pad_heads = bool(kv.get(f'{arch}.attention.kv_lora_rank')) or arch in ('kimi-linear', 'glm5-next')
      if not pad_heads and not kv.get(f'{arch}.attention.kv_lora_rank') and n_kv_heads < shard and shard % n_kv_heads == 0 and (1):
        # too few KV heads to split: repeat each one so every device holds one (master's gguf_shard repeats)
        repeats = {k: shard // n_kv_heads for k in entries if k.split('.', 2)[-1] in ('attn_k.weight', 'attn_v.weight', 'attn_k.bias', 'attn_v.bias')}
        n_kv_heads = shard
      if not pad_heads: assert n_kv_heads % shard == 0, f"tensor parallel needs the attention heads to split over {shard} devices"
      # shard heads, FFNs and routed experts while replicating latent projections, KV cache and the small kda projections
      head_rules = {**{w: 0 for w in ('attn_q.weight', 'attn_k.weight', 'attn_v.weight', 'attn_q_b.weight', 'attn_k_b.weight', 'attn_v_b.weight')},
                    'attn_output.weight': 1}
      # the kda layers split by heads too (the gated deltanet of qwen3.5 is copied to every device)
      if arch in ('kimi-linear', 'glm5-next'): head_rules.update({w: 0 for w in ('ssm_conv1d_q.weight', 'ssm_conv1d_k.weight',
        'ssm_conv1d_v.weight', 'ssm_dt.bias', 'ssm_a', 'ssm_beta.weight', 'ssm_f_b.weight', 'ssm_g_b.weight', 'ssm_alpha.weight',
        'attn_gate.weight')})
      padded_heads = n_heads
      if pad_heads:
        # a device's heads of a weight split on its inputs (attn_output) are whole quant blocks: a Q6_K kda output (heads of 128 columns, blocks
        # of 256) on 6 devices takes 72 heads (12 a device), not 66
        in_split = [(shape[-1] // n_heads, _GGML_QUANT.get(typ, (1,))[0]) for name,(_,shape,typ) in entries.items()
                    if head_rules.get(re.sub(r"^blk\.\d+\.", "", name)) == len(shape)-1]
        padded_heads = round_up(n_heads, shard)
        while any((padded_heads // shard * cols) % block for cols, block in in_split): padded_heads += shard
      ffn_rules = {**{w: 0 for w in ('ffn_gate.weight', 'ffn_up.weight', 'ffn_gate_shexp.weight', 'ffn_up_shexp.weight', 'output.weight')},
                   **{w: 1 for w in ('ffn_down.weight', 'ffn_down_shexp.weight')}}
      if len(kv['tokenizer.ggml.tokens']) % shard == 0: ffn_rules['token_embd.weight'] = 0
      # experts are split by their hidden dim, or by experts when it doesn't split (expert parallel)
      exp_hidden, exp_count = kv.get(f'{arch}.expert_feed_forward_length', 0), kv.get(f'{arch}.expert_count', 0)
      down_block = max([_GGML_QUANT.get(t, (1,))[0] for k,(_,_,t) in entries.items() if k.endswith('ffn_down_exps.weight')], default=1)
      exp_names = ('ffn_gate_exps.weight', 'ffn_up_exps.weight', 'ffn_down_exps.weight')
      if exp_hidden % shard == 0 and (exp_hidden // shard) % down_block == 0:
        ffn_rules.update({'ffn_gate_exps.weight': 1, 'ffn_up_exps.weight': 1, 'ffn_down_exps.weight': 2})
      elif exp_count and exp_hidden % down_block == 0 and (1) and \
           (padded:=next((e for e in range(exp_count, exp_count + shard*shard) if expert_slicing(e, exp_hidden // down_block, shard)), 0)):
        # slices of the experts' hidden dim (whole quant blocks of the down projection) spread over the devices: the work of a token is balanced
        # (padded: experts past the router's never selected, so the slices split evenly. 512 on 6 devices: 516. their units load expert 0's bytes)
        expert_slices, expert_pad = exp_hidden // down_block, padded if padded != exp_count else 0
        slicing = [[u if u < exp_count*expert_slices else u % expert_slices for u in us]
                   for us in unwrap(expert_slicing(padded, expert_slices, shard))]
        units_map = {name: (expert_slices, slicing) for name in entries if re.sub(r"^blk\.\d+\.", "", name) in exp_names}
      elif exp_count % shard == 0: ffn_rules.update({w: 0 for w in exp_names})
      # the down projections' blocks of all the layers: an FFN's hidden dim pads the same in every layer (UD quants mix Q6_K and Q8_0 ones)
      down_units:dict[str, int] = {}
      for name, (_, _, typ) in entries.items():
        if (m:=re.search(r"(ffn_down(_shexp)?)\.weight$", name)):
          down_units[m.group(1)] = math.lcm(down_units.get(m.group(1), 1), _GGML_QUANT.get(typ, (1,))[0])
      for name, (_, shape, typ) in entries.items():
        if (k:=re.sub(r"^blk\.\d+\.", "", name)) in head_rules:
          axis = shard_map[name] = head_rules[k]
          sizes[name] = shape[axis] // n_heads * padded_heads if padded_heads != n_heads else shape[axis]
        elif k in ffn_rules:
          axis = shard_map[name] = ffn_rules[k]
          # FFN hidden dims pad to whole quant blocks of the down projection on every device
          down = entries.get(re.sub(r"ffn_(gate|up)(_shexp)?\.", r"ffn_down\2.", name), entries[name])
          # (the output rows: whole tiles of 32 on every device, the tensor cores take a few tokens' logits)
          mk = re.match(r"ffn_(?:gate|up|down)(_shexp)?\.weight$", k)
          unit = 32 if k == 'output.weight' else 1 if '_exps' in k else down_units["ffn_down" + (mk.group(1) or "")] if mk else \
            _GGML_QUANT.get(down[2], (1,))[0]
          sizes[name] = round_up(shape[axis], shard*unit)
      pads = {k: sizes[k] for k in sizes if sizes[k] != entries[k][1][shard_map[k]]}
    devices = tuple(Device.canonicalize(f'{Device.DEFAULT}:{i}') for i in range(shard))
    _open_devices(devices)
    _lap("devices opened")
    # (the compile workers started and warmed up while the weights are read)
    warm_worker_pool(devices[0])
    _lap("pool started")
    # gate and up experts of the same format load as one (fuse_gate_up)
    # (a pair in different files of a split gguf isn't: a fused read is of the gate's file. prepare fuses it on the devices)
    def file_of(data:Tensor) -> str|None:
      try: return cast(Buffer, data.uop.buffer).device
      except Exception: return None
    fuse = {g: u for g in units_map if g.endswith("ffn_gate_exps.weight") and (u:=g.replace("gate", "up")) in units_map and
            entries[g][2] == entries[u][2] and file_of(entries[g][0]) == file_of(entries[u][0]) and (1)}
    # a token embedding copied to every device (a vocab that doesn't split) stays in host memory: a call looks up a few of its rows
    host_embd = None
    if shard > 1 and 'token_embd.weight' not in shard_map and 'output.weight' in entries and amd_custom_kernels_supported(devices) and \
       (1):
      data, shape, typ = entries.pop('token_embd.weight')
      mv, raw = _host_buffer(devices, int(data.nbytes()))
      mv[:] = data.to("CPU").data()
      host_embd = ggml_data_to_tensor(raw, prod(shape), typ).reshape(shape)
    pending:list[Callable[[], None]] = []
    # (LOAD_DEFER: the files read while the jits compile, see seq_warmup)
    state_dict = gguf_shard_units(entries, devices, shard_map, repeats, units_map, pads if shard > 1 else {}, fuse, row_sel=rows_map,
                                  pending=pending if getenv("LOAD_DEFER", 0) else None)
    _lap("weights")
    if host_embd is not None: state_dict['token_embd.weight'] = host_embd

    # all state items should be float16, not float32
    state_dict = {k:v.cast('float16') for k,v in state_dict.items()}

    # some models like Llama 3.2 don't have an output.weight, they just tie to the token_embd.weight
    if 'output.weight' not in state_dict: state_dict['output.weight'] = state_dict['token_embd.weight']

    hc = HCConfig(kv[f'{arch}.hyper_connection.count'], kv.get(f'{arch}.hyper_connection.sinkhorn_iterations', 0),
                  kv.get(f'{arch}.hyper_connection.epsilon', 0.0), kv.get(f'{arch}.hyper_connection.low_rank', 0)) \
      if f'{arch}.hyper_connection.count' in kv else None
    state_dict = {re.sub(r'(hc_attn|hc_ffn|hc_head|output_hc|\.ple)_', r'\1.', k):v for k,v in state_dict.items()}

    kpool = kv.get(f'{arch}.attention.indexer.block_size' if arch == 'minimax-m3' else f'{arch}.attention.indexer.kpool')
    indexer = IndexerConfig(kv[f'{arch}.attention.indexer.head_count'], kv[f'{arch}.attention.indexer.key_length'],
                            kv[f'{arch}.attention.indexer.top_k'], kpool,
                            kv.get(f'{arch}.attention.layer_norm_epsilon') or kv[f'{arch}.attention.layer_norm_rms_epsilon']) if kpool else None
    ple = PLEConfig(kv[f'{arch}.ple.layers'][0], ple_entry[1][1], tuple(kv[f'{arch}.ple.layer_multipliers']),
                    tuple(kv[f'{arch}.ple.head_vocab_sizes']), tuple(kv[f'{arch}.ple.head_offsets']), kv[f'{arch}.ple.eos_token_id'],
                    kv[f'{arch}.ple.conv_kernel'], ple_entry[2]) if ple_entry is not None else None
    if indexer is not None: state_dict = {k.replace('.indexer_compressor_', '.indexer.compressor_'):v for k,v in state_dict.items()}
    if arch == 'minimax-m3':
      indexer = replace(unwrap(indexer), rope_dim=kv[f'{arch}.rope.dimension_count'], rope_interleave=False, msa=True,
                        top_k=kv[f'{arch}.attention.indexer.top_k'] * kv[f'{arch}.attention.indexer.block_size'])
      state_dict = {k.replace('.indexer.q_proj', '.indexer.attn_q_b').replace('.indexer.k_proj', '.indexer.attn_k'):v for k,v in state_dict.items()}
    if mtp: state_dict = {re.sub(rf"^blk\.{num_blocks}\.(nextn\.)?", "mtp.", k):v for k,v in state_dict.items()}

    max_context = min(max_context, kv[f'{arch}.context_length']) if max_context is not None else kv[f'{arch}.context_length']

    ssm = None
    ssm_layers: tuple[bool, ...] = ()
    if arch in ('qwen35', 'qwen35moe'):
      ssm = SSMConfig(**{k: kv[f'{arch}.ssm.{k}'] for k in ('conv_kernel','state_size','group_count','time_step_rank','inner_size')}, silu_gate=True)
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
        state_dict[name] = permute_rope(state_dict[name], n_heads, rope_dim)
      elif arch == 'llama' and 'attn_k.weight' in name:
        state_dict[name] = permute_rope(state_dict[name], n_kv_heads, state_dict[name].shape[0]//n_kv_heads)
      elif kv_lora_rank and 'attn_kv_a_mqa.weight' in name:
        state_dict[name] = permute_rope(state_dict[name], 1, state_dict[name].shape[0]-kv_lora_rank)
    swiglu_oai = arch in ('gpt-oss', 'minimax-m3')
    glm_seq = (1)
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
      routed_scaling_factor=kv.get(f'{arch}.expert_weights_scale', 1.0), attn_output_gate=arch in ('qwen35', 'qwen35moe'),
      ssm=ssm,
      ssm_layers=ssm_layers,
      qkv_bias='blk.0.attn_q.bias' in state_dict,
      expert_bias=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.exp_probs_b.bias" in state_dict, expert_slices=expert_slices,
      expert_proj_bias='blk.0.ffn_gate_exps.bias' in state_dict, attn_output_bias='blk.0.attn_output.bias' in state_dict,
      swiglu_alpha=1.702 if swiglu_oai else 1.0, swiglu_limit=7.0 if swiglu_oai else kv.get(f'{arch}.swiglu_clamp_exp', [None])[0],
      swiglu_up_bias=1.0 if swiglu_oai else 0.0, attn_sinks='blk.0.attn_sinks.weight' in state_dict,
      sliding_window=kv.get(f'{arch}.attention.sliding_window', 0),
      sliding_window_pattern=kv.get(f'{arch}.attention.sliding_window_pattern', 2 if arch == 'gpt-oss' else 0), hc=hc, indexer=indexer, mtp=mtp,
      spec_drafts=getenv("SPEC_DRAFTS", 2) if mtp else 0, ple=ple, expert_pad=expert_pad,
      # (SEQ_ROWS: the rows of the caches of the sequences sharing them, past a context: more conversations kept)
      # (GLM-5.3-Flash on several devices: GLM_SEQ on by default (the server's SeqEngine: conversations keep their caches, a host stash), its
      # rows two contexts (262144: 524288 rows, with 16 slots and SEQ_STASH_GB=4 in a 6x24 GB C5's memory; C6/C7 have more))
      cache_rows=round_up(getenv("SEQ_ROWS", 2*max_context if glm_seq else 0), REGION_ALIGN) if glm_seq else 0)
    with Linear.empty_init(): model = Transformer(config)  # (the weights are loaded)
    _lap("model made")
    # (the file's KV heads: repeated so every device holds one, device d holds head d*n//devices)
    model._kv_real = kv[f'{arch}.attention.head_count_kv']
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
    _lap("state loaded")
    model._pending_load, model._loaded = pending, True
    # (VISION: GLM 5.3 Flash's image pad: the chat template's <|image|> between <|begin_of_image|> and <|end_of_image|>)
    if arch == 'glm5-next' and '<|image|>' in (toks:=kv.get('tokenizer.ggml.tokens', [])): model.image_pad_id = toks.index('<|image|>')
    # (the n-gram table: the bytes in the gguf file, mmapped, or a copy in host memory)
    if ple_entry is not None:
      model.ple_table = _disk_bytes(ple_entry[0]) or cast(Buffer, ple_entry[0].to("CPU").realize().uop.base.buffer).as_memoryview()
      model._ple_file = _disk_location(ple_entry[0])
    # NOTE: without this contiguous, it unpacks the weights from the model every time. we shouldn't need this, but for now it's faster
    if realize:
      for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
      Tensor.realize(*params)
    else: model.prepare()
    _lap("prepared")
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
    for m in modules([self.blk, self.output] + ([self.output_hc] if hasattr(self, "output_hc") else []) +
                     ([self.mtp] if hasattr(self, "mtp") else []), set()):
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
    # With the weights still to read: the small ones' memory now, their values after.
    if self._pending_load: _realize_later(small, self._pending_load, self._scratch)
    Tensor.realize(*tables, *([] if self._pending_load else small))

  def _finish_load(self):
    """the work left of loading (deferred): the weights read, the values computed from them"""
    self._derive_weights()
    while self._pending_load: self._pending_load.pop(0)()

  def _derive_weights(self):
    """the weights computed from the file's: their memory now, their values after the weights are read if they aren't yet (_pending_load)"""
    def realize(ts:list[Tensor]):
      if self._pending_load: _realize_later(ts, self._pending_load, self._scratch)
      else: Tensor.realize(*ts)
    # (SKINNY) the float weights of few outputs that decoding multiplies by (the routers, the indexers' head weights) stored transposed (the
    # outputs the fast axis), the Linear's a view of it: skinny_matmul reads every input's weights of all the outputs at once
    blocks = self.blk + ([self.mtp] if hasattr(self, "mtp") else [])
    lins = [l for b in blocks for l in (getattr(b, "ffn_gate_inp", None), getattr(getattr(b, "indexer", None), "proj", None),
                                        *(getattr(getattr(b, "indexer", None), p, None) for p in ("q_proj", "k_proj")),
                                        *(getattr(getattr(b, h, None), "inject", None) for h in ("hc_attn", "hc_ffn")))
            if isinstance(l, Linear) and l.ggml_type is None and l.weight.dtype in (dtypes.float32, dtypes.half) and l.weight.ndim == 2 and
            getattr(l, "bias", None) is None and not hasattr(l, "weight_t") and amd_custom_kernels_supported(l.weight.device) and (1)]
    if lins:
      realize(wts:=[l.weight.T.contiguous() for l in dedup(lins)])
      for l, wt in zip(dedup(lins), wts): l.weight_t, l.weight = wt, wt.T
    # (gated hyper-connections: the inject weights in float for ghc_pre, zeros without (a mix alone))
    ghcs = [g for b in blocks for g in (getattr(b, "hc_attn", None), getattr(b, "hc_ffn", None), getattr(b, "hc_head", None))
            if isinstance(g, GatedResidual)] + ([self.output_hc] if self.gated else [])
    winj = []
    for g in ghcs:
      if g._winj is not None: continue
      if (inj:=getattr(g, "inject", None)) is None: g._winj = Tensor.zeros(1, g.count*g.dim, device=g.norm["weight"].device).contiguous()
      elif inj.ggml_type is None: g._winj = inj.weight.float().contiguous()
      else:
        raw = inj.weight.bitcast(dtypes.uint8).flatten()
        g._winj = ggml_data_to_tensor(raw, g.count*g.count*g.dim, inj.ggml_type).reshape(g.count, -1).contiguous()
      winj.append(g._winj)
    if winj: realize(winj)

  def sequence_parallel(self, rows:int) -> bool:
    """the hyper-connection streams of a prefill chunk of `rows` are split over the devices: tiles of 16 rows on every device"""
    devs = self.token_embd.weight.device
    return isinstance(devs, tuple) and self.config.hc is not None and rows % (16*len(devs)) == 0 and bool((1))

  def prefill_chunks(self, chunk_size:int|None=None) -> tuple[int, ...]:
    """the prefill chunk sizes, ascending: the custom kernels compute chunks padded to a static size, a piece of a prompt runs the smallest
    one it fits (its own jit), large chunks are efficient. sequence parallel: whole tiles of rows on every device"""
    fast = amd_custom_kernels_supported(self.token_embd.weight.device)
    default_chunk = 4096 if self.max_context > 65536 else 2048
    chunk_size = chunk_size or min(getenv("PREFILL_CHUNK", default_chunk if fast else 32), self.max_context)
    if self.has_recurrent_block and not fast: chunk_size = 1
    sizes = [chunk_size] + [min(chunk_size, n) for n in (((192), (512), (2048),
                                                          4096) if fast else ())]
    # (PREFILL_SIZES: more sizes, less padding of a prompt's last piece (the turns of agents: a few hundred tokens), a jit each)
    if fast and (more:=getenv("PREFILL_SIZES", "")): sizes += [min(chunk_size, int(n)) for n in str(more).split(",")]
    if fast and (devs:=self.token_embd.weight.device) and isinstance(devs, tuple) and self.config.hc is not None and (1):
      # whole tiles of 16 rows on every device, the kernels of the whole chunk on tiles of 64
      tile = math.lcm(16*len(devs), 64)
      def tiled(n:int) -> int: return min(max(tile, round(n / tile) * tile), self.max_context // tile * tile or tile)
      # (PREFILL_CHUNK_UP, default on: the biggest chunk whole tiles up, not the nearest (6 devices: 4096 -> 4224, not 4032): a 4k
      # prompt (4096 tokens and its template) one chunk, not 4032 and a padded 192. other chunk ends: other greedy tokens (4k, 262k);
      # Quality bar passed (8,064-position NLL inside the chunk-noise floor, KL within the noise pair's))
      up = [min(-(-chunk_size // tile) * tile, self.max_context // tile * tile or tile)]
      sizes = [tiled(n) for n in sizes] + up
    return tuple(sorted(set(sizes)))

  def warmup(self):
    """compile and capture every jit a request uses (they capture on their first call): a prompt of each prefill chunk size, decoding"""
    # (GLM_SEQ: the sequences sharing the caches' jits)
    if self.config.cache_rows and self.seqs_supported(): return self.seq_warmup()
    sizes = self.prefill_chunks()
    # (a sequence sharded cache (a long context): one after the other, the weights read before the caches are made (their memory is the device's))
    self._capture_parallel(sizes)
    self._finish_load()
    # the big chunk first: its scratch memory is the most, the other jits share it. a prompt one past a size runs the next one (1: the smallest)
    for n in sorted({1, *(s + 1 for s in sizes[:-1])}, reverse=True):
      # (WARMUP_JITS names: prefill<chunk>, step<m>, bpf<P>_<G>, slots and host; other graphs are captured on their first call)
      if not warmup_wanted(f"prefill{next((s for s in sizes if s >= n), sizes[-1])}"): continue
      self._cached_tokens, self._ckpt_tokens = [], []
      list(zip(range(4), self.generate([0] * min(n, self.max_context - 8))))
    if self._ckpt_tokens: self._checkpoint(self._ckpt_tokens, save=False)  # and a checkpoint's restore
    self._cached_tokens, self._ckpt_tokens = [], []

  def _capture_parallel(self, sizes:tuple[int, ...]):
    """the jits of the prefill chunk sizes and of speculative decoding compiled at the same time (in forked processes), called as generate does"""
    if not (devs:=self.token_embd.weight.device) or not amd_custom_kernels_supported(devs): return
    self._init_states()
    from tinygrad.llm.kernels.amd_glm import host_allreduce, p2p_allreduce
    # (the device memory the jits use is made here: not in the processes. the resources they share too: the n-gram tables, the collectives')
    if isinstance(devs, tuple): self._seq_shared()
    if isinstance(devs, tuple) and p2p_allreduce(devs) is None: host_allreduce(devs)
    if not hasattr(self, "_token_buf"): self._token_host, self._token_buf = _host_ints(devs, self._token_rows())
    self._token_prev:list[int]|None = None  # (the captures write the buffer: the next generate writes all of it)
    # the random state of every device (sampling's noise) made here: one made in a capture's process isn't the one the jits run with, its
    # counter wouldn't advance (every step and request the same noise)
    for d in to_tuple(devs):
      key, ctr = Tensor._next_counter(d, 0)
      Tensor.realize(key, ctr, Tensor._device_rng_counters[d])
    v_start_pos, calls = UOp.variable("start_pos", 0, self.max_context-1), cast(list[tuple[Any, Callable[[], Any]]], [])
    for n in sizes:
      sp, nt = v_start_pos.bind(0), UOp.variable("toks", 1, n).bind(n)
      jits = self.__dict__.setdefault("jits" + ("_dense" if self._dense_ctx(sp, nt) else "") + "_greedy", {})
      if n not in jits: jits[n] = TinyJit(self._forward_effects, capture_first=True, scratch=self._scratch)
      setattr(jits[n], "warmup_name", f"prefill{n}")
      calls.append((jits[n], functools.partial(self, self._token_buf[:, sp:sp+nt], sp, None)))
    # the computations left of loading (their jits) too: run after the weights are read
    calls += [(j, j) for j in self._pending_load if isinstance(j, _TinyJit)]
    if self.config.spec_drafts > 0:
      self._spec_init()
      if not hasattr(self, "_spec_jit_greedy"): self._spec_jit_greedy = TinyJit(self.forward_spec, capture_first=True, scratch=self._scratch)
      setattr(self._spec_jit_greedy, "warmup_name", "spec")
      # include the dense-indexer speculative JIT when this context needs it
      if self._dense_ctx(0, 1):
        self._spec_jit_dense_greedy = TinyJit(self.forward_spec, capture_first=True, scratch=self._scratch)
        setattr(self._spec_jit_dense_greedy, "warmup_name", "spec_dense")
        calls.append((self._spec_jit_dense_greedy, functools.partial(self._spec_call, 0, 0, 1, None)))
        if (top:=unwrap(self.config.indexer).top_k) + 2*self.config.spec_drafts + 2 < self.max_context:
          calls.append((self._spec_jit_greedy, functools.partial(self._spec_call, top, 0, 1, None)))
      else: calls.append((self._spec_jit_greedy, functools.partial(self._spec_call, 0, 0, 1, None)))
    # the weights are read (and what's computed from them) while the jits compile, and a checkpoint is saved and restored (its kernels)
    def meanwhile(take):
      # (the jits of the computations left run after: they're compiled meanwhile)
      while self._pending_load and not isinstance(self._pending_load[0], _TinyJit): self._pending_load.pop(0)()
      if self.has_recurrent_block and (1): self._checkpoint([0], save=True), self._checkpoint([0], save=False)
    # the work left of loading as soon as its jits are taken, then the slots' memory (the memory loading used is free): the jits of batched
    # decoding that come after are linked as they come (while the others compile)
    loads, load_jits = {id(j) for j in self._pending_load if isinstance(j, _TinyJit)}, [j for j in self._pending_load if isinstance(j, _TinyJit)]
    def taken(jit:_TinyJit):
      loads.discard(id(jit))
      if loads: return
      self._finish_load()
      # (run once: their graphs and the buffers they read are freed)
      for j in load_jits: j.reset()
      load_jits.clear()
      gc.collect()
      for j, _ in calls:
        if j.captured is None and getattr(j, "link_later", False): j.link_later = False
    # (WARMUP_JITS: the jits of the names wanted (see warmup), the others captured on their first call; the loads' always)
    calls = [(j, c) for j, c in calls if j in load_jits or warmup_wanted(getattr(j, "warmup_name", ""))]
    capture_parallel(calls, meanwhile, taken)

  def _recurrent_states(self) -> list[tuple[Tensor, bool]]:
    # the state that isn't indexed by position: the recurrent and conv states (slotted: the ones of the last positions, a position p's in slot
    # p % (rewind+1)), the indexers' rings of the last tokens, the hidden states of speculative decoding
    states:list[tuple[Tensor, bool]] = [(self._spec_h, False)] if hasattr(self, "_spec_h") else []
    for b in self.blk + ([self.mtp] if hasattr(self, "mtp") else []):
      if isinstance(b, GatedDeltaNetBlock) and hasattr(b, "recurrent_state"):
        states += [(s, True) for s in (b.recurrent_state, *getattr(b, "conv_states", [getattr(b, "conv_state", None)])) if s is not None]
      if (idx:=getattr(b, "indexer", None)) is not None and hasattr(idx, "ring"): states.append((idx.ring, False))
    return states

  def _checkpoint(self, tokens:list[int], save:bool):
    """
    a copy of the recurrent state at the end of a prompt (save) or back from it: the next prompt has this one, but the model's reply can come
    back different (re-rendered by a chat template, re-tokenized): the reply and what follows are prefilled again, not everything
    """
    if not (states:=self._recurrent_states()): return
    # only the slot of the prompt's end of a slotted state (a variable: the same kernels for every prompt)
    slot = UOp.variable("ckpt_slot", 0, self.config.rewind).bind(len(tokens if save else self._ckpt_tokens) % (self.config.rewind + 1))
    if not hasattr(self, "_ckpt"):
      self._ckpt = [v.clone().realize() for v in self._ckpt_views(states, slot)]
      # (a jit each: a request saves one, they're launched, not scheduled and linked every time)
      self._ckpt_jits = {sv: TinyJit(functools.partial(self._ckpt_copy, sv), capture_first=True) for sv in (True, False)}
    self._ckpt_jits[save](slot)
    if save: self._ckpt_tokens = tokens[:]

  @staticmethod
  def _ckpt_views(states:list[tuple[Tensor, bool]], slot:UOp) -> list[Tensor]:
    return [s.shrink(((slot, slot+1),) + (None,)*(s.ndim-1)) if slotted else s for s, slotted in states]
  def _ckpt_copy(self, save:bool, slot:UOp):
    views = self._ckpt_views(self._recurrent_states(), slot)
    pairs = zip(self._ckpt, views) if save else zip(views, self._ckpt)
    Tensor.realize(*[dst.assign(src) for dst, src in pairs])

  def _reuse(self, tokens:list[int]) -> tuple[int, bool]:
    # recurrent state can't be partially reused after divergence: the cached prefix if the tokens extend it, else the checkpoint's (restored)
    cached, ckpt = self._cached_tokens, getattr(self, "_ckpt_tokens", [])
    if cached and len(cached) < len(tokens) and tokens[:len(cached)] == cached: return len(cached), False
    if ckpt and len(ckpt) <= len(tokens) and tokens[:len(ckpt)] == ckpt: return len(ckpt), True
    return 0, False

  def get_start_pos(self, tokens:list[int]) -> int:
    if self.has_recurrent_block: return self._reuse(tokens)[0]
    prefix_len = sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], zip(tokens[:-1], self._cached_tokens)))
    return min(block._reusable_prefix_len(prefix_len, len(self._cached_tokens)) for block in self.blk)

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
    temp = None
    if temperature != 0 and (temp:=self._temps.get((temperature, top_p))) is None:
      temp = self._temps[(temperature, top_p)] = Tensor([temperature, top_p]).to(self.token_embd.weight.device).realize()
    if fast:=amd_custom_kernels_supported(self.token_embd.weight.device):
      # in host memory: every step writes its sampled token after its input tokens (see forward), the host polls it
      if not hasattr(self, "_token_buf"):
        self._token_host, self._token_buf = _host_ints(cast(str|tuple[str, ...], self.token_embd.weight.device), self._token_rows())
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
    restore = False
    if self.has_recurrent_block:
      start_pos, restore = self._reuse(tokens)
      if restore: self._checkpoint(tokens, save=False)
      # a prefill from before the checkpoint overwrites the caches it relies on
      elif start_pos < len(getattr(self, "_ckpt_tokens", [])): self._ckpt_tokens = []
    if spec:=self.config.spec_drafts > 0 and fast:
      self._spec_init()
      # the MTP layer's input of the first prompt token: the hidden state of the one before (a zero one at the start)
      if start_pos == 0 or restore: self._spec_row = 0
      # (in a jit: a realize outside one walks every tensor of the model)
      if start_pos == 0 or self._spec_row: self._spec_row0(0 if start_pos == 0 else self._spec_row)
    out, prompt_len, ahead = None, len(tokens), False
    # (n-gram embeddings: the rows of the prompt's positions, then of every token sampled before the step that reads it: no run ahead)
    if (ple:=self.config.ple is not None and not (fast and self._ple_dev())): self._write_ple_rows(tokens, start_pos)
    while len(tokens) < self.max_context:
      n_toks = min(chunk_size, len(tokens) - start_pos)
      # the prompt in chunks (symbolic sizes), then one token at a time (the decode jit)
      sp, nt = v_start_pos.bind(start_pos), 1 if start_pos >= prompt_len else v_toks[next(n for n in sizes if n_toks <= n)].bind(n_toks)
      # with run ahead, the previous step already submitted this one
      if not ahead:
        # the same prompt again: its checkpoint already has the last token's state and hidden state; sample its reply, don't prefill it twice
        if not n_toks and restore: out = self._sample(self._spec_h[0:1], temp, devs).realize()
        else: out = self(t[:, sp:sp+nt] if start_pos < prompt_len or out is None or fast else out, sp, temp).realize()
      # the token is polled only if the call wrote it to the token buffer (a replaced __call__ doesn't)
      if fast and (out is None or out.uop.base is not self._token_buf.uop.base): fast = False
      start_pos += n_toks
      # chunked prefill: keep processing until all prompt tokens are consumed
      if start_pos < len(tokens): continue
      # the end of the prompt: a checkpoint of the recurrent state (before any step of the reply)
      if self.has_recurrent_block and start_pos == prompt_len and (1): self._checkpoint(tokens, save=True)
      if spec:
        tok = self._wait_host(lambda: self._token_host[start_pos], "a token of generate") if fast else int(cast(Tensor, out).item())
        tokens.append(tok)
        self._cached_tokens, self._spec_row = tokens[:-1], 0
        yield tok
        yield from self._spec_steps(tokens, temp)
        return
      if fast:
        # run ahead: submit the next step (it reads this token on the device) before waiting for this one. it writes the recurrent states of
        # the next position, the ones before it are kept (see GatedDeltaNetBlock._attention): the caches still hold every token but the last
        if ahead:=len(tokens) + 1 < self.max_context and not ple: self(t[:, (sp:=v_start_pos.bind(start_pos)):sp+1], sp, temp)
        tok = self._wait_host(lambda: self._token_host[start_pos], "a token of generate")
      tokens.append(tok if fast else int(cast(Tensor, out).item()))
      if ple: self._write_ple_rows(tokens, len(tokens) - 1)
      self._cached_tokens = tokens[:-1]
      yield tokens[-1]
