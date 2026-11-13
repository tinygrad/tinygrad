from __future__ import annotations
import array, collections, contextlib, enum, functools, gc, itertools, math, os, pathlib, random, re, time
from dataclasses import dataclass, replace, field
from typing import Any, Callable, Iterator, cast
from tinygrad import Tensor, nn, UOp, TinyJit, getenv, function, dtypes, Device
from tinygrad.engine.realize import warm_worker_pool
from tinygrad.engine.jit import capture_parallel, warmup_select, warmup_wanted, _TinyJit, run_as_they_come
from tinygrad.engine.realize import run_linear, compile_linear, link_linear
from tinygrad.llm.kernels.amd_glm import rows_shifted, rows_pick, pool_expand
from tinygrad.llm.kernels.amd_glm import Linear, gated_delta_prefill, flash_attention, amd_custom_kernels_supported, find_packed, moe_linear, \
  hc_coeffs, hc_pre, q8_quantize, q8_linear, QUANT_SIZES, q8_quantize_cache_clear, amd_flash_attention_decode as flash_attention_decode, moe_route, \
  moe_grouped_linear, amd_flash_attention_mla_prefill as flash_attention_mla_prefill, partial_sum, fuse_gate_up, kda_decode, moe_router, \
  indexer_scores, _wmma_rdna4, pool_keys, pool_index_rows, slot_scatter, slot_gather, \
  indexer_decode_index, amd_flash_attention_mla_sparse as flash_attention_mla_sparse, moe_combine, token_sharded, shard_rows, reduce_scatter, \
  all_gather, all_gather_q8, sharded_argmax, skinny_matmul, moe_combine_scatter, kda_conv, kda_replay, tables_ready, moe_router_local, \
  hc_post_quant, \
  indexer_decode_index_rows, Q8_ROWS_I8, MOE_TILE, I8_WMMA_QUANTS, flash_attention_rows, \
  grouped_rows, q8_quantize_known, SLOT_SPAN, ple_gather, ghc_pre, ghc_post, ghc_prefill_norm, ghc_prefill_mix, moe_softmax_router, Q8_0, IQ3_S, \
  mla_bmm, bmm_wmma, weighted_sum, ghc_decode_mix, qsa_prefill_mask, qsa_sparse_prefill, rows_gather, rows_scatter, seq_scores, seq_index, \
  kv_q8_pack, kv_q8_unpack, qsa_decode_attention, FLAG_LANE, DEC_LANE, IDX_LANE, \
  flag_allreduce, ring_mailbox, ring_queue_of, RING_QUEUE_ON, mb_queue, state_token, f32t_gemm, few_out_rows, index_scores, after_kernel, \
  host_allreduce, p2p_allreduce, _argmax_parts, _device_index, _unbind, f16t_gemm, \
  nucleus, seq_store, mla_sparse_seq, seq_rows, SEQ_BLK, seq_gather, seq_views, seq_store_grouped, fp8_rows, FP8_ROW, IDX_FP8_ROW, q4_rows, Q4_ROW, \
  pool_index_shard, top_k_mask
from tinygrad.llm.gguf import gguf_parse, ggml_data_to_tensor, _GGML_QUANT
from tinygrad.llm.gguf_load import gguf_shard_units, _disk_bytes, _disk_location
from tinygrad.uop.ops import resolve, Ops, AxisType, KernelInfo
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

def _qwen_mtp_file(model:pathlib.Path) -> pathlib.Path|None:
  """the MTP layer's gguf of a qwen4exp model (QWEN_MTP, else MTP/mtp-*-shared-*.gguf in its directory or the one above)"""
  if (env:=getenv("QWEN_MTP", "")): return pathlib.Path(env)
  for d in (model.parent, model.parent.parent):
    if (found:=sorted(d.glob("MTP/mtp-*-shared-*.gguf"))): return found[-1]
  return None

def _qwen4exp_entries(kv:dict, entries:dict[str, tuple[Tensor, tuple[int, ...], int]], shard:int) -> tuple[dict, int, dict]:
  """
  qwen4exp's tensors as the model's: the KV heads repeated so every device holds one (shard a multiple of them), the gated deltanets in the kda
  layout (attn_qkv and its conv split in q, k and v, the q and k heads repeated for the v heads). on devices, as the rows of every device of the
  file's tensors (gguf_shard's rows: read straight from the file): a device's v heads use a run of the k heads. returns the entries, the KV
  heads and the rows
  """
  entries, arch = dict(entries), kv['general.architecture']
  n_kv, rows = kv[f"{arch}.attention.head_count_kv"], {}
  hk, hv, K = kv[f'{arch}.ssm.group_count'], kv[f'{arch}.ssm.time_step_rank'], kv[f'{arch}.ssm.state_size']
  def cut(data:Tensor, r0:int, r1:int, n:int) -> Tensor:
    rb = int(data.shape[0]) // n
    return data[r0*rb:r1*rb]
  if shard > n_kv and shard % n_kv == 0:
    for name in [k for k in entries if k.endswith(('attn_k.weight', 'attn_v.weight', 'attn_k.bias', 'attn_v.bias'))]:
      data, shape, typ = entries[name]
      hd = shape[0] // n_kv
      entries[name], rows[name] = (data, (shard * hd, *shape[1:]), typ), (shape[0], [(d * n_kv // shard * hd, hd) for d in range(shard)])
    n_kv = shard
  for name in [k for k in entries if k.endswith('attn_qkv.weight')]:
    blk = name[:-len('attn_qkv.weight')]
    for src, dst in (('attn_qkv.weight', 'attn_{}.weight'), ('ssm_conv1d.weight', 'ssm_conv1d_{}.weight')):
      data, shape, typ = entries.pop(blk + src)
      n = shape[0]
      # (the v heads share the k heads by tiling: v head h's q and k are k head h % hk's, as GatedDeltaNetBlock's repeat)
      for c, (r0, r1) in zip("qk", ((0, hk*K), (hk*K, 2*hk*K))):
        part = cut(data, r0, r1, n)
        if shard > 1 and hv % shard == 0 and all((d*hv//shard) % hk + hv//shard <= hk for d in range(shard)):
          entries[blk + dst.format(c)] = (part, (hv * K, *shape[1:]), typ)
          rows[blk + dst.format(c)] = (hk*K, [((d*hv//shard) % hk * K, hv//shard * K) for d in range(shard)])
        else:
          t = part.to("CPU").reshape(hk*K, -1).repeat(hv // hk, 1)
          entries[blk + dst.format(c)] = (t.flatten(), (hv * K, *shape[1:]), typ)
      entries[blk + dst.format("v")] = (cut(data, 2*hk*K, n, n), (n - 2*hk*K, *shape[1:]), typ)
  return entries, n_kv, rows

def expert_slice_table(units:list[int], experts:int, slices:int) -> list[list[int]]:
  """(experts, w): the local index of every slice a device holds of an expert, -1 padded"""
  table:list[list[int]] = [[] for _ in range(experts)]
  for i, u in enumerate(units): table[u // slices].append(i)
  w = max(len(t) for t in table)
  return [t + [-1]*(w-len(t)) for t in table]

_block_q8:list[tuple[UOp, tuple[Tensor, ...]]|None] = [None]  # a decoding block's output and its quantization (the next block's input's)
INLINE_ACT = getenv("INLINE_ACT", 1)  # a decoding activation computed in the kernel that quantizes it (not a kernel of its own)

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

def short_conv(state:Tensor, x:Tensor, weight:Tensor, initial:Tensor, dilation:int=1) -> tuple[Tensor, UOp]:
  assert state.shape[1] == (weight.shape[1]-1)*dilation, f"state has {state.shape[1]} rows, expected {(weight.shape[1]-1)*dilation}"
  # assemble the conv window in a static-size buffer: [state | x rows | zero-pad], state is zeroed when initial
  T, T_pad = x.shape[1], x.max_shape[1]
  window = initial.where(0, state).cat(x.cast(state.dtype), dim=1).pad_to((x.shape[0], state.shape[1] + T_pad, x.shape[-1])).contiguous()
  # the last state.shape[1] rows before the padding become the next state
  store = state.uop.store(window[:, T:T+state.shape[1]].uop)
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
  # (the qwen sparse attention's caches: rows of the regions of the sequences sharing them (see Transformer.regions), at least max_context)
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
  qsa: bool = False  # the attention layers are qwen sparse attention (QSABlock)
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
# batched decoding with the streams' rows split over the devices (SEQ_SPLIT, see Transformer._forward_seqs): the real rows of a device (its
# rows are padded to whole tiles of 32), 0: not split
_seq_split = [0]
SEQ_SPLIT_MAX = 128  # (the most real rows a device)
def _split_rows() -> int: return round_up(_seq_split[0], 32)
def _split_real(x:Tensor) -> Tensor:
  """(B, N*P, ...) split padded rows: the real rows (B, N*R, ...), split"""
  N, R, B, rest = len(cast(tuple, x.device)), _seq_split[0], x.shape[0], x.shape[2:]
  return x.reshape(B, N, _split_rows(), *rest)[:, :, :R].reshape(B, N*R, *rest)
def _split_pad(x:Tensor) -> Tensor:
  """(B, N*R, ...) split real rows: the padded rows (B, N*P, ...), split (zeros past the real ones)"""
  N, R, B, rest = len(cast(tuple, x.device)), _seq_split[0], x.shape[0], x.shape[2:]
  P = _split_rows()
  return x.reshape(B, N, R, *rest).pad((None, None, (0, P - R), *[None]*len(rest))).reshape(B, N*P, *rest)
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
MOE_DECODE_ROWS = getenv("MOE_DECODE_ROWS", 96)
# batched decoding: rows of experts decoded (in bundles) up to this many, grouped (the prefill kernels) past it
MOE_BATCH_ROWS = getenv("MOE_BATCH_ROWS", 512)
def _skinny(lin:Linear, x:Tensor) -> Tensor|None:
  """lin(x) by skinny_matmul (few outputs, few rows) if its weight is stored for it (see Transformer._finish_load), else None"""
  if (wt:=getattr(lin, "weight_t", None)) is None or not isinstance(x.shape[-2], int): return None
  # (rows split over the devices (a prefill's): a few a device)
  if token_sharded(x) and int(x.uop.shard_shape[-2]) > getenv("SKINNY_SPLIT_ROWS", 512): return None
  return skinny_matmul(x, wt)

def _decode_rows() -> int: return MOE_BATCH_ROWS if _batch_rows[0] is not None else MOE_DECODE_ROWS

# batched decoding (Transformer.forward_batch): the rows are groups of a sequence each, group g (its rows the tokens at positions pos.. of the
# sequence) in the state slot g: (pos, rows) of every group
_batch_rows:list[list[tuple[UOp, int]]|None] = [None]

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
MIX_SHARED = getenv("MIX_SHARED", 1)
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

# SEQ_REPLAY: a sequence slot's recurrent state is one (after the first token of its last step) and the inputs of the tokens after it, replayed
# by the next step (see kda_decode): a step writes one state, not k+1 (their memory and their writes)
SEQ_REPLAY = getenv("SEQ_REPLAY", 1)
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
  return (k + 1 if k and getenv("SLOT_STATES_TIGHT", 1) else k + 2), k + 2

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

def _unallocated(*sizes:int|UOp, dtype:DType, device:str|tuple[str, ...]|None, axis:int|None=None) -> Tensor:
  """a tensor whose buffer is made but not allocated: a jit captured before its memory is (it holds it, not an intermediate of its own)"""
  shape, dev = tuple(int(s) for s in sizes), cast(str|tuple[str, ...], unwrap(device))
  if axis is not None and isinstance(dev, tuple):
    local = tuple(s//len(dev) if i == axis else s for i, s in enumerate(shape))
    return Tensor(UOp.new_buffer(dev, prod(local), dtype).reshape(local).unshard(axis))
  return Tensor(UOp.new_buffer(dev, prod(shape), dtype).reshape(shape))

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

def _shared_host(owner:str, devs:tuple[str, ...], nbytes:int) -> list[Buffer]:
  """nbytes of host memory (made when used) mapped on every device: its buffer of every device (the owner's, the others' its mappings)"""
  host = Buffer(owner, max(nbytes, 1), options=BufferSpec(host=True))
  class _Alias(Buffer):
    # (the owner's mapping on another device: allocated with the owner's)
    def allocate(self, opaque=None, external_ptr=None):
      a = host.ensure_allocated().get_buf(self.device)
      return super().allocate(opaque=BufferStorage(a), external_ptr=a)
  return [host if d == owner else _Alias(d, max(nbytes, 1)) for d in devs]

@functools.cache
def _stash_save_kernel(dst:UOp, src:UOp, dev:UOp, row:UOp, at:UOp, n:int, writers:int) -> UOp:
  # rows row..row+n of a cache (a device's: (.., rows, D)) into rows at.. of its copy in host memory, by the devices of writers (8 bits a device:
  # its part and the count of the devices that write the copy (0: none), a row of every count of them by each)
  D = cast(int, src.shape[-1])
  def flat(t:UOp) -> UOp: return t.reshape(prod(cast(tuple[int, ...], t.shape)))
  s, d = flat(src), flat(dst)
  # (a thread 16 bytes next to each other: a wave's stores whole lines of the host memory (wider ones are split, half lines it writes slowly))
  V = 16 // src.dtype.itemsize if D % (16 // src.dtype.itemsize) == 0 else 1
  r, w = UOp.range(n, 0, AxisType.GLOBAL), UOp.range(D // V, 1, AxisType.LOCAL)
  code = (UOp.const(writers, dtypes.int64) >> (dev[0].load().cast(dtypes.int64) * 8)) & 0xFF
  part, count = (code & 0xF).cast(dtypes.weakint), (code >> 4).cast(dtypes.weakint)
  ok = count.ne(0) & ((r % count.maximum(1)).eq(part))
  stores = [d[((at + r) * D + w * V + j).valid(ok)].store(s[(row + r) * D + w * V + j].load()) for j in range(V)]
  return UOp.group(*stores).end(r, w).sink(arg=KernelInfo(name="stash_save", opts_to_apply=()))

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

def _temp_pair(t:float|tuple[float, ...]) -> array.array:
  """a row's temperature and top_p (a temperature alone: top_p 1) for a step's host buffer"""
  return array.array('f', t[:2] if isinstance(t, tuple) else (t, 1.0))  # (a top_k: the batch steps have none)

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
  return isinstance(device, tuple) and len(device) > 1 and amd_custom_kernels_supported(device) and bool(getenv("MLA_SEQ_SHARD", 0))

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
      if isinstance(norm, nn.RMSNorm) and norm.weight is not None and not split and not rows and xs.shape[0] == 1 and getenv("HC_POST_QUANT", 1):
        coef, hn = hc_pre(mix, sq, base, scale, xs, norm.weight, self.count, self.iters, self.eps, self.norm_eps, norm.eps, features, raw=True)
        return hc_post_quant(sublayer(hn[:, :T]), coef, xs, self.count)
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
    if isinstance(B, int) and isinstance(T, int) and getenv("GHC_ROWS", 8) < B*T <= getenv("GHC_BATCH_ROWS", 1024) and \
       amd_custom_kernels_supported(xs.device) and \
       not _wmma_rdna4(xs.device) and self.down.ggml_type == self.up.ggml_type == Q8_0 and self.count == 4 and self.dim % 256 == 0 and \
       getattr(self, "_winj", None) is not None and not token_sharded(xs) and getenv("GHC_BATCH", 1):
      # (the rows of batched decoding: the prefill's kernels on whole tiles of 32 rows (the rows past them padding), not a dozen of generic ones)
      n = B*T
      flat = xs.reshape(n, self.count*self.dim).float().pad(((0, round_up(n, 32) - n), None))
      xn, xhalf, part = ghc_prefill_norm(flat, self.norm["weight"], unwrap(self._winj), self.count, self.eps)
      lo = q8_linear(self.down, xhalf) * (1 / self.count)
      mixed = ghc_prefill_mix((lo * lo.sigmoid()).contiguous(), self.up.weight, xn)
      return mixed[:n].reshape(B, T, self.dim), (part.sum(1) * (1 / self.count))[:n].reshape(B, T, -1)[..., :self.count]
    if not (isinstance(B, int) and isinstance(T, int) and B*T <= getenv("GHC_ROWS", 8) and getenv("GHC_FUSED", 1) and
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
    if B*T <= getenv("GHC_POST_ROWS", 1):
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
    if _seq_split[0] and token_sharded(xs): return self._split_call(xs, sublayer)
    if (fused:=self._fused(xs)) is not None: mixed, inj = fused
    else:
      mixed, xn = self.mix(xs)
      inj = (s if (s:=_skinny(self.inject, xn)) is not None else self.inject(xn)) / self.count
    out = sublayer(mixed)
    if out.shape[1] is not xs.shape[1]: out = out.pad_to((out.shape[0], xs.shape[1], out.shape[2]))
    return xs + out.unsqueeze(2) * (inj.sigmoid() * 2).unsqueeze(-1)

  def _split_call(self, xs:Tensor, sublayer:Callable[[Tensor], Tensor]) -> Tensor:
    """
    batched decoding with the streams' rows split over the devices (padded, see _seq_split): every device the mix and coefficients of its rows,
    the sublayer sees all the real rows (gathered) and returns its partial sums, every device sums its rows of them (as a prefill's phases)
    """
    mixed, post, _ = self.pre_split(xs)
    own = _split_real(mixed).contiguous()
    x = all_gather_q8(own, 1) if getenv("AG_Q8", 1) and own.shape[-1] % 256 == 0 else all_gather(own, 1, dtypes.half)
    _partial_out[0], _sp_rows[0] = True, own
    try: out = sublayer(x)
    finally: _partial_out[0], _sp_rows[0] = False, None
    # (summed and split already: moe_combine_scatter)
    done = out.ndim == own.ndim and token_sharded(out)
    out = out if done else reduce_scatter(out, 1) if out.ndim == own.ndim + 1 else shard_rows(out, 1)
    return self.post_split(_split_pad(out), post, post, xs)

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
      elif isinstance(T:=xr.shape[1], int) and 16 < T < 128 and T % 16 and getenv("ROUTER_PAD", 1):
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
         amd_custom_kernels_supported(x.device) and self.config.num_experts <= 1024 and getenv("MOE_ROUTER", 1):
        k, scale, exps = self.config.num_experts_per_tok, self.config.routed_scaling_factor, self.ffn_gate_exps
        exps.set_quantized()
        if rows is None and xr.shape[1] <= _decode_rows() and exps.slices and exps.ggml_type is not None and getenv("MOE_ROUTER_LOCAL", 1):
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
    if (pf:=_batch_prefill[0]) is not None: sel = pf.real.reshape(-1, 1).to(sel.device).where(sel, -1)
    if (T:=_ffn_rows[0]) is not None and sel.ndim >= 3 and isinstance(R:=sel.shape[-2], int) and getenv("MOE_SKIP_PAD", 1):
      sel = (Tensor.arange(R).reshape(R, 1) < (Tensor(T) if isinstance(T, UOp) else T)).to(sel.device).where(sel, -1)
    # many tokens: grouped by expert (prefill), a few (the drafts of speculative decoding) as decoding
    grouped = (not isinstance(x.shape[1], int) or x.shape[1] > _decode_rows()) and all(e.ggml_type is not None for e in exps)
    shexp:Tensor|None = None
    shexp_gate:Tensor|None = None
    if grouped and hasattr(self, 'ffn_gate_shexp') and self.ffn_gate_exps.expert_parallel and \
       isinstance(x.shape[1], int) and self.ffn_down_shexp.shard_axis == 1 and (not hasattr(self, 'ffn_gate_inp_shexp') or getenv("SHEXP_FUSED", 1)):
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
                                  dtypes.half if _batch_rows[0] is not None and self.ffn_down_exps.slices and
                                  (_seq_batch[0] is None or getenv("SEQ_HALF_DOWN", 0)) else dtypes.float32)
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
    i8 = gate_exps.ggml_type in I8_WMMA_QUANTS and gate_exps.ggml_type != IQ3_S and getenv("I8_WMMA_RDNA3", 1)
    tile = getenv("MOE_BATCH_TILE", 16) if _batch_rows[0] is not None and _batch_prefill[0] is None else \
      MOE_TILE if _wmma_rdna4(x.device) or getenv("MOE_TILE", 0) or i8 else 32
    # (MOE_TILE32_ROWS: RDNA4 tiles of 32 rows while the experts have at most this many rows each on average (a 64-row tile mostly padding))
    if tile == 64 and _wmma_rdna4(x.device) and isinstance(T, int) and (t32:=getenv("MOE_TILE32_ROWS", 0)) and \
       B * T * int(sel.shape[-1]) <= t32 * self.config.num_experts: tile = 32
    # (MOE_TILE32_TOKENS: RDNA3 tiles of 32 rows for a chunk of at most this many tokens (192: ~5 rows an expert, a 64-row tile mostly
    #  padding: the MoE layer 1.8x faster on a 7900 XTX at 192, 1.3x at 384, a chunk 4% at 768 and 1% at 1152, the same bits). 0: off)
    Tm = T if isinstance(T, int) else int(T.vmax)
    if tile == 64 and not _wmma_rdna4(x.device) and not getenv("MOE_TILE", 0) and B * Tm <= getenv("MOE_TILE32_TOKENS", 1152): tile = 32
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
    sorted_down = bool(getenv("MOE_SORTED_DOWN", 1))
    down = down_exps.grouped(act.contiguous(), entries, tiles, selections=tokens*k, out_dtype=dtypes.half if down_exps.slices else dtypes.float32,
                             sorted_out=sorted_down)
    # sequence parallel: summed over the devices and split on the tokens around the ring as they're combined
    if not self.gated and half and gate_exps.expert_parallel and _sp_rows[0] is not None and isinstance(T, int) and \
       tokens % len(cast(tuple, x.device)) == 0 and \
       tokens == T and getenv("MOE_SCATTER", 1):
      if add_scale is None: return moe_combine_scatter(down, pos, probs, tokens, k, add, by_selection=not sorted_down)
    y = moe_combine(down, pos, probs, tokens, k, add, add_scale=add_scale,
                    out_dtype=dtypes.half if half and gate_exps.expert_parallel else dtypes.float32,
                    by_selection=not sorted_down)
    y = y.reshape(-1, *sel_shape[:-1], D)[:, :, :x.shape[1] if isinstance(T, int) else T]
    return y if gate_exps.expert_parallel else y[0]

  # given the token-prefix match, return how much cached state this block can still reuse
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return prefix_len
  def _init_slots(self, n:int, ctx:int): pass  # the states of batched decoding (a block with states of its own)
  def _init_state(self, x:Tensor): raise NotImplementedError
  def _attention(self, x:Tensor, start_pos:int|UOp, T:int|UOp|None=None) -> Tensor: raise NotImplementedError

  def _ple(self, x:Tensor, emb:Tensor, start_pos:int|UOp, T:int|UOp|None, dep:UOp|None=None) -> Tensor: raise NotImplementedError

  def __call__(self, x: Tensor, start_pos: int|UOp, T:int|UOp|None=None, ple_emb:Tensor|None=None):
    # T (hyper-connections): x has static padded rows of which the first T are the tokens, the output too
    # ple_emb: the n-gram embedding rows of the tokens (the block with the PLE layer)
    self._init_state(x)
    # we pass in the weights implicitly so we unpack the GGUF on the fly
    # (decoding: the output's quantization made by its last kernel is returned too, the next block's input's: BLOCK_Q8_OUT)
    q8_out = T is None and self.config.hc is not None and not self.gated and bool(getenv("BLOCK_Q8_OUT", 1)) and not _seq_split[0]
    # (the block before's output's quantization, if x is that output: its function's arguments, the quantization of x in it)
    xq = _block_q8[0][1] if q8_out and _block_q8[0] is not None and _block_q8[0][0] is x.uop else ()
    if ple_emb is not None and not hasattr(self, "ple"): ple_emb = None
    @function(precompile=True, allow_implicit=True)
    def _run(x:Tensor, start_pos:int|UOp, T:int|UOp|None, ple_emb:Tensor|None, *xq:Tensor):
      if ple_emb is not None:
        if _seq_split[0] and token_sharded(x):
          # (split rows: the n-gram embedding of all the real rows, every device adds its rows of it)
          x = x + _split_pad(shard_rows(self._ple(all_gather(_split_real(x).contiguous(), 1, dtypes.half).float(), ple_emb, start_pos, T), 1))
        else: x = x + self._ple(x, ple_emb, start_pos, T)
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
      x = all_gather_q8(hn, 1) if getenv("AG_Q8", 1) and hn.shape[-1] % 256 == 0 else all_gather(hn, 1, dtypes.half)
      # Route each device's own rows; gather the small selections/probabilities rather than repeat the router projection everywhere.
      _partial_out[0], _sp_rows[0] = True, hn
      # (in half: the layers' kernels read it so, a cast to float and back is two copies)
      xin = x if getenv("SUB_HALF_IN", 1) else x.float()
      if sub != "attn": _ffn_rows[0] = T
      if state: _state_writes[0] = []
      try: out = self._attention(xin, start_pos, T) if sub == "attn" else self._feed_forward(xin)
      finally: _partial_out[0], _sp_rows[0], _ffn_rows[0] = False, None, None
      writes, _state_writes[0] = _state_writes[0], None
      # partial sums, or the whole output
      # (summed and split already: moe_combine_scatter)
      done = out.ndim == hn.ndim and token_sharded(out)
      # (MB_DEP=compute: the token after the sublayer's compute (a value of its output before its collective))
      if state and getenv("MB_DEP", "sub") == "compute": writes = [out.contiguous() if done else out]
      out = _fn_out(out if done else reduce_scatter(out, 1) if out.ndim == hn.ndim + 1 else shard_rows(out, 1))
      return (out, state_token(hn, writes or [])) if state else out
    state = (getenv("MB_DEP", "sub") == "compute" or (getenv("MB_DEP", "sub") == "state" and sub == "attn")) and _mb_active[0]
    if not state: return _sub(hn, start_pos, T)
    out, token = _sub(hn, start_pos, T)
    self._token:Tensor|None = token
    return out
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
      return (ret.cast(dtypes.half) if getenv("HC_STREAMS_HALF", 1) else ret).contiguous()
    return _post(out, post, comb, xs)

def _fn_out(t:Tensor) -> Tensor:
  """
  a function's output: a kernel's output as it is (its buffer: contiguous() of it in a function is a copy kernel, ~0.1 ms of 1152 rows of
  2560 floats, 500 a 13824 chunk), else made contiguous
  """
  return t if getenv("FN_OUT_DIRECT", 1) and t.uop.has_buffer_identity(after_ok=True) else t.contiguous()

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
    return _on_queue(outs, f"COMPUTE:{q}") if (q:=mb_queue(i)) and getenv("MB_QUEUES", 1) else outs
  lane_calls = getenv("MB_LANE_CALLS", 0)  # (1: a batch's ordering reads, 2: its n-gram embedding calls on its queue, else the first's)
  def after(i:int, x:Tensor, dep:Tensor|None, rows:bool=True) -> Tensor:
    if dep is None: return x
    # (MB_DEP_EFFECT: in the schedule's order only, the queues wait for what the kernels' buffers share: the states and caches, not the output)
    if getenv("MB_DEP_EFFECT", 0) and rows: return Tensor(x.uop.after(*[u for u in dep.uop.toposort() if u.op is Ops.CALL][-1:]))
    if getenv("MB_QUEUES", 1) and (not lane_calls & 1 or (i == 0 and getenv("MB_AFTER0_INLINE", 0))):
      z = (dep[..., :1] if rows else dep.flatten()[:1]).cast(x.dtype) * 0
      while rows and z.ndim < x.ndim: z = z.unsqueeze(-1)
      return x + (z != z).where(0, z)
    if getenv("MB_QUEUES", 1) and getenv("MB_AFTER_KERNEL", 1):
      # (a kernel of its own on the batch's queue: x and a value of dep (zero, also from a NaN or inf), every device its part)
      return queued(i, after_kernel(x, dep))[0]
    if getenv("MB_QUEUES", 1):
      # (a call of its own on the batch's queue (precompiled: a function that isn't is inlined, its kernel on the first queue). rows: dep is
      # split on its rows as x, a value of every row read)
      @function(precompile=True, allow_implicit=True)
      def _after(x:Tensor, dep:Tensor) -> Tensor:
        # (zero, also from a NaN or inf of dep (the padding rows of a chunk's last piece): not a value of it in x)
        z = (dep[..., :1] if rows else dep.flatten()[:1]).cast(x.dtype) * 0
        while rows and z.ndim < x.ndim: z = z.unsqueeze(-1)
        return (x + (z != z).where(0, z)).contiguous()
      return queued(i, _after(x, dep))[0]
    return Tensor(x.uop.after(*dep.uop.src[1:]))
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
      if lane_calls & 2:
        # (the gather, the embedding and its add: calls on the batch's queue)
        @function(precompile=True, allow_implicit=True)
        def _gather(x:Tensor) -> Tensor: return all_gather(x, 1).contiguous()
        p = queued(i, block._phase_ple(queued(i, _gather(xi))[0], ple[:, i*n:(i+1)*n].contiguous(), starts[i], ts[i]))[0]
        ple_done.append(p)
        @function(precompile=True, allow_implicit=True)
        def _add(x:Tensor, p:Tensor) -> Tensor: return (x + shard_rows(p, 1)).contiguous()
        xs[i] = queued(i, _add(xs[i], p))[0]
      else:
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
  skew = getenv("MB_SKEW", -1)
  if skew < 0: skew = len(subs)  # (a batch's sublayers all before the next batch's)
  for step in range(len(subs) + skew*(len(xs)-1)):
    for i in range(len(xs)):
      if not 0 <= (k:=step - i*skew) < len(subs): continue
      block, hc, norm, sub = subs[k]
      FLAG_LANE[0] = i
      if k == 0: phases[i] = pre(i, 0)
      hn, post, comb = phases[i]
      # (the batch before it ran this sublayer: its output's call)
      # (MB_LAG: the first batch's sublayer k after the last's k-MB_LAG (with MB_SKEW=0): the batches stay together, the collectives of one
      # run while the other computes, not a first batch ahead and the last alone at the end)
      lag = getenv("MB_LAG", 0)
      # (MB_DEP: what of the batch before a batch's sublayer waits for: its sublayer's output (sub), its streams after it (post), its next
      # sublayer's coefficients (pre))
      dep = deps[i-1][k] if i else done[-1][k-lag] if lag and k >= lag and len(xs) > 1 else None
      block._token = None
      out = queued(i, block._phase_sub(sub, after(i, hn, dep), starts[i], ts[i]))[0]
      done[i][k] = deps[i][k] = out
      # (MB_DEP=state: the next batch waits for the states this one wrote (none: a feed forward's), not its output)
      if getenv("MB_DEP", "sub") in ("state", "compute"): deps[i][k] = queued(i, block._token)[0] if block._token is not None else None
      xs[i] = queued(i, block._phase_post(hc, out, post, comb, xs[i]))[0]
      if getenv("MB_DEP", "sub") == "post": deps[i][k] = xs[i]
      if k + 1 < len(subs):
        phases[i] = pre(i, k+1)
        if getenv("MB_DEP", "sub") == "pre": deps[i][k] = phases[i][0]
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
    self.indexer:Indexer|None = Indexer(config) if config.indexer and not isinstance(self, QSABlock) else None

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

    #self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v))
    #k = self.cache_kv[0, :, :, 0:start_pos+T, :]
    #v = self.cache_kv[1, :, :, 0:start_pos+T, :]

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

# set while a call whose context the indexer's top_k covers is traced (its jit): the qwen sparse attention attends all the keys (see QSABlock)
_qsa_dense:list[bool] = [False]
# the first row of the sequence's region of the qwen sparse attention's caches (shared by sequences, see Transformer.regions) while its call is
# traced: a bound variable (or 0), a multiple of REGION_ALIGN
_seq_base:list[UOp|int] = [0]
# (a prefill chunk's time is about that of this many rows more than its padded ones: its kernels' fixed time, see Transformer._chunk_plan)
CHUNK_FIXED_ROWS = getenv("CHUNK_FIXED_ROWS", 256)
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
  return (sb:=_seq_batch[0]) is not None and sb.G == 1 and bool(getenv("SEQ_G1_MAIN", 1))

@dataclass(frozen=True)
class SeqPrompt:
  """
  a mixed step's prompt (Transformer.forward_seqs): P rows after the off rows of the sequences decoding, its count of tokens from start of the
  sequence of slot (its region from base), the slot's states of start continued (the variables, and int32 tensors (1,) of the devices). the
  effects of the blocks (its states into the slot) realized with the step
  """
  P:int; off:int; slot:UOp; base:UOp; start:UOp; count:UOp; slots:Tensor; starts:Tensor; counts:Tensor # noqa: E702
  effects:list = field(default_factory=list)
_seq_prompt:list[SeqPrompt|None] = [None]

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

class QSAIndexer:
  """
  qwen sparse attention's indexer: the keys of a pool of kpool tokens are averaged, normed and roped at the pool's first position, a query's (its
  heads normed and roped) score of a pool is the sum of its heads' rectified dots. its cache holds every token's key (the pools are made of them)
  """
  pool_k:Tensor  # batched decoding: the keys of every slot, their pooled keys (QSABlock._init_slots)
  pool_pk:Tensor
  def __init__(self, config:TransformerConfig):
    self.idx, self.max_context, self.cache_rows = (idx:=unwrap(config.indexer)), config.max_context, config.cache_rows
    self.q_proj, self.k_proj = Linear(config.dim, idx.n_heads * idx.head_dim, bias=False), Linear(config.dim, idx.head_dim, bias=False)
    self.q_norm, self.k_norm = nn.RMSNorm(idx.head_dim, idx.norm_eps), nn.RMSNorm(idx.head_dim, idx.norm_eps)

  def _init_state(self, x:Tensor):
    if hasattr(self, "cache_k"): return
    # every token's key, and the pooled keys (a call pools the pools of its tokens: the ones before them are cached). the rows of every sequence's
    # region (see Transformer.regions), REGION_ALIGN more than a context: the region of the padding sequences of a step
    rows = max(self.max_context, self.cache_rows) + REGION_ALIGN
    self.cache_k = _zeros(rows, self.idx.head_dim, dtype=dtypes.half, device=x.device)
    self.cache_pk = _zeros(rows // self.idx.kpool, self.idx.head_dim, dtype=dtypes.half, device=x.device)
    if getenv("QSA_DUMP"):
      self._dump = _zeros(getenv("PREFILL_CHUNK", 4608), self.idx.top_k // self.idx.kpool + 2, dtype=dtypes.int32, device=x.device)

  def _pooled(self, keys:Tensor, cos:Tensor, sin:Tensor) -> Tensor:
    """keys (..., pools, kpool, D): the pooled keys (..., pools, D), roped with cos and sin (pools, rope/2) of the pools' first positions"""
    idx = self.idx
    k = self.k_norm(keys.float().mean(-2))
    return _rope(k[..., :idx.rope_dim], cos, sin).cat(k[..., idx.rope_dim:], dim=-1)

  def _queries(self, x:Tensor, cos:Tensor, sin:Tensor) -> Tensor:
    """x (R, dim): the (R, H, D) queries, roped with cos and sin (R, 1, rope/2)"""
    idx = self.idx
    q = self.q_norm(self._proj(self.q_proj, x).reshape(x.shape[0], idx.n_heads, idx.head_dim).float())
    return _rope(q[..., :idx.rope_dim], cos, sin).cat(q[..., idx.rope_dim:], dim=-1)

  @staticmethod
  def _proj(lin:Linear, x:Tensor) -> Tensor:
    """the projections (float weights) of many rows on the tensor cores: half copies of the input and the weight, a float accumulator"""
    if lin.ggml_type is not None or not isinstance(x.shape[0], int) or x.shape[0] < 32 or x.ndim != 2: return lin(x)
    # (its weight stored transposed (_derive_weights): the gemm of the rows (a hundred or more of batched decoding), the generic matmul slower)
    if (wt:=getattr(lin, "weight_t", None)) is not None and (r:=f16t_gemm(x, wt)) is not None: return r
    return x.half().contiguous().matmul(lin.weight.half().T, dtype=dtypes.float).contiguous()

  def scores(self, x:Tensor, start_pos:int|UOp, T:int|UOp, freqs:Tensor) -> tuple[Tensor, int|UOp]:
    """
    x (1, Tq, dim) has the T tokens at start_pos (its rows past them padding): their keys and pools cached, the scores (Tq, pools) of every
    query on the pools up to the last token's, and that count of pools
    """
    idx, Tq = self.idx, x.shape[1]
    S = start_pos + T
    pk, pools = self.update(x, start_pos, T, freqs)
    qf = freqs.shrink(((start_pos, S), None)).pad_to((Tq, freqs.shape[-1])).reshape(Tq, 1, -1)
    q = self._queries(x[0], *qf.chunk(2, dim=-1))
    if isinstance(Tq, int) and Tq >= 32:
      # many queries: the heads' scores of all the pools (a static count: the pools past the last token's aren't seen) on the tensor cores
      sc = q.reshape(Tq * idx.n_heads, idx.head_dim).half().contiguous().matmul(self._static(pk).T, dtype=dtypes.float).contiguous()
      return sc.reshape(Tq, idx.n_heads, -1).relu().sum(1).shrink((None, (0, pools))) * idx.head_dim**-0.5, pools
    sc = (q.transpose(0, 1) @ pk.float().T).relu().sum(0) * idx.head_dim**-0.5  # (Tq, pools)
    return sc.contiguous(), pools

  def _static(self, pk:Tensor) -> Tensor:
    """the pooled keys of a sequence (pools, D) as a static (max_context//kpool, D) copy (zeros past them)"""
    return pk.pad_to((self.max_context // self.idx.kpool, self.idx.head_dim)).contiguous()

  def update(self, x:Tensor, start_pos:int|UOp, T:int|UOp, freqs:Tensor) -> tuple[Tensor, int|UOp]:
    """
    the keys of the T tokens of x at start_pos and their pools cached (the sequence's region, from _seq_base): the pooled keys (pools, D) of the
    sequence (after) and that count of pools up to the last token's
    """
    idx, kp, b = self.idx, self.idx.kpool, _seq_base[0]
    S, pb = start_pos + T, b // kp
    keys = self._proj(self.k_proj, x[0])[:T].half()
    cache = Tensor(self.cache_k.uop.after(self.cache_k.shrink(((b+start_pos, b+S), None)).uop.store(keys.uop)))
    # the pools of the tokens (the first one's tokens before them are cached), from the cached keys
    first, new = start_pos // kp, (start_pos % kp + T + kp - 1) // kp
    pf = freqs.reshape(-1, kp, freqs.shape[-1])[:, 0].shrink(((first, first + new), None))
    pk_new = self._pooled(cache.reshape(-1, kp, idx.head_dim).shrink(((pb + first, pb + first + new), None, None)), *pf.chunk(2, dim=-1))
    pk = Tensor(self.cache_pk.uop.after(self.cache_pk.shrink(((pb + first, pb + first + new), None)).uop.store(pk_new.half().uop)))
    _state_written(cache, pk)
    pools = (S + kp - 1) // kp
    return pk.shrink(((pb, pb + pools), None)), pools

  def mask(self, x:Tensor, start_pos:int|UOp, T:int|UOp, freqs:Tensor) -> Tensor:
    """the mask (1, 1, Tq, S) of the keys every query sees"""
    Tq, N, S = x.shape[1], self.max_context, start_pos + T
    scores, _ = self.scores(x, start_pos, T, freqs)
    q_pos, k_pos = (Tensor(start_pos) + Tensor.arange(Tq)).reshape(Tq, 1), Tensor.arange(N)[:S].reshape(1, -1)
    return _pool_select(scores, q_pos, k_pos, self.idx).reshape(1, 1, Tq, -1).cast(x.dtype)

  def prefill_mask(self, x:Tensor, start_pos:UOp, T:int|UOp, freqs:Tensor) -> Tensor:
    """One uint32 per pool and 32 queries; score disjoint query rows on the devices, then gather the compact masks."""
    idx, Tq = self.idx, cast(int, x.shape[1])
    pk, pools = self.update(x, start_pos, T, freqs)
    qf = freqs.shrink(((start_pos, start_pos+T), None)).pad_to((Tq, freqs.shape[-1])).reshape(Tq, 1, -1)
    xx = x[0]
    if isinstance(x.device, tuple) and Tq % (32*len(x.device)) == 0: xx, qf = shard_rows(xx, 0), shard_rows(qf, 0)
    q = self._queries(xx, *qf.chunk(2, dim=-1))
    sc = q.reshape(Tq*idx.n_heads, idx.head_dim).half().contiguous().matmul(self._static(pk).T, dtype=dtypes.float).contiguous()
    scores = (sc.reshape(Tq, idx.n_heads, -1).relu().sum(1) * idx.head_dim**-0.5).contiguous()
    return qsa_prefill_mask(scores, start_pos, pools, idx.top_k//idx.kpool, idx.kpool)

  def prefill_picks(self, x:Tensor, start_pos:UOp, T:int|UOp, freqs:Tensor) -> Tensor:
    """
    the lists (Tq, top+kpool+1) of the keys every query of a prefill chunk sees (as picks): the scores and the lists of disjoint rows on the
    devices, then gathered
    """
    idx, Tq = self.idx, cast(int, x.shape[1])
    pk, pools = self.update(x, start_pos, T, freqs)
    qf = freqs.shrink(((start_pos, start_pos+T), None)).pad_to((Tq, freqs.shape[-1])).reshape(Tq, 1, -1)
    xx, split = x[0], isinstance(x.device, tuple) and Tq % (32*len(x.device)) == 0
    if split: xx, qf = shard_rows(xx, 0), shard_rows(qf, 0)
    q = self._queries(xx, *qf.chunk(2, dim=-1))
    # (the scores of the pools so far only, the heads' relu summed in the kernel: index_scores)
    if (scores:=index_scores(q.reshape(Tq, idx.n_heads, idx.head_dim), self._static(pk), pools, idx.head_dim**-0.5)) is None:
      sc = q.reshape(Tq*idx.n_heads, idx.head_dim).half().contiguous().matmul(self._static(pk).T, dtype=dtypes.float).contiguous()
      scores = (sc.reshape(Tq, idx.n_heads, -1).relu().sum(1) * idx.head_dim**-0.5).contiguous()
    # (compact (the pools, not their keys): a quarter of the bytes gathered)
    lists = pool_index_rows(scores, start_pos, pools, idx.top_k // idx.kpool, idx.kpool, compact=bool(getenv("PICKS_COMPACT", 1)))
    out = all_gather(lists, 0) if split else lists
    # (QSA_DUMP: debugging, the lists of a chunk kept in a buffer of the layer (the last chunk's))
    if getenv("QSA_DUMP") and Tq == self._dump.shape[0]:
      out = Tensor(self._dump.uop.after(self._dump.uop.store(out.contiguous().uop)))
    return out

  def picks(self, x:Tensor, start_pos:UOp, T:int|UOp, freqs:Tensor) -> Tensor:
    """the lists (Tq, top+kpool+1) of the keys every query sees (its top pools, its tail, the count last): one kernel picks them"""
    scores, pools = self.scores(x, start_pos, T, freqs)
    scores = scores.pad_to((scores.shape[0], self.max_context // self.idx.kpool)).contiguous()
    return pool_index_rows(scores, start_pos, pools, self.idx.top_k // self.idx.kpool, self.idx.kpool)

class QSABlock(TransformerBlock):
  cache_k:Tensor
  """
  qwen sparse attention: gated GQA attention on the keys its indexer picks. the heads split over the devices with one KV head each (repeated at
  load), so a device's cache is the shared row [k|v] of its KV head (2*head_dim) and its queries [q|0]: the MLA flash kernels' latent attention
  (their output's second half is the attention's). the cache (KV, B, N, 2*head_dim) split on KV: a device's (1, B, N, D) is the latent cache
  (1, B, 1, N, D) of the kernels. batched decoding: the caches of the slots (KV, n, ctx, D), the slot of a group its index
  """
  pool_k:Tensor
  indexer:QSAIndexer|None  # type: ignore[assignment]  # (its own indexer, not TransformerBlock's)
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    self.indexer = QSAIndexer(config) if config.indexer else None

  @property
  def sparse(self) -> bool: return self.indexer is not None and self.config.max_context > self.indexer.idx.top_k
  def _flash(self, x:Tensor) -> bool:
    return amd_custom_kernels_supported(x.device) and isinstance(x.device, tuple) and self.config.n_kv_heads == len(x.device)

  def _qkv(self, x:Tensor, cos:Tensor, sin:Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """x (B, Tq, dim): q (B, H, Tq, hd), the output gate (B, Tq, H*hd), k and v (B, KV, Tq, hd); roped with cos and sin broadcast to them"""
    B, Tq, _ = x.shape
    H, hd, rd = self.config.n_heads, self.config.head_dim, self.config.rope_dim
    qg = self.attn_q(x).reshape(B, Tq, H, 2, hd)
    q, gate = qg[:, :, :, 0, :], qg[:, :, :, 1, :].reshape(B, Tq, H * hd)
    q = self.attn_q_norm(q).transpose(1, 2)
    q = _rope(q[..., :rd], cos, sin).cat(q[..., rd:], dim=-1)
    k, v = self._kv(x, cos, sin)
    return q, gate, k, v

  def _kv(self, x:Tensor, cos:Tensor, sin:Tensor) -> tuple[Tensor, Tensor]:
    B, Tq, _ = x.shape
    KV, hd, rd = self.config.n_kv_heads, self.config.head_dim, self.config.rope_dim
    k, v = self.attn_k(x).reshape(B, Tq, KV, hd), self.attn_v(x).reshape(B, Tq, KV, hd)
    k = self.attn_k_norm(k).transpose(1, 2)
    return _rope(k[..., :rd], cos, sin).cat(k[..., rd:], dim=-1), v.transpose(1, 2)

  def _out_gated(self, attn:Tensor, gate:Tensor) -> Tensor:
    """the attention (B, H, Tq, hd) gated, through the output projection"""
    B, H, Tq, hd = attn.shape
    return _out(self.attn_output, attn.transpose(1, 2).reshape(B, Tq, H * hd) * gate.sigmoid())

  def _attention(self, x:Tensor, start_pos:int|UOp, T:int|UOp|None=None) -> Tensor:
    if (sb:=_seq_batch[0]) is not None: return self._attention_seqs(x, sb)
    if (groups:=_batch_rows[0]) is not None: return self._attention_batch(x, groups)
    if T is not None: return self._attention_rows(x, start_pos, T)
    # a symbolic chunk computes padded to its max size (static shapes), it stores and attends with its real tokens
    if not isinstance(T:=x.shape[1], int): return self._attention_rows(x.pad_to(x.max_shape), start_pos, T)[:, :T]
    return self._attention_rows(x, start_pos, T)

  def _attention_rows(self, x:Tensor, start_pos:int|UOp, T:int|UOp) -> Tensor:
    # x has Tq rows, the first T are the tokens at start_pos
    B, Tq, _ = x.shape
    hd, N = self.config.head_dim, self.config.max_context
    freqs = self.freqs_cis.shrink(((start_pos, start_pos + T), None)).pad_to((Tq, self.freqs_cis.shape[-1]))
    cos, sin = (f.reshape(1, 1, Tq, -1) for f in freqs.chunk(2, dim=-1))
    q, gate, k, v = self._qkv(x, cos, sin)
    # (decoding with the indexer: the lists of the keys a query sees, the attention reads only them)
    picks = mask = tile_mask = row_picks = None
    upd:Tensor|None = None
    if self.indexer is not None and self._flash(x) and _qsa_dense[0]:
      # (a context the top_k covers: every key is seen, causal attention. the indexer's caches kept for a longer one)
      upd = self.indexer.update(x, start_pos, T, self.freqs_cis)[0]
    elif self.sparse and self.indexer is not None:
      if B == 1 and isinstance(Tq, int) and Tq % 32 == 0 and isinstance(start_pos, UOp) and self._flash(x) and self.indexer.idx.kpool == 4 and \
         2*hd == (self.cache_k.shape[-1]*32//9 if self._q8 else self.cache_k.shape[-1]) and hd == 256 and getenv("QSA_ROW_ATTN", 1):
        # (every query on the keys of its list (qsa_decode's kernel): its keys alone are read, not the tiles of every key of its block's)
        row_picks = self.indexer.prefill_picks(x, start_pos, T, self.freqs_cis)
      elif B == 1 and isinstance(Tq, int) and Tq % 32 == 0 and N % 128 == 0 and 64 <= hd <= 256 and hd % 32 == 0 and \
         isinstance(start_pos, UOp) and self._flash(x) and not _wmma_rdna4(x.device) and self.indexer.idx.kpool == 4:
        tile_mask = self.indexer.prefill_mask(x, start_pos, T, self.freqs_cis)
      elif isinstance(Tq, int) and Tq <= getenv("QSA_PICKS_ROWS", 8) and self._flash(x) and isinstance(start_pos, UOp) and getenv("QSA_PICKS", 1):
        picks = self.indexer.picks(x, start_pos, T, self.freqs_cis)
      else: mask = self.indexer.mask(x, start_pos, T, self.freqs_cis)
    b = _seq_base[0]  # (the sequence's region of the cache)
    if not self._flash(x):
      # (the generic path: the cache's halves are the keys and the values)
      rows = k.cat(v, dim=-1)[:, :, :T].transpose(0, 1).cast(self.cache_k.dtype)
      store = self.cache_k.shrink((None, None, (b+start_pos, b+start_pos+T), None)).uop.store(rows.uop)
      cache = Tensor(self.cache_k.uop.after(store)).shrink((None, None, (b, b+start_pos+T), None)).transpose(0, 1).float()
      causal = Tensor.full((1, 1, Tq, start_pos+T), float("-inf")).triu(start_pos+1)
      attn = q.scaled_dot_product_attention(cache[..., :hd], cache[..., hd:], attn_mask=causal if mask is None else causal + mask, enable_gqa=True)
      return self._out_gated(attn, gate)
    k_store = k.cat(v, dim=-1)[:, :, :T].transpose(0, 1)  # (KV, B, T, 2*hd)
    store = self.cache_k.shrink((None, None, (b+start_pos, b+start_pos+T), None)).uop.store(self._kv_rows(k_store).uop)
    cache = Tensor(self.cache_k.uop.after(store))
    _state_written(cache)
    if row_picks is not None:
      bases = (Tensor.zeros(Tq, dtype=dtypes.int32, device=x.device) + (Tensor(b).cast(dtypes.int32) if isinstance(b, UOp) else b)).contiguous()
      attn = qsa_decode_attention(q[0].transpose(0, 1), cache.unsqueeze(2), row_picks, bases, hd**-0.5,
                                  kpool=self.indexer.idx.kpool if self.indexer is not None and getenv("PICKS_COMPACT", 1) else 0)  # (Tq, H, 1, hd)
      return self._out_gated(attn.permute(2, 1, 0, 3), gate)
    if tile_mask is not None:
      return self._out_gated(qsa_sparse_prefill(q, cache, tile_mask, cast(UOp, start_pos), start_pos+T, unwrap(self.indexer).idx.kpool,
                                                base=b, N=N), gate)
    if upd is not None and isinstance(Tq, int) and Tq >= 32 and Tq % 32 == 0 and self.indexer is not None and getenv("QSA_FLASH_PREFILL", 1) and \
       (getenv("QSA_FLASH_MTP", 1) or not isinstance(self, QwenMTPBlock)):
      # (a chunk of a prompt the top_k covers: its keys are the first top_k positions of the cache. their keys and values apart, the tensor
      # cores' flash attention of the GQA heads)
      top = self.indexer.idx.top_k
      kvc = Tensor(cache.uop.after(upd.uop)).shrink((None, None, (b, b+top), None)).transpose(0, 1)  # (B, KV, top, 2*hd)
      if self._q8: kvc = kv_q8_unpack(kvc)
      kv = Tensor.stack(kvc[..., :hd], kvc[..., hd:]).contiguous()
      if (fa:=flash_attention_rows(q, kv, start_pos, start_pos+T)) is not None: return self._out_gated(fa, gate)
    latent, q, scale = cache.unsqueeze(2), q.cat(q.zeros_like(), dim=-1), hd**-0.5
    assert not self._q8, "an int8 cache: the sparse prefill, the dense prefill's flash kernel and the steps of sequences only"
    if upd is not None: latent = Tensor(latent.uop.after(upd.uop))
    if picks is not None:
      # (a few queries (the drafts of speculative decoding): every one on the keys of its list, a decode each)
      attn = Tensor.cat(*[flash_attention_decode(q[:, :, r:r+1].half(), latent, start_pos+T, N, shared=True, scale=scale,
                                                 index=picks[r].contiguous(), base=b) for r in range(Tq)], dim=2)
    elif Tq == 1:
      if mask is not None: mask = mask.reshape(B, -1).pad_to((B, N)).float().contiguous()
      attn = flash_attention_decode(q.half(), latent, start_pos+T, N, shared=True, scale=scale, mask=mask, base=b)
    else:
      if mask is not None: mask = mask.reshape(B, Tq, -1).pad_to((B, Tq, N)).float().contiguous()
      # (the region's keys: a copy of them from its first row)
      if not (isinstance(b, int) and b == 0):
        latent = latent.shrink((None, None, None, (b, b+start_pos+T), None))
        latent = latent.pad_to(latent.shape[:3] + (N,) + latent.shape[4:]).contiguous()
      attn = flash_attention_mla_prefill(q, latent, start_pos, start_pos+T, N, scale, mask)
    return self._out_gated(attn[..., hd:], gate)

  def _attention_seqs(self, x:Tensor, sb:SeqBatch) -> Tensor:
    """
    a step of G sequences sharing the caches (each its region of rows from its base): R rows of every one from its position, every query on
    the keys of its region its indexer picks (every key before it with a context of top_k or less)
    """
    assert self.indexer is not None and self._flash(x), "sequences sharing the caches: the indexer's and the flash kernels"
    if (sp:=_seq_prompt[0]) is not None:
      # (a mixed step: the prompt's rows as a prefill chunk's in its region, then the decoding rows' (their writes of the caches after the
      # prompt's reads, one order of the writes and the reads))
      _seq_prompt[0], base, _out_defer[0] = None, _seq_base[0], True
      try:
        dec = self._attention_seqs(x[:, :sp.off], sb)
        _seq_base[0] = sp.base
        with self.caches_set(*self._seq_written): pout = self._attention_rows(x[:, sp.off:], sp.start, sp.count)
      finally: _seq_prompt[0], _seq_base[0], _out_defer[0] = sp, base, False
      # (one output projection: one allreduce, the parts' independent ones would be the collectives' state written twice)
      return _out(self.attn_output, dec.cat(pout.cast(dec.dtype), dim=1))
    G, R, hd, N = sb.G, sb.R, self.config.head_dim, int(self.cache_k.shape[2])
    rows, idx, ic = G*R, self.indexer, self.indexer.idx
    kp = ic.kpool
    pos, phys = sb.row_positions, sb.row_phys
    freqs = rows_gather(self.freqs_cis, pos)  # (rows, rope)
    cos, sin = (f.reshape(1, 1, rows, -1) for f in freqs.chunk(2, dim=-1))
    q, gate, k, v = self._qkv(x, cos, sin)
    # the keys and values of the rows into their rows of the cache (a device's of its KV head)
    W = int(self.cache_k.shape[-1])
    kv = self._kv_rows(k.cat(v, dim=-1).transpose(0, 1).reshape(-1, 2*hd))
    cache = rows_scatter(self.cache_k.reshape(-1, W), kv, phys).reshape(self.cache_k.shape)
    # the indexer: the keys of the rows, then the pools they're in (NP from a sequence's first row's, the ones past its last row's are
    # incomplete: never picked, made again when they are) pooled from the cached keys
    ck = rows_scatter(idx.cache_k, idx._proj(idx.k_proj, x[0]).half(), phys)
    NP = (R - 1) // kp + 2
    pools = ((sb.positions // kp).reshape(G, 1) + Tensor.arange(NP, dtype=dtypes.int32).reshape(1, NP).to(x.device)).reshape(-1)  # (G*NP,)
    pbase = sb.bases.reshape(G, 1).expand(G, NP).reshape(-1) // kp
    krows = (((pbase + pools) * kp).reshape(-1, 1) + Tensor.arange(kp, dtype=dtypes.int32).reshape(1, kp).to(x.device)).reshape(-1)
    pkeys = rows_gather(ck, krows.contiguous()).reshape(G*NP, kp, ic.head_dim)
    pf = rows_gather(self.freqs_cis, (pools * kp).contiguous())
    pooled = idx._pooled(pkeys, *pf.chunk(2, dim=-1))
    pk = rows_scatter(idx.cache_pk, pooled.half(), (pbase + pools).contiguous())
    self._seq_written = (cache, ck, pk)
    # every query's scores of its sequence's pools, the lists of the keys it sees
    iq = idx._queries(x[0], *(f.reshape(rows, 1, -1) for f in freqs.chunk(2, dim=-1)))  # (rows, H, D)
    scores = seq_scores(iq, pk, sb.bases, sb.positions, R, kp, self.config.max_context // kp)
    picks = seq_index(scores, sb.positions, ic.top_k // kp, kp)
    # every row its own list of the keys of its region: [q|0] on the latent [k|v] rows (the second half of the output is the attention's)
    qr = q.permute(2, 1, 0, 3)  # (rows, H, 1, hd)
    latent = Tensor(cache.uop.after(pk.uop)).unsqueeze(2)
    if getenv("QSA_DECODE", 1):
      # (a lane a key: its scores of the heads without reductions over the lanes, the value half only of the rows read for the output)
      attn = qsa_decode_attention(qr[:, :, 0], latent, picks, sb.row_bases, hd**-0.5, parts=getenv("QSA_PARTS", 8), waves=getenv("QSA_WAVES", 4))
      return self._out_gated(attn.permute(2, 1, 0, 3), gate)
    attn = flash_attention_decode(qr.cat(qr.zeros_like(), dim=-1).half(), latent, N, N, shared=True, scale=hd**-0.5, index=picks,
                                  bases=sb.row_bases)
    return self._out_gated(attn[..., hd:].permute(2, 1, 0, 3), gate)

  def _attention_batch(self, x:Tensor, groups:list[tuple[UOp, int]]) -> Tensor:
    """batched decoding: a row a sequence at its position, its caches in its slot (slot g the group g's), one kernel for all of them"""
    assert all(t == 1 for _, t in groups) and _batch_prefill[0] is None, "batched qwen sparse attention: a token a sequence"
    G, hd, ctx = len(groups), self.config.head_dim, int(self.pool_k.shape[2])
    positions = _group_positions(groups, x.device)
    freqs = self.freqs_cis[:ctx][positions.cast(dtypes.int32)]  # (G, rope)
    cos, sin = (f.reshape(G, 1, 1, -1) for f in freqs.chunk(2, dim=-1))
    q, gate, k, v = self._qkv(x.reshape(G, 1, -1), cos, sin)
    # the rows into the slots (a device's rows of its KV head: (n*ctx, D) of every device)
    rows = k.cat(v, dim=-1).permute(1, 0, 2, 3).reshape(-1, 2*hd).cast(self.pool_k.dtype)
    pool = grouped_rows(self.pool_k.reshape(-1, 2*hd), rows, positions, ctx)
    k_pos, q_pos = Tensor.arange(ctx).reshape(1, ctx), positions.reshape(G, 1).cast(dtypes.int32)
    if self.indexer is not None:
      idx, ic = self.indexer, self.indexer.idx
      keys = grouped_rows(idx.pool_k.reshape(-1, ic.head_dim), idx.k_proj(x)[0].half(), positions, ctx).reshape(idx.pool_k.shape)
      # (the pool of every group's token pooled again from its keys, into its slot's pooled keys: the model's when a slot is loaded back)
      P, kp = ctx // ic.kpool, ic.kpool
      pidx = (positions.cast(dtypes.int32) // kp).reshape(G, 1)
      onehot = (pidx == Tensor.arange(P, dtype=dtypes.int32).reshape(1, P)).cast(dtypes.half)  # (G, P)
      grp = (onehot.unsqueeze(-1) * keys[:G].reshape(G, P, kp * ic.head_dim)).sum(1).reshape(G, 1, kp, ic.head_dim)
      pc = (onehot.float() @ self.freqs_cis[:ctx].reshape(P, kp, -1)[:, 0]).reshape(G, 1, -1)
      pooled = idx._pooled(grp, *pc.chunk(2, dim=-1)).reshape(G, ic.head_dim)
      pks = grouped_rows(idx.pool_pk.reshape(-1, ic.head_dim), pooled.half(), positions // kp, P)
      keys = Tensor(keys.uop.after(pks.uop))
    if self.indexer is not None and ctx > ic.top_k:
      pf = self.freqs_cis[:ctx].reshape(ctx // ic.kpool, ic.kpool, -1)[:, 0]
      pk = idx._pooled(keys[:G].reshape(G, -1, ic.kpool, ic.head_dim), *pf.chunk(2, dim=-1))  # (G, pools, D)
      iq = idx._queries(x[0], *(f.reshape(G, 1, -1) for f in freqs.chunk(2, dim=-1)))  # (G, H, D)
      scores = (iq @ pk.transpose(-1, -2)).relu().sum(1) * ic.head_dim**-0.5  # (G, pools)
      mask = _pool_select(scores, q_pos, k_pos, ic)
    else:
      mask = (k_pos <= q_pos).where(0.0, -math.inf)
      if self.indexer is not None: pool = Tensor(pool.uop.after(keys.uop))
    latent = pool.reshape(self.pool_k.shape)[:, :G].unsqueeze(2)  # (KV, G, 1, ctx, D): a device's (1, G, 1, ctx, D)
    attn = flash_attention_decode(q.cat(q.zeros_like(), dim=-1).half(), latent, ctx, ctx, shared=True, scale=hd**-0.5, mask=mask.float().contiguous())
    return self._out_gated(attn[..., hd:], gate.reshape(G, 1, -1)).reshape(1, G, -1)

  def _init_slots(self, n:int, ctx:int):
    """the caches of n sequences of batched decoding of at most ctx tokens (one slot each)"""
    if hasattr(self, "pool_k") and int(self.pool_k.shape[1]) >= n: return
    KV, D, devs = self.config.n_kv_heads, 2*self.config.head_dim, self.cache_k.device
    self.pool_k = _unallocated(KV, n, ctx, D, dtype=dtypes.half, device=devs, axis=0 if isinstance(devs, tuple) else None)
    if (idx:=self.indexer) is not None:
      idx.pool_k = _unallocated(n, ctx, idx.idx.head_dim, dtype=dtypes.half, device=devs)
      idx.pool_pk = _unallocated(n, ctx // idx.idx.kpool, idx.idx.head_dim, dtype=dtypes.half, device=devs)

  _seq_written:tuple[Tensor, Tensor, Tensor]  # (the caches after a step's writes: the attention's, the indexer's keys and pooled keys)
  @contextlib.contextmanager
  def caches_set(self, cache:Tensor, ck:Tensor, pk:Tensor):
    """the attention and indexer caches as these (a step's written ones: a mixed step's prompt's writes and reads after the step's)"""
    idx = unwrap(self.indexer)
    saved = (self.cache_k, idx.cache_k, idx.cache_pk)
    self.cache_k, idx.cache_k, idx.cache_pk = cache, ck, pk
    try: yield
    finally: self.cache_k, idx.cache_k, idx.cache_pk = saved

  @property
  def _q8(self) -> bool: return self.cache_k.dtype == dtypes.uint32
  def _kv_rows(self, kv:Tensor) -> Tensor:
    """rows (..., 2*head_dim) of keys and values as the cache's (in int8 groups: SEQ_KV_Q8)"""
    return kv_q8_pack(kv) if self._q8 else kv.cast(self.cache_k.dtype)

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_k"):
      # (the rows of every sequence's region, see QSAIndexer._init_state)
      KV, N, D = self.config.n_kv_heads, max(self.config.max_context, self.config.cache_rows) + REGION_ALIGN, 2*self.config.head_dim
      # (zeroed: the flash kernels read whole tiles past the valid region. SEQ_KV_Q8: in int8 groups of 32 and their scales (kv_q8_pack), the
      # bytes of a row 0.56x: more rows in memory)
      q8 = bool(getenv("SEQ_KV_Q8", 0)) and D % 32 == 0 and amd_custom_kernels_supported(x.device)
      self.cache_k = _zeros(KV, 1, N, D//4 + D//32 if q8 else D, dtype=dtypes.uint32 if q8 else dtypes.half, device=x.device,
                            axis=0 if isinstance(x.device, tuple) else None)
      self.freqs_cis = _freqs(self.config.rope_dim, self.config.max_context, self.config.rope_theta, x.device, self.config.yarn)
    if self.indexer is not None: self.indexer._init_state(x)

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
    self.k_norm = (nn.RMSNorm if idx.msa or idx.qsa else nn.LayerNorm)(idx.head_dim, eps=idx.norm_eps)
    if idx.msa or idx.qsa: self.q_norm = nn.RMSNorm(idx.head_dim, eps=idx.norm_eps)
    else: self.proj = Linear(config.dim, idx.n_heads, bias=False)
    if idx.kpool > 1 and not (idx.msa or idx.qsa):
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
      idx.head_dim % 16 == 0 and max_pools % 64 == 0 and bool(getenv("INDEXER_SPARSE", 1))
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
    if Tq <= 8 and idx.n_heads % 16 == 0 and getenv("INDEXER_DECODE_ROWS", 1):
      return indexer_decode_index_rows(iq, pk, weights, start_pos, pools, k, idx.kpool, pbase=pb, width=self.max_pools)
    # (IDX_SPLIT, default on: a prefill chunk's queries split on their rows over the devices (every device had all of them: the same lists six
    # times), each device the scores and the lists of its rows, the lists gathered. the same lists: exact)
    n = len(x.device) if isinstance(x.device, tuple) else 1
    if n > 1 and getenv("IDX_SPLIT", 1) and not self.shard and Tq % (16*n) == 0 and Tq // n <= getenv("INDEXER_ROWS", 1056) and \
       idx.n_heads % 16 == 0 and amd_custom_kernels_supported(x.device):
      if getenv("IDX_SPLIT_QB", 0):
        # (IDX_SPLIT_QB: the q projection of the device's rows too (q8_linear's split: the weight whole, the rows split). its kernel's tiles
        # differ from the whole chunk's (a 32-row part is the f16 kernel's): not the same bits)
        lin, qs = self.attn_q_b, shard_rows(q[0], 0)
        if lin.ggml_type is None and lin.use_custom_quant: lin.set_quantized(lin.weight)
        iq_s = q8_linear(lin, qs, split=True) if lin.ggml_type in QUANT_SIZES and lin.shard_axis is None else lin(qs)
        iq_s = iq_s.reshape(Tq, idx.n_heads, idx.head_dim)
      else: iq_s = shard_rows(iq.contiguous(), 0)  # (the q projection of every row as before, then split: the same queries)
      w_s = shard_rows(weights.contiguous(), 0)
      sc = indexer_scores(iq_s, pk, w_s, pools, idx.kpool, pbase=pb, width=self.max_pools)
      # (compact: the pools gathered, a quarter of the bytes, then expanded to the keys on every device (pool_expand))
      if not getenv("IDX_SPLIT_COMPACT", 1): return all_gather(pool_index_rows(sc, start_pos, pools, k, idx.kpool), 0)[:Tq]
      return pool_expand(all_gather(pool_index_rows(sc, start_pos, pools, k, idx.kpool, compact=True), 0), k, idx.kpool)[:Tq]
    # the scores on the tensor cores (rows of 16 queries), a kernel per query picks its pools
    rows = round_up(Tq, 16)
    iq, weights = iq.pad(((0, rows-Tq), None, None)), weights.pad(((0, rows-Tq), None))
    # blocks of rows one after the other: the scores of a block (rows x the max pools) are its memory, not the ones of the chunk
    block, picks = round_up(min(rows, getenv("INDEXER_ROWS", 1056)), 16), None
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
    k, iq = self.attn_k(x) if idx.qsa else self.k_norm(self.attn_k(x)), self.attn_q_b(q).reshape(B, T, idx.n_heads, idx.head_dim)
    if idx.msa or idx.qsa: iq = self.q_norm(iq)
    if idx.rope_dim:
      rot = unwrap(freqs)[start_pos:start_pos+T]
      if not idx.qsa: k = apply_rope(k[..., :idx.rope_dim].unsqueeze(1), rot, idx.rope_interleave).squeeze(1).cat(k[..., idx.rope_dim:], dim=-1)
      iq = apply_rope(iq[..., :idx.rope_dim].transpose(1, 2), rot, idx.rope_interleave).transpose(1, 2).cat(iq[..., idx.rope_dim:], dim=-1)
    if hasattr(self, "compressor_gate"):
      # cache an indexer key and a pooling gate per token, a pool's key is the gate softmax weighted sum of its keys
      kg = k.stack(self.compressor_gate(x), dim=2)  # (B, T, kpool, 2, D)
      kg = Tensor(self.cache_kg.uop.after(self.cache_kg[:, start_pos:S].uop.store(kg.half().uop))).float().reshape(B, -1, idx.kpool, 2, idx.head_dim)
      k = ((kg[..., 1, :] + self.compressor_ape["weight"]).softmax(2) * kg[..., 0, :]).sum(2)  # (B, S, D)
    else:
      k = Tensor(self.cache_kg.uop.after(self.cache_kg[:, start_pos:S].uop.store(k.half().uop))).float()  # (B, S, D)
    if idx.qsa:
      k, rot = self.k_norm(k.reshape(B, -1, idx.kpool, idx.head_dim).mean(2)), unwrap(freqs)[:self.max_context:idx.kpool]
      k = apply_rope(k[..., :idx.rope_dim].unsqueeze(1), rot, idx.rope_interleave).squeeze(1).cat(k[..., idx.rope_dim:], dim=-1)
    scores = iq @ k.transpose(-1, -2).unsqueeze(1)
    if idx.msa:   scores = scores.reshape(B, T, idx.n_heads, scores.shape[-1] // idx.kpool, idx.kpool).max(-1).transpose(1, 2)
    elif idx.qsa: scores = (scores.relu() * idx.head_dim**-0.5).sum(2).unsqueeze(1)
    else:         scores = (scores.relu() * (self.proj(x) * (idx.n_heads * idx.head_dim)**-0.5).unsqueeze(-1)).sum(2).unsqueeze(1)
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
  _regions:tuple[int, int, int]|None = None  # (SLOT_ALIAS: see Transformer._aliased)
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
    bmm = half and B == 1 and isinstance(Tq, int) and Tq <= 8 and amd_custom_kernels_supported(x.device) and bool(getenv("MLA_BMM", 1))
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
      if isinstance(Tq, int) and Tq > getenv("SEQ_GATHER_ROWS", 16) and getenv("SEQ_GATHER", 1):
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
    tc = Tq > getenv("MLA_BMM_ROWS", 32) and self.attn_k_b["weight"].dtype == dtypes.half
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
    def core(q:Tensor, k_store:Tensor, kg:Tensor, iq:Tensor, iw:Tensor, groups:list[tuple[UOp, int]], pf:BatchPrefill|None,
             pools:tuple[Tensor, ...]) -> tuple[Tensor, tuple[Tensor, ...]]:
      # one kernel for all the sequences (the same rows each, the caches of slot g the g-th of the pools): their positions in a tensor. returns
      # the attention and the pools (pooled keys, ring, cache) after it (another part's writes after its reads)
      assert len({T for _, T in groups}) == 1, "batched decoding: groups of the same rows"
      Tq, scale = int(q.shape[2]), 1.0/self.config.head_dim**0.5
      if (sb:=_seq_batch[0]) is not None:
        # sequences sharing the caches (GLM_SEQ): every one's region of the caches from its base, its ring in its state slot. the regions
        # as slots of REGION_ALIGN rows (a block on every device, see seq_store): slot base//REGION_ALIGN, a sequence's span of them
        assert pf is None and self.seq_sharded, "sequences sharing the caches: decoding steps, a sequence sharded cache"
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
    pools:tuple[Tensor, ...] = (idx.cache_pk, idx.pool_ring, self.cache_s) if _seq_batch[0] is not None else \
      (idx.pool_pk, idx.pool_ring, self.pool_s if self.seq_sharded else self.pool_k)
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
    if layer.bias is None and layer.ggml_type is None and R >= 32 and getenv("FEW_BATCH", 1):
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
    if (sb:=_seq_batch[0]) is not None and (sp:=_seq_prompt[0]) is not None:
      # (a mixed step: the decoding rows', then the prompt's from its slot's state of its start, the one of its end into the slot)
      _seq_prompt[0] = None
      try: dec = self._ple(x[:, :sp.off], emb[:, :sp.off], start_pos, T, dep)
      finally: _seq_prompt[0] = sp
      _seq_prompt[0], _seq_batch[0], groups = None, None, _batch_rows[0]
      _batch_rows[0] = None
      try: pp = self._ple(x[:, sp.off:], emb[:, sp.off:], sp.start, sp.count, dep)
      finally: _seq_prompt[0], _seq_batch[0], _batch_rows[0] = sp, sb, groups
      return dec.cat(pp, dim=1)
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
      assert all(t == 1 for _, t in groups) and _batch_prefill[0] is None, "batched n-gram embeddings: a token a sequence"
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

  def _prefill_batch(self, x:Tensor, pf:BatchPrefill, pre:tuple[list[Tensor], Tensor, Tensor, Tensor]|None=None,
                     deps:tuple[UOp, ...]=()) -> Tensor:
    """
    prefills of G prompts (P rows each, their first counts[g] the tokens) from position 0: the scan of a batch of G (rows past a prompt are no-ops),
    the states after the tokens into the slots (the one of position count in a slot's S)
    pre: the projections of the rows (q, k and v, the gate, alpha, beta) made before (a mixed step's): the outputs before ssm_out. deps: the
    kernels the states' pools are read and written after (a mixed step's decoding)
    """
    def pool(t:Tensor, rows:int) -> Tensor:
      v = t.reshape(t.shape[0], rows, -1)
      return Tensor(v.uop.after(*deps)) if deps else v
    pool_rec, pool_conv = pool(self.pool_rec, 1), [pool(p, self.ssm_conv_kernel-1) for p in self.pool_conv[:3]]
    P, (Bt, Tq, Dm) = pf.P, x.shape
    G, H, V, K, C = Tq // P, self.num_v_heads, self.head_v_dim, self.head_k_dim, self.ssm_conv_kernel
    xg, real = x.reshape(G, P, Dm).half(), pf.real.reshape(G, P, 1).to(x.device)
    def rows(t:Tensor) -> Tensor: return t.reshape(G, P, -1)
    out_gate = (rows(pre[1]) if pre is not None else self._gate(xg)).reshape(G, P, H, V)
    beta = (rows(pre[3]) if pre is not None else self.ssm_beta(xg)).sigmoid().reshape(G, P, H)
    g, a = ((rows(pre[2]) if pre is not None else self._alpha(xg)).float() + self._dt_ch).reshape(G, P, H, -1), self.ssm_a.reshape(H, -1)
    log_alpha = g.softplus() * a if (lb:=unwrap(self.config.ssm).gate_lower_bound) is None else (g * -a).sigmoid() * lb
    beta, log_alpha = real.where(beta, 0), real.unsqueeze(-1).where(log_alpha, 0)
    weights = [w["weight"] for w in (self.ssm_conv1d_q, self.ssm_conv1d_k, self.ssm_conv1d_v)]
    slots = pf.slots.to(x.device)
    starts = pf.starts.to(x.device) if pf.starts is not None else None
    # the windows of the prompts: the C-1 inputs before a prompt (its slot's conv state at its start, zeros at position 0) and its rows, the conv
    # state of a slot the last C-1 inputs before its prompt's end
    projs = [rows(p) for p in pre[0]] if pre is not None else [proj(xg) for proj in (self.attn_q, self.attn_k, self.attn_v)]
    if starts is None: windows = [p.float().pad(((0, 0), (C-1, 0), (0, 0))).contiguous() for p in projs]
    else:
      cont = (starts > 0).reshape(G, 1, 1)
      SC = int(self.slot_conv[0][0].shape[0])
      before = [cont.where(slot_gather(p, slots, starts, SC, dtypes.float), 0) for p in pool_conv]
      windows = [b.cat(p.float(), dim=1).contiguous() for b, p in zip(before, projs)]
    if self.head_k_dim == self.head_v_dim and self.num_v_heads == self.num_k_heads and K % 32 == 0 and amd_custom_kernels_supported(x.device):
      # the convolutions and norms of q, k and v in a kernel (as _attention's): the windows one after another, a prompt's output rows the ones
      # of its window (the rows across two windows are dropped)
      W = P + C - 1
      flat = [w.reshape(1, G*W, -1) for w in windows]
      outs = kda_conv([f[:, :C-1] for f in flat], [f[:, C-1:] for f in flat], weights, K, K**-0.5)
      q, k, v, kq = (o.pad(((0, 0), (0, 0), (0, C-1)) + ((0, 0),)*(o.ndim-3)).reshape(H, G, W, *o.shape[3:])[:, :, :P]
                     .transpose(0, 1).contiguous() for o in outs)
    else:
      convs = [functools.reduce(lambda acc, i: acc + w[:, i:i+P] * cw[:, i], range(1, C), w[:, 0:P] * cw[:, 0]).silu()
               for w, cw in zip(windows, weights)]
      q, k = ((z.reshape(G, P, self.num_k_heads, K).rmsnorm(eps=1e-6/K) / K**0.5).repeat(1, 1, H//self.num_k_heads, 1) for z in convs[:2])
      v = convs[2].reshape(G, P, H, V)
      q, k, v = (z.transpose(1, 2).float() for z in (q, k, v))
      q, kq = q * K**-0.5, None
    S = int(self.slot_rec[0].shape[0])
    if starts is None:
      state = Tensor.empty(G, H, V, K, dtype=dtypes.float32, device=x.device, axis=self.recurrent_state.uop.axis and 1).zeros_like().contiguous()
    else:
      # (a continuation: its slot's state of its start position, zeros at position 0)
      rec = slot_gather(pool_rec, slots, starts, S, dtypes.float).reshape(G, H, V, K)
      state = (starts > 0).reshape(G, 1, 1, 1).where(rec, 0).contiguous()
    core = gated_delta_prefill(q, k, v, beta.transpose(1, 2).float(), log_alpha.transpose(1, 2).exp(), state, kq=kq).transpose(1, 2)
    # the states into the slots (the scan updated state in place: after it)
    state = Tensor(state.uop.after(core.uop))
    counts = pf.counts.to(x.device)
    pf.effects.append(slot_scatter(pool_rec, state.reshape(G, 1, -1), slots, counts, S, starts=starts))
    # (a conv state: the window's rows count.. of the prompt, the C-1 inputs before its end)
    pf.effects.extend(slot_scatter(p, window, slots, counts, int(self.slot_conv[0][0].shape[0]), shift=1, starts=starts)
                      for p, window in zip(pool_conv, windows))
    z = (self.ssm_norm(core) * out_gate.sigmoid()).cast(x.dtype).contiguous()
    return z.reshape(Bt, Tq, -1) if pre is not None else _out(self.ssm_out, z.reshape(Bt, Tq, -1))

  def _attention_batch(self, x:Tensor, groups:list[tuple[UOp, int]]) -> Tensor:
    """batched decoding: a group of rows a sequence, its states in its slot (the projections of all of them in one kernel each)"""
    if (pf:=_batch_prefill[0]) is not None and not pf.off: return self._prefill_batch(x, pf)
    xin, x = x, x.half()
    projs, gate = [self.attn_q(x), self.attn_k(x), self.attn_v(x)], self._gate(x)
    alpha, beta = self._alpha(x, rows=True), self._few_rows(self.ssm_beta, x)
    weights = [w["weight"] for w in (self.ssm_conv1d_q, self.ssm_conv1d_k, self.ssm_conv1d_v)]
    if (sb:=_seq_batch[0]) is not None and (sp:=_seq_prompt[0]) is not None:
      # (a mixed step of sequences sharing the caches: the decoding rows' outputs (their kernel reads the first rows of the projections), the
      # prompt's from the rest, its states from its slot's of its start, into the slot after)
      S, SC, T = int(self.slot_rec[0].shape[0]), int(self.slot_conv[0][0].shape[0]), sb.R
      zd = kda_decode(self.pool_rec, self.pool_conv[:3], projs, weights, gate, alpha, beta, self._dt_ch, self.ssm_a,
                      unwrap(self.ssm_norm.weight), Tensor(groups[0][0]), unwrap(self.config.ssm).gate_lower_bound, self.ssm_norm.eps, row=0, rows=T,
                      positions=sb.positions, groups=sb.G, S=S, SC=SC, slots=sb.slots)
      # (the prompt's rows as a prefill chunk's: the model's states, its slot's copied to them before the step, back after)
      _seq_prompt[0], _batch_rows[0], _seq_batch[0], _out_defer[0] = None, None, None, True
      try: zp = self._attention(xin[:, sp.off:], sp.start, sp.count)
      finally: _seq_prompt[0], _batch_rows[0], _seq_batch[0], _out_defer[0] = sp, groups, sb, False
      # (one output projection: one allreduce)
      return self.ssm_out(zd.cat(zp.cast(zd.dtype), dim=1))
    if pf is not None:
      # (a mixed step: the decoding rows' outputs (their kernel reads the first rows of the projections), the prompts' from the rest)
      o, nd = pf.off, len(groups) - len(pf.count_vars)
      S, SC, T = int(self.slot_rec[0].shape[0]), int(self.slot_conv[0][0].shape[0]), groups[0][1]
      assert len({T for _, T in groups[:nd]}) == 1, "a mixed step: decoding groups of the same rows"
      calls:list[UOp] = []
      zd = kda_decode(self.pool_rec, self.pool_conv[:3], projs, weights, gate, alpha, beta, self._dt_ch, self.ssm_a,
                      unwrap(self.ssm_norm.weight), Tensor(groups[0][0]), unwrap(self.config.ssm).gate_lower_bound, self.ssm_norm.eps, row=0, rows=T,
                      positions=_group_positions(groups[:nd], self.pool_rec.device), groups=nd, S=S, SC=SC, calls_out=calls)
      def part(t:Tensor) -> Tensor: return t[:, o:]
      zp = self._prefill_batch(xin[:, o:], replace(pf, off=0), ([part(p) for p in projs], part(gate), part(alpha), part(beta)), tuple(calls))
      return self.ssm_out(zd.cat(zp.cast(zd.dtype), dim=1))
    # groups of the same size: a kernel for all of them (their states in the pools, their positions in a tensor)
    if len({T for _, T in groups}) == 1 and getenv("KDA_GROUPS", 1):
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
    if half is None: half = getenv("SLOT_STATE_HALF", -1) == 1 or (getenv("SLOT_STATE_HALF", -1) == -1 and n > 8)
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
       self.head_k_dim == self.head_v_dim == 128 and amd_custom_kernels_supported(x.device) and getenv("KDA_DECODE", 1):
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
      not symbolic and amd_custom_kernels_supported(x.device) and bool(getenv("KDA_CONV", 1))
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

class QwenMTPBlock(QSABlock):
  """
  qwen4exp's next token prediction: every stream of the model's residual (before its last mix) normed (hnorm) next to the normed embedding of the
  next token (enorm), projected (eh_proj) into the streams of a qwen sparse attention block with gated hyper-connections; its streams are the
  next prediction's input, their mix (hc_head) the output head's
  """
  def __init__(self, config:TransformerConfig):
    super().__init__(replace(config, ple=None))
    self.eh_proj = Linear(2 * config.dim, config.dim, bias=False)
    self.enorm = nn.RMSNorm(config.dim, config.norm_eps)
    self.hnorm = {"weight": Tensor.zeros(unwrap(config.hc).count * config.dim)}
    self.hc_head = GatedResidual(config, inject=False)
  def _project_input_split(self, emb:Tensor, h:Tensor) -> Tensor:
    """_project_input of rows split over the devices (the projection a device's rows: q8_linear split)"""
    count, dim = unwrap(self.config.hc).count, self.config.dim
    hs = h.reshape(*h.shape[:2], count, dim).float()
    hn = hs.rmsnorm(eps=self.config.norm_eps) * self.hnorm["weight"].reshape(count, dim)
    en = self.enorm(emb.float()).unsqueeze(2).expand(*hn.shape)
    inp = en.cat(hn, dim=-1)
    B, R = int(inp.shape[0]), int(inp.shape[1])
    return q8_linear(self.eh_proj, inp.reshape(B*R*count, 2*dim).half().contiguous(), split=True).reshape(B, R, count, dim)

  def _project_input(self, emb:Tensor, h:Tensor) -> Tensor:
    count, dim = unwrap(self.config.hc).count, self.config.dim
    hs = h.reshape(*h.shape[:2], count, dim).float()
    hn = hs.rmsnorm(eps=self.config.norm_eps) * self.hnorm["weight"].reshape(count, dim)
    en = self.enorm(emb.float()).unsqueeze(2).expand(*hn.shape)
    return self.eh_proj(en.cat(hn, dim=-1))

  def predict(self, emb:Tensor, h:Tensor, start_pos:int|UOp, T:int|UOp|None=None) -> Tensor:
    """emb (B, Tq, dim) the embeddings of the next tokens, h (B, Tq, count*dim) the streams of their positions: the streams (B, Tq, count, dim)"""
    return self(self._project_input(emb, h), start_pos, T)

  def prefill_cache(self, emb:Tensor, h:Tensor, start_pos:int|UOp, T:int|UOp) -> Tensor:
    # Prompt outputs are unused. KV/indexer caches depend only on the attention input, not Q, attention output, or the stateless FFN.
    xs = self._project_input(emb, h)
    self._init_state(xs)
    return self._cache_input(cast(GatedResidual, self.hc_attn).mixed(xs), start_pos, T)

  def _cache_input(self, x:Tensor, start_pos:int|UOp, T:int|UOp) -> Tensor:
    Tq = x.shape[1]
    freqs = self.freqs_cis.shrink(((start_pos, start_pos + T), None)).pad_to((Tq, self.freqs_cis.shape[-1]))
    k, v = self._kv(x, *(f.reshape(1, 1, Tq, -1) for f in freqs.chunk(2, dim=-1)))
    kv, b = self._kv_rows(k.cat(v, dim=-1)[:, :, :T].transpose(0, 1)), _seq_base[0]
    deps = [self.cache_k.shrink((None, None, (b+start_pos, b+start_pos+T), None)).uop.store(kv.uop)]
    if self.indexer is not None: deps.append(self.indexer.update(x, start_pos, T, self.freqs_cis)[0].uop)
    return Tensor(self.cache_k.uop.after(*deps))

  def predict_selected(self, emb:Tensor, h:Tensor, start_pos:int|UOp, row:int|UOp) -> Tensor:
    # Catch up all KV/indexer rows, but only the accepted position needs Q, attention output, and the stateless FFN.
    if not (_qsa_dense[0] and self._flash(emb)):
      return self.predict(emb, h, start_pos).shrink((None, (row, row+1), None, None))
    xs = self._project_input(emb, h)
    self._init_state(xs)
    hc = cast(GatedResidual, self.hc_attn)
    if (fused:=hc._fused(xs)) is not None: mixed, inj = fused
    else:
      mixed, xn = hc.mix(xs)
      inj = (s if (s:=_skinny(hc.inject, xn)) is not None else hc.inject(xn)) / hc.count
    cache = self._cache_input(mixed, start_pos, mixed.shape[1])
    pos, hd = start_pos + row, self.config.head_dim
    freqs = self.freqs_cis.shrink(((pos, pos+1), None)).reshape(1, 1, 1, -1)
    q, gate, _, _ = self._qkv(mixed[:, row:row+1], *freqs.chunk(2, dim=-1))
    attn = flash_attention_decode(q.cat(q.zeros_like(), dim=-1).half(), cache.unsqueeze(2), pos+1,
                                  self.config.max_context, shared=True, scale=hd**-0.5, base=_seq_base[0])
    out = self._out_gated(attn[..., hd:], gate)
    x = xs[:, row:row+1] + out.unsqueeze(2) * (inj[:, row:row+1].sigmoid()*2).unsqueeze(-1)
    return self.hc_ffn(x, self._feed_forward, self.ffn_norm)

  def head(self, streams:Tensor) -> Tensor:
    """the streams' mix: the output head's input"""
    return self.hc_head.mixed(streams)

class Transformer:
  def __init__(self, config:TransformerConfig):
    dense_config = replace(config, num_experts=0, num_experts_per_tok=0, shared_expert_dim=0, hidden_dim=config.dense_hidden_dim or config.hidden_dim)
    if config.ssm: config = replace(config, qk_norm=config.head_dim)
    self.config = config
    block_cls = MLATransformerBlock if config.kv_lora_rank > 0 else QSABlock if config.qsa else TransformerBlock
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
    if config.mtp: self.mtp = QwenMTPBlock(config) if config.qsa else MTPBlock(config)
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
    mv, self._batch_ple = _host_buffer(devs, max(self.BATCH_SIZES) * width)
    self._batch_ple_host = memoryview(mv).cast('B')
    self._slot_hist:dict[int, list[int]] = {}  # (batched decoding: the tokens before a slot's next one, for its n-grams)

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
      bool(getenv("PLE_DEVICE", 1))

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
    mb = getenv("MICRO_BATCHES", 1)
    if k <= 1 or self.config.hc is None or not amd_custom_kernels_supported(self.token_embd.weight.device) or \
       not (self.sequence_parallel(rows) and (mb > 1 or getenv("MB_PHASES", 1)) and self.sequence_parallel(rows // mb) and rows % mb == 0):
      return [(0, n)]
    return [(i*n//k, (i+1)*n//k) for i in range(k)]

  def _carry(self, rows:int) -> list[Tensor]:
    """the streams of every micro batch of a prefill chunk of rows between its parts (made before the jits are captured)"""
    carry = self.__dict__.setdefault("_pf_carry", {})
    if rows not in carry:
      hc, mb, devs = unwrap(self.config.hc), getenv("MICRO_BATCHES", 1), cast(tuple[str, ...], self.token_embd.weight.device)
      dt = dtypes.half if getenv("HC_STREAMS_HALF", 1) else dtypes.float32
      carry[rows] = [Tensor.zeros(1, rows // mb, hc.count, self.config.dim, dtype=dt).shard(devs, axis=1).contiguous().realize() for _ in range(mb)]
    return carry[rows]

  def _forward(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor|None, base:int|UOp, part:int|None=None) -> Tensor:
    # (a view of the token buffer from a variable base: copied, the lookup of the embedding's rows of a view at two variables is a reduce over
    # the vocabulary)
    x = emb = self.token_embd((tokens.contiguous() if isinstance(base, UOp) else tokens).to(self.token_embd.weight.device)).float()  # (B, T, D)
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
       all(isinstance(b, (MLATransformerBlock, GatedDeltaNetBlock, QSABlock)) for b in self.blk):
      x = x.pad_to(x.max_shape).contiguous()
      R = int(x.shape[1])
      # sequence parallel: every device keeps the streams of its rows, the blocks gather and scatter the sublayers' inputs and outputs
      # micro batches (on queues of their own, AM_MEC_PIPES: the copies of one run with the kernels of the other)
      mb = getenv("MICRO_BATCHES", 1)
      if (sp:=self.sequence_parallel(R)) and (mb > 1 or getenv("MB_PHASES", 1)) and self.sequence_parallel(R // mb) and R % mb == 0:
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
          if self.config.spec_drafts and getenv("MTP_NOGATHER", 1):
            # (the MTP layer's input of every device's rows from its own (a row the streams of the one before: the devices' last rows
            # gathered), the last row's streams picked: not every row's streams gathered)
            x = self._mtp_prefill_rows(emb, xs, cast(UOp, start_pos), T, n)
          else:
            x = Tensor.cat(*[_on_lane(i, all_gather, xi, 1) for i, xi in enumerate(xs)], dim=1).float()
            if self.config.spec_drafts: x = self._mtp_prefill(emb, x, cast(UOp, start_pos), T)
        else: x = Tensor.cat(*[_on_lane(i, all_gather, xi.mean(2), 1) for i, xi in enumerate(xs)], dim=1)
      else:
        # (gated hyper-connections split their rows in the phases only: without them every device has all the rows)
        if sp and self.gated: sp = False
        if sp: x = shard_rows(x, 1)
        for block in self.blk: x = block(x, start_pos, T, ple_emb=ple)
        if self.gated:
          if self.config.spec_drafts: x = self._mtp_prefill(emb, x, cast(UOp, start_pos), T)
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

  def _mtp_prefill_rows(self, emb:Tensor, xs:list[Tensor], start_pos:UOp, T:UOp, n:int) -> Tensor:
    """_mtp_prefill (gated) of the micro batches' streams xs (1, n, count, dim) split on the rows. returns (1, T, count, dim) whose last row is
    the last row's streams (the others zeros: only the last one is read)"""
    qm, hc, mb = cast(QwenMTPBlock, self.mtp), unwrap(self.config.hc), len(xs)
    W = hc.count * self.config.dim
    flat = [x.reshape(n, W) for x in xs]
    prev = rows_shifted(flat, self._spec_h[0])
    embp = emb.pad_to((1, n*mb, emb.shape[-1]))
    mixed = []
    for i in range(mb):
      e_l = shard_rows(embp[:, i*n:(i+1)*n].contiguous(), 1)
      xs_l = qm._project_input_split(e_l, prev[i].reshape(1, n, W))
      qm._init_state(xs_l)
      mixed.append(_on_lane(i, all_gather, cast(GatedResidual, qm.hc_attn).pre_split(xs_l)[0], 1, dtypes.half))
    m = qm._cache_input(Tensor.cat(*mixed, dim=1).float(), start_pos, T)
    last = rows_pick(flat, T - 1, T, SEL_LANE)
    effects = [m[:, :, :1, :1].contiguous(), self._spec_h[0:1].assign(last)]
    if self._effects is None: Tensor.realize(*effects)
    else: self._effects += effects
    return last.reshape(1, 1, hc.count, self.config.dim)

  def _mtp_prefill(self, emb:Tensor, x:Tensor, start_pos:UOp, T:UOp, sp:bool=False) -> Tensor:
    """
    the MTP layer on a prompt chunk (for its caches): emb (1, T, D) the embeddings of its tokens, x (1, rows, D) the model's (padded) output.
    the hidden state of the position before the chunk is row 0 of _spec_h, the one of its last token is written there. returns x
    """
    if self.gated:
      # (the streams of every position: the MTP layer's input of the next one, the ones before the chunk's first in _spec_h's row 0)
      hs = x.flatten(2).float()
      emb, carry = emb.pad_to((1, x.shape[1], emb.shape[-1])), self._spec_h[0:1].unsqueeze(0).contiguous()
      qm, R = cast(QwenMTPBlock, self.mtp), int(x.shape[1])
      if getenv("MTP_SPLIT", 1) and self.sequence_parallel(R) and amd_custom_kernels_supported(x.device):
        # (the MTP layer's input projection and mix of every device's rows only (of the gathered streams: its part kept), its mixed input
        # gathered for the caches: not the projections of every row on every device)
        e_l, h_l = shard_rows(emb.contiguous(), 1), shard_rows(carry.cat(hs[:, :-1], dim=1).contiguous(), 1)
        xs_l = qm._project_input_split(e_l, h_l)
        qm._init_state(xs_l)
        mixed = cast(GatedResidual, qm.hc_attn).pre_split(xs_l)[0]
        m = qm._cache_input(all_gather(mixed, 1, dtypes.half).float(), start_pos, T)
      else: m = qm.prefill_cache(emb, carry.cat(hs[:, :-1], dim=1), start_pos, T)
      effects = [m[:, :, :1, :1].contiguous(), self._spec_h[0:1].assign(hs[0].shrink(((T-1, T), None)))]
      if self._effects is None: Tensor.realize(*effects)
      else: self._effects += effects
      return x
    hn = self.output_norm(x).float()
    emb, carry = emb.pad_to((1, x.shape[1], emb.shape[-1])), self._spec_h[0:1].unsqueeze(0).contiguous()
    # the MTP layer reads the carry before it's written (the schedule orders an assign after the reads). realized with the call's output
    inp = self.mtp.enorm(emb).cat(cast(MTPBlock, self.mtp).hnorm(carry.cat(hn[:, :-1], dim=1)), dim=-1)
    if sp and getenv("MTP_SP", 1):
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
    if self.gated: return self._forward_spec_gated(last, j, start_pos, par, temperature, mtp_pos, *chain)
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

  def _forward_spec_gated(self, last:Tensor, j:UOp, start_pos:UOp, par:UOp, temperature:Tensor|None, mtp_pos:UOp, *chain:UOp) -> Tensor:
    """forward_spec of gated hyper-connections (qwen4exp): the MTP layer reads and writes the streams, the drafts' n-gram rows on the devices"""
    k, devs, y = self.config.spec_drafts, self.token_embd.weight.device, self._spec_in
    mtp, hc = cast(QwenMTPBlock, self.mtp), unwrap(self.config.hc)
    m = mtp.predict_selected(self.token_embd(y.to(devs)).float(), last, mtp_pos, j)
    drafts:list[Tensor] = []
    for i in range(k):
      drafts.append(self._sample(mtp.head(m)[:, 0, :], None, y.device))
      if i < k-1: m = mtp.predict(self.token_embd(drafts[-1].to(devs)).float(), m.flatten(2), chain[i])
    # verify: the model on the token at start_pos and the drafts, their streams kept
    vt = y.shrink((None, (j, j+1))).cat(*drafts, dim=1).contiguous()
    x = self.token_embd(vt.to(devs)).float()
    x = x.unsqueeze(2).expand(*x.shape[:2], hc.count, x.shape[-1]).contiguous()
    ple = None
    if (pc:=self.config.ple) is not None:
      # (the n-gram rows of the positions: the tokens before start_pos from the token buffer, then the verified ones)
      n = len(pc.multipliers)
      win = unwrap(self._token_buf).pad(((0, 0), (n-1, 0)), value=pc.eos)[0].shrink(((start_pos, start_pos + n-1),)).cat(vt[0].to(devs))
      ple = self._ple_emb(self._ple_rows_dev(win).unsqueeze(0))
    for block in self.blk: x = block(x, start_pos, ple_emb=ple)
    hs = self._spec_h.shrink(((par*(k+1), (par+1)*(k+1)), None)).assign(x.flatten(2)[0].float())
    tok = self._sample(self._hc_out(hs.reshape(1, k+1, hc.count, -1))[0], temperature, y.device)
    q8_quantize_cache_clear()  # don't keep the graph alive
    out = tok.reshape(1, k+1).cat(*drafts, dim=1).cast(dtypes.int32)
    return self._spec_out.assign(Tensor(out.uop.mselect(0)) if isinstance(out.device, tuple) else out)

  def _spec_call(self, mtp_pos:int, j:int, par:int, temperature:Tensor|None):
    """a step of speculative decoding (forward_spec, its jit)"""
    k, pos = self.config.spec_drafts, mtp_pos + j
    # (the step's positions mtp_pos..pos+k covered by the top_k of the qwen sparse attention: its jit of all the keys, see _qsa_dense)
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
    _qsa_dense[0] = dense
    try: jit(v_mtp.bind(mtp_pos), v_j.bind(j), v_pos.bind(pos), v_par.bind(par), temperature, *(v.bind(pos+1+i) for i,v in enumerate(v_chain)))
    finally: _qsa_dense[0] = False

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

  # ******** batched decoding: the next token of many sequences in one step (a row each), every one with its own states (a slot) ********
  # a sequence is prefilled as usual (the states of the model), then its states are copied into a free slot (slot_store); steps take the
  # sequences of the slots 0..n-1 (the rows of a step), the projections, experts and the head read their weights once for all of them

  BATCH_SIZES = tuple(int(x) for x in str(getenv("BATCH_SIZES", "1,2,4,16,32,64,80")).split(","))
  # (MIX: the landing slots of mixed steps past the slots decoded, see batch_mixed_step)
  MIX = getenv("MIX", 0)
  def batch_slots(self) -> int:
    """the slots: the ones decoded (decode_slots) and the landing ones of mixed steps"""
    # (the memory of the slots: a shorter context of the model's own frees some)
    if not (self.has_recurrent_block and (self.config.kv_lora_rank > 0 or self.config.qsa)) or self.config.cache_rows: return 0
    return min(getenv("BATCH_SLOTS", 80 if self.max_context <= 65536 else 32), max(self.BATCH_SIZES)) + self.mix_landing()
  def mix_landing(self) -> int:
    """the landing slots of mixed steps (MIX of them, with speculative decoding): a step's prompts prefilled into them, then moved"""
    return self.MIX if self.config.spec_drafts and self.has_recurrent_block and self.config.kv_lora_rank > 0 else 0
  def decode_slots(self) -> int:
    """the slots of the sequences decoded (the first ones)"""
    return n - self.mix_landing() if (n:=self.batch_slots()) else 0
  def batch_ctx(self) -> int:
    ctx = min(getenv("BATCH_CTX", 2048), self.max_context)
    # (aliased: whole blocks of the split sequence, the slots' regions are the model's cache's positions)
    if self._aliased(): ctx = ctx // (q:=SEQ_BLK*len(to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device)))) * q
    return ctx
  def _aliased(self) -> bool:
    """
    (SLOT_ALIAS, a sequence sharded cache) the slots' caches and the model's are one buffer: the model's first batch_ctx positions, then the
    slots' (the model's past them too: a long sequence alone overwrites the slots, see main_overlaps)
    """
    return seq_sharded(self.token_embd.weight.device) and bool(self.batch_slots()) and bool(getenv("SLOT_ALIAS", 1))
  def main_overlaps(self, length:int) -> list[int]:
    """the slots the model's cache of `length` positions overwrites (aliased: slot i's are the model's past (SLOT_SPAN+i)*batch_ctx)"""
    if not self._aliased(): return []
    return list(range(min(self.batch_slots(), max(0, -(-length // self.batch_ctx()) - SLOT_SPAN))))
  def _slot_writes(self):
    """(aliased) the slots written: the model's conversation past its own positions (and its checkpoint) is gone, see main_own"""
    if not self._aliased(): return
    if len(getattr(self, "_cached_tokens", [])) > (S:=self.main_own()): self._cached_tokens = []
    if len(getattr(self, "_ckpt_tokens", [])) > S: self._ckpt_tokens = []
  def batch_sizes(self) -> tuple[int, ...]:
    """
    the rows of the steps of batched decoding (a jit each): the sizes below the slots and the slots (BATCH_SLOTS of another size). none with
    BATCH_DECODE=0 (no batched decoding, the slots only keep conversations: its steps aren't captured)
    """
    if not getenv("BATCH_DECODE", 1): return ()
    return tuple(sorted({s for s in self.BATCH_SIZES if s < n} | {n})) if (n:=self.decode_slots()) else ()

  def _init_slots(self, alloc:bool=True):
    """the slots of batched decoding (alloc: their memory too, else when the jits using them are linked: after the peak of loading)"""
    if not (n:=self.batch_slots()): return
    # (made once: the jits captured use them)
    if not hasattr(self, "_batch_in"):
      self._init_states()
      for b in self._slot_blocks(): b._init_slots(n, self.batch_ctx())
      devs = cast(str|tuple[str, ...], self.token_embd.weight.device)
      if (k:=self.config.spec_drafts):
        # speculative decoding: the hidden states of every slot's last verified tokens, the tokens of a step's groups (k+1 each) and its
        # outputs (the k+1 sampled tokens and the k drafts of every group)
        self._spec_init()
        self.pool_spec_h, self.slot_spec_h = _slot_pool(n, self._spec_h)
        self._bspec_host_in, self._bspec_in = _host_ints(devs, max(self.BATCH_SIZES)*(k+1))
        self._bspec_host_out, self._bspec_out = _host_ints(devs[0] if isinstance(devs, tuple) else devs, max(self.BATCH_SIZES)*(2*k+1))
      # in host memory: the tokens and temperatures of a step's rows (the host writes them), the sampled tokens (one device writes them)
      self._batch_host_in, self._batch_in = _host_ints(devs, max(self.BATCH_SIZES))
      # (a row's temperature and top_p)
      tmv, self._batch_temp = _host_buffer(devs, 8*max(max(self.BATCH_SIZES) + self.mix_landing(), self.BPF_G), dtypes.float32)
      self._batch_host_temp = tmv.cast('f')
      self._batch_host_out, self._batch_out = _host_ints(devs[0] if isinstance(devs, tuple) else devs, max(self.BATCH_SIZES))
      # prefills in a step: the tokens of the prompts, the token after each
      self._bpf_host_in, self._bpf_in = _host_ints(devs, max(self.BPF_G * self.BPF_P, self.mix_landing() * max(self.MIX_PS)))
      self._bpf_host_out, self._bpf_out = _host_ints(devs[0] if isinstance(devs, tuple) else devs, max(self.BPF_G, self.mix_landing()))
    if alloc: self._alloc_slots()

  def _alloc_slots(self):
    if getattr(self, "_slots_ready", False) or not self.batch_slots(): return
    self._init_slots(alloc=False)  # (made, not allocated)
    # (the buffers freed by loading and capturing are cached for reuse: their memory is the slots')
    for d in to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device)): Device[d].allocator.free_cache()
    todo:list[Tensor] = [self.pool_spec_h] if hasattr(self, "pool_spec_h") else []
    for blk in self._slot_blocks():
      if isinstance(blk, GatedDeltaNetBlock): todo += [blk.pool_rec, *blk.pool_conv]
      if isinstance(blk, MLATransformerBlock):
        todo += [blk.pool_s if blk.seq_sharded else blk.pool_k] + ([blk.indexer.pool_pk, blk.indexer.pool_ring] if blk.indexer is not None else [])
      if isinstance(blk, QSABlock): todo += [blk.pool_k] + ([blk.indexer.pool_k, blk.indexer.pool_pk] if blk.indexer is not None else [])
    for t in todo:
      for u in t.uop.toposort():
        if u.op is Ops.BUFFER:
          for x in (b.bufs if isinstance(b:=u.buffer, MultiBuffer) else [b]): x.ensure_allocated()
    # (zeros: a step's rows of nothing read states and caches no sequence wrote (WIDE: a slot spanned at its last positions), finite ones)
    for t in todo: t.assign(t.zeros_like()).realize()
    self._slots_ready = True

  def _slot_blocks(self) -> list[FFNBlock]: return self.blk + ([self.mtp] if self.config.spec_drafts else [])
  def _slot_copy(self, slot:UOp, rp:UOp, wp:UOp, wpc:UOp, hrow:UOp, span:int=1):
    """
    the states of the model at the end of its tokens (the recurrent ones in rp) into a slot (its recurrent ones in wp, its conv ones in wpc): in
    the pools.
    speculative decoding: the hidden state of the last token (row hrow of _spec_h) is the slot's row 0
    """
    copies:list[Tensor] = []
    def rows(t:Tensor, start:UOp|int, n:int=1) -> Tensor: return t.shrink(((start, start+n),) + (None,)*(t.ndim-1))
    if (k:=self.config.spec_drafts): copies.append(rows(self.pool_spec_h, slot*2*(k+1)).assign(rows(self._spec_h, hrow)))
    for b in self._slot_blocks():
      if isinstance(b, GatedDeltaNetBlock):
        S, SC = int(b.slot_rec[0].shape[0]), int(b.slot_conv[0][0].shape[0])
        copies.append(rows(b.pool_rec, slot*S + wp).assign(rows(b.recurrent_state, rp).cast(b.pool_rec.dtype)))
        copies += [rows(d, slot*SC + wpc).assign(rows(c, rp).cast(d.dtype)) for d, c in zip(b.pool_conv, b.conv_states)]
      if isinstance(b, MLATransformerBlock):
        # (span: the slots a sequence spans, see SLOT_SPAN)
        if b.seq_sharded:
          n = b.slot_rows()
          copies.append(rows(b.slot_rows_view(b.pool_s), slot*n, span*n).assign(rows(b.slot_rows_view(b.cache_s), 0, span*n)))
        else: copies.append(rows(b.pool_k, slot).assign(b.cache_k[:, :, :int(b.slot_k[0].shape[2])]))
        if (idx:=b.indexer) is not None:
          P, R = idx.pk_slot(), int(idx.slot_ring[0].shape[0])
          copies += [rows(idx.pk_rows(idx.pool_pk), slot*P, span*P).assign(rows(idx.pk_rows(idx.cache_pk), 0, span*P)),
                     rows(idx.pool_ring, slot*R, R).assign(idx.ring)]
      if isinstance(b, QSABlock):
        ctx = int(b.pool_k.shape[2])
        copies.append(b.pool_k.shrink((None, (slot, slot+1), None, None)).assign(b.cache_k[:, :, :ctx]))
        if (qi:=b.indexer) is not None:
          copies += [rows(qi.pool_k, slot).assign(qi.cache_k[:ctx].unsqueeze(0)),
                     rows(qi.pool_pk, slot).assign(qi.cache_pk[:ctx // qi.idx.kpool].unsqueeze(0))]
    Tensor.realize(*copies)

  def _slot_load_copy(self, slot:UOp, rp:UOp, wp:UOp, wpc:UOp, hrow:UOp, span:int=1):
    """
    the reverse of _slot_copy: a slot's states back into the model's (a sequence stored there continues alone). speculative decoding: the hidden
    state of the last token is the slot's row hrow (0 if stored by _slot_copy, a batched step's last verified row else), the model's row 0
    """
    copies:list[Tensor] = []
    def rows(t:Tensor, start:UOp|int, n:int=1) -> Tensor: return t.shrink(((start, start+n),) + (None,)*(t.ndim-1))
    if (k:=self.config.spec_drafts): copies.append(rows(self._spec_h, 0).assign(rows(self.pool_spec_h, slot*2*(k+1) + hrow)))
    for b in self._slot_blocks():
      if isinstance(b, GatedDeltaNetBlock):
        S, SC = int(b.slot_rec[0].shape[0]), int(b.slot_conv[0][0].shape[0])
        copies.append(rows(b.recurrent_state, rp).assign(rows(b.pool_rec, slot*S + wp).cast(b.recurrent_state.dtype)))
        copies += [rows(c, rp).assign(rows(d, slot*SC + wpc).cast(c.dtype)) for d, c in zip(b.pool_conv, b.conv_states)]
      if isinstance(b, MLATransformerBlock):
        if b.seq_sharded:
          n = b.slot_rows()
          copies.append(rows(b.slot_rows_view(b.cache_s), 0, span*n).assign(rows(b.slot_rows_view(b.pool_s), slot*n, span*n)))
        else: copies.append(b.cache_k[:, :, :int(b.slot_k[0].shape[2])].assign(rows(b.pool_k, slot)))
        if (idx:=b.indexer) is not None:
          P, R = idx.pk_slot(), int(idx.slot_ring[0].shape[0])
          copies += [rows(idx.pk_rows(idx.cache_pk), 0, span*P).assign(rows(idx.pk_rows(idx.pool_pk), slot*P, span*P)),
                     idx.ring.assign(rows(idx.pool_ring, slot*R, R))]
      if isinstance(b, QSABlock):
        ctx = int(b.pool_k.shape[2])
        copies.append(b.cache_k[:, :, :ctx].assign(b.pool_k.shrink((None, (slot, slot+1), None, None))))
        if (qi:=b.indexer) is not None:
          copies += [qi.cache_k[:ctx].assign(rows(qi.pool_k, slot)[0]), qi.cache_pk[:ctx // qi.idx.kpool].assign(rows(qi.pool_pk, slot)[0])]
    Tensor.realize(*copies)

  def _slot_pools(self) -> list[tuple[Tensor, int]]:
    """every pool of the slots' states and the rows of a slot in it"""
    pools:list[tuple[Tensor, int]] = [(self.pool_spec_h, int(self.slot_spec_h[0].shape[0]))] if hasattr(self, "pool_spec_h") else []
    for b in self._slot_blocks():
      if isinstance(b, GatedDeltaNetBlock):
        pools += [(b.pool_rec, int(b.slot_rec[0].shape[0]))] + [(p, int(b.slot_conv[0][0].shape[0])) for p in b.pool_conv]
      if isinstance(b, MLATransformerBlock):
        pools.append((b.slot_rows_view(b.pool_s), b.slot_rows()) if b.seq_sharded else (b.pool_k, 1))
        if (idx:=b.indexer) is not None: pools += [(idx.pk_rows(idx.pool_pk), idx.pk_slot()), (idx.pool_ring, int(idx.slot_ring[0].shape[0]))]
      if isinstance(b, QSABlock):
        pools.append((b.pool_k.permute(1, 0, 2, 3), 1))
        if b.indexer is not None: pools += [(b.indexer.pool_k, 1), (b.indexer.pool_pk, 1)]
    return pools
  def _slot_move_copy(self, src:UOp, dst:UOp):
    def rows(t:Tensor, start:UOp, n:int) -> Tensor: return t.shrink(((start, start+n),) + (None,)*(t.ndim-1))
    Tensor.realize(*[rows(p, dst*n, n).assign(rows(p, src*n, n).contiguous()) for p, n in self._slot_pools()])
  def _slot_jits(self):
    # (their temporaries in the memory of the model's intermediates: a slot's copy is as big as its caches)
    if not hasattr(self, "_slot_jit"): self._slot_jit = TinyJit(self._slot_copy, capture_first=True, scratch=self._scratch)
    if not hasattr(self, "_slot_move_jit"): self._slot_move_jit = TinyJit(self._slot_move_copy, capture_first=True, scratch=self._scratch)
    if not hasattr(self, "_slot_load_jit"): self._slot_load_jit = TinyJit(self._slot_load_copy, capture_first=True, scratch=self._scratch)
    # (the copies of a sequence spanning more slots, see slot_span)
    if not hasattr(self, "_slot_span_jits"):
      self._slot_span_jits = {n: tuple(TinyJit(functools.partial(f, span=n), capture_first=True, scratch=self._scratch)
                                       for f in (self._slot_copy, self._slot_load_copy)) for n in range(2, self.max_span()+1)}
  def main_own(self) -> int:
    """(aliased) the positions of the model's cache that aren't a slot's (SLOT_SPAN slots' worth)"""
    return SLOT_SPAN*self.batch_ctx()
  def max_span(self) -> int:
    """(WIDE, aliased slots) the slots a sequence of batched decoding spans at most: its positions past its slot's are the next ones'"""
    return SLOT_SPAN if self._aliased() else 1
  def slot_span(self, length:int) -> int:
    """the slots a sequence of `length` positions spans"""
    return max(1, -(-length // self.batch_ctx())) if self._aliased() else 1
  def _slot_move_vars(self, src:int, dst:int) -> tuple[UOp, ...]:
    v = functools.partial(UOp.variable, min_val=0, max_val=self.batch_slots()-1)
    return v("slot_src").bind(src), v("slot_dst").bind(dst)
  def _slot_store_vars(self, i:int, length:int, hrow:int) -> tuple[UOp, ...]:
    S, SC = slot_states(self.config.spec_drafts)
    rp = UOp.variable("slot_rp", 0, self.config.rewind).bind(length % (self.config.rewind + 1))
    return (UOp.variable("slot", 0, self.batch_slots()-1).bind(i), rp, UOp.variable("slot_wp", 0, S-1).bind(length % S),
            UOp.variable("slot_wpc", 0, SC-1).bind(length % SC), UOp.variable("slot_hrow", 0, 2*self.config.spec_drafts+1).bind(hrow))

  def prefix_to_slot(self, tokens:list[int], slot:int) -> int:
    """
    the states of the longest prefix of tokens the model has (its conversation, or its checkpoint restored) into slot: the position the rest
    starts at (a batch_prefill continues it there, or slot_first when tokens is exactly its checkpoint).
    """
    start, restore = self._reuse(tokens) if self.has_recurrent_block else (self.get_start_pos(tokens), False)
    if restore: self._checkpoint(tokens, save=False)
    self.slot_store(slot, start)
    self._slot_hist_set(slot, tokens[:start])
    return start

  def slot_first(self, slot:int, temp:float|tuple[float, ...]) -> int:
    """the token after the prompt whose checkpoint is in slot: sampled from its hidden state, without another recurrent prefill"""
    self._init_slots()
    pair = _temp_pair(temp)
    temperature = None if pair[0] == 0 else Tensor([pair], device=self.token_embd.weight.device)
    hrow = self.pool_spec_h.shrink(((slot*2*(self.config.spec_drafts+1), slot*2*(self.config.spec_drafts+1)+1), None))
    return int(self._sample(hrow, temperature, self.token_embd.weight.device).item())

  def _ckpt_from_slot_copy(self, slot:UOp, wp:UOp, wpc:UOp):
    """the states of a slot at a position (its recurrent ones in wp, its conv ones in wpc) as the model's checkpoint (_ckpt, in its order)"""
    k = self.config.spec_drafts
    def rows(t:Tensor, start:UOp|int, n:int=1) -> Tensor: return t.shrink(((start, start+n),) + (None,)*(t.ndim-1))
    srcs:list[Tensor] = [rows(self.pool_spec_h, slot*2*(k+1), 2*(k+1))] if hasattr(self, "_spec_h") else []
    for b in self.blk + ([self.mtp] if hasattr(self, "mtp") else []):
      if isinstance(b, GatedDeltaNetBlock) and hasattr(b, "recurrent_state"):
        S, SC = int(b.slot_rec[0].shape[0]), int(b.slot_conv[0][0].shape[0])
        srcs += [rows(b.pool_rec, slot*S + wp)] + [rows(p, slot*SC + wpc) for p in b.pool_conv]
      if (idx:=getattr(b, "indexer", None)) is not None and hasattr(idx, "ring"):
        R = int(idx.slot_ring[0].shape[0])
        srcs.append(rows(idx.pool_ring, slot*R, R))
    assert len(srcs) == len(self._ckpt), f"{len(srcs)} states of a slot, a checkpoint of {len(self._ckpt)}"
    Tensor.realize(*[d.assign(s.cast(d.dtype)) for d, s in zip(self._ckpt, srcs)])
  def ckpt_from_slot(self, i:int, length:int):
    """
    the states of slot i at position `length` (a batch_prefill's prompt end) as the model's checkpoint (the caller sets _ckpt_tokens): the
    checkpoint of a prompt prefilled in a slot, see the server's ckpt_keep
    """
    if not hasattr(self, "_ckpt"): return
    if not hasattr(self, "_ckpt_slot_jit"): self._ckpt_slot_jit = TinyJit(self._ckpt_from_slot_copy, capture_first=True)
    v = self._slot_store_vars(i, length, 0)
    self._ckpt_slot_jit(v[0], v[2], v[3])

  def slot_move(self, src:int, dst:int):
    """the states of slot src into slot dst (the sequences of a batch in the slots of its first rows)"""
    self._slot_jits()
    self._slot_writes()
    self._slot_move_jit(*self._slot_move_vars(src, dst))
    if self.config.ple is not None: self._slot_hist[dst] = list(self._slot_hist.get(src, []))

  def slot_store(self, i:int, length:int, hrow:int=0):
    """the states of the sequence just prefilled (or decoded) by the model, of `length` tokens, into slot i (hrow: the row of _spec_h)"""
    self._init_slots()
    self._slot_jits()
    for d in to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device)): Device[d].synchronize()  # (run ahead steps of the model)
    span = self.slot_span(length)
    assert span <= self.max_span(), f"a sequence of {length} tokens into a slot of {self.batch_ctx()}"
    self._slot_writes()
    (self._slot_jit if span == 1 else self._slot_span_jits[span][0])(*self._slot_store_vars(i, length, hrow))
    self._slot_hist_set(i, self._cached_tokens[:length])

  def slot_load(self, i:int, length:int, hrow:int=0):
    """
    the states of slot i (a sequence of `length` tokens, stored by slot_store or decoded batched there) back into the model's: it continues alone
    from them (the cache of a conversation kept in a slot). hrow: the slot's row of the last token's hidden state. the caller sets _cached_tokens
    """
    self._init_slots()
    self._slot_jits()
    for d in to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device)): Device[d].synchronize()
    span = self.slot_span(length)
    (self._slot_load_jit if span == 1 else self._slot_span_jits[span][1])(*self._slot_store_vars(i, length, hrow))
    self._ckpt_tokens, self._spec_row = [], 0

  # ** a conversation's states in host memory (HOST_STASH in the server: the ones longer than a slot): its bytes copied, no kernel
  HOST_BUCKET = getenv("HOST_BUCKET", 16384)  # the positions host_save copies: a conversation's rounded up to a multiple (the same copies)
  def _host_parts(self, length:int, ckpt:bool=True) -> list[tuple[Buffer, int, int]]:
    """
    the bytes of the model's states at the end of its `length` tokens: (a device's buffer, offset, size) each. the positions of the caches up
    to `length` rounded up to HOST_BUCKET, the row of the recurrent ones at its end (the one a call after it reads), every row of the other
    states (the hidden states of speculative decoding), the checkpoint's if it has one (the states at the end of the prompt of the last reply:
    the caches up to it are the conversation's).
    the same for every conversation of a bucket: host_save's copies are a program made once
    """
    rows = self._host_len(length)
    parts:list[tuple[Buffer, int, int]] = []
    def add(t:Tensor, axis:int=0, n:int|None=None): parts.extend(self._state_parts(t, axis, n))
    if self.config.spec_drafts: add(self._spec_h)
    for b in self._slot_blocks():
      if isinstance(b, MLATransformerBlock):
        if b.seq_sharded:
          n = len(cast(tuple[str, ...], b.cache_s.device))
          add(b.cache_s, 0, (rows + SEQ_BLK*n - 1) // (SEQ_BLK*n) * SEQ_BLK)
        else: add(b.cache_k, 2, rows)
        if (idx:=b.indexer) is not None:
          pools = (rows + idx.idx.kpool - 1) // idx.idx.kpool
          # (split: every device's pools of the positions, see seq_rows)
          if idx.shard: pools = (rows + SEQ_BLK*idx.shard - 1) // (SEQ_BLK*idx.shard) * (SEQ_BLK // idx.idx.kpool)
          if idx.sparse: add(idx.cache_pk, 0, pools), add(idx.ring)
          else: add(idx.cache_pk, 1, pools), add(idx.cache_kg, 1, pools * idx.idx.kpool)
      if isinstance(b, QSABlock):
        add(b.cache_k, 2, rows)
        if b.indexer is not None: add(b.indexer.cache_k, 0, rows), add(b.indexer.cache_pk, 0, (rows + b.indexer.idx.kpool - 1) // b.indexer.idx.kpool)
    # (the recurrent states' row of the end only: through the staging rows, see _host_rows)
    for t in self._host_rows(): add(t)
    if ckpt:
      for t in getattr(self, "_ckpt", []): add(t)
    return parts

  @staticmethod
  def _state_parts(t:Tensor, axis:int=0, n:int|None=None) -> list[tuple[Buffer, int, int]]:
    """the first n rows of a state's axis (all of it): a prefix of every device's buffer, past the axes of one before it"""
    bufs = [u.buffer for u in t.uop.toposort() if u.op is Ops.BUFFER]
    assert len(bufs) == 1, f"a state of {len(bufs)} buffers"
    per = bufs[0].bufs if isinstance(bufs[0], MultiBuffer) else [bufs[0]]
    shape = list(t.shape)
    if isinstance(t.device, tuple) and (sa:=t.uop.axis) is not None: shape[sa] //= len(t.device)
    assert prod(shape[:axis]) == 1 and per[0].nbytes == prod(shape) * t.dtype.itemsize, f"{t.shape} isn't rows of a buffer on axis {axis}"
    row = int(prod(shape[axis+1:])) * t.dtype.itemsize
    return [(b, 0, (int(shape[axis]) if n is None else n)*row) for b in per]

  def _host_rows(self, length:int|None=None, save:bool=True) -> list[Tensor]:
    """
    the staging rows of the slotted recurrent states (the row of a position p is p%S): host_save copies the row of a conversation's end into
    them (length, save) then them to host memory, host_load back from them (save False). their copies are the same program for every row
    """
    states:list[tuple[Tensor, bool]] = [(t, sl) for t, sl in self._recurrent_states() if sl]
    if not states: return []
    slot = UOp.variable("hrow_slot", 0, self.config.rewind).bind((length or 0) % (self.config.rewind + 1))
    if not hasattr(self, "_hrows"):
      self._hrows = [v.clone().realize() for v in self._ckpt_views(states, slot)]
      self._hrow_jits = {sv: TinyJit(functools.partial(self._hrow_copy, sv), capture_first=True) for sv in (True, False)}
    if length is not None: self._hrow_jits[save](slot)
    return self._hrows
  def _hrow_copy(self, save:bool, slot:UOp):
    views = self._ckpt_views([x for x in self._recurrent_states() if x[1]], slot)
    Tensor.realize(*[dst.assign(src) for dst, src in (zip(self._hrows, views) if save else zip(views, self._hrows))])

  HOST_CHUNK = getenv("HOST_CHUNK", 1 << 30)  # the bytes of a buffer of host memory (host_alloc): a conversation takes as many as it needs
  def _host_layout(self, parts:list[tuple[Buffer, int, int]]) -> list[tuple[int, int]]:
    """every part's (chunk, offset): one after the other, the next chunk when it doesn't fit (aligned to 256 bytes)"""
    out, chunk, off = [], 0, 0
    for _, _, n in parts:
      assert n <= self.HOST_CHUNK, f"a part of {n} bytes, chunks of {self.HOST_CHUNK}"
      if off + n > self.HOST_CHUNK: chunk, off = chunk + 1, 0
      out.append((chunk, off))
      off = round_up(off + n, 256)
    return out
  def host_chunks(self, length:int, ckpt:bool=True) -> int:
    """the chunks of host memory (host_alloc) host_save needs for a conversation of `length` tokens"""
    return self._host_layout(self._host_parts(length, ckpt))[-1][0] + 1
  def host_ckpt(self, tokens:list[int]) -> int:
    """the tokens of the model's checkpoint (the prompt of its last reply) if it is one of the conversation `tokens` (host_save keeps it)"""
    ck = self._ckpt_tokens if hasattr(self, "_ckpt") else []
    return len(ck) if 0 < len(ck) < len(tokens) and tokens[:len(ck)] == ck else 0
  def host_alloc(self, nbytes:int|None=None) -> Buffer:
    """a chunk of host memory for host_save (CPU memory: the devices copy to it, mapped on them here once: used again, not made again)"""
    buf = Buffer("CPU", nbytes or self.HOST_CHUNK, options=BufferSpec(nolru=True)).allocate()
    for d in to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device)): buf.get_buf(d)
    return buf

  def _host_copy(self, save:bool, chunks:list[Buffer], length:int, ckpt:bool=True):
    if save: self._host_rows(length, True)  # (the row of its end to the staging rows, before the copies)
    # (the bucket of its positions, its checkpoint or not)
    self._host_run((save, len(chunks), self._host_len(length), ckpt), save, chunks, lambda: self._host_parts(length, ckpt))
    if not save: self._host_rows(length, False)  # (the staging rows to the row of its end)

  def _host_run(self, key:tuple, save:bool, chunks:list[Buffer], parts_of:Callable[[], list[tuple[Buffer, int, int]]],
                layout_of:Callable[[list[tuple[Buffer, int, int]]], list[tuple[int, int]]]|None=None):
    # the copies between the model's states (parts_of) and chunks (of host memory: inputs of the program, the same program for every chunks
    # of these sizes): a program made once (building and compiling ~2000 copies takes ~0.5 s), the devices copy at once (CPU memory is the
    # devices' to copy to, none staged)
    if not hasattr(self, "_host_programs"): self._host_programs:dict[tuple, UOp] = {}
    key = (*key, tuple(c.nbytes for c in chunks))
    if (prg:=self._host_programs.pop(key, None)) is None:
      parts = parts_of()
      layout = (layout_of or self._host_layout)(parts)
      assert len(chunks) > layout[-1][0], f"{len(chunks)} chunks for {layout[-1][0]+1}"
      ins = [UOp.param(i, dtypes.uint8, c.nbytes, c.device) for i, c in enumerate(chunks)]
      pairs = [(ins[c][off:off+n], UOp.from_buffer(b.view(n, o).ensure_allocated(), dtypes.uint8)) for (b, o, n), (c, off) in zip(parts, layout)]
      calls = [h.store_call(b) if save else b.store_call(h) for h, b in pairs]
      prg = link_linear(compile_linear(UOp(Ops.LINEAR, src=tuple(calls))))
      while len(self._host_programs) >= getenv("HOST_PROGRAMS", 512): self._host_programs.pop(next(iter(self._host_programs)))
    self._host_programs[key] = prg  # (the most recently used last)
    devs = to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device))
    for d in devs: Device[d].synchronize()  # (run ahead steps of the model)
    run_linear(prg, input_uops=[UOp.from_buffer(c, dtypes.uint8) for c in chunks], jit=True, update_stats=False)
    for d in devs + ("CPU",): Device[d].synchronize()

  # ** a request's checkpoint while it's decoded in a slot (the model's own is the next prompt's by then): copied to host memory after its
  # prefill, back into the model's before its conversation's host_save. a client that doesn't send the model's reasoning back has the
  # conversation of its next turn differ right after this prompt: it goes on from there, not from the start
  def _ckpt_parts(self) -> list[tuple[Buffer, int, int]]: return [p for t in getattr(self, "_ckpt", []) for p in self._state_parts(t)]
  @staticmethod
  def _one_layout(parts:list[tuple[Buffer, int, int]]) -> list[tuple[int, int]]:
    """every part's (0, offset): one after the other in one buffer (aligned to 256 bytes), whatever its size"""
    offs = list(itertools.accumulate((round_up(n, 256) for _, _, n in parts), initial=0))
    return [(0, o) for o in offs[:-1]]
  def ckpt_alloc(self) -> Buffer|None:
    """host memory for a copy of the model's checkpoint (ckpt_copy): its size, in one buffer (not chunks). None without one"""
    if not (parts:=self._ckpt_parts()): return None
    return self.host_alloc(self._one_layout(parts)[-1][1] + parts[-1][2])
  def ckpt_copy(self, save:bool, buf:Buffer):
    """the model's checkpoint into buf (save), or back from it (the caller sets _ckpt_tokens)"""
    self._host_run(("ckpt", save), save, [buf], self._ckpt_parts, self._one_layout)

  def _host_len(self, length:int) -> int:
    """the positions of a conversation host_save copies: rounded up to HOST_BUCKET, (aliased) not past the model's own if it fits (the slots')"""
    n = min(round_up(max(length, 1), self.HOST_BUCKET), self.max_context)
    return min(n, self.main_own()) if self._aliased() and length <= self.main_own() else n

  def host_save(self, length:int, chunks:list[Buffer], ckpt:bool=True):
    """
    the states of the model's conversation of `length` tokens copied to host memory: chunks (host_alloc'd, as many as host_chunks: the same
    ones every time its copies are the same program). its checkpoint's too (ckpt), its hidden states of speculative decoding (every row, see
    host_load)
    """
    self._host_copy(True, chunks, length, ckpt)

  def host_load(self, chunks:list[Buffer], length:int, hrow:int, ckpt:bool=True):
    """
    the states of a conversation of `length` tokens saved by host_save back into the model's (it continues from them, or from its checkpoint:
    ckpt, saved with it). hrow: the row of _spec_h of its last token's hidden state when saved (_spec_row). sets no _cached_tokens, _ckpt_tokens
    """
    self._host_copy(False, chunks, length, ckpt)
    self._ckpt_tokens, self._spec_row = [], hrow

  def forward_batch(self, *positions:UOp) -> Tensor:
    """a step of batched decoding: row i the sequence of slot i at positions[i], its token in _batch_in, its temperature in _batch_temp"""
    n, devs = len(positions), self.token_embd.weight.device
    _batch_rows[0], Q8_ROWS_I8[0] = [(p, 1) for p in positions], bool(getenv("BATCH_I8", 1))
    try:
      x = self.token_embd(self._batch_in[:, :n].to(devs)).float()  # (1, n, D)
      if (hc:=self.config.hc) is not None: x = x.unsqueeze(2).expand(*x.shape[:2], hc.count, x.shape[-1]).contiguous()
      ple = None
      if (pc:=self.config.ple) is not None: ple = self._ple_emb(self._batch_ple.reshape(-1, len(pc.vocab_sizes) * self._ple_rb)[:n].unsqueeze(0))
      for block in self.blk: x = block(x, 0, ple_emb=ple)
      if hc is not None: x = self._hc_out(x)
      tok = self._sample(self.output_norm(x)[0].float(), self._batch_temp[:2*n].reshape(n, 2), self._batch_in.device)
    finally: _batch_rows[0], Q8_ROWS_I8[0] = None, False
    q8_quantize_cache_clear()  # don't keep the graph alive
    out = tok.reshape(1, n).cast(dtypes.int32)
    return self._batch_out[:, :n].assign(Tensor(out.uop.mselect(0)) if isinstance(out.device, tuple) else out)

  def _batch_jit(self, m:int):
    if not hasattr(self, "_batch_jits"): self._batch_jits:dict[int, Any] = {}
    if m not in self._batch_jits: self._batch_jits[m] = TinyJit(self.forward_batch, capture_first=True, scratch=self._scratch)
    return self._batch_jits[m]
  def _batch_positions(self, positions:list[int]) -> list[UOp]:
    return [UOp.variable(f"batch_pos{i}", 0, SLOT_SPAN*self.batch_ctx()-1).bind(pos) for i, pos in enumerate(positions)]

  def batch_step(self, rows:list[tuple[int, int, float|tuple[float, ...]]]) -> list[int]:
    """
    a step of batched decoding: rows (token, position, temperature (or temperature and top_p)) of the slots 0..len(rows)-1, their sampled tokens
    """
    self._init_slots()
    self._slot_writes()
    n = len(rows)
    m = next(s for s in self.batch_sizes() if s >= n)
    assert m <= self.batch_slots(), f"{n} rows, {self.batch_slots()} slots"
    # (the rows past n are the free slots' at position 0: what they write is overwritten when a sequence takes the slot)
    rows = rows + [(0, 0, 0.0)] * (m - n)
    for i, (tok, _, temp) in enumerate(rows): self._batch_host_in[i], self._batch_host_temp[2*i:2*i+2] = tok, _temp_pair(temp)
    if self.config.ple is not None: self._batch_ple_write([(tok, pos) for tok, pos, _ in rows])
    self._batch_host_out[:m] = array.array('i', [-1] * m)
    self._batch_jit(m)(*self._batch_positions([pos for _, pos, _ in rows]))
    st = time.perf_counter()
    while min(self._batch_host_out[:m]) < 0:
      # (a step that doesn't come: the devices' errors, or none)
      if time.perf_counter() - st > 10:
        for d in to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device)): Device[d].synchronize()
        if min(self._batch_host_out[:m]) < 0: raise RuntimeError(f"batched step of {m} rows: no tokens after the devices finished")
    return list(self._batch_host_out[:n])

  def _batch_ple_write(self, rows:list[tuple[int, int]]):
    """the n-gram rows of a step's rows (token, position) of the slots 0..: from the tokens before them in their slots (_slot_hist)"""
    ple = unwrap(self.config.ple)
    self._ple_init()
    n, width, ctxs = len(ple.multipliers), len(ple.vocab_sizes) * self._ple_rb, []
    for i, (tok, pos) in enumerate(rows):
      hist = self._slot_hist.get(i, [])
      ctxs.append([tok] + [ple.eos if pos < j or len(hist) < j or ple.eos in hist[len(hist)-j:] else hist[len(hist)-j] for j in range(1, n)])
      self._slot_hist[i] = (hist + [tok])[-(n-1):] if n > 1 else []
    import numpy as np
    np.frombuffer(self._batch_ple_host, dtype=np.uint8)[:len(rows)*width] = self._ple_rows(self._ple_ids(ctxs)).reshape(-1)
  def _slot_hist_set(self, i:int, tokens:list[int]):
    """(n-gram embeddings) the tokens of the sequence in slot i, before its next one"""
    if (ple:=self.config.ple) is None: return
    self._ple_init()
    self._slot_hist[i] = tokens[max(0, len(tokens) - (len(ple.multipliers) - 1)):] if len(ple.multipliers) > 1 else []

  def forward_batch_spec(self, *seqs:UOp, prompts:tuple[int, int]|None=None) -> Tensor:
    """
    a step of batched speculative decoding (forward_spec of every sequence, its slot g the group g): seqs (mtp_pos, j, par) of every group,
    its k+1 tokens in _bspec_in, its temperature in _batch_temp. the MTP layer catches up on the k+1 positions from mtp_pos and drafts k tokens
    after start_pos = mtp_pos+j, the model verifies them. returns (1, n*(2k+1)): the k+1 sampled tokens and the k drafts of every group
    prompts (G, P): a mixed step, the prefills of G prompts of P rows (as forward_batch_prefill's continuations) with the verification, the last
    3G of seqs their token counts, slots and starts, their tokens in _bpf_in, their temperatures in _batch_temp after the groups', the token
    after every one in _bpf_out (the weights read once for both)
    """
    G, P = prompts or (0, 0)
    pvs, seqs = (seqs[len(seqs)-3*G:], seqs[:len(seqs)-3*G]) if G else ((), seqs)
    k, n, devs = self.config.spec_drafts, len(seqs) // 3, self.token_embd.weight.device
    groups = [(seqs[3*g], seqs[3*g+1], seqs[3*g+2]) for g in range(n)]
    y = self._bspec_in[:, :n*(k+1)]
    Q8_ROWS_I8[0] = bool(getenv("BATCH_I8", 1))
    try:
      # the MTP layer on the last step's tokens with their hidden states (the rows of the other parity), the row j of every group
      last = Tensor.cat(*[self.slot_spec_h[g].shrink((((1-par)*(k+1), (2-par)*(k+1)), None)) for g, (_, _, par) in enumerate(groups)])
      _batch_rows[0] = [(mp, k+1) for mp, _, _ in groups]
      m = self.mtp.predict(self.token_embd(y.to(devs)).float(), last.unsqueeze(0), 0)
      m = Tensor.cat(*[m.shrink((None, (g*(k+1)+j, g*(k+1)+j+1), None)) for g, (_, j, _) in enumerate(groups)], dim=1)
      # (SPEC_SHARED: the drafts sampled with the noise of the position they're verified at, see _hash_uniform. else the most likely)
      shared = bool(getenv("SPEC_SHARED", 1))
      seed = (Tensor.rand_like(self._batch_temp[:1]) * (1 << 24)).cast(dtypes.uint32) if shared else None
      temps = self._batch_temp[:2*n].reshape(n, 1, 2).expand(n, k+1, 2).reshape(n*(k+1), 2)
      drafts:list[Tensor] = []
      for i in range(k):
        # (draft i of group g: verified at row i of the group)
        # (at their temperature, not their top_p: no nucleus of theirs, about as many accepted)
        drafts.append(self._sample(m[0], self._batch_temp[:2*n].reshape(n, 2)[:, :1] if shared else None, y.device,
                                   (seed, Tensor.arange(n) * (k+1) + i) if seed is not None else None).reshape(1, n))
        if i < k-1:
          _batch_rows[0] = [(mp + j + 1 + i, 1) for mp, j, _ in groups]
          m = self.mtp.predict(self.token_embd(drafts[-1].to(devs)).float(), m, 0)
      # verify: the model on every group's token at start_pos and its drafts, their hidden states kept (the rows of parity par)
      toks = Tensor.cat(*[y.shrink((None, (g*(k+1)+j, g*(k+1)+j+1))).cat(*[d[:, g:g+1] for d in drafts], dim=1)
                          for g, (_, j, _) in enumerate(groups)], dim=1)
      x = self.token_embd(toks.contiguous().to(devs)).float()
      _batch_rows[0] = [(mp + j, k+1) for mp, j, _ in groups] + [(st, P) for st in pvs[2*G:]]
      if G:
        # (the prompts' rows after the groups')
        pf = BatchPrefill(P, self._ints(pvs[:G]), self._ints(pvs[G:2*G]), pvs[:G], pvs[G:2*G], starts=self._ints(pvs[2*G:]), off=n*(k+1))
        x = x.cat(emb:=self.token_embd(self._bpf_in[:, :G*P].to(devs)).float(), dim=1)
        _batch_prefill[0] = pf
      if (hc:=self.config.hc) is not None: x = x.unsqueeze(2).expand(*x.shape[:2], hc.count, x.shape[-1]).contiguous()
      for block in self.blk: x = block(x, 0)
      if hc is not None: x = x.mean(2)
      hn_all = self.output_norm(x)[0].float()
      hn = Tensor.cat(*[self.slot_spec_h[g].shrink(((par*(k+1), (par+1)*(k+1)), None)).assign(hn_all[g*(k+1):(g+1)*(k+1)])
                        for g, (_, _, par) in enumerate(groups)])
      tok = self._sample(hn, temps, y.device, (seed, Tensor.arange(n*(k+1))) if seed is not None else None)
      if G:
        # the prompts: their MTP layer's caches, the token after every one (their hidden states written after the groups')
        _batch_rows[0], _batch_prefill[0] = [(st, P) for st in pvs[2*G:]], (pf:=replace(pf, off=0))
        ptok = self._prompt_ends(hn_all[n*(k+1):], emb, pf, self._batch_temp[2*n:2*(n+G)].reshape(G, 2))
        pf.effects.append(self._bpf_out[:, :G].assign(Tensor(o.uop.mselect(0)) if isinstance((o:=ptok.reshape(1, G).cast(dtypes.int32)).device, tuple)
                                                      else o))
    finally: _batch_rows[0], _batch_prefill[0], Q8_ROWS_I8[0] = None, None, False
    q8_quantize_cache_clear()  # don't keep the graph alive
    out = tok.reshape(n, k+1).cat(*[d.reshape(n, 1) for d in drafts], dim=1).reshape(1, n*(2*k+1)).cast(dtypes.int32)
    ret = self._bspec_out[:, :n*(2*k+1)].assign(Tensor(out.uop.mselect(0)) if isinstance(out.device, tuple) else out)
    if G: Tensor.realize(ret, *pf.effects)
    return ret

  def _ints(self, xs:tuple[UOp, ...]) -> Tensor:
    """variables as an int32 tensor of the devices (a stack of variables has no device: onto zeros of the devices)"""
    devs = self.token_embd.weight.device
    return (Tensor.zeros(len(xs), dtype=dtypes.int32, device=devs) + Tensor.stack(*[Tensor(x) for x in xs]).cast(dtypes.int32)).contiguous()
  def _prompt_ends(self, hn:Tensor, emb:Tensor, pf:BatchPrefill, temps:Tensor) -> Tensor:
    """
    prompts prefilled into slots (pf's, _batch_rows theirs): the MTP layer on their rows (its caches, a token's input the hidden state hn of the
    one before, zero before the first), the hidden state of every one's last token its slot's (speculative decoding's row 0). returns (G,) the
    token after every one (sampled at temps (G, 2))
    """
    G, P, k = len(pf.count_vars), pf.P, self.config.spec_drafts
    before = hn.reshape(G, P, -1)[:, :P-1].pad(((0, 0), (1, 0), (0, 0))).reshape(1, G*P, -1)
    if k: pf.effects.append(self.mtp(self.mtp.eh_proj(self.mtp.enorm(emb).cat(cast(MTPBlock, self.mtp).hnorm(before), dim=-1)), 0)[:, :1, :1]
                            .contiguous())
    last = Tensor.cat(*[hn.shrink(((g*P + c - 1, g*P + c), None)) for g, c in enumerate(pf.count_vars)])
    if k: pf.effects.extend([self.pool_spec_h.shrink(((sl*2*(k+1), sl*2*(k+1)+1), None)).assign(last[g:g+1]) for g, sl in enumerate(pf.slot_vars)])
    return self._sample(last, temps, self._bpf_in.device)

  BPF_G = getenv("BPF_G", 8)  # prefills in a step of them: prompts at most
  # (BPF_PS: the rows of a prompt of a step, a jit each (a step's prompts padded to the smallest that fits them), BPF_P the most)
  BPF_PS = tuple(sorted({int(x) for x in str(getenv("BPF_PS", getenv("BPF_P", 168))).split(",") if x}))
  BPF_P = max(BPF_PS)
  BPF_ROWS = getenv("BPF_ROWS", BPF_G * min(BPF_PS))  # (a step's rows at most: its memory)
  def forward_batch_prefill(self, *vs:UOp, P:int|None=None) -> Tensor:
    """
    prefills of G prompts into slots (from position 0): vs the token counts and the slots of the G, their P tokens (padded) in _bpf_in, their
    temperatures in _batch_temp. the model and the MTP layer on all of them (the caches and states of the slots), the token after every one's
    last. returns (1, G) the tokens
    """
    G, P, devs = len(vs) // 3, P or self.BPF_P, self.token_embd.weight.device
    cvars, svars, stvars = vs[:G], vs[G:2*G], vs[2*G:3*G]
    pf = BatchPrefill(P, self._ints(cvars), self._ints(svars), cvars, svars, starts=self._ints(stvars) if stvars else None)
    _batch_rows[0] = [(st, P) for st in stvars] if stvars else [(UOp.const(0, dtypes.weakint), P)] * G
    _batch_prefill[0], Q8_ROWS_I8[0] = pf, False
    try:
      emb = self.token_embd(self._bpf_in[:, :G*P].to(devs)).float()
      x = emb
      if (hc:=self.config.hc) is not None: x = x.unsqueeze(2).expand(*x.shape[:2], hc.count, x.shape[-1]).contiguous()
      if hc is not None and self.sequence_parallel(G*P):
        # sequence parallel (as a prefill chunk): every device the streams of its rows, the sublayers on all of them
        x = all_gather(micro_batch_blocks(self.blk, [shard_rows(x, 1)], [0], [G*P])[0].mean(2), 1)
      else:
        for block in self.blk: x = block(x, 0)
        if hc is not None: x = x.mean(2)
      tok = self._prompt_ends(self.output_norm(x).float()[0], emb, pf, self._batch_temp[:2*G].reshape(G, 2))
    finally: _batch_rows[0], _batch_prefill[0] = None, None
    q8_quantize_cache_clear()  # don't keep the graph alive
    out = tok.reshape(1, G).cast(dtypes.int32)
    ret = self._bpf_out[:, :G].assign(Tensor(out.uop.mselect(0)) if isinstance(out.device, tuple) else out)
    Tensor.realize(ret, *pf.effects)
    return ret
  def batch_prefills(self) -> int:
    """
    the prompts prefilled together into slots (batch_prefill) at most: 0 if not (sequence parallel prefill of hybrid blocks). its memory: with a
    context of 64K or less (as the slots of more sequences)
    """
    return self.BPF_G if self.batch_sizes() and self.BPF_P and not self.config.qsa and self.sequence_parallel(self.BPF_G * self.BPF_P) and \
      (not seq_sharded(self.token_embd.weight.device) or getenv("BPF_SEQ", 0)) and \
      self.max_context <= getenv("BPF_MAX_CONTEXT", 65536) else 0
  def bpf_sizes(self, P:int|None=None) -> tuple[int, ...]:
    """
    the prompts of a batch_prefill step of prompts of P rows (a jit each): BPF_GS (sizes up to BPF_G, at most BPF_ROWS rows), a batch of n runs
    the smallest one of n or more
    """
    P = P or min(self.BPF_PS)
    gs = {g for g in (int(x) for x in str(getenv("BPF_GS", "")).split(",") if x) if 0 < g <= self.BPF_G} | {self.BPF_G}
    return tuple(sorted({g for g in gs if g * P <= self.BPF_ROWS and self.sequence_parallel(g * P)} or {1}))
  def bpf_class(self, n:int) -> int|None:
    """the rows of the batch_prefill steps of prompts of n tokens at most (the smallest of BPF_PS), None: more than BPF_P"""
    return next((p for p in self.BPF_PS if p >= n), None)
  def _bpf_jit(self, G:int|None=None, P:int|None=None):
    if not hasattr(self, "_bpf_jits"): self._bpf_jits:dict[tuple[int, int], Any] = {}
    if (key:=(G or self.BPF_G, P or self.BPF_P)) not in self._bpf_jits:
      self._bpf_jits[key] = TinyJit(functools.partial(self.forward_batch_prefill, P=key[1]), capture_first=True, scratch=self._scratch)
    return self._bpf_jits[key]
  def _bpf_vars(self, counts:list[int], slots:list[int], starts:list[int]|None=None) -> list[UOp]:
    return [UOp.variable(f"bpf_count{g}", 1, max(self.BPF_P, *self.MIX_PS)).bind(c) for g, c in enumerate(counts)] + \
           [UOp.variable(f"bpf_slot{g}", 0, self.batch_slots()-1).bind(s) for g, s in enumerate(slots)] + \
           [UOp.variable(f"bpf_start{g}", 0, SLOT_SPAN*self.batch_ctx()-1).bind(st) for g, st in enumerate(starts or [0]*len(counts))]
  def batch_prefill(self, prompts:list[tuple[list[int], int, float|tuple[float, ...]]], starts:list[int]|None=None, P:int|None=None) -> list[int]:
    """
    prefills of prompts (tokens, slot, temperature) (at most BPF_G of BPF_P tokens) into their slots in one step: the token after every one.
    starts: the positions they start at (a conversation continued from its states in the slot at the position), None: 0. P: the step's rows
    of a prompt (one of BPF_PS), None: the smallest that fits them
    """
    self._init_slots()
    P, n = P or cast(int, self.bpf_class(max(len(t) for t, _, _ in prompts))), len(prompts)
    G = next(g for g in self.bpf_sizes(P) if g >= n)
    assert 0 < n <= G and all(0 < len(t) <= P for t, _, _ in prompts)
    starts = (starts or [0] * n) + [(starts or [0] * n)[-1]] * (G - n)
    assert all(st + P <= self.batch_ctx() for st in starts), "a prompt past its slot"
    # (the rest of the G: the last one again, the same writes)
    prompts = prompts + [prompts[-1]] * (G - n)
    for g, (toks, _, temp) in enumerate(prompts):
      self._bpf_host_in[g*P:(g+1)*P] = array.array('i', toks + [0] * (P - len(toks)))
      self._batch_host_temp[2*g:2*g+2] = _temp_pair(temp)
    self._bpf_host_out[:G] = array.array('i', [-1] * G)
    self._bpf_jit(G, P)(*self._bpf_vars([len(t) for t, _, _ in prompts], [s for _, s, _ in prompts], starts))
    st = time.perf_counter()
    while min(self._bpf_host_out[:G]) < 0:
      if time.perf_counter() - st > 10:
        for d in to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device)): Device[d].synchronize()
        if min(self._bpf_host_out[:G]) < 0: raise RuntimeError("batched prefill: no tokens after the devices finished")
    return list(self._bpf_host_out[:n])

  def _bspec_jit(self, m:int):
    if not hasattr(self, "_bspec_jits"): self._bspec_jits:dict[int, Any] = {}
    if m not in self._bspec_jits: self._bspec_jits[m] = TinyJit(self.forward_batch_spec, capture_first=True, scratch=self._scratch)
    return self._bspec_jits[m]
  def _bspec_vars(self, seqs:list[tuple[int, int, int]]) -> list[UOp]:
    k, ctx = self.config.spec_drafts, SLOT_SPAN*self.batch_ctx()
    return [v for g, (mp, j, par) in enumerate(seqs) for v in (UOp.variable(f"bspec_mtp{g}", 0, ctx-1).bind(mp),
            UOp.variable(f"bspec_j{g}", 0, k).bind(j), UOp.variable(f"bspec_par{g}", 0, 1).bind(par))]

  def batch_spec_step(self, seqs:list[tuple[list[int], int, int, int, float|tuple[float, ...]]]) -> list[tuple[list[int], list[int]]]:
    """
    a step of batched speculative decoding of the slots 0..len(seqs)-1: seqs (the k+1 tokens of the last step, mtp_pos, j, par, temperature)
    of every one (see _spec_steps), their k+1 sampled tokens and k drafts
    """
    self._init_slots()
    self._slot_writes()
    k, n = self.config.spec_drafts, len(seqs)
    m = next(s for s in self.batch_sizes() if s >= n)
    assert m <= self.batch_slots(), f"{n} sequences, {self.batch_slots()} slots"
    # (the groups past n are the free slots' at position 0: what they write is overwritten when a sequence takes the slot)
    seqs = seqs + [([0]*(k+1), 0, 0, 1, 0.0)] * (m - n)
    # (the step's tokens and temperatures written at once, the host's time is the devices' idle time between steps)
    self._bspec_host_in[:m*(k+1)] = array.array('i', [t for y, _, _, _, _ in seqs for t in y])
    self._batch_host_temp[:2*m] = array.array('f', [v for *_, temp in seqs for v in _temp_pair(temp)])
    self._bspec_host_out[:m*(2*k+1)] = array.array('i', [-1] * (m*(2*k+1)))
    jit = self._bspec_jit(m)
    if (cap:=getattr(jit, "captured", None)) is not None and getenv("BSPEC_FAST_CALL", 1):
      # (captured: its variables' values as they are, no bound variables made a step)
      cap([], {name: v for g, (_, mp, j, par, _) in enumerate(seqs)
               for name, v in ((f"bspec_mtp{g}", mp), (f"bspec_j{g}", j), (f"bspec_par{g}", par))})
    else: jit(*self._bspec_vars([(mp, j, par) for _, mp, j, par, _ in seqs]))
    st = time.perf_counter()
    while min(self._bspec_host_out[:m*(2*k+1)]) < 0:
      # (a step that doesn't come: the devices' errors, or none)
      if time.perf_counter() - st > 10:
        for d in to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device)): Device[d].synchronize()
        if min(self._bspec_host_out[:m*(2*k+1)]) < 0: raise RuntimeError(f"batched step of {m} sequences: no tokens after the devices finished")
    out = list(self._bspec_host_out[:n*(2*k+1)])
    return [(out[g*(2*k+1):g*(2*k+1)+k+1], out[g*(2*k+1)+k+1:(g+1)*(2*k+1)]) for g in range(n)]

  # (mixed steps: the rows of a prompt (a jit each, a step's prompts padded to the smallest that fits them), the steps of decoding they're of)
  MIX_PS = tuple(sorted({int(x) for x in str(getenv("MIX_PS", "192,384")).split(",") if x}))
  def mix_sizes(self) -> tuple[int, ...]:
    """the sizes of the steps of decoding with mixed steps (MIX_SIZES, default the largest): fewer sequences step without prompts"""
    sizes = self.batch_sizes()
    return tuple(sorted({int(x) for x in str(getenv("MIX_SIZES", "")).split(",") if x and int(x) in sizes} or set(sizes[-1:])))
  def mix_class(self, n:int) -> int|None:
    """the rows of the mixed step of a prompt of n tokens (the smallest of MIX_PS), None: more"""
    return next((p for p in self.MIX_PS if p >= n), None)
  def _mix_jit(self, m:int, G:int, P:int):
    if not hasattr(self, "_mix_jits"): self._mix_jits:dict[tuple[int, int, int], Any] = {}
    if (key:=(m, G, P)) not in self._mix_jits:
      self._mix_jits[key] = TinyJit(functools.partial(self.forward_batch_spec, prompts=(G, P)), capture_first=True, scratch=self._scratch)
    return self._mix_jits[key]

  def batch_mixed_step(self, seqs:list[tuple[list[int], int, int, int, float|tuple[float, ...]]],
                       prompts:list[tuple[list[int], int, float|tuple[float, ...], int]], P:int|None=None) \
                       -> tuple[list[tuple[list[int], list[int]]], list[int]]:
    """
    a step of batched speculative decoding (as batch_spec_step's) with the prefills of prompts (tokens, slot, temperature, start): their
    conversations continued from their states in the slots at start (landing slots, see mix_landing). returns the sequences' step and the
    token after every prompt. P: the rows of a prompt, None: the smallest of MIX_PS that fits them
    """
    self._init_slots()
    self._slot_writes()
    k, n, G = self.config.spec_drafts, len(seqs), len(prompts)
    m = next(s for s in self.mix_sizes() if s >= n)
    P = P or cast(int, self.mix_class(max(len(t) for t, _, _, _ in prompts)))
    assert 0 < G <= self.mix_landing() and all(0 < len(t) <= P and st + P <= self.batch_ctx() for t, _, _, st in prompts)
    seqs = seqs + [([0]*(k+1), 0, 0, 1, 0.0)] * (m - n)
    def pair(temp:float|tuple[float, ...]) -> tuple[float, ...]: return temp[:2] if isinstance(temp, tuple) else (temp, 1.0)
    self._bspec_host_in[:m*(k+1)] = array.array('i', [t for y, _, _, _, _ in seqs for t in y])
    self._batch_host_temp[:2*(m+G)] = array.array('f', [v for *_, temp in seqs for v in pair(temp)] + [v for _, _, temp, _ in prompts
                                                                                                     for v in pair(temp)])
    for g, (toks, _, _, _) in enumerate(prompts): self._bpf_host_in[g*P:(g+1)*P] = array.array('i', toks + [0] * (P - len(toks)))
    self._bspec_host_out[:m*(2*k+1)] = array.array('i', [-1] * (m*(2*k+1)))
    self._bpf_host_out[:G] = array.array('i', [-1] * G)
    jit = self._mix_jit(m, G, P)
    pvals = [len(t) for t, _, _, _ in prompts] + [s for _, s, _, _ in prompts] + [st for _, _, _, st in prompts]
    if (cap:=getattr(jit, "captured", None)) is not None and getenv("BSPEC_FAST_CALL", 1):
      names = [f"bpf_{x}{g}" for x in ("count", "slot", "start") for g in range(G)]
      cap([], {name: v for g, (_, mp, j, par, _) in enumerate(seqs)
               for name, v in ((f"bspec_mtp{g}", mp), (f"bspec_j{g}", j), (f"bspec_par{g}", par))} | dict(zip(names, pvals)))
    else: jit(*self._bspec_vars([(mp, j, par) for _, mp, j, par, _ in seqs]), *self._bpf_vars(pvals[:G], pvals[G:2*G], pvals[2*G:]))
    st = time.perf_counter()
    while min(self._bspec_host_out[:m*(2*k+1)]) < 0 or min(self._bpf_host_out[:G]) < 0:
      if time.perf_counter() - st > 10:
        for d in to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device)): Device[d].synchronize()
        if min(self._bspec_host_out[:m*(2*k+1)]) < 0 or min(self._bpf_host_out[:G]) < 0:
          raise RuntimeError(f"mixed step of {m} sequences: no tokens after the devices finished")
    out = list(self._bspec_host_out[:n*(2*k+1)])
    return [(out[g*(2*k+1):g*(2*k+1)+k+1], out[g*(2*k+1)+k+1:(g+1)*(2*k+1)]) for g in range(n)], list(self._bpf_host_out[:G])

  # ******** sequences sharing the caches (qwen4exp): every one its region of the attention caches' rows (the token buffer's too) and a slot of
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
    glm = bool(self.config.cache_rows and self.config.kv_lora_rank and not self.config.qsa and seq_sharded(devs))
    return bool((self.config.qsa and self.gated or glm) and self.config.spec_drafts and self.config.indexer is not None and
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
    return getenv("SEQ_SLOTS", 16 if self.config.kv_lora_rank and not self.config.qsa else max(self.SEQ_SIZES))
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
      if isinstance(b, GatedDeltaNetBlock): b._init_slots(n, 0, half=bool(getenv("SEQ_STATE_HALF", 0)), replay=bool(SEQ_REPLAY))
    # (GLM: the indexer's ring of every slot, the MLA caches the regions')
    for b in self.blk + [self.mtp]:
      if isinstance(b, MLATransformerBlock): b._init_seq_slots(n)
    # every slot's hidden states (the streams) of the last step's tokens, two parities of k+1
    self.seq_spec_h = _zeros(n*2*(k+1), self._spec_dim, device=devs)
    # every slot's checkpoint: its states of a position (the end of its last prompt), the hidden state of the last token
    self.seq_ckpt_h = _zeros(n, self._spec_dim, device=devs)
    # (SEQ_CKPT_HOST: the checkpoints' recurrent states in host memory: a copy a turn, the device memory more slots')
    self.seq_ckpt_states = [[_slot_pool(n, b.pool_rec, (1, *b.pool_rec.shape[1:]), host=bool(getenv("SEQ_CKPT_HOST", 1)))[0]] +
                            [_slot_pool(n, p, (1, *p.shape[1:]))[0] for p in b.pool_conv] for b in self.blk if isinstance(b, GatedDeltaNetBlock)]
    # (GLM: the checkpoints' indexer rings (the last tokens of the prompt, its last pool's), see _seq_ckpt_copy)
    self.seq_ckpt_rings = [(b.indexer, _slot_pool(n, b.indexer.ring)[0]) for b in self.blk + [self.mtp]
                           if isinstance(b, MLATransformerBlock) and b.indexer is not None]
    self._spec_init()
    if not hasattr(self, "_token_buf"): self._token_host, self._token_buf = _host_ints(devs, self._token_rows())
    # in host memory: a step's input tokens, the temperature and top_p of every sequence, its sampled tokens and drafts (written by one device)
    self._seq_host_in, self._seq_in = _host_ints(devs, m*(k+1))
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
    if save and (k:=self.config.spec_drafts) and getenv("SEQ_SPEC_CLEAR", 1):
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
      if not isinstance(b, QSABlock): continue
      copies.append(mv(b.cache_k, src, dst, n, 2))
      if (idx:=b.indexer) is not None:
        kp = idx.idx.kpool
        copies += [mv(idx.cache_k, src, dst, n, 0), mv(idx.cache_pk, src // kp, dst // kp, n // kp, 0)]
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
    qsa = [b for b in self.blk + [self.mtp] if isinstance(b, QSABlock)]
    if not qsa:
      # (other model families: no stash, an evicted sequence is prefilled again)
      self.stash_entries = self.stash_chunks = self.stash_chunk_nbytes = self.stash_entry_nbytes = 0
      return
    def per_row(t:Tensor, axis:int) -> int:
      # (a device's bytes of a row: split over the devices or not)
      return t.nbytes() // int(t.shape[axis]) // (len(t.device) if isinstance(t.device, tuple) and t.uop.axis is not None else 1)
    # (the host memory of a chunk: every distinct KV head's rows and the indexers' (see below))
    kvr = self._kv_real
    chunk = C * sum(kvr * per_row(b.cache_k, 2) + (per_row(b.indexer.cache_k, 0) + per_row(b.indexer.cache_pk, 0) // b.indexer.idx.kpool
                                                    if b.indexer is not None else 0) for b in qsa)
    self.stash_entries = getenv("SEQ_STASH_ENTRIES", 32) if gb > 0 else 0
    gdn = [b for b in self.blk if isinstance(b, GatedDeltaNetBlock)]
    entry = sum(per_row(t, 0) for b in gdn for t in (b.pool_rec, *b.pool_conv)) + per_row(self.seq_spec_h, 0) + \
            sum(per_row(t, 0) for c in self.seq_ckpt_states for t in c) + per_row(self.seq_ckpt_h, 0)
    self.stash_chunk_nbytes, self.stash_entry_nbytes = chunk, entry  # exact host-copy units for serving telemetry
    self.stash_chunks = max(0, int(gb * 1e9 - self.stash_entries * entry) // chunk) if gb > 0 else 0
    if not self.stash_chunks: self.stash_entries = 0
    if not self.stash_entries: return
    H, E = self.stash_chunks * C, self.stash_entries
    # (the attention caches: a copy of every distinct one in host memory every device maps, written by one device: the KV heads repeated over
    # the devices are the same, the indexers' caches are on every device)
    devs, kvr = cast(tuple[str, ...], self.token_embd.weight.device), self._kv_real
    head = [d * kvr // len(devs) for d in range(len(devs))]
    # (the devices of a copy write their part of its rows: SEQ_STASH_SPLIT of them at most)
    def code(group:list[list[int]]) -> int:
      ret = 0
      for g in group:
        g = g[:max(1, getenv("SEQ_STASH_SPLIT", 3))]
        for i, d in enumerate(g): ret |= (i | (len(g) << 4)) << (8*d)
      return ret
    self._stash_writers = code([[d for d in range(len(devs)) if head[d] == h] for h in range(kvr)]), code([list(range(len(devs)))])
    self._stash_kv = []
    kv_specs = [[(b.cache_k, H)] + ([(b.indexer.cache_k, H), (b.indexer.cache_pk, H // b.indexer.idx.kpool)] if b.indexer is not None else [])
                for b in qsa]
    owners = [devs[head.index(h)] for h in range(kvr)] + [devs[0]]
    def row_bytes(t:Tensor) -> int: return per_row(t, 2 if t is not None and t.ndim == 4 else 0)
    sizes = [sum(round_up(rows * row_bytes(t), 4096) for b in kv_specs for t, rows in b[:1]) for _ in range(kvr)] + \
            [sum(round_up(rows * row_bytes(t), 4096) for b in kv_specs for t, rows in b[1:])]
    arenas = [_shared_host(o, devs, sz) for o, sz in zip(owners, sizes)]
    offs = [0] * len(arenas)
    def take(t:Tensor, rows:int, arena_of_dev:list[int]) -> Tensor:
      local = (rows, *[int(x) for x in t.shape[3 if t.ndim == 4 else 1:]]) if t.ndim == 4 else (rows, *[int(x) for x in t.shape[1:]])
      nb, views = prod(local) * t.dtype.itemsize, []
      for d in range(len(devs)): views.append(arenas[arena_of_dev[d]][d].view(nb, offs[arena_of_dev[d]]))
      for a in set(arena_of_dev): offs[a] += round_up(nb, 4096)
      (mb:=MultiBuffer.__new__(MultiBuffer)).bufs = views
      u = UOp.from_buffer(mb, t.dtype, devs).reshape((1, 1, *local) if t.ndim == 4 else local)
      return Tensor(u.unshard(0) if t.ndim == 4 else u)
    for b in kv_specs:
      self._stash_kv.append(tuple(take(t, rows, head if i == 0 else [kvr]*len(devs)) for i, (t, rows) in enumerate(b)))
    # (a sequence's states of its last position: the recurrent and conv ones, the hidden state of its last token. and its checkpoint's)
    states = [t for b in gdn for t in (b.pool_rec, *b.pool_conv)] + [self.seq_spec_h] + \
             [t for c in self.seq_ckpt_states + [[self.seq_ckpt_h]] for t in c]
    self._stash_state = _host_like([(t, 0, E) for t in states])
    self._stash_state_count, self._stash_rings, self._stash_replays = len(states), [], []
    self._stash_calls = self._chunk_calls(min((self.seq_rows() + self.REGION_ALIGN) // C - 1, self.stash_chunks))
    self._stash_kv_jits = {(sv, n): TinyJit(functools.partial(self._stash_kv_copy, sv, n), capture_first=True)
                           for sv in (True, False) for n in self._stash_calls}
    self._stash_state_jits = {sv: TinyJit(functools.partial(self._stash_state_copy, sv), capture_first=True) for sv in (True, False)}
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
    # (the stash copies a region's pools whole on every card: not the split ones)
    assert not any(index(b).shard for b in kb), "GLM stash with SEQ_IDX_SHARD: no (SEQ_STASH_GB=0)"
    # Latents use seq_move's (local rows, devices, features) view: C/nd rows a device. Index pools are replicated.
    self._stash_kv = []
    for b in kb:
      latent = _host_like([(b.slot_rows_view(b.cache_s), 0, self.stash_chunks * C // nd)])[0]
      pools = _host_like([(index(b).cache_pk, 0, self.stash_chunks * C // index(b).idx.kpool)])[0]
      self._stash_kv.append((latent, pools))
    # Whole ring/replay windows, not one position: every row after a checkpoint restore must have the same state as before eviction.
    self._stash_rings = list(self.seq_ckpt_rings)
    self._stash_replays = replay
    states = [t for b in gdn for t in (b.pool_rec, *b.pool_conv)] + [self.seq_spec_h] + \
             [t for c in self.seq_ckpt_states + [[self.seq_ckpt_h]] for t in c]
    extra:list[Tensor] = []
    for b, ck in self._stash_rings:
      for t in (b.pool_ring, ck): extra.append(_host_like([(t, 0, self.stash_entries * int(b.ring.shape[0]))])[0])
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
        glm_idx, kp = gb.indexer, gb.indexer.idx.kpool
        # (the stash's rows from its chunk, the caches' from the region's row: a sequence comes back to another region)
        items = [(rows(st[0], chunk*(C//nd), C*chunks//nd, 0), rows(gb.slot_rows_view(gb.cache_s), row//nd, C*chunks//nd, 0)),
                 (rows(st[1], chunk*(C//kp), C*chunks//kp, 0), rows(glm_idx.cache_pk, row//kp, C*chunks//kp, 0))]
        for dst, src in items:
          # The host and device have identical region-relative row layout; the grouped kernels' absolute base stays out of this copy.
          if save: saves.append(dst.assign(src))
          else: pairs.append((src, dst))
      # (a restore's copy kernels read the host rows: no device copy of them first (a stash call's rows of every cache: ~126 MB on C5))
      Tensor.realize(*saves, *[x.assign(d) for x, d in pairs])
      return
    for b, st in zip([b for b in self.blk + [self.mtp] if isinstance(b, QSABlock)], self._stash_kv):
      ts = [(st[0], b.cache_k, row, chunk*C, C*chunks, self._stash_writers[0])]
      if (idx:=b.indexer) is not None:
        kp = idx.idx.kpool
        ts += [(st[1], idx.cache_k, row, chunk*C, C*chunks, self._stash_writers[1]),
               (st[2], idx.cache_pk, row//kp, chunk*(C//kp), (C//kp)*chunks, self._stash_writers[1])]
      for dst, src, r, c, n, writers in ts:
        axis = 2 if src.ndim == 4 else 0
        if save:
          # (a device of every distinct copy writes it: the host memory takes a few GB/s of the devices' writes at once)
          # (the row and chunk: bound on an input, the kernel's variables)
          s = Tensor(src.uop.after(r, c))
          saves.append(Tensor.custom_kernel(dst, s, _device_index(devs), fxn=functools.partial(_stash_save_kernel, row=_unbind(r), at=_unbind(c),
                                                                                              n=n, writers=writers))[0])
        else: pairs.append((rows(dst, c, n, axis), rows(src, r, n, axis)))
    Tensor.realize(*saves, *[x.assign(d) for d, x in pairs])
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

  def _prefill_jit(self, n:int, dense:bool) -> Any:
    """the jit of a prefill chunk of n tokens on a sequence's region (dense: a context the indexer's top_k covers), see __call__"""
    jits = self.__dict__.setdefault("jits" + ("_dense" if dense else "") + "_seq", {})
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
    for lane in range(getenv("MICRO_BATCHES", 1)):
      if mb_queue(lane) and getenv("MB_QUEUES", 1):
        for d in devs: getattr(Device[d], "compute_queue_n", lambda i: None)(mb_queue(lane))
      # (and the decoding steps' lanes, see DEC_LANE)
      for ln in (lane, lane + SEL_LANE) + ((DEC_LANE, DEC_LANE + SEL_LANE) if lane == 0 and getenv("SEQ_DEC_LANE", 1) else ()) + \
                ((IDX_LANE,) if lane == 0 and getenv("SEQ_IDX_SHARD", 0) else ()):
        FLAG_LANE[0] = ln
        for g in (4, 16, 64): flag_allreduce(devs, g)
        # (the ring's mailbox: made before the jits are captured (in a fork, it'd be the child's))
        ring_mailbox(devs)
    FLAG_LANE[0] = 0
    # (the ring collectives' queue of their own, see ring_queue)
    for lane in range(getenv("MICRO_BATCHES", 1)):
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
    for m in sorted(self.SEQ_SIZES, reverse=True):
      add(f"step{m}", self._seq_jit(m), functools.partial(self.seq_step, [(0, 0, 1, 0, 1, [0]*(k+1), 0.0)] * m))
      for d in self._step_drafts(m)[1:]:
        add(f"step{m}k{d}", self._seq_jit(m, k=d), functools.partial(self.seq_step, [(0, 0, 1, 0, 1, [0]*(k+1), 0.0)] * m, drafts=d))
    # (a mixed step's prompt: of slot 1 in the region after the sequences', a context past top_k)
    mix_prompt = (1, self.REGION_ALIGN * (-(-(top + self.SEQ_MIX_ROWS) // self.REGION_ALIGN)), top, 1, 0.0)
    for m in self.SEQ_MIX_SIZES:
      add(f"mixed{m}", self._seq_jit(m, True), functools.partial(self.seq_step, [(0, 0, 1, 0, 1, [0]*(k+1), 0.0)] * m, mix_prompt))
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
    mp_vars = dict(zip(("seqp_slot", "seqp_base", "seqp_start", "seqp_count"), mix_prompt[:4]))
    for m in self.SEQ_MIX_SIZES:
      par.append((jit:=self._seq_jit(m, True), functools.partial(lambda j, m: j(*self._seq_vars([(0, 0, 1, 0, 1)]*m), *self._prompt_vars(mp_vars)),
                                                                 jit, m)))
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
    # (SEQ_WARMUP_ONLY=step: the steps' jits only (experiments: the others captured when first called))
    if (only:=getenv("SEQ_WARMUP_ONLY", "")):
      steps = {self._seq_jit(m, k=d) for m in self.SEQ_SIZES for d in self._step_drafts(m)}
      par = [(j, c) for j, c in par if j in steps or any(j is lj for lj, _ in loads)]
      calls = {k: v for k, v in calls.items() if k.startswith(str(only))}
    if DEBUG >= 1:
      for i, (_, call) in enumerate(par): print(f"seq_warmup: jit {i}: {getattr(call, 'func', call)} {getattr(call, 'args', '')}"[:160])
    _lap("capture_parallel")
    if getenv("WARMUP_PARALLEL", 1): capture_parallel(par, meanwhile)
    else: meanwhile(lambda j, wait=True: False)
    _lap("captured")
    if DEBUG >= 1: print(f"seq_warmup: {len(par)} jits compiled at {time.perf_counter()-st:.2f}s")
    for name, (jit, call) in calls.items():
      if name in ran: continue
      st = time.perf_counter()
      call()
      if DEBUG >= 1: print(f"seq_warmup: {name} run in {time.perf_counter()-st:.2f}s")

  def seq_prefill(self, slot:int, base:int, tokens:list[int], start:int, hrow:int, temp:float|tuple[float, ...]=0.0,
                  upto:int|None=None) -> int|None:
    """
    the tokens from start of the sequence of slot `slot` (its region of the caches from base) prefilled, from its states of position start (the
    slot's, and its row hrow the hidden state of the token before it; start 0: none): its states of position len(tokens) into the slot (the
    hidden state of the last token its row 0). returns the token sampled after the last. upto: only the tokens up to upto (a slice of the
    prompt, the rest from there): None returned
    """
    self._finish_load()
    self._seq_init()
    if upto is not None:
      assert 0 <= start < upto < len(tokens) and base % self.REGION_ALIGN == 0 and base + len(tokens) <= self.seq_rows()
      for d in to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device)): Device[d].synchronize()
      # (the token after the slice the prompt's: the slice's sampled token isn't written over it, the next slice's tokens are its)
      self._token_host[base+start:base+upto+1] = array.array('i', tokens[start:upto+1])
      self._seq_copy_call(False, slot, start, hrow, keep=start > 0)
      pos = start
      for n in self._seq_chunks(upto - start):
        self._seq_chunk(base, pos, n)
        pos += n
      self._seq_copy_call(True, slot, upto, 0)
      return None
    assert 0 <= start < len(tokens) and base % self.REGION_ALIGN == 0 and base + len(tokens) <= self.seq_rows()
    L, devs = len(tokens), to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device))
    for d in devs: Device[d].synchronize()  # (a step may still read the token buffer)
    self._token_host[base+start:base+L+1] = array.array('i', tokens[start:] + [-1])
    self._seq_copy_call(False, slot, start, hrow, keep=start > 0)
    self._seq_pf_host_temp[:TEMP_W] = _temp_row(temp)
    pos = start
    for n in self._seq_chunks(L - start):
      self._seq_chunk(base, pos, n)
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
    assert not self.config.qsa, "qwen4exp: see qwen4exp.py"
    self._seq_init()
    if not hasattr(self, "_seq_first_jit"): self._seq_first_jit = TinyJit(self._seq_first, capture_first=True)
    for d in to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device)): Device[d].synchronize()  # (a step may read the token buffer)
    self._token_host[base+pos] = -1
    self._seq_pf_host_temp[:TEMP_W] = _temp_row(temp)
    self._seq_first_jit(*self._first_vars(slot, base+pos))
    return self._wait_host(lambda: self._token_host[base + pos], "the token after a prompt")

  def seq_slice(self, n:int) -> int:
    """a prefill slice of about n tokens (see the server's PREFILL_SLICE): GLM's whole chunks of generate's (its prefill the same chunks)"""
    return n if self.config.qsa or getenv("SEQ_CHUNK_PLAN", 0) else round_up(n, self.prefill_chunks()[-1])
  def _seq_chunks(self, n:int) -> list[int]:
    """
    the chunks of a prefill of n tokens: GLM as generate's (the biggest size, the rest: the same chunks, its greedy tokens the main path's;
    SEQ_CHUNK_PLAN=1 the least time's, see _chunk_plan), qwen4exp the least time's
    """
    if self.config.qsa or getenv("SEQ_CHUNK_PLAN", 0): return self._chunk_plan(n)
    big, ret = self.prefill_chunks()[-1], []
    # (PREFILL_BIG_ALONE=1, default off: the biggest chunk size only while no other sequence decodes (the server sets _pf_alone before a
    # slice), else the next size: a long chunk is a long wait for the decoding ones. the chunks then depend on the load: a reorder)
    if getenv("PREFILL_BIG_ALONE", 0) and not getattr(self, "_pf_alone", True) and len(sizes:=self.prefill_chunks()) > 1: big = sizes[-2]
    while n > 0:
      ret.append(min(big, n))
      n -= ret[-1]
    return ret
  def _chunk_plan(self, n:int) -> list[int]:
    """
    the chunks of n tokens of a prefill: the biggest size while more than twice it is left, the rest in the sizes of the least time (a chunk's
    time about a fixed part and its padded rows': 700 tokens are 576+192, not 2112)
    """
    sizes, ret = self.prefill_chunks(), []
    while n > 2*sizes[-1]:
      ret.append(sizes[-1])
      n -= sizes[-1]
    def cost(s:int) -> float: return CHUNK_FIXED_ROWS + s
    best:list[tuple[float, int]] = [(0.0, 0)] + [(math.inf, 0)] * n
    for i in range(1, n+1): best[i] = min((cost(s) + best[max(0, i - s)][0], s) for s in sizes)
    while n > 0:
      ret.append(min(best[n][1], n))
      n -= ret[-1]
    return ret

  def _seq_chunk(self, base:int, pos:int, n:int, part:int|None=None):
    """a prefill chunk: the n tokens from pos of the region from base in the token buffer (the jit of the smallest chunk size of n or more)"""
    sizes = self.prefill_chunks()
    vb, sp = UOp.variable("seq_base", 0, self.seq_rows()).bind(base), UOp.variable("start_pos", 0, self.max_context-1).bind(pos)
    nt = UOp.variable("toks", 1, next(s for s in sizes if n <= s)).bind(n)
    self(self._token_buf[:, vb+sp:vb+sp+nt], sp, self._seq_pf_temp, vb, part)

  def forward_seqs(self, *vs:UOp) -> Tensor: return self._forward_seqs(vs, ())
  def forward_seqs_mixed(self, *vs:UOp) -> Tensor:
    """forward_seqs and a prompt's tokens (a mixed step): the last 4 of vs its slot, base, start and count (see seq_step)"""
    return self._forward_seqs(vs[:-4], vs[-4:])
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
    Q8_ROWS_I8[0] = bool(getenv("SEQ_I8", 0))
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

  def _forward_seqs(self, vs:tuple[UOp, ...], pv:tuple[UOp, ...]) -> Tensor:
    """
    a step of speculative decoding of G sequences sharing the caches: vs the slot, the base of the region, mtp_pos, j and par of every one (as
    forward_spec's), its k+1 tokens (the last step's sampled ones) in _seq_in, its temperature and top_p in _seq_temp. returns (1, G*(2k+1)):
    every one's k+1 sampled tokens and k drafts. (the drafts are sampled with the noise of the row they're verified at: an exact sample of the
    model, more of them accepted at a temperature)
    pv (a mixed step): a prompt's chunk of SEQ_MIX_ROWS rows at most (its tokens in the token buffer) prefilled with the step, its rows after
    the sequences' (the projections and the experts of all the rows at once: no prefill of its own), the token after it last in the output
    """
    if not self.config.qsa:
      assert not pv, "GLM: no mixed steps"
      return self._forward_seqs_glm(vs)
    k, G, devs = self.config.spec_drafts, len(vs) // 6, self.token_embd.weight.device
    hc, mtp, ple = unwrap(self.config.hc), cast(QwenMTPBlock, self.mtp), unwrap(self.config.ple)
    sl, bs, mp, js, ps, kbs = (list(vs[i::6]) for i in range(6))
    slots, bases, H = self._ints(tuple(sl)), self._ints(tuple(bs)), 2*(k+1)
    y = self._seq_in[:, :G*(k+1)]
    temps = self._seq_temp[:TEMP_W*G].reshape(G, TEMP_W)
    def batch(pos:list[UOp], R:int) -> SeqBatch: return SeqBatch(G, R, slots, bases, self._ints(tuple(pos)))
    # the hidden states of the last step's tokens: every slot's rows of parity 1-par
    last = Tensor.cat(*[self.seq_spec_h.shrink(((s*H + (1-p)*(k+1), s*H + (2-p)*(k+1)), None)) for s, p in zip(sl, ps)])
    # (the step's seed of the noise from the host: a random tensor of the devices is a different one on every device, a sampled token of one
    # not the others')
    seed = self._seq_temp[-1:].bitcast(dtypes.uint32)
    Q8_ROWS_I8[0] = bool(getenv("SEQ_I8", 0))
    # (SEQ_DEC_LANE: the step's collectives on the decoding lanes, see DEC_LANE)
    prev_lane = FLAG_LANE[0]
    if getenv("SEQ_DEC_LANE", 1): FLAG_LANE[0] = DEC_LANE
    try:
      # the MTP layer catches up on the k+1 positions from mtp_pos, row j of every sequence drafts
      _seq_batch[0], _batch_rows[0] = batch(mp, k+1), [(p, k+1) for p in mp]
      emb = self.token_embd(y.to(devs)).float()
      Nd = len(devs) if isinstance(devs, tuple) else 1
      if getenv("SEQ_SPLIT", 0) and getenv("SEQ_SPLIT_MTP", 1) and not pv and Nd > 1 and (G*(k+1)) % Nd == 0 and G*(k+1)//Nd <= SEQ_SPLIT_MAX:
        # (SEQ_SPLIT: the MTP layer's rows split over the devices as the model's, its streams gathered for the drafts' head and the next pass)
        _seq_split[0] = G*(k+1) // Nd
        try:
          xs = mtp._project_input_split(_split_pad(shard_rows(emb, 1)).contiguous(), _split_pad(shard_rows(last.unsqueeze(0), 1)).contiguous())
          m = all_gather(_split_real(mtp(xs, 0)).contiguous(), 1, dtypes.half).float()
        finally: _seq_split[0] = 0
      else: m = mtp.predict(emb, last.unsqueeze(0), 0)
      m = Tensor.cat(*[m.shrink((None, (g*(k+1)+j, g*(k+1)+j+1), None, None)) for g, j in enumerate(js)], dim=1)
      drafts:list[Tensor] = []
      for i in range(k):
        # (draft i of sequence g: verified at its row i, sampled with that row's noise at its temperature, not its top_p)
        drafts.append(self._sample(mtp.head(m)[0], temps[:, :1], y.device, (seed, Tensor.arange(G) * (k+1) + i)).reshape(1, G))
        if i < k-1:
          pos = [p + j + 1 + i for p, j in zip(mp, js)]
          _seq_batch[0], _batch_rows[0] = batch(pos, 1), [(p, 1) for p in pos]
          m = mtp.predict(self.token_embd(drafts[-1].to(devs)).float(), m.flatten(2), 0)
      # verify: the model on every sequence's token at start_pos (mtp_pos+j) and its drafts
      vt = Tensor.cat(*[y.shrink((None, (g*(k+1)+j, g*(k+1)+j+1))).cat(*[d[:, g:g+1] for d in drafts], dim=1) for g, j in enumerate(js)],
                      dim=1).contiguous()
      starts = [p + j for p, j in zip(mp, js)]
      x = self.token_embd(vt.to(devs)).float()
      # (the n-gram rows: every sequence's tokens before its start from the token buffer, then its verified ones)
      n = len(ple.multipliers)
      wins = Tensor.cat(*[self._token_window(self._token_buf, b, s, n-1).reshape(1, n-1) for b, s in zip(bs, starts)])
      ple_rows = self._ple_rows_dev(wins.cat(vt.reshape(G, k+1).to(devs), dim=1))
      if pv:
        # the prompt's rows after the sequences' (its tokens from the token buffer, past its count: not tokens, their rows' outputs not used)
        psl, pb, pst, pcnt = pv
        P = self.SEQ_MIX_ROWS
        ptok = self._token_buf.shrink((None, (pb + pst, pb + pst + P))).contiguous().maximum(0)
        x = x.cat(pemb:=self.token_embd(ptok.to(devs)).float(), dim=1)
        ple_rows = ple_rows.cat(self._ple_rows_dev(self._token_window(self._token_buf, pb, pst, P + n-1)))
        _seq_prompt[0] = SeqPrompt(P, G*(k+1), psl, pb, pst, pcnt, self._ints((psl,)), self._ints((pst,)), self._ints((pcnt,)))
      x = x.unsqueeze(2).expand(*x.shape[:2], hc.count, x.shape[-1]).contiguous()
      ple_emb = self._ple_emb(ple_rows.unsqueeze(0))
      _seq_batch[0], _batch_rows[0] = batch(starts, k+1), [(s, k+1) for s in starts]
      # (SEQ_REPLAY: the positions of the slots' recurrent states, their replayed tokens up to a sequence's start)
      if SEQ_REPLAY:
        assert not pv, "SEQ_REPLAY: no mixed steps (SEQ_MIX_SIZES)"
        _seq_batch[0] = replace(cast(SeqBatch, _seq_batch[0]), kbases=self._ints(tuple(kbs)))
      # SEQ_SPLIT: the streams' rows split over the devices (whole sequences on each, padded to tiles of 32): the hyper-connections and the
      # router of a device's rows only, a sublayer's input gathered and its output reduce-scattered (the bytes of an allreduce)
      N = len(devs) if isinstance(devs, tuple) else 1
      split = bool(getenv("SEQ_SPLIT", 0)) and not pv and N > 1 and (G*(k+1)) % N == 0 and G*(k+1) // N <= SEQ_SPLIT_MAX
      if split:
        _seq_split[0] = G*(k+1) // N
        x = _split_pad(shard_rows(x, 1)).contiguous()
      try:
        for block in self.blk: x = block(x, 0, ple_emb=ple_emb)
        if split: x = all_gather(_split_real(x).contiguous(), 1, dtypes.half).float()
      finally: _seq_split[0] = 0
      effects:list[Tensor] = []
      if pv:
        sp, _seq_prompt[0], _seq_batch[0], _batch_rows[0] = cast(SeqPrompt, _seq_prompt[0]), None, None, None
        effects += sp.effects
        xp, x = x[:, G*(k+1):], x[:, :G*(k+1)]
        hp = xp.flatten(2).float()
        # (its last token's hidden state: the token after it sampled with the sequences' (one sampling: its shared counts written once))
        last = hp[0].shrink(((pcnt - 1, pcnt), None))
        # the MTP layer's caches of the prompt's positions (as _mtp_prefill: the hidden state before it the model's row 0, its last token's
        # there after), written after the drafts' writes
        carry = self._spec_h[0:1].unsqueeze(0).contiguous()
        _seq_base[0] = pb
        try:
          with mtp.caches_set(*mtp._seq_written): mm = mtp.prefill_cache(pemb, carry.cat(hp[:, :-1], dim=1), pst, pcnt)
        finally: _seq_base[0] = 0
        effects += [mm[:, :, :1, :1].contiguous(), self._spec_h[0:1].assign(last)]
      hsv = x.flatten(2)[0].float()
      hs = Tensor.cat(*[self.seq_spec_h.shrink(((s*H + p*(k+1), s*H + (p+1)*(k+1)), None)).assign(hsv[g*(k+1):(g+1)*(k+1)])
                        for g, (s, p) in enumerate(zip(sl, ps))])
      tt = temps.reshape(G, 1, TEMP_W).expand(G, k+1, TEMP_W).reshape(G*(k+1), TEMP_W)
      hrows, keys = hs, Tensor.arange(G*(k+1))
      if pv:
        # (the prompt's at its temperature, the noise of a row of no sequence's)
        hrows, keys = hs.cat(last), Tensor.arange(G*(k+1) + 1)
        tt = tt.cat(self._seq_temp[TEMP_W*self._seq_m:TEMP_W*(self._seq_m+1)].reshape(1, TEMP_W))
      tok = self._sample(self._hc_out(hrows.reshape(1, -1, hc.count, hrows.shape[-1] // hc.count))[0], tt, y.device, (seed, keys))
    finally: _seq_batch[0], _batch_rows[0], _seq_prompt[0], Q8_ROWS_I8[0], FLAG_LANE[0] = None, None, None, False, prev_lane
    q8_quantize_cache_clear()  # don't keep the graph alive
    ptok_out = tok[G*(k+1):] if pv else None
    tok = tok[:G*(k+1)]
    out = tok.reshape(G, k+1).cat(*[d.reshape(G, 1) for d in drafts], dim=1).reshape(1, G*(2*k+1)).cast(dtypes.int32)
    if ptok_out is not None: out = out.cat(ptok_out.reshape(1, 1).cast(dtypes.int32), dim=1)
    W = G*(2*k+1) + (1 if pv else 0)
    ret = self._seq_out[:, :W].assign(Tensor(out.uop.mselect(0)) if isinstance(out.device, tuple) else out)
    if effects: Tensor.realize(ret, *effects)
    return ret

  _step_k:int|None = None  # (the drafts of the step being captured, see _seq_jit)
  def _step_drafts(self, m:int) -> list[int]:
    """the drafts of the steps of m sequences: the size's, then the ones of a sequence whose last step (of another size) drafted more"""
    d = self.drafts_of(m)
    return [d] + [x for x in sorted({self.drafts_of(s) for s in self.SEQ_SIZES}) if x > d]
  def _seq_jit(self, m:int, mixed:bool=False, k:int|None=None):
    """the jit of a step of m sequences and k drafts (default the size's, see drafts_of)"""
    if not hasattr(self, "_seq_jits"): self._seq_jits:dict[tuple, Any] = {}
    k = self.drafts_of(m) if k is None else k
    # (a step of fewer drafts than the layouts': captured with _step_k set)
    key = (m, mixed) if k == self.config.spec_drafts else (m, mixed, k)
    if key not in self._seq_jits:
      fn = self.forward_seqs_mixed if mixed else self.forward_seqs
      if k != self.config.spec_drafts:
        assert not mixed and not self.has_recurrent_block, "SPEC_DRAFTS_BY_SIZE: GLM_SEQ steps of MLA models only (no mixed steps, no replay)"
        def forward_seqs_k(*vs:UOp, fn=fn, k=k) -> Tensor:
          self._step_k = k
          try: return fn(*vs)
          finally: self._step_k = None
        fn = forward_seqs_k
      self._seq_jits[key] = TinyJit(fn, capture_first=True, scratch=self._scratch)
    return self._seq_jits[key]
  # (mixed steps: a prompt's chunk of SEQ_MIX_ROWS rows at most with a step of the sequences, padded to one of SEQ_MIX_SIZES, see seq_step)
  SEQ_MIX_ROWS = getenv("SEQ_MIX_ROWS", 576)
  SEQ_MIX_SIZES = tuple(int(x) for x in str(getenv("SEQ_MIX_SIZES", "")).split(",") if x)
  def mix_size(self, n:int, count:int, start:int) -> int|None:
    """the size of a mixed step of n sequences and a prompt of count tokens from start, None: not one (its own prefill)"""
    if count > self.SEQ_MIX_ROWS or start < unwrap(self.config.indexer).top_k or count < 1: return None
    return next((m for m in self.SEQ_MIX_SIZES if n <= m <= 2*max(n, 4)), None)
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
               prompt:tuple[int, int, int, int, float|tuple[float, ...]]|None=None, drafts:int|None=None) -> list[tuple[list[int], list[int]]]:
    """
    a step of speculative decoding of sequences sharing the caches: seqs (slot, base, mtp_pos, j, par, the k+1 tokens of the last step,
    temperature) of every one (see _spec_steps), their k+1 sampled tokens and k drafts. prompt (a mixed step, see mix_size): (slot, base, start,
    count, temperature) of a sequence whose count tokens from start (in the token buffer) are prefilled with the step (on the model's states:
    the host copies its slot's in before, out after), the token sampled after them the step's prompt_token
    drafts: the step's (default the size's, see drafts_of); a sequence's j past them (its last step drafted more): the step drafts j or more
    (its MTP catches up on j+1 tokens, see _step_drafts), y as long as the step's k+1 (the tokens past j not read). the layouts of the
    states are spec_drafts'
    """
    self._seq_init()
    n = len(seqs)
    m = next(s for s in self.SEQ_SIZES if s >= n) if prompt is None else cast(int, self.mix_size(n, prompt[3], prompt[2]))
    k = max([self.drafts_of(m) if drafts is None else drafts, *(j for _, _, _, j, *_ in seqs)])
    # (more than the size's: the fewest of the warmed steps' that are enough)
    if drafts is None: k = next((d for d in self._step_drafts(m) if d >= k), k)
    assert k <= self.config.spec_drafts, f"a step of {k} drafts, the layouts have {self.config.spec_drafts}"
    # (the sequences past n: the padding slot's at position 0 in the padding region)
    seqs = seqs + [(self.seq_slots(), self.seq_rows(), 0, 0, 1, [0]*(k+1), 0.0)] * (m - n)
    self._seq_host_in[:m*(k+1)] = array.array('i', [t for *_, y, _ in seqs for t in (list(y) + [0]*k)[:k+1]])
    self._seq_host_temp[:TEMP_W*m] = array.array('f', [v for *_, temp in seqs for v in _temp_row(temp)])
    # (the step's seed, see forward_seqs)
    self._seq_host_temp[-1] = array.array('f', bytes(random.getrandbits(8) for _ in range(4)) if not getenv("SEQ_SEED") else
                                          int(getenv("SEQ_SEED")).to_bytes(4, "little"))[0]
    W = m*(2*k+1) + (prompt is not None)
    self._seq_host_out[:W] = array.array('i', [-1] * W)
    pvals:dict[str, int] = {}
    if prompt is not None:
      psl, pb, pst, pcnt, ptemp = prompt
      self._seq_host_temp[TEMP_W*self._seq_m:TEMP_W*(self._seq_m+1)] = _temp_row(ptemp)
      pvals = {"seqp_slot": psl, "seqp_base": pb, "seqp_start": pst, "seqp_count": pcnt}
    jit, st = self._seq_jit(m, prompt is not None, k), time.perf_counter()
    if (cap:=getattr(jit, "captured", None)) is not None:
      # (captured: the variables' values as they are, no bound variables made a step)
      cap([], {**{name: v for g, (sl, b, mp, j, par, _, _) in enumerate(seqs)
                  for name, v in ((f"seq_slot{g}", sl), (f"seq_base{g}", b), (f"seq_mtp{g}", mp), (f"seq_j{g}", j), (f"seq_par{g}", par),
                                  (f"seq_kb{g}", self._kbase_of(sl, mp + j)))}, **pvals})
    else: jit(*self._seq_vars([(sl, b, mp, j, par) for sl, b, mp, j, par, _, _ in seqs], k), *self._prompt_vars(pvals))
    # (SEQ_REPLAY: every sequence's recurrent states now the ones after its step's first token)
    if SEQ_REPLAY:
      for sl, _, mp, j, _, _, _ in seqs[:n]: self._seq_kbase[sl] = mp + j + 1
    launched = time.perf_counter()
    out = self._wait_host(lambda: min(o:=self._seq_host_out[:W].tolist()) >= 0 and o or -1, f"a step of {m} sequences")
    self.step_times[0] += launched - st
    self.step_times[1] += time.perf_counter() - launched
    self.prompt_token = out[m*(2*k+1)] if prompt is not None else None
    return [(out[g*(2*k+1):g*(2*k+1)+k+1], out[g*(2*k+1)+k+1:(g+1)*(2*k+1)]) for g in range(n)]
  prompt_token:int|None = None  # (the token after a mixed step's prompt)
  def _prompt_vars(self, pv:dict[str, int]) -> list[UOp]:
    if not pv: return []
    rows = self.seq_rows()
    bounds = {"seqp_slot": self.seq_slots(), "seqp_base": rows, "seqp_start": self.max_context-1, "seqp_count": self.SEQ_MIX_ROWS}
    return [UOp.variable(name, 1 if name == "seqp_count" else 0, bounds[name]).bind(v) for name, v in pv.items()]

  def _init_states(self):
    """the states and caches of the blocks, realized (the jits capture on their first call: not the one time setup)"""
    if getattr(self, "_states_ready", False): return
    devs = self.token_embd.weight.device
    if devs is not None and amd_custom_kernels_supported(devs): tables_ready(devs)
    x = Tensor.empty(1, 1, self.config.dim, device=devs)
    holders:list = []
    if self._aliased():
      # (the regions of the slots and the model's cache: as many as both need)
      n, S = self.batch_slots(), self.batch_ctx()
      for b in self.blk + ([self.mtp] if hasattr(self, "mtp") else []):
        if isinstance(b, MLATransformerBlock):
          b._regions = (n, S, max(n + SLOT_SPAN, -(-self.max_context // S)))
          if b.indexer is not None: b.indexer._regions = b._regions
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
    # (a context the qwen sparse attention's top_k covers: the jits of all the keys, see _qsa_dense)
    dense = self._dense_ctx(start_pos, T)
    # (a sequence's region from a variable base: jits of their own, see seq_prefill)
    jits = self.__dict__.setdefault("jits" + ("_dense" if dense else "") + ("_greedy" if temperature is None else "") +
                                    ("_seq" if isinstance(base, UOp) else ""), {})
    if (jit:=jits.get(key)) is None:
      # the states and caches are set up before: a jit captures on its first call
      self._init_states()
      jit = jits[key] = self._new_jit(key)
    _qsa_dense[0] = dense
    try:
      if not isinstance(jit, list): return jit(tokens, start_pos, temperature, base)[0]
      # (a prefill in parts: one of them (part, see seq_warmup), else all in order)
      for p, j in enumerate(jit):
        if part is None or p == part: out = j(tokens, start_pos, temperature, base)[0]
      return out
    finally: _qsa_dense[0] = False

  def _new_jit(self, rows:int) -> Any:
    """the jit of a call of rows (a list of the parts' jits: PREFILL_PARTS)"""
    if len(parts:=self.prefill_parts(rows)) == 1: return TinyJit(self._forward_effects, capture_first=True, scratch=self._scratch)
    self._carry(rows)
    return [TinyJit(functools.partial(self._forward_effects, part=p), capture_first=True, scratch=self._scratch) for p in range(len(parts))]

  def _dense_ctx(self, start_pos:int|UOp, T:int|UOp) -> bool:
    """the context of a call (start_pos and T: bound variables or ints) is covered by the qwen sparse attention's top_k: it sees every key"""
    if not (self.config.qsa and (idx:=self.config.indexer) is not None and self.max_context > idx.top_k and getenv("QSA_DENSE", 1)): return False
    def val(u:int|UOp) -> int: return u if isinstance(u, int) else int(u.unbind()[1])
    return val(start_pos) + val(T) <= idx.top_k

  def _forward_effects(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor|None, base:int|UOp=0, part:int|None=None) -> tuple[Tensor, ...]:
    # forward and the tensors realized with its output for their effects (e.g. the caches of the MTP layer)
    self._effects = []
    out = self.forward(tokens, start_pos, temperature, base, part)
    effects, self._effects = self._effects, None
    return (out, *effects)

  @staticmethod
  def from_gguf(gguf:Tensor|str|pathlib.Path, max_context:int|None=None,
                realize=bool(getenv("REALIZE", 0)), shard:int=1, defer_load:bool=False) -> tuple[Transformer, dict]:
    """defer_load: ignored, the weights are read before the warmup (no loading in the background)"""
    defer_load = False
    # TODO: remove the need for copy to default device
    kv, entries = gguf_parse(gguf.to(None).realize() if isinstance(gguf, Tensor) else gguf)
    _lap("parsed")
    # deferred: the files are read ahead from now (the disk works while the model is made, the loader finds them in memory)
    # (not the n-gram table: read to host memory by its loader, see _ple_dev_init)
    if defer_load: _read_ahead(entries, skip=("per_layer_token_embd.weight",))
    arch = kv['general.architecture']
    n_heads, n_kv_heads = kv[f'{arch}.attention.head_count'], kv[f'{arch}.attention.head_count_kv']
    assert shard >= 1, f"shard must be at least 1, got {shard}"
    # the next token prediction (MTP) layers: the first is loaded with MTP=1 (model.mtp), the others aren't used
    num_blocks = kv[f'{arch}.block_count'] - kv.get(f'{arch}.nextn_predict_layers', 0)
    mtp = bool(getenv("MTP", int(arch == "glm5-next"))) and kv.get(f"{arch}.nextn_predict_layers", 0) > 0
    # (NUM_BLOCKS: only the first blocks for small-kernel profiling, no MTP; unset keeps the exact deployed model)
    # (NUM_BLOCKS_MTP=1: the MTP layer kept, as the block after the first NUM_BLOCKS: a decode step's rows (drafts) as deployed)
    if (nb:=getenv("NUM_BLOCKS", 0)) and nb < num_blocks:
      if mtp and getenv("NUM_BLOCKS_MTP", 0):
        entries = {re.sub(rf"^blk\.{num_blocks}\.", f"blk.{nb}.", k):v for k,v in entries.items()
                   if not ((m:=re.match(r"blk\.(\d+)\.", k)) and nb <= int(m.group(1)) < num_blocks)}
        num_blocks = nb
      else: num_blocks, mtp = nb, False
    # (qwen4exp: the MTP layer in a gguf of its own (its block after the model's), found next to the model's)
    if arch == 'qwen4exp' and getenv("MTP", 1) and not isinstance(gguf, Tensor) and (mf:=_qwen_mtp_file(pathlib.Path(gguf))) is not None:
      mkv, mentries = gguf_parse(mf)
      mb = mkv[f'{arch}.block_count'] - 1
      entries.update({re.sub(rf"^blk\.{mb}\.", f"blk.{num_blocks}.", k): v for k, v in mentries.items() if k.startswith(f"blk.{mb}.")})
      mtp = True
    entries = {k:v for k,v in entries.items() if not ((m:=re.match(r"blk\.(\d+)\.", k)) and int(m.group(1)) >= num_blocks + mtp)}
    # (qwen4exp: the n-gram table stays in host memory, the gated deltanets in the kda layout, a KV head a device)
    ple_entry = entries.pop('per_layer_token_embd.weight', None)
    rows_map:dict[str, tuple[int, list[tuple[int, int]]]] = {}
    if arch == 'qwen4exp': entries, n_kv_heads, rows_map = _qwen4exp_entries(kv, entries, shard)
    shard_map:dict[str, int] = {}
    units_map:dict[str, tuple[int, list[list[int]]]] = {}
    repeats:dict[str, int] = {}
    pads:dict[str, int] = {}
    padded_heads, sizes, expert_slices, expert_pad = n_heads, {}, 0, 0
    if shard > 1:
      # MLA and kda attention split all their tensors by the n_heads heads: those are zero padded to a multiple of the devices
      pad_heads = bool(kv.get(f'{arch}.attention.kv_lora_rank')) or arch in ('kimi-linear', 'glm5-next')
      if not pad_heads and not kv.get(f'{arch}.attention.kv_lora_rank') and n_kv_heads < shard and shard % n_kv_heads == 0 and getenv("KV_REPEAT", 1):
        # too few KV heads to split: repeat each one so every device holds one (master's gguf_shard repeats)
        repeats = {k: shard // n_kv_heads for k in entries if k.split('.', 2)[-1] in ('attn_k.weight', 'attn_v.weight', 'attn_k.bias', 'attn_v.bias')}
        n_kv_heads = shard
      if not pad_heads: assert n_kv_heads % shard == 0, f"tensor parallel needs the attention heads to split over {shard} devices"
      # shard heads, FFNs and routed experts while replicating latent projections, KV cache and the small kda projections
      head_rules = {**{w: 0 for w in ('attn_q.weight', 'attn_k.weight', 'attn_v.weight', 'attn_q_b.weight', 'attn_k_b.weight', 'attn_v_b.weight')},
                    'attn_output.weight': 1}
      # the kda layers split by heads too (the gated deltanet of qwen3.5 is copied to every device)
      if arch in ('kimi-linear', 'glm5-next', 'qwen4exp'): head_rules.update({w: 0 for w in ('ssm_conv1d_q.weight', 'ssm_conv1d_k.weight',
        'ssm_conv1d_v.weight', 'ssm_dt.bias', 'ssm_a', 'ssm_beta.weight', 'ssm_f_b.weight', 'ssm_g_b.weight', 'ssm_alpha.weight',
        'attn_gate.weight')})
      if arch == 'qwen4exp': head_rules['ssm_out.weight'] = 1
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
      # tinyr9: five 128-wide slices per Qwen expert reduce routing and down-projection intermediates.
      if arch == 'qwen4exp' and shard in (4, 6):
        down_block = math.lcm(down_block, 128)
        assert exp_hidden % down_block == 0
      exp_names = ('ffn_gate_exps.weight', 'ffn_up_exps.weight', 'ffn_down_exps.weight')
      if exp_hidden % shard == 0 and (exp_hidden // shard) % down_block == 0:
        ffn_rules.update({'ffn_gate_exps.weight': 1, 'ffn_up_exps.weight': 1, 'ffn_down_exps.weight': 2})
      elif exp_count and exp_hidden % down_block == 0 and getenv("EXPERT_SLICES", 1) and \
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
            entries[g][2] == entries[u][2] and file_of(entries[g][0]) == file_of(entries[u][0]) and getenv("FUSE_GATE_UP", 1)}
    # a token embedding copied to every device (a vocab that doesn't split) stays in host memory: a call looks up a few of its rows
    host_embd = None
    if shard > 1 and 'token_embd.weight' not in shard_map and 'output.weight' in entries and amd_custom_kernels_supported(devices) and \
       getenv("HOST_EMBEDDING", 1):
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
    state_dict = {k:v.cast('float16') if getenv("HALF", 1) else v for k,v in state_dict.items()}

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
    if arch == 'qwen4exp':  # QSA: the indexer pools compress_ratio keys
      indexer = IndexerConfig(kv[f'{arch}.attention.indexer.head_count'], kv[f'{arch}.attention.indexer.key_length'],
                              kv[f'{arch}.attention.indexer.top_k'], max(kv[f'{arch}.attention.compress_ratios']),
                              kv[f'{arch}.attention.layer_norm_rms_epsilon'], kv[f'{arch}.rope.dimension_count'], False)
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
    elif arch == 'qwen4exp':
      # (the kda layout: the q and k heads repeated for the v heads at load, see _qwen4exp_entries)
      hv = kv[f'{arch}.ssm.time_step_rank']
      ssm = SSMConfig(kv[f'{arch}.ssm.conv_kernel'], kv[f'{arch}.ssm.state_size'], hv, hv, kv[f'{arch}.ssm.inner_size'], kda=True, head_decay=True)
      ssm_layers = tuple((i+1) % kv[f'{arch}.full_attention_interval'] != 0 for i in range(num_blocks))
      for i, is_ssm in enumerate(ssm_layers):
        if is_ssm: state_dict[f"blk.{i}.ssm_a"] = state_dict[f"blk.{i}.ssm_a"].reshape(hv, 1)
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
    glm_seq = getenv("GLM_SEQ", int(arch == "glm5-next" and mtp and shard > 1))
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
      norm_topk_prob=kv.get(f'{arch}.expert_weights_norm', arch in ('qwen3moe', 'qwen35moe', 'kimi-linear', 'qwen4exp')),
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
      routed_scaling_factor=kv.get(f'{arch}.expert_weights_scale', 1.0), attn_output_gate=arch in ('qwen35', 'qwen35moe', 'qwen4exp'),
      ssm=ssm,
      ssm_layers=ssm_layers,
      qkv_bias='blk.0.attn_q.bias' in state_dict,
      expert_bias=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.exp_probs_b.bias" in state_dict, expert_slices=expert_slices,
      expert_proj_bias='blk.0.ffn_gate_exps.bias' in state_dict, attn_output_bias='blk.0.attn_output.bias' in state_dict,
      swiglu_alpha=1.702 if swiglu_oai else 1.0, swiglu_limit=7.0 if swiglu_oai else kv.get(f'{arch}.swiglu_clamp_exp', [None])[0],
      swiglu_up_bias=1.0 if swiglu_oai else 0.0, attn_sinks='blk.0.attn_sinks.weight' in state_dict,
      sliding_window=kv.get(f'{arch}.attention.sliding_window', 0),
      sliding_window_pattern=kv.get(f'{arch}.attention.sliding_window_pattern', 2 if arch == 'gpt-oss' else 0), hc=hc, indexer=indexer, mtp=mtp,
      spec_drafts=getenv("SPEC_DRAFTS", 4 if arch == "qwen4exp" else 2) if mtp else 0, ple=ple, qsa=arch == 'qwen4exp', expert_pad=expert_pad,
      # (SEQ_ROWS: the rows of the caches of the sequences sharing them, past a context: more conversations kept)
      # (GLM-5.3-Flash on several devices: GLM_SEQ on by default (the server's SeqEngine: conversations keep their caches, a host stash), its
      # rows two contexts (262144: 524288 rows, with 16 slots and SEQ_STASH_GB=4 in a 6x24 GB C5's memory; C6/C7 have more))
      cache_rows=round_up(getenv("SEQ_ROWS", 2*max_context if glm_seq else 0), REGION_ALIGN) if arch == 'qwen4exp' or glm_seq else 0)
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
         amd_custom_kernels_supported(g.weight.device) and getenv("FUSE_GATE_UP", 1):
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
            getattr(l, "bias", None) is None and not hasattr(l, "weight_t") and amd_custom_kernels_supported(l.weight.device) and getenv("SKINNY", 1)]
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
    return isinstance(devs, tuple) and self.config.hc is not None and rows % (16*len(devs)) == 0 and bool(getenv("SEQ_PARALLEL", 1))

  def prefill_chunks(self, chunk_size:int|None=None) -> tuple[int, ...]:
    """the prefill chunk sizes, ascending: the custom kernels compute chunks padded to a static size, a piece of a prompt runs the smallest
    one it fits (its own jit), large chunks are efficient. sequence parallel: whole tiles of rows on every device"""
    fast = amd_custom_kernels_supported(self.token_embd.weight.device)
    # Qwen's gated streams keep the 2K chunk at long contexts too: larger chunks substantially increase prefill workspace.
    default_chunk = 2048 if self.config.qsa and self.config.hc is not None else (4096 if self.max_context > 65536 else 2048)
    chunk_size = chunk_size or min(getenv("PREFILL_CHUNK", default_chunk if fast else 32), self.max_context)
    if self.has_recurrent_block and not fast: chunk_size = 1
    sizes = [chunk_size] + [min(chunk_size, n) for n in ((getenv("PREFILL_TINY", 192), getenv("PREFILL_SMALL", 512), getenv("PREFILL_MID", 2048),
                                                          4096) if fast else ())]
    # (PREFILL_SIZES: more sizes, less padding of a prompt's last piece (the turns of agents: a few hundred tokens), a jit each)
    if fast and (more:=getenv("PREFILL_SIZES", "")): sizes += [min(chunk_size, int(n)) for n in str(more).split(",")]
    if fast and (devs:=self.token_embd.weight.device) and isinstance(devs, tuple) and self.config.hc is not None and getenv("SEQ_PARALLEL", 1):
      # whole tiles of 16 rows on every device, the kernels of the whole chunk on tiles of 64
      tile = math.lcm(16*len(devs), 64)
      def tiled(n:int) -> int: return min(max(tile, round(n / tile) * tile), self.max_context // tile * tile or tile)
      # (PREFILL_CHUNK_UP, default on: the biggest chunk whole tiles up, not the nearest (6 devices: 4096 -> 4224, not 4032): a 4k
      # prompt (4096 tokens and its template) one chunk, not 4032 and a padded 192. other chunk ends: other greedy tokens (4k, 262k);
      # quality bar PASS 2026-11-10 on C7 and C5 (8,064-position NLL inside the chunk-noise floor, KL as the noise pair's))
      up = [min(-(-chunk_size // tile) * tile, self.max_context // tile * tile or tile)] \
        if getenv("PREFILL_CHUNK_UP", 1) and not self.config.qsa else []
      sizes = [tiled(n) for n in sizes] + up
    return tuple(sorted(set(sizes)))

  def warmup(self):
    """compile and capture every jit a request uses (they capture on their first call): a prompt of each prefill chunk size, decoding"""
    # (GLM_SEQ: the sequences sharing the caches' jits)
    if self.config.cache_rows and not self.config.qsa and self.seqs_supported(): return self.seq_warmup()
    sizes = self.prefill_chunks()
    # (a sequence sharded cache (a long context): one after the other, the weights read before the caches are made (their memory is the device's))
    if getenv("WARMUP_PARALLEL", 0 if seq_sharded(self.token_embd.weight.device) else 1): self._capture_parallel(sizes)
    self._finish_load()
    # (mixed steps first, the most rows first: their scratch memory can be more than a prefill's (the slots' memory made for them))
    if (land:=self.mix_landing()):
      self._alloc_slots()
      k = self.config.spec_drafts
      for m, G, P in sorted(itertools.product(self.mix_sizes(), range(1, land+1), self.MIX_PS), key=lambda x: -x[0]*(k+1)-x[1]*x[2]):
        if not warmup_wanted(f"mixed{m}_{G}_{P}"): continue
        self.batch_mixed_step([([0]*(k+1), 1, 0, 1, 0.0)] * m, [([0], self.decode_slots() + g, 0.0, 0) for g in range(G)], P=P)
    # the big chunk first: its scratch memory is the most, the other jits share it. a prompt one past a size runs the next one (1: the smallest)
    for n in sorted({1, *(s + 1 for s in sizes[:-1])}, reverse=True):
      # (WARMUP_JITS names: prefill<chunk> (the chunk a prompt of n runs: n=1 the smallest, its decoding too), step<m>, mixed<m>_<G>_<P>,
      # bpf<P>_<G>, slots (the copies of the slots), host (the staging rows'); the others captured on their first call)
      if not warmup_wanted(f"prefill{next((s for s in sizes if s >= n), sizes[-1])}"): continue
      self._cached_tokens, self._ckpt_tokens = [], []
      list(zip(range(4), self.generate([0] * min(n, self.max_context - 8))))
    if self._ckpt_tokens: self._checkpoint(self._ckpt_tokens, save=False)  # and a checkpoint's restore
    # batched decoding: the slots' memory, a copy into a slot, a move between two and a step of every size (their jits linked)
    self._alloc_slots()
    slots = self.batch_slots() if warmup_wanted("slots") else 0
    if slots: self.slot_store(0, 1)
    if slots > 1: self.slot_move(1, 0)
    # (and the copies of a sequence spanning more slots)
    for n in range(2, self.max_span()+1 if slots else 0):
      self.slot_store(0, (n-1)*self.batch_ctx() + 1), self.slot_load(0, (n-1)*self.batch_ctx() + 1)
    for m in self.batch_sizes():
      if m > self.batch_slots() or not warmup_wanted(f"step{m}"): continue
      if self.config.spec_drafts: self.batch_spec_step([([0]*(self.config.spec_drafts+1), 1, 0, 1, 0.0)] * m)
      else: self.batch_step([(0, 1, 0.0)] * m)
    if self.batch_prefills():
      for P in self.BPF_PS:
        for G in self.bpf_sizes(P):
          if warmup_wanted(f"bpf{P}_{G}"): self.batch_prefill([([0], 0, 0.0)] * G, P=P)
    if (self.batch_prefills() or land) and slots: self.ckpt_from_slot(0, 1)  # (a prompt prefilled in a slot: its checkpoint, see the server)
    # (the staging rows of host_save and host_load: their copies captured)
    if warmup_wanted("host"): self._host_rows(1, True), self._host_rows(1, False)
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
      # (and the one of a context the qwen sparse attention's top_k covers, see _qsa_dense)
      if self._dense_ctx(0, 1):
        self._spec_jit_dense_greedy = TinyJit(self.forward_spec, capture_first=True, scratch=self._scratch)
        setattr(self._spec_jit_dense_greedy, "warmup_name", "spec_dense")
        calls.append((self._spec_jit_dense_greedy, functools.partial(self._spec_call, 0, 0, 1, None)))
        if (top:=unwrap(self.config.indexer).top_k) + 2*self.config.spec_drafts + 2 < self.max_context:
          calls.append((self._spec_jit_greedy, functools.partial(self._spec_call, top, 0, 1, None)))
      else: calls.append((self._spec_jit_greedy, functools.partial(self._spec_call, 0, 0, 1, None)))
    # batched decoding: a jit of every step size (the slots made here: the processes don't allocate)
    if (slots:=self.batch_slots()):
      # (their memory isn't made yet, the jits are linked on their first call: after the weights are read, the memory loading used is free)
      self._init_slots(alloc=False)
      def step(m:int) -> Callable[[], Any]:
        if self.config.spec_drafts: return lambda: self._bspec_jit(m)(*self._bspec_vars([(0, 0, 1)]*m))
        return lambda: self._batch_jit(m)(*self._batch_positions([0]*m))
      for m in self.batch_sizes():
        jit = self._bspec_jit(m) if self.config.spec_drafts else self._batch_jit(m)
        if m <= slots: calls.append((jit, step(m)))
        jit.link_later = True
        setattr(jit, "warmup_name", f"step{m}")
      # (and the copies into a slot, back from one and between two)
      self._slot_jits()
      calls += [(self._slot_jit, lambda: self._slot_jit(*self._slot_store_vars(0, 1, 0))),
                (self._slot_load_jit, lambda: self._slot_load_jit(*self._slot_store_vars(0, 1, 0)))]
      if slots > 1: calls += [(self._slot_move_jit, lambda: self._slot_move_jit(*self._slot_move_vars(1, 0)))]
      for j in (self._slot_jit, self._slot_move_jit, self._slot_load_jit):
        setattr(j, "link_later", True)
        setattr(j, "warmup_name", "slots")
      # (and the mixed steps)
      for m, G, P in itertools.product(self.mix_sizes(), range(1, self.mix_landing()+1), self.MIX_PS):
        mix = self._mix_jit(m, G, P)
        calls.append((mix, functools.partial(mix, *self._bspec_vars([(0, 0, 1)]*m),
                                             *self._bpf_vars([1]*G, [self.decode_slots() + g for g in range(G)], [0]*G))))
        mix.link_later = True
        setattr(mix, "warmup_name", f"mixed{m}_{G}_{P}")
      # (and the prefills of several prompts)
      if self.batch_prefills():
        for P in self.BPF_PS:
          for G in self.bpf_sizes(P):
            bpf = self._bpf_jit(G, P)
            calls.append((bpf, functools.partial(bpf, *self._bpf_vars([1]*G, [0]*G, [0]*G))))
            bpf.link_later = True
            setattr(bpf, "warmup_name", f"bpf{P}_{G}")
    # the weights are read (and what's computed from them) while the jits compile, and a checkpoint is saved and restored (its kernels)
    def meanwhile(take):
      # (the jits of the computations left run after: they're compiled meanwhile)
      while self._pending_load and not isinstance(self._pending_load[0], _TinyJit): self._pending_load.pop(0)()
      if self.has_recurrent_block and getenv("CHECKPOINT", 1): self._checkpoint([0], save=True), self._checkpoint([0], save=False)
    # the work left of loading as soon as its jits are taken, then the slots' memory (the memory loading used is free): the jits of batched
    # decoding that come after are linked as they come (while the others compile)
    loads, load_jits = {id(j) for j in self._pending_load if isinstance(j, _TinyJit)}, [j for j in self._pending_load if isinstance(j, _TinyJit)]
    def taken(jit:_TinyJit):
      loads.discard(id(jit))
      if loads or not slots or getattr(self, "_slots_ready", False) or not getenv("SLOTS_EARLY", 1): return
      self._finish_load()
      # (run once: their graphs and the buffers they read are freed)
      for j in load_jits: j.reset()
      load_jits.clear()
      gc.collect()
      self._alloc_slots()
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
      if self.has_recurrent_block and start_pos == prompt_len and getenv("CHECKPOINT", 1): self._checkpoint(tokens, save=True)
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

# KPROF=n (with PROFILE=1): every n batched decoding steps the GPU kernel times of them printed (a device's, per step), the server's real steps
if (KPROF:=getenv("KPROF", 0)):
  def _kprof_wrap(f:Callable) -> Callable:
    def w(self, *args, **kwargs):
      ret = f(self, *args, **kwargs)
      import extra.kprof as kp
      devs = to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device))
      c = self.__dict__.setdefault("_kprof", {"steps": 0, "st": time.perf_counter(), "names": collections.Counter()})
      c["steps"] += 1
      c["names"][f.__name__] += 1
      if c["steps"] >= KPROF:
        t, dt = kp.times(*devs), time.perf_counter() - c["st"]
        tot = sum(us for _, us in t.values()) / len(devs) / c["steps"]
        print(f"KPROF {c['steps']} steps {dict(c['names'])} in {dt*1e3:.0f} ms: {dt/c['steps']*1e3:.2f} ms/step, GPU {tot/1e3:.3f} ms/step/device")
        for n, (cnt, us) in sorted(t.items(), key=lambda x: -x[1][1])[:getenv("KPROF_TOP", 30)]:
          print(f"KPROF  {us/len(devs)/c['steps']:9.1f} us/step  {cnt/len(devs)/c['steps']:6.1f}/step  {n}")
        kp.reset(*devs)
        c.update(steps=0, st=time.perf_counter(), names=collections.Counter())
      return ret
    return w
  for _n in ("batch_step", "batch_spec_step", "batch_mixed_step"): setattr(Transformer, _n, _kprof_wrap(getattr(Transformer, _n)))
