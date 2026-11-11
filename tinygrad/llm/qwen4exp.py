from __future__ import annotations
import array, contextlib, enum, functools, itertools, math, os, pathlib, random, re, time
from dataclasses import dataclass, replace
from typing import Any, Callable, cast
from tinygrad import Tensor, nn, UOp, TinyJit, getenv, function, dtypes, Device
from tinygrad.engine.realize import warm_worker_pool
from tinygrad.engine.jit import capture_parallel, warmup_select, run_as_they_come, _TinyJit
from tinygrad.llm.kernels import amd_qwen
from tinygrad.llm.kernels.amd_qwen import rows_shifted, rows_pick
from tinygrad.llm.kernels.amd_qwen import EXTRA_QUEUES, _wmma_rdna4, ring_errors, init_deferred, _init_buffers
from tinygrad.llm.kernels.amd_qwen import Linear, gated_delta_prefill, amd_custom_kernels_supported, find_packed, moe_linear, q8_linear, \
  q8_quantize_cache_clear, moe_route, moe_grouped_linear, partial_sum, fuse_gate_up, kda_decode, pool_index_rows, slot_scatter, \
  slot_gather, moe_combine, token_sharded, shard_rows, reduce_scatter, all_gather, all_gather_q8, sharded_argmax, skinny_matmul, kda_conv, \
  kda_replay, tables_ready, MOE_TILE, I8_WMMA_QUANTS, ple_gather, ghc_pre, ghc_post, ghc_prefill_norm, ghc_prefill_mix, moe_softmax_router, \
  Q8_0, IQ3_S, q8_embed, weighted_sum, ghc_decode_mix, rows_gather, rows_scatter, seq_scores, seq_index, kv_q8_pack, kv_q4_pack, \
  qsa_decode_attention, FLAG_LANE, DEC_LANE, flag_allreduce, ring_mailbox, ring_queue_of, RING_QUEUE_ON, mb_queue, state_token, f32t_gemm, \
  few_out_rows, index_scores, fwht, merged_linear, q8_linear_parts, after_kernel, _argmax_parts, _device_index, _unbind, f16t_gemm, nucleus, \
  picks_local, pool_window, seq_index_split
from tinygrad.llm.gguf import gguf_parse, ggml_data_to_tensor, _GGML_QUANT
from tinygrad.llm.direct_read import DirectFile, read_many, close_rings
from tinygrad.llm import gguf_load
from tinygrad.llm.gguf_load import gguf_shard_units, _disk_bytes, _disk_location
from tinygrad.uop.ops import Ops, AxisType, KernelInfo
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
    # (not realized: made from the read weights, see _prepare)
    self.weight = fuse_gate_up(self.weight, up.weight, self.num_experts, rows).contiguous()
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

def _qwen4exp_entries(kv:dict, entries:dict[str, tuple[Tensor, tuple[int, ...], int]], shard:int, kv_split:int=1) -> tuple[dict, int, dict]:
  """
  qwen4exp's tensors as the model's: the KV heads repeated so every device holds one (shard a multiple of them), the gated deltanets in the kda
  layout (attn_qkv and its conv split in q, k and v, the q and k heads repeated for the v heads). on devices, as the rows of every device of the
  file's tensors (gguf_shard's rows: read straight from the file): a device's v heads use a run of the k heads. kv_split: the query heads of a
  KV head's group (and their columns of the output projection) on every device of it (see TransformerConfig.kv_split). returns the entries,
  the KV heads and the rows
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
    if kv_split > 1:
      for name in [k for k in entries if k.endswith('attn_q.weight')]:
        data, shape, typ = entries[name]
        g = shape[0] // n_kv
        entries[name], rows[name] = (data, (shard * g, *shape[1:]), typ), (shape[0], [(d * n_kv // shard * g, g) for d in range(shard)])
      for name in [k for k in entries if k.endswith('attn_output.weight')]:
        # (its columns of a group repeated: every row's bytes of a group's columns (whole quant blocks) kv_split times)
        data, shape, typ = entries[name]
        t = data.to("CPU").reshape(shape[0], n_kv, -1).repeat_interleave(kv_split, 1)
        entries[name] = (t.flatten(), (shape[0], shape[1] * kv_split), typ)
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
    # (the output gate's weight after v: q, k, v and the gate adjacent on every device, batched decoding's one kernel of them, see merged_linear)
    if blk + 'attn_gate.weight' in entries: entries[blk + 'attn_gate.weight'] = entries.pop(blk + 'attn_gate.weight')
  return entries, n_kv, rows

def expert_slice_table(units:list[int], experts:int, slices:int) -> list[list[int]]:
  """(experts, w): the local index of every slice a device holds of an expert, -1 padded"""
  table:list[list[int]] = [[] for _ in range(experts)]
  for i, u in enumerate(units): table[u // slices].append(i)
  w = max(len(t) for t in table)
  return [t + [-1]*(w-len(t)) for t in table]


def gated_activation(gate:Tensor, up:Tensor, config:TransformerConfig|None=None, *, alpha:float=1.0, limit:float|None=None,
                     up_bias:float=0.0) -> Tensor:
  # (config: its swiglu alpha, limit and up bias, else the keywords')
  if config is not None: alpha, limit, up_bias = config.swiglu_alpha, config.swiglu_limit, config.swiglu_up_bias
  if limit is not None and limit > 0: gate, up = gate.clamp(max_=limit), up.clamp(-limit, limit)
  # TODO: remove the need for this contiguous (gate and up from the custom kernels: one kernel)
  sig = (gate * alpha).sigmoid()
  return gate * (sig if amd_custom_kernels_supported(gate.device) else sig.contiguous()) * (up + up_bias)


def pairwise_topk(x: Tensor, k: int) -> tuple[Tensor, Tensor]:
  n = x.shape[-1]
  vals = Tensor.arange(n).reshape(1,1,n).cast(x.dtype).expand(x.shape)
  cmp = (x.unsqueeze(-1) > x.unsqueeze(-2)) | ((x.unsqueeze(-1) == x.unsqueeze(-2)) & \
    (Tensor.arange(n).reshape(1,1,n,1) < Tensor.arange(n).reshape(1,1,1,n)))
  # the element of rank r (the count of the elements below it) is the r-th smallest: the top k are the ranks n-k..n-1, in that order
  picked = cmp.sum(axis=-1).unsqueeze(-2) == Tensor.arange(n-k, n).reshape(k, 1)
  return picked.where(x.unsqueeze(-2), 0).sum(-1), picked.where(vals.unsqueeze(-2), 0).sum(-1).cast('int32')


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
  # the devices that split the rows of a KV head (its positions by pools, see QSABlock): the heads of the queries of its group on every one
  # of them (n_heads its group's times kv_split), each attends its keys, the outputs combined by their log sum exps
  kv_split: int = 1
  # the devices the indexer's pooled keys are split over (pool p on device p%idx_split, its row p//idx_split there, see QSAIndexer): a device
  # scores its pools, the top ones of every device exchanged
  idx_split: int = 1
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
# QTF: batched decoding teacher forced (a quality tool's, extra/qdecq.py): the drafts and the step's tokens forced from the host, the rows'
# log probs written to the host
QTF = getenv("QTF", 0)
# SEQ_MIX_TOKENS: a short prompt suffix forced through the ordinary decode rows, no standalone prefill (0: off)
SEQ_MIX_TOKENS = getenv("SEQ_MIX_TOKENS", 0)
# AG_REAL: batched decoding's q8 all-gather packs a device's real rows from the padded mix, not from their copy (tr8 B=8 +0.8%: off)
AG_REAL = getenv("AG_REAL", 0)
# HC_REAL: batched decoding's hyper-connections read a device's real rows only and the mix stores them (no copy of them), its input not
# materialized in float (2 kernels less a sublayer): the mix's epilogue and the activation's kernel round differently (FMA, rcp: ulps)
HC_REAL = getenv("HC_REAL", 1)
GDN_MERGED, HC_NORM_REAL = getenv("GDN_MERGED", 0), getenv("HC_NORM_REAL", 1)
# QSA_ROT_FWHT: QSA_KV_ROT's rotations (keys, values, queries, the output back) by a fast Walsh-Hadamard kernel (fwht), not 256x256 matmuls
QSA_ROT_FWHT = getenv("QSA_ROT_FWHT", 1)
# HC_DEC: batched decoding's hyper-connections of a device's few real rows (1-HC_DEC_ROWS (<=16), split) by the decode kernels (ghc_pre,
# the decode linear, ghc_decode_mix/ghc_post: the B=1 path's) instead of the prefill ones on padded tiles of 32 (numerics: the B=1 path's,
# q8 activations). tr8 IQ4 with the default numerics (HC_REAL): B=4 (2 rows a device) 16.59 -> 16.24 ms, B=8 (4) 0%: HC_DEC_ROWS=2.
# extra/qdecq.py vs HC_DEC=0: G=4 NLL -0.008 KL 0.021 top-1 95.6%, G=8 NLL +0.014 KL 0.026 top-1 94.4%)
HC_DEC, HC_DEC_ROWS = getenv("HC_DEC", 1), getenv("HC_DEC_ROWS", 2)
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
MOE_DECODE_ROWS = 96
# batched decoding: rows of experts decoded (in bundles) up to this many, grouped (the prefill kernels) past it
MOE_BATCH_ROWS = 512
def _skinny(lin:Linear, x:Tensor) -> Tensor|None:
  """lin(x) by skinny_matmul (few outputs, few rows) if its weight is stored for it (see Transformer._finish_load), else None"""
  if (wt:=getattr(lin, "weight_t", None)) is None or not isinstance(x.shape[-2], int): return None
  # (rows split over the devices (a prefill's): a few a device)
  if token_sharded(x) and int(x.uop.shard_shape[-2]) > 512: return None
  return skinny_matmul(x, wt)

def _decode_rows() -> int: return MOE_BATCH_ROWS if _batch_rows[0] is not None else MOE_DECODE_ROWS

# batched decoding (Transformer.forward_batch): the rows are groups of a sequence each, group g (its rows the tokens at positions pos.. of the
# sequence) in the state slot g: (pos, rows) of every group
_batch_rows:list[list[tuple[UOp, int]]|None] = [None]


# a sequence slot's recurrent state is one (after the first token of its last step) and the inputs of the tokens after it, replayed by the next
# step (see kda_decode): a step writes one state, not k+1 (their memory and their writes)
def replay_q(k:int) -> int: return 2*(k+1)
def slot_states(k:int) -> tuple[int, int]:
  """
  the S of a slot's recurrent states and of its conv states (the state of a position p is p%S, see kda_decode): a step of k+1 tokens reads
  one and writes k+1, the recurrent ones can overwrite the one read (SLOT_STATES_TIGHT: a quarter less memory, more slots)
  """
  return (k + 1 if k else k + 2), k + 2


def _slot_pool(n:int, like:Tensor, shape:tuple[int|UOp, ...]|None=None, dtype:DType|None=None, host:bool=False,
               views:int|None=None) -> tuple[Tensor, list[Tensor]]:
  """
  n slots of a state like `like` (of shape, or like's, of dtype or like's) in one buffer of every device, made but not allocated (a jit
  captured before its memory is holds it): the pool (n*shape[0], ...) and the n slots, views of it (buffers of their own: a kernel writes one
  in place). host: in host memory (every device its part), the devices read and write it directly. views: only the first that many slots
  (the kernels index the pool: a view a slot is python before the start-up's forks, thousands of them, not used)
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
  return tensor(base, n), [tensor(UOp.from_buffer(view(i), dtype, devs), 1) for i in range(n if views is None else min(views, n))]


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
      parts = b.bufs if isinstance(b, MultiBuffer) else [b]
      # (ZERO_ALLOC_LATE: deferred zeroing allocates them later too: when a jit is linked or the zeroing runs (in the start-up's window))
      if not (_zero_later and ZERO_ALLOC_LATE):
        for x in parts: x.ensure_allocated()
      if _zero_later: _zero_later[-1].extend(parts)
      else:
        for x in parts: _zero_buffer(x)

def _drop_jit(j:_TinyJit):
  """a jit never called again (a derivation of weights): its capture and its function's tensors dropped"""
  j.captured, j.fxn = None, None
  j.__dict__.pop("_linears", None)

def _drop_apart(final=True):
  """
  LOAD_APART: the buffers of the file's tensors the weights are derived from, freed (they are once nothing holds them: the derivations run).
  a line of the bytes freed and held (LOAD_APART=2: who holds them); not final: tried again later if held
  """
  if not (refs:=gguf_load.apart_buffers): return
  import gc
  st = time.perf_counter()
  nbytes = sum(b.nbytes for r in refs if (b:=r()) is not None)
  if any(r() is not None for r in refs): gc.collect()
  held = [b for r in refs if (b:=r()) is not None]
  print(f"LOAD_APART: {(nbytes - sum(b.nbytes for b in held))/1e6:.1f} MB freed, {sum(b.nbytes for b in held)/1e6:.1f} MB held "
        f"({time.perf_counter()-st:.3f}s{'' if final else ', tried again after the warm-up'})")
  if held and getenv("LOAD_APART") >= 2:
    # (a chain of referrers of a buffer held: the first one not a frame each)
    import types
    def what(o) -> str:
      if isinstance(o, dict): return f"dict {len(o)} keys {[str(k)[:40] for k in list(o)[:4]]}"
      return f"{type(o).__name__} {len(o)}" if isinstance(o, (list, tuple, set)) else type(o).__qualname__
    obj, seen = held[0], {id(held)}
    for _ in range(12):
      if not (up:=[r for r in gc.get_referrers(obj) if id(r) not in seen and not isinstance(r, types.FrameType)]): break
      seen.add(id(obj:=up[0]))
      print(f"LOAD_APART: held by {what(obj)}")
  if final or not held: refs.clear()

# (while not empty: _fill_zero allocates, the buffers are zeroed later, see _zero_deferred)
_zero_later:list[list[Buffer]] = []
ZERO_ALLOC_LATE = getenv("ZERO_ALLOC_LATE", 1)
def _realize_later(ts:list[Tensor], pending:list[Callable[[], None]]):
  """the tensors are buffers now (allocated, their devices and split kept), their values are computed by the function added to pending"""
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
  if pairs: pending.append(TinyJit(compute, capture_first=True))

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

def _mem_available() -> int:
  """the host memory available (bytes, MemAvailable: 0 if unknown)"""
  try:
    with open("/proc/meminfo") as f: return next((int(l.split()[1]) * 1024 for l in f if l.startswith("MemAvailable:")), 0)
  except OSError: return 0

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

@functools.cache
def _rot_host(hd:int) -> tuple[Tensor, Tensor]:
  """QSA_KV_ROT: the randomized Hadamard matrix of hd dims and its signs (in host memory, made once: every block's the same)"""
  sign = [1 if random.Random(hd + i).random() < 0.5 else -1 for i in range(hd)]
  m = [[sign[i] * (-1)**bin(i & j).count("1") * hd**-0.5 for j in range(hd)] for i in range(hd)]
  return Tensor(m, dtype=dtypes.float32, device="CPU").realize(), Tensor([float(s) for s in sign], dtype=dtypes.float32, device="CPU").realize()

@functools.cache
def _rot_dev(hd:int, device:str|tuple[str, ...]) -> tuple[Tensor, Tensor]:
  """_rot_host's matrix and signs on the devices (made once: every block's the same, not a copy a block)"""
  m, sign = _rot_host(hd)
  return m.to(device).contiguous().realize(), sign.to(device).realize()

_t_import = time.perf_counter()
def _process_age() -> float:
  """the seconds since this process started (its start in /proc), 0 if unknown"""
  try:
    with open("/proc/self/stat") as f: start = int(f.read().rsplit(")", 1)[1].split()[19])
    with open("/proc/uptime") as f: return float(f.read().split()[0]) - start / os.sysconf("SC_CLK_TCK")
  except (OSError, ValueError, IndexError): return 0.0
# (DEBUG_LOAD: where the clock of the laps starts: the process's age then and the wall clock, for the launcher's time before it)
if getenv("DEBUG_LOAD"): print(f"load: qwen4exp imported at process age {_process_age():.2f}s, epoch {time.time():.3f}", flush=True)
def _lap(what:str):
  """DEBUG_LOAD=1: a line when loading gets past `what` (the time is the log's)"""
  if getenv("DEBUG_LOAD"):
    from tinygrad.device import alloc_stats
    try: rss = f", rss {int(open('/proc/self/statm').read().split()[1]) * 4096 / 1e9:.1f} GB"
    except OSError: rss = ""
    print(f"load: {what} at {time.perf_counter()-_t_import:.2f}s ({alloc_stats[0]} allocations in {alloc_stats[1]:.2f}s{rss})", flush=True)

def _open_devices(devices:tuple[str, ...]):
  """the devices opened at the same time (most of opening one is waiting for it): the first alone (what they share is made once)"""
  import threading
  if len(devices) < 2: return
  Device[devices[0]]
  threads = [threading.Thread(target=Device.__getitem__, args=(d,), daemon=True) for d in devices[1:]]
  for t in threads: t.start()
  for t in threads: t.join()

def _bytes_by_device(ts:list[Tensor], devs:tuple[str, ...]) -> dict[str, int]:
  """the bytes of the realized buffers of ts on every device of devs (a buffer once, host memory not)"""
  seen:set[int] = set()
  out = dict.fromkeys(devs, 0)
  for t in ts:
    # (a realized tensor's buffer: its base's, or its own (a view of one))
    try: b = t.uop.base.buffer
    except Exception:
      try: b = t.uop.buffer
      except Exception: continue
    for x in (b.bufs if isinstance(b, MultiBuffer) else [b]):
      x = x.base if x._base is not None else x
      if id(x) in seen or (x.options is not None and x.options.host) or x.device not in out: continue
      seen.add(id(x))
      out[x.device] += x.nbytes
  return out

def _is_buffer(u:UOp) -> bool:
  """the uop is a buffer's (realized: its reshapes, split over devices or not), not a computation"""
  while u.op in (Ops.RESHAPE, Ops.UNSHARD): u = u.src[0]
  return u.op is Ops.BUFFER

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


class GatedResidual:
  """
  qwen4exp hyper-connections: the sublayer input is a sigmoid gated low-rank mix of the normed streams (their mean), its output is added to every
  stream with a 2*sigmoid weight of its own
  """
  _winj:Tensor|None = None  # (the inject weights in float for ghc_pre, see Transformer._finish_load)
  _wnorm:Tensor|None = None  # (the norm weight in float for ghc_pre: not cast by every step)
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
    if isinstance(B, int) and isinstance(T, int) and 8 < B*T <= 1024 and amd_custom_kernels_supported(xs.device) and \
       self.down.ggml_type == self.up.ggml_type == Q8_0 and self.count == 4 and self.dim % 256 == 0 and \
       getattr(self, "_winj", None) is not None and not token_sharded(xs):
      # (the rows of batched decoding: the prefill's kernels on whole tiles of 32 rows (the rows past them padding), not a dozen of generic ones)
      n = B*T
      flat = xs.reshape(n, self.count*self.dim).float().pad(((0, round_up(n, 32) - n), None))
      xn, xhalf, part = ghc_prefill_norm(flat, self.norm["weight"], unwrap(self._winj), self.count, self.eps)
      lo = q8_linear(self.down, xhalf) * (1 / self.count)
      mixed = ghc_prefill_mix((lo * lo.sigmoid()).contiguous(), self.up.weight, xn)
      return mixed[:n].reshape(B, T, self.dim), (part.sum(1) * (1 / self.count))[:n].reshape(B, T, -1)[..., :self.count]
    if not (isinstance(B, int) and isinstance(T, int) and B*T <= 8 and amd_custom_kernels_supported(xs.device)): return None
    for lin in (self.down, self.up):
      if lin.ggml_type is None and lin.use_custom_quant: lin.set_quantized(lin.weight)
    if self.down.ggml_type != Q8_0 or self.up.ggml_type != Q8_0 or getattr(self, "_winj", None) is None: return None
    w = self._wnorm if self._wnorm is not None else self.norm["weight"]
    xn, xq, part = ghc_pre(xs.reshape(B*T, self.count*self.dim), w, unwrap(self._winj), self.count, self.eps)
    down = q8_linear(self.down, xn, xq=xq)
    if 1 < B*T <= 8 and self.count == 4 and self.dim % 32 == 0 and self.up.in_features % 32 == 0 and 0 < self.up.in_features <= 512:
      mixed, inj = ghc_decode_mix(down, self.up.weight, xn, part, self.count)
      return mixed.reshape(B, T, self.dim), inj.reshape(B, T, -1)[..., :self.count]
    if B*T == 1:
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
    if HC_DEC and (dec:=self._split_dec(xs)) is not None: own, post = dec
    elif HC_REAL: own, post, _ = self.pre_split(xs, real=True)
    else:
      mixed, post, _ = self.pre_split(xs)
      # (own: lazy, a copy only if the sublayer reads it (the router): the gather packs the real rows from mixed)
      own = _split_real(mixed).contiguous()
    x = all_gather_q8(mixed, 1, real=_seq_split[0]) if AG_REAL and not HC_REAL and own.shape[-1] % 256 == 0 and mixed.shape[0] == 1 else \
      all_gather_q8(own, 1) if own.shape[-1] % 256 == 0 else all_gather(own, 1, dtypes.half)
    _partial_out[0], _sp_rows[0] = True, own
    try: out = sublayer(x)
    finally: _partial_out[0], _sp_rows[0] = False, None
    # (summed and split already: moe_combine_scatter)
    done = out.ndim == own.ndim and token_sharded(out)
    out = out if done else reduce_scatter(out, 1) if out.ndim == own.ndim + 1 else shard_rows(out, 1)
    return self.post_split(_split_pad(out), post, post, xs)

  def _split_dec(self, xs:Tensor) -> tuple[Tensor, Tensor]|None:
    """
    HC_DEC: the sublayer input of every device's real rows (B, N*R, dim) and the output weights of its padded rows (B, N*P, count) by the decode
    kernels (_fused's few rows path on a device's R rows). None: not those rows or weights
    """
    R, devs = _seq_split[0], xs.device
    if not (isinstance(devs, tuple) and 1 <= R <= HC_DEC_ROWS and self.count == 4 and self.dim % 32 == 0 and self.up.in_features % 32 == 0 and
            0 < self.up.in_features <= 512 and self.down.ggml_type == Q8_0 and self.up.ggml_type == Q8_0 and
            getattr(self, "_winj", None) is not None and amd_custom_kernels_supported(devs)): return None
    B, N = int(xs.shape[0]), len(devs)
    w = self._wnorm if self._wnorm is not None else self.norm["weight"]
    xn, xq, part = ghc_pre(_split_real(xs).reshape(B*N*R, self.count*self.dim), w, unwrap(self._winj), self.count, self.eps)
    down = q8_linear(self.down, xn, xq=xq, split=True)
    mixed, inj = (ghc_decode_mix if R > 1 else ghc_post)(down, self.up.weight, xn, part, self.count)
    return mixed.reshape(B, N*R, self.dim), _split_pad(inj.reshape(B, N*R, -1)[..., :self.count].sigmoid() * 2)

  def pre_split(self, xs:Tensor, norm:Callable[[Tensor], Tensor]|None=None, real:bool=False) -> tuple[Tensor, Tensor, Tensor]:
    """
    the streams' rows split over the devices (sequence parallel, static rows): the sublayer's input of every device's rows and their output
    weights (2*sigmoid(inject)), twice (the phases' coefficients are three, see post_split). real: the sublayer's input of every device's
    real rows only (batched decoding's, _split_real's), the padding rows not read
    """
    B, R = xs.shape[:2]
    rows = int(B * R)
    if self._winj is not None and self.dim % 256 == 0 and amd_custom_kernels_supported(xs.device):
      # (HC_NORM_REAL with real: the norm of a device's real rows only)
      xn, xhalf, part = ghc_prefill_norm(xs.reshape(rows, -1), self.norm["weight"], self._winj, self.count, self.eps,
                                         rows=_seq_split[0] if real and HC_NORM_REAL else 0)
      # (the streams' sums added, not a reduce: no kernel of their own, the residual's kernel computes post from them, in the same order)
      inj = functools.reduce(lambda a, b: a + b, [part[:, s] for s in range(self.count)]).reshape(B, R, -1)
    else:
      xn = (xs.float().rmsnorm(eps=self.eps) * self.norm["weight"].reshape(self.count, self.dim)).flatten(2).contiguous()
      xhalf, inj = xn.reshape(rows, -1).half().contiguous(), self.inject(xn)
    R_real = _seq_split[0] if real else 0
    # (m_rows: batched decoding's real rows a device, the tiles of the others not computed (TILE16))
    lo = q8_linear(self.down, xhalf, split=True, rows=R_real, m_rows=_seq_split[0]) * (1 / self.count)
    # (HC_REAL, batched decoding's real rows (a prefill's bits as they were): not materialized in float: the kernel summing the down
    # projection's splits stores the half the mix reads (its rcp can round differently))
    act = lo * lo.sigmoid() if HC_REAL and real else (lo * lo.sigmoid()).contiguous()
    if self.up.ggml_type == Q8_0 and self.count == 4 and self.dim % 32 == 0 and \
       int(xhalf.uop.shard_shape[0]) % 32 == 0:
      shape = (int(B), len(cast(tuple, xs.device))*R_real, self.dim) if real else (int(B), int(R), self.dim)
      mixed = ghc_prefill_mix(act, self.up.weight, xn.reshape(rows, -1), shape, rows=R_real, m_rows=_seq_split[0])
    else:
      g = q8_linear(self.up, act.contiguous(), split=True).reshape(B, R, self.count, self.dim)
      mixed = (g.sigmoid() * xn.reshape(B, R, self.count, self.dim)).mean(2)
      if real: mixed = _split_real(mixed).contiguous()
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

    # --- the gated hyper-connections (they norm the streams themselves) --
    assert config.hc is not None and config.hc.low_rank > 0, "qwen4exp: gated hyper-connections"
    self.gated, self.attn_norm, self.ffn_norm = True, _identity, _identity
    self.hc_attn, self.hc_ffn = GatedResidual(config), GatedResidual(config)
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
    if (T:=_ffn_rows[0]) is not None and sel.ndim >= 3 and isinstance(R:=sel.shape[-2], int):
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
      x_down = self.ffn_down_exps(sel, act,
                                  dtypes.half if _batch_rows[0] is not None and self.ffn_down_exps.slices and _seq_batch[0] is None else
                                  dtypes.float32)
      ws = weighted_sum(x_down, probs) if _batch_rows[0] is not None else None
      out = ws if ws is not None else (x_down * probs.unsqueeze(-1)).sum(axis=-2)  # (B, T, D), expert parallel: (devices, B, T, D)
    partial = out.ndim == x.ndim + 1
    keep = partial and _partial_out[0]  # the partial sums are returned (sequence parallel)
    if shexp is not None:
      if partial and not keep: out = partial_sum(out)
    elif hasattr(self, 'ffn_gate_shexp'):
      shexp_act = gated_activation(self.ffn_gate_shexp(x), self.ffn_up_shexp(x), limit=self.config.swiglu_limit)
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
    i8 = gate_exps.ggml_type in I8_WMMA_QUANTS and gate_exps.ggml_type != IQ3_S
    tile = MOE_TILE if _wmma_rdna4(x.device) or i8 else 32
    # (MOE_TILE16 (RDNA3, IQ3_S gate|up with the K256 sub-blocks unrolled by XH or UNROLL): tiles of 16 rows when the experts have few
    # rows each (a warm turn's chunk): half the tensor core work of a tile of 32 mostly padding, the same bits per row)
    if tile == 32 and (amd_qwen.K256_XH or amd_qwen.K256_UNROLL) and gate_exps.ggml_type == IQ3_S and isinstance(T, int) and \
       _moe_use_tile16(cast(int, B) * T * int(sel.shape[-1]), self.config.num_experts): tile = 16
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
    # (a row per sorted row, the selections read through pos: the rows written are together, not spread over every selection's)
    down = down_exps.grouped(act.contiguous(), entries, tiles, selections=tokens*k, out_dtype=dtypes.half if down_exps.slices else dtypes.float32,
                             sorted_out=True)
    y = moe_combine(down, pos, probs, tokens, k, add, add_scale=add_scale,
                    out_dtype=dtypes.half if half and gate_exps.expert_parallel else dtypes.float32)
    y = y.reshape(-1, *sel_shape[:-1], D)[:, :, :x.shape[1] if isinstance(T, int) else T]
    return y if gate_exps.expert_parallel else y[0]

  # given the token-prefix match, return how much cached state this block can still reuse
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return prefix_len
  def _init_state(self, x:Tensor): raise NotImplementedError
  def _attention(self, x:Tensor, start_pos:int|UOp, T:int|UOp|None=None) -> Tensor: raise NotImplementedError
  def _ple(self, x:Tensor, emb:Tensor, start_pos:int|UOp, T:int|UOp|None) -> Tensor: raise NotImplementedError  # (GatedDeltaNetBlock's)

  def __call__(self, x: Tensor, start_pos: int|UOp, T:int|UOp|None=None, ple_emb:Tensor|None=None):
    # T (hyper-connections): x has static padded rows of which the first T are the tokens, the output too
    # ple_emb: the n-gram embedding rows of the tokens (the block with the PLE layer)
    self._init_state(x)
    if ple_emb is not None and not hasattr(self, "ple"): ple_emb = None
    # we pass in the weights implicitly so we unpack the GGUF on the fly
    @function(precompile=True, allow_implicit=True)
    def _run(x:Tensor, start_pos:int|UOp, T:int|UOp|None, ple_emb:Tensor|None):
      if ple_emb is not None:
        if _seq_split[0] and token_sharded(x):
          # (split rows: the n-gram embedding of all the real rows, every device adds its rows of it)
          x = x + _split_pad(shard_rows(self._ple(all_gather(_split_real(x).contiguous(), 1, dtypes.half).float(), ple_emb, start_pos, T), 1))
        else: x = x + self._ple(x, ple_emb, start_pos, T)
      xs = self.hc_attn(x, lambda h: self._attention(h, start_pos, *(() if T is None else (T,))), self.attn_norm, rows=T is not None)
      _ffn_rows[0] = T
      try: return self.hc_ffn(xs, self._feed_forward, self.ffn_norm, rows=T is not None).contiguous()
      finally: _ffn_rows[0] = None
    return _run(x, start_pos, T, ple_emb)

  # the phases of a sublayer on a micro batch (see micro_batch_blocks): functions, scheduled once
  def _phase_pre(self, hc:GatedResidual, norm:Callable[[Tensor], Tensor], xs:Tensor) -> tuple[Tensor, Tensor, Tensor]:
    @function(precompile=True, allow_implicit=True)
    def _pre(xs:Tensor) -> tuple[Tensor, Tensor, Tensor]:
      hn, post, comb = hc.pre_split(xs, norm)
      return _fn_out(hn), _fn_out(post), _fn_out(comb)
    return _pre(xs)
  def _phase_sub(self, sub:str, hn:Tensor, start_pos:int|UOp, T:int|UOp) -> Tensor:
    @function(precompile=True, allow_implicit=True)
    def _sub(hn:Tensor, start_pos:int|UOp, T:int|UOp) -> Any:
      # (sent as int8 groups of 32, as the quantized layers would make them)
      x = all_gather_q8(hn, 1) if hn.shape[-1] % 256 == 0 else all_gather(hn, 1, dtypes.half)
      # Route each device's own rows; gather the small selections/probabilities rather than repeat the router projection everywhere.
      _partial_out[0], _sp_rows[0] = True, hn
      # (in half: the layers' kernels read it so, a cast to float and back is two copies)
      if sub != "attn": _ffn_rows[0] = T
      if state: _state_writes[0] = []
      try: out = self._attention(x, start_pos, T) if sub == "attn" else self._feed_forward(x)
      finally: _partial_out[0], _sp_rows[0], _ffn_rows[0] = False, None, None
      writes, _state_writes[0] = _state_writes[0], None
      # partial sums, or the whole output
      # (summed and split already: moe_combine_scatter)
      done = out.ndim == hn.ndim and token_sharded(out)
      out = _fn_out(out if done else reduce_scatter(out, 1) if out.ndim == hn.ndim + 1 else shard_rows(out, 1))
      return (out, state_token(hn, writes or [])) if state else out
    # (the attention's state writes: the token the other micro batch's attention waits for, see micro_batch_blocks)
    state = sub == "attn" and _mb_active[0]
    if not state: return _sub(hn, start_pos, T)
    out, token = _sub(hn, start_pos, T)
    self._token:Tensor|None = token
    return out
  def _phase_ple(self, x:Tensor, emb:Tensor, start_pos:int|UOp, T:int|UOp) -> Tensor:
    """the n-gram embedding of all the rows x of a micro batch (a function: its conv state's write is the call's, the next batch's after it)"""
    @function(precompile=True, allow_implicit=True)
    def _ple(x:Tensor, emb:Tensor, start_pos:int|UOp, T:int|UOp) -> Tensor: return self._ple(x, emb, start_pos, T).contiguous()
    return _ple(x, emb, start_pos, T)
  def _phase_post(self, hc:GatedResidual, out:Tensor, post:Tensor, comb:Tensor, xs:Tensor) -> Tensor:
    @function(precompile=True, allow_implicit=True)
    def _post(out:Tensor, post:Tensor, comb:Tensor, xs:Tensor) -> Tensor:
      # (the streams kept in half: their kernels read half the bytes)
      return hc.post_split(out, post, comb, xs).cast(dtypes.half).contiguous()
    return _post(out, post, comb, xs)

def _fn_out(t:Tensor) -> Tensor:
  """
  a function's output: a kernel's output as it is (its buffer: contiguous() of it in a function is a copy kernel, ~0.1 ms of 1152 rows of
  2560 floats, 500 a 13824 chunk), else made contiguous
  """
  return t if t.uop.has_buffer_identity(after_ok=True) else t.contiguous()

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
  # (every batch's calls on its own queue (AM_MEC_PIPES): the copies of one run with the kernels of the other)
  def queued(i:int, *outs:Tensor) -> tuple[Tensor, ...]: return _on_queue(outs, f"COMPUTE:{q}") if (q:=mb_queue(i)) else outs
  def after(i:int, x:Tensor, dep:Tensor|None) -> Tensor:
    """x after dep: a kernel of its own on the batch's queue (x and a value of dep (zero, also from a NaN or inf), every device its part)"""
    return x if dep is None else queued(i, after_kernel(x, dep))[0]
  subs = [(block, hc, norm, sub) for block in blocks for hc, norm, sub in ((block.hc_attn, block.attn_norm, "attn"),
                                                                           (block.hc_ffn, block.ffn_norm, "ffn"))]
  for block in blocks: block._init_state(xs[0])
  ple_done:list[Tensor] = []
  def pre(i:int, k:int) -> tuple[Tensor, ...]:
    block, hc, norm, sub = subs[k]
    if sub == "attn" and ple is not None and hasattr(block, "ple"):
      # (the n-gram embedding's conv runs over the rows in order: on all of them of the batch, every device keeps its rows. a batch's conv
      # state is the one the batch before it wrote: after it. the gather, the embedding and its add: calls on the batch's queue)
      n = int(ple.shape[1]) // len(xs)
      xi = after(i, xs[i], ple_done[-1] if ple_done else None)
      @function(precompile=True, allow_implicit=True)
      def _gather(x:Tensor) -> Tensor: return all_gather(x, 1).contiguous()
      p = queued(i, block._phase_ple(queued(i, _gather(xi))[0], ple[:, i*n:(i+1)*n].contiguous(), starts[i], ts[i]))[0]
      ple_done.append(p)
      @function(precompile=True, allow_implicit=True)
      def _add(x:Tensor, p:Tensor) -> Tensor: return (x + shard_rows(p, 1)).contiguous()
      xs[i] = queued(i, _add(xs[i], p))[0]
    return queued(i, *block._phase_pre(hc, norm, xs[i]))
  # the batches' sublayers in step: batch i's sublayer k after batch i-1's (its states the one before wrote), the collectives of one run while
  # the other computes
  # (one micro batch: its collectives on its compute queue, nothing to run under them: no waits between queues (tr9 576/1152-row warm
  # chunks -7%/-5%). RING_QUEUE_MB1=1: on their own queue as two batches' are)
  RING_QUEUE_ON[0] = len(xs) > 1 or bool(getenv("RING_QUEUE_MB1", 0))
  _mb_active[0] = True
  deps:list[list[Tensor|None]] = [[None]*len(subs) for _ in xs]
  phases:list[tuple[Tensor, ...]] = [() for _ in xs]
  for k in range(len(subs)):
    for i in range(len(xs)):
      block, hc, norm, sub = subs[k]
      FLAG_LANE[0] = i
      if k == 0: phases[i] = pre(i, 0)
      hn, post, comb = phases[i]
      block._token = None
      out = queued(i, block._phase_sub(sub, after(i, hn, deps[i-1][k] if i else None), starts[i], ts[i]))[0]
      # (the next batch waits for the states this one wrote (none: a feed forward's), not its output)
      deps[i][k] = queued(i, block._token)[0] if block._token is not None else None
      xs[i] = queued(i, block._phase_post(hc, out, post, comb, xs[i]))[0]
      if k + 1 < len(subs): phases[i] = pre(i, k+1)
  FLAG_LANE[0], RING_QUEUE_ON[0], _mb_active[0] = 0, False, False
  return xs

class TransformerBlock(FFNBlock):
  """the attention projections of the qwen sparse attention block (QSABlock)"""
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


# the first row of the sequence's region of the qwen sparse attention's caches (shared by sequences, see Transformer.regions) while its call is
# traced: a bound variable (or 0), a multiple of SEQ_BASE_ALIGN
_seq_base:list[UOp|int] = [0]
# (split indexer: the pools of a device a prefill chunk's sequence is gathered of while its call is traced, None: a context's, see
# Transformer._pk_bucket)
_pk_local:list[int|None] = [None]
# PREFILL_PK_CTX: the contexts (tokens, ascending) a prefill chunk smaller than the largest is also traced for (its jit of the shortest one its
# context fits): the split indexer's pooled keys of that many tokens gathered, not of a whole context (a small chunk's gather of 262k a
# constant ~6 ms on tr8)
PREFILL_PK_CTX = tuple(int(c) for c in getenv("PREFILL_PK_CTX", "").split(",") if c)
# (a prefill chunk's time is about that of this many rows more than its padded ones: its kernels' fixed time, see Transformer._chunk_plan)
CHUNK_FIXED_ROWS = getenv("CHUNK_FIXED_ROWS", 256)
# (a multiple of the pools of the KV heads' devices (kpool*kv_split): a region's pools on the devices as its positions)
REGION_ALIGN = 3072
# Sampled harnesses and packed serving may use the split KV/indexer alignment; moves/stash still use REGION_ALIGN.
SEQ_BASE_ALIGN = getenv("SEQ_BASE_ALIGN", REGION_ALIGN)
def _region_rows(config:TransformerConfig) -> int:
  """the rows of the caches the regions of the sequences are of (the REGION_ALIGN after them: the padding sequences' region)"""
  return round_up(max(config.max_context, config.cache_rows), REGION_ALIGN)
# a prefill chunk's rows in this many micro batches (sequence parallel: the copies of one run with the kernels of the other)
MICRO_BATCHES = getenv("MICRO_BATCHES", 2)
PREFILL_MB1_ROWS = getenv("PREFILL_MB1_ROWS", 0)
PREFILL_PARTS = getenv("PREFILL_PARTS", 1)
WARM_ASYNC = getenv("WARM_ASYNC", 1)
LANES_LATE = getenv("LANES_LATE", 1)
PREFILL_TILE32 = getenv("PREFILL_TILE32", 1)
# MOE_TILE16: grouped experts in tiles of 16 rows if a chunk has at most this many selections an expert on average (see _grouped_experts)
MOE_TILE16 = getenv("MOE_TILE16", 40)
def _moe_use_tile16(selections:int, experts:int) -> bool:
  """fewer padded rows and weight reads with 16-row tiles (17-32 rows/expert: both tile sizes pad to 32, so keep 32)"""
  if not MOE_TILE16 or selections > MOE_TILE16 * experts: return False
  return round_up(selections, 16 * experts) < round_up(selections, 32 * experts)
PREFILL_SIZES = tuple(int(c) for c in getenv("PREFILL_SIZES", "192,512,1152,2048,4096").split(",") if c)
# the attention cache's keys (1) and values (2) rotated, see QSABlock._rot
QSA_KV_ROT = getenv("QSA_KV_ROT", 0)

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


def _rope(x:Tensor, cos:Tensor, sin:Tensor) -> Tensor:
  """half-split rope of x (..., d) with cos and sin (..., d/2) broadcast to it"""
  x1, x2 = x.chunk(2, dim=-1)
  return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)


class QSAIndexer:
  """
  qwen sparse attention's indexer: the keys of a pool of kpool tokens are averaged, normed and roped at the pool's first position, a query's (its
  heads normed and roped) score of a pool is the sum of its heads' rectified dots. its cache holds the pooled keys only: the keys of the pool a
  position is in come from its state (the kpool-1 keys before it, slotted as the gated deltanets' conv states) and the call's
  """
  state:Tensor  # (S, kpool-1, D) the keys of the kpool-1 positions before a position p at p%S (S: the model's states', see update)
  pool_state:Tensor  # batched decoding: every slot's (n*SC, kpool-1, D), see slot_states
  def __init__(self, config:TransformerConfig):
    self.idx, self.max_context, self.rows = (idx:=unwrap(config.indexer)), config.max_context, _region_rows(config) + REGION_ALIGN
    self.S, self.SC, self.split = config.rewind + 1, slot_states(config.spec_drafts)[1], config.idx_split
    # (split: a sequence's pools on a device, at most: its region's pools of every device gathered for a prefill)
    self.local_pools = -(-config.max_context // idx.kpool // self.split)
    self.q_proj, self.k_proj = Linear(config.dim, idx.n_heads * idx.head_dim, bias=False), Linear(config.dim, idx.head_dim, bias=False)
    self.q_norm, self.k_norm = nn.RMSNorm(idx.head_dim, idx.norm_eps), nn.RMSNorm(idx.head_dim, idx.norm_eps)

  def _init_state(self, x:Tensor):
    if hasattr(self, "cache_pk"): return
    # the pooled keys (a call pools the pools of its tokens) of the rows of every sequence's region (see Transformer.regions), REGION_ALIGN more
    # than a context: the region of the padding sequences of a step
    n = self.split
    self.cache_pk = _zeros(self.rows // self.idx.kpool + (n * self.local_pools if n > 1 else 0), self.idx.head_dim, dtype=dtypes.half,
                           device=x.device, axis=0 if n > 1 else None)
    self.state = _zeros(self.S, self.idx.kpool - 1, self.idx.head_dim, dtype=dtypes.half, device=x.device)

  def _init_slots(self, n:int):
    """the states of n sequences of batched decoding (a slot each): SC of them a slot, the one of a position p is p%SC (made, not zeroed)"""
    if hasattr(self, "pool_state") and int(self.pool_state.shape[0]) >= n * self.SC: return
    self.pool_state = _slot_pool(n, self.state, (self.SC, *self.state.shape[1:]), views=0)[0]

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

  def _static(self, pk:Tensor) -> Tensor:
    """the pooled keys of a sequence (pools, D) as a static (max_context//kpool, D) copy (zeros past them)"""
    return pk.pad_to((self.max_context // self.idx.kpool, self.idx.head_dim)).contiguous()

  def update(self, x:Tensor, start_pos:int|UOp, T:int|UOp, freqs:Tensor, gather:bool=True) -> tuple[Tensor, int|UOp]:
    """
    the keys of the T tokens of x at start_pos and their pools cached (the sequence's region, from _seq_base): the pooled keys (pools, D) of the
    sequence (after) and that count of pools up to the last token's. split: the sequence's pooled keys gathered (gather: else the cache after)
    """
    idx, kp, b, Tq = self.idx, self.idx.kpool, _seq_base[0], int(x.shape[1])
    S, pb = start_pos + T, b // kp
    assert Tq % kp == 0, "the rows of a call: whole pools"
    keys = self._proj(self.k_proj, x[0]).half()  # (Tq, D): the first T are the tokens'
    # the keys of the kpool-1 positions before start_pos (its state, start_pos%S) and the tokens': window row i is position start_pos-(kp-1)+i
    # (read with the keys, a value of the call's input: a micro batch's state is the one the batch before it wrote, see GatedDeltaNetBlock)
    states, NS = self.state, int(self.state.shape[0])
    rp, wp = start_pos % NS, S % NS
    prev = states.shrink(((rp, rp+1), None, None)).reshape(kp-1, idx.head_dim)
    window = (prev + keys[:1] * 0).cat(keys, dim=0).pad(((0, kp), None)).contiguous()
    # the state of the position after the tokens: the window's rows of the kpool-1 positions before it
    store = states.shrink(((wp, wp+1), None, None)).uop.store(window.shrink(((T, T+kp-1), None)).unsqueeze(0).uop)
    # the pools of the tokens (the first one's tokens before them from the state): the window from the first one's first position
    first, new = start_pos // kp, (start_pos % kp + T + kp - 1) // kp
    # (pooled after the state is written: the next micro batch waits for the pooled keys (see micro_batch_blocks), the state then too)
    at = kp - 1 - start_pos % kp
    keys_p = Tensor(window.uop.after(store)).shrink(((at, at + Tq + kp), None)).reshape(Tq // kp + 1, kp, idx.head_dim)
    pf = freqs.reshape(-1, kp, freqs.shape[-1])[:, 0].shrink(((first, first + new), None))
    pools = (S + kp - 1) // kp
    if self.split > 1:
      # (every pool of the chunk (whole chunks of them), the ones of the tokens into their devices' rows)
      pf = freqs.reshape(-1, kp, freqs.shape[-1])[:, 0].pad(((0, Tq // kp + 1), None)).shrink(((first, first + Tq // kp + 1), None))
      pk_all = self._pooled(keys_p, *pf.chunk(2, dim=-1))
      j = Tensor.arange(Tq // kp + 1, dtype=dtypes.int32).to(x.device)
      p0 = pb + first
      prow = j + (Tensor(p0).cast(dtypes.int32) if isinstance(p0, UOp) else p0)
      pk = rows_scatter(self.cache_pk, pk_all.half(), self._pk_index(prow, j < Tensor(new)))
      _state_written(pk)
      return (self._gathered(pk, pb) if gather else pk), pools
    pk_new = self._pooled(keys_p.shrink(((0, new), None, None)), *pf.chunk(2, dim=-1))
    pk = Tensor(self.cache_pk.uop.after(self.cache_pk.shrink(((pb + first, pb + first + new), None)).uop.store(pk_new.half().uop)))
    _state_written(pk)
    return pk.shrink(((pb, pb + pools), None)), pools

  def _pk_index(self, rows:Tensor, valid:Tensor|None=None) -> Tensor:
    """the rows of the pooled keys of every device of the pools' rows (n,) (valid ones) (split: its own ones', -1 for the others)"""
    if self.split == 1: return rows if valid is None else valid.where(rows, -1)
    n, devs = self.split, cast(tuple[str, ...], self.cache_pk.device)
    own = (rows % n).reshape(1, -1) == _device_index(devs).reshape(-1, 1)
    if valid is not None: own = own & valid.reshape(1, -1)
    return own.where((rows // n).reshape(1, -1), -1).reshape(-1).contiguous()

  def _gathered(self, pk:Tensor, pb:int|UOp) -> Tensor:
    """the pooled keys (split) of the sequence of the region from pool pb, whole on every device: (local_pools*split, D), row p its pool p"""
    n, L = self.split, _pk_local[0] or self.local_pools
    at = Tensor.arange(L, dtype=dtypes.int32).to(pk.device) + (Tensor(pb // n).cast(dtypes.int32) if isinstance(pb, UOp) else pb // n)
    mine = rows_gather(pk, at.contiguous(), split=True)  # (n*L, D) split: every device its rows of the sequence's pools
    return all_gather(mine, 0).reshape(n, L, -1).permute(1, 0, 2).reshape(n * L, -1).contiguous()

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
    spk = self._static(pk) if self.split == 1 else pk[:min(self.max_context // idx.kpool, int(pk.shape[0]))]
    if (scores:=index_scores(q.reshape(Tq, idx.n_heads, idx.head_dim), spk, pools, idx.head_dim**-0.5)) is None:
      sc = q.reshape(Tq*idx.n_heads, idx.head_dim).half().contiguous().matmul(spk.T, dtype=dtypes.float).contiguous()
      scores = (sc.reshape(Tq, idx.n_heads, -1).relu().sum(1) * idx.head_dim**-0.5).contiguous()
    # (compact (the pools, not their keys): a quarter of the bytes gathered)
    lists = pool_index_rows(scores, start_pos, pools, idx.top_k // idx.kpool, idx.kpool, compact=True)
    out = all_gather(lists, 0) if split else lists
    return out


class QSABlock(TransformerBlock):
  cache_k:Tensor
  """
  qwen sparse attention: gated GQA attention on the keys its indexer picks. the heads split over the devices with one KV head each (repeated at
  load), so a device's cache is the shared row [k|v] of its KV head (2*head_dim) and its queries [q|0]: the MLA flash kernels' latent attention
  (their output's second half is the attention's). the cache (KV, B, N, 2*head_dim) split on KV: a device's (1, B, N, D) is the latent cache
  (1, B, 1, N, D) of the kernels. batched decoding: the caches of the slots (KV, n, ctx, D), the slot of a group its index
  """
  pool_k:Tensor
  indexer:QSAIndexer|None
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
    if T is not None: return self._attention_rows(x, start_pos, T)
    # a symbolic chunk computes padded to its max size (static shapes), it stores and attends with its real tokens
    if not isinstance(T:=x.shape[1], int): return self._attention_rows(x.pad_to(x.max_shape), start_pos, T)[:, :T]
    return self._attention_rows(x, start_pos, T)

  def _attention_rows(self, x:Tensor, start_pos:int|UOp, T:int|UOp) -> Tensor:
    # x has Tq rows, the first T are the tokens at start_pos
    B, Tq, _ = x.shape
    hd = self.config.head_dim
    freqs = self.freqs_cis.shrink(((start_pos, start_pos + T), None)).pad_to((Tq, self.freqs_cis.shape[-1]))
    cos, sin = (f.reshape(1, 1, Tq, -1) for f in freqs.chunk(2, dim=-1))
    q, gate, k, v = self._qkv(x, cos, sin)
    b = _seq_base[0]  # (the sequence's region of the cache)
    if not self._flash(x):
      # (without the custom kernels: every key, causal (model.py's Indexer is the reference of the sparse attention). the cache's halves are the
      # keys and the values)
      rows = k.cat(v, dim=-1)[:, :, :T].transpose(0, 1).cast(self.cache_k.dtype)
      store = self.cache_k.shrink((None, None, (b+start_pos, b+start_pos+T), None)).uop.store(rows.uop)
      cache = Tensor(self.cache_k.uop.after(store)).shrink((None, None, (b, b+start_pos+T), None)).transpose(0, 1).float()
      causal = Tensor.full((1, 1, Tq, start_pos+T), float("-inf")).triu(start_pos+1)
      return self._out_gated(q.scaled_dot_product_attention(cache[..., :hd], cache[..., hd:], attn_mask=causal, enable_gqa=True), gate)
    # (every query on the keys of its list (qsa_decode's kernel): its keys alone are read)
    assert self.sparse and self.indexer is not None and B == 1 and isinstance(Tq, int) and Tq % 32 == 0 and isinstance(start_pos, UOp) and \
      self.indexer.idx.kpool == 4 and hd == 256, "the qwen sparse attention on the custom kernels: a prefill chunk of tiles of 32 rows"
    row_picks = self.indexer.prefill_picks(x, start_pos, T, self.freqs_cis)
    cache = self._cache_store(k, v, start_pos, T)
    _state_written(cache)
    bases = (Tensor.zeros(Tq, dtype=dtypes.int32, device=x.device) + (Tensor(b).cast(dtypes.int32) if isinstance(b, UOp) else b)).contiguous()
    attn = self._attend(q[0].transpose(0, 1), cache, row_picks, bases)  # (Tq, H, 1, hd)
    return self._out_gated(attn.permute(2, 1, 0, 3), gate)

  def _cache_store(self, k:Tensor, v:Tensor, start_pos:int|UOp, T:int|UOp) -> Tensor:
    """the keys and values (B, KV, Tq, hd) of the T tokens at start_pos (of the region at _seq_base) into the cache: the cache after"""
    b, Tq = _seq_base[0], int(k.shape[2])
    if self.config.kv_split == 1:
      k_store = k.cat(v, dim=-1)[:, :, :T].transpose(0, 1)  # (KV, B, T, 2*hd)
      store = self.cache_k.shrink((None, None, (b+start_pos, b+start_pos+T), None)).uop.store(self._kv_rows(k_store).uop)
      return Tensor(self.cache_k.uop.after(store))
    # (the rows split over the devices of a KV head: every device its pools' rows of them)
    t = Tensor.arange(Tq, dtype=dtypes.int32).to(k.device)
    at = b + start_pos
    phys = t + (Tensor(at).cast(dtypes.int32) if isinstance(at, UOp) else at)
    W = int(self.cache_k.shape[-1])
    kv = self._kv_rows(k.cat(v, dim=-1).transpose(0, 1).reshape(-1, 2*self.config.head_dim))
    return rows_scatter(self.cache_k.reshape(-1, W), kv, self._kv_index(phys, t < Tensor(T))).reshape(self.cache_k.shape)

  def _kv_index(self, phys:Tensor, valid:Tensor|None=None) -> Tensor:
    """
    the rows of the cache of every device of the positions' rows phys (n,) of the regions (valid ones): (KV*n,) split, a device's row of the
    ones of its pools, -1 for the others (not written). kv_split 1: phys
    """
    S, kp = self.config.kv_split, unwrap(self.indexer).idx.kpool
    if S == 1: return phys if valid is None else valid.where(phys, -1)
    devs = cast(tuple[str, ...], self.cache_k.device)
    part = (_device_index(devs) % S).reshape(-1, 1)
    p = (phys // kp).reshape(1, -1)
    own = (p % S) == part
    if valid is not None: own = own & valid.reshape(1, -1)
    return own.where((p // S) * kp + (phys % kp).reshape(1, -1), -1).reshape(-1).contiguous()

  def _attend(self, q:Tensor, cache:Tensor, picks:Tensor, bases:Tensor, compact:bool=True, **kw) -> Tensor:
    """
    every row's heads q (rows, H, hd) on the keys of its list picks (compact: (rows, top+2) its pools, see _pool_index_kernel) of the cache from
    bases: (rows, H, 1, hd). kv_split (compact lists): a device's heads (its group's) on its keys, the devices' outputs weighted by their log
    sum exps (exchanged)
    """
    if (rot:=self._rot) is not None:
      # (QSA_KV_ROT: the cache's keys and values rotated: the queries alike, the output back)
      if QSA_ROT_FWHT and (sign:=getattr(self, "_rot_sign", None)) is not None:
        out = self._attend_cache(fwht(q, sign) if QSA_KV_ROT & 1 else q, cache, picks, bases, compact, **kw)
        return fwht(out, sign, transpose=True) if QSA_KV_ROT & 2 else out
      out = self._attend_cache(q.float() @ rot if QSA_KV_ROT & 1 else q, cache, picks, bases, compact, **kw)
      return out @ rot.T if QSA_KV_ROT & 2 else out
    return self._attend_cache(q, cache, picks, bases, compact, **kw)

  def _attend_cache(self, q:Tensor, cache:Tensor, picks:Tensor, bases:Tensor, compact:bool=True, **kw) -> Tensor:
    kp, S, hd = unwrap(self.indexer).idx.kpool, self.config.kv_split, self.config.head_dim
    kw["bits"] = self._kv_bits
    if S == 1: return cast(Tensor, qsa_decode_attention(q, cache.unsqueeze(2), picks, bases, hd**-0.5, kpool=kp if compact else 0, **kw))
    local = picks_local(picks, S, kp)
    attn, lse = cast(tuple[Tensor, Tensor], qsa_decode_attention(q, cache.unsqueeze(2), local, bases // S, hd**-0.5, kpool=kp, lse=True, **kw))
    rows, H = int(lse.shape[0]), int(lse.shape[1])
    N = len(cast(tuple, lse.device))
    # the log sum exp of every head of a group: of its devices' (the heads of device d are the group's, d's part of its keys)
    # (gathered with the devices' parts outermost: one contiguous part a device)
    la = all_gather(lse.T.contiguous(), 0).T.reshape(rows, N // S, S, H // N)
    m = la.max(2, keepdim=True)
    tot = (m + (la - m).exp().sum(2, keepdim=True).log()).expand(rows, N // S, S, H // N).reshape(rows, H)
    w = (lse - shard_rows(tot.contiguous(), 1)).exp()
    return attn * w.reshape(rows, H, 1, 1)

  def _attention_seqs(self, x:Tensor, sb:SeqBatch) -> Tensor:
    """
    a step of G sequences sharing the caches (each its region of rows from its base): R rows of every one from its position, every query on
    the keys of its region its indexer picks (every key before it with a context of top_k or less)
    """
    assert self.indexer is not None and self._flash(x), "sequences sharing the caches: the indexer's and the flash kernels"
    G, R, hd = sb.G, sb.R, self.config.head_dim
    rows, idx, ic = G*R, self.indexer, self.indexer.idx
    kp = ic.kpool
    pos, phys = sb.row_positions, sb.row_phys
    freqs = rows_gather(self.freqs_cis, pos, const=True)  # (rows, rope)
    cos, sin = (f.reshape(1, 1, rows, -1) for f in freqs.chunk(2, dim=-1))
    q, gate, k, v = self._qkv(x, cos, sin)
    # the keys and values of the rows into their rows of the cache (a device's of its KV head)
    W = int(self.cache_k.shape[-1])
    kv = self._kv_rows(k.cat(v, dim=-1).transpose(0, 1).reshape(-1, 2*hd))
    cache = rows_scatter(self.cache_k.reshape(-1, W), kv, self._kv_index(phys)).reshape(self.cache_k.shape)
    # the indexer: the keys of the rows, then the pools they're in (NP from a sequence's first row's, the ones past its last row's are
    # incomplete: never picked, made again when they are) pooled from a sequence's window: the keys of the kpool-1 positions before its first
    # row (its slot's state of the position) and its rows' (window row i is position positions[g]-(kp-1)+i)
    keys = idx._proj(idx.k_proj, x[0]).half().reshape(G, R, ic.head_dim)
    NP = (R - 1) // kp + 2
    # (and every row's state of the position after it written: the window's rows of the kpool-1 positions before that)
    pkeys, pool_after = pool_window(idx.pool_state, keys, sb.slots, sb.positions, idx.SC, NP, kp)
    pkeys = pkeys.reshape(G*NP, kp, ic.head_dim)
    pools = ((sb.positions // kp).reshape(G, 1) + Tensor.arange(NP, dtype=dtypes.int32).reshape(1, NP).to(x.device)).reshape(-1)  # (G*NP,)
    pbase = sb.bases.reshape(G, 1).expand(G, NP).reshape(-1) // kp
    pf = rows_gather(self.freqs_cis, (pools * kp).contiguous(), const=True)
    pooled = idx._pooled(pkeys, *pf.chunk(2, dim=-1))
    pk = Tensor(rows_scatter(idx.cache_pk, pooled.half(), idx._pk_index((pbase + pools).contiguous())).uop.after(pool_after.uop))
    # every query's scores of its sequence's pools, the lists of the keys it sees
    iq = idx._queries(x[0], *(f.reshape(rows, 1, -1) for f in freqs.chunk(2, dim=-1)))  # (rows, H, D)
    scores = seq_scores(iq, pk, sb.bases, sb.positions, R, kp, self.config.max_context // kp, n=idx.split)
    compact = self.config.kv_split > 1 or idx.split > 1
    if idx.split > 1: picks = seq_index_split(scores, sb.positions, ic.top_k // kp, kp)
    else: picks = seq_index(scores, sb.positions, ic.top_k // kp, kp, compact=compact)
    # every row its own list of the keys of its region: [q|0] on the latent [k|v] rows (the second half of the output is the attention's)
    qr = q.permute(2, 1, 0, 3)  # (rows, H, 1, hd)
    latent = Tensor(cache.uop.after(pk.uop))
    # (a lane a key: its scores of the heads without reductions over the lanes, the value half only of the rows read for the output)
    attn = self._attend(qr[:, :, 0], latent, picks, sb.row_bases, compact=compact, parts=8, waves=4)
    return self._out_gated(attn.permute(2, 1, 0, 3), gate)

  @property
  def _kv_bits(self) -> int:
    """the bits of a value of the cache: 16 (half), 8 (kv_q8_pack's int8 groups) or 4 (kv_q4_pack's, QSA_KV_BITS=4)"""
    if self.cache_k.dtype != dtypes.uint32: return 16
    return 4 if int(self.cache_k.shape[-1]) == 2*self.config.head_dim * 9 // 64 else 8
  def _kv_rows(self, kv:Tensor) -> Tensor:
    """rows (..., 2*head_dim) of keys and values as the cache's (QSA_KV_ROT: rotated, see _rot)"""
    if (rot:=self._rot) is not None:
      hd = self.config.head_dim
      k, v = kv[..., :hd], kv[..., hd:]
      if QSA_ROT_FWHT and (sign:=getattr(self, "_rot_sign", None)) is not None:
        kv = (fwht(k, sign) if QSA_KV_ROT & 1 else k).cat(fwht(v, sign) if QSA_KV_ROT & 2 else v, dim=-1)
      else: kv = (k.float() @ rot if QSA_KV_ROT & 1 else k).cat(v.float() @ rot if QSA_KV_ROT & 2 else v, dim=-1)
    return {4: kv_q4_pack, 8: kv_q8_pack}[b](kv) if (b:=self._kv_bits) != 16 else kv.cast(self.cache_k.dtype)
  @property
  def _rot(self) -> Tensor|None:
    """
    QSA_KV_ROT (1: the keys, 2: the values, 3: both): the cache's rows of them rotated (head_dim a Hadamard matrix of random signs: their
    outliers spread over the dims of a group, quantized with less error). the queries rotated alike (the scores the same), the attention's
    output back (the values')
    """
    return getattr(self, "_rot_m", None)

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_k"):
      # (the rows of every sequence's region, see QSAIndexer._init_state)
      KV, N, D = self.config.n_kv_heads, (_region_rows(self.config) + REGION_ALIGN) // self.config.kv_split, 2*self.config.head_dim
      # (zeroed: the flash kernels read whole tiles past the valid region. in int8 groups of 32 and their scales (kv_q8_pack), the bytes of a
      # row 0.56x: more rows in memory. QSA_KV_BITS=4: in 4-bit groups (kv_q4_pack), 0.28x. 16: half)
      bits = getenv("QSA_KV_BITS", 8) if D % 32 == 0 and amd_custom_kernels_supported(x.device) else 16
      assert bits in (4, 8, 16) and (bits != 4 or D % 64 == 0), f"QSA_KV_BITS {bits}"
      W = {4: D//8 + D//64, 8: D//4 + D//32, 16: D}[bits]
      self.cache_k = _zeros(KV, 1, N, W, dtype=dtypes.half if bits == 16 else dtypes.uint32, device=x.device,
                            axis=0 if isinstance(x.device, tuple) else None)
      if QSA_KV_ROT and bits != 16:
        hd = self.config.head_dim
        assert hd & (hd - 1) == 0, "QSA_KV_ROT: a Hadamard matrix of a power of two dims"
        self._rot_m = _rot_dev(hd, x.device)[0]
        if QSA_ROT_FWHT: self._rot_sign = _rot_dev(hd, x.device)[1]
      self.freqs_cis = _freqs(self.config.rope_dim, self.config.max_context, self.config.rope_theta, x.device, self.config.yarn)
    if self.indexer is not None: self.indexer._init_state(x)


class GatedDeltaNetBlock(FFNBlock):
  _qkvg:Linear|None = None  # (GDN_MERGED: q, k, v and the gate's weights as one, see Transformer._derive_weights)
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
    if layer.bias is None and layer.ggml_type is None and R >= 32:
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
    """
    a step of the sequences sharing the caches: a group of rows a sequence, its states in its slot (the projections of all of them in one
    kernel each, the delta rule of all of them in one, see kda_decode)
    """
    sb = cast(SeqBatch, _seq_batch[0])
    x = x.half()
    # (GDN_MERGED: q, k, v and the gate by one kernel (their weights adjacent: merged_linear), no split sums: the sums' order changes)
    if (m:=getattr(self, "_qkvg", None)) is not None and (parts:=q8_linear_parts(m, x, 4)) is not None: projs, gate = parts[:3], parts[3]
    else: projs, gate = [self.attn_q(x), self.attn_k(x), self.attn_v(x)], self._gate(x)
    alpha, beta = self._alpha(x, rows=True), self._few_rows(self.ssm_beta, x)
    weights = [w["weight"] for w in (self.ssm_conv1d_q, self.ssm_conv1d_k, self.ssm_conv1d_v)]
    S, SC, T = int(self.slot_rec[0].shape[0]), int(self.slot_conv[0][0].shape[0]), sb.R
    z = kda_decode(self.pool_rec, self.pool_conv[:3], projs, weights, gate, alpha, beta, self._dt_ch, self.ssm_a,
                   unwrap(self.ssm_norm.weight), Tensor(groups[0][0]), unwrap(self.config.ssm).gate_lower_bound, self.ssm_norm.eps, row=0, rows=T,
                   positions=sb.positions, groups=sb.G, S=S, SC=SC, slots=sb.slots, replay=self.pool_rp, kbases=sb.kbases,
                   replay_q=replay_q(self.config.spec_drafts))
    return _out(self.ssm_out, z)

  def _init_slots(self, n:int, ctx:int, half:bool=False, replay:bool=False):
    """
    the states of n sequences of batched decoding (one slot each): S of each state, the one of a position p is p%S (see slot_states). replay:
    one recurrent state a slot and the replayed inputs (see SEQ_REPLAY)
    """
    if getattr(self, "_slots_n", 0) >= n: return
    self._slots_n = n
    S, SC = slot_states(self.config.spec_drafts)
    if replay:
      S = 1
      K, V = self.head_k_dim, self.head_v_dim
      # (a row of a head: its k, decay, v and beta)
      self.pool_rp = _slot_pool(n, self.recurrent_state, (replay_q(self.config.spec_drafts), *self.recurrent_state.shape[1:3], 2*K + V + 1),
                                dtypes.float32, views=0)[0]
    # (in one buffer of every device: slot_store writes what a step reads)
    # (in float as the model's: a step reads and writes it, in half its rounding adds up over a long reply. in half for many slots
    # (SLOT_STATE_HALF, default past 8): their memory, a long context's model and 32 slots of it don't fit)
    # (slot_rec / slot_conv: the first slot's views only, their shapes read)
    self.pool_rec, self.slot_rec = _slot_pool(n, self.recurrent_state, (S, *self.recurrent_state.shape[1:]), dtypes.half if half else None,
                                              views=1)
    pools = [_slot_pool(n, c, (SC, *c.shape[1:]), views=1) for c in self.conv_states]
    self.pool_conv, self.slot_conv = [p for p, _ in pools], [[slots[0] for _, slots in pools]]

  def _attention(self, x:Tensor, start_pos:int|UOp, T:int|UOp|None=None) -> Tensor:
    if (groups:=_batch_rows[0]) is not None: return self._attention_batch(x, groups)
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
       self.head_k_dim == self.head_v_dim == 128 and amd_custom_kernels_supported(x.device):
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
      # (read with the projection, a value of the sublayer's input: a read of the state alone is of the function's implicit inputs only, made
      # before its call, not after the micro batch before it writes it (see micro_batch_blocks))
      state = state + proj[:, :1].cast(state.dtype) * 0
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
      not symbolic and amd_custom_kernels_supported(x.device)
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

  def _cache_input(self, x:Tensor, start_pos:int|UOp, T:int|UOp) -> Tensor:
    Tq = x.shape[1]
    freqs = self.freqs_cis.shrink(((start_pos, start_pos + T), None)).pad_to((Tq, self.freqs_cis.shape[-1]))
    k, v = self._kv(x, *(f.reshape(1, 1, Tq, -1) for f in freqs.chunk(2, dim=-1)))
    cache = self._cache_store(k, v, start_pos, T)
    if self.indexer is None: return cache
    return Tensor(cache.uop.after(self.indexer.update(x, start_pos, T, self.freqs_cis, gather=False)[0].uop))

  def head(self, streams:Tensor) -> Tensor:
    """the streams' mix: the output head's input"""
    return self.hc_head.mixed(streams)

class Transformer:
  _embd_q8:Tensor|None = None  # (the token embedding's Q8_0 bytes in host memory, see _embed)
  _weight_bytes:dict[str, int] = {}  # (the weights' memory of every device, see from_gguf)
  def __init__(self, config:TransformerConfig):
    dense_config = replace(config, num_experts=0, num_experts_per_tok=0, shared_expert_dim=0, hidden_dim=config.dense_hidden_dim or config.hidden_dim)
    if config.ssm: config = replace(config, qk_norm=config.head_dim)
    self.config = config
    self.blk:list[FFNBlock] = []
    for i in range(config.num_blocks):
      c = dense_config if i < config.leading_dense_blocks else config
      if config.sliding_window_pattern != 0 and (i+1) % config.sliding_window_pattern == 0: c = replace(c, sliding_window=0)
      if config.indexer_shared and config.indexer_shared[i]: c = replace(c, indexer=None)
      if config.ple is not None and i != config.ple.layer: c = replace(c, ple=None)
      self.blk.append(GatedDeltaNetBlock(c, config.ssm) if config.ssm and config.ssm_layers[i] else QSABlock(c))
    self.token_embd  = nn.Embedding(config.vocab_size, config.dim)
    # (gated hyper-connections: the last mix of the streams norms them)
    self.gated, self.output_norm, self.output_hc = True, _identity, GatedResidual(config, inject=False)
    self.output = Linear(config.dim, config.padded_vocab_size or config.vocab_size, bias=False)
    if config.mtp: self.mtp = QwenMTPBlock(config)
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
    self._loaded = False  # the weights are loaded (from_gguf)
    self._pending_load:list[Callable[[], None]] = []  # the work left of loading (from_gguf with LOAD_DEFER)
    self._scratch:dict = {}  # the memory of the jits' intermediates: they run one at a time

  @property
  def _spec_dim(self) -> int:
    """the width of a position's hidden state the MTP layer reads: the streams (gated hyper-connections), else the normed hidden state"""
    return unwrap(self.config.hc).count * self.config.dim if self.gated else self.config.dim

  def _hc_out(self, x:Tensor) -> Tensor:
    """the streams (B, T, count, dim) into the output's hidden state"""
    return self.output_hc.mixed(x) if self.gated else x.mean(2)

  # ** the n-gram embeddings (qwen4exp): the rows of every position's n-grams hashed from the tokens before it and gathered on the devices,
  # from the table (in the gguf file) in host memory they read
  @property
  def _ple_rb(self) -> int:
    """the bytes of a packed row of a head's table"""
    ple = unwrap(self.config.ple)
    return ple.head_dim // _GGML_QUANT[ple.ggml_type][0] * _GGML_QUANT[ple.ggml_type][1]

  def _ple_dev_init(self, read:bool=True):
    """
    the table in host memory of the devices (the heads' tables views of it: their rows' bytes are 32 bit offsets), read from the file (direct:
    not kept in the page cache besides, the table is 29GB) or copied from its bytes. read=False: its buffers only, the memory made and read by
    _ple_dev_read (jits that read it can capture before)
    """
    if hasattr(self, "_ple_heads"): return
    ple, devs = unwrap(self.config.ple), to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device))
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

  def _ple_dev_read(self):
    """the table read into its host memory (see _ple_dev_init): from the file (io_uring, O_DIRECT, see direct_read), or copied from its bytes"""
    if not hasattr(self, "_ple_heads"): self._ple_dev_init(read=False)
    if getattr(self, "_ple_read", False): return
    self._ple_dev_alloc()
    self._ple_read, table, loc, mv, pre = True, unwrap(self.ple_table), self._ple_file, self._ple_mv, self._ple_pre
    if loc is None:
      mv[:table.nbytes] = table
      return
    f = DirectFile(loc[0])
    try: read_many([(f, loc[1] - pre, mv[:min(len(mv), f.size - (loc[1] - pre))])])
    finally:
      f.close()
      close_rings()

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

  # ******** PREFILL_PARTS (as glm5next's): a prefill chunk's jit in parts of the blocks, jits captured at once in processes of their own (a
  # jit's capture is ~30 s of python for all the blocks). the streams of the micro batches between parts in buffers of the model (in half, as
  # the blocks keep them) ********

  def _prefill_mb(self, rows:int) -> int: return 1 if rows <= PREFILL_MB1_ROWS else MICRO_BATCHES
  def prefill_parts(self, rows:int) -> list[tuple[int, int]]:
    """the blocks of the parts of a prefill chunk of rows (one part: not split)"""
    k, n, mb = PREFILL_PARTS, len(self.blk), self._prefill_mb(rows)
    if k <= 1 or self.config.hc is None or not amd_custom_kernels_supported(self.token_embd.weight.device) or \
       not (self.sequence_parallel(rows) and self.sequence_parallel(rows // mb) and rows % mb == 0): return [(0, n)]
    return [(i*n//k, (i+1)*n//k) for i in range(k)]

  def _carry(self, rows:int) -> list[Tensor]:
    """the streams of every micro batch of a prefill chunk of rows between its parts (made before the jits are captured): views of one buffer a
    micro batch, of the most rows of the chunk sizes (the chunks run one at a time; a part writes them before the next reads them)"""
    carry = self.__dict__.setdefault("_pf_carry", {})
    if rows not in carry:
      hc, mb, devs = unwrap(self.config.hc), self._prefill_mb(rows), cast(tuple[str, ...], self.token_embd.weight.device)
      # (the elements a device has of a micro batch's stream of a chunk of r rows)
      def local(r:int) -> int: return r // self._prefill_mb(r) // len(devs) * hc.count * self.config.dim
      if (bases:=self.__dict__.get("_pf_carry_bases")) is None:
        sizes = [r for r in self.prefill_chunks() if len(self.prefill_parts(r)) > 1] + [rows]
        n, lanes = max(local(r) for r in sizes), max(self._prefill_mb(r) for r in sizes)
        bases = self._pf_carry_bases = [Tensor.zeros(len(devs) * n, dtype=dtypes.half).shard(devs, axis=0).contiguous().realize()
                                        for _ in range(lanes)]
      def view(base:Tensor) -> Tensor:
        (m:=MultiBuffer.__new__(MultiBuffer)).bufs = [b.view(local(rows) * 2, 0).ensure_allocated()
                                                      for b in cast(MultiBuffer, base.uop.base.buffer).bufs]
        return Tensor(UOp.from_buffer(m, dtypes.half, devs).reshape(1, rows // mb // len(devs), hc.count, self.config.dim).unshard(1))
      carry[rows] = [view(bases[k]) for k in range(mb)]
    return carry[rows]

  def _new_prefill_jit(self, rows:int) -> Any:
    """the jit of a prefill chunk of rows (a list of the parts' jits: PREFILL_PARTS)"""
    if len(parts:=self.prefill_parts(rows)) == 1: return TinyJit(self._forward_effects, capture_first=True, scratch=self._scratch)
    self._carry(rows)
    return [TinyJit(functools.partial(self._forward_effects, part=p), capture_first=True, scratch=self._scratch) for p in range(len(parts))]

  def _forward(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor|None, base:int|UOp, part:int|None=None) -> Tensor:
    # (a view of the token buffer from a variable base: copied, the lookup of the embedding's rows of a view at two variables is a reduce over
    # the vocabulary)
    x = emb = self.token_embd((tokens.contiguous() if isinstance(base, UOp) else tokens).to(self.token_embd.weight.device)).float()  # (B, T, D)
    if (hc:=self.config.hc) is not None: x = x.unsqueeze(2).expand(*x.shape[:2], hc.count, x.shape[-1]).contiguous()
    ple = None
    if (pc:=self.config.ple) is not None:
      # (the tokens of the rows and the n-1 before them from the token buffer, EOS before the start)
      buf = getattr(self, "_token_buf", None)
      assert buf is not None and tokens.uop.base is buf.uop.base, "n-gram rows: the tokens of the token buffer"
      ple = self._ple_emb(self._ple_rows_dev(self._token_window(buf, base, start_pos, int(x.max_shape[1]) + len(pc.multipliers)-1)).unsqueeze(0))
    # a symbolic chunk with hyper-connections runs every block on static padded rows (their real count is T), sliced at the end
    if hc is not None and not isinstance(T:=x.shape[1], int) and amd_custom_kernels_supported(x.device) and \
       all(isinstance(b, (GatedDeltaNetBlock, QSABlock)) for b in self.blk):
      x = x.pad_to(x.max_shape).contiguous()
      R = int(x.shape[1])
      # sequence parallel: every device keeps the streams of its rows, the blocks gather and scatter the sublayers' inputs and outputs
      # micro batches (on queues of their own, AM_MEC_PIPES: the copies of one run with the kernels of the other)
      # (PREFILL_MB1_ROWS: a chunk of this many rows or fewer in one batch: a small chunk's collectives are latency, not hidden by compute)
      mb = self._prefill_mb(R)
      if (sp:=self.sequence_parallel(R)) and self.sequence_parallel(R // mb) and R % mb == 0:
        # the blocks in phases (micro_batch_blocks), on micro batches of rows (the copies of one overlap the kernels of the next). a batch
        # starts at the end of the tokens before it (all padding if there are none): its states follow theirs
        n, xs, ts, starts = R // mb, [], [], []
        a, b = self.prefill_parts(R)[part or 0]
        for i in range(mb):
          if not a: xs.append(shard_rows(x[:, i*n:(i+1)*n].contiguous(), 1))
          ts.append((T - i*n).maximum(0).minimum(n))
          starts.append(start_pos + (T.minimum(i*n) if i else 0))
        if a: xs = list(self._carry(R))
        xs = micro_batch_blocks(self.blk[a:b], xs, starts, ts, ple)
        if b < len(self.blk):
          # (not the last part: the streams to the model's buffers, the output one of them)
          outs = [Tensor(c.uop.after(c.uop.store(xi.cast(c.dtype).uop))) for c, xi in zip(self._carry(R), xs)]
          return functools.reduce(lambda a, o: a + o[:, :, :1, :1].float(), outs[1:], outs[0][:, :, :1, :1].float()).contiguous()
        if self.gated:
          # (the MTP layer's input of every device's rows from its own (a row the streams of the one before: the devices' last rows gathered),
          # the last row's streams picked: not every row's streams gathered)
          if self.config.spec_drafts: x = self._mtp_prefill_rows(emb, xs, cast(UOp, start_pos), T, n)
          else: x = Tensor.cat(*[_on_lane(i, all_gather, xi, 1) for i, xi in enumerate(xs)], dim=1).float()
        else: x = Tensor.cat(*[_on_lane(i, all_gather, xi.mean(2), 1) for i, xi in enumerate(xs)], dim=1)
      else:
        # (gated hyper-connections split their rows in the phases only: without them every device has all the rows. their MTP layer's prefill
        # is the phases')
        assert not (self.gated and self.config.spec_drafts), "the MTP layer of gated hyper-connections prefills sequence parallel"
        if sp and self.gated: sp = False
        if sp: x = shard_rows(x, 1)
        for block in self.blk: x = block(x, start_pos, T, ple_emb=ple)
        if not self.gated: x = all_gather(x.mean(2), 1) if sp else self._hc_out(x)
      if x.shape[1] != 1: x = x[:, :T]
    else:
      for block in self.blk: x = block(x, start_pos, ple_emb=ple)
    if hc is not None and x.ndim == 4: x = self._hc_out(x[:, -1:])
    # Only the last token needs the final stream mix and vocabulary projection.
    tok = self._sample(self.output_norm(x[:, -1:])[:, -1, :], temperature, tokens.device)
    if (lp:=getattr(self, "_lp_buf", None)) is not None:
      # (QSA_LP (extra/qwen_kv_quality.py): the logits of the last token into _lp_buf, its effect)
      eff = lp.assign(self.output(self.output_norm(x[:, -1:])[:, -1, :]).float())
      if self._effects is None: eff.realize()
      else: self._effects.append(eff)
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

  def _embed(self, tokens:Tensor) -> Tensor:
    """the embeddings (..., dim) in float of tokens on the devices: a Q8_0 table in host memory read by its rows (not the generic gather)"""
    if (raw:=getattr(self, "_embd_q8", None)) is not None and amd_custom_kernels_supported(tokens.device) and isinstance(tokens.numel(), int):
      return q8_embed(raw, tokens.reshape(-1), self.config.dim).reshape(*tokens.shape, self.config.dim)
    return self.token_embd(tokens).float()

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
      if tp.shape[1] == 2: logits = (nucleus(z, self.config.vocab_size, tp[:, 1:]) | hot.logical_not()).where(logits, -math.inf)
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

  def _mtp_prefill_rows(self, emb:Tensor, xs:list[Tensor], start_pos:UOp, T:UOp, n:int) -> Tensor:
    """_mtp_prefill (gated) of the micro batches' streams xs (1, n, count, dim) split on the rows. returns (1, T, count, dim) whose last row is
    the last row's streams (the others zeros: only the last one is read)"""
    qm, hc, mb = self.mtp, unwrap(self.config.hc), len(xs)
    W = hc.count * self.config.dim
    flat = [x.reshape(n, W) for x in xs]
    prev = rows_shifted(flat, self._spec_h[0])
    embp = emb.pad_to((1, n*mb, emb.shape[-1]))
    mixed = []
    for i in range(mb):
      e_l = shard_rows(embp[:, i*n:(i+1)*n].contiguous(), 1)
      xs_l = qm._project_input_split(e_l, prev[i].reshape(1, n, W))
      qm._init_state(xs_l)
      mixed.append(_on_lane(i, all_gather, qm.hc_attn.pre_split(xs_l)[0], 1, dtypes.half))
    m = qm._cache_input(Tensor.cat(*mixed, dim=1).float(), start_pos, T)
    last = rows_pick(flat, T - 1, T, SEL_LANE)
    effects = [m[:, :, :1, :1].contiguous(), self._spec_h[0:1].assign(last)]
    if self._effects is None: Tensor.realize(*effects)
    else: self._effects += effects
    return last.reshape(1, 1, hc.count, self.config.dim)

  def _wait_host(self, get:Callable[[], Any], what:str) -> Any:
    """the value a step writes to host memory (polled: -1 until then). a step that doesn't come (the devices' errors): an error, not a hang"""
    st = time.perf_counter()
    while isinstance(v:=get(), int) and v < 0:
      if time.perf_counter() - st > 10:
        # (the devices' timelines: done / submitted, the ones behind the others are where it hangs)
        devs = to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device))
        print(f"{what}: none after 10 s, timelines " + " ".join(f"{d}={(tl:=Device[d].timeline.host.view(fmt='Q'))[0]}/{tl[1]}" for d in devs),
              flush=True)
        for d in devs:
          if (rep:=getattr(getattr(Device[d], "iface", None), "hang_report", None)) is not None:
            with contextlib.suppress(Exception): print("  " + rep(), flush=True)
        if (err:=ring_errors()): print(f"  ring exchanges timed out (RING_TIMEOUT_MS): {err}", flush=True)
        for d in devs: Device[d].synchronize()
        if isinstance(v:=get(), int) and v < 0: raise RuntimeError(f"{what}: none after the devices finished")
        break
    # (RING_TIMEOUT_MS: an exchange a wave gave up on made garbage of the values after it: an error)
    if (err:=ring_errors()): raise RuntimeError(f"{what}: a ring exchange timed out (RING_TIMEOUT_MS): {err}")
    return v

  def _ints(self, xs:tuple[UOp, ...]) -> Tensor:
    """variables as an int32 tensor of the devices (a stack of variables has no device: onto zeros of the devices)"""
    devs = self.token_embd.weight.device
    return (Tensor.zeros(len(xs), dtype=dtypes.int32, device=devs) + Tensor.stack(*[Tensor(x) for x in xs]).cast(dtypes.int32)).contiguous()

  # ******** sequences sharing the caches (qwen4exp): every one its region of the attention caches' rows (the token buffer's too) and a slot of
  # the recurrent states. a sequence is prefilled by the model's calls on its region (its states loaded from its slot before, stored after),
  # a step of speculative decoding takes every sequence decoding at once (forward_seqs). the regions are the caller's (see SeqEngine) ********

  REGION_ALIGN = REGION_ALIGN  # expose the model's move/stash alignment to SeqEngine rather than importing qwen4exp there
  SEQ_BASE_ALIGN = SEQ_BASE_ALIGN  # region allocation may be packed more tightly than whole moves/stash chunks
  SEQ_SIZES = tuple(int(x) for x in str(getenv("SEQ_SIZES", "1,2,4,8")).split(","))
  SEQ_MIX_TOKENS = SEQ_MIX_TOKENS
  def clock_caps(self, default:dict[str, str]) -> dict[str, str]:
    """
    the gfx clock caps of the server's modes (fast: few users, many, prefill). RDNA4 (R9700): none, its board's power limit is the cap (at
    2000 MHz 48 users are 15% slower for 22% less energy a token)
    """
    return {mode: "" for mode in default} if _wmma_rdna4(self.token_embd.weight.device) else default
  def seqs_supported(self) -> bool:
    devs = self.token_embd.weight.device
    return bool(self.config.qsa and self.gated and self.config.spec_drafts and self.config.indexer is not None and isinstance(devs, tuple) and
                amd_custom_kernels_supported(devs))
  def seq_slots(self) -> int:
    """the slots of the sequences (the one after them is the padding sequences' of a step)"""
    return getenv("SEQ_SLOTS", max(self.SEQ_SIZES))
  def _token_rows(self) -> int:
    """the token buffer: a token of every row of the caches (the regions of the sequences sharing them), and one past"""
    return _region_rows(self.config) + REGION_ALIGN + 1
  def seq_rows(self) -> int:
    """the rows of the caches the regions of the sequences are of (the REGION_ALIGN after them: the padding sequences' region)"""
    return _region_rows(self.config)

  def _seq_init(self):
    if hasattr(self, "seq_spec_h"): return
    if _zero_later: return self._seq_init_states()
    # (the states zeroed by one jit (a kernel a buffer), not by a jit's call a chunk of every buffer)
    with _zero_deferred() as zero_bufs: self._seq_init_states()
    _zero_jit(dedup(zero_bufs))[1]()
    _lap("seq init: zeroed by a jit")

  def _seq_init_states(self):
    # (the buffers freed by loading are cached for reuse: their memory is the caches')
    for d in to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device)): Device[d].allocator.free_cache()
    _lap("seq init: cache freed")
    self._init_states()
    _lap("seq init: states")
    k, n, devs, m = self.config.spec_drafts, self.seq_slots() + 1, cast(tuple[str, ...], self.token_embd.weight.device), max(self.SEQ_SIZES)
    # (the slots' recurrent states in float, SEQ_STATE_HALF: in half)
    for b in self.blk:
      if isinstance(b, GatedDeltaNetBlock): b._init_slots(n, 0, half=False, replay=True)
    for idx in self._indexers(): idx._init_slots(n)
    # every slot's hidden states (the streams) of the last step's tokens, two parities of k+1
    self.seq_spec_h = _zeros(n*2*(k+1), self._spec_dim, device=devs)
    # every slot's checkpoint: its states of a position (the end of its last prompt), the hidden state of the last token
    self.seq_ckpt_h = _zeros(n, self._spec_dim, device=devs)
    # (the checkpoints' recurrent states in host memory: a copy a turn, the device memory more slots')
    self.seq_ckpt_states = [[_slot_pool(n, b.pool_rec, (1, *b.pool_rec.shape[1:]), host=True, views=0)[0]] +
                            [_slot_pool(n, p, (1, *p.shape[1:]), views=0)[0] for p in b.pool_conv]
                            for b in self.blk if isinstance(b, GatedDeltaNetBlock)]
    # (and the indexers' states)
    self.seq_ckpt_idx = [_slot_pool(n, idx.pool_state, (1, *idx.pool_state.shape[1:]), views=0)[0] for idx in self._indexers()]
    self._spec_init()
    if not hasattr(self, "_token_buf"): self._token_host, self._token_buf = _host_ints(devs, self._token_rows())
    # in host memory: a step's input tokens, the temperature and top_p of every sequence, its sampled tokens and drafts (written by one device)
    self._seq_host_in, self._seq_in = _host_ints(devs, m*(k+1))
    # (a mixed step's prompt's at 2m, the step's seed last)
    self._seq_m = m
    tmv, self._seq_temp = _host_buffer(devs, 8*m + 12, dtypes.float32)
    self._seq_host_temp = tmv.cast('f')
    # (a prefill's temperature and top_p: in host memory, a new value isn't a new tensor (its realize walks every tensor of the model))
    tmv, self._seq_pf_temp = _host_buffer(devs, 8, dtypes.float32)
    self._seq_pf_host_temp = tmv.cast('f')
    self._seq_host_out, self._seq_out = _host_ints(devs[0], m*(2*k+1) + 1)
    if QTF or SEQ_MIX_TOKENS:
      # (QTF / SEQ_MIX_TOKENS: forced drafts and step tokens from the host; QTF also returns every row's log probs)
      self._seq_host_tf, self._seq_tf = _host_ints(devs, m*(2*k+1))
    if QTF:
      V = self.config.vocab_size
      tmv, self._seq_tfo = _host_buffer(devs[0], 4*m*(k+1)*V, dtypes.float32)
      self._seq_host_tfo = tmv.cast('f')
    self._seq_copy_jits = {sv: TinyJit(functools.partial(self._seq_copy, sv), capture_first=True) for sv in (True, False)}
    self._seq_ckpt_jits = {sv: TinyJit(functools.partial(self._seq_ckpt_copy, sv), capture_first=True) for sv in (True, False)}
    # (SEQ_REPLAY: a slot's recurrent states advanced to a position, see _seq_materialize; the position of every slot's state)
    self._seq_mat_jit = TinyJit(self._seq_mat, capture_first=True)
    self._seq_first_jit = TinyJit(self._seq_first, capture_first=True)
    self._seq_kbase:dict[int, int] = {}
    # (the calls' chunk counts that fit the caches: a short max context's are a few chunks)
    self._move_calls = self._chunk_calls((self.seq_rows() + REGION_ALIGN) // self.SEQ_MOVE - 1)
    self._seq_move_jits = {m: TinyJit(functools.partial(self._seq_move_copy, m), capture_first=True) for m in self._move_calls}
    _lap("seq init: slots")
    self._seq_stash_init()
    _lap("seq init: stash")
    todo = [self.seq_spec_h, self.seq_ckpt_h] + [p for b in self.blk if isinstance(b, GatedDeltaNetBlock) for p in (b.pool_rec, *b.pool_conv)] + \
      [b.pool_rp for b in self.blk if isinstance(b, GatedDeltaNetBlock) and hasattr(b, "pool_rp")] + \
      [t for c in self.seq_ckpt_states for t in c] + [idx.pool_state for idx in self._indexers()] + self.seq_ckpt_idx
    _fill_zero(*todo)
    _lap("seq init: zeroed")

  def _seq_copy(self, save:bool, slot:UOp, pm:UOp, ps:UOp, pc:UOp, hrow:UOp, keep:UOp):
    """
    the recurrent states of the model of a position (its rows pm) into slot `slot` (its rows ps of the recurrent states, pc of the conv ones) and
    the hidden state of the last token (the model's row 0) into the slot's row hrow (save), or back (the hidden state times keep: 0 none)
    """
    def rows(t:Tensor, start:UOp|int) -> Tensor: return t.shrink(((start, start+1),) + (None,)*(t.ndim-1))
    H = 2*(self.config.spec_drafts+1)
    pairs:list[tuple[Tensor, Tensor]] = [(rows(self.seq_spec_h, slot*H + hrow), rows(self._spec_h, 0))]
    for b in self.blk:
      if isinstance(b, GatedDeltaNetBlock):
        S, SC = int(b.slot_rec[0].shape[0]), int(b.slot_conv[0][0].shape[0])
        pairs.append((rows(b.pool_rec, slot*S + ps), rows(b.recurrent_state, pm)))
        pairs += [(rows(d, slot*SC + pc), rows(c, pm)) for d, c in zip(b.pool_conv, b.conv_states)]
    pairs += [(rows(idx.pool_state, slot*idx.SC + pc), rows(idx.state, pm)) for idx in self._indexers()]
    if save: Tensor.realize(*[d.assign(s.cast(d.dtype)) for d, s in pairs])
    else:
      (h, hs), rest = pairs[0], pairs[1:]
      Tensor.realize(hs.assign(h.contiguous() * Tensor(keep).cast(dtypes.float)), *[s.assign(d.cast(s.dtype)) for d, s in rest])

  def _indexers(self) -> list[QSAIndexer]:
    """the indexers of the attention blocks (and the MTP layer's)"""
    return [b.indexer for b in self.blk + ([self.mtp] if hasattr(self, "mtp") else []) if isinstance(b, QSABlock) and b.indexer is not None]

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
    if save: self._seq_kbase[slot] = pos
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
      if not isinstance(b, QSABlock): continue
      S = self.config.kv_split
      copies.append(mv(b.cache_k, src // S, dst // S, n // S, 2))
      if (idx:=b.indexer) is not None:
        kp = idx.idx.kpool
        # (split: every device's rows, (devices, rows, D) split on the devices)
        kn, pk = kp * idx.split, idx.cache_pk if idx.split == 1 else idx.cache_pk.reshape(idx.split, -1, idx.idx.head_dim)
        copies.append(mv(pk, src // kn, dst // kn, n // kn, 0 if idx.split == 1 else 1))
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
    qsa = [b for b in self.blk + [self.mtp] if isinstance(b, QSABlock)]
    def per_row(t:Tensor, axis:int) -> int:
      # (a device's bytes of a row: split over the devices or not)
      return t.nbytes() // int(t.shape[axis]) // (len(t.device) if isinstance(t.device, tuple) and t.uop.axis is not None else 1)
    # (the host memory of a chunk: every distinct KV head's rows and the indexers' (see below))
    kvr = self._kv_real
    chunk = C * sum(kvr * per_row(b.cache_k, 2) + (per_row(b.indexer.cache_pk, 0) // b.indexer.idx.kpool if b.indexer is not None else 0)
                    for b in qsa)
    gdn = [b for b in self.blk if isinstance(b, GatedDeltaNetBlock)]
    # (an entry: a copy on every device of its part of the states)
    entry = len(to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device))) * \
      (sum(per_row(t, 0) for b in gdn for t in (b.pool_rec, *b.pool_conv)) + per_row(self.seq_spec_h, 0) +
       sum(per_row(t, 0) for c in self.seq_ckpt_states for t in c) + per_row(self.seq_ckpt_h, 0) +
       sum(per_row(t, 0) for t in [idx.pool_state for idx in self._indexers()] + self.seq_ckpt_idx))
    self.stash_chunk_nbytes, self.stash_entry_nbytes = chunk, entry  # exact host-copy units for serving telemetry
    # (the conversations the stash holds at most: SEQ_STASH_ENTRIES, by default as many as fit of SEQ_STASH_CTX tokens each (an agent's
    # conversation, 32k), at least 8)
    ctx = getenv("SEQ_STASH_CTX", 32768)
    self.stash_entries = max(8, getenv("SEQ_STASH_ENTRIES", int(gb * 1e9) // (entry + chunk * ctx // C))) if gb > 0 else 0
    self.stash_chunks = max(0, int(gb * 1e9 - self.stash_entries * entry) // chunk) if gb > 0 else 0
    print(f"stash: {self.stash_chunks} chunks of {C} rows ({chunk/1e6:.1f} MB), {self.stash_entries} entries ({entry/1e6:.1f} MB)")
    if not self.stash_chunks: self.stash_entries = 0
    if not self.stash_entries: return
    H, E = self.stash_chunks * C, self.stash_entries
    # (the attention caches: a copy of every distinct one in host memory every device maps, written by one device: the KV heads repeated over
    # the devices are the same, the indexers' caches are on every device)
    devs, kvr, S, n = cast(tuple[str, ...], self.token_embd.weight.device), self._kv_real, self.config.kv_split, self.config.idx_split
    N, head = len(devs), [d * kvr // len(devs) for d in range(len(devs))]
    # (the devices of a copy write their part of its rows: SEQ_STASH_SPLIT of them at most)
    def code(group:list[list[int]]) -> int:
      ret = 0
      for g in group:
        g = g[:3]
        for i, d in enumerate(g): ret |= (i | (len(g) << 4)) << (8*d)
      return ret
    # (a cache split over the devices (kv_split, idx_split): a copy of every device's rows, written by it)
    self._stash_writers = code([[d] for d in range(N)] if S > 1 else [[d for d in range(N) if head[d] == h] for h in range(kvr)]), \
      code([[d] for d in range(N)] if n > 1 else [list(range(N))])
    # the arenas: every device's (split caches), every KV head's, the indexers' (on every device)
    owners = list(devs) + [devs[head.index(h)] for h in range(kvr)] + [devs[0]]
    kv_arena = list(range(N)) if S > 1 else [N + h for h in head]
    pk_arena = list(range(N)) if n > 1 else [N + kvr] * N
    kv_specs = [[(b.cache_k, H // S, kv_arena)] + ([(b.indexer.cache_pk, H // (b.indexer.idx.kpool * n), pk_arena)] if b.indexer is not None else [])
                for b in qsa]
    def local_of(t:Tensor, rows:int) -> tuple[int, ...]:
      return (rows, *[int(x) for x in t.shape[3 if t.ndim == 4 else 1:]])
    sizes = [0] * len(owners)
    for b in kv_specs:
      for t, rows, ar in b:
        for a in set(ar): sizes[a] += round_up(prod(local_of(t, rows)) * t.dtype.itemsize, 4096)
    arenas = [_shared_host(o, devs, sz) for o, sz in zip(owners, sizes)]
    offs = [0] * len(arenas)
    def take(t:Tensor, rows:int, arena_of_dev:list[int]) -> Tensor:
      local = local_of(t, rows)
      nb, views = prod(local) * t.dtype.itemsize, []
      for d in range(N): views.append(arenas[arena_of_dev[d]][d].view(nb, offs[arena_of_dev[d]]))
      for a in set(arena_of_dev): offs[a] += round_up(nb, 4096)
      (mb:=MultiBuffer.__new__(MultiBuffer)).bufs = views
      u = UOp.from_buffer(mb, t.dtype, devs).reshape((1, 1, *local) if t.ndim == 4 else local)
      return Tensor(u.unshard(0) if t.ndim == 4 or (n > 1 and arena_of_dev is pk_arena) else u)
    self._stash_kv = [tuple(take(t, rows, ar) for t, rows, ar in b) for b in kv_specs]
    # (a sequence's states of its last position: the recurrent and conv ones, the hidden state of its last token. and its checkpoint's)
    states = [t for b in gdn for t in (b.pool_rec, *b.pool_conv)] + [self.seq_spec_h] + \
             [t for c in self.seq_ckpt_states + [[self.seq_ckpt_h]] for t in c] + [idx.pool_state for idx in self._indexers()] + self.seq_ckpt_idx
    self._stash_state = _host_like([(t, 0, E) for t in states])
    self._stash_calls = self._chunk_calls(min((self.seq_rows() + REGION_ALIGN) // C - 1, self.stash_chunks))
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
    for b, st in zip([b for b in self.blk + [self.mtp] if isinstance(b, QSABlock)], self._stash_kv):
      S = self.config.kv_split
      ts = [(st[0], b.cache_k, row // S, chunk*(C // S), (C // S)*chunks, self._stash_writers[0])]
      if (idx:=b.indexer) is not None:
        kp = idx.idx.kpool
        kn = kp * idx.split
        ts.append((st[1], idx.cache_pk, row//kn, chunk*(C//kn), (C//kn)*chunks, self._stash_writers[1]))
      for dst, src, r, c, n, writers in ts:
        axis = 2 if src.ndim == 4 else 0
        if save:
          # (a device of every distinct copy writes it: the host memory takes a few GB/s of the devices' writes at once)
          # (the row and chunk: bound on an input, the kernel's variables)
          s = Tensor(src.uop.after(r, c))
          saves.append(Tensor.custom_kernel(dst, s, _device_index(devs), fxn=functools.partial(_stash_save_kernel, row=_unbind(r), at=_unbind(c),
                                                                                              n=n, writers=writers))[0])
        elif src.ndim == 2 and src.uop.axis == 0:
          # (split on its rows: every device's, (devices, rows, D))
          N = len(devs)
          pairs.append((rows(dst.reshape(N, -1, int(dst.shape[-1])), c, n, 1), rows(src.reshape(N, -1, int(src.shape[-1])), r, n, 1)))
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
    # (the indexers' states of the position, and of the checkpoint)
    slots += [rows(idx.pool_state, slot*idx.SC + pc) for idx in self._indexers()] + [rows(t, slot) for t in self.seq_ckpt_idx]
    pairs = [(rows(st, entry), x) for st, x in zip(self._stash_state, slots)]
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
    else: self._seq_kbase[slot] = pos
    self._stash_state_jits[save](*self._stash_state_vars(slot, entry, pos, hrow))

  def _seq_ckpt_copy(self, save:bool, slot:UOp, ps:UOp, pc:UOp):
    """the checkpoint of slot `slot`: its recurrent states (rows ps, the conv ones pc) and its hidden state of row 0 (save), or back"""
    def rows(t:Tensor, start:UOp|int) -> Tensor: return t.shrink(((start, start+1),) + (None,)*(t.ndim-1))
    H = 2*(self.config.spec_drafts+1)
    pairs:list[tuple[Tensor, Tensor]] = [(rows(self.seq_ckpt_h, slot), rows(self.seq_spec_h, slot*H))]
    for b, c in zip([b for b in self.blk if isinstance(b, GatedDeltaNetBlock)], self.seq_ckpt_states):
      S, SC = int(b.slot_rec[0].shape[0]), int(b.slot_conv[0][0].shape[0])
      pairs += [(rows(c[0], slot), rows(b.pool_rec, slot*S + ps))] + [(rows(d, slot), rows(p, slot*SC + pc)) for d, p in zip(c[1:], b.pool_conv)]
    pairs += [(rows(c, slot), rows(idx.pool_state, slot*idx.SC + pc)) for c, idx in zip(self.seq_ckpt_idx, self._indexers())]
    Tensor.realize(*[(d.assign(s) if save else s.assign(d)) for d, s in pairs])
  def _seq_S(self) -> tuple[int, int]:
    """the S of a sequence slot's recurrent states (one with SEQ_REPLAY) and of its conv states"""
    S, SC = slot_states(self.config.spec_drafts)
    return 1, SC
  def seq_ckpt(self, save:bool, slot:int, pos:int):
    """the checkpoint of slot `slot`: its states of position pos (and the hidden state of row 0) saved, or restored"""
    S, SC = self._seq_S()
    if save: self._seq_materialize(slot, pos)
    else: self._seq_kbase[slot] = pos
    self._seq_ckpt_jits[save](UOp.variable("seq_slot", 0, self.seq_slots()).bind(slot), UOp.variable("seq_ps", 0, S-1).bind(pos % S),
                              UOp.variable("seq_pc", 0, SC-1).bind(pos % SC))

  def _prefill_jit(self, n:int, lp:int|None=None) -> Any:
    """the jit of a prefill chunk of n tokens on a sequence's region (lp: of a context of that many pools a device, see _pk_bucket), see __call__"""
    jits, key = self.__dict__.setdefault("jits_seq", {}), n if lp is None else (n, lp)
    if key not in jits: jits[key] = self._new_prefill_jit(n)
    return jits[key]

  def _pk_buckets(self, n:int) -> list[int]:
    """the pools a device of the contexts (PREFILL_PK_CTX) a prefill chunk of size n is traced for (split indexer, not the largest size)"""
    if (idx:=unwrap(self.config.indexer)).top_k >= self.max_context or (s:=self.config.idx_split) == 1 or n >= self.prefill_chunks()[-1]: return []
    # (whole tiles of the scores: n*lp a multiple of 128)
    lps = [round_up(-(-c // idx.kpool // s), 128 // math.gcd(s, 128)) for c in PREFILL_PK_CTX if c < self.max_context]
    return [lp for lp in dedup(lps) if lp < -(-self.max_context // idx.kpool // s)]

  def _pk_bucket(self, n:int, end:int) -> int|None:
    """the pools a device of the jit of a prefill chunk of size n ending at position end (None: a whole context's)"""
    kp, s = unwrap(self.config.indexer).kpool, self.config.idx_split
    return next((lp for lp in self._pk_buckets(n) if lp * s >= -(-end // kp)), None)

  def _seq_shared(self, read:bool=True):
    """the resources the jits share, made before they're captured (or loaded): the random state of the devices, the n-gram tables, the
    collectives' mailboxes, the argmax's counts"""
    devs = cast(tuple[str, ...], self.token_embd.weight.device)
    # (one realize: one walks every tensor)
    Tensor.realize(*[x for d in devs for x in (*Tensor._next_counter(d, 0), Tensor._device_rng_counters[d])])
    if self.config.ple is not None: self._ple_dev_init(read)
    _lap("shared: ple")
    # (a lane of the collectives a micro batch, and its compute queue (MB_QUEUES), see micro_batch_blocks. their state copied in at once)
    # (LANES_LATE, in seq_warmup: their memory made now, their state copied in first in the start-up's window, not before the forks)
    with init_deferred(self.__dict__.setdefault("_lanes_init", []) if not read and LANES_LATE else None): self._seq_lanes(devs)
    _lap("shared: lanes")
    _device_index(devs)
    _argmax_parts(devs, (self.config.padded_vocab_size or self.config.vocab_size) // len(devs), self.config.vocab_size)

  def _seq_lanes(self, devs:tuple[str, ...]):
    for lane in range(MICRO_BATCHES):
      if mb_queue(lane):
        for d in devs: getattr(Device[d], "compute_queue_n", lambda i: None)(mb_queue(lane))
      # (and the decoding steps' lanes, see DEC_LANE)
      for ln in (lane, lane + SEL_LANE) + ((DEC_LANE, DEC_LANE + SEL_LANE) if lane == 0 else ()):
        FLAG_LANE[0] = ln
        for g in (4, 16, 64): flag_allreduce(devs, g)
        # (the ring's mailbox: made before the jits are captured (in a fork, it'd be the child's))
        ring_mailbox(devs)
    FLAG_LANE[0] = 0
    # (the ring collectives' queue of their own, see ring_queue)
    for lane in range(MICRO_BATCHES):
      if ring_queue_of(lane, any_time=True):
        for d in devs: getattr(Device[d], "compute_queue_n", lambda i: None)(ring_queue_of(lane, any_time=True))

  def memory_report(self) -> dict:
    """
    the memory of every device (GB): the weights, the attention caches (KV, the indexer's), the slots' states, the rest used (workspace), free
    (the AM driver's), and the caches' bytes a token and the full contexts (max_context) their rows hold
    """
    def gb(x:float) -> float: return round(x / 1e9, 3)
    devs = to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device))
    rep:dict[str, Any]|None = getattr(self, "_mem_report", None)
    if rep is None:
      qsa = [b for b in self.blk + ([self.mtp] if hasattr(self, "mtp") else []) if isinstance(b, QSABlock) and hasattr(b, "cache_k")]
      gdn = [b for b in self.blk if isinstance(b, GatedDeltaNetBlock)]
      groups = {"kv": [b.cache_k for b in qsa],
                "indexer": [t for i in self._indexers() for t in (getattr(i, "cache_pk", None), getattr(i, "state", None),
                                                                  getattr(i, "pool_state", None)) if t is not None],
                "slots": [t for b in gdn for t in (getattr(b, "pool_rec", None), getattr(b, "pool_rp", None), *getattr(b, "pool_conv", []))
                          if t is not None] + [t for t in (getattr(self, "seq_spec_h", None), getattr(self, "seq_ckpt_h", None)) if t is not None]}
      rows = self.seq_rows()
      grp = {"weights": self._weight_bytes} | {k: _bytes_by_device(v, devs) for k, v in groups.items()}
      per_token = (grp["kv"][devs[0]] + grp["indexer"][devs[0]]) // (rows + REGION_ALIGN)
      rep = self._mem_report = {"groups": grp, "rows": rows, "bytes_per_token": per_token, "full_contexts": round(rows / self.max_context, 2)}
    def free(d:str) -> int:
      try: return sum(sz for sz, _, _, f in Device[d].iface.dev_impl.mm.pa_allocator.blocks.values() if f)  # type: ignore[attr-defined]
      except Exception: return -1
    def vram(d:str) -> int:
      try: return int(Device[d].iface.dev_impl.vram_size)  # type: ignore[attr-defined]
      except Exception: return 0
    # (cached: the device memory of the buffers freed into the allocator's cache (LRU), reused by a buffer of their size and freed when an
    # allocation fails: free to the server too. card 0 holds ~90 MB of them after the start-up (the load's), the others few)
    def cached(d:str) -> int:
      try: return sum(sz*len(c) for (sz, o), c in Device[d].allocator.cache.items() if o is None or not o.host)  # type: ignore[attr-defined]
      except Exception: return 0
    grp = cast(dict[str, dict[str, int]], rep["groups"])
    per = {d: {k: gb(v[d]) for k, v in grp.items()} for d in devs}
    # (the rest of the memory: the activations, the collectives' buffers, the jits')
    for d in devs:
      per[d] |= {"workspace": gb(vram(d) - free(d) - cached(d) - sum(v[d] for v in grp.values())), "free": gb(free(d)), "cached": gb(cached(d))}
    return {"devices": per, "rows": rep["rows"], "bytes_per_token": rep["bytes_per_token"], "full_contexts": rep["full_contexts"],
            "kv_split": self.config.kv_split}

  def seq_warmup(self):
    """
    every jit of the sequences sharing the caches (a prefill of every chunk size, a context the indexer's top_k covers and longer, a step of
    every size, the copies of the slots' states and of a region), captured and run once (linked)
    """
    # the memory of everything the jits read made first (the derived weights', the states'), the states zeroed by a jit compiled with the others
    st = time.perf_counter()
    self._derive_weights()
    _lap("warmup: derived")
    with _zero_deferred() as zero_bufs: self._seq_init()
    _lap("warmup: seq init")
    # (the n-gram table read while the jits compile: its buffers made, the jits that read it capture before)
    self._seq_shared(read=False)
    _lap("warmup: shared")
    # (the computations from the weights (LOAD_DEFER) and the zeroing of the states: jits compiled with the others, run after the reads)
    derive = [j for j in self._pending_load if isinstance(j, _TinyJit)]
    loads = [(j, j) for j in derive] + [_zero_jit(dedup(zero_bufs))]
    if DEBUG >= 1: print(f"seq_warmup: memory made at {time.perf_counter()-st:.2f}s")
    ran:set[str] = set()
    def meanwhile(take:Callable[..., bool|None]):
      # (LANES_LATE: the collectives' state first: no jit runs before)
      _init_buffers(self.__dict__.pop("_lanes_init", []))
      # the files' reads first (LOAD_DEFER: the weights'), then the n-gram table's, then what's computed from them
      while self._pending_load and not isinstance(self._pending_load[0], _TinyJit): self._pending_load.pop(0)()
      _lap("warmup: weights read")
      if self.config.ple is not None:
        self._ple_dev_read()
        _lap("warmup: n-gram table read")
      for j, call in loads:
        if j in self._pending_load: self._pending_load.remove(j)
        take(j)
        call()
      # (LOAD_APART: the derivations ran, their jits drop what they hold: the buffers they read freed before the warm-up's arenas are made)
      if gguf_load.apart_buffers:
        for j in derive: _drop_jit(j)
        _drop_apart(final=False)
      # (the warm-up's runs as their jits come: while the others compile; the stash's jits linked meanwhile when the host memory has room)
      run_as_they_come(take, calls, ran, idle=functools.partial(link_late, take))
    top, k = unwrap(self.config.indexer).top_k, self.config.spec_drafts
    calls:dict[str, tuple[Any, Callable[[], Any]]] = {}
    def add(name:str, jit:Any, call:Callable[[], Any]):
      if all(j is not jit for j, _ in calls.values()): calls[name] = (jit, call)
    # every chunk size's jits: of a context top_k covers (its fewest tokens from the start: a piece of a prompt it's the size for) and longer
    chunks = sorted(self.prefill_chunks(), reverse=True)
    # (a chunk size's jits of the contexts of PREFILL_PK_CTX and of a whole one: a chunk of each)
    pk_jits = [(c, ctx) for c in chunks if (lps:=self._pk_buckets(c)) for ctx in [self.max_context] + [lp * self.config.idx_split *
               unwrap(self.config.indexer).kpool for lp in lps]]
    def pk_call(c:int, ctx:int): return functools.partial(self._seq_chunk, 0, 0, min(c, top), ctx) if top + c > self.seq_rows() else \
                                        functools.partial(self._seq_chunk, 0, top, c, ctx)
    for c, ctx in pk_jits: add(f"prefill{c}_ctx{ctx}", self._prefill_jit(c, self._pk_bucket(c, ctx)), pk_call(c, ctx))
    for c in chunks:
      if self._pk_buckets(c): continue
      # (WARM_ASYNC: the warm-up's runs of the prefills and steps are their jits' calls (as their captures'), the devices not waited for: the
      # parent takes the jits that come meanwhile. 0: a prefill and a step of the server's, their tokens read)
      if WARM_ASYNC: add(f"prefill{c}", self._prefill_jit(c), functools.partial(self._seq_chunk, 0, 0, min(c, top)) if top + c > self.seq_rows()
                         else functools.partial(self._seq_chunk, 0, top, c))
      elif top + c > self.seq_rows(): add(f"prefill{c}", self._prefill_jit(c), functools.partial(self.seq_prefill, 0, 0, [0]*min(c, top), 0, 0))
      else: add(f"prefill{c}", self._prefill_jit(c), functools.partial(self.seq_prefill, 0, 0, [0]*(top+c), top, 0))
    for m in sorted(self.SEQ_SIZES, reverse=True):
      seqs:list[tuple[int, int, int, int, int, list[int], float|tuple[float, float]]] = [(0, 0, 1, 0, 1, [0]*(k+1), 0.0)] * m
      add(f"step{m}", jit:=self._seq_jit(m), functools.partial(lambda j, m: j(*self._seq_vars([(0, 0, 1, 0, 1)]*m)), jit, m) if WARM_ASYNC else
                                             functools.partial(self.seq_step, seqs))
      # mixed admission requires another decoding sequence; a one-row forced step is unreachable and its degenerate graph fails rangeify
      if SEQ_MIX_TOKENS and m > 1:
        add(f"step{m}_forced", self._seq_jit(m, True), functools.partial(self.seq_step, seqs, forced=[[0]*(k+1)]*m))
    for sv in (True, False):
      add(f"copy{int(sv)}", self._seq_copy_jits[sv], functools.partial(self._seq_copy_call, sv, 0, 1, 0))
      add(f"ckpt{int(sv)}", self._seq_ckpt_jits[sv], functools.partial(self.seq_ckpt, sv, 0, 1))
    add("move", self._seq_move_jits[1], functools.partial(self.seq_move, 0, REGION_ALIGN, 1))
    add("first", self._seq_first_jit, functools.partial(self.seq_first, 0, 0, 1))
    if self._seq_mat_jit is not None and self.config.spec_drafts:
      add("mat", self._seq_mat_jit, functools.partial(lambda: cast(Any, self._seq_mat_jit)(*self._mat_vars(0, 1, 1))))
    # the prefills' and steps' jits captured and compiled at once (in forked processes, see capture_parallel): their jits called alone (no
    # device work but theirs)
    par:list[tuple[Any, Callable[[], Any]]] = []
    for c in chunks:
      if self._pk_buckets(c): continue
      par.append((self._prefill_jit(c), functools.partial(self._seq_chunk, 0, 0, min(c, top)) if top + c > self.seq_rows() else
                                       functools.partial(self._seq_chunk, 0, top, c)))
    par += [(self._prefill_jit(c, self._pk_bucket(c, ctx)), pk_call(c, ctx)) for c, ctx in pk_jits]
    # (a prefill in parts (PREFILL_PARTS): a process a part's jit)
    par = [x for jit, call in par for x in ([(j, functools.partial(cast(Any, call), part=p)) for p, j in enumerate(jit)] if isinstance(jit, list) else
                                            [(jit, call)])]
    for m in self.SEQ_SIZES:
      par.append((jit:=self._seq_jit(m), functools.partial(lambda j, m: j(*self._seq_vars([(0, 0, 1, 0, 1)]*m)), jit, m)))
      if SEQ_MIX_TOKENS and m > 1:
        par.append((jit:=self._seq_jit(m, True), functools.partial(lambda j, m: j(*self._seq_vars([(0, 0, 1, 0, 1)]*m)), jit, m)))
    # (and the copies of the states: their jits alone, not seq_move's write of the token buffer)
    for sv in (True, False):
      par += [(self._seq_copy_jits[sv], functools.partial(self._seq_copy_call, sv, 0, 1, 0)),
              (self._seq_ckpt_jits[sv], functools.partial(self.seq_ckpt, sv, 0, 1))]
    v = functools.partial(UOp.variable, min_val=0, max_val=self.seq_rows())
    par += [(j, functools.partial(lambda j: j(v("move_src").bind(0), v("move_dst").bind(REGION_ALIGN)), j)) for j in self._seq_move_jits.values()]
    par.append((self._seq_first_jit, functools.partial(lambda j: j(*self._first_vars(0, 1)), self._seq_first_jit)))
    if self._seq_mat_jit is not None and self.config.spec_drafts:
      par.append((self._seq_mat_jit, functools.partial(lambda j: j(*self._mat_vars(0, 1, 1)), self._seq_mat_jit)))
    warm_named:dict[str, list[Any]] = {}
    if self.stash_entries:
      for sv in (True, False):
        par += [(j:=self._stash_kv_jits[(sv, n)], functools.partial(lambda j: j(*self._stash_kv_vars(0, 0)), j)) for n in self._stash_calls]
        par.append((self._stash_state_jits[sv], functools.partial(lambda j: j(*self._stash_state_vars(0, 0, 0, 0)), self._stash_state_jits[sv])))
        warm_named |= {f"stash{int(sv)}_kv{n}": [self._stash_kv_jits[(sv, n)]] for n in self._stash_calls}
        warm_named[f"stash{int(sv)}_state"] = [self._stash_state_jits[sv]]
    # (WARMUP_JITS: named calls and capture-only named jits (stash copies), the others captured on their first call)
    calls, par = warmup_select(calls, par, warm_named)
    par += loads
    # (STASH_LINK_LATE: the stash's jits linked after the others are compiled: its host memory (SEQ_STASH_GB) isn't made while the capturing
    # processes and the compilers hold theirs, the start-up's peak of host memory)
    late = [*self._stash_kv_jits.values(), *self._stash_state_jits.values()] if self.stash_entries and getenv("STASH_LINK_LATE", 1) else []
    for j in late: setattr(j, "link_later", True)
    # (STASH_LINK_FREE_GB: linked while the jits left compile (the parent waits for them) once the host memory available holds the stash and
    # that many GB more, else after them)
    need = (self.stash_chunks * self.stash_chunk_nbytes + self.stash_entries * self.stash_entry_nbytes + getenv("STASH_LINK_FREE_GB", 10) * 1e9) \
      if late else 0
    def link_late(take:Callable[..., bool|None]|None=None):
      if not late or not getattr(late[0], "link_later", False): return
      if take is not None and (_mem_available() < need or any(take(j, wait=False) is None for j in late)): return
      for j in late:
        setattr(j, "link_later", False)
        if j.captured is not None: j.captured.linked()
      _lap("warmup: stash linked")
    # (the compiles earliest first: qwen's 30 or more jits come one after another, the parent's ~1 s takes don't pile up (all-warm C2 tr8: 62.8 s,
    # latest first 70.9 s))
    capture_parallel(par, meanwhile, compile_fifo=True)
    _init_buffers(self.__dict__.pop("_lanes_init", []))
    if DEBUG >= 1: print(f"seq_warmup: {len(par)} jits compiled at {time.perf_counter()-st:.2f}s")
    for name, (jit, call) in calls.items():
      if name in ran: continue
      st = time.perf_counter()
      call()
      if DEBUG >= 1: print(f"seq_warmup: {name} run in {time.perf_counter()-st:.2f}s")
    link_late()
    # (the derivations' jits dropped: the buffers they read freed)
    loads.clear()
    par.clear()
    _drop_apart()

  def seq_prefill(self, slot:int, base:int, tokens:list[int], start:int, hrow:int, temp:float|tuple[float, float]=0.0,
                  upto:int|None=None) -> int|None:
    """
    the tokens from start of the sequence of slot `slot` (its region of the caches from base) prefilled, from its states of position start (the
    slot's, and its row hrow the hidden state of the token before it; start 0: none): its states of position len(tokens) into the slot (the
    hidden state of the last token its row 0). returns the token sampled after the last. upto: only the tokens up to upto (a slice of the
    prompt, the rest from there): None returned, the devices not waited for (the slice before it is: one slice queued while one runs)
    """
    self._finish_load()
    self._seq_init()
    devs = to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device))
    if upto is not None:
      assert 0 <= start < upto < len(tokens) and base % SEQ_BASE_ALIGN == 0 and base + len(tokens) <= self.seq_rows()
      # (the token after the slice the prompt's: the slice's sampled token isn't written over it, the next slice's tokens are its)
      self._token_host[base+start:base+upto+1] = array.array('i', tokens[start:upto+1])
      self._seq_copy_call(False, slot, start, hrow, keep=start > 0)
      pos = start
      for n in self._chunk_plan(upto - start):
        self._seq_chunk(base, pos, n)
        pos += n
      self._seq_copy_call(True, slot, upto, 0)
      self._slice_wait()
      self._slice_done = [(t:=Device[d].timeline.host.view(fmt='Q'), t[1]) for d in devs]
      return None
    assert 0 <= start < len(tokens) and base % SEQ_BASE_ALIGN == 0 and base + len(tokens) <= self.seq_rows()
    L = len(tokens)
    for d in devs: Device[d].synchronize()  # (a step may still read the token buffer)
    self._slice_done = []
    self._token_host[base+start:base+L+1] = array.array('i', tokens[start:] + [-1])
    self._seq_copy_call(False, slot, start, hrow, keep=start > 0)
    self._seq_pf_host_temp[:2] = array.array('f', temp if isinstance(temp, tuple) else (temp, 1.0))
    pos = start
    for n in self._chunk_plan(L - start):
      self._seq_chunk(base, pos, n)
      pos += n
    self._seq_copy_call(True, slot, L, 0)
    return self._wait_host(lambda: self._token_host[base + L], "the token after a prefill")

  def _slice_wait(self):
    """the last slice (seq_prefill's upto) done on the devices"""
    for tl, v in getattr(self, "_slice_done", []):
      while tl[0] < v: time.sleep(0.0002)
    self._slice_done = []

  def _seq_first(self, slot:UOp, at:UOp):
    """the token after the hidden state of row 0 of slot `slot` (a prompt's last token's) sampled into the token buffer at row `at`"""
    hc, H = unwrap(self.config.hc), 2*(self.config.spec_drafts+1)
    h = self.seq_spec_h.shrink(((slot*H, slot*H+1), None)).reshape(1, 1, hc.count, self.config.dim)
    tok = self._sample(self.output_norm(self._hc_out(h))[:, -1, :], self._seq_pf_temp, self._token_buf.device)
    self._token_buf[:, at:at+1].assign(tok.cast(self._token_buf.dtype)).realize()
  def _first_vars(self, slot:int, at:int) -> list[UOp]:
    return [UOp.variable("first_slot", 0, self.seq_slots()).bind(slot), UOp.variable("first_at", 0, self._token_rows()-1).bind(at)]
  def seq_first(self, slot:int, base:int, pos:int, temp:float|tuple[float, float]=0.0) -> int:
    """
    the token after the sequence of slot `slot` (its region from base) of pos tokens, from its states there (its hidden state of the last token
    its row 0: a prompt's checkpoint restored, see SeqEngine.prefill): a prompt that is all in the caches has no token to prefill
    """
    self._seq_init()
    for d in to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device)): Device[d].synchronize()  # (a step may read the token buffer)
    self._token_host[base+pos] = -1
    self._seq_pf_host_temp[:2] = array.array('f', temp if isinstance(temp, tuple) else (temp, 1.0))
    self._seq_first_jit(*self._first_vars(slot, base+pos))
    return self._wait_host(lambda: self._token_host[base + pos], "the token after a prompt")

  def _chunk_plan(self, n:int) -> list[int]:
    """
    the chunks of n tokens of a prefill: the biggest size while more than twice it is left, the rest in the sizes of the least time (a chunk's
    time about a fixed part and its padded rows': 700 tokens are 576+192, not 2112)
    """
    sizes, ret = self.prefill_chunks(), []
    while n > 2*sizes[-1]:
      ret.append(sizes[-1])
      n -= sizes[-1]
    best = self._chunk_best(tuple(sizes))
    while n > 0:
      ret.append(min(best[n][1], n))
      n -= ret[-1]
    return ret

  @staticmethod
  @functools.cache
  def _chunk_best(sizes:tuple[int, ...]) -> list[tuple[float, int]]:
    """the least time of a prefill of n tokens (up to twice the biggest size) and its last chunk's size, every n (once: ~0.1 s of python)"""
    best:list[tuple[float, int]] = [(0.0, 0)] + [(math.inf, 0)] * (2 * sizes[-1])
    for i in range(1, len(best)): best[i] = min((CHUNK_FIXED_ROWS + s + best[max(0, i - s)][0], s) for s in sizes)
    return best

  def _seq_chunk(self, base:int, pos:int, n:int, ctx:int|None=None, part:int|None=None):
    """
    a prefill chunk: the n tokens from pos of the region from base in the token buffer (the jit of the smallest chunk size of n or more, and of
    the shortest context (PREFILL_PK_CTX) ctx fits: pos+n if None)
    """
    sizes = self.prefill_chunks()
    vb, sp = UOp.variable("seq_base", 0, self.seq_rows()).bind(base), UOp.variable("start_pos", 0, self.max_context-1).bind(pos)
    nt = UOp.variable("toks", 1, c:=next(s for s in sizes if n <= s)).bind(n)
    _pk_local[0] = self._pk_bucket(c, pos + n if ctx is None else ctx)
    try: self(self._token_buf[:, vb+sp:vb+sp+nt], sp, self._seq_pf_temp, vb, part)
    finally: _pk_local[0] = None

  def forward_seqs(self, *vs:UOp) -> Tensor: return self._forward_seqs(vs, False)
  def forward_seqs_forced(self, *vs:UOp) -> Tensor: return self._forward_seqs(vs, True)
  def _forward_seqs(self, vs:tuple[UOp, ...], mix:bool) -> Tensor:
    """
    a step of speculative decoding of G sequences sharing the caches: vs the slot, the base of the region, mtp_pos, j, par and the position of
    the recurrent state of every one, its k+1 tokens (the last step's sampled ones) in _seq_in, its temperature and top_p in _seq_temp. returns
    (1, G*(2k+1)): every one's k+1 sampled tokens and k drafts. (the drafts are sampled with the noise of the row they're verified at: an exact
    sample of the model, more of them accepted at a temperature)
    """
    k, G, devs = self.config.spec_drafts, len(vs) // 6, self.token_embd.weight.device
    hc, mtp, ple = unwrap(self.config.hc), self.mtp, unwrap(self.config.ple)
    sl, bs, mp, js, ps, kbs = (list(vs[i::6]) for i in range(6))
    slots, bases, H = self._ints(tuple(sl)), self._ints(tuple(bs)), 2*(k+1)
    y = self._seq_in[:, :G*(k+1)]
    temps = self._seq_temp[:2*G].reshape(G, 2)
    def batch(pos:list[UOp], R:int) -> SeqBatch: return SeqBatch(G, R, slots, bases, self._ints(tuple(pos)))
    # the hidden states of the last step's tokens: every slot's rows of parity 1-par
    # (the slots' rows by one gather kernel: a cat of G slices at variable rows is a kernel of G threads (221 us at G=16))
    gather = amd_custom_kernels_supported(devs)
    def spec_rows(par:Callable[[UOp], UOp|int]) -> Tensor:
      return self._ints(tuple(s*H + par(p)*(k+1) + r for s, p in zip(sl, ps) for r in range(k+1)))
    last = rows_gather(self.seq_spec_h, spec_rows(lambda p: 1-p)) if gather else \
      Tensor.cat(*[self.seq_spec_h.shrink(((s*H + (1-p)*(k+1), s*H + (2-p)*(k+1)), None)) for s, p in zip(sl, ps)])
    # (the step's seed of the noise from the host: a random tensor of the devices is a different one on every device, a sampled token of one
    # not the others')
    seed = self._seq_temp[-1:].bitcast(dtypes.uint32)
    # the step's collectives on the decoding lanes (see DEC_LANE)
    prev_lane, FLAG_LANE[0] = FLAG_LANE[0], DEC_LANE
    # (the streams' rows split over the devices if they're whole tiles there: the hyper-connections and the router of a device's rows only, a
    # sublayer's input gathered and its output reduce-scattered (the bytes of an allreduce))
    N = len(devs) if isinstance(devs, tuple) else 1
    split = N > 1 and (G*(k+1)) % N == 0 and G*(k+1) // N <= SEQ_SPLIT_MAX
    try:
      # the MTP layer catches up on the k+1 positions from mtp_pos, row j of every sequence drafts
      _seq_batch[0], _batch_rows[0] = batch(mp, k+1), [(p, k+1) for p in mp]
      emb = self._embed(y.to(devs))
      if split:
        # (its rows split over the devices as the model's, its streams gathered for the drafts' head and the next pass)
        _seq_split[0] = G*(k+1) // N
        try:
          xs = mtp._project_input_split(_split_pad(shard_rows(emb, 1)).contiguous(), _split_pad(shard_rows(last.unsqueeze(0), 1)).contiguous())
          m = all_gather(_split_real(mtp(xs, 0)).contiguous(), 1, dtypes.half)
        finally: _seq_split[0] = 0
      else: m = mtp.predict(emb, last.unsqueeze(0), 0)
      if gather and split:
        # (row j of every sequence by one gather kernel (in float), not a G-way cat of variable slices)
        m = rows_gather(m.reshape(G*(k+1), -1), self._ints(tuple(g*(k+1) + j for g, j in enumerate(js))), dtypes.float32).reshape(1, G, *m.shape[2:])
      else: m = Tensor.cat(*[m.float().shrink((None, (g*(k+1)+j, g*(k+1)+j+1), None, None)) for g, j in enumerate(js)], dim=1)
      drafts:list[Tensor] = []
      for i in range(k):
        # (draft i of sequence g: verified at its row i, sampled with that row's noise at its temperature, not its top_p)
        drafts.append(self._sample(mtp.head(m)[0], temps[:, :1], y.device, (seed, Tensor.arange(G) * (k+1) + i)).reshape(1, G))
        # (QTF: the forced draft instead, the MTP layer's next pass on it)
        if QTF: drafts[-1] = self._seq_tf[:, :G*(2*k+1)].reshape(G, 2*k+1)[:, i].reshape(1, G).to(drafts[-1].device).contiguous()
        elif mix:
          f = self._seq_tf[:, :G*(2*k+1)].reshape(G, 2*k+1)[:, i].reshape(1, G).to(drafts[-1].device).contiguous()
          drafts[-1] = (f >= 0).where(f, drafts[-1]).contiguous()
        if i < k-1:
          pos = [p + j + 1 + i for p, j in zip(mp, js)]
          _seq_batch[0], _batch_rows[0] = batch(pos, 1), [(p, 1) for p in pos]
          m = mtp.predict(self._embed(drafts[-1].to(devs)), m.flatten(2), 0)
      # verify: the model on every sequence's token at start_pos (mtp_pos+j) and its drafts
      vt = Tensor.cat(*[y.shrink((None, (g*(k+1)+j, g*(k+1)+j+1))).cat(*[d[:, g:g+1] for d in drafts], dim=1) for g, j in enumerate(js)],
                      dim=1).contiguous()
      starts = [p + j for p, j in zip(mp, js)]
      x = self._embed(vt.to(devs))
      # (the n-gram rows: every sequence's tokens before its start from the token buffer, then its verified ones)
      n = len(ple.multipliers)
      wins = Tensor.cat(*[self._token_window(self._token_buf, b, s, n-1).reshape(1, n-1) for b, s in zip(bs, starts)])
      ple_rows = self._ple_rows_dev(wins.cat(vt.reshape(G, k+1).to(devs), dim=1))
      x = x.unsqueeze(2).expand(*x.shape[:2], hc.count, x.shape[-1]).contiguous()
      ple_emb = self._ple_emb(ple_rows.unsqueeze(0))
      # (the positions of the slots' recurrent states: their replayed tokens up to a sequence's start, see kda_decode)
      _seq_batch[0], _batch_rows[0] = replace(batch(starts, k+1), kbases=self._ints(tuple(kbs))), [(s, k+1) for s in starts]
      if split:
        _seq_split[0] = G*(k+1) // N
        x = _split_pad(shard_rows(x, 1)).contiguous()
      try:
        for block in self.blk: x = block(x, 0, ple_emb=ple_emb)
        if split: x = all_gather(_split_real(x).contiguous(), 1, dtypes.half).float()
      finally: _seq_split[0] = 0
      hsv = x.flatten(2)[0].float()
      if gather:
        # (stored by one scatter kernel and read back by a gather, not G assigns and a cat)
        at = spec_rows(lambda p: p)
        hs = rows_gather(rows_scatter(self.seq_spec_h, hsv.cast(self.seq_spec_h.dtype), at), at)
      else:
        hs = Tensor.cat(*[self.seq_spec_h.shrink(((s*H + p*(k+1), s*H + (p+1)*(k+1)), None)).assign(hsv[g*(k+1):(g+1)*(k+1)])
                          for g, (s, p) in enumerate(zip(sl, ps))])
      tt = temps.reshape(G, 1, 2).expand(G, k+1, 2).reshape(G*(k+1), 2)
      hout = self._hc_out(hs.reshape(1, -1, hc.count, hs.shape[-1] // hc.count))[0]
      tok = self._sample(hout, tt, y.device, (seed, Tensor.arange(G*(k+1))))
      if QTF:
        # (QTF: the rows' log probs (the vocab's) to the host, the step's tokens the forced ones: every draft accepted)
        V = self.config.vocab_size
        lp = self.output(hout).to(devs[0] if isinstance(devs, tuple) else devs).float()[:, :V].log_softmax(-1)
        lpw = self._seq_tfo[:G*(k+1)*V].assign(lp.flatten())
        forced = self._seq_tf[:, :G*(2*k+1)].reshape(G, 2*k+1)[:, k:].reshape(G*(k+1), 1).to(tok.device)
        tok = Tensor(forced.contiguous().uop.after(lpw.uop))
      elif mix:
        forced = self._seq_tf[:, :G*(2*k+1)].reshape(G, 2*k+1)[:, k:].reshape(G*(k+1), 1).to(tok.device).contiguous()
        tok = (forced >= 0).where(forced, tok).contiguous()
    finally: _seq_batch[0], _batch_rows[0], FLAG_LANE[0] = None, None, prev_lane
    q8_quantize_cache_clear()  # don't keep the graph alive
    out = tok.reshape(G, k+1).cat(*[d.reshape(G, 1) for d in drafts], dim=1).reshape(1, G*(2*k+1)).cast(dtypes.int32)
    return self._seq_out[:, :G*(2*k+1)].assign(Tensor(out.uop.mselect(0)) if isinstance(out.device, tuple) else out)

  def _seq_jit(self, m:int, mix:bool=False):
    if not hasattr(self, "_seq_jits"): self._seq_jits:dict[tuple[int, bool], Any] = {}
    if (m, mix) not in self._seq_jits:
      self._seq_jits[(m, mix)] = TinyJit(self.forward_seqs_forced if mix else self.forward_seqs, capture_first=True, scratch=self._scratch)
    return self._seq_jits[(m, mix)]
  def _seq_vars(self, seqs:list[tuple[int, int, int, int, int]]) -> list[UOp]:
    k, n = self.config.spec_drafts, self.seq_slots()
    return [v for g, (sl, b, mp, j, par) in enumerate(seqs) for v in (
      UOp.variable(f"seq_slot{g}", 0, n).bind(sl), UOp.variable(f"seq_base{g}", 0, self.seq_rows()).bind(b),
      UOp.variable(f"seq_mtp{g}", 0, self.max_context-1).bind(mp), UOp.variable(f"seq_j{g}", 0, k).bind(j),
      UOp.variable(f"seq_par{g}", 0, 1).bind(par), UOp.variable(f"seq_kb{g}", 0, self.max_context).bind(self._kbase_of(sl, mp + j)))]
  def _kbase_of(self, slot:int, start:int) -> int:
    """the position of slot `slot`'s recurrent state for a step at start (SEQ_REPLAY: the replayed tokens' before it), else start"""
    if (b:=getattr(self, "_seq_kbase", {}).get(slot)) is None or not 0 <= start - b <= self.config.spec_drafts: return start
    return b

  def seq_step(self, seqs:list[tuple[int, int, int, int, int, list[int], float|tuple[float, float]]],
               forced:list[list[int]]|None=None) -> list[tuple[list[int], list[int]]]:
    """
    a step of speculative decoding of sequences sharing the caches: seqs (slot, base, mtp_pos, j, par, the k+1 tokens of the last step,
    temperature) of every one, their k+1 sampled tokens and k drafts. forced (SEQ_MIX_TOKENS): each one's next suffix tokens to force, at
    most k+1; -1 in the other slots leaves their ordinary samples and drafts unchanged
    """
    self._seq_init()
    k, n = self.config.spec_drafts, len(seqs)
    m = next(s for s in self.SEQ_SIZES if s >= n)
    if forced is not None:
      assert SEQ_MIX_TOKENS and len(forced) == n and all(len(f) <= k+1 for f in forced)
      rows = []
      for f in forced + [[]] * (m-n):
        assert all(t >= 0 for t in f)
        rows += f[:k] + [-1]*(k-min(k, len(f))) + f + [-1]*(k+1-len(f))
      self._seq_host_tf[:m*(2*k+1)] = array.array('i', rows)
    # (the sequences past n: the padding slot's at position 0 in the padding region)
    seqs = seqs + [(self.seq_slots(), self.seq_rows(), 0, 0, 1, [0]*(k+1), 0.0)] * (m - n)
    self._seq_host_in[:m*(k+1)] = array.array('i', [t for *_, y, _ in seqs for t in y])
    self._seq_host_temp[:2*m] = array.array('f', [v for *_, temp in seqs for v in (temp if isinstance(temp, tuple) else (temp, 1.0))])
    # (the step's seed, see forward_seqs)
    self._seq_host_temp[-1] = array.array('f', bytes(random.getrandbits(8) for _ in range(4)))[0]
    W = m*(2*k+1)
    self._seq_host_out[:W] = array.array('i', [-1] * W)
    jit = self._seq_jit(m, forced is not None)
    if (cap:=getattr(jit, "captured", None)) is not None:
      # (captured: the variables' values as they are, no bound variables made a step)
      cap([], {name: v for g, (sl, b, mp, j, par, _, _) in enumerate(seqs)
               for name, v in ((f"seq_slot{g}", sl), (f"seq_base{g}", b), (f"seq_mtp{g}", mp), (f"seq_j{g}", j), (f"seq_par{g}", par),
                               (f"seq_kb{g}", self._kbase_of(sl, mp + j)))})
    else: jit(*self._seq_vars([(sl, b, mp, j, par) for sl, b, mp, j, par, _, _ in seqs]))
    # (every sequence's recurrent states now the ones after its step's first token)
    for sl, _, mp, j, _, _, _ in seqs[:n]: self._seq_kbase[sl] = mp + j + 1
    out = self._wait_host(lambda: min(o:=self._seq_host_out[:W].tolist()) >= 0 and o or -1, f"a step of {m} sequences")
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
    if todo: Tensor.realize(*todo)
    self._states_ready = True

  def __call__(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor|None, base:int|UOp=0, part:int|None=None) -> Tensor:
    # a view of the token buffer is already realized: no schedule for the jit input
    if (buf:=getattr(self, "_token_buf", None)) is None or tokens.uop.base is not buf.uop.base: tokens = tokens.contiguous()
    key:Any = int(T.vmax) if isinstance(T:=tokens.shape[1], UOp) else T
    if _pk_local[0] is not None: key = (key, _pk_local[0])
    # (a sequence's region from a variable base: jits of their own, see seq_prefill)
    jits = self.__dict__.setdefault("jits" + ("_greedy" if temperature is None else "") + ("_seq" if isinstance(base, UOp) else ""), {})
    if (jit:=jits.get(key)) is None:
      # the states and caches are set up before: a jit captures on its first call
      self._init_states()
      jit = jits[key] = self._new_prefill_jit(int(T.vmax) if isinstance(T, UOp) else T) if isinstance(base, UOp) else \
        TinyJit(self._forward_effects, capture_first=True, scratch=self._scratch)
    if not isinstance(jit, list): return jit(tokens, start_pos, temperature, base)[0]
    # (a prefill in parts: one of them (part, see seq_warmup), else all in order)
    for p, j in enumerate(jit):
      if part is None or p == part: out = j(tokens, start_pos, temperature, base)[0]
    return out

  def _forward_effects(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor|None, base:int|UOp=0, part:int|None=None) -> tuple[Tensor, ...]:
    # forward and the tensors realized with its output for their effects (e.g. the caches of the MTP layer)
    self._effects = []
    out = self.forward(tokens, start_pos, temperature, base, part)
    effects, self._effects = self._effects, None
    return (out, *effects)

  @staticmethod
  def from_gguf(gguf:Tensor|str|pathlib.Path, max_context:int|None=None,
                realize=bool(getenv("REALIZE", 0)), shard:int=1, parsed:tuple[dict, dict]|None=None) -> tuple[Transformer, dict]:
    """the qwen4exp model of a gguf file on `shard` devices (its weights read, the n-gram table too). parsed: the file's (kv, entries)"""
    kv, entries = parsed if parsed is not None else gguf_parse(gguf.to(None).realize() if isinstance(gguf, Tensor) else gguf)
    arch = kv['general.architecture']
    assert arch == 'qwen4exp' and shard > 1, f"the qwen4exp model on tensor parallel devices, not {arch} on {shard}"
    n_heads, n_kv_heads = kv[f'{arch}.attention.head_count'], kv[f'{arch}.attention.head_count_kv']
    num_blocks = kv[f'{arch}.block_count'] - kv.get(f'{arch}.nextn_predict_layers', 0)
    # the next token prediction (MTP) layer: in a gguf of its own (its block after the model's), found next to the model's
    mtp = False
    if not isinstance(gguf, Tensor) and (mf:=_qwen_mtp_file(pathlib.Path(gguf))) is not None:
      mkv, mentries = gguf_parse(mf)
      mb = mkv[f'{arch}.block_count'] - 1
      entries.update({re.sub(rf"^blk\.{mb}\.", f"blk.{num_blocks}.", k): v for k, v in mentries.items() if k.startswith(f"blk.{mb}.")})
      mtp = True
    entries = {k:v for k,v in entries.items() if not ((m:=re.match(r"blk\.(\d+)\.", k)) and int(m.group(1)) >= num_blocks + mtp)}
    # the n-gram table stays in host memory, the gated deltanets in the kda layout, a KV head a device
    ple_entry = entries.pop('per_layer_token_embd.weight', None)
    # (QSA_KV_SPLIT: the rows of a KV head split over its devices, see TransformerConfig.kv_split)
    kv_real = n_kv_heads
    kv_split = shard // kv_real if getenv("QSA_KV_SPLIT", 0) and shard > kv_real and shard % kv_real == 0 else 1
    # (QSA_IDX_SPLIT: the indexer's pooled keys split over the devices, see TransformerConfig.idx_split)
    idx_split = shard if getenv("QSA_IDX_SPLIT", 0) and REGION_ALIGN % (4 * shard) == 0 else 1
    entries, n_kv_heads, rows_map = _qwen4exp_entries(kv, entries, shard, kv_split)
    units_map:dict[str, tuple[int, list[list[int]]]] = {}
    repeats:dict[str, int] = {}
    sizes:dict[str, int] = {}
    if n_kv_heads < shard and shard % n_kv_heads == 0:
      # too few KV heads to split: repeat each one so every device holds one
      repeats = {k: shard // n_kv_heads for k in entries if k.split('.', 2)[-1] in ('attn_k.weight', 'attn_v.weight')}
      n_kv_heads = shard
    assert n_kv_heads % shard == 0, f"tensor parallel needs the attention heads to split over {shard} devices"
    # the attention and the gated deltanets split by their heads
    head_rules = {**{w: 0 for w in ('attn_q.weight', 'attn_k.weight', 'attn_v.weight', 'ssm_conv1d_q.weight', 'ssm_conv1d_k.weight',
                                    'ssm_conv1d_v.weight', 'ssm_dt.bias', 'ssm_a', 'ssm_beta.weight', 'ssm_alpha.weight', 'attn_gate.weight')},
                  'attn_output.weight': 1, 'ssm_out.weight': 1}
    ffn_rules = {**{w: 0 for w in ('ffn_gate_shexp.weight', 'ffn_up_shexp.weight', 'output.weight')}, 'ffn_down_shexp.weight': 1}
    if len(kv['tokenizer.ggml.tokens']) % shard == 0: ffn_rules['token_embd.weight'] = 0
    # the experts: slices of their hidden dim (whole 128 wide quant blocks of the down projection) spread over the devices, the work of a token
    # balanced (padded: experts past the router's never selected, so the slices split evenly. 512 on 6 devices: 516. their units load expert
    # 0's bytes)
    exp_hidden, exp_count = kv[f'{arch}.expert_feed_forward_length'], kv[f'{arch}.expert_count']
    down_block = math.lcm(128, *[_GGML_QUANT.get(t, (1,))[0] for k,(_,_,t) in entries.items() if k.endswith('ffn_down_exps.weight')])
    assert exp_hidden % down_block == 0
    expert_slices = exp_hidden // down_block
    padded = next(e for e in range(exp_count, exp_count + shard*shard) if expert_slicing(e, expert_slices, shard))
    expert_pad = padded if padded != exp_count else 0
    slicing = [[u if u < exp_count*expert_slices else u % expert_slices for u in us] for us in unwrap(expert_slicing(padded, expert_slices, shard))]
    units_map = {name: (expert_slices, slicing) for name in entries
                 if re.sub(r"^blk\.\d+\.", "", name) in ('ffn_gate_exps.weight', 'ffn_up_exps.weight', 'ffn_down_exps.weight')}
    # the shared expert's hidden dim pads to whole quant blocks of its down projection on every device (UD quants mix Q6_K and Q8_0 ones),
    # the output rows to whole tiles of 32 (the tensor cores take a few tokens' logits)
    down_unit = math.lcm(*[_GGML_QUANT.get(t, (1,))[0] for k,(_,_,t) in entries.items() if k.endswith('ffn_down_shexp.weight')])
    shard_map:dict[str, int] = {}
    for name, (_, shape, typ) in entries.items():
      if (k:=re.sub(r"^blk\.\d+\.", "", name)) in head_rules: shard_map[name] = head_rules[k]
      elif k in ffn_rules:
        axis = shard_map[name] = ffn_rules[k]
        sizes[name] = round_up(shape[axis], shard*(32 if k == 'output.weight' else _GGML_QUANT.get(typ, (1,))[0] if k == 'token_embd.weight' else
                                                   down_unit))
    pads = {k: sizes[k] for k in sizes if sizes[k] != entries[k][1][shard_map[k]]}
    devices = tuple(Device.canonicalize(f'{Device.DEFAULT}:{i}') for i in range(shard))
    _open_devices(devices)
    # (the compile workers started and warmed up while the weights are read)
    warm_worker_pool(devices[0])
    # (the micro batches on compute queues of their own if the devices have four)
    EXTRA_QUEUES[0] = all(getattr(Device[d], "compute_queue_count", lambda: 0)() >= 4 for d in devices)
    # gate and up experts of the same format load as one (fuse_gate_up)
    # (a pair in different files of a split gguf isn't: a fused read is of the gate's file. prepare fuses it on the devices)
    def file_of(data:Tensor) -> str|None:
      try: return cast(Buffer, data.uop.buffer).device
      except Exception: return None
    fuse = {g: u for g in units_map if g.endswith("ffn_gate_exps.weight") and (u:=g.replace("gate", "up")) in units_map and
            entries[g][2] == entries[u][2] and file_of(entries[g][0]) == file_of(entries[u][0])}
    # a token embedding copied to every device (a vocab that doesn't split) stays in host memory: a call looks up a few of its rows
    host_embd, embd_q8 = None, None
    if 'token_embd.weight' not in shard_map and 'output.weight' in entries:
      data, shape, typ = entries.pop('token_embd.weight')
      mv, raw = _host_buffer(devices, int(data.nbytes()))
      mv[:] = data.to("CPU").data()
      host_embd = ggml_data_to_tensor(raw, prod(shape), typ).reshape(shape)
      if typ == Q8_0: embd_q8 = raw
    # (LOAD_DEFER (default): the weights' memory made now, the files read later: while the jits compile, see seq_warmup. 0: read now)
    pending:list[Callable[[], None]]|None = [] if getenv("LOAD_DEFER", 1) else None
    # (LOAD_APART: the file's tensors the model only derives others from in buffers of their own, freed once derived (see _drop_apart), not in
    # the weights' buffer for good: the float ones (F32, BF16: prepare's half copies, _derive_weights' transposed and float ones) and a gate and
    # up pair the load didn't fuse (prepare's fuse_up). C2: ~0.5 GB a device)
    apart = ({k for k,(_,_,t) in entries.items() if t in (0, 30) and k not in ('token_embd.weight', 'output.weight')} |
             {n for g, u in ((g, g.replace("gate", "up")) for g in units_map if g.endswith("ffn_gate_exps.weight"))
              if g not in fuse and u in units_map and entries[g][2] == entries[u][2] for n in (g, u)}) if getenv("LOAD_APART", 0) else None
    state_dict = gguf_shard_units(entries, devices, shard_map, repeats, units_map, pads, fuse, row_sel=rows_map, pending=pending, apart=apart)
    _lap("weights")
    if host_embd is not None: state_dict['token_embd.weight'] = host_embd
    state_dict = {k:v.cast('float16') for k,v in state_dict.items()}
    if 'output.weight' not in state_dict: state_dict['output.weight'] = state_dict['token_embd.weight']
    state_dict = {re.sub(r'(hc_attn|hc_ffn|hc_head|output_hc|\.ple)_', r'\1.', k):v for k,v in state_dict.items()}
    if mtp: state_dict = {re.sub(rf"^blk\.{num_blocks}\.(nextn\.)?", "mtp.", k):v for k,v in state_dict.items()}

    hc = HCConfig(kv[f'{arch}.hyper_connection.count'], kv.get(f'{arch}.hyper_connection.sinkhorn_iterations', 0),
                  kv.get(f'{arch}.hyper_connection.epsilon', 0.0), kv[f'{arch}.hyper_connection.low_rank'])
    # QSA: the indexer pools compress_ratio keys
    indexer = IndexerConfig(kv[f'{arch}.attention.indexer.head_count'], kv[f'{arch}.attention.indexer.key_length'],
                            kv[f'{arch}.attention.indexer.top_k'], max(kv[f'{arch}.attention.compress_ratios']),
                            kv[f'{arch}.attention.layer_norm_rms_epsilon'], kv[f'{arch}.rope.dimension_count'], False)
    ple = PLEConfig(kv[f'{arch}.ple.layers'][0], ple_entry[1][1], tuple(kv[f'{arch}.ple.layer_multipliers']),
                    tuple(kv[f'{arch}.ple.head_vocab_sizes']), tuple(kv[f'{arch}.ple.head_offsets']), kv[f'{arch}.ple.eos_token_id'],
                    kv[f'{arch}.ple.conv_kernel'], ple_entry[2]) if ple_entry is not None else None
    max_context = min(max_context, kv[f'{arch}.context_length']) if max_context is not None else kv[f'{arch}.context_length']
    # (the kda layout: the q and k heads repeated for the v heads at load, see _qwen4exp_entries)
    hv = kv[f'{arch}.ssm.time_step_rank']
    ssm = SSMConfig(kv[f'{arch}.ssm.conv_kernel'], kv[f'{arch}.ssm.state_size'], hv, hv, kv[f'{arch}.ssm.inner_size'], kda=True, head_decay=True)
    ssm_layers = tuple((i+1) % kv[f'{arch}.full_attention_interval'] != 0 for i in range(num_blocks))
    for i, is_ssm in enumerate(ssm_layers):
      if is_ssm: state_dict[f"blk.{i}.ssm_a"] = state_dict[f"blk.{i}.ssm_a"].reshape(hv, 1)
    head_dim = kv.get(f'{arch}.attention.key_length', kv[f'{arch}.embedding_length'] // n_heads)
    yarn = YaRNConfig(factor=kv[f'{arch}.rope.scaling.factor'],
                      orig_ctx_len=kv.get(f'{arch}.rope.scaling.original_context_length', kv[f'{arch}.context_length']),
                      beta_fast=kv.get(f'{arch}.rope.scaling.yarn_beta_fast', 32.0),
                      beta_slow=kv.get(f'{arch}.rope.scaling.yarn_beta_slow', 1.0)) if kv.get(f'{arch}.rope.scaling.type') == 'yarn' else None
    config = TransformerConfig(
      num_blocks=num_blocks, dim=kv[f'{arch}.embedding_length'], hidden_dim=exp_hidden, n_heads=n_heads * kv_split, n_kv_heads=n_kv_heads,
      norm_eps=kv[f'{arch}.attention.layer_norm_rms_epsilon'], vocab_size=len(kv['tokenizer.ggml.tokens']),
      padded_vocab_size=int(state_dict['output.weight'].shape[0]), head_dim=head_dim, rope_theta=kv.get(f'{arch}.rope.freq_base', 10000.0),
      rope_dim=kv.get(f'{arch}.rope.dimension_count', head_dim), yarn=yarn, v_head_dim=kv.get(f'{arch}.attention.value_length', head_dim),
      max_context=max_context, qk_norm=int(state_dict['blk.0.attn_q_norm.weight'].shape[0]) if 'blk.0.attn_q_norm.weight' in state_dict else 0,
      num_experts=exp_count, num_experts_per_tok=kv[f'{arch}.expert_used_count'], norm_topk_prob=kv.get(f'{arch}.expert_weights_norm', True),
      shared_expert_dim=sizes.get("blk.0.ffn_gate_shexp.weight", kv.get(f'{arch}.expert_shared_feed_forward_length', 0)),
      shared_expert_gate="blk.0.ffn_gate_inp_shexp.weight" in state_dict,
      routed_scaling_factor=kv.get(f'{arch}.expert_weights_scale', 1.0), attn_output_gate=True, ssm=ssm, ssm_layers=ssm_layers,
      expert_slices=expert_slices, swiglu_limit=kv.get(f'{arch}.swiglu_clamp_exp', [None])[0], hc=hc, indexer=indexer, mtp=mtp,
      # (SPEC_DRAFTS: the drafts of a step (fewer for many sequences at once))
      spec_drafts=getenv("SPEC_DRAFTS", 4) if mtp else 0, ple=ple, qsa=True, expert_pad=expert_pad,
      # (SEQ_ROWS: the rows of the caches of the sequences sharing them, past a context: more conversations kept)
      cache_rows=round_up(getenv("SEQ_ROWS", 0), REGION_ALIGN), kv_split=kv_split, idx_split=idx_split)
    with Linear.empty_init(): model = Transformer(config)  # (the weights are loaded)
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
    model._loaded, model._pending_load = True, pending or []
    model._embd_q8 = embd_q8
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
    # (the weights' memory: a walk of the model later meets the jits' and the engine's references)
    model._weight_bytes = _bytes_by_device(nn.state.get_parameters(model), devices)
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
    # packed gate and up experts of the same format are one weight: their rows interleaved (the ones the load didn't fuse: a pair in two files)
    fused:list[Tensor] = []
    for b in self.blk + ([self.mtp] if hasattr(self, "mtp") else []):
      if (g:=getattr(b, 'ffn_gate_exps', None)) is not None and hasattr(b, 'ffn_up_exps') and g.ggml_type is not None and \
         g.ggml_type == b.ffn_up_exps.ggml_type and amd_custom_kernels_supported(g.weight.device):
        packed.discard(id(b.ffn_up_exps.weight))
        packed.discard(id(g.weight))
        g.fuse_up(b.ffn_up_exps, b.config.hidden_dim // (b.config.expert_slices or 1))
        del b.ffn_up_exps
        packed.add(id(g.weight))
        fused.append(g.weight)
    if fused:
      if self._pending_load: _realize_later(fused, self._pending_load)
      else: Tensor.realize(*fused)
    # (LOAD_APART: the skinny Linears' weights not copied here: _derive_weights' transposed copy is made from the file's, the same values (a
    # cast, then the copy), the copy here held dead in its group's buffer (C2: ~180 MB a device))
    if getenv("LOAD_APART", 0): packed |= {id(l.weight) for l in self._skinny_lins()}
    small = [p for p in nn.state.get_parameters(self) if id(p) not in packed and p is not self.token_embd.weight and not p.uop.is_realized]
    for p in small: p.replace(p.contiguous())
    tables = dedup([t for m in modules([self.blk] + ([self.mtp] if hasattr(self, "mtp") else []), set()) if isinstance(m, ExpertWeights)
                    for t in (getattr(m, "first", None), getattr(m, "slice_table", None)) if t is not None and not t.uop.is_realized])
    Tensor.realize(*tables)
    # (the weights still to read: the small ones' memory now, their values after)
    if self._pending_load: return _realize_later(small, self._pending_load)
    # (a block's at a time: the blocks' graphs are the same, scheduled once (the schedule cache), not one graph of all of them)
    ids = {id(p) for p in small}
    for b in self.blk + ([self.mtp] if hasattr(self, "mtp") else []):
      if mine:=[p for p in nn.state.get_parameters(b) if id(p) in ids]:
        Tensor.realize(*mine)
        ids -= {id(p) for p in mine}
    Tensor.realize(*[p for p in small if id(p) in ids])

  def _finish_load(self):
    """the work left of loading: the values computed from the weights, the weights read if deferred"""
    self._derive_weights()
    ran = bool(self._pending_load)
    while self._pending_load: self._pending_load.pop(0)()
    _lap("derived")
    if ran: _drop_apart()

  def _skinny_lins(self) -> list[Linear]:
    """the Linears of float weights _derive_weights stores transposed (weight_t) for skinny_matmul"""
    blocks = self.blk + ([self.mtp] if hasattr(self, "mtp") else [])
    return [l for b in blocks for l in (getattr(b, "ffn_gate_inp", None), getattr(getattr(b, "indexer", None), "proj", None),
                                        *(getattr(getattr(b, "indexer", None), p, None) for p in ("q_proj", "k_proj")),
                                        *(getattr(getattr(b, h, None), "inject", None) for h in ("hc_attn", "hc_ffn")))
            if isinstance(l, Linear) and l.ggml_type is None and l.weight.dtype in (dtypes.float32, dtypes.half) and l.weight.ndim == 2 and
            getattr(l, "bias", None) is None and not hasattr(l, "weight_t") and amd_custom_kernels_supported(l.weight.device)]

  def _derive_weights(self):
    """the weights computed from the file's"""
    def realize(ts:list[Tensor]):
      if self._pending_load: _realize_later(ts, self._pending_load)
      else: Tensor.realize(*ts)
    # (SKINNY) the float weights of few outputs that decoding multiplies by (the routers, the indexers' head weights) stored transposed (the
    # outputs the fast axis), the Linear's a view of it: skinny_matmul reads every input's weights of all the outputs at once
    blocks = self.blk + ([self.mtp] if hasattr(self, "mtp") else [])
    if lins:=self._skinny_lins():
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
    for g in ghcs:
      if g._wnorm is None:
        g._wnorm = g.norm["weight"].float().contiguous()
        winj.append(g._wnorm)
    if winj: realize(winj)
    # (GDN_MERGED: the gated deltanets' q, k, v and gate weights as one (adjacent, see _qwen4exp_entries): batched decoding's one kernel)
    if GDN_MERGED:
      for b in self.blk:
        if isinstance(b, GatedDeltaNetBlock) and b.is_kda and hasattr(b, "attn_gate"):
          b._qkvg = merged_linear([b.attn_q, b.attn_k, b.attn_v, b.attn_gate])
      if DEBUG >= 1: print(f"GDN_MERGED: {sum(getattr(b, '_qkvg', None) is not None for b in self.blk)} merged q|k|v|gate weights")

  def sequence_parallel(self, rows:int) -> bool:
    """the hyper-connection streams of a prefill chunk of `rows` are split over the devices: tiles of 16 rows on every device"""
    devs = self.token_embd.weight.device
    return isinstance(devs, tuple) and self.config.hc is not None and rows % (16*len(devs)) == 0

  def prefill_chunks(self, chunk_size:int|None=None) -> tuple[int, ...]:
    """the prefill chunk sizes, ascending: the custom kernels compute chunks padded to a static size, a piece of a prompt runs the smallest
    one it fits (its own jit), large chunks are efficient. sequence parallel: whole tiles of rows on every device"""
    fast = amd_custom_kernels_supported(self.token_embd.weight.device)
    # Qwen's gated streams keep the 2K chunk at long contexts too: larger chunks substantially increase prefill workspace.
    # (PREFILL_CHUNK: the largest, 13824: a 262k prompt in 19 chunks, the micro batches' collectives hidden by their compute)
    chunk_size = chunk_size or min(getenv("PREFILL_CHUNK", 13824) if fast else 32, self.max_context)
    if self.has_recurrent_block and not fast: chunk_size = 1
    # (and smaller ones, a jit each: less padding of a prompt's last piece (the turns of agents: a few hundred tokens))
    # (PREFILL_SIZES: the smaller sizes, e.g. 192,512,1152,2048,4096: a warm turn of ~1k tokens in one chunk, not two of 576)
    sizes = [chunk_size] + [min(chunk_size, n) for n in (PREFILL_SIZES if fast else ())]
    if fast and (devs:=self.token_embd.weight.device) and isinstance(devs, tuple) and self.config.hc is not None:
      # whole tiles of 16 rows on every device, the kernels of the whole chunk on tiles of 64
      tile = math.lcm(16*len(devs), 64)
      def tiled(n:int, t:int=tile) -> int: return min(max(t, round(n / t) * t), self.max_context // t * t or t)
      sizes = [tiled(n) for n in sizes]
      # (PREFILL_TILE32: a chunk of micro batches (past PREFILL_MB1_ROWS) in whole tiles of 32 rows a device a micro batch: the sequence
      # parallel hyper-connections' gemms take tiles of 32 rows (q8_gemm), 2112 rows (176 a device a batch) ran them on the slow f16 kernel)
      if PREFILL_TILE32 and MICRO_BATCHES > 1:
        t2 = math.lcm(32*len(devs)*MICRO_BATCHES, 64)
        sizes = [tiled(n, t2) if n > PREFILL_MB1_ROWS and n % t2 else n for n in sizes]
    return tuple(sorted(set(sizes)))

  def warmup(self):
    if self.seqs_supported(): return self.seq_warmup()
    for _ in range(2): list(zip(range(2), self.generate([0])))

  def get_start_pos(self, tokens:list[int]) -> int:
    # recurrent state can't be partially reused after divergence: reuse it only when tokens extend the cached prefix
    if self.has_recurrent_block:
      return len(self._cached_tokens) if self._cached_tokens and len(self._cached_tokens) < len(tokens) \
        and tokens[:len(self._cached_tokens)] == self._cached_tokens else 0
    prefix_len = sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], zip(tokens[:-1], self._cached_tokens)))
    return min(block._reusable_prefix_len(prefix_len, len(self._cached_tokens)) for block in self.blk)

  def generate(self, tokens:list[int], chunk_size:int=32, temperature:float=0.0, top_p:float=1.0):
    if self.seqs_supported():
      yield from self._generate_seq(tokens, temperature, top_p)
      return
    if self.has_recurrent_block and not amd_custom_kernels_supported(self.token_embd.weight.device): chunk_size = 1
    v_start_pos = UOp.variable("start_pos", 0, self.max_context-1)
    v_toks = UOp.variable("toks", 1, chunk_size)
    # TODO: use UOp.variable for temperature once float variables are supported. (greedy: no temperature, the jits of no sampling)
    temp = Tensor([[temperature, top_p]]) if temperature != 0 else None
    # assign all input tokens once, then slice from start_pos for the model call
    t = Tensor(tokens + [0] * (self.max_context - len(tokens)), dtype="int32").reshape(1, self.max_context)
    # recompute start_pos from what's currently valid in the caches
    start_pos = self.get_start_pos(tokens)
    out, prompt_len = None, len(tokens)
    while len(tokens) < self.max_context:
      n_toks = min(chunk_size, len(tokens) - start_pos)
      sp, nt = v_start_pos.bind(start_pos), v_toks.bind(n_toks)
      out = self(t[:, sp:sp+nt] if start_pos < prompt_len or out is None else out, sp, temp).realize()
      start_pos += n_toks
      # chunked prefill: keep processing until all prompt tokens are consumed
      if start_pos < len(tokens): continue
      tokens.append(int(out.item()))
      self._cached_tokens = tokens[:-1]
      yield tokens[-1]

  def _generate_seq(self, tokens:list[int], temperature:float, top_p:float):
    """generate of a model whose sequences share its caches: a sequence of the engine (the one its prompt continues), decoded alone"""
    from tinygrad.llm.engine import SeqEngine
    if (eng:=getattr(self, "_engine", None)) is None: eng = self._engine = SeqEngine(self)
    if (got:=eng.take(list(tokens), force=True)) is None: raise RuntimeError(f"a prompt of {len(tokens)} tokens doesn't fit the caches")
    s, start, restore = got
    kept, k = len(tokens), self.config.spec_drafts
    try:
      tokens.append(eng.prefill(s, list(tokens), start, restore, (temperature, top_p)))
      kept += 1
      yield tokens[-1]
      while len(tokens) < self.max_context and eng.room(s, len(s.tokens) + 2*(k+1)):
        for t in eng.step([s])[0]:
          tokens.append(t)
          kept += 1
          yield t
    finally: eng.stop(s, kept)
