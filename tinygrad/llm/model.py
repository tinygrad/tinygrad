from __future__ import annotations
import array, enum, functools, gc, itertools, math, pathlib, re, time
from dataclasses import dataclass, replace, field
from typing import Any, Callable, Iterator, cast
from tinygrad import Tensor, nn, UOp, TinyJit, getenv, function, dtypes, Device
from tinygrad.engine.jit import capture_parallel, _TinyJit
from tinygrad.engine.realize import run_linear
from tinygrad.llm.kernels.amd import Linear, gated_delta_prefill, flash_attention, amd_custom_kernels_supported, find_packed, moe_linear, \
  hc_coeffs, hc_pre, q8_quantize, q8_linear, QUANT_SIZES, q8_quantize_cache_clear, amd_flash_attention_decode as flash_attention_decode, moe_route, \
  moe_grouped_linear, amd_flash_attention_mla_prefill as flash_attention_mla_prefill, partial_sum, fuse_gate_up, kda_decode, moe_router, \
  indexer_scores, _wmma_rdna4, pool_keys, pool_index_rows, slot_scatter, \
  indexer_decode_index, amd_flash_attention_mla_sparse as flash_attention_mla_sparse, moe_combine, token_sharded, shard_rows, reduce_scatter, \
  all_gather, all_gather_q8, sharded_argmax, moe_combine_scatter, kda_conv, tables_ready, moe_router_local, hc_post_quant, \
  indexer_decode_index_rows, Q8_ROWS_I8, MOE_TILE, grouped_rows, q8_quantize_known, \
  mla_bmm, nucleus
from tinygrad.llm.gguf import gguf_parse, gguf_shard, ggml_data_to_tensor, _GGML_QUANT
from tinygrad.uop.ops import resolve, Ops
from tinygrad.dtype import DType
from tinygrad.helpers import unwrap, round_up, to_tuple, prod, dedup
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

_block_q8:list[tuple[UOp, tuple[Tensor, ...]]|None] = [None]  # a decoding block's output and its quantization (the next block's input's)
INLINE_ACT = getenv("INLINE_ACT", 1)  # a decoding activation computed in the kernel that quantizes it (not a kernel of its own)

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

@dataclass(frozen=True)
class HCConfig:  # manifold-constrained hyper-connections: count residual streams, mixed by a doubly stochastic matrix
  count: int
  sinkhorn_iters: int
  eps: float

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
def _out(layer:Linear, x:Tensor) -> Tensor:
  """a sublayer's output projection: its partial sums if they are kept (_partial_out), in half (they're sent so)"""
  return layer.partial(x, half=True) if _partial_out[0] else layer(x)

def _identity(x:Tensor) -> Tensor: return x

# the experts of a few tokens are the ones of every token (a decode kernel each), more are grouped by expert (prefill)
MOE_DECODE_ROWS = getenv("MOE_DECODE_ROWS", 96)
# batched decoding: rows of experts decoded (in bundles) up to this many, grouped (the prefill kernels) past it
MOE_BATCH_ROWS = getenv("MOE_BATCH_ROWS", 512)
def _decode_rows() -> int: return MOE_BATCH_ROWS if _batch_rows[0] is not None else MOE_DECODE_ROWS

# batched decoding (Transformer.forward_batch): the rows are groups of a sequence each, group g (its rows the tokens at positions pos.. of the
# sequence) in the state slot g: (pos, rows) of every group
_batch_rows:list[list[tuple[UOp, int]]|None] = [None]

@dataclass(frozen=True)
class BatchPrefill:
  """a step of prefills of G prompts into slots (Transformer.forward_batch_prefill): P rows each, their token counts and slots (tensors and the
  variables they're of), the effects of the blocks (the states into the slots) realized with the step"""
  P:int; counts:Tensor; slots:Tensor; count_vars:tuple[UOp, ...]; slot_vars:tuple[UOp, ...]; effects:list = field(default_factory=list) # noqa: E702
  @property
  def real(self) -> Tensor:
    """(G*P,) the rows that are tokens (the others pad a prompt to P rows)"""
    return (Tensor.arange(self.P).reshape(1, self.P).to(self.counts.device) < self.counts.reshape(-1, 1)).reshape(-1)
_batch_prefill:list[BatchPrefill|None] = [None]

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

def _host_ints(devs:str|tuple[str, ...], n:int) -> tuple[memoryview, Tensor]:
  """n int32 in host memory that the host polls: a (1, n) tensor every device reads and writes directly"""
  mv, t = _host_buffer(devs, 4*n, dtypes.int32)
  return mv.cast('i'), t[None]

def _temp_pair(t:float|tuple[float, float]) -> array.array:
  """a row's temperature and top_p (a temperature alone: top_p 1) for a step's host buffer"""
  return array.array('f', t if isinstance(t, tuple) else (t, 1.0))

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
        k, scale, exps = self.config.num_experts_per_tok, self.config.routed_scaling_factor, self.ffn_gate_exps
        exps.set_quantized()
        if rows is None and xr.shape[1] <= _decode_rows() and exps.slices and exps.ggml_type is not None and getenv("MOE_ROUTER_LOCAL", 1):
          # decoding: the device's slices of the selections in the router's kernel
          return self._experts(x, h, *moe_router_local(logits, bias, k, scale, exps.slice_table), local=True)
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
    if (T:=_ffn_rows[0]) is not None and sel.ndim >= 3 and isinstance(R:=sel.shape[-2], int) and getenv("MOE_SKIP_PAD", 1):
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
      out = self._grouped_experts(x, sel, probs, add=shexp, half=_partial_out[0] and (shexp is not None or not hasattr(self, 'ffn_gate_shexp')))
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
      out = (x_down * probs.unsqueeze(-1)).sum(axis=-2)  # (B, T, D), expert parallel: (devices, B, T, D)
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

  def _grouped_experts(self, x:Tensor, sel:Tensor, probs:Tensor, add:Tensor|None=None, half:bool=False) -> Tensor:
    """prefill: the selections sorted by expert, every expert reads its weights once for all its tokens. (devices, B, T, D) partials if split"""
    B, T, D = x.shape
    # a symbolic token count runs padded to the max size, the garbage tokens are sliced off
    if not isinstance(T, int): x, sel, probs = x.pad_to(x.max_shape), sel.pad_to(sel.max_shape), probs.pad_to(probs.max_shape)
    gate_exps, down_exps = self.ffn_gate_exps, self.ffn_down_exps
    # (batched decoding: a few rows an expert, small tiles)
    tile = getenv("MOE_BATCH_TILE", 16) if _batch_rows[0] is not None and _batch_prefill[0] is None else MOE_TILE
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
    if half and gate_exps.expert_parallel and _sp_rows[0] is not None and isinstance(T, int) and tokens % len(cast(tuple, x.device)) == 0 and \
       tokens == T and getenv("MOE_SCATTER", 1):
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
    # T (hyper-connections): x has static padded rows of which the first T are the tokens, the output too
    self._init_state(x)
    # we pass in the weights implicitly so we unpack the GGUF on the fly
    # (decoding: the output's quantization made by its last kernel is returned too, the next block's input's: BLOCK_Q8_OUT)
    q8_out = T is None and self.config.hc is not None and bool(getenv("BLOCK_Q8_OUT", 1))
    # (the block before's output's quantization, if x is that output: its function's arguments, the quantization of x in it)
    xq = _block_q8[0][1] if q8_out and _block_q8[0] is not None and _block_q8[0][0] is x.uop else ()
    @function(precompile=True, allow_implicit=True)
    def _run(x:Tensor, start_pos:int|UOp, T:int|UOp|None, *xq:Tensor):
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
    ret = _run(x, start_pos, T, *xq)
    if not q8_out: return ret
    out, *q = ret
    _block_q8[0] = (out.uop, tuple(q))
    return out

  # the phases of a sublayer on a micro batch (see micro_batch_blocks): functions, scheduled once
  def _phase_pre(self, hc:HyperConnection, norm:nn.RMSNorm, xs:Tensor) -> tuple[Tensor, Tensor, Tensor]:
    @function(precompile=True, allow_implicit=True)
    def _pre(xs:Tensor) -> tuple[Tensor, Tensor, Tensor]:
      hn, post, comb = hc.pre_split(xs, norm)
      return hn.contiguous(), post.contiguous(), comb.contiguous()
    return _pre(xs)
  def _phase_sub(self, sub:str, hn:Tensor, start_pos:int|UOp, T:int|UOp) -> Tensor:
    @function(precompile=True, allow_implicit=True)
    def _sub(hn:Tensor, start_pos:int|UOp, T:int|UOp) -> Tensor:
      # (AG_Q8: sent as int8 groups of 32, as the quantized layers would make them)
      x = all_gather_q8(hn, 1) if getenv("AG_Q8", 1) and hn.shape[-1] % 256 == 0 else all_gather(hn, 1, dtypes.half)
      _partial_out[0], _sp_rows[0] = True, hn
      # (in half: the layers' kernels read it so, a cast to float and back is two copies)
      xin = x if getenv("SUB_HALF_IN", 1) else x.float()
      if sub != "attn": _ffn_rows[0] = T
      try: out = self._attention(xin, start_pos, T) if sub == "attn" else self._feed_forward(xin)
      finally: _partial_out[0], _sp_rows[0], _ffn_rows[0] = False, None, None
      # partial sums, or the whole output
      # (summed and split already: moe_combine_scatter)
      done = out.ndim == hn.ndim and token_sharded(out)
      return (out if done else reduce_scatter(out, 1) if out.ndim == hn.ndim + 1 else shard_rows(out, 1)).contiguous()
    return _sub(hn, start_pos, T)
  def _phase_post(self, hc:HyperConnection, out:Tensor, post:Tensor, comb:Tensor, xs:Tensor) -> Tensor:
    @function(precompile=True, allow_implicit=True)
    def _post(out:Tensor, post:Tensor, comb:Tensor, xs:Tensor) -> Tensor:
      # (HC_STREAMS_HALF: the streams kept in half, their kernels read half the bytes)
      ret = hc.post_split(out, post, comb, xs)
      return (ret.cast(dtypes.half) if getenv("HC_STREAMS_HALF", 1) else ret).contiguous()
    return _post(out, post, comb, xs)

def _on_queue(ts:tuple[Tensor, ...], queue:str) -> tuple[Tensor, ...]:
  """the outputs of a function call, its kernels on the compute queue `queue` (its copies on the copy queue of that index)"""
  def call_of(u:UOp) -> UOp: return u.src[1] if u.op is Ops.AFTER and u.src[1].op is Ops.CALL else call_of(u.src[0])
  call = call_of(ts[0].uop)
  def requeue(u:UOp) -> UOp:
    if u.op is Ops.AFTER and u.src[1] is call: return u.replace(src=(u.src[0], call.replace(arg=replace(call.arg, queue=queue)), *u.src[2:]))
    return u.replace(src=(requeue(u.src[0]), *u.src[1:]))
  return tuple(Tensor(requeue(t.uop)) for t in ts)

def micro_batch_blocks(blocks:list, xs:list[Tensor], starts:list, ts:list) -> list[Tensor]:
  """
  the blocks on micro batches of rows (sequence parallel streams, a batch starts at the end of the tokens before it). a sublayer is three
  phases: the coefficients and its normed input (then gathered), its compute (its partial sums then scattered), its output added to the streams.
  a sublayer of a batch runs after the one of the batch before it: it reads the state and the cache it writes (implicit, not graph inputs)
  """
  def after(x:Tensor, dep:Tensor|None) -> Tensor: return x if dep is None else Tensor(x.uop.after(*dep.uop.src[1:]))
  def queued(i:int, *outs:Tensor) -> tuple[Tensor, ...]: return _on_queue(outs, f"COMPUTE:{i}") if i and getenv("MB_QUEUES", 1) else outs
  subs = [(block, hc, norm, sub) for block in blocks for hc, norm, sub in ((block.hc_attn, block.attn_norm, "attn"),
                                                                           (block.hc_ffn, block.ffn_norm, "ffn"))]
  for block in blocks: block._init_state(xs[0])
  done:list[Tensor] = []
  for i in range(len(xs)):
    for k, (block, hc, norm, sub) in enumerate(subs):
      if k == 0: hn, post, comb = queued(i, *block._phase_pre(hc, norm, xs[i]))
      # (the batch before it ran this sublayer: its output's call, sublayer k of every batch is kept in done)
      out = queued(i, block._phase_sub(sub, after(hn, done[k] if i else None), starts[i], ts[i]))[0]
      if i == 0: done.append(out)
      else: done[k] = out
      xs[i] = queued(i, block._phase_post(hc, out, post, comb, xs[i]))[0]
      if k + 1 < len(subs): hn, post, comb = queued(i, *subs[k+1][0]._phase_pre(subs[k+1][1], subs[k+1][2], xs[i]))
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

  def _attention(self, x:Tensor, start_pos:int|UOp, T:int|UOp|None=None) -> Tensor:
    assert T is None, "rows mode is for MLA and gated deltanet blocks"
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
  pool_pk:Tensor  # batched decoding: the caches of every slot, and of all of them (MLATransformerBlock._init_slots)
  pool_ring:Tensor
  slot_pk:list[Tensor]
  slot_ring:list[Tensor]
  def __init__(self, config:TransformerConfig):
    self.idx, self.max_context, self.rewind = (idx:=unwrap(config.indexer)), config.max_context, config.rewind
    self.attn_q_b, self.attn_k = Linear(config.q_lora_rank, idx.n_heads * idx.head_dim, bias=False), Linear(config.dim, idx.head_dim, bias=False)
    self.proj, self.k_norm = Linear(config.dim, idx.n_heads, bias=False), nn.LayerNorm(idx.head_dim, eps=idx.norm_eps)
    if idx.kpool > 1:
      self.compressor_gate, self.compressor_ape = Linear(config.dim, idx.head_dim, bias=False), {"weight": Tensor.zeros(idx.kpool, idx.head_dim)}

  def _init_state(self, x:Tensor):
    if hasattr(self, "cache_pk"): return
    idx, max_pools = self.idx, round_up(self.max_context, self.idx.kpool) // self.idx.kpool
    # the sparse path (RDNA4): no mask, the pools a query picks are lists (decode) or per 16 queries (prefill, see pool_union)
    self.sparse = x.shape[0] == 1 and amd_custom_kernels_supported(x.device) and _wmma_rdna4(x.device) and 16 % idx.kpool == 0 and \
      idx.head_dim % 16 == 0 and max_pools % 64 == 0 and bool(getenv("INDEXER_SPARSE", 1))
    # the pooled keys, a call pools its tokens' pools only. the sparse path keeps them in half and the tokens of the last incomplete pool in a
    # ring (the last tokens: the next call can start before the end, see pool_keys), the other one every token's key and gate
    self.cache_pk = Tensor.zeros(*(() if self.sparse else (x.shape[0],)), max_pools, idx.head_dim,
                                 dtype=dtypes.half if self.sparse else dtypes.float32, device=x.device).contiguous()
    ring = round_up(max(2*idx.kpool, self.rewind + idx.kpool - 1), idx.kpool)
    if self.sparse: self.ring = Tensor.zeros(ring, 2, idx.head_dim, dtype=dtypes.half, device=x.device).contiguous()
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
    # a few (the drafts of speculative decoding): the same, a row a query
    if Tq <= 8 and idx.n_heads % 16 == 0 and getenv("INDEXER_DECODE_ROWS", 1):
      return indexer_decode_index_rows(iq, pk, weights, start_pos, pools, k, idx.kpool)
    # the scores on the tensor cores (rows of 16 queries), a kernel per query picks its pools
    rows = round_up(Tq, 16)
    iq, weights = iq.pad(((0, rows-Tq), None, None)), weights.pad(((0, rows-Tq), None))
    # blocks of rows one after the other: the scores of a block (rows x the max pools) are its memory, not the ones of the chunk
    block, picks = round_up(min(rows, getenv("INDEXER_ROWS", 1056)), 16), None
    for r0 in range(0, rows, block):
      qb = iq[r0:r0+block].contiguous()
      if picks is not None: qb = Tensor(qb.uop.after(*picks.uop.src[1:]))  # after the last block's lists: its scores are done
      if picks is None: picks = Tensor.empty(rows, k*idx.kpool + idx.kpool + 1, dtype=dtypes.int32, device=x.device)
      picks = pool_index_rows(indexer_scores(qb, pk, weights[r0:r0+block], pools), start_pos, pools, k, idx.kpool, row0=r0, out=picks)
    return unwrap(picks)[:Tq]

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
    if (groups:=_batch_rows[0]) is not None: return self._attention_batch(x, groups)
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
    # a few tokens: a thread an output (the same values)
    bmm = half and B == 1 and isinstance(Tq, int) and Tq <= 8 and amd_custom_kernels_supported(x.device) and bool(getenv("MLA_BMM", 1))
    if bmm: q_nope = mla_bmm(q_nope, self.attn_k_b["weight"])
    else: q_nope = q_nope.half().contiguous().matmul(k_b, dtype=dtypes.float).contiguous() if half else q_nope @ k_b
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
    tc = Tq > getenv("MLA_BMM_ROWS", 64) and self.attn_k_b["weight"].dtype == dtypes.half
    if tc: q = q_nope.half().contiguous().matmul(self.attn_k_b["weight"].transpose(-1, -2), dtype=dtypes.float).contiguous()
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
      iw = (idx.proj(x) * (ic.n_heads * ic.head_dim)**-0.5)[0]
    # one kernel for all the sequences (the same rows each, the caches of slot g the g-th of the pools): their positions in a tensor
    assert idx is not None and len({T for _, T in groups}) == 1, "batched decoding: the sparse attention of the indexer, groups of the same rows"
    scale, positions = 1.0/self.config.head_dim**0.5, _group_positions(groups, x.device)
    ctx, D, P, R = int(self.pool_k.shape[2]), int(self.pool_k.shape[-1]), int(idx.slot_pk[0].shape[0]), int(idx.slot_ring[0].shape[0])
    # (prefills: the slots of the prompts, their token counts)
    slots, counts = (pf.slots.to(x.device), pf.counts.to(x.device)) if (pf:=_batch_prefill[0]) is not None else (None, None)
    cache = grouped_rows(self.pool_k.reshape(-1, D), k_store[0, 0], positions, ctx, slots)
    pk = pool_keys(idx.pool_pk, idx.pool_ring, kg.half(), idx.compressor_ape["weight"], None, Tq, grouped=(positions, P, R, slots, counts))
    picks = indexer_decode_index_rows(iq, pk, iw, None, None, ic.top_k // ic.kpool, ic.kpool, grouped=(positions, P, slots))
    attn = flash_attention_mla_sparse(q, cache, picks, scale, None, grouped=(positions, ctx, slots))[..., :self.config.kv_lora_rank]
    if tc: attn = attn.half().contiguous().matmul(self.attn_v_b["weight"].transpose(-1, -2), dtype=dtypes.float).contiguous()
    else: attn = mla_bmm(attn, self.attn_v_b["weight"])
    return _out(self.attn_output, attn.transpose(1, 2).reshape(B, Tq, -1))

  def _init_slots(self, n:int, ctx:int):
    """the caches of n sequences of batched decoding of at most ctx tokens (one slot each)"""
    if len(getattr(self, "slot_k", [])) >= n: return
    self.pool_k, self.slot_k = _slot_pool(n, self.cache_k, (1, 1, ctx, self.cache_k.shape[-1]))
    if (idx:=self.indexer) is not None:
      pools = round_up(ctx, idx.idx.kpool) // idx.idx.kpool
      idx.pool_pk, idx.slot_pk = _slot_pool(n, idx.cache_pk, (pools, idx.cache_pk.shape[-1]))
      idx.pool_ring, idx.slot_ring = _slot_pool(n, idx.ring)

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

  def _prefill_batch(self, x:Tensor, pf:BatchPrefill) -> Tensor:
    """
    prefills of G prompts (P rows each, their first counts[g] the tokens) from position 0: the scan of a batch of G (rows past a prompt are no-ops),
    the states after the tokens into the slots (the one of position count in a slot's S)
    """
    P, (Bt, Tq, Dm) = pf.P, x.shape
    G, H, V, K, C = Tq // P, self.num_v_heads, self.head_v_dim, self.head_k_dim, self.ssm_conv_kernel
    xg, real = x.reshape(G, P, Dm).half(), pf.real.reshape(G, P, 1).to(x.device)
    out_gate = self.ssm_g_b(self.ssm_g_a(xg)).reshape(G, P, H, V)
    beta = self.ssm_beta(xg).sigmoid().reshape(G, P, H)
    g, a = (self.ssm_f_b(self.ssm_f_a(xg)).float() + self.ssm_dt["bias"]).reshape(G, P, H, -1), self.ssm_a.reshape(H, -1)
    log_alpha = g.softplus() * a if (lb:=unwrap(self.config.ssm).gate_lower_bound) is None else (g * -a).sigmoid() * lb
    beta, log_alpha = real.where(beta, 0), real.unsqueeze(-1).where(log_alpha, 0)
    weights = [w["weight"] for w in (self.ssm_conv1d_q, self.ssm_conv1d_k, self.ssm_conv1d_v)]
    # the windows of the prompts from position 0 (zeros before), the conv state of a slot the last C-1 inputs before its prompt's end
    windows = [proj(xg).float().pad(((0, 0), (C-1, 0), (0, 0))).contiguous() for proj in (self.attn_q, self.attn_k, self.attn_v)]
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
    state = Tensor.empty(G, H, V, K, dtype=dtypes.float32, device=x.device, axis=self.recurrent_state.uop.axis and 1).zeros_like().contiguous()
    core = gated_delta_prefill(q, k, v, beta.transpose(1, 2).float(), log_alpha.transpose(1, 2).exp(), state, kq=kq).transpose(1, 2)
    # the states into the slots (the scan updated state in place: after it)
    S = int(self.slot_rec[0].shape[0])
    state = Tensor(state.uop.after(core.uop))
    slots, counts = pf.slots.to(x.device), pf.counts.to(x.device)
    pf.effects.append(slot_scatter(self.pool_rec.reshape(self.pool_rec.shape[0], 1, -1), state.reshape(G, 1, -1), slots, counts, S))
    # (a conv state: the window's rows count.. of the prompt, the C-1 inputs before its end)
    pf.effects.extend(slot_scatter(pool.reshape(pool.shape[0], C-1, -1), window, slots, counts, int(self.slot_conv[0][0].shape[0]), shift=1)
                      for pool, window in zip(self.pool_conv, windows))
    z = (self.ssm_norm(core) * out_gate.sigmoid()).cast(x.dtype).contiguous()
    return _out(self.ssm_out, z.reshape(Bt, Tq, -1))

  def _attention_batch(self, x:Tensor, groups:list[tuple[UOp, int]]) -> Tensor:
    """batched decoding: a group of rows a sequence, its states in its slot (the projections of all of them in one kernel each)"""
    if (pf:=_batch_prefill[0]) is not None: return self._prefill_batch(x, pf)
    x = x.half()
    projs, gate = [self.attn_q(x), self.attn_k(x), self.attn_v(x)], self.ssm_g_b(self.ssm_g_a(x))
    alpha, beta = self.ssm_f_b(self.ssm_f_a(x)), self.ssm_beta(x)
    weights = [w["weight"] for w in (self.ssm_conv1d_q, self.ssm_conv1d_k, self.ssm_conv1d_v)]
    # groups of the same size: a kernel for all of them (their states in the pools, their positions in a tensor)
    if len({T for _, T in groups}) == 1 and getenv("KDA_GROUPS", 1):
      S, SC, T = int(self.slot_rec[0].shape[0]), int(self.slot_conv[0][0].shape[0]), groups[0][1]
      z = kda_decode(self.pool_rec, self.pool_conv, projs, weights, gate, alpha, beta, self.ssm_dt["bias"], self.ssm_a,
                     unwrap(self.ssm_norm.weight), Tensor(groups[0][0]), unwrap(self.config.ssm).gate_lower_bound, self.ssm_norm.eps, row=0, rows=T,
                     positions=_group_positions(groups, self.pool_rec.device), groups=len(groups), S=S, SC=SC)
      return self.ssm_out(z)
    # (every row's kernel reads its row of the inputs)
    starts = list(itertools.accumulate((t for _, t in groups), initial=0))
    zs = [kda_decode(self.slot_rec[g], self.slot_conv[g], projs, weights, gate, alpha, beta, self.ssm_dt["bias"], self.ssm_a,
                     unwrap(self.ssm_norm.weight), Tensor(pos), unwrap(self.config.ssm).gate_lower_bound, self.ssm_norm.eps, row=starts[g], rows=T,
                     SC=int(self.slot_conv[g][0].shape[0]))
          for g, (pos, T) in enumerate(groups)]
    return self.ssm_out(zs[0].cat(*zs[1:], dim=1))

  def _init_slots(self, n:int, ctx:int):
    """
    the states of n sequences of batched decoding (one slot each): S of each state, the one of a position p is p%S (see slot_states)
    """
    if len(getattr(self, "slot_rec", [])) >= n: return
    S, SC = slot_states(self.config.spec_drafts)
    # (in one buffer of every device: slot_store writes what a step reads)
    # (in float as the model's: a step reads and writes it, in half its rounding adds up over a long reply. in half for many slots
    # (SLOT_STATE_HALF, default past 8): their memory, a long context's model and 32 slots of it don't fit)
    half = getenv("SLOT_STATE_HALF", -1) == 1 or (getenv("SLOT_STATE_HALF", -1) == -1 and n > 8)
    self.pool_rec, self.slot_rec = _slot_pool(n, self.recurrent_state, (S, *self.recurrent_state.shape[1:]), dtypes.half if half else None)
    pools = [_slot_pool(n, c, (SC, *c.shape[1:])) for c in self.conv_states]
    self.pool_conv, self.slot_conv = [p for p, _ in pools], [[slots[i] for _, slots in pools] for i in range(n)]

  def _attention(self, x:Tensor, start_pos:int|UOp, T:int|UOp|None=None) -> Tensor:
    if (groups:=_batch_rows[0]) is not None: return self._attention_batch(x, groups)
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
    if is_kda and isinstance(Tq, int) and Tq <= self.config.spec_drafts + 1 and B == 1 and not rows and self.ssm_conv_kernel == 4 and \
       self.head_k_dim == self.head_v_dim == 128 and amd_custom_kernels_supported(x.device) and getenv("KDA_DECODE", 1):
      # a few tokens (one, or the drafts of speculative decoding): a kernel for the convolutions, the delta rule, the norm and the gate
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
      core = gated_delta_prefill(q, k, v, beta, alpha, state, Tensor(start_pos), end_pos=start_pos + T, kq=kq).transpose(1, 2)
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
      # S of each state: the state at a position p is p%S (see _attention). S-1 steps can be taken back (config.rewind)
      S = self.config.rewind + 1
      def state(*shape:int, axis:int) -> Tensor:
        return (Tensor.zeros(S, *shape).shard(x.device, axis+1) if split else Tensor.zeros(S, *shape, device=x.device)).clone()
      if hasattr(self, "attn_qkv"): self.conv_state = state(x.shape[0], self.ssm_conv_kernel-1, self.conv_channels, axis=2)
      else: self.conv_states = [state(x.shape[0], self.ssm_conv_kernel-1, d, axis=2) for d in self.qkv_dims]
      self.recurrent_state = state(x.shape[0], self.num_v_heads, self.head_v_dim, self.head_k_dim, axis=1)

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
      self.blk.append(GatedDeltaNetBlock(c, config.ssm) if config.ssm and config.ssm_layers[i] else block_cls(c))
    self.token_embd  = nn.Embedding(config.vocab_size, config.dim)
    self.output_norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.output = Linear(config.dim, config.padded_vocab_size or config.vocab_size, bias=False)
    if config.mtp: self.mtp = MTPBlock(config)
    self.max_context = config.max_context
    self.has_recurrent_block = any(isinstance(b, GatedDeltaNetBlock) for b in self.blk)
    self._cached_tokens: list[int] = []
    self._effects: list[Tensor]|None = None  # tensors that forward realizes with its output (see _forward_effects)
    self._ckpt_tokens: list[int] = []  # the prompt of the recurrent state's checkpoint (_checkpoint)
    # a jit for every token count of a call (the max size of a symbolic chunk, 1 for decoding): static padded shapes of a small chunk don't pay
    # for a big one
    self.jits:dict[int, Any] = {}
    self._pending_load:list[Callable[[], None]] = []  # the work left of loading (from_gguf defer_load)
    self._loaded = False  # the weights are loaded (from_gguf)
    self._scratch:dict = {}  # the memory of the jits' intermediates: they run one at a time

  def forward(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    x = emb = self.token_embd(tokens.to(self.token_embd.weight.device)).float()  # (B, T, D)
    if (hc:=self.config.hc) is not None: x = x.unsqueeze(2).expand(*x.shape[:2], hc.count, x.shape[-1]).contiguous()
    # a symbolic chunk with hyper-connections runs every block on static padded rows (their real count is T), sliced at the end
    if hc is not None and not isinstance(T:=x.shape[1], int) and amd_custom_kernels_supported(x.device) and \
       all(isinstance(b, (MLATransformerBlock, GatedDeltaNetBlock)) for b in self.blk):
      x = x.pad_to(x.max_shape).contiguous()
      R = int(x.shape[1])
      # sequence parallel: every device keeps the streams of its rows, the blocks gather and scatter the sublayers' inputs and outputs
      # micro batches (on queues of their own, AM_MEC_PIPES: the copies of one run with the kernels of the other)
      mb = getenv("MICRO_BATCHES", 1)
      if (sp:=self.sequence_parallel(R)) and (mb > 1 or getenv("MB_PHASES", 1)) and self.sequence_parallel(R // mb) and R % mb == 0:
        # the blocks in phases (micro_batch_blocks), on micro batches of rows if MICRO_BATCHES > 1 (the copies of one overlap the kernels of
        # the next). a batch starts at the end of the tokens before it (all padding if there are none): its states follow theirs
        n, xs, ts, starts = R // mb, [], [], []
        for i in range(mb):
          xs.append(shard_rows(x[:, i*n:(i+1)*n].contiguous(), 1))
          ts.append((T - i*n).maximum(0).minimum(n))
          starts.append(start_pos + (T.minimum(i*n) if i else 0))
        xs = micro_batch_blocks(self.blk, xs, starts, ts)
        x = Tensor.cat(*[all_gather(xi.mean(2), 1) for xi in xs], dim=1)
      else:
        if sp: x = shard_rows(x, 1)
        for block in self.blk: x = block(x, start_pos, T)
        x = all_gather(x.mean(2), 1) if sp else x.mean(2)
      if self.config.spec_drafts: x = self._mtp_prefill(emb, x, cast(UOp, start_pos), T, sp=bool(sp))
      x = x[:, :T]
    else:
      for block in self.blk: x = block(x, start_pos)
    if hc is not None and x.ndim == 4: x = x.mean(2)
    # only run the output projection on the last token
    tok = self._sample(self.output_norm(x[:, -1:])[:, -1, :], temperature, tokens.device)
    q8_quantize_cache_clear()  # don't keep the graph alive
    # the tokens are a view of generate's token buffer: the sampled token goes after them unless that is a prompt token (chunked prefill)
    if (buf:=getattr(self, "_token_buf", None)) is not None and tokens.uop.base is buf.uop.base:
      tok = (dst:=buf[:, (nxt:=start_pos + tokens.shape[1]):nxt+1]).assign((dst < 0).where(tok.cast(buf.dtype), dst))
    return tok

  def _sample(self, h:Tensor, temperature:Tensor|None, device:str|tuple[str, ...]|None) -> Tensor:
    """
    the token (B, 1) after every normed hidden state h (B, D), sampled (greedy without a temperature). on device unless the output is split.
    temperature: (rows, 1) the temperature of every row (or one), or (rows, 2) and its top_p (only the tokens of the top_p nucleus are sampled)
    """
    logits = self.output(h)
    # Gumbel-max trick: argmax(logits/temp - log(-log(uniform))) is equivalent to sampling from softmax(logits/temp)
    if temperature is not None:
      tp = temperature.to(logits.device).reshape(-1, temperature.shape[-1])
      z = logits.float() / tp[:, :1].maximum(1e-12)
      logits = z - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()
      if tp.shape[1] == 2: logits = nucleus(z, self.config.vocab_size, tp[:, 1:]).where(logits, -math.inf)
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
    hn = self.output_norm(x).float()
    emb, carry = emb.pad_to((1, x.shape[1], emb.shape[-1])), self._spec_h[0:1].unsqueeze(0).contiguous()
    # the MTP layer reads the carry before it's written (the schedule orders an assign after the reads). realized with the call's output
    inp = self.mtp.enorm(emb).cat(self.mtp.hnorm(carry.cat(hn[:, :-1], dim=1)), dim=-1)
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
    if (hc:=self.config.hc) is not None: x = x.unsqueeze(2).expand(*x.shape[:2], hc.count, x.shape[-1]).contiguous()
    for block in self.blk: x = block(x, start_pos)
    if hc is not None: x = x.mean(2)
    tok = self._sample(self._spec_h.shrink(((par*(k+1), (par+1)*(k+1)), None)).assign(self.output_norm(x)[0].float()), temperature, y.device)
    q8_quantize_cache_clear()  # don't keep the graph alive
    out = tok.reshape(1, k+1).cat(*drafts, dim=1).cast(dtypes.int32)
    return self._spec_out.assign(Tensor(out.uop.mselect(0)) if isinstance(out.device, tuple) else out)

  def _spec_call(self, mtp_pos:int, j:int, par:int, temperature:Tensor):
    """a step of speculative decoding (forward_spec, its jit)"""
    k, pos = self.config.spec_drafts, mtp_pos + j
    if not hasattr(self, "_spec_jit"): self._spec_jit = TinyJit(self.forward_spec, capture_first=getattr(self, "_states_ready", False),
                                                                scratch=self._scratch)
    if not hasattr(self, "_spec_vars"):
      # (made once: a step binds them)
      pos_var = functools.partial(UOp.variable, min_val=0, max_val=self.max_context-1)
      self._spec_vars = (pos_var("mtp_pos"), UOp.variable("spec_j", 0, k), pos_var("start_pos"), UOp.variable("spec_par", 0, 1),
                         tuple(pos_var(f"spec_pos{i}") for i in range(k-1)))
    v_mtp, v_j, v_pos, v_par, v_chain = self._spec_vars
    self._spec_jit(v_mtp.bind(mtp_pos), v_j.bind(j), v_pos.bind(pos), v_par.bind(par), temperature, *(v.bind(pos+1+i) for i,v in enumerate(v_chain)))

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
      while min(out:=host_out.tolist()) < 0: pass
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

  # ******** batched decoding: the next token of many sequences in one step (a row each), every one with its own states (a slot) ********
  # a sequence is prefilled as usual (the states of the model), then its states are copied into a free slot (slot_store); steps take the
  # sequences of the slots 0..n-1 (the rows of a step), the projections, experts and the head read their weights once for all of them

  BATCH_SIZES = tuple(int(x) for x in str(getenv("BATCH_SIZES", "1,2,4,16,32,64,80")).split(","))
  def batch_slots(self) -> int:
    # (the memory of the slots: a shorter context of the model's own frees some)
    if not (self.has_recurrent_block and self.config.kv_lora_rank > 0): return 0
    return min(getenv("BATCH_SLOTS", 80 if self.max_context <= 65536 else 32), max(self.BATCH_SIZES))
  def batch_ctx(self) -> int: return min(getenv("BATCH_CTX", 2048), self.max_context)
  def batch_sizes(self) -> tuple[int, ...]:
    """
    the rows of the steps of batched decoding (a jit each): the sizes below the slots and the slots (BATCH_SLOTS of another size). none with
    BATCH_DECODE=0 (no batched decoding, the slots only keep conversations: its steps aren't captured)
    """
    if not getenv("BATCH_DECODE", 1): return ()
    return tuple(sorted({s for s in self.BATCH_SIZES if s < n} | {n})) if (n:=self.batch_slots()) else ()

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
      tmv, self._batch_temp = _host_buffer(devs, 8*max(*self.BATCH_SIZES, self.BPF_G), dtypes.float32)
      self._batch_host_temp = tmv.cast('f')
      self._batch_host_out, self._batch_out = _host_ints(devs[0] if isinstance(devs, tuple) else devs, max(self.BATCH_SIZES))
      # prefills in a step: the tokens of the prompts, the token after each
      self._bpf_host_in, self._bpf_in = _host_ints(devs, self.BPF_G * self.BPF_P)
      self._bpf_host_out, self._bpf_out = _host_ints(devs[0] if isinstance(devs, tuple) else devs, self.BPF_G)
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
        todo += [blk.pool_k] + ([blk.indexer.pool_pk, blk.indexer.pool_ring] if blk.indexer is not None else [])
    for t in todo:
      for u in t.uop.toposort():
        if u.op is Ops.BUFFER:
          for x in (b.bufs if isinstance(b:=u.buffer, MultiBuffer) else [b]): x.ensure_allocated()
    self._slots_ready = True

  def _slot_blocks(self) -> list[FFNBlock]: return self.blk + ([self.mtp] if self.config.spec_drafts else [])
  def _slot_copy(self, slot:UOp, rp:UOp, wp:UOp, wpc:UOp, hrow:UOp):
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
        ctx = int(b.slot_k[0].shape[2])
        copies.append(rows(b.pool_k, slot).assign(b.cache_k[:, :, :ctx]))
        if (idx:=b.indexer) is not None:
          P, R = int(idx.slot_pk[0].shape[0]), int(idx.slot_ring[0].shape[0])
          copies += [rows(idx.pool_pk, slot*P, P).assign(idx.cache_pk[:P]), rows(idx.pool_ring, slot*R, R).assign(idx.ring)]
    Tensor.realize(*copies)

  def _slot_load_copy(self, slot:UOp, rp:UOp, wp:UOp, wpc:UOp, hrow:UOp):
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
        ctx = int(b.slot_k[0].shape[2])
        copies.append(b.cache_k[:, :, :ctx].assign(rows(b.pool_k, slot)))
        if (idx:=b.indexer) is not None:
          P, R = int(idx.slot_pk[0].shape[0]), int(idx.slot_ring[0].shape[0])
          copies += [idx.cache_pk[:P].assign(rows(idx.pool_pk, slot*P, P)), idx.ring.assign(rows(idx.pool_ring, slot*R, R))]
    Tensor.realize(*copies)

  def _slot_pools(self) -> list[tuple[Tensor, int]]:
    """every pool of the slots' states and the rows of a slot in it"""
    pools:list[tuple[Tensor, int]] = [(self.pool_spec_h, int(self.slot_spec_h[0].shape[0]))] if hasattr(self, "pool_spec_h") else []
    for b in self._slot_blocks():
      if isinstance(b, GatedDeltaNetBlock):
        pools += [(b.pool_rec, int(b.slot_rec[0].shape[0]))] + [(p, int(b.slot_conv[0][0].shape[0])) for p in b.pool_conv]
      if isinstance(b, MLATransformerBlock):
        pools.append((b.pool_k, 1))
        if (idx:=b.indexer) is not None: pools += [(idx.pool_pk, int(idx.slot_pk[0].shape[0])), (idx.pool_ring, int(idx.slot_ring[0].shape[0]))]
    return pools
  def _slot_move_copy(self, src:UOp, dst:UOp):
    def rows(t:Tensor, start:UOp, n:int) -> Tensor: return t.shrink(((start, start+n),) + (None,)*(t.ndim-1))
    Tensor.realize(*[rows(p, dst*n, n).assign(rows(p, src*n, n).contiguous()) for p, n in self._slot_pools()])
  def _slot_jits(self):
    if not hasattr(self, "_slot_jit"): self._slot_jit = TinyJit(self._slot_copy, capture_first=True)
    if not hasattr(self, "_slot_move_jit"): self._slot_move_jit = TinyJit(self._slot_move_copy, capture_first=True)
    if not hasattr(self, "_slot_load_jit"): self._slot_load_jit = TinyJit(self._slot_load_copy, capture_first=True)
  def _slot_move_vars(self, src:int, dst:int) -> tuple[UOp, ...]:
    v = functools.partial(UOp.variable, min_val=0, max_val=self.batch_slots()-1)
    return v("slot_src").bind(src), v("slot_dst").bind(dst)
  def _slot_store_vars(self, i:int, length:int, hrow:int) -> tuple[UOp, ...]:
    S, SC = slot_states(self.config.spec_drafts)
    rp = UOp.variable("slot_rp", 0, self.config.rewind).bind(length % (self.config.rewind + 1))
    return (UOp.variable("slot", 0, self.batch_slots()-1).bind(i), rp, UOp.variable("slot_wp", 0, S-1).bind(length % S),
            UOp.variable("slot_wpc", 0, SC-1).bind(length % SC), UOp.variable("slot_hrow", 0, 2*self.config.spec_drafts+1).bind(hrow))

  def slot_move(self, src:int, dst:int):
    """the states of slot src into slot dst (the sequences of a batch in the slots of its first rows)"""
    self._slot_jits()
    self._slot_move_jit(*self._slot_move_vars(src, dst))

  def slot_store(self, i:int, length:int, hrow:int=0):
    """the states of the sequence just prefilled (or decoded) by the model, of `length` tokens, into slot i (hrow: the row of _spec_h)"""
    self._init_slots()
    self._slot_jits()
    for d in to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device)): Device[d].synchronize()  # (run ahead steps of the model)
    self._slot_jit(*self._slot_store_vars(i, length, hrow))

  def slot_load(self, i:int, length:int, hrow:int=0):
    """
    the states of slot i (a sequence of `length` tokens, stored by slot_store or decoded batched there) back into the model's: it continues alone
    from them (the cache of a conversation kept in a slot). hrow: the slot's row of the last token's hidden state. the caller sets _cached_tokens
    """
    self._init_slots()
    self._slot_jits()
    for d in to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device)): Device[d].synchronize()
    self._slot_load_jit(*self._slot_store_vars(i, length, hrow))
    self._ckpt_tokens, self._spec_row = [], 0

  # ** a conversation's states in host memory (HOST_STASH in the server: the ones longer than a slot): its bytes copied, no kernel
  def _host_parts(self, length:int, hrow:int, ckpt:bool=False) -> list[tuple[Buffer, int, int]]:
    """
    the bytes of the model's states at the end of its `length` tokens: (a device's buffer, offset, size) each. hrow: the row of _spec_h of the
    last token's hidden state (row 0 when loaded). ckpt: and the checkpoint's (the states at the end of its prompt, _checkpoint: the caches
    up to it are the conversation's)
    """
    parts:list[tuple[Buffer, int, int]] = []
    def add(t:Tensor, axis:int, start:int, n:int):
      # (the rows start..start+n of axis, a prefix of every device's buffer past the axes of one before it)
      bufs = [u.buffer for u in t.uop.toposort() if u.op is Ops.BUFFER]
      assert len(bufs) == 1, f"a state of {len(bufs)} buffers"
      per = bufs[0].bufs if isinstance(bufs[0], MultiBuffer) else [bufs[0]]
      shape = list(t.shape)
      if isinstance(t.device, tuple) and (sa:=t.uop.axis) is not None: shape[sa] //= len(t.device)
      assert prod(shape[:axis]) == 1 and per[0].nbytes == prod(shape) * t.dtype.itemsize, f"{t.shape} isn't rows of a buffer on axis {axis}"
      row = int(prod(shape[axis+1:])) * t.dtype.itemsize
      parts.extend((b, start*row, n*row) for b in per)
    if self.config.spec_drafts: add(self._spec_h, 0, hrow, 1)
    rp = length % (self.config.rewind + 1)
    for b in self._slot_blocks():
      if isinstance(b, GatedDeltaNetBlock):
        for t in [b.recurrent_state] + ([b.conv_state] if hasattr(b, "conv_state") else b.conv_states): add(t, 0, rp, 1)
      if isinstance(b, MLATransformerBlock):
        add(b.cache_k, 2, 0, length)
        if (idx:=b.indexer) is not None:
          pools = (length + idx.idx.kpool - 1) // idx.idx.kpool
          if idx.sparse: add(idx.cache_pk, 0, 0, pools), add(idx.ring, 0, 0, int(idx.ring.shape[0]))
          else: add(idx.cache_pk, 1, 0, pools), add(idx.cache_kg, 1, 0, pools * idx.idx.kpool)
    if ckpt:
      for t in self._ckpt: add(t, 0, 0, int(t.shape[0]))
    return parts

  def _host_copy(self, pairs:list[tuple[Buffer, Buffer]]):
    # (in one go: the devices copy at once, CPU memory is the devices' to copy to (none staged))
    run_linear(UOp(Ops.LINEAR, src=tuple(UOp.from_buffer(d, dtypes.uint8).store_call(UOp.from_buffer(s, dtypes.uint8)) for d, s in pairs)),
               update_stats=False)
    for d in dedup([b.device for p in pairs for b in p]): Device[d].synchronize()

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
  def host_chunks(self, length:int, ckpt:bool=False) -> int:
    """the chunks of host memory (host_alloc) host_save needs for a conversation of `length` tokens (and its checkpoint)"""
    return self._host_layout(self._host_parts(length, 0, ckpt))[-1][0] + 1
  def host_ckpt(self, tokens:list[int]) -> int:
    """the tokens of the model's checkpoint (the prompt of its last reply) if it is one of the conversation `tokens` (host_save can keep it)"""
    ck = self._ckpt_tokens if hasattr(self, "_ckpt") else []
    return len(ck) if 0 < len(ck) < len(tokens) and tokens[:len(ck)] == ck else 0
  def host_alloc(self) -> Buffer:
    """a chunk of host memory for host_save (CPU memory: the devices copy to it, mapped once: used again, not made again)"""
    return Buffer("CPU", self.HOST_CHUNK, options=BufferSpec(nolru=True)).allocate()

  def host_save(self, length:int, hrow:int, chunks:list[Buffer], ckpt:bool=False):
    """
    the states of the model's conversation of `length` tokens (hrow: the row of _spec_h) copied to host memory: chunks (host_alloc'd, as many
    as host_chunks). ckpt: and its checkpoint's (see host_ckpt)
    """
    for d in to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device)): Device[d].synchronize()  # (run ahead steps of the model)
    parts = self._host_parts(length, hrow, ckpt)
    layout = self._host_layout(parts)
    assert len(chunks) > layout[-1][0], f"{len(chunks)} chunks for {layout[-1][0]+1}"
    self._host_copy([(chunks[c].view(n, off).ensure_allocated(), b.view(n, o).ensure_allocated()) for (b, o, n), (c, off) in zip(parts, layout)])

  def host_load(self, chunks:list[Buffer], length:int, ckpt:bool=False):
    """
    the states of a conversation of `length` tokens saved by host_save back into the model's (it continues from them, or from its checkpoint).
    sets no _cached_tokens, _ckpt_tokens
    """
    for d in to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device)): Device[d].synchronize()  # (run ahead steps of the model)
    parts = self._host_parts(length, 0, ckpt)
    layout = self._host_layout(parts)
    assert len(chunks) > layout[-1][0], "saved by another model"
    self._host_copy([(b.view(n, o).ensure_allocated(), chunks[c].view(n, off).ensure_allocated()) for (b, o, n), (c, off) in zip(parts, layout)])
    self._ckpt_tokens, self._spec_row = [], 0

  def forward_batch(self, *positions:UOp) -> Tensor:
    """a step of batched decoding: row i the sequence of slot i at positions[i], its token in _batch_in, its temperature in _batch_temp"""
    n, devs = len(positions), self.token_embd.weight.device
    _batch_rows[0], Q8_ROWS_I8[0] = [(p, 1) for p in positions], bool(getenv("BATCH_I8", 1))
    try:
      x = self.token_embd(self._batch_in[:, :n].to(devs)).float()  # (1, n, D)
      if (hc:=self.config.hc) is not None: x = x.unsqueeze(2).expand(*x.shape[:2], hc.count, x.shape[-1]).contiguous()
      for block in self.blk: x = block(x, 0)
      if hc is not None: x = x.mean(2)
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
    return [UOp.variable(f"batch_pos{i}", 0, self.batch_ctx()-1).bind(pos) for i, pos in enumerate(positions)]

  def batch_step(self, rows:list[tuple[int, int, float|tuple[float, float]]]) -> list[int]:
    """
    a step of batched decoding: rows (token, position, temperature (or temperature and top_p)) of the slots 0..len(rows)-1, their sampled tokens
    """
    self._init_slots()
    n = len(rows)
    m = next(s for s in self.batch_sizes() if s >= n)
    assert m <= self.batch_slots(), f"{n} rows, {self.batch_slots()} slots"
    # (the rows past n are the free slots' at position 0: what they write is overwritten when a sequence takes the slot)
    rows = rows + [(0, 0, 0.0)] * (m - n)
    for i, (tok, _, temp) in enumerate(rows): self._batch_host_in[i], self._batch_host_temp[2*i:2*i+2] = tok, _temp_pair(temp)
    self._batch_host_out[:m] = array.array('i', [-1] * m)
    self._batch_jit(m)(*self._batch_positions([pos for _, pos, _ in rows]))
    st = time.perf_counter()
    while min(self._batch_host_out[:m]) < 0:
      # (a step that doesn't come: the devices' errors, or none)
      if time.perf_counter() - st > 10:
        for d in to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device)): Device[d].synchronize()
        if min(self._batch_host_out[:m]) < 0: raise RuntimeError(f"batched step of {m} rows: no tokens after the devices finished")
    return list(self._batch_host_out[:n])

  def forward_batch_spec(self, *seqs:UOp) -> Tensor:
    """
    a step of batched speculative decoding (forward_spec of every sequence, its slot g the group g): seqs (mtp_pos, j, par) of every group,
    its k+1 tokens in _bspec_in, its temperature in _batch_temp. the MTP layer catches up on the k+1 positions from mtp_pos and drafts k tokens
    after start_pos = mtp_pos+j, the model verifies them. returns (1, n*(2k+1)): the k+1 sampled tokens and the k drafts of every group
    """
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
      drafts:list[Tensor] = []
      for i in range(k):
        drafts.append(self._sample(m[0], None, y.device).reshape(1, n))
        if i < k-1:
          _batch_rows[0] = [(mp + j + 1 + i, 1) for mp, j, _ in groups]
          m = self.mtp.predict(self.token_embd(drafts[-1].to(devs)).float(), m, 0)
      # verify: the model on every group's token at start_pos and its drafts, their hidden states kept (the rows of parity par)
      toks = Tensor.cat(*[y.shrink((None, (g*(k+1)+j, g*(k+1)+j+1))).cat(*[d[:, g:g+1] for d in drafts], dim=1)
                          for g, (_, j, _) in enumerate(groups)], dim=1)
      x = self.token_embd(toks.contiguous().to(devs)).float()
      if (hc:=self.config.hc) is not None: x = x.unsqueeze(2).expand(*x.shape[:2], hc.count, x.shape[-1]).contiguous()
      _batch_rows[0] = [(mp + j, k+1) for mp, j, _ in groups]
      for block in self.blk: x = block(x, 0)
      if hc is not None: x = x.mean(2)
      hn = self.output_norm(x)[0].float()
      hn = Tensor.cat(*[self.slot_spec_h[g].shrink(((par*(k+1), (par+1)*(k+1)), None)).assign(hn[g*(k+1):(g+1)*(k+1)])
                        for g, (_, _, par) in enumerate(groups)])
      temps = self._batch_temp[:2*n].reshape(n, 1, 2).expand(n, k+1, 2).reshape(n*(k+1), 2)
      tok = self._sample(hn, temps, y.device)
    finally: _batch_rows[0], Q8_ROWS_I8[0] = None, False
    q8_quantize_cache_clear()  # don't keep the graph alive
    out = tok.reshape(n, k+1).cat(*[d.reshape(n, 1) for d in drafts], dim=1).reshape(1, n*(2*k+1)).cast(dtypes.int32)
    return self._bspec_out[:, :n*(2*k+1)].assign(Tensor(out.uop.mselect(0)) if isinstance(out.device, tuple) else out)

  BPF_G, BPF_P = getenv("BPF_G", 8), getenv("BPF_P", 168)  # prefills in a step of them: prompts, their rows (at most)
  def forward_batch_prefill(self, *vs:UOp) -> Tensor:
    """
    prefills of G prompts into slots (from position 0): vs the token counts and the slots of the G, their P tokens (padded) in _bpf_in, their
    temperatures in _batch_temp. the model and the MTP layer on all of them (the caches and states of the slots), the token after every one's
    last. returns (1, G) the tokens
    """
    G, P, devs, k = self.BPF_G, self.BPF_P, self.token_embd.weight.device, self.config.spec_drafts
    cvars, svars = vs[:G], vs[G:2*G]
    # (a stack of variables has no device: onto zeros of the devices)
    def ints(xs:tuple[UOp, ...]) -> Tensor:
      return (Tensor.zeros(len(xs), dtype=dtypes.int32, device=devs) + Tensor.stack(*[Tensor(x) for x in xs]).cast(dtypes.int32)).contiguous()
    pf = BatchPrefill(P, ints(cvars), ints(svars), cvars, svars)
    _batch_rows[0], _batch_prefill[0], Q8_ROWS_I8[0] = [(UOp.const(0, dtypes.weakint), P)] * G, pf, False
    try:
      emb = self.token_embd(self._bpf_in.to(devs)).float()
      x = emb
      if (hc:=self.config.hc) is not None: x = x.unsqueeze(2).expand(*x.shape[:2], hc.count, x.shape[-1]).contiguous()
      if hc is not None and self.sequence_parallel(G*P):
        # sequence parallel (as a prefill chunk): every device the streams of its rows, the sublayers on all of them
        x = all_gather(micro_batch_blocks(self.blk, [shard_rows(x, 1)], [0], [G*P])[0].mean(2), 1)
      else:
        for block in self.blk: x = block(x, 0)
        if hc is not None: x = x.mean(2)
      hn = self.output_norm(x).float()  # (1, G*P, D)
      # the MTP layer on every prompt (its caches): a token's input is the hidden state of the one before (zero before the first)
      before = hn.reshape(G, P, -1)[:, :P-1].pad(((0, 0), (1, 0), (0, 0))).reshape(1, G*P, -1)
      if k: pf.effects.append(self.mtp(self.mtp.eh_proj(self.mtp.enorm(emb).cat(self.mtp.hnorm(before), dim=-1)), 0)[:, :1, :1].contiguous())
      # the last token of every prompt: the next token, its hidden state the slot's (speculative decoding's row 0)
      last = Tensor.cat(*[hn[0].shrink(((g*P + c - 1, g*P + c), None)) for g, c in enumerate(cvars)])
      if k: pf.effects.extend([self.pool_spec_h.shrink(((sl*2*(k+1), sl*2*(k+1)+1), None)).assign(last[g:g+1]) for g, sl in enumerate(svars)])
      tok = self._sample(last, self._batch_temp[:2*G].reshape(G, 2), self._bpf_in.device)
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
    return self.BPF_G if self.batch_sizes() and self.BPF_P and self.sequence_parallel(self.BPF_G * self.BPF_P) and \
      self.max_context <= getenv("BPF_MAX_CONTEXT", 65536) else 0
  def _bpf_jit(self):
    if not hasattr(self, "_bpf_jit_"): self._bpf_jit_ = TinyJit(self.forward_batch_prefill, capture_first=True, scratch=self._scratch)
    return self._bpf_jit_
  def _bpf_vars(self, counts:list[int], slots:list[int]) -> list[UOp]:
    return [UOp.variable(f"bpf_count{g}", 1, self.BPF_P).bind(c) for g, c in enumerate(counts)] + \
           [UOp.variable(f"bpf_slot{g}", 0, self.batch_slots()-1).bind(s) for g, s in enumerate(slots)]
  def batch_prefill(self, prompts:list[tuple[list[int], int, float|tuple[float, float]]]) -> list[int]:
    """prefills of prompts (tokens, slot, temperature) (at most BPF_G of BPF_P tokens) into their slots in one step: the token after every one"""
    self._init_slots()
    G, P, n = self.BPF_G, self.BPF_P, len(prompts)
    assert 0 < n <= G and all(0 < len(t) <= P for t, _, _ in prompts)
    # (the rest of the G: the last one again, the same writes)
    prompts = prompts + [prompts[-1]] * (G - n)
    for g, (toks, _, temp) in enumerate(prompts):
      self._bpf_host_in[g*P:(g+1)*P] = array.array('i', toks + [0] * (P - len(toks)))
      self._batch_host_temp[2*g:2*g+2] = _temp_pair(temp)
    self._bpf_host_out[:G] = array.array('i', [-1] * G)
    self._bpf_jit()(*self._bpf_vars([len(t) for t, _, _ in prompts], [s for _, s, _ in prompts]))
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
    k, ctx = self.config.spec_drafts, self.batch_ctx()
    return [v for g, (mp, j, par) in enumerate(seqs) for v in (UOp.variable(f"bspec_mtp{g}", 0, ctx-1).bind(mp),
            UOp.variable(f"bspec_j{g}", 0, k).bind(j), UOp.variable(f"bspec_par{g}", 0, 1).bind(par))]

  def batch_spec_step(self, seqs:list[tuple[list[int], int, int, int, float|tuple[float, float]]]) -> list[tuple[list[int], list[int]]]:
    """
    a step of batched speculative decoding of the slots 0..len(seqs)-1: seqs (the k+1 tokens of the last step, mtp_pos, j, par, temperature)
    of every one (see _spec_steps), their k+1 sampled tokens and k drafts
    """
    self._init_slots()
    k, n = self.config.spec_drafts, len(seqs)
    m = next(s for s in self.batch_sizes() if s >= n)
    assert m <= self.batch_slots(), f"{n} sequences, {self.batch_slots()} slots"
    # (the groups past n are the free slots' at position 0: what they write is overwritten when a sequence takes the slot)
    seqs = seqs + [([0]*(k+1), 0, 0, 1, 0.0)] * (m - n)
    for g, (y, _, _, _, temp) in enumerate(seqs):
      self._bspec_host_in[g*(k+1):(g+1)*(k+1)] = array.array('i', y)
      self._batch_host_temp[2*g:2*g+2] = _temp_pair(temp)
    self._bspec_host_out[:m*(2*k+1)] = array.array('i', [-1] * (m*(2*k+1)))
    self._bspec_jit(m)(*self._bspec_vars([(mp, j, par) for _, mp, j, par, _ in seqs]))
    st = time.perf_counter()
    while min(self._bspec_host_out[:m*(2*k+1)]) < 0:
      # (a step that doesn't come: the devices' errors, or none)
      if time.perf_counter() - st > 10:
        for d in to_tuple(cast(str|tuple[str, ...], self.token_embd.weight.device)): Device[d].synchronize()
        if min(self._bspec_host_out[:m*(2*k+1)]) < 0: raise RuntimeError(f"batched step of {m} sequences: no tokens after the devices finished")
    out = list(self._bspec_host_out[:n*(2*k+1)])
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
      if isinstance(v, Tensor): return [v] if not v.uop.is_realized else []
      return [t for x in v for t in lazy(x)] if isinstance(v, (list, tuple)) else []
    # (and the parameters of a model initialized but not loaded. a loaded one's are: the quantized weights' are their decoding, not used)
    todo = [t for h in holders for v in vars(h).values() for t in lazy(v)] + ([] if self._loaded else lazy(nn.state.get_parameters(self)))
    # with the weights still to read: their memory now, their values after
    if self._pending_load: _realize_later(todo, self._pending_load)
    else: Tensor.realize(*todo)
    self._states_ready = True

  def __call__(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    # a view of the token buffer is already realized: no schedule for the jit input
    if (buf:=getattr(self, "_token_buf", None)) is None or tokens.uop.base is not buf.uop.base: tokens = tokens.contiguous()
    key = int(T.vmax) if isinstance(T:=tokens.shape[1], UOp) else T
    if (jit:=self.jits.get(key)) is None:
      # the states and caches are set up before: a jit captures on its first call
      self._init_states()
      jit = self.jits[key] = TinyJit(self._forward_effects, capture_first=True, scratch=self._scratch)
    return jit(tokens, start_pos, temperature)[0]

  def _forward_effects(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> tuple[Tensor, ...]:
    # forward and the tensors realized with its output for their effects (e.g. the caches of the MTP layer)
    self._effects = []
    out = self.forward(tokens, start_pos, temperature)
    effects, self._effects = self._effects, None
    return (out, *effects)

  @staticmethod
  def from_gguf(gguf:Tensor|str|pathlib.Path, max_context:int|None=None,
                realize=bool(getenv("REALIZE", 0)), shard:int=1, defer_load:bool=False) -> tuple[Transformer, dict]:
    """defer_load: the weights are read later (warmup compiles meanwhile), before the first call"""
    # TODO: remove the need for copy to default device
    kv, entries = gguf_parse(gguf.to(None).realize() if isinstance(gguf, Tensor) else gguf)
    # deferred: the files are read ahead from now (the disk works while the model is made, the loader finds them in memory)
    if defer_load: _read_ahead(entries)
    arch = kv['general.architecture']
    n_heads, n_kv_heads = kv[f'{arch}.attention.head_count'], kv[f'{arch}.attention.head_count_kv']
    assert shard >= 1, f"shard must be at least 1, got {shard}"
    # the next token prediction (MTP) layers: the first is loaded with MTP=1 (model.mtp), the others aren't used
    num_blocks = kv[f'{arch}.block_count'] - kv.get(f'{arch}.nextn_predict_layers', 0)
    mtp = bool(getenv("MTP", int(arch == "glm5-next"))) and kv.get(f"{arch}.nextn_predict_layers", 0) > 0
    entries = {k:v for k,v in entries.items() if not ((m:=re.match(r"blk\.(\d+)\.", k)) and int(m.group(1)) >= num_blocks + mtp)}
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
          # (the output rows: whole tiles of 32 on every device, the tensor cores take a few tokens' logits)
          unit = 32 if k == 'output.weight' else 1 if '_exps' in k else _GGML_QUANT.get(down[2], (1,))[0]
          sizes[name] = round_up(shape[axis], shard*unit)
      pads = {k: sizes[k] for k in sizes if sizes[k] != entries[k][1][shard_map[k]]}
    devices = tuple(Device.canonicalize(f'{Device.DEFAULT}:{i}') for i in range(shard))
    _open_devices(devices)
    # gate and up experts of the same format load as one (fuse_gate_up)
    fuse = {g: u for g in units_map if g.endswith("ffn_gate_exps.weight") and (u:=g.replace("gate", "up")) in units_map and
            entries[g][2] == entries[u][2] and getenv("FUSE_GATE_UP", 1)}
    # a token embedding copied to every device (a vocab that doesn't split) stays in host memory: a call looks up a few of its rows
    host_embd = None
    if shard > 1 and 'token_embd.weight' not in shard_map and 'output.weight' in entries and amd_custom_kernels_supported(devices) and \
       getenv("HOST_EMBEDDING", 1):
      data, shape, typ = entries.pop('token_embd.weight')
      mv, raw = _host_buffer(devices, int(data.nbytes()))
      mv[:] = data.to("CPU").data()
      host_embd = ggml_data_to_tensor(raw, prod(shape), typ).reshape(shape)
    pending:list[Callable[[], None]] = []
    state_dict = gguf_shard(entries, devices, shard_map, units_map, pads if shard > 1 else {}, fuse, pending if defer_load else None)
    if host_embd is not None: state_dict['token_embd.weight'] = host_embd

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
    if mtp: state_dict = {re.sub(rf"^blk\.{num_blocks}\.(nextn\.)?", "mtp.", k):v for k,v in state_dict.items()}

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
        state_dict[name] = permute_rope(state_dict[name], n_heads, rope_dim)
      elif arch == 'llama' and 'attn_k.weight' in name:
        state_dict[name] = permute_rope(state_dict[name], n_kv_heads, state_dict[name].shape[0]//n_kv_heads)
      elif kv_lora_rank and 'attn_kv_a_mqa.weight' in name:
        state_dict[name] = permute_rope(state_dict[name], 1, state_dict[name].shape[0]-kv_lora_rank)
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
      sliding_window_pattern=kv.get(f'{arch}.attention.sliding_window_pattern', 2 if arch == 'gpt-oss' else 0), hc=hc, indexer=indexer, mtp=mtp,
      spec_drafts=getenv("SPEC_DRAFTS", 2) if mtp else 0)
    with Linear.empty_init(): model = Transformer(config)  # (the weights are loaded)
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
    for m in modules([self.blk, self.output] + ([self.mtp] if hasattr(self, "mtp") else []), set()):
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
    # with the weights still to read: the small ones' memory now, their values after
    if self._pending_load: _realize_later(small, self._pending_load)
    Tensor.realize(*tables, *([] if self._pending_load else small))

  def _finish_load(self):
    """the work left of loading (deferred): the weights read, the values computed from them"""
    while self._pending_load: self._pending_load.pop(0)()

  def sequence_parallel(self, rows:int) -> bool:
    """the hyper-connection streams of a prefill chunk of `rows` are split over the devices: tiles of 16 rows on every device"""
    devs = self.token_embd.weight.device
    return isinstance(devs, tuple) and self.config.hc is not None and rows % (16*len(devs)) == 0 and bool(getenv("SEQ_PARALLEL", 1))

  def prefill_chunks(self, chunk_size:int|None=None) -> tuple[int, ...]:
    """the prefill chunk sizes, ascending: the custom kernels compute chunks padded to a static size, a piece of a prompt runs the smallest
    one it fits (its own jit), large chunks are efficient. sequence parallel: whole tiles of rows on every device"""
    fast = amd_custom_kernels_supported(self.token_embd.weight.device)
    # (a context of 64K or less (many sequences batched): chunks of 2048, their memory the slots')
    chunk_size = chunk_size or min(getenv("PREFILL_CHUNK", (4096 if self.max_context > 65536 else 2048) if fast else 32), self.max_context)
    if self.has_recurrent_block and not fast: chunk_size = 1
    sizes = [chunk_size] + [min(chunk_size, n) for n in ((getenv("PREFILL_TINY", 192), getenv("PREFILL_SMALL", 512), getenv("PREFILL_MID", 2048),
                                                          4096) if fast else ())]
    if fast and (devs:=self.token_embd.weight.device) and isinstance(devs, tuple) and self.config.hc is not None and getenv("SEQ_PARALLEL", 1):
      # whole tiles of 16 rows on every device, the kernels of the whole chunk on tiles of 64
      tile = math.lcm(16*len(devs), 64)
      def tiled(n:int) -> int: return min(max(tile, round(n / tile) * tile), self.max_context // tile * tile or tile)
      sizes = [tiled(n) for n in sizes]
    return tuple(sorted(set(sizes)))

  def warmup(self):
    """compile and capture every jit a request uses (they capture on their first call): a prompt of each prefill chunk size, decoding"""
    sizes = self.prefill_chunks()
    if getenv("WARMUP_PARALLEL", 1): self._capture_parallel(sizes)
    self._finish_load()
    # the big chunk first: its scratch memory is the most, the other jits share it. a prompt one past a size runs the next one (1: the smallest)
    for n in sorted({1, *(s + 1 for s in sizes[:-1])}, reverse=True):
      self._cached_tokens, self._ckpt_tokens = [], []
      list(zip(range(4), self.generate([0] * min(n, self.max_context - 8))))
    if self._ckpt_tokens: self._checkpoint(self._ckpt_tokens, save=False)  # and a checkpoint's restore
    # batched decoding: the slots' memory, a copy into a slot, a move between two and a step of every size (their jits linked)
    self._alloc_slots()
    if self.batch_slots(): self.slot_store(0, 1)
    if self.batch_slots() > 1: self.slot_move(1, 0)
    for m in self.batch_sizes():
      if m > self.batch_slots(): continue
      if self.config.spec_drafts: self.batch_spec_step([([0]*(self.config.spec_drafts+1), 1, 0, 1, 0.0)] * m)
      else: self.batch_step([(0, 1, 0.0)] * m)
    if self.batch_prefills(): self.batch_prefill([([0], 0, 0.0)])
    self._cached_tokens, self._ckpt_tokens = [], []

  def _capture_parallel(self, sizes:tuple[int, ...]):
    """the jits of the prefill chunk sizes and of speculative decoding compiled at the same time (in forked processes), called as generate does"""
    if not (devs:=self.token_embd.weight.device) or not amd_custom_kernels_supported(devs): return
    self._init_states()
    from tinygrad.llm.kernels.amd import host_allreduce, p2p_allreduce
    # (the device memory the jits use is made here: not in the processes)
    if isinstance(devs, tuple) and p2p_allreduce(devs) is None: host_allreduce(devs)
    if not hasattr(self, "_token_buf"): self._token_host, self._token_buf = _host_ints(devs, self.max_context+1)
    self._token_prev:list[int]|None = None  # (the captures write the buffer: the next generate writes all of it)
    # the random state of every device (sampling's noise) made here: one made in a capture's process isn't the one the jits run with, its
    # counter wouldn't advance (every step and request the same noise)
    for d in to_tuple(devs):
      key, ctr = Tensor._next_counter(d, 0)
      Tensor.realize(key, ctr, Tensor._device_rng_counters[d])
    temp = Tensor([0.0, 1.0]).to(devs).realize()  # (temperature, top_p)
    v_start_pos, calls = UOp.variable("start_pos", 0, self.max_context-1), cast(list[tuple[Any, Callable[[], Any]]], [])
    for n in sizes:
      if n not in self.jits: self.jits[n] = TinyJit(self._forward_effects, capture_first=True, scratch=self._scratch)
      sp, nt = v_start_pos.bind(0), UOp.variable("toks", 1, n).bind(n)
      calls.append((self.jits[n], functools.partial(self, self._token_buf[:, sp:sp+nt], sp, temp)))
    # the computations left of loading (their jits) too: run after the weights are read
    calls += [(j, j) for j in self._pending_load if isinstance(j, _TinyJit)]
    if self.config.spec_drafts > 0:
      self._spec_init()
      if not hasattr(self, "_spec_jit"): self._spec_jit = TinyJit(self.forward_spec, capture_first=True, scratch=self._scratch)
      calls.append((self._spec_jit, functools.partial(self._spec_call, 0, 0, 1, temp)))
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
      # (and the copies into a slot, back from one and between two)
      self._slot_jits()
      calls += [(self._slot_jit, lambda: self._slot_jit(*self._slot_store_vars(0, 1, 0))),
                (self._slot_load_jit, lambda: self._slot_load_jit(*self._slot_store_vars(0, 1, 0)))]
      if slots > 1: calls += [(self._slot_move_jit, lambda: self._slot_move_jit(*self._slot_move_vars(1, 0)))]
      for j in (self._slot_jit, self._slot_move_jit, self._slot_load_jit): setattr(j, "link_later", True)
      # (and the prefills of several prompts)
      if self.batch_prefills():
        calls.append((bpf:=self._bpf_jit(), lambda: bpf(*self._bpf_vars([1]*self.BPF_G, [0]*self.BPF_G))))
        bpf.link_later = True
    # the weights are read (and what's computed from them) while the jits compile, and a checkpoint is saved and restored (its kernels)
    def meanwhile():
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
    if ckpt and len(ckpt) < len(tokens) and tokens[:len(ckpt)] == ckpt: return len(ckpt), True
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
    if (temp:=self._temps.get((temperature, top_p))) is None:
      temp = self._temps[(temperature, top_p)] = Tensor([temperature, top_p]).to(self.token_embd.weight.device).realize()
    if fast:=amd_custom_kernels_supported(self.token_embd.weight.device):
      # in host memory: every step writes its sampled token after its input tokens (see forward), the host polls it
      if not hasattr(self, "_token_buf"):
        self._token_host, self._token_buf = _host_ints(cast(str|tuple[str, ...], self.token_embd.weight.device), self.max_context+1)
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
      if self.has_recurrent_block and start_pos == prompt_len and getenv("CHECKPOINT", 1): self._checkpoint(tokens, save=True)
      if spec:
        while (tok:=self._token_host[start_pos]) < 0: pass
        tokens.append(tok)
        self._cached_tokens, self._spec_row = tokens[:-1], 0
        yield tok
        yield from self._spec_steps(tokens, temp)
        return
      if fast:
        # run ahead: submit the next step (it reads this token on the device) before waiting for this one. it writes the recurrent states of
        # the next position, the ones before it are kept (see GatedDeltaNetBlock._attention): the caches still hold every token but the last
        if ahead:=len(tokens) + 1 < self.max_context: self(t[:, (sp:=v_start_pos.bind(start_pos)):sp+1], sp, temp)
        while (tok:=self._token_host[start_pos]) < 0: pass
      tokens.append(tok if fast else int(cast(Tensor, out).item()))
      self._cached_tokens = tokens[:-1]
      yield tokens[-1]
