from __future__ import annotations
import functools, itertools, math
from typing import Callable, cast
from tinygrad import Tensor, UOp, nn, Device, Context
from tinygrad.dtype import AddrSpace, DType, dtypes
from tinygrad.helpers import prod, getenv, unwrap, round_up, dedup
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, resolve, smin, sint
from tinygrad.renderer.cstyle import HIPRenderer

BLOCK_M, BLOCK_N, WARP_SIZE = 32, 32, 32
WMMA_M, WMMA_N, WMMA_K = 16, 16, 16
WAVES_M, WAVES_N, LANES_PER_WAVE_M, LANES_PER_WAVE_N = 2, 2, 2, 16
WMMA_ACC, THREADS_PER_BLOCK = WMMA_M // LANES_PER_WAVE_M, WARP_SIZE * WAVES_M * WAVES_N
LDS_PAD, WMMA_ARG, LOG2E = 4, ((WMMA_N, WMMA_M, WMMA_K), 32), math.log2(math.e)

def _unbind(v:int|UOp) -> int|UOp: return v.unbind_all()[0] if isinstance(v, UOp) else v

@functools.cache
def amd_custom_kernels_supported(device:str|tuple[str, ...]|None) -> bool:
  if getenv("DISABLE_AMD_KERNELS"): return False
  # Wave32 and WMMA kernels support RDNA3/4 and CDNA3/4 (CDNA packs the 32-lane groups into wave64s, ds_swizzle stays within 32 lanes).
  if isinstance(device, tuple): device = device[0]
  if device is None or device.split(":")[0] != "AMD": return False
  # @function contexts set ALLOW_DEVICE_USAGE=0 (scheduling must not open devices); the device is always open here
  with Context(ALLOW_DEVICE_USAGE=1):
    return (t:=getattr(Device[device], "target", None)) is not None and t[0] in (11, 12, 9) and isinstance(Device[device].renderer, HIPRenderer)

@functools.cache
def _gfx_version(device:str|tuple[str, ...]|None) -> tuple[int, ...]:
  if isinstance(device, tuple): device = device[0]
  with Context(ALLOW_DEVICE_USAGE=1): return getattr(Device[unwrap(device)], "target")

def _is_rdna4(device:str|tuple[str, ...]|None) -> bool: return _gfx_version(device)[0] == 12
def ordered_expert_sum_supported(device:str|tuple[str, ...]|None) -> bool:
  return amd_custom_kernels_supported(device) and _is_rdna4(device)

_ORDERED_EXPERT_SUM = """
typedef unsigned long size_t;
extern "C" __attribute__((device, const)) size_t __ockl_get_local_id(unsigned int);
extern "C" __attribute__((device, const)) size_t __ockl_get_group_id(unsigned int);
extern "C" __attribute__((device)) void __ockl_barrier(void);

#pragma clang fp contract(off)
extern "C" __attribute__((global)) __attribute__((amdgpu_flat_work_group_size(128,128)))
void sorted_expert_sum(float* out, const float* first, const float* second, const float* w1, const float* w2) {
  const int lane=__ockl_get_local_id(0);
  const int token=__ockl_get_group_id(0);
  const int c=128*__ockl_get_group_id(1)+lane;
  __attribute__((shared)) int order[K1+K2];
  __attribute__((shared)) float scale[K1+K2];
  const int entry=lane;
  if(entry<K1+K2) {
    float weights[K1+K2];
    #pragma unroll
    for(int k=0;k<K1;k++) weights[k]=w1[k*WIDTH+token];
    #pragma unroll
    for(int k=0;k<K2;k++) weights[K1+k]=w2[k*WIDTH+token];
    int rank=0;
    #pragma unroll
    for(int k=0;k<K1+K2;k++) {
      if(k!=entry) rank += weights[k]>weights[entry] || (k<entry && weights[k]==weights[entry]);
    }
    order[rank]=entry;
    scale[rank]=weights[entry];
  }
  __builtin_amdgcn_s_barrier();
  float sum=0.0f;
  #pragma unroll
  for(int k=0;k<K1+K2;k++) {
    const int ent=order[k];
    const float weighted=(ent<K1 ? first[(ent*WIDTH+token)*FEATURES+c] : second[((ent-K1)*WIDTH+token)*FEATURES+c])*scale[k];
    // The stock rank reduction sums the selected product with +0 from
    // the other entries before adding it to the running sum.
    sum += weighted+0.0f;
  }
  out[token*FEATURES+c]=sum;
}
"""

@functools.cache
def _ordered_expert_sum_kernel(out:UOp, first:UOp, second:UOp, w1:UOp, w2:UOp, *, device:str) -> UOp:
  (width, features), k1, k2 = out.shape[-2:], w1.shape[0] // out.shape[-2], w2.shape[0] // out.shape[-2]
  src = f"#define WIDTH {width}\n#define FEATURES {features}\n#define K1 {k1}\n#define K2 {k2}\n" + _ORDERED_EXPERT_SUM
  sink = UOp.sink(out.base, first.base, second.base, w1.base, w2.base, UOp.special(128, "lidx0"),
                  UOp.special(width, "gidx0"), UOp.special(features // 128, "gidx1"), arg=KernelInfo("sorted_expert_sum"))
  with Context(ALLOW_DEVICE_USAGE=1): binary = Device[device].compiler.compile_cached(src)
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=(*sink.src, sink)),
                             UOp(Ops.SOURCE, arg=src), UOp(Ops.BINARY, arg=binary)))

def ordered_expert_sum(first:Tensor, second:Tensor, w1:Tensor, w2:Tensor, T:int) -> Tensor:
  """(gpus, T, features): the T tokens' expert outputs of the two parts first (gpus, k1*T, features) and second (gpus, k2*T, features), entry
  j*T+t token t's, weighted by w1 (k1*T,) and w2 (k2*T,), finite, and summed in descending weight order, ties by part and entry"""
  peers, features = first.shape[0], first.shape[-1]
  assert first.shape[1] == w1.shape[0] and second.shape[1] == w2.shape[0] and second.shape[-1] == features and features % 128 == 0
  assert first.uop.shard_shape[0] == second.uop.shard_shape[0] == 1 and w1.uop.axis is None and w2.uop.axis is None
  assert all(t.dtype == dtypes.float32 for t in (first, second, w1, w2))
  resolved = unwrap(first.device)
  device = resolved if isinstance(resolved, str) else tuple(resolved)
  out = Tensor.empty(peers, T, features, device=device, dtype=dtypes.float32, axis=first.uop.axis)
  return Tensor.custom_kernel(out, first.contiguous(), second.contiguous(), w1.contiguous(), w2.contiguous(),
                              fxn=functools.partial(_ordered_expert_sum_kernel, device=device[0] if isinstance(device, tuple) else device))[0]


def _is_cdna(device:str|tuple[str, ...]|None) -> bool: return _gfx_version(device)[0] == 9

def warp_reduce(val:UOp, maximum:bool=False, full_wave:bool=False) -> UOp:
  for offset in ((16, 8, 4, 2, 1) if full_wave else (8, 4, 2, 1)):
    if val.op is Ops.INDEX and val.addrspace == AddrSpace.REG: val = val.load()
    other = UOp.custom_function(f"__builtin_bit_cast(float, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, {{0}}), {0x1f | offset<<10}))",
                                shape=val.shape, dtype=dtypes.float, fmt=True).call(val)
    val = val.maximum(other) if maximum else val + other
  return val

def _reg(shape:tuple[int, ...], value:float, dep:UOp|None=None) -> UOp:
  ret = UOp.alloc(shape, dtypes.float, addrspace=AddrSpace.REG)
  return ret.after((ret if dep is None else ret.after(dep)).store(ret.const_like(value)))

def _amd_load(ptr:UOp, lanes:int|None=None, stream:bool=False) -> UOp:
  assert ptr.op is Ops.INDEX
  # nontemporal scalar load: streamed weights must not evict the activations/KV cache from L2
  if lanes is None: return ptr.load(arg="nontemporal")
  buf, coords = ptr.src[0], ptr.src[1:]
  idx = sum((coord*math.prod(buf.shape[i+1:]) for i,coord in enumerate(coords)), UOp.const(0))
  return UOp(Ops.SHRINK, src=(buf.flatten(), idx, UOp.const(lanes))).load(arg="nontemporal" if stream else None)

def fp8_matmul(x:Tensor, weight:Tensor, scale:Tensor, block_size:tuple[int, int]) -> Tensor:
  block_n, block_k = block_size
  n, k = weight.shape[-2:]
  x_blocks = x.float().pad((0, -k % block_k)).unflatten(-1, (-1, block_k)).rearrange("... t g k -> ... g t k")
  w_blocks = weight.pad((0, -k % block_k)).unflatten(-1, (-1, block_k)).rearrange("... n g k -> ... g k n")
  limit = 224.0 if weight.dtype == dtypes.fp8e4m3fnuz else 448.0
  x_scale = x_blocks.abs().max(-1, keepdim=True).clamp(min_=1e-10) / limit
  quantized = (x_blocks / x_scale).clamp(-limit, limit).cast(weight.dtype).contiguous()
  partials = (quantized.float() @ w_blocks.float()).contiguous()
  w_scale = scale.transpose(-1, -2).repeat_interleave(block_n, -1)[..., None, :n]
  return (partials * x_scale * w_scale).sum(-3).cast(x.dtype)

class Linear(nn.Linear):
  ggml_type:int|None = None
  use_custom_quant = True
  shard_axis:int|None = None
  weight_scale_inv:Tensor|None = None
  weight_block_size:tuple[int, int] = (128, 128)
  def __init__(self, in_features:int, out_features:int, bias=True):
    super().__init__(in_features, out_features, bias)
    self.in_features, self.out_features = in_features, out_features
  def set_quantized(self, decoded:Tensor):
    from tinygrad.llm.kernels.amd_gguf import find_packed
    if (found:=find_packed(decoded, self.in_features)) is None: return
    self.ggml_type, self.weight = found
    self.shard_axis = decoded.uop.axis
  def __call__(self, x:Tensor) -> Tensor:
    from tinygrad.llm.kernels.amd_gguf import QUANT_SIZES, q8_linear
    if self.weight_scale_inv is not None:
      out = fp8_matmul(x, self.weight, self.weight_scale_inv, self.weight_block_size)
      return out + self.bias if self.bias is not None else out
    supported = self.use_custom_quant and amd_custom_kernels_supported(self.weight.device)
    if self.ggml_type is None and supported:
      self.set_quantized(self.weight)
      if self.ggml_type is None:
        # tiny dense fp16 matmul (e.g. the ssm beta/alpha head rows): single fp16 gemv kernel instead of a
        # generic matmul schedule, and realize the densely packed weight once if it is still a lazy ggml view
        if self.weight.dtype in (dtypes.half, dtypes.float, dtypes.bfloat16) and self.out_features <= 2048 \
          and self.in_features % (WARP_SIZE*4) == 0 and self.weight.uop.axis is None:
          numel, max_shape = x.numel(), x.max_shape
          if isinstance(numel, int) or prod(max_shape) // self.in_features <= 32:
            out = f16_gemv(self, x if isinstance(numel, int) else x.pad_to(max_shape))
            return out if isinstance(numel, int) else out.shrink(tuple((0, s) for s in (*x.shape[:-1], self.out_features)))
        self.use_custom_quant = supported = False  # not a supported quant format
    if self.ggml_type in QUANT_SIZES and supported:
      if isinstance(x.numel(), int): return q8_linear(self, x)
      # symbolic token count: padded to the max chunk size (static shapes), the kernel's rows those there are, the rest sliced off
      out = q8_linear(self, x.pad_to(x.max_shape), rows=cast(UOp, prod(x.shape[:-1])))
      return out.shrink(tuple((0, s) for s in (*out.shape[:out.ndim - x.ndim], *x.shape[:-1], self.out_features)))
    if isinstance(x.numel(), int): return self.dense(x)
    # (a symbolic token count: the matmul on the max rows, static and its own kernel, as the quantized ones)
    return self.dense(x.pad_to(x.max_shape)).contiguous().shrink(tuple((0, s) for s in (*x.shape[:-1], self.out_features)))
  def dense(self, x:Tensor) -> Tensor:
    if x.dtype == dtypes.float and self.weight.dtype == dtypes.half and x.numel() >= 16 * self.in_features and \
       amd_custom_kernels_supported(self.weight.device):
      return (y:=x.half().dot(self.weight.T, dtype=dtypes.float)) if self.bias is None else y + self.bias
    return super().__call__(x)

class ExpertWeights:
  """Like Linear but with num_experts dimension. Weight shape: (num_experts, out_features, in_features)."""
  ggml_type:int|None = None
  use_custom_quant = True
  weight_scale_inv:Tensor|None = None
  weight_block_size:tuple[int, int] = (128, 128)
  def __init__(self, num_experts:int, in_features:int, out_features:int, bias:bool=False):
    self.in_features = in_features
    self.weight = Tensor.zeros(num_experts, out_features, in_features)
    if bias: self.bias = Tensor.zeros(num_experts, out_features)
  def set_quantized(self):
    from tinygrad.llm.kernels.amd_gguf import find_packed
    if (found:=find_packed(self.weight, self.in_features)) is None: return
    self.ggml_type, self.packed = found
  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
    from tinygrad.llm.kernels.amd_gguf import moe_linear
    # sel: (B, T, k), x: (B, T, 1, in) or (B, T, k, in) -> output: (B, T, k, out)
    supported = self.weight_scale_inv is None and self.use_custom_quant and amd_custom_kernels_supported(self.weight.device)
    if self.ggml_type is None and supported:
      self.set_quantized()
      if self.ggml_type is None: self.use_custom_quant = supported = False  # not a supported quant format
    assert self.weight.uop.axis != 0 or (self.ggml_type is not None and supported and not hasattr(self, 'bias')), "split experts need the moe kernel"
    if self.weight_scale_inv is not None:
      ret = fp8_matmul(x.unsqueeze(-2), self.weight[sel], self.weight_scale_inv[sel], self.weight_block_size).squeeze(-2)
    elif self.ggml_type is not None and supported: ret = moe_linear(self, sel, x)
    else: ret = (x.unsqueeze(-2) @ self.weight[sel].transpose(-1, -2)).contiguous().squeeze(-2)
    return ret + self.bias[sel] if hasattr(self, 'bias') else ret

# ******** tiny dense fp16 gemv ********

@functools.cache
def _amd_f16_gemv_kernel(out:UOp, w:UOp, x:UOp, *rest:UOp, in_features:int, out_features:int, tokens:int) -> UOp:
  bias: UOp|None = rest[0] if rest else None
  # one block per (token, output row), 32 lanes accumulate 4-wide chunks of the row
  lanes, val_chunk = WARP_SIZE, 4
  token, out_row = UOp.range(tokens, 0, AxisType.GLOBAL), UOp.range(out_features, 1, AxisType.GLOBAL)
  lane = UOp.range(lanes, 2, axis_type=AxisType.LOCAL)
  per = in_features // (lanes * val_chunk)
  assert per * lanes * val_chunk == in_features
  w = w.reshape((out_features, per, lanes*val_chunk))
  x = x.reshape((tokens, per, lanes*val_chunk))
  acc = UOp.const(0, dtypes.float32)
  for i in range(per):
    for j in range(val_chunk):
      acc = acc + w[out_row, i, lane*val_chunk + j].load().float() * x[token, i, lane*val_chunk + j].load().float()
  total = warp_reduce(acc, full_wave=True)
  if bias is not None: total = total + bias[out_row].load().float()
  return out[token, out_row.valid(lane.eq(0))].store(total).end(token, out_row, lane).sink(arg=KernelInfo(name="linear_f16_gemv", opts_to_apply=()))

def _view_back(t:Tensor) -> Tensor:
  # Widening half to float is exact; preserve casts that round or change the values.
  uop = t.uop
  while uop.op is Ops.CAST and uop.dtype == dtypes.float32 and uop.src[0].dtype in (dtypes.half, dtypes.bfloat16): uop = uop.src[0]
  return Tensor(uop).reshape(t.shape)

def f16_gemv(layer:Linear, x:Tensor) -> Tensor:
  tokens = prod(x.shape[:-1])
  assert isinstance(tokens, int)
  weight = _view_back(layer.weight)
  x = x.contiguous()
  out = Tensor.empty(tokens, layer.out_features, dtype=dtypes.float32, device=x.device)
  fxn = functools.partial(_amd_f16_gemv_kernel, in_features=layer.in_features, out_features=layer.out_features, tokens=tokens)
  srcs = (out, weight.reshape(-1), x.reshape(tokens, layer.in_features)) + (() if layer.bias is None else (_view_back(layer.bias),))
  return Tensor.custom_kernel(*srcs, fxn=fxn)[0].reshape(*x.shape[:-1], layer.out_features)

# ******** flash attention on the KV cache ********

def _vec_load(ptr:UOp, lanes:int) -> tuple[UOp, ...]:
  if lanes == 1: return (ptr.load().float(),)
  vec = _amd_load(ptr, lanes)
  return tuple(vec[i].float() for i in range(lanes))

@functools.cache
def _amd_flash_attention_decode_partial(out, stats, q, cache_kv, valid_kv_len, max_kv_len, block_n, waves=4):
  valid_kv_len = _unbind(valid_kv_len)
  _, B, H_KV, N, D = cast(tuple[int, int, int, int, int], cache_kv.shape)
  _, H, M, _ = cast(tuple[int, int, int, int], q.shape)
  assert M == 1 and H % H_KV == 0 and D % WARP_SIZE == 0 and max_kv_len <= N and max_kv_len % block_n == 0
  G, CHUNK, DPL, WAVES, PARTIALS = H // H_KV, block_n, D // WARP_SIZE, waves, out.shape[2]
  assert CHUNK % WAVES == 0
  SEC = CHUNK // WAVES  # keys each wave scans independently
  total_chunks = (valid_kv_len+CHUNK-1)//CHUNK
  live_chunks = min(total_chunks, PARTIALS) if isinstance(total_chunks, int) else total_chunks.minimum(PARTIALS)
  block_bhkv, block_chunk = UOp.range(B*H_KV, 0, AxisType.GLOBAL), UOp.range(live_chunks, 1, AxisType.GLOBAL)
  lane, wave = UOp.range(WARP_SIZE, -1, axis_type=AxisType.WARP), UOp.range(WAVES, 3, axis_type=AxisType.LOCAL)
  b, kv_head = block_bhkv // H_KV, block_bhkv % H_KV
  # per-lane query fragments for every GQA head, kept packed in registers; unpacked at use
  qf = tuple(_vec_load(q[b, kv_head*G+h, 0, lane*DPL], DPL) for h in range(G))
  zerof = UOp.const(0, dtypes.float)
  # Each block scans every PARTIALS-th chunk, keeping an online softmax across rounds.
  chunk_round = UOp.range((total_chunks-1-block_chunk)//PARTIALS+1, 4, AxisType.LOOP)
  chunk_id = block_chunk + chunk_round*PARTIALS
  valids: list[UOp] = []
  scores: list[list[UOp]] = [[zerof]*G for _ in range(SEC)]
  vfrags: list[tuple[UOp, ...]] = [()]*SEC
  for j in range(SEC):
    key = chunk_id*CHUNK + wave*SEC + j
    valid = key < valid_kv_len
    valids.append(valid)
    kfrag = _vec_load(cache_kv[0, b, kv_head, key, lane*DPL], DPL)
    # V is prefetched in the score pass so both streams are in flight together
    vfrags[j] = tuple(valid.where(v, zerof) for v in _vec_load(cache_kv[1, b, kv_head, key, lane*DPL], DPL))
    for h in range(G):
      s = warp_reduce(sum((qf[h][i]*kfrag[i] for i in range(DPL)), UOp.const(0, dtypes.float)), full_wave=True) * (1/math.sqrt(D))
      scores[j][h] = valid.where(s, UOp.const(-1e30, dtypes.float))
  # A finite initial max keeps fully masked waves from computing exp(-inf - -inf).
  acc_reg, max_reg, sum_reg = _reg((G, DPL), 0), _reg((G,), -1e30), _reg((G,), 0)
  prev_acc, prev_max, prev_sum = acc_reg.after(chunk_round), max_reg.after(chunk_round), sum_reg.after(chunk_round)
  row_max = [functools.reduce(UOp.maximum, (scores[j][h] for j in range(SEC)), prev_max[h].load()) for h in range(G)]
  # Rescale the previous rounds to the new max, then accumulate this round's keys.
  alpha = [((prev_max[h].load()-row_max[h])*LOG2E).exp2() for h in range(G)]
  accs = [[alpha[h]*prev_acc[h, i].load() for i in range(DPL)] for h in range(G)]
  row_sums = [alpha[h]*prev_sum[h].load() for h in range(G)]
  for j in range(SEC):
    for h in range(G):
      beta = valids[j].where(((scores[j][h]-row_max[h])*LOG2E).exp2(), zerof)
      accs[h] = [a + beta*v for a, v in zip(accs[h], vfrags[j])]
      row_sums[h] = row_sums[h] + beta
  update = UOp.group(acc_reg.store(UOp.stack(*(x for acc in accs for x in acc)).reshape(G, DPL)),
                     max_reg.store(UOp.stack(*row_max)), sum_reg.store(UOp.stack(*row_sums))).end(chunk_round)
  acc_reg, max_reg, sum_reg = acc_reg.after(update), max_reg.after(update), sum_reg.after(update)
  # exchange across the block's waves through LDS (fp16 halves LDS so more blocks fit per CU)
  # Matching cache/LDS strides can reuse a loop-local cache index outside the loop. Pad that layout.
  acc_lds = UOp.alloc((WAVES, G, D + (LDS_PAD if G == SEC else 0)), dtypes.half, addrspace=AddrSpace.LOCAL)[:, :, :D]
  ml_lds = UOp.alloc((WAVES, G, 2), dtypes.float, addrspace=AddrSpace.LOCAL)
  lds_acc = acc_lds.reshape(WAVES, G, WARP_SIZE, DPL)
  # Normalize before fp16 to avoid overflow. Nonempty waves have sum >= 1; empty waves keep their zero accumulator.
  stores = [lds_acc[wave, h, lane].store((acc_reg[h].load() / sum_reg[h].load().maximum(1)).cast(dtypes.half)) for h in range(G)]
  # NOTE: duplicate stores of the same value from every lane are harmless here
  stores += [ml_lds[wave, h, i].store(x) for h in range(G) for i, x in enumerate((max_reg[h].load(), sum_reg[h].load()))]
  acc_lds, ml_lds = acc_lds.after(*stores), ml_lds.after(*stores)
  tid = wave*WARP_SIZE + lane
  final_stores:list[UOp] = []
  for i in range(-(-G*D//(WAVES*WARP_SIZE))):
    flat = tid + i*WAVES*WARP_SIZE
    h, d = flat // D, flat % D
    M = functools.reduce(UOp.maximum, (ml_lds[w, h, 0].load() for w in range(WAVES)))
    # LDS holds normalized values; restore each wave's sum before combining.
    val = sum((((ml_lds[w, h, 0].load()-M)*LOG2E).exp2() * ml_lds[w, h, 1].load() * acc_lds[w, h, d].load().float()
               for w in range(WAVES)), zerof)
    oidx = out[b, kv_head*G + h, block_chunk, d]
    if G*D % (WAVES*WARP_SIZE): oidx = out[b, (kv_head*G + h).valid(flat < G*D), block_chunk, d]
    final_stores.append(oidx.store(val))
  hstat = tid
  M = functools.reduce(UOp.maximum, (ml_lds[w, hstat, 0].load() for w in range(WAVES)))
  L = sum((((ml_lds[w, hstat, 0].load()-M)*LOG2E).exp2() * ml_lds[w, hstat, 1].load() for w in range(WAVES)), zerof)
  q_head = (kv_head*G + hstat).valid(hstat < G) if WAVES*WARP_SIZE > G else kv_head*G + hstat
  final_stores += [stats[b, q_head, block_chunk, 0].store(M), stats[b, q_head, block_chunk, 1].store(L)]
  return UOp.group(*final_stores).end(lane, wave, block_chunk, block_bhkv).sink(arg=KernelInfo(name="flash_decode_partial", opts_to_apply=()))

@functools.cache
def _amd_flash_decode_combine(o:UOp, partial:UOp, stats:UOp, live:int|UOp) -> UOp:
  # one wave per (batch, head, 64-dim tile): every lane redundantly weights its chunks; no cross-lane traffic
  live = _unbind(live)
  B, H, C, D = cast(tuple[int, int, int, int], partial.shape)
  DT = 64 if D % 64 == 0 else WARP_SIZE  # dims per block
  assert D % DT == 0
  block_bh, block_dt = UOp.range(B*H, 0, AxisType.GLOBAL), UOp.range(D//DT, 1, AxisType.GLOBAL)
  lane = UOp.range(WARP_SIZE, 2, axis_type=AxisType.LOCAL)
  b, h = block_bh // H, block_bh % H
  NPD = DT // WARP_SIZE  # output dims per lane
  dims = tuple(block_dt*DT + lane*NPD + i for i in range(NPD))
  chunk = UOp.range(live, 100, AxisType.LOOP)
  def iloop(ph, val): return ph.store(ph.const_like(val))
  chunk_max = UOp.alloc((1,), dtypes.float, addrspace=AddrSpace.REG)
  chunk_max_i = chunk_max.after(iloop(chunk_max, -math.inf))
  update0 = chunk_max_i.store(chunk_max_i.after(chunk).maximum(stats[b, h, chunk, 0].load())).end(chunk)
  chunk_max = chunk_max_i.after(update0)
  chunk2 = UOp.range(live, 101, AxisType.LOOP)
  acc = UOp.alloc((NPD,), dtypes.float, addrspace=AddrSpace.REG)
  weight_sum = UOp.alloc((1,), dtypes.float, addrspace=AddrSpace.REG)
  acc_i, weight_sum_i = acc.after(iloop(acc, 0)), weight_sum.after(iloop(weight_sum, 0))
  w = ((stats[b, h, chunk2, 0].load()-chunk_max)*LOG2E).exp2()
  update1 = UOp.group(*[acc_i[i].store(acc_i.after(chunk2)[i].load() + w*partial[b, h, chunk2, d].load()) for i, d in enumerate(dims)],
                      weight_sum_i[0].store(weight_sum_i.after(chunk2)[0].load() + w*stats[b, h, chunk2, 1].load())).end(chunk2)
  acc, weight_sum = acc_i.after(update1), weight_sum_i.after(update1)
  inv = 1 / weight_sum[0].load()
  return UOp.group(*[o[b, h, 0, d].store(acc[i].load() * inv) for i, d in enumerate(dims)]) \
    .end(lane, block_dt, block_bh).sink(arg=KernelInfo(name="flash_decode_combine", opts_to_apply=()))

def amd_flash_attention_decode(q:Tensor, cache_kv:Tensor, valid_kv_len:int|UOp, max_kv_len:int) -> Tensor:
  # Carry length bindings even when the cache was populated independently of this call.
  if isinstance(valid_kv_len, UOp): cache_kv = Tensor(cache_kv.uop.after(valid_kv_len))
  B, H, D = cache_kv.shape[1], q.shape[1], cache_kv.shape[4]
  chunks, axis = min(48, max_kv_len // 64), q.uop.axis
  partial = Tensor.empty(B, H, chunks, D, dtype="float32", device=q.device, axis=axis)
  stats = Tensor.empty(B, H, chunks, 2, dtype="float32", device=q.device, axis=axis)
  waves, group = 16, H // cache_kv.shape[2]
  while waves * group * ((D+LDS_PAD)*2 + 8) > 65536: waves //= 2
  assert waves > 0, "attention head group exceeds shared memory capacity"
  fxn = functools.partial(_amd_flash_attention_decode_partial, valid_kv_len=valid_kv_len, max_kv_len=max_kv_len, block_n=64, waves=waves)
  partial, stats = Tensor.custom_kernel(partial, stats, q, cache_kv, fxn=fxn)[:2]
  live = (valid_kv_len+63)//64
  live = min(live, chunks) if isinstance(live, int) else live.minimum(chunks)
  out = Tensor.empty(B, H, 1, D, dtype="float32", device=q.device, axis=axis)
  fxn = functools.partial(_amd_flash_decode_combine, live=live)
  return Tensor.custom_kernel(out, partial, stats, fxn=fxn)[0]

def _wmma_fragment(fragment:UOp, lane:UOp, rdna4:bool) -> UOp:
  return fragment.reshape(2, 2, 4)[:, lane//16, :].reshape(8) if rdna4 else fragment

@functools.cache
def _amd_flash_attention(o:UOp, q:UOp, cache:UOp, valid_kv_len:int|UOp, q_start:int|UOp|None=None, rdna4:bool=False) -> UOp:
  valid_kv_len, q_start = _unbind(valid_kv_len), _unbind(q_start) if q_start is not None else None
  BH, M, D = q.shape
  _, B, H_KV, physical_n, cache_dim = cache.shape
  k, v = cache[0].reshape(B*H_KV, physical_n, cache_dim), cache[1].reshape(B*H_KV, physical_n, cache_dim)
  assert k.shape == v.shape and BH % k.shape[0] == 0 and k.shape[2] == D
  gqa_group = BH // k.shape[0]
  if isinstance(M, int): assert M % BLOCK_M == 0
  assert isinstance(D, int) and D % WMMA_K == 0 and D % LANES_PER_WAVE_N == 0
  TM, TN, TD, SCALE = BLOCK_M//(WAVES_M*LANES_PER_WAVE_M), BLOCK_N//LANES_PER_WAVE_N, D//(WAVES_N*LANES_PER_WAVE_N), 1/math.sqrt(D)
  # query row 0 sits at sequence position q_base (the queries may be padded beyond valid_kv_len - q_base rows)
  q_base = valid_kv_len - M if q_start is None else q_start
  block_bh, block_m = UOp.range(BH, 0, AxisType.GLOBAL), UOp.range(M // BLOCK_M, 1, AxisType.GLOBAL)
  kv_head = block_bh // gqa_group
  q, o = (x.reshape(BH, M//BLOCK_M, BLOCK_M, D)[block_bh, block_m] for x in (q, o))
  k, v = k[kv_head], v[kv_head]
  wave_m, wave_n, lane = UOp.range(WAVES_M, 2, AxisType.LOCAL), UOp.range(WAVES_N, 3, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  tid, lane_m, lane_n = (wave_m * WAVES_N + wave_n) * WARP_SIZE + lane, lane // LANES_PER_WAVE_N, lane % LANES_PER_WAVE_N
  Q_ELEMS_PER_THREAD, KV_ELEMS_PER_THREAD = BLOCK_M * D // THREADS_PER_BLOCK, BLOCK_N * D // THREADS_PER_BLOCK
  QP_lds = UOp.alloc((BLOCK_M, D + LDS_PAD), dtypes.half, addrspace=AddrSpace.LOCAL)
  KV_lds = UOp.alloc((BLOCK_N, D + LDS_PAD), dtypes.half, addrspace=AddrSpace.LOCAL)[:, :D]
  acc, m_i, l_i = _reg((TM, TD), 0), _reg((TM,), -math.inf), _reg((TM,), 0)
  n_tile = UOp.range(((q_base + (block_m + 1) * BLOCK_M).minimum(valid_kv_len) + BLOCK_N - 1) // BLOCK_N, 100, AxisType.LOOP)
  Q_lds = QP_lds[:, :D]
  Q_store = Q_lds.after(n_tile).reshape(THREADS_PER_BLOCK, Q_ELEMS_PER_THREAD)[tid].store(q.reshape(THREADS_PER_BLOCK, Q_ELEMS_PER_THREAD)[tid])
  load_k = UOp.range(KV_ELEMS_PER_THREAD, 90)
  kval = k.reshape(physical_n*D)[n_tile*BLOCK_N*D + tid*KV_ELEMS_PER_THREAD + load_k]
  K_store = KV_lds.reshape(THREADS_PER_BLOCK, KV_ELEMS_PER_THREAD)[tid, load_k].store(kval).end(load_k)
  Q_lds, KV_lds_k = Q_lds.after(Q_store, K_store), KV_lds.after(Q_store, K_store)
  S_reg = _reg((TM, TN), 0, n_tile)
  k_qk, tm1, tn1 = UOp.range(D//WMMA_K, 101, AxisType.LOOP), UOp.range(TM//WMMA_ACC, 200), UOp.range(TN, 201)
  S_frag = S_reg.reshape(TM // WMMA_ACC, WMMA_ACC, TN).permute(0, 2, 1)[tm1, tn1]
  q_frag = Q_lds.reshape(WAVES_M, TM // WMMA_ACC, WMMA_M, D // WMMA_K, WMMA_K)[wave_m, tm1, lane_n, k_qk]
  k_frag = KV_lds_k.reshape(TN, WMMA_N, D // WMMA_K, WMMA_K)[tn1, lane_n, k_qk]
  # All waves must finish reading Q/K before their shared memory is reused for P/V.
  q_frag, k_frag = (_wmma_fragment(f, lane, rdna4) for f in (q_frag, k_frag))
  qk_done = S_frag.store(UOp.wmma(q_frag, k_frag, S_frag.after(k_qk), *WMMA_ARG)).end(tm1, tn1).end(k_qk).barrier()
  S_reg = S_reg.after(qk_done, S_reg.store(S_reg * SCALE))
  rm, rn = UOp.range(TM, 250), UOp.range(TN, 251)
  q_idx = q_base + block_m * BLOCK_M + wave_m * WMMA_M + (lane_m*TM + rm if rdna4 else rm*LANES_PER_WAVE_M + lane_m)
  k_idx = n_tile * BLOCK_N + rn * LANES_PER_WAVE_N + lane_n
  causal = (k_idx <= q_idx) & (k_idx < valid_kv_len)
  S_reg = S_reg.after(S_reg[rm, rn].store(causal.where(S_reg[rm, rn], S_reg[rm, rn].const_like(-math.inf))).end(rm, rn))
  m_ij, rm2 = _reg((TM,), -math.inf, n_tile), UOp.range(TN, 261, AxisType.LOOP)
  m_ij = m_ij.after(m_ij.store(m_ij.after(rm2).maximum(S_reg[:, rm2])).end(rm2))
  ri_w = UOp.range(TM, 270)
  m_ij = m_ij.after(m_ij[ri_w].store(warp_reduce(m_ij[ri_w], maximum=True)).end(ri_w))
  tile_max = m_ij.reshape(TM, 1).expand(TM, TN).maximum(-1e30)
  S_reg = S_reg.after(S_reg.store(((S_reg - tile_max) * LOG2E).exp2()))
  p_local, ri_ws = _reg((TM,), 0, n_tile), UOp.range(TM, 295)
  p_sum = p_local.after(p_local[ri_ws].store(sum((warp_reduce(S_reg[ri_ws, rn]) for rn in range(TN)), S_reg.const_like(0))).end(ri_ws))
  P_lds = QP_lds.flatten()[:WAVES_N * BLOCK_M * BLOCK_N].reshape(WAVES_N, BLOCK_M, BLOCK_N)
  # gfx11 distributes even/odd rows between half-waves; gfx12 distributes the low/high eight rows.
  row_shape = (LANES_PER_WAVE_M, TM) if rdna4 else (TM, LANES_PER_WAVE_M)
  row_lane, row_elem = (2, 3) if rdna4 else (3, 2)
  P_write = P_lds.reshape(WAVES_N, WAVES_M, *row_shape, TN, LANES_PER_WAVE_N).permute(1, 0, row_lane, 5, row_elem, 4) \
    .reshape(THREADS_PER_BLOCK, TM, TN)
  P_store = P_write[tid].store(S_reg.cast(dtypes.half))
  beta_i, ri4, rj4 = UOp.alloc((TM,), dtypes.float, addrspace=AddrSpace.REG), UOp.range(TM, 330), UOp.range(TD, 331)
  m_new = m_i[ri4].maximum(m_ij[ri4])
  alpha_val, beta_val = ((m_i[ri4] - m_new) * LOG2E).exp2(), ((m_ij[ri4] - m_new) * LOG2E).exp2()
  correction = UOp.group(acc[ri4, rj4].store(alpha_val * acc[ri4, rj4]).end(rj4),
                         l_i[ri4].store(alpha_val * l_i[ri4] + beta_val * p_sum[ri4]),
                         m_i[ri4].store(m_new), beta_i[ri4].store(beta_val)).end(ri4)
  acc, l_i, m_i, beta_i = acc.after(correction), l_i.after(correction), m_i.after(correction), beta_i.after(correction)
  V_lds = UOp.alloc((D, BLOCK_N + LDS_PAD), dtypes.half, addrspace=AddrSpace.LOCAL)[:, :BLOCK_N]
  V_copy, load_v = V_lds.after(qk_done).permute(1, 0), UOp.range(KV_ELEMS_PER_THREAD, 390)
  v_pos = n_tile*BLOCK_N + (tid*KV_ELEMS_PER_THREAD + load_v)//D
  vval = (v_pos < valid_kv_len).where(v.reshape(physical_n*D)[n_tile*BLOCK_N*D + tid*KV_ELEMS_PER_THREAD + load_v], 0)
  V_store = V_copy.reshape(THREADS_PER_BLOCK, KV_ELEMS_PER_THREAD)[tid, load_v].store(vval).end(load_v)
  P_lds, V_lds = P_lds.after(P_store, V_store), V_lds.after(P_store, V_store)
  pv_acc = _reg((TM, TD), 0, n_tile)
  k_pv, tm2, tn2 = UOp.range(BLOCK_N//WMMA_K, 400, AxisType.LOOP), UOp.range(TM//WMMA_ACC, 401), UOp.range(TD, 402)
  pv_frag = pv_acc.reshape(TM // WMMA_ACC, WMMA_ACC, TD).permute(0, 2, 1)[tm2, tn2]
  p_frag = P_lds[wave_n].reshape(WAVES_M, TM // WMMA_ACC, WMMA_M, BLOCK_N // WMMA_K, WMMA_K)[wave_m, tm2, lane_n, k_pv]
  v_frag = V_lds.reshape(WAVES_N, TD, WMMA_N, BLOCK_N // WMMA_K, WMMA_K)[wave_n, tn2, lane_n, k_pv]
  p_frag, v_frag = (_wmma_fragment(f, lane, rdna4) for f in (p_frag, v_frag))
  pv_done = pv_frag.store(UOp.wmma(p_frag, v_frag, pv_frag.after(k_pv), *WMMA_ARG)).end(tm2, tn2).end(k_pv)
  pv_acc = pv_acc.after(pv_done)
  ri5, rj5 = UOp.range(TM, 410), UOp.range(TD, 411)
  n_tile_end = acc[ri5, rj5].store(acc[ri5, rj5] + beta_i[ri5] * pv_acc[ri5, rj5]).end(ri5, rj5).end(n_tile)
  acc, l_i, m_i = acc.after(n_tile_end), l_i.after(n_tile_end), m_i.after(n_tile_end)
  acc = acc.after(acc.store(acc * (1 / l_i).reshape(TM, 1).expand(TM, TD)))
  o = o.reshape(WAVES_M, *row_shape, WAVES_N, TD, LANES_PER_WAVE_N) \
    .permute(0, 3, row_lane-1, 5, row_elem-1, 4).reshape(THREADS_PER_BLOCK, TM, TD)
  return o[tid].store(acc).end(wave_m, wave_n, lane).end(block_m, block_bh).sink(arg=KernelInfo(opts_to_apply=()))

def flash_attention(q:Tensor, assigned_kv:Tensor, valid_end:int|UOp) -> Tensor:
  # cached flash attention on the half KV cache (already written through assigned_kv); valid_end stays bound at the graph level
  T_real, q_start = q.shape[2], None
  D, N, group = q.shape[3], assigned_kv.shape[3], q.shape[1] // assigned_kv.shape[2]
  decode = resolve(T_real == 1, False)
  # Non-power-of-two decode dimensions can lose tail-store masks. Q/P, K, and V use separate LDS allocations.
  supported = D % 32 == 0 and (D & (D-1) == 0 and N % 64 == 0 and group*((D+LDS_PAD)*2+8) <= 65536 if decode else
    D >= 64 and 2*(2*BLOCK_M*(D+LDS_PAD) + D*(BLOCK_N+LDS_PAD)) <= 65536 and N % BLOCK_N == 0 and q.max_shape[2] % BLOCK_M == 0)
  if not decode and _is_cdna(q.device): supported = False  # the prefill kernel is RDNA WMMA only
  if not supported:
    k, v = (assigned_kv[i, :, :, :valid_end].float() for i in range(2))
    mask = None if decode else Tensor.full((T_real, valid_end), -math.inf, dtype=dtypes.float32, device=q.device).triu(valid_end-T_real+1)
    return q.float().scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)
  if decode: return amd_flash_attention_decode(q.half(), assigned_kv, valid_end, cast(int, N))
  if isinstance(T_real, UOp):
    # symbolic chunk: pad the queries to the static tile; garbage rows are sliced off
    T_pad = q.max_shape[2]
    assert T_pad % BLOCK_M == 0, "chunk_size must be a multiple of 32"
    q, q_start = q.pad_to((*q.shape[:2], T_pad, q.shape[3])), valid_end - T_real
  B, H, T, D = q.shape
  out = Tensor.empty(B, H, T, D, dtype="float32", device=q.device, axis=q.uop.axis).reshape(B*H, T, D)
  fxn = functools.partial(_amd_flash_attention, valid_kv_len=valid_end, q_start=q_start, rdna4=_is_rdna4(q.device))
  if isinstance(valid_end, UOp): assigned_kv = Tensor(assigned_kv.uop.after(valid_end))
  out = Tensor.custom_kernel(out, q.half().reshape(B*H, T, D), assigned_kv, fxn=fxn)[0].reshape(B, H, T, D)
  return out if q_start is None else out[:, :, :T_real]

# ******** gated delta net: fused recurrent scan ********

@functools.cache
def _gated_delta_prefill_kernel(core:UOp, q:UOp, k:UOp, v:UOp, beta:UOp, alpha:UOp, state:UOp, kq:UOp, start_pos:UOp|None=None,
                                kept:UOp|None=None, live:UOp|None=None) -> UOp:
  batch, heads, tokens, value_dim, row_tile = *core.shape, 4
  key_dim, alpha_dim = q.shape[-1], alpha.shape[-1] if len(alpha.shape) == 4 else 1
  assert all(isinstance(x, int) for x in (batch, heads, tokens, value_dim, key_dim)) and key_dim % 32 == 0 and value_dim % row_tile == 0
  batch, heads, tokens, value_dim, key_dim = cast(tuple[int, int, int, int, int], (batch, heads, tokens, value_dim, key_dim))
  core, v = (x.reshape(batch*heads, tokens, value_dim) for x in (core, v))
  q, k = (x.reshape(batch*heads, tokens, key_dim) for x in (q, k))
  beta, kq = (x.reshape(batch*heads, tokens) for x in (beta, kq))
  alpha, state = alpha.reshape(batch*heads, tokens, alpha_dim), state.reshape(batch*heads, value_dim, key_dim)
  bh_row, lane = UOp.range(batch*heads*value_dim//row_tile, 0), UOp.range(32, 1, axis_type=AxisType.LOCAL)
  bh, row_base = bh_row // (value_dim//row_tile), (bh_row % (value_dim//row_tile))*row_tile
  rows, cols = tuple(row_base+i for i in range(row_tile)), tuple(lane + i*32 for i in range(key_dim//32))
  current = UOp.alloc((row_tile*key_dim//32,), dtypes.float32, addrspace=AddrSpace.REG)
  initial = None if start_pos is None else start_pos.eq(0)
  current = current.after(current.store(UOp.stack(*(state[bh, row, col].float() if initial is None else
    initial.where(0, state[bh, row, col].float()) for row in rows for col in cols))))
  token = UOp.range(tokens if live is None else live, 2, AxisType.LOOP)  # (live: the tokens there are, the rest padding)
  keys = tuple(k[bh, token, col].load() for col in cols)
  queries = tuple(q[bh, token, col].load() for col in cols)
  updates, stores = [], []
  for row_idx,row in enumerate(rows):
    previous = tuple(current.after(token)[row_idx*key_dim//32+i].load() for i in range(key_dim//32))
    decayed, bv = tuple(x * alpha[bh, token, col if alpha_dim > 1 else 0].load() for x,col in zip(previous, cols)), beta[bh, token].load()
    state_k = warp_reduce(sum((x*y for x,y in zip(decayed, keys)), UOp.const(0, dtypes.float32)), full_wave=True)
    state_q = warp_reduce(sum((x*y for x,y in zip(decayed, queries)), UOp.const(0, dtypes.float32)), full_wave=True)
    delta = (v[bh, token, row].load() - state_k) * bv
    updates += [x + delta*y for x,y in zip(decayed, keys)]
    stores.append(core[bh, token, row.valid(lane.eq(0))].store(state_q + delta*kq[bh, token]))
  if kept is not None:  # the states after each of the first kept tokens (a short step's, for a rollback)
    kept = kept.reshape(nk:=kept.shape[0], batch*heads, value_dim, key_dim)
    stores += [kept[token.valid(token < nk), bh, row, col].store(u.cast(kept.dtype)) for (row, col), u in zip(itertools.product(rows, cols), updates)]
  step = UOp.group(*stores, current.store(UOp.stack(*updates))).end(token)
  state_stores = (state[bh, row, col].store(current.after(step)[row_idx*key_dim//32+i].load().cast(state.dtype))
                  for row_idx,row in enumerate(rows) for i,col in enumerate(cols))
  return UOp.group(*state_stores).end(lane, bh_row).sink(arg=KernelInfo(name="gated_delta_prefill", opts_to_apply=()))

def gated_delta_prefill(q:Tensor, k:Tensor, v:Tensor, beta:Tensor, alpha:Tensor, state:Tensor, start_pos:Tensor|None=None,
                        kept:Tensor|None=None, live:int|UOp|None=None) -> Tensor:
  """the scan's outputs; state updated in place, kept (n, *state.shape) the states after each of the first n tokens. live: the tokens scanned
  (a symbolic count, the rest padding: their outputs garbage)"""
  batch, heads, tokens, key_dim = q.shape
  value_dim = v.shape[-1]
  assert q.shape == k.shape and v.shape[:3] == beta.shape == (batch, heads, tokens) and state.shape == (batch, heads, value_dim, key_dim)
  assert alpha.shape[:3] == (batch, heads, tokens) and (len(alpha.shape) == 3 or alpha.shape[-1] in (1, key_dim))
  assert key_dim % 32 == 0 and value_dim % 4 == 0
  assert q.dtype == k.dtype == dtypes.float32, "recurrent Q/K must be float32"
  assert state.uop.contiguous_view_offset() is not None, "recurrent state must be contiguous"
  if start_pos is not None:
    assert start_pos.uop.is_bound_var
    state = Tensor(state.uop.after(start_pos.uop))
  core, kq = Tensor.empty_like(v), (q*k).sum(-1).contiguous()
  assert kept is None or kept.uop.contiguous_view_offset() is not None, "kept states must be contiguous"
  srcs = (core, _carry(q.contiguous(), *([] if live is None else [live])), k.contiguous(), v.contiguous(), beta.contiguous(), alpha.contiguous(),
          state, kq, *([] if kept is None else [kept]))
  contig = tuple(x.uop if x.uop.op is Ops.AFTER or x is kept else x.uop.contiguous() for x in srcs)
  params = tuple(UOp.placeholder_like(x, slot=i) for i,x in enumerate(contig))
  call = _gated_delta_prefill_kernel(*params[:8], None if start_pos is None else start_pos.uop.unbound(), *params[8:],
                                     **({} if live is None else {"live": _unbind(live)})).call(*contig)
  return Tensor(contig[0].after(call))

KDA_ROWS = 8  # the rows of a head's state a wave updates in kda_decode

@functools.cache
def _kda_decode_kernel(z:UOp, q8:UOp, scale:UOp, xsum:UOp, state:UOp, cq:UOp, ck:UOp, cv:UOp, q:UOp, k:UOp, v:UOp, gate:UOp, beta:UOp,
                       wq:UOp, wk:UOp, wv:UOp, alpha:UOp, dt_bias:UOp, ssm_a:UOp, norm_w:UOp, start_pos:UOp, *src:UOp, lb:float, eps:float,
                       token:int, slot:int=0, src_slot:int=0) -> UOp:
  """a kda token (token's row of the step's inputs and of z), a workgroup a head: a thread a channel's short convs and silu of q, k and v and
  its decay (to LDS, the convs' windows shifted into cq, ck, cv), then a wave KDA_ROWS rows of the state, a lane 32 apart columns of them:
  the l2 norms of q and k, the delta rule on the rows (into state), the output's rms norm and sigmoid gate into z and its q8 (q8_quantize's:
  a wave a group). the states read from src (state, cq, ck, cv) if given, else in place (from k31's engine)"""
  s_in, *c_in = src or (state, cq, ck, cv)
  H, hv, hk = cast(tuple[int, int, int], state.shape[-3:])
  s_in, s_out = s_in.reshape(-1, H, hv, hk), state.reshape(-1, H, hv, hk)  # (a buffer's state slot and src_slot: a kept one's)
  R, W, zero, initial = cast(int, cq.shape[1]), hv // KDA_ROWS, UOp.const(0, dtypes.float), start_pos.eq(0)
  assert hv == hk and hk % 32 == 0 and hv % KDA_ROWS == 0 and W*WARP_SIZE >= hk
  head, wave, lane = UOp.range(H, 0, AxisType.GLOBAL), UOp.range(W, 1, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  t = wave*WARP_SIZE + lane
  tc, live = t.minimum(hk-1), t < hk
  ch = head*hk + tc
  def row(x:UOp, i:UOp) -> UOp: return x.reshape(-1)[token*(int(prod(x.shape))//int(z.shape[1])) + i]  # (token's row of the step's x)
  wins = [[initial.where(0, c[0, i, ch].load().float()) for i in range(R)] + [row(x, ch).load().float()] for c, x in zip(c_in, (q, k, v))]
  qv, kv, vv = (sum((wn[i] * w[ch, i].load().float() for i in range(R + 1)), zero).silu() for wn, w in zip(wins, (wq, wk, wv)))
  al = (((row(alpha, ch).load().float() + dt_bias[ch].load().float()) * ssm_a[head, 0].load().float().abs()).sigmoid() * lb).exp()
  lds = UOp.alloc((4, hk), dtypes.float, addrspace=AddrSpace.LOCAL)
  lds = lds.after(UOp.group(*[lds[j, tc.valid(live)].store(x) for j, x in enumerate((qv, kv, vv, al))]).barrier())
  # the next windows, once the thread read these
  stores = [c.after(UOp.group(*[x for wn in wins for x in wn]))[0, i, ch.valid(live)].store(wn[i+1].cast(c.dtype))
            for c, wn in zip((cq, ck, cv), wins) for i in range(R)]
  cols, rows = [lane + 32*i for i in range(hk//32)], [wave*KDA_ROWS + j for j in range(KDA_ROWS)]
  def l2(x:list[UOp]) -> list[UOp]:  # (x/sqrt(sum(x^2)+1e-6))
    r = (warp_reduce(sum((e*e for e in x), zero), full_wave=True) + 1e-6).rsqrt()
    return [e * r for e in x]
  qn, kn = [e * hk**-0.5 for e in l2([lds[0, c].load() for c in cols])], l2([lds[1, c].load() for c in cols])
  kq = warp_reduce(sum((x*y for x, y in zip(qn, kn)), zero), full_wave=True)
  b, decay = row(beta, head).load().float().sigmoid(), [lds[3, c].load() for c in cols]
  core = []
  for r in rows:
    decayed = [initial.where(0, s_in[src_slot, head, r, c].load().float()) * d for c, d in zip(cols, decay)]
    sk, sq = (warp_reduce(sum((d*x for d, x in zip(decayed, xs)), zero), full_wave=True) for xs in (kn, qn))
    delta = (lds[2, r].load() - sk) * b
    stores += [s_out[slot, head, r, c].store((d + delta*x).cast(state.dtype)) for c, d, x in zip(cols, decayed, kn)]
    core.append(sq + delta*kq)
  # the head's outputs through LDS: their mean square, then a thread an output (every lane's store the same value: no lane gates)
  cl, ss = UOp.alloc((hv,), dtypes.float, addrspace=AddrSpace.LOCAL), UOp.alloc((W,), dtypes.float, addrspace=AddrSpace.LOCAL)
  cl = cl.after(bar:=UOp.group(*[cl[r].store(c) for r, c in zip(rows, core)], ss[wave].store(sum((c*c for c in core), zero))).barrier())
  rs = (sum((ss.after(bar)[w].load() for w in range(1, W)), ss.after(bar)[0].load()) * (1/hv) + eps).rsqrt()
  out = (cl[tc].load() * rs * norm_w[tc].load().float() * row(gate, head*hv + tc).load().float().sigmoid()).cast(z.dtype)
  stores.append(row(z, (head*hv + tc).valid(live)).store(out))
  # (every lane of a quad its word, of a half its sum: the same values stored again, no lane gates)
  value, g = out.float(), (token*H + head)*(hv//32) + wave
  d = UOp.custom_function("({0}/127.0f)", shape=(), dtype=dtypes.float, fmt=True).call(warp_reduce(value.abs(), maximum=True, full_wave=True))
  d = d.maximum(1e-8)
  quant = UOp.custom_function("__builtin_nearbyintf({0}/{1})", shape=(), dtype=dtypes.float, fmt=True).call(value, d).clip(-127, 127)
  word = quant.cast(dtypes.int8).cast(dtypes.uint8).cast(dtypes.uint32) << ((lane%4)*8).cast(dtypes.uint32)
  for offset in (1, 2):
    word |= UOp.custom_function(f"__builtin_amdgcn_ds_swizzle({{0}}, {0x1f|offset<<10})", shape=(), dtype=dtypes.uint32, fmt=True).call(word)
  stores += [q8.reshape(-1)[(g*8 + lane//4).valid(live)].store(word), scale.reshape(-1)[g.valid(live)].store(d),
             xsum.reshape(-1)[(g*2 + lane//16).valid(live)].store(warp_reduce(quant.cast(dtypes.int8).float()))]
  return UOp.group(*stores).end(lane, wave, head).sink(arg=KernelInfo(name="kda_decode", opts_to_apply=()))

def kda_decode(q:Tensor, k:Tensor, v:Tensor, gate:Tensor, beta:Tensor, alpha:Tensor, conv_w:list[Tensor], dt_bias:Tensor, ssm_a:Tensor,
               norm_w:Tensor, start_pos:UOp, lb:float, eps:float, states:list[Tensor], src:list[Tensor]|None=None, token:int=0,
               z:tuple[Tensor, ...]=(), slot:int=0, src_slot:int=0) -> tuple[tuple[Tensor, ...], list[Tensor]]:
  """GatedDeltaNetBlock._attention's normed and gated outputs z (B, T, inner) and their q8 (q8_quantize's) for token of a step's T from the
  outputs of its linears (q, k, v, gate, beta, the decay's alpha), in one kernel: (z, its q8) (the earlier tokens' rows those of z) and the
  states [recurrent, conv q, k, v] it wrote in place (read from src instead if given). the recurrent states whole buffers, slot and src_slot
  their (heads, value_dim, key_dim) states (a kept one's: a slice of a buffer would be a copy, the kernel's writes lost)"""
  if not z:
    T, inner, local = int(q.shape[1]), int(gate.shape[-1]), int(gate.uop.shard_shape[-1])
    z = (Tensor.empty(*q.shape[:2], inner, dtype=dtypes.half, device=q.device, axis=None if gate.uop.axis is None else 2),
         Tensor.empty(T, local//32, 8, dtype=dtypes.uint32, device=q.device), Tensor.empty(T, local//32, dtype=dtypes.float32, device=q.device),
         Tensor.empty(T, local//32, 2, dtype=dtypes.float32, device=q.device))
  assert all(s.uop.has_buffer_identity(after_ok=True) for s in states), "KDA destinations must have buffer identities"
  if src is None: src_slot = slot
  state = states[0]
  if src is not None and state.uop.storage_base is src[0].uop.storage_base: state = src[0]
  srcs = (*z, Tensor(state.uop.after(start_pos)), *states[1:], q, k, v, gate, beta, *conv_w, alpha, dt_bias, ssm_a, norm_w)
  contig = tuple(x.uop if i < 8 or x.uop.has_buffer_identity(after_ok=True) else x.uop.contiguous() for i, x in enumerate(srcs))
  ins = tuple(x.uop for x in src or ())
  params = [UOp.placeholder_like(x, slot=i) for i, x in enumerate(contig + ins)]
  call = _kda_decode_kernel(*params[:20], _unbind(start_pos), *params[20:], lb=lb, eps=eps, token=token,
                            slot=slot, src_slot=src_slot).call(*contig, *ins)
  return tuple(Tensor(x.after(call)) for x in contig[:4]), [Tensor(x.after(call)) for x in contig[4:8]]

# *** top-k: a workgroup a row, a radix select of the k-th key (4 passes of 8 bit LDS histograms), then the picks compacted by ballots

DECODE_THREADS = 1024

def _cfn(inline:bool, src:tuple[UOp, ...], arg:tuple[str, DType]) -> UOp:
  fmt, dt = arg
  return UOp.custom_function(fmt, shape=src[0].shape if src and dt != dtypes.void else (), dtype=dt, fmt=True).call(*src, inline=inline)

def _ballot(pred:UOp) -> UOp: return _cfn(False, src=(pred,), arg=("__builtin_amdgcn_ballot_w32({0})", dtypes.uint32))
def _popcount(x:UOp) -> UOp: return _cfn(True, src=(x,), arg=("__builtin_popcount({})", dtypes.int32))
def _bpermute(x:UOp, src_lane:UOp) -> UOp:
  return _cfn(False, src=(x, src_lane), arg=("__builtin_amdgcn_ds_bpermute((int)({1})*4, (int)({0}))", dtypes.int32))
def _readlane(x:UOp, lane:int) -> UOp: return _cfn(False, src=(x,), arg=(f"__builtin_amdgcn_readlane((int)({{0}}), {lane})", dtypes.int32))
def _wave_scan(x:UOp, lane:UOp) -> UOp:
  for d in (1, 2, 4, 8, 16): x = x + (lane >= d).where(_bpermute(x, (lane - d).maximum(0)), UOp.const(0, dtypes.int32))
  return x

def _topk_key(x:UOp) -> UOp:  # (a float's bits in its order)
  u = x.bitcast(dtypes.uint32)
  return (u >> 31).eq(1).where(u ^ 0xFFFFFFFF, u | 0x80000000)

def _radix_kth(key:Callable[[UOp], UOp], npc:UOp, k:int, pools:UOp, t:UOp, wave:UOp, lane:UOp) -> UOp:
  zero = UOp.const(0, dtypes.int32)
  hist = UOp.alloc((4*256,), dtypes.int32, addrspace=AddrSpace.LOCAL)
  hist = hist.after(UOp.group(*(hist[(t + i*DECODE_THREADS).valid(t + i*DECODE_THREADS < 4*256)].store(zero)
                                for i in range((4*256 + DECODE_THREADS - 1) // DECODE_THREADS))).barrier())
  res = UOp.alloc((4, 3), dtypes.int32, addrspace=AddrSpace.LOCAL)
  prefix, left = UOp.const(0, dtypes.uint32), UOp.const(k, dtypes.int32)
  for d in range(4):
    c = UOp.range(npc, 20 + d, AxisType.LOOP)
    kv = key(c)
    match = (kv >> (32 - 8*d)).eq(prefix) if d else UOp.const(True, dtypes.bool)
    digit = ((kv >> (24 - 8*d)) & 255).cast(dtypes.weakint)
    add = _sys("__hip_atomic_fetch_add({0}, {1}, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);", dtypes.void, hist.index(d*256 + digit),
               match.cast(dtypes.int32))
    hd = hist.after(UOp.group(add).end(c).barrier())
    bins = [hd[d*256 + 255 - lane*8 - j].load() for j in range(8)]
    lane_sum = functools.reduce(lambda a, b: a + b, bins)
    incl = _wave_scan(lane_sum, lane)
    cum, dsel, before = incl - lane_sum, zero, zero
    for j in range(8):
      hit = (cum < left) & (cum + bins[j] >= left)
      dsel, before = hit.where((255 - lane*8 - j).cast(dtypes.int32), dsel), hit.where(cum, before)
      cum = cum + bins[j]
    found = ((incl - lane_sum) < left) & (incl >= left) & wave.eq(0)
    res = res.after(UOp.group(res[d, found.where(0, 2)].store(dsel), res[d, found.where(1, 2)].store(left - before)).barrier())
    prefix, left = (prefix << 8) | res[d, 0].load().cast(dtypes.uint32), res[d, 1].load()
  return (pools >= k).where(prefix, UOp.const(0, dtypes.uint32)).maximum(1)

def _pick_stores(key:Callable[[UOp], UOp], npc:UOp, per:int, thr:UOp, t:UOp, wave:UOp, lane:UOp,
                 write:Callable[[UOp, UOp, UOp], UOp], wc:UOp, rid:int) -> tuple[list[UOp], UOp]:
  W, zero = DECODE_THREADS // WARP_SIZE, UOp.const(0, dtypes.int32)
  c2 = UOp.range(npc, rid, AxisType.LOOP)
  wc = wc.after(wc[c2, wave.valid(lane.eq(0))].store(_popcount(_ballot(key(c2) >= thr))).end(c2).barrier())
  def count(ci:UOp) -> UOp: return (ci < npc).where(wc[ci.minimum(per-1), lane].load(), zero)
  ct = UOp.alloc((per,), dtypes.int32, addrspace=AddrSpace.LOCAL)
  st_a = []
  for r in range((per + W - 1) // W):
    ci = (wave + r * W).minimum(per - 1)
    n = count(wave + r * W)
    incl = _wave_scan(n, lane)
    ok = wave + r * W < per
    st_a += [wc[ci.valid(ok), lane].store(incl - n), ct[ci.valid(ok & lane.eq(W - 1))].store(incl)]
  wpre, ct = (x.after(UOp.group(*st_a).barrier()) for x in (wc, ct))
  cb = UOp.alloc((per + 1,), dtypes.int32, addrspace=AddrSpace.LOCAL)
  carry, st_b = zero, []
  for r in range((per + W - 1) // W):
    n = (lane + r * W < per).where(ct[(lane + r * W).minimum(per - 1)].load(), zero)
    incl = _wave_scan(n, lane)
    st_b.append(cb[(lane + r * W).valid(wave.eq(0) & (lane + r * W < per))].store(carry + incl - n))
    carry = carry + _readlane(incl, W - 1)
  st_b.append(cb[UOp.const(per, dtypes.weakint).valid(wave.eq(0) & lane.eq(0))].store(carry))
  cb = cb.after(UOp.group(*st_b).barrier())
  c3 = UOp.range(npc, rid + 1, AxisType.LOOP)
  pk = key(c3) >= thr
  pos = cb[c3].load() + wpre[c3, wave].load() + _popcount(_ballot(pk) & ((UOp.const(1, dtypes.uint32) << lane.cast(dtypes.uint32)) - 1))
  return [write(c3, pos, pk).end(c3)], cb[per].load()

@functools.cache
def _pool_index_kernel(index:UOp, scores:UOp, valid:UOp, k:int) -> UOp:
  """index (R, K): row r's positions of the keys above the k-th of its first valid scores (by position) then of the ones equal to it (to K),
  the slots past min(K, valid) unwritten. a workgroup a row, thread t holds position c*THREADS + t of every chunk c"""
  rows, P = cast(tuple[int, int], scores.shape)
  THREADS = DECODE_THREADS
  per, W = (P + THREADS - 1) // THREADS, THREADS // WARP_SIZE
  row, wave, lane = UOp.range(rows, 0, AxisType.GLOBAL), UOp.range(W, 1, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  t = wave * WARP_SIZE + lane
  npc = ((valid + THREADS - 1) // THREADS).minimum(per)
  def key(c:UOp) -> UOp:
    p = t + c * THREADS
    ok = p < valid
    return ok.where(_topk_key(scores[row, p.minimum(P-1).valid(ok)].load()), UOp.const(0, dtypes.uint32))
  thr = _radix_kth(key, npc, k, valid, t, wave, lane)
  picks = UOp.alloc((k + 1,), dtypes.int32, addrspace=AddrSpace.LOCAL)
  def write(base:UOp) -> Callable[[UOp, UOp, UOp], UOp]:
    return lambda c3, pos, pk: picks[(pk & (base + pos < k)).where((base + pos).cast(dtypes.weakint), k)].store((t + c3 * THREADS).cast(dtypes.int32))
  wc = UOp.alloc((per, W), dtypes.int32, addrspace=AddrSpace.LOCAL)
  above, ngt = _pick_stores(key, npc, per, thr + 1, t, wave, lane, write(UOp.const(0, dtypes.int32)), wc, 5)
  equal, _ = _pick_stores(lambda c: key(c).eq(thr).cast(dtypes.uint32), npc, per, UOp.const(1, dtypes.uint32), t, wave, lane, write(ngt),
                          wc.after(UOp.group(*above).barrier()), 7)
  picks = picks.after(UOp.group(*above, *equal).barrier())
  return UOp.group(*[index[row, i.valid(i < k)].store(picks[i.minimum(k - 1)].load()) for j in range(-(-k // THREADS)) for i in [t + j * THREADS]]) \
    .end(row, wave, lane).sink(arg=KernelInfo(name="pool_index", opts_to_apply=()))

def _carry(t:Tensor, *vals:int|UOp) -> Tensor:  # (t after the bindings of the variables in vals: the kernels take them unbound)
  return Tensor(t.uop.after(*binds)) if (binds:=dedup(u for v in vals if isinstance(v, UOp) for u in v.toposort() if u.is_bound_var)) else t

def _xor16(x:UOp) -> UOp:  # (x of the lane lane ^ 16, a float)
  return UOp.custom_function(f"__builtin_bit_cast(float, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, {{0}}), {0x1f | 16 << 10}))",
                             shape=x.shape, dtype=dtypes.float, fmt=True).call(x)

INDEX_ROWS = 16  # (a workgroup's rows: their scores of its keys, the keys' fragments read once)

@functools.cache
def _indexer_scores_kernel(out:UOp, q:UOp, keys:UOp, w:UOp, valid:int|UOp, start:int|UOp, T:int, rdna4:bool) -> UOp:
  """out (B*T, N): of row r (batch r // T, its token t = r % T at start + t) every key's score, the heads' sum of w[r, h] relu(q[r, h] . key),
  -inf past the token (and anything past valid). q (B*T, H, D) half, keys (B, N, D) half, w (B*T, H) float. WMMA: a wave 16 keys (their
  fragments read once), a 16 head tile of a row a step: the heads a lane's accumulators, the other half's through a swizzle"""
  R, H, D = cast(tuple[int, int, int], q.shape)
  N = keys.shape[1]
  assert H % 16 == 0 and D % 16 == 0
  WAVES = 4
  rows = min(INDEX_ROWS, R)
  kb, rg = UOp.range((valid + 16*WAVES - 1) // (16*WAVES), 0, AxisType.GLOBAL), UOp.range(-(-R // rows), 1, AxisType.GLOBAL)
  wave, lane = UOp.range(WAVES, 2, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  key = (kb*WAVES + wave)*16 + lane % 16
  rr = UOp.range(rows, 3, AxisType.LOOP)
  r = (rg*rows + rr).minimum(R - 1)
  kf = [_wmma_fragment(keys.reshape(-1, N, D // 16, 16)[r // T, key.minimum(N - 1), ks], lane, rdna4) for ks in range(D // 16)]
  total = UOp.const(0.0, dtypes.float)
  for tile in range(H // 16):
    acc = _reg((8,), 0, rr)
    acc = acc.after(acc.store(functools.reduce(lambda c, ks: UOp.wmma(_wmma_fragment(q.reshape(R, H, D // 16, 16)[r, tile*16 + lane % 16, ks],
                                                                                     lane, rdna4), kf[ks], c, *WMMA_ARG), range(D // 16), acc)))
    heads = [tile*16 + ((lane // 16)*8 + i if rdna4 else 2*i + lane // 16) for i in range(8)]
    total = total + functools.reduce(UOp.__add__, [w[r, h].load() * acc[i].load().maximum(0) for i, h in enumerate(heads)])
  total = total + _xor16(total)
  score = (key > start + r % T).where(-math.inf, total)
  return out[r.valid(rg*rows + rr < R), key.valid((lane < 16) & (key < N))].store(score).end(rr, lane, wave, rg, kb) \
    .sink(arg=KernelInfo(name="indexer_scores", opts_to_apply=()))

def indexer_scores(q:Tensor, keys:Tensor, w:Tensor, start:int|UOp, valid:int|UOp) -> Tensor:
  """(B*T, N) of the keys (B, N, D) cache's first valid ones: a row (token t at start + t) a score of each, the heads' sum of w (B, T, H) of
  relu(q (B, T, H, D) . key), -inf past the token"""
  B, T, H, D = q.shape
  q, w = q.pad((None, None, (0, -H % 16), None)), w.pad((None, None, (0, -H % 16)))  # (a 16 head tile: the padded heads weigh 0)
  H, N = H + -H % 16, round_up(cast(int, keys.shape[1]), DECODE_THREADS)
  out = Tensor.empty(B*T, N, dtype=dtypes.float32, device=q.device)
  keys = _carry(keys, start, valid)
  fxn = functools.partial(_indexer_scores_kernel, valid=_unbind(valid), start=_unbind(start), T=T, rdna4=_is_rdna4(q.device))
  return Tensor.custom_kernel(out, q.half().reshape(B*T, H, D).contiguous(), keys, w.float().reshape(B*T, H).contiguous(), fxn=fxn)[0]

def topk_indices(scores:Tensor, k:int, valid:int|UOp) -> Tensor:
  """(R, k): the positions of the k largest of each row's first valid scores (R, N) (ties: the earlier), the slots past min(k, valid) unset"""
  R, N = scores.shape
  scores, valid = _carry(scores, valid), _unbind(valid)
  return Tensor.custom_kernel(Tensor.empty(R, k, dtype=dtypes.int32, device=scores.device), scores,
                              fxn=functools.partial(_pool_index_kernel, valid=valid, k=k))[0]

# *** mla attention over each query's top-k positions: flash-decode's partials, a block a query row and head, the keys read by position

@functools.cache
def _mla_topk_partial(out:UOp, stats:UOp, q:UOp, cache:UOp, *idx:UOp, start:int|UOp, scale:float, heads:int, waves:int=4, unroll:int=2) -> UOp:
  """out (B*T, H, P, V), stats (B*T, H, P, 2): of query row b*T + t (at position start + t) and each of a group of heads, block p's softmax
  partial over its slots of idx[b, t] (a wave every P*waves-th, a key a step of each head's online softmax: a key read once for the group) of
  the first start + T at or before the query. a key is the cache's row (B, 1, N, D) at a slot's position, its value the row's first V; a lane
  holds dims [lane*V/32, ..) of the values (vector loads) and V + lane + 32*i of the rest"""
  start = _unbind(start)
  B, H, T, D = cast(tuple[int, int, int, int], q.shape)
  N, P, V = cast(int, cache.shape[2]), cast(int, out.shape[2]), cast(int, out.shape[3])
  K = cast(int, idx[0].shape[2]) if idx else N  # (no positions: slot j is position j, every key there is)
  DV, DR, G = V // WARP_SIZE, -(-(D - V) // WARP_SIZE), P * waves
  assert V % WARP_SIZE == 0 and H % heads == 0
  row, hg, part = UOp.range(B*T, 0, AxisType.GLOBAL), UOp.range(H // heads, 1, AxisType.GLOBAL), UOp.range(P, 2, AxisType.GLOBAL)
  lane, wave = UOp.range(WARP_SIZE, -1, axis_type=AxisType.WARP), UOp.range(waves, 3, axis_type=AxisType.LOCAL)
  b, t, zero, hs = row // T, row % T, UOp.const(0, dtypes.float), [hg*heads + i for i in range(heads)]
  rest = [V + lane + WARP_SIZE*i for i in range(DR)]
  def load(x:UOp, d:UOp) -> UOp: return (d < D).where(x[d.minimum(D - 1)].load().float(), zero)  # (the rest's lanes past D: 0)
  def values(x:UOp) -> list[UOp]: return [v for c in range(0, DV, 8) for v in _vec_load(x[lane*DV + c], min(8, DV - c))]
  qf = [[load(q[b, h, t], lane*DV + i) for i in range(DV)] + [load(q[b, h, t], d) for d in rest] for h in hs]
  keys = smin(start + T, K)
  step = UOp.range((keys + G*unroll - 1) // (G*unroll), 4, AxisType.LOOP)
  acc_reg, max_reg, sum_reg = _reg((heads, DV), 0), _reg((heads,), -1e30), _reg((heads,), 0)
  acc = [[acc_reg.after(step)[i, j].load() for j in range(DV)] for i in range(heads)]
  m, l = [max_reg.after(step)[i].load() for i in range(heads)], [sum_reg.after(step)[i].load() for i in range(heads)]
  for u in range(unroll):  # (an online softmax step a key, a finite initial max)
    slot = (step*unroll + u)*G + part*waves + wave
    pos = idx[0][b, t, slot.minimum(K - 1)].load() if idx else slot
    valid = (slot < keys) & (pos <= start + t) & (pos >= 0)
    row_k = cache[b, 0, valid.where(pos, 0).minimum(N - 1)]  # (a slot not a key's: the first key, never an unwritten row)
    kf = values(row_k) + [load(row_k, d) for d in rest]
    for i in range(heads):
      s = valid.where(warp_reduce(sum((x*y for x, y in zip(qf[i], kf)), zero), full_wave=True) * scale, UOp.const(-1e30, dtypes.float))
      mn = m[i].maximum(s)
      alpha, beta = ((m[i] - mn)*LOG2E).exp2(), valid.where(((s - mn)*LOG2E).exp2(), zero)
      acc[i], l[i], m[i] = [a*alpha + beta*v for a, v in zip(acc[i], kf[:DV])], l[i]*alpha + beta, mn
  update = UOp.group(acc_reg.store(UOp.stack(*(a for ai in acc for a in ai)).reshape(heads, DV)), max_reg.store(UOp.stack(*m)),
                     sum_reg.store(UOp.stack(*l))).end(step)
  acc_reg, max_reg, sum_reg = acc_reg.after(update), max_reg.after(update), sum_reg.after(update)
  # the block's waves merged through LDS
  acc_lds = UOp.alloc((waves, heads, V), dtypes.float, addrspace=AddrSpace.LOCAL)
  ml_lds = UOp.alloc((waves, heads, 2), dtypes.float, addrspace=AddrSpace.LOCAL)
  stores = [acc_lds[wave, i, lane*DV + j].store(acc_reg[i, j].load()) for i in range(heads) for j in range(DV)]
  stores += [ml_lds[wave, i, k].store(x) for i in range(heads) for k, x in enumerate((max_reg[i].load(), sum_reg[i].load()))]
  acc_lds, ml_lds = acc_lds.after(*stores), ml_lds.after(*stores)
  tid, final = wave*WARP_SIZE + lane, []
  for i, h in enumerate(hs):
    M = functools.reduce(UOp.maximum, (ml_lds[w, i, 0].load() for w in range(waves)))
    ws = [((ml_lds[w, i, 0].load() - M)*LOG2E).exp2() for w in range(waves)]
    for c in range(-(-V // (waves*WARP_SIZE))):
      d = tid + c*waves*WARP_SIZE
      val = sum((w*acc_lds[x, i, d.minimum(V - 1)].load() for x, w in enumerate(ws)), zero)
      final.append(out[row, h, part, d.valid(d < V)].store(val))
    final += [stats[row, h, part, tid.valid(tid.eq(k))].store(x) for k, x in enumerate((M, sum((w*ml_lds[x, i, 1].load() for x, w in enumerate(ws)),
                                                                                               zero)))]
  return UOp.group(*final).end(lane, wave, part, hg, row).sink(arg=KernelInfo(name="mla_topk_partial", opts_to_apply=()))

def _lane_of(x:UOp, src:UOp) -> UOp:  # (x of lane src, a float)
  return UOp.custom_function("__builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute({1}, __builtin_bit_cast(int, {0})))", shape=x.shape,
                             dtype=dtypes.float, fmt=True).call(x, (src*4).cast(dtypes.int32))

@functools.cache
def _mla_wmma_kernel(out:UOp, *srcs:UOp, start:int|UOp, scale:float, rows:int|UOp, split:int=1) -> UOp:
  """out (B*T, H, V): query row b*T + t (at position start + t) attending to the cache's rows (B, 1, N, D halves) at its first
  start + t + 1 slots of idx[b, t] (or every position) at or before it (its keys first, as topk_indices has them), on the f16 tensor cores
  (RDNA4). a workgroup a row: its heads in tiles of 16 (the WMMA rows), a block of 16 keys at a time gathered into LDS (loaded into registers
  the block before). a tile's 4 waves each the scores of a quarter of D (summed through LDS), then each its online softmax of the block's
  scores and the values of a quarter of V. split: a workgroup a row's every split-th block, out (B*T, H, split, V) and stats (B*T, H, split, 2)
  (srcs: stats first) its flash-decode partials"""
  stats, (q, cache, *idx) = (srcs[0], srcs[1:]) if split > 1 else (None, srcs)
  start = _unbind(start)
  B, H, T, D = cast(tuple[int, int, int, int], q.shape)
  N, DW, V = cast(int, cache.shape[2]), cast(int, cache.shape[3]) // 2, cast(int, out.shape[-1])
  K, RW = cast(int, idx[0].shape[2]) if idx else N, DW + 4  # (a row's words in LDS: the 16 rows of a fragment on different banks)
  tiles = -(-H // 16)
  assert tiles <= 2 and D == 2*DW and D % 64 == 0 and V % 128 == 0 and DW % (8*tiles) == 0
  row, lane, wave = UOp.range(rows, 0, AxisType.GLOBAL), UOp.range(WARP_SIZE, -1, axis_type=AxisType.WARP), UOp.range(4*tiles, 1, AxisType.LOCAL)
  b, t, tid, tile, part, ph, col = row // T, row % T, wave*WARP_SIZE + lane, wave // 4, wave % 4, lane // 16, lane % 16
  ks = [4*ph + i for i in range(4)] + [8 + 4*ph + i for i in range(4)]  # (a lane's 8 of a fragment's 16 inputs)
  def halves(w:UOp) -> list[UOp]: return [(w >> (16*i)).cast(dtypes.uint16).bitcast(dtypes.half) for i in range(2)]
  head = tile*16 + col
  qfrags = [UOp.stack(*((head < H).where(q[b, head.minimum(H - 1), t, part*(D//4) + i*16 + k].load(), 0.0).cast(dtypes.half) for k in ks))
            for i in range(D//64)]
  keys = (t + start + 1).minimum(K)
  blocks = (keys + 15) // 16
  first = UOp.range(split, 3, AxisType.GLOBAL) if split > 1 else UOp.const(0, dtypes.weakint)
  kb = UOp.range((blocks - first + split - 1) // split, 2, AxisType.LOOP)
  # the block's keys into LDS: 8*tiles threads a row, a contiguous run of its words each (vector loads of vec), loaded into registers the block
  # before; a slot not a key's reads row 0
  lds_k = UOp.alloc((16*RW,), dtypes.uint32, addrspace=AddrSpace.LOCAL)
  lds_ok = UOp.alloc((16,), dtypes.int32, addrspace=AddrSpace.LOCAL)
  r, wi, per = tid // (8*tiles), tid % (8*tiles), DW // (8*tiles)
  vec = math.gcd(per, 4)
  def block_words(blk:UOp) -> list[UOp]:
    slot = blk*16 + r
    pos = idx[0][b, t, slot.minimum(K - 1)].load() if idx else slot
    ok = (slot < keys) & (pos <= start + t) & (pos >= 0)
    at = ok.where(pos, 0).minimum(N - 1)
    # Gather the selected cache halves directly, then pack their bits for LDS.
    words = [_amd_load(cache[b, 0, at, 2*(wi*per + i)], 2*vec).bitcast(dtypes.uint32) for i in range(0, per, vec)]
    return [ok.cast(dtypes.uint32)] + [w[j] for w in words for j in range(vec)]
  ahead = UOp.alloc((per + 1,), dtypes.uint32, addrspace=AddrSpace.REG)
  ahead = ahead.after(UOp.group(*(ahead[i].store(w) for i, w in enumerate(block_words(kb.const_like(0)*split + first)))))
  cur = [ahead.after(kb)[i].load() for i in range(per + 1)]
  staged = [lds_k[r*RW + wi*per + i].store(cur[1 + i]) for i in range(per)]
  staged.append(lds_ok[r.valid(wi.eq(0))].store(cur[0].cast(dtypes.int32)))
  barrier = UOp.group(*staged).barrier()
  lds_k, lds_ok = lds_k.after(barrier), lds_ok.after(barrier)
  def kword(key:UOp, w:UOp) -> UOp: return lds_k[key*RW + w].load()
  # the wave's scores of its quarter of D: (its tile's heads, the block's keys), summed over the quarters through LDS
  S = UOp.stack(*[UOp.const(0.0, dtypes.float)]*8)
  for i, qf in enumerate(qfrags):
    w0 = (part*(D//4) + i*16) // 2
    S = UOp.wmma(qf, UOp.stack(*(h for w in (2*ph, 2*ph + 1, 4 + 2*ph, 5 + 2*ph) for h in halves(kword(col, w0 + w)))), S, WMMA_ARG[0], WARP_SIZE)
  lds_s = UOp.alloc((tiles, 4, 16, 16), dtypes.float, addrspace=AddrSpace.LOCAL)
  lds_s = lds_s.after(UOp.group(*(lds_s[tile, part, ph*8 + j, col].store(S[j]) for j in range(8))).barrier())
  # the online softmax of a lane's head row col over its 8 keys and its partner's (lane ^ 16), the values of a quarter of V
  m_reg, l_reg, acc_reg = _reg((1,), -1e30), _reg((1,), 0), _reg((V//64, 8), 0)
  m, l = m_reg.after(kb)[0].load(), l_reg.after(kb)[0].load()
  s = [lds_ok[k].load().ne(0).where(sum((lds_s[tile, p, col, k].load() for p in range(4)), UOp.const(0.0, dtypes.float))*scale, -math.inf)
       for k in ks]
  mn = m.maximum(functools.reduce(UOp.maximum, s).maximum(_xor16(functools.reduce(UOp.maximum, s))))
  alpha, p = ((m - mn)*LOG2E).exp2(), [((x - mn)*LOG2E).exp2() for x in s]
  psum = functools.reduce(UOp.__add__, p)
  ln = l*alpha + psum + _xor16(psum)
  pfrag = UOp.stack(*(x.cast(dtypes.half) for x in p))
  arow = [_lane_of(alpha, ph*8 + j) for j in range(8)]
  def vdim(o:int) -> UOp: return part*(V//4) + (o//2)*32 + 2*col + o % 2
  acc = []
  for o in range(V//64):
    vfrag = UOp.stack(*(halves(kword(k, vdim(o) // 2))[o % 2] for k in ks))
    acc.append(UOp.wmma(pfrag, vfrag, UOp.stack(*(acc_reg.after(kb)[o, j].load()*arow[j] for j in range(8))), WMMA_ARG[0], WARP_SIZE))
  fetch = [ahead.after(*cur)[i].store(w) for i, w in enumerate(block_words(((kb + 1)*split + first).minimum(blocks - 1)))]
  update = UOp.group(m_reg[0].store(mn), l_reg[0].store(ln), *(acc_reg[o, j].store(a[j]) for o, a in enumerate(acc) for j in range(8)),
                     *fetch).end(kb)
  l_end = l_reg.after(update)[0].load()
  if stats is None:
    lrow = [_lane_of(l_end, ph*8 + j) for j in range(8)]
    stores = [out[row, (tile*16 + ph*8 + j).valid(tile*16 + ph*8 + j < H), vdim(o)].store(acc_reg.after(update)[o, j].load() / lrow[j])
              for o in range(V//64) for j in range(8)]
    return UOp.group(*stores).end(lane, wave, row).sink(arg=KernelInfo(name="mla_wmma", opts_to_apply=()))
  stores = [out[row, (tile*16 + ph*8 + j).valid(tile*16 + ph*8 + j < H), first, vdim(o)].store(acc_reg.after(update)[o, j].load())
            for o in range(V//64) for j in range(8)]
  stores += [stats[row, head.valid((head < H) & ph.eq(0) & part.eq(0)), first, k].store(x)
             for k, x in enumerate((m_reg.after(update)[0].load(), l_end))]
  return UOp.group(*stores).end(lane, wave, first, row).sink(arg=KernelInfo(name="mla_wmma_split", opts_to_apply=()))

def mla_wmma_attention(q:Tensor, cache:Tensor, idx:Tensor|None, start:int|UOp, values:int, scale:float) -> Tensor:
  """mla_topk_attention on the f16 tensor cores (RDNA4, up to 32 heads a device): the queries and probabilities in half, sums in float. a symbolic
  T: the buffers its max's, the rows those there are"""
  B, H, T, _ = q.shape
  N = T if isinstance(T, int) else T.vmax
  out = Tensor.empty(B*N, H, values, dtype=dtypes.float32, device=q.device, axis=None if q.uop.axis is None else 1)
  if isinstance(start, UOp): cache = Tensor(cache.uop.after(start))
  if idx is not None and not isinstance(T, int): idx = idx.pad_to(idx.max_shape)
  out = Tensor.custom_kernel(out, _carry(q.pad_to(q.max_shape).contiguous(), T), cache,
                             *([] if idx is None else [idx.cast(dtypes.int32).contiguous()]),
                             fxn=functools.partial(_mla_wmma_kernel, start=start, scale=scale, rows=_unbind(B*T)))[0]
  return out.reshape(B, N, H, values)[:, :T].transpose(1, 2)
def mla_topk_supported(device:str|tuple[str, ...]|None, values:int) -> bool:
  """mla_topk_attention's gpus and shapes: a value a lane's dims of a wave"""
  return amd_custom_kernels_supported(device) and values % WARP_SIZE == 0

def mla_topk_attention(q:Tensor, cache:Tensor, idx:Tensor|None, start:int|UOp, values:int, scale:float) -> Tensor:
  """(B, H, T, values): the queries q (B, H, T, D) at positions start.. attending to the cache's rows (B, 1, N, D) at their indexer's positions
  idx (B, T, K) (of the first start + T slots, at or before the query, those first; None: every position, causal), a row's value its first
  values dims"""
  B, H, T, D = q.shape
  local = q.uop.shard_shape[1] if q.uop.axis == 1 else H
  wmma = _is_rdna4(q.device) and local <= 32 and D % 64 == 0 and values % 128 == 0
  if wmma and q.max_shape[2] >= 16: return mla_wmma_attention(q, cache, idx, start, values, scale)  # (a chunk's rows: the tensor cores)
  parts = 16 if wmma else 4 if B*T > 8 else 32  # (a few rows: more blocks a row)
  partial = Tensor.empty(B*T, H, parts, values, dtype=dtypes.float32, device=q.device, axis=None if q.uop.axis is None else 1)
  stats = Tensor.empty(B*T, H, parts, 2, dtype=dtypes.float32, device=q.device, axis=None if q.uop.axis is None else 1)
  if isinstance(start, UOp): cache = Tensor(cache.uop.after(start))
  ids = [] if idx is None else [idx.cast(dtypes.int32).contiguous()]
  if wmma:
    fxn = functools.partial(_mla_wmma_kernel, start=start, scale=scale, rows=B*T, split=parts)
    partial, stats = Tensor.custom_kernel(partial, stats, q.contiguous(), cache, *ids, fxn=fxn)[:2]
  else:
    # (a key read once for up to 3 heads: their queries and sums in a lane's registers)
    fxn = functools.partial(_mla_topk_partial, start=start, scale=scale, heads=max(n for n in (1, 2, 3) if H % n == 0))
    partial, stats = Tensor.custom_kernel(partial, stats, q.contiguous(), cache, *ids, fxn=fxn)[:2]
  out = Tensor.empty(B*T, H, 1, values, dtype=dtypes.float32, device=q.device, axis=None if q.uop.axis is None else 1)
  out = Tensor.custom_kernel(out, partial, stats, fxn=functools.partial(_amd_flash_decode_combine, live=parts))[0]
  return out.reshape(B, T, H, values).transpose(1, 2)

# ******** small allreduces: the gpus write their partials into each other's memory ********

# every gpu has a mailbox in its memory, mapped into the others: cached, a writer's l2 combines its stores into lines and a system release
# writes them back (rdna4's system scope loads miss the reader's l2, rdna3's don't see the others' stores). a workgroup has a region of each:
# two parities of a slot per gpu and of the sums, AR_WORDS words each, a value and its epoch in a word (no torn reads, no flags). the epochs
# are the workgroups' counters on each gpu, the same on all of them: every gpu runs the same allreduces. no gpu is an epoch ahead of another's
# reads (it waits for their values)
AR_GROUPS, AR_WORDS, AR_TIMEOUT_MS, AR_TWO_SHOT = 64, 1024, 1000, 4096

@functools.cache
def _mailboxes(devs:tuple[str, ...]) -> tuple[Tensor, ...]:
  """each gpu's epoch counters (a workgroup's, then the gpu's index) and mailboxes (k: the one of gpu (d+k)%N), zeroed: no word has an epoch"""
  from tinygrad.device import Buffer, MultiBuffer
  N, nbytes = len(devs), AR_GROUPS * 2 * (len(devs) + 1) * AR_WORDS * 8
  def multi(bufs:list[Buffer], dtype) -> Tensor:
    (mb:=MultiBuffer.__new__(MultiBuffer)).bufs = bufs
    return Tensor(UOp.from_buffer(mb, dtype, devs))
  with Context(ALLOW_DEVICE_USAGE=1):
    own = [Buffer(d, nbytes, initial_value=bytes(nbytes)) for d in devs]
    counters = [Buffer(d, (AR_GROUPS + 1) * 4, initial_value=bytes(AR_GROUPS * 4) + i.to_bytes(4, "little")) for i, d in enumerate(devs)]
  return multi(counters, dtypes.int32), *(multi([own[(d + k) % N] for d in range(N)], dtypes.uint64) for k in range(N))

def _mailbox_shape(x:Tensor, n:sint) -> tuple[int, int]|None:
  """the mailbox kernels' (T, EP) for n floats a gpu: small ones on RDNA4's gpus (system scope atomics over the peers' memory), else None"""
  if not (isinstance(x.device, tuple) and isinstance(n, int) and amd_custom_kernels_supported(x.device) and _is_rdna4(x.device)): return None
  return _ar_shape(n)

def _ar_shape(n:int) -> tuple[int, int]|None:
  """(threads, values a thread) of a workgroup of an allreduce of n: small ones in narrow workgroups, larger in up to 24 of 256 threads"""
  for T, EP in (((128, 1),) if n <= 2048 else ()) + ((256, 1), (256, 2), (256, 4)):
    if n % (T * EP) == 0 and n // (T * EP) <= (AR_GROUPS if n <= 2048 else 24): return T, EP
  return next(((T, EP) for T, EP in ((256, 4), (128, 4), (64, 4), (128, 1)) if n % (T * EP) == 0 and n // (T * EP) <= AR_GROUPS), None)

def _sys(fmt:str, dtype, *src:UOp) -> UOp: return UOp.custom_function(fmt, dtype=dtype, fmt=True).call(*src)

@functools.cache
def _allreduce_kernel(out:UOp, x:UOp, counter:UOp, *mb:UOp, n:int, T:int, EP:int) -> UOp:
  # one shot: a gpu stores its values into its slot of every other gpu's mailbox and polls its own for theirs. two shot: workgroup g's go to
  # gpu g%N (its owner) only, the owner sums them and stores the sums into the others' mailboxes. a sum adds the gpus' values in gpu order on
  # every gpu: the same bits on all. a poll gives up at a deadline: the words it didn't get are NaN (and the sums), not a gpu that never ends
  N, G, two_shot = len(mb), n // (T * EP), n >= AR_TWO_SHOT
  wg, t = UOp.range(G, 0, AxisType.GLOBAL), UOp.range(T, 1, AxisType.LOCAL)
  li = [t * EP + i for i in range(EP)]
  els = [wg * T * EP + l for l in li]
  epoch, me = (counter[wg].load() + 1).cast(dtypes.uint32), counter[AR_GROUPS].load().cast(dtypes.weakint)
  deadline = _sys("__builtin_readsteadycounter()", dtypes.uint64, epoch) + AR_TIMEOUT_MS * 100000  # (100 MHz)
  par = (epoch & 1).cast(dtypes.weakint)
  def reg(s:int|UOp, l:UOp) -> UOp: return ((wg * 2 + par) * (N + 1) + s) * AR_WORDS + l
  vals = [x.reshape(-1)[e].load().float() for e in els]
  def word(v:UOp) -> UOp: return (epoch.cast(dtypes.uint64) << 32) | v.bitcast(dtypes.uint32).cast(dtypes.uint64)
  def value(w:UOp) -> UOp: return (w >> 32).cast(dtypes.uint32).eq(epoch).where(w.cast(dtypes.uint32).bitcast(dtypes.float32), math.nan)
  def push(stores:list[UOp], after:UOp|None=None) -> UOp:
    # the stores in the l2, the workgroup's, then one wave writes the l2 back to the others
    fenced = _sys('__builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");', dtypes.void, UOp.const(0, dtypes.int).after(*stores))
    once = UOp.range((t < 32).where(1, 0).after(UOp.group(fenced, *([] if after is None else [after])).barrier()), next(UOp.unique_num),
                     AxisType.LOOP)
    return _sys('__builtin_amdgcn_fence(__ATOMIC_RELEASE, "");', dtypes.void, once).end(once).barrier()
  def poll(after:UOp, slots:list[UOp], waits:list[UOp]) -> list[UOp]:
    # the slots' words (system scope loads of this gpu's mailbox) until those it waits for have the epoch
    loop = UOp.loop(next(UOp.unique_num)).replace(src=(UOp(Ops.NOOP).after(after),))
    got = [_sys("__hip_atomic_load({0}, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)", dtypes.uint64, mb[0].after(loop).index(s)) for s in slots]
    stale = functools.reduce(lambda a, b: a | b, [w & (g >> 32).cast(dtypes.uint32).ne(epoch) for g, w in zip(got, waits)])
    words = UOp.alloc((len(slots),), dtypes.uint64, addrspace=AddrSpace.REG)
    kept = UOp.group(*[words[i].store(g) for i, g in enumerate(got)])
    words = words.after(kept.backedge(loop, stale & (_sys("__builtin_readsteadycounter()", dtypes.uint64, stale) < deadline)))
    return [words[i].load() for i in range(len(slots))]
  def ordered_sum(parts:list[UOp]) -> UOp: return functools.reduce(lambda a, b: a + b, parts)
  if not two_shot:
    sent = push([mb[k][reg(me, l)].store(word(v)) for k in range(1, N) for l, v in zip(li, vals)])
    got = poll(sent, [reg(s, l) for s in range(N) for l in li], [me.ne(s) for s in range(N) for _ in li])
    stores = [out[e].store(ordered_sum([me.eq(s).where(v, value(got[s * EP + i])) for s in range(N)])) for i, (e, v) in enumerate(zip(els, vals))]
  else:
    owner = (wg % N).cast(dtypes.weakint)
    k_owner, is_owner = (owner - me + N) % N, owner.eq(me)
    sent = push([mb[k][reg(me, l).valid(k_owner.eq(k))].store(word(v)) for k in range(1, N) for l, v in zip(li, vals)])
    got = poll(sent, [reg(s, l) for s in range(N) for l in li], [is_owner & me.ne(s) for s in range(N) for _ in li])
    sums = [ordered_sum([me.eq(s).where(v, value(got[s * EP + i])) for s in range(N)]) for i, v in enumerate(vals)]
    published = push([mb[k][reg(N, l).valid(is_owner)].store(word(sm)) for k in range(1, N) for l, sm in zip(li, sums)], sent)
    got = poll(published, [reg(N, l) for l in li], [is_owner.logical_not() for _ in li])
    stores = [out[e].store(is_owner.where(sm, value(g))) for e, sm, g in zip(els, sums, got)]
  # the next epoch, once every thread of the workgroup read the counter
  done = counter.after(UOp.group(*stores).barrier())[wg.valid(t.eq(0))].store(epoch.cast(dtypes.int32))
  return UOp.group(*stores, done).end(wg, t).sink(arg=KernelInfo(name="allreduce", opts_to_apply=()))

PARTIALS = [False]  # (the row parallel quantized linears' partials (devices, ...) as they are: their caller sums them)

def _per_device(x:Tensor) -> UOp:  # (the buffer of a kernel's output split over the devices, a device's part its mselect, after the kernel)
  u = x.uop
  if u.op is Ops.UNSHARD: return u.src[0]
  assert u.op is Ops.AFTER and u.src[0].op is Ops.UNSHARD, f"not a split kernel output: {u.op}"
  return u.src[0].src[0].after(*u.src[1:])


def scatter(p:Tensor, rows:int|None=None) -> list[Tensor]:
  """the partials p (devices, R, ...) split on axis 0 in steps, each (R, ...) split on its rows over the devices (in half, as partial_sum sums):
  in step k device t has device t+k's part of its rows, copied straight to it (step 0: its own). rows: of each part its first rows"""
  devs, S = cast(tuple[str, ...], p.device), p.shape[1:]
  n, size = len(devs), S[0] // len(devs)
  rest, kept = tuple((0, s) for s in S[1:]), size if rows is None else rows
  p = p.cast(dtypes.half if p.dtype == dtypes.float32 and amd_custom_kernels_supported(p.device) else p.dtype)
  local = _per_device(p if p.uop.op is Ops.UNSHARD else Tensor.empty(*p.shape, dtype=p.dtype, device=devs, axis=0).assign(p))
  def part(j:int, i:int) -> UOp: return local.mselect(j).reshape(S).shrink(((i * size, i * size + kept), *rest))
  own = Tensor(local.reshape(S)._shard(0, UOp.range(n, 0, AxisType.DEVICE)).shrink(((0, kept), *rest)).unshard(0))
  return [own, *(Tensor(UOp.mstack(*[part((t + k) % n, t).copy_to_device(devs[t]) for t in range(n)]).unshard(0)) for k in range(1, n))]

def ring_sum(steps:list[Tensor]) -> Tensor:  # (scatter's steps summed in the ring's order: as they come, the device's own last)
  return functools.reduce(Tensor.add, [*steps[1:], steps[0]])

def reduce_scatter(p:Tensor) -> Tensor:
  """the sum over the devices of their partials p (devices, R, ...) split on axis 0, split on its rows over the devices (R, ...): each device's
  part of the rows of every device copied straight to it, summed in the ring's order (in half, as partial_sum sums)"""
  return ring_sum(scatter(p)).contiguous().cast(p.dtype)

def all_to_all(x:Tensor) -> Tensor:
  """x (R, ..., F) split on its last axis over the devices, split on its rows instead (R, ..., F): each device's rows of every device's
  columns copied to it"""
  devs, S = cast(tuple[str, ...], x.device), x.uop.shard_shape
  n, size, w = len(devs), S[0] // len(devs), S[-1]
  local = _per_device(Tensor.empty(*x.shape, dtype=x.dtype, device=devs, axis=x.ndim - 1).assign(x))
  def part(j:int, i:int) -> UOp:
    p = local.mselect(j).reshape(S).shrink(((i * size, (i + 1) * size), *((0, s) for s in S[1:])))
    return (p if i == j else p.copy_to_device(devs[i])).pad(tuple((0, 0) for _ in S[:-1]) + ((j * w, (n - 1 - j) * w),))
  return Tensor(UOp.mstack(*[functools.reduce(lambda a, b: a + b, [part(j, i) for j in range(n)]).contiguous() for i in range(n)]).unshard(0))

def partial_sum(x:Tensor) -> Tensor:
  """the sum over the devices of their partials x (devices, ...) split on axis 0, on every device: small ones in a kernel of the gpus' mailboxes,
  larger float ones on the gpus in half (their copies half the bytes)"""
  if not all(isinstance(d, int) for d in x.shape): return partial_sum(x.pad_to(x.max_shape)).shrink(tuple((0, d) for d in x.shape[1:]))
  if x.uop.axis != 0 or not (shape:=_mailbox_shape(x, n:=prod(x.shape[1:]))):
    return x.half().sum(0).cast(x.dtype) if x.dtype == dtypes.float32 and amd_custom_kernels_supported(x.device) else x.sum(0)
  fxn = functools.partial(_allreduce_kernel, n=n, T=shape[0], EP=shape[1])
  out = Tensor.empty(n, dtype=dtypes.float32, device=x.device)
  return Tensor.custom_kernel(out, x.float().reshape(x.shape[0], n).contiguous(), *_mailboxes(x.device), fxn=fxn)[0].reshape(x.shape[1:])

def keyed_empty(key:tuple, shape:tuple[sint, ...], dtype, device:str|tuple[str, ...]|None, axis:int|None=None) -> Tensor:
  """an empty tensor (sharded on axis) whose buffer is key's: the kernels that write it from the same inputs are one"""
  local = tuple(s // len(device) if i == axis and isinstance(device, tuple) else s for i, s in enumerate(shape))
  return Tensor(UOp.alloc(local, dtype, hash((key, shape, dtype, device, axis)) % (1 << 62), device=device, bind_on_realize=True,
                          axis=axis if isinstance(device, tuple) else None))

def _gather_offset(el:UOp, peer:int, peers:int, widths:tuple[int, ...], rows:int) -> UOp:
  width = sum(widths)
  row, col = el // width, el % width
  if len(widths) == 1: return (row*peers + peer)*width + col
  offsets = list(itertools.accumulate(widths, initial=0))
  addresses = [rows*peers*off + (row*peers + peer)*w + col-off for w,off in zip(widths, offsets)]
  return functools.reduce(lambda addr,i: (col < offsets[i+1]).where(addresses[i], addr), reversed(range(len(widths)-1)), addresses[-1])

@functools.cache
def _allgather_kernel(out:UOp, counter:UOp, *src:UOp, n:int, widths:tuple[int, ...], T:int, EP:int) -> UOp:
  # Each GPU broadcasts its compact local slices (rows of widths, side by side) once. Mailbox slots/epochs share the allreduce protocol; no padded
  # zero vectors or sums.
  xs, mb, width, offs = src[:len(widths)], src[len(widths):], sum(widths), list(itertools.accumulate(widths, initial=0))
  N, G = len(mb), n // (T * EP)
  wg, t = UOp.range(G, 0, AxisType.GLOBAL), UOp.range(T, 1, AxisType.LOCAL)
  li = [t*EP+i for i in range(EP)]
  els = [wg*T*EP+l for l in li]
  epoch, me = (counter[wg].load()+1).cast(dtypes.uint32), counter[AR_GROUPS].load().cast(dtypes.weakint)
  deadline = _sys("__builtin_readsteadycounter()", dtypes.uint64, epoch) + AR_TIMEOUT_MS*100000
  par = (epoch & 1).cast(dtypes.weakint)
  def reg(s:int|UOp, l:UOp) -> UOp: return ((wg*2+par)*(N+1)+s)*AR_WORDS+l
  def load(e:UOp) -> UOp:
    if len(xs) == 1: return xs[0].reshape(-1)[e].load()
    row, col = e // width, e % width
    vals = [x.reshape(-1)[(row*w + col - o).valid((col >= o) & (col < o + w))].load() for x, w, o in zip(xs, widths, offs)]
    return functools.reduce(lambda acc, i: (col < offs[i + 1]).where(vals[i], acc), reversed(range(len(vals) - 1)), vals[-1])
  vals = [load(e) for e in els]
  words = [(epoch.cast(dtypes.uint64)<<32) | v.bitcast(dtypes.uint32).cast(dtypes.uint64) for v in vals]
  sent = [mb[k][reg(me, l)].store(v) for k in range(1, N) for l, v in zip(li, words)]
  fenced = _sys('__builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");', dtypes.void, UOp.const(0, dtypes.int).after(*sent))
  once = UOp.range((t < 32).where(1, 0).after(UOp.group(fenced).barrier()), next(UOp.unique_num), AxisType.LOOP)
  pushed = _sys('__builtin_amdgcn_fence(__ATOMIC_RELEASE, "");', dtypes.void, once).end(once).barrier()
  loop = UOp.loop(next(UOp.unique_num)).replace(src=(UOp(Ops.NOOP).after(pushed),))
  got = [_sys("__hip_atomic_load({0}, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)", dtypes.uint64,
              mb[0].after(loop).index(reg(s, l))) for s in range(N) for l in li]
  stale = functools.reduce(lambda a,b: a | b, [me.ne(s) & (g>>32).cast(dtypes.uint32).ne(epoch)
                                             for s in range(N) for g in got[s*EP:(s+1)*EP]])
  kept = UOp.alloc((N*EP,), dtypes.uint64, addrspace=AddrSpace.REG)
  stores = UOp.group(*[kept[i].store(g) for i,g in enumerate(got)])
  kept = kept.after(stores.backedge(loop, stale & (_sys("__builtin_readsteadycounter()", dtypes.uint64, stale) < deadline)))
  def value(i:int) -> UOp:
    word = kept[i].load()
    return (word>>32).cast(dtypes.uint32).eq(epoch).where(word.cast(dtypes.uint32).bitcast(dtypes.float32), math.nan)
  outputs = [out.reshape(-1)[_gather_offset(e, s, N, widths, n//width)].store(me.eq(s).where(v, value(s*EP+i)))
             for s in range(N) for i,(e,v) in enumerate(zip(els, vals))]
  done = counter.after(UOp.group(*outputs).barrier())[wg.valid(t.eq(0))].store(epoch.cast(dtypes.int32))
  return UOp.group(*outputs, done).end(wg, t).sink(arg=KernelInfo(name="allgather", opts_to_apply=()))

def allgather_last(x:Tensor, *, exact:bool=False) -> Tensor: return allgather_lasts(x, exact=exact)[0]

def allgather_lasts(*xs:Tensor, exact:bool=False) -> list[Tensor]:
  """xs sharded on their last axes on every device whole, each device's parts side by side: small float ones in a kernel of the gpus'
  mailboxes, else each shard copied to the other devices once (larger float ones on the gpus in half unless exact)"""
  if not xs: return []
  if len(xs) > 1 and not all(x.uop.axis == x.ndim-1 and isinstance(x.device, tuple) and x.device == xs[0].device
                             and x.shape[:-1] == xs[0].shape[:-1] for x in xs):
    return [allgather_last(x, exact=exact) for x in xs]
  if xs[0].uop.axis != xs[0].ndim-1 or not isinstance(xs[0].device, tuple): return list(xs)
  if not all(isinstance(d, int) for x in xs for d in x.shape):
    return [g.shrink(tuple((0, d) for d in x.shape)) for g, x in zip(allgather_lasts(*(x.pad_to(x.max_shape) for x in xs), exact=exact), xs)]
  devs = xs[0].device
  widths = tuple(int(x.shape[-1]) // len(devs) for x in xs)
  n = sum(prod(x.uop.max_shard_shape) for x in xs)
  shape = _mailbox_shape(xs[0], n)
  if shape is None and len(xs) > 1 and amd_custom_kernels_supported(devs) and _is_rdna4(devs):
    shape = next(((n // g, 1) for g in range(1, AR_GROUPS + 1) if n % g == 0 and n // g <= min(256, AR_WORDS)), None)
  if any(x.dtype != dtypes.float32 for x in xs) or shape is None:
    def whole(x:Tensor, w:int) -> Tensor:
      y = x.half() if not exact and x.dtype == dtypes.float32 and amd_custom_kernels_supported(devs) else x
      return Tensor.cat(*(y[..., i*w:(i+1)*w] for i in range(len(devs))), dim=-1).cast(x.dtype)
    return [whole(x, w) for x, w in zip(xs, widths)]
  fxn = functools.partial(_allgather_kernel, n=n, widths=widths, T=shape[0], EP=shape[1])
  counter, *mbs = _mailboxes(devs)
  out = keyed_empty(("allgather", *(x.uop.key for x in xs)), (*xs[0].shape[:-1], len(devs) * sum(widths)), dtypes.float32, devs)
  out = Tensor.custom_kernel(out, counter, *(x.contiguous() for x in xs), *mbs, fxn=fxn)[0]
  if len(xs) == 1: return [out]
  rows = int(prod(out.shape[:-1]))
  return [part.reshape(*out.shape[:-1], w*len(devs)) for part,w in zip(out.flatten().split([rows*len(devs)*w for w in widths]), widths)]

# ******** the residual streams' mix and the sublayer's rmsnorm, a workgroup a token ********

RES_PER = 8  # a thread's neighboring columns
RES_MAX_STREAMS = 14  # (2*streams sums of a token, by the first wave's lanes)

@functools.cache
def _res_mix_kernel(y:UOp, wm:UOp, wn:UOp, xs:UOp, tokens:int|UOp, streams:int, dim:int, eps:float, norm_eps:float, count:UOp|None,
                    start:int|UOp) -> UOp:
  """y = rmsnorm(sum_s p_s xs_s) * wn, p the softmax over the streams of (rmsnorm(xs_s) * wm).sum(): a workgroup a token, every thread its
  RES_PER columns of each stream once: their sums of squares and dots with wm (LDS), then their mix and its sum of squares (LDS).
  count: the streams mixed (the rest zero, left out)"""
  W = dim // (RES_PER * WARP_SIZE)
  token, wave, lane = UOp.range(tokens, 0, AxisType.GLOBAL), UOp.range(W, 1, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  t = wave * WARP_SIZE + lane
  cols, zero = [t * RES_PER + i for i in range(RES_PER)], UOp.const(0, dtypes.float)
  x = [[xs.reshape(-1, streams, dim)[start + token, s, d].load().float() for d in cols] for s in range(streams)]
  w = [wm[d].load().float() for d in cols]
  vals = [sum((v * v for v in x[s]), zero) for s in range(streams)] + [sum((v * u for v, u in zip(x[s], w)), zero) for s in range(streams)]
  red = UOp.alloc((W, 2*streams), dtypes.float, addrspace=AddrSpace.LOCAL)
  red = red.after(UOp.group(*[red[wave, j].store(warp_reduce(v, full_wave=True)) for j, v in enumerate(vals)]).barrier())
  tot = UOp.alloc((2*streams,), dtypes.float, addrspace=AddrSpace.LOCAL)
  col = t.valid(t < 2*streams)
  tot = tot.after(tot[col].store(sum((red[k, col].load() for k in range(1, W)), red[0, col].load())).barrier())
  score = [tot[streams + s].load() * (tot[s].load() * (1/dim) + eps).rsqrt() for s in range(streams)]
  if count is not None: score = [(s < count).where(sc, -math.inf) for s, sc in enumerate(score)]
  m = functools.reduce(UOp.maximum, score)
  e = [((sc - m) * LOG2E).exp2() for sc in score]
  esum = sum(e[1:], e[0])
  h = [sum((e[s] * x[s][i] for s in range(streams)), zero) for i in range(RES_PER)]
  lds_s = UOp.alloc((W,), dtypes.float, addrspace=AddrSpace.LOCAL)
  lds_s = lds_s.after(lds_s[wave.valid(lane.eq(0))].store(warp_reduce(sum((v * v for v in h), zero), full_wave=True)).barrier())
  ss = warp_reduce(lds_s[lane % W].load() * (lane < W).cast(dtypes.float), full_wave=True)
  inv = 1 / esum * (ss / (esum * esum) * (1/dim) + norm_eps).rsqrt()
  return UOp.group(*[y.reshape(-1, dim)[token, d].store(v * inv * wn[d].load().float()) for v, d in zip(h, cols)]) \
    .end(token, wave, lane).sink(arg=KernelInfo(name="res_mix", opts_to_apply=()))

def res_mix_norm(xs:Tensor, wm:Tensor, wn:Tensor, eps:float, norm_eps:float, count:UOp|None=None, rows:tuple[int|UOp, int]|None=None) -> Tensor:
  """rmsnorm(the mix of the residual streams xs (B, T, streams, dim) by wm) * wn: (B, T, dim), of its first count streams (the others zero).
  rows: (start, n) the mix of xs's rows start.. (B 1, read in place; xs split on its rows over the devices: each device's own rows start..)"""
  B, T, streams, dim = cast(tuple[int, int, int, int], xs.shape)
  assert 1 <= streams <= RES_MAX_STREAMS and dim % (RES_PER * WARP_SIZE) == 0 and dim <= RES_PER * WARP_SIZE**2 and (rows is None or B == 1)
  split = len(xs.device) if xs.uop.axis == 1 and isinstance(xs.device, tuple) else 1
  start, n = rows or (0, T // split)
  assert split == 1 or isinstance(n, int)
  N = n if isinstance(n, int) else n.vmax  # (a symbolic n: the buffers its max's, its rows those there are)
  y = Tensor.empty(B, N * split, dim, dtype=dtypes.float32, device=xs.device, axis=1 if split > 1 else None)
  fxn = functools.partial(_res_mix_kernel, tokens=_unbind(B*n), streams=streams, dim=dim, eps=eps, norm_eps=norm_eps,
                          count=None if count is None else _unbind(count), start=_unbind(start))
  y = Tensor.custom_kernel(y, wm, wn, _carry(xs, start, n, *([] if count is None else [count])), fxn=fxn)[0]
  return y if isinstance(n, int) else y[:, :n]
