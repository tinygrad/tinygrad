from __future__ import annotations
import functools, math, weakref
from typing import Callable, cast
from tinygrad import Tensor, UOp, nn, Device, Context
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.dtype import AddrSpace, DType, dtypes
from tinygrad.helpers import prod, getenv, unwrap
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, resolve
from tinygrad.renderer.cstyle import HIPRenderer

BLOCK_M, BLOCK_N, WARP_SIZE = 32, 32, 32
WMMA_M, WMMA_N, WMMA_K = 16, 16, 16
WAVES_M, WAVES_N, LANES_PER_WAVE_M, LANES_PER_WAVE_N = 2, 2, 2, 16
WMMA_ACC, THREADS_PER_BLOCK = WMMA_M // LANES_PER_WAVE_M, WARP_SIZE * WAVES_M * WAVES_N
LDS_PAD, WMMA_ARG, LOG2E = 4, ((WMMA_N, WMMA_M, WMMA_K), 32), math.log2(math.e)
GGML_BLOCK_SIZE, Q8_GROUP_SIZE = 256, 32
Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K = 8, 10, 11, 12, 13, 14
IQ2_XS, IQ3_XXS, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS = 17, 18, 20, 21, 22, 23
QUANT_SIZES = {Q8_0: 272, Q2_K: 84, Q3_K: 110, Q4_K: 144, Q5_K: 176, Q6_K: 210, IQ2_XS: 74,
               IQ3_XXS: 98, IQ4_NL: 144, IQ3_S: 110, IQ2_S: 82, IQ4_XS: 136}  # bytes per 256 weights
HALFWORD_QUANTS = (Q8_0, Q6_K, Q2_K, Q3_K, IQ2_XS, IQ3_XXS, IQ4_NL, IQ3_S, IQ2_S)
SMALL_BLOCK_QUANTS = {Q8_0: 17, IQ4_NL: 9}
QUANT_NAMES = {Q8_0: "q8_0", Q2_K: "q2_k", Q3_K: "q3_k", Q4_K: "q4_k", Q5_K: "q5_k", Q6_K: "q6", IQ4_XS: "iq4_xs",
               IQ2_XS: "iq2_xs", IQ3_XXS: "iq3_xxs", IQ4_NL: "iq4_nl", IQ3_S: "iq3_s", IQ2_S: "iq2_s"}

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
def _is_cdna(device:str|tuple[str, ...]|None) -> bool: return _gfx_version(device)[0] == 9

def warp_reduce(val:UOp, maximum:bool=False, full_wave:bool=False, width:int=0) -> UOp:
  # (width: over each aligned group of width lanes)
  for offset in [o for o in (16, 8, 4, 2, 1) if o < (width or (32 if full_wave else 16))]:
    if val.op is Ops.INDEX and val.addrspace == AddrSpace.REG: val = val.load()
    other = UOp(Ops.CUSTOM, src=(val,), arg=
      (f"__builtin_bit_cast(float, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, {{0}}), {0x1f | offset<<10}))", dtypes.float))
    val = val.maximum(other) if maximum else val + other
  return val

def _reg(shape:tuple[int, ...], value:float, dep:UOp|None=None) -> UOp:
  ret = UOp.alloc(shape, dtypes.float, addrspace=AddrSpace.REG)
  return ret.after((ret if dep is None else ret.after(dep)).store(ret.const_like(value)))

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

# ******** quant linear: q8-activation kernels over packed ggml weights ********

def find_packed(decoded:Tensor, in_features:int) -> tuple[int, Tensor]|None:
  """(ggml type, packed words) if `decoded` is exactly the dequantization of packed ggml weights, rows of in_features"""
  if in_features % Q8_GROUP_SIZE: return None
  packed_sizes = {typ: decoded.numel() * type_size // 256 for typ,type_size in QUANT_SIZES.items()
                  if in_features % GGML_BLOCK_SIZE == 0 or typ in SMALL_BLOCK_QUANTS}
  graph = decoded.uop.toposort()
  raw = next((u for u in graph if u.op in (Ops.SHRINK, Ops.BUFFER, Ops.UNSHARD) and u.dtype == dtypes.uint8 and
              prod(u.shape) in packed_sizes.values()), None)
  if raw is None: return None
  # Only unwrap storage/order-preserving views, then require the exact dequantization expression.
  # This rejects subsequent arithmetic and permutations, including RoPE's concatenated query weights.
  # a permute keeps the order if the axes not of size 1 on a device stay sorted (gguf_shard moves the devices)
  def unwrapped(u:UOp) -> UOp:
    while u.op in (Ops.RESHAPE, Ops.STAGE) or (u.op is Ops.CAST and dtypes.is_float(u.dtype) and dtypes.is_float(u.src[0].dtype)) or \
      (u.op is Ops.PERMUTE and (kept:=[a for a in u.marg if u.src[0].shard_shape[a] != 1]) == sorted(kept)):
      u = u.src[0]
    return u
  # Several formats have the same byte count (Q3_K/IQ3_S and Q4_K/IQ4_NL). Match the expression, not just the size.
  for ggml_type, size in packed_sizes.items():
    if size != prod(raw.shape): continue
    expected = ggml_data_to_tensor(Tensor(raw).flatten(), int(decoded.numel()), ggml_type)
    if unwrapped(decoded.uop).key == unwrapped(expected.uop).key: break
  else: return None
  # Some blocks are only halfword-aligned; keep all formats as zero-copy views of the GGUF storage.
  word_dtype = dtypes.uint16 if ggml_type in HALFWORD_QUANTS else dtypes.uint32
  raw_offset = raw.contiguous_view_offset()
  if raw_offset is None or raw_offset % word_dtype.itemsize or raw.buf_uop.dtype != dtypes.uint8: return None
  return ggml_type, Tensor(raw).flatten().bitcast(word_dtype).contiguous()

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
    if (found:=find_packed(decoded, self.in_features)) is None: return
    self.ggml_type, self.weight = found
    self.shard_axis = decoded.uop.axis
  def __call__(self, x:Tensor) -> Tensor:
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
      # symbolic token count: pad to the max chunk size so the kernels see static shapes, garbage rows are sliced off
      out = q8_linear(self, x.pad_to(x.max_shape))
      return out.shrink(tuple((0, s) for s in (*x.shape[:-1], self.out_features)))
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
    if self.weight.uop.axis == 0 or (found:=find_packed(self.weight, self.in_features)) is None: return
    self.ggml_type, self.packed = found
  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
    # sel: (B, T, k), x: (B, T, 1, in) or (B, T, k, in) -> output: (B, T, k, out)
    supported = self.weight_scale_inv is None and self.use_custom_quant and amd_custom_kernels_supported(self.weight.device)
    if self.ggml_type is None and supported:
      self.set_quantized()
      if self.ggml_type is None: self.use_custom_quant = supported = False  # not a supported quant format
    if self.weight_scale_inv is not None:
      ret = fp8_matmul(x.unsqueeze(-2), self.weight[sel], self.weight_scale_inv[sel], self.weight_block_size).squeeze(-2)
    elif self.ggml_type is not None and supported and isinstance(x.shape[1], int) and x.shape[1] == 1: ret = moe_linear(self, sel, x)
    else: ret = (x.unsqueeze(-2) @ self.weight[sel].transpose(-1, -2)).contiguous().squeeze(-2)
    return ret + self.bias[sel] if hasattr(self, 'bias') else ret

def _amd_dp4a(a:UOp, b:UOp, c:UOp, cdna:bool) -> UOp:
  if cdna: return UOp(Ops.CUSTOMI, src=(a, b, c), arg=("__builtin_amdgcn_sdot4({}, {}, {}, false)", dtypes.int32))  # CDNA has no sudot4 (dot8-insts)
  return UOp(Ops.CUSTOMI, src=(a, b, c), arg=("__builtin_amdgcn_sudot4(true, {}, true, {}, {}, false)", dtypes.int32))

def _amd_byte_perm(a:UOp, b:UOp, selectors:UOp) -> UOp:
  return UOp(Ops.CUSTOMI, src=tuple(x.cast(dtypes.uint32) for x in (a, b, selectors)), arg=("__builtin_amdgcn_perm({}, {}, {})", dtypes.uint32))

def _amd_load(ptr:UOp, lanes:int|None=None, stream:bool=False) -> UOp:
  assert ptr.op is Ops.INDEX
  # nontemporal scalar load: streamed weights must not evict the activations/KV cache from L2
  if lanes is None: return ptr.load(arg="nontemporal")
  buf, coords = ptr.src[0], ptr.src[1:]
  idx = sum((coord*math.prod(buf.shape[i+1:]) for i,coord in enumerate(coords)), UOp.const(0))
  return UOp(Ops.SHRINK, src=(buf.flatten(), idx, UOp.const(lanes))).load(arg="nontemporal" if stream else None)

def _load_byte(raw:UOp, base:UOp, offset:int|UOp) -> UOp:
  size = raw.dtype.itemsize
  return (raw[base + offset//size].cast(dtypes.uint32) >> ((offset%size)*8)) & 255
def _load_u32(raw:UOp, base:UOp, offset:int|UOp, stream:bool=False) -> UOp:
  # Halfword-aligned formats cannot load u32 directly.
  lo, hi = (raw[base+offset//2+i] for i in range(2))
  if stream: lo, hi = _amd_load(lo), _amd_load(hi)
  return lo.cast(dtypes.uint32) | (hi.cast(dtypes.uint32) << 16)

def _half(value:UOp) -> UOp: return value.cast(dtypes.uint16).bitcast(dtypes.float16).float()

def _iq4_bytes(packed:UOp, shift:int|UOp) -> UOp:
  # the non-linear iq4nl table as a byte lookup: 3 byte_perms beat any arithmetic/select-tree form (~60% decode)
  selectors = (packed >> shift) & 0x0f0f0f0f
  low = _amd_byte_perm(UOp.const(0xf6eaddcf, dtypes.uint32), UOp.const(0xbfad9881, dtypes.uint32), selectors)
  high = _amd_byte_perm(UOp.const(0x71594535, dtypes.uint32), UOp.const(0x26190d01, dtypes.uint32), selectors & 0x07070707)
  return _amd_byte_perm(high, low, 0x03020100 | ((selectors & 0x08080808) >> 1))

def _q5_scales(raw:UOp, base:UOp, subgroup:UOp) -> tuple[UOp, UOp, UOp, UOp]:
  # scales/mins (6-bit each) live in block bytes 4-15: three words total, same for the whole super-block's lanes
  w1, w2, w3 = _amd_load(raw[base+1]), _amd_load(raw[base+2]), _amd_load(raw[base+3])
  sb = (subgroup & 3) * 8  # byte within word
  byte1, byte2, byte3 = (w1 >> sb) & 255, (w2 >> sb) & 255, (w3 >> sb) & 255
  scale = (subgroup < 4).where(byte1 & 63, (byte3 & 15) | ((byte1 >> 6) << 4))
  minimum = (subgroup < 4).where(byte2 & 63, (byte3 >> 4) | ((byte2 >> 6) << 4))
  d, dmin = (raw[base] & 0xffff).cast(dtypes.uint16), (raw[base] >> 16).cast(dtypes.uint16)
  return _half(d), _half(dmin), scale.float(), minimum.float()

def _iq4_scale(raw:UOp, base:UOp, subgroup:UOp) -> UOp:
  header, low = raw[base].load(), raw[base+1].load()
  scale = ((low >> (4*subgroup)) & 15) | (((header >> (16+2*subgroup)) & 3) << 4)
  return _half(header) * (scale.cast(dtypes.int32)-32).float()

def iq4_half_lut(device:str|tuple[str, ...]|None) -> Tensor:
  from tinygrad.runtime.autogen.ggml_common import kvalues_iq4nl
  return Tensor.const(tuple(x for j in range(16) for i in range(16) for x in (kvalues_iq4nl[i], kvalues_iq4nl[j])),
                      dtypes.float16).to(device, force=True).bitcast(dtypes.uint32)

def _q8_group(q:UOp, scale:UOp, xsum:UOp, token:UOp, group:UOp, lane:UOp, value:UOp) -> list[UOp]:
  """the stores of the q8 of a wave's group (lane l its element l)"""
  # Quantize each input once, then pack four neighboring lanes into one word.
  # Keep divisions intact: reciprocal multiplication can move half-precision inputs across rounding ties.
  d = UOp(Ops.CUSTOM, src=(warp_reduce(value.abs(), maximum=True, full_wave=True),), arg=("({0}/127.0f)", dtypes.float)).maximum(1e-8)
  rounded = UOp(Ops.CUSTOM, src=(value, d), arg=("__builtin_nearbyintf({0}/{1})", dtypes.float))
  quant = rounded.clip(-127, 127).cast(dtypes.int8)
  word = quant.cast(dtypes.uint8).cast(dtypes.uint32) << ((lane%4)*8).cast(dtypes.uint32)
  for offset in (1, 2):
    word |= UOp(Ops.CUSTOM, src=(word,), arg=(f"__builtin_amdgcn_ds_swizzle({{0}}, {0x1f | offset<<10})", dtypes.uint32))
  return [q[token, group, (lane//4).valid((lane%4).eq(0))].store(word), scale[token, group.valid(lane.eq(0))].store(d),
          xsum[token, group, (lane//16).valid((lane%16).eq(0))].store(warp_reduce(quant.float()))]

@functools.cache
def _q8_quantize_kernel(q:UOp, scale:UOp, xsum:UOp, *xs:UOp, tokens:int, in_features:int, act:Callable|None=None) -> UOp:
  # act: elementwise, of the parts of the xs' rows (each of in_features)
  groups = in_features//Q8_GROUP_SIZE
  token_group, lane = UOp.range(tokens*groups, 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP)
  token, group = token_group//groups, token_group%groups
  parts = [int(prod(x.shape))//(tokens*in_features) for x in xs]
  vals = [x.reshape(tokens, p, groups, 32)[token, h, group, lane].load().float() for x, p in zip(xs, parts) for h in range(p)]
  value = vals[0] if act is None else act(*vals)
  return UOp.group(*_q8_group(q, scale, xsum, token, group, lane, value)).end(token_group, lane) \
    .sink(arg=KernelInfo(name="q8_quantize", opts_to_apply=()))

# linears of the same input share its quantization: one kernel, not one per linear. an entry is dropped once its outputs are realized
_q8_cache:weakref.WeakKeyDictionary[UOp, tuple[Tensor, Tensor, Tensor]] = weakref.WeakKeyDictionary()
def q8_quantize(x:Tensor, tokens:int, in_features:int) -> tuple[Tensor, Tensor, Tensor]:
  # (casts, reshapes and a device's select of a cached input take its q8, that device's: the q8 is coarser than a half)
  u, dev = x.uop, None
  while u not in _q8_cache and (u.op in (Ops.CAST, Ops.RESHAPE) or u.op is Ops.MSELECT and dev is None):
    if u.op is Ops.MSELECT: dev = u.arg
    u = u.src[0]
  if (hit:=_q8_cache.get(u)) is not None and not hit[0].uop.is_realized and hit[0].shape[:2] == (tokens, in_features//Q8_GROUP_SIZE):
    return hit if dev is None else cast(tuple[Tensor, Tensor, Tensor], tuple(Tensor(t.uop.mselect(dev)) for t in hit))
  _q8_cache[x.uop] = ret = _q8_quantize(x, tokens=tokens, in_features=in_features)
  return ret

def _q8_empty(x:Tensor, tokens:int, in_features:int) -> tuple[Tensor, Tensor, Tensor]:
  groups = in_features//Q8_GROUP_SIZE
  return (Tensor.empty(tokens, groups, 8, dtype=dtypes.uint32, device=x.device), Tensor.empty(tokens, groups, dtype=dtypes.float32, device=x.device),
          Tensor.empty(tokens, groups, 2, dtype=dtypes.float32, device=x.device))

def _q8_quantize(*xs:Tensor, tokens:int, in_features:int, act:Callable|None=None) -> tuple[Tensor, Tensor, Tensor]:
  fxn = functools.partial(_q8_quantize_kernel, tokens=tokens, in_features=in_features, act=act)
  q, scale, xsum = Tensor.custom_kernel(*_q8_empty(xs[0], tokens, in_features), *xs, fxn=fxn)[:3]
  return q, scale, xsum

def q8_act(act:Callable, *xs:Tensor, parts:int=1) -> Tensor:
  """act (elementwise, of tensors and uops) of the parts of the xs' rows (e.g. a gate and up gemv's halves), its q8 made from the xs in one
  kernel: the linears that take it read the q8 (q8 cache), it isn't computed unless something else reads it"""
  xs = tuple(x.reshape(-1, parts, x.shape[-1] // parts) for x in xs)
  ret = act(*(x[:, h] for x in xs for h in range(parts)))
  _q8_cache[ret.uop] = _q8_quantize(*xs, tokens=int(xs[0].shape[0]), in_features=int(xs[0].uop.shard_shape[-1]), act=act)
  return ret

@functools.cache
def _rms_norm_q8_kernel(q:UOp, scale:UOp, xsum:UOp, x:UOp, w:UOp, tokens:int, dim:int, eps:float) -> UOp:
  """the q8 of rms_norm(x) * w, a wave a group: a workgroup's waves (of its token's groups) sum all of the token's squares, in LDS"""
  waves = math.gcd(groups:=dim//Q8_GROUP_SIZE, 4)
  tw, wave, lane = UOp.range(tokens*groups//waves, 0, AxisType.GLOBAL), UOp.range(waves, 1, AxisType.LOCAL), UOp.range(32, -1, AxisType.WARP)
  t, g, tid = tw//(groups//waves), tw%(groups//waves)*waves + wave, wave*32 + lane
  sq = [x.reshape(tokens, dim)[t, tid + k*waves*32].load().float() for k in range(dim//(waves*32))]
  ss = UOp.alloc((waves,), dtypes.float, addrspace=AddrSpace.LOCAL)
  ss = ss.after(ss[wave.valid(lane.eq(0))].store(warp_reduce(sum((v*v for v in sq[1:]), sq[0]*sq[0]), full_wave=True)).barrier())
  rn = 1 / (sum((ss[k].load() for k in range(1, waves)), ss[0].load()) * (1/dim) + eps).sqrt()
  value = x.reshape(tokens, dim)[t, g*Q8_GROUP_SIZE + lane].load().float() * rn * w[g*Q8_GROUP_SIZE + lane].load().float()
  return UOp.group(*_q8_group(q, scale, xsum, t, g, lane, value)).end(tw, wave, lane).sink(arg=KernelInfo(name="rms_norm_q8", opts_to_apply=()))

def q8_rms_norm(norm:nn.RMSNorm, x:Tensor) -> Tensor:
  """norm(x), its q8 made from x in one kernel (as q8_act's)"""
  assert norm.weight is not None
  ret, dim = norm(x), int(x.shape[-1])
  fxn = functools.partial(_rms_norm_q8_kernel, tokens=(T:=int(x.numel())//dim), dim=dim, eps=norm.eps)
  q, scale, xsum = Tensor.custom_kernel(*_q8_empty(x, T, dim), x, norm.weight, fxn=fxn)[:3]
  _q8_cache[ret.uop] = (q, scale, xsum)
  return ret

# ******** residual stream mix + rmsnorm + q8, one token ********

RES_WAVES, RES_PER = 4, 2
RES_GROUPS = RES_WAVES * RES_PER  # the q8 groups of a workgroup: RES_PER a wave, lane l element l of each (the q8 is wave local)
def _res_slice(tokens:int, dim:int) -> tuple[int, UOp, UOp, UOp, UOp, list[UOp], list[UOp]]:
  """the workgroups of a token's slices: G a token, (token, workgroup, lane, wave, its q8 groups, its elements)"""
  G = dim // (RES_GROUPS * Q8_GROUP_SIZE)
  assert G * RES_GROUPS * Q8_GROUP_SIZE == dim
  tw, lane, wave = UOp.range(tokens*G, 0, AxisType.GLOBAL), UOp.range(WARP_SIZE, -1, AxisType.WARP), UOp.range(RES_WAVES, 1, AxisType.LOCAL)
  groups = [(tw%G*RES_WAVES + wave)*RES_PER + k for k in range(RES_PER)]
  return G, tw, tw//G, lane, wave, groups, [g*Q8_GROUP_SIZE + lane for g in groups]

@functools.cache
def _res_mix_partial_kernel(part:UOp, xs:UOp, wm:UOp, *upd:UOp, tokens:int, streams_in:int, new_block:int, dim:int) -> UOp:
  """a workgroup's slice of a token's streams' gram matrix (upper triangle) and of their dots with the mix weight. upd: (xs_new, out), the
  streams updated by out (new_block: as a new stream, else onto the last) are written to xs_new"""
  _, tw, t, lane, wave, _, idx = _res_slice(tokens, dim)
  xv = [[xs.reshape(tokens, streams_in, dim)[t, s_, i].load() for i in idx] for s_ in range(streams_in)]
  if upd:
    o = [upd[1].reshape(tokens, dim)[t, i].load() for i in idx]
    xv = xv + [o] if new_block else xv[:-1] + [[x + oi for x, oi in zip(xv[-1], o)]]
  S, w, zero = len(xv), [wm[i].load() for i in idx], UOp.const(0, dtypes.float)
  stores = [upd[0].reshape(tokens, S, dim)[t, s_, i].store(x) for s_ in range(S) for i, x in zip(idx, xv[s_])] if upd else []
  vals = [sum((a*b for a, b in zip(xv[s_], xv[r])), zero) for s_ in range(S) for r in range(s_, S)] + \
         [sum((a*b for a, b in zip(xv[s_], w)), zero) for s_ in range(S)]
  red = UOp.alloc((RES_WAVES, len(vals)), dtypes.float, addrspace=AddrSpace.LOCAL)
  red = red.after(UOp.group(*[red[wave, j].store(warp_reduce(v, full_wave=True)) for j, v in enumerate(vals)]).barrier())
  tid = wave*WARP_SIZE + lane
  assert len(vals) <= RES_WAVES*WARP_SIZE
  col = tid.minimum(len(vals)-1)
  stores.append(part[tw, tid.valid(tid < len(vals))].store(sum((red[w_, col].load() for w_ in range(1, RES_WAVES)), red[0, col].load())))
  return UOp.group(*stores).end(lane, wave, tw).sink(arg=KernelInfo(name="res_mix_partial", opts_to_apply=()))

@functools.cache
def _res_mix_norm_kernel(y:UOp, q:UOp, scale:UOp, xsum:UOp, part:UOp, wn:UOp, xs:UOp, tokens:int, streams:int, dim:int, eps:float,
                         norm_eps:float) -> UOp:
  """y = rmsnorm(sum_s p_s xs_s) * wn, p = softmax over the streams of (rmsnorm(xs_s) * wm).sum(), and the q8 of y (q8_quantize's layout).
  a workgroup sums its token's workgroups' partials (sum(h^2) is p G p of the gram matrix G) and does its slice"""
  G, tw, t, lane, wave, groups, idx = _res_slice(tokens, dim)
  NP, zero, tid = streams*(streams+1)//2, UOp.const(0, dtypes.float), wave*WARP_SIZE + lane
  tot_l, col = UOp.alloc((NP + streams,), dtypes.float, addrspace=AddrSpace.LOCAL), tid.minimum(NP + streams - 1)
  tot_l = tot_l.after(tot_l[tid.valid(tid < NP + streams)].store(sum((part[t*G + g, col].load() for g in range(1, G)), part[t*G, col].load()))
                      .barrier())
  tot = [tot_l[j].load() for j in range(NP + streams)]
  pair = {st: k for k, st in enumerate((s_, r) for s_ in range(streams) for r in range(s_, streams))}
  def gram(s_:int, r:int) -> UOp: return tot[pair[min(s_, r), max(s_, r)]]
  score = [tot[NP+s_] / (gram(s_, s_) * (1/dim) + eps).sqrt() for s_ in range(streams)]
  m = functools.reduce(UOp.maximum, score)
  e = [((sc - m) * LOG2E).exp2() for sc in score]
  esum = sum(e[1:], e[0])
  hh = sum((e[s_] * e[r] * gram(s_, r) for s_ in range(streams) for r in range(streams)), zero) / (esum * esum)
  rn = 1 / (hh * (1/dim) + norm_eps).sqrt()
  stores = []
  for g, i in zip(groups, idx):
    value = sum((e[s_] * xs.reshape(tokens, streams, dim)[t, s_, i].load() for s_ in range(streams)), zero) / esum * rn * wn[i].load()
    stores += [y.reshape(tokens, dim)[t, i].store(value), *_q8_group(q, scale, xsum, t, g, lane, value)]
  return UOp.group(*stores).end(lane, wave, tw).sink(arg=KernelInfo(name="res_mix_norm", opts_to_apply=()))

def res_mix_norm(xs:Tensor, wm:Tensor, wn:Tensor, eps:float, norm_eps:float, out:Tensor|None=None, new_block:bool=False) -> tuple[Tensor, Tensor]:
  """the (1, T, dim) input of a sublayer from the residual streams xs (1, T, streams, dim): their mix then the sublayer's rmsnorm. with out
  (1, T, 1, dim), the output of the sublayer before, the streams are first updated with it (AttentionResidual.update), in the kernels.
  returns the input and the streams. its q8 goes to the q8 cache, the linears that take it don't quantize it again. two kernels over
  slices of the tokens' streams: the partials of the reductions, then each slice's mix from all of its token's"""
  _, T, streams_in, dim = xs.shape
  new_block = new_block and out is not None
  streams = streams_in + new_block
  part = Tensor.empty(T * dim // (RES_GROUPS*Q8_GROUP_SIZE), streams*(streams+1)//2 + streams, dtype=dtypes.float32, device=xs.device)
  upd = [] if out is None else [Tensor.empty(1, T, streams, dim, dtype=dtypes.float32, device=xs.device), out]
  part, _, _, *upd = Tensor.custom_kernel(part, xs, wm, *upd, fxn=functools.partial(_res_mix_partial_kernel, tokens=T, streams_in=streams_in,
                                          new_block=int(new_block), dim=dim))
  if upd: xs = upd[0]
  y = Tensor.empty(1, T, dim, dtype=dtypes.float32, device=xs.device)
  q = Tensor.empty(T, dim//Q8_GROUP_SIZE, 8, dtype=dtypes.uint32, device=xs.device)
  scale = Tensor.empty(T, dim//Q8_GROUP_SIZE, dtype=dtypes.float32, device=xs.device)
  xsum = Tensor.empty(T, dim//Q8_GROUP_SIZE, 2, dtype=dtypes.float32, device=xs.device)
  fxn = functools.partial(_res_mix_norm_kernel, tokens=T, streams=streams, dim=dim, eps=eps, norm_eps=norm_eps)
  y, q, scale, xsum = Tensor.custom_kernel(y, q, scale, xsum, part, wn, xs, fxn=fxn)[:4]
  _q8_cache[y.uop] = (q, scale, xsum)
  return y, xs

def _decode_linear(out:UOp, out_features:int, group_count:int, group_dot, name:str, per:int=0, tokens:int=0, bias:UOp|None=None,
                   tables:tuple[UOp, ...]=(), live:Callable[[UOp], UOp]|None=None, lane_outs:int=1, joint:bool=False) -> UOp:
  # a wave per output and chunk of its groups (a lane a group, per of them), neighboring rows share the workgroup. the chunks' waves of an
  # output add up their sums in LDS (from tinygrad PR 18863): out is (tokens, out_features), no chunk partials to sum after
  # short rows (at most 32 groups): 8 lanes an output, all its groups, 4 outputs a wave
  # tokens: a lane takes every token of its groups (out (rows*tokens, out_features), row r token t at r*tokens+t): the tokens share the
  # weight loads, a row is read once. live: a token's live groups (a prefix of its row's), a lane's groups past them not read (a loop).
  # lane_outs: a lane takes that many neighboring outputs of its groups, they share the input loads. joint: the tokens' live groups are the
  # same (live of the first), one loop: their weight loads shared
  n_rows, of = out.shape[0] // max(tokens, 1) * out_features // lane_outs, out_features // lane_outs
  lanes = 8 if 8 <= group_count <= 32 and n_rows % 16 == 0 else 32
  slots = (group_count + lanes - 1) // lanes
  per = (min(per, slots) or (slots + 15) // 16) if lanes == 32 else slots
  chunks = (slots + per - 1) // per
  rows = math.gcd(of, 4)
  while rows > 1 and rows * chunks > 16: rows //= 2
  outs = rows * 32 // lanes  # the output groups of a workgroup
  row = UOp.range(n_rows//outs, 0, AxisType.GLOBAL)
  wave, chunk, lr = UOp.range(rows, 1, AxisType.LOCAL), UOp.range(chunks, 3, AxisType.LOCAL), UOp.range(32, -1, AxisType.WARP)
  token_output, lane = (row*rows+wave)*(32//lanes) + lr//lanes, lr % lanes
  toks = [token_output // of * max(tokens, 1) + t for t in range(max(tokens, 1))]
  outputs = [token_output % of * lane_outs + r for r in range(lane_outs)]
  if tables:  # (group_dot's lookup tables in LDS: their loads off the path of the weights')
    threads, tid = rows*chunks*32, (wave*chunks + chunk)*32 + lr
    lts = [UOp.alloc((int(t.numel()),), t.dtype, addrspace=AddrSpace.LOCAL) for t in tables]
    fill = UOp.group(*[lt[(tid + k*threads).valid(tid + k*threads < n)].store(t[(tid + k*threads).minimum(n-1)].load())
                       for lt, t in zip(lts, tables) for n in [int(t.numel())] for k in range(-(-n // threads))]).barrier()
    tables = tuple(lt.after(fill) for lt in lts)
  def dot(token:UOp, output:UOp, group:UOp) -> UOp:
    return (group < group_count).where(group_dot(token, output, group.minimum(group_count-1), tables), UOp.const(0, dtypes.float32))
  def lane_sums(j:int, ts:list[UOp]) -> list[UOp]:  # (the sums of the tokens ts, each of its outputs)
    if live is None:
      return [functools.reduce(lambda a, i: a + dot(t, o, lane + (chunk*per + i)*lanes), range(1, per), dot(t, o, lane + chunk*per*lanes))
              for t in ts for o in outputs]
    i = UOp.range(((live(ts[0]) - lane - chunk*per*lanes + lanes - 1) // lanes).maximum(0).minimum(per), 10 + j, AxisType.LOOP)
    acc = _reg((len(ts)*len(outputs),), 0)
    acc = acc.after(UOp.group(*[acc[r].store(acc.after(i)[r].load() + dot(t, o, lane + (chunk*per + i)*lanes))
                                for r, (t, o) in enumerate((t, o) for t in ts for o in outputs)]).end(i))
    return [acc[r].load() for r in range(len(ts)*len(outputs))]
  pairs = [(t, o) for t in toks for o in outputs]
  totals = [warp_reduce(s_, full_wave=True, width=lanes) for j, ts in enumerate([toks] if joint else [[t] for t in toks])
            for s_ in lane_sums(j, ts)]
  if chunks > 1:
    lds = UOp.alloc((rows, chunks, len(pairs)), dtypes.float32, addrspace=AddrSpace.LOCAL)
    lds = lds.after(UOp.group(*[lds[wave, chunk.valid(lane.eq(0)), j].store(tot) for j, tot in enumerate(totals)]).barrier())
    totals = [functools.reduce(lambda a, c: a + lds[wave, c, j].load(), range(1, chunks), lds[wave, 0, j].load()) for j in range(len(pairs))]
  if bias is not None: totals = [tot + bias.reshape(-1)[t*out_features + o].load() for (t, o), tot in zip(pairs, totals)]
  return UOp.group(*[out[t, o.valid(lane.eq(0) & chunk.eq(0))].store(tot.cast(out.dtype)) for (t, o), tot in zip(pairs, totals)]) \
    .end(row, wave, chunk, lr).sink(arg=KernelInfo(name=name, opts_to_apply=()))

@functools.cache
def _iq_grid(device:str|tuple[str, ...]|None, ggml_type:int) -> Tensor:
  from tinygrad.runtime.autogen import ggml_common as ggml
  grid, words = {IQ2_XS: (ggml.iq2xs_grid, 2), IQ2_S: (ggml.iq2s_grid, 2),
                 IQ3_XXS: (ggml.iq3xxs_grid, 1), IQ3_S: (ggml.iq3s_grid, 1)}[ggml_type]
  # a buffer once: a const table is a kernel in every graph that takes it
  with Context(ALLOW_DEVICE_USAGE=1):
    return Tensor.const(tuple((v >> (32*i)) & 0xffffffff for v in grid for i in range(words)), dtypes.uint32).to(device, force=True).realize()

@functools.cache
def _iq_signs(device:str|tuple[str, ...]|None) -> Tensor:
  """the byte masks of the 7 sign bits of 8 weights (the 8th makes the count even): 2 words each, a byte 0xff for a negative weight"""
  def mask(bits:int) -> int: return sum(255 << (8*b) for b in range(4) if bits >> b & 1)
  even = [k | (bin(k).count("1") & 1) << 7 for k in range(128)]
  with Context(ALLOW_DEVICE_USAGE=1):
    return Tensor.const(tuple(mask(e >> (4*h) & 15) for e in even for h in range(2)), dtypes.uint32).to(device, force=True).realize()

def _iq_tables(device:str|tuple[str, ...]|None, ggml_type:int) -> tuple[Tensor, ...]:
  """a decode kernel's lookup tables: the grid, and the sign masks of the formats with 7 sign bits"""
  if ggml_type not in (IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S): return ()
  return (_iq_grid(device, ggml_type), *((_iq_signs(device),) if ggml_type in (IQ2_XS, IQ3_XXS) else ()))

def _iq_even_signs(signs:UOp) -> UOp:  # (the 8th sign bit makes the count even)
  return signs | ((UOp(Ops.CUSTOMI, src=(signs.cast(dtypes.uint32),), arg=("__builtin_popcount({})", dtypes.uint32)) & 1) << 7)

def _iq_signed_word(word:UOp, signs:UOp) -> UOp:
  # sign bit i to bit 0 of byte i (the shifted copies don't overlap), each byte's mask a byte_perm of it (selector 12: 0x00, 13: 0xff)
  low = ((signs & 15) * 0x00204081) & 0x01010101
  # Grid magnitudes are nonzero and <128, so each byte can be negated without a carry into its neighbor.
  return (word ^ _amd_byte_perm(UOp.const(0, dtypes.uint32), UOp.const(0, dtypes.uint32), low | 0x0c0c0c0c)) + low

def _quant_word(raw:UOp, base:UOp, subgroup:UOp, i:int|UOp, ggml_type:int, grid:UOp|None, sign_masks:UOp|None=None) -> UOp:
  # Four packed weight bytes, shared by integer-dot decode and FP16 WMMA prefill.
  def byte(offset): return _load_byte(raw, base, offset)
  def word(offset): return _load_u32(raw, base, offset, stream=ggml_type == Q6_K)
  if ggml_type == Q8_0: return _load_u32(raw, base, 2+i*4, stream=True)
  if ggml_type in (Q4_K, Q5_K):
    offset = (4 if ggml_type == Q4_K else 12) + (subgroup//2)*8
    weights = (_amd_load(raw[base+offset], 8)[i] >> ((subgroup&1)*4).cast(dtypes.uint32)) & 0x0f0f0f0f
    if ggml_type == Q5_K: weights |= ((_amd_load(raw[base+4], 8)[i] >> subgroup.cast(dtypes.uint32)) & 0x01010101) << 4
    return weights
  if ggml_type == Q6_K:
    low = word((subgroup//4)*64 + (subgroup%2)*32 + i*4) >> ((subgroup%4//2)*4).cast(dtypes.uint32)
    high = word(128 + (subgroup//4)*32 + i*4) >> ((subgroup%4)*2).cast(dtypes.uint32)
    return (low & 0x0f0f0f0f) | ((high & 0x03030303) << 4)
  if ggml_type in (Q2_K, Q3_K):
    offset = (16 if ggml_type == Q2_K else 32) + (subgroup//4)*32 + i*4
    weights = (word(offset) >> ((subgroup%4)*2)) & 0x03030303
    if ggml_type == Q3_K: weights |= ((word(i*4) >> subgroup) & 0x01010101) << 2
    return weights
  if ggml_type in (IQ4_NL, IQ4_XS):
    packed = word(2+(i%4)*4) if ggml_type == IQ4_NL else _amd_load(raw[base+2+subgroup*4+i%4])
    return _iq4_bytes(packed, 4*(i//4))
  assert grid is not None
  grid_words = 2 if ggml_type in (IQ2_XS, IQ2_S) else 1
  if ggml_type in (IQ3_S, IQ3_XXS):
    index = byte(2+subgroup*8+i)
    if ggml_type == IQ3_S:
      index |= ((byte(66+subgroup) >> i) & 1) << 8
      signs = byte(74+subgroup*4+i//2)
    else: signs = (word(66+subgroup*4) >> (7*(i//2))) & 127
  elif ggml_type == IQ2_XS:
    packed = raw[base+1+subgroup*4+i//2].cast(dtypes.uint32)
    index, signs = packed & 511, packed >> 9
  else:
    index = byte(2+subgroup*4+i//2) | (((byte(66+subgroup) >> (2*(i//2))) & 3) << 8)
    signs = byte(34+subgroup*4+i//2)
  if ggml_type in (IQ2_XS, IQ3_XXS):  # 7 sign bits: a table's masks, or the 8th made
    if sign_masks is not None:
      m = sign_masks[signs*2 + i%2]
      # Grid magnitudes are nonzero and <128, so each byte can be negated without a carry into its neighbor.
      return (grid[index*grid_words+i%grid_words] ^ m) + (m & 0x01010101)
    signs = _iq_even_signs(signs)
  return _iq_signed_word(grid[index*grid_words+i%grid_words], signs >> (4*(i%2)))

@functools.cache
def _quant_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, *extra:UOp, out_features:int, in_features:int, ggml_type:int,
                         gather:tuple[int, int]|None=None, tokens:int=0, weighted:bool=False, summed:bool=False, bias:bool=False,
                         shared:bool=False, cdna:bool=False, gate:Callable|None=None, xtoken:Callable|None=None, name:str="linear_") -> UOp:
  # gather=(stride, offset): the rows of output token t are the weights at raw[slots[t]*stride + offset] (in raw elements),
  # extra[0] is the slots table. with fewer input rows than output rows, output row r reads input row r % rows (the experts of one
  # token share their input). summed: one output row, the sum over the slots s of their rows on input row s weighted by extra[1][s] (a
  # row's groups go over the slots); with tokens an output row a token t, slots, input rows and weights s*tokens+t. weighted: extra[1] the
  # slots' weights, a slot of weight 0 (padding) skipped (its slots a token's last). bias: the next extra (out's shape) added to the outputs.
  # shared: a slot every tokens rows (weights): each slot's weights read once for all of its tokens (a slot of weights all 0 skipped)
  # gate(token, base) -> base: moves the weight rows, xtoken(token) -> the activation row of an output row
  slots, grids = (extra[0], extra[1:]) if gather is not None else (None, extra)
  w, grids = (grids[0], grids[1:]) if weighted else (None, grids)
  b, grids = (grids[0], grids[1:]) if bias else (None, grids)
  if gather is None and gate is None and 1 < xq.shape[0] <= 8: tokens = int(xq.shape[0])  # (a few tokens of a linear share its weight loads)
  # (in LDS: the grid table, and the slots and weights of a summed row: a lane's slot isn't a load from memory before its weights')
  tables = (*grids, *((slots, w) if summed and slots is not None and w is not None else ()))
  def group_dot(token:UOp, output:UOp, group:UOp, lds:tuple[UOp, ...]) -> UOp:
    table, sign_masks = lds[0] if grids else None, lds[1] if len(grids) > 1 else None
    if w is None or not summed: return slot_dot(token, output, group, table, sign_masks, slots)
    k, group = group // (in_features//32), group % (in_features//32)
    if tokens: k = k*tokens + token
    return slot_dot(k, output, group, table, sign_masks, lds[-2]) * lds[-1][k].load()
  def slot_dot(token:UOp, output:UOp, group:UOp, table:UOp|None, sign_masks:UOp|None, slots:UOp|None) -> UOp:
    block, subgroup = group//8, group%8
    base = (output*in_features//256 + block) * (QUANT_SIZES[ggml_type]//raw.dtype.itemsize)
    if ggml_type in SMALL_BLOCK_QUANTS: base = (output*in_features//32 + group)*SMALL_BLOCK_QUANTS[ggml_type]
    # a cache pool can pass 2**31 elements: the slot offset is 64 bit
    if slots is not None and gather is not None:
      base = base.cast(dtypes.int64) + slots[token // tokens if shared else token].load().cast(dtypes.int64)*gather[0] + gather[1]
    if gate is not None: base = gate(token, base)
    if xtoken is not None: token = xtoken(token)
    elif xq.shape[0] != out.shape[0]: token = token % xq.shape[0]
    def byte(offset): return _load_byte(raw, base, offset)
    def word(offset): return _load_u32(raw, base, offset)
    xwords = _amd_load(xq[token, group, 0], 8)
    # One accumulator per scale group: 32 weights for Q4/Q5/IQ4_XS, two groups of 16 otherwise.
    dots = [UOp.const(0, dtypes.int32)] * (1 if ggml_type in (Q4_K, Q5_K, IQ4_XS) else 2)
    for i in range(8):
      weights = _quant_word(raw, base, subgroup, i, ggml_type, table, sign_masks)
      acc = i//(8//len(dots))
      dots[acc] = _amd_dp4a(weights, xwords[i], dots[acc], cdna)
    if ggml_type in (Q4_K, Q5_K):
      d, dmin, scale, minimum = _q5_scales(raw, base, subgroup)
      total = dots[0].float()*d*scale - (xs[token, group, 0].load()+xs[token, group, 1].load())*dmin*minimum
    elif ggml_type == IQ4_XS:
      total = dots[0].float() * _iq4_scale(raw, base, subgroup)
    elif ggml_type == Q6_K:
      # Subtract the quant offset via the activation sums instead of unpacking signed bytes.
      scales = [(raw[base+96+subgroup] >> (h*8)).cast(dtypes.uint8).bitcast(dtypes.int8).float() for h in range(2)]
      total = sum((dots[h].float()-32*xs[token, group, h].load())*scales[h] for h in range(2))
      return total * xd[token, group] * _half(raw[base+104])
    elif ggml_type in (Q2_K, Q3_K):
      total = UOp.const(0, dtypes.float32)
      for half in range(2):
        j = subgroup*2+half
        if ggml_type == Q2_K:
          scale = byte(j)
          total += dots[half].float() * (scale & 15).float() * _half(raw[base+40]) - \
                   xs[token, group, half] * (scale >> 4).float() * _half(raw[base+41])
        else:
          scale = ((byte(96+j%8) >> ((j//8)*4)) & 15) | (((byte(104+j%4) >> ((j//4)*2)) & 3) << 4)
          total += (dots[half].float() - 4*xs[token, group, half]) * (scale.cast(dtypes.int32)-32).float() * _half(raw[base+54])
    elif ggml_type in (IQ2_XS, IQ2_S):
      scales = byte((66 if ggml_type == IQ2_XS else 74)+subgroup)
      total = sum(dots[h].float() * (((scales >> (h*4)) & 15).float()+0.5) for h in range(2)) * 0.25 * _half(raw[base])
    else:
      total = (dots[0]+dots[1]).float() * _half(raw[base])
      if ggml_type == IQ3_S: total *= (1+2*((byte(106+subgroup//2) >> ((subgroup%2)*4)) & 15)).float()
      if ggml_type == IQ3_XXS: total *= ((word(66+subgroup*4) >> 28).float()+0.5)*0.5
    return total * xd[token, group]
  # (the expert gemvs: 2 groups a lane, profiled best)
  gpk, n = in_features//32, int(xq.shape[0]) // max(tokens, 1) if summed else 1
  def live(t:UOp) -> UOp:  # the groups of a row's slots of weight not 0: a slot's, or the summed slots' of a token (shared: of any token)
    assert w is not None
    def on(k:UOp|int) -> UOp:
      if not shared: return w[k].load().ne(0)
      u = k // tokens if isinstance(k, UOp) else k
      return functools.reduce(UOp.__or__, [w[u*tokens + i].load().ne(0) for i in range(tokens)])
    if not summed: return on(t).where(gpk, 0)
    return sum(((on(k*max(tokens, 1) + t) if not shared else on(k)).cast(dtypes.int) for k in range(n)), UOp.const(0, dtypes.int)) * gpk
  return _decode_linear(out, out_features, gpk * n, group_dot, name+QUANT_NAMES[ggml_type], per=2 if gather is not None else 0,
                        tokens=tokens, bias=b, tables=tables, live=live if w is not None else None,
                        lane_outs=2 if summed else 1, joint=shared)

def _quant_decode_body(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, grids:tuple[UOp, ...], out_features:int, in_features:int, ggml_type:int,
                       cdna:bool, gate:Callable|None=None, xtoken:Callable|None=None, name:str="linear_") -> UOp:
  return _quant_decode_kernel(out, raw, xq, xd, xs, *grids, out_features=out_features, in_features=in_features, ggml_type=ggml_type,
                              cdna=cdna, gate=gate, xtoken=xtoken, name=name)

@functools.cache
def _moe_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, sel:UOp, *grids:UOp,
                       out_features:int, in_features:int, ggml_type:int, x_share:int, cdna:bool=False) -> UOp:
  # out rows are (token, slot): the slot's expert reads its weight rows, x_share slots share an activation row
  expert_words = out_features * in_features * QUANT_SIZES[ggml_type] // 256 // raw.dtype.itemsize
  def gate(row:UOp, base:UOp) -> UOp: return base + sel[row].cast(dtypes.weakint).maximum(0).minimum(raw.shape[0]//expert_words-1) * expert_words
  def xtoken(row:UOp) -> UOp: return row // x_share
  return _quant_decode_body(out, raw, xq, xd, xs, grids, out_features, in_features, ggml_type, cdna, gate, xtoken, "moe_")

def moe_linear(layer:ExpertWeights, sel:Tensor, x:Tensor) -> Tensor:
  B, T, slots = cast(tuple[int, int, int], sel.shape[-3:]) # sel: (B, T, k), weight: (E, O, I)
  in_features, x_slots = int(x.uop.shard_shape[-1]), int(x.shape[-2]) # gate/up: (B, T, 1, D), down: (B, T, k, H)
  assert x_slots in (1, slots), f"x has {x_slots} rows per token, expected 1 or {slots}"
  local_out, global_out, ggml_type = int(layer.weight.uop.max_shard_shape[1]), int(layer.weight.shape[1]), unwrap(layer.ggml_type)
  splits = layer.in_features // in_features
  assert (layer.weight.uop.axis == 2) == (splits > 1), f"input features split over {splits} devices, weight split on axis {layer.weight.uop.axis}"
  xq, xd, xs = q8_quantize(x.contiguous(), B*T*x_slots, in_features)
  # input features split: every device writes the partial sums of its rows, the sum over the devices is the allreduce
  out = Tensor.empty(splits*B*T*slots, global_out, dtype=dtypes.float32, device=x.device, axis=0 if splits > 1 else layer.weight.uop.axis)
  fxn = functools.partial(_moe_decode_kernel, out_features=local_out, in_features=in_features, ggml_type=ggml_type, x_share=slots//x_slots,
                          cdna=_is_cdna(x.device))
  extra = _iq_tables(x.device, ggml_type)
  result = Tensor.custom_kernel(out, layer.packed, xq, xd, xs, sel.cast(dtypes.int32).flatten(), *extra, fxn=fxn)[0]
  if splits > 1: result = partial_sum(result.reshape(splits, B*T*slots, global_out))
  return result.reshape(B, T, slots, global_out)

# kgroups: groups of 16 lanes that split k in a 16x16x16 fragment, RDNA3 copies a and b in both halves
def _wmma_words(lane_group:UOp, kgroups:int) -> tuple[UOp|int, ...]:
  return tuple(range(4)) if kgroups == 1 else tuple(lane_group + kgroups*j for j in range(4//kgroups))

def _wmma_layout(out:UOp, out_features:int, token_tile:int, output_tiles:int, lanes:int):
  if out_features % (16*output_tiles): output_tiles = 1
  output_waves = 2 if out_features % (32*output_tiles) == 0 else 1
  token_block, output_block = UOp.range(out.shape[0]//token_tile, 0), UOp.range(out_features//(16*output_tiles*output_waves), 1)
  # lane is a hardware WARP range (like the flash kernel): the fragment math stays visible without being
  # range-split into nested loops, which would scramble the WMMA fragment layout
  lane, wave = UOp.range(lanes, -1, axis_type=AxisType.WARP), UOp.range(output_waves, 3, axis_type=AxisType.LOCAL)
  col, lane_group, rows = lane % 16, lane // 16, 256 // lanes
  outputs = tuple((output_block*output_waves+wave)*(16*output_tiles) + tile*16 + col for tile in range(output_tiles))
  inputs = tuple(token_block*token_tile + tile*16 + col for tile in range(token_tile//16))
  tokens = tuple(tuple(token_block*token_tile + tile*16 + lane_group*rows + i for i in range(rows)) for tile in range(token_tile//16))
  return output_waves, token_block, output_block, lane, wave, lane_group, outputs, inputs, tokens

def _wmma_stores(out, outputs, tokens, accs, update, lane, wave, output_waves, kgroups):
  # RDNA4 and CDNA (kgroups > 1) own consecutive rows per lane group, matching tokens directly.
  if kgroups > 1:
    return [out[token, output].store(acc.after(update)[i].load().cast(out.dtype)) for output,output_accs in zip(outputs, accs)
            for tile_tokens,acc in zip(tokens, output_accs) for i,token in enumerate(tile_tokens)]
  # the accumulator fragment halves are exchanged between lane pairs (l, l^16) through LDS (a ds_swizzle without CUSTOM)
  flat_accs = [acc for output_accs in accs for acc in output_accs]
  lds = UOp.alloc((output_waves, 32, len(flat_accs)*8), dtypes.float32, addrspace=AddrSpace.LOCAL)
  stores = [lds[wave, lane, a*8+i].store(acc.after(update)[i].load()) for a,acc in enumerate(flat_accs) for i in range(8)]
  lds = lds.after(*stores)
  def values(ai:int) -> tuple[UOp, ...]:
    own = tuple(lds[wave, lane, ai*8+i].load() for i in range(8))
    peer = tuple(lds[wave, lane ^ 16, ai*8+i].load() for i in range(8))
    low = (lane // 16).eq(0)
    return tuple(low.where(own[i], peer[i+4]) if j == 0 else low.where(peer[i], own[i+4]) for i in range(4) for j in range(2))
  tt = len(tokens)
  return [out[token, output].store(value.cast(out.dtype)) for ot,(output,output_accs) in enumerate(zip(outputs, accs))
          for tile,(tile_tokens,_acc) in enumerate(zip(tokens, output_accs)) for token,value in zip(tile_tokens, values(ot*tt+tile))]

def _quant_linear_wmma(out, x, out_features, in_features, type_words, layout, dequant, name, kgroups, lanes):
  x = x.reshape(out.shape[0], in_features)
  output_waves, token_block, output_block, lane, wave, lane_group, outputs, input_tokens, tokens = layout
  token_tile, output_tiles = len(tokens)*16, len(outputs)
  # IQ4_NL blocks are 32 wide: a device can hold part of a 256 weight row group
  output_words = in_features * type_words // GGML_BLOCK_SIZE
  accs = tuple(tuple(UOp.alloc((len(tokens[0]),), dtypes.float32, addrspace=AddrSpace.REG)
                     for tile in range(token_tile // 16)) for ot in range(output_tiles))
  accs = tuple(tuple(acc.after(acc.store(acc.const_like(0))) for acc in output_accs) for output_accs in accs)
  group = UOp.range(in_features // Q8_GROUP_SIZE, 4, AxisType.LOOP)
  block, subgroup = group // 8, group % 8
  wmma_accs = [list(output_accs) for output_accs in accs]
  ks = tuple(word*4 + b for word in _wmma_words(lane_group, kgroups) for b in range(4))
  for half in range(2):
    afrags = tuple(UOp.stack(*(x[input_token, group*32 + half*16 + i].cast(dtypes.float16) for i in ks))
                   for input_token in input_tokens)
    for output_tile,output in enumerate(outputs):
      bfrag = UOp.stack(*dequant(output*output_words + block*type_words, subgroup, half))
      for tile,afrag in enumerate(afrags):
        previous = accs[output_tile][tile].after(group) if half == 0 else wmma_accs[output_tile][tile]
        wmma_accs[output_tile][tile] = UOp.wmma(afrag, bfrag, previous, WMMA_ARG[0], lanes)
  update = UOp.group(*(acc.store(value) for output_accs,output_values in zip(accs, wmma_accs)
                       for acc,value in zip(output_accs, output_values))).end(group)
  stores = _wmma_stores(out, outputs, tokens, accs, update, lane, wave, output_waves, kgroups)
  return UOp.group(*stores).end(token_block, output_block, lane, wave).sink(arg=KernelInfo(name=name, opts_to_apply=()))

@functools.cache
def _q5_linear_f16_wmma_kernel(out:UOp, raw:UOp, x:UOp, out_features:int, in_features:int, ggml_type:int, kgroups:int, lanes:int) -> UOp:
  token_tile, output_tiles = (64, 1 if out_features <= 1024 else 2) if out.shape[0] % 64 == 0 else \
    (32 if out.shape[0] % 32 == 0 else 16, 2)
  layout = _wmma_layout(out, out_features, token_tile, output_tiles, lanes)
  word_indices = _wmma_words(layout[5], kgroups)
  def dequant(base:UOp, subgroup:UOp, half:int) -> tuple[UOp, ...]:
    d, dmin, scale, minimum = _q5_scales(raw, base, subgroup)
    qs_base = base + (4 if ggml_type == Q4_K else 12) + (subgroup // 2)*8 + half*4
    words = tuple((raw[qs_base+i] >> ((subgroup&1)*4).cast(dtypes.uint32) & 0x0f0f0f0f) |
      (((raw[base+4+half*4+i] >> subgroup.cast(dtypes.uint32) & 0x01010101) << 4) if ggml_type == Q5_K else 0) for i in word_indices)
    return tuple(((word >> (byte*8) & 255).float()*d*scale-dmin*minimum).cast(dtypes.float16) for word in words for byte in range(4))
  return _quant_linear_wmma(out, x, out_features, in_features, QUANT_SIZES[ggml_type]//4,
                            layout, dequant, f"linear_q{4 if ggml_type == Q4_K else 5}_k_f16_wmma", kgroups, lanes)

@functools.cache
def _iq4_linear_f16_wmma_kernel(out:UOp, raw:UOp, x:UOp, lut:UOp, out_features:int, in_features:int, kgroups:int, lanes:int) -> UOp:
  token_tile = 32 if out_features <= 1024 and out.shape[0] % 32 == 0 else 64 if out.shape[0] % 64 == 0 and \
    out_features <= 6144 else 128 if out.shape[0] % 128 == 0 else \
    32 if out.shape[0] % 32 == 0 else 16
  output_tiles = 1 if out_features <= 1024 else 2 if out_features <= 6144 else 1 if out_features < 8192 else 2
  layout = _wmma_layout(out, out_features, token_tile, output_tiles, lanes)
  output_waves, _, _, lane, wave, lane_group, _, _, _ = layout
  word_indices = _wmma_words(lane_group, kgroups)
  local_lut = UOp.alloc((256,), dtypes.uint32, addrspace=AddrSpace.LOCAL)
  tid, lut_items = wave*lanes+lane, 256//(lanes*output_waves)
  lut = local_lut.after(*(local_lut[tid*lut_items+i].store(lut[tid*lut_items+i]) for i in range(lut_items)))
  def dequant(base:UOp, subgroup:UOp, half:int) -> tuple[UOp, ...]:
    scale = _iq4_scale(raw, base, subgroup)
    pairs = tuple(lut[((raw[base + 2 + subgroup*4 + word] >> (byte*8)) & 255).cast(dtypes.weakint)]
                  for word in word_indices for byte in range(4))
    return tuple((_half((pair >> (half*16)) & 0xffff)*scale).cast(dtypes.float16) for pair in pairs)
  return _quant_linear_wmma(out, x, out_features, in_features, QUANT_SIZES[IQ4_XS]//4, layout, dequant, "linear_iq4_xs_f16_wmma", kgroups, lanes)

@functools.cache
def _quant_linear_f16_wmma_kernel(out:UOp, raw:UOp, x:UOp, *grids:UOp,
                                  out_features:int, in_features:int, ggml_type:int, kgroups:int, lanes:int) -> UOp:
  token_tile = 64 if out.shape[0] % 64 == 0 else 32 if out.shape[0] % 32 == 0 else 16
  layout = _wmma_layout(out, out_features, token_tile, 2, lanes)
  output_waves, _, _, lane, wave, lane_group, _, _, _ = layout
  word_indices = _wmma_words(lane_group, kgroups)
  grid = None
  if grids:
    grid = UOp.alloc((int(grids[0].numel()),), dtypes.uint32, addrspace=AddrSpace.LOCAL)
    tid, threads = wave*lanes+lane, output_waves*lanes
    grid = grid.after(*(grid[tid+i*threads].store(grids[0][tid+i*threads]) for i in range(int(grid.numel())//threads)))
  def dequant(base:UOp, subgroup:UOp, half:int) -> tuple[UOp, ...]:
    if ggml_type in SMALL_BLOCK_QUANTS: base += subgroup*SMALL_BLOCK_QUANTS[ggml_type]  # eight independent blocks per 256 weights
    def byte(offset): return _load_byte(raw, base, offset)
    zero, minimum = 0, UOp.const(0, dtypes.float)
    if ggml_type == Q2_K:
      sc = byte(subgroup*2+half)
      scale, minimum = _half(raw[base+40])*(sc & 15).float(), _half(raw[base+41])*(sc >> 4).float()
    elif ggml_type == Q3_K:
      j = subgroup*2+half
      sc = ((byte(96+j%8) >> ((j//8)*4)) & 15) | (((byte(104+j%4) >> ((j//4)*2)) & 3) << 4)
      scale, zero = _half(raw[base+54])*(sc.cast(dtypes.int32)-32).float(), 4
    elif ggml_type == Q6_K:
      scale, zero = _half(raw[base+104])*byte(192+subgroup*2+half).cast(dtypes.uint8).bitcast(dtypes.int8).float(), 32
    else:
      scale = _half(raw[base])
      if ggml_type in (IQ2_XS, IQ2_S):
        sc = byte((66 if ggml_type == IQ2_XS else 74)+subgroup)
        scale *= (((sc >> (half*4)) & 15).float()+0.5)*0.25
      elif ggml_type == IQ3_S: scale *= (1+2*((byte(106+subgroup//2) >> ((subgroup%2)*4)) & 15)).float()
      elif ggml_type == IQ3_XXS: scale *= ((_load_u32(raw, base, 66+subgroup*4) >> 28).float()+0.5)*0.5
    words = tuple(_quant_word(raw, base, subgroup, half*4+i, ggml_type, grid) for i in word_indices)
    return tuple((((word >> (b*8)).cast(dtypes.uint8).bitcast(dtypes.int8).float()-zero)*scale-minimum).cast(dtypes.half)
                 for word in words for b in range(4))
  return _quant_linear_wmma(out, x, out_features, in_features, QUANT_SIZES[ggml_type]//raw.dtype.itemsize,
                            layout, dequant, f"linear_{QUANT_NAMES[ggml_type]}_f16_wmma", kgroups, lanes)

def q8_linear(layer:Linear, x:Tensor) -> Tensor:
  assert layer.ggml_type in QUANT_SIZES
  # a row parallel layer fed a replicated input takes the slice of its device
  if layer.shard_axis == 1 and isinstance(x.device, tuple) and x.uop.axis is None:
    x = Tensor(x.uop._shard(x.ndim-1, rng:=UOp.range(len(x.device), 0, AxisType.DEVICE)).unshard(x.ndim-1, rng))
  tokens = int(x.numel()) // layer.in_features
  out_features, in_features = layer.out_features, int(x.uop.shard_shape[-1])
  splits = layer.in_features // in_features
  assert (layer.shard_axis == 1) == (splits > 1), f"input features split over {splits} devices, weight split on axis {layer.shard_axis}"
  out_shape:tuple[int, ...] = (splits*tokens, out_features)
  fxn:Callable[..., UOp]
  extra = (_iq_grid(x.device, layer.ggml_type),) if layer.ggml_type in (IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S) else ()
  # the kernels write the output features of their own device
  local_out = out_features // len(dev) if isinstance(dev:=layer.weight.device, tuple) and layer.shard_axis == 0 else out_features
  cdna = _is_cdna(x.device)
  if tokens % 16 == 0 and local_out % 16 == 0:
    if layer.ggml_type == IQ4_XS:
      fxn, extra = _iq4_linear_f16_wmma_kernel, (iq4_half_lut(x.device),)
    else:
      fxn = functools.partial(_q5_linear_f16_wmma_kernel if layer.ggml_type in (Q4_K, Q5_K) else _quant_linear_f16_wmma_kernel,
                              ggml_type=layer.ggml_type)
    fxn = functools.partial(fxn, kgroups=4 if cdna else 2 if _is_rdna4(x.device) else 1, lanes=64 if cdna else 32)
    srcs = (x.cast(dtypes.float16).contiguous(), *extra)
  else:
    fxn = functools.partial(_quant_decode_kernel, ggml_type=layer.ggml_type, cdna=cdna)
    srcs = (*q8_quantize(x, tokens, in_features), *_iq_tables(x.device, layer.ggml_type))
  out = Tensor.empty(out_shape, dtype=dtypes.float32, device=x.device, axis=None if layer.shard_axis is None else 1-layer.shard_axis)
  result = Tensor.custom_kernel(out, layer.weight, *srcs, fxn=functools.partial(fxn, out_features=local_out, in_features=in_features))[0]
  # row parallel: every device wrote the partial sum of its input features, the sum over the devices is the allreduce
  if splits > 1: result = partial_sum(result.reshape(splits, tokens, out_features))
  result = result.reshape(*x.shape[:-1], out_features)
  return result if layer.bias is None else result + layer.bias

def expert_linear(pool:Tensor, slots:Tensor, x:Tensor, out_features:int, in_features:int, ggml_type:int, stride:int, offset:int=0,
                  tokens:int=0, w:Tensor|None=None, bias:Tensor|None=None, summed:bool=True, shared:bool=False) -> Tensor:
  """gemv of the quantized experts in the slots of a cache pool: out[k] = W[slots[k]] @ x[k % rows] (x rows rows). with w (slots of
  weight 0, a token's last, are padding: not computed) their sum unless not summed: out[0] = sum_k w[k] * W[slots[k]] @ x[k], with tokens a
  sum a token: out[t] = sum_s w[s*tokens+t] * W[slots[s*tokens+t]] @ x[s*tokens+t]. bias (out's shape) is added to the outputs.
  shared: a slot for all the tokens (w and the rows of x and out s*tokens+t, slots s): its weights read once, slots of weights all 0 skipped
  pool holds raw gguf blocks as words (uint16 for halfword formats, else uint32), a slot starts every stride words, W at offset in it"""
  rows = int(x.numel()) // in_features
  extra = _iq_tables(x.device, ggml_type)
  summed = summed and w is not None
  assert not tokens or summed or shared, "tokens split the weighted sum"
  out = Tensor.empty(max(tokens, 1) if summed else slots.shape[0] * (tokens if shared else 1), out_features, dtype=dtypes.float32, device=x.device)
  fxn = functools.partial(_quant_decode_kernel, ggml_type=ggml_type, out_features=out_features, in_features=in_features, gather=(stride, offset),
                          tokens=tokens, weighted=w is not None, summed=summed, bias=bias is not None, shared=shared)
  return Tensor.custom_kernel(out, pool, *q8_quantize(x, rows, in_features), slots, *[t for t in (w, bias) if t is not None], *extra, fxn=fxn)[0]

# ******** grouped expert gemm: the prefill's routed experts, WMMA ********

@functools.cache
def _iq3_half_tables(device:str|tuple[str, ...]|None) -> Tensor:
  """the grouped gemm's IQ3_XXS lookup tables, 1024 words: the grid as halves 1024 + magnitude (2 words an entry, bytes 0-1 then 2-3), then
  for every 7 bit sign pattern (the 8th its parity) the sign bits of its two words' halves (4 words)"""
  from tinygrad.runtime.autogen.ggml_common import iq3xxs_grid
  def halves(v:int) -> int: return (0x6400 | (v & 255)) | (0x6400 | (v >> 8 & 255)) << 16
  grid = [halves(g >> (16*i) & 0xffff) for g in iq3xxs_grid for i in range(2)]
  even = [p | (bin(p).count("1") & 1) << 7 for p in range(128)]
  signs = [(e >> (2*i) & 1) << 15 | (e >> (2*i+1) & 1) << 31 for e in even for i in range(4)]
  with Context(ALLOW_DEVICE_USAGE=1): return Tensor.const(tuple(grid + signs), dtypes.uint32).to(device, force=True).realize()

def _pk_fma_f16(a:UOp, b:UOp, c:UOp) -> UOp:  # a*b+c of the half pairs in words, one rounding
  return UOp(Ops.CUSTOM, src=(a, b, c), arg=("({{ typedef _Float16 __h2 __attribute__((ext_vector_type(2))); __builtin_bit_cast(unsigned int, "
    "__builtin_elementwise_fma(__builtin_bit_cast(__h2, {0}), __builtin_bit_cast(__h2, {1}), __builtin_bit_cast(__h2, {2}))); }})", dtypes.uint32))

@functools.cache
def _expert_gemm_kernel(out:UOp, raw:UOp, x:UOp, tslot:UOp, ntiles:UOp, *rest:UOp, out_features:int, in_features:int, stride:int,
                        offset:int, gather:bool, rdna4:bool, G:int=4, OT:int=2, W:int=4, QH:int=8, SPLIT:int=1,
                        ORDER:int=1) -> UOp:
  """out[p] = W[tslot[p//16]] @ x[rows[p]] (rows = rest[0] if gather, else p) for the tiles p//16 < ntiles[0]. a workgroup a chunk of up to
  G tiles of a slot (it starts at a tile i of a slot's run or of i % G 0, its tiles to the next of those: the others' workgroups exit) and
  its W waves' OT output tiles each, a loop over the weights' 256 blocks: a weight fragment dequantized once for the chunk's tiles (QH k
  steps of them at a time in registers, the WMMA of a tile past the chunk's skipped: a loop of 0 or 1 rounds). the IQ3_XXS weights to
  half fragments by tables in LDS: a grid word's halves 1024 + magnitude, fma'd by the scale less 1024 times it (exact: the product
  rounded once), the signs xor'd in. the k slots are permuted (the same in x): k step q of a block is its grid word pairs q and 16+q (8
  neighboring input features each, the same signs), on RDNA4 a half wave each: a lane's grid indices, scales and x of a block are neighbors"""
  rows, tables = (rest[0], rest[1]) if gather else (None, rest[0])
  P, nblocks, bw = int(out.shape[0]), in_features // 256, QUANT_SIZES[IQ3_XXS]//2
  while out_features % (16*OT*W): W //= 2
  G = min(G, P//16)
  assert W and in_features % 256 == 0 and P % 16 == 0 and 16 % QH == 0
  # (64 bit offsets only for a pool past 2**31 words)
  itype = dtypes.int64 if int(raw.numel()) >= 2**31 else dtypes.int32
  # (ORDER 0: the tiles the fastest, a slot's chunks at once: its weights read from memory once; 1: the output blocks, a tile's x once)
  oblock, tile = UOp.range(out_features//(16*OT*W), ORDER, AxisType.GLOBAL), UOp.range(P//16, 1-ORDER, AxisType.GLOBAL)
  wave, lane = UOp.range(W, 2, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  col, h = lane % 16, lane // 16
  nt, slot = ntiles[0].load(), tslot[tile].load()
  start = (tile % G).eq(0) | slot.ne(tslot[(tile - 1).maximum(0)].load()) if G > 1 else UOp.const(True, dtypes.bool)
  lives = [start & (tile < nt)]
  for j in range(1, G):
    tj = (tile + j).minimum(P//16 - 1)
    lives.append(lives[-1] & ((tile % G) + j < G) & (tile + j < nt) & tslot[tj].load().eq(slot))
  lut = UOp.alloc((1024,), dtypes.uint32, addrspace=AddrSpace.LOCAL)
  tid, threads = wave*WARP_SIZE + lane, W*WARP_SIZE
  lut = lut.after(UOp.group(*(lut[tid+i*threads].store(tables[tid+i*threads]) for i in range(1024//threads))).barrier())
  slot_base = slot.cast(itype)*stride + offset
  tiles = [(tile + j).minimum(P//16 - 1) for j in range(G)]
  xrows = [rows[t*16 + col].load() if rows is not None else t*16 + col for t in tiles]
  outputs = [(oblock*W + wave)*(16*OT) + t*16 + col for t in range(OT)]
  accs = [[_reg((8,), 0) for _ in range(OT)] for _ in range(G)]
  blk = UOp.range(lives[0].where(nblocks, 0), 3, AxisType.LOOP)
  vals = [acc.after(blk) for acc in accs[0]]
  curs = [[acc.after(blk) for acc in accs_j] for accs_j in accs[1:]]
  ends = []
  # a lane's grid word pairs of k step q: (pair, its index among the lane's subgroups' scales)
  lane_pairs = [[(h*16 + q, q//4)] if rdna4 else [(q, q//4), (16+q, 4 + q//4)] for q in range(16)]
  subs = [h*4 + i for i in range(4)] if rdna4 else list(range(8))
  xbases = [xr.cast(itype)*in_features + blk*256 for xr in xrows]
  wq = []
  for o in outputs:
    base = slot_base + (o*nblocks + blk).cast(itype)*bw
    d = _half(raw[base])
    idx = {p: raw[base + 1 + p].cast(dtypes.uint32) for q in range(16) for p, _ in lane_pairs[q]}
    auxs, scales = [], []
    for sub in subs:
      aux = _load_u32(raw, base, 66 + sub*4)
      s16 = (d * ((aux >> 28).float()*0.5 + 0.25)).cast(dtypes.half)
      s2 = s16.bitcast(dtypes.uint16).cast(dtypes.uint32) * 0x10001
      auxs.append(aux)
      scales.append((s2, (s16 * -1024).bitcast(dtypes.uint16).cast(dtypes.uint32) * 0x10001))
    wq.append((idx, auxs, scales))
  def bfrag(q:int, t:int) -> UOp:
    idx, auxs, scales = wq[t]
    words = []
    for p, si in lane_pairs[q]:
      s2, o2 = scales[si]
      sign = ((auxs[si] >> (7*(q % 4))) & 127) * 4 + 512
      for w, gi in enumerate((idx[p] & 255, idx[p] >> 8)):
        for hw in range(2):
          words.append(_pk_fma_f16(lut[(gi*2 + hw).cast(dtypes.weakint)].load(), s2, o2) ^ lut[(sign + w*2 + hw).cast(dtypes.weakint)].load())
    return UOp.stack(*((w >> (16*i)).cast(dtypes.uint16).bitcast(dtypes.half) for w in words for i in range(2)))
  def afrag(j:int, q:int) -> UOp: return UOp.stack(*(e for p, _ in lane_pairs[q] for e in _vec_halves(x, xbases[j] + p*8)))
  zero = UOp.stack(*[UOp.const(0, dtypes.float)]*8)
  def chains(start:list[UOp], j:int, qs:range, bf:dict) -> list[UOp]:
    # (SPLIT chains of WMMAs an accumulator, k step q on chain q % SPLIT, added at the end: independent WMMAs in flight)
    cs = [[v] + [zero]*(SPLIT-1) for v in start]
    for q in qs:
      a = afrag(j, q)
      for t in range(OT): cs[t][q % SPLIT] = UOp.wmma(a, bf[q, t], cs[t][q % SPLIT], *WMMA_ARG)
    return [functools.reduce(UOp.__add__, c) for c in cs]
  for qh in range(16 // QH):
    qs = range(qh*QH, (qh+1)*QH)
    bf = {(q, t): bfrag(q, t) for q in qs for t in range(OT)}
    vals = chains(vals, 0, qs, bf)
    for j in range(1, G):  # (the chunk's other tiles: a round if live)
      r = UOp.range(lives[j].where(1, 0), 10 + qh*G + j, AxisType.LOOP)
      cv = chains([cur.after(r) for cur in curs[j-1]], j, qs, bf)
      e = UOp.group(*(cur.store(v) for cur, v in zip(curs[j-1], cv))).end(r)
      curs[j-1] = [acc.after(e) for acc in accs[j]]
      ends.append(e)
  update = UOp.group(*(acc.store(v) for acc, v in zip(accs[0], vals)), *ends).end(blk)
  stores = []
  for j in range(G):
    tokens = (tuple((tiles[j]*16 + h*8 + i).valid(lives[j]) for i in range(8)),)
    stores += _wmma_stores(out, outputs, tokens, [[acc] for acc in accs[j]], update, lane, wave, W, 2 if rdna4 else 1)
  return UOp.group(*stores).end(oblock, tile, wave, lane).sink(arg=KernelInfo(name="expert_gemm_iq3_xxs", opts_to_apply=()))

def _vec_halves(x:UOp, idx:UOp) -> list[UOp]:  # 8 neighboring halves, one load
  v = _amd_load(x[idx], 8)
  return [v[i] for i in range(8)]

EG_OPTS = {"G": 4, "OT": 2, "W": 4, "QH": 8, "SPLIT": 1, "ORDER": 1}  # (the gemm's knobs, EG_<knob> to override)

def expert_gemm(pool:Tensor, x:Tensor, tslot:Tensor, ntiles:Tensor, out_features:int, in_features:int, stride:int, offset:int=0,
                rows:Tensor|None=None, dtype=dtypes.float32) -> Tensor:
  """the grouped gemm of the routed experts' IQ3_XXS weights in the slots of a cache pool: out[p] = W[tslot[p//16]] @ x[rows[p]] (rows
  None: x[p]), (P, out_features) of dtype (float or half; the sums float), for the 16 row tiles p//16 < ntiles[0] (P static: the rows of
  the tiles past them not written). pool holds raw gguf blocks as uint16 words, a slot starts every stride words, W at offset in it
  (expert_linear's). x is cast to half"""
  P = int(rows.shape[0]) if rows is not None else int(x.shape[0])
  out = Tensor.empty(P, out_features, dtype=dtype, device=x.device)
  fxn = functools.partial(_expert_gemm_kernel, out_features=out_features, in_features=in_features, stride=stride, offset=offset,
                          gather=rows is not None, rdna4=_is_rdna4(x.device), **{k: getenv(f"EG_{k}", v) for k, v in EG_OPTS.items()})
  srcs = (pool, x.cast(dtypes.float16).reshape(-1).contiguous(), tslot, ntiles, *([rows] if rows is not None else []), _iq3_half_tables(x.device))
  return Tensor.custom_kernel(out, *srcs, fxn=fxn)[0]

@functools.cache
def _expert_combine_kernel(out:UOp, y:UOp, dest:UOp, w:UOp, K:int) -> UOp:
  # a workgroup a token, a thread 4 neighboring columns of every 512: the sum of its K rows of y in order (deterministic)
  T, D = cast(tuple[int, int], out.shape)
  P = int(y.shape[0])
  assert D % 4 == 0
  tok, tid = UOp.range(T, 0, AxisType.GLOBAL), UOp.range(128, 1, AxisType.LOCAL)
  ents = [(dest[tok*K + k].load().cast(dtypes.int64)*D, w[tok*K + k].load()) for k in range(K)]
  stores = []
  for c in range(-(-D // 512)):
    col = (c*512 + tid*4)
    ok = col < D if D % 512 else None
    colc = col.minimum(D - 4) if ok is not None else col
    acc = [UOp.const(0, dtypes.float)]*4
    for base, wk in ents:
      v = _amd_load(y.reshape(P*D)[base + colc], 4)
      acc = [a + wk*v[i].float() for i, a in enumerate(acc)]
    stores += [out[tok, (colc + i) if ok is None else (colc + i).valid(ok)].store(a) for i, a in enumerate(acc)]
  return UOp.group(*stores).end(tok, tid).sink(arg=KernelInfo(name="expert_combine", opts_to_apply=()))

def expert_combine(y:Tensor, dest:Tensor, w:Tensor, T:int, K:int) -> Tensor:
  """the routed experts' weighted sum of a token: out[t] = sum over k (in order) of w[t*K+k] * y[dest[t*K+k]], y (P, D) float or half (the
  down gemm's output in pair order), dest int32 and w float (T*K,): (T, D) float. a gather of K rows a token (not a one-hot)"""
  out = Tensor.empty(T, int(y.shape[1]), dtype=dtypes.float32, device=y.device)
  return Tensor.custom_kernel(out, y, dest, w.float(), fxn=functools.partial(_expert_combine_kernel, K=K))[0]

# ******** mla absorb matmuls, a few tokens ********

@functools.cache
def _mla_bmm_kernel(out:UOp, x:UOp, w:UOp) -> UOp:
  # out[h, t, o] = sum over i of x[h, t, i] * w[h, o, i] in float: a wave a row (h, o) of w, a lane 4 neighboring i of every 128, the tokens
  # on the row's loads
  H, T, I = cast(tuple[int, int, int], x.shape)
  O = cast(int, w.shape[1])
  assert I % (WARP_SIZE*4) == 0 and H*O % 4 == 0
  row, wave, lane = UOp.range(H*O // 4, 0, AxisType.GLOBAL), UOp.range(4, 1, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  h, o = (row*4 + wave) // O, (row*4 + wave) % O
  idx = [k*WARP_SIZE*4 + lane*4 + j for k in range(I // (WARP_SIZE*4)) for j in range(4)]
  wv = [w[h, o, i].load().float() for i in idx]
  return UOp.group(*[out[h, t, o.valid(lane.eq(0))].store(warp_reduce(sum((a*x[h, t, i].load().float() for a, i in zip(wv, idx)),
    UOp.const(0, dtypes.float)), full_wave=True)) for t in range(T)]).end(row, wave, lane).sink(arg=KernelInfo(name="mla_bmm", opts_to_apply=()))

def mla_bmm(x:Tensor, w:Tensor) -> Tensor:
  """(1, H, T, I) @ (H, O, I)^T -> (1, H, T, O) in float: the absorb matmuls of mla attention for a few tokens"""
  _, H, T, I = x.shape
  axis = x.uop.axis if isinstance(x.device, tuple) else None
  out = Tensor.empty(int(H), int(T), int(w.shape[1]), dtype=dtypes.float32, device=x.device, axis=None if axis is None else 0)
  return Tensor.custom_kernel(out, x.reshape(H, T, I).contiguous(), w, fxn=_mla_bmm_kernel)[0].reshape(1, H, T, w.shape[1])

# ******** small allreduces: the devices write to each other's memory ********

# every device has a mailbox in its memory (uncached), mapped into every other device: a device writes its partial of an element to the
# mailbox of the element's owner, the owner polls its own mailbox for the partials of every device, sums them and writes the sums to the
# mailboxes of all the devices, they poll their own. no reads over the links (posted writes only), no host memory

class P2PAllreduce:
  # every device has a mailbox in its memory, mapped into every other device (cached: a writer's l2 combines its stores into lines, written
  # back by a system release). a workgroup has a region of it: two parities of a slot per sender and of the sums, P2P_AR_MAXW words each (a
  # value and its epoch in a word: no torn reads, no flags), its own epoch counter. the regions never move with the size of the allreduce: a
  # word another one left can't have an epoch this one waits for
  def __init__(self, devs:tuple[str, ...]):
    from tinygrad.device import Buffer, BufferSpec, BufferStorage, MultiBuffer
    N = len(devs)
    nbytes = P2P_AR_MAXG * 2*(N+1)*P2P_AR_MAXW * 8
    with Context(ALLOW_DEVICE_USAGE=1):
      # (zeroed from host memory: no kernel to compile. a word the last process left could have an epoch this one waits for)
      own = [Buffer(d, nbytes, initial_value=bytes(nbytes)) for d in devs]
      def alias(b:Buffer, d:str) -> Buffer:
        return Buffer(d, nbytes, opaque=BufferStorage(va:=b.get_buf(d)), options=BufferSpec(external_ptr=va)).ensure_allocated()
      # mailbox k of device d is the one of device (d+k)%N (0: its own)
      self.mboxes = []
      for k in range(N):
        (mb:=MultiBuffer.__new__(MultiBuffer)).bufs = [own[d] if k == 0 else alias(own[(d+k) % N], devs[d]) for d in range(N)]
        self.mboxes.append(Tensor(UOp.from_buffer(mb, dtypes.uint64, devs)))
      # the epoch counter of every workgroup, then the device's index
      counters = [Buffer(d, (P2P_AR_MAXG+1)*4, initial_value=bytes(P2P_AR_MAXG*4)+i.to_bytes(4, "little")) for i,d in enumerate(devs)]
      (cb:=MultiBuffer.__new__(MultiBuffer)).bufs = counters
      self.counter = Tensor(UOp.from_buffer(cb, dtypes.int32, devs))
      for d in devs: Device[d].synchronize()

P2P_AR_MAXG, P2P_AR_MAXW = 64, 1024  # workgroups, the words of a workgroup's slot
P2P_AR_TIMEOUT_MS, P2P_AR_TWO_SHOT = 50, 4096  # a poll's deadline, the allreduces in two shots from these many values

def p2p_allreduce(devs:tuple[str, ...]) -> P2PAllreduce|None: return _p2p_allreduce(tuple(Device.canonicalize(d) for d in devs))
@functools.cache
def _p2p_allreduce(devs:tuple[str, ...]) -> P2PAllreduce|None:
  if len(devs) < 2 or not amd_custom_kernels_supported(devs): return None
  return P2PAllreduce(devs)

def p2p_ar_shape(n:int) -> tuple[int, int]|None:
  """(threads, values a thread) of a p2p allreduce of n values: a workgroup's at most P2P_AR_MAXW, at most P2P_AR_MAXG of them"""
  # (measured: small ones in many narrow workgroups, larger ones in up to 24 of 256 threads)
  for T, EP in (((128, 1),) if n <= 2048 else ()) + ((256, 1), (256, 2), (256, 4)):
    if n % (T*EP) == 0 and n // (T*EP) <= (P2P_AR_MAXG if n <= 2048 else 24): return T, EP
  for T, EP in ((256, 4), (128, 4), (64, 4)):
    if n % (T*EP) == 0 and n // (T*EP) <= P2P_AR_MAXG: return T, EP
  return None

def _sys(fmt:str, dtype:DType, *src:UOp) -> UOp: return UOp(Ops.CUSTOM, src=src, arg=(fmt, dtype))
_ATOMIC_LOAD = "__hip_atomic_load({0}, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)"

@functools.cache
def _p2p_allreduce_kernel(out:UOp, x:UOp, counter:UOp, *mb:UOp, N:int, n:int, T:int, EP:int, two_shot:bool=False, gather:bool=False) -> UOp:
  # one shot: every device stores its values into its slot in the mailboxes of all the others and polls its own mailbox for theirs (gather:
  # out (N, n) is all of them). two shot: workgroup g's values go to device g%N (its owner) only, the owner sums them and stores the sums
  # into the mailboxes of the others. the sums add the devices' values in device order on every device: the same bits everywhere.
  # stores: plain (the l2 combines them into lines), a fence (in the l2), the workgroup, a system release (the l2 written back to the others).
  # polls: system scope loads of the device's own mailbox (not its l2), every word of a thread in every iteration. a poll gives up at a
  # deadline (P2P_AR_TIMEOUT_MS): the words it didn't get are NaN (the sums too, loudly), not a device that never finishes
  G = n // (T*EP)
  assert n % (T*EP) == 0 and G <= P2P_AR_MAXG and T*EP <= P2P_AR_MAXW, f"p2p allreduce of {n} in {G} workgroups of {T}x{EP}: not in the mailbox"
  wg, t = UOp.range(G, 0, AxisType.GLOBAL), UOp.range(T, 1, AxisType.LOCAL)
  li = [t*EP + i for i in range(EP)]
  els = [wg*T*EP + l for l in li]
  epoch, me = (counter[wg].load() + 1).cast(dtypes.uint32), counter[P2P_AR_MAXG].load().cast(dtypes.weakint)
  deadline = _sys("__builtin_readsteadycounter()", dtypes.uint64, epoch) + P2P_AR_TIMEOUT_MS * 100000  # (100 MHz)
  par = (epoch & 1).cast(dtypes.weakint)
  def reg(s:int|UOp, l:UOp) -> UOp: return ((wg*2 + par)*(N+1) + s)*P2P_AR_MAXW + l
  vals = [x.reshape(-1)[e].load().float() for e in els]  # (a device's n)
  def word(v:UOp) -> UOp: return (epoch.cast(dtypes.uint64) << 32) | v.bitcast(dtypes.uint32).cast(dtypes.uint64)
  def value(w:UOp) -> UOp:  # (of an older epoch: past the deadline, NaN)
    return (w >> 32).cast(dtypes.uint32).eq(epoch).where((w & 0xFFFFFFFF).cast(dtypes.uint32).bitcast(dtypes.float32), float("nan"))
  def push(stores:list[UOp], after:UOp|None=None) -> UOp:
    # the stores in the l2, the workgroup, the l2 written back to the others
    fenced = _sys('__builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");', dtypes.void, *stores)
    bar = UOp.group(fenced, *([] if after is None else [after])).barrier()
    # (one wave writes the l2 back: a loop that runs once in it, not at all in the others)
    once = UOp.range((t < 32).where(1, 0), next(UOp.unique_num), AxisType.LOOP)
    return UOp.group(UOp.group(_sys('__builtin_amdgcn_fence(__ATOMIC_RELEASE, "");', dtypes.void, once, bar)).end(once)).barrier()
  def poll(after:UOp, slots:list[UOp], waits:list[UOp]) -> list[UOp]:
    # the words of the slots (system scope loads of this device's mailbox, all of them in an iteration) until those it waits for have the epoch
    loop = UOp.range(UOp(Ops.NOOP).after(after), next(UOp.unique_num), dtype=dtypes.void)
    words = UOp.alloc((len(slots),), dtypes.uint64, addrspace=AddrSpace.REG)
    loaded = words.after(loop).store(UOp.stack(*(_sys(_ATOMIC_LOAD, dtypes.uint64, mb[0].after(loop)[s]) for s in slots)))
    got = [words.after(loaded)[i].load() for i in range(len(slots))]
    stale = functools.reduce(lambda a,b: a | b, [w & (g >> 32).cast(dtypes.uint32).ne(epoch) for g, w in zip(got, waits)])
    waiting = stale & (_sys("__builtin_readsteadycounter()", dtypes.uint64, stale) < deadline)
    words = words.after(UOp(Ops.CUSTOM, src=(waiting,), arg=("", dtypes.void)).backedge(loop, waiting))
    return [words[i].load() for i in range(len(slots))]
  def ordered_sum(parts:list[UOp]) -> UOp: return functools.reduce(lambda a,b: a+b, parts)
  if not two_shot:
    sent = push([mb[k][reg(me, l)].store(word(v)) for k in range(1, N) for l, v in zip(li, vals)])
    got = poll(sent, [reg(s, l) for s in range(N) for l in li], [me.ne(s) for s in range(N) for _ in li])
    parts = [[me.eq(s).where(v, value(got[s*EP + i])) for s in range(N)] for i, v in enumerate(vals)]
    if gather: stores = [out[s*n + e].store(p) for e, ps in zip(els, parts) for s, p in enumerate(ps)]
    else: stores = [out[e].store(ordered_sum(ps)) for e, ps in zip(els, parts)]
  else:
    owner = (wg % N).cast(dtypes.weakint)
    k_owner, is_owner = (owner - me + N) % N, owner.eq(me)
    sent = push([mb[k][reg(me, l).valid(k_owner.eq(k))].store(word(v)) for k in range(1, N) for l, v in zip(li, vals)])
    got = poll(sent, [reg(s, l) for s in range(N) for l in li], [is_owner & me.ne(s) for s in range(N) for _ in li])
    sums = [ordered_sum([me.eq(s).where(v, value(got[s*EP + i])) for s in range(N)]) for i, v in enumerate(vals)]
    published = push([mb[k][reg(N, l).valid(is_owner)].store(word(sm)) for k in range(1, N) for l, sm in zip(li, sums)], sent)
    got2 = poll(published, [reg(N, l) for l in li], [is_owner.logical_not() for _ in li])
    stores = [out[e].store(is_owner.where(sm, value(g))) for e, sm, g in zip(els, sums, got2)]
  # the next epoch, once every thread of the workgroup read the counter
  done = counter.after(UOp.group(*stores).barrier())[wg.valid(t.eq(0))].store(epoch.cast(dtypes.int32))
  return UOp.group(*stores, done).end(wg, t).sink(arg=KernelInfo(name="p2p_allgather" if gather else "p2p_allreduce", opts_to_apply=()))


def partial_sum(x:Tensor) -> Tensor:
  """the sum over the leading device axis of x (devices, ...), each device its partial: an allreduce. small ones are a kernel in which the
  devices write their partials to each other's memory (P2PAllreduce, from tinygrad PR 18863), the rest the copies of the tensor sum"""
  n = prod(x.shape[1:])
  if isinstance(x.device, tuple) and isinstance(n, int) and (shape:=p2p_ar_shape(n)) is not None and (p2p:=p2p_allreduce(x.device)):
    # (two shot from P2P_AR_TWO_SHOT values: fewer bytes over the links)
    out = Tensor.empty(n, dtype=dtypes.float32, device=x.device)
    fxn = functools.partial(_p2p_allreduce_kernel, N=len(x.device), n=n, T=shape[0], EP=shape[1], two_shot=n >= P2P_AR_TWO_SHOT)
    return Tensor.custom_kernel(out, x.float().reshape(x.shape[0], n).contiguous(), p2p.counter, *p2p.mboxes, fxn=fxn)[0].reshape(x.shape[1:])
  return x.sum(0)

def all_gather_p2p(x:Tensor) -> Tensor|None:
  """x split over the devices on its last axis, whole on every device: the p2p kernel in its gather mode (None: not supported)"""
  devs, N = cast(tuple[str, ...], x.device), len(cast(tuple, x.device))
  n = prod(x.shape) // N
  if not isinstance(n, int) or x.uop.axis != x.ndim - 1 or (shape:=p2p_ar_shape(n)) is None or (p2p:=p2p_allreduce(devs)) is None: return None
  out = Tensor.empty(N*n, dtype=dtypes.float32, device=devs)
  fxn = functools.partial(_p2p_allreduce_kernel, N=N, n=n, T=shape[0], EP=shape[1], gather=True)
  # (a device's part its slice of every token, out the parts in device order)
  lead, part = int(prod(x.shape[:-1])), x.shape[-1] // N
  return Tensor.custom_kernel(out, x.float(), p2p.counter, *p2p.mboxes, fxn=fxn)[0].reshape(N, lead, part).permute(1, 0, 2).reshape(x.shape)

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

MLA_KEYS, MLA_PARTIALS, MLA_GROUPS = 32, 64, 4  # keys a round (two 16 key tiles), the most rounds a step's partials, waves a row tile

def _swizzle_xor(x:UOp, mask:int) -> UOp:  # x of the lane lane ^ mask (a float)
  return UOp(Ops.CUSTOM, src=(x,), arg=(f"__builtin_bit_cast(float, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, {{0}}), {0x1f | mask<<10}))",
                                       dtypes.float))

@functools.cache
def _mla_decode_partial(out:UOp, stats:UOp, q:UOp, cache:UOp, *mask:UOp, valid:int|UOp, dv:int, scale:float, tokens:int,
                        rdna4:bool) -> UOp:
  """mla attention of a few tokens on the latent cache (mqa: one kv head, the values the first dv of its dims), flash decoding: a workgroup
  every MLA_PARTIALS-th round of MLA_KEYS keys, for all of the rows (q (1, rows, D) half: head h token t at h*tokens + t), the round's keys
  read once into LDS for them all. WMMA: the scores keys x rows (a wave a 16 key tile and a 16 row tile, the keys in LDS, q from memory),
  through LDS to the online softmax, then the values transposed (dv x keys, strided LDS loads) times the weights (keys x rows), a wave a
  row tile and its share of the dv tiles. out (1, rows, partials, dv) the unnormalized sums, stats their max and sum
  (_amd_flash_decode_combine's). the mask (1, 1, T, keys) added, else causal"""
  _, M, D = cast(tuple[int, int, int], q.shape)
  N, valid = int(cache.shape[2]), _unbind(valid)
  # a workgroup's row tiles (at most 3: the rows past them another workgroup's, with the keys again), key tiles, k steps of the scores, dv tiles
  RB = -(-M // 48)
  RT, KT, KS, DT = -(-M // (16*RB)), MLA_KEYS // 16, D // 16, dv // 16
  W, threads = RT*MLA_GROUPS, RT*MLA_GROUPS*WARP_SIZE
  DW, SP, E = DT // MLA_GROUPS, MLA_GROUPS // KT, MLA_KEYS*D // threads  # (a wave's dv tiles, the scores' k splits, a thread's round halves)
  assert D % (16*SP) == 0 and dv % (16*MLA_GROUPS) == 0 and dv <= D and D % E == 0 and MLA_GROUPS % KT == 0
  total = (valid + MLA_KEYS - 1) // MLA_KEYS
  live = min(total, MLA_PARTIALS) if isinstance(total, int) else total.minimum(MLA_PARTIALS)
  rb, block = UOp.range(RB, 2, AxisType.GLOBAL), UOp.range(live, 0, AxisType.GLOBAL)
  wave, lane = UOp.range(W, 1, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  tid, col = wave*WARP_SIZE + lane, lane % 16
  def acc_row(i:int) -> UOp: return (lane//16)*8 + i if rdna4 else lane//16 + 2*i  # (the M row of a lane's i-th accumulator)
  def frag(view:UOp) -> UOp: return _wmma_fragment(view, lane, rdna4)
  neg = UOp.const(-1e30, dtypes.float)
  rnd = UOp.range((total - 1 - block) // MLA_PARTIALS + 1, 4, AxisType.LOOP)
  first = (block + rnd*MLA_PARTIALS)*MLA_KEYS
  # the round's keys into LDS (each thread its E halves of a key; keys past the valid ones are the last valid one's: masked)
  K = UOp.alloc((MLA_KEYS, D), dtypes.half, addrspace=AddrSpace.LOCAL)
  e = UOp.range(E, 5)
  key = (first + tid*E // D).minimum(valid - 1)
  fill = K.after(rnd).reshape(threads, E)[tid, e].store(cache.reshape(N*D)[key*D + tid*E % D + e]).end(e).barrier()
  K = K.after(fill)
  # the scores (keys x rows) of the wave's key tile kt and row tile rt over its k split sp, into LDS (the splits' sums: rows x keys, float)
  kt, rt, sp = wave // RT % KT, wave % RT, wave // (RT*KT)
  S = _reg((8,), 0, rnd)
  ksr = UOp.range(KS // SP, 7, AxisType.LOOP)
  ks = ksr + sp*(KS // SP)
  lrow = rt*16 + col  # (the lane's row in the workgroup's, and in all)
  row = rb*RT*16 + lrow
  qrow = row.minimum(M - 1)
  sdone = S.store(UOp.wmma(frag(K.reshape(MLA_KEYS, KS, 16)[kt*16 + col, ks]), frag(q.reshape(M, KS, 16)[qrow, ks]), S.after(ksr),
                           *WMMA_ARG)).end(ksr)
  S = S.after(sdone)
  SL = UOp.alloc((SP, RT*16, MLA_KEYS + 1), dtypes.float, addrspace=AddrSpace.LOCAL)
  t = (row % tokens) if tokens > 1 else UOp.const(0, dtypes.int)
  SL = SL.after(UOp.group(*[SL.after(rnd)[sp, lrow, kt*16 + acc_row(i)].store(S[i].load()) for i in range(8)]).barrier())
  def score(j:int) -> UOp:  # (the lane's row and key j, masked and scaled)
    k, v = first + j, functools.reduce(UOp.__add__, [SL[x, lrow, j].load() for x in range(SP)])
    if mask: return (k < valid).where(v*scale + mask[0][0, 0, t, k.minimum(N - 1)].load().float(), neg)
    return (k < valid - (tokens - 1) + t).where(v*scale, neg)
  # the online softmax of the wave's row tile: a lane a row (as the values' accumulators: a lane a row), half of the round's keys
  c = wave // RT
  half = lane // 16
  sv = [score(half*(MLA_KEYS//2) + j) for j in range(MLA_KEYS//2)]
  acc, mx, sm = _reg((DW*8,), 0), _reg((1,), -1e30), _reg((1,), 0)
  pacc, pmx, psm = acc.after(rnd), mx.after(rnd), sm.after(rnd)
  rmax = functools.reduce(UOp.maximum, sv, pmx[0].load())
  rmax = rmax.maximum(_swizzle_xor(rmax, 16))
  alpha, p = ((pmx[0].load() - rmax) * LOG2E).exp2(), [((v - rmax) * LOG2E).exp2() for v in sv]
  psum = functools.reduce(UOp.__add__, p)
  psum = psum + _swizzle_xor(psum, 16)
  PL = UOp.alloc((RT*16, MLA_KEYS), dtypes.half, addrspace=AddrSpace.LOCAL)
  PL = PL.after(UOp.group(*[PL.after(rnd)[lrow.valid(c.eq(0)), half*(MLA_KEYS//2) + j].store(v.cast(dtypes.half)) for j, v in enumerate(p)])
                .barrier())
  # the values: dv x keys (strided loads of the keys' first dv) times the weights keys x rows, a wave its row tile's dv tiles c*DW..+DW, onto
  # the sums so far scaled by alpha
  cur = acc.after(UOp.group(*[acc[i].store(alpha*pacc[i].load()) for i in range(DW*8)]))
  for kk in range(KT):
    pf = frag(PL.reshape(RT*16, KT, 16)[lrow, kk])
    cur = acc.after(UOp.group(*[cur.reshape(DW, 8)[d].store(UOp.wmma(frag(K.reshape(KT, 16, D).permute(0, 2, 1)[kk, (c*DW + d)*16 + col]), pf,
                                                                     cur.reshape(DW, 8)[d], *WMMA_ARG)) for d in range(DW)]))
  upd = UOp.group(cur, mx[0].store(rmax), sm[0].store(alpha*psm[0].load() + psum))
  acc, mx, sm = (x.after(upd.barrier().end(rnd)) for x in (acc, mx, sm))
  ok = row < M
  stores = [out[0, row.valid(ok), block, (c*DW + dt)*16 + acc_row(i)].store(acc[dt*8 + i].load()) for dt in range(DW) for i in range(8)]
  stores += [stats[0, row.valid(ok & c.eq(0) & half.eq(0)), block, k].store(x[0].load()) for k, x in enumerate((mx, sm))]
  return UOp.group(*stores).end(lane, wave, block, rb).sink(arg=KernelInfo(name="mla_decode_partial", opts_to_apply=()))

def mla_decode_attention(q:Tensor, cache:Tensor, valid:int|UOp, dv:int, scale:float, mask:Tensor|None=None) -> Tensor:
  """softmax(q k^T * scale + mask) v of a few tokens' q (1, H, T, D) on the cache (1, 1, N, D) of mla (k its first valid keys, v their first
  dv dims), causal without a mask: (1, H, T, dv). flash decoding, the partial sums of the chunks then their combine"""
  if isinstance(valid, UOp): cache = Tensor(cache.uop.after(valid))
  _, H, T, D = q.shape
  chunks, axis = min(MLA_PARTIALS, -(-int(cache.shape[2]) // MLA_KEYS)), q.uop.axis
  partial = Tensor.empty(1, H*T, chunks, dv, dtype=dtypes.float32, device=q.device, axis=None if axis is None else 1)
  stats = Tensor.empty(1, H*T, chunks, 2, dtype=dtypes.float32, device=q.device, axis=None if axis is None else 1)
  fxn = functools.partial(_mla_decode_partial, valid=valid, dv=dv, scale=scale, tokens=T, rdna4=_is_rdna4(q.device))
  partial, stats = Tensor.custom_kernel(partial, stats, q.half().reshape(1, H*T, D).contiguous(), cache,
                                        *([] if mask is None else [mask.float()]), fxn=fxn)[:2]
  live = (valid + MLA_KEYS - 1) // MLA_KEYS
  live = min(live, chunks) if isinstance(live, int) else live.minimum(chunks)
  out = Tensor.empty(1, H*T, 1, dv, dtype=dtypes.float32, device=q.device, axis=None if axis is None else 1)
  return Tensor.custom_kernel(out, partial, stats, fxn=functools.partial(_amd_flash_decode_combine, live=live))[0].reshape(1, H, T, dv)

MLAP_KEYS = 32  # the prefill's keys a round
# a workgroup's query rows, K's row pad (halves), waves, k splits of the scores, q in registers: MLAP_<knob>
MLAP_OPTS = {"R": 64, "PAD": 4, "W": 16, "DS": 2, "QREG": 1}

def _exp2(x:UOp) -> UOp: return UOp(Ops.CUSTOM, src=(x,), arg=("__builtin_amdgcn_exp2f({0})", dtypes.float))  # (v_exp_f32, no range fixup)

@functools.cache
def _mla_prefill_kernel(out:UOp, q:UOp, cache:UOp, start_pos:int|UOp, dv:int, scale:float, rdna4:bool, R:int=64, PAD:int=4, W:int=16,
                        DS:int=2, QREG:int=1) -> UOp:
  """causal mla attention of a chunk of T tokens at start_pos (token t sees keys 0..start_pos+t of the latent cache, v their first dv dims):
  flash attention, a workgroup R query rows (row r: token r // H, head r % H: the heads of a token share its keys) and a round of
  MLAP_KEYS keys at a time in LDS (read once for all the rows). R//8 waves, wave w: the scores of row tile w % (R//16), both key tiles,
  over half the dims (keys x rows, q from memory once a round) to LDS; the online softmax of 8 rows (a lane 8 keys of a row: 4 lanes a
  row; the weights and the rescale to LDS); then its dv tiles of the values transposed (dv x keys, strided LDS loads: a fragment for all
  the row tiles) times the weights (keys x rows; a lane a row: a row tile's rescale one value). out (1, H, T, dv) float"""
  _, H, T, D = cast(tuple[int, int, int, int], q.shape)
  N, sp, M = int(cache.shape[2]), _unbind(start_pos), H*T
  RT, KT, KS = R // 16, MLAP_KEYS // 16, D // 16
  threads, DW, DP, RW = W*WARP_SIZE, dv // (16*W), D + PAD, R // W
  LPR = WARP_SIZE // RW  # (the softmax's lanes a row)
  KL = MLAP_KEYS // LPR
  assert D % 16 == 0 and dv % (16*W) == 0 and dv <= D and R % W == 0 and WARP_SIZE % RW == 0 and MLAP_KEYS % LPR == 0
  rb = UOp.range(-(-M // R), 0, AxisType.GLOBAL)
  wave, lane = UOp.range(W, 1, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  tid, col, half = wave*WARP_SIZE + lane, lane % 16, lane // 16
  def acc_row(i:int) -> UOp: return half*8 + i if rdna4 else half + 2*i  # (the M row of a lane's i-th accumulator)
  def frag(view:UOp) -> UOp: return _wmma_fragment(view, lane, rdna4)
  def rowtok(r:UOp) -> tuple[UOp, UOp]:  # (the token and head of a workgroup row)
    g = (rb*R + r).minimum(M - 1)
    return g // H, g % H
  last = (rb*R + R - 1).minimum(M - 1) // H  # (the workgroup's last token: its keys the rounds')
  rnd = UOp.range((sp + last) // MLAP_KEYS + 1, 2, AxisType.LOOP)
  first = rnd*MLAP_KEYS
  # the round's keys into LDS, 8 halves a load, a thread a key's every 8*TK-th 8 halves (one address, offsets); keys past the cache's end:
  # the last one's, masked
  TK = threads // MLAP_KEYS
  K = UOp.alloc((MLAP_KEYS, DP), dtypes.half, addrspace=AddrSpace.LOCAL)
  fills = []
  for i in range(-(-D // (8*TK))):
    key, d0 = tid // TK, (tid % TK)*8 + i*TK*8
    ok = d0 < D if D % (8*TK) and (i+1)*TK*8 > D else None
    vals = _amd_load(cache.reshape(N*D)[(first + key).minimum(N - 1)*D + (d0 if ok is None else d0.minimum(D - 8))], 8)
    fills += [K.after(rnd)[key, d0 + e if ok is None else (d0 + e).valid(ok)].store(vals[e]) for e in range(8)]
  K = K.after(UOp.group(*fills).barrier())
  # the scores (keys x rows) of row tile srt, KTW key tiles from ktb, over k split sd (DS of them; a q fragment loaded once for the key
  # tiles; QREG: the wave's q fragments in registers the whole kernel, else loaded a round), into LDS (the splits' partial sums: rows x
  # keys, float)
  groups = W // RT
  KTW = KT*DS // groups
  assert W % RT == 0 and KS % DS == 0 and KTW >= 1 and KT % KTW == 0 and groups == DS*(KT // KTW)
  srt, sg = wave % RT, wave // RT
  sd, ktb = sg // (KT // KTW), sg % (KT // KTW) * KTW
  stok, shead = rowtok(srt*16 + col)
  qv = q.reshape(H, T, KS, 16)[shead, stok]
  if QREG:
    qfs = [frag(qv[sd*(KS // DS) + i]) for i in range(KS // DS)]
  S = _reg((KTW*8,), 0, rnd)
  if QREG:
    Sv = [S.reshape(KTW, 8)[k] for k in range(KTW)]
    for i in range(KS // DS):
      dk = sd*(KS // DS) + i
      Sv = [UOp.wmma(frag(K[:, :D].reshape(MLAP_KEYS, KS, 16)[(ktb + k)*16 + col, dk]), qfs[i], Sv[k], *WMMA_ARG) for k in range(KTW)]
    sdone = UOp.group(*[S.reshape(KTW, 8)[k].store(Sv[k]) for k in range(KTW)])
  else:
    ks = UOp.range(KS // DS, 3, AxisType.LOOP)
    dk, Ss = sd*(KS // DS) + ks, S.after(ks).reshape(KTW, 8)
    sdone = UOp.group(*[S.reshape(KTW, 8)[k].store(UOp.wmma(frag(K[:, :D].reshape(MLAP_KEYS, KS, 16)[(ktb + k)*16 + col, dk]), frag(qv[dk]),
                                                            Ss[k], *WMMA_ARG)) for k in range(KTW)]).end(ks)
  S = S.after(sdone)
  SL = UOp.alloc((DS, R, MLAP_KEYS + 1), dtypes.float, addrspace=AddrSpace.LOCAL)
  SL = SL.after(UOp.group(*[SL.after(rnd)[sd, srt*16 + col, (ktb + k)*16 + acc_row(i)].store(S[k*8 + i].load())
                            for k in range(KTW) for i in range(8)]).barrier())
  # the online softmax: lane l of wave w row w*RW + l//LPR, keys (l%LPR)*KL..+KL; the weights (half) and the row's rescale to LDS
  srow, part = wave*RW + lane // LPR, lane % LPR
  rtok, _ = rowtok(srow)
  neg = UOp.const(-1e30, dtypes.float)
  sv = [(first + part*KL + j <= sp + rtok).where(sum((SL[x, srow, part*KL + j].load() for x in range(1, DS)), SL[0, srow, part*KL + j].load())
        *(scale*LOG2E), neg) for j in range(KL)]
  mx, sm = _reg((1,), -1e30), _reg((1,), 0)
  pmx, psm = mx.after(rnd), sm.after(rnd)
  rmax = functools.reduce(UOp.maximum, sv, pmx[0].load())
  for m in (1 << b for b in range(LPR.bit_length() - 1)): rmax = rmax.maximum(_swizzle_xor(rmax, m))
  alpha, p = _exp2(pmx[0].load() - rmax), [_exp2(v - rmax) for v in sv]
  psum = functools.reduce(UOp.__add__, p)
  for m in (1 << b for b in range(LPR.bit_length() - 1)): psum = psum + _swizzle_xor(psum, m)
  PL = UOp.alloc((R, MLAP_KEYS + PAD), dtypes.half, addrspace=AddrSpace.LOCAL)
  AL = UOp.alloc((R,), dtypes.float, addrspace=AddrSpace.LOCAL)
  wr = UOp.group(*[PL.after(rnd)[srow, part*KL + j].store(v.cast(dtypes.half)) for j, v in enumerate(p)],
                 AL.after(rnd)[srow.valid(part.eq(0))].store(alpha)).barrier()
  PL, AL = PL.after(wr), AL.after(wr)
  # the values: the wave's dv tiles (dv x keys, strided loads of the keys' first dv) times the weights of each row tile (keys x rows),
  # onto the sums so far scaled by the row's alpha
  acc = _reg((RT*DW*8,), 0)
  pacc = acc.after(rnd)
  cur = acc.after(UOp.group(*[acc[(rt*DW + d)*8 + i].store(AL[rt*16 + col].load()*pacc[(rt*DW + d)*8 + i].load())
                              for rt in range(RT) for d in range(DW) for i in range(8)]))
  for kk in range(KT):
    pfs = [frag(PL[:, :MLAP_KEYS].reshape(R, KT, 16)[rt*16 + col, kk]) for rt in range(RT)]
    vfs = [frag(K.reshape(KT, 16, DP).permute(0, 2, 1)[kk, (wave*DW + d)*16 + col]) for d in range(DW)]
    cur = acc.after(UOp.group(*[cur.reshape(RT*DW, 8)[rt*DW + d].store(UOp.wmma(vfs[d], pfs[rt], cur.reshape(RT*DW, 8)[rt*DW + d], *WMMA_ARG))
                                for rt in range(RT) for d in range(DW)]))
  upd = UOp.group(cur, mx[0].store(rmax), sm[0].store(alpha*psm[0].load() + psum))
  acc, mx, sm = (x.after(upd.barrier().end(rnd)) for x in (acc, mx, sm))
  LL = UOp.alloc((R,), dtypes.float, addrspace=AddrSpace.LOCAL)
  LL = LL.after(LL[srow.valid(part.eq(0))].store(sm[0].load()).barrier())
  stores = []
  for rt in range(RT):
    # (the row's gate opaque: else simplified by the bounds of the workgroup's index it implies, then dropped)
    g = UOp(Ops.CUSTOMI, src=(rb*R + rt*16 + col,), arg=("{0}", dtypes.int))
    tok, head, inv = g.minimum(M - 1) // H, g.minimum(M - 1) % H, 1 / LL[rt*16 + col].load()
    stores += [out[0, head, tok.valid(g < M), (wave*DW + d)*16 + acc_row(i)].store(acc[(rt*DW + d)*8 + i].load() * inv)
               for d in range(DW) for i in range(8)]
  return UOp.group(*stores).end(lane, wave, rb).sink(arg=KernelInfo(name="mla_prefill", opts_to_apply=()))

def mla_prefill_attention(q:Tensor, cache:Tensor, start_pos:int|UOp, dv:int, scale:float) -> Tensor:
  """causal softmax(q k^T * scale) v of a prompt chunk's q (1, H, T, D) at start_pos on the mla cache (1, 1, N, D) holding its keys (token t
  at start_pos + t sees the keys 0..start_pos+t, v their first dv dims): (1, H, T, dv) float. one kernel, no scores in memory"""
  if isinstance(start_pos, UOp): cache = Tensor(cache.uop.after(start_pos))
  _, H, T, _ = q.shape
  axis = q.uop.axis
  out = Tensor.empty(1, H, T, dv, dtype=dtypes.float32, device=q.device, axis=axis)
  fxn = functools.partial(_mla_prefill_kernel, start_pos=start_pos, dv=dv, scale=scale, rdna4=_is_rdna4(q.device),
                          **{k: getenv(f"MLAP_{k}", v) for k, v in MLAP_OPTS.items()})
  return Tensor.custom_kernel(out, q.half().contiguous(), cache, fxn=fxn)[0]

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
def _gated_delta_prefill_kernel(core:UOp, q:UOp, k:UOp, v:UOp, beta:UOp, alpha:UOp, state:UOp, kq:UOp, start_pos:UOp|None=None) -> UOp:
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
  token = UOp.range(tokens, 2, AxisType.LOOP)
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
  step = UOp.group(*stores, current.store(UOp.stack(*updates))).end(token)
  state_stores = (state[bh, row, col].store(current.after(step)[row_idx*key_dim//32+i].load().cast(state.dtype))
                  for row_idx,row in enumerate(rows) for i,col in enumerate(cols))
  return UOp.group(*state_stores).end(lane, bh_row).sink(arg=KernelInfo(name="gated_delta_prefill", opts_to_apply=()))

KDA_ROWS = 8  # the rows of a head's state a wave updates in kda_decode

@functools.cache
def _kda_decode_kernel(z:UOp, q8:UOp, scale:UOp, xsum:UOp, state:UOp, conv:UOp, q:UOp, k:UOp, v:UOp, gate:UOp, beta:UOp, conv_w:UOp,
                       alpha_lin:UOp, dt_bias:UOp, ssm_a:UOp, norm_w:UOp, start_pos:UOp, lb:float, norm_eps:float, token:int=0,
                       state_src:UOp|None=None, conv_src:UOp|None=None) -> UOp:
  """a kda token (GatedDeltaNetBlock._attention), a workgroup a head: a thread a channel's conv and silu of q, k and v and its decay
  (to LDS, the conv's window in place), then a wave KDA_ROWS rows of the state, a lane 32 apart columns of them: the l2 norms of q and k,
  the delta rule on the rows (the state in place), the output norm and gate: z and its q8 (a wave a group). the inputs, z and its q8
  are of the step's tokens, token's row read and written. state_src, conv_src: read from these instead"""
  state_in, conv_in = state if state_src is None else state_src, conv if conv_src is None else conv_src
  _, H, hv, hk = cast(tuple[int, int, int, int], state.shape)
  W, zero, initial = hv // KDA_ROWS, UOp.const(0, dtypes.float), start_pos.eq(0)
  assert hv == hk and hk % 32 == 0 and hv % KDA_ROWS == 0 and W*WARP_SIZE >= hk
  head, wave, lane = UOp.range(H, 0, AxisType.GLOBAL), UOp.range(W, 1, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  t = wave*WARP_SIZE + lane
  tc, live = t.minimum(hk-1), t < hk
  ch = head*hk + tc
  def row(x:UOp, i:UOp) -> UOp: return x.reshape(-1)[token*(int(prod(x.shape))//int(z.shape[1])) + i]  # (token's row of a step's x)
  wins = [[initial.where(0, conv_in[0, i, part, ch].load()) for i in range(3)] + [row(x, ch).load()] for part, x in enumerate((q, k, v))]
  qv, kv, vv = (sum((w[i] * conv_w[part, ch, i].load().float() for i in range(4)), zero).silu() for part, w in enumerate(wins))
  a = -ssm_a[head, 0].load().float()
  al = (((row(alpha_lin, ch).load().float() + dt_bias[ch].load().float()) * a).sigmoid() * lb).exp()
  lds = UOp.alloc((4, hk), dtypes.float, addrspace=AddrSpace.LOCAL)
  lds = lds.after(UOp.group(*[lds[j, tc.valid(live)].store(x) for j, x in enumerate((qv, kv, vv, al))]).barrier())
  # the next window, once the thread read this one
  stores = [conv.after(UOp.group(*[x for w in wins for x in w]))[0, i, part, ch.valid(live)].store(w[i+1]) for part, w in enumerate(wins)
            for i in range(3)]
  cols, rows = [lane + 32*i for i in range(hk//32)], [wave*KDA_ROWS + j for j in range(KDA_ROWS)]
  def l2(x:list[UOp]) -> list[UOp]:  # as rmsnorm(eps=1e-6/hk) / hk**0.5
    r = (warp_reduce(sum((e*e for e in x), zero), full_wave=True) * (1/hk) + 1e-6/hk).rsqrt()
    return [e * r / hk**0.5 for e in x]
  qn, kn = [e * hk**-0.5 for e in l2([lds[0, c].load() for c in cols])], l2([lds[1, c].load() for c in cols])
  kq = warp_reduce(sum((x*y for x, y in zip(qn, kn)), zero), full_wave=True)
  b, alpha = row(beta, head).load().sigmoid(), [lds[3, c].load() for c in cols]
  core = []
  for r in rows:
    decayed = [initial.where(0, state_in[0, head, r, c].load()) * al_ for c, al_ in zip(cols, alpha)]
    sk, sq = (warp_reduce(sum((d*x for d, x in zip(decayed, xs)), zero), full_wave=True) for xs in (kn, qn))
    delta = (lds[2, r].load() - sk) * b
    stores += [state[0, head, r, c].store(d + delta*x) for c, d, x in zip(cols, decayed, kn)]
    core.append(sq + delta*kq)
  # the head's outputs through LDS: their mean square, then a wave a q8 group of z
  cl, ss = UOp.alloc((hv,), dtypes.float, addrspace=AddrSpace.LOCAL), UOp.alloc((W,), dtypes.float, addrspace=AddrSpace.LOCAL)
  cl = cl.after(bar:=UOp.group(*[cl[r.valid(lane.eq(0))].store(c) for r, c in zip(rows, core)],
                               ss[wave.valid(lane.eq(0))].store(sum((c*c for c in core), zero))).barrier())
  rs = (sum((ss.after(bar)[w].load() for w in range(1, W)), ss.after(bar)[0].load()) * (1/hv) + norm_eps).rsqrt()
  e, live = wave*32 + lane, wave < hv//32
  ec = e.minimum(hv-1)
  zh = (cl[ec].load() * rs * norm_w[ec].load().float() * row(gate, head*hv + ec).load().sigmoid()).cast(z.dtype)
  value, g = zh.float(), (token*H + head)*(hv//32) + wave
  stores.append(row(z, (head*hv + e).valid(live)).store(zh))
  d = UOp(Ops.CUSTOM, src=(warp_reduce(value.abs(), maximum=True, full_wave=True),), arg=("({0}/127.0f)", dtypes.float)).maximum(1e-8)
  quant = UOp(Ops.CUSTOM, src=(value, d), arg=("__builtin_nearbyintf({0}/{1})", dtypes.float)).clip(-127, 127).cast(dtypes.int8)
  word = quant.cast(dtypes.uint8).cast(dtypes.uint32) << ((lane%4)*8).cast(dtypes.uint32)
  for offset in (1, 2):
    word |= UOp(Ops.CUSTOM, src=(word,), arg=(f"__builtin_amdgcn_ds_swizzle({{0}}, {0x1f | offset<<10})", dtypes.uint32))
  # (a valid on the flat index: of a coordinate it doesn't stop a group past the buffer)
  stores += [q8.reshape(-1)[(g*8 + lane//4).valid(live & (lane%4).eq(0))].store(word), scale.reshape(-1)[g.valid(live & lane.eq(0))].store(d),
             xsum.reshape(-1)[(g*2 + lane//16).valid(live & (lane%16).eq(0))].store(warp_reduce(quant.float()))]
  return UOp.group(*stores).end(lane, wave, head).sink(arg=KernelInfo(name="kda_decode", opts_to_apply=()))

def kda_decode(q:Tensor, k:Tensor, v:Tensor, gate:Tensor, beta:Tensor, conv:Tensor, conv_w:Tensor, alpha_lin:Tensor, dt_bias:Tensor,
               ssm_a:Tensor, state:Tensor, norm_w:Tensor, start_pos:UOp, lb:float, norm_eps:float, src:tuple[Tensor, Tensor]|None=None,
               token:int=0, z:Tensor|None=None) -> tuple[Tensor, Tensor, Tensor]:
  """GatedDeltaNetBlock._attention's z (1, T, inner) for a token of a step's T from their linears' outputs (q, k, v, gate, beta, the
  decay's alpha_lin), one kernel, and the conv window and state it wrote. those in place (src: the conv window and state read instead).
  token's row of z (the z of the step's earlier tokens: the rows they wrote) and of its q8, in the q8 cache: the out proj reads that"""
  T, inner, local = int(q.shape[1]), int(alpha_lin.shape[-1]), int(alpha_lin.uop.shard_shape[-1])
  if z is None: z, q8, scale, xsum = Tensor.empty(1, T, inner, dtype=dtypes.half, device=q.device, axis=None if alpha_lin.uop.axis is None else 2), \
    Tensor.empty(T, local//Q8_GROUP_SIZE, 8, dtype=dtypes.uint32, device=q.device), Tensor.empty(T, local//Q8_GROUP_SIZE, dtype=dtypes.float32,
    device=q.device), Tensor.empty(T, local//Q8_GROUP_SIZE, 2, dtype=dtypes.float32, device=q.device)
  else: q8, scale, xsum = _q8_cache[z.uop]
  srcs = (z, q8, scale, xsum, Tensor(state.uop.after(start_pos)), conv, q, k, v, gate, beta, conv_w, alpha_lin, dt_bias, ssm_a, norm_w,
          *(() if src is None else (src[1], src[0])))
  contig = tuple(x.uop if x.uop.has_buffer_identity(after_ok=True) else x.uop.contiguous() for x in srcs)
  params = [UOp.placeholder_like(x, slot=i) for i, x in enumerate(contig)]
  call = _kda_decode_kernel(*params[:16], _unbind(start_pos), lb, norm_eps, token, *params[16:]).call(*contig)
  z, q8, scale, xsum, state, conv = (Tensor(x.after(call)) for x in contig[:6])
  _q8_cache[z.uop] = (q8, scale, xsum)
  return z, conv, state

def gated_delta_prefill(q:Tensor, k:Tensor, v:Tensor, beta:Tensor, alpha:Tensor, state:Tensor, start_pos:Tensor|None=None) -> Tensor:
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
  srcs = (core, q.contiguous(), k.contiguous(), v.contiguous(), beta.contiguous(), alpha.contiguous(), state, kq)
  contig = tuple(x.uop if x.uop.op is Ops.AFTER else x.uop.contiguous() for x in srcs)
  params = tuple(UOp.placeholder_like(x, slot=i) for i,x in enumerate(contig))
  call = _gated_delta_prefill_kernel(*params, None if start_pos is None else start_pos.uop.unbound()).call(*contig)
  return Tensor(contig[0].after(call))
