from __future__ import annotations
import functools, math
from dataclasses import replace
from typing import Callable, cast
from tinygrad import Tensor, UOp, nn, Device, Context
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import prod, getenv, round_up
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, resolve
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.renderer import Estimates

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
# formats of independent 32 weight blocks: (halfwords per block). a device may hold part of a 256 weight row group
SMALL_BLOCK_QUANTS = {Q8_0: 17, IQ4_NL: 9}
QUANT_NAMES = {Q8_0: "q8_0", Q2_K: "q2_k", Q3_K: "q3_k", Q4_K: "q4_k", Q5_K: "q5_k", Q6_K: "q6", IQ4_XS: "iq4_xs",
               IQ2_XS: "iq2_xs", IQ3_XXS: "iq3_xxs", IQ4_NL: "iq4_nl", IQ3_S: "iq3_s", IQ2_S: "iq2_s"}

def _unbind(v:int|UOp) -> int|UOp: return v.unbind_all()[0] if isinstance(v, UOp) else v

@functools.cache
def amd_custom_kernels_supported(device:str|tuple[str, ...]|None) -> bool:
  if getenv("DISABLE_AMD_KERNELS"): return False
  # Wave32 and WMMA kernels support RDNA3/4. CDNA's wave64/MFMA is not supported.
  if isinstance(device, tuple): device = device[0]
  if device is None or device.split(":")[0] != "AMD": return False
  # @function contexts set ALLOW_DEVICE_USAGE=0 (scheduling must not open devices); the device is always open here
  with Context(ALLOW_DEVICE_USAGE=1):
    return (t:=getattr(Device[device], "target", None)) is not None and t[0] in (11, 12) and isinstance(Device[device].renderer, HIPRenderer)

# int8 WMMA on q8 activations pays a rescale per 32 inputs of every output: it wins where the f16 decode is expensive, the grid formats
I8_WMMA_QUANTS = (IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S)
# formats with a single scale per 32 weights and no offset
GROUP_SCALE_QUANTS = (Q8_0, IQ4_NL, IQ4_XS, IQ3_S, IQ3_XXS, Q4_K, Q5_K)
def _use_i8_wmma(device:str|tuple[str, ...]|None, ggml_type:int) -> bool:
  return bool(getenv("I8_WMMA", 1)) and _wmma_rdna4(device) and (ggml_type in I8_WMMA_QUANTS or getenv("I8_WMMA", 1) == 2)

@functools.cache
def _wmma_rdna4(device:str|tuple[str, ...]) -> bool:
  if isinstance(device, tuple): device = device[0]
  with Context(ALLOW_DEVICE_USAGE=1): return getattr(Device[device], "target")[0] == 12

def warp_reduce(val:UOp, maximum:bool=False, full_wave:bool=False) -> UOp:
  for offset in ((16, 8, 4, 2, 1) if full_wave else (8, 4, 2, 1)):
    if val.op is Ops.INDEX and val.addrspace == AddrSpace.REG: val = val.load()
    other = UOp(Ops.CUSTOM, src=(val,), arg=
      (f"__builtin_bit_cast(float, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, {{0}}), {0x1f | offset<<10}))", dtypes.float))
    val = val.maximum(other) if maximum else val + other
  return val

def _reg(shape:tuple[int, ...], value:float, dep:UOp|None=None) -> UOp:
  ret = UOp.alloc(shape, dtypes.float, addrspace=AddrSpace.REG)
  return ret.after((ret if dep is None else ret.after(dep)).store(ret.const_like(value)))

# ******** quant linear: q8-activation kernels over packed ggml weights ********

def find_packed(decoded:Tensor, in_features:int) -> tuple[int, Tensor]|None:
  """(ggml type, packed words) if `decoded` is exactly the dequantization of packed ggml weights, rows of in_features"""
  if in_features % Q8_GROUP_SIZE: return None
  packed_sizes = {typ: decoded.numel() // 256 * type_size for typ,type_size in QUANT_SIZES.items()
                  if (in_features % GGML_BLOCK_SIZE == 0 or typ in SMALL_BLOCK_QUANTS) and decoded.numel() % 256 == 0}
  packed_sizes.update({typ: decoded.numel() // 32 * hw * 2 for typ,hw in SMALL_BLOCK_QUANTS.items()})
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
  def __init__(self, in_features:int, out_features:int, bias=True):
    super().__init__(in_features, out_features, bias)
    self.in_features, self.out_features = in_features, out_features
  def set_quantized(self, decoded:Tensor):
    if (found:=find_packed(decoded, self.in_features)) is None: return
    self.ggml_type, self.weight = found
    self.shard_axis = decoded.uop.axis
  def partial(self, x:Tensor) -> Tensor:
    """row parallel layers return the partial sums of each device, (devices, ..., out_features) sharded on the devices, the others the output"""
    if self.ggml_type is None and self.use_custom_quant and amd_custom_kernels_supported(self.weight.device): self.set_quantized(self.weight)
    if self.ggml_type in QUANT_SIZES and self.shard_axis == 1 and self.bias is None and isinstance(x.numel(), int): return q8_linear(self, x, False)
    return self(x)
  def __call__(self, x:Tensor) -> Tensor:
    supported = self.use_custom_quant and amd_custom_kernels_supported(self.weight.device)
    if self.ggml_type is None and supported:
      self.set_quantized(self.weight)
      if self.ggml_type is None:
        # tiny dense fp16 matmul (e.g. the ssm beta/alpha head rows): single fp16 gemv kernel instead of a
        # generic matmul schedule, and realize the densely packed weight once if it is still a lazy ggml view
        if self.weight.dtype in (dtypes.half, dtypes.float, dtypes.bfloat16) and self.out_features <= 2048 \
          and self.in_features % (WARP_SIZE*4) == 0 and self.weight.uop.axis is None:
          numel, max_shape = x.numel(), x.max_shape
          if prod(max_shape) // self.in_features <= 32:
            out = f16_gemv(self, x if isinstance(numel, int) else x.pad_to(max_shape))
            return out if isinstance(numel, int) else out.shrink(tuple((0, s) for s in (*x.shape[:-1], self.out_features)))
        self.use_custom_quant = supported = False  # not a supported quant format
    if self.ggml_type in QUANT_SIZES and supported:
      if isinstance(x.numel(), int): return q8_linear(self, x)
      # symbolic token count: pad to the max chunk size so the kernels see static shapes, garbage rows are sliced off
      out = q8_linear(self, x.pad_to(x.max_shape))
      return out.shrink(tuple((0, s) for s in (*x.shape[:-1], self.out_features)))
    return super().__call__(x)

def _amd_dp4a(a:UOp, b:UOp, c:UOp) -> UOp:
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

@functools.cache
def _q8_quantize_kernel(q:UOp, scale:UOp, xsum:UOp, x:UOp, *sq:UOp, tokens:int, in_features:int) -> UOp:
  groups = in_features//Q8_GROUP_SIZE
  token_group, lane = UOp.range(tokens*groups, 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP)
  token, group = token_group//groups, token_group%groups
  value = x.reshape(tokens, groups, 32)[token, group, lane].load().float()
  # Quantize each input once, then pack four neighboring lanes into one word.
  # Keep divisions intact: reciprocal multiplication can move half-precision inputs across rounding ties.
  d = UOp(Ops.CUSTOM, src=(warp_reduce(value.abs(), maximum=True, full_wave=True),), arg=("({0}/127.0f)", dtypes.float)).maximum(1e-8)
  rounded = UOp(Ops.CUSTOM, src=(value, d), arg=("__builtin_nearbyintf({0}/{1})", dtypes.float))
  quant = rounded.clip(-127, 127).cast(dtypes.int8)
  word = quant.cast(dtypes.uint8).cast(dtypes.uint32) << ((lane%4)*8).cast(dtypes.uint32)
  for offset in (1, 2):
    word |= UOp(Ops.CUSTOM, src=(word,), arg=(f"__builtin_amdgcn_ds_swizzle({{0}}, {0x1f | offset<<10})", dtypes.uint32))
  stores:tuple[UOp, ...] = (q[token, group, (lane//4).valid((lane%4).eq(0))].store(word),
            scale[token, group.valid(lane.eq(0))].store(d),
            xsum[token, group, (lane//16).valid((lane%16).eq(0))].store(warp_reduce(quant.float())))
  # optionally the sum of squares of each group, for an rms norm applied after a linear layer
  if sq: stores += (sq[0][token, group.valid(lane.eq(0))].store(warp_reduce(value*value, full_wave=True)),)
  return UOp.group(*stores).end(token_group, lane).sink(arg=KernelInfo(name="q8_quantize", opts_to_apply=()))

def q8_quantize(x:Tensor, tokens:int, in_features:int, sumsq:bool=False, split:bool=False) -> tuple[Tensor, ...]:
  """
  int8 groups of 32 with their scales and sums (and optionally sums of squares). an input shared by several layers is quantized once.
  NOTE: the cache holds graphs, the model clears it after building one (q8_quantize_cache_clear)
  """
  return _q8_quantize(x.uop, tokens, in_features, sumsq, split)

def q8_quantize_cache_clear(): _q8_quantize.cache_clear()

@functools.lru_cache(maxsize=32)
def _q8_quantize(x_uop:UOp, tokens:int, in_features:int, sumsq:bool, split:bool) -> tuple[Tensor, ...]:
  x, groups = Tensor(x_uop), in_features//Q8_GROUP_SIZE
  # split: the rows are split over the devices, so are the outputs, every device quantizes its rows
  axis, local = (0, tokens // len(cast(tuple, x.device))) if split else (None, tokens)
  q = Tensor.empty(tokens, groups, 8, dtype=dtypes.uint32, device=x.device, axis=axis)
  scale = Tensor.empty(tokens, groups, dtype=dtypes.float32, device=x.device, axis=axis)
  xsum = Tensor.empty(tokens, groups, 2, dtype=dtypes.float32, device=x.device, axis=axis)
  sq = (Tensor.empty(tokens, groups, dtype=dtypes.float32, device=x.device, axis=axis),) if sumsq else ()
  outs = Tensor.custom_kernel(q, scale, xsum, x, *sq, fxn=functools.partial(_q8_quantize_kernel, tokens=local, in_features=in_features))
  return tuple(outs[:3]) + ((outs[4],) if sumsq else ())

def _decode_linear(out:UOp, out_features:int, group_count:int, group_dot, name:str, active:Callable|None=None,
                   estimates:Estimates|None=None) -> UOp:
  # One wave per output/chunk of 32 groups; group neighboring rows to amortize workgroup scheduling. the waves of the chunks of an output
  # share the workgroup: they add up their sums in LDS
  # a wave serves 32//lanes outputs with `lanes` lanes each, whichever leaves fewer lanes idle. at most 16 waves in a workgroup: a wave sums
  # `per` slots of its lanes
  # active(token) -> bool: inactive tokens (experts of other devices) skip their loop and store 0
  def waste(lanes:int) -> int: return lanes*((group_count+lanes-1)//lanes) - group_count
  lanes = min((l for l in (32, 16, 8) if out_features % (32//l) == 0), key=lambda l: (waste(l), -l))
  outs_per_wave, slots = 32 // lanes, (group_count+lanes-1)//lanes
  per = (slots + 15) // 16
  chunks = (slots + per-1) // per
  rows = math.gcd(out_features // outs_per_wave, 4)
  while rows > 1 and rows*chunks > 16: rows //= 2
  row = UOp.range(out.shape[0]*out_features//(rows*outs_per_wave), 0, AxisType.GLOBAL)
  wave, chunk, lane = UOp.range(rows, 1, AxisType.LOCAL), UOp.range(chunks, 3, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  sub, which = lane % lanes, lane // lanes
  token_output = (row*rows+wave)*outs_per_wave + which
  token, output = token_output // out_features, token_output % out_features
  def dot(group:UOp) -> UOp:
    return (group < group_count).where(group_dot(token, output, group.minimum(group_count-1)), UOp.const(0, dtypes.float32))
  if active is None: total = functools.reduce(lambda a,s: a + dot(sub+(chunk*per+s)*lanes), range(1, per), dot(sub+chunk*per*lanes))
  else:
    # a data dependent loop: inactive tokens run it 0 times
    s = UOp.range(active(token).where(per, 0), 5, AxisType.LOOP)
    acc = _reg((1,), 0)
    total = acc.after(acc[0].store(acc.after(s)[0].load() + dot(sub+(chunk*per+s)*lanes)).end(s))[0].load()
  for offset in (16, 8, 4, 2, 1)[(32//lanes).bit_length()-1:]:
    total = total + UOp(Ops.CUSTOM, src=(total,), arg=
      (f"__builtin_bit_cast(float, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, {{0}}), {0x1f | offset<<10}))", dtypes.float))
  if chunks > 1:
    lds = UOp.alloc((rows, chunks, outs_per_wave), dtypes.float32, addrspace=AddrSpace.LOCAL)
    lds = lds.after(lds[wave, chunk, which.valid(sub.eq(0))].store(total).barrier())
    total = functools.reduce(lambda a,c: a + lds[wave, c, which].load(), range(1, chunks), lds[wave, 0, which].load())
  return out[token, output.valid(sub.eq(0) & chunk.eq(0))].store(total.cast(out.dtype)).end(row, wave, chunk, lane).sink(
    arg=KernelInfo(name=name, opts_to_apply=(), estimates=estimates))

def _decode_linear_loop(out:UOp, out_features:int, group_count:int, group_dot, name:str, waves:int=4, active:Callable|None=None,
                        estimates:Estimates|None=None) -> UOp:
  # out (rows, inner rows, out_features): one wave per output feature of a row, looping over the inner rows and the groups.
  # no partial sums to add up afterwards, and the inner rows (the experts of a token) share a workgroup
  rows, inner = out.shape[0], out.shape[1]
  waves = math.gcd(out_features, waves)
  row, block = UOp.range(rows, 0, AxisType.GLOBAL), UOp.range(out_features//waves, 1, AxisType.GLOBAL)
  lane, wave = UOp.range(32, 2, AxisType.LOCAL), UOp.range(waves, 3, AxisType.LOCAL)
  output = block*waves + wave
  inner_row, chunks = UOp.range(inner, 4, AxisType.LOOP), (group_count+31)//32
  # rows that aren't active (experts of other devices) skip their loop
  chunk = UOp.range(chunks if active is None else active(row*inner+inner_row).where(chunks, 0), 5, AxisType.LOOP)
  acc = _reg((1,), 0, inner_row)
  group = lane+chunk*32
  value = (group < group_count).where(group_dot(row*inner+inner_row, output, group.minimum(group_count-1)), UOp.const(0, dtypes.float32))
  acc = acc.after(acc[0].store(acc.after(chunk)[0].load() + value).end(chunk))
  total = warp_reduce(acc[0].load(), full_wave=True)
  # a data dependent loop has no symbolic estimates
  return out[row, inner_row, output.valid(lane.eq(0))].store(total.cast(out.dtype)).end(inner_row).end(row, block, lane, wave).sink(
    arg=KernelInfo(name=name, opts_to_apply=(), estimates=estimates))

def _iq_grid(device:str|tuple[str, ...]|None, ggml_type:int) -> Tensor:
  from tinygrad.runtime.autogen import ggml_common as ggml
  grid, words = {IQ2_XS: (ggml.iq2xs_grid, 2), IQ2_S: (ggml.iq2s_grid, 2),
                 IQ3_XXS: (ggml.iq3xxs_grid, 1), IQ3_S: (ggml.iq3s_grid, 1)}[ggml_type]
  return Tensor.const(tuple((v >> (32*i)) & 0xffffffff for v in grid for i in range(words)),
                      dtypes.uint32).to(device, force=True)

def _iq_even_signs(signs:UOp) -> UOp:
  parity = signs ^ (signs >> 4)
  parity ^= parity >> 2
  parity ^= parity >> 1
  return signs | ((parity & 1) << 7)

def _iq_signed_word(word:UOp, signs:UOp) -> UOp:
  mask = sum(((signs >> i) & 1) * (255 << (8*i)) for i in range(4))
  # Grid magnitudes are nonzero and <128, so each byte can be negated without a carry into its neighbor.
  return (word ^ mask) + (mask & 0x01010101)

def _quant_word(raw:UOp, base:UOp, subgroup:UOp, i:int|UOp, ggml_type:int, grid:UOp|None) -> UOp:
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
    else: signs = _iq_even_signs((word(66+subgroup*4) >> (7*(i//2))) & 127)
  elif ggml_type == IQ2_XS:
    packed = raw[base+1+subgroup*4+i//2].cast(dtypes.uint32)
    index, signs = packed & 511, _iq_even_signs(packed >> 9)
  else:
    index = byte(2+subgroup*4+i//2) | (((byte(66+subgroup) >> (2*(i//2))) & 3) << 8)
    signs = byte(34+subgroup*4+i//2)
  return _iq_signed_word(grid[index*grid_words+i%grid_words], signs >> (4*(i%2)))

@functools.cache
def _quant_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, *grids:UOp,
                         out_features:int, in_features:int, ggml_type:int) -> UOp:
  return _quant_decode_body(out, raw, xq, xd, xs, grids, out_features, in_features, ggml_type)

def _quant_decode_body(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, grids:tuple[UOp, ...], out_features:int, in_features:int, ggml_type:int,
                       gate:Callable|None=None, xtoken:Callable|None=None, name:str="linear_", active:Callable|None=None) -> UOp:
  # gate(token, output, base) -> base: moves (and predicates) the weight rows, xtoken(token) -> the activation row of an output row
  def group_dot(token:UOp, output:UOp, group:UOp) -> UOp:
    block, subgroup = group//8, group%8
    base = (output*in_features//256 + block) * (QUANT_SIZES[ggml_type]//raw.dtype.itemsize)
    if ggml_type in SMALL_BLOCK_QUANTS: base = (output*in_features//32 + group)*SMALL_BLOCK_QUANTS[ggml_type]
    if gate is not None: base = gate(token, output, base)
    def byte(offset): return _load_byte(raw, base, offset)
    def word(offset): return _load_u32(raw, base, offset)
    xwords = _amd_load(xq[token if xtoken is None else xtoken(token), group, 0], 8)
    # One accumulator per scale group: 32 weights for Q4/Q5/IQ4_XS/Q8_0, two groups of 16 otherwise.
    dots = [UOp.const(0, dtypes.int32)] * (1 if ggml_type in (Q4_K, Q5_K, IQ4_XS, Q8_0) else 2)
    for i in range(8):
      weights = _quant_word(raw, base, subgroup, i, ggml_type, grids[0] if grids else None)
      acc = i//(8//len(dots))
      dots[acc] = _amd_dp4a(weights, xwords[i], dots[acc])
    if xtoken is not None: token = xtoken(token)
    if ggml_type == Q8_0: total = dots[0].float() * _half(raw[base])
    elif ggml_type in (Q4_K, Q5_K):
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
  if len(out.shape) == 2 and active is not None:
    # moe on a grid: a row per (token, slot), inactive rows skip their loop. an upper bound estimate
    weights, nbytes = (w:=out.shape[0]*out_features*in_features), w*QUANT_SIZES[ggml_type]//256
    return _decode_linear(out, out_features, in_features//32, group_dot, name+QUANT_NAMES[ggml_type], active=active,
                          estimates=Estimates(2*weights, nbytes, nbytes))
  if len(out.shape) == 3 and out.shape[2] == out_features:
    # an upper bound: the rows of inactive experts are skipped
    weights, nbytes = (w:=out.shape[0]*out.shape[1]*out_features*in_features), w*QUANT_SIZES[ggml_type]//256
    return _decode_linear_loop(out, out_features, in_features//32, group_dot, name+QUANT_NAMES[ggml_type], active=active,
                               estimates=None if active is None else Estimates(2*weights, nbytes, nbytes))
  return _decode_linear(out, out_features, in_features//32, group_dot, name+QUANT_NAMES[ggml_type])

def _wmma_layout(out:UOp, out_features:int, token_tile:int, output_tiles:int, group:tuple[UOp, ...]=(), rows:int|None=None):
  # grouped experts: the token blocks cover the sorted rows (group[0]), not the output. rows: the rows of out to cover (default all)
  # output_tiles: at most that many, as many as divide the outputs
  output_tiles = next(t for t in (output_tiles, 4, 2, 1) if t <= output_tiles and out_features % (16*t) == 0)
  output_waves = 2 if out_features % (32*output_tiles) == 0 else 1
  token_block = UOp.range((group[0].shape[0] if group else rows or out.shape[0])//token_tile, 0, AxisType.GLOBAL)
  output_block = UOp.range(out_features//(16*output_tiles*output_waves), 1, AxisType.GLOBAL)
  # lane is a hardware WARP range (like the flash kernel): the fragment math stays visible without being
  # range-split into nested loops, which would scramble the WMMA fragment layout
  lane, wave = UOp.range(WARP_SIZE, -1, axis_type=AxisType.WARP), UOp.range(output_waves, 3, axis_type=AxisType.LOCAL)
  col, half = lane % 16, lane // 16
  outputs = tuple((output_block*output_waves+wave)*(16*output_tiles) + tile*16 + col for tile in range(output_tiles))
  inputs = tuple(token_block*token_tile + tile*16 + col for tile in range(token_tile//16))
  tokens = tuple(tuple(token_block*token_tile + tile*16 + half*8 + i for i in range(8)) for tile in range(token_tile//16))
  return output_waves, token_block, output_block, lane, wave, half, outputs, inputs, tokens

def _gated(gate:UOp, up:UOp, act:tuple[float, float, float]) -> UOp:
  # gated_activation in a kernel: act is (alpha, limit, up_bias), limit <= 0 for none
  alpha, limit, up_bias = act
  if limit > 0: gate, up = gate.minimum(limit), up.maximum(-limit).minimum(limit)
  return gate * (1 + (gate * (-alpha*LOG2E)).exp2()).reciprocal() * (up + up_bias)

def _act_stores(out, outputs, tokens, accs, update, act):
  # the paired output tiles are the gate and up rows of the same 16 columns (interleaved weights): the activation is stored
  assert len(outputs) == 2, "a gated activation needs the gate and up tiles of a column"
  col = outputs[0] - (outputs[0] // 32) * 16
  return [out[token, col].store(_gated(g.after(update)[i].load(), u.after(update)[i].load(), act).cast(out.dtype))
          for tile_tokens, g, u in zip(tokens, accs[0], accs[1]) for i, token in enumerate(tile_tokens)]

def _wmma_stores(out, outputs, tokens, accs, update, half, lane, wave, output_waves, rdna4):
  # RDNA4 owns eight consecutive rows per half-wave, matching tokens directly.
  if rdna4:
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
    low = half.eq(0)
    return tuple(low.where(own[i], peer[i+4]) if j == 0 else low.where(peer[i], own[i+4]) for i in range(4) for j in range(2))
  tt = len(tokens)
  return [out[token, output].store(value.cast(out.dtype)) for ot,(output,output_accs) in enumerate(zip(outputs, accs))
          for tile,(tile_tokens,_acc) in enumerate(zip(tokens, output_accs)) for token,value in zip(tile_tokens, values(ot*tt+tile))]

def _quant_linear_wmma(out, x, out_features, in_features, type_words, layout, dequant, name, rdna4, group:tuple[UOp, ...]=(),
                       expert_words:int=0, x_div:int=0, act:tuple[float, float, float]|None=None):
  # group (entries, tiles): grouped experts, the row r is the selection entries[r] (padding: -1) of the expert tiles[r//16] (-1: none).
  # x_div: the x row is entries[r]//x_div and the output row r, else the x row is r and the output row the selection
  x = x.reshape(-1, in_features)
  output_waves, token_block, output_block, lane, wave, physical_half, outputs, input_tokens, tokens = layout
  token_tile, output_tiles = len(tokens)*16, len(outputs)
  # IQ4_NL blocks are 32 wide: a device can hold part of a 256 weight row group
  output_words = in_features * type_words // GGML_BLOCK_SIZE
  accs = tuple(tuple(UOp.alloc((8,), dtypes.float32, addrspace=AddrSpace.REG)
                     for tile in range(token_tile // 16)) for ot in range(output_tiles))
  accs = tuple(tuple(acc.after(acc.store(acc.const_like(0))) for acc in output_accs) for output_accs in accs)
  estimates, expert_base = None, UOp.const(0, dtypes.weakint)
  if group:
    rows, tiles = group
    assert token_tile == MOE_TILE, "grouped experts have tiles of MOE_TILE rows"
    expert = tiles[token_block].load()
    # a tile without an expert skips its loop: no estimate from the data dependent loop
    group_rng = UOp.range((expert >= 0).where(in_features // Q8_GROUP_SIZE, 0), 4, AxisType.LOOP)
    # clamped: loads that don't depend on the loop can be hoisted above it, a tile without an expert reads expert 0
    expert_base = expert.maximum(0).cast(dtypes.weakint) * expert_words
    weights = rows.shape[0]*out_features*in_features
    estimates = Estimates(2*weights, weights*type_words*2//GGML_BLOCK_SIZE, weights*type_words*2//GGML_BLOCK_SIZE)
    x_rows = tuple((r:=rows[t].load()).valid(r >= 0) // x_div for t in input_tokens) if x_div else input_tokens
    if not x_div: tokens = tuple(tuple((r:=rows[t].load()).valid(r >= 0) for t in tile_tokens) for tile_tokens in tokens)
  else:
    group_rng = UOp.range(in_features // Q8_GROUP_SIZE, 4, AxisType.LOOP)
    x_rows = input_tokens
  block, subgroup = group_rng // 8, group_rng % 8
  wmma_accs = [list(output_accs) for output_accs in accs]
  # gfx12 moves k2 from the element index to lane bit 4: each lane supplies two groups of four.
  ks = tuple(i%4 + physical_half*4 + (i//4)*8 for i in range(8)) if rdna4 else tuple(range(16))
  for half in range(2):
    afrags = tuple(UOp.stack(*(x[input_token, group_rng*32 + half*16 + i].cast(dtypes.float16) for i in ks))
                   for input_token in x_rows)
    for output_tile,output in enumerate(outputs):
      bfrag = UOp.stack(*dequant(expert_base + output*output_words + block*type_words, subgroup, half))
      for tile,afrag in enumerate(afrags):
        previous = accs[output_tile][tile].after(group_rng) if half == 0 else wmma_accs[output_tile][tile]
        wmma_accs[output_tile][tile] = UOp.wmma(afrag, bfrag, previous, *WMMA_ARG)
  update = UOp.group(*(acc.store(value) for output_accs,output_values in zip(accs, wmma_accs)
                       for acc,value in zip(output_accs, output_values))).end(group_rng)
  if act is not None:
    assert rdna4, "the fused activation needs the RDNA4 accumulator layout"
    stores = _act_stores(out, outputs, tokens, accs, update, act)
  else: stores = _wmma_stores(out, outputs, tokens, accs, update, physical_half, lane, wave, output_waves, rdna4)
  return UOp.group(*stores).end(token_block, output_block, lane, wave).sink(arg=KernelInfo(name=name, opts_to_apply=(), estimates=estimates))

@functools.cache
def _q5_linear_f16_wmma_kernel(out:UOp, raw:UOp, x:UOp, *group:UOp, out_features:int, in_features:int, ggml_type:int, rdna4:bool=False,
                               expert_words:int=0, x_div:int=0, act:tuple|None=None) -> UOp:
  token_tile, output_tiles = (64, 1 if out_features <= 1024 else 2) if out.shape[0] % 64 == 0 else \
    (32 if out.shape[0] % 32 == 0 else 16, 2)
  if group: token_tile = MOE_TILE
  if act is not None: output_tiles = 2
  layout = _wmma_layout(out, out_features, token_tile, output_tiles, group)
  word_indices = (layout[5], layout[5]+2) if rdna4 else tuple(range(4))
  def dequant(base:UOp, subgroup:UOp, half:int) -> tuple[UOp, ...]:
    d, dmin, scale, minimum = _q5_scales(raw, base, subgroup)
    qs_base = base + (4 if ggml_type == Q4_K else 12) + (subgroup // 2)*8 + half*4
    words = tuple((raw[qs_base+i] >> ((subgroup&1)*4).cast(dtypes.uint32) & 0x0f0f0f0f) |
      (((raw[base+4+half*4+i] >> subgroup.cast(dtypes.uint32) & 0x01010101) << 4) if ggml_type == Q5_K else 0) for i in word_indices)
    return tuple(((word >> (byte*8) & 255).float()*d*scale-dmin*minimum).cast(dtypes.float16) for word in words for byte in range(4))
  return _quant_linear_wmma(out, x, out_features, in_features, QUANT_SIZES[ggml_type]//4,
                            layout, dequant, f"linear_q{4 if ggml_type == Q4_K else 5}_k_f16_wmma", rdna4, group, expert_words, x_div, act)

@functools.cache
def _iq4_linear_f16_wmma_kernel(out:UOp, raw:UOp, x:UOp, lut:UOp, *group:UOp, out_features:int, in_features:int, rdna4:bool=False,
                                expert_words:int=0, x_div:int=0, act:tuple|None=None) -> UOp:
  token_tile = 32 if out_features <= 1024 and out.shape[0] % 32 == 0 else 64 if out.shape[0] % 64 == 0 and \
    out_features <= 6144 else 128 if out.shape[0] % 128 == 0 else \
    32 if out.shape[0] % 32 == 0 else 16
  if group: token_tile = MOE_TILE
  output_tiles = 1 if out_features <= 1024 else 2 if out_features <= 6144 else 1 if out_features < 8192 else 2
  if act is not None: output_tiles = 2
  layout = _wmma_layout(out, out_features, token_tile, output_tiles, group)
  output_waves, _, _, lane, wave, half, _, _, _ = layout
  word_indices = (half, half+2) if rdna4 else tuple(range(4))
  local_lut = UOp.alloc((256,), dtypes.uint32, addrspace=AddrSpace.LOCAL)
  tid, lut_items = wave*32+lane, 256//(32*output_waves)
  lut = local_lut.after(*(local_lut[tid*lut_items+i].store(lut[tid*lut_items+i]) for i in range(lut_items)))
  def dequant(base:UOp, subgroup:UOp, half:int) -> tuple[UOp, ...]:
    scale = _iq4_scale(raw, base, subgroup)
    pairs = tuple(lut[((raw[base + 2 + subgroup*4 + word] >> (byte*8)) & 255).cast(dtypes.weakint)]
                  for word in word_indices for byte in range(4))
    return tuple((_half((pair >> (half*16)) & 0xffff)*scale).cast(dtypes.float16) for pair in pairs)
  return _quant_linear_wmma(out, x, out_features, in_features, QUANT_SIZES[IQ4_XS]//4, layout, dequant, "linear_iq4_xs_f16_wmma", rdna4,
                            group, expert_words, x_div, act)

def _wmma_scale(raw:UOp, base:UOp, subgroup:UOp, half:int, ggml_type:int) -> tuple[UOp, int, UOp]:
  # the weights of half `half` (16) of group `subgroup` (32) of a block are (q - zero)*scale - minimum, q the bytes of _quant_word
  def byte(offset): return _load_byte(raw, base, offset)
  zero, minimum = 0, UOp.const(0, dtypes.float)
  if ggml_type in (Q4_K, Q5_K):
    d, dmin, sc, mn = _q5_scales(raw, base, subgroup)
    scale, minimum = d*sc, dmin*mn
  elif ggml_type == IQ4_XS: scale = _iq4_scale(raw, base, subgroup)
  elif ggml_type == Q2_K:
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
  return scale, zero, minimum

def _i8_frag(words:tuple[UOp, ...]) -> UOp:
  # NOTE: a bitcast of the words to int8 vectors loops in devectorize, the compiler packs the bytes back
  return UOp.stack(*(((w >> (8*i)) & 0xff).cast(dtypes.uint8).bitcast(dtypes.int8) for w in words for i in range(4)))

@functools.cache
def _quant_linear_i8_wmma_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, *srcs:UOp, out_features:int, in_features:int, ggml_type:int,
                                 expert_words:int=0, x_div:int=0, act:tuple|None=None, splits:int=1) -> UOp:
  # RDNA4 int8 WMMA on q8 activations: the weights are their packed bytes, the scales apply per 16 inputs (see _wmma_scale), like the decode.
  # splits: the inputs are split in that many ranges (workgroups for few outputs), out has the rows of every split (summed after)
  grids, group = (srcs[:1], srcs[1:]) if ggml_type in (IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S) else ((), srcs)
  rows = group[0].shape[0] if group else out.shape[0] // splits
  token_tile = MOE_TILE if group else 64 if rows % 64 == 0 else 32 if rows % 32 == 0 else 16
  # outputs past a multiple of 32 (a few rows, e.g. the 24 of a hyper-connection): their tiles read the last row and aren't stored
  real_out, out_features = out_features, round_up(out_features, 32)
  layout = _wmma_layout(out, out_features, token_tile, 2, group, rows=int(rows))
  output_waves, token_block, output_block, lane, wave, ph, outputs, input_tokens, tokens = layout
  stored = outputs
  if real_out != out_features: outputs = tuple(o.minimum(real_out-1) for o in outputs)
  grid = None
  if grids:
    grid = UOp.alloc((int(grids[0].numel()),), dtypes.uint32, addrspace=AddrSpace.LOCAL)
    tid, threads = wave*32+lane, output_waves*32
    grid = grid.after(*(grid[tid+i*threads].store(grids[0][tid+i*threads]) for i in range(int(grid.numel())//threads)))
  type_words = QUANT_SIZES[ggml_type]//raw.dtype.itemsize
  output_words = in_features * type_words // GGML_BLOCK_SIZE
  n_groups, estimates, expert_base = in_features // Q8_GROUP_SIZE, None, UOp.const(0, dtypes.weakint)
  # the x row of the fragment rows (input_tokens) and of the accumulator rows (tokens), and the output row of the accumulator rows
  def xrow(r:UOp) -> UOp: return r
  store_rows = tokens
  if group:
    entries, tiles = group
    expert = tiles[token_block].load()
    group_rng = UOp.range((expert >= 0).where(n_groups, 0), 4, AxisType.LOOP)
    # clamped: loads that don't depend on the loop can be hoisted above it, a tile without an expert reads expert 0
    expert_base = expert.maximum(0).cast(dtypes.weakint) * expert_words
    weights = rows*out_features*in_features
    estimates = Estimates(2*weights, weights*type_words*2//GGML_BLOCK_SIZE, weights*type_words*2//GGML_BLOCK_SIZE)
    if x_div:
      def xrow(r:UOp) -> UOp: return (e:=entries[r].load()).valid(e >= 0) // x_div
    else: store_rows = tuple(tuple((e:=entries[t].load()).valid(e >= 0) for t in tile_tokens) for tile_tokens in tokens)
  else: group_rng = UOp.range(n_groups // splits, 4, AxisType.LOOP)
  split = UOp.range(splits, 5, AxisType.GLOBAL)
  g = group_rng + split * (n_groups // splits)
  block, subgroup = g // 8, g % 8
  accs = [[UOp.alloc((8,), dtypes.float32, addrspace=AddrSpace.REG) for _ in tokens] for _ in outputs]
  accs = [[acc.after(acc.store(acc.const_like(0))) for acc in row] for row in accs]
  a_rows = [xrow(t) for t in input_tokens]
  s_rows = [[xrow(t) for t in tile_tokens] for tile_tokens in tokens]
  d_x = [[xd[r, g].load() for r in tile_rows] for tile_rows in s_rows]
  zero_acc = UOp.stack(*[UOp.const(0, dtypes.int32)]*8)
  new_vals = [[[accs[o][t].after(group_rng)[j].load() for j in range(8)] for t in range(len(tokens))] for o in range(len(outputs))]
  # one scale per 32 inputs: both halves accumulate in int32 and the group is scaled once, else every half has its own scale
  halves = ((0, 1),) if ggml_type in GROUP_SCALE_QUANTS else ((0,), (1,))
  for hs in halves:
    afrags = [[_i8_frag((xq[r, g, half*4+ph].load(), xq[r, g, half*4+2+ph].load())) for half in hs] for r in a_rows]
    s_x = [[sum((xs[r, g, half].load() for half in hs), UOp.const(0, dtypes.float)) for r in tile_rows] for tile_rows in s_rows]
    for o, output in enumerate(outputs):
      base = expert_base + output*output_words + block*type_words
      if ggml_type in SMALL_BLOCK_QUANTS: base = base + subgroup*SMALL_BLOCK_QUANTS[ggml_type]
      scale, zero, minimum = _wmma_scale(raw, base, subgroup, hs[0], ggml_type)
      bfrags = [_i8_frag(tuple(_quant_word(raw, base, subgroup, half*4+i, ggml_type, grid) for i in (ph, ph+2))) for half in hs]
      for t, tile_afrags in enumerate(afrags):
        acc_i = zero_acc
        for afrag, bfrag in zip(tile_afrags, bfrags): acc_i = UOp.wmma(afrag, bfrag, acc_i, *WMMA_ARG)
        dot = UOp.alloc((8,), dtypes.int32, addrspace=AddrSpace.REG)
        dot = dot.after(dot.store(acc_i))
        for j in range(8):
          sx = s_x[t][j]
          term = scale * dot[j].load().float() if not zero else scale * (dot[j].load().float() - zero*sx)
          new_vals[o][t][j] = new_vals[o][t][j] + d_x[t][j] * (term if ggml_type not in (Q4_K, Q5_K, Q2_K) else term - minimum*sx)
  update = UOp.group(*(accs[o][t].store(UOp.stack(*new_vals[o][t])) for o in range(len(outputs)) for t in range(len(tokens)))).end(group_rng)
  if act is not None: stores = _act_stores(out, outputs, store_rows, accs, update, act)
  else: stores = [out[row + split * rows, output.valid(output < real_out) if real_out != out_features else output].store(
                    accs[o][t].after(update)[j].load().cast(out.dtype))
                  for o, output in enumerate(stored) for t, tile_rows in enumerate(store_rows) for j, row in enumerate(tile_rows)]
  return UOp.group(*stores).end(token_block, output_block, split, lane, wave).sink(
    arg=KernelInfo(name=f"linear_{QUANT_NAMES[ggml_type]}_i8_wmma", opts_to_apply=(), estimates=estimates))

@functools.cache
def _quant_linear_f16_wmma_kernel(out:UOp, raw:UOp, x:UOp, *srcs:UOp,
                                  out_features:int, in_features:int, ggml_type:int, rdna4:bool=False, expert_words:int=0, x_div:int=0,
                                  act:tuple|None=None) -> UOp:
  # srcs: the iq grid if the format has one, then the expert group (rows, tiles) if grouped
  grids, group = (srcs[:1], srcs[1:]) if ggml_type in (IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S) else ((), srcs)
  token_tile = MOE_TILE if group else 64 if out.shape[0] % 64 == 0 else 32 if out.shape[0] % 32 == 0 else 16
  # four output tiles of a wave share its token fragments (two: grouped experts, a tile's rows are few)
  layout = _wmma_layout(out, out_features, token_tile, 2 if group else 4, group)
  output_waves, _, _, lane, wave, physical_half, _, _, _ = layout
  word_indices = (physical_half, physical_half+2) if rdna4 else tuple(range(4))
  grid = None
  if grids:
    grid = UOp.alloc((int(grids[0].numel()),), dtypes.uint32, addrspace=AddrSpace.LOCAL)
    tid, threads = wave*32+lane, output_waves*32
    grid = grid.after(*(grid[tid+i*threads].store(grids[0][tid+i*threads]) for i in range(int(grid.numel())//threads)))
  def dequant(base:UOp, subgroup:UOp, half:int) -> tuple[UOp, ...]:
    if ggml_type in SMALL_BLOCK_QUANTS: base += subgroup*SMALL_BLOCK_QUANTS[ggml_type]  # eight independent blocks per 256 weights
    scale, zero, minimum = _wmma_scale(raw, base, subgroup, half, ggml_type)
    words = tuple(_quant_word(raw, base, subgroup, half*4+i, ggml_type, grid) for i in word_indices)
    return tuple((((word >> (b*8)).cast(dtypes.uint8).bitcast(dtypes.int8).float()-zero)*scale-minimum).cast(dtypes.half)
                 for word in words for b in range(4))
  return _quant_linear_wmma(out, x, out_features, in_features, QUANT_SIZES[ggml_type]//raw.dtype.itemsize,
                            layout, dequant, f"linear_{QUANT_NAMES[ggml_type]}_f16_wmma", rdna4, group, expert_words, x_div, act)

def q8_linear(layer:Linear, x:Tensor, reduce:bool=True, xq:tuple[Tensor, ...]|None=None, split:bool=False) -> Tensor:
  assert layer.ggml_type in QUANT_SIZES
  tokens = int(x.numel()) // layer.in_features
  out_features, in_features = layer.out_features, int(x.uop.shard_shape[-1])
  splits = layer.in_features // in_features
  assert (layer.shard_axis == 1) == (splits > 1), f"input features split over {splits} devices, weight split on axis {layer.shard_axis}"
  out_shape:tuple[int, ...] = (splits*tokens, out_features)
  fxn:Callable[..., UOp]
  ksplits = 1  # the input groups split over workgroups, their partial outputs summed after
  extra = (_iq_grid(x.device, layer.ggml_type),) if layer.ggml_type in (IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S) else ()
  # the kernels write the output features of their own device
  local_out = out_features // len(dev) if isinstance(dev:=layer.weight.device, tuple) and layer.shard_axis == 0 else out_features
  if tokens % 16 == 0 and local_out % 32 == 0 and xq is None and _use_i8_wmma(x.device, layer.ggml_type):
    fxn = functools.partial(_quant_linear_i8_wmma_kernel, ggml_type=layer.ggml_type)
    srcs = (*q8_quantize(x, tokens, in_features)[:3], *extra)
  elif tokens >= 64 and tokens % 16 == 0 and xq is not None and layer.ggml_type in GROUP_SCALE_QUANTS and layer.shard_axis is None and \
       _wmma_rdna4(x.device) and not split:
    # many tokens already quantized (a few outputs, e.g. a hyper-connection's): the int8 tensor cores, the inputs split over workgroups
    ksplits = next(s for s in (16, 8, 4, 2, 1) if (in_features // Q8_GROUP_SIZE) % s == 0 and in_features // Q8_GROUP_SIZE // s >= 8)
    fxn = functools.partial(_quant_linear_i8_wmma_kernel, ggml_type=layer.ggml_type, splits=ksplits)
    srcs = (*xq[:3], *extra)
    out_shape = (ksplits*tokens, out_features)
  elif tokens % 16 == 0 and local_out % 16 == 0 and xq is None:
    if layer.ggml_type == IQ4_XS:
      fxn, extra = _iq4_linear_f16_wmma_kernel, (iq4_half_lut(x.device),)
    else:
      fxn = functools.partial(_q5_linear_f16_wmma_kernel if layer.ggml_type in (Q4_K, Q5_K) else _quant_linear_f16_wmma_kernel,
                              ggml_type=layer.ggml_type)
    fxn = functools.partial(fxn, rdna4=_wmma_rdna4(x.device))
    srcs = (x.cast(dtypes.float16).contiguous(), *extra)
  else:
    fxn = functools.partial(_quant_decode_kernel, ggml_type=layer.ggml_type)
    srcs = (*(xq or q8_quantize(x, tokens, in_features))[:3], *extra)
  # split: x has its rows split over the devices (the weight is whole on every device), so has the output
  out_axis = 0 if split and layer.shard_axis is None else None if layer.shard_axis is None else 1-layer.shard_axis
  out = Tensor.empty(out_shape, dtype=dtypes.float32, device=x.device, axis=out_axis)
  result = Tensor.custom_kernel(out, layer.weight, *srcs, fxn=functools.partial(fxn, out_features=local_out, in_features=in_features))[0]
  if ksplits > 1: result = result.reshape(ksplits, tokens, out_features).sum(0)
  # row parallel: every device wrote the partial sum of its input features, the sum over the devices is the allreduce
  if splits > 1:
    if not reduce: return result.reshape(splits, *x.shape[:-1], out_features)  # the partial sums, sharded on the device axis
    result = partial_sum(result.reshape(splits, tokens, out_features))
  result = result.reshape(*x.shape[:-1], out_features)
  return result if layer.bias is None else result + layer.bias

# ******** allreduce through host memory, in a kernel ********

# a queue level allreduce through a hub costs copies and cross device signals (~45us), a kernel on every device exchanges through a mailbox in
# host memory instead (every device maps it): it writes its partial, then spins until the partials of all the devices are there and sums them
HOST_AR_THREADS, HOST_AR_MAXN = 256, 1 << 16

class HostAllreduce:
  def __init__(self, devs:tuple[str, ...]):
    from tinygrad.device import Buffer, BufferSpec, BufferStorage, MultiBuffer
    def multi(bufs:list, dtype) -> Tensor:
      (mb:=MultiBuffer.__new__(MultiBuffer)).bufs = bufs
      return Tensor(UOp.from_buffer(mb, dtype, devs))
    N, self.groups = len(devs), HOST_AR_MAXN // HOST_AR_THREADS
    nbytes = 2*N*HOST_AR_MAXN*8  # two parities of a slot per device, 64-bit words: the value and the epoch
    with Context(ALLOW_DEVICE_USAGE=1):
      host = Buffer(devs[0], nbytes, options=BufferSpec(host=True, uncached=True, cpu_access=True)).ensure_allocated()
      host.as_memoryview(allow_zero_copy=True)[:] = bytes(nbytes)
      mbox = [host] + [Buffer(d, nbytes, opaque=BufferStorage(host.get_buf(d))) for d in devs[1:]]
      # the epoch of every workgroup, then the index of the device
      counters = [Buffer(d, (self.groups+1)*4, initial_value=bytes(self.groups*4)+i.to_bytes(4, "little")) for i,d in enumerate(devs)]
    self.mbox, self.counter = multi(mbox, dtypes.uint64), multi(counters, dtypes.int32)

@functools.cache
def host_allreduce(devs:tuple[str, ...]) -> HostAllreduce|None:
  if not getenv("HOST_ALLREDUCE", 1) or len(devs) < 2 or not amd_custom_kernels_supported(devs): return None
  return HostAllreduce(devs)

@functools.cache
def _host_allreduce_kernel(out:UOp, x:UOp, mbox:UOp, counter:UOp, N:int, n:int) -> UOp:
  # workgroup g owns its epoch counter and the elements g*THREADS..: an epoch in every word, a reader spins until the word has its epoch.
  # the parity of the epoch picks the slot: a device can't get two allreduces ahead, the next one needs the partials of all the devices.
  # every device sums the same values in the same order: the same bits everywhere
  wg, t = UOp.range(n // HOST_AR_THREADS, 0, AxisType.GLOBAL), UOp.range(HOST_AR_THREADS, 1, AxisType.LOCAL)
  e = wg * HOST_AR_THREADS + t
  epoch, me = counter[wg].load() + 1, counter[HOST_AR_MAXN // HOST_AR_THREADS].load()
  def slot(p:UOp|int) -> UOp: return ((epoch & 1) * N + p) * HOST_AR_MAXN + e
  val = x[0, e].load().float()
  # a value and its epoch in one 64-bit word (one store, one load): no torn reads
  mbox = mbox.replace(arg=replace(mbox.arg, volatile=True))
  write = mbox[slot(me)].store((epoch.cast(dtypes.uint64) << 32) | val.bitcast(dtypes.uint32).cast(dtypes.uint64))
  # spin until the words of all the devices have this epoch. the loop comes after the write: it is issued before this device waits
  loop = UOp.range(UOp(Ops.NOOP).after(write), next(UOp.unique_num), dtype=dtypes.void)
  words = UOp.alloc((N,), dtypes.uint64, addrspace=AddrSpace.REG)
  loaded = words.after(loop).store(UOp.stack(*(mbox.after(loop)[slot(p)].load() for p in range(N))))
  waiting = functools.reduce(lambda a,b: a | b, [(words.after(loaded)[p].load() >> 32).cast(dtypes.uint32).ne(epoch.cast(dtypes.uint32))
                                                for p in range(N)])
  pause = UOp(Ops.CUSTOM, src=(waiting,), arg=("__builtin_amdgcn_s_sleep(1);", dtypes.void))
  words = words.after(pause.backedge(loop, waiting))
  def read(p:int) -> UOp: return (words[p].load() & 0xFFFFFFFF).cast(dtypes.uint32).bitcast(dtypes.float32)
  total = functools.reduce(lambda a,b: a+b, [me.eq(p).where(val, read(p)) for p in range(N)])
  store = out[e].store(total)
  # the next epoch is stored once every thread of the workgroup read the counter
  done = counter.after(UOp.group(store).barrier())[wg.valid(t.eq(0))].store(epoch)
  return UOp.group(store, done).end(wg, t).sink(arg=KernelInfo(name="host_allreduce", opts_to_apply=()))

# ******** sequence parallel: the rows of a tensor split over the devices ********

def token_sharded(x:Tensor) -> bool:
  """x is split over devices on an axis before its last: its rows if the caller says so (not a leading device axis of partials)"""
  return isinstance(x.device, tuple) and (axis:=x.uop.axis) is not None and axis < x.ndim - 1

def _per_device(x:Tensor) -> UOp:
  # the value on every device of a split kernel output (_materialized): its buffer's view of the device, after the kernel
  u = x.uop
  assert u.op is Ops.AFTER and u.src[0].op is Ops.UNSHARD, f"not a split kernel output: {u.op}"
  return u.src[0].src[0].after(*u.src[1:])

def _cast_kernel(out:UOp, x:UOp) -> UOp:
  o, s = out.flatten(), x.flatten()
  i = UOp.range(o.shape[0], 0)
  return o[i].store(s[i].load().cast(o.dtype)).end(i).sink(arg=KernelInfo(name="cast"))

def _materialized(x:Tensor, dtype) -> Tensor:
  """
  x (split over devices) cast into a new buffer by a kernel: x and what it is after (e.g. the stores of a recurrent state) are the kernel's
  input, the result is just a buffer (its per device values are taken apart without its producers)
  """
  out = Tensor.empty(*x.shape, dtype=dtype, device=x.device, axis=x.uop.axis)
  return Tensor.custom_kernel(out, x, fxn=_cast_kernel)[0]

def shard_rows(x:Tensor, axis:int) -> Tensor:
  """a tensor on every device split on axis: each device keeps its part, nothing is copied"""
  return Tensor(x.uop._shard(axis, UOp.range(len(cast(tuple, x.device)), 0, AxisType.DEVICE)).unshard(axis))

def reduce_scatter(p:Tensor, axis:int) -> Tensor:
  """
  the sum over the leading device axis of p (devices, *S), split on it, as S split on axis (of S): device i sums part i of every device's
  partial. the parts cross in half through host memory, every device on its own link, the sums in float
  """
  from tinygrad.schedule.allreduce import _host_of
  devs = cast(tuple[str, ...], p.device)
  n, S, local = len(devs), p.shape[1:], _per_device(_materialized(p, dtypes.half))
  size, host = S[axis] // n, _host_of(devs[0])
  def part(j:int, i:int) -> UOp:
    return local.mselect(j).reshape(S).shrink(tuple((0, s) if a != axis else (i*size, (i+1)*size) for a, s in enumerate(S)))
  # summed in half, in device order: the sums of the allreduce of partial_sum
  parts = [functools.reduce(lambda a, b: a + b, [part(j, i).cast(dtypes.half) if j == i else
                            part(j, i).cast(dtypes.half).contiguous().copy_to_device(host).copy_to_device(devs[i]) for j in range(n)]).float()
           for i in range(n)]
  return Tensor(UOp.mstack(*[x.contiguous() for x in parts]).unshard(axis))

def all_gather(x:Tensor, axis:int, dtype=None) -> Tensor:
  """a tensor split on axis over the devices, whole on every device: every part goes through host memory once (in dtype)"""
  from tinygrad.schedule.allreduce import _host_of
  devs = cast(tuple[str, ...], x.device)
  local = _per_device(_materialized(x, dtype or x.dtype))
  mine = [local.mselect(i).contiguous() for i in range(len(devs))]
  staged = [m.copy_to_device(_host_of(devs[0])) for m in mine]
  whole = [Tensor.cat(*[Tensor(mine[i] if i == j else staged[i].copy_to_device(devs[j])) for i in range(len(devs))], dim=axis).uop
           for j in range(len(devs))]
  return Tensor(UOp.mstack(*whole))

def partial_sum(x:Tensor) -> Tensor:
  """the sum over the leading device axis (an allreduce). many tokens (prefill) are summed in half: the transfers are half the size"""
  n = prod(x.shape[1:])
  if isinstance(x.device, tuple) and isinstance(n, int) and n % HOST_AR_THREADS == 0 and n <= HOST_AR_MAXN and (comm:=host_allreduce(x.device)):
    out = Tensor.empty(n, dtype=dtypes.float32, device=x.device)
    fxn = functools.partial(_host_allreduce_kernel, N=len(x.device), n=n)
    return Tensor.custom_kernel(out, x.float().reshape(x.shape[0], n).contiguous(), comm.mbox, comm.counter, fxn=fxn)[0].reshape(x.shape[1:])
  if prod(x.shape[1:-1]) > 1 and getenv("ALLREDUCE_HALF", 1): return x.half().sum(0, dtype=dtypes.half).float()
  return x.sum(0)

# ******** mixture of experts: the selected experts of each token ********

@functools.cache
def _moe_router_kernel(sel:UOp, probs:UOp, logits:UOp, bias:UOp, k:int, scale:float) -> UOp:
  # sigmoid gating, a workgroup per token and a thread per expert: the rank of an expert is the count of those below it (a lower score, or
  # the same and a higher index), the top k are the ranks n-k..n-1 in that order (as pairwise_topk). the probabilities are normalized
  n = cast(int, logits.shape[-1])
  tokens = cast(int, logits.numel()) // n
  logits, sel, probs = logits.reshape(tokens, n), sel.reshape(tokens, k), probs.reshape(tokens, k)
  token, i = UOp.range(tokens, 0, AxisType.GLOBAL), UOp.range(n, 1, AxisType.LOCAL)
  score = (1 + (logits[token, i].load().float() * (-1/math.log(2))).exp2()).reciprocal()
  key = score + bias[i].load().float()
  lds = UOp.alloc((n,), dtypes.float32, addrspace=AddrSpace.LOCAL)
  lds = lds.after(lds[i].store(key).barrier())
  j = UOp.range(n, 2, AxisType.LOOP)
  other = lds[j].load()
  below = ((key > other) | (key.eq(other) & (i < j))).where(UOp.const(1, dtypes.int32), UOp.const(0, dtypes.int32))
  rank = _reg_int(below, j)
  slot = rank - (n - k)
  picked = slot >= 0
  lds_p = UOp.alloc((k,), dtypes.float32, addrspace=AddrSpace.LOCAL)
  lds_p = lds_p.after(UOp.group(lds_p[slot.valid(picked)].store(score), sel[token, slot.valid(picked)].store(i.cast(dtypes.int32))).barrier())
  total = functools.reduce(lambda a,s: a + lds_p[s].load(), range(1, k), lds_p[0].load())
  out = probs[token, i.valid(i < k)].store(lds_p[i.minimum(k-1)].load() / total * scale)
  return UOp.group(out, lds_p).end(token, i).sink(arg=KernelInfo(name="moe_router", opts_to_apply=()))

def _reg_int(value:UOp, rng:UOp) -> UOp:
  # the sum of value over the loop rng in an int register
  acc = UOp.alloc((1,), dtypes.int32, addrspace=AddrSpace.REG)
  acc = acc.after(acc.store(acc.const_like(0)))
  return acc.after(acc[0].store(acc.after(rng)[0].load() + value).end(rng))[0].load()

def moe_router(logits:Tensor, bias:Tensor, k:int, scale:float) -> tuple[Tensor, Tensor]:
  """sigmoid gating with a selection bias, normalized: the top k experts (B, T, k) int32 and their probabilities times scale"""
  shape = (*logits.shape[:-1], k)
  sel = Tensor.empty(*shape, dtype=dtypes.int32, device=logits.device)
  probs = Tensor.empty(*shape, dtype=dtypes.float32, device=logits.device)
  fxn = functools.partial(_moe_router_kernel, k=k, scale=scale)
  sel, probs = Tensor.custom_kernel(sel, probs, logits.contiguous(), bias, fxn=fxn)[:2]
  return sel, probs

@functools.cache
def _moe_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, sel:UOp, first:UOp, *grids:UOp,
                       out_features:int, in_features:int, ggml_type:int, experts:int, slots:int, x_slots:int) -> UOp:
  # out rows are (token, slot): the slot's expert reads its weight rows, experts of other devices read nothing
  expert_words = out_features * in_features * QUANT_SIZES[ggml_type] // 256 // raw.dtype.itemsize
  def local(row:UOp) -> UOp: return ((expert:=sel[row].load() - first[0].load()) >= 0) & (expert < experts)
  def gate(row:UOp, output:UOp, base:UOp) -> UOp: return base + (sel[row].load() - first[0].load()).cast(dtypes.weakint) * expert_words
  def xtoken(row:UOp) -> UOp: return row if x_slots == slots else row // slots
  return _quant_decode_body(out, raw, xq, xd, xs, grids, out_features, in_features, ggml_type, gate, xtoken, "moe_", active=local)

def moe_linear(weight:Tensor, ggml_type:int, experts:int, x:Tensor, sel:Tensor, first:Tensor) -> Tensor:
  """
  weight: packed (experts, out, in) local to each device, x: (..., x_slots, in), sel: (B, T, slots) expert ids, first: the first expert of a device.
  returns (..., slots, out). on devices holding some of the experts the output is (devices, B, T, slots, out), every device has its own partial
  """
  B, T, slots = sel.shape[-3:]  # sel can have a leading device axis: the selections of every device
  in_features, x_slots, tokens = int(x.uop.shard_shape[-1]), int(x.shape[-2]), int(B*T)
  out_features = weight.uop.max_shard_shape[0] * weight.dtype.itemsize * 256 // QUANT_SIZES[ggml_type] // in_features // experts
  devices = len(weight.device) if isinstance(weight.device, tuple) else 1
  extra = (_iq_grid(x.device, ggml_type),) if ggml_type in (IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S) else ()
  xq = q8_quantize(x.contiguous(), tokens*x_slots, in_features)
  # a grid over the (token, slot) rows, or a workgroup loops over the slots of a token
  grid = getenv("MOE_GRID", 1)
  out = Tensor.empty((devices*tokens*slots, out_features) if grid else (devices*tokens, slots, out_features), dtype=dtypes.float32,
                     device=x.device, axis=0 if devices > 1 else None)
  fxn = functools.partial(_moe_decode_kernel, out_features=out_features, in_features=in_features, ggml_type=ggml_type, experts=experts,
                          slots=slots, x_slots=x_slots)
  result = Tensor.custom_kernel(out, weight, *xq, sel.cast(dtypes.int32).flatten(), first, *extra, fxn=fxn)[0]
  return result.reshape(*((devices,) if devices > 1 else ()), B, T, slots, out_features)

MOE_TILE = getenv("MOE_TILE", 64)  # grouped experts: rows of an expert are padded to whole tiles, a workgroup decodes an expert once per tile

def moe_grouped_linear(weight:Tensor, ggml_type:int, experts:int, x:Tensor, entries:Tensor, tiles:Tensor, slots:int=0, selections:int=0,
                       act:tuple[float, float, float]|None=None, out_dtype=dtypes.float32) -> Tensor:
  """
  grouped experts (prefill): every expert reads its weights once for all its rows, the selections sorted by expert (moe_route). entries (R,): the
  selection of each row (-1: padding), tiles (R/16,): the local expert of each tile of 16 rows (-1: none).
  slots: x has a row per token, the row of a selection is selection//slots, the output (R, out) is sorted.
  else: x has a row per sorted row, the output (selections, out) has a row per selection (the rows of other devices' experts aren't written).
  on devices holding some of the experts, every input has a leading device axis and so does the output.
  act (alpha, limit, up_bias): the weight rows are gate and up interleaved by 16 (fuse_gate_up), the output is their gated activation in half
  """
  devices = len(weight.device) if isinstance(weight.device, tuple) else 1
  rows = entries
  R, in_features = int(rows.uop.max_shard_shape[-1]), int(x.uop.shard_shape[-1])
  out_features = weight.uop.max_shard_shape[0] * weight.dtype.itemsize * 256 // QUANT_SIZES[ggml_type] // in_features // experts
  expert_words = out_features * in_features * QUANT_SIZES[ggml_type] // 256 // weight.dtype.itemsize
  extra:tuple[Tensor, ...] = (_iq_grid(x.device, ggml_type),) if ggml_type in (IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S) else ()
  fxn:Callable[..., UOp]
  group = (rows.cast(dtypes.int32).flatten().contiguous(), tiles.cast(dtypes.int32).flatten().contiguous())
  rows_out, cols = R if slots else selections, out_features // 2 if act else out_features
  out = Tensor.empty((devices*rows_out, cols), dtype=dtypes.half if act else out_dtype, device=x.device, axis=0 if devices > 1 else None)
  if _use_i8_wmma(x.device, ggml_type):
    x_rows = int(x.uop.shard_shape[0]) if x.ndim == 2 else prod(x.uop.shard_shape[:-1])
    fxn = functools.partial(_quant_linear_i8_wmma_kernel, ggml_type=ggml_type, out_features=out_features, in_features=in_features,
                            expert_words=expert_words, x_div=slots, act=act)
    result = Tensor.custom_kernel(out, weight, *q8_quantize(x.contiguous(), int(x_rows), in_features)[:3], *extra, *group, fxn=fxn)[0]
    return result.reshape(devices, rows_out, cols) if devices > 1 else result
  if ggml_type == IQ4_XS: fxn, extra = _iq4_linear_f16_wmma_kernel, (iq4_half_lut(x.device),)
  else: fxn = functools.partial(_q5_linear_f16_wmma_kernel if ggml_type in (Q4_K, Q5_K) else _quant_linear_f16_wmma_kernel, ggml_type=ggml_type)
  fxn = functools.partial(fxn, rdna4=_wmma_rdna4(x.device), out_features=out_features, in_features=in_features, expert_words=expert_words,
                          x_div=slots, act=act)
  srcs = (x.cast(dtypes.float16).contiguous(), *extra, *group)
  result = Tensor.custom_kernel(out, weight, *srcs, fxn=fxn)[0]
  return result.reshape(devices, rows_out, cols) if devices > 1 else result

def fuse_gate_up(gate:Tensor, up:Tensor, experts:int, rows:int) -> Tensor:
  """
  the packed (flat) weights of all `experts` with `rows` rows each, gate and up rows interleaved by 16: [gate 0..15, up 0..15, gate 16..31, ...].
  the expert axis keeps its split over devices
  """
  g, u = (w.reshape(experts, rows // 16, -1) for w in (gate, up))
  return g.stack(u, dim=2).flatten()

@functools.cache
def _exclusive_cumsum_kernel(out:UOp, x:UOp, block:int=64) -> UOp:
  # x, out (n, T, C) int32: out[i, t, c] is the sum of x[i, :t, c]. a workgroup per column, a thread per block of rows: the totals of the
  # blocks before it through LDS, then the block's rows
  n, T, C = cast(tuple[int, int, int], x.shape)
  blocks = T // block
  col, b = UOp.range(n*C, 0, AxisType.GLOBAL), UOp.range(blocks, 1, AxisType.LOCAL)
  i, c = col // C, col % C
  vals = [x[i, b*block + r, c].load() for r in range(block)]
  lds = UOp.alloc((blocks,), dtypes.int32, addrspace=AddrSpace.LOCAL)
  lds = lds.after(lds[b].store(functools.reduce(lambda a,v: a+v, vals[1:], vals[0])).barrier())
  start = functools.reduce(lambda a,j: a + (j < b).where(lds[j].load(), UOp.const(0, dtypes.int32)), range(blocks), UOp.const(0, dtypes.int32))
  stores, run = [], start
  for r in range(block):
    stores.append(out[i, b*block + r, c].store(run))
    run = run + vals[r]
  return UOp.group(*stores).end(col, b).sink(arg=KernelInfo(name="exclusive_cumsum", opts_to_apply=()))

def exclusive_cumsum(x:Tensor) -> Tensor:
  """(n, T, C) int32: the sum of the rows before every row"""
  out = Tensor.empty(*x.shape, dtype=dtypes.int32, device=x.device, axis=x.uop.axis)
  return Tensor.custom_kernel(out, x.cast(dtypes.int32).contiguous(), fxn=_exclusive_cumsum_kernel)[0]

@functools.cache
def _scatter_rows_kernel(out:UOp, pos:UOp) -> UOp:
  # out[pos[i]] = i for every i with pos[i] >= 0 (pos is injective)
  i = UOp.range(pos.shape[0], 0, AxisType.GLOBAL)
  p = pos[i].load()
  return out[p.valid(p >= 0)].store(i.cast(dtypes.int32)).end(i).sink(arg=KernelInfo(name="scatter_rows", opts_to_apply=()))

def moe_route(sel:Tensor, first:Tensor, experts:int) -> tuple[Tensor, Tensor, Tensor]:
  """
  sorts the (token, slot) selections of sel (B, T, k) by local expert, each expert's rows padded to whole tiles.
  returns valid (..., B*T*k) if a selection is an expert of this device, entries (..., R) the selection of each row (-1: padding)
  and tiles (..., R/MOE_TILE) the local expert of each tile (-1: none). first: the first expert of each device, a leading device axis if sharded
  """
  k, N = sel.shape[-1], int(prod(sel.shape[-3:]))  # sel can have a leading device axis
  tokens, R = N // k, round_up(N + (MOE_TILE-1)*experts, MOE_TILE)
  e = sel.reshape(-1, tokens, k).cast(dtypes.int32) - first.reshape(-1, 1, 1)
  valid = (e >= 0) & (e < experts)
  onehot = (e.unsqueeze(-1) == Tensor.arange(experts, dtype=dtypes.int32)) & valid.unsqueeze(-1)  # (n, tokens, k, experts)
  # a token selects an expert at most once: the rank of a selection is the count of earlier tokens of its expert
  per_token = onehot.sum(2, dtype=dtypes.int32)  # (n, tokens, experts)
  earlier = exclusive_cumsum(per_token) if tokens % 64 == 0 and amd_custom_kernels_supported(sel.device) else per_token.cumsum(1) - per_token
  padded = (per_token.sum(1) + MOE_TILE-1) // MOE_TILE * MOE_TILE
  offsets = padded.cumsum(-1) - padded
  pos = valid.where(onehot.where((offsets.unsqueeze(1) + earlier).unsqueeze(2), 0).sum(-1), -1).reshape(-1, N)
  # the selection of every sorted row: a scatter of the positions into rows of -1 (padding)
  n = pos.shape[0]
  entries = Tensor.full((n, R), -1, dtype=dtypes.int32)
  entries = (entries.shard(first.device, 0) if isinstance(first.device, tuple) else entries.to(first.device)).contiguous()
  entries = Tensor.custom_kernel(entries.reshape(-1), pos.cast(dtypes.int32).reshape(-1).contiguous(), fxn=_scatter_rows_kernel)[0].reshape(n, R)
  start = Tensor.arange(R // MOE_TILE, dtype=dtypes.int32) * MOE_TILE
  inside = (offsets.unsqueeze(1) <= start.unsqueeze(-1)) & (start.unsqueeze(-1) < (offsets + padded).unsqueeze(1))
  tiles = inside.where(Tensor.arange(experts, dtype=dtypes.int32) + 1, 0).sum(-1) - 1
  return valid.reshape(-1, N), entries, tiles.cast(dtypes.int32)

# ******** manifold-constrained hyper-connections: the mixing coefficients ********

def _swizzle_xor(val:UOp, mask:int) -> UOp:
  swizzle = f"__builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, {{0}}), {0x1f | mask<<10})"
  return UOp(Ops.CUSTOM, src=(val,), arg=(f"__builtin_bit_cast(float, {swizzle})", dtypes.float))

@functools.cache
def _hc_coeffs_kernel(out:UOp, mix:UOp, sq:UOp, base:UOp, scale:UOp, count:int, iters:int, eps:float, norm_eps:float, features:int) -> UOp:
  # one wave per token. the mix is a linear layer of the streams before their rms norm: scale it by the rms from the sums of squares.
  # lanes [0, count*count) hold comb[lane//count][lane%count], the sums over a row/column are lane swizzles. the next 2*count lanes pre and post
  assert count == 4, "the swizzles assume 4 streams"
  token, lane = UOp.range(out.shape[0], 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP)
  groups = sq.shape[1]
  total = sum((sq[token, (lane + 32*i).valid(lane + 32*i < groups)].load() for i in range((groups+31)//32)), UOp.const(0, dtypes.float))
  rms = (warp_reduce(total, full_wave=True) * (1/features) + norm_eps).rsqrt()
  is_comb = lane < count*count
  idx = is_comb.where(lane + 2*count, (lane - count*count).minimum(2*count-1).maximum(0))
  which = is_comb.where(2, (lane < count*count + count).where(0, 1))
  s = which.eq(0).where(scale[0].load(), which.eq(1).where(scale[1].load(), scale[2].load()))
  v = mix[token, idx].load() * rms * s + base[idx].load()
  # pre, post
  sig = (1 + (v * -LOG2E).exp2()).reciprocal()
  prepost = which.eq(0).where(sig + eps, sig * 2)
  # comb: softmax over the row, then sinkhorn: normalize the columns then the rows, alternating
  def rowsum(x:UOp, op=lambda a,b: a+b) -> UOp: return op(y:=op(x, _swizzle_xor(x, 1)), _swizzle_xor(y, 2))
  def colsum(x:UOp) -> UOp: return (y:=x + _swizzle_xor(x, 4)) + _swizzle_xor(y, 8)
  e = ((v - rowsum(v, UOp.maximum)) * LOG2E).exp2()
  c = e * rowsum(e).reciprocal() + eps
  for it in range(2 * iters - 1): c = c * ((colsum(c) if it % 2 == 0 else rowsum(c)) + eps).reciprocal()
  valid = lane < count*count + 2*count
  out_idx = is_comb.where(lane + 2*count, lane - count*count)
  return out[token, out_idx.valid(valid)].store(is_comb.where(c, prepost)).end(token, lane).sink(
    arg=KernelInfo(name="hc_coeffs", opts_to_apply=()))

def hc_coeffs(mix:Tensor, sq:Tensor, base:Tensor, scale:Tensor, count:int, iters:int, eps:float, norm_eps:float,
              features:int, split:bool=False) -> tuple[Tensor, Tensor, Tensor]:
  """
  pre (..., count), post (..., count) and comb (..., count, count) of mHC from the (..., (2+count)*count) mix of the streams before their rms norm,
  and the sums of squares of the streams in (rows, groups)
  """
  rows = prod(mix.shape[:-1])
  out = Tensor.empty(rows, mix.shape[-1], dtype=dtypes.float32, device=mix.device, axis=0 if split else None)
  fxn = functools.partial(_hc_coeffs_kernel, count=count, iters=iters, eps=eps, norm_eps=norm_eps, features=features)
  out = Tensor.custom_kernel(out, mix.float().reshape(rows, -1).contiguous(), sq, base.float().contiguous(), scale.float().contiguous(), fxn=fxn)[0]
  out = out.reshape(*mix.shape[:-1], -1)
  return out[..., :count], out[..., count:2*count], out[..., 2*count:].reshape(*mix.shape[:-1], count, count)

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
def _amd_flash_attention_decode_partial(out, stats, q, cache_kv, *mask, valid_kv_len, max_kv_len, block_n, waves=4, group=None, shared=False,
                                        scale=None, causal_rows=0, q_start=None, indexed=False):
  # shared: one kv head for every query head that is also the value (MLA latent), group: query heads per block. mask: additive (B, N)
  # causal_rows (prefill): the query heads are (head, token) rows of causal_rows tokens from q_start, mask: (B, tokens, N)
  # indexed: instead of the mask, the list of the keys to attend to (count last): a data dependent count of chunks
  valid_kv_len, q_start = _unbind(valid_kv_len), _unbind(q_start) if q_start is not None else None
  index = mask[0] if indexed else None
  if index is not None:
    mask, n_max = (), cast(int, index.shape[0]) - 1
    valid_kv_len = index[n_max].load().cast(dtypes.weakint)
  _, B, H_KV, N, D = cast(tuple[int, int, int, int, int], cache_kv.shape)
  _, H, M, _ = cast(tuple[int, int, int, int], q.shape)
  if group is not None: H_KV = H // group
  assert M == 1 and H % H_KV == 0 and D % WARP_SIZE == 0 and max_kv_len <= N and max_kv_len % block_n == 0
  G, CHUNK, DPL, WAVES, PARTIALS = H // H_KV, block_n, D // WARP_SIZE, waves, out.shape[2]
  scale = 1/math.sqrt(D) if scale is None else scale
  assert CHUNK % WAVES == 0
  SEC = CHUNK // WAVES  # keys each wave scans independently
  total_chunks = (valid_kv_len+CHUNK-1)//CHUNK
  live_chunks = min(total_chunks, PARTIALS) if isinstance(total_chunks, int) else total_chunks.minimum(PARTIALS)
  if index is not None: live_chunks = min((n_max+CHUNK-1)//CHUNK, PARTIALS)
  block_bhkv, block_chunk = UOp.range(B*H_KV, 0, AxisType.GLOBAL), UOp.range(live_chunks, 1, AxisType.GLOBAL)
  lane, wave = UOp.range(WARP_SIZE, -1, axis_type=AxisType.WARP), UOp.range(WAVES, 3, axis_type=AxisType.LOCAL)
  b, kv_head = block_bhkv // H_KV, block_bhkv % H_KV
  # per-lane query fragments for every GQA head, kept packed in registers; unpacked at use
  qf = tuple(_vec_load(q[b, kv_head*G+h, 0, lane*DPL], DPL) for h in range(G))
  zerof = UOp.const(0, dtypes.float)
  # Each block scans every PARTIALS-th chunk, keeping an online softmax across rounds.
  # (blocks past the chunks run no round: the count is never negative)
  chunk_round = UOp.range((total_chunks-1-block_chunk)//PARTIALS+1 if index is None else
                          ((total_chunks-block_chunk+PARTIALS-1)//PARTIALS).maximum(0), 4, AxisType.LOOP)
  chunk_id = block_chunk + chunk_round*PARTIALS
  valids: list[UOp] = []
  scores: list[list[UOp]] = [[zerof]*G for _ in range(SEC)]
  vfrags: list[tuple[UOp, ...]] = [()]*SEC
  # the keys a row sees: a prefill row is the token at q_start + t, it sees the keys up to itself (padding rows past valid_kv_len see them all)
  row_tok = [(kv_head*G + h) % causal_rows for h in range(G)] if causal_rows else [None]*G
  row_len = [(q_start + t + 1).minimum(valid_kv_len) if causal_rows else valid_kv_len for t in row_tok]
  row_valids: list[list[UOp]] = []
  for j in range(SEC):
    key = chunk_id*CHUNK + wave*SEC + j
    valid = key < valid_kv_len
    valids.append(valid)
    row_valids.append([key < n for n in row_len])
    # (the list past the count isn't written: clamped)
    if index is not None: key = index[key.minimum(n_max-1)].load().cast(dtypes.weakint).maximum(0).minimum(N-1)
    kfrag = _vec_load(cache_kv[0, b, 0 if shared else kv_head, key, lane*DPL], DPL)
    # V is prefetched in the score pass so both streams are in flight together
    vfrags[j] = tuple(valid.where(v, zerof) for v in (kfrag if shared else _vec_load(cache_kv[1, b, kv_head, key, lane*DPL], DPL)))
    for h in range(G):
      s = warp_reduce(sum((qf[h][i]*kfrag[i] for i in range(DPL)), UOp.const(0, dtypes.float)), full_wave=True) * scale
      if mask: s = s + (mask[0][b, row_tok[h], key.minimum(N-1)] if causal_rows else mask[0][b, key.minimum(N-1)]).load().float()
      scores[j][h] = row_valids[j][h].where(s, UOp.const(-1e30, dtypes.float))
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
      beta = row_valids[j][h].where(((scores[j][h]-row_max[h])*LOG2E).exp2(), zerof)
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
  # indexed: a data dependent count of keys, an upper bound estimate
  est = None if index is None else Estimates(4*n_max*B*H*D, n_max*B*D*cache_kv.dtype.itemsize, n_max*B*D*cache_kv.dtype.itemsize)
  return UOp.group(*final_stores).end(lane, wave, block_chunk, block_bhkv).sink(arg=KernelInfo(name="flash_decode_partial", opts_to_apply=(),
                                                                                              estimates=est))

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

def amd_flash_attention_decode(q:Tensor, cache_kv:Tensor, valid_kv_len:int|UOp, max_kv_len:int, shared:bool=False, scale:float|None=None,
                               mask:Tensor|None=None, causal_rows:int=0, group:int=1, partials:int=48, block_n:int=64,
                               q_start:int|UOp|None=None, index:Tensor|None=None) -> Tensor:
  """
  shared: the cache (1, B, 1, N, D) is the key and the value of every head (MLA latent), mask: additive (B, N) over the keys.
  causal_rows: the query heads are (head, token) rows of that many tokens from q_start, `group` of them per block, mask (B, tokens, N)
  """
  # Carry length bindings even when the cache was populated independently of this call.
  if isinstance(valid_kv_len, UOp): cache_kv = Tensor(cache_kv.uop.after(valid_kv_len))
  B, H, D = cache_kv.shape[1], q.shape[1], cache_kv.shape[4]
  chunks, axis = min(partials, max_kv_len // block_n), q.uop.axis
  partial = Tensor.empty(B, H, chunks, D, dtype="float32", device=q.device, axis=axis)
  stats = Tensor.empty(B, H, chunks, 2, dtype="float32", device=q.device, axis=axis)
  # shared: a block per `group` query heads, they read the same keys
  waves, group = 16, group if shared else int(H // cache_kv.shape[2])
  while waves * group * ((D+LDS_PAD)*2 + 8) > 65536 or (waves > 1 and block_n // waves < 2): waves //= 2
  assert waves > 0, "attention head group exceeds shared memory capacity"
  fxn = functools.partial(_amd_flash_attention_decode_partial, valid_kv_len=valid_kv_len, max_kv_len=max_kv_len, block_n=block_n, waves=waves,
                          group=group if shared else None, shared=shared, scale=scale, causal_rows=causal_rows, q_start=q_start,
                          indexed=index is not None)
  extra = (index,) if index is not None else () if mask is None else (mask,)
  partial, stats = Tensor.custom_kernel(partial, stats, q, cache_kv, *extra, fxn=fxn)[:2]
  live = (valid_kv_len+block_n-1)//block_n
  live = min(live, chunks) if isinstance(live, int) else live.minimum(chunks)
  # indexed: the partials of every block (the empty ones weigh nothing)
  if index is not None: live = min((int(index.shape[0])-1+block_n-1)//block_n, chunks)
  out = Tensor.empty(B, H, 1, D, dtype="float32", device=q.device, axis=axis)
  fxn = functools.partial(_amd_flash_decode_combine, live=live)
  return Tensor.custom_kernel(out, partial, stats, fxn=fxn)[0]

@functools.cache
def _mla_flash_wmma_kernel(out:UOp, q:UOp, cache:UOp, *mask:UOp, tokens:int, q_start:int|UOp, valid_len:int|UOp, scale:float) -> UOp:
  # RDNA4 WMMA flash attention of the (head, token) rows of q on the shared latent cache (keys = values). a workgroup: 16 tokens of a head, 4 waves
  # each own a quarter of the latent dims: they sum their partial scores through LDS, then update their quarter of the output with P (via LDS)
  q_start, valid_len = _unbind(q_start), _unbind(valid_len)
  R, D = cast(tuple[int, int], q.shape)
  N, WAVES = cache.shape[0], 4
  DW = D // WAVES
  KT = DW // 16
  blk = UOp.range(R // 16, 0, AxisType.GLOBAL)
  wave, lane = UOp.range(WAVES, 1, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  col, half = lane % 16, lane // 16
  t0, dbase = (blk % (tokens // 16)) * 16, wave * DW
  ks = tuple(e % 4 + half * 4 + (e // 4) * 8 for e in range(8))  # the k of the 8 fragment elements of a lane (RDNA4)
  # the query fragments of this wave's dims (A: lane row = col)
  qfrag = [UOp.stack(*(q[blk*16 + col, dbase + kk*16 + k].load() for k in ks)) for kk in range(KT)]
  # the keys up to the causal end of the 16 tokens
  n_tile = UOp.range(((q_start + t0 + 16).minimum(valid_len) + 15) // 16, 2, AxisType.LOOP)
  def key_at(j) -> tuple[UOp, UOp]: return n_tile * 16 + j, UOp.const(True, dtypes.bool)
  key, key_ok = key_at(col)
  zero8 = UOp.stack(*[UOp.const(0, dtypes.float)]*8)
  # partial scores of this wave's dims (C: lane key = col, rows half*8+e)
  s_acc = zero8
  for kk in range(KT): s_acc = UOp.wmma(qfrag[kk], UOp.stack(*(cache[key.minimum(N-1), dbase + kk*16 + k].load() for k in ks)), s_acc, *WMMA_ARG)
  s_reg = UOp.alloc((8,), dtypes.float, addrspace=AddrSpace.REG)
  s_reg = s_reg.after(s_reg.store(s_acc))
  lds_s = UOp.alloc((WAVES, WARP_SIZE, 8), dtypes.float, addrspace=AddrSpace.LOCAL)
  lds_s = lds_s.after(UOp.group(*(lds_s[wave, lane, e].store(s_reg[e].load()) for e in range(8))).barrier())
  m_reg, l_reg = _reg((8,), -1e30), _reg((8,), 0)
  o_regs = [_reg((8,), 0) for _ in range(KT)]
  m_old, l_old = m_reg.after(n_tile), l_reg.after(n_tile)
  p_vals, m_new, l_new, alpha = [], [], [], []
  for e in range(8):
    row = t0 + half * 8 + e
    vis = (key <= q_start + row) & (key < valid_len) & key_ok
    s = sum((lds_s[w, lane, e].load() for w in range(WAVES)), UOp.const(0, dtypes.float)) * scale
    if mask: s = s + mask[0][row, key.minimum(N-1)].load().float()
    s = vis.where(s, UOp.const(-1e30, dtypes.float))
    mx = s
    for sh in (1, 2, 4, 8): mx = mx.maximum(_swizzle_xor(mx, sh))
    mn = m_old[e].load().maximum(mx)
    p = vis.where(((s - mn) * LOG2E).exp2(), UOp.const(0, dtypes.float))
    ps = p
    for sh in (1, 2, 4, 8): ps = ps + _swizzle_xor(ps, sh)
    a = ((m_old[e].load() - mn) * LOG2E).exp2()
    p_vals.append(p)
    m_new.append(mn)
    alpha.append(a)
    l_new.append(l_old[e].load() * a + ps)
  # P (rows x keys) to the A layout (lane row = col, elements keys) through LDS
  lds_p = UOp.alloc((WAVES, 16, 16 + 2), dtypes.half, addrspace=AddrSpace.LOCAL)
  lds_p = lds_p.after(UOp.group(*(lds_p[wave, half*8 + e, col].store(p_vals[e].cast(dtypes.half)) for e in range(8))).barrier())
  pfrag = UOp.stack(*(lds_p[wave, col, k].load() for k in ks))
  o_new = []
  for dt in range(KT):
    # B: lane dim = col, elements keys. keys past the valid length are zero: p is 0 there but the cache isn't written (0*nan)
    vkeys = [key_at(k) for k in ks]
    vfrag = UOp.stack(*(((vk < valid_len) & vok).where(cache[vk.minimum(N-1), dbase + dt*16 + col].load(), UOp.const(0, dtypes.half))
                        for vk, vok in vkeys))
    prev = o_regs[dt].after(n_tile)
    o_new.append(UOp.wmma(pfrag, vfrag, UOp.stack(*(prev[e].load() * alpha[e] for e in range(8))), *WMMA_ARG))
  update = UOp.group(m_reg.store(UOp.stack(*m_new)), l_reg.store(UOp.stack(*l_new)), *(o.store(v) for o, v in zip(o_regs, o_new)))
  update = update.barrier().end(n_tile)
  l_fin = l_reg.after(update)
  stores = [out[blk*16 + half*8 + e, dbase + dt*16 + col].store(o.after(update)[e].load() / l_fin[e].load().maximum(1e-30))
            for dt, o in enumerate(o_regs) for e in range(8)]
  return UOp.group(*stores).end(blk, wave, lane).sink(arg=KernelInfo(name="mla_flash_wmma", opts_to_apply=()))

@functools.cache
def _mla_sparse_wmma_kernel(out:UOp, q:UOp, cache:UOp, index:UOp, scale:float, valid_len:UOp) -> UOp:
  # RDNA4 WMMA attention of every token on the keys of its list (pool_index_rows): a workgroup per token, its 16 (padded) heads are the rows,
  # they share the keys. 4 waves own a quarter of the latent dims each: they sum their partial scores through LDS, then update their quarter of
  # the output with P (via LDS). keys past valid_len aren't seen (padding rows of a chunk past its tokens: the cache isn't written there)
  R, D = cast(tuple[int, int], q.shape)
  T, n_max = R // 16, cast(int, index.shape[1]) - 1
  N, WAVES = cache.shape[0], 4
  DW = D // WAVES
  KT = DW // 16
  tok = UOp.range(T, 0, AxisType.GLOBAL)
  wave, lane = UOp.range(WAVES, 1, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  col, half = lane % 16, lane // 16
  dbase = wave * DW
  ks = tuple(e % 4 + half * 4 + (e // 4) * 8 for e in range(8))
  qfrag = [UOp.stack(*(q[tok*16 + col, dbase + kk*16 + k].load() for k in ks)) for kk in range(KT)]
  count = index[tok, n_max].load().cast(dtypes.weakint).minimum(n_max)
  n_tile = UOp.range((count + 15) // 16, 2, AxisType.LOOP)
  def key_at(j) -> tuple[UOp, UOp]:
    i = n_tile * 16 + j
    key = index[tok, i.minimum(n_max-1)].load().cast(dtypes.weakint)
    return key.maximum(0).minimum(N-1), (i < count) & (key < valid_len)
  key, key_ok = key_at(col)
  zero8 = UOp.stack(*[UOp.const(0, dtypes.float)]*8)
  s_acc = zero8
  for kk in range(KT): s_acc = UOp.wmma(qfrag[kk], UOp.stack(*(cache[key, dbase + kk*16 + k].load() for k in ks)), s_acc, *WMMA_ARG)
  s_reg = UOp.alloc((8,), dtypes.float, addrspace=AddrSpace.REG)
  s_reg = s_reg.after(s_reg.store(s_acc))
  lds_s = UOp.alloc((WAVES, WARP_SIZE, 8), dtypes.float, addrspace=AddrSpace.LOCAL)
  lds_s = lds_s.after(UOp.group(*(lds_s[wave, lane, e].store(s_reg[e].load()) for e in range(8))).barrier())
  m_reg, l_reg = _reg((8,), -1e30), _reg((8,), 0)
  o_regs = [_reg((8,), 0) for _ in range(KT)]
  m_old, l_old = m_reg.after(n_tile), l_reg.after(n_tile)
  p_vals, m_new, l_new, alpha = [], [], [], []
  for e in range(8):
    s = key_ok.where(sum((lds_s[w, lane, e].load() for w in range(WAVES)), UOp.const(0, dtypes.float)) * scale, UOp.const(-1e30, dtypes.float))
    mx = s
    for sh in (1, 2, 4, 8): mx = mx.maximum(_swizzle_xor(mx, sh))
    mn = m_old[e].load().maximum(mx)
    p = key_ok.where(((s - mn) * LOG2E).exp2(), UOp.const(0, dtypes.float))
    ps = p
    for sh in (1, 2, 4, 8): ps = ps + _swizzle_xor(ps, sh)
    a = ((m_old[e].load() - mn) * LOG2E).exp2()
    p_vals.append(p)
    m_new.append(mn)
    alpha.append(a)
    l_new.append(l_old[e].load() * a + ps)
  lds_p = UOp.alloc((WAVES, 16, 16 + 2), dtypes.half, addrspace=AddrSpace.LOCAL)
  lds_p = lds_p.after(UOp.group(*(lds_p[wave, half*8 + e, col].store(p_vals[e].cast(dtypes.half)) for e in range(8))).barrier())
  pfrag = UOp.stack(*(lds_p[wave, col, k].load() for k in ks))
  vkeys = [key_at(k) for k in ks]
  o_new = []
  for dt in range(KT):
    # (keys past the count are zero: p is 0 there, but 0*nan)
    vfrag = UOp.stack(*(vok.where(cache[vk, dbase + dt*16 + col].load(), UOp.const(0, dtypes.half)) for vk, vok in vkeys))
    prev = o_regs[dt].after(n_tile)
    o_new.append(UOp.wmma(pfrag, vfrag, UOp.stack(*(prev[e].load() * alpha[e] for e in range(8))), *WMMA_ARG))
  update = UOp.group(m_reg.store(UOp.stack(*m_new)), l_reg.store(UOp.stack(*l_new)), *(o.store(v) for o, v in zip(o_regs, o_new)))
  update = update.barrier().end(n_tile)
  l_fin = l_reg.after(update)
  stores = [out[tok*16 + half*8 + e, dbase + dt*16 + col].store(o.after(update)[e].load() / l_fin[e].load().maximum(1e-30))
            for dt, o in enumerate(o_regs) for e in range(8)]
  # a data dependent count of keys, an upper bound estimate
  est = Estimates(4*R*n_max*D, R*D*2 + T*n_max*D*2, R*D*4)
  return UOp.group(*stores).end(tok, wave, lane).sink(arg=KernelInfo(name="mla_sparse_wmma", opts_to_apply=(), estimates=est))

def amd_flash_attention_mla_sparse(q:Tensor, cache:Tensor, index:Tensor, scale:float, valid_len:int|UOp) -> Tensor:
  """
  the attention of the (B=1, H, T, D) queries on the shared latent cache (1, 1, 1, N, D), every token on the keys of its list (T, n_max+1)
  (pool_index_rows): (1, H, T, D). the heads (at most 16) of a token are the rows of a workgroup
  """
  B, H, T, D = q.shape
  # the heads split over n devices: (n, T, 16, D) rows, a device's heads padded to 16
  n = len(q.device) if isinstance(q.device, tuple) and q.uop.axis == 1 else 1
  hl = H // n
  assert B == 1 and hl <= 16 and D % 64 == 0
  rows = q.reshape(n, hl, T, D).pad((None, (0, 16 - hl), None, None)).permute(0, 2, 1, 3).half().reshape(n*T*16, D).contiguous()
  out = Tensor.empty(n*T*16, D, dtype=dtypes.float32, device=q.device, axis=0 if n > 1 else None)
  vl = _unbind(valid_len) if isinstance(valid_len, UOp) else UOp.const(valid_len, dtypes.weakint)
  if isinstance(valid_len, UOp): index = Tensor(index.uop.after(valid_len))
  out = Tensor.custom_kernel(out, rows, cache.reshape(cache.shape[-2], D), index,
                             fxn=functools.partial(_mla_sparse_wmma_kernel, scale=scale, valid_len=vl))[0]
  return out.reshape(n, T, 16, D)[:, :, :hl].permute(0, 2, 1, 3).reshape(1, H, T, D)

def amd_flash_attention_mla_prefill(q:Tensor, cache:Tensor, q_start:int|UOp, valid_kv_len:int|UOp, max_kv_len:int, scale:float,
                                    mask:Tensor|None=None, group:int=4, partials:int=4, block_n:int=32) -> Tensor:
  """
  causal attention of the (B, H, T, D) queries at q_start.. on the shared latent cache (1, B, 1, N, D) of valid_kv_len keys: (B, H, T, D).
  mask: additive (B, T, N). every (head, token) is a row of the flash decode kernel, `group` rows of a head share their key loads
  """
  B, H, T, D = q.shape
  if B == 1 and T % 16 == 0 and D % 64 == 0 and _wmma_rdna4(q.device) and getenv("MLA_WMMA", 1):
    rows = q.half().reshape(H*T, D).contiguous()
    if isinstance(valid_kv_len, UOp): cache = Tensor(cache.uop.after(valid_kv_len))
    out = Tensor.empty(H*T, D, dtype=dtypes.float32, device=q.device, axis=q.uop.axis if q.uop.axis is None else 0)
    fxn = functools.partial(_mla_flash_wmma_kernel, tokens=T, q_start=q_start, valid_len=valid_kv_len, scale=scale)
    srcs = (rows, cache.reshape(cache.shape[-2], D)) + (() if mask is None else (mask.reshape(T, -1),))
    return Tensor.custom_kernel(out, *srcs, fxn=fxn)[0].reshape(B, H, T, D)
  group, partials, block_n = getenv("MLA_G", group), getenv("MLA_P", partials), getenv("MLA_BN", block_n)
  assert isinstance(T, int) and (H*T) % group == 0 and T % group == 0
  rows = q.half().reshape(B, H*T, 1, D).contiguous()
  out = amd_flash_attention_decode(rows, cache, valid_kv_len, max_kv_len, shared=True, scale=scale, mask=mask, causal_rows=T, group=group,
                                   partials=partials, block_n=block_n, q_start=q_start)
  return out.reshape(B, H, T, D)

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
  fxn = functools.partial(_amd_flash_attention, valid_kv_len=valid_end, q_start=q_start, rdna4=_wmma_rdna4(q.device))
  if isinstance(valid_end, UOp): assigned_kv = Tensor(assigned_kv.uop.after(valid_end))
  out = Tensor.custom_kernel(out, q.half().reshape(B*H, T, D), assigned_kv, fxn=fxn)[0].reshape(B, H, T, D)
  return out if q_start is None else out[:, :, :T_real]

# ******** gated delta net: fused recurrent scan ********

@functools.cache
def _gated_delta_prefill_kernel(core:UOp, q:UOp, k:UOp, v:UOp, beta:UOp, alpha:UOp, state:UOp, kq:UOp, start_pos:UOp|None=None,
                                end_pos:UOp|None=None) -> UOp:
  # end_pos: the state is (2, B, H, V, K), the state at a position p is p%2: read at start_pos, written at end_pos
  batch, heads, tokens, value_dim, row_tile = *core.shape, 4
  key_dim, alpha_dim = q.shape[-1], alpha.shape[-1] if len(alpha.shape) == 4 else 1
  assert all(isinstance(x, int) for x in (batch, heads, tokens, value_dim, key_dim)) and key_dim % 32 == 0 and value_dim % row_tile == 0
  batch, heads, tokens, value_dim, key_dim = cast(tuple[int, int, int, int, int], (batch, heads, tokens, value_dim, key_dim))
  core, v = (x.reshape(batch*heads, tokens, value_dim) for x in (core, v))
  q, k = (x.reshape(batch*heads, tokens, key_dim) for x in (q, k))
  beta, kq = (x.reshape(batch*heads, tokens) for x in (beta, kq))
  alpha = alpha.reshape(batch*heads, tokens, alpha_dim)
  # the rows of the state read and written: offsets in the (2*B*H, V, K) state with parities
  read_off, write_off = (0, 0) if end_pos is None else ((cast(UOp, start_pos) % 2) * (batch*heads), (end_pos % 2) * (batch*heads))
  state = state.reshape(-1, value_dim, key_dim)
  bh_row, lane = UOp.range(batch*heads*value_dim//row_tile, 0), UOp.range(32, 1, axis_type=AxisType.LOCAL)
  bh, row_base = bh_row // (value_dim//row_tile), (bh_row % (value_dim//row_tile))*row_tile
  rows, cols = tuple(row_base+i for i in range(row_tile)), tuple(lane + i*32 for i in range(key_dim//32))
  current = UOp.alloc((row_tile*key_dim//32,), dtypes.float32, addrspace=AddrSpace.REG)
  initial = None if start_pos is None else start_pos.eq(0)
  current = current.after(current.store(UOp.stack(*(state[read_off+bh, row, col].float() if initial is None else
    initial.where(0, state[read_off+bh, row, col].float()) for row in rows for col in cols))))
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
  state_stores = (state[write_off+bh, row, col].store(current.after(step)[row_idx*key_dim//32+i].load().cast(state.dtype))
                  for row_idx,row in enumerate(rows) for i,col in enumerate(cols))
  return UOp.group(*state_stores).end(lane, bh_row).sink(arg=KernelInfo(name="gated_delta_prefill", opts_to_apply=()))

@functools.cache
def _indexer_scores_kernel(out:UOp, q:UOp, k:UOp, w:UOp, pools:UOp, NP:int=4) -> UOp:
  # RDNA4 WMMA: out[t, p] = sum over the heads h of w[t, h] * relu(q[t, h] . k[p]). a wave per 16 tokens and G of the tiles of 16*NP
  # pools: it loops over the tiles g, g+G, .. of the pools (the work follows the context, not its max)
  T, H, D = cast(tuple[int, int, int], q.shape)
  P = cast(int, k.shape[0])
  G = min(64, P // (16*NP))
  tt, g, lane = UOp.range(T // 16, 0, AxisType.GLOBAL), UOp.range(G, 1, AxisType.GLOBAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  col, half = lane % 16, lane // 16
  tiles = (pools + 16*NP - 1) // (16*NP)
  active = UOp.range(((tiles - g + G - 1) // G).maximum(0), 2, AxisType.LOOP)
  t0, p0 = tt * 16, (g + active * G) * (16*NP)
  ks = tuple(e % 4 + half * 4 + (e // 4) * 8 for e in range(8))
  kfrags = [[UOp.stack(*(k[p0 + n*16 + col, kk*16 + j].load() for j in ks)) for kk in range(D // 16)] for n in range(NP)]
  zero8 = UOp.stack(*[UOp.const(0, dtypes.float)]*8)
  # a loop over the heads (unrolled, the loads of all the heads would be hoisted): the sums in registers
  accs = [_reg((8,), 0, active) for _ in range(NP)]
  h = UOp.range(H, 3, AxisType.LOOP)
  qfrags = [UOp.stack(*(q[t0 + col, h, kk*16 + j].load() for j in ks)) for kk in range(D // 16)]
  wt = [w[t0 + half*8 + e, h].load() for e in range(8)]
  updates = []
  for n in range(NP):
    s = zero8
    for kk in range(D // 16): s = UOp.wmma(qfrags[kk], kfrags[n][kk], s, *WMMA_ARG)
    s_reg = UOp.alloc((8,), dtypes.float, addrspace=AddrSpace.REG)
    s_reg = s_reg.after(s_reg.store(s))
    prev = accs[n].after(h)
    updates.append(accs[n].store(UOp.stack(*(prev[e].load() + s_reg[e].load().maximum(0) * wt[e] for e in range(8)))))
  done = UOp.group(*updates).end(h)
  stores = [out[t0 + half*8 + e, p0 + n*16 + col].store(accs[n].after(done)[e].load()) for n in range(NP) for e in range(8)]
  # a data dependent count of tiles, an upper bound estimate
  est = Estimates(2*T*H*P*D, P*D*2 + T*H*D*2, T*P*4)
  return UOp.group(*stores).end(active).end(tt, g, lane).sink(arg=KernelInfo(name="indexer_scores", opts_to_apply=(), estimates=est))

def indexer_scores(q:Tensor, k:Tensor, w:Tensor, pools:int|UOp) -> Tensor:
  """q (T, H, D), k (P, D) the pooled keys (P static, the first `pools` used), w (T, H): (T, P) scores, the columns past pools unwritten"""
  T, H, D = q.shape
  P = k.shape[0]
  out = Tensor.empty(T, P, dtype=dtypes.float32, device=q.device)
  kh = k if k.dtype == dtypes.half else k.half().contiguous()
  if isinstance(pools, UOp): kh = Tensor(kh.uop.after(pools))
  fxn = functools.partial(_indexer_scores_kernel, pools=_unbind(pools) if isinstance(pools, UOp) else UOp.const(pools, dtypes.weakint))
  return Tensor.custom_kernel(out, q.half().contiguous(), kh, w.float().contiguous(), fxn=fxn)[0]

@functools.cache
def _pool_keys_kernel(pk:UOp, ring:UOp, kg:UOp, ape:UOp, start_pos:UOp, T:UOp) -> UOp:
  # the pooled key of every pool with a token of this call: the softmax of the gates (+ape) over the pool's tokens weighs their keys. the
  # tokens before start_pos in the first pool come from the ring (the last pool's tokens), the others from kg (T rows of key and gate)
  P, D = cast(tuple[int, int], pk.shape)
  kpool, rows = cast(int, ring.shape[0]), cast(int, kg.shape[0])
  j, d = UOp.range(rows // kpool + 1, 0, AxisType.GLOBAL), UOp.range(D, 1, AxisType.LOCAL)
  first, end = start_pos // kpool, start_pos + T
  pool = first + j
  keys:list[UOp] = []
  gates:list[UOp] = []
  for s in range(kpool):
    pos = pool * kpool + s
    old, new = pos < start_pos, (pos >= start_pos) & (pos < end)
    src = (pos - start_pos).maximum(0).minimum(rows-1)
    for i, lst in ((0, keys), (1, gates)):
      lst.append(old.where(ring[s, i, d].load().float(), new.where(kg[src, i, d].load().float(), UOp.const(0, dtypes.float))))
  logits = [g + ape[s, d].load().float() for s, g in enumerate(gates)]
  mx = functools.reduce(UOp.maximum, logits)
  ws = [((l - mx) * LOG2E).exp2() for l in logits]
  pooled = functools.reduce(lambda a, b: a + b, (w * k for w, k in zip(ws, keys))) / functools.reduce(lambda a, b: a + b, ws)
  ok = (pool * kpool < end) & (pool < P)
  return pk[pool.valid(ok).minimum(P-1), d].store(pooled.cast(pk.dtype)).end(j, d).sink(arg=KernelInfo(name="pool_keys", opts_to_apply=()))

@functools.cache
def _pool_ring_kernel(ring:UOp, kg:UOp, start_pos:UOp, T:UOp) -> UOp:
  # the tokens of the last pool (if not complete) to the ring: the next call's first pool needs them
  kpool, _, D = cast(tuple[int, int, int], ring.shape)
  rows = cast(int, kg.shape[0])
  s, i, d = UOp.range(kpool, 0, AxisType.GLOBAL), UOp.range(2, 1, AxisType.GLOBAL), UOp.range(D, 2, AxisType.LOCAL)
  end = start_pos + T
  pos = end // kpool * kpool + s
  ok = (pos >= start_pos) & (pos < end)
  return ring[s, i, d.valid(ok)].store(kg[(pos - start_pos).maximum(0).minimum(rows-1), i, d].load().cast(ring.dtype)).end(s, i, d).sink(
    arg=KernelInfo(name="pool_ring", opts_to_apply=()))

def pool_keys(pk:Tensor, ring:Tensor, kg:Tensor, ape:Tensor, start_pos:UOp, T:int|UOp) -> Tensor:
  """
  updates the (P, D) pooled key cache with the pools of the T tokens at start_pos (kg: their (rows, 2, D) keys and gates, rows >= T), the ring
  (kpool, 2, D) holds the tokens of the last incomplete pool. returns the cache after the update
  """
  ints = [_unbind(x) if isinstance(x, UOp) else UOp.const(x, dtypes.weakint) for x in (start_pos, T)]
  pk, ring = (Tensor(x.uop.after(*(v for v in (start_pos, T) if isinstance(v, UOp)))) for x in (pk, ring))
  pk = Tensor.custom_kernel(pk, ring, kg.contiguous(), ape.contiguous(), fxn=functools.partial(_pool_keys_kernel, start_pos=ints[0], T=ints[1]))[0]
  ring = Tensor(ring.uop.after(pk.uop))
  ring = Tensor.custom_kernel(ring, kg.contiguous(), fxn=functools.partial(_pool_ring_kernel, start_pos=ints[0], T=ints[1]))[0]
  return Tensor(pk.uop.after(ring.uop))

@functools.cache
def _indexer_scores_decode_kernel(out:UOp, q:UOp, k:UOp, w:UOp, pools:UOp, NP:int=2) -> UOp:
  # one query (RDNA4 WMMA): the heads are the rows, out[p] = sum over the heads h of w[h] * relu(q[h] . k[p]). G waves, a wave loops over the
  # tiles of 16*NP pools g, g+G, .. of the pools: the work follows the context, not its max
  H, D = cast(tuple[int, int], q.shape)
  P = cast(int, k.shape[0])
  G = min(256, P // (16*NP))
  g, lane = UOp.range(G, 0, AxisType.GLOBAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  col, half = lane % 16, lane // 16
  tiles = (pools + 16*NP - 1) // (16*NP)
  active = UOp.range(((tiles - g + G - 1) // G).maximum(0), 1, AxisType.LOOP)
  p0 = (g + active * G) * (16*NP)
  ks = tuple(e % 4 + half * 4 + (e // 4) * 8 for e in range(8))
  zero8 = UOp.stack(*[UOp.const(0, dtypes.float)]*8)
  totals = [UOp.const(0, dtypes.float)] * NP
  for ht in range(H // 16):
    qfrags = [UOp.stack(*(q[ht*16 + col, kk*16 + j].load() for j in ks)) for kk in range(D // 16)]
    wt = [w[ht*16 + half*8 + e].load() for e in range(8)]
    for n in range(NP):
      s = zero8
      for kk in range(D // 16): s = UOp.wmma(qfrags[kk], UOp.stack(*(k[p0 + n*16 + col, kk*16 + j].load() for j in ks)), s, *WMMA_ARG)
      s_reg = UOp.alloc((8,), dtypes.float, addrspace=AddrSpace.REG)
      s_reg = s_reg.after(s_reg.store(s))
      totals[n] = totals[n] + functools.reduce(lambda a, e: a + s_reg[e].load().maximum(0) * wt[e], range(1, 8), s_reg[0].load().maximum(0) * wt[0])
  # the rows of a lane's half, then the other half's (lane ^ 16)
  stores = [out[(p0 + n*16 + col).valid(half.eq(0))].store(tot + _swizzle_xor(tot, 16)) for n, tot in enumerate(totals)]
  # a data dependent count of tiles, an upper bound estimate
  est = Estimates(2*H*P*D, P*D*2, P*4)
  return UOp.group(*stores).end(active).end(g, lane).sink(arg=KernelInfo(name="indexer_scores_decode", opts_to_apply=(), estimates=est))

DECODE_THREADS = 1024

# a ballot is across the wave: a CUSTOM (a statement of its own), not inlined where it is used (in the branch of a gated store)
def _ballot(pred:UOp) -> UOp: return UOp(Ops.CUSTOM, src=(pred,), arg=("__builtin_amdgcn_ballot_w32({0})", dtypes.uint32))
def _popcount(x:UOp) -> UOp: return UOp(Ops.CUSTOMI, src=(x,), arg=("__builtin_popcount({})", dtypes.int32))
# across the wave: CUSTOMs (statements of their own, not inlined in the branch of a gated store)
def _bpermute(x:UOp, src_lane:UOp) -> UOp:
  return UOp(Ops.CUSTOM, src=(x, src_lane), arg=("__builtin_amdgcn_ds_bpermute((int)({1})*4, (int)({0}))", dtypes.int32))
def _readlane(x:UOp, lane:int) -> UOp: return UOp(Ops.CUSTOM, src=(x,), arg=(f"__builtin_amdgcn_readlane((int)({{0}}), {lane})", dtypes.int32))
def _wave_scan(x:UOp, lane:UOp) -> UOp:
  """the inclusive prefix sums of x (int32) over the lanes of a wave"""
  for d in (1, 2, 4, 8, 16): x = x + (lane >= d).where(_bpermute(x, (lane - d).maximum(0)), UOp.const(0, dtypes.int32))
  return x

@functools.cache
def _pool_index_kernel(index:UOp, scores:UOp, q_start:UOp, pools:UOp, k:int, kpool:int) -> UOp:
  # a workgroup per query row (the query of row r is at q_start+r), thread t holds pool c*THREADS + t of every chunk c of THREADS pools. the
  # key of the k-th best by a bisection (the counts of keys >= a candidate: per thread, the waves' through LDS), then the picked pools in
  # order: the picks of every (chunk, wave) (ballots), scans of them for their offsets. the tail of the query's pool after them, the count
  # after the list. the loops over the chunks run over the chunks with pools: the work follows the context, not its max
  rows, P = cast(tuple[int, int], scores.shape)
  n_max, THREADS = cast(int, index.shape[1]) - 1, DECODE_THREADS
  per, W = (P + THREADS - 1) // THREADS, THREADS // WARP_SIZE
  assert W == WARP_SIZE, "a scan of a chunk's waves is a wave"
  row, wave, lane = UOp.range(rows, 0, AxisType.GLOBAL), UOp.range(W, 1, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  t, q = wave * WARP_SIZE + lane, q_start + row
  npc = ((pools + THREADS - 1) // THREADS).minimum(per)
  def key(c:UOp) -> UOp:
    # the order preserving key of a complete pool before the query, 0 (the lowest) for the others
    p = t + c * THREADS
    ok = (p < pools) & ((p + 1) * kpool <= q + 1)
    bits = scores[row, p.minimum(P-1).valid(ok)].load().bitcast(dtypes.uint32)
    return ok.where((bits >= 0x80000000).where(bits ^ 0xFFFFFFFF, bits | 0x80000000), UOp.const(0, dtypes.uint32))
  zero = UOp.const(0, dtypes.int32)
  # the bisection: bit by bit from the top, no loop with no more pools than k (they are all picked)
  lds = UOp.alloc((32, W), dtypes.float32, addrspace=AddrSpace.LOCAL)
  thr_reg = _reg_u32()
  bit = UOp.range((pools > k).where(32, 0), 3, AxisType.LOOP)
  prev = thr_reg.after(bit)[0].load()
  cand = prev | (UOp.const(1, dtypes.uint32) << (31 - bit).cast(dtypes.uint32))
  c = UOp.range(npc, 4, AxisType.LOOP)
  cnt_reg = _reg((1,), 0, bit)  # zeroed for every bit
  cnt = cnt_reg.after(cnt_reg[0].store(cnt_reg.after(c)[0].load() + (key(c) >= cand).cast(dtypes.float32)).end(c))[0].load()
  stage = lds.after(lds[bit, wave.valid(lane.eq(0))].store(warp_reduce(cnt, full_wave=True)).barrier())
  total = warp_reduce(stage[bit, lane % W].load() * (lane < W).cast(dtypes.float32), full_wave=True)  # the sum of the waves' counts
  thr = thr_reg.after(thr_reg[0].store((total >= k).where(cand, prev)).end(bit))[0].load().maximum(1)
  # the picks of every (chunk, wave): the counts of a ballot
  wc = UOp.alloc((per, W), dtypes.int32, addrspace=AddrSpace.LOCAL)
  c2 = UOp.range(npc, 5, AxisType.LOOP)
  wc = wc.after(wc[c2, wave.valid(lane.eq(0))].store(_popcount(_ballot(key(c2) >= thr))).end(c2).barrier())
  # the offsets of the (chunk, wave)s in pool order: in every chunk a scan of its waves' counts (a wave per chunk), then a scan of the
  # chunks' totals (one wave). the chunks without pools count 0
  def count(ci:UOp) -> UOp: return (ci < npc).where(wc[ci.minimum(per-1), lane].load(), zero)
  wpre, ct = UOp.alloc((per, W), dtypes.int32, addrspace=AddrSpace.LOCAL), UOp.alloc((per,), dtypes.int32, addrspace=AddrSpace.LOCAL)
  st_a = []
  for r in range((per + W - 1) // W):
    ci = (wave + r * W).minimum(per - 1)
    n = count(wave + r * W)
    incl = _wave_scan(n, lane)
    ok = wave + r * W < per
    st_a += [wpre[ci.valid(ok), lane].store(incl - n), ct[ci.valid(ok & lane.eq(W - 1))].store(incl)]
  wpre, ct = (x.after(UOp.group(*st_a).barrier()) for x in (wpre, ct))
  cb = UOp.alloc((per + 1,), dtypes.int32, addrspace=AddrSpace.LOCAL)
  carry, st_b = zero, []
  for r in range((per + W - 1) // W):
    n = (lane + r * W < per).where(ct[(lane + r * W).minimum(per - 1)].load(), zero)
    incl = _wave_scan(n, lane)
    st_b.append(cb[(lane + r * W).valid(wave.eq(0) & (lane + r * W < per))].store(carry + incl - n))
    carry = carry + _readlane(incl, W - 1)
  st_b.append(cb[UOp.const(per, dtypes.weakint).valid(wave.eq(0) & lane.eq(0))].store(carry))
  cb = cb.after(UOp.group(*st_b).barrier())
  # the picked pools of every chunk at their offsets
  c3 = UOp.range(npc, 6, AxisType.LOOP)
  pk = key(c3) >= thr
  pos = cb[c3].load() + wpre[c3, wave].load() + _popcount(_ballot(pk) & ((UOp.const(1, dtypes.uint32) << lane.cast(dtypes.uint32)) - 1))
  p = t + c3 * THREADS
  stores = [UOp.group(*(index[row, (pos * kpool + j).valid(pk & (pos < k))].store((p * kpool + j).cast(dtypes.int32))
                        for j in range(kpool))).end(c3)]
  total_picked = cb[per].load().minimum(k)  # the count of picked pools
  tail0 = (q + 1) // kpool * kpool
  ntail = q + 1 - tail0
  stores.append(index[row, (total_picked * kpool + t).valid(t < ntail)].store((tail0 + t).cast(dtypes.int32)))
  stores.append(index[row, UOp.const(n_max, dtypes.weakint).valid(t.eq(0))].store((total_picked * kpool + ntail).cast(dtypes.int32)))
  est = Estimates(rows*P*32, rows*P*4*4, rows*(k+1)*kpool*4)  # data dependent loops: an upper bound
  return UOp.group(*stores).end(row, wave, lane).sink(arg=KernelInfo(name="pool_index", opts_to_apply=(), estimates=est))

def _reg_u32() -> UOp:
  ret = UOp.alloc((1,), dtypes.uint32, addrspace=AddrSpace.REG)
  return ret.after(ret.store(ret.const_like(0)))

def pool_index_rows(scores:Tensor, q_start:UOp, pools:int|UOp, top:int, kpool:int) -> Tensor:
  """(rows, top*kpool+kpool+1) int32: for every query row (at q_start+r) the list of the keys it sees (its top pools, its tail), the count last"""
  rows = int(scores.shape[0])
  out = Tensor.empty(rows, top*kpool + kpool + 1, dtype=dtypes.int32, device=scores.device)
  pu = _unbind(pools) if isinstance(pools, UOp) else UOp.const(pools, dtypes.weakint)
  sc = Tensor(scores.uop.after(q_start, *([pools] if isinstance(pools, UOp) else [])))
  return Tensor.custom_kernel(out, sc, fxn=functools.partial(_pool_index_kernel, q_start=_unbind(q_start), pools=pu, k=top, kpool=kpool))[0]

def indexer_decode_index(q:Tensor, k:Tensor, w:Tensor, q_start:UOp, pools:int|UOp, top:int, kpool:int) -> Tensor:
  """
  one query: q (H, D), the pooled keys k (P, D) half (P static, the first `pools` used), w (H,). the (top*kpool+kpool+1,) int32 list of the
  keys it sees (the top pools of the scores, the tail of its own pool), the count last
  """
  P = int(k.shape[0])
  scores = Tensor.empty(P, dtype=dtypes.float32, device=q.device)
  if isinstance(pools, UOp): k = Tensor(k.uop.after(pools))
  pu = _unbind(pools) if isinstance(pools, UOp) else UOp.const(pools, dtypes.weakint)
  scores = Tensor.custom_kernel(scores, q.half().contiguous(), k, w.float().contiguous(),
                                fxn=functools.partial(_indexer_scores_decode_kernel, pools=pu))[0]
  return pool_index_rows(scores.reshape(1, P), q_start, pools, top, kpool).reshape(-1)

KDA_WAVES = 8

@functools.cache
def _kda_decode_kernel(z:UOp, state:UOp, cq:UOp, ck:UOp, cv:UOp, pq:UOp, pk:UOp, pv:UOp, wq:UOp, wk:UOp, wv:UOp, gate:UOp, alpha:UOp, beta:UOp,
                       dt:UOp, a:UOp, norm:UOp, start_pos:UOp, lb:float|None, eps:float) -> UOp:
  # one token of a kda layer, a workgroup per head: the short convolutions of q, k and v (shifting their states), the l2 norms of q and k,
  # the decay and the delta rule update of the head's state, the rms norm of the output and its sigmoid gate.
  # the states (2, ...) at a position p are p%2: read at start_pos, written at start_pos+1 (the other one keeps the state before the token)
  heads, V, K = cast(tuple[int, int, int], state.shape[-3:])
  W, conv = KDA_WAVES, cast(int, wq.shape[-1])
  assert K == V == W*32//2 and conv == 4, "a thread per channel of q, k and v"
  state, z = state.reshape(2*heads, V, K), z.reshape(heads*V)
  cq, ck, cv = (x.reshape(2*(conv-1), heads*K) for x in (cq, ck, cv))
  rp, wp = start_pos % 2, (start_pos + 1) % 2
  pq, pk, pv, gate, alpha = (x.reshape(heads*K) for x in (pq, pk, pv, gate, alpha))
  h, wave, lane = UOp.range(heads, 0, AxisType.GLOBAL), UOp.range(W, 1, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  tid, initial = wave * WARP_SIZE + lane, start_pos.eq(0)
  def silu(x:UOp) -> UOp: return x / ((-x * LOG2E).exp2() + 1)
  def sigmoid(x:UOp) -> UOp: return ((-x * LOG2E).exp2() + 1).reciprocal()
  # the convolutions: thread tid < K is channel tid of q, k and v. the state of a channel is its last conv-1 inputs
  lds = UOp.alloc((3, K), dtypes.float, addrspace=AddrSpace.LOCAL)
  ch, active = h * K + tid % K, tid < K
  stores = []
  for i, (cs, p, w) in enumerate(((cq, pq, wq), (ck, pk, wk), (cv, pv, wv))):
    window = [initial.where(0, cs[rp*(conv-1) + j, ch].load().float()) for j in range(conv-1)] + [p[ch].load().float()]
    out = silu(sum((w[ch, j].load().float() * window[j] for j in range(1, conv)), w[ch, 0].load().float() * window[0]))
    stores += [lds[i, tid.valid(active)].store(out)] + [cs[wp*(conv-1) + j, ch.valid(active)].store(window[j+1].cast(cs.dtype))
                                                       for j in range(conv-1)]
  lds = lds.after(UOp.group(*stores).barrier())
  # every wave: q and k of the head (key column lane + 32*i), l2 normalized, the decay of every column
  cols = [lane + 32*i for i in range(K // 32)]
  def l2(x:list[UOp]) -> list[UOp]:
    r = (warp_reduce(functools.reduce(lambda s,y: s + y*y, x[1:], x[0]*x[0]), full_wave=True) + 1e-6).sqrt().reciprocal()
    return [y * r for y in x]
  qs = [y * K**-0.5 for y in l2([lds[0, c].load() for c in cols])]
  ks = l2([lds[1, c].load() for c in cols])
  kq = warp_reduce(functools.reduce(lambda s,xy: s + xy[0]*xy[1], zip(ks[1:], qs[1:]), ks[0]*qs[0]), full_wave=True)
  b = sigmoid(beta.reshape(heads)[h].load().float())
  A = a.reshape(heads)[h].load().float()
  def decay(c:UOp) -> UOp:
    g = alpha[h*K + c].load().float() + dt.reshape(heads*K)[h*K + c].load().float()
    if lb is None: return ((g.maximum(0) + ((-g.abs() * LOG2E).exp2() + 1).log2() * (1/LOG2E)) * A * LOG2E).exp2()
    return (sigmoid(g * -A) * lb * LOG2E).exp2()
  decays = [decay(c) for c in cols]
  # the delta rule on the rows of this wave
  rows_per_wave = V // W
  lds_o = UOp.alloc((V,), dtypes.float, addrspace=AddrSpace.LOCAL)
  ostores, sstores = [], []
  for j in range(rows_per_wave):
    r = wave * rows_per_wave + j
    decayed = [initial.where(0, state[rp*heads + h, r, c].load().float()) * d for c, d in zip(cols, decays)]
    sk = warp_reduce(functools.reduce(lambda s,xy: s + xy[0]*xy[1], zip(decayed[1:], ks[1:]), decayed[0]*ks[0]), full_wave=True)
    sq = warp_reduce(functools.reduce(lambda s,xy: s + xy[0]*xy[1], zip(decayed[1:], qs[1:]), decayed[0]*qs[0]), full_wave=True)
    delta = (lds[2, r].load() - sk) * b
    sstores += [state[wp*heads + h, r, c].store((x + delta * kk).cast(state.dtype)) for c, x, kk in zip(cols, decayed, ks)]
    ostores.append(lds_o[r.valid(lane.eq(0))].store(sq + delta * kq))
  lds_o = lds_o.after(UOp.group(*ostores).barrier())
  # rms norm of the head's output, gated
  o = [lds_o[c].load() for c in cols]
  rinv = (warp_reduce(functools.reduce(lambda s,y: s + y*y, o[1:], o[0]*o[0]), full_wave=True) * (1/V) + eps).sqrt().reciprocal()
  r = tid % V
  out = lds_o[r].load() * rinv * norm[r].load().float() * sigmoid(gate[h*V + r].load().float())
  zs = z[(h*V + r).valid(tid < V)].store(out.cast(z.dtype))
  return UOp.group(zs, *sstores).end(h, wave, lane).sink(arg=KernelInfo(name="kda_decode", opts_to_apply=()))

def kda_decode(state:Tensor, conv_states:list[Tensor], projs:list[Tensor], conv_weights:list[Tensor], gate:Tensor, alpha:Tensor, beta:Tensor,
               dt:Tensor, a:Tensor, norm:Tensor, start_pos:Tensor, lb:float|None, eps:float) -> Tensor:
  """
  one token of a kda layer: returns the gated, normed output (B, 1, heads*V) in half. the states (2, B, ...) at a position p are p%2:
  the ones of start_pos are read, the ones of start_pos+1 written
  """
  _, B, heads, V, K = state.shape
  assert B == 1 and start_pos.uop.is_bound_var
  z = Tensor.empty(1, 1, heads*V, dtype=dtypes.half, device=state.device, axis=None if state.uop.axis is None else 2)
  # the states are written in place
  inplace = tuple(Tensor(s.uop.after(start_pos.uop)) for s in (state, *conv_states))
  srcs = (z, *inplace, *projs, *conv_weights, gate, alpha, beta, dt, a, norm)
  contig = tuple(x.uop if x.uop.op is Ops.AFTER else x.uop.contiguous() for x in srcs)
  params = tuple(UOp.placeholder_like(x, slot=i) for i,x in enumerate(contig))
  call = _kda_decode_kernel(*params, start_pos.uop.unbound(), lb, eps).call(*contig)
  return Tensor(contig[0].after(call))

def gated_delta_prefill(q:Tensor, k:Tensor, v:Tensor, beta:Tensor, alpha:Tensor, state:Tensor, start_pos:Tensor|None=None,
                        end_pos:UOp|None=None) -> Tensor:
  """
  the scan of the (B, H, T, ...) tokens updating the recurrent state in place. end_pos: the state is (2, B, H, V, K), the state at position p
  is p%2: the scan reads the state of start_pos and writes the one of end_pos (the other one keeps the state before)
  """
  batch, heads, tokens, key_dim = q.shape
  value_dim = v.shape[-1]
  assert q.shape == k.shape and v.shape[:3] == beta.shape == (batch, heads, tokens)
  assert state.shape == ((2,) if end_pos is not None else ()) + (batch, heads, value_dim, key_dim)
  assert alpha.shape[:3] == (batch, heads, tokens) and (len(alpha.shape) == 3 or alpha.shape[-1] in (1, key_dim))
  assert key_dim % 32 == 0 and value_dim % 4 == 0
  assert q.dtype == k.dtype == dtypes.float32, "recurrent Q/K must be float32"
  assert state.uop.contiguous_view_offset() is not None, "recurrent state must be contiguous"
  if start_pos is not None:
    assert start_pos.uop.is_bound_var
    state = Tensor(state.uop.after(start_pos.uop, *([end_pos] if isinstance(end_pos, UOp) else [])))
  core, kq = Tensor.empty_like(v), (q*k).sum(-1).contiguous()
  srcs = (core, q.contiguous(), k.contiguous(), v.contiguous(), beta.contiguous(), alpha.contiguous(), state, kq)
  contig = tuple(x.uop if x.uop.op is Ops.AFTER else x.uop.contiguous() for x in srcs)
  params = tuple(UOp.placeholder_like(x, slot=i) for i,x in enumerate(contig))
  end = None if end_pos is None else end_pos.unbind_all()[0] if isinstance(end_pos, UOp) else UOp.const(end_pos, dtypes.weakint)
  call = _gated_delta_prefill_kernel(*params, None if start_pos is None else start_pos.uop.unbound(), end).call(*contig)
  return Tensor(contig[0].after(call))
