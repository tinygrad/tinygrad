from __future__ import annotations
import contextlib, functools, itertools, math
from dataclasses import replace
from typing import Any, Callable, cast
from tinygrad import Tensor, UOp, nn, Device, Context
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.dtype import AddrSpace, DType, dtypes
from tinygrad.helpers import prod, getenv, round_up, unwrap, dedup
from tinygrad.uop.ops import AxisType, KernelInfo, Ops
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.renderer import Estimates

BLOCK_M, BLOCK_N, WARP_SIZE = 32, 32, 32
WMMA_M, WMMA_N, WMMA_K = 16, 16, 16
WAVES_M, WAVES_N, LANES_PER_WAVE_M, LANES_PER_WAVE_N = 2, 2, 2, 16
WMMA_ACC, THREADS_PER_BLOCK = WMMA_M // LANES_PER_WAVE_M, WARP_SIZE * WAVES_M * WAVES_N
LDS_PAD, WMMA_ARG, LOG2E = 4, ((WMMA_N, WMMA_M, WMMA_K), 32), math.log2(math.e)
GGML_BLOCK_SIZE, Q8_GROUP_SIZE = 256, 32
Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K = 8, 10, 11, 12, 13, 14
IQ2_XS, IQ3_XXS, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS, Q2_0 = 17, 18, 20, 21, 22, 23, 42
QUANT_SIZES = {Q8_0: 272, Q2_K: 84, Q3_K: 110, Q4_K: 144, Q5_K: 176, Q6_K: 210, IQ2_XS: 74,
               IQ3_XXS: 98, IQ4_NL: 144, IQ3_S: 110, IQ2_S: 82, IQ4_XS: 136, Q2_0: 72}  # bytes per 256 weights
HALFWORD_QUANTS = (Q8_0, Q6_K, Q2_K, Q3_K, IQ2_XS, IQ3_XXS, IQ4_NL, IQ3_S, IQ2_S, Q2_0)
# formats of independent blocks of 32 or 64 weights: (halfwords, weights) of a block. a device may hold part of a 256 weight row group
SMALL_BLOCK_QUANTS = {Q8_0: (17, 32), IQ4_NL: (9, 32), Q2_0: (9, 64)}
def _small_block(ggml_type:int, group:UOp|int) -> UOp|int:
  """the halfword offset of the block of the 32 weight group `group` (of a row, or of a 256 weight group) of a small block format"""
  hw, w = SMALL_BLOCK_QUANTS[ggml_type]
  return group // (w // 32) * hw
QUANT_NAMES = {Q8_0: "q8_0", Q2_K: "q2_k", Q3_K: "q3_k", Q4_K: "q4_k", Q5_K: "q5_k", Q6_K: "q6", IQ4_XS: "iq4_xs",
               IQ2_XS: "iq2_xs", IQ3_XXS: "iq3_xxs", IQ4_NL: "iq4_nl", IQ3_S: "iq3_s", IQ2_S: "iq2_s", Q2_0: "q2_0"}

def _unbind(v:int|UOp) -> int|UOp: return v.unbind_all()[0] if isinstance(v, UOp) else v

DECODE_MULTI = 32  # up to that many tokens of a decode linear share its weight loads

@functools.cache
def _gfx(device:str|tuple[str, ...]|None) -> int:
  """the major gfx version of an AMD device with HIP kernels (11: RDNA3, 12: RDNA4), 0 if not one"""
  if isinstance(device, tuple): device = device[0]
  if device is None or device.split(":")[0] != "AMD": return 0
  # @function contexts set ALLOW_DEVICE_USAGE=0 (scheduling must not open devices); the device is always open here
  with Context(ALLOW_DEVICE_USAGE=1):
    return t[0] if (t:=getattr(Device[device], "target", None)) is not None and isinstance(Device[device].renderer, HIPRenderer) else 0

def amd_custom_kernels_supported(device:str|tuple[str, ...]|None) -> bool:
  """the kernels are RDNA3's and RDNA4's (wave32): the WMMA ones in the layouts of either (rdna4, see _wmma_rdna4)"""
  return not getenv("DISABLE_AMD_KERNELS") and _gfx(device) in (11, 12)

def _wmma_rdna4(device:str|tuple[str, ...]|None) -> bool:
  """
  RDNA4's WMMA layouts (gfx12): a lane's A/B fragment is 8 halves (two groups of four of its row's 16 inputs, by its half of the wave), its
  accumulator rows half*8+j (RDNA3: 16 halves, the rows 2j+half). the kernels with tensor cores take rdna4, the RDNA3-only ones aren't used
  """
  return _gfx(device) == 12

# int8 WMMA on q8 activations pays a rescale per 32 inputs of every output: it wins where the f16 decode is expensive, the grid formats
I8_WMMA_QUANTS = (IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S)
# formats with a single scale per 32 weights and no offset
GROUP_SCALE_QUANTS = (Q8_0, IQ4_NL, IQ4_XS, IQ3_S, IQ3_XXS, Q4_K, Q5_K, Q2_0)

def _cfn(inline:bool, src:tuple[UOp, ...], arg:tuple[str, DType]) -> UOp:
  """a call of inline C (the old CUSTOM (a statement, or a value in a variable) and CUSTOMI (inline=True: an expression)): arg (format, dtype)"""
  fmt, dt = arg
  return UOp.custom_function(fmt, shape=src[0].shape if src and dt != dtypes.void else (), dtype=dt, fmt=True).call(*src, inline=inline)

def _lane_xor(val:UOp, mask:int) -> UOp:
  """the float of lane ^ mask (wave32): DPP within rows of 16 lanes, permlanex16 across them (RDNA, a few cycles), else a ds_swizzle"""
  if 0 < mask < 16: op = f"__builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, {{0}}), {0x160 | mask}, 15, 15, false)"
  elif mask == 16:
    op = "__builtin_amdgcn_permlanex16(0, __builtin_bit_cast(int, {0}), 0x76543210, 0xfedcba98, false, false)"
  else: op = f"__builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, {{0}}), {0x1f | mask<<10})"
  return _cfn(False, src=(val,), arg=(f"__builtin_bit_cast(float, {op})", dtypes.float))

@functools.cache
def _fwht_kernel(out:UOp, x:UOp, sign:UOp, *, pre:bool) -> UOp:
  # out (n, D) = x times a Hadamard matrix of random signs (QSA_KV_ROT's rotation m[i][j] = sign[i]*(-1)^popcount(i&j)/sqrt(D)): pre, x @ m (the
  # signs on the inputs), else x @ m.T (on the outputs). a wave a row, a lane its elements lane+32e: the butterflies of the low 5 bits across
  # the lanes (lane_xor), of the others in registers. D log D adds, not a D x D matmul (other roundings than one: QSA_ROT_FWHT)
  D = cast(int, out.shape[-1])
  out, x = out.reshape(-1, D), x.reshape(-1, D)
  n, E = cast(int, out.shape[0]), D // WARP_SIZE
  assert D & (D - 1) == 0 and E >= 1
  r, lane = UOp.range(n, 0, AxisType.GLOBAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  vals = [x[r, e*WARP_SIZE + lane].load().float() for e in range(E)]
  if pre: vals = [v * sign[e*WARP_SIZE + lane].load() for e, v in enumerate(vals)]
  for b in range(5):
    m, hi = 1 << b, (lane & (1 << b)).ne(0)
    vals = [hi.where(o - v, v + o) for v, o in zip(vals, [_lane_xor(v, m) for v in vals])]
  for b in range(E.bit_length() - 1):
    m = 1 << b
    vals = [vals[e ^ m] - vals[e] if e & m else vals[e] + vals[e ^ m] for e in range(E)]
  vals = [v * D**-0.5 if pre else v * D**-0.5 * sign[e*WARP_SIZE + lane].load() for e, v in enumerate(vals)]
  return UOp.group(*[out[r, e*WARP_SIZE + lane].store(v.cast(out.dtype)) for e, v in enumerate(vals)]).end(r, lane).sink(
    arg=KernelInfo(name="fwht", opts_to_apply=()))

def fwht(x:Tensor, sign:Tensor, transpose:bool=False) -> Tensor:
  """x (..., D) @ m (transpose: @ m.T), m = diag(sign) H / sqrt(D) (H the Hadamard matrix of D, a power of two), in float"""
  # (every device its part of x, whatever axis it's split on: the kernel takes its rows of D)
  xf = x.float().contiguous()
  out = Tensor.empty(*xf.shape, dtype=dtypes.float32, device=x.device, axis=xf.uop.axis if isinstance(x.device, tuple) else None)
  return Tensor.custom_kernel(out, xf, sign.float().contiguous(), fxn=functools.partial(_fwht_kernel, pre=not transpose))[0]

def warp_reduce(val:UOp, maximum:bool=False, full_wave:bool=False) -> UOp:
  for offset in ((16, 8, 4, 2, 1) if full_wave else (8, 4, 2, 1)):
    if val.op is Ops.INDEX and val.addrspace == AddrSpace.REG: val = val.load()
    other = _lane_xor(val, offset)
    val = val.maximum(other) if maximum else val + other
  return val

def _reg(shape:tuple[int, ...], value:float, dep:UOp|None=None) -> UOp:
  ret = UOp.alloc(shape, dtypes.float, addrspace=AddrSpace.REG)
  return ret.after((ret if dep is None else ret.after(dep)).store(ret.const_like(value)))

# ******** quant linear: q8-activation kernels over packed ggml weights ********

# (the most rows the split-K gemm of tiles of 32 rows takes (batched decoding, the smallest prefill chunk), see q8_linear)
SKINNY_ROWS = 256
SKINNY_WG, HC_KS = getenv("SKINNY_WG", 192), getenv("HC_KS", 32)
# AG_FUSE: all_gather_q8's packing kernel sends the words to the peers too (the flag exchange's send kernel folded into it)
AG_FUSE = getenv("AG_FUSE", 1)
# TILE16: batched decoding's hyper-connection gemms (a device's few rows: 4-16 of a tile of 32) of its real rows' tiles alone, in tiles of 16
TILE16 = getenv("TILE16", 1)

def unwrap_storage_views(u:UOp) -> UOp:
  # Only unwrap storage/order-preserving views, then require the exact dequantization expression.
  # This rejects subsequent arithmetic and permutations, including RoPE's concatenated query weights.
  # a permute keeps the order if the axes not of size 1 on a device stay sorted (gguf_shard moves the devices)
  while u.op in (Ops.RESHAPE, Ops.STAGE) or (u.op is Ops.CAST and dtypes.is_float(u.dtype) and dtypes.is_float(u.src[0].dtype)) or \
    (u.op is Ops.PERMUTE and (kept:=[a for a in u.marg if u.src[0].shard_shape[a] != 1]) == sorted(kept)):
    u = u.src[0]
  return u

def _known_decoded(u:UOp) -> tuple[int, UOp]|None:
  from tinygrad.llm.gguf_load import decoded_weights
  return decoded_weights.get(unwrap_storage_views(u))

def find_packed(decoded:Tensor, in_features:int) -> tuple[int, Tensor]|None:
  """(ggml type, packed words) if `decoded` is exactly the dequantization of packed ggml weights, rows of in_features"""
  if in_features % Q8_GROUP_SIZE: return None
  packed_sizes = {typ: decoded.numel() // 256 * type_size for typ,type_size in QUANT_SIZES.items()
                  if (in_features % GGML_BLOCK_SIZE == 0 or typ in SMALL_BLOCK_QUANTS) and decoded.numel() % 256 == 0}
  packed_sizes.update({typ: decoded.numel() // w * hw * 2 for typ,(hw, w) in SMALL_BLOCK_QUANTS.items()})
  # the loader's dequantizations (gguf.decoded_weight) are known: no graph to build and match
  raw:UOp|None
  if (known:=_known_decoded(decoded.uop)) is not None and prod(known[1].shape) == packed_sizes.get(known[0]):
    ggml_type, raw = known
  else:
    graph = decoded.uop.toposort()
    raw = next((u for u in graph if u.op in (Ops.SHRINK, Ops.BUFFER, Ops.UNSHARD) and u.dtype == dtypes.uint8 and
                prod(u.shape) in packed_sizes.values()), None)
    if raw is None: return None
    # Several formats have the same byte count (Q3_K/IQ3_S and Q4_K/IQ4_NL). Match the expression, not just the size.
    for ggml_type, size in packed_sizes.items():
      if size != prod(raw.shape): continue
      expected = ggml_data_to_tensor(Tensor(raw).flatten(), int(decoded.numel()), ggml_type)
      if unwrap_storage_views(decoded.uop).key == unwrap_storage_views(expected.uop).key: break
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
  _empty_init = False
  def __init__(self, in_features:int, out_features:int, bias=True):
    if Linear._empty_init:
      # (weights that are loaded after: no random initialization, its graph is most of the time to make a big model)
      self.weight, self.bias = Tensor.empty(out_features, in_features), Tensor.empty(out_features) if bias else None
    else: super().__init__(in_features, out_features, bias)
    self.in_features, self.out_features = in_features, out_features
  @staticmethod
  @contextlib.contextmanager
  def empty_init():
    """the Linears made in it have empty weights (to be loaded)"""
    old, Linear._empty_init = Linear._empty_init, True
    try: yield
    finally: Linear._empty_init = old
  def set_quantized(self, decoded:Tensor):
    if (found:=find_packed(decoded, self.in_features)) is None: return
    self.ggml_type, self.weight = found
    self.shard_axis = decoded.uop.axis
  def partial(self, x:Tensor, half:bool=False) -> Tensor:
    """
    row parallel layers return the partial sums of each device, (devices, ..., out_features) sharded on the devices, the others the output.
    half: the partial sums may be in half (they're sent so)
    """
    if self.ggml_type is None and self.use_custom_quant and amd_custom_kernels_supported(self.weight.device): self.set_quantized(self.weight)
    if self.ggml_type in QUANT_SIZES and self.shard_axis == 1 and self.bias is None and isinstance(x.numel(), int):
      return q8_linear(self, x, False, half_out=half)
    return self(x)
  def __call__(self, x:Tensor) -> Tensor:
    supported = self.use_custom_quant and amd_custom_kernels_supported(self.weight.device)
    if self.ggml_type is None and supported:
      self.set_quantized(self.weight)
      if self.ggml_type is None:
        # tiny dense fp16 matmul (e.g. the ssm beta/alpha head rows): single fp16 gemv kernel instead of a
        # generic matmul schedule, and realize the densely packed weight once if it is still a lazy ggml view
        # (a weight split on its outputs over the devices: every device its outputs)
        if self.weight.dtype in (dtypes.half, dtypes.float, dtypes.bfloat16) and self.out_features <= 2048 \
          and self.in_features % (WARP_SIZE*4) == 0 and (self.weight.uop.axis is None or (self.weight.uop.axis == 0 and
          isinstance(self.weight.device, tuple) and self.out_features % len(self.weight.device) == 0)):
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
  return _cfn(True, src=(a, b, c), arg=("__builtin_amdgcn_sudot4(true, {}, true, {}, {}, false)", dtypes.int32))

def _amd_sudot4u(a:UOp, b:UOp, c:UOp) -> UOp:  # c + the dot of a's 4 signed bytes with b's 4 unsigned ones
  return _cfn(True, src=(a, b, c), arg=("__builtin_amdgcn_sudot4(true, {}, false, {}, {}, false)", dtypes.int32))

_H2 = "_Float16 __attribute__((ext_vector_type(2)))"
def _amd_pkrtz(a:UOp, b:UOp) -> UOp:  # two floats to a half2 (its bits)
  return _cfn(True, src=(a, b), arg=("__builtin_bit_cast(unsigned int, __builtin_amdgcn_cvt_pkrtz({}, {}))", dtypes.uint32))
def _amd_fdot2_less1152(a:UOp, b:UOp, c:UOp, less:int=1152) -> UOp:  # c + the dot of the half2s a - 1152 (less) and b (in float)
  return _cfn(True, src=(a, b, c), arg=(f"__builtin_amdgcn_fdot2(__builtin_bit_cast({_H2}, {{0}}) - ({_H2})((_Float16){less:.1f}f), "
                                             f"__builtin_bit_cast({_H2}, {{1}}), {{2}}, false)", dtypes.float))

def _amd_byte_perm(a:UOp, b:UOp, selectors:UOp) -> UOp:
  return _cfn(True, src=tuple(x.cast(dtypes.uint32) for x in (a, b, selectors)), arg=("__builtin_amdgcn_perm({}, {}, {})", dtypes.uint32))

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

_tables:dict[tuple, Tensor] = {}
def _table(key:tuple, device:str|tuple[str, ...]|None, values:tuple[int, ...], dtype:DType) -> Tensor:
  """
  a constant table, a buffer made once (not a kernel in every jit that reads it: tables_ready makes them before the jits capture), copied from
  host memory (no kernel to compile). made while the devices can't be used (a function's capture) it's its kernel in that graph, not kept
  """
  if (t:=_tables.get(key)) is not None: return t
  from tinygrad.helpers import ALLOW_DEVICE_USAGE
  if not ALLOW_DEVICE_USAGE: return Tensor.const(values, dtype).to(device, force=True).contiguous()
  # (its bytes copied into a buffer on every device: no schedule, nothing to link)
  import struct
  from tinygrad.device import Buffer, MultiBuffer
  data = struct.pack(f"<{len(values)}{dtype.fmt}", *values)
  devs = device if isinstance(device, tuple) else (Device.canonicalize(device),)
  bufs = [Buffer(d, len(data), initial_value=data) for d in devs]
  if isinstance(device, tuple): (mb:=MultiBuffer.__new__(MultiBuffer)).bufs = bufs
  t = _tables[key] = Tensor(UOp.from_buffer(mb if isinstance(device, tuple) else bufs[0], dtype, device if isinstance(device, tuple) else devs[0]))
  return t

def iq4_half_lut(device:str|tuple[str, ...]|None) -> Tensor:
  from tinygrad.runtime.autogen.ggml_common import kvalues_iq4nl
  import struct
  # (the half pairs as uint32 words)
  words = tuple(int.from_bytes(struct.pack("<ee", kvalues_iq4nl[i], kvalues_iq4nl[j]), "little") for j in range(16) for i in range(16))
  return _table(("iq4_half_lut", device), device, words, dtypes.uint32)

@functools.cache
def _q8_quantize_kernel(q:UOp, scale:UOp, xsum:UOp, x:UOp, *sq:UOp, tokens:int, in_features:int) -> UOp:
  groups = in_features//Q8_GROUP_SIZE
  if groups % 8 == 0: return _q8_quantize_wide(q, scale, xsum, x, *sq, tokens=tokens, in_features=in_features)
  token_group, lane = UOp.range(tokens*groups, 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP)
  token, group = token_group//groups, token_group%groups
  value = x.reshape(tokens, groups, 32)[token, group, lane].load().float()
  stores = _q8_narrow_stores(value, q, scale, xsum, sq[0] if sq else None, token, group, lane)
  return UOp.group(*stores).end(token_group, lane).sink(arg=KernelInfo(name="q8_quantize", opts_to_apply=()))

def _q8_narrow_stores(value:UOp, q:UOp, scale:UOp, xsum:UOp, sq:UOp|None, token:UOp, group:UOp, lane:UOp) -> list[UOp]:
  # a lane's value of a group of 32 (a wave): its words, scale, sums (and squares)
  # Quantize each input once, then pack four neighboring lanes into one word.
  # Keep divisions intact: reciprocal multiplication can move half-precision inputs across rounding ties.
  d = _cfn(False, src=(warp_reduce(value.abs(), maximum=True, full_wave=True),), arg=("({0}/127.0f)", dtypes.float)).maximum(1e-8)
  rounded = _cfn(False, src=(value, d), arg=("__builtin_nearbyintf({0}/{1})", dtypes.float))
  quant = rounded.clip(-127, 127).cast(dtypes.int8)
  word = quant.cast(dtypes.uint8).cast(dtypes.uint32) << ((lane%4)*8).cast(dtypes.uint32)
  for offset in (1, 2):
    word |= _cfn(False, src=(word,), arg=(f"__builtin_amdgcn_ds_swizzle({{0}}, {0x1f | offset<<10})", dtypes.uint32))
  stores:list[UOp] = [q[token, group, (lane//4).valid((lane%4).eq(0))].store(word),
            scale[token, group.valid(lane.eq(0))].store(d),
            xsum[token, group, (lane//16).valid((lane%16).eq(0))].store(warp_reduce(quant.float()))]
  # optionally the sum of squares of each group, for an rms norm applied after a linear layer
  if sq is not None: stores.append(sq[token, group.valid(lane.eq(0))].store(warp_reduce(value*value, full_wave=True)))
  return stores

def _q8_quantize_wide(q:UOp, scale:UOp, xsum:UOp, x:UOp, *sq:UOp, tokens:int, in_features:int) -> UOp:
  # as _q8_quantize_kernel, a wave quantizes 8 groups: a lane eight neighboring inputs (wide loads), four lanes a group. the same values: the
  # max and the integer sums don't depend on the order, the sums of squares add in the order of the full wave butterfly (16, 8, 4, 2, 1 apart)
  groups = in_features//Q8_GROUP_SIZE
  token_g8, lane = UOp.range(tokens*(groups//8), 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP)
  token, group, sub = token_g8//(groups//8), (token_g8 % (groups//8))*8 + lane//4, lane % 4
  values = [x.reshape(tokens, in_features)[token, group*32 + sub*8 + i].load().float() for i in range(8)]
  stores = _q8_wide_stores(values, q, scale, xsum, sq[0] if sq else None, token, group, sub)
  return UOp.group(*stores).end(token_g8, lane).sink(arg=KernelInfo(name="q8_quantize", opts_to_apply=()))

def _q8_wide_stores(values:list[UOp], q:UOp, scale:UOp, xsum:UOp, sq:UOp|None, token:UOp, group:UOp, sub:UOp) -> list[UOp]:
  # a lane's 8 neighboring values of a group of 32 (four lanes a group, sub the lane of the four): their words, scale, sums (and squares)
  mx = functools.reduce(lambda a, b: a.maximum(b), [v.abs() for v in values])
  for off in (1, 2): mx = mx.maximum(_lane_xor(mx, off))
  d = _cfn(False, src=(mx,), arg=("({0}/127.0f)", dtypes.float)).maximum(1e-8)
  quants = [_cfn(False, src=(v, d), arg=("__builtin_nearbyintf({0}/{1})", dtypes.float)).clip(-127, 127).cast(dtypes.int8) for v in values]
  words = [functools.reduce(lambda a, b: a | b, [quants[w*4+i].cast(dtypes.uint8).cast(dtypes.uint32) << (8*i) for i in range(4)]) for w in range(2)]
  qs = functools.reduce(lambda a, b: a + b, [qv.float() for qv in quants])
  qs = qs + _lane_xor(qs, 1)
  stores:list[UOp] = [q[token, group, sub*2 + w].store(words[w]) for w in range(2)]
  stores += [scale[token, group.valid(sub.eq(0))].store(d), xsum[token, group, (sub//2).valid((sub%2).eq(0))].store(qs)]
  if sq is not None:
    sqs = [v*v for v in values]
    sqs = [a + _lane_xor(a, 2) for a in sqs]  # 16 apart: the lane two over
    sqs = [a + _lane_xor(a, 1) for a in sqs]  # 8 apart
    for k in (4, 2, 1): sqs = [sqs[i] + sqs[i ^ k] for i in range(8)]
    stores.append(sq[token, group.valid(sub.eq(0))].store(sqs[0]))
  return stores

def q8_quantize(x:Tensor, tokens:int, in_features:int, sumsq:bool=False, split:bool=False) -> tuple[Tensor, ...]:
  """
  int8 groups of 32 with their scales and sums (and optionally sums of squares). an input shared by several layers is quantized once.
  NOTE: the cache holds graphs, the model clears it after building one (q8_quantize_cache_clear)
  """
  if (ret:=_q8_cache.get(key:=_q8_key(x, tokens, in_features, sumsq, split))) is None: ret = _q8_cache[key] = _q8_quantize(*key)
  return ret

def _q8_key(x:Tensor, tokens:int, in_features:int, sumsq:bool, split:bool) -> tuple:
  # (keyed by the rows: the views of an input are quantized once)
  u = x.uop
  if not split and x.shape[-1] == in_features and prod(x.shape) == tokens*in_features:
    while u.op is Ops.RESHAPE: u = u.src[0]
    u = u.reshape((tokens, in_features))
  return (u, tokens, in_features, sumsq, split)

_q8_cache:dict[tuple, tuple[Tensor, ...]] = {}

def q8_quantize_known(x:Tensor, tokens:int, in_features:int, sumsq:bool, outs:tuple[Tensor, ...]):
  """the quantization of x is outs (the kernel that made x made it too)"""
  _q8_cache[_q8_key(x, tokens, in_features, sumsq, False)] = outs

def q8_quantize_cache_clear():
  # (the caches of a forward's graph: kept, they'd keep its buffers from the memory planner of the jits captured after)
  _q8_cache.clear()
  _bundle_cache.clear()
  _gather_cache.clear()

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

def _prefetched(body:UOp, rng:UOp, last:UOp|int, params:tuple[UOp, ...]|None=None) -> tuple[UOp, list[UOp]]:
  """
  software pipelining of the loop over rng: the loads in body from the global params (default: the ones body doesn't store to) that depend on
  rng are issued an iteration early, into registers (the first ones before the loop). returns body reading the registers and the stores of the
  next iteration's loads (in the loop)
  """
  topo, deps = body.toposort(), set()
  if params is None:
    stored = {u.src[0].src[0] for u in topo if u.op is Ops.STORE and u.src[0].op is Ops.INDEX}
    params = tuple(u for u in topo if u.op is Ops.PARAM and u not in stored)
  for u in topo:
    if u is rng or any(s in deps for s in u.src): deps.add(u)
  targets_of = {u.src[0] for u in topo if u.op in (Ops.LOAD, Ops.STORE)}
  def scalar(u:UOp) -> bool: return u.shape == ()
  targets = [u for u in topo if u in deps and (
    (u.op is Ops.LOAD and u.src[0].op is Ops.INDEX and u.src[0].src[0] in params) or
    (u.op is Ops.INDEX and u.src[0] in params and u not in targets_of)) and scalar(u)]
  # the next iteration's index through a placeholder (the replacement holds rng)
  hole = UOp.variable("prefetch_index", 0, 1 << 30, rng.dtype)
  repl, stores = {}, []
  for t in targets:
    reg = UOp.alloc((1,), t.dtype, addrspace=AddrSpace.REG)
    reg = reg.after(reg[0].store(t.substitute({rng: rng.const_like(0)})))
    repl[t] = cur = reg.after(rng)[0].load()
    stores.append(reg.after(cur)[0].store(t.substitute({rng: hole}).substitute({hole: (rng + 1).minimum(last)})))
  return body.substitute(repl), stores

def _decode_linear(out:UOp, out_features:int, group_count:int, group_dot, name:str, active:Callable|None=None,
                   estimates:Estimates|None=None, multi:int=1, bundles:tuple[int, Callable]|None=None) -> UOp:
  # One wave per output/chunk of 32 groups; group neighboring rows to amortize workgroup scheduling. the waves of the chunks of an output
  # share the workgroup: they add up their sums in LDS
  # a wave serves 32//lanes outputs with `lanes` lanes each, whichever leaves fewer lanes idle. at most 16 waves in a workgroup: a wave sums
  # `per` slots of its lanes
  # active(token) -> bool: inactive tokens (experts of other devices) skip their loop and store 0
  # multi: the tokens (all of out's rows) a lane computes for its output, they share its weight loads (a few tokens: the drafts of speculative
  # decoding read the weights once)
  # bundles (count, store_row): the grid's rows are bundles of `multi` tokens (b*multi+m) that share the weights (the rows of an expert),
  # active(token) of its first token, a token's output row store_row(token) (-1: none)
  def waste(lanes:int) -> int: return lanes*((group_count+lanes-1)//lanes) - group_count
  # Four-group projections (128-wide expert slices and KDA gates) otherwise waste half the lanes.
  choices = (32, 16, 8, 4) if group_count == 4 else (32, 16, 8)
  lanes = min((l for l in choices if out_features % (32//l) == 0), key=lambda l: (waste(l), -l))
  outs_per_wave, slots = 32 // lanes, (group_count+lanes-1)//lanes
  per = (slots + 15) // 16
  # Five slots fit one wave: splitting them into two four-slot chunks wastes three slots and needs an LDS reduction.
  if active is not None: per = min(5 if slots == 5 else 4, slots)
  elif group_count <= 48 and out_features >= 2048: per = min(max(per, 4), slots)
  chunks = (slots + per-1) // per
  rows = math.gcd(out_features // outs_per_wave, 4)
  while rows > 1 and rows*chunks > (32 if active is not None else 16): rows //= 2
  assert multi == 1 or bundles is not None or (active is None and out.shape[0] == multi), "multi: every row of out, no inactive tokens"
  # rep: the outputs a lane computes (active rows: a wave reads more than the few hundred bytes of one output of a narrow input)
  rep = 2 if active is not None else 1
  while rep > 1 and out_features % (rows*outs_per_wave*rep): rep //= 2
  row = UOp.range((bundles[0] if bundles else out.shape[0]//multi)*out_features//(rows*outs_per_wave*rep), 0, AxisType.GLOBAL)
  wave, chunk, lane = UOp.range(rows, 1, AxisType.LOCAL), UOp.range(chunks, 3, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  sub, which = lane % lanes, lane // lanes
  token_outputs = [((row*rows+wave)*rep + r)*outs_per_wave + which for r in range(rep)]
  outputs = [t % out_features for t in token_outputs]
  tokens = [token_outputs[0] // out_features] if multi == 1 else [UOp.const(t, dtypes.weakint) for t in range(multi)]
  if bundles: tokens = [token_outputs[0] // out_features * multi + t for t in range(multi)]
  def dot(token:UOp, group:UOp, output:UOp=outputs[0]) -> UOp:
    return (group < group_count).where(group_dot(token, output, group.minimum(group_count-1)), UOp.const(0, dtypes.float32))
  if active is None:
    totals = [functools.reduce(lambda a,s: a + dot(t, sub+(chunk*per+s)*lanes), range(1, per), dot(t, sub+chunk*per*lanes)) for t in tokens]
  else:
    # a data dependent loop: inactive tokens run it 0 times
    s = UOp.range(active(tokens[0]).where(per, 0), 5, AxisType.LOOP)
    acc = _reg((rep*len(tokens),), 0)
    group = sub+(chunk*per+s)*lanes
    acc = acc.after(UOp.group(*[acc[t*rep+r].store(acc.after(s)[t*rep+r].load() + dot(tok, group, outputs[r]))
                                for t, tok in enumerate(tokens) for r in range(rep)]).end(s))
    totals = [acc[i].load() for i in range(rep*len(tokens))]
  for offset in (16, 8, 4, 2, 1)[(32//lanes).bit_length()-1:]:
    totals = [total + _cfn(False, src=(total,), arg=
      (f"__builtin_bit_cast(float, __builtin_amdgcn_ds_swizzle(__builtin_bit_cast(int, {{0}}), {0x1f | offset<<10}))", dtypes.float))
      for total in totals]
  if chunks > 1:
    lds = UOp.alloc((len(totals), rows, chunks, outs_per_wave), dtypes.float32, addrspace=AddrSpace.LOCAL)
    lds = lds.after(UOp.group(*(lds[t, wave, chunk, which.valid(sub.eq(0))].store(total) for t, total in enumerate(totals))).barrier())
    totals = [functools.reduce(lambda a,c: a + lds[t, wave, c, which].load(), range(1, chunks), lds[t, wave, 0, which].load())
              for t in range(len(totals))]
  first = sub.eq(0) & chunk.eq(0)
  if active is None: stores = [out[token, outputs[0].valid(first)].store(total.cast(out.dtype)) for token, total in zip(tokens, totals)]
  elif bundles:
    stores = [out[(o:=bundles[1](tok)).maximum(0), outputs[r].valid(first & (o >= 0))].store(totals[t*rep+r].cast(out.dtype))
              for t, tok in enumerate(tokens) for r in range(rep)]
  else: stores = [out[tokens[0], outputs[r].valid(first)].store(totals[r].cast(out.dtype)) for r in range(rep)]
  return UOp.group(*stores).end(row, wave, chunk, lane).sink(arg=KernelInfo(name=name, opts_to_apply=(), estimates=estimates))

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
  return _table(("iq_grid", device, ggml_type), device, tuple((v >> (32*i)) & 0xffffffff for v in grid for i in range(words)), dtypes.uint32)

def tables_ready(device:str|tuple[str, ...]):
  """the constant tables of the kernels, made (outside a jit: its runs read them)"""
  iq4_half_lut(device)
  for t in (IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S): _iq_grid(device, t)

def _iq_even_signs(signs:UOp) -> UOp:
  parity = signs ^ (signs >> 4)
  parity ^= parity >> 2
  parity ^= parity >> 1
  return signs | ((parity & 1) << 7)

def _iq_signed_word(word:UOp, signs:UOp) -> UOp:
  # sign bit i to bit 0 of byte i (the shifted copies of the 4 bits don't overlap: no carries), all of the byte's bits for the mask
  ones = ((signs & 15) * 0x00204081) & 0x01010101
  # Grid magnitudes are nonzero and <128, so each byte can be negated without a carry into its neighbor.
  # (ones*255 by a byte permute, selector 12 a 0x00 byte, 13 a 0xff one: a 32 bit multiply is quarter rate)
  mask = _amd_byte_perm(UOp.const(0, dtypes.uint32), UOp.const(0, dtypes.uint32), ones + 0x0C0C0C0C)
  return (word ^ mask) + ones

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
  if ggml_type == Q2_0:
    # (base: the 64 weight block, the group its half) four 2 bit q of a byte as bytes, q-1 signed: +0x7f then ^0x80 a byte (no carries)
    q = byte(2 + (subgroup%2)*8 + i)
    q4 = (q & 3) | ((q & 12) << 6) | ((q & 48) << 12) | ((q & 192) << 18)
    return (q4 + 0x7F7F7F7F) ^ 0x80808080
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


def _quant_frag_words(raw:UOp, base:UOp, subgroup:UOp, ph:UOp, ggml_type:int, grid:UOp|None, signed:bool=False) -> list[tuple[UOp, UOp]]:
  """
  the words of a WMMA fragment's lane (half ph) of a group: (_quant_word h*4+ph, h*4+ph+2) for h 0, 1, the bytes they share loaded once (IQ3_S).
  signed: grid is the IQ3_S grid words with their signs applied (index | signs << 9, _signed_grid)
  """
  if ggml_type == IQ3_S and raw.dtype.itemsize == 2 and grid is not None:
    # words i = 2j+ph (j 0..3): index byte ph of qs halfword j, its 9th bit bit i of qh, its signs nibble ph of signs byte j
    sh = (ph * 8).cast(dtypes.uint32)
    qs = [(raw[base + 1 + subgroup*4 + j].load().cast(dtypes.uint32) >> sh) & 255 for j in range(4)]
    qh = (raw[base + 33 + subgroup//2].load().cast(dtypes.uint32) >> ((subgroup % 2)*8).cast(dtypes.uint32)) & 255
    sg = [raw[base + 37 + subgroup*2 + k].load().cast(dtypes.uint32) for k in range(2)]
    def word(j:int) -> UOp:
      index = qs[j] | (((qh >> (ph.cast(dtypes.uint32) + 2*j)) & 1) << 8)
      signs = (sg[j//2] >> (8*(j % 2))) & 255
      if signed: return grid[index | (((signs >> (4*ph).cast(dtypes.uint32)) & 15) << 9)]
      return _iq_signed_word(grid[index], signs >> (4*ph).cast(dtypes.uint32))
    return [(word(2*h), word(2*h+1)) for h in range(2)]
  return [tuple(_quant_word(raw, base, subgroup, h*4 + i, ggml_type, grid) for i in (ph, ph+2)) for h in range(2)]  # type: ignore[misc]

def _quant_group_words(raw:UOp, base:UOp, subgroup:UOp, ggml_type:int, grid:UOp|None) -> list[UOp]:
  """the 8 words of _quant_word of a group (32 weights) of a block: the bytes they share loaded once (IQ3_S, IQ4_XS), else word by word"""
  if ggml_type == IQ3_S and raw.dtype.itemsize == 2 and grid is not None:
    # qs (8 index bytes), qh (the 9th bits), signs (a nibble a word): halfwords
    qs = [raw[base + 1 + subgroup*4 + k].load().cast(dtypes.uint32) for k in range(4)]
    qh = (raw[base + 33 + subgroup//2].load().cast(dtypes.uint32) >> ((subgroup % 2)*8).cast(dtypes.uint32)) & 255
    sg = [raw[base + 37 + subgroup*2 + k].load().cast(dtypes.uint32) for k in range(2)]
    words = []
    for i in range(8):
      index = ((qs[i//2] >> (8*(i % 2))) & 255) | (((qh >> i) & 1) << 8)
      signs = (sg[i//4] >> (8*((i//2) % 2))) & 255
      words.append(_iq_signed_word(grid[index], signs >> (4*(i % 2))))
    return words
  if ggml_type == IQ4_XS and raw.dtype.itemsize == 4:
    packed = [raw[base + 2 + subgroup*4 + k].load() for k in range(4)]
    return [_iq4_bytes(packed[i % 4], 4*(i//4)) for i in range(8)]
  return [_quant_word(raw, base, subgroup, i, ggml_type, grid) for i in range(8)]

@functools.cache
def _quant_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, *grids:UOp,
                         out_features:int, in_features:int, ggml_type:int) -> UOp:
  return _quant_decode_body(out, raw, xq, xd, xs, grids, out_features, in_features, ggml_type)

def _quant_decode_body(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, grids:tuple[UOp, ...], out_features:int, in_features:int, ggml_type:int,
                       gate:Callable|None=None, xtoken:Callable|None=None, name:str="linear_", active:Callable|None=None,
                       bundle:tuple[int, int, Callable]|None=None) -> UOp:
  # gate(token, output, base) -> base: moves (and predicates) the weight rows, xtoken(token) -> the activation row of an output row
  def group_dot(token:UOp, output:UOp, group:UOp) -> UOp:
    block, subgroup = group//8, group%8
    base = (output*in_features//256 + block) * (QUANT_SIZES[ggml_type]//raw.dtype.itemsize)
    if ggml_type in SMALL_BLOCK_QUANTS: base = _small_block(ggml_type, output*in_features//32 + group)
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
  if bundle is not None:
    # bundles of rows of an expert (moe_bundles): the rows of a bundle share the weight loads
    bw = bundle[0]*out_features*in_features
    return _decode_linear(out, out_features, in_features//32, group_dot, name+QUANT_NAMES[ggml_type], active=active, multi=bundle[1],
                          bundles=(bundle[0], bundle[2]), estimates=Estimates(2*bw*bundle[1], bw*QUANT_SIZES[ggml_type]//256, 0))
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
  # a few tokens share the weight loads
  multi = int(out.shape[0]) if gate is None and xtoken is None and 1 < out.shape[0] <= DECODE_MULTI else 1
  return _decode_linear(out, out_features, in_features//32, group_dot, name+QUANT_NAMES[ggml_type], multi=multi)

def _is_buffer(x:Tensor) -> bool:
  # x is a kernel's output (its views): every load of it in a kernel is a load, not a computation (each element is read by many outputs)
  u = x.uop
  while u.op in (Ops.RESHAPE, Ops.SHRINK, Ops.EXPAND, Ops.PERMUTE) or (u.op is Ops.CAST and dtypes.is_float(u.dtype)): u = u.src[0]
  return u.op in (Ops.AFTER, Ops.BUFFER, Ops.PARAM, Ops.ALLOC, Ops.STAGE)


def _wmma_layout(out:UOp, out_features:int, token_tile:int, output_tiles:int, group:tuple[UOp, ...]=(), rows:int|None=None, waves:int=2):
  # grouped experts: the token blocks cover the sorted rows (group[0]), not the output. rows: the rows of out to cover (default all)
  # output_tiles: at most that many, as many as divide the outputs
  output_tiles = next(t for t in (output_tiles, 4, 2, 1) if t <= output_tiles and out_features % (16*t) == 0)
  output_waves = next(w for w in (waves, 8, 4, 2, 1) if w <= waves and out_features % (16*output_tiles*w) == 0)
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

def _wmma_values(accs, update, half, lane, wave, output_waves, rdna4:bool=False) -> list[list[tuple[UOp, ...]]]:
  # the accumulators' values of every output tile and token tile, in the order of the tile's tokens
  # (RDNA4: a half-wave's are eight consecutive rows, the tokens' order already)
  if rdna4: return [[tuple(acc.after(update)[i].load() for i in range(8)) for acc in output_accs] for output_accs in accs]
  # the accumulator fragment halves are exchanged between lane pairs (l, l^16) through LDS (a ds_swizzle without CUSTOM), or by permlanex16
  # when that LDS doesn't fit (the gate and up tiles of grouped experts with 8 waves: 64 KB)
  flat_accs = [acc for output_accs in accs for acc in output_accs]
  if output_waves*32*len(flat_accs)*8*4 > 32768:
    def lane_values(ai:int) -> tuple[UOp, ...]:
      own = tuple(flat_accs[ai].after(update)[i].load() for i in range(8))
      peer = tuple(_lane_xor(v, 16) for v in own)
      low = half.eq(0)
      return tuple(low.where(own[i], peer[i+4]) if j == 0 else low.where(peer[i], own[i+4]) for i in range(4) for j in range(2))
    return [[lane_values(ot*len(accs[0])+tile) for tile in range(len(accs[0]))] for ot in range(len(accs))]
  lds = UOp.alloc((output_waves, 32, len(flat_accs)*8), dtypes.float32, addrspace=AddrSpace.LOCAL)
  stores = [lds[wave, lane, a*8+i].store(acc.after(update)[i].load()) for a,acc in enumerate(flat_accs) for i in range(8)]
  lds = lds.after(*stores)
  def values(ai:int) -> tuple[UOp, ...]:
    own = tuple(lds[wave, lane, ai*8+i].load() for i in range(8))
    peer = tuple(lds[wave, lane ^ 16, ai*8+i].load() for i in range(8))
    low = half.eq(0)
    return tuple(low.where(own[i], peer[i+4]) if j == 0 else low.where(peer[i], own[i+4]) for i in range(4) for j in range(2))
  tt = len(accs[0])
  return [[values(ot*tt+tile) for tile in range(tt)] for ot in range(len(accs))]

def _act_stores(out, outputs, tokens, vals, act):
  # the paired output tiles are the gate and up rows of the same 16 columns (interleaved weights): the activation is stored
  assert len(outputs) == 2, "a gated activation needs the gate and up tiles of a column"
  col = outputs[0] - (outputs[0] // 32) * 16
  return [out[token, col].store(_gated(g[i], u[i], act).cast(out.dtype))
          for tile_tokens, g, u in zip(tokens, vals[0], vals[1]) for i, token in enumerate(tile_tokens)]

def _wmma_stores(out, outputs, tokens, accs, update, half, lane, wave, output_waves, rdna4:bool=False):
  vals = _wmma_values(accs, update, half, lane, wave, output_waves, rdna4)
  return [out[token, output].store(value.cast(out.dtype)) for output,output_vals in zip(outputs, vals)
          for tile_tokens,tile_vals in zip(tokens, output_vals) for token,value in zip(tile_tokens, tile_vals)]

def _quant_linear_wmma(out, x, out_features, in_features, type_words, layout, dequant, name, group:tuple[UOp, ...]=(),
                       expert_words:int=0, x_div:int=0, act:tuple[float, float, float]|None=None, rdna4:bool=False):
  # group (entries, tiles): grouped experts, the row r is the selection entries[r] (padding: -1) of the expert tiles[r//16] (-1: none).
  # x_div > 0: the x row is entries[r]//x_div and the output row r, 0: the x row is r and the output row the selection, -1: both are r
  # (the padding rows aren't stored)
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
    assert token_tile == _moe_tile(group), "grouped experts have tiles of the routing's rows"
    expert = tiles[token_block].load()
    # a tile without an expert skips its loop: no estimate from the data dependent loop
    group_rng = UOp.range((expert >= 0).where(in_features // Q8_GROUP_SIZE, 0), 4, AxisType.LOOP)
    # clamped: loads that don't depend on the loop can be hoisted above it, a tile without an expert reads expert 0
    expert_base = expert.maximum(0).cast(dtypes.weakint) * expert_words
    weights = rows.shape[0]*out_features*in_features
    estimates = Estimates(2*weights, weights*type_words*2//GGML_BLOCK_SIZE, weights*type_words*2//GGML_BLOCK_SIZE)
    x_rows = tuple((r:=rows[t].load()).valid(r >= 0) // x_div for t in input_tokens) if x_div > 0 else input_tokens
    if x_div == 0: tokens = tuple(tuple((r:=rows[t].load()).valid(r >= 0) for t in tile_tokens) for tile_tokens in tokens)
    if x_div == -1: tokens = tuple(tuple(t.valid(rows[t].load() >= 0) for t in tile_tokens) for tile_tokens in tokens)
  else:
    group_rng = UOp.range(in_features // Q8_GROUP_SIZE, 4, AxisType.LOOP)
    x_rows = input_tokens
  block, subgroup = group_rng // 8, group_rng % 8
  wmma_accs = [list(output_accs) for output_accs in accs]
  # (RDNA4: k2 of the element index is lane bit 4, a lane has two groups of four of the 16)
  ks = tuple(i%4 + physical_half*4 + (i//4)*8 for i in range(8)) if rdna4 else tuple(range(16))
  # (RDNA4, f16 activations, 4 output waves at least: the tile's activations of the group through LDS (the waves share them), 8 halves a thread)
  stage = rdna4 and x.dtype == dtypes.float16 and output_waves >= 4
  if stage:
    tile_rows, threads, tid = token_tile, output_waves*32, wave*32 + lane
    lds_a = UOp.alloc((tile_rows, 40), dtypes.float16, addrspace=AddrSpace.LOCAL)  # padded rows
    st = []
    for k in range((tile_rows*4 + threads - 1) // threads):
      e = tid + k*threads
      rr, c = (e // 4).minimum(tile_rows-1), e % 4
      xr = ((r:=rows[token_block*tile_rows + rr].load()).valid(r >= 0) // x_div) if group and x_div > 0 else token_block*tile_rows + rr
      st += [lds_a[rr.valid(e < tile_rows*4), c*8 + i].store(x[xr, group_rng*32 + c*8 + i]) for i in range(8)]
    lds_a = lds_a.after(UOp.group(*st).barrier())
  for half in range(2):
    if stage: afrags = tuple(UOp.stack(*(lds_a[t*16 + lane % 16, half*16 + i].load() for i in ks)) for t in range(len(x_rows)))
    else: afrags = tuple(UOp.stack(*(x[input_token, group_rng*32 + half*16 + i].cast(dtypes.float16) for i in ks)) for input_token in x_rows)
    for output_tile,output in enumerate(outputs):
      bfrag = UOp.stack(*dequant(expert_base + output*output_words + block*type_words, subgroup, half))
      for tile,afrag in enumerate(afrags):
        previous = accs[output_tile][tile].after(group_rng) if half == 0 else wmma_accs[output_tile][tile]
        wmma_accs[output_tile][tile] = UOp.wmma(afrag, bfrag, previous, *WMMA_ARG)
  update = UOp.group(*(acc.store(value) for output_accs,output_values in zip(accs, wmma_accs)
                       for acc,value in zip(output_accs, output_values)))
  # a long loop: the next group's weights and activations are loaded while this one computes (a short one doesn't pay for the registers)
  if group and in_features // Q8_GROUP_SIZE >= 32:
    update, prefetch = _prefetched(update, group_rng, in_features // Q8_GROUP_SIZE - 1)
    update = UOp.group(update, *prefetch)
  update = update.end(group_rng)
  if act is not None:
    stores = _act_stores(out, outputs, tokens, _wmma_values(accs, update, physical_half, lane, wave, output_waves, rdna4), act)
  else: stores = _wmma_stores(out, outputs, tokens, accs, update, physical_half, lane, wave, output_waves, rdna4)
  return UOp.group(*stores).end(token_block, output_block, lane, wave).sink(arg=KernelInfo(name=name, opts_to_apply=(), estimates=estimates))

@functools.cache
def _q5_linear_f16_wmma_kernel(out:UOp, raw:UOp, x:UOp, *group:UOp, out_features:int, in_features:int, ggml_type:int,
                               expert_words:int=0, x_div:int=0, act:tuple|None=None, rdna4:bool=False) -> UOp:
  token_tile, output_tiles = (64, 1 if out_features <= 1024 else 2) if out.shape[0] % 64 == 0 else \
    (32 if out.shape[0] % 32 == 0 else 16, 2)
  if group: token_tile = _moe_tile(group)
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
                            layout, dequant, f"linear_q{4 if ggml_type == Q4_K else 5}_k_f16_wmma", group, expert_words, x_div, act, rdna4)

@functools.cache
def _iq4_linear_f16_wmma_kernel(out:UOp, raw:UOp, x:UOp, lut:UOp, *group:UOp, out_features:int, in_features:int,
                                expert_words:int=0, x_div:int=0, act:tuple|None=None, rdna4:bool=False) -> UOp:
  token_tile = 32 if out_features <= 1024 and out.shape[0] % 32 == 0 else 64 if out.shape[0] % 64 == 0 and \
    out_features <= 6144 else 128 if out.shape[0] % 128 == 0 else \
    32 if out.shape[0] % 32 == 0 else 16
  if group: token_tile = _moe_tile(group)
  output_tiles = 1 if out_features <= 1024 else 2 if out_features <= 6144 else 1 if out_features < 8192 else 2
  if act is not None: output_tiles = 2
  # grouped experts: eight waves share a tile's activations (through LDS)
  layout = _wmma_layout(out, out_features, token_tile, output_tiles, group, waves=8 if group else 2)
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
  return _quant_linear_wmma(out, x, out_features, in_features, QUANT_SIZES[IQ4_XS]//4, layout, dequant, "linear_iq4_xs_f16_wmma",
                            group, expert_words, x_div, act, rdna4)

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
def _q8k_quantize_kernel(q:UOp, scale:UOp, x:UOp, tokens:int, in_features:int) -> UOp:
  # int8 groups of 32 with one scale per 256 inputs (q8_K, as the CPU dots of IQ formats): a wave a (token, block), eight inputs a lane
  blocks = in_features // 256
  token_block, lane = UOp.range(tokens*blocks, 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP)
  token, block = token_block // blocks, token_block % blocks
  values = [x.reshape(tokens, in_features)[token, block*256 + lane*8 + i].load().float() for i in range(8)]
  local = functools.reduce(lambda a, b: a.maximum(b), [v.abs() for v in values])
  d = _cfn(False, src=(warp_reduce(local, maximum=True, full_wave=True),), arg=("({0}/127.0f)", dtypes.float)).maximum(1e-8)
  quants = [_cfn(False, src=(v, d), arg=("__builtin_nearbyintf({0}/{1})", dtypes.float)).clip(-127, 127).cast(dtypes.int8) for v in values]
  words = [functools.reduce(lambda a, b: a | b, [quants[w*4+i].cast(dtypes.uint8).cast(dtypes.uint32) << (8*i) for i in range(4)]) for w in range(2)]
  group = block*8 + lane//4
  stores = [q[token, group, (lane%4)*2 + w].store(words[w]) for w in range(2)] + [scale[token, block.valid(lane.eq(0))].store(d)]
  return UOp.group(*stores).end(token_block, lane).sink(arg=KernelInfo(name="q8k_quantize", opts_to_apply=()))

def q8k_quantize(x:Tensor, tokens:int, in_features:int) -> tuple[Tensor, Tensor]:
  """int8 groups of 32 (as q8_quantize) and one scale per 256 inputs"""
  q = Tensor.empty(tokens, in_features//32, 8, dtype=dtypes.uint32, device=x.device)
  scale = Tensor.empty(tokens, in_features//256, dtype=dtypes.float32, device=x.device)
  outs = Tensor.custom_kernel(q, scale, x, fxn=functools.partial(_q8k_quantize_kernel, tokens=tokens, in_features=in_features))
  return outs[0], outs[1]

# a*b+c of the low 24 bits of a and b (a full rate multiply, the 32 bit one is quarter rate)
MAD_I24 = "({{ int r_; __asm__(\"v_mad_i32_i24 %0, %1, %2, %3\" : \"=v\"(r_) : \"v\"({0}), \"v\"({1}), \"v\"({2})); r_; }})"

@functools.cache
def _grouped_k256_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, *srcs:UOp, out_features:int, in_features:int, ggml_type:int, expert_words:int,
                         x_div:int, act:tuple|None=None, rdna4:bool=False) -> UOp:
  # grouped experts (IQ3_S, IQ4_XS) on q8_K activations: per block of 256 inputs, the int8 WMMA dots of a sub-block (32) times its integer sub-scale
  # accumulate in int32, the block is scaled once (weight d times activation d). a workgroup is a tile of MOE_TILE rows x 256 weight rows, 8
  # waves of 2 output tiles (act: the gate and up of 16 columns) by the tile's 4 token tiles. the tile's activations of a block are in LDS
  grid_src, (rows, tiles) = (srcs[0] if ggml_type == IQ3_S else None), srcs[-2:]
  TOK, WAVES, OT = _moe_tile((rows, tiles)), 8, 2
  TT, OUTB, blocks = TOK // 16, WAVES * OT * 16, in_features // 256
  type_words = QUANT_SIZES[ggml_type] // raw.dtype.itemsize
  output_words = in_features * type_words // GGML_BLOCK_SIZE
  token_block, output_block = UOp.range(rows.shape[0] // TOK, 0, AxisType.GLOBAL), UOp.range(out_features // OUTB, 1, AxisType.GLOBAL)
  lane, wave = UOp.range(WARP_SIZE, -1, axis_type=AxisType.WARP), UOp.range(WAVES, 3, axis_type=AxisType.LOCAL)
  col, ph, tid = lane % 16, lane // 16, wave * 32 + lane
  # (RDNA4 IQ3_S: the grid words with their signs applied, a lookup instead of the sign math)
  grid, signed = None, rdna4 and ggml_type == IQ3_S
  assert rdna4 or TOK <= 64, "RDNA3's K256 staging fits in 32 KiB of LDS"
  if grid_src is not None and signed:
    # the grid words with every one of the 16 sign patterns applied (index | signs << 9): a lookup instead of the sign math
    n = int(grid_src.numel())
    grid = UOp.alloc((n*16,), dtypes.uint32, addrspace=AddrSpace.LOCAL)
    entries = [tid + i*256 for i in range(n*16 // 256)]
    grid = grid.after(UOp.group(*(grid[e].store(_iq_signed_word(grid_src[e % n].load(), (e // n).cast(dtypes.uint32))) for e in entries)).barrier())
  elif grid_src is not None:
    grid = UOp.alloc((int(grid_src.numel()),), dtypes.uint32, addrspace=AddrSpace.LOCAL)
    grid = grid.after(UOp.group(*(grid[tid + i*256].store(grid_src[tid + i*256]) for i in range(int(grid.numel()) // 256))).barrier())
  expert = tiles[token_block].load()
  b = UOp.range((expert >= 0).where(blocks, 0), 4, AxisType.LOOP)
  expert_base = expert.maximum(0).cast(dtypes.weakint) * expert_words
  def xrow(r:UOp) -> UOp: return rows[r].load().maximum(0) // x_div if x_div > 0 else r
  # the tile's int8 activations of the block: rows of 8 groups of 8 words (a group's words at (w%2)*4+w//2, a stride of 10 words)
  lds_q = UOp.alloc((TOK, 8, 10), dtypes.uint32, addrspace=AddrSpace.LOCAL)
  lds_d = UOp.alloc((TOK,), dtypes.float32, addrspace=AddrSpace.LOCAL)
  st = []
  for k in range(TOK * 64 // 256):
    e = tid + k * 256
    rr, gi, wi = e // 64, (e // 8) % 8, e % 8
    st.append(lds_q[rr, gi, (wi % 2) * 4 + wi // 2].store(xq[xrow(token_block * TOK + rr), b * 8 + gi, wi].load()))
  st.append(lds_d[tid % TOK].store(xd[xrow(token_block * TOK + tid % TOK), b].load()))
  staged = UOp.group(*st).barrier()
  lds_q, lds_d = lds_q.after(staged), lds_d.after(staged)
  outputs = [output_block * OUTB + wave * OT * 16 + o * 16 + col for o in range(OT)]
  bases = [expert_base + output * output_words + b * type_words for output in outputs]
  zero = UOp.stack(*[UOp.const(0, dtypes.int32)]*8)
  accs = [[UOp.alloc((8,), dtypes.float32, addrspace=AddrSpace.REG) for _ in range(TT)] for _ in range(OT)]
  accs = [[acc.after(acc.store(acc.const_like(0))) for acc in row] for row in accs]
  # the int32 sums of the block over its sub-blocks, times their integer sub-scales (a 24 bit multiply-add: a dot of 32 int8 products of
  # weights of at most 127 (IQ3_S 15) is below 2**23, a sub-scale at most 32 in magnitude, 8 of them below 2**31)
  ints = [[UOp.alloc((8,), dtypes.int32, addrspace=AddrSpace.REG) for _ in range(TT)] for _ in range(OT)]
  ints = [[r.after(r.after(b).store(zero)) for r in row] for row in ints]
  dws = [_half(raw[base]) for base in bases]
  def sub_scale(base:UOp, sb:UOp) -> UOp:
    # the integer scale of sub-block sb: IQ3_S 1+2*s (odd, at most 31), IQ4_XS the 6 bits minus 32
    if ggml_type == IQ3_S: return (1 + 2*((_load_byte(raw, base, 106 + sb//2) >> ((sb % 2) * 4).cast(dtypes.uint32)) & 15)).cast(dtypes.int32)
    header, low = raw[base].load(), raw[base+1].load()
    return (((low >> (4*sb).cast(dtypes.uint32)) & 15) | (((header >> (16+2*sb).cast(dtypes.uint32)) & 3) << 4)).cast(dtypes.int32) - 32
  sb = UOp.range(8, 5, AxisType.LOOP)
  def acols(h:int) -> tuple: return (ph*4+2*h, ph*4+2*h+1) if rdna4 else (2*h, 2*h+4, 2*h+1, 2*h+5)
  afr = [[_i8_frag(tuple(lds_q[t*16 + col, sb, c].load() for c in acols(h))) for h in range(2)] for t in range(TT)]
  sums = []
  for o, base in enumerate(bases):
    sub = sub_scale(base, sb)
    words = _quant_frag_words(raw, base, sb, ph, ggml_type, grid, signed=signed) if rdna4 else \
      [tuple(_quant_word(raw, base, sb, h*4+i, ggml_type, grid) for i in range(4)) for h in range(2)]
    bfr = [_i8_frag(w) for w in words]
    for t in range(TT):
      dot = UOp.wmma(afr[t][1], bfr[1], UOp.wmma(afr[t][0], bfr[0], zero, *WMMA_ARG), *WMMA_ARG)
      dreg = UOp.alloc((8,), dtypes.int32, addrspace=AddrSpace.REG)
      dreg = dreg.after(dreg.store(dot))
      prev = ints[o][t].after(sb)
      sums.append(ints[o][t].store(UOp.stack(*[_cfn(True, src=(dreg[j].load(), sub, prev[j].load()), arg=(MAD_I24, dtypes.int32))
                                              for j in range(8)])))
  done = UOp.group(*sums).end(sb)
  new = [[UOp.stack(*(accs[o][t].after(b)[j].load() + ints[o][t].after(done)[j].load().float() *
                      (dws[o] * lds_d[t*16 + (ph*8 + j if rdna4 else 2*j + ph)].load())
                      for j in range(8))) for t in range(TT)] for o in range(OT)]
  # (a barrier after the block's reads: the next block's stage doesn't overwrite what a slower wave still reads)
  update = UOp.group(*(accs[o][t].store(new[o][t]) for o in range(OT) for t in range(TT))).barrier().end(b)
  def row(t:int, j:int) -> UOp: return token_block * TOK + t * 16 + (ph * 8 + j if rdna4 else 2*j + ph)
  def out_row(r:UOp) -> UOp:
    # x_div > 0: the output row is the sorted row, 0: the selection, -1: the sorted row (the padding rows aren't stored)
    if x_div > 0: return r
    return (e:=rows[r].load()).valid(e >= 0) if x_div == 0 else r.valid(rows[r].load() >= 0)
  def val(o:int, t:int, j:int) -> UOp: return accs[o][t].after(update)[j].load()
  if act is not None:
    stores = [out[out_row(row(t, j)), output_block * (OUTB // 2) + wave * 16 + col].store(_gated(val(0, t, j), val(1, t, j), act).cast(out.dtype))
              for t in range(TT) for j in range(8)]
  else:
    stores = [out[out_row(row(t, j)), output].store(val(o, t, j).cast(out.dtype))
              for o, output in enumerate(outputs) for t in range(TT) for j in range(8)]
  weights = rows.shape[0] * out_features * in_features
  est = Estimates(2*weights, weights*type_words*2//GGML_BLOCK_SIZE, weights*type_words*2//GGML_BLOCK_SIZE)
  return UOp.group(*stores).end(token_block, output_block, lane, wave).sink(
    arg=KernelInfo(name=f"grouped_{QUANT_NAMES[ggml_type]}_k256", opts_to_apply=(), estimates=est))

@functools.cache
def _quant_linear_i8_wmma_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, *srcs:UOp, out_features:int, in_features:int, ggml_type:int,
                                 expert_words:int=0, x_div:int=0, act:tuple|None=None, splits:int=1, kwaves:int=1, rdna4:bool=False) -> UOp:
  # (RDNA3's layouts: a lane's fragment is its row's 16 inputs (four words), its accumulator the rows 2j+half. RDNA4: half*8+j)
  # RDNA4 int8 WMMA on q8 activations: the weights are their packed bytes, the scales apply per 16 inputs (see _wmma_scale), like the decode.
  # splits: the inputs are split in that many ranges (workgroups for few outputs), out has the rows of every split (summed after)
  # kwaves: the inputs split over that many waves of a workgroup too (a few rows: more waves read the weights), summed in LDS
  grids, group = (srcs[:1], srcs[1:]) if ggml_type in (IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S) else ((), srcs)
  rows = group[0].shape[0] if group else out.shape[0] // splits
  token_tile = _moe_tile(group) if group else 64 if rows % 64 == 0 else 32 if rows % 32 == 0 else 16
  # (the inputs split over waves: a tile of all the rows of batched decoding, the weights read once)
  if not group and kwaves > 1 and rows % 48 == 0: token_tile = 48
  # outputs past a multiple of 32 (a few rows, e.g. the 24 of a hyper-connection): their tiles read the last row and aren't stored
  real_out, out_features = out_features, round_up(out_features, 32)
  # grouped experts: eight waves share a tile's activations (through LDS)
  layout = _wmma_layout(out, out_features, token_tile, 2, group, rows=int(rows), waves=8 if group else 2)
  output_waves, token_block, output_block, lane, wave, ph, outputs, input_tokens, tokens = layout
  if not rdna4: tokens = tuple(tuple(token_block*token_tile + t*16 + 2*i + ph for i in range(8)) for t in range(len(tokens)))
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
    if x_div > 0:
      def xrow(r:UOp) -> UOp: return (e:=entries[r].load()).valid(e >= 0) // x_div
    elif x_div == -1: store_rows = tuple(tuple(t.valid(entries[t].load() >= 0) for t in tile_tokens) for tile_tokens in tokens)
    else: store_rows = tuple(tuple((e:=entries[t].load()).valid(e >= 0) for t in tile_tokens) for tile_tokens in tokens)
  else: group_rng = UOp.range(n_groups // (splits*kwaves), 4, AxisType.LOOP)
  split = UOp.range(splits, 5, AxisType.GLOBAL)
  kw = UOp.range(kwaves, 6, AxisType.LOCAL) if kwaves > 1 else UOp.const(0, dtypes.weakint)
  g = group_rng + (split*kwaves + kw) * (n_groups // (splits*kwaves))
  block, subgroup = g // 8, g % 8
  accs = [[UOp.alloc((8,), dtypes.float32, addrspace=AddrSpace.REG) for _ in tokens] for _ in outputs]
  accs = [[acc.after(acc.store(acc.const_like(0))) for acc in row] for row in accs]
  a_rows = [xrow(t) for t in input_tokens]
  s_rows = [[xrow(t) for t in tile_tokens] for tile_tokens in tokens]
  # grouped experts of formats without a zero point: the tile's activations of the group (and their scales) through LDS, loaded once for the
  # waves (a word or a scale a thread), a lane's words of a fragment next to each other
  stage_x = bool(group) and ggml_type in GROUP_SCALE_QUANTS and ggml_type not in (Q4_K, Q5_K)
  if stage_x:
    tile_rows, threads, tid = len(input_tokens)*16, output_waves*32, wave*32 + lane
    # two buffers by the parity of the group: a wave fills the next group's while the slower ones still read this one's (a barrier between)
    par = group_rng % 2
    # rows of 10 words (8 used): the 16 rows of a fragment's lanes on different banks (2 words a lane, 8 byte aligned)
    lds_q = UOp.alloc((2*tile_rows, 10), dtypes.uint32, addrspace=AddrSpace.LOCAL)
    lds_d = UOp.alloc((2*tile_rows,), dtypes.float32, addrspace=AddrSpace.LOCAL)
    st = []
    # every thread loads words of rows and a row's scale, unconditionally (no branches): a padding row reads row 0, the threads of the same
    # scale store the same value
    def srow(r:UOp) -> UOp: return entries[r].load().maximum(0) // x_div if x_div > 0 else r
    assert (tile_rows*8) % threads == 0 and threads % tile_rows == 0, "whole words and scales for the threads"
    for k in range(tile_rows*8 // threads):
      rr, wi = (e:=tid + k*threads) // 8, (tid + k*threads) % 8
      st.append(lds_q[par*tile_rows + rr, (wi%2)*4 + wi//2].store(xq[srow(token_block*tile_rows + rr), g, wi].load()))
    st.append(lds_d[par*tile_rows + tid % tile_rows].store(xd[srow(token_block*tile_rows + tid % tile_rows), g].load()))
    staged = UOp.group(*st).barrier()
    lds_q, lds_d = lds_q.after(staged), lds_d.after(staged)
    d_x = [[lds_d[par*tile_rows + t*16 + (ph*8 + j if rdna4 else 2*j + ph)].load() for j in range(8)] for t in range(len(tokens))]
  else: d_x = [[xd[r, g].load() for r in tile_rows] for tile_rows in s_rows]
  zero_acc = UOp.stack(*[UOp.const(0, dtypes.int32)]*8)
  new_vals = [[[accs[o][t].after(group_rng)[j].load() for j in range(8)] for t in range(len(tokens))] for o in range(len(outputs))]
  # one scale per 32 inputs: both halves accumulate in int32 and the group is scaled once, else every half has its own scale
  halves = ((0, 1),) if ggml_type in GROUP_SCALE_QUANTS else ((0,), (1,))
  for hs in halves:
    # (the staged words of a group: word w at column (w%2)*4 + w//2. RDNA3: the four words of a half in order, columns 2h, 2h+4, 2h+1, 2h+5)
    def acols(half:int) -> tuple: return (ph*4 + 2*half, ph*4 + 2*half + 1) if rdna4 else (2*half, 2*half + 4, 2*half + 1, 2*half + 5)
    def awords(half:int) -> tuple: return (half*4+ph, half*4+2+ph) if rdna4 else tuple(half*4+i for i in range(4))
    if stage_x: afrags = [[_i8_frag(tuple(lds_q[par*tile_rows + t*16 + lane % 16, c].load() for c in acols(half)))
                           for half in hs] for t in range(len(a_rows))]
    else: afrags = [[_i8_frag(tuple(xq[r, g, w].load() for w in awords(half))) for half in hs] for r in a_rows]
    s_x = [[sum((xs[r, g, half].load() for half in hs), UOp.const(0, dtypes.float)) for r in tile_rows] for tile_rows in s_rows]
    for o, output in enumerate(outputs):
      base = expert_base + output*output_words + block*type_words
      if ggml_type in SMALL_BLOCK_QUANTS: base = base + _small_block(ggml_type, subgroup)
      scale, zero, minimum = _wmma_scale(raw, base, subgroup, hs[0], ggml_type)
      bfrags = [_i8_frag(tuple(_quant_word(raw, base, subgroup, half*4+i, ggml_type, grid) for i in ((ph, ph+2) if rdna4 else range(4))))
                for half in hs]
      for t, tile_afrags in enumerate(afrags):
        acc_i = zero_acc
        for afrag, bfrag in zip(tile_afrags, bfrags): acc_i = UOp.wmma(afrag, bfrag, acc_i, *WMMA_ARG)
        dot = UOp.alloc((8,), dtypes.int32, addrspace=AddrSpace.REG)
        dot = dot.after(dot.store(acc_i))
        for j in range(8):
          sx = s_x[t][j]
          term = scale * dot[j].load().float() if not zero else scale * (dot[j].load().float() - zero*sx)
          new_vals[o][t][j] = new_vals[o][t][j] + d_x[t][j] * (term if ggml_type not in (Q4_K, Q5_K, Q2_K) else term - minimum*sx)
  update = UOp.group(*(accs[o][t].store(UOp.stack(*new_vals[o][t])) for o in range(len(outputs)) for t in range(len(tokens))))
  if len(group):
    # the next group's weights and activations are loaded while this one computes
    update, prefetch = _prefetched(update, group_rng, n_groups - 1, (raw, xq, xd, xs))
    update = UOp.group(update, *prefetch)
  update = update.end(group_rng)
  def acc(o:int, t:int, j:int) -> UOp: return accs[o][t].after(update)[j].load()
  first = UOp.const(True, dtypes.bool)
  if kwaves > 1:
    # the waves' sums of their inputs through LDS, added in wave order by the first
    n = len(outputs)*len(tokens)*8
    lds = UOp.alloc((kwaves, output_waves, WARP_SIZE, n), dtypes.float32, addrspace=AddrSpace.LOCAL)
    lds = lds.after(UOp.group(*(lds[kw, wave, lane, (o*len(tokens)+t)*8+j].store(acc(o, t, j))
                                for o in range(len(outputs)) for t in range(len(tokens)) for j in range(8))).barrier())
    def acc(o:int, t:int, j:int) -> UOp:
      return functools.reduce(lambda a,b: a+b, [lds[k, wave, lane, (o*len(tokens)+t)*8+j].load() for k in range(kwaves)])
    first = kw.eq(0)
  # (the accumulators' rows are the tokens' already (store_rows): their values as they are)
  if act is not None:
    stores = _act_stores(out, outputs, store_rows, [[tuple(acc.after(update)[i].load() for i in range(8)) for acc in o_accs] for o_accs in accs], act)
  else: stores = [out[row + split * rows, (output.valid(first & (output < real_out)) if real_out != out_features else output.valid(first))].store(
                    acc(o, t, j).cast(out.dtype))
                  for o, output in enumerate(stored) for t, tile_rows in enumerate(store_rows) for j, row in enumerate(tile_rows)]
  return UOp.group(*stores).end(token_block, output_block, split, lane, wave, *([kw] if kwaves > 1 else [])).sink(
    arg=KernelInfo(name=f"linear_{QUANT_NAMES[ggml_type]}_i8_wmma", opts_to_apply=(), estimates=estimates))

@functools.cache
def _quant_linear_f16_wmma_kernel(out:UOp, raw:UOp, x:UOp, *srcs:UOp,
                                  out_features:int, in_features:int, ggml_type:int, expert_words:int=0, x_div:int=0,
                                  act:tuple|None=None, rdna4:bool=False) -> UOp:
  # srcs: the iq grid if the format has one, then the expert group (rows, tiles) if grouped
  grids, group = (srcs[:1], srcs[1:]) if ggml_type in (IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S) else ((), srcs)
  token_tile = _moe_tile(group) if group else 64 if out.shape[0] % 64 == 0 else 32 if out.shape[0] % 32 == 0 else 16
  # four output tiles of a wave share its token fragments (two: grouped experts, a tile's rows are few), four waves its activations (LDS),
  # eight for grouped experts (a tile's rows are few, the outputs many)
  tiles, waves = (2, 8) if group else \
    next(((t, w) for t, w in ((4, 4), (2, 4), (4, 2), (2, 2)) if out_features % (16*t*w) == 0), (4, 2))
  layout = _wmma_layout(out, out_features, token_tile, tiles, group, waves=waves)
  output_waves, _, _, lane, wave, physical_half, _, _, _ = layout
  word_indices = (physical_half, physical_half+2) if rdna4 else tuple(range(4))
  grid = None
  if grids:
    grid = UOp.alloc((int(grids[0].numel()),), dtypes.uint32, addrspace=AddrSpace.LOCAL)
    tid, threads = wave*32+lane, output_waves*32
    grid = grid.after(*(grid[tid+i*threads].store(grids[0][tid+i*threads]) for i in range(int(grid.numel())//threads)))
  def dequant(base:UOp, subgroup:UOp, half:int) -> tuple[UOp, ...]:
    if ggml_type in SMALL_BLOCK_QUANTS: base += _small_block(ggml_type, subgroup)  # independent blocks in the 256 weights
    words = tuple(_quant_word(raw, base, subgroup, half*4+i, ggml_type, grid) for i in word_indices)
    if ggml_type == IQ3_S:
      # the bytes to halves without floats: with the sign bit flipped a byte is the mantissa of the half 1024+u (0x64 above it, a byte
      # permute for two), minus 1152 the signed byte. q times the integer sub-scale (at most 15*31) is exact in half, times d rounds once:
      # the same half as (q*(d*sub-scale)) in float rounded
      d = raw[base].cast(dtypes.uint16).bitcast(dtypes.half)
      sub = (1+2*((_load_byte(raw, base, 106+subgroup//2) >> ((subgroup%2)*4)) & 15)).cast(dtypes.half)
      def halves(word:UOp) -> list[UOp]:
        u = word ^ 0x80808080
        pairs = [_amd_byte_perm(UOp.const(0x64646464, dtypes.uint32), u, UOp.const(sel, dtypes.uint32)) for sel in (0x04010400, 0x04030402)]
        return [((p >> sh) & 0xffff).cast(dtypes.uint16).bitcast(dtypes.half) for p in pairs for sh in (0, 16)]
      return tuple((h - 1152) * sub * d for word in words for h in halves(word))
    scale, zero, minimum = _wmma_scale(raw, base, subgroup, half, ggml_type)
    return tuple((((word >> (b*8)).cast(dtypes.uint8).bitcast(dtypes.int8).float()-zero)*scale-minimum).cast(dtypes.half)
                 for word in words for b in range(4))
  return _quant_linear_wmma(out, x, out_features, in_features, QUANT_SIZES[ggml_type]//raw.dtype.itemsize,
                            layout, dequant, f"linear_{QUANT_NAMES[ggml_type]}_f16_wmma", group, expert_words, x_div, act, rdna4)

@functools.cache
def _q8_gemm_kernel(out:UOp, raw:UOp, x:UOp, *hc:UOp, token_tile:int,
                    out_features:int=0, in_features:int=0, ggml_type:int=Q8_0, block_n:int=128, k_splits:int=1, rdna4:bool=False,
                    rows:int=0, n_out:int=1, wtile:bool=False, pfd:int=0, m_rows:int=0) -> UOp:
  # n_out: the outputs are n_out tensors (the first n_out sources), of the weight's rows in n_out equal parts (projections of the same input,
  # their weights adjacent: one kernel). a workgroup's outputs are in one of them
  srcs = (out, raw, x, *hc)
  outs, raw, x, hc = srcs[:n_out], srcs[n_out], srcs[n_out+1], srcs[n_out+2:]
  # Q8_0 weights times f16 x on RDNA3/4 WMMA: four waves, two along tokens and two along outputs.
  # rows: x's rows past them aren't read, the last one's again (in the caches; the outputs of the rows past them aren't used)
  # Stage activations and dequantized weights in LDS: each weight is dequantized once for token_tile tokens.
  # RDNA3 uses 16-element input fragments and interleaved accumulator rows, with LDS capped at 32 KiB.
  M, N = cast(int, out.shape[0]), n_out*cast(int, out.shape[-1])
  K, streams = cast(int, x.shape[-1]), 4 if hc else 1
  # (hc: out may have fewer rows than x, its first ones (a device's real rows of batched decoding): the others aren't stored)
  out_rows, M = M, (cast(int, x.numel()) // K if hc else round_up(M, token_tile) if n_out > 1 else M)
  assert k_splits > 0 and M % k_splits == 0 and K % (32*k_splits) == 0
  assert k_splits == 1 or ggml_type == Q8_0 or (ggml_type == Q6_K and (K // (32*k_splits)) % 8 == 0), "a split of whole blocks"
  M //= k_splits
  # (x of fewer rows than the tiles: not padded, the rows past them read its last one)
  if (xrows:=cast(int, x.numel()) // K) < M: rows = min(rows or xrows, xrows)
  x = x.reshape(xrows, K)
  # (token_tile 16: a tile of the tensor cores' 16 rows, two waves (along the outputs) a workgroup: a device's few rows of batched decoding)
  assert block_n in ((32, 64) if hc else (64, 128)) and token_tile in (16, *range(32, 1 << 16, 32)) and M % token_tile == 0
  assert not hc or (len(hc) == 1 and hc[0].shape == (M, 4*N) and k_splits == 1 and ggml_type == Q8_0)
  OW, TW, OT, groups = 2, 1 if token_tile == 16 else 2, streams*block_n // 32, K // (32*k_splits)
  NT = 32 * OW * TW  # threads a workgroup
  TT = token_tile // (16 * TW)  # token tiles of a wave
  # Outputs past N read the last weight row and aren't stored. (m_rows: the tiles of the first m_rows rows alone, the others not computed:
  # their outputs not written)
  m_tiles = round_up(min(m_rows, M), token_tile) // token_tile if m_rows else M // token_tile
  token_block, output_block = UOp.range(m_tiles, 0, AxisType.GLOBAL), UOp.range(round_up(N, block_n) // block_n, 1, AxisType.GLOBAL)
  split = UOp.range(k_splits, 2, AxisType.GLOBAL)
  lane, wave = UOp.range(WARP_SIZE, -1, axis_type=AxisType.WARP), UOp.range(OW * TW, 3, axis_type=AxisType.LOCAL)
  col, half = lane % 16, lane // 16
  wt, wo, tid = wave // OW, wave % OW, wave * 32 + lane
  # GS groups of 32 inputs a stage (a barrier a stage)
  # (the hyper-connections' mix of a few rows (decoding's): two groups a stage, the next group's weights prefetched)
  GS = next(gs for gs in (2 if M <= 64 else 1, 1) if groups % gs == 0) if hc else \
    next(gs for gs in (2, 1) if groups % gs == 0 and (block_n*gs % NT == 0 or gs == 1) and
                        (token_tile * (32*gs + 8) <= 128 * 72 if rdna4 else (token_tile+max(block_n, NT//gs))*(32*gs+8)*2 <= 32768))
  stage_rows = max(streams*block_n, NT//GS)  # full waves stage weights, including unused rows when block_n=64 and GS=1
  assert not wtile or (not hc and stage_rows == block_n and ggml_type == Q8_0), "tile-major weights: a stage's rows are the block's"
  assert rdna4 or (token_tile+stage_rows)*(32*GS+8)*2 <= 32768
  gr = UOp.range(groups // GS, 4, AxisType.LOOP)
  g = gr * GS + split*groups
  # the stage: x (token_tile, 32*GS) in eight halves a thread, the weights (128, 32*GS): the blocks of output tid
  lds_x = UOp.alloc((token_tile, 32*GS + 8), dtypes.float16, addrspace=AddrSpace.LOCAL)
  lds_w = UOp.alloc((stage_rows, 32*GS + 8), dtypes.float16, addrspace=AddrSpace.LOCAL)
  # Q8_PREFETCH (Q8_0): the next group's x and weight words loaded into registers before this group's tensor core work, converted and stored
  # to LDS after it (their latency hidden by it, not a stall of every group)
  # (ggml_type F32T: raw is a float weight transposed (K, N), the router's: converted to half as it's staged)
  prefetch = ggml_type in (Q8_0, F32T) and (not hc or M <= 64)
  # pfd: a stage's loads issued pfd stages before it (the loop unrolled pfd times, a set of registers a stage): more bytes in flight than the
  # next stage's (a stage's tensor core work is short next to the memory's latency)
  dist = pfd if prefetch and pfd > 1 and (groups // GS) % pfd == 0 and groups // GS > pfd else 0
  def pf_loads(gg:UOp) -> list[UOp]:
    out:list[UOp] = []
    for k in range((token_tile * 4 * GS + NT-1) // NT):
      e = tid + k * NT
      r, c = (e // (4*GS)).minimum(token_tile - 1), e % (4*GS)
      xr = token_block * token_tile + r
      out += [x[xr.minimum(rows-1) if rows else xr, gg * 32 + c * 8 + i].load() for i in range(8)]
    for bi in range(stage_rows * GS // NT):
      gi, wr = (bi*NT + tid) // stage_rows, (bi*NT + tid) % stage_rows
      row = (output_block * block_n + (wr % block_n if hc else wr)).minimum(N-1) + (wr // block_n * N if hc else 0)
      if ggml_type == F32T:
        out += [raw.reshape(K*k_splits, N)[(gg + gi) * 32 + i, row].load() for i in range(32)]
        continue
      # (wtile: the weights tile-major: a stage's block_n rows x GS blocks contiguous, the threads' blocks one after the other)
      base = ((((output_block*(groups*k_splits//GS) + gg//GS)*GS + gi)*block_n + wr) if wtile else (row * groups*k_splits + gg + gi)) * 17
      out += [raw[base + w].load() for w in range(17)]
    return out
  def pf_stores(vals:list[UOp], bx:UOp, bw:UOp) -> list[UOp]:
    out, j = [], 0
    for k in range((token_tile * 4 * GS + NT-1) // NT):
      e = tid + k * NT
      r, c = (e // (4*GS)).minimum(token_tile - 1), e % (4*GS)
      out += [bx[r.valid(e < token_tile * 4 * GS), c * 8 + i].store(vals[j + i]) for i in range(8)]
      j += 8
    for bi in range(stage_rows * GS // NT):
      gi, wr = (bi*NT + tid) // stage_rows, (bi*NT + tid) % stage_rows
      if ggml_type == F32T:
        out += [bw[wr, gi * 32 + i].store(vals[j + i].cast(dtypes.half)) for i in range(32)]
        j += 32
        continue
      wv, j = vals[j:j+17], j + 17
      scale = wv[0].cast(dtypes.uint16).bitcast(dtypes.float16)
      for w in range(8):
        u = (wv[1 + 2*w].cast(dtypes.uint32) | (wv[2 + 2*w].cast(dtypes.uint32) << 16)) ^ 0x80808080
        pairs = [_amd_byte_perm(UOp.const(0x64646464, dtypes.uint32), u, UOp.const(sel, dtypes.uint32)) for sel in (0x04010400, 0x04030402)]
        halves = [((p >> sh) & 0xffff).cast(dtypes.uint16).bitcast(dtypes.half) for p in pairs for sh in (0, 16)]
        out += [bw[wr, gi * 32 + w * 4 + i].store((h - 1152) * scale) for i, h in enumerate(halves)]
    return out
  lds_x0, lds_w0 = lds_x, lds_w
  if dist: pass
  elif prefetch:
    first = UOp.group(*pf_stores(pf_loads(UOp.const(0, dtypes.weakint) + split*groups), lds_x, lds_w)).barrier()
    nxt_vals = pf_loads((gr + 1).minimum(groups // GS - 1) * GS + split*groups)
    lds_x, lds_w = lds_x.after(first, gr), lds_w.after(first, gr)
  else:
    st = []
    for k in range((token_tile * 4 * GS + NT-1) // NT):
      e = tid + k * NT
      r, c = (e // (4*GS)).minimum(token_tile - 1), e % (4*GS)
      xr = token_block * token_tile + r
      st += [lds_x[r.valid(e < token_tile * 4 * GS), c * 8 + i].store(x[xr.minimum(rows-1) if rows else xr, g * 32 + c * 8 + i].load())
             for i in range(8)]
    for bi in range(stage_rows * GS // NT):
      gi, wr = (bi*NT + tid) // stage_rows, (bi*NT + tid) % stage_rows
      row = (output_block * block_n + (wr % block_n if hc else wr)).minimum(N-1) + (wr // block_n * N if hc else 0)
      if ggml_type == Q6_K:
        # (Q6_K: a group's 32 weights of a block of 256, their 6 bit values less 32 times the scale of their half (16), in half as the f16 path's)
        gg = g + gi
        bbase = (row * (groups*k_splits // 8) + gg // 8) * (QUANT_SIZES[Q6_K] // raw.dtype.itemsize)
        words = _quant_group_words(raw, bbase, gg % 8, Q6_K, None)
        scales = [_wmma_scale(raw, bbase, gg % 8, h, Q6_K)[0].cast(dtypes.float16) for h in (0, 1)]
        st += [lds_w[wr, gi * 32 + w * 4 + i].store((((words[w] >> (8*i)) & 255).cast(dtypes.int32) - 32).cast(dtypes.float16) * scales[w // 4])
               for w in range(8) for i in range(4)]
        continue
      base = ((((output_block*(groups*k_splits//GS) + g//GS)*GS + gi)*block_n + wr) if wtile else (row * groups*k_splits + g + gi)) * 17
      scale = raw[base].load().cast(dtypes.uint16).bitcast(dtypes.float16)
      # the bytes to halves without floats (as F16_FAST_CVT): with the sign bit flipped a byte is the mantissa of the half 1024+u, minus 1152 the
      # signed byte (exact), times the scale rounds once: the same halves
      for w in range(8):
        u = (raw[base + 1 + 2*w].load().cast(dtypes.uint32) | (raw[base + 2 + 2*w].load().cast(dtypes.uint32) << 16)) ^ 0x80808080
        pairs = [_amd_byte_perm(UOp.const(0x64646464, dtypes.uint32), u, UOp.const(sel, dtypes.uint32)) for sel in (0x04010400, 0x04030402)]
        halves = [((p >> sh) & 0xffff).cast(dtypes.uint16).bitcast(dtypes.half) for p in pairs for sh in (0, 16)]
        st += [lds_w[wr, gi * 32 + w * 4 + i].store((h - 1152) * scale) for i, h in enumerate(halves)]
    staged = UOp.group(*st).barrier()
    lds_x, lds_w = lds_x.after(staged), lds_w.after(staged)
  ks = tuple(i % 4 + half * 4 + (i // 4) * 8 for i in range(8)) if rdna4 else tuple(range(16))
  accs = [[UOp.alloc((8,), dtypes.float32, addrspace=AddrSpace.REG) for _ in range(TT)] for _ in range(OT)]
  accs = [[acc.after(acc.store(acc.const_like(0))) for acc in row] for row in accs]
  def group_wmma(cur:list[list[UOp]], lx:UOp, lw:UOp) -> list[list[UOp]]:
    for kh in range(2*GS):
      afr = [UOp.stack(*(lx[wt * TT * 16 + t * 16 + col, kh * 16 + i].load() for i in ks)) for t in range(TT)]
      bfr = [UOp.stack(*(lw[(o//(OT//streams))*block_n + wo*(OT//streams)*16 + (o%(OT//streams))*16 + col,
                            kh * 16 + i].load() for i in ks)) for o in range(OT)]
      cur = [[UOp.wmma(afr[t], bfr[o], cur[o][t], *WMMA_ARG) for t in range(TT)] for o in range(OT)]
    return cur
  if dist:
    iters = groups // GS
    gr = UOp.range(iters // dist, 4, AxisType.LOOP)
    init = [pf_loads(UOp.const(d*GS, dtypes.weakint) + split*groups) for d in range(dist)]
    regs = [[UOp.alloc((1,), v.dtype, addrspace=AddrSpace.REG) for v in vs] for vs in init]
    regs = [[r.after(r[0].store(v)) for r, v in zip(rs, vs)] for rs, vs in zip(regs, init)]
    update = gr
    for d in range(dist):
      vals = [r.after(update)[0].load() for r in regs[d]]
      # (slot d's stage of the next iteration: loaded now, pfd stages ahead)
      nxt = pf_loads(((gr + 1) * dist + d).minimum(iters - 1) * GS + split*groups)
      staged = UOp.group(*pf_stores(vals, lds_x0.after(update), lds_w0.after(update))).barrier()
      cur = group_wmma([[accs[o][t].after(update) for t in range(TT)] for o in range(OT)], lds_x0.after(staged), lds_w0.after(staged))
      update = UOp.group(*(accs[o][t].store(cur[o][t]) for o in range(OT) for t in range(TT)),
                         *(r.after(v)[0].store(n) for r, v, n in zip(regs[d], vals, nxt))).barrier()
    update = update.end(gr)
  else:
    cur = group_wmma([[accs[o][t].after(gr) for t in range(TT)] for o in range(OT)], lds_x, lds_w)
    # (a barrier after the group's reads: the next group's stage doesn't overwrite what a slower wave still reads)
    update = UOp.group(*(accs[o][t].store(cur[o][t]) for o in range(OT) for t in range(TT))).barrier()
    if prefetch: update = UOp.group(*pf_stores(nxt_vals, lds_x0.after(update), lds_w0.after(update))).barrier()
    update = update.end(gr)
  def ocol(o:int) -> UOp:
    c = output_block * block_n + wo * (OT//streams) * 16 + o * 16 + col
    return c.valid(c < N) if N % block_n else c
  stores = []
  for o, t, i in itertools.product(range(OT//streams), range(TT), range(8)):
    r = token_block * token_tile + wt * TT * 16 + t * 16 + (half * 8 + i if rdna4 else 2*i + half)
    value = sum(_sigmoid(accs[s*(OT//streams)+o][t].after(update)[i].load()) * hc[0][r.minimum(rows-1) if rows else r, s*N+ocol(o)].load()
                for s in range(streams)) / streams if hc else accs[o][t].after(update)[i].load()
    orow = (split*M + r).valid(r < out_rows) if out_rows < M else split*M + r
    if n_out == 1: stores.append(out[orow, ocol(o)].store(value.cast(out.dtype)))
    else:
      ne, which = N // n_out, output_block * block_n // (N // n_out)
      stores += [o_[orow, (ocol(o) - j*ne).valid(which.eq(j))].store(value.cast(o_.dtype)) for j, o_ in enumerate(outs)]
  # (LIN_NAMES: the shapes in the name, for profiles)
  name = f"q8_gemm_{M}_{N}_{K}{'_hc' if hc else ''}_k{k_splits}_t{token_tile}_b{block_n}" if getenv("LIN_NAMES") else "q8_gemm"
  return UOp.group(*stores).end(token_block, output_block, split, lane, wave).sink(arg=KernelInfo(name=name, opts_to_apply=()))

F32T = -1  # (_q8_gemm_kernel: a float weight transposed)

def f32t_gemm(x:Tensor, wt:Tensor) -> Tensor|None:
  """
  x (..., K) @ wt (K, N) float (a transposed float weight, a router's) on the tensor cores in float (the weight in half as it's staged): rows
  split over the devices (wt on all), (..., N) float. None: not those shapes (the generic matmul's kernel: a few dozen workgroups)
  """
  K, N = int(wt.shape[0]), int(wt.shape[1])
  R = int(x.uop.shard_shape[0]) if isinstance(x.device, tuple) and x.uop.axis == 0 and x.ndim == 2 else None
  if not amd_custom_kernels_supported(x.device) or R is None or R % 64 or N % 64 or K % 32 or \
     isinstance(wt.device, tuple) and wt.uop.axis is not None: return None
  out = Tensor.empty(int(x.shape[0]), N, dtype=dtypes.float32, device=x.device, axis=0)
  fxn = functools.partial(_q8_gemm_kernel, token_tile=64, ggml_type=F32T, block_n=64, rdna4=_wmma_rdna4(x.device))
  return Tensor.custom_kernel(out, wt, x.half().contiguous(), fxn=fxn)[0]

def f16t_gemm(x:Tensor, wt:Tensor) -> Tensor|None:
  """
  x (R, K) @ wt (K, N) (a transposed float weight) on the tensor cores, the rows whole on every device (f32t_gemm's: split over them): the rows
  of batched decoding (a few hundred), padded to tiles of 32. (R, N) float. None: not those shapes
  """
  if x.ndim != 2 or not isinstance(R:=x.shape[0], int) or not 32 <= R <= 1024 or not amd_custom_kernels_supported(x.device): return None
  K, N = int(wt.shape[0]), int(wt.shape[1])
  if (isinstance(x.device, tuple) and x.uop.axis is not None) or (isinstance(wt.device, tuple) and wt.uop.axis is not None) or N % 64 or K % 32:
    return None
  # (TILE16: rows in fewer tiles of 16 than their padding to 32 (48 of 64): tiles of 16, more workgroups, the same sums)
  t16 = TILE16 and not _wmma_rdna4(x.device) and round_up(R, 16) < round_up(R, 32)
  P = round_up(R, 16 if t16 else 32)
  xp = x.half()
  if P != R: xp = xp.pad(((0, P - R), None))
  out = Tensor.empty(P, N, dtype=dtypes.float32, device=x.device)
  fxn = functools.partial(_q8_gemm_kernel, token_tile=16 if t16 else 32, ggml_type=F32T, block_n=64, rdna4=_wmma_rdna4(x.device))
  return Tensor.custom_kernel(out, wt, xp.contiguous(), fxn=fxn)[0][:R]

def merged_linear(layers:list[Linear]) -> Linear|None:
  """
  one Linear of the layers' rows (Q8_0, the same input, split on their outputs over the devices) if their weights are views of adjacent bytes
  of every device's memory (the loader's): a view of them (no copy). None if not
  """
  from tinygrad.device import MultiBuffer
  for l in layers:
    if l.ggml_type is None and l.use_custom_quant and amd_custom_kernels_supported(l.weight.device): l.set_quantized(l.weight)
  l0 = layers[0]
  if l0.ggml_type != Q8_0 or any(l.ggml_type != Q8_0 or l.shard_axis != 0 or l.in_features != l0.in_features or l.out_features != l0.out_features
                                 or l.bias is not None for l in layers): return None
  # (the buffer every weight is a view of)
  bufs = [[u.buffer for u in l.weight.uop.toposort() if u.op is Ops.BUFFER and isinstance(u.buffer, MultiBuffer)] for l in layers]
  if any(len(b) != 1 for b in bufs): return None
  mbs = [b[0] for b in bufs]
  parts = []
  for d in range(len(mbs[0].bufs)):
    bs = [mb.bufs[d] for mb in mbs]
    if any(x.base is not bs[0].base or x.nbytes != bs[0].nbytes or x.offset != bs[0].offset + j*bs[0].nbytes for j, x in enumerate(bs)): return None
    parts.append(bs[0].base.view(len(bs)*bs[0].nbytes, bs[0].offset))
  (mb:=MultiBuffer.__new__(MultiBuffer)).bufs = parts
  devs = cast(tuple[str, ...], l0.weight.device)
  with Linear.empty_init(): merged = Linear(l0.in_features, l0.out_features*len(layers), bias=False)
  merged.weight = Tensor(UOp.from_buffer(mb, dtypes.uint8, devs).reshape(1, -1).unshard(0)).flatten().bitcast(l0.weight.dtype).contiguous()
  merged.ggml_type, merged.shard_axis = Q8_0, 0
  return merged

def q8_linear_parts(merged:Linear, x:Tensor, n:int) -> list[Tensor]|None:
  """
  the n projections of x of a merged Linear (merged_linear's) by one kernel (the skinny gemm, its outputs in n tensors: no split sums):
  batched decoding's rows (17-96). None: not those rows
  """
  tokens = int(x.numel()) // merged.in_features
  devs, K = x.device, merged.in_features
  if not isinstance(devs, tuple) or not 16 < tokens <= 96 or K % 32 or not amd_custom_kernels_supported(devs): return None
  ne = merged.out_features // n
  P = round_up(tokens, 32)
  # (MERGED_TT: the token tile, default the rows' (a tile of 32 more workgroups, the weights dequantized once a tile))
  tt = next(t for t in (getenv("MERGED_TT", 96), 96, 64, 32) if P % t == 0)
  outs = [Tensor.empty(tokens, ne, dtype=dtypes.float32, device=devs, axis=1) for _ in range(n)]
  fxn = functools.partial(_q8_gemm_kernel, token_tile=tt, ggml_type=Q8_0, block_n=64, k_splits=1, rdna4=_wmma_rdna4(devs), n_out=n,
                          out_features=merged.out_features // len(devs), in_features=K)
  res = Tensor.custom_kernel(*outs, merged.weight, x.reshape(tokens, K).cast(dtypes.float16).contiguous(), fxn=fxn)[:n]
  return [r.reshape(*x.shape[:-1], ne) for r in res]

def q8_linear(layer:Linear, x:Tensor, reduce:bool=True, xq:tuple[Tensor, ...]|None=None, split:bool=False, half_out:bool=False,
              rows:int=0, m_rows:int=0) -> Tensor:
  """
  rows (split): every device's real rows, its first ones (batched decoding's, the others padding): the others aren't read. m_rows (split,
  TILE16): the outputs of the tiles of every device's first m_rows rows alone (the others' not computed: garbage), in tiles of 16 if they fit
  """
  assert layer.ggml_type in QUANT_SIZES
  tokens = int(x.numel()) // layer.in_features
  out_features, in_features = layer.out_features, int(x.uop.shard_shape[-1])
  splits = layer.in_features // in_features
  rdna4 = _wmma_rdna4(x.device)
  # (RDNA3: more rows than share a decode kernel's weight loads (batched decoding's): Q8_0 in the gemm of tiles of 32 rows (skinny), the others
  # padded to the tiles of 16 of the tensor cores, the extra rows sliced off)
  # (Q6_K too, whole blocks of 256 a split of the input: its gemm's few output blocks of a device's few outputs are few workgroups. not the
  # output head's many)
  dev_out = out_features // len(d) if isinstance(d:=layer.weight.device, tuple) and layer.shard_axis == 0 else out_features
  skinny = (layer.ggml_type == Q8_0 or (layer.ggml_type == Q6_K and dev_out <= 16384)) and xq is None and not split and \
    16 < tokens <= SKINNY_ROWS and layer.in_features % (256 if layer.ggml_type == Q6_K else 32) == 0
  # (RDNA3: Q6_K (the output head) of many rows in the gemm, its weights dequantized once a tile of 32 or 64 rows)
  q6_gemm = not skinny and layer.ggml_type == Q6_K and xq is None and not split and tokens >= 17 and layer.in_features % 256 == 0
  # (the output head of 65-96 rows (22-32 sequences of 3 drafts): a tile of 96 rows, the gemm reads the last row again past them, not 128)
  q6_rows = q6_gemm and 64 < tokens <= 96
  pad = round_up(tokens, 64 if tokens > 32 else 32) if q6_gemm and tokens % (64 if tokens > 32 else 32) and not q6_rows else \
        round_up(tokens, 64) if tokens > 64 and tokens % 64 and not skinny and not q6_rows else \
        round_up(tokens, 16) if 8 < tokens < 64 and tokens % 16 and layer.ggml_type in (Q8_0, Q6_K) and not skinny else 0
  # (skinny: the gemm's tiles of 32 rows past the rows read the last row again (no padded copy of x), the outputs past them aren't summed)
  P = round_up(tokens, 32) if skinny else 96 if q6_rows else tokens
  if not pad and not skinny and not rdna4 and tokens > 16 and tokens % 16 and xq is None and not split: pad = round_up(tokens, 16)
  if pad and xq is None and not split:
    out = q8_linear(layer, x.reshape(tokens, x.shape[-1]).pad(((0, pad - tokens), (0, 0))), reduce, None, split, half_out)
    out = out[:, :tokens] if splits > 1 and not reduce else out[:tokens]
    return out.reshape(*((splits,) if splits > 1 and not reduce else ()), *x.shape[:-1], out_features)
  assert (layer.shard_axis == 1) == (splits > 1), f"input features split over {splits} devices, weight split on axis {layer.shard_axis}"
  out_shape:tuple[int, ...] = (splits*tokens, out_features)
  fxn:Callable[..., UOp]
  ksplits = 1  # the input groups split over workgroups, their partial outputs summed after
  extra = (_iq_grid(x.device, layer.ggml_type),) if layer.ggml_type in (IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S) else ()
  # (split: every device its rows, the kernels' tiles of them)
  local_rows = tokens // len(x.device) if split and isinstance(x.device, tuple) else tokens
  # the kernels write the output features of their own device
  local_out = out_features // len(dev) if isinstance(dev:=layer.weight.device, tuple) and layer.shard_axis == 0 else out_features
  # (a few outputs of a long input (a hyper-connection's down projection) of a few rows a device (a small prefill's): the gemm with the input
  # split over workgroups, the tensor cores' tiles of them are a few workgroups that loop over the whole input)
  few_out_long = layer.ggml_type == Q8_0 and layer.shard_axis is None and not half_out and local_rows % 32 == 0 and \
    local_rows <= 1024 and local_out <= 512 and in_features >= 4096
  if skinny:
    # (the inputs split over workgroups: a few row tiles and output blocks are few workgroups for the GPU, their partial outputs summed after)
    groups, blocks = in_features // 32, (P // 32) * round_up(local_out, 64) // 64
    ks = [s for s in (1, 2, 4, 8, 16) if groups % s == 0 and (s == 1 or groups // s >= 8) and
          (layer.ggml_type != Q6_K or (groups // s) % 8 == 0)]
    # (SKINNY_WG: the workgroups the splits aim for)
    ksplits = next((s for s in ks if blocks * s >= SKINNY_WG), ks[-1])
    # (a tile of up to 96 rows: the weights dequantized in LDS once for them, 3 tiles of the 192 rows of 48 sequences of 3 drafts: -2% a step)
    tt = next(t for t in (96, 64, 32) if P % t == 0)
    # (TILE16: rows in fewer whole tiles of 16 than of the padding to 32 (48 of 64: 16 sequences of 3 drafts): their tiles of 16 alone, the
    # same splits (the same sums))
    mr = tokens if TILE16 and not rdna4 and layer.ggml_type == Q8_0 and round_up(tokens, 16) < P else 0
    if mr: tt = 16
    fxn = functools.partial(_q8_gemm_kernel, token_tile=tt, ggml_type=layer.ggml_type, block_n=64, k_splits=ksplits, rdna4=rdna4, m_rows=mr)
    srcs = (x.cast(dtypes.float16).contiguous(),)
    out_shape = (splits*ksplits*P, out_features)
  elif (layer.ggml_type == Q8_0 or q6_gemm) and xq is None and (local_out % 128 == 0 or tokens >= 1024 or q6_gemm or few_out_long) and \
       in_features % 32 == 0 and (rdna4 or local_rows >= 256 or few_out_long or q6_gemm) and \
       (tt:=next((t for t in ((192, 128, 96, 64, 32) if q6_gemm else (96, 128, 64, 32) if rdna4 else
                              (128 if local_out % 128 == 0 and local_out >= 512 else 64, 64, 32)) if (P if q6_rows else local_rows) % t == 0), 0)):
    # many tokens: the weights dequantized in LDS for token tiles of tt
    if layer.ggml_type == Q8_0 and layer.shard_axis is None and not half_out and local_rows <= 2048 and local_out <= 512 and in_features >= 4096:
      # (a few outputs of a long input: its parts on workgroups of their own, summed. 768 rows of 320 outputs are 60 tiles: 152 -> 99 us)
      # (HC_KS: the splits of a few rows a device (batched decoding's hyper-connection down projection))
      ksplits = next(s for s in (HC_KS if local_rows <= 64 else 16 if local_rows <= 512 else 8, 8, 4, 2, 1) if in_features % (32*s) == 0)
      out_shape = (ksplits*tokens, out_features)
    mr = m_rows if TILE16 and split and not rdna4 and 0 < m_rows < local_rows else 0
    if mr and mr <= 16: tt = 16
    fxn = functools.partial(_q8_gemm_kernel, token_tile=tt, ggml_type=layer.ggml_type, rdna4=rdna4,
                            block_n=128 if rdna4 or (local_out % 128 == 0 and ksplits == 1 and tt == 128) else 64, k_splits=ksplits,
                            rows=rows if split and 0 < rows < local_rows else 0, m_rows=mr)
    if q6_rows: out_shape = (splits*P, out_features)
    srcs = (x.cast(dtypes.float16).contiguous(),)
  elif tokens % 16 == 0 and local_out % 16 == 0 and xq is None and not (tokens <= DECODE_MULTI and in_features >= 1024):
    # (a few rows of a long input (batched decoding's) in the decode kernel, its weight loads shared by the rows: the f16 tensor cores' tiles of
    # them are few workgroups that loop over the input, 2-4x slower. a short input's many outputs are the tensor cores')
    if layer.ggml_type == IQ4_XS: fxn, extra = _iq4_linear_f16_wmma_kernel, (iq4_half_lut(x.device),)
    else: fxn = functools.partial(_q5_linear_f16_wmma_kernel if layer.ggml_type in (Q4_K, Q5_K) else _quant_linear_f16_wmma_kernel,
                                  ggml_type=layer.ggml_type)
    fxn = functools.partial(fxn, rdna4=rdna4)
    srcs = (x.cast(dtypes.float16).contiguous(), *extra)
  else:
    fxn = functools.partial(_quant_decode_kernel, ggml_type=layer.ggml_type)
    srcs = (*(xq or q8_quantize(x, tokens, in_features))[:3], *extra)
  # split: x has its rows split over the devices (the weight is whole on every device), so has the output
  out_axis = 0 if split and layer.shard_axis is None else None if layer.shard_axis is None else 1-layer.shard_axis
  # half_out: q8_gemm stores the sums in half (the others in float, then cast)
  half_out = half_out and getattr(fxn, "func", None) is _q8_gemm_kernel
  out = Tensor.empty(out_shape, dtype=dtypes.half if half_out else dtypes.float32, device=x.device, axis=out_axis)
  result = Tensor.custom_kernel(out, layer.weight, *srcs, fxn=functools.partial(fxn, out_features=local_out, in_features=in_features))[0]
  if ksplits > 1:
    # (split: every device's rows of the partial outputs of the input splits)
    if split: result = result.reshape(n:=len(cast(tuple, x.device)), ksplits, tokens // n, out_features).sum(1).reshape(tokens, out_features)
    elif splits > 1: result = result.reshape(splits, ksplits, P, out_features)[:, :, :tokens].sum(1).reshape(splits*tokens, out_features)
    else: result = result.reshape(ksplits, P, out_features)[:, :tokens].sum(0)
  elif P != tokens: result = result.reshape(splits, P, out_features)[:, :tokens].reshape(splits*tokens, out_features)
  # row parallel: every device wrote the partial sum of its input features, the sum over the devices is the allreduce
  if splits > 1:
    if not reduce: return result.reshape(splits, *x.shape[:-1], out_features)  # the partial sums, sharded on the device axis
    result = partial_sum(result.reshape(splits, tokens, out_features))
  result = result.reshape(*x.shape[:-1], out_features)
  return result if layer.bias is None else result + layer.bias

def _sys(fmt:str, dtype:DType, *src:UOp) -> UOp: return _cfn(False, src=src, arg=(fmt, dtype))


# ******** allreduce peer to peer: the values in half into the inboxes of the peers, then a flag a peer ********

# every device has an inbox in its memory (uncached, mapped into every other device): two parities of a slot per sender for the values (two
# halves in a word) and for a flag. an exchange is two kernels. the first stores a device's values into its slot of every peer's inbox through the
# peers' mappings cached read write shared (gfx11 mtype 4: the writes combined in the l2, ~25 GB/s to all the peers at once. uncached writes
# to more than one root complex at once crawl at ~1 GB/s: 7900 XTXs behind EPYC root ports). the l2 is written back before the second starts
# (AMDComputeQueue._exec): its workgroup 0 stores a flag (the epoch) into every peer's inbox, behind the values on their path. every workgroup
# polls this device's inbox for the peers' flags, then sums in device order (the same bits everywhere). a device is at most one call ahead
# of the others: two parities. Each pool has a fixed workgroup count so all its epoch counters move together across varying payload sizes.

FLAG_AR_MAXN = 1 << 20  # words (two half values or one float32 each)

# the collectives' mailboxes and pools (name, object), read by comm_state_report
_COMM_STATE:list[tuple[str, int, Any]] = []
def _peek(b:Any, off:int, n:int) -> bytes:
  """n bytes at off of a buffer in an AM device's memory, read through its BAR (not by the device: a hung one too)"""
  adev, out = cast(Any, Device[b.device]).iface.dev_impl, bytearray()
  for pa, sz in b.get_storage(b.device).meta.mapping.paddrs:
    if off >= sz:
      off -= sz
      continue
    out += bytes(adev.vram.view(pa + off, min(sz - off, n - len(out)))[:])
    off = 0
    if len(out) >= n: break
  return bytes(out)
def _overlaps(b:Any, offs:list[int], n:int) -> str:
  """the physical places of a buffer's ranges [off, off+n) and the other live buffers of its device whose memory has them (none: it's its own)"""
  import gc
  from tinygrad.device import Buffer
  def phys(m:Any, off:int) -> int:
    for pa, sz in m.paddrs:
      if off < sz: return pa + off
      off -= sz
    return -1
  m = b.get_storage(b.device).meta.mapping
  rng = [(phys(m, o), phys(m, o) + n) for o in offs]
  hits = []
  for o in gc.get_objects():
    if not isinstance(o, Buffer) or o is b or o.device != b.device or o._base is not None or not o.is_allocated(): continue
    om = getattr(getattr(o._storage, "meta", None), "mapping", None)
    if om is None or not hasattr(om, "paddrs") or om.aspace is not m.aspace: continue
    for pa, sz in om.paddrs:
      if any(pa < hi and lo < pa + sz for lo, hi in rng): hits.append(f"{o.nbytes}B {o.options} at {pa:#x}+{sz:#x}")
  return f"phys {[hex(lo) for lo, _ in rng]} other buffers there: {hits[:6]}"
def comm_state_report() -> str:
  """
  where every device's collectives are (a hang): a ring lane's epoch counters (the exchanges of a chunk done, min..max of its chunks), the
  flags of the next exchange's parity in its inbox for the first chunk; a flag pool's counters (a workgroup's) and its flags (a sender's)
  """
  import struct
  out = []
  for name, lane, st in _COMM_STATE:
    with contextlib.suppress(Exception):
      N = len(st.devs)
      if isinstance(st, RingMailbox):
        CH = _ring_chunks(8 if lane < DEC_LANE else 32)
        cnt = [struct.unpack(f"<{RING_CMAX+1}I", _peek(b, 0, (RING_CMAX+1)*4)) for b in st.counter_bufs]
        line = f"{name}: chunks' epochs " + " ".join(f"d{c[RING_CMAX]}:{min(c[:CH])}..{max(c[:CH])}" for c in cnt)
        # (the chunks a device is behind in; the first two the devices differ in (else the first one): every device's flags of its next epoch's
        # parity, a slot's rounds run length encoded (epoch x rounds))
        top = [max(c[i] for c in cnt) for i in range(CH)]
        line += " | behind: " + " ".join(f"d{d}:{len(b)}{b[:6]}" for d, c in enumerate(cnt) if (b:=[i for i in range(CH) if c[i] < top[i]]))
        if (diff:=[i for i in range(CH) if len({c[i] for c in cnt}) > 1]) or any(min(c[:CH]) != max(c[:CH]) for c in cnt):
          for ch in (diff[:2] or [0]):
            line += f"\n  chunk {ch}: " + " ".join(f"d{d}:{cnt[d][ch]}" for d in range(N))
            for d, fb in enumerate(st.flag_bufs):
              par, rle = (cnt[d][ch] + 1) & 1, []
              for sl in range(N):
                R = RING_CMAX // CH
                eps = [w >> 32 for w in struct.unpack(f"<{R}Q", _peek(fb, ((par*N + sl)*RING_CMAX + ch*R)*8, R*8))]
                rle.append(" ".join(f"{e}x{len(list(g))}" for e, g in itertools.groupby(eps)))
              line += f"\n    d{d} flags(par {par}) " + " | ".join(f"s{sl}: {r}" for sl, r in enumerate(rle))
              # (a device behind: its inbox's words of the chunk against the ones the device before it has of the same part (what it forwarded,
              # an all gather's), a slot's differing words: count, first ones (word in the chunk's region))
              if cnt[d][ch] < top[ch]:
                W, cmp = RING_MAXW // CH, []
                for sl in range(N):
                  if sl in (d, (d-1) % N): continue
                  at = ((par*N + sl)*RING_MAXW + ch*W)*4
                  mine, prev = (struct.unpack(f"<{W}I", _peek(b, at, W*4)) for b in (st.inbox_bufs[d], st.inbox_bufs[(d-1) % N]))
                  bad = [i for i in range(W) if mine[i] != prev[i]]
                  cmp.append(f"s{sl}: {len(bad)} {[(i, hex(mine[i]), hex(prev[i])) for i in bad[:4]]}")
                line += f"\n    d{d} inbox vs d{(d-1) % N}: " + " | ".join(cmp)
                line += f"\n    d{d} {_overlaps(st.inbox_bufs[d], [((par*N + sl)*RING_MAXW + ch*W)*4 for sl in range(N)], W*4)}"
        out.append(line)
      elif isinstance(st, FlagAllreduce):
        cnt = [struct.unpack(f"<{st.groups+1}I", _peek(b, 0, (st.groups+1)*4)) for b in st.counter_bufs]
        fls = [struct.unpack(f"<{2*N}I", _peek(b, 0, 2*N*4)) for b in st.flag_bufs]
        if any(min(c[:-1]) != max(c[:-1]) for c in cnt) or len({c[:-1] for c in cnt}) > 1 or lane == 0:
          out.append(f"{name}: epochs " + " ".join(f"d{c[-1]}:{min(c[:-1])}..{max(c[:-1])}" for c in cnt) + " flags " +
                     " ".join(f"d{d}:{list(f)}" for d, f in enumerate(fls)))
  return "\n".join(out)

def flag_ar_groups(n:int) -> int: return 4 if n <= 8192 else 16 if n <= 65536 else 64

def _init_buffers(pairs:list[tuple[Any, bytes]]):
  """the buffers allocated, their bytes copied in at once: one launch of copies from host memory every device maps (not a copy each)"""
  from tinygrad.device import Buffer, BufferSpec, BufferStorage
  from tinygrad.engine.realize import run_linear
  for b, _ in pairs: b.ensure_allocated()
  if not pairs: return
  offs = list(itertools.accumulate([round_up(len(v), 256) for _, v in pairs], initial=0))
  host = Buffer(pairs[0][0].device, offs[-1], options=BufferSpec(host=True, cpu_access=True)).ensure_allocated()
  mv = host.as_memoryview(allow_zero_copy=True)
  for (_, v), o in zip(pairs, offs): mv[o:o+len(v)] = v
  def alias(d:str) -> Any:
    if d == host.device: return host
    return Buffer(d, host.nbytes, opaque=BufferStorage(va:=host.get_buf(d)), options=BufferSpec(external_ptr=va)).ensure_allocated()
  aliases = {d: alias(d) for d in dedup(b.device for b, _ in pairs)}
  calls = [UOp.from_buffer(b, dtypes.uint8).store_call(UOp.from_buffer(aliases[b.device].view(len(v), o).ensure_allocated(), dtypes.uint8))
           for (b, v), o in zip(pairs, offs)]
  run_linear(UOp(Ops.LINEAR, src=tuple(calls)), update_stats=False)
  for d in aliases: Device[d].synchronize()

class FlagAllreduce:
  def __init__(self, devs:tuple[str, ...], groups:int):
    from tinygrad.device import Buffer, BufferSpec, BufferStorage, MultiBuffer
    N = len(devs)
    def mapped(nbytes:int, slot:int=0, peer_mtype:int|None=None) -> list[Tensor]:
      # (zeroed: a flag the last process left could have an epoch this one waits for)
      own = [Buffer(d, nbytes, options=BufferSpec(uncached=True)).ensure_allocated() for d in devs]
      inits.extend((b, bytes(nbytes)) for b in own)
      for b in own: setattr(b, "peer_mtype", peer_mtype)
      def alias(b:Buffer, d:int) -> Buffer:
        # (slot: the alias from the slot of device d on, its bytes)
        va = b.get_buf(devs[d]) + d*slot
        return Buffer(devs[d], slot or nbytes, opaque=BufferStorage(va), options=BufferSpec(external_ptr=va)).ensure_allocated()
      ret = []
      for k in range(N):  # inbox k of device d is the one of device (d+k)%N (0: its own)
        (mb:=MultiBuffer.__new__(MultiBuffer)).bufs = [own[d] if k == 0 else alias(own[(d+k) % N], d) for d in range(N)]
        ret.append(Tensor(UOp.from_buffer(mb, dtypes.uint32, devs)))
      return ret
    inits:list[tuple[Buffer, bytes]] = []
    with Context(ALLOW_DEVICE_USAGE=1):
      # (the values: the peers' aliases at the slot of the device (its two parities), what the send kernel writes back from the l2)
      self.vals = mapped(2*N*FLAG_AR_MAXN*4, 2*FLAG_AR_MAXN*4, 4)
      self.flags = mapped(2*N*4)
      self.flag_bufs = [b for b, _ in inits[-N:]]
      # (an epoch counter a workgroup, then the device's index)
      counters = [Buffer(d, (groups+1)*4) for d in devs]
      self.devs, self.groups, self.counter_bufs = devs, groups, counters
      _init_buffers(inits + [(b, bytes(groups*4) + i.to_bytes(4, "little")) for i, b in enumerate(counters)])
      (cb:=MultiBuffer.__new__(MultiBuffer)).bufs = counters
      self.counter = Tensor(UOp.from_buffer(cb, dtypes.uint32, devs))
      for d in devs: Device[d].synchronize()

# (the mailboxes and counters collectives use: lanes of their own (micro batches' collectives are independent, they'd write a lane's state at
# once), see micro_batch_blocks)
FLAG_LANE = [0]
def flag_allreduce(devs:tuple[str, ...], groups:int) -> FlagAllreduce|None:
  return _flag_allreduce(tuple(Device.canonicalize(d) for d in devs), groups, FLAG_LANE[0])
@functools.cache
def _flag_allreduce(devs:tuple[str, ...], groups:int, lane:int=0) -> FlagAllreduce|None:
  if len(devs) < 2 or not amd_custom_kernels_supported(devs): return None
  if any(_gfx(d) not in (11, 12) for d in devs): return None
  _COMM_STATE.append((f"flag pool lane {lane} groups {groups}", lane, ret:=FlagAllreduce(devs, groups)))
  return ret

def flag_ar_shape(n:int, groups:int) -> tuple[int, int]|None:
  """(threads, words a thread) over a fixed number of workgroups (words past n masked)."""
  if n <= 0 or n > FLAG_AR_MAXN or groups <= 0: return None
  T = 256 if n >= groups*256 else 128 if n >= groups*128 else 64
  return T, (n + groups*T - 1) // (groups*T)

_LOAD_SYS32 = "__hip_atomic_load({0}, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)"

def _flag_ar_common(N:int, n:int, T:int, EP:int, counter:UOp):
  groups = counter.shape[0] - 1
  wg, t = UOp.range(groups, 0, AxisType.GLOBAL), UOp.range(T, 1, AxisType.LOCAL)
  # (a thread's words T apart: every store and load of a wave on consecutive addresses)
  els = [(wg*EP + i)*T + t for i in range(EP)]
  epoch, me = (counter[wg].load() + 1).cast(dtypes.uint32), counter[groups].load().cast(dtypes.weakint)
  par = (epoch & 1).cast(dtypes.weakint)
  # A sender's two parities are interleaved in blocks of 64 words.
  def pos(e:UOp) -> UOp: return (e // 64)*128 + par*64 + e % 64
  def spos(e:UOp) -> UOp: return pos(e).valid(e < n)
  def vslot(s:UOp|int, e:UOp) -> UOp: return (pos(e) + s*2*FLAG_AR_MAXN).valid(e < n)
  def fslot(s:UOp|int) -> UOp: return par*N + s
  return wg, t, els, epoch, me, spos, vslot, fslot

def _h2(x:UOp, e:UOp, n:int, clip:bool=True) -> UOp:
  # Reduce partials saturate to fp16; pure allgather preserves the original half bits, including infinities.
  def h(v:UOp) -> UOp:
    return (v.float().maximum(-65504.0).minimum(65504.0) if clip else v).cast(dtypes.half).bitcast(dtypes.uint16).cast(dtypes.uint32)
  return h(x[0, (2*e).valid(e < n)].load()) | (h(x[0, (2*e+1).valid(e < n)].load()) << 16)

@functools.cache
def _flag_send_kernel(*args:UOp, N:int, n:int, T:int, EP:int, scatter:int=0, raw:bool=False, packed:bool=False) -> UOp:
  # a device's values into its slot of every peer's inbox (args: the inboxes 1..N-1, x, the counters)
  vals_mb, x, counter = args[:N-1], args[N-1], args[N]
  wg, t, els, epoch, me, spos, vslot, fslot = _flag_ar_common(N, n, T, EP, counter)
  def value(k:int, e:UOp) -> UOp:
    if raw: return x[0, e.valid(e < n)].load().bitcast(dtypes.uint32)
    return _h2(x, ((me+k) % N)*n + e if scatter else e, scatter or n, clip=not packed)
  sent = [vals_mb[k-1][spos(e)].store(value(k, e)) for k in range(1, N) for e in els]
  return UOp.group(*sent).end(wg, t).sink(arg=KernelInfo(name=f"flag_ar_send_{N-1}_{n*4}", opts_to_apply=()))

def _flag_poll(wg:UOp, t:UOp, epoch:UOp, me:UOp, flags_mb:tuple[UOp, ...], N:int) -> UOp:
  def fslot(s:UOp) -> UOp: return (epoch & 1).cast(dtypes.weakint)*N + s
  flagged = [flags_mb[k][fslot(me).valid(t.eq(k) & wg.eq(0))].store(epoch) for k in range(1, N)]
  # thread s of a workgroup polls the flag of device s in this device's inbox until it has the epoch
  s_t = t.cast(dtypes.weakint)
  loop = UOp.range(UOp(Ops.NOOP).after(*flagged), next(UOp.unique_num), dtype=dtypes.void)
  word = UOp.alloc((1,), dtypes.uint32, addrspace=AddrSpace.REG)
  loaded = word.after(loop).store(_sys(_LOAD_SYS32, dtypes.uint32, flags_mb[0].after(loop)[fslot(s_t).valid(t < N)]))
  got = word.after(loaded)[0].load()
  stale = (t < N) & s_t.ne(me) & got.ne(epoch)
  # Never consume stale payloads on timeout. A missing peer is reported by the host queue's wait timeout.
  word = word.after(_cfn(False, src=(stale,), arg=("", dtypes.void)).backedge(loop, stale))
  return UOp.group(word[0].load()).barrier()

@functools.cache
def _flag_sum_kernel(out:UOp, x:UOp, counter:UOp, *mb:UOp, N:int, n:int, T:int, EP:int, scatter:int=0, gather:int=0, gather_half:bool=False,
                     broadcast:bool=False, half:bool=False) -> UOp:
  # half: the broadcast reduced chunk in fp16 pairs (saturating, as the partials are), gathered so (half the bytes of float32)
  vals_mb, flags_mb = mb[:N], mb[N:]
  wg, t, els, epoch, me, spos, vslot, fslot = _flag_ar_common(N, n, T, EP, counter)
  def unh(b:UOp) -> UOp: return b.cast(dtypes.uint16).bitcast(dtypes.half).float()
  arrived = _flag_poll(wg, t, epoch, me, flags_mb, N)
  # this device's own values from x, the others' from its inbox (uncached: the memory)
  stores = []
  for e in els:
    v = x[0, e.valid(e < n)].load().bitcast(dtypes.uint32) if gather and not gather_half else \
      _h2(x, me*n + e if scatter else e, scatter or n, clip=not gather_half or half)
    ps = [me.eq(s).where(v, vals_mb[0].after(arrived)[vslot(s, e)].load()) for s in range(N)]
    if gather:
      if gather_half:
        for s, p in enumerate(ps):
          for j in (0, 1):
            value = (p >> (16*j)).cast(dtypes.uint16).bitcast(dtypes.half).cast(out.dtype)
            stores.append(out[(2*(s*n+e)+j).valid((e < n) & (2*(s*n+e)+j < gather))].store(value))
      else:
        stores += [out[(s*n+e).valid((e < n) & (s*n+e < gather))].store(p.bitcast(out.dtype) if out.dtype.itemsize == 4 else
                                                                        p.bitcast(dtypes.float32).cast(out.dtype)) for s, p in enumerate(ps)]
    else:
      values = [functools.reduce(lambda a,b: a+b, [unh(p >> (16*j)) for p in ps]) for j in (0, 1)]
      stores += [out[(2*e+j).valid(e < n)].store(v) for j, v in enumerate(values)]
      if broadcast:
        # Send the reduced float32 chunk directly into the next epoch's parity. The gather's execution barrier/L2 writeback precedes its flags.
        def h(v:UOp) -> UOp: return v.maximum(-65504.0).minimum(65504.0).cast(dtypes.half).bitcast(dtypes.uint16).cast(dtypes.uint32)
        for k in range(1, N):
          if half:
            at = (e//64)*128 + ((epoch+1)&1).cast(dtypes.weakint)*64 + e%64
            stores.append(vals_mb[k][at.valid(e < n)].store(h(values[0]) | (h(values[1]) << 16)))
            continue
          for j, v in enumerate(values):
            raw_e = 2*e+j
            at = (raw_e//64)*128 + ((epoch+1)&1).cast(dtypes.weakint)*64 + raw_e%64
            stores.append(vals_mb[k][at.valid(e < n)].store(v.bitcast(dtypes.uint32)))
  done = counter.after(UOp.group(*stores).barrier())[wg.valid(t.eq(0))].store(epoch)
  return UOp.group(*stores, done).end(wg, t).sink(arg=KernelInfo(name="flag_allreduce_push", opts_to_apply=()))

def flag_partial_sum(x:Tensor) -> Tensor|None:
  """the sum over the leading device axis of x (devices, ...) on every device, peer to peer (FlagAllreduce). None: not supported"""
  if not isinstance(x.device, tuple) or x.uop.axis != 0 or x.shape[0] != len(x.device) or x.dtype not in (dtypes.half, dtypes.float32): return None
  if not isinstance(n:=prod(x.shape[1:]), int) or n % 2 or n == 0: return None
  # (many rows (batched decoding): the bytes are the time. the partials sent as int8 groups of 32 with their scale, the sums gathered so (the
  # values the quantized layers' kernels would make of them, a prefill's exchanges are of them too))
  if n >= 65536 and (r:=_flag_partial_sum_q8(x, n)) is not None: return r
  N, groups = len(x.device), flag_ar_groups(n//2)
  # (a few rows (a sequence's drafts): one exchange, the sums of the whole of every device's partials (the latency of the links, not the bytes).
  # RDNA4 (R9700): every device's stores to all the others at once are slow (a 2560*3 one: 57 us), the two exchanges of a reduce scatter from
  # 2048 values (21 us))
  scatter = n >= (2048 if _wmma_rdna4(x.device) else 16384)
  chunk = round_up((n//2 + N-1)//N, 64)
  if (shape:=flag_ar_shape(2*chunk if scatter else n//2, groups)) is None or (ar:=flag_allreduce(x.device, groups)) is None: return None
  out = Tensor.empty(n, dtype=dtypes.float32, device=x.device)
  xf = x.reshape(x.shape[0], n).contiguous()
  if scatter:
    # The first exchange sends only each destination's chunk; the second broadcasts reduced chunks in float32 (no second rounding).
    t1, ep1 = unwrap(flag_ar_shape(chunk, groups))
    kw1 = dict(N=N, n=chunk, T=t1, EP=ep1, scatter=n//2)
    sent = Tensor.custom_kernel(*ar.vals[1:], xf, ar.counter, fxn=functools.partial(_flag_send_kernel, **kw1))[:N-1]
    # (the reduced chunks broadcast in fp16, saturating as the partials are: half the bytes of the second exchange)
    half = True
    reduced = Tensor.custom_kernel(Tensor.empty(2*chunk, dtype=dtypes.float32, device=x.device), xf, ar.counter, ar.vals[0], *sent, *ar.flags,
                                   fxn=functools.partial(_flag_sum_kernel, broadcast=True, half=half, **kw1))
    xf, counter = reduced[0].reshape(1, 2*chunk), reduced[2]
    t2, ep2 = unwrap(flag_ar_shape(chunk if half else 2*chunk, groups))
    kw = dict(N=N, n=chunk if half else 2*chunk, T=t2, EP=ep2)
    sent = reduced[4:4+N-1]
    return Tensor.custom_kernel(out, xf, counter, ar.vals[0], *sent, *ar.flags,
                                fxn=functools.partial(_flag_sum_kernel, gather=n, gather_half=half, half=half, **kw))[0].reshape(x.shape[1:])
  kw = dict(N=N, n=n//2, T=shape[0], EP=shape[1])
  sent = Tensor.custom_kernel(*ar.vals[1:], xf, ar.counter, fxn=functools.partial(_flag_send_kernel, **kw))[:N-1]
  return Tensor.custom_kernel(out, xf, ar.counter, ar.vals[0], *sent, *ar.flags,
                              fxn=functools.partial(_flag_sum_kernel, **kw))[0].reshape(x.shape[1:])

@functools.cache
def _flag_q8_send_kernel(*args:UOp, N:int, groups:int) -> UOp:
  # Quantize directly into the destination inbox: eight int8 words plus a float32 scale per 32 values.
  # All pool counters advance together; the send grid can read counter zero without matching the receive grid.
  mb, x, counter = args[:N-1], args[N-1], args[N]
  block, lane = UOp.range(N*groups//8, 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP)
  group, sub = block*8 + lane//4, lane % 4
  values = [x.reshape(-1)[group*32 + sub*8 + i].load().float() for i in range(8)]
  words, d = _q8_pack_values(values)
  epoch = (counter[0].load() + 1).cast(dtypes.uint32)
  me, par = counter[int(counter.shape[0])-1].load().cast(dtypes.weakint), (epoch & 1).cast(dtypes.weakint)
  def pos(e:UOp) -> UOp: return (e//64*2+par)*64 + e%64
  stores:list[UOp] = []
  for k, buf in enumerate(mb, 1):
    valid, base = (group//groups).eq((me+k)%N), (group%groups)*9
    stores.extend(buf[pos(base+sub*2+w).valid(valid)].store(words[w]) for w in range(2))
    stores.append(buf[pos(base+8).valid(valid & sub.eq(0))].store(d.bitcast(dtypes.uint32)))
  return UOp.group(*stores).end(block, lane).sink(arg=KernelInfo(name=f"flag_ar_send_{N-1}_{groups*9*4}", opts_to_apply=()))

@functools.cache
def _flag_q8_sum_kernel(out:UOp, x:UOp, counter:UOp, *mb:UOp, N:int, groups:int) -> UOp:
  # Sum remote Q8 partials in device order, then add the unquantized local contribution.
  wg, _, _, epoch, me, _, vslot, _ = _flag_ar_common(N, groups*9, 256, 0, counter)
  warp, lane = UOp.range(8, 1, AxisType.LOCAL), UOp.range(32, -1, AxisType.WARP)
  t = warp*32 + lane
  arrived = _flag_poll(wg, t, epoch, me, mb[N:], N)
  inbox, mine = mb[0].after(arrived), x.after(arrived).reshape(-1)
  EP, sub = round_up(groups, (int(counter.shape[0])-1)*64)//((int(counter.shape[0])-1)*64), t%4
  stores = []
  for j in range(EP):
    group = wg*EP*64 + t//4 + j*64
    acc = [UOp.const(0, dtypes.float32)]*8
    for s in range(N):
      valid = (group < groups) & me.ne(s)
      d = inbox[vslot(s, group*9+8).valid(valid)].load().bitcast(dtypes.float32)
      words = [inbox[vslot(s, group*9+sub*2+w).valid(valid)].load() for w in range(2)]
      vals = [((words[i//4] >> (8*(i%4))).cast(dtypes.uint8).bitcast(dtypes.int8).float()*d) for i in range(8)]
      acc = [a + me.ne(s).where(v, 0.0) for a, v in zip(acc, vals)]
    for i in range(8):
      at = group*32+sub*8+i
      val = acc[i] + mine[(me*groups*32+at).valid(group < groups)].load().float()
      stores.append(out.reshape(-1)[at.valid(group < groups)].store(val))
  done = counter.after(UOp.group(*stores))[wg.valid(t.eq(0))].store(epoch)
  return UOp.group(*stores, done).end(wg, warp, lane).sink(arg=KernelInfo(name="flag_allreduce_push_q8", opts_to_apply=()))

def _flag_partial_sum_q8(x:Tensor, n:int) -> Tensor|None:
  """flag_partial_sum in two exchanges of int8 groups: a reduce scatter of the partials, an all gather of the sums (the same bits everywhere)"""
  N = len(cast(tuple[str, ...], x.device))
  n2 = round_up(n, N*256)
  xf = x.reshape(N, n)
  if n2 != n: xf = xf.pad((None, (0, n2 - n)))
  if (rs:=flag_reduce_scatter(xf, 0)) is None: return None
  return all_gather_q8(rs.reshape(1, n2 // 256, 256), 1).reshape(n2)[:n].float().reshape(x.shape[1:])

def ring_min(devs:str|tuple[str, ...]) -> int:
  """
  the bytes of an exchange from which it goes around the ring (RING_MIN), not one exchange with every peer: 7900 XTX: 512 KB (a decoding step
  of 16/32/48 sequences: 31.3/37.3/46.7 -> 25.8/35.5/45.9 ms; a ring's ~70 us of latency is more than the links' time of its bytes), RDNA4
  (R9700, its one exchange slow): 64 KB
  """
  return getenv("RING_MIN", (1 << 16) if _wmma_rdna4(devs) else (1 << 19))

def flag_reduce_scatter(p:Tensor, axis:int) -> Tensor|None:
  """One P2P exchange: each rank receives its contiguous chunk, summed in float32 from Q8 (or unaligned fp16) partials."""
  if not isinstance(p.device, tuple) or p.uop.axis != 0 or p.shape[0] != len(p.device) or p.dtype not in (dtypes.half, dtypes.float32): return None
  N, S = len(p.device), p.shape[1:]
  if not all(isinstance(s, int) for s in S) or prod(S[:axis]) != 1 or S[axis] % N: return None
  size = int(prod(S))
  if size % (2*N): return None
  # (big ones around the ring: see RingMailbox)
  if size*9//8 >= ring_min(p.device) and (r:=ring_reduce_scatter_q8(p, size, tuple(int(s) for s in S), axis)) is not None: return r.reshape(S)
  if size and size % (N*256) == 0:
    qgroups, groups = size//(N*32), flag_ar_groups(size//2)
    if qgroups*9 > FLAG_AR_MAXN or (ar:=flag_allreduce(p.device, groups)) is None: return None
    xf = p.reshape(N, size).contiguous()
    sent = Tensor.custom_kernel(*ar.vals[1:], xf, ar.counter, fxn=functools.partial(_flag_q8_send_kernel, N=N, groups=qgroups))[:N-1]
    out = Tensor.empty(size, dtype=dtypes.float32, device=p.device, axis=0)
    return Tensor.custom_kernel(out, xf, ar.counter, ar.vals[0], *sent, *ar.flags,
                                fxn=functools.partial(_flag_q8_sum_kernel, N=N, groups=qgroups))[0].reshape(S)
  chunk, groups = size//(2*N), flag_ar_groups(size//2)
  if (shape:=flag_ar_shape(chunk, groups)) is None or (ar:=flag_allreduce(p.device, groups)) is None: return None
  kw = dict(N=N, n=chunk, T=shape[0], EP=shape[1], scatter=size//2)
  xf = p.reshape(N, size).contiguous()
  sent = Tensor.custom_kernel(*ar.vals[1:], xf, ar.counter, fxn=functools.partial(_flag_send_kernel, **kw))[:N-1]
  out = Tensor.empty(size, dtype=dtypes.float32, device=p.device, axis=0)
  return Tensor.custom_kernel(out, xf, ar.counter, ar.vals[0], *sent, *ar.flags,
                              fxn=functools.partial(_flag_sum_kernel, **kw))[0].reshape(S)

def flag_all_gather(x:Tensor, axis:int, dtype=None) -> Tensor|None:
  """One P2P exchange of contiguous shards, in float32 or fp16, without a reduction."""
  dtype = dtype or x.dtype
  # (around the ring: see RingMailbox. 2 byte values as pairs in words)
  if isinstance(x.device, tuple) and x.uop.axis == axis and prod(x.shape[:axis]) == 1 and \
     isinstance(size:=x.numel(), int) and (w:=dtype.itemsize) in (2, 4) and size % (len(x.device)*(4//w)) == 0 and \
     size*w >= ring_min(x.device) and ring_mailbox(x.device) is not None:
    wds = x.cast(dtype).reshape(len(x.device), size//len(x.device)).contiguous()
    if w == 2: wds = wds.bitcast(dtypes.uint32)
    if (r:=ring_all_gather_words(wds)) is not None: return r.bitcast(dtype).reshape(x.shape)
  # (words (int32, uint32: the q8 packs of all_gather_q8): as float32's, the words as they are, not bitcast copies before and after)
  if not isinstance(x.device, tuple) or x.uop.axis != axis or dtype not in (dtypes.half, dtypes.float32, dtypes.int32, dtypes.uint32): return None
  N, S = len(x.device), x.shape
  if not all(isinstance(s, int) for s in S) or prod(S[:axis]) != 1 or S[axis] % N: return None
  size, half = int(prod(S)), dtype == dtypes.half
  if size % (N*(2 if half else 1)): return None
  chunk = size//N
  words, groups = chunk//(2 if half else 1), flag_ar_groups(size//(2 if half else 1))
  if (shape:=flag_ar_shape(words, groups)) is None or (ar:=flag_allreduce(x.device, groups)) is None: return None
  kw = dict(N=N, n=words, T=shape[0], EP=shape[1])
  xf = x.cast(dtype).reshape(N, chunk).contiguous()
  sent = Tensor.custom_kernel(*ar.vals[1:], xf, ar.counter, fxn=functools.partial(_flag_send_kernel, raw=not half, packed=half, **kw))[:N-1]
  out = Tensor.empty(size, dtype=dtype, device=x.device)
  return Tensor.custom_kernel(out, xf, ar.counter, ar.vals[0], *sent, *ar.flags,
                              fxn=functools.partial(_flag_sum_kernel, gather=size, gather_half=half, **kw))[0].reshape(S)

# ******** ring collectives: every device writes to the next one only (0->1->..->N-1->0) ********

# the links: every 7900 XTX has ~25 GB/s out and in at once, behind root complexes shared by two of them (EPYC: the root complex to root
# complex paths ~27 GB/s each way). all to all (every device to every other at once) gets ~11.5 GB/s a device, a ring ~25 GB/s (every device
# to the next: the devices are in bus order, the ring crosses every root complex once each way). an exchange in one kernel a device: a wave
# moves its chunk of every part around the ring, a hop a step (the next device's inbox through an uncached mapping, then a flag with the
# epoch and a checksum of the words). the writes of a peer land in no order (a flag can be seen before the words before it): the receiver
# reads the words until their checksum is the flag's. the polls sleep between loads: polling waves starve the links' writes into the memory
# (more waves: no progress at all). a device is at most one exchange ahead of the next one: two parities of the inbox

SEL_LANE_BASE = 8  # (the lanes from it are the micro batches' second ones (model.SEL_LANE): the same queue)
RING_MAXW = 1 << 20  # words a part (a slot of the inbox: 4 MB, a prefill's part of 4608 rows of q8 is 2.2 MB)

class RingMailbox:
  def __init__(self, devs:tuple[str, ...], chunks:int):
    from tinygrad.device import Buffer, BufferSpec, BufferStorage, MultiBuffer
    N = len(devs)
    def mapped(nbytes:int, dtype:DType) -> tuple[Tensor, Tensor]:
      # (zeroed: a flag the last process left could have an epoch this one waits for. uncached on the device, uncached in the previous one's)
      own = [Buffer(d, nbytes, options=BufferSpec(uncached=True)).ensure_allocated() for d in devs]
      inits.extend((b, bytes(nbytes)) for b in own)
      for b in own: setattr(b, "peer_mtype", 3)
      def alias(b:Buffer, d:int) -> Buffer:
        va = b.get_buf(devs[d])
        return Buffer(devs[d], nbytes, opaque=BufferStorage(va), options=BufferSpec(external_ptr=va)).ensure_allocated()
      (m0:=MultiBuffer.__new__(MultiBuffer)).bufs = own
      (m1:=MultiBuffer.__new__(MultiBuffer)).bufs = [alias(own[(d+1) % N], d) for d in range(N)]
      return Tensor(UOp.from_buffer(m0, dtype, devs)), Tensor(UOp.from_buffer(m1, dtype, devs))
    inits:list[tuple[Buffer, bytes]] = []
    with Context(ALLOW_DEVICE_USAGE=1):
      self.inbox, self.inbox_next = mapped(2*N*RING_MAXW*4, dtypes.uint32)
      self.inbox_bufs = [b for b, _ in inits[-N:]]
      self.flags, self.flags_next = mapped(2*N*chunks*8, dtypes.uint64)
      self.flag_bufs = [b for b, _ in inits[-N:]]
      # (an epoch counter a chunk, then the device's place in the ring)
      counters = [Buffer(d, (chunks+1)*4) for d in devs]
      self.devs, self.counter_bufs = devs, counters
      _init_buffers(inits + [(b, bytes(chunks*4) + i.to_bytes(4, "little")) for i, b in enumerate(counters)])
      (cb:=MultiBuffer.__new__(MultiBuffer)).bufs = counters
      self.counter = Tensor(UOp.from_buffer(cb, dtypes.uint32, devs))
      # (RING_TIMEOUT_MS: what the waves that gave up waited for, in host memory: ring_errors reads it without the devices)
      dbg = [Buffer(d, 64*4, options=BufferSpec(host=True, cpu_access=True)) if RING_TIMEOUT_MS else Buffer(d, 4) for d in devs]
      if RING_TIMEOUT_MS:
        for b in dbg: b.ensure_allocated().as_memoryview(allow_zero_copy=True)[:] = bytes(64*4)
      self.dbg_bufs = dbg
      (db:=MultiBuffer.__new__(MultiBuffer)).bufs = dbg
      self.dbg = Tensor(UOp.from_buffer(db, dtypes.uint32, devs))
      for d in devs: Device[d].synchronize()

RING_W, RING_CMAX = 8, 4096  # waves a workgroup (a wave a chunk), the most (chunk, round) flags of a slot
# (the workgroups of a kernel: few, a kernel spinning with many waves on a device can keep the ones another device waits for from starting.
# a wave's chunk in rounds of a few values a lane (no spills), a flag a round)
# (the collectives of decoding steps on lanes of their own (from DEC_LANE): their exchanges are small, more workgroups (RING_G_DEC) move them in
# fewer rounds. a lane's mailboxes always have the same workgroups: their epoch counters move together)
DEC_LANE = 16
def ring_g(units:int=0) -> int:
  # (the decoding steps' lanes: RING_G_DEC, more for a unit a wave (units: a reduce scatter's), two a wave are ~2.5x the time (78 sequences))
  if FLAG_LANE[0] < DEC_LANE: return 8
  return max(G:=32, min(-(-units // RING_W), 2*G))
# (the slots of a mailbox a chunk's own: its flags (a round each) and words at the same place whatever the kernel's workgroups (a decoding lane's
# exchanges have 32 to 64 of them: their chunks' epochs move apart, a slot shared by two chunks can be overwritten by a device a kernel ahead
# before the next one read it: a hang). the most chunks of a lane: 8 workgroups (prefill), 64 (decoding))
def _ring_chunks(G:int) -> int: return (G if G <= 8 else 64) * RING_W
def _ring_fits(G:int, R:int, span:int) -> bool:
  """the rounds and words of a chunk fit in its slots"""
  return R <= RING_CMAX // _ring_chunks(G) and span <= RING_MAXW // _ring_chunks(G)
def _ring_rounds(units:int, per_round:int, G:int) -> tuple[int, int, int]:
  """(workgroups, rounds, units of a wave a round) of units over the waves of G workgroups (per_round: the most a wave a round)"""
  per_wave = -(-units // (G * RING_W))
  R = -(-per_wave // per_round)
  assert R * G * RING_W <= RING_CMAX, f"ring: {units} units in {R} rounds of {G*RING_W} waves"
  return G, R, -(-per_wave // R)

def ring_mailbox(devs:tuple[str, ...]) -> RingMailbox|None:
  return _ring_mailbox(tuple(Device.canonicalize(d) for d in devs), FLAG_LANE[0])
@functools.cache
def _ring_mailbox(devs:tuple[str, ...], lane:int=0) -> RingMailbox|None:
  if len(devs) < 3 or not amd_custom_kernels_supported(devs): return None
  if any(_gfx(d) not in (11, 12) for d in devs): return None
  _COMM_STATE.append((f"ring lane {lane}", lane, ret:=RingMailbox(devs, RING_CMAX)))
  return ret

RING_DEADLINE:list[UOp|None] = [None]
RING_TIMEOUT_MS = getenv("RING_TIMEOUT_MS", 0)
def ring_errors() -> str:
  """
  the ring exchanges a wave gave up on (RING_TIMEOUT_MS: a peer's flag that didn't come or words that never matched its checksum, e.g. bad memory),
  cleared: a mailbox's device and (exchange's epoch, chunk, the flag's epoch, slot | device<<24 | late<<30 | bad<<31) of a step's first wave
  """
  if not RING_TIMEOUT_MS: return ""
  import struct
  out = []
  for name, _, st in _COMM_STATE:
    if not isinstance(st, RingMailbox): continue
    for d, b in zip(st.devs, st.dbg_bufs):
      mv = b.as_memoryview(allow_zero_copy=True)
      if any(w := struct.unpack("<64I", mv[:256])):
        out.append(f"{name} {d}: " + " ".join(f"({w[4*k]},{w[4*k+1]:#x},{w[4*k+2]},{w[4*k+3]:#x})" for k in range(16) if any(w[4*k:4*k+4])))
        mv[:] = bytes(64*4)
  return "; ".join(out)
def _ring_spin(after:UOp, loads:Callable[[UOp], list[UOp]], stale:Callable[[list[UOp]], UOp], dtype:DType) -> tuple[list[UOp], UOp]:
  # loads (of the loop) again until stale of them (the same in every lane) is false, a sleep between. (the values, still stale (RING_DEBUG))
  loop = UOp.range(UOp(Ops.NOOP).after(after), next(UOp.unique_num), dtype=dtypes.void)
  ld = loads(loop)
  reg = UOp.alloc((len(ld),), dtype, addrspace=AddrSpace.REG)
  loaded = reg.after(loop).store(UOp.stack(*ld)) if len(ld) > 1 else reg.after(loop)[0].store(ld[0])
  got = [reg.after(loaded)[i].load() for i in range(len(ld))]
  st = stale(got)
  if (dl:=RING_DEADLINE[0]) is not None: st = st & (_sys("__builtin_readsteadycounter()", dtypes.uint64, st) < dl)
  s = _sys(f"({{0}} ? (__builtin_amdgcn_s_sleep({getenv('RING_SLEEP', 16)}), true) : false)", dtypes.bool, st)
  reg = reg.after(_cfn(False, src=(s,), arg=("", dtypes.void)).backedge(loop, s))
  vals = [reg[i].load() for i in range(len(ld))]
  return vals, stale(vals)

def _wave_sum(c:UOp) -> UOp:
  for off in (16, 8, 4, 2, 1): c = c + _lane_xor(c.bitcast(dtypes.float32), off).bitcast(dtypes.uint32)
  return c

_LOAD_SYS64 = "__hip_atomic_load({0}, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)"

def _ring_common(counter:UOp, N:int, dbg:UOp, G:int, tag:int=0, R:int=1, span:int=0):
  # (span: the words of a chunk in the kernel's layout, from ch*span)
  W, C, CH = RING_W, G*RING_W, _ring_chunks(G)
  assert G <= 8 or 32 <= G <= 64, f"ring: {G} workgroups"
  assert _ring_fits(G, R, span), f"ring: {R} rounds, {span} words a chunk of {CH}"
  cid = [UOp.const(0, dtypes.weakint)]  # (the round, set by the kernel)
  wg, warp, lane = UOp.range(G, 0, AxisType.GLOBAL), UOp.range(W, 1, AxisType.LOCAL), UOp.range(32, -1, AxisType.WARP)
  ch = wg*W + warp
  # (the chunks' epochs: a chunk's moves with the kernels that have it, the same ones on every device)
  epoch, me = (counter[ch].load() + 1).cast(dtypes.uint32), counter[RING_CMAX].load().cast(dtypes.weakint)
  # (RING_TIMEOUT_MS: a wave waits at most that long for a peer's flag or words, then goes on (its wait in dbg): an error, not a hang)
  RING_DEADLINE[0] = _sys("__builtin_readsteadycounter()", dtypes.uint64, epoch) + RING_TIMEOUT_MS * 100000 if RING_TIMEOUT_MS else None
  dbgs:list[UOp] = []
  par = (epoch & 1).cast(dtypes.weakint)
  def fslot(s:UOp|int) -> UOp: return (par*N + s)*RING_CMAX + ch*(RING_CMAX // CH) + cid[0]
  def islot(s:UOp|int, e:UOp, valid:UOp) -> UOp: return ((par*N + s)*RING_MAXW + ch*(RING_MAXW // CH) + (e - ch*span)).valid(valid)
  def recv(after:UOp, flags:UOp, inbox:UOp, s:UOp|int, at:list[tuple[UOp, UOp]], mult:list[UOp], step:int=0) -> list[UOp]:
    # this device's inbox: the flag of slot s for the epoch, then the words at (index, valid) until their checksum is the flag's (the words
    # not valid are 0, as the sender's)
    (fl,), late = _ring_spin(after, lambda loop: [_sys(_LOAD_SYS64, dtypes.uint64, flags.after(loop)[fslot(s)])],
                             lambda g: (g[0] >> 32).cast(dtypes.uint32).ne(epoch), dtypes.uint64)
    # (a gated custom load reads the slot's start: masked)
    vals, bad = _ring_spin(fl, lambda loop: [v.where(_sys(_LOAD_SYS32, dtypes.uint32, inbox.after(loop)[islot(s, e, v)]), 0) for e, v in at],
                           lambda g: csum(g, mult).ne(fl.cast(dtypes.uint32)), dtypes.uint32)
    if RING_DEADLINE[0] is not None:
      # (the first wave that gave up: its epoch, chunk, flag's epoch | slot<<24 | late<<30 | bad<<31)
      k, u32 = step % 16, dtypes.uint32
      su = UOp.const(s, u32) if isinstance(s, int) else s.cast(u32)
      row = (epoch, UOp.const(tag, u32) + ch.cast(u32), (fl >> 32).cast(u32),
             su | (late.cast(u32) << 30) | (bad.cast(u32) << 31) | (me.cast(u32) << 24))
      dbgs.extend([dbg[UOp.const(4*k + i, dtypes.weakint).valid((late | bad) & lane.eq(0))].store(v) for i, v in enumerate(row)])
    return vals
  def csum(words:list[UOp], mult:list[UOp]) -> UOp: return _wave_sum(functools.reduce(lambda a, b: a + b, [w*m for w, m in zip(words, mult)]))
  def send(after:list[UOp], inbox_next:UOp, flags_next:UOp, s:UOp|int, at:list[tuple[UOp, UOp]], words:list[UOp], mult:list[UOp]) -> UOp:
    sent = [inbox_next.after(*after)[islot(s, e, v)].store(w) for (e, v), w in zip(at, words)]
    # (the fence: the flag after the words' writes are done. the checksum doesn't need it, a flag seen early is a reread)
    fenced = _sys('__builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");', dtypes.void, *sent)
    word = (epoch.cast(dtypes.uint64) << 32) | csum([v.where(w, 0) for (_, v), w in zip(at, words)], mult).cast(dtypes.uint64)
    return flags_next.after(fenced)[fslot(s).valid(lane.eq(0))].store(word)
  def finish(last:UOp) -> UOp: return counter.after(last)[ch.valid(lane.eq(0))].store(epoch)
  def at_round(r:UOp) -> None: cid[0] = r
  def end_round(x:UOp, r:UOp) -> UOp:  # (the round's debug stores in it)
    ret = UOp.group(x, *dbgs).end(r)
    dbgs.clear()
    return ret
  return C, wg, warp, lane, ch, me, recv, send, finish, at_round, end_round

@functools.cache
def _ring_ag_kernel(out:UOp, x:UOp, counter:UOp, inbox:UOp, inbox_next:UOp, flags:UOp, flags_next:UOp, dbg:UOp, *, N:int, n:int, G:int=8) -> UOp:
  # out (N*n words): part s from the device at s, x (n words) this device's. a wave's words in R rounds of EP a lane
  G, R, EP = _ring_rounds(-(-n // 32), 32, G)  # (rows of 32 words: a word a lane)
  C, wg, warp, lane, ch, me, recv, send, finish, at_round, end_round = _ring_common(counter, N, dbg, G, n << 10, R, R*EP*32)
  mult = [(2*(i*32 + lane) + 1).cast(dtypes.uint32) for i in range(EP)]
  def step(k:UOp|int, dep:list[UOp], rid:int, first:bool, last:bool) -> UOp:
    s = (me - k + N) % N
    # (the rounds a loop: a round after the one before, the step after the rounds of the one before)
    r = UOp.range(R, rid, AxisType.LOOP)
    at_round(r)
    at = [(e, e < n) for e in [((ch*R + r)*EP + i)*32 + lane for i in range(EP)]]
    # (x and out of any 4 byte dtype: their words as they are, bitcast here, not by copies of them)
    vals = [x.after(*dep, r)[0, e.valid(v)].load().bitcast(dtypes.uint32) for e, v in at] if first else \
      recv(UOp.group(*dep, r), flags, inbox, s, at, mult, k if isinstance(k, int) else 1)
    st = [out.after(*dep)[(s*n + e).valid(v)].store(w.bitcast(out.dtype)) for (e, v), w in zip(at, vals)]
    return end_round(UOp.group(*st, send(dep, inbox_next, flags_next, s, at, vals, mult)) if not last else UOp.group(*st), r)
  # (the steps between the first and the last a loop, see _ring_rs_q8_kernel)
  prev = step(0, [], 10, True, False)
  if N > 3:
    kk = UOp.range(N-2, 11, AxisType.LOOP)
    prev = step(kk + 1, [prev, kk], 12, False, False).end(kk)
  else:
    for k in range(1, N-1): prev = step(k, [prev], 10 + k, False, False)
  prev = step(N-1, [prev], 10 + N, False, True)
  return UOp.group(prev, finish(prev)).end(wg, warp, lane).sink(arg=KernelInfo(name=f"ring_allgather_{n*4}", opts_to_apply=()))

@functools.cache
def _ring_rs_q8_kernel(out:UOp, p:UOp, counter:UOp, inbox:UOp, inbox_next:UOp, flags:UOp, flags_next:UOp, dbg:UOp, *, N:int, size:int,
                       G:int=8) -> UOp:
  # p (size) this device's partials, out (size/N) the sum of part me (float). in step k a device adds its partials of part me-k-1 to what
  # came (the sum of the devices before it), sends it on as int8 groups of 32 with their scale (q8_pack), the last its own. a wave's groups in
  # R rounds of EP (8 groups a wave, four lanes a group: eight values a lane)
  blk = size // N
  BG = blk // 32
  out = out.reshape(-1)
  G, R, EP = _ring_rounds(-(-BG // 8), 4, G)
  C, wg, warp, lane, ch, me, recv, send, finish, at_round, end_round = _ring_common(counter, N, dbg, G, (1 << 31) | (size >> 8 << 10), R,
                                                                                    R*EP*72)
  sub = lane % 4
  def step(k:UOp|int, dep:list[UOp], rid:int, last:bool, first:bool) -> UOp:
    # step k: the partials of part me-k-1 plus what came (k > 0), sent on (or stored: the last step)
    b = (me - k - 1 + N) % N
    r = UOp.range(R, rid, AxisType.LOOP)
    at_round(r)
    dep = dep + [r]
    groups = [((ch*R + r)*EP + i)*8 + lane//4 for i in range(EP)]
    # the wave's 8 groups on the wire: the words of the lanes (two a lane, lane after lane: a store a line), then the 8 scales (uncached
    # stores of a peer's memory are its writes: lanes' words apart are a write each)
    base = [((ch*R + r)*EP + i)*72 for i in range(EP)]
    at = [(b0 + w*32 + lane, g < BG) for b0, g in zip(base, groups) for w in range(2)] + \
         [(b0 + 64 + lane//4, (g < BG) & sub.eq(0)) for b0, g in zip(base, groups)]
    mult = [(2*(j*32 + lane) + 1).cast(dtypes.uint32) for j in range(len(at))]
    # (the partials loaded after the round before: not all of them at the start, in registers)
    acc = [[p.after(*dep).reshape(-1)[(b*blk + g*32 + sub*8 + j).valid(g < BG)].load().float() for j in range(8)] for g in groups]
    if not first:
      # (the partials' loads before the wait for what came: their latency in it)
      got = recv(UOp.group(*dep, *[a for g in acc for a in g]), flags, inbox, k-1, at, mult,
                 k if isinstance(k, int) else 1)
      for i in range(EP):
        # (the scale from the group's first lane: every lane loads its word, the others' are masked)
        sc = got[2*EP + i].bitcast(dtypes.float32)
        d = _lane_xor(_lane_xor(sc, 1), 2) + _lane_xor(sc, 1) + _lane_xor(sc, 2) + sc
        ws = got[2*i:2*i+2]
        acc[i] = [a + (ws[j//4] >> (8*(j%4))).cast(dtypes.uint8).bitcast(dtypes.int8).float()*d for j, a in enumerate(acc[i])]
    if not last:
      packed = [_q8_pack_values(a, recip=True) for a in acc]
      words = [ws[w] for ws, _ in packed for w in range(2)] + [d.bitcast(dtypes.uint32) for _, d in packed]
      return end_round(send(dep, inbox_next, flags_next, k, at, words, mult), r)
    outs = [out.after(*dep)[(g*32 + sub*8 + j).valid(g < BG)].store(a[j]) for g, a in zip(groups, acc) for j in range(8)]
    return end_round(UOp.group(*outs), r)
  # (the steps between the first and the last a loop: unrolled, every step's graph has all the ones before it (their order), the codegen's
  # simplifications of them ~quadratic: ~25 s a kernel)
  prev = step(0, [], 10, False, True)
  if N > 3:
    kk = UOp.range(N-2, 11, AxisType.LOOP)
    prev = step(kk + 1, [prev, kk], 12, False, False).end(kk)
  else:
    for k in range(1, N-1): prev = step(k, [prev], 10 + k, False, False)
  prev = step(N-1, [prev], 10 + N, True, False)
  assert prev is not None
  return UOp.group(prev, finish(prev)).end(wg, warp, lane).sink(arg=KernelInfo(name=f"ring_reduce_scatter_q8_{size}", opts_to_apply=()))

# the micro batches' calls on compute queues of their own (1.., the first's 0), their collectives on two more (2, 3): set if the devices have
# them (AM_MEC_PIPES=2 at least on am), else all on queue 0 (the copies of one micro batch don't run with the kernels of the other)
EXTRA_QUEUES = [False]
def ring_queue() -> int:
  """
  the compute queue of the ring collectives (0: the calls' own), a queue of their own for the micro batches on queues of theirs (a kernel that
  spins on its peers holds its queue and its waves: two at once on a device (one a batch) can wait on each other through the devices' queues.
  on one queue they run one at a time, in the same order on every device)
  """
  return 2 if EXTRA_QUEUES[0] else 0
def mb_queue(i:int) -> int:
  """the compute queue of micro batch i"""
  return i if EXTRA_QUEUES[0] else 0
RING_QUEUE_ON = [False]  # (the collectives on queues of their own only in the micro batches' prefill: a step's on its queue)
def ring_queue_of(lane:int, any_time:bool=False) -> int:
  """the queue of a lane's collectives (the ring's queue + the lane), 0: the calls' own"""
  if not (any_time or RING_QUEUE_ON[0]): return 0
  return ring_queue() + (lane if ring_queue() else 0)

def _comm_queued(outs:list[Tensor]) -> list[Tensor]:
  """the outputs of a collective's kernel that spins on its peers, its call on the comm queue of the lane (a micro batch's own: lane i on
  RING_QUEUE+i, its exchanges in its order. the compute queues never spin: a queue waiting for one can't make a cycle through the devices)"""
  if not (q:=ring_queue_of(FLAG_LANE[0] % SEL_LANE_BASE)): return outs
  def call_of(u:UOp) -> UOp: return u.src[1] if u.op is Ops.AFTER and u.src[1].op is Ops.CALL else call_of(u.src[0])
  call = call_of(outs[0].uop)
  new = call.replace(arg=replace(call.arg, queue=f"COMPUTE:{q}"))
  # (the call in the outputs' chains of views and afters: not a walk of the graph before it)
  def requeue(u:UOp) -> UOp:
    if u.op is Ops.AFTER and u.src[1] is call: return u.replace(src=(u.src[0], new, *u.src[2:]))
    return u.replace(src=(requeue(u.src[0]), *u.src[1:]))
  return [Tensor(requeue(o.uop)) for o in outs]
def _ring_queued(t:Tensor) -> Tensor: return _comm_queued([t])[0]

def ring_all_gather_words(x:Tensor) -> Tensor|None:
  """x (N, n) 32 bit words split on the devices: (N*n,) on every device, around the ring. None: not supported"""
  devs = x.device
  if not isinstance(devs, tuple) or x.uop.axis != 0 or x.ndim != 2 or x.dtype.itemsize != 4 or not isinstance(n:=x.shape[1], int): return None
  if n > RING_MAXW: return None
  G, R, EP = _ring_rounds(-(-n // 32), 32, ring_g())
  if not _ring_fits(G, R, R*EP*32) or (mb:=ring_mailbox(devs)) is None: return None
  # (out of x's dtype: no bitcast copies of x and out (a kernel each))
  out = Tensor.empty(len(devs)*n, dtype=x.dtype, device=devs)
  return _ring_queued(Tensor.custom_kernel(out, x.contiguous(), mb.counter, mb.inbox, mb.inbox_next, mb.flags,
                                          mb.flags_next, mb.dbg, fxn=functools.partial(_ring_ag_kernel, N=len(devs), n=n, G=ring_g()))[0])

def ring_reduce_scatter_q8(p:Tensor, size:int, shape:tuple[int, ...]|None=None, axis:int=0) -> Tensor|None:
  """
  p (N, size) partials split on the devices: the sum of part i (size/N, float) on device i, around the ring in int8 groups. shape: the output's
  (split on axis), made so (a view of the flat one isn't a buffer: a copy of it)
  """
  devs = p.device
  if not isinstance(devs, tuple) or size % (len(devs)*256): return None
  units = -(-(size//len(devs)//32) // 8)
  G, R, EP = _ring_rounds(units, 4, ring_g(units))
  if G*RING_W*R*EP*72 > RING_MAXW or not _ring_fits(G, R, R*EP*72) or (mb:=ring_mailbox(devs)) is None: return None
  out = Tensor.empty(*(shape or (size,)), dtype=dtypes.float32, device=devs, axis=axis if shape else 0)
  return _ring_queued(Tensor.custom_kernel(out, p.reshape(len(devs), size).contiguous(), mb.counter, mb.inbox, mb.inbox_next, mb.flags,
                                          mb.flags_next, mb.dbg, fxn=functools.partial(_ring_rs_q8_kernel, N=len(devs), size=size, G=G))[0])


# ******** sequence parallel: the rows of a tensor split over the devices ********

def token_sharded(x:Tensor) -> bool:
  """x is split over devices on an axis before its last: its rows if the caller says so (not a leading device axis of partials)"""
  return isinstance(x.device, tuple) and (axis:=x.uop.axis) is not None and axis < x.ndim - 1

def _per_device(x:Tensor) -> UOp:
  # the value on every device of a split kernel output (_materialized): its buffer's view of the device, after the kernel
  u = x.uop
  assert u.op is Ops.AFTER and u.src[0].op is Ops.UNSHARD, f"not a split kernel output: {u.op}"
  return u.src[0].src[0].after(*u.src[1:])

@functools.cache
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
  partial. the parts cross in half (peer to peer), the sums in float
  """
  if (r:=flag_reduce_scatter(p, axis)) is not None: return r
  devs = cast(tuple[str, ...], p.device)
  n, S = len(devs), p.shape[1:]
  # partials that are a kernel's half output split on the devices are taken as they are (no kernel to copy them)
  u = p.uop
  while u.op is Ops.RESHAPE: u = u.src[0]
  direct = p.dtype == dtypes.half and u.op is Ops.AFTER and u.src[0].op is Ops.UNSHARD and u.src[0].src[0].max_numel() == prod(S)
  local = _per_device(Tensor(u)) if direct else _per_device(_materialized(p, dtypes.half))
  size = S[axis] // n
  def part(j:int, i:int) -> UOp:
    return local.mselect(j).reshape(S).shrink(tuple((0, s) if a != axis else (i*size, (i+1)*size) for a, s in enumerate(S)))
  if len(S) == 3 and S[0] == 1 and axis == 1 and S[2] % 256 == 0:
    # around the ring as below, the sums sent as int8 groups of 32 with their scale (q8_pack, a bit over half the bytes): a step adds its
    # part to what it got and quantizes the sum, the device of the rows adds its own (exact) to the sum of the others
    T, D = S[1], S[2]
    pieces = 2 if size % 32 == 0 else 1
    rows = size // pieces
    xs = [Tensor(local.mselect(j)).reshape(T, D) for j in range(n)]
    outs_q:list[list[UOp]] = []
    for pc in range(pieces):
      gq:dict[tuple[int, int], Tensor] = {}
      for k in range(1, n):
        for j in range(n):
          t = (j - k) % n
          pk = Tensor.empty(rows, D//32, 9, dtype=dtypes.uint32, device=devs[j])
          pk = Tensor.custom_kernel(pk, xs[j], *([gq[(t, j)]] if k > 1 else []), fxn=functools.partial(_q8_pack_kernel, row0=t*size + pc*rows))[0]
          gq[(t, (j+1) % n)] = pk.to(devs[(j+1) % n])
      outs_q.append([Tensor.custom_kernel(Tensor.empty(rows, D, dtype=dtypes.float32, device=devs[t]), xs[t], gq[(t, t)],
                                          fxn=functools.partial(_q8_sum_kernel, row0=t*size + pc*rows))[0].uop for t in range(n)])
    whole = [functools.reduce(lambda a, b: a.cat(b, dim=0), [Tensor(o[t]) for o in outs_q]) if pieces > 1 else Tensor(outs_q[0][t]) for t in range(n)]
    return Tensor(UOp.mstack(*[w.reshape(1, size, D).contiguous().uop for w in whole]).unshard(axis))
  # around the ring: in step k device j adds its part of the sum of device t=j-k to what it got (the sum of the parts of t+1..j-1) and
  # sends it to j+1 (the links between neighbours are the fast ones). device t adds its own last. the parts in two pieces, a ring
  # each: one's copy runs while another's sum is computed
  pieces = 2 if size % 32 == 0 and all(s == 1 for s in S[:axis]) else 1
  def piece(u:UOp, pc:int) -> UOp:
    return u if pieces == 1 else u.shrink(tuple((0, s) if a != axis else (pc*(size//pieces), (pc+1)*(size//pieces)) for a, s in enumerate(u.shape)))
  outs = []
  for pc in range(pieces):
    got:dict[tuple[int, int], UOp] = {}
    for k in range(1, n):
      for j in range(n):
        t = (j - k) % n
        mine = piece(part(j, t), pc) if k == 1 else (got[(t, j)] + piece(part(j, t), pc).cast(dtypes.half)).contiguous()
        got[(t, (j+1) % n)] = mine.copy_to_device(devs[(j+1) % n])
    outs.append([(got[(t, t)] + piece(part(t, t), pc).cast(dtypes.half)).float() for t in range(n)])
  return Tensor(UOp.mstack(*[functools.reduce(lambda a, b: a.cat(b, dim=axis), [o[t] for o in outs]).contiguous() for t in range(n)]).unshard(axis))

def _q8_unpacked(pk:UOp, token:UOp, group:UOp, sub:UOp) -> list[UOp]:
  # a lane's eight values of a packed group (q8_pack)
  d = pk[token, group, 8].load().bitcast(dtypes.float32)
  words = [pk[token, group, sub*2 + w].load() for w in range(2)]
  return [(words[i//4] >> (8*(i%4))).cast(dtypes.uint8).bitcast(dtypes.int8).float() * d for i in range(8)]

def _q8_pack_values(values:list[UOp], recip:bool=False) -> tuple[list[UOp], UOp]:
  # (recip: times the scale's reciprocal, not divided by it: a value at a half can round the other way)
  mx = functools.reduce(lambda a, b: a.maximum(b), [v.abs() for v in values])
  for off in (1, 2): mx = mx.maximum(_lane_xor(mx, off))
  d = _cfn(False, src=(mx,), arg=("({0}/127.0f)", dtypes.float)).maximum(1e-8)
  rd = _cfn(False, src=(d,), arg=("(1.0f/{0})", dtypes.float)) if recip else None
  quants = [(_cfn(False, src=(v, rd), arg=("__builtin_nearbyintf({0}*{1})", dtypes.float)) if rd is not None else
             _cfn(False, src=(v, d), arg=("__builtin_nearbyintf({0}/{1})", dtypes.float))).clip(-127, 127).cast(dtypes.int8) for v in values]
  words = [functools.reduce(lambda a, b: a | b, [quants[w*4+i].cast(dtypes.uint8).cast(dtypes.uint32) << (8*i) for i in range(4)]) for w in range(2)]
  return words, d

@functools.cache
def _q8_pack_kernel(out:UOp, x:UOp, *add:UOp, row0:int=0) -> UOp:
  # a row's groups of 32 as q8_quantize's int8 words (rounded the same), the float scale the ninth word: a lane eight neighboring inputs.
  # x from row row0 (plus the packed add)
  T, G = cast(tuple[int, int, int], out.shape)[:2]
  tg8, lane = UOp.range(T*(G//8), 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP)
  token, group, sub = tg8//(G//8), (tg8 % (G//8))*8 + lane//4, lane % 4
  values = [x.reshape(-1, G*32)[row0 + token, group*32 + sub*8 + i].load().float() for i in range(8)]
  if add: values = [a + b for a, b in zip(_q8_unpacked(add[0], token, group, sub), values)]
  words, d = _q8_pack_values(values)
  stores = [out[token, group, sub*2 + w].store(words[w]) for w in range(2)]
  stores.append(out[token, group, (sub*0+8).valid(sub.eq(0))].store(d.bitcast(dtypes.uint32)))
  return UOp.group(*stores).end(tg8, lane).sink(arg=KernelInfo(name="q8_pack", opts_to_apply=()))

@functools.cache
def _q8_pack_send_kernel(out:UOp, *args:UOp, N:int, n:int, T:int, EP:int) -> UOp:
  # _q8_pack_kernel and _flag_send_kernel (raw words) in one: the packed words stored, and sent into this device's slot of every peer's inbox
  # (args: the peers' inboxes 1..N-1, x, the counters). a word e is the flag exchange's workgroup e//(EP*T)'s, its parity that one's epoch's
  vals_mb, x, counter = args[:N-1], args[N-1], args[N]
  TT, G = cast(tuple[int, int, int], out.shape)[:2]
  tg8, lane = UOp.range(TT*(G//8), 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP)
  token, group, sub = tg8//(G//8), (tg8 % (G//8))*8 + lane//4, lane % 4
  values = [x.reshape(-1, G*32)[token, group*32 + sub*8 + i].load().float() for i in range(8)]
  words, d = _q8_pack_values(values)
  def put(w:UOp, word:UOp, ok:UOp|None=None) -> list[UOp]:
    e = (token*G + group)*9 + w
    epoch = (counter[e // (EP*T)].load() + 1).cast(dtypes.uint32)
    at = (e // 64)*128 + (epoch & 1).cast(dtypes.weakint)*64 + e % 64
    return [out[token, group, w if ok is None else w.valid(ok)].store(word)] + \
      [vals_mb[k][at if ok is None else at.valid(ok)].store(word) for k in range(N-1)]
  stores = [st for w in range(2) for st in put(sub*2 + w, words[w])]
  stores += put(sub*0 + 8, d.bitcast(dtypes.uint32), sub.eq(0))
  return UOp.group(*stores).end(tg8, lane).sink(arg=KernelInfo(name="q8_pack_send", opts_to_apply=()))

@functools.cache
def _q8_unpack_kernel(out:UOp, *parts:UOp) -> UOp:
  # the halves of the packed groups: a lane eight inputs (two words), times the group's scale. parts: the rows in that many parts (the part of
  # a row is the workgroup's: its loads from the others are skipped)
  T, D = cast(tuple[int, int], out.shape)
  G, rows = D // 32, T // len(parts)
  tg8, lane = UOp.range(T*(G//8), 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP)
  token, group, sub = tg8//(G//8), (tg8 % (G//8))*8 + lane//4, lane % 4
  def load(w:int) -> UOp:
    if len(parts) == 1: return parts[0][token, group, w].load()
    pt, r = token // rows, token % rows
    return functools.reduce(lambda a, b: a + b, [(pt.eq(i)).where(p.reshape(rows, G, 9)[r.valid(pt.eq(i)), group, w].load(), 0)
                                                 for i, p in enumerate(parts)])
  d = load(8).bitcast(dtypes.float32)
  words = [load(sub*2 + w) for w in range(2)]
  vals = [((words[i//4] >> (8*(i%4))).cast(dtypes.uint8).bitcast(dtypes.int8).float() * d).cast(dtypes.half) for i in range(8)]
  stores = [out[token, group*32 + sub*8 + i].store(vals[i]) for i in range(8)]
  return UOp.group(*stores).end(tg8, lane).sink(arg=KernelInfo(name="q8_unpack", opts_to_apply=()))

@functools.cache
def _q8_sum_kernel(out:UOp, mine:UOp, *packed:UOp, row0:int=0) -> UOp:
  # the packed groups (q8_pack) of the others plus mine (from row row0), in float: a lane eight inputs
  T, D = cast(tuple[int, int], out.shape)
  G = D // 32
  tg8, lane = UOp.range(T*(G//8), 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP)
  token, group, sub = tg8//(G//8), (tg8 % (G//8))*8 + lane//4, lane % 4
  acc = [UOp.const(0, dtypes.float)]*8
  for pk in packed: acc = [a + b for a, b in zip(acc, _q8_unpacked(pk, token, group, sub))]
  acc = [acc[i] + mine.reshape(-1, D)[row0 + token, group*32 + sub*8 + i].load().float() for i in range(8)]
  stores = [out[token, group*32 + sub*8 + i].store(acc[i].cast(out.dtype)) for i in range(8)]
  return UOp.group(*stores).end(tg8, lane).sink(arg=KernelInfo(name="q8_sum", opts_to_apply=()))

def all_gather_q8(x:Tensor, axis:int, real:int=0) -> Tensor:
  """
  all_gather of x (1, T, D) split on its rows (axis 1) in half, sent as int8 groups of 32 with their scale (as q8_quantize: the activations
  the quantized layers' kernels would make), a bit over half the bytes. every device's own rows are quantized too (the same values everywhere).
  real: x's rows of every device are padded, its first `real` ones gathered (packed from x as it is, not a copy of them)
  """
  assert axis == 1 and x.ndim == 3 and x.shape[0] == 1
  devs, (T, D) = cast(tuple[str, ...], x.device), cast(tuple[int, int], x.shape[1:])
  src = x.reshape(T, D)
  if real: T = len(devs) * real
  packed = Tensor.empty(T, D//32, 9, dtype=dtypes.uint32, device=devs, axis=0)
  N, n = len(devs), (T // len(devs)) * (D//32) * 9
  # AG_FUSE: the flag exchange's send in the packing kernel (one kernel less)
  if AG_FUSE and N*n*4 < ring_min(devs) and (shape:=flag_ar_shape(n, groups:=flag_ar_groups(N*n))) is not None and \
     (ar:=flag_allreduce(devs, groups)) is not None:
    kw = dict(N=N, n=n, T=shape[0], EP=shape[1])
    res = Tensor.custom_kernel(packed, *ar.vals[1:], src, ar.counter, fxn=functools.partial(_q8_pack_send_kernel, **kw))
    words = Tensor.empty(N*n, dtype=dtypes.uint32, device=devs)
    gath = Tensor.custom_kernel(words, res[0].reshape(N, n), ar.counter, ar.vals[0], *res[1:N], *ar.flags,
                                fxn=functools.partial(_flag_sum_kernel, gather=N*n, gather_half=False, **kw))[0].reshape(T, D//32, 9)
    return Tensor.custom_kernel(Tensor.empty(T, D, dtype=dtypes.half, device=devs), gath, fxn=_q8_unpack_kernel)[0].reshape(1, T, D)
  packed = Tensor.custom_kernel(packed, src, fxn=_q8_pack_kernel)[0]
  if (gathered:=flag_all_gather(packed, 0)) is not None:
    out = Tensor.empty(T, D, dtype=dtypes.half, device=devs)
    return Tensor.custom_kernel(out, gathered, fxn=_q8_unpack_kernel)[0].reshape(1, T, D)
  # the parts around the ring from the packed buffers, every device unpacks them as they are (no buffer of the whole packed)
  local = _per_device(packed)
  mine, n = [local.mselect(i) for i in range(len(devs))], len(devs)
  sent = _ring(mine, devs)
  def unpacked(j:int) -> UOp:
    parts = [Tensor(mine[i] if i == j else sent[(i, j)]) for i in range(n)]
    return Tensor.custom_kernel(Tensor.empty(T, D, dtype=dtypes.half, device=devs[j]), *parts, fxn=_q8_unpack_kernel)[0].reshape(1, T, D).uop
  outs = [unpacked(j) for j in range(n)]
  return Tensor(UOp.mstack(*outs))

def _ring(mine:list[UOp], devs:tuple[str, ...]) -> dict[tuple[int, int], UOp]:
  """
  part i (on device i) on every other device j, around the ring: in step k device j sends the part it got in step k-1 to device j+1 (the
  links between neighbours are the fast ones: every copy is on one). a step's copies wait for the ones before (their data)
  """
  n, got = len(devs), cast(dict[tuple[int, int], UOp], {})
  for k in range(1, n):
    for j in range(n):
      i = (j - k + 1) % n  # the part device j forwards in step k
      got[(i, (j+1) % n)] = (mine[j] if k == 1 else got[(i, j)]).copy_to_device(devs[(j+1) % n])
  return got

def _flag_chunked_all_gather(x:Tensor, axis:int, dtype=None) -> Tensor|None:
  # Large residual streams exceed a mailbox slot. Exchange slices of each rank's shard, then restore rank-major order.
  dtype = dtype or x.dtype
  if not isinstance(x.device, tuple) or x.uop.axis != axis or prod(x.shape[:axis]) != 1 or \
     dtype not in (dtypes.half, dtypes.float32, dtypes.int32, dtypes.uint32) or not isinstance(x.numel(), int): return None
  N, pack = len(x.device), 2 if dtype == dtypes.half else 1
  chunk = int(x.numel()) // N
  if chunk <= FLAG_AR_MAXN*pack: return None
  parts = next((p for p in (2, 4, 8, 16, 32, 64) if chunk % (p*pack) == 0 and chunk//p <= FLAG_AR_MAXN*pack), 0)
  if not parts: return None
  source = x.reshape(N, parts, chunk//parts)
  got:list[Tensor] = []
  for p in range(parts):
    piece = source[:, p].reshape(N*chunk//parts).contiguous()
    # serialize exchanges sharing the counter and mailbox (after the last one's kernels: an AFTER is of effects)
    if got: piece = Tensor(piece.uop.after(*[u for u in got[-1].uop.toposort() if u.op is Ops.CALL][-2:]))
    if (y:=flag_all_gather(piece, 0, dtype)) is None: return None
    got.append(y.reshape(N, chunk//parts))
  return got[0].stack(*got[1:], dim=1).reshape(x.shape).contiguous()

def all_gather(x:Tensor, axis:int, dtype=None) -> Tensor:
  """a tensor split on axis over the devices, whole on every device (in dtype): every part goes to the others peer to peer"""
  if (r:=flag_all_gather(x, axis, dtype)) is not None: return r
  if (r:=_flag_chunked_all_gather(x, axis, dtype)) is not None: return r
  devs = cast(tuple[str, ...], x.device)
  local = _per_device(_materialized(x, dtype or x.dtype))
  # (the parts are sent from the buffer)
  mine, n = [local.mselect(i) for i in range(len(devs))], len(devs)
  sent = _ring(mine, devs)
  # (every device's whole a buffer: its readers don't each make one)
  whole = [Tensor.cat(*[Tensor(mine[i] if i == j else sent[(i, j)]) for i in range(n)], dim=axis) for j in range(n)]
  return Tensor(UOp.mstack(*[w.contiguous().uop for w in whole]))

def host_allgather(x:Tensor) -> Tensor|None:
  """
  x (devices, n) floats, a row on every device: the whole (devices, n) on every device, written to the others' memory (or through the host
  mailbox. None: not supported)
  """
  return flag_all_gather(x, 0) if isinstance(x.device, tuple) else None

@functools.cache
def _local_argmax_kernel(out:UOp, x:UOp, count:UOp) -> UOp:
  # a workgroup per row: the max of the first count values of the row (the device's real ones) and its first index, as (value, index bits)
  B, V = cast(tuple[int, int], x.shape)
  T = 256
  row, t = UOp.range(B, 0, AxisType.GLOBAL), UOp.range(T, 1, AxisType.LOCAL)
  n = count[0].load()
  best, idx = UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG), UOp.alloc((1,), dtypes.int32, addrspace=AddrSpace.REG)
  best = best.after(best[0].store(UOp.const(-math.inf, dtypes.float32)))
  idx = idx.after(idx[0].store(UOp.const(0, dtypes.int32)))
  i = UOp.range((V + T - 1) // T, 2, AxisType.LOOP)
  col = i * T + t
  v = (col < n).where(x[row, col.minimum(V-1)].load().float(), UOp.const(-math.inf, dtypes.float32))
  cb, ci = best.after(i)[0].load(), idx.after(i)[0].load()
  better = v > cb
  upd = UOp.group(best[0].store(better.where(v, cb)), idx[0].store(better.where(col.cast(dtypes.int32), ci))).end(i)
  vb, vi = best.after(upd)[0].load(), idx.after(upd)[0].load()
  # the workgroup's max: the best of every thread in LDS, then a thread picks the first max (the lowest index of the max)
  lds = UOp.alloc((T, 2), dtypes.float32, addrspace=AddrSpace.LOCAL)
  lds = lds.after(UOp.group(lds[t, 0].store(vb), lds[t, 1].store(vi.bitcast(dtypes.float32))).barrier())
  win_v, win_i = lds[0, 0].load(), lds[0, 1].load().bitcast(dtypes.int32)
  for j in range(1, T):
    v2, i2 = lds[j, 0].load(), lds[j, 1].load().bitcast(dtypes.int32)
    take = (v2 > win_v) | (v2.eq(win_v) & (i2 < win_i))
    win_v, win_i = take.where(v2, win_v), take.where(i2, win_i)
  return UOp.group(out[row, UOp.const(0, dtypes.weakint).valid(t.eq(0))].store(win_v),
                   out[row, UOp.const(1, dtypes.weakint).valid(t.eq(0))].store(win_i.bitcast(dtypes.float32))).end(row, t) \
    .sink(arg=KernelInfo(name="local_argmax", opts_to_apply=()))

@functools.cache
def _argmax_parts(devs:tuple[str, ...], local:int, n:int) -> tuple[Tensor, Tensor]:
  # the count of the first n columns on every device (split), the index of its first column (on every device)
  N = len(devs)
  return (Tensor([max(0, min(local, n - i*local)) for i in range(N)], dtype=dtypes.int32).shard(devs, 0).realize(),
          Tensor([[i*local] for i in range(N)], dtype=dtypes.int32).to(devs).realize())

@functools.cache
def _argmax_chunks_kernel(out:UOp, x:UOp, count:UOp) -> UOp:
  # out (B, C, 2): the max of every chunk of a row's first count values and its first index (as local_argmax, a workgroup a chunk)
  B, V = cast(tuple[int, int], x.shape)
  C, T = int(out.shape[1]), 256
  CW = round_up((V + C - 1) // C, T)
  row, c, t = UOp.range(B, 0, AxisType.GLOBAL), UOp.range(C, 1, AxisType.GLOBAL), UOp.range(T, 2, AxisType.LOCAL)
  n = count[0].load()
  best, idx = UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG), UOp.alloc((1,), dtypes.int32, addrspace=AddrSpace.REG)
  best = best.after(best[0].store(UOp.const(-math.inf, dtypes.float32)))
  idx = idx.after(idx[0].store(UOp.const(0, dtypes.int32)))
  i = UOp.range(CW // T, 3, AxisType.LOOP)
  col = c*CW + i*T + t
  v = ((col < n) & (col < V)).where(x[row, col.minimum(V-1)].load().float(), UOp.const(-math.inf, dtypes.float32))
  cb, ci = best.after(i)[0].load(), idx.after(i)[0].load()
  better = v > cb
  upd = UOp.group(best[0].store(better.where(v, cb)), idx[0].store(better.where(col.cast(dtypes.int32), ci))).end(i)
  vb, vi = best.after(upd)[0].load(), idx.after(upd)[0].load()
  lds = UOp.alloc((T, 2), dtypes.float32, addrspace=AddrSpace.LOCAL)
  lds = lds.after(UOp.group(lds[t, 0].store(vb), lds[t, 1].store(vi.bitcast(dtypes.float32))).barrier())
  win_v, win_i = lds[0, 0].load(), lds[0, 1].load().bitcast(dtypes.int32)
  for j in range(1, T):
    v2, i2 = lds[j, 0].load(), lds[j, 1].load().bitcast(dtypes.int32)
    take = (v2 > win_v) | (v2.eq(win_v) & (i2 < win_i))
    win_v, win_i = take.where(v2, win_v), take.where(i2, win_i)
  return UOp.group(out[row, c, UOp.const(0, dtypes.weakint).valid(t.eq(0))].store(win_v),
                   out[row, c, UOp.const(1, dtypes.weakint).valid(t.eq(0))].store(win_i.bitcast(dtypes.float32))).end(row, c, t) \
    .sink(arg=KernelInfo(name="argmax_chunks", opts_to_apply=()))

@functools.cache
def _argmax_final_kernel(out:UOp, parts:UOp) -> UOp:
  # out (B, 2): the first max of the chunks of a row (in order: the lowest index of the max)
  B, C = int(parts.shape[0]), int(parts.shape[1])
  row = UOp.range(B, 0, AxisType.GLOBAL)
  win_v, win_i = parts[row, 0, 0].load(), parts[row, 0, 1].load()
  for c in range(1, C):
    v2, i2 = parts[row, c, 0].load(), parts[row, c, 1].load()
    take = v2 > win_v
    win_v, win_i = take.where(v2, win_v), take.where(i2, win_i)
  return UOp.group(out[row, 0].store(win_v), out[row, 1].store(win_i)).end(row).sink(arg=KernelInfo(name="argmax_final", opts_to_apply=()))

def sharded_argmax(x:Tensor, n:int) -> Tensor|None:
  """
  x (B, V) split on its last axis over the devices: the argmax (B, 1) int32 of the first n columns (the first max) on every device. every
  device takes the max of its part, the (max, index) pairs are exchanged through host memory. None: not supported
  """
  devs = x.device
  if not (isinstance(devs, tuple) and x.uop.axis == x.ndim-1 and amd_custom_kernels_supported(devs)): return None
  N, (B, V) = len(devs), x.shape
  count, first = _argmax_parts(devs, local:=int(V) // N, n)
  part = Tensor.empty(N*B, 2, dtype=dtypes.float32, device=devs, axis=0)
  if local >= (C:=32)*256:
    # (a row's chunks a workgroup each, then their max: one workgroup a row is a long serial scan)
    chunks = Tensor.empty(N*B, C, 2, dtype=dtypes.float32, device=devs, axis=0)
    chunks = Tensor.custom_kernel(chunks, x.contiguous(), count, fxn=_argmax_chunks_kernel)[0]
    part = Tensor.custom_kernel(part, chunks, fxn=_argmax_final_kernel)[0]
  else: part = Tensor.custom_kernel(part, x.contiguous(), count, fxn=_local_argmax_kernel)[0]
  if (g:=host_allgather(part.reshape(N, B*2))) is None: return None
  g = g.reshape(N, B, 2)
  vals, idxs = g[..., 0], g[..., 1].bitcast(dtypes.int32) + first
  # the lowest index of the max
  return vals.eq(vals.max(0, keepdim=True)).where(idxs, n).min(0).reshape(B, 1)

@functools.cache
def _nucleus_round_kernel(out:UOp, z:UOp, e:UOp, t:UOp) -> UOp:
  # out (1, B, G): the sums of e over the values of z at or above each of the G thresholds of its row (a device's part of the vocab): a
  # workgroup a row, a thread every 256th value (all the thresholds), the threads' sums added in LDS in thread order
  B, _, C, c = cast(tuple[int, int, int, int], z.shape)
  G = int(t.shape[-1])
  row, tid = UOp.range(B, 0, AxisType.GLOBAL), UOp.range(256, 1, AxisType.LOCAL)
  ts = [t[row, g].load() for g in range(G)]
  acc = UOp.alloc((G,), dtypes.float32, addrspace=AddrSpace.REG)
  acc = acc.after(UOp.group(*[acc[g].store(UOp.const(0.0, dtypes.float32)) for g in range(G)]))
  i = UOp.range(C * c // 256, 2, AxisType.LOOP)
  pos = i * 256 + tid
  zv, ev = z[row, 0, pos // c, pos % c].load(), e[row, 0, pos // c, pos % c].load()
  upd = UOp.group(*[acc[g].store(acc.after(i)[g].load() + (zv >= ts[g]).where(ev, 0.0)) for g in range(G)]).end(i)
  lds = UOp.alloc((256, G), dtypes.float32, addrspace=AddrSpace.LOCAL)
  lds = lds.after(UOp.group(*[lds[tid, g].store(acc.after(upd)[g].load()) for g in range(G)]).barrier())
  total = functools.reduce(lambda a, b: a + b, [lds[k, tid.minimum(G-1)].load() for k in range(256)])
  return out[0, row, tid.valid(tid < G)].store(total).end(row, tid).sink(arg=KernelInfo(name="nucleus_round", opts_to_apply=()))

def nucleus(z:Tensor, n:int, top_p:Tensor, rounds:int|None=None, G:int|None=None) -> Tensor:
  """
  the tokens (B, V) of the top_p nucleus of softmax(z[:, :n]) (top-p sampling): the ones at least as likely as a threshold t, the largest with
  the probability of the ones at or above it at least top_p (B, 1). t is found by `rounds` searches of G thresholds (to 50/G**rounds of a log
  probability: the next ones as likely as the last in, to that). z split on its last axis over the devices: every device sums its part, the sums
  are exchanged (host_allgather). top_p >= 1: every token
  """
  rounds, G = rounds or 4, G or 16
  B, V = int(z.shape[0]), int(z.shape[1])
  z = (Tensor.arange(V).to(z.device) < n).where(z, -math.inf)
  split = isinstance(devs:=z.device, tuple) and z.uop.axis == 1
  N = len(devs) if isinstance(devs, tuple) and split else 1
  zl = z.reshape(B, N, V // N)
  # (the sums of a device's part in two steps, chunks of 256 then theirs: one sum over all of them is a few workgroups)
  c = 256
  zc = zl.pad((None, None, (0, round_up(V // N, c) - V // N)), value=-math.inf).reshape(B, N, -1, c)
  def vsum(x:Tensor, op:str="sum") -> Tensor: return (x.sum(-1).contiguous().sum(-1) if op == "sum" else x.max(-1).contiguous().max(-1))
  def parts(part:Tensor) -> Tensor:
    # part (B, N, ...) a device's in its row of N: every device's (N, B, ...) on every device
    if not split: return part.permute(1, 0, *range(2, part.ndim))
    g = host_allgather(part.permute(1, 0, *range(2, part.ndim)).reshape(N, -1))
    assert g is not None, "no host_allgather on these devices"
    return g.reshape(N, *part.shape[:1], *part.shape[2:])
  # the max and the sum of exp past it of every device's part, at once: the max, the sum of all
  pm = vsum(zc, "max")
  g = parts(pm.stack(vsum((zc - pm.reshape(B, N, 1, 1)).exp()), dim=-1))  # (N, B, 2)
  m = g[..., 0].max(0)
  mass = ((g[..., 1] * (g[..., 0] - m).exp()).sum(0) * top_p.reshape(-1)).reshape(B, 1)
  m = m.reshape(B, 1)
  e = (zc - m.reshape(B, 1, 1, 1)).exp().contiguous()  # (once, not a round's every threshold)
  steps = Tensor.arange(G).float().reshape(1, G).to(z.device) / G
  lo, w = m - 50.0, 50.0
  for _ in range(rounds):
    t = lo + w * steps  # (B, G) up from lo: the mass at or above them goes down
    if split and G <= 256:
      # (a device's sums in one kernel: the generic one of the (B, G, V) comparisons is ~0.2 TFLOPS)
      out = Tensor.empty(N, B, G, dtype=dtypes.float32, device=devs, axis=0)
      local = Tensor.custom_kernel(out, zc.contiguous(), e, t.contiguous(), fxn=_nucleus_round_kernel)[0].permute(1, 0, 2)  # (B, N, G)
      above = parts(local).sum(0)
    else: above = parts(vsum(e.unsqueeze(2) * (zc.unsqueeze(2) >= t.reshape(B, 1, G, 1, 1)))).sum(0)  # (B, G)
    k = (above >= mass).sum(-1, keepdim=True).maximum(1)  # (the first is lo: at least the mass)
    lo, w = lo + w * (k - 1).float() / G, w / G
  return ((zl >= lo.reshape(B, 1, 1)) | (top_p.reshape(-1, 1, 1) >= 1)).reshape(B, V)


def partial_sum(x:Tensor) -> Tensor:
  """the sum over the leading device axis (an allreduce). many tokens (prefill) are summed in half: the transfers are half the size"""
  # (peer to peer, the values in half and a flag a workgroup: a write over the links and a poll of this device's memory)
  if (r:=flag_partial_sum(x)) is not None: return r
  n = prod(x.shape[1:])
  if prod(x.shape[1:-1]) > 1:
    if isinstance(x.device, tuple) and isinstance(n, int) and n <= 256_000 and x.uop.axis == 0:
      # every device's half partial copied to the others (as the naive allreduce), summed in float in device order: the same bits everywhere
      h = x.half().contiguous()
      return functools.reduce(lambda a,b: a + b, [h[i].to(x.device).float() for i in range(x.shape[0])])
    return x.half().sum(0, dtype=dtypes.half).float()
  return x.sum(0)

# ******** mixture of experts: the selected experts of each token ********


@functools.cache
def _skinny_kernel(out:UOp, x:UOp, wt:UOp, RB:int, KC:int, U:int) -> UOp:
  # out (KS, R, E) partial sums: workgroup (r, s) rows r*RB.. over the inputs s*KC.., a thread an expert (wt (K, E): every load of a step
  # one row of it for all the threads), U inputs a step (their loads together), RB accumulators a thread
  KS, R, E = cast(tuple[int, int, int], out.shape)
  rb, s, e = UOp.range(R // RB, 0, AxisType.GLOBAL), UOp.range(KS, 1, AxisType.GLOBAL), UOp.range(E, 2, AxisType.LOCAL)
  acc = UOp.alloc((RB,), dtypes.float32, addrspace=AddrSpace.REG)
  acc = acc.after(UOp.group(*[acc[i].store(UOp.const(0.0, dtypes.float32)) for i in range(RB)]))
  k = UOp.range(KC // U, 3, AxisType.LOOP)
  col = s * KC + k * U
  ws = [wt[col + u, e].load().float() for u in range(U)]
  def dot(i:int) -> UOp: return functools.reduce(lambda a, b: a + b, [x[rb * RB + i, col + u].load() * ws[u] for u in range(U)])
  upd = UOp.group(*[acc[i].store(acc.after(k)[i].load() + dot(i)) for i in range(RB)]).end(k)
  return UOp.group(*[out[s, rb * RB + i, e].store(acc.after(upd)[i].load()) for i in range(RB)]).end(rb, s, e) \
    .sink(arg=KernelInfo(name="skinny_matmul", opts_to_apply=()))

@functools.cache
def _sum_parts_kernel(out:UOp, part:UOp) -> UOp:
  # out (n,) the sum of part (S, n) over S in its order: a thread 4 neighboring outputs (the loads 16 bytes), the generic reduction of a leading
  # axis is several times slower
  S, n = cast(tuple[int, int], part.shape)
  g, t = UOp.range((n // 4 + 255) // 256, 0, AxisType.GLOBAL), UOp.range(256, 1, AxisType.LOCAL)
  i4 = g * 256 + t
  i = (i4 * 4).minimum(n - 4)
  vals = [functools.reduce(lambda a, b: a + b, [part[s, i + j].load() for s in range(S)]) for j in range(4)]
  return UOp.group(*[out[(i4 * 4 + j).valid(i4 * 4 < n)].store(v) for j, v in enumerate(vals)]).end(g, t).sink(
    arg=KernelInfo(name="sum_parts", opts_to_apply=()))

@functools.cache
def _few_out_rows_kernel(out:UOp, x:UOp, w:UOp, *, RW:int, W:int) -> UOp:
  # out (R, E) = x (R, K) @ w (E, K)^T in float: a wave RW rows, its lanes 8 inputs of every 256 (a chunk), the weights of a chunk loaded once
  # for the wave's rows, a wave's sums reduced over its lanes
  R, E = cast(tuple[int, int], out.shape)
  K = cast(int, x.shape[-1])
  x, w = x.reshape(R, K), w.reshape(E, K)
  wg, wave, lane = UOp.range(R // (W*RW), 0, AxisType.GLOBAL), UOp.range(W, 1, AxisType.LOCAL), UOp.range(32, -1, AxisType.WARP)
  rows = [(wg*W + wave)*RW + r for r in range(RW)]
  acc = [[UOp.const(0.0, dtypes.float)]*E for _ in range(RW)]
  for c in range(K // 256):
    k0 = c*256 + lane*8
    wv = [[w[e, k0 + j].load().float() for j in range(8)] for e in range(E)]
    for r, row in enumerate(rows):
      xv = [x[row, k0 + j].load().float() for j in range(8)]
      acc[r] = [functools.reduce(lambda a, j: a + xv[j]*wv[e][j], range(8), acc[r][e]) for e in range(E)]
  stores = [out[row, UOp.const(e, dtypes.weakint).valid(lane.eq(0))].store(warp_reduce(acc[r][e], full_wave=True))
            for r, row in enumerate(rows) for e in range(E)]
  return UOp.group(*stores).end(wg, wave, lane).sink(arg=KernelInfo(name="few_out_rows", opts_to_apply=()))

@functools.cache
def _after_kernel(out:UOp, x:UOp, dep:UOp) -> UOp:
  # (workgroups of 256 threads, 8 values a thread: a range alone is a workgroup a value, ~0.5 ms for a chunk's rows)
  n = int(prod(out.shape))
  g, t = UOp.range((n + 2047) // 2048, 0, AxisType.GLOBAL), UOp.range(256, 1, AxisType.LOCAL)
  d = dep.reshape(-1)[0].load().float() * 0
  z = d.ne(d).where(UOp.const(0.0, dtypes.float), d)  # (zero: also from a NaN or inf)
  idx = [g * 2048 + j * 256 + t for j in range(8)]
  return UOp.group(*[out.reshape(-1)[i.valid(i < n)].store((x.reshape(-1)[i.minimum(n-1)].load().float() + z).cast(out.dtype)) for i in idx]).end(
    g, t).sink(arg=KernelInfo(name="after_read", opts_to_apply=()))

def _state_token_kernel(out:UOp, *deps:UOp) -> UOp:
  z = functools.reduce(lambda a, d: a + d.reshape(-1)[0].load().float() * 0, deps, UOp.const(0.0, dtypes.float))
  return out.reshape(-1)[0].store(z.ne(z).where(UOp.const(0.0, dtypes.float), z)).sink(arg=KernelInfo(name="state_token", opts_to_apply=()))

def state_token(like:Tensor, deps:list[Tensor]) -> Tensor:
  """a zero on every device of like after reading a value of each of deps (on every device its part): a kernel after their writes"""
  devs = like.device
  out = Tensor.empty(len(devs), dtype=dtypes.float32, device=devs, axis=0) if isinstance(devs, tuple) else Tensor.empty(1, dtype=dtypes.float32)
  return Tensor.custom_kernel(out, *deps, fxn=_state_token_kernel)[0]

def after_kernel(x:Tensor, dep:Tensor) -> Tensor:
  """x, after the call of dep (a kernel that reads a value of it): a micro batch's order on another queue (see micro_batch_blocks)"""
  out = Tensor.empty(*x.shape, dtype=x.dtype, device=x.device, axis=x.uop.axis)
  return Tensor.custom_kernel(out, x.contiguous(), dep, fxn=_after_kernel)[0]

def few_out_rows(x:Tensor, w:Tensor) -> Tensor|None:
  """
  x (..., K) @ w (E, K)^T in float: few outputs (a gated deltanet's alpha and beta, a device its heads') of the many rows of a prefill on every
  device (the generic kernel is a few workgroups of 32 threads: ~20x slower). None: not those shapes
  """
  R, K = int(prod(x.shape[:-1])), int(x.shape[-1])
  E = int(w.uop.shard_shape[0]) if isinstance(w.device, tuple) and w.uop.axis == 0 else int(w.shape[0])
  if not amd_custom_kernels_supported(x.device) or K % 256 or E > 16 or R % 32 or R < 32:
    return None
  if isinstance(x.device, tuple) and (x.uop.axis is not None or w.uop.axis not in (0, None)): return None
  out = Tensor.empty(R, int(w.shape[0]), dtype=dtypes.float32, device=x.device, axis=1 if isinstance(w.device, tuple) and w.uop.axis == 0 else None)
  # (a few hundred rows (batched decoding's): a wave a row, 4 a workgroup: 64 rows were 2 workgroups (11 us, 0.8 ms a step of 16 sequences). a
  # row's sums are the same either way)
  RW, W = (1, 4) if R <= 256 and R % 4 == 0 else (4, 8)
  out = Tensor.custom_kernel(out, x.reshape(R, K).half().contiguous(), w, fxn=functools.partial(_few_out_rows_kernel, RW=RW, W=W))[0]
  return out.reshape(*x.shape[:-1], int(w.shape[0]))

def skinny_matmul(x:Tensor, wt:Tensor) -> Tensor|None:
  """x (B, T, K) @ wt (K, E) in float, few outputs of few rows (see skinny_parts), the parts summed. None: not those shapes"""
  if (part:=skinny_parts(x, wt)) is None: return None
  KS, R, E = cast(tuple[int, int, int], part.shape)
  # (x's rows split over the devices (a prefill's, sequence parallel): every device its rows)
  logits = Tensor.empty(R*E, dtype=dtypes.float32, device=x.device, axis=None if part.uop.axis is None else 0)
  return Tensor.custom_kernel(logits, part.reshape(KS, R*E), fxn=_sum_parts_kernel)[0].reshape(*x.shape[:-1], E)

def skinny_parts(x:Tensor, wt:Tensor) -> Tensor|None:
  """
  the partial sums (KS, B*T, E) of x (B, T, K) @ wt (K, E) (float or half) in float over KS parts of the inputs, few outputs (a router's, an
  indexer's head weights) of the few rows of decoding: the inputs split over workgroups, a thread an output (the generic kernels are 3-4x
  slower). None: not those shapes
  """
  K, E = cast(tuple[int, int], wt.shape)
  R = int(prod(x.shape[:-1]))
  # (rows split over the devices: x (1, R, K) on its rows, the kernel of every device's)
  split = token_sharded(x)
  if split and (x.uop.axis != x.ndim - 2 or prod(x.shape[:-2]) != 1 or wt.uop.axis is not None): return None
  RL = int(x.uop.shard_shape[-2]) if split else R
  if not amd_custom_kernels_supported(x.device) or E > 1024 or RL > 1024: return None
  RB = next((r for r in (8, 8, 6, 4, 3, 2, 1) if RL % r == 0), 1)
  KS = next((ks for ks in (16, 16, 8, 4, 2, 1) if K % ks == 0), 1)
  U = next((u for u in (8, 8, 4, 2, 1) if (K // KS) % u == 0), 1)
  out = Tensor.empty(KS, R, E, dtype=dtypes.float32, device=x.device, axis=1 if split else None)
  fxn = functools.partial(_skinny_kernel, RB=RB, KC=K // KS, U=U)
  return Tensor.custom_kernel(out, x.float().reshape(R, K).contiguous(), wt, fxn=fxn)[0]


@functools.cache
def _moe_softmax_router_kernel(sel:UOp, probs:UOp, logits:UOp, *table:UOp, k:int, scale:float) -> UOp:
  # One wave per token. Select in O(E*k), rather than ranking every expert against all E others.
  n = cast(int, logits.shape[-1])
  tokens, width = cast(int, logits.numel()) // n, cast(int, table[0].shape[-1]) if table else 1
  assert n % 32 == 0 and 0 < k <= n and k*width <= 32
  logits, sel, probs = logits.reshape(tokens, n), sel.reshape(tokens, k*width), probs.reshape(tokens, k*width)
  token, lane = UOp.range(tokens, 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP)
  es = [(lane + 32*m).cast(dtypes.int32) for m in range(n // 32)]
  scores = [logits[token, e].load().float() for e in es]
  taken = [UOp.const(False, dtypes.bool)] * len(es)
  def better(a:UOp, i:UOp, b:UOp, j:UOp) -> UOp: return (a > b) | (a.eq(b) & (i < j))
  won = []
  for _ in range(k):
    best, index = UOp.const(-math.inf, dtypes.float32), UOp.const(1 << 30, dtypes.int32)
    for e, sc, used in zip(es, scores, taken):
      value, idx = used.where(-math.inf, sc), used.where(1 << 30, e)
      choose = better(value, idx, best, index)
      best, index = choose.where(value, best), choose.where(idx, index)
    for off in (16, 8, 4, 2, 1):
      other, oi = _lane_xor(best, off), _lane_xor(index.bitcast(dtypes.float32), off).bitcast(dtypes.int32)
      choose = better(other, oi, best, index)
      best, index = choose.where(other, best), choose.where(oi, index)
    won.append((index, best))
    taken = [used | e.eq(index) for e, used in zip(es, taken)]
  # pairwise_topk returns the selected experts in ascending score order, with lower expert IDs last on ties.
  slots, slot = won[::-1], lane // width
  expert, score = slots[0]
  for i, (e, sc) in enumerate(slots[1:], 1): expert, score = slot.eq(i).where(e, expert), slot.eq(i).where(sc, score)
  p = ((score - won[0][1]) * LOG2E).exp2()
  total = warp_reduce(((lane < k*width) & (lane % width).eq(0)).where(p, 0), full_wave=True)
  selected = table[0].reshape(-1, width)[expert.cast(dtypes.weakint).maximum(0).minimum(n-1), lane % width].load() if table else expert
  at = lane.valid(lane < k*width)
  return UOp.group(sel[token, at].store(selected), probs[token, at].store(p / total * scale)).end(token, lane).sink(
    arg=KernelInfo(name="moe_softmax_router", opts_to_apply=()))

@functools.cache
def _moe_softmax_router_bitonic_kernel(sel:UOp, probs:UOp, logits:UOp, *table:UOp, k:int, scale:float) -> UOp:
  # Small decode batches need more than one wave per token. Sort (score, -expert) keys with wave shuffles and double-buffered LDS.
  n, width = int(logits.shape[-1]), int(table[0].shape[-1]) if table else 1
  assert n >= 32 and n & (n-1) == 0 and 0 < k <= 32
  tokens = int(logits.numel()) // n
  logits, sel, probs = logits.reshape(tokens, n), sel.reshape(tokens, k*width), probs.reshape(tokens, k*width)
  token, wave, lane = UOp.range(tokens, 0, AxisType.GLOBAL), UOp.range(n//32, 1, AxisType.LOCAL), UOp.range(32, -1, AxisType.WARP)
  idx = wave*32 + lane
  sc = logits[token, idx].load().float()
  bits = sc.eq(0).where(0, sc.bitcast(dtypes.uint32))  # signed zeros compare equal
  ordered = (bits & 0x80000000).ne(0).where(bits ^ 0xffffffff, bits ^ 0x80000000)
  key = (ordered.cast(dtypes.uint64) << 32) | (n-1-idx).cast(dtypes.uint64)
  lds, parity = UOp.alloc((2, n), dtypes.uint64, addrspace=AddrSpace.LOCAL), 0
  for level in range(1, n.bit_length()):
    for j in reversed(range(level)):
      stride = 1 << j
      if stride < 32:
        lo = _lane_xor(key.cast(dtypes.uint32).bitcast(dtypes.float32), stride).bitcast(dtypes.uint32)
        hi = _lane_xor((key >> 32).cast(dtypes.uint32).bitcast(dtypes.float32), stride).bitcast(dtypes.uint32)
        other = lo.cast(dtypes.uint64) | (hi.cast(dtypes.uint64) << 32)
      else:
        lds = lds.after(lds[parity, idx].store(key).barrier())
        other = lds[parity, idx ^ stride].load()
        parity ^= 1
      maximum = (idx & (1 << level)).ne(0) ^ (idx & stride).ne(0)
      key = maximum.eq(key < other).where(other, key)
  ordered = (key >> 32).cast(dtypes.uint32)
  sc = (ordered & 0x80000000).ne(0).where(ordered ^ 0x80000000, ordered ^ 0xffffffff).bitcast(dtypes.float32)
  valid = idx >= n-k  # all selected experts are in the last wave, in pairwise_topk's order
  best = warp_reduce(valid.where(sc, -math.inf), maximum=True, full_wave=True)
  p = ((sc-best)*LOG2E).exp2()
  total = warp_reduce(valid.where(p, 0), full_wave=True)
  expert = (n-1-key.cast(dtypes.uint32)).cast(dtypes.int32)
  stores:list[UOp] = []
  for w in range(width):
    selected = table[0].reshape(-1, width)[expert.cast(dtypes.weakint), w].load() if table else expert
    at = ((idx-(n-k))*width+w).valid(valid)
    stores.extend((sel[token, at].store(selected), probs[token, at].store(p/total*scale)))
  return UOp.group(*stores).end(token, wave, lane).sink(arg=KernelInfo(name="moe_softmax_router_bitonic", opts_to_apply=()))

def moe_softmax_router(logits:Tensor, k:int, scale:float, table:Tensor|None=None) -> tuple[Tensor, Tensor]:
  shape = (*logits.shape[:-1], k)
  axis = logits.uop.axis if token_sharded(logits) else None
  if table is not None: shape, axis = (len(cast(tuple, logits.device)), *logits.shape[:-1], k*int(table.shape[-1])), 0
  sel = Tensor.empty(shape, dtype=dtypes.int32, device=logits.device, axis=axis)
  probs = Tensor.empty(shape, dtype=dtypes.float32, device=logits.device, axis=axis)
  n = int(logits.shape[-1])
  kernel = _moe_softmax_router_bitonic_kernel if isinstance(rows:=prod(logits.shape[:-1]), int) and 0 < rows <= 128 and \
    32 <= n <= 1024 and n & (n-1) == 0 else _moe_softmax_router_kernel
  res = Tensor.custom_kernel(sel, probs, logits.contiguous(), *([table] if table is not None else []),
                             fxn=functools.partial(kernel, k=k, scale=scale))
  return res[0], res[1]


@functools.cache
def _moe_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, sel:UOp, first:UOp, *grids:UOp,
                       out_features:int, in_features:int, ggml_type:int, experts:int, slots:int, x_slots:int) -> UOp:
  # out rows are (token, slot): the slot's expert reads its weight rows, experts of other devices read nothing
  expert_words = out_features * in_features * QUANT_SIZES[ggml_type] // 256 // raw.dtype.itemsize
  def local(row:UOp) -> UOp: return ((expert:=sel[row].load() - first[0].load()) >= 0) & (expert < experts)
  def gate(row:UOp, output:UOp, base:UOp) -> UOp: return base + (sel[row].load() - first[0].load()).cast(dtypes.weakint) * expert_words
  def xtoken(row:UOp) -> UOp: return row if x_slots == slots else row // slots
  return _quant_decode_body(out, raw, xq, xd, xs, grids, out_features, in_features, ggml_type, gate, xtoken, "moe_", active=local)

@functools.cache
def _moe_bundled_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, bexp:UOp, brow:UOp, *grids:UOp,
                        out_features:int, in_features:int, ggml_type:int, slots:int, x_slots:int, M:int) -> UOp:
  # the bundles of moe_bundles: a bundle's M rows (selections) of an expert read its weights once, a row is stored to its selection's row
  expert_words = out_features * in_features * QUANT_SIZES[ggml_type] // 256 // raw.dtype.itemsize
  def gate(token:UOp, output:UOp, base:UOp) -> UOp: return base + bexp[token // M].load().cast(dtypes.weakint).maximum(0) * expert_words
  def xtoken(token:UOp) -> UOp: return brow[token].load().cast(dtypes.weakint).maximum(0) // (1 if x_slots == slots else slots)
  def active(token:UOp) -> UOp: return bexp[token // M].load() >= 0
  def store_row(token:UOp) -> UOp: return brow[token].load().cast(dtypes.weakint)
  return _quant_decode_body(out, raw, xq, xd, xs, grids, out_features, in_features, ggml_type, gate, xtoken, "moe_bundled_", active=active,
                            bundle=(int(bexp.shape[0]), M, store_row))


@functools.cache
def _moe_bundle_atomic_kernel(bexp:UOp, brow:UOp, sel:UOp, first:UOp, experts:int, M:int) -> UOp:
  """
  the bundles with a thread a selection: its rank in its expert by an atomic add of the expert's count (the order of the rows of an expert
  is the order the adds came in: a row's output doesn't depend on its bundle), the bundles before every expert by a scan of the threads' counts
  (a thread the experts tid*per..), then every selection to its place
  """
  N, THREADS = cast(int, sel.shape[0]), 256
  per = (experts + THREADS - 1) // THREADS
  # (the lanes of a wave: hardware lanes, the scans permute across them)
  wave, lane = UOp.range(THREADS // WARP_SIZE, 0, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  tid = wave * WARP_SIZE + lane
  zero = UOp.const(0, dtypes.int32)
  cnt = UOp.alloc((per*THREADS,), dtypes.int32, addrspace=AddrSpace.LOCAL)
  cnt = cnt.after(UOp.group(*(cnt[tid*per + m].store(zero) for m in range(per))).barrier())
  # the ranks (in LDS)
  Q = (N + THREADS - 1) // THREADS
  rank = UOp.alloc((Q*THREADS,), dtypes.int32, addrspace=AddrSpace.LOCAL)
  def expert(i:UOp) -> tuple[UOp, UOp]:
    e = (sel[i.minimum(N-1)].load() - first[0].load()).cast(dtypes.weakint)
    return e.maximum(0).minimum(experts-1), (i < N) & (e >= 0) & (e < experts)
  adds = []
  for q in range(Q):
    e, ok = expert(i:=tid + THREADS*q)
    r = _sys("__hip_atomic_fetch_add({0}, {1}, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP)", dtypes.int32, cnt[e], ok.cast(dtypes.int32))
    adds.append(rank[i].store(r))
  rank, cnt = rank.after(UOp.group(*adds).barrier()), cnt.after(UOp.group(*adds).barrier())
  # the bundles of the thread's experts, their exclusive scan over the threads (the waves', then the waves' totals)
  nbs = [(cnt[tid*per + m].load() + (M-1)) // M for m in range(per)]
  local = [functools.reduce(lambda a, b: a + b, nbs[:m], zero) for m in range(per)]
  total = functools.reduce(lambda a, b: a + b, nbs, zero)
  incl = _wave_scan(total, lane)
  wt = UOp.alloc((THREADS // WARP_SIZE,), dtypes.int32, addrspace=AddrSpace.LOCAL)
  wt = wt.after(wt[wave.valid(lane.eq(WARP_SIZE-1))].store(incl).barrier())
  woff = functools.reduce(lambda a, b: a + b, [(w < wave).where(wt[w].load(), zero) for w in range(THREADS // WARP_SIZE)], zero)
  start = UOp.alloc((per*THREADS,), dtypes.int32, addrspace=AddrSpace.LOCAL)
  start = start.after(UOp.group(*(start[tid*per + m].store(woff + incl - total + local[m]) for m in range(per))).barrier())
  # every selection to its place, the first of a bundle its expert
  stores = []
  for q in range(Q):
    e, ok = expert(i:=tid + THREADS*q)
    rk, st = rank[i].load().cast(dtypes.weakint), start[e].load().cast(dtypes.weakint)
    stores += [brow[(st*M + rk).valid(ok)].store(i.cast(dtypes.int32)), bexp[(st + rk // M).valid(ok & (rk % M).eq(0))].store(e.cast(dtypes.int32))]
  return UOp.group(*stores).end(wave, lane).sink(arg=KernelInfo(name="moe_bundle", opts_to_apply=()))

@functools.cache
def _moe_bundled_wmma_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, bexp:UOp, brow:UOp, *grids:UOp, out_features:int, in_features:int,
                             ggml_type:int, slots:int, x_slots:int, waves:int=2, oloop:int=1) -> UOp:
  # (RDNA4) the bundles of moe_bundles (16 rows of an expert each) on the int8 tensor cores: a workgroup a bundle and oloop blocks of 16*waves outputs
  # (a wave 16 of a block) in a loop. per block of 256 inputs the threads decode the outputs' weights (a thread a group's 8 words: the loads of a
  # row's block are coalesced) and their scales to LDS once for the bundle's rows, then a wave's fragments are its outputs' words (as
  # _quant_linear_i8_wmma_kernel's), the rows' q8 activations the other fragment. a row's output is stored to its selection's row (brow, -1:
  # none)
  NB, OB, threads = cast(int, bexp.shape[0]), 16*waves, 32*waves
  RS = GGML_BLOCK_SIZE // 4 + 1  # a row's words in LDS (+1: the 16 rows of a fragment's lanes on different banks)
  bundle, output_block = UOp.range(NB, 0, AxisType.GLOBAL), UOp.range(out_features // (OB*oloop), 1, AxisType.GLOBAL)
  lane, wave = UOp.range(WARP_SIZE, -1, axis_type=AxisType.WARP), UOp.range(waves, 3, axis_type=AxisType.LOCAL)
  tid, col, ph = wave*32 + lane, lane % 16, lane // 16
  grid = None
  if grids:
    grid = UOp.alloc((int(grids[0].numel()),), dtypes.uint32, addrspace=AddrSpace.LOCAL)
    grid = grid.after(*(grid[tid+i*threads].store(grids[0][tid+i*threads]) for i in range(int(grid.numel())//threads)))
  type_words = QUANT_SIZES[ggml_type]//raw.dtype.itemsize
  output_words, n_kb = in_features * type_words // GGML_BLOCK_SIZE, in_features // GGML_BLOCK_SIZE
  expert = bexp[bundle].load()
  ob = UOp.range((expert >= 0).where(oloop, 0), 5, AxisType.LOOP)
  kb = UOp.range(n_kb, 4, AxisType.LOOP)
  expert_base = expert.maximum(0).cast(dtypes.weakint) * (out_features * output_words)
  first_out = (output_block*oloop + ob)*OB
  # the block's weights and scales to LDS: a thread a (row, group), its 8 words and its scale (8 threads a row's block)
  lds_w = UOp.alloc((OB*RS,), dtypes.uint32, addrspace=AddrSpace.LOCAL)
  lds_s = UOp.alloc((OB*8,), dtypes.float32, addrspace=AddrSpace.LOCAL)
  st = []
  for k in range(OB*8 // threads):
    row, sg = (e:=tid + k*threads) // 8, (tid + k*threads) % 8
    base = expert_base + (first_out + row)*output_words + kb*type_words
    st += [lds_w[row*RS + sg*8 + i].store(w) for i, w in enumerate(_quant_group_words(raw, base, sg, ggml_type, grid))]
    st.append(lds_s[e].store(_wmma_scale(raw, base, sg, 0, ggml_type)[0]))
  staged = UOp.group(*st).barrier()
  lds_w, lds_s = lds_w.after(staged), lds_s.after(staged)
  def srow(t:UOp) -> UOp: return brow[bundle*16 + t].load().cast(dtypes.weakint)
  def xrow(t:UOp) -> UOp: return srow(t).maximum(0) // (1 if x_slots == slots else slots)
  a_row, s_rows = xrow(col), [xrow(ph*8 + j) for j in range(8)]
  acc = UOp.alloc((8,), dtypes.float32, addrspace=AddrSpace.REG)
  acc = acc.after(acc.after(ob).store(acc.const_like(0)))
  vals = [acc.after(kb)[j].load() for j in range(8)]
  orow = wave*16 + col
  for sg in range(8):
    g = kb*8 + sg
    afrags = [_i8_frag((xq[a_row, g, half*4+ph].load(), xq[a_row, g, half*4+2+ph].load())) for half in (0, 1)]
    bfrags = [_i8_frag(tuple(lds_w[orow*RS + sg*8 + half*4 + i].load() for i in (ph, ph+2))) for half in (0, 1)]
    dot_i = UOp.wmma(afrags[1], bfrags[1], UOp.wmma(afrags[0], bfrags[0], UOp.stack(*[UOp.const(0, dtypes.int32)]*8), *WMMA_ARG), *WMMA_ARG)
    dot = UOp.alloc((8,), dtypes.int32, addrspace=AddrSpace.REG)
    dot = dot.after(dot.store(dot_i))
    scale = lds_s[orow*8 + sg].load()
    vals = [vals[j] + xd[s_rows[j], g].load() * (scale * dot[j].load().float()) for j in range(8)]
  # (a barrier after the block's reads: the next block's stage doesn't overwrite what a slower wave still reads)
  update = acc.store(UOp.stack(*vals)).barrier().end(kb)
  output = first_out + orow
  stores = [out[(r:=srow(ph*8 + j)).maximum(0), output.valid(r >= 0)].store(acc.after(update)[j].load().cast(out.dtype)) for j in range(8)]
  weights = NB*16*out_features*in_features
  est = Estimates(2*weights, weights*type_words*2//16//GGML_BLOCK_SIZE, 0)  # (an upper bound: every bundle used)
  return UOp.group(*stores).end(ob).end(bundle, output_block, lane, wave).sink(arg=KernelInfo(
    name=f"moe_bundled_{QUANT_NAMES[ggml_type]}_wmma", opts_to_apply=(), estimates=est))

def moe_bundles(sel:Tensor, first:Tensor, experts:int, M:int, N:int) -> tuple[Tensor, Tensor]:
  """the N selections sel of every device in bundles of at most M rows of an expert (_moe_bundle_kernel): (bexp, brow) of every device"""
  NB = (N + M - 1) // M + min(experts, N)
  devs = sel.device
  def filled(n:int) -> Tensor:
    shape, axis = ((len(devs)*n,), 0) if isinstance(devs, tuple) else ((n,), None)
    return Tensor.empty(*shape, dtype=dtypes.int32, device=devs, axis=axis).full_like(-1).contiguous()
  kernel = _moe_bundle_atomic_kernel
  bexp, brow = Tensor.custom_kernel(filled(NB), filled(NB*M), sel.cast(dtypes.int32).reshape(-1).contiguous(), first,
                                    fxn=functools.partial(kernel, experts=experts, M=M))[:2]
  return bexp, brow

def moe_linear(weight:Tensor, ggml_type:int, experts:int, x:Tensor, sel:Tensor, first:Tensor, out_dtype:DType=dtypes.float32) -> Tensor:
  """
  weight: packed (experts, out, in) local to each device, x: (..., x_slots, in), sel: (B, T, slots) expert ids, first: the first expert of a device.
  returns (..., slots, out). on devices holding some of the experts the output is (devices, B, T, slots, out), every device has its own partial
  """
  B, T, slots = sel.shape[-3:]  # sel can have a leading device axis: the selections of every device
  in_features, x_slots, tokens = int(x.uop.shard_shape[-1]), int(x.shape[-2]), int(B*T)
  out_features = weight.uop.max_shard_shape[0] * weight.dtype.itemsize * 256 // QUANT_SIZES[ggml_type] // in_features // experts
  devices = len(weight.device) if isinstance(weight.device, tuple) else 1
  extra = (_iq_grid(x.device, ggml_type),) if ggml_type in (IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S) else ()
  xq = q8_quantize(x, tokens*x_slots, in_features)
  # (RDNA4: bundles of 16 rows on the int8 tensor cores)
  wmma = _wmma_rdna4(x.device) and ggml_type in MOE_WMMA_QUANTS and tokens >= MOE_WMMA_MIN and out_features % 32 == 0 and \
    in_features % GGML_BLOCK_SIZE == 0
  if tokens >= MOE_BUNDLE_MIN:
    # many rows (batched decoding): the rows of an expert in bundles of M, a bundle reads the expert's weights once
    # (the gate and up rows' bundles are the down rows' too: the same selections)
    M = 16 if wmma else MOE_BUNDLE
    key = (sel.uop, first.uop, experts, M)
    if (bundles:=_bundle_cache.get(key)) is None:
      bundles = _bundle_cache[key] = moe_bundles(sel, first, experts, M, tokens*int(slots))
    bexp, brow = bundles
    out = Tensor.empty(devices*tokens*slots, out_features, dtype=out_dtype, device=x.device, axis=0 if devices > 1 else None)
    out = out.zeros_like().contiguous()
    fxn = functools.partial(_moe_bundled_wmma_kernel, out_features=out_features, in_features=in_features, ggml_type=ggml_type, slots=int(slots),
                            x_slots=x_slots, oloop=16 if in_features == GGML_BLOCK_SIZE else 1) if wmma else \
          functools.partial(_moe_bundled_kernel, out_features=out_features, in_features=in_features, ggml_type=ggml_type, slots=int(slots),
                            x_slots=x_slots, M=M)
    result = Tensor.custom_kernel(out, weight, *xq, bexp, brow, *extra, fxn=fxn)[0]
    return result.reshape(*((devices,) if devices > 1 else ()), B, T, slots, out_features)
  # a grid over the (token, slot) rows
  out = Tensor.empty((devices*tokens*slots, out_features), dtype=dtypes.float32, device=x.device, axis=0 if devices > 1 else None)
  fxn = functools.partial(_moe_decode_kernel, out_features=out_features, in_features=in_features, ggml_type=ggml_type, experts=experts,
                          slots=slots, x_slots=x_slots)
  result = Tensor.custom_kernel(out, weight, *xq, sel.cast(dtypes.int32).flatten(), first, *extra, fxn=fxn)[0]
  return result.reshape(*((devices,) if devices > 1 else ()), B, T, slots, out_features)

_bundle_cache:dict[tuple, tuple[Tensor, Tensor]] = {}
# batched decoding of at least MOE_BUNDLE_MIN tokens: the rows of an expert in bundles of MOE_BUNDLE that share its weight loads
MOE_BUNDLE, MOE_BUNDLE_MIN = 4, 40
# (RDNA4: the bundles of at least MOE_WMMA_MIN tokens on the int8 tensor cores, of these formats)
MOE_WMMA_MIN, MOE_WMMA_QUANTS = 64, (IQ3_S, IQ4_XS, IQ4_NL)
MOE_TILE = 32  # grouped experts: rows of an expert are padded to whole tiles, a workgroup decodes an expert once per tile
def _moe_tile(group:tuple[UOp, ...]) -> int:
  """the rows of a tile of grouped experts (moe_route): its rows (entries) over its tiles"""
  return int(group[0].shape[0]) // int(group[1].shape[0])

def moe_grouped_linear(weight:Tensor, ggml_type:int, experts:int, x:Tensor, entries:Tensor, tiles:Tensor, slots:int=0, selections:int=0,
                       act:tuple[float, float, float]|None=None, out_dtype=dtypes.float32, sorted_out:bool=False) -> Tensor:
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
  rows_out, cols = R if slots or sorted_out else selections, out_features // 2 if act else out_features
  x_div = slots or (-1 if sorted_out else 0)  # sorted_out: x and the output have a row per sorted row
  out = Tensor.empty((devices*rows_out, cols), dtype=dtypes.half if act else out_dtype, device=x.device, axis=0 if devices > 1 else None)
  rdna4 = _wmma_rdna4(x.device)
  # (IQ4_XS too: as fast as its f16 kernel for the experts' down rows, which read their activations exactly)
  # (RDNA3 IQ4_XS: K256 too, its f16 kernel ~4x the time of an IQ3_S layer's. RDNA4: IQ3_S only)
  if ggml_type in ((IQ3_S,) if rdna4 else (IQ3_S, IQ4_XS)) and in_features % 256 == 0 and out_features % 256 == 0:
    # activations with a scale per 256 inputs (q8_K, as llama.cpp's dots of the IQ formats): the sub-blocks accumulate in int32
    x_rows = int(x.uop.shard_shape[0]) if x.ndim == 2 else prod(x.uop.shard_shape[:-1])
    xq, xd = q8k_quantize(x.contiguous(), int(x_rows), in_features)
    fxn = functools.partial(_grouped_k256_kernel, out_features=out_features, in_features=in_features, ggml_type=ggml_type,
                            expert_words=expert_words, x_div=x_div, act=act, rdna4=rdna4)
    result = Tensor.custom_kernel(out, weight, xq, xd, *extra, *group, fxn=fxn)[0]
    return result.reshape(devices, rows_out, cols) if devices > 1 else result
  # (the int8 tensor cores: the grid formats, IQ4_NL and Q2_0)
  if ggml_type in I8_WMMA_QUANTS or ggml_type in (IQ4_NL, Q2_0):
    x_rows = int(x.uop.shard_shape[0]) if x.ndim == 2 else prod(x.uop.shard_shape[:-1])
    fxn = functools.partial(_quant_linear_i8_wmma_kernel, ggml_type=ggml_type, out_features=out_features, in_features=in_features,
                            expert_words=expert_words, x_div=x_div, act=act, rdna4=rdna4)
    result = Tensor.custom_kernel(out, weight, *q8_quantize(x.contiguous(), int(x_rows), in_features)[:3], *extra, *group, fxn=fxn)[0]
    return result.reshape(devices, rows_out, cols) if devices > 1 else result
  if ggml_type == IQ4_XS: fxn, extra = _iq4_linear_f16_wmma_kernel, (iq4_half_lut(x.device),)
  else: fxn = functools.partial(_q5_linear_f16_wmma_kernel if ggml_type in (Q4_K, Q5_K) else _quant_linear_f16_wmma_kernel, ggml_type=ggml_type)
  fxn = functools.partial(fxn, out_features=out_features, in_features=in_features, expert_words=expert_words,
                          x_div=x_div, act=act, rdna4=rdna4)
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

# ******** moe_route of many tokens: a counting sort in three kernels (the generic one-hot and cumsum kernels: ~0.3 ms a layer) ********

ROUTE_G, ROUTE_T = 64, 256  # workgroups, threads

@functools.cache
def _route_init_kernel(counts:UOp, entries:UOp, tiles:UOp) -> UOp:
  # counts 0, the rows -1, the tiles -1
  n = max(int(prod(counts.shape)), int(prod(entries.shape)), int(prod(tiles.shape)))
  i = UOp.range(n, 0, AxisType.GLOBAL)
  st = [counts.reshape(-1)[i.valid(i < int(prod(counts.shape)))].store(UOp.const(0, dtypes.int32)),
        entries.reshape(-1)[i.valid(i < int(prod(entries.shape)))].store(UOp.const(-1, dtypes.int32)),
        tiles.reshape(-1)[i.valid(i < int(prod(tiles.shape)))].store(UOp.const(-1, dtypes.int32))]
  return UOp.group(*st).end(i).sink(arg=KernelInfo(name="route_init", opts_to_apply=(), estimates=Estimates()))

def _route_expert(sel:UOp, first:UOp, i:UOp, N:int, experts:int) -> tuple[UOp, UOp]:
  e = (sel.reshape(-1)[i.minimum(N-1)].load() - first.reshape(-1)[0].load()).cast(dtypes.weakint)
  return e.maximum(0).minimum(experts-1), (i < N) & (e >= 0) & (e < experts)

@functools.cache
def _route_count_kernel(rank:UOp, counts:UOp, sel:UOp, first:UOp, *, experts:int) -> UOp:
  # the rank of every selection among its expert's (a workgroup's in LDS, its workgroup's base from the global counts)
  N = int(prod(rank.shape))
  Q = (N + ROUTE_G*ROUTE_T - 1) // (ROUTE_G*ROUTE_T)
  per = (experts + ROUTE_T - 1) // ROUTE_T
  wg, t = UOp.range(ROUTE_G, 0, AxisType.GLOBAL), UOp.range(ROUTE_T, 1, AxisType.LOCAL)
  hist = UOp.alloc((per*ROUTE_T,), dtypes.int32, addrspace=AddrSpace.LOCAL)
  hist = hist.after(UOp.group(*(hist[t*per + m].store(UOp.const(0, dtypes.int32)) for m in range(per))).barrier())
  locs = []
  for q in range(Q):
    e, ok = _route_expert(sel, first, i:=(wg*Q + q)*ROUTE_T + t, N, experts)
    locs.append(_sys("__hip_atomic_fetch_add({0}, {1}, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP)", dtypes.int32, hist[e],
                     ok.cast(dtypes.int32)))
  hist = hist.after(UOp.group(*locs).barrier())
  base = UOp.alloc((per*ROUTE_T,), dtypes.int32, addrspace=AddrSpace.LOCAL)
  bst = [base[t*per + m].store(_sys("__hip_atomic_fetch_add({0}, {1}, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)", dtypes.int32,
                                    counts.reshape(-1)[(t*per + m).minimum(experts-1)], (t*per + m < experts).where(hist[t*per + m].load(), 0)))
         for m in range(per)]
  base = base.after(UOp.group(*bst).barrier())
  stores = []
  for q, loc in enumerate(locs):
    e, ok = _route_expert(sel, first, i:=(wg*Q + q)*ROUTE_T + t, N, experts)
    stores.append(rank.reshape(-1)[i.valid(i < N)].store(ok.where(base[e].load() + loc, -1)))
  return UOp.group(*stores).end(wg, t).sink(arg=KernelInfo(name="route_count", opts_to_apply=(), estimates=Estimates()))

@functools.cache
def _route_place_kernel(pos:UOp, entries:UOp, tiles:UOp, rank:UOp, counts:UOp, sel:UOp, first:UOp, *, experts:int, tile:int) -> UOp:
  # the offsets of the experts' tiles (every workgroup scans the padded counts), every selection its row, the first workgroup the tiles
  N = int(prod(pos.shape))
  Q = (N + ROUTE_G*ROUTE_T - 1) // (ROUTE_G*ROUTE_T)
  per = (experts + ROUTE_T - 1) // ROUTE_T
  wg = UOp.range(ROUTE_G, 0, AxisType.GLOBAL)
  wave, lane = UOp.range(ROUTE_T // WARP_SIZE, 1, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  t, zero = wave * WARP_SIZE + lane, UOp.const(0, dtypes.int32)
  pads = [(t*per + m < experts).where((counts.reshape(-1)[(t*per + m).minimum(experts-1)].load() + (tile-1)) // tile * tile, zero)
          for m in range(per)]
  local = [functools.reduce(lambda a, b: a + b, pads[:m], zero) for m in range(per)]
  total = functools.reduce(lambda a, b: a + b, pads, zero)
  incl = _wave_scan(total, lane)
  wt = UOp.alloc((ROUTE_T // WARP_SIZE,), dtypes.int32, addrspace=AddrSpace.LOCAL)
  wt = wt.after(wt[wave.valid(lane.eq(WARP_SIZE-1))].store(incl).barrier())
  woff = functools.reduce(lambda a, b: a + b, [(w < wave).where(wt[w].load(), zero) for w in range(ROUTE_T // WARP_SIZE)], zero)
  off = UOp.alloc((per*ROUTE_T,), dtypes.int32, addrspace=AddrSpace.LOCAL)
  offs = [woff + incl - total + local[m] for m in range(per)]
  off = off.after(UOp.group(*(off[t*per + m].store(offs[m]) for m in range(per))).barrier())
  stores = []
  for q in range(Q):
    e, ok = _route_expert(sel, first, i:=(wg*Q + q)*ROUTE_T + t, N, experts)
    p = off[e].load() + rank.reshape(-1)[i.minimum(N-1)].load()
    stores += [pos.reshape(-1)[i.valid(i < N)].store(ok.where(p, -1)),
               entries.reshape(-1)[p.cast(dtypes.weakint).valid(ok)].store(i.cast(dtypes.int32))]
  # (the first workgroup: an expert's tiles its index)
  for m in range(per):
    x = t*per + m
    nt = UOp.range(((x < experts) & wg.eq(0)).where(pads[m] // tile, 0), 2 + m, AxisType.LOOP)
    stores.append(UOp.group(tiles.reshape(-1)[(offs[m] // tile + nt).cast(dtypes.weakint)].store(x.cast(dtypes.int32))).end(nt))
  return UOp.group(*stores).end(wg, wave, lane).sink(arg=KernelInfo(name="route_place", opts_to_apply=(), estimates=Estimates()))

def _moe_route_sort(sel:Tensor, first:Tensor, experts:int, tile:int) -> tuple[Tensor, Tensor, Tensor]|None:
  """moe_route's entries, tiles and pos of many tokens by the counting sort (None: not those shapes)"""
  if not amd_custom_kernels_supported(sel.device) or experts > 2048: return None
  N = int(prod(sel.shape[-3:]))
  R = round_up(N + (tile-1)*experts, tile)
  devs = first.device
  if isinstance(devs, tuple) and first.uop.axis != 0: return None
  n = len(devs) if isinstance(devs, tuple) else 1
  axis = 0 if isinstance(devs, tuple) else None
  counts = Tensor.empty(n, experts, dtype=dtypes.int32, device=devs, axis=axis)
  entries = Tensor.empty(n, R, dtype=dtypes.int32, device=devs, axis=axis)
  tiles = Tensor.empty(n, R // tile, dtype=dtypes.int32, device=devs, axis=axis)
  counts, entries, tiles = Tensor.custom_kernel(counts, entries, tiles, fxn=_route_init_kernel)
  sel_f = sel.reshape(-1).cast(dtypes.int32).contiguous()
  rank = Tensor.empty(n, N, dtype=dtypes.int32, device=devs, axis=axis)
  rank, counts = Tensor.custom_kernel(rank, counts, sel_f, first.cast(dtypes.int32), fxn=functools.partial(_route_count_kernel, experts=experts))[:2]
  pos = Tensor.empty(n, N, dtype=dtypes.int32, device=devs, axis=axis)
  pos, entries, tiles = Tensor.custom_kernel(pos, entries, tiles, rank, counts, sel_f, first.cast(dtypes.int32),
                                             fxn=functools.partial(_route_place_kernel, experts=experts, tile=tile))[:3]
  return entries, tiles, pos

def moe_route(sel:Tensor, first:Tensor, experts:int, tile:int=MOE_TILE) -> tuple[Tensor, Tensor, Tensor, Tensor]:
  """
  sorts the (token, slot) selections of sel (B, T, k) by local expert, each expert's rows padded to whole tiles.
  returns valid (..., B*T*k) if a selection is an expert of this device, entries (..., R) the selection of each row (-1: padding)
  tiles (..., R/tile) the local expert of each tile (-1: none) and pos (..., B*T*k) the row of each selection (-1: not local). a few tokens
  (batched decoding) have few rows an expert: small tiles (an expert's weights decoded once for its rows, few padding rows computed)
  first: the first expert of each device, a leading device axis if sharded
  """
  k, N = sel.shape[-1], int(prod(sel.shape[-3:]))  # sel can have a leading device axis
  tokens, R = N // k, round_up(N + (tile-1)*experts, tile)
  # (many tokens: the counting sort, valid not made (its callers don't use it))
  # (sel: (B, T, k) the same on every device, or (devices, B, T, k) split on the devices)
  if tokens >= 256 and (sel.ndim == 3 or (sel.ndim == 4 and isinstance(sel.device, tuple) and sel.uop.axis == 0)) and \
     (r:=_moe_route_sort(sel, first, experts, tile)) is not None:
    return Tensor.zeros(1, dtype=dtypes.bool), *r
  e = sel.reshape(-1, tokens, k).cast(dtypes.int32) - first.reshape(-1, 1, 1)
  valid = (e >= 0) & (e < experts)
  onehot = (e.unsqueeze(-1) == Tensor.arange(experts, dtype=dtypes.int32)) & valid.unsqueeze(-1)  # (n, tokens, k, experts)
  # a token selects an expert at most once: the rank of a selection is the count of earlier tokens of its expert
  per_token = onehot.sum(2, dtype=dtypes.int32)  # (n, tokens, experts)
  earlier = exclusive_cumsum(per_token) if tokens % 64 == 0 and amd_custom_kernels_supported(sel.device) else per_token.cumsum(1) - per_token
  padded = (per_token.sum(1) + tile-1) // tile * tile
  offsets = padded.cumsum(-1) - padded
  pos = valid.where(onehot.where((offsets.unsqueeze(1) + earlier).unsqueeze(2), 0).sum(-1), -1).reshape(-1, N)
  # the selection of every sorted row: a scatter of the positions into rows of -1 (padding)
  n = pos.shape[0]
  entries = Tensor.full((n, R), -1, dtype=dtypes.int32)
  entries = (entries.shard(first.device, 0) if isinstance(first.device, tuple) else entries.to(first.device)).contiguous()
  entries = Tensor.custom_kernel(entries.reshape(-1), pos.cast(dtypes.int32).reshape(-1).contiguous(), fxn=_scatter_rows_kernel)[0].reshape(n, R)
  start = Tensor.arange(R // tile, dtype=dtypes.int32) * tile
  inside = (offsets.unsqueeze(1) <= start.unsqueeze(-1)) & (start.unsqueeze(-1) < (offsets + padded).unsqueeze(1))
  tiles = inside.where(Tensor.arange(experts, dtype=dtypes.int32) + 1, 0).sum(-1) - 1
  return valid.reshape(-1, N), entries, tiles.cast(dtypes.int32), pos.cast(dtypes.int32)

@functools.cache
def _moe_combine_kernel(y:UOp, down:UOp, pos:UOp, probs:UOp, *add:UOp, k:int, by_selection:bool=False, row0:int=0,
                        add_rows:tuple[int, ...]=(), add_scale:bool=False) -> UOp:
  # (add_scale: the last src the scale (T,) of the first add's rows (a gated shared expert's))
  scale_src, add = (add[-1], add[:-1]) if add_scale else (None, add)
  # y[t] = the sum over the slots j of token t of probs[t, j] * down[pos[t, j]] (the sorted rows): a workgroup per token, a thread its columns.
  # the slots of other devices' experts (pos -1) are skipped by the whole workgroup. the tokens from row0 (of the inputs), each add from its row
  T, D = cast(tuple[int, int], y.shape)
  threads = 128 if D == 2560 else 256
  per = D // threads
  t, th, j = UOp.range(T, 0, AxisType.GLOBAL), UOp.range(threads, 1, AxisType.LOCAL), UOp.range(k, 2, AxisType.LOOP)
  p, w = pos[(t + row0) * k + j].load(), probs[(t + row0) * k + j].load()
  if by_selection: p = (p >= 0).where((t + row0) * k + j, -1)  # down has a row per selection
  accs = UOp.alloc((per,), dtypes.float32, addrspace=AddrSpace.REG)
  accs = accs.after(accs.store(UOp.stack(*[UOp.const(0, dtypes.float)]*per)))
  vals = [(p >= 0).where(down[p.valid(p >= 0), th * per + c].load().float() * w, UOp.const(0, dtypes.float)) for c in range(per)]
  step = accs.store(UOp.stack(*[accs.after(j)[c].load() + v for c, v in enumerate(vals)]))
  if D == 2560:
    # Aligned eight-byte loads per lane, pipelined across the independently gathered expert rows.
    step, prefetch = _prefetched(step, j, k-1, (down, pos, probs))
    step = UOp.group(step, *prefetch)
  accs = accs.after(step.end(j))
  def plus(c:int) -> UOp:
    def term(i:int, a:UOp) -> UOp:
      v = a[t + (add_rows[i] if add_rows else 0), th * per + c].load().float()
      return v * scale_src.reshape(-1)[t + row0].load().float() if scale_src is not None and i == 0 else v
    return functools.reduce(lambda a, b: a + b, [term(i, a) for i, a in enumerate(add)], UOp.const(0, dtypes.float))
  return UOp.group(*[y[t, th * per + c].store((accs[c].load() + plus(c)).cast(y.dtype)) for c in range(per)]).end(t, th).sink(
    arg=KernelInfo(name="moe_combine", opts_to_apply=()))

def moe_combine(down:Tensor, pos:Tensor, probs:Tensor, tokens:int, k:int, add:Tensor|None=None, out_dtype=dtypes.float32,
                by_selection:bool=False, add_scale:Tensor|None=None) -> Tensor:
  """
  down (..., R, D) the outputs of the sorted rows, pos (..., tokens*k) the row of every selection (-1: none), probs (tokens*k) their weights:
  (..., tokens, D) the weighted sums of every token's rows (plus add, of that shape), summed in float. a leading device axis (split) is kept
  """
  devices = len(down.device) if isinstance(down.device, tuple) and down.uop.axis == 0 else 1
  D = int(down.shape[-1])
  y = Tensor.empty(devices*tokens, D, dtype=out_dtype, device=down.device, axis=0 if devices > 1 else None)
  extra:tuple[Tensor, ...] = (add.reshape(-1, D).contiguous(),) if add is not None else ()
  if add is not None and add_scale is not None: extra += (add_scale.float().reshape(-1).contiguous(),)
  y = Tensor.custom_kernel(y, down.reshape(-1, D).contiguous(), pos.reshape(-1).contiguous(), probs.float().reshape(-1).contiguous(), *extra,
                           fxn=functools.partial(_moe_combine_kernel, k=k, by_selection=by_selection,
                                                 add_scale=add is not None and add_scale is not None))[0]
  return y.reshape(devices, tokens, D) if devices > 1 else y


# ******** manifold-constrained hyper-connections: the mixing coefficients ********


@functools.cache
def _weighted_sum_kernel(out:UOp, x:UOp, p:UOp) -> UOp:
  # out (R, D) = the sum over S of x (R, S, D) times p (R, S), in float in the order of S: a thread 8 neighboring outputs of a row
  R, S, D = cast(tuple[int, int, int], x.shape)
  g, t = UOp.range(R * D // 8 // 256, 0, AxisType.GLOBAL), UOp.range(256, 1, AxisType.LOCAL)
  e = g * 256 + t
  r, d0 = e // (D // 8), (e % (D // 8)) * 8
  ps = [p[r, s].load().float() for s in range(S)]
  vals = [functools.reduce(lambda a, b: a + b, [x[r, s, d0 + i].load().float() * ps[s] for s in range(S)]) for i in range(8)]
  return UOp.group(*[out[r, d0 + i].store(v) for i, v in enumerate(vals)]).end(g, t).sink(arg=KernelInfo(name="weighted_sum", opts_to_apply=()))

def weighted_sum(x:Tensor, p:Tensor) -> Tensor|None:
  """
  (x * p.unsqueeze(-1)).sum(-2) of x (..., S, D) (the experts' rows of a token) and p (..., S) in float: a kernel of the rows of decoding (the
  generic reduction is ~2.5x slower). a leading device axis (partials) stays. None: not those shapes
  """
  *lead, S, D = x.shape
  # (a leading device axis of partials, or none: not rows split over the devices (sequence parallel))
  dev_axis = isinstance(x.device, tuple) and x.uop.axis == 0
  # (p of every row on every device (a sequence parallel step's gathered selections): its rows are each device's partials' too)
  p_all = dev_axis and tuple(p.shape) == tuple(x.shape[1:-1]) and isinstance(p.device, tuple) and p.uop.axis is None
  if isinstance(x.device, tuple) and x.uop.axis not in (0, None) or (tuple(p.shape) != tuple(x.shape[:-1]) and not p_all): return None
  R = int(prod(lead[1:] if dev_axis else lead))
  if not all(isinstance(v, int) for v in (*lead, S, D)) or int(D) % 8 or (R * int(D) // 8) % 256 or not amd_custom_kernels_supported(x.device) or \
     not getenv("WEIGHTED_SUM", 1): return None
  n = len(cast(tuple, x.device)) if dev_axis else 1
  out = Tensor.empty(n*R, int(D), dtype=dtypes.float32, device=x.device, axis=0 if dev_axis else None)
  res = Tensor.custom_kernel(out, x.reshape(n*R, int(S), int(D)), p.reshape(R if p_all else n*R, int(S)), fxn=_weighted_sum_kernel)[0]
  return res.reshape(*lead, int(D))


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
  # (the weight split on its outputs: every device's of its own, the output split so)
  split = weight.uop.axis == 0 and isinstance(weight.device, tuple)
  local = layer.out_features // len(cast(tuple, weight.device)) if split else layer.out_features
  out = Tensor.empty(tokens, layer.out_features, dtype=dtypes.float32, device=x.device, axis=1 if split else None)
  fxn = functools.partial(_amd_f16_gemv_kernel, in_features=layer.in_features, out_features=local, tokens=tokens)
  srcs = (out, weight.reshape(-1), x.reshape(tokens, layer.in_features)) + (() if layer.bias is None else (_view_back(layer.bias),))
  return Tensor.custom_kernel(*srcs, fxn=fxn)[0].reshape(*x.shape[:-1], layer.out_features)

# ******** flash attention on the KV cache ********

def _vec_load(ptr:UOp, lanes:int) -> tuple[UOp, ...]:
  if lanes == 1: return (ptr.load().float(),)
  vec = _amd_load(ptr, lanes)
  return tuple(vec[i].float() for i in range(lanes))


@functools.cache
def _amd_flash_decode_combine(o:UOp, partial:UOp, stats:UOp, *lse:UOp, live:int|UOp) -> UOp:
  # (lse: the log of the sum of the exps of the scores of every (row, head) too, (B, H) (-1e30 without keys: an output of zeros), see
  # qsa_decode_attention's split)
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
  ws = weight_sum[0].load()
  inv = (ws > 0).where(1 / ws.maximum(1e-30), UOp.const(0.0, dtypes.float)) if lse else 1 / ws
  stores = [o[b, h, 0, d].store(acc[i].load() * inv) for i, d in enumerate(dims)]
  if lse:
    ls = (ws > 0).where(chunk_max + ws.maximum(1e-30).log2() * (1 / LOG2E), -1e30)
    stores.append(lse[0][b, h.valid(lane.eq(0) & block_dt.eq(0))].store(ls))
  return UOp.group(*stores).end(lane, block_dt, block_bh).sink(arg=KernelInfo(name="flash_decode_combine", opts_to_apply=()))

@functools.cache
def _qsa_decode_kernel(out:UOp, stats:UOp, q:UOp, cache:UOp, index:UOp, bases:UOp, *, scale:float, W:int, kpool:int=0, bits:int=8) -> UOp:
  # the partials (B, H, P, DV) and their (max, sum) (B, H, P, 2) of every row b's heads (q (B, H, DK): the GQA group of a KV head) on the keys
  # of its list index[b] (B, n+1, the count last), the rows of the cache (.., N, row) from bases[b]: a row's first DK dims the key, the next DV
  # the value (half, or kv_q8_pack's words). a workgroup (b, part) of W waves, a wave a tile of 32 keys a round: a lane a key, its scores of
  # all the heads (the queries in LDS: no reduction over the lanes a key), then the tile's values into the lanes' dims of the output
  B, H, P, DV = cast(tuple[int, int, int, int], out.shape)
  DK = cast(int, q.shape[-1])
  q8, N, n_max = cache.dtype == dtypes.uint32, cast(int, cache.shape[-2]), cast(int, index.shape[1]) - 1
  # (bits 4: kv_q4_pack's rows: 4-bit values, the same groups of 32, their scales half)
  q4 = q8 and bits == 4
  # (kpool: compact lists (pool_index_rows'): the picked pools, their count, the query's position)
  if kpool: n_max = (cast(int, index.shape[1]) - 2) * kpool + kpool
  W4, DPL = (DK + DV) // (8 if q4 else 4), DV // WARP_SIZE
  assert DK % 32 == 0 and DV % (WARP_SIZE * 8) == 0 and DPL == 8
  cache = cache.reshape(N, cast(int, cache.shape[-1]))
  b, part = UOp.range(B, 0, AxisType.GLOBAL), UOp.range(P, 1, AxisType.GLOBAL)
  wave, lane = UOp.range(W, 2, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  tid, T = wave * WARP_SIZE + lane, W * WARP_SIZE
  # the queries in LDS
  n = H * DK
  qf = q.reshape(B * n)
  # QSA_DOT (an int8 cache): the queries int8 a (head, group of 32 dims) with its scale and sum (the integer dots of sudot4)
  # (the (head, group)s quantized by the threads, a few each if they're more)
  dot = q8 and DK % 32 == 0 and H * (DK // 32) <= 4 * W * WARP_SIZE
  if dot:
    G = DK // 32
    qw = UOp.alloc((H*G*8,), dtypes.int32, addrspace=AddrSpace.LOCAL)
    qsc, qsm = UOp.alloc((H*G,), dtypes.float, addrspace=AddrSpace.LOCAL), UOp.alloc((H*G,), dtypes.int32, addrspace=AddrSpace.LOCAL)
    st = []
    for it in range((H*G + T - 1) // T):
      t_, mine = (tid + it*T).minimum(H*G - 1), tid + it*T < H*G
      qv = [qf[b*n + (t_ // G)*DK + (t_ % G)*32 + j].load().float() for j in range(32)]
      mx = functools.reduce(UOp.maximum, [x.maximum(-x) for x in qv])
      inv = (mx > 0).where(127.0 / mx.maximum(1e-30), UOp.const(0.0, dtypes.float))
      qi = [_sys("__builtin_rintf({0})", dtypes.float, x * inv).cast(dtypes.int32) for x in qv]
      words = [functools.reduce(lambda a, k: a | ((qi[4*w + k] & 255) << (8*k)), range(4), UOp.const(0, dtypes.int32)) for w in range(8)]
      st += [qw[(t_*8 + w).valid(mine)].store(words[w]) for w in range(8)]
      st += [qsc[t_.valid(mine)].store(mx / 127.0), qsm[t_.valid(mine)].store(functools.reduce(lambda a, x: a + x, qi))]
    barrier_q = UOp.group(*st).barrier()
    qw, qsc, qsm = qw.after(barrier_q), qsc.after(barrier_q), qsm.after(barrier_q)
  else:
    ql = UOp.alloc((n,), dtypes.float, addrspace=AddrSpace.LOCAL)
    ld = [ql[(tid + i*T).valid(tid + i*T < n)].store(qf[b*n + (tid + i*T).minimum(n-1)].load().float()) for i in range((n + T - 1) // T)]
    ql = ql.after(UOp.group(*ld).barrier())
  if kpool:
    kp = cast(int, index.shape[1]) - 2
    npk, qpos = index[b, kp].load().cast(dtypes.weakint), index[b, kp + 1].load().cast(dtypes.weakint)
    tail0 = (qpos + 1) // kpool * kpool
    count = (npk * kpool + qpos + 1 - tail0).minimum(n_max)
  else: count = index[b, n_max].load().cast(dtypes.weakint).minimum(n_max)
  koff = bases[b].load().cast(dtypes.weakint)
  tiles = (count + T - 1) // T
  rnd = UOp.range(((tiles - part + P - 1) // P).maximum(0), 3, AxisType.LOOP)
  i = (part + rnd * P) * T + wave * WARP_SIZE + lane
  valid = i < count
  # (the lanes past the list a copy of its last key: not the rows past it, of positions not written yet (a draft row's later ones, a past
  # sequence's): masked, they still changed the outputs)
  ic = i.minimum((count - 1).maximum(0))
  if kpool:
    keyi = (ic < npk * kpool).where(index[b, (ic // kpool).minimum(kp - 1)].load().cast(dtypes.weakint) * kpool + ic % kpool,
                                    tail0 + ic - npk * kpool)
  else: keyi = index[b, ic].load().cast(dtypes.weakint)
  row = (koff + keyi.maximum(0)).minimum(N - 1)
  def gscale(r:UOp, grp:UOp|int) -> UOp:
    # the scale of the group grp of row r (bits 4: halves, two a word)
    if not q4: return cache[r, W4 + grp].load().bitcast(dtypes.float)
    w = cache[r, W4 + grp // 2].load()
    hi = w >> ((grp % 2) * 16).cast(dtypes.uint32) if isinstance(grp, UOp) else w >> ((grp % 2) * 16)
    return hi.cast(dtypes.uint16).bitcast(dtypes.half).float()
  def nibbles(w:UOp) -> list[UOp]:
    # a word of 8 values of kv_q4_pack (a byte the values j, j+4 of 8) as the words of their bytes (the values + 8) in order
    return [w & 0x0f0f0f0f, (w >> 4) & 0x0f0f0f0f]
  def vals(r:UOp, d0:UOp|int, m:int) -> list[UOp]:
    # m values from dim d0 of row r (int8: the bytes are the values + 128, a group of 32 a scale; 4 bits: + 8)
    if not q8: return list(_vec_load(cache[r, d0], m))
    sc = gscale(r, d0 // 32)
    wv = _amd_load(cache[r, d0 // 8], m // 8) if q4 else _amd_load(cache[r, d0 // 4], m // 4)
    words = [x for i in range(m // 8) for x in nibbles(wv[i])] if q4 else [wv[i] for i in range(m // 4)]
    return [((words[j//4] >> (8*(j%4))) & 0xFF).cast(dtypes.float) * sc - sc * (8 if q4 else 128) for j in range(m)]
  if dot:
    # the scores of the lane's key: its bytes (the values + 128) dotted with the int8 queries 4 a step, less 128 the queries' sum, times scales
    # (a loop over the groups: unrolled, the queries' loads would be hoisted out of the keys' loop, 256 registers)
    sacc = _reg((H,), 0, rnd)
    g = UOp.range(G, 4, AxisType.LOOP)
    kv4 = _amd_load(cache[row, g * 4], 4) if q4 else _amd_load(cache[row, g * 8], 8)
    kw = [x for i in range(4) for x in nibbles(kv4[i])] if q4 else [kv4[i] for i in range(8)]
    ks, prev_s, bias = gscale(row, g), sacc.after(g), 8 if q4 else 128
    new_s = []
    for h in range(H):
      acc_i = functools.reduce(lambda a, w: _amd_sudot4u(qw[(h*G + g)*8 + w].load(), kw[w].bitcast(dtypes.int32), a), range(8),
                               UOp.const(0, dtypes.int32))
      new_s.append(prev_s[h].load() + (acc_i - qsm[h*G + g].load() * bias).cast(dtypes.float) * (qsc[h*G + g].load() * ks))
    sacc = sacc.after(sacc.store(UOp.stack(*new_s)).end(g))
    sc_ = [valid.where(sacc[h].load() * scale, UOp.const(-1e30, dtypes.float)) for h in range(H)]
  else:
    # the scores of the lane's key: 8 dims a step
    acc = _reg((H,), 0, rnd)
    dch = UOp.range(DK // 8, 4, AxisType.LOOP)
    kv, prev = vals(row, dch * 8, 8), acc.after(dch)
    new = [functools.reduce(lambda a, j: a + ql[h*DK + dch*8 + j].load() * kv[j], range(8), prev[h].load()) for h in range(H)]
    acc = acc.after(acc.store(UOp.stack(*new)).end(dch))
    sc_ = [valid.where(acc[h].load() * scale, UOp.const(-1e30, dtypes.float)) for h in range(H)]
  # the online softmax of the wave's keys
  m_reg, l_reg, o_reg = _reg((H,), -1e30), _reg((H,), 0), _reg((H, DPL), 0)
  pm, plr, po = m_reg.after(rnd), l_reg.after(rnd), o_reg.after(rnd)
  m_new = [pm[h].load().maximum(warp_reduce(sc_[h], maximum=True, full_wave=True)) for h in range(H)]
  pr = [valid.where(((sc_[h] - m_new[h]) * LOG2E).exp2(), UOp.const(0.0, dtypes.float)) for h in range(H)]
  alpha = [((pm[h].load() - m_new[h]) * LOG2E).exp2() for h in range(H)]
  l_new = [alpha[h] * plr[h].load() + warp_reduce(pr[h], full_wave=True) for h in range(H)]
  # the tile's weights and rows to every lane (LDS), then its values into the lane's DPL dims
  pl = UOp.alloc((W, H, WARP_SIZE), dtypes.float, addrspace=AddrSpace.LOCAL)
  rl = UOp.alloc((W, WARP_SIZE), dtypes.int32, addrspace=AddrSpace.LOCAL)
  st = [pl[wave, h, lane].store(pr[h]) for h in range(H)] + [rl[wave, lane].store(row.cast(dtypes.int32))]
  barrier = UOp.group(*st).barrier()
  pl, rl = pl.after(barrier), rl.after(barrier)
  o = [[alpha[h] * po[h, d].load() for d in range(DPL)] for h in range(H)]
  if dot:
    # two keys at once: a dim's two bytes the half2 (1024+u1, 1024+u2) (a perm), less 1152 their values, dotted with the two keys' weights
    # times their scales (of the lane's group of 32 dims) by v_dot2_f32_f16
    # (a loop over the pairs, QSA_PV_UNROLL of them a step: unrolled, their loads all at once spill)
    vg, U = W4 + DK // 32 + lane * DPL // 32, 2
    oacc = _reg((H, DPL), 0, rnd)
    oacc = oacc.after(oacc.store(UOp.stack(*[x for oh in o for x in oh]).reshape(H, DPL)))
    jr = UOp.range(WARP_SIZE // (2*U), 5, AxisType.LOOP)
    prev_o = oacc.after(jr)
    o = [[prev_o[h, d].load() for d in range(DPL)] for h in range(H)]
    for u in range(U):
      jj = jr * U + u
      r1, r2 = (rl[wave, 2*jj + k].load().cast(dtypes.weakint) for k in (0, 1))
      # (4 bits: the lane's 8 dims a word)
      w1, w2 = (nibbles(cache[r, (DK + lane * DPL) // 8].load()) if q4 else _amd_load(cache[r, (DK + lane * DPL) // 4], DPL // 4)
                for r in (r1, r2))
      s1, s2 = (gscale(r, vg - W4) for r in (r1, r2))
      pp = [_amd_pkrtz(pl[wave, h, 2*jj].load() * s1, pl[wave, h, 2*jj + 1].load() * s2) for h in range(H)]
      for d in range(DPL):
        sel = (d % 4) | 0x0c00 | ((4 + d % 4) << 16) | 0x0c000000
        pair = _amd_byte_perm(w2[d // 4], w1[d // 4], UOp.const(sel, dtypes.uint32)) | 0x64006400
        for h in range(H): o[h][d] = _amd_fdot2_less1152(pair, pp[h], o[h][d], 1032 if q4 else 1152)
    oacc = oacc.after(oacc.store(UOp.stack(*[x for oh in o for x in oh]).reshape(H, DPL)).end(jr))
    o = [[oacc[h, d].load() for d in range(DPL)] for h in range(H)]
  else:
    for j in range(WARP_SIZE):
      v = vals(rl[wave, j].load().cast(dtypes.weakint), DK + lane * DPL, DPL)
      for h in range(H):
        p = pl[wave, h, j].load()
        o[h] = [o[h][d] + p * v[d] for d in range(DPL)]
  update = UOp.group(m_reg.store(UOp.stack(*m_new)), l_reg.store(UOp.stack(*l_new)),
                     o_reg.store(UOp.stack(*[x for oh in o for x in oh]).reshape(H, DPL))).end(rnd)
  m_reg, l_reg, o_reg = m_reg.after(update), l_reg.after(update), o_reg.after(update)
  # the waves combined through LDS
  ml = UOp.alloc((W, H, 2), dtypes.float, addrspace=AddrSpace.LOCAL)
  ol = UOp.alloc((W, H, DV), dtypes.float, addrspace=AddrSpace.LOCAL)
  st = [ol[wave, h, lane * DPL + d].store(o_reg[h, d].load()) for h in range(H) for d in range(DPL)]
  st += [ml[wave, h, k].store(x) for h in range(H) for k, x in enumerate((m_reg[h].load(), l_reg[h].load()))]
  barrier = UOp.group(*st).barrier()
  ml, ol = ml.after(barrier), ol.after(barrier)
  final = []
  for k in range((H * DV + T - 1) // T):
    flat = tid + k * T
    h, d = (flat // DV).minimum(H - 1), flat % DV
    M = functools.reduce(UOp.maximum, (ml[w_, h, 0].load() for w_ in range(W)))
    val = functools.reduce(lambda a, w_: a + ((ml[w_, h, 0].load() - M) * LOG2E).exp2() * ol[w_, h, d].load(), range(W), UOp.const(0.0, dtypes.float))
    final.append(out[b, h.valid(flat < H * DV), part, d].store(val))
  hs = tid.minimum(H - 1)
  M = functools.reduce(UOp.maximum, (ml[w_, hs, 0].load() for w_ in range(W)))
  L = functools.reduce(lambda a, w_: a + ((ml[w_, hs, 0].load() - M) * LOG2E).exp2() * ml[w_, hs, 1].load(), range(W), UOp.const(0.0, dtypes.float))
  final += [stats[b, hs.valid(tid < H), part, 0].store(M), stats[b, hs.valid(tid < H), part, 1].store(L)]
  rb = ((DK + DV) // 2 + (DK + DV) // 16 if q4 else (DK + DV) + (DK + DV) // 8) if q8 else (DK + DV) * 2
  est = Estimates(4 * n_max * B * H * (DK + DV), n_max * B * rb, n_max * B * rb)
  return UOp.group(*final).end(lane, wave, part, b).sink(arg=KernelInfo(name="qsa_decode", opts_to_apply=(), estimates=est))

# QSA_PREFILL: a prefill's rows (many) on qsa_prefill's kernel (the tensor cores), not qsa_decode's (a lane a key: vector ALU bound)
QSA_PREFILL = getenv("QSA_PREFILL", 0)

@functools.cache
def _qsa_prefill_kernel(out:UOp, stats:UOp, q:UOp, cache:UOp, index:UOp, bases:UOp, *, scale:float, kpool:int, rdna4:bool, ns:int=1) -> UOp:
  # as _qsa_decode_kernel (its outputs, compact lists of pools, kv_q4_pack's 4-bit cache) on the tensor cores: a workgroup a row b, its heads
  # (at most 16) the rows of the tiles. a tile of 16*ns keys into LDS (keys: their words; values: dequantized in half; scales in float), the
  # scores of a group of 32 dims a product of the half queries and those (2 WMMA), times the keys' scales of the group (a lane's key); a wave a
  # quarter of the dims, the scores summed through LDS. the values (times their groups' scales, in half) by the weights (in half)
  B, H, _, DV = cast(tuple[int, int, int, int], out.shape)
  DK, N = cast(int, q.shape[-1]), cast(int, cache.shape[-2])
  WAVES, D, TK = 4, DK + DV, 16 * ns
  W4, T = D // 8, WAVES * WARP_SIZE
  assert H <= 16 and DK % (32 * WAVES) == 0 and DV % (32 * WAVES) == 0 and cast(int, cache.shape[-1]) == W4 + D // 64
  cache = cache.reshape(N, cast(int, cache.shape[-1]))
  DW, VW = DK // WAVES, DV // WAVES
  b = UOp.range(B, 0, AxisType.GLOBAL)
  wave, lane = UOp.range(WAVES, 1, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  col, half, tid = lane % 16, lane // 16, wave * WARP_SIZE + lane
  ks = tuple(e % 4 + half * 4 + (e // 4) * 8 for e in range(8)) if rdna4 else tuple(range(16))
  def crow(e:int) -> UOp: return half * 8 + e if rdna4 else 2 * e + half  # (the row of a lane's accumulator element e)
  # the queries' fragments of the wave's dims: a head a row (rows past H zeros)
  qv = col < H
  qfrag = [UOp.stack(*(qv.where(q[b, col.minimum(H-1), wave*DW + kk*16 + k].load(), UOp.const(0, dtypes.half)) for k in ks))
           for kk in range(DW // 16)]
  # the keys of the row's list (see _qsa_decode_kernel): its pools' kpool keys, then the ones after the last whole pool up to its position
  kp = cast(int, index.shape[1]) - 2
  n_max = kp * kpool + kpool
  npk, qpos = index[b, kp].load().cast(dtypes.weakint), index[b, kp + 1].load().cast(dtypes.weakint)
  tail0 = (qpos + 1) // kpool * kpool
  count = (npk * kpool + qpos + 1 - tail0).minimum(n_max)
  koff = bases[b].load().cast(dtypes.weakint)
  ntiles = (count + TK - 1) // TK
  tile = UOp.range(ntiles, 2, AxisType.LOOP)
  def key_at(t:UOp, j:UOp) -> tuple[UOp, UOp]:
    i = t * TK + j
    ic = i.minimum((count - 1).maximum(0))
    keyi = (ic < npk * kpool).where(index[b, (ic // kpool).minimum(kp - 1)].load().cast(dtypes.weakint) * kpool + ic % kpool,
                                    tail0 + ic - npk * kpool)
    return (koff + keyi.maximum(0)).minimum(N - 1), i < count
  # the tile: a thread D*TK/T dims of a key (words of 8 values, their groups' scales), its values less 8 in half (exact). the next tile's words
  # loaded (registers) before this one's tensor core work, stored to LDS after it
  TPR = T // TK  # threads a key
  DPT = D // TPR  # dims a thread
  assert DK % DPT == 0, "a thread's dims all keys' or all values'"
  lrow, lpart = tid // TPR, tid % TPR
  # (the keys' halves: their words as they are (a value converted at its fragment); the values': n - 8 in half, a value's 16 keys strided)
  lds_k = UOp.alloc((TK, DK // 8 + 1), dtypes.uint32, addrspace=AddrSpace.LOCAL)
  lds_v = UOp.alloc((TK, DV + 8), dtypes.half, addrspace=AddrSpace.LOCAL)
  lds_sc = UOp.alloc((D // 32, TK + 4), dtypes.float, addrspace=AddrSpace.LOCAL)  # (a group's keys together)
  def word_halves(w:UOp) -> list[UOp]:
    # a word's 8 values (kv_q4_pack: byte i the values i (low nibble) and i+4) in order: the nibbles into halves 1024+n, less 1032: n - 8 exactly
    nib = [w & 0x0f0f0f0f, (w >> 4) & 0x0f0f0f0f]
    pairs = [_amd_byte_perm(UOp.const(0x64646464, dtypes.uint32), x, UOp.const(sel, dtypes.uint32)) for x in nib for sel in (0x04010400, 0x04030402)]
    return [((p_ >> sh) & 0xffff).cast(dtypes.uint16).bitcast(dtypes.half) - 1032 for p_ in pairs for sh in (0, 16)]
  def tile_loads(t:UOp) -> tuple[list[UOp], UOp]:
    lkey, lok = key_at(t, lrow)
    words = [v[i] for h4 in range(DPT // 32) if (v:=_amd_load(cache[lkey, lpart*(DPT//8) + h4*4], 4)) is not None for i in range(4)]
    return words + [cache[lkey, W4 + lpart*(DPT//64) + i].load() for i in range(DPT // 64)], lok
  def tile_stores(vals:tuple[list[UOp], UOp], lk:UOp, lv:UOp, lsc:UOp) -> list[UOp]:
    ws, lok = vals
    words, sws = ws[:DPT // 8], ws[DPT // 8:]
    st = []
    scs = [lok.where((sws[gi // 2] >> (16 * (gi % 2))).cast(dtypes.uint16).bitcast(dtypes.half).float(), UOp.const(0.0, dtypes.float))
           for gi in range(DPT // 32)]
    for wi, w in enumerate(words):
      d0 = lpart*DPT + wi*8  # (a thread's dims are all keys' or all values')
      kdim = lpart*DPT < DK
      st.append(lk[lrow, (lpart*(DPT//8) + wi).valid(kdim)].store(w))
      # (the values times their group's scale, in half: the weights' fragments one for all the value groups)
      st += [lv[lrow, (d0 - DK + j).valid(kdim.ne(True))].store((h_.float() * scs[wi // 4]).cast(dtypes.half)) for j, h_ in enumerate(word_halves(w))]
    for gi in range(DPT // 32): st.append(lsc[lpart * (DPT // 32) + gi, lrow].store(scs[gi]))
    return st
  lds_k0, lds_v0, lds_sc0 = lds_k, lds_v, lds_sc
  first = UOp.group(*tile_stores(tile_loads(UOp.const(0, dtypes.weakint)), lds_k, lds_v, lds_sc)).barrier()
  nxt = tile_loads((tile + 1).minimum((ntiles - 1).maximum(0)))
  lds_k, lds_v, lds_sc = lds_k.after(first, tile), lds_v.after(first, tile), lds_sc.after(first, tile)
  key_ok = [key_at(tile, st_ * 16 + col)[1] for st_ in range(ns)]
  # the wave's scores of a subtile of 16 keys: a group's two WMMA, times the scales of the lane's key
  zero = UOp.stack(*[UOp.const(0, dtypes.float)]*8)
  lds_s = UOp.alloc((WAVES, ns, WARP_SIZE, 8), dtypes.float, addrspace=AddrSpace.LOCAL)
  sst = []
  for st_ in range(ns):
    s_acc = [UOp.const(0, dtypes.float)] * 8
    for gl in range(DW // 32):
      t = zero
      for h2 in range(2):
        kk = gl * 2 + h2
        d0 = wave*DW + kk*16
        kh = word_halves(lds_k[st_*16 + col, d0 // 8].load()) + word_halves(lds_k[st_*16 + col, d0 // 8 + 1].load())
        # (RDNA4: a lane half's 8 of the 16 dims, see ks)
        kf = [half.eq(0).where(kh[e % 4 + (e // 4) * 8], kh[4 + e % 4 + (e // 4) * 8]) for e in range(8)] if rdna4 else kh
        t = UOp.wmma(qfrag[kk], UOp.stack(*kf), t, *WMMA_ARG)
      treg = UOp.alloc((8,), dtypes.float, addrspace=AddrSpace.REG)
      treg = treg.after(treg.store(t))
      ksc = lds_sc[(wave*DW) // 32 + gl, st_*16 + col].load()
      s_acc = [s_acc[e] + treg[e].load() * ksc for e in range(8)]
    sst += [lds_s[wave, st_, lane, e].store(s_acc[e]) for e in range(8)]
  lds_s = lds_s.after(UOp.group(*sst).barrier())
  VT, ES = VW // 16, 8 // WAVES
  o_regs = [_reg((8,), 0) for _ in range(VT)]
  m_reg, l_reg = _reg((ES,), -1e30), _reg((ES,), 0)
  m_old, l_old = m_reg.after(tile), l_reg.after(tile)
  lds_p = UOp.alloc((16, TK + 8), dtypes.half, addrspace=AddrSpace.LOCAL)
  lds_a = UOp.alloc((16,), dtypes.float, addrspace=AddrSpace.LOCAL)
  pst, m_new, l_new = [], [], []
  for i in range(ES):
    e = wave * ES + i
    sv = [key_ok[st_].where(functools.reduce(lambda a, w_: a + lds_s[w_, st_, lane, e].load(), range(1, WAVES), lds_s[0, st_, lane, e].load())
                            * scale, UOp.const(-1e30, dtypes.float)) for st_ in range(ns)]
    mx = functools.reduce(UOp.maximum, sv)
    for sh in (1, 2, 4, 8): mx = mx.maximum(_lane_xor(mx, sh))
    mn = m_old[i].load().maximum(mx)
    pv = [key_ok[st_].where(((sv[st_] - mn) * LOG2E).exp2(), UOp.const(0, dtypes.float)) for st_ in range(ns)]
    ps = functools.reduce(lambda a, x: a + x, pv)
    for sh in (1, 2, 4, 8): ps = ps + _lane_xor(ps, sh)
    a = ((m_old[i].load() - mn) * LOG2E).exp2()
    pst += [lds_p[crow(e), st_*16 + col].store(pv[st_].cast(dtypes.half)) for st_ in range(ns)]
    pst.append(lds_a[crow(e).valid(col.eq(0))].store(a))
    m_new.append(mn)
    l_new.append(l_old[i].load() * a + ps)
  shared = UOp.group(*pst, m_reg.store(UOp.stack(*m_new)), l_reg.store(UOp.stack(*l_new))).barrier()
  lds_p, lds_a = lds_p.after(shared), lds_a.after(shared)
  alpha = [lds_a[crow(e)].load() for e in range(8)]
  o_new = []
  # (the weights times the keys' scales of a values' group, in half: a lane's head's row of them, a subtile a WMMA; a group two dim tiles)
  # (a lane's head's row of the weights, a subtile a WMMA)
  pfrags = [UOp.stack(*(lds_p[col, st_*16 + k].load() for k in ks)) for st_ in range(ns)]
  for dt in range(VT):
    acc = UOp.stack(*(o_regs[dt].after(tile)[e].load() * alpha[e] for e in range(8)))
    for st_ in range(ns):
      pfrag = pfrags[st_]
      vfrag = UOp.stack(*(lds_v.after(shared)[st_*16 + k, wave*VW + dt*16 + col].load() for k in ks))
      acc = UOp.wmma(pfrag, vfrag, acc, *WMMA_ARG)
    o_new.append(acc)
  update = UOp.group(*(o.store(v) for o, v in zip(o_regs, o_new)), shared).barrier()
  update = UOp.group(*tile_stores(nxt, lds_k0.after(update), lds_v0.after(update), lds_sc0.after(update))).barrier().end(tile)
  stores = [out[b, crow(e).valid(crow(e) < H), 0, wave*VW + dt*16 + col].store(o.after(update)[e].load())
            for dt, o in enumerate(o_regs) for e in range(8)]
  m_fin, l_fin = m_reg.after(update), l_reg.after(update)
  for i in range(ES):
    r = crow(wave * ES + i)
    ok = col.eq(0) & (r < H)
    stores += [stats[b, r.valid(ok), 0, 0].store(m_fin[i].load()), stats[b, r.valid(ok), 0, 1].store(l_fin[i].load())]
  est = Estimates(4 * n_max * B * 16 * D, n_max * B * (D // 2 + D // 16), n_max * B * (D // 2 + D // 16))
  return UOp.group(*stores).end(b, wave, lane).sink(arg=KernelInfo(name="qsa_prefill", opts_to_apply=(), estimates=est))

def qsa_decode_attention(q:Tensor, cache:Tensor, index:Tensor, bases:Tensor, scale:float, parts:int=0, waves:int=0, kpool:int=0,
                         lse:bool=False, bits:int=8) -> Tensor|tuple[Tensor, Tensor]:
  """
  q (B, H, DK) every row's heads (a KV head's group: H on a device) on the keys of its list index (B, n+1) of the cache's rows (.., N, row) from
  bases (B,): a row's first DK dims its key, the next DV its value (half, or kv_q8_pack's). returns (B, H, 1, DV). lse: and the log of the sum
  of the exps of the scores of every (row, head) (B, H) (the keys split over devices: their outputs combined by them). bits: of a uint32 cache's
  values (8: kv_q8_pack's rows, 4: kv_q4_pack's)
  """
  B, H, DK = cast(tuple[int, int, int], q.shape)
  W = int(cache.shape[-1])
  DV = (W * 64 // 9 if bits == 4 else W * 32 // 9 if cache.dtype == dtypes.uint32 else W) - DK
  if QSA_PREFILL and B > 256 and bits == 4 and kpool and H <= 16 and not parts and not waves and amd_custom_kernels_supported(q.device):
    # (a prefill's rows: the tensor cores, see _qsa_prefill_kernel)
    axis = q.uop.axis
    out = Tensor.empty(B, H, 1, DV, dtype=dtypes.float32, device=q.device, axis=axis)
    stats = Tensor.empty(B, H, 1, 2, dtype=dtypes.float32, device=q.device, axis=axis)
    out, stats = Tensor.custom_kernel(out, stats, q.half().contiguous(), cache, index, bases,
                                      fxn=functools.partial(_qsa_prefill_kernel, scale=scale, kpool=kpool, rdna4=_wmma_rdna4(q.device),
                                                         ns=getenv("QSA_PREFILL_NS", 1)))[:2]
    if not lse: return Tensor.custom_kernel(Tensor.empty(B, H, 1, DV, dtype=dtypes.float32, device=q.device, axis=axis), out, stats,
                                            fxn=functools.partial(_amd_flash_decode_combine, live=1))[0]
    ls = Tensor.empty(B, H, dtype=dtypes.float32, device=q.device, axis=axis)
    res, _, _, ls = Tensor.custom_kernel(Tensor.empty(B, H, 1, DV, dtype=dtypes.float32, device=q.device, axis=axis), out, stats, ls,
                                         fxn=functools.partial(_amd_flash_decode_combine, live=1))[:4]
    return res, ls
  # (many rows (a prefill's): a workgroup of a wave a row, all its keys (enough workgroups, its queries loaded once, no combining))
  # (the rows of batched decoding (a few hundred): their keys split over parts and waves, a workgroup a row alone is too few for the GPU)
  many = B > 256
  parts = parts or (1 if many else 8)
  waves = waves or (1 if many else 4)
  axis = q.uop.axis
  partial = Tensor.empty(B, H, parts, DV, dtype=dtypes.float32, device=q.device, axis=axis)
  stats = Tensor.empty(B, H, parts, 2, dtype=dtypes.float32, device=q.device, axis=axis)
  partial, stats = Tensor.custom_kernel(partial, stats, q.half().contiguous(), cache, index, bases,
                                        fxn=functools.partial(_qsa_decode_kernel, scale=scale, W=waves, kpool=kpool, bits=bits))[:2]
  out = Tensor.empty(B, H, 1, DV, dtype=dtypes.float32, device=q.device, axis=axis)
  if not lse: return Tensor.custom_kernel(out, partial, stats, fxn=functools.partial(_amd_flash_decode_combine, live=parts))[0]
  ls = Tensor.empty(B, H, dtype=dtypes.float32, device=q.device, axis=axis)
  out, _, _, ls = Tensor.custom_kernel(out, partial, stats, ls, fxn=functools.partial(_amd_flash_decode_combine, live=parts))[:4]
  return out, ls


@functools.cache
def _slot_scatter_kernel(dst:UOp, src:UOp, slots:UOp, counts:UOp, *starts:UOp, S:int, shift:int) -> UOp:
  # row slots[g]*S + (starts[g]+counts[g])%S of dst (n*S, rows, F) <- rows shift*counts[g].. (rows of them) of src[g] (G, R, F), cast to
  # dst's dtype (no starts: 0)
  G, _, F = cast(tuple[int, int, int], src.shape)
  rows = cast(int, dst.shape[1])
  L = math.gcd(F, 256)
  g, r, blk, lane = (UOp.range(G, 0, AxisType.GLOBAL), UOp.range(rows, 1, AxisType.GLOBAL), UOp.range(F // L, 2, AxisType.GLOBAL),
                     UOp.range(L, 3, AxisType.LOCAL))
  f, cnt = blk * L + lane, counts[g].load().cast(dtypes.weakint)
  value = src[g, shift * cnt + r, f].load().cast(dst.dtype)
  pos = cnt + starts[0][g].load().cast(dtypes.weakint) if starts else cnt
  store = dst[slots[g].load().cast(dtypes.weakint) * S + pos % S, r, f].store(value)
  return store.end(g, r, blk, lane).sink(arg=KernelInfo(name="slot_scatter", opts_to_apply=()))

def slot_scatter(pool:Tensor, src:Tensor, slots:Tensor, counts:Tensor, S:int, shift:int=0, starts:Tensor|None=None) -> Tensor:
  """
  the states of G sequences into their slots: pool (n*S, rows, F) row slots[g]*S + (starts[g]+counts[g])%S (a slot's S rows, the one of
  position starts[g]+counts[g], no starts: 0) is rows shift*counts[g].. (rows of them) of src[g] (G, R, F). returns pool after
  """
  return Tensor.custom_kernel(pool, src, slots, counts, *([starts] if starts is not None else []),
                              fxn=functools.partial(_slot_scatter_kernel, S=S, shift=shift))[0]

@functools.cache
def _pool_window_kernel(pkeys:UOp, pool:UOp, keys:UOp, slots:UOp, positions:UOp, *, S:int, NP:int, kpool:int) -> UOp:
  # the keys (G*NP*kpool, D) of the NP pools from every sequence g's first row's (at positions[g]) from its window: the kpool-1 keys before it
  # (its slot's state of the position: row slots[g]*S + positions[g]%S of pool (n*S, kpool-1, D)) and the keys of its R rows (keys (G, R, D)),
  # the keys past them its last (an incomplete pool's). and the state of the position after every row written (the window's kpool-1 rows
  # before it, rows positions[g]+r+1 of the slot: not the one read, R < S)
  G, R, D = cast(tuple[int, int, int], keys.shape)
  K1, W0, NK = kpool - 1, kpool - 1 + R, NP * kpool
  g, item, lane = UOp.range(G, 0, AxisType.GLOBAL), UOp.range(NK + R * K1, 1, AxisType.GLOBAL), UOp.range(D, 2, AxisType.LOCAL)
  slot, pos = slots[g].load().cast(dtypes.weakint), positions[g].load().cast(dtypes.weakint)
  def win(w:UOp) -> UOp:
    sv = pool[slot * S + pos % S, w.minimum(K1 - 1), lane].load()
    kv = keys[g, (w - K1).maximum(0).minimum(R - 1), lane].load()
    return (w < K1).where(sv, kv)
  is_pk = item < NK
  pk = pkeys[(g * NK + item).valid(is_pk), lane].store(win((K1 - pos % kpool + item).minimum(W0 - 1)).cast(pkeys.dtype))
  si = (item - NK).maximum(0)
  r, i = si // K1, si % K1
  st = pool[(slot * S + (pos + r + 1) % S).valid(item >= NK), i, lane].store(win(r + 1 + i).cast(pool.dtype))
  return UOp.group(pk, st).end(g, item, lane).sink(arg=KernelInfo(name="pool_window", opts_to_apply=()))

def pool_window(pool:Tensor, keys:Tensor, slots:Tensor, positions:Tensor, S:int, NP:int, kpool:int) -> tuple[Tensor, Tensor]:
  """
  the keys (G*NP*kpool, D) of the NP pools from the first row's of G sequences of R rows (keys (G, R, D)), the ones before them from the
  sequences' slots' states (pool (n*S, kpool-1, D), see QSAIndexer), and the pool after the states of their rows' next positions are written
  """
  G, R, D = (int(x) for x in keys.shape)
  out = Tensor.empty(G * NP * kpool, D, dtype=keys.dtype, device=keys.device)
  out, pool = Tensor.custom_kernel(out, pool, keys.contiguous(), slots, positions,
                                   fxn=functools.partial(_pool_window_kernel, S=S, NP=NP, kpool=kpool))[:2]
  return out, pool

@functools.cache
def _slot_gather_kernel(dst:UOp, pool:UOp, slots:UOp, starts:UOp, S:int) -> UOp:
  # dst[g] (G, rows, F) <- row slots[g]*S + starts[g]%S of pool (n*S, rows, F), cast to dst's dtype
  G, rows, F = cast(tuple[int, int, int], dst.shape)
  L = math.gcd(F, 256)
  g, r, blk, lane = (UOp.range(G, 0, AxisType.GLOBAL), UOp.range(rows, 1, AxisType.GLOBAL), UOp.range(F // L, 2, AxisType.GLOBAL),
                     UOp.range(L, 3, AxisType.LOCAL))
  f = blk * L + lane
  row = slots[g].load().cast(dtypes.weakint) * S + starts[g].load().cast(dtypes.weakint) % S
  return dst[g, r, f].store(pool[row, r, f].load().cast(dst.dtype)).end(g, r, blk, lane).sink(arg=KernelInfo(name="slot_gather", opts_to_apply=()))

# ******** gated hyper-connections (qwen4exp) of a few rows: the streams' norm, quantization and inject partials in a kernel, the up projection,
# its sigmoid gates and the mean of the gated streams in another (the down projection between them is the decode linear) ********

def _sigmoid(x:UOp) -> UOp: return ((-x * LOG2E).exp2() + 1).reciprocal()

@functools.cache
def _ghc_pre_kernel(xn:UOp, q:UOp, scale:UOp, xsum:UOp, part:UOp, xs:UOp, w:UOp, winj:UOp, *, count:int, D:int, eps:float) -> UOp:
  # a workgroup a (token, stream) of xs (T, count*D): its rms norm (times w) in xn, quantized in groups of 32 (q, scale, xsum: q8_quantize's),
  # and its inject partials part[token, stream, j] = winj[j, stream's] . xn of the stream
  NJ, W = cast(int, winj.shape[0]), 8
  wg, wave = UOp.range(cast(int, xs.shape[0])*count, 0, AxisType.GLOBAL), UOp.range(W, 1, AxisType.LOCAL)
  lane = UOp.range(WARP_SIZE, -1, AxisType.WARP)
  tok, st, t = wg // count, wg % count, wave * WARP_SIZE + lane
  vals = [xs[tok, st*D + it*W*WARP_SIZE + t].load().float() for it in range(D // (W*WARP_SIZE))]
  def block_sum(lds:UOp, j:int, v:UOp) -> tuple[UOp, UOp]:
    return lds, lds[j, wave.valid(lane.eq(0))].store(warp_reduce(v, full_wave=True))
  red = UOp.alloc((1 + NJ, W), dtypes.float, addrspace=AddrSpace.LOCAL)
  red = red.after(red[0, wave.valid(lane.eq(0))].store(warp_reduce(functools.reduce(lambda a,b: a+b, [v*v for v in vals]), full_wave=True)).barrier())
  rinv = (functools.reduce(lambda a,b: a+b, [red[0, i].load() for i in range(W)]) * (1/D) + eps).sqrt().reciprocal()
  xns = [v * rinv * w[st*D + it*W*WARP_SIZE + t].load().float() for it, v in enumerate(vals)]
  stores:list[UOp] = []
  for it, x in enumerate(xns):
    stores.append(xn[tok, st*D + it*W*WARP_SIZE + t].store(x.cast(xn.dtype)))
    stores += _q8_narrow_stores(x, q, scale, xsum, None, tok, st*(D//32) + it*W + wave, lane)
  ps = [red[1+j, wave.valid(lane.eq(0))].store(warp_reduce(functools.reduce(lambda a,b: a+b, [
          winj[j, st*D + it*W*WARP_SIZE + t].load().float() * x for it, x in enumerate(xns)]), full_wave=True)) for j in range(NJ)]
  red2 = red.after(UOp.group(*ps).barrier())
  stores += [part[tok, st, (UOp.const(j, dtypes.weakint)).valid(t.eq(0))].store(
               functools.reduce(lambda a,b: a+b, [red2[1+j, i].load() for i in range(W)])) for j in range(NJ)]
  return UOp.group(*stores).end(wg, wave, lane).sink(arg=KernelInfo(name="ghc_pre", opts_to_apply=()))

@functools.cache
def _ghc_prefill_norm_kernel(xn:UOp, xhalf:UOp, part:UOp, xs:UOp, w:UOp, winj:UOp, *, count:int, D:int, eps:float, rows:int=0) -> UOp:
  # Keep each normalized stream in registers for all inject dot products. Also write the half GEMM input, avoiding a separate cast.
  # rows: the rows (of a device) past them aren't read nor their xn and xhalf stored (padding: their inject partials are zeros)
  NJ, W = cast(int, winj.shape[0]), 8
  wg, wave = UOp.range(cast(int, xs.shape[0])*count, 0, AxisType.GLOBAL), UOp.range(W, 1, AxisType.LOCAL)
  lane = UOp.range(WARP_SIZE, -1, AxisType.WARP)
  tok, st, t = wg // count, wg % count, wave * WARP_SIZE + lane
  tr = tok.valid(tok < rows) if rows else tok
  vals = [xs[tr, st*D + it*W*WARP_SIZE + t].load().float() for it in range(D // (W*WARP_SIZE))]
  red = UOp.alloc((1 + NJ, W), dtypes.float32, addrspace=AddrSpace.LOCAL)
  red = red.after(red[0, wave.valid(lane.eq(0))].store(warp_reduce(sum(v*v for v in vals), full_wave=True)).barrier())
  rinv = (sum(red[0, i].load() for i in range(W)) * (1/D) + eps).sqrt().reciprocal()
  norm = [v * rinv * w[st*D + it*W*WARP_SIZE + t].load().float() for it, v in enumerate(vals)]
  stores:list[UOp] = []
  for it, v in enumerate(norm):
    at = st*D + it*W*WARP_SIZE + t
    stores += [xn[tr, at].store(v), xhalf[tr, at].store(v.cast(dtypes.half))]
  ps = [red[1+j, wave.valid(lane.eq(0))].store(warp_reduce(sum(
          winj[j, st*D + it*W*WARP_SIZE + t].load().float() * v for it, v in enumerate(norm)), full_wave=True)) for j in range(NJ)]
  red2 = red.after(UOp.group(*ps).barrier())
  stores += [part[tok, st, UOp.const(j, dtypes.weakint).valid(t.eq(0))].store(sum(red2[1+j, i].load() for i in range(W))) for j in range(NJ)]
  return UOp.group(*stores).end(wg, wave, lane).sink(arg=KernelInfo(name="ghc_prefill_norm", opts_to_apply=()))

def ghc_prefill_norm(xs:Tensor, w:Tensor, winj:Tensor, count:int, eps:float, rows:int=0) -> tuple[Tensor, Tensor, Tensor]:
  """rows: a device's rows past them aren't read (batched decoding's padding: xn and xhalf not written, their inject partials zeros)"""
  T, CD = cast(tuple[int, int], xs.shape)
  D = CD // count
  assert CD % count == 0 and D % 256 == 0
  axis = 0 if token_sharded(xs) else None
  xn = Tensor.empty(T, CD, dtype=dtypes.float32, device=xs.device, axis=axis)
  xhalf = Tensor.empty(T, CD, dtype=dtypes.half, device=xs.device, axis=axis)
  part = Tensor.empty(T, count, winj.shape[0], dtype=dtypes.float32, device=xs.device, axis=axis)
  res = Tensor.custom_kernel(xn, xhalf, part, xs.contiguous(), w.contiguous(), winj.contiguous(),
                             fxn=functools.partial(_ghc_prefill_norm_kernel, count=count, D=D, eps=eps, rows=rows))
  return res[0], res[1], res[2]

def _q8_gemm_2d(out:UOp, *srcs:UOp, **kw) -> UOp: return _q8_gemm_kernel(out.reshape(-1, out.shape[-1]), *srcs, **kw)

def ghc_prefill_mix(lo:Tensor, raw:Tensor, xn:Tensor, shape:tuple[int, ...]|None=None, rows:int=0, m_rows:int=0) -> Tensor:
  """
  shape: the output's (rows its second axis, split on it: made so, a view of the (rows, D) one isn't a buffer: a copy of it). fewer rows a
  device than lo's: its first ones. rows: a device's rows of lo and xn past them aren't read. m_rows (TILE16): the tiles of a device's first
  m_rows rows alone (the other rows of the output not written), tiles of 16 if they fit
  """
  # Four stream projections in each workgroup: sigmoid and weighted mean stay in registers, not a full-size gate tensor.
  T, CD = cast(tuple[int, int], xn.shape)
  assert CD % 128 == 0
  sharded = token_sharded(xn)
  if shape is not None:
    out = Tensor.empty(*shape, dtype=dtypes.float32, device=xn.device, axis=len(shape) - 2 if sharded else None)
    mr = m_rows if TILE16 and not _wmma_rdna4(xn.device) and 0 < m_rows < T // (len(cast(tuple, xn.device)) if sharded else 1) else 0
    return Tensor.custom_kernel(out, raw, lo.half().contiguous(), xn,
                                fxn=functools.partial(_q8_gemm_2d, token_tile=16 if mr and mr <= 16 else 32, block_n=32,
                                                      rdna4=_wmma_rdna4(xn.device), rows=rows, m_rows=mr))[0]
  out = Tensor.empty(T, CD//4, dtype=dtypes.float32, device=xn.device, axis=0 if sharded else None)
  ret = Tensor.custom_kernel(out, raw, lo.half().contiguous(), xn,
                             fxn=functools.partial(_q8_gemm_kernel, token_tile=32, block_n=32, rdna4=_wmma_rdna4(xn.device)))[0]
  return ret.reshape(shape) if shape is not None else ret

def ghc_pre(xs:Tensor, w:Tensor, winj:Tensor, count:int, eps:float) -> tuple[Tensor, tuple[Tensor, Tensor, Tensor], Tensor]:
  """xs (T, count*D) the streams of T rows: their norms (T, count*D) half, its quantization (q8_quantize's), the inject partials (T, count, NJ)"""
  T, CD = cast(tuple[int, int], xs.shape)
  D, NJ, G = CD // count, int(winj.shape[0]), CD // Q8_GROUP_SIZE
  # (rows split over the devices (batched decoding's, HC_DEC): every device its rows, the outputs split so)
  ax = 0 if token_sharded(xs) else None
  outs = [Tensor.empty(T, CD, dtype=dtypes.half, device=xs.device, axis=ax), Tensor.empty(T, G, 8, dtype=dtypes.uint32, device=xs.device, axis=ax),
          Tensor.empty(T, G, dtype=dtypes.float32, device=xs.device, axis=ax), Tensor.empty(T, G, 2, dtype=dtypes.float32, device=xs.device, axis=ax),
          Tensor.empty(T, count, NJ, dtype=dtypes.float32, device=xs.device, axis=ax)]
  res = Tensor.custom_kernel(*outs, xs.float().contiguous(), w.float().contiguous(), winj.float().contiguous(),
                             fxn=functools.partial(_ghc_pre_kernel, count=count, D=D, eps=eps))
  return res[0], (res[1], res[2], res[3]), res[4]

@functools.cache
def _ghc_post_kernel(mixed:UOp, inj:UOp, lo:UOp, raw:UOp, xn:UOp, part:UOp, *, count:int, D:int, R:int) -> UOp:
  # mixed[t, d] = mean over the streams s of sigmoid(up[s*D+d] . silu(lo[t]/count)) * xn[t, s*D+d], up the Q8_0 rows (count*D, R) (blocks of 17
  # halfwords); inj[t, j] = sum over the streams of part[t, s, j] / count. 16 lanes a row (a block each), a workgroup the count rows of OUT
  # outputs: their gated values through LDS
  T, NJ, NB, TH = cast(int, lo.shape[0]), cast(int, part.shape[2]), R // 32, 256
  OUT = TH // 16 // count
  assert NB <= 16 and D % OUT == 0
  blk, wave = UOp.range(D // OUT, 0, AxisType.GLOBAL), UOp.range(TH // WARP_SIZE, 1, AxisType.LOCAL)
  lane = UOp.range(WARP_SIZE, -1, AxisType.WARP)
  t = wave * WARP_SIZE + lane
  g16, b = t // 16, t % 16
  dl, st = g16 // count, g16 % count
  row = st * D + blk * OUT + dl
  act = UOp.alloc((T, R), dtypes.float, addrspace=AddrSpace.LOCAL)
  ast = []
  for tok in range(T):
    for i in range((R + TH - 1) // TH):
      j = i * TH + t
      z = lo[tok, j.valid(j < R)].load() * (1/count)
      ast.append(act[tok, j.valid(j < R)].store(z * _sigmoid(z)))
  act = act.after(UOp.group(*ast).barrier())
  on = b < NB
  base = row * (NB * 17) + b.valid(on) * 17
  dd = raw[base].load().bitcast(dtypes.half).float()
  dots = [UOp.const(0.0, dtypes.float)] * T
  for h in range(16):
    hw = raw[base + 1 + h].load()
    q0, q1 = (((hw >> sh) & 0xff).cast(dtypes.uint8).bitcast(dtypes.int8).float() for sh in (0, 8))
    bi = b.valid(on) * 32 + 2*h
    dots = [dots[tok] + q0 * act[tok, bi].load() + q1 * act[tok, bi + 1].load() for tok in range(T)]
  sums = [warp_reduce(on.where(dd * dots[tok], 0.0)) for tok in range(T)]  # (16 lanes)
  gl = UOp.alloc((T, TH // 16), dtypes.float, addrspace=AddrSpace.LOCAL)
  gst = [gl[tok, g16.valid(b.eq(0))].store(_sigmoid(sums[tok]) * xn[tok, row].load().float()) for tok in range(T)]
  gl = gl.after(UOp.group(*gst).barrier())
  # (thread dl*count of the first (OUT*count) ... the first OUT threads of every token store the mean of their output's streams)
  ot = t.valid(t < OUT)
  stores = [mixed[tok, blk * OUT + ot].store(functools.reduce(lambda a,c: a+c, [gl[tok, t * count + s2].load() for s2 in range(count)]) * (1/count))
            for tok in range(T)]
  first = blk.eq(0) & t.eq(0)
  stores += [inj[tok, UOp.const(jj, dtypes.weakint).valid(first)].store(functools.reduce(lambda a,c: a+c, [part[tok, s2, jj].load()
             for s2 in range(count)]) * (1/count)) for tok in range(T) for jj in range(NJ)]
  return UOp.group(*stores).end(blk, wave, lane).sink(arg=KernelInfo(name="ghc_post", opts_to_apply=()))

def ghc_post(lo:Tensor, up_raw:Tensor, xn:Tensor, part:Tensor, count:int) -> tuple[Tensor, Tensor]:
  """the down outputs lo (T, R), the up projection's Q8_0 words, the norms xn and inject partials of ghc_pre: the mix (T, D), the inject (T, NJ)"""
  T, R = cast(tuple[int, int], lo.shape)
  D, NJ = int(xn.shape[1]) // count, int(part.shape[2])
  ax = 0 if token_sharded(lo) else None
  mixed = Tensor.empty(T, D, dtype=dtypes.float32, device=lo.device, axis=ax)
  inj = Tensor.empty(T, NJ, dtype=dtypes.float32, device=lo.device, axis=ax)
  res = Tensor.custom_kernel(mixed, inj, lo.float().contiguous(), up_raw, xn, part, fxn=functools.partial(_ghc_post_kernel, count=count, D=D, R=R))
  return res[0], res[1]

@functools.cache
def _ghc_decode_mix_kernel(mixed:UOp, inj:UOp, mq:UOp, md:UOp, ms:UOp, lo:UOp, raw:UOp, xn:UOp, part:UOp,
                           *, count:int, D:int, R:int) -> UOp:
  # Quantize silu(lo/count) in LDS, share Q8 up weights across the rows, and emit the mix plus its next-layer Q8 representation.
  T, NJ, NB = int(lo.shape[0]), int(part.shape[2]), R//32
  assert count == 4 and R % 32 == 0 and 0 < NB <= 16 and D % 32 == 0 and 1 < T <= 16 and T*NJ <= 64
  blk, wave, lane = UOp.range(D//32, 0, AxisType.GLOBAL), UOp.range(16, 1, AxisType.LOCAL), UOp.range(32, -1, AxisType.WARP)
  q = UOp.alloc((T, 8, 16), dtypes.uint32, addrspace=AddrSpace.LOCAL)
  scales = UOp.alloc((T, 16), dtypes.float32, addrspace=AddrSpace.LOCAL)
  prep:list[UOp] = []
  for batch in range((T*NB+15)//16):
    tg = wave + batch*16
    tok, group, valid = (tg//NB).minimum(T-1), tg % NB, tg < T*NB
    z = lo[tok, group*32+lane].load()*(1/count)
    value = z*_sigmoid(z)
    d = _cfn(False, src=(warp_reduce(value.abs(), maximum=True, full_wave=True),), arg=("({0}/127.0f)", dtypes.float32)).maximum(1e-8)
    quant = _cfn(False, src=(value, d), arg=("__builtin_nearbyintf({0}/{1})", dtypes.float32)).clip(-127, 127).cast(dtypes.int8)
    word = quant.cast(dtypes.uint8).cast(dtypes.uint32) << ((lane%4)*8).cast(dtypes.uint32)
    for off in (1, 2): word |= _cfn(False, src=(word,), arg=(f"__builtin_amdgcn_ds_swizzle({{0}}, {0x1f | off<<10})", dtypes.uint32))
    prep.extend((q[tok, (lane//4).valid(valid & (lane%4).eq(0)), group].store(word), scales[tok, group.valid(valid & lane.eq(0))].store(d)))
  ready = UOp.group(*prep).barrier()
  q, scales, raw = q.after(ready), scales.after(ready), raw.after(ready)
  shared = UOp.alloc((T, 32, count), dtypes.float32, addrspace=AddrSpace.LOCAL)
  written:list[UOp] = []
  for output in range(2):
    b, row = lane%8, lane//8*D + blk*32 + wave + output*16
    totals = [UOp.const(0., dtypes.float32)]*T
    for offset in range((NB+7)//8):
      g = b + 8*offset
      bg = g.minimum(NB-1)
      base = row*NB*17 + bg*17
      dd = _half(raw[base].load())
      acc = UOp.alloc((T,), dtypes.int32, addrspace=AddrSpace.REG)
      acc = acc.after(acc.store(acc.const_like(0)))
      i = UOp.range(8, 3+offset+output*2, AxisType.LOOP)
      w = _quant_word(raw, base, UOp.const(0), i, Q8_0, None)
      acc = acc.after(UOp.group(*[acc[tok].store(_amd_dp4a(w, q[tok, i, bg].load(), acc.after(i)[tok].load())) for tok in range(T)]).end(i))
      totals = [totals[tok] + (g < NB).where(acc[tok].load().float()*dd*scales[tok, bg].load(), 0) for tok in range(T)]
    for off in (4, 2, 1): totals = [v + _lane_xor(v, off) for v in totals]
    vals = [_sigmoid(totals[tok])*xn[tok, row].load().float() for tok in range(T)]
    # Store all four stream contributions: a cross-half-wave shuffle feeding only lane zero can read inactive lanes.
    written.extend(shared[tok, wave+output*16, (lane//8).valid(b.eq(0))].store(vals[tok]) for tok in range(T))
  shared = shared.after(UOp.group(*written).barrier())
  only = UOp.range(wave.eq(0).where(1, 0), 2, AxisType.LOOP)
  stores:list[UOp] = []
  for tok in range(T):
    value = sum(shared.after(only)[tok, lane, s].load() for s in range(count))*(1/count)
    stores.append(mixed[tok, blk*32+lane].store(value))
    stores.extend(_q8_narrow_stores(value, mq, md, ms, None, UOp.const(tok), blk, lane))
  for k in range((T*NJ + 31)//32):
    li = lane + 32*k
    valid = blk.eq(0) & (li < T*NJ)
    it, j = (li//NJ).valid(valid), li % NJ
    stores.append(inj[it, j].store(sum(part.after(only)[it, s, j].load() for s in range(count))*(1/count)))
  return UOp.group(*stores).end(only, blk, wave, lane).sink(arg=KernelInfo(name="ghc_decode_mix", opts_to_apply=()))

def ghc_decode_mix(lo:Tensor, up_raw:Tensor, xn:Tensor, part:Tensor, count:int) -> tuple[Tensor, Tensor]:
  T, R = cast(tuple[int, int], lo.shape)
  D, NJ = int(xn.shape[1])//count, int(part.shape[2])
  ax = 0 if (sharded:=token_sharded(lo)) else None
  outs = (Tensor.empty(T, D, dtype=dtypes.float32, device=lo.device, axis=ax), Tensor.empty(T, NJ, dtype=dtypes.float32, device=lo.device, axis=ax),
          Tensor.empty(T, D//32, 8, dtype=dtypes.uint32, device=lo.device, axis=ax),
          Tensor.empty(T, D//32, dtype=dtypes.float32, device=lo.device, axis=ax),
          Tensor.empty(T, D//32, 2, dtype=dtypes.float32, device=lo.device, axis=ax))
  ret = Tensor.custom_kernel(*outs, lo.float().contiguous(), up_raw, xn, part,
                            fxn=functools.partial(_ghc_decode_mix_kernel, count=count, D=D, R=R))[:5]
  if not sharded: q8_quantize_known(ret[0], T, D, False, tuple(ret[2:]))
  return ret[0], ret[1]

@functools.cache
def _ple_gather_kernel(out:UOp, ids:UOp, *heads:UOp, rb:int) -> UOp:
  # out (R, H*rb) <- row ids[r, j] (rb bytes) of head j's table, for every row r and head j (a lane a byte)
  R, H = cast(tuple[int, int], ids.shape)
  r, b = UOp.range(R, 0, AxisType.GLOBAL), UOp.range(rb, 1, AxisType.LOCAL)
  stores = [out[r, j*rb + b].store(heads[j][ids[r, j].load().cast(dtypes.weakint) * rb + b].load()) for j in range(H)]
  return UOp.group(*stores).end(r, b).sink(arg=KernelInfo(name="ple_gather", opts_to_apply=()))

def ple_gather(ids:Tensor, heads:list[Tensor], rb:int) -> Tensor:
  """the packed rows (R, H*rb) uint8 of the rows ids (R, H) int32 of every head's table (heads: (rows*rb,) uint8 each, in host memory)"""
  out = Tensor.empty(int(ids.shape[0]), len(heads) * rb, dtype=dtypes.uint8, device=ids.device)
  return Tensor.custom_kernel(out, ids.cast(dtypes.int32).contiguous(), *heads, fxn=functools.partial(_ple_gather_kernel, rb=rb))[0]

def slot_gather(pool:Tensor, slots:Tensor, starts:Tensor, S:int, dtype:DType) -> Tensor:
  """the states of G sequences at their positions from their slots: (G, rows, F) of rows slots[g]*S + starts[g]%S of pool (n*S, rows, F)"""
  out = Tensor.empty(int(slots.shape[0]), *pool.shape[1:], dtype=dtype, device=pool.device,
                     axis=None if pool.uop.axis is None else pool.uop.axis)
  return Tensor.custom_kernel(out, pool, slots, starts, fxn=functools.partial(_slot_gather_kernel, S=S))[0]


# ** an fp8 latent cache (SEQ_FP8): a row of 512 latent dims is 128 words of e4m3 values and 4 float scales (of 128 dims each, their absolute
# max over 448), 528 bytes instead of 1024


# ******** the MLA latent cache split on its positions over the devices (sequence sharded): a block of SEQ_BLK positions a device in turn ********
# position p is on device (p // SEQ_BLK) % n, row (p // (SEQ_BLK*n)) * SEQ_BLK + p % SEQ_BLK of its part: the positions of a context of any length
# are on every device. a device attends all the heads over its keys, the devices' outputs are combined by their max and sum

SEQ_BLK = 64

# (WIDE) the slots a sequence of batched decoding spans at most: its positions past its slot's are the next slots' (not decoded meanwhile)


@functools.cache
@functools.cache
def _rows_shift_kernel(out:UOp, x:UOp, bounds:UOp, carry:UOp, dev:UOp, *, lane:int, N:int) -> UOp:
  # out row r: x's row r-1 (a device's first: the last row of the device before it (bounds (lanes*devices, W), lane-major), the first device's
  # of lane 0 the carry, of another lane the last device's of the lane before it)
  m, W = cast(int, x.shape[0]), cast(int, x.shape[1])
  # (the columns in workgroups of 256 threads: a thread a row's W columns in a loop was 0.9 ms a 576-row prefill chunk)
  rngs = (UOp.range(W // 256, 1, AxisType.GLOBAL), UOp.range(256, 2, AxisType.LOCAL)) if W % 256 == 0 else (UOp.range(W, 1, AxisType.LOOP),)
  c = rngs[0] * 256 + rngs[1] if len(rngs) == 2 else rngs[0]
  r = UOp.range(m, 0, AxisType.GLOBAL)
  d = dev[0].load().cast(dtypes.weakint)
  prev_b = (d > 0).where(lane*N + d - 1, (lane - 1)*N + N - 1) if lane else (d - 1).maximum(0)
  first = r.eq(0)
  from_carry = first & d.eq(0) & UOp.const(lane == 0, dtypes.bool)
  v = from_carry.where(carry[c].load().float(), first.where(bounds[prev_b.maximum(0), c].load().float(), x[(r - 1).maximum(0), c].load().float()))
  return out[r, c].store(v.cast(out.dtype)).end(r, *rngs).sink(arg=KernelInfo(name="rows_shift", opts_to_apply=()))

@functools.cache
def _last_row_kernel(out:UOp, x:UOp) -> UOp:
  W = cast(int, x.shape[1])
  c = UOp.range(W, 0, AxisType.GLOBAL)
  return out[0, c].store(x[cast(int, x.shape[0]) - 1, c].load()).end(c).sink(arg=KernelInfo(name="last_row", opts_to_apply=()))

@functools.cache
def _row_pick_kernel(out:UOp, x:UOp, dev:UOp, *, row:UOp, row0:int) -> UOp:
  # out (1, W): x's row (of the rows split over the devices, a device's m from row0 + its index * m) row if it's this device's, else zeros
  m, W = cast(int, x.shape[0]), cast(int, x.shape[1])
  c = UOp.range(W, 0, AxisType.GLOBAL)
  local = row - row0 - dev[0].load().cast(dtypes.weakint) * m
  ok = (local >= 0) & (local < m)
  return out[0, c].store(ok.where(x[local.maximum(0).minimum(m-1), c].load().float(), 0.0).cast(out.dtype)).end(c).sink(
    arg=KernelInfo(name="row_pick", opts_to_apply=()))

def _lane_call(lane:int, fn:Callable, *args):
  # (an exchange on a lane of its own: two that can run at once on one lane's mailbox and counter hang)
  prev, FLAG_LANE[0] = FLAG_LANE[0], lane
  try: return fn(*args)
  finally: FLAG_LANE[0] = prev

def rows_shifted(xs:list[Tensor], carry:Tensor) -> list[Tensor]:
  """
  xs: the micro batches' rows (lane after lane), each (rows, W) split on the devices: each one row later (a row the one before it, the very first
  carry (W,)): every device's last row gathered (small, a lane's on its lane), every device its own rows
  """
  devs = cast(tuple[str, ...], xs[0].device)
  W = int(xs[0].shape[1])
  lasts = [Tensor.custom_kernel(Tensor.empty(len(devs), W, dtype=x.dtype, device=devs, axis=0), x, fxn=_last_row_kernel)[0] for x in xs]
  bounds = Tensor.cat(*[_lane_call(i, all_gather, lt, 0) for i, lt in enumerate(lasts)], dim=0).contiguous()
  di = _device_index(devs)
  return [Tensor.custom_kernel(Tensor.empty(*x.shape, dtype=x.dtype, device=devs, axis=0), x, bounds, carry.contiguous(), di,
                               fxn=functools.partial(_rows_shift_kernel, lane=i, N=len(devs)))[0] for i, x in enumerate(xs)]

def rows_pick(xs:list[Tensor], row:UOp, var:UOp, lane0:int) -> Tensor:
  """the row `row` (an expression of the bound variable var) of the micro batches' rows xs ((rows, W) each, split on the devices, lane after
  lane): (1, W) on every device (every device its part, zeros if not its row, gathered on lanes from lane0 and summed)"""
  devs = cast(tuple[str, ...], xs[0].device)
  di, n, W, tot = _device_index(devs), int(xs[0].shape[0]), int(xs[0].shape[1]), None
  for i, x in enumerate(xs):
    part = Tensor.custom_kernel(Tensor.empty(len(devs), W, dtype=dtypes.float32, device=devs, axis=0), Tensor(x.uop.after(var)), di,
                                fxn=functools.partial(_row_pick_kernel, row=_unbind(row), row0=i*n))[0]
    g = _lane_call(lane0 + i, all_gather, part, 0).sum(axis=0, keepdim=True)
    tot = g if tot is None else tot + g
  return cast(Tensor, tot)

@functools.cache
def _device_index(devs:tuple[str, ...]) -> Tensor:
  # the index of every device on it (a kernel's own part). (once: a realize while a jit is traced walks every tensor)
  return Tensor(list(range(len(devs))), dtype=dtypes.int32).shard(devs, 0).realize()


# ******** gated delta net: fused recurrent scan ********

@functools.cache
def _gated_delta_prefill_lds_kernel(core:UOp, q:UOp, k:UOp, v:UOp, beta:UOp, alpha:UOp, state:UOp, kq:UOp, start_pos:UOp|None=None,
                                    end_pos:UOp|None=None, TC:int=16, W:int=8) -> UOp:
  # _gated_delta_prefill_kernel with W waves a workgroup (W*4 value rows of a head): the operands of TC tokens are staged in LDS once for the
  # waves (every wave of a head read them all from L2), the waves step through them. the same arithmetic
  # a value row over L lanes (key_dim/L keys a lane, its dots summed over them), row_tile rows a lane
  L = 16
  batch, heads, tokens, value_dim, row_tile = *core.shape, (4 if L == 32 else 2)
  key_dim, alpha_dim = q.shape[-1], alpha.shape[-1] if len(alpha.shape) == 4 else 1
  batch, heads, tokens, value_dim, key_dim = cast(tuple[int, int, int, int, int], (batch, heads, tokens, value_dim, key_dim))
  core, v = (x.reshape(batch*heads, tokens, value_dim) for x in (core, v))
  q, k = (x.reshape(batch*heads, tokens, key_dim) for x in (q, k))
  beta, kq = (x.reshape(batch*heads, tokens) for x in (beta, kq))
  alpha = alpha.reshape(batch*heads, tokens, alpha_dim)
  S = cast(int, state.shape[0]) if end_pos is not None else 1
  read_off, write_off = (0, 0) if end_pos is None else ((cast(UOp, start_pos) % S) * (batch*heads), (end_pos % S) * (batch*heads))
  state = state.reshape(-1, value_dim, key_dim)
  RW = W * (32 // L) * row_tile  # the rows of a workgroup
  bh_blk = UOp.range(batch*heads*value_dim//RW, 0, AxisType.GLOBAL)
  wave, lane = UOp.range(W, 1, AxisType.LOCAL), UOp.range(32, -1, AxisType.WARP)
  bh, blk_row = bh_blk // (value_dim//RW), (bh_blk % (value_dim//RW))*RW
  tid = wave*32 + lane
  kl, rsub, per = lane % L, lane // L, key_dim // L
  wrow = (wave * (32 // L) + rsub) * row_tile  # the first row of the lane in the workgroup's
  rows = tuple(blk_row + wrow + i for i in range(row_tile))
  cols = tuple(kl + i*L for i in range(per))
  def tree(xs:list[UOp]) -> UOp:
    while len(xs) > 1: xs = [xs[i] + xs[i+1] if i+1 < len(xs) else xs[i] for i in range(0, len(xs), 2)]
    return xs[0]
  def lane_sum(x:UOp) -> UOp:
    for off in (16, 8, 4, 2, 1):
      if off < L: x = x + _lane_xor(x, off)
    return x
  current = UOp.alloc((row_tile*per,), dtypes.float32, addrspace=AddrSpace.REG)
  initial = None if start_pos is None else start_pos.eq(0)
  current = current.after(current.store(UOp.stack(*(state[read_off+bh, row, col].float() if initial is None else
    initial.where(0, state[read_off+bh, row, col].float()) for row in rows for col in cols))))
  chunk = UOp.range(tokens // TC, 2, AxisType.LOOP)
  lds_k, lds_q = (UOp.alloc((TC, key_dim), dtypes.float32, addrspace=AddrSpace.LOCAL) for _ in range(2))
  lds_a = UOp.alloc((TC, alpha_dim), dtypes.float32, addrspace=AddrSpace.LOCAL)
  lds_v = UOp.alloc((TC, RW), dtypes.float32, addrspace=AddrSpace.LOCAL)
  lds_s = UOp.alloc((TC, 2), dtypes.float32, addrspace=AddrSpace.LOCAL)  # beta, kq
  threads = W*32
  def stage(c:UOp) -> list[tuple[UOp, tuple[UOp, UOp], UOp]]:
    # the operands of chunk c: (LDS buffer, index, value loaded)
    out = []
    for src, lds, width in ((k, lds_k, key_dim), (q, lds_q, key_dim), (alpha, lds_a, alpha_dim)):
      for e0 in range(0, TC*width, threads):
        e = tid + e0
        out.append((lds, ((e//width).valid(e < TC*width), e % width), src[bh, c*TC + (e//width).minimum(TC-1), e % width].load()))
    for e0 in range(0, TC*RW, threads):
      e = tid + e0
      out.append((lds_v, ((e//RW).valid(e < TC*RW), e % RW), v[bh, c*TC + (e//RW).minimum(TC-1), blk_row + e % RW].load()))
    out += [(lds_s, (tid.valid(tid < TC), UOp.const(0, dtypes.weakint)), beta[bh, c*TC + tid.minimum(TC-1)].load()),
            (lds_s, (tid.valid(tid < TC), UOp.const(1, dtypes.weakint)), kq[bh, c*TC + tid.minimum(TC-1)].load())]
    return out
  bufs = (lds_k, lds_q, lds_a, lds_v, lds_s)
  staged = UOp.group(*[l[i].store(val) for l, i, val in stage(chunk)]).barrier()
  lds_k, lds_q, lds_a, lds_v, lds_s = (x.after(staged) for x in bufs)
  t = UOp.range(TC, 3, AxisType.LOOP)
  token = chunk*TC + t
  keys = tuple(lds_k[t, col].load() for col in cols)
  queries = tuple(lds_q[t, col].load() for col in cols)
  updates, stores = [], []
  for row_idx,row in enumerate(rows):
    previous = tuple(current.after(t)[row_idx*per+i].load() for i in range(per))
    decayed, bv = tuple(x * lds_a[t, col if alpha_dim > 1 else 0].load() for x,col in zip(previous, cols)), lds_s[t, 0].load()
    if L == 32:
      state_k = warp_reduce(sum((x*y for x,y in zip(decayed, keys)), UOp.const(0, dtypes.float32)), full_wave=True)
      state_q = warp_reduce(sum((x*y for x,y in zip(decayed, queries)), UOp.const(0, dtypes.float32)), full_wave=True)
    else:
      state_k = lane_sum(tree([x*y for x,y in zip(decayed, keys)]))
      state_q = lane_sum(tree([x*y for x,y in zip(decayed, queries)]))
    delta = (lds_v[t, wrow + row_idx].load() - state_k) * bv
    updates += [x + delta*y for x,y in zip(decayed, keys)]
    stores.append(core[bh, token, row.valid(kl.eq(0))].store(state_q + delta*lds_s[t, 1].load()))
  else: step = UOp.group(*stores, current.store(UOp.stack(*updates))).end(t).barrier().end(chunk)
  state_stores = (state[write_off+bh, row, col].store(current.after(step)[row_idx*per+i].load().cast(state.dtype))
                  for row_idx,row in enumerate(rows) for i,col in enumerate(cols))
  return UOp.group(*state_stores).end(lane, wave, bh_blk).sink(arg=KernelInfo(name="gated_delta_prefill", opts_to_apply=()))

# ******** gated delta rule prefill in chunks: the tokens of a chunk at once from the state before it (exact in float) ********
# within a chunk of C tokens from the state S0 (rows the values, columns the keys), with G_t the sums of log2 alpha of its tokens to t (every
# key its own) and u_t the update of token t (its delta): u = P (v - W), W_t = S0 (exp2(G_t) k_t), the outputs o = Q + R (v - W) with
# Q_t = S0 (exp2(G_t) q_t), the state after it S0 exp2(G_C) + sum_t u_t (k_t exp2(G_C - G_t)). P = (I + diag(beta) A)^-1 diag(beta) and R = B P
# of A_tj = sum_c k_t k_j exp2(G_t - G_j) (j < t), B_tj = sum_c q_t k_j exp2(G_t - G_j) (j < t, B_tt = q_t k_t): every exponent at most 0


@functools.cache
def _gated_delta_prefill_kernel(core:UOp, q:UOp, k:UOp, v:UOp, beta:UOp, alpha:UOp, state:UOp, kq:UOp, start_pos:UOp|None=None,
                                end_pos:UOp|None=None) -> UOp:
  # end_pos: the state is (S, B, H, V, K), the state at a position p is p%S: read at start_pos, written at end_pos
  batch, heads, tokens, value_dim, row_tile = *core.shape, 4
  key_dim, alpha_dim = q.shape[-1], alpha.shape[-1] if len(alpha.shape) == 4 else 1
  assert all(isinstance(x, int) for x in (batch, heads, tokens, value_dim, key_dim)) and key_dim % 32 == 0 and value_dim % row_tile == 0
  batch, heads, tokens, value_dim, key_dim = cast(tuple[int, int, int, int, int], (batch, heads, tokens, value_dim, key_dim))
  core, v = (x.reshape(batch*heads, tokens, value_dim) for x in (core, v))
  q, k = (x.reshape(batch*heads, tokens, key_dim) for x in (q, k))
  beta, kq = (x.reshape(batch*heads, tokens) for x in (beta, kq))
  alpha = alpha.reshape(batch*heads, tokens, alpha_dim)
  # the rows of the state read and written: offsets in the (2*B*H, V, K) state with parities
  S = cast(int, state.shape[0]) if end_pos is not None else 1
  read_off, write_off = (0, 0) if end_pos is None else ((cast(UOp, start_pos) % S) * (batch*heads), (end_pos % S) * (batch*heads))
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


DECODE_THREADS = 1024

# a ballot is across the wave: a CUSTOM (a statement of its own), not inlined where it is used (in the branch of a gated store)
def _ballot(pred:UOp) -> UOp: return _cfn(False, src=(pred,), arg=("__builtin_amdgcn_ballot_w32({0})", dtypes.uint32))
def _popcount(x:UOp) -> UOp: return _cfn(True, src=(x,), arg=("__builtin_popcount({})", dtypes.int32))
# across the wave: CUSTOMs (statements of their own, not inlined in the branch of a gated store)
def _bpermute(x:UOp, src_lane:UOp) -> UOp:
  return _cfn(False, src=(x, src_lane), arg=("__builtin_amdgcn_ds_bpermute((int)({1})*4, (int)({0}))", dtypes.int32))
def _readlane(x:UOp, lane:int) -> UOp: return _cfn(False, src=(x,), arg=(f"__builtin_amdgcn_readlane((int)({{0}}), {lane})", dtypes.int32))
def _wave_scan(x:UOp, lane:UOp) -> UOp:
  """the inclusive prefix sums of x (int32) over the lanes of a wave"""
  for d in (1, 2, 4, 8, 16): x = x + (lane >= d).where(_bpermute(x, (lane - d).maximum(0)), UOp.const(0, dtypes.int32))
  return x

def _radix_kth(key:Callable[[UOp], UOp], npc:UOp, k:int, pools:UOp, t:UOp, wave:UOp, lane:UOp) -> UOp:
  """
  the k-th largest of the keys of a workgroup's threads (key(c) a thread's of chunk c of npc, uint32, 1 at least, all of them with no more pools
  than k): a radix select, 4 passes of 8 bits from the top. a pass counts the digits of the keys whose higher digits are the prefix found (a
  histogram in LDS, atomics), one wave scans it from the top for the bin of the k-th (the keys left), the next pass that bin's. the same as a
  bisection bit by bit, 8 barriers
  """
  zero = UOp.const(0, dtypes.int32)
  hist = UOp.alloc((4*256,), dtypes.int32, addrspace=AddrSpace.LOCAL)
  hist = hist.after(UOp.group(*(hist[(t + i*DECODE_THREADS).valid(t + i*DECODE_THREADS < 4*256)].store(zero)
                                for i in range((4*256 + DECODE_THREADS - 1) // DECODE_THREADS))).barrier())
  res = UOp.alloc((4, 2), dtypes.int32, addrspace=AddrSpace.LOCAL)
  prefix, left = UOp.const(0, dtypes.uint32), UOp.const(k, dtypes.int32)
  for d in range(4):
    c = UOp.range(npc, 20 + d, AxisType.LOOP)
    kv = key(c)
    match = (kv >> (32 - 8*d)).eq(prefix) if d else UOp.const(True, dtypes.bool)
    digit = ((kv >> (24 - 8*d)) & 255).cast(dtypes.weakint)
    add = _sys("__hip_atomic_fetch_add({0}, {1}, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);", dtypes.void, hist[d*256 + digit],
               match.cast(dtypes.int32))
    hd = hist.after(UOp.group(add).end(c).barrier())
    # lane L the bins 255-8L-j (j 0..7): from the top
    bins = [hd[d*256 + 255 - lane*8 - j].load() for j in range(8)]
    lane_sum = functools.reduce(lambda a, b: a + b, bins)
    incl = _wave_scan(lane_sum, lane)
    cum, dsel, before = incl - lane_sum, zero, zero
    for j in range(8):
      hit = (cum < left) & (cum + bins[j] >= left)
      dsel, before = hit.where((255 - lane*8 - j).cast(dtypes.int32), dsel), hit.where(cum, before)
      cum = cum + bins[j]
    found = ((incl - lane_sum) < left) & (incl >= left) & wave.eq(0)
    res = res.after(UOp.group(res[d, UOp.const(0, dtypes.weakint).valid(found)].store(dsel),
                              res[d, UOp.const(1, dtypes.weakint).valid(found)].store(left - before)).barrier())
    prefix, left = (prefix << 8) | res[d, 0].load().cast(dtypes.uint32), res[d, 1].load()
  return (pools > k).where(prefix, UOp.const(0, dtypes.uint32)).maximum(1)

@functools.cache
def _pool_index_kernel(index:UOp, scores:UOp, *grouped:UOp, q_start:UOp, pools:UOp, k:int, kpool:int, row0:int=0, dev_rows:int=0,
                       compact:bool=False) -> UOp:
  # (compact: the list (k+2) the picked pools, their count at k, the query's position at k+1 (the keys: a pool's kpool, then the tail of the
  # query's pool: see qsa_decode's kpool))
  # a workgroup per query row of scores (the query of row r is at q_start+r, its list row row0+r of index), thread t holds pool c*THREADS + t
  # of every chunk c of THREADS pools. the
  # key of the k-th best by a bisection (the counts of keys >= a candidate: per thread, the waves' through LDS), then the picked pools in
  # order: the picks of every (chunk, wave) (ballots), scans of them for their offsets. the tail of the query's pool after them, the count
  # after the list. the loops over the chunks run over the chunks with pools: the work follows the context, not its max
  rows, P = cast(tuple[int, int], scores.shape)
  n_max, THREADS = cast(int, index.shape[1]) - 1, DECODE_THREADS
  per, W = (P + THREADS - 1) // THREADS, THREADS // WARP_SIZE
  assert W == WARP_SIZE, "a scan of a chunk's waves is a wave"
  row, wave, lane = UOp.range(rows, 0, AxisType.GLOBAL), UOp.range(W, 1, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  t, q, index = wave * WARP_SIZE + lane, q_start + row, index.shrink(((row0, row0 + rows), None))
  if dev_rows:
    # (the rows split over the devices: this one's are dev_rows from its index's, the device's index the source)
    q, grouped = q + grouped[0][0].load().cast(dtypes.weakint) * dev_rows, ()
  if grouped:
    # (positions): the rows of G sequences (rows//G each, the first at positions[g])
    Tg = rows // cast(int, grouped[0].shape[0])
    start = grouped[0][row // Tg].load().cast(dtypes.weakint)
    q, pools = start + row % Tg, (start + Tg + kpool - 1) // kpool
  npc = ((pools + THREADS - 1) // THREADS).minimum(per)
  def key(c:UOp) -> UOp:
    # the order preserving key of a complete pool before the query, 0 (the lowest) for the others
    p = t + c * THREADS
    ok = (p < pools) & ((p + 1) * kpool <= q + 1)
    bits = scores[row, p.minimum(P-1).valid(ok)].load().bitcast(dtypes.uint32)
    return ok.where(_order_key(bits), UOp.const(0, dtypes.uint32))
  thr = _radix_kth(key, npc, k, pools, t, wave, lane)
  def write(c3:UOp, pos:UOp, pk:UOp) -> UOp:
    p = t + c3 * THREADS
    if compact: return index[row, pos.valid(pk & (pos < k))].store(p.cast(dtypes.int32))
    return UOp.group(*(index[row, (pos * kpool + j).valid(pk & (pos < k))].store((p * kpool + j).cast(dtypes.int32)) for j in range(kpool)))
  stores, picked = _pick_stores(key, npc, per, thr, t, wave, lane, write)
  total_picked = picked.minimum(k)  # the count of picked pools
  if compact:
    stores += [index[row, UOp.const(k, dtypes.weakint).valid(t.eq(0))].store(total_picked.cast(dtypes.int32)),
               index[row, UOp.const(k+1, dtypes.weakint).valid(t.eq(0))].store(q.cast(dtypes.int32))]
    est = Estimates(rows*P*32, rows*P*4*4, rows*(k+2)*4)
    return UOp.group(*stores).end(row, wave, lane).sink(arg=KernelInfo(name="pool_index", opts_to_apply=(), estimates=est))
  tail0 = (q + 1) // kpool * kpool
  ntail = q + 1 - tail0
  stores.append(index[row, (total_picked * kpool + t).valid(t < ntail)].store((tail0 + t).cast(dtypes.int32)))
  stores.append(index[row, UOp.const(n_max, dtypes.weakint).valid(t.eq(0))].store((total_picked * kpool + ntail).cast(dtypes.int32)))
  est = Estimates(rows*P*32, rows*P*4*4, rows*(k+1)*kpool*4)  # data dependent loops: an upper bound
  return UOp.group(*stores).end(row, wave, lane).sink(arg=KernelInfo(name="pool_index", opts_to_apply=(), estimates=est))

def _pick_stores(key:Callable[[UOp], UOp], npc:UOp, per:int, thr:UOp, t:UOp, wave:UOp, lane:UOp,
                 write:Callable[[UOp, UOp, UOp], UOp]) -> tuple[list[UOp], UOp]:
  """
  the elements (thread t of chunk c of a workgroup of DECODE_THREADS, npc chunks of the per) whose key is at least thr, in their order:
  write(c, pos, picked) the stores of a chunk's (pos: its offset). the stores and the count. (the picks of every (chunk, wave) by ballots,
  scans of them for their offsets)
  """
  W, zero = DECODE_THREADS // WARP_SIZE, UOp.const(0, dtypes.int32)
  # the picks of every (chunk, wave): the counts of a ballot
  wc = UOp.alloc((per, W), dtypes.int32, addrspace=AddrSpace.LOCAL)
  ballots = UOp.alloc((per, W), dtypes.uint32, addrspace=AddrSpace.LOCAL)
  c2 = UOp.range(npc, 5, AxisType.LOOP)
  bits = _ballot(key(c2) >= thr)
  init = UOp.group(wc[c2, wave.valid(lane.eq(0))].store(_popcount(bits)),
                   ballots[c2, wave.valid(lane.eq(0))].store(bits)).end(c2).barrier()
  wc, ballots = wc.after(init), ballots.after(init)
  # the offsets of the (chunk, wave)s in pool order: in every chunk a scan of its waves' counts (a wave per chunk), then a scan of the
  # chunks' totals (one wave). the chunks without pools count 0
  # (the offsets of the waves in place of their counts: a thread reads its count, then writes its offset. LDS for a long context's pools)
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
  # the picked ones of every chunk at their offsets
  c3 = UOp.range(npc, 6, AxisType.LOOP)
  # Materialize the wave ballot before predicating stores. Simplifying an address under `pk` must not turn ballot(pk) into ballot(true).
  bits = ballots[c3, wave].load()
  pk = ((bits >> lane.cast(dtypes.uint32)) & 1) != 0
  pos = cb[c3].load() + wpre[c3, wave].load() + _popcount(bits & ((UOp.const(1, dtypes.uint32) << lane.cast(dtypes.uint32)) - 1))
  return [write(c3, pos, pk).end(c3)], cb[per].load()

@functools.cache
def _picks_local_kernel(out:UOp, index:UOp, dev:UOp, *, S:int, kpool:int) -> UOp:
  # a compact list of every row (rows, k+2: its pools, their count, the query's position, see _pool_index_kernel) as the one of this device's
  # keys: its pools (the ones p%S of its part, as p//S: its rows of them), their count, and the position whose tail (the keys of the query's
  # pool) is its: the rows of the query's pool if it's this device's, none if not
  rows, k = cast(int, index.shape[0]), cast(int, index.shape[1]) - 2
  per = (k + DECODE_THREADS - 1) // DECODE_THREADS
  row, wave, lane = UOp.range(rows, 0, AxisType.GLOBAL), UOp.range(DECODE_THREADS // WARP_SIZE, 1, AxisType.LOCAL), \
    UOp.range(WARP_SIZE, -1, AxisType.WARP)
  t = wave * WARP_SIZE + lane
  part, n = dev[0].load().cast(dtypes.weakint) % S, index[row, k].load().cast(dtypes.weakint)
  def pool(c:UOp) -> UOp: return index[row, (t + c * DECODE_THREADS).minimum(k - 1)].load().cast(dtypes.weakint)
  def key(c:UOp) -> UOp:
    return ((t + c * DECODE_THREADS < n) & (pool(c) % S).eq(part)).where(UOp.const(1, dtypes.uint32), UOp.const(0, dtypes.uint32))
  def write(c:UOp, pos:UOp, pk:UOp) -> UOp: return out[row, pos.valid(pk & (pos < k))].store((pool(c) // S).cast(dtypes.int32))
  stores, count = _pick_stores(key, UOp.const(per, dtypes.weakint), per, UOp.const(1, dtypes.uint32), t, wave, lane, write)
  qpos = index[row, k + 1].load().cast(dtypes.weakint)
  tp = (qpos + 1) // kpool
  local = (tp // S) * kpool + (tp % S).eq(part).where(qpos + 1 - tp * kpool, 0) - 1
  stores += [out[row, UOp.const(k, dtypes.weakint).valid(t.eq(0))].store(count.cast(dtypes.int32)),
             out[row, UOp.const(k + 1, dtypes.weakint).valid(t.eq(0))].store(local.cast(dtypes.int32))]
  return UOp.group(*stores).end(row, wave, lane).sink(arg=KernelInfo(name="picks_local", opts_to_apply=()))

def picks_local(index:Tensor, S:int, kpool:int) -> Tensor:
  """
  compact lists (rows, k+2) of the pools of every row (on every device) as every device's of its keys (split on the positions over the S
  devices of a KV head: pool p is the one of part p%S, its row (p//S)*kpool of that device's region, see QSABlock.kv_split)
  """
  devs = cast(tuple[str, ...], index.device)
  out = Tensor.empty(len(devs) * int(index.shape[0]), int(index.shape[1]), dtype=dtypes.int32, device=devs, axis=0)
  return Tensor.custom_kernel(out, index, _device_index(devs), fxn=functools.partial(_picks_local_kernel, S=S, kpool=kpool))[0]

# ** the indexer's pooled keys split over the devices (QSAIndexer.split): pool p on device p%n, its row p//n there. a device scores its pools,
# its top k of them (their scores and pools) are exchanged, every device picks the top k of all (the same lists everywhere)

@functools.cache
def _pool_cand_kernel(cand:UOp, scores:UOp, dev:UOp, positions:UOp, *, k:int, kpool:int, n:int) -> UOp:
  # a workgroup per query row of the device's scores of its pools (rows, PL) (the rows of G sequences, the first at positions[g]): its top k
  # complete pools before the query (_pool_index_kernel's selection), their (score, pool) in cand (rows, k, 2), the rest (0, -1)
  rows, PL = cast(tuple[int, int], scores.shape)
  THREADS, W = DECODE_THREADS, DECODE_THREADS // WARP_SIZE
  per = (PL + THREADS - 1) // THREADS
  row, wave, lane = UOp.range(rows, 0, AxisType.GLOBAL), UOp.range(W, 1, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  t = wave * WARP_SIZE + lane
  Tg = rows // cast(int, positions.shape[0])
  start = positions[row // Tg].load().cast(dtypes.weakint)
  q, pools = start + row % Tg, (start + Tg + kpool - 1) // kpool
  me = dev[0].load().cast(dtypes.weakint)
  npl = ((pools - me + n - 1) // n).maximum(0).minimum(PL)
  npc = ((npl + THREADS - 1) // THREADS).minimum(per)
  def key(c:UOp) -> UOp:
    lp = t + c * THREADS
    p = lp * n + me
    ok = (lp < npl) & ((p + 1) * kpool <= q + 1)
    return ok.where(_order_key(scores[row, lp.minimum(PL-1).valid(ok)].load().bitcast(dtypes.uint32)), UOp.const(0, dtypes.uint32))
  thr = _radix_kth(key, npc, k, npl, t, wave, lane)
  def write(c3:UOp, pos:UOp, pk:UOp) -> UOp:
    lp, ok = t + c3 * THREADS, pk & (pos < k)
    return UOp.group(cand[row, pos.valid(ok), 0].store(scores[row, lp.minimum(PL-1)].load()),
                     cand[row, pos.valid(ok), 1].store((lp * n + me).cast(dtypes.float32)))
  stores, picked = _pick_stores(key, npc, per, thr, t, wave, lane, write)
  fill = (t < k) & (t >= picked.minimum(k))
  stores += [cand[row, t.valid(fill), 0].store(UOp.const(0.0, dtypes.float32)), cand[row, t.valid(fill), 1].store(UOp.const(-1.0, dtypes.float32))]
  est = Estimates(rows*PL*32, rows*PL*4*4, rows*k*8)  # data dependent loops: an upper bound
  return UOp.group(*stores).end(row, wave, lane).sink(arg=KernelInfo(name="pool_cand", opts_to_apply=(), estimates=est))

@functools.cache
def _pool_merge_kernel(index:UOp, cand:UOp, positions:UOp, *, k:int, kpool:int) -> UOp:
  # the compact lists (rows, k+2) of _pool_index_kernel (the pools, their count, the query's position) from the candidates of every device
  # (n, rows, c, 2): the top k of all of them (in the devices' order)
  n, rows, c = cast(int, cand.shape[0]), cast(int, cand.shape[1]), cast(int, cand.shape[2])
  THREADS, W = DECODE_THREADS, DECODE_THREADS // WARP_SIZE
  NC = n * c
  per = (NC + THREADS - 1) // THREADS
  row, wave, lane = UOp.range(rows, 0, AxisType.GLOBAL), UOp.range(W, 1, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  t = wave * WARP_SIZE + lane
  Tg = rows // cast(int, positions.shape[0])
  start = positions[row // Tg].load().cast(dtypes.weakint)
  q, pools = start + row % Tg, (start + Tg + kpool - 1) // kpool
  def at(ci:UOp) -> tuple[UOp, UOp, UOp]:
    i = t + ci * THREADS
    return (i // c).minimum(n-1), i % c, i < NC
  def key(ci:UOp) -> UOp:
    j, m, inb = at(ci)
    ok = inb & (cand[j, row, m, 1].load() >= 0)
    return ok.where(_order_key(cand[j, row, m, 0].load().bitcast(dtypes.uint32)), UOp.const(0, dtypes.uint32))
  npc = UOp.const(per, dtypes.weakint)
  thr = _radix_kth(key, npc, k, pools, t, wave, lane)
  def write(c3:UOp, pos:UOp, pk:UOp) -> UOp:
    j, m, _ = at(c3)
    return index[row, pos.valid(pk & (pos < k))].store(cand[j, row, m, 1].load().cast(dtypes.int32))
  stores, picked = _pick_stores(key, npc, per, thr, t, wave, lane, write)
  stores += [index[row, UOp.const(k, dtypes.weakint).valid(t.eq(0))].store(picked.minimum(k).cast(dtypes.int32)),
             index[row, UOp.const(k + 1, dtypes.weakint).valid(t.eq(0))].store(q.cast(dtypes.int32))]
  est = Estimates(rows*NC*32, rows*NC*8*4, rows*(k+2)*4)  # data dependent loops: an upper bound
  return UOp.group(*stores).end(row, wave, lane).sink(arg=KernelInfo(name="pool_merge", opts_to_apply=(), estimates=est))

def seq_index_split(scores:Tensor, positions:Tensor, top:int, kpool:int) -> Tensor:
  """
  seq_index (compact) of the scores of split pooled keys (seq_scores' n): every device's top of its pools, exchanged, the top of all (the same
  lists (rows, top+2) on every device)
  """
  devs = cast(tuple[str, ...], scores.device)
  n, rows = len(devs), int(scores.shape[0]) // len(devs)
  cand = Tensor.empty(n*rows, top, 2, dtype=dtypes.float32, device=devs, axis=0)
  cand = Tensor.custom_kernel(cand, scores, _device_index(devs), positions, fxn=functools.partial(_pool_cand_kernel, k=top, kpool=kpool, n=n))[0]
  allc = all_gather(cand.reshape(n, rows*top*2), 0, dtypes.float32)
  out = Tensor.empty(rows, top + 2, dtype=dtypes.int32, device=devs)
  return Tensor.custom_kernel(out, allc.reshape(n, rows, top, 2).contiguous(), positions,
                              fxn=functools.partial(_pool_merge_kernel, k=top, kpool=kpool))[0]


def _order_key(bits:UOp) -> UOp:
  """an order preserving uint32 key of a float's bits"""
  return (bits >= 0x80000000).where(bits ^ 0xFFFFFFFF, bits | 0x80000000)


@functools.cache
def _index_scores_kernel(out:UOp, q:UOp, pk:UOp, *, H:int, scale:float, pools:UOp, rdna4:bool=False) -> UOp:
  # out (R, P) = the sum over the H heads of relu(q[r, h] . pk[p]) * scale for the pools below `pools` (the others not written), q (R*H, D),
  # pk (P, D) half on the RDNA3 tensor cores: a workgroup 64 (row, head) rows (16 rows) and 128 pools, four waves (2 along each), a K stage
  # of 64 in LDS. the heads of a row are 4 consecutive rows of a 16 row tile: a lane's two of them and its other half's (lanes ^16)
  R, P = cast(tuple[int, int], out.shape)
  D = cast(int, q.shape[-1])
  assert H == 4 and D % 64 == 0 and (R*H) % 64 == 0 and P % 128 == 0
  MT, NT, TT, OT = 64, 128, 2, 4
  mb, nb = UOp.range(R*H // MT, 0, AxisType.GLOBAL), UOp.range(P // NT, 1, AxisType.GLOBAL)
  lane, wave = UOp.range(WARP_SIZE, -1, axis_type=AxisType.WARP), UOp.range(4, 3, axis_type=AxisType.LOCAL)
  col, half, wt, wo, tid = lane % 16, lane // 16, wave // 2, wave % 2, wave * 32 + lane
  # (the pools of the block: none past `pools`, a loop of one or none)
  live = UOp.range((nb * NT < pools).where(1, 0), 5, AxisType.LOOP)
  q, pk = q.reshape(R*H, D), pk.reshape(P, D)
  accs = [[UOp.alloc((8,), dtypes.float32, addrspace=AddrSpace.REG) for _ in range(TT)] for _ in range(OT)]
  accs = [[acc.after(acc.after(live).store(acc.const_like(0))) for acc in row] for row in accs]
  ks = UOp.range(D // 64, 4, AxisType.LOOP)
  lds_a = UOp.alloc((MT, 64 + 8), dtypes.float16, addrspace=AddrSpace.LOCAL)
  lds_b = UOp.alloc((NT, 64 + 8), dtypes.float16, addrspace=AddrSpace.LOCAL)
  st = []
  for k in range(MT * 8 // 128):  # (8 halves a thread a load)
    e = tid + k * 128
    st += [lds_a[e // 8, (e % 8) * 8 + i].store(q[mb * MT + e // 8, ks * 64 + (e % 8) * 8 + i].load()) for i in range(8)]
  for k in range(NT * 8 // 128):
    e = tid + k * 128
    st += [lds_b[e // 8, (e % 8) * 8 + i].store(pk[nb * NT + e // 8, ks * 64 + (e % 8) * 8 + i].load()) for i in range(8)]
  staged = UOp.group(*st).barrier()
  lds_a, lds_b = lds_a.after(staged), lds_b.after(staged)
  cur = [[accs[o][t].after(ks) for t in range(TT)] for o in range(OT)]
  # (RDNA4: a lane's fragment is 8 of the 16, two groups of four by its half of the wave)
  kis = tuple(i % 4 + half * 4 + (i // 4) * 8 for i in range(8)) if rdna4 else tuple(range(16))
  for kh in range(4):
    afr = [UOp.stack(*(lds_a[wt * TT * 16 + t * 16 + col, kh * 16 + i].load() for i in kis)) for t in range(TT)]
    bfr = [UOp.stack(*(lds_b[wo * OT * 16 + o * 16 + col, kh * 16 + i].load() for i in kis)) for o in range(OT)]
    cur = [[UOp.wmma(afr[t], bfr[o], cur[o][t], *WMMA_ARG) for t in range(TT)] for o in range(OT)]
  update = UOp.group(*(accs[o][t].store(cur[o][t]) for o in range(OT) for t in range(TT))).barrier().end(ks)
  stores = []
  for o, t in itertools.product(range(OT), range(TT)):
    a = accs[o][t].after(update)
    pool = nb * NT + wo * OT * 16 + o * 16 + col
    if rdna4:
      # (RDNA4: the rows half*8+j of the tile, a lane's 8 are two rows' 4 heads: their sum in the lane)
      for jj in range(2):
        tot = functools.reduce(lambda x, y: x + y, [a[jj*4 + h].load().maximum(0.0) for h in range(4)]) * scale
        stores.append(out[(mb * MT + wt * TT * 16 + t * 16) // H + half * 2 + jj, pool].store(tot))
      continue
    for j in range(4):
      # (rows 2i+half of the tile: row j's heads half and 2+half in this lane, the other two in lane ^16)
      sj = a[2*j].load().maximum(0.0) + a[2*j+1].load().maximum(0.0)
      # (the other half's lane by bpermute: permlanex16's selection isn't lane ^ 16)
      other = _bpermute(sj.bitcast(dtypes.int32), lane ^ 16).bitcast(dtypes.float32)
      tot = (sj + other) * scale
      row = (mb * MT + wt * TT * 16 + t * 16) // H + j
      stores.append(out[row.valid(half.eq(0)), pool].store(tot))
  return UOp.group(*stores).end(live).end(mb, nb, lane, wave).sink(arg=KernelInfo(name="index_scores", opts_to_apply=()))

def index_scores(q:Tensor, pk:Tensor, pools:int|UOp, scale:float) -> Tensor|None:
  """(R, P) float: the sum over the heads of relu(q (R, H, D) . pk (P, D)) * scale for the pools below `pools` (the others not written)"""
  R, H, D = cast(tuple[int, int, int], q.shape)
  P = int(pk.shape[0])
  if H != 4 or D % 64 or P % 128 or (int(q.uop.shard_shape[0]) if isinstance(q.device, tuple) and q.uop.axis == 0 else R) % 16 or \
     not amd_custom_kernels_supported(q.device): return None
  pu = _unbind(pools) if isinstance(pools, UOp) else UOp.const(pools, dtypes.weakint)
  out = Tensor.empty(R, P, dtype=dtypes.float32, device=q.device, axis=0 if isinstance(q.device, tuple) and q.uop.axis == 0 else None)
  qh = q.reshape(R*H, D).half().contiguous()
  if isinstance(pools, UOp): qh = Tensor(qh.uop.after(pools))
  return Tensor.custom_kernel(out, qh, pk.half(), fxn=functools.partial(_index_scores_kernel, H=H, scale=scale, pools=pu,
                                                                         rdna4=_wmma_rdna4(q.device)))[0]

def pool_index_rows(scores:Tensor, q_start:UOp, pools:int|UOp, top:int, kpool:int, row0:int=0, out:Tensor|None=None,
                    compact:bool=False) -> Tensor:
  """
  (rows, top*kpool+kpool+1) int32: for every query row (at q_start+r) the list of the keys it sees (its top pools, its tail), the count last.
  out: the lists of more rows, the ones of the scores are rows row0.. of it (the others kept). compact: (rows, top+2) the pools (see
  _pool_index_kernel)
  """
  rows = int(scores.shape[0])
  # (scores split on their rows over the devices: every device the lists of its rows, split so)
  split = isinstance(scores.device, tuple) and scores.uop.axis == 0
  if out is None:
    out = Tensor.empty(rows, top + 2 if compact else top*kpool + kpool + 1, dtype=dtypes.int32, device=scores.device, axis=0 if split else None)
  pu = _unbind(pools) if isinstance(pools, UOp) else UOp.const(pools, dtypes.weakint)
  sc = Tensor(scores.uop.after(q_start, *([pools] if isinstance(pools, UOp) else [])))
  dev_rows = rows // len(cast(tuple, scores.device)) if split else 0
  fxn = functools.partial(_pool_index_kernel, q_start=_unbind(q_start) + row0, pools=pu, k=top, kpool=kpool, row0=row0, dev_rows=dev_rows,
                          compact=compact)
  return Tensor.custom_kernel(out, sc, *([_device_index(cast(tuple[str, ...], scores.device))] if split else []), fxn=fxn)[0]


# ******** sequences sharing the caches (each its region of rows, see Transformer.regions): rows gathered and scattered by index, the indexer's
# scores of every sequence's own pools ********

@functools.cache
def _rows_gather_kernel(out:UOp, table:UOp, idx:UOp) -> UOp:
  # out[r] (n, F) <- table[idx[r]] (P, F), cast to out's dtype
  n, F = cast(tuple[int, int], out.shape)
  P, L = cast(int, table.shape[0]), math.gcd(F, 256)
  r, blk, lane = UOp.range(n, 0, AxisType.GLOBAL), UOp.range(F // L, 1, AxisType.GLOBAL), UOp.range(L, 2, AxisType.LOCAL)
  f, row = blk * L + lane, idx[r].load().cast(dtypes.weakint).maximum(0).minimum(P-1)
  return out[r, f].store(table[row, f].load().cast(out.dtype)).end(r, blk, lane).sink(arg=KernelInfo(name="rows_gather", opts_to_apply=()))

def kv_q4_pack(x:Tensor) -> Tensor:
  """
  rows x (..., D) as 4-bit groups of 32 (Q4_0's: the value of the largest magnitude -8, the scale its -1/8, the rest rounded into -8..7) in
  words (..., D//8 + D//64) uint32: the values + 8, 8 a word (value j of a group: word j//8, byte j%4, its high nibble if j%8 >= 4), then the
  groups' half scales, two a word (the attention's 4-bit cache: QSA_KV_BITS=4)
  """
  D = int(x.shape[-1])
  lead = tuple(x.shape[:-1])
  g = x.float().reshape(*lead, D//32, 32)
  mx, mn = g.max(-1, keepdim=True), g.min(-1, keepdim=True)
  d = ((mx >= -mn).where(mx, mn) / -8).cast(dtypes.half)
  df = d.float()
  q = (g * (df != 0).where(1 / df, 0)).round().clip(-8, 7).cast(dtypes.int32) + 8
  b = q.cast(dtypes.uint32).reshape(*lead, D//8, 2, 4)
  words = functools.reduce(lambda a, i: a | (b[..., i // 4, i % 4] << (8*(i % 4) + 4*(i // 4))), range(1, 8), b[..., 0, 0])
  s = d.reshape(*lead, D//64, 2).bitcast(dtypes.uint16).cast(dtypes.uint32)
  return words.cat(s[..., 0] | (s[..., 1] << 16), dim=-1)

def kv_q8_pack(x:Tensor) -> Tensor:
  """
  rows x (..., D) as int8 groups of 32 with their scale (absmax/127), in words (..., D//4 + D//32) uint32: the values' (4 a word, in order, each
  + 128: a byte), then the groups' float scales (the attention's int8 cache: SEQ_KV_Q8)
  """
  D = int(x.shape[-1])
  g = x.float().reshape(*x.shape[:-1], D//32, 32)
  sc = g.abs().max(-1, keepdim=True) / 127
  q = (g / sc.maximum(1e-30)).round().clip(-127, 127).cast(dtypes.int32).reshape(*x.shape[:-1], D//4, 4)
  b = (q + 128).cast(dtypes.uint32)
  words = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16) | (b[..., 3] << 24)
  return words.cat(sc.reshape(*x.shape[:-1], D//32).bitcast(dtypes.uint32), dim=-1)


@functools.cache
def _q8_embed_kernel(out:UOp, raw:UOp, ids:UOp) -> UOp:
  # out (R, D) <- the rows ids (R,) of a Q8_0 table raw (bytes: 34 a block of 32), dequantized as the gguf loader does (in float, then half)
  R, D = cast(tuple[int, int], out.shape)
  r, blk, lane = UOp.range(R, 0, AxisType.GLOBAL), UOp.range(D//32, 1, AxisType.GLOBAL), UOp.range(32, 2, AxisType.LOCAL)
  base = ids[r].load().cast(dtypes.weakint)*(D//32*34) + blk*34
  d = (raw[base].load().cast(dtypes.uint16) | (raw[base+1].load().cast(dtypes.uint16) << 8)).bitcast(dtypes.half).float()
  q = raw[base+2+lane].load().bitcast(dtypes.int8).float()
  return out[r, blk*32+lane].store((d*q).cast(dtypes.half).float()).end(r, blk, lane).sink(arg=KernelInfo(name="q8_embed", opts_to_apply=()))

def q8_embed(raw:Tensor, ids:Tensor, dim:int) -> Tensor:
  """the rows ids (n,) of the Q8_0 table raw (uint8, rows of dim) as float (n, dim)"""
  out = Tensor.empty(int(ids.shape[0]), dim, dtype=dtypes.float32, device=ids.device)
  return Tensor.custom_kernel(out, raw, ids.cast(dtypes.int32).contiguous(), fxn=_q8_embed_kernel)[0]

def rows_gather(table:Tensor, idx:Tensor, dtype:DType|None=None, const:bool=False, split:bool=False) -> Tensor:
  """
  the rows idx (n,) int32 of table (P, F): (n, F) (a table split on its rows: every device's of its own). const: the table isn't written while
  a graph is made, the same rows of it gathered once (the blocks' rope rows of a step). split: (devices*n, F) split on its rows (every
  device's rows of its part of the table)
  """
  if const and (ret:=_gather_cache.get(key:=(table.uop, idx.uop, dtype))) is not None: return ret
  N = len(table.device) if split and isinstance(table.device, tuple) else 1
  out = Tensor.empty(N * int(idx.shape[0]), int(table.shape[1]), dtype=dtype or table.dtype, device=table.device, axis=0 if N > 1 else None)
  ret = Tensor.custom_kernel(out, table, idx, fxn=_rows_gather_kernel)[0]
  if const: _gather_cache[key] = ret
  return ret
_gather_cache:dict[tuple, Tensor] = {}

@functools.cache
def _rows_scatter_kernel(buf:UOp, rows:UOp, idx:UOp) -> UOp:
  # buf[idx[r]] (P, F) <- rows[r] (n, F), cast to buf's dtype (an index out of buf's rows isn't written)
  n, F = cast(tuple[int, int], rows.shape)
  P, L = cast(int, buf.shape[0]), math.gcd(F, 256)
  r, blk, lane = UOp.range(n, 0, AxisType.GLOBAL), UOp.range(F // L, 1, AxisType.GLOBAL), UOp.range(L, 2, AxisType.LOCAL)
  f, row = blk * L + lane, idx[r].load().cast(dtypes.weakint)
  return buf[row.valid((row >= 0) & (row < P)), f].store(rows[r, f].load().cast(buf.dtype)).end(r, blk, lane) \
    .sink(arg=KernelInfo(name="rows_scatter", opts_to_apply=()))

def rows_scatter(buf:Tensor, rows:Tensor, idx:Tensor) -> Tensor:
  """rows (n, F) into the rows idx (n,) int32 of buf (P, F): buf after (split on the rows of both: every device its own)"""
  return Tensor.custom_kernel(buf, rows.contiguous(), idx, fxn=_rows_scatter_kernel)[0]

SEQ_SCORE_TILE = 256

@functools.cache
def _seq_scores_kernel(out:UOp, q:UOp, pk:UOp, bases:UOp, positions:UOp, *dev:UOp, R:int, kpool:int, WG:int, n:int=1) -> UOp:
  # (n: the pooled keys split over n devices (pool p on device p%n, its row p//n of the device's, see QSAIndexer.split): a device's scores of
  # its pools, column lp its pool lp*n+device)
  # out[g*R + r, p] (rows, P) = sum over the heads h of relu(q[g*R+r, h] . pk[bases[g]//kpool + p]) * D**-0.5, for the pools p of sequence g up
  # to its last row's (the rows of g at positions[g]..; the others aren't written). WG workgroups a sequence loop over its tiles of pools, a
  # thread a pool (its key in registers), the sequence's queries in LDS
  rows, P = cast(tuple[int, int], out.shape)
  _, H, D = cast(tuple[int, int, int], q.shape)
  PT, G, TILE = cast(int, pk.shape[0]), rows // R, SEQ_SCORE_TILE
  g, w, t = UOp.range(G, 0, AxisType.GLOBAL), UOp.range(WG, 1, AxisType.GLOBAL), UOp.range(TILE, 2, AxisType.LOCAL)
  start = positions[g].load().cast(dtypes.weakint)
  pools = (start + R + kpool - 1) // kpool
  if n > 1: pools = ((pools - dev[0][0].load().cast(dtypes.weakint) + n - 1) // n).maximum(0)
  pools = pools.minimum(P)
  tiles = (pools + TILE - 1) // TILE
  nq = R*H*D
  ql = UOp.alloc((nq,), dtypes.float, addrspace=AddrSpace.LOCAL)
  qf = q.reshape(rows*H*D)
  ld = [ql[(t + i*TILE).valid(t + i*TILE < nq)].store(qf[g*nq + (t + i*TILE).minimum(nq-1)].load().float())
        for i in range((nq + TILE - 1) // TILE)]
  ql = ql.after(UOp.group(*ld).barrier())
  active = UOp.range(((tiles - w + WG - 1) // WG).maximum(0), 3, AxisType.LOOP)
  p = (w + active * WG) * TILE + t
  prow = (bases[g].load().cast(dtypes.weakint) // (kpool * n) + p).minimum(PT-1)
  # the dots of every (row, head) in registers, 8 dims of the key a round
  acc = _reg((R*H,), 0, active)
  dch = UOp.range(D // 8, 4, AxisType.LOOP)
  kv = _vec_load(pk[prow, dch*8], 8)
  prev = acc.after(dch)
  new = [functools.reduce(lambda a, j: a + ql[i*D + dch*8 + j].load() * kv[j], range(8), prev[i].load()) for i in range(R*H)]
  acc = acc.after(acc.store(UOp.stack(*new)).end(dch))
  stores = []
  for r in range(R):
    s = functools.reduce(lambda a, h: a + acc[r*H + h].load().maximum(0), range(1, H), acc[r*H].load().maximum(0))
    stores.append(out[g*R + r, p.valid(p < pools)].store(s * D**-0.5))
  est = Estimates(2*rows*H*D*P, P*D*2*G, rows*P*4)  # (a data dependent count of tiles: an upper bound)
  return UOp.group(*stores).end(active).end(g, w, t).sink(arg=KernelInfo(name="seq_scores", opts_to_apply=(), estimates=est))

def seq_scores(q:Tensor, pk:Tensor, bases:Tensor, positions:Tensor, R:int, kpool:int, P:int, n:int=1) -> Tensor:
  """
  the indexer's scores (G*R, P) of the queries q (G*R, H, D) of G sequences (R rows each, the first at positions[g]) on their pooled keys, the
  rows of pk (all the sequences') from bases[g]//kpool: the columns of the pools up to a sequence's last row written. n: pk split over the n
  devices (pool p on device p%n): every device's of its pools (n*G*R, ceil(P/n)) split on its rows
  """
  rows, devs = int(q.shape[0]), q.device
  if n == 1:
    out = Tensor.empty(rows, P, dtype=dtypes.float32, device=devs)
    return Tensor.custom_kernel(out, q.float().contiguous(), pk, bases, positions, fxn=functools.partial(_seq_scores_kernel, R=R, kpool=kpool,
                                                                                                        WG=16))[0]
  out = Tensor.empty(n*rows, (P + n - 1) // n, dtype=dtypes.float32, device=devs, axis=0)
  fxn = functools.partial(_seq_scores_kernel, R=R, kpool=kpool, WG=16, n=n)
  return Tensor.custom_kernel(out, q.float().contiguous(), pk, bases, positions, _device_index(cast(tuple[str, ...], devs)), fxn=fxn)[0]

def seq_index(scores:Tensor, positions:Tensor, top:int, kpool:int, compact:bool=False) -> Tensor:
  """
  the lists (G*R, top*kpool+kpool+1) of the keys (positions of its sequence) of every query row of seq_scores' scores (the rows of G
  sequences, the first at positions[g]): its top pools, its tail, the count last. compact: (G*R, top+2) the pools (see _pool_index_kernel)
  """
  rows, zero = int(scores.shape[0]), UOp.const(0, dtypes.weakint)
  out = Tensor.empty(rows, top + 2 if compact else top*kpool + kpool + 1, dtype=dtypes.int32, device=scores.device)
  fxn = functools.partial(_pool_index_kernel, q_start=zero, pools=zero, k=top, kpool=kpool, compact=compact)
  return Tensor.custom_kernel(out, scores, positions, fxn=fxn)[0]

KDA_WAVES = 8
# KDA_R, KDA_RW: batched decoding's kda_decode: a head's rows over KDA_R workgroups of KDA_RW waves (0: 4 at 8 or more sequences, else 8)
KDA_R, KDA_RW = getenv("KDA_R", 4), getenv("KDA_RW", 0)

@functools.cache
def _kda_decode_kernel(z:UOp, state:UOp, cq:UOp, ck:UOp, cv:UOp, pq:UOp, pk:UOp, pv:UOp, wq:UOp, wk:UOp, wv:UOp, gate:UOp, alpha:UOp, beta:UOp,
                       dt:UOp, a:UOp, norm:UOp, start_pos:UOp, lb:float|None, eps:float, R:int=0, row0:int=0, G:int=1,
                       positions:UOp|None=None, S:int=0, SC:int=0, slots:UOp|None=None, rp:UOp|None=None, kb:UOp|None=None, Q:int=0) -> UOp:
  # row0: the tokens are rows row0.. of the inputs (a row of batched decoding: the inputs of all of them, not a copy of its own)
  # G: groups (batched decoding: a sequence each, its tokens the rows g*T.., its states the S of g*S.. in the states, its start position the
  # g-th of positions)
  # R: the rows of a head split over R workgroups (each makes the convolutions, norms and decays of the head), z the outputs before their norm
  # (float, kda_out normalizes them): more workgroups, fewer rows in a row
  # a few tokens (T) of a kda layer, a workgroup per head: the short convolutions of q, k and v (shifting their states), the l2 norms of q and
  # k, the decay and the delta rule update of the head's state token by token, the rms norm of the outputs and their sigmoid gates.
  # the states (S, ...) at a position p are p%S: read at start_pos, the ones after every token written (the others keep the states before)
  # (S: the states of a group, the state buffer can hold more groups than the call's)
  # (SC: the conv states' S, default S. a recurrent state of S == T: a thread reads its rows of the one of start_pos before it writes them, the
  # conv states (read by every workgroup of a head, written by the first) need SC > T)
  # (rp: replay (SEQ_REPLAY): a group's state (S=1) is the one after the first token of its last step (at kb[g], its base), and rp the inputs of
  # the tokens after it (k, decay, v, beta of a head, a row (slot*Q + position%Q)*heads + head): the ones up to start_pos replayed, then the
  # step's tokens. the state after its first token written (the next base) and the inputs of its others: reads and writes of the inputs at
  # positions start_pos-T+1.. and ..start_pos+T-1 apart (Q >= 2T): no workgroup reads what another writes)
  S, T, N = S or cast(int, state.shape[0]), cast(int, z.shape[-2]) // G, cast(int, pq.shape[-2])
  SC = SC or S
  heads, V, K = cast(tuple[int, int, int], state.shape[-3:])
  # more tokens: more waves, a wave runs fewer rows through the tokens
  W, conv = KDA_WAVES * (1 if T == 1 else 2), cast(int, wq.shape[-1])
  # (4 waves a workgroup at 8 or more sequences: faster on tr8; it rounds differently from 8 (a numerics change, checked by extra/qdecq.py))
  if R: W = KDA_RW or (4 if G >= 8 else 8)
  assert K == V and W*32 >= K and V % W == 0 and conv == 4, "a thread per channel of q, k and v"
  assert (T <= S or rp is not None) and T < SC, "the states of the T tokens and the one before"
  state, z = state.reshape(-1, V, K), z.reshape(G*T, heads*V)
  cq, ck, cv = (x.reshape(-1, heads*K) for x in (cq, ck, cv))
  pq, pk, pv, gate, alpha = (x.reshape(N, heads*K) for x in (pq, pk, pv, gate, alpha))
  beta = beta.reshape(N, heads)
  ghr, wave, lane = UOp.range(G*heads*max(R, 1), 0, AxisType.GLOBAL), UOp.range(W, 1, AxisType.LOCAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  grp, hr = (ghr // (heads*max(R, 1)), ghr % (heads*max(R, 1))) if G > 1 else (UOp.const(0, dtypes.weakint), ghr)
  if positions is not None: start_pos = positions[grp].load().cast(dtypes.weakint)
  # (slots: the states of group g are slot slots[g]'s, else g's)
  slot = slots[grp].load().cast(dtypes.weakint) if slots is not None else grp
  rpos, wps = start_pos % S, [(start_pos + t + 1) % S for t in range(T)]
  rpc, wpcs, cbase = start_pos % SC, [(start_pos + t + 1) % SC for t in range(T)], slot*SC
  sbase, prow = slot*S, [grp*T + row0 + t for t in range(T)]  # (the group's states and the rows of its tokens)
  qrows, krows, vrows, gate_rows, alpha_rows, beta_rows = ([x[r] for r in prow] for x in (pq, pk, pv, gate, alpha, beta))
  hr = ghr if G == 1 else hr
  h, chunk = (hr // R, hr % R) if R else (hr, UOp.const(0, dtypes.weakint))
  tid, initial = wave * WARP_SIZE + lane, start_pos.eq(0)
  def silu(x:UOp) -> UOp: return x / ((-x * LOG2E).exp2() + 1)
  def sigmoid(x:UOp) -> UOp: return ((-x * LOG2E).exp2() + 1).reciprocal()
  # the convolutions of every token: thread tid < K is channel tid of q, k and v. the state of a channel is its last conv-1 inputs
  lds = UOp.alloc((T, 3, K), dtypes.float, addrspace=AddrSpace.LOCAL)
  ch, active = h * K + tid % K, tid < K
  stores = []
  for i, (cs, p, w) in enumerate(((cq, qrows, wq), (ck, krows, wk), (cv, vrows, wv))):
    inputs = [initial.where(0, cs[(cbase+rpc)*(conv-1) + j, ch].load().float()) for j in range(conv-1)] + [p[t][ch].load().float() for t in range(T)]
    weights = [w[ch, j].load().float() for j in range(conv)]
    for t in range(T):
      window = inputs[t:t+conv]
      out = silu(sum((weights[j] * window[j] for j in range(1, conv)), weights[0] * window[0]))
      stores += [lds[t, i, tid.valid(active)].store(out)] + [cs[(cbase+wpcs[t])*(conv-1) + j, ch.valid(active & chunk.eq(0))].store(
                   window[j+1].cast(cs.dtype)) for j in range(conv-1)]
  lds = lds.after(UOp.group(*stores).barrier())
  # every wave: q and k of the head (key column lane + 32*i), l2 normalized, the decay of every column
  cols = [lane + 32*i for i in range(K // 32)]
  def l2(x:list[UOp]) -> list[UOp]:
    r = (warp_reduce(functools.reduce(lambda s,y: s + y*y, x[1:], x[0]*x[0]), full_wave=True) + 1e-6).sqrt().reciprocal()
    return [y * r for y in x]
  def dot(x:list[UOp], y:list[UOp]) -> UOp: return warp_reduce(functools.reduce(lambda s,ab: s + ab[0]*ab[1], zip(x[1:], y[1:]), x[0]*y[0]),
                                                               full_wave=True)
  qs = [[y * K**-0.5 for y in l2([lds[t, 0, c].load() for c in cols])] for t in range(T)]
  ks = [l2([lds[t, 1, c].load() for c in cols]) for t in range(T)]
  kq = [dot(ks[t], qs[t]) for t in range(T)]
  b = [sigmoid(beta_rows[t][h].load().float()) for t in range(T)]
  A = a.reshape(heads)[h].load().float()
  def decay(t:int, c:UOp) -> UOp:
    g = alpha_rows[t][h*K + c].load().float() + dt.reshape(heads*K)[h*K + c].load().float()
    if lb is None: return ((g.maximum(0) + ((-g.abs() * LOG2E).exp2() + 1).log2() * (1/LOG2E)) * A * LOG2E).exp2()
    return (sigmoid(g * -A) * lb * LOG2E).exp2()
  decays = [[decay(t, c) for c in cols] for t in range(T)]
  # the delta rule on the rows of this wave, token by token
  rows_per_wave = V // (W * max(R, 1))
  lds_o = UOp.alloc((T, V), dtypes.float, addrspace=AddrSpace.LOCAL)
  ostores, sstores, istores = [], [], []
  replay = rp is not None
  if replay:
    assert S == 1 and kb is not None and Q >= 2*T
    rp = cast(UOp, rp).reshape(-1, cast(int, cast(UOp, rp).shape[-1]))
    # (the host's bases are 0..T-1 tokens before start_pos: no clamps in the indexing, see Transformer._kbase_of)
    kbase = kb[grp].load().cast(dtypes.weakint)
    nrep = initial.where(0, start_pos - kbase)
    def irow(p:UOp) -> UOp: return (slot*Q + p % Q)*heads + h
    first = wave.eq(0) & chunk.eq(0)  # (a head's vectors written by one wave)
    for t in range(1, T):
      at = irow(start_pos + t)
      rpu = rp
      istores += [rpu[at, c.valid(first)].store(x) for c, x in zip(cols, ks[t])] + \
                 [rpu[at, (K + c).valid(first)].store(x) for c, x in zip(cols, decays[t])] + \
                 [rpu[at, UOp.const(2*K + V, dtypes.weakint).valid(first & lane.eq(0))].store(b[t])]
  for j in range(rows_per_wave):
    r = (chunk * W + wave) * rows_per_wave + j
    row = [initial.where(0, state[(sbase+rpos)*heads + h, r, c].load().float()) for c in cols]
    if replay:
      for tt in range(T-1):
        at = irow(kbase + tt)
        rpu = cast(UOp, rp)
        kk, dd = [rpu[at, c].load() for c in cols], [rpu[at, K + c].load() for c in cols]
        decayed = [x * d for x, d in zip(row, dd)]
        delta = (rpu[at, 2*K + r].load() - dot(decayed, kk)) * rpu[at, 2*K + V].load()
        on = nrep > tt
        row = [on.where(x + delta * k_, o) for x, k_, o in zip(decayed, kk, row)]
    for t in range(T):
      decayed = [x * d for x, d in zip(row, decays[t])]
      sk, sq = dot(decayed, ks[t]), dot(decayed, qs[t])
      delta = (lds[t, 2, r].load() - sk) * b[t]
      row = [x + delta * kk for x, kk in zip(decayed, ks[t])]
      if not replay or t == 0: sstores += [state[(sbase+wps[t])*heads + h, r, c].store(x.cast(state.dtype)) for c, x in zip(cols, row)]
      if replay and t > 0: istores.append(cast(UOp, rp)[irow(start_pos + t), (2*K + r).valid(lane.eq(0))].store(lds[t, 2, r].load()))
      ostores.append((z[grp*T + t, (h*V + r).valid(lane.eq(0))] if R else lds_o[t, r.valid(lane.eq(0))]).store(sq + delta * kq[t]))
  if R: return UOp.group(*ostores, *sstores, *istores).end(ghr, wave, lane).sink(arg=KernelInfo(name="kda_decode", opts_to_apply=()))
  lds_o = lds_o.after(UOp.group(*ostores).barrier())
  # rms norm of the head's outputs, gated
  zs = []
  for t in range(T):
    o = [lds_o[t, c].load() for c in cols]
    rinv = (warp_reduce(functools.reduce(lambda s,y: s + y*y, o[1:], o[0]*o[0]), full_wave=True) * (1/V) + eps).sqrt().reciprocal()
    r = tid % V
    out = lds_o[t, r].load() * rinv * norm[r].load().float() * sigmoid(gate_rows[t][h*V + r].load().float())
    zs.append(z[grp*T + t, (h*V + r).valid(tid < V)].store(out.cast(z.dtype)))
  return UOp.group(*zs, *sstores, *istores).end(ghr, wave, lane).sink(arg=KernelInfo(name="kda_decode", opts_to_apply=()))

def _kda_replay_kernel(state:UOp, rp:UOp, slot:UOp, base:UOp, n:UOp, *, T:int, Q:int, W:int=8) -> UOp:
  """
  replay (SEQ_REPLAY): the state of slot `slot` (S=1, the one at base) after its n (< T) next tokens of its inputs in rp, written back: the
  state of position base+n (see _kda_decode_kernel)
  """
  heads, V, K = cast(tuple[int, int, int], state.shape[-3:])
  state = state.reshape(-1, V, K)
  RW = cast(int, rp.shape[-1])
  rp = rp.reshape(-1, RW)
  hw, lane = UOp.range(heads * (V // W), 0, AxisType.GLOBAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  h, r = hw // (V // W), hw % (V // W) * W
  cols = [lane + 32*i for i in range(K // 32)]
  def dot(x:list[UOp], y:list[UOp]) -> UOp: return warp_reduce(functools.reduce(lambda s,ab: s + ab[0]*ab[1], zip(x[1:], y[1:]), x[0]*y[0]),
                                                               full_wave=True)
  stores = []
  for j in range(W):
    row = [state[slot*heads + h, r + j, c].load().float() for c in cols]
    for tt in range(T-1):
      at = (slot*Q + (base + tt) % Q)*heads + h
      kk, dd = [rp[at, c].load() for c in cols], [rp[at, K + c].load() for c in cols]
      decayed = [x * d for x, d in zip(row, dd)]
      delta = (rp[at, 2*K + r + j].load() - dot(decayed, kk)) * rp[at, 2*K + V].load()
      on = n > tt
      row = [on.where(x + delta * k_, o) for x, k_, o in zip(decayed, kk, row)]
    stores += [state[slot*heads + h, r + j, c].store(x.cast(state.dtype)) for c, x in zip(cols, row)]
  return UOp.group(*stores).end(hw, lane).sink(arg=KernelInfo(name="kda_replay", opts_to_apply=()))

def kda_replay(state:Tensor, rp:Tensor, slot:UOp, base:UOp, n:UOp, T:int, Q:int) -> Tensor:
  """the state of a slot (replay's, see _kda_replay_kernel) advanced n tokens from its base: the state after the kernel"""
  # (the variables bound by the state input: its kernel the unbound ones)
  st = Tensor(state.uop.after(slot, base, n))
  return Tensor.custom_kernel(st, rp, fxn=functools.partial(_kda_replay_kernel, slot=cast(UOp, _unbind(slot)), base=cast(UOp, _unbind(base)),
                                                            n=cast(UOp, _unbind(n)), T=T, Q=Q))[0]

@functools.cache
def _kda_out_kernel(z:UOp, q:UOp, scale:UOp, xsum:UOp, o:UOp, norm:UOp, gate:UOp, eps:float, heads:int) -> UOp:
  # the rms norm of a head's outputs (as _kda_decode_kernel: a lane its columns lane+32i) gated, in half, and its quantization as
  # _q8_quantize_kernel's: a wave a group of 32
  T, HV = cast(tuple[int, int], z.shape)
  V = HV // heads
  G = V // 32
  tg, lane = UOp.range(T*heads*G, 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP)
  t, h, g = tg // (heads*G), tg // G % heads, tg % G
  def sigmoid(x:UOp) -> UOp: return ((-x * LOG2E).exp2() + 1).reciprocal()
  ov = [o[t, h*V + lane + 32*i].load() for i in range(V // 32)]
  rinv = (warp_reduce(functools.reduce(lambda s,y: s + y*y, ov[1:], ov[0]*ov[0]), full_wave=True) * (1/V) + eps).sqrt().reciprocal()
  r = g*32 + lane
  out = (o[t, h*V + r].load() * rinv * norm[r].load().float() * sigmoid(gate[t, h*V + r].load().float())).cast(z.dtype)
  stores = [z[t, h*V + r].store(out)] + _q8_narrow_stores(out.float(), q, scale, xsum, None, t, h*G + g, lane)
  return UOp.group(*stores).end(tg, lane).sink(arg=KernelInfo(name="kda_out", opts_to_apply=()))

def kda_decode(state:Tensor, conv_states:list[Tensor], projs:list[Tensor], conv_weights:list[Tensor], gate:Tensor, alpha:Tensor, beta:Tensor,
               dt:Tensor, a:Tensor, norm:Tensor, start_pos:Tensor, lb:float|None, eps:float, row:int|None=None, rows:int=1,
               positions:Tensor|None=None, groups:int=1, S:int=0, SC:int=0, calls_out:list[UOp]|None=None, slots:Tensor|None=None,
               replay:Tensor|None=None, kbases:Tensor|None=None, replay_q:int=0) -> Tensor:
  """
  a few tokens (B=1, T) of a kda layer: returns the gated, normed outputs (1, T, heads*V) in half. the states (S, B, ...) at a position p
  are p%S: the ones of start_pos are read, the ones after every token written (T < S). calls_out: the kernel appended (a later write of
  the states waits for it)
  """
  _, B, heads, V, K = state.shape
  # row: `rows` tokens, rows row.. of the inputs (batched decoding: the inputs of every sequence's rows)
  T = projs[0].shape[1] if row is None else rows
  assert B == 1, "a sequence"  # (start_pos: a bound variable, or an expression of some)
  # groups (batched decoding): a sequence each, its states the S of g*S.. of the states (of every slot), its start the g-th of positions
  G = groups
  axis = None if state.uop.axis is None else 2
  # (R: the rows of a head over R workgroups, the outputs normed and quantized by kda_out)
  R = KDA_R if V % 32 == 0 else 0
  z = Tensor.empty(1, G*T, heads*V, dtype=dtypes.float if R else dtypes.half, device=state.device, axis=axis)
  # the states are written in place
  inplace = tuple(Tensor(s.uop.after(start_pos.uop)) for s in (state, *conv_states))
  assert slots is None or positions is not None, "the slots of groups"
  assert replay is None or (slots is not None and kbases is not None), "replay: the slots' bases"
  extra = ((positions,) if positions is not None else ()) + ((slots,) if slots is not None else ()) + \
    ((Tensor(replay.uop.after(start_pos.uop)), cast(Tensor, kbases)) if replay is not None else ())
  srcs = (z, *inplace, *projs, *conv_weights, gate, alpha, beta, dt, a, norm) + extra
  contig = tuple(x.uop if x.uop.op is Ops.AFTER else x.uop.contiguous() for x in srcs)
  params = tuple(UOp.placeholder_like(x, slot=i) for i,x in enumerate(contig))
  nx = len(extra)
  pos_param = params[len(params)-nx] if positions is not None else None
  slot_param = params[len(params)-nx+1] if slots is not None else None
  rp_param, kb_param = (params[-2], params[-1]) if replay is not None else (None, None)
  call = _kda_decode_kernel(*params[:len(params)-nx], cast(UOp, _unbind(start_pos.uop)), lb, eps, R, row or 0, G,
                            pos_param, S, SC, slot_param, rp_param, kb_param, replay_q).call(*contig)
  out = Tensor(contig[0].after(call))
  if calls_out is not None: calls_out.append(call)
  if not R: return out
  # the norm, the gate and the quantization of the outputs (the output projection's input)
  n = int(heads*V)  # (heads of every device)
  local = n // len(cast(tuple, state.device)) if axis is not None else n
  T = int(G*T)
  zh = Tensor.empty(1, T, n, dtype=dtypes.half, device=state.device, axis=axis)
  # (as _q8_quantize's of an input split on its features: every device its groups, a tensor of their shape)
  q = Tensor.empty(T, local // 32, 8, dtype=dtypes.uint32, device=state.device)
  scale = Tensor.empty(T, local // 32, dtype=dtypes.float32, device=state.device)
  xsum = Tensor.empty(T, local // 32, 2, dtype=dtypes.float32, device=state.device)
  if row is not None: gate = gate.shrink((None, (row, row+T), None))
  outs = Tensor.custom_kernel(zh.reshape(T, n), q, scale, xsum, out.reshape(T, n), norm.contiguous(), gate.reshape(T, n).contiguous(),
                              fxn=functools.partial(_kda_out_kernel, eps=eps, heads=local // V))
  zh = outs[0].reshape(1, T, n)
  q8_quantize_known(zh, T, local, False, (outs[1], outs[2], outs[3]))
  return zh

@functools.cache
def _kda_conv_kernel(q:UOp, k:UOp, v:UOp, kq:UOp, sq:UOp, sk:UOp, sv:UOp, pq:UOp, pk:UOp, pv:UOp, cq:UOp, ck:UOp, cv:UOp, taps:int,
                     qscale:float) -> UOp:
  """
  the short convolutions of q, k and v (their windows: the conv states s (taps-1, H*K) then the projections p (T, H*K), weights (H*K, taps))
  and their silu, q and k l2 normed (q scaled by qscale): (H, T, K) floats, and kq (H, T) the dot of q and k. a wave per head and token, a
  lane 4 channels
  """
  H, T, K = cast(tuple[int, int, int], q.shape)
  per = K // WARP_SIZE
  t, h, lane = UOp.range(T, 0, AxisType.GLOBAL), UOp.range(H, 1, AxisType.GLOBAL), UOp.range(WARP_SIZE, -1, AxisType.WARP)
  def conv(st:UOp, pr:UOp, cw:UOp, j:int) -> UOp:
    c = h * K + lane * per + j
    def win(r:UOp) -> UOp:  # the window's row r: the state's, then the projection's
      return (r < taps - 1).where(st[r.valid(r < taps - 1), c].load().float(), pr[(r - (taps - 1)).valid(r >= taps - 1), c].load().float())
    x = functools.reduce(lambda a, b: a + b, [win(t + i) * cw[c, i].load().float() for i in range(taps)])
    return x * (1 + (x * -LOG2E).exp2()).reciprocal()
  qs, ks, vs = ([conv(st, pr, cw, j) for j in range(per)] for st, pr, cw in ((sq, pq, cq), (sk, pk, ck), (sv, pv, cv)))
  def total(xs:list[UOp]) -> UOp: return warp_reduce(functools.reduce(lambda a, b: a + b, xs), full_wave=True)
  qn, kn = (total([x * x for x in qs]) + 1e-6).rsqrt() * qscale, (total([x * x for x in ks]) + 1e-6).rsqrt()
  qs, ks = [x * qn for x in qs], [x * kn for x in ks]
  dot = total([a * b for a, b in zip(qs, ks)])
  stores = [o[h, t, lane * per + j].store(xs[j]) for o, xs in ((q, qs), (k, ks), (v, vs)) for j in range(per)]
  stores.append(kq[h, t.valid(lane.eq(0))].store(dot))
  return UOp.group(*stores).end(t, h, lane).sink(arg=KernelInfo(name="kda_conv", opts_to_apply=()))

def kda_conv(states:list[Tensor], projs:list[Tensor], weights:list[Tensor], dim:int, qscale:float) -> tuple[Tensor, ...]:
  """
  the short convolutions of q, k and v (_kda_conv_kernel): their conv states (1, taps-1, heads*dim), projections (1, T, heads*dim) and weights,
  the heads split over the devices: q, k, v (1, heads, T, dim), kq (1, heads, T)
  """
  T, taps = int(projs[0].shape[1]), int(weights[0].shape[-1])
  devs = projs[0].device
  n, axis = (len(devs), 0) if isinstance(devs, tuple) else (1, None)
  heads = int(projs[0].uop.shard_shape[-1] if axis is not None else projs[0].shape[-1]) // dim
  outs = [Tensor.empty(heads * n, T, dim, dtype=dtypes.float32, device=devs, axis=axis) for _ in range(3)]
  kq = Tensor.empty(heads * n, T, dtype=dtypes.float32, device=devs, axis=axis)
  res = Tensor.custom_kernel(*outs, kq, *[x.reshape(x.shape[1], x.shape[2]).contiguous() for x in states + projs],
                             *[c.contiguous() for c in weights], fxn=functools.partial(_kda_conv_kernel, taps=taps, qscale=qscale))
  return tuple(x.unsqueeze(0) for x in res[:4])

def gated_delta_prefill(q:Tensor, k:Tensor, v:Tensor, beta:Tensor, alpha:Tensor, state:Tensor, start_pos:Tensor|None=None,
                        end_pos:UOp|None=None, kq:Tensor|None=None) -> Tensor:
  """
  the scan of the (B, H, T, ...) tokens updating the recurrent state in place. end_pos: the state is (2, B, H, V, K), the state at position p
  is p%2: the scan reads the state of start_pos and writes the one of end_pos (the other one keeps the state before)
  """
  batch, heads, tokens, key_dim = q.shape
  value_dim = v.shape[-1]
  assert q.shape == k.shape and v.shape[:3] == beta.shape == (batch, heads, tokens)
  assert state.shape[1 if end_pos is not None else 0:] == (batch, heads, value_dim, key_dim)
  assert alpha.shape[:3] == (batch, heads, tokens) and (len(alpha.shape) == 3 or alpha.shape[-1] in (1, key_dim))
  assert key_dim % 32 == 0 and value_dim % 4 == 0
  assert q.dtype == k.dtype == dtypes.float32, "recurrent Q/K must be float32"
  # (the view of the buffer, not of the writes it's after: a rewrite of their graphs)
  sv = state.uop
  while sv.op is Ops.AFTER: sv = sv.src[0]
  assert sv.contiguous_view_offset() is not None, "recurrent state must be contiguous"
  if start_pos is not None:
    # start_pos: a bound variable or an expression of them
    state = Tensor(state.uop.after(start_pos.uop, *([end_pos] if isinstance(end_pos, UOp) else [])))
  core, kq = Tensor.empty_like(v), (q*k).sum(-1).contiguous() if kq is None else kq
  srcs = (core, q.contiguous(), k.contiguous(), v.contiguous(), beta.contiguous(), alpha.contiguous(), state, kq)
  contig = tuple(x.uop if x.uop.op is Ops.AFTER else x.uop.contiguous() for x in srcs)
  params = tuple(UOp.placeholder_like(x, slot=i) for i,x in enumerate(contig))
  end = None if end_pos is None else end_pos.unbind_all()[0] if isinstance(end_pos, UOp) else UOp.const(end_pos, dtypes.weakint)
  sp = None if start_pos is None else start_pos.uop.unbind_all()[0]
  # many tokens: the waves of a head share the operands of 16 of them through LDS
  W = 4
  if tokens % 16 == 0 and value_dim % (32*W) == 0 and key_dim % 32 == 0:
    tc = (32 if key_dim == 128 and value_dim == 128 and tokens % 32 == 0 else 16)
    call = _gated_delta_prefill_lds_kernel(*params, sp, end, W=W, TC=tc).call(*contig)
  else: call = _gated_delta_prefill_kernel(*params, sp, end).call(*contig)
  return Tensor(contig[0].after(call))
