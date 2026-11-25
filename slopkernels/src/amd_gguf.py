from __future__ import annotations

import functools
from collections.abc import Callable

from tinygrad import UOp
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import unwrap
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

from slopkernels.src.amd import WMMA_ARG, _amd_load, warp_reduce

GGML_BLOCK_SIZE, Q8_GROUP_SIZE = 256, 32
Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K = 8, 10, 11, 12, 13, 14
IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS, MXFP4, NVFP4 = 16, 17, 18, 19, 20, 21, 22, 23, 39, 40
QUANT_SIZES = {Q8_0: 272, Q2_K: 84, Q3_K: 110, Q4_K: 144, Q5_K: 176, Q6_K: 210, IQ2_XXS: 66, IQ2_XS: 74,
               IQ3_XXS: 98, IQ1_S: 50, IQ4_NL: 144, IQ3_S: 110, IQ2_S: 82, IQ4_XS: 136, MXFP4: 136, NVFP4: 144}  # bytes per 256 weights
HALFWORD_QUANTS = (Q8_0, Q6_K, Q2_K, Q3_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ4_NL, IQ3_S, IQ2_S)
SMALL_BLOCK_QUANTS = {Q8_0: 17, IQ4_NL: 9, MXFP4: 17}  # block size in packed words (halfwords, mxfp4 bytes)
QUANT_BLOCK = {Q8_0: 32, IQ4_NL: 32, MXFP4: 32, NVFP4: 64}  # weights per block, GGML_BLOCK_SIZE otherwise
QUANT_NAMES = {Q8_0: "q8_0", Q2_K: "q2_k", Q3_K: "q3_k", Q4_K: "q4_k", Q5_K: "q5_k", Q6_K: "q6", IQ4_XS: "iq4_xs", IQ2_XXS: "iq2_xxs",
               IQ2_XS: "iq2_xs", IQ3_XXS: "iq3_xxs", IQ1_S: "iq1_s", IQ4_NL: "iq4_nl", IQ3_S: "iq3_s", IQ2_S: "iq2_s", MXFP4: "mxfp4", NVFP4: "nvfp4"}

def _amd_dp4a(a:UOp, b:UOp, c:UOp, cdna:bool) -> UOp:
  # CDNA has no sudot4 (dot8-insts)
  return UOp.native("__builtin_amdgcn_sdot4({}, {}, {}, false)" if cdna else "__builtin_amdgcn_sudot4(true, {}, true, {}, {}, false)",
                    shape=a.shape, dtype=dtypes.int32, fmt=True).call(a, b, c, inline=True)

def _amd_byte_perm(a:UOp, b:UOp, selectors:UOp) -> UOp:
  return UOp.native("__builtin_amdgcn_perm({}, {}, {})", shape=a.shape, dtype=dtypes.uint32,
                    fmt=True).call(*[x.cast(dtypes.uint32) for x in (a, b, selectors)], inline=True)

def _load_byte(raw:UOp, base:UOp, offset:int|UOp) -> UOp:
  size = raw.dtype.itemsize
  sh = (offset % size) * 8
  return (raw[base + offset // size].cast(dtypes.uint32) >> (sh.cast(dtypes.uint32) if isinstance(sh, UOp) else sh)) & 255
def _load_u32(raw:UOp, base:UOp, offset:int|UOp, stream:bool=False) -> UOp:
  # Halfword-aligned (mxfp4 byte-aligned) formats cannot load u32 directly.
  size = raw.dtype.itemsize
  parts = [raw[base+offset//size+i] for i in range(4//size)]
  if stream: parts = [_amd_load(p) for p in parts]
  return functools.reduce(lambda x,y: x|y, (p.cast(dtypes.uint32) << (8*size*i) for i,p in enumerate(parts)))

def _half(value:UOp) -> UOp: return value.cast(dtypes.uint16).bitcast(dtypes.float16).float()

IQ4NL_LUT = (0xbfad9881, 0xf6eaddcf, 0x26190d01, 0x71594535)
MXFP4_LUT = (0x03020100, 0x0c080604, 0xfdfeff00, 0xf4f8fafc)  # e2m1 doubled

def _nibble_lut(packed:UOp, shift:int|UOp, lut:tuple[int, int, int, int]) -> UOp:
  # a 16 entry int8 table looked up by the nibbles: 3 byte_perms beat any arithmetic/select-tree form (~60% decode)
  selectors = (packed >> shift) & 0x0f0f0f0f
  low = _amd_byte_perm(UOp.const(lut[1], dtypes.uint32), UOp.const(lut[0], dtypes.uint32), selectors)
  high = _amd_byte_perm(UOp.const(lut[3], dtypes.uint32), UOp.const(lut[2], dtypes.uint32), selectors & 0x07070707)
  return _amd_byte_perm(high, low, 0x03020100 | ((selectors & 0x08080808) >> 1))

def _e8m0_halved(e:UOp) -> UOp:
  # 2**(e-128) as float32 bits, e < 2 is a denormal
  e = e.cast(dtypes.uint32)
  return (e < 2).where(UOp.const(0x00200000, dtypes.uint32) << (e & 1), (e - 1) << 23).bitcast(dtypes.float32)

def _ue4m3_halved(e:UOp) -> UOp:
  e = e.cast(dtypes.uint32)
  exponent, mantissa = (e >> 3) & 15, e & 7
  normal = ((exponent + 119) << 23) | (mantissa << 20)
  value = exponent.eq(0).where(mantissa.float() * 2**-10, normal.bitcast(dtypes.float32))
  return e.eq(0x7f).where(0.0, value)

def _q5_scales(raw:UOp, base:UOp, subgroup:UOp, stream:bool=True) -> tuple[UOp, UOp, UOp, UOp]:
  # scales/mins (6-bit each) live in block bytes 4-15: three words total, same for the whole super-block's lanes
  ld = _amd_load if stream else (lambda p: p.load())
  w1, w2, w3 = ld(raw[base+1]), ld(raw[base+2]), ld(raw[base+3])
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

def _iq1_scale(raw:UOp, base:UOp, subgroup:UOp) -> tuple[UOp, UOp]:
  qh = raw[base+17+subgroup].cast(dtypes.uint32)
  return (((qh >> 12) & 7)*2+1).float() * _half(raw[base]), (qh >> 15).eq(0).where(-0.125, 0.125)

@functools.cache
def _q8_quantize_kernel(q:UOp, scale:UOp, xsum:UOp, x:UOp, tokens:int|UOp, in_features:int|UOp) -> UOp:
  groups = in_features//Q8_GROUP_SIZE
  token_group, lane = UOp.range(tokens*groups, 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP)
  token, group = token_group//groups, token_group%groups
  value = x.reshape(tokens, groups, 32)[token, group, lane].load().float()
  # Quantize each input once, then pack four neighboring lanes into one word.
  # Keep divisions intact: reciprocal multiplication can move half-precision inputs across rounding ties.
  warp_max = warp_reduce(value.abs(), Ops.MAX, full_wave=True)
  d = UOp.native("({0}/127.0f)", shape=warp_max.shape, dtype=dtypes.float, fmt=True).call(warp_max).maximum(1e-8)
  rounded = UOp.native("__builtin_nearbyintf({0}/{1})", shape=value.shape, dtype=dtypes.float, fmt=True).call(value, d)
  quant = rounded.clip(-127, 127).cast(dtypes.int8)
  word = quant.cast(dtypes.uint8).cast(dtypes.uint32) << ((lane%4)*8).cast(dtypes.uint32)
  word = warp_reduce(word, Ops.OR, masks=(1, 2))
  stores = (q[token, group, (lane//4).valid((lane%4).eq(0))].store(word),
            scale[token, group.valid(lane.eq(0))].store(d),
            xsum[token, group, (lane//16).valid((lane%16).eq(0))].store(warp_reduce(quant.float())))
  return UOp.group(*stores).end(token_group, lane).sink(arg=KernelInfo(name="q8_quantize", opts_to_apply=()))

def _decode_linear(out:UOp, out_features:int|UOp, group_count:int|UOp, group_dot, name:str) -> UOp:
  chunks = out.shape[2]
  row = UOp.range(out.shape[0] * out_features, 0, AxisType.GLOBAL)
  chunk = UOp.range(chunks, 1, AxisType.GLOBAL)
  lane = UOp.range(32, 2, AxisType.LOCAL)
  token, output = row // out_features, row % out_features
  steps = (group_count + chunks * 32 - 1) // (chunks * 32)
  acc = UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG)
  acc = acc.after(acc[0].store(UOp.const(0, dtypes.float32)))
  s = UOp.range(steps, 4)
  group = lane + (chunk + s * chunks) * 32
  upd = acc[0].store(acc.after(s)[0].load() +
                     (group < group_count).where(group_dot(token, output, group.minimum(group_count - 1), s), UOp.const(0, dtypes.float32))).end(s)
  val = warp_reduce(acc.after(upd)[0].load(), full_wave=True)
  return out[token, output, chunk.valid(lane.eq(0))].store(val.cast(out.dtype)).end(row, chunk, lane).sink(
    arg=KernelInfo(name=name, opts_to_apply=()))

def _iq_even_signs(signs:UOp) -> UOp:
  parity = signs ^ (signs >> 4)
  parity ^= parity >> 2
  parity ^= parity >> 1
  return signs | ((parity & 1) << 7)

def _iq_signed_word(word:UOp, signs:UOp) -> UOp:
  ones = ((signs.cast(dtypes.uint32) & 15) * 0x00204081) & 0x01010101
  # Grid magnitudes are nonzero and <128, so each byte can be negated without a carry into its neighbor.
  return (word ^ (ones * 255)) + ones

def _sh(x:int|UOp) -> int|UOp: return x.cast(dtypes.uint32) if isinstance(x, UOp) else x

def _quant_word(raw:UOp, base:UOp, subgroup:UOp, i:int|UOp, ggml_type:int, grid:UOp|None, stream:bool=True) -> UOp:
  # Four packed weight bytes, shared by integer-dot decode and FP16 WMMA prefill.
  def byte(offset): return _load_byte(raw, base, offset)
  def word(offset): return _load_u32(raw, base, offset, stream=stream and ggml_type == Q6_K)
  if ggml_type == Q8_0: return _load_u32(raw, base, 2+i*4, stream=stream)
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
    weights = (word(offset) >> ((subgroup%4)*2).cast(dtypes.uint32)) & 0x03030303
    if ggml_type == Q3_K: weights |= ((word(i*4) >> subgroup.cast(dtypes.uint32)) & 0x01010101) << 2
    return weights
  if ggml_type == NVFP4:
    block64, subblock = subgroup//2, (subgroup%2)*2 + i//4
    packed = word(block64*36 + 4 + subblock*8 + (i%2)*4)
    return _nibble_lut(packed, _sh(4*((i%4)//2)), MXFP4_LUT)
  if ggml_type in (IQ4_NL, IQ4_XS, MXFP4):
    packed = _amd_load(raw[base+2+subgroup*4+i%4]) if ggml_type == IQ4_XS else word((1 if ggml_type == MXFP4 else 2)+(i%4)*4)
    return _nibble_lut(packed, _sh(4*(i//4)), MXFP4_LUT if ggml_type == MXFP4 else IQ4NL_LUT)
  assert grid is not None
  grid_words = 2 if ggml_type in (IQ2_XXS, IQ2_XS, IQ1_S, IQ2_S) else 1
  if ggml_type in (IQ3_S, IQ3_XXS, IQ2_XXS):
    index = byte(2+subgroup*8+i//grid_words)
    if ggml_type == IQ3_S:
      index |= ((byte(66+subgroup) >> _sh(i)) & 1) << 8
      signs = byte(74+subgroup*4+i//2)
    else: signs = _iq_even_signs((word(66+subgroup*4 if ggml_type == IQ3_XXS else 6+subgroup*8) >> _sh(7*(i//2))) & 127)
  elif ggml_type == IQ2_XS:
    packed = raw[base+1+subgroup*4+i//2].cast(dtypes.uint32)
    index, signs = packed & 511, _iq_even_signs(packed >> 9)
  elif ggml_type == IQ1_S:
    index = byte(2+subgroup*4+i//2) | (((raw[base+17+subgroup].cast(dtypes.uint32) >> _sh(3*(i//2))) & 7) << 8)
    signs = index.const_like(0)
  else:
    index = byte(2+subgroup*4+i//2) | (((byte(66+subgroup) >> _sh(2*(i//2))) & 3) << 8)
    signs = byte(34+subgroup*4+i//2)
  return _iq_signed_word(grid[index*grid_words+i%grid_words].load(), signs >> _sh(4*(i%2)))

@functools.cache
def _quant_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, *grids:UOp, out_features:UOp, in_features:UOp, ggml_type:int,
                         cdna:bool=False, group_rows:UOp) -> UOp:
  words = out_features * in_features * QUANT_SIZES[ggml_type] // GGML_BLOCK_SIZE // raw.dtype.itemsize
  return _quant_decode_body(out, raw, xq, xd, xs, grids, out_features, in_features, ggml_type, cdna,
                            lambda token, base: base + (group_rows > 0).where(token // group_rows.maximum(1) * words, 0))

def _quant_base(output:UOp, group:UOp, in_features:int|UOp, ggml_type:int, itemsize:int) -> UOp:
  if ggml_type == NVFP4: return (output*in_features//64 + (group//8)*4) * (36//itemsize)
  if ggml_type in SMALL_BLOCK_QUANTS: return (output*in_features//32 + group) * SMALL_BLOCK_QUANTS[ggml_type]
  return (output*in_features//256 + group//8) * (QUANT_SIZES[ggml_type]//itemsize)

def _fma(a:UOp, b:UOp, c:UOp) -> UOp:
  args = tuple(x.load() if x.op is Ops.INDEX else x for x in (a, b, c))
  return UOp.native("__builtin_fmaf({0},{1},{2})", shape=args[0].shape, dtype=dtypes.float32, fmt=True).call(*args)

def _dp4a_loop(n:int, dep:UOp, word_fn:Callable[[UOp], UOp], x_reg:UOp, cdna:bool, x_off:int|UOp=0, rid:int=11) -> UOp:
  acc = UOp.alloc((1,), dtypes.int32, addrspace=AddrSpace.REG)
  acc = acc.after(acc.after(dep)[0].store(UOp.const(0, dtypes.int32)))
  i = UOp.range(n, rid)
  upd = acc[0].store(_amd_dp4a(word_fn(i), x_reg[x_off + i].load(), acc.after(i)[0].load(), cdna)).end(i)
  return acc.after(upd)[0].load()

def _dp4a_2loop(dep:UOp, word_fn:Callable[[UOp, UOp], UOp], x_reg:UOp, cdna:bool) -> tuple[UOp, UOp]:
  acc = UOp.alloc((2,), dtypes.int32, addrspace=AddrSpace.REG)
  h0 = UOp.range(2, 11, AxisType.UPCAST)
  acc = acc.after(acc.after(dep)[h0].store(UOp.const(0, dtypes.int32)).end(h0))
  hl, wi = UOp.range(2, 12), UOp.range(4, 13)
  upd = acc[hl].store(_amd_dp4a(word_fn(hl, wi), x_reg[hl * 4 + wi].load(), acc.after(hl, wi)[hl].load(), cdna)).end(wi, hl)
  acc = acc.after(upd)
  return acc[0].load(), acc[1].load()

def _quant_group_dot(raw:UOp, base:UOp, subgroup:UOp, x_reg:UOp, xd_val:UOp, xs_fn:Callable[[int], UOp],
                     ggml_type:int, cdna:bool, grid:UOp|None, dep:UOp) -> UOp:
  if ggml_type == Q8_0 and raw.dtype == dtypes.uint16:
    w_reg = UOp.alloc((8,), dtypes.uint32, addrspace=AddrSpace.REG)
    w_reg = w_reg.after(w_reg[:4].store(_amd_load(raw[base + 1], 8, stream=True).bitcast(dtypes.uint32)),
                        w_reg[4:].store(_amd_load(raw[base + 9], 8, stream=True).bitcast(dtypes.uint32)))
    return (_dp4a_loop(8, dep, lambda i: w_reg[i].load(), x_reg, cdna).float() * _half(raw[base])) * xd_val
  if ggml_type == IQ4_NL and raw.dtype == dtypes.uint16:
    p_reg = UOp.alloc((4,), dtypes.uint32, addrspace=AddrSpace.REG)
    p_reg = p_reg.after(p_reg.store(_amd_load(raw[base + 1], 8, stream=True).bitcast(dtypes.uint32)))
    acc_i = _dp4a_loop(8, dep, lambda i: _nibble_lut(p_reg[i % 4].load(), _sh((i // 4) * 4), IQ4NL_LUT), x_reg, cdna)
    return (acc_i.float() * _half(raw[base])) * xd_val
  if ggml_type in (Q4_K, Q5_K) and raw.dtype == dtypes.uint32:
    q_reg = UOp.alloc((8,), dtypes.uint32, addrspace=AddrSpace.REG)
    q_reg = q_reg.after(q_reg.store(_amd_load(raw[base + (4 if ggml_type == Q4_K else 12) + (subgroup // 2) * 8], 8, stream=True)))
    h_reg = UOp.alloc((8,), dtypes.uint32, addrspace=AddrSpace.REG) if ggml_type == Q5_K else None
    if h_reg is not None: h_reg = h_reg.after(h_reg.store(_amd_load(raw[base + 4], 8, stream=True)))
    q5_sh = ((subgroup & 1) * 4).cast(dtypes.uint32)
    def q5_w(i:UOp) -> UOp:
      w = (q_reg[i].load() >> q5_sh) & 0x0f0f0f0f
      return w if h_reg is None else (w | (((h_reg[i].load() >> subgroup.cast(dtypes.uint32)) & 0x01010101) << 4))
    acc_i = _dp4a_loop(8, dep, q5_w, x_reg, cdna)
    d, dmin, scale, minimum = _q5_scales(raw, base, subgroup)
    return (acc_i.float() * (d * scale) - (xs_fn(0) + xs_fn(1)) * (dmin * minimum)) * xd_val
  if ggml_type == IQ4_XS and raw.dtype == dtypes.uint32:
    p_reg = UOp.alloc((4,), dtypes.uint32, addrspace=AddrSpace.REG)
    p_reg = p_reg.after(p_reg.store(_amd_load(raw[base + 2 + subgroup * 4], 4, stream=True)))
    acc_i = _dp4a_loop(8, dep, lambda i: _nibble_lut(p_reg[i % 4].load(), _sh((i // 4) * 4), IQ4NL_LUT), x_reg, cdna)
    return (acc_i.float() * _iq4_scale(raw, base, subgroup)) * xd_val
  if ggml_type == NVFP4 and raw.dtype == dtypes.uint32:
    b64, h64 = subgroup // 2, subgroup % 2
    sc_w = raw[base + b64 * 9].load()
    p_reg = UOp.alloc((4,), dtypes.uint32, addrspace=AddrSpace.REG)
    p_reg = p_reg.after(p_reg.store(_amd_load(raw[base + b64 * 9 + 1 + h64 * 4], 4, stream=True)))
    d0, d1 = _dp4a_2loop(dep, lambda h, i: _nibble_lut(p_reg[h * 2 + i % 2].load(), _sh((i // 2) * 4), MXFP4_LUT), x_reg, cdna)
    ns0 = _ue4m3_halved((sc_w >> (h64 * 16).cast(dtypes.uint32)) & 255)
    ns1 = _ue4m3_halved((sc_w >> (h64 * 16 + 8).cast(dtypes.uint32)) & 255)
    return (d0.float() * ns0 + d1.float() * ns1) * xd_val
  if ggml_type == IQ3_XXS and grid is not None:
    q_reg = UOp.alloc((2,), dtypes.uint32, addrspace=AddrSpace.REG)
    if raw.dtype == dtypes.uint16:
      sc_w = _amd_load(raw[base + 33 + subgroup * 2], 2, stream=True).bitcast(dtypes.uint32)[0]
      q_reg = q_reg.after(q_reg.store(_amd_load(raw[base + 1 + subgroup * 4], 4, stream=True).bitcast(dtypes.uint32)))
    else:
      sc_w = _load_u32(raw, base, 66 + subgroup * 4)
      q_reg = q_reg.after(q_reg[0].store(_load_u32(raw, base, 2 + subgroup * 8)), q_reg[1].store(_load_u32(raw, base, 6 + subgroup * 8)))
    acc = UOp.alloc((1,), dtypes.int32, addrspace=AddrSpace.REG)
    acc = acc.after(acc.after(dep)[0].store(UOp.const(0, dtypes.int32)))
    p = UOp.range(4, 11)
    sgn = _iq_even_signs((sc_w >> _sh(7 * p)) & 127)
    qw, sh = q_reg[p // 2].load(), (p & 1) * 16
    a1 = _amd_dp4a(_iq_signed_word(grid[(qw >> _sh(sh)) & 255].load(), sgn), x_reg[2 * p].load(), acc.after(p)[0].load(), cdna)
    a2 = _amd_dp4a(_iq_signed_word(grid[(qw >> _sh(sh + 8)) & 255].load(), sgn >> 4), x_reg[2 * p + 1].load(), a1, cdna)
    acc_i = acc.after(acc[0].store(a2).end(p))[0].load()
    return acc_i.float() * (_half(raw[base]) * ((sc_w >> 28).float() + 0.5) * 0.5) * xd_val
  if ggml_type == IQ2_S and grid is not None:
    if raw.dtype == dtypes.uint16:
      q0 = _amd_load(raw[base + 1 + subgroup * 2], 2, stream=True).bitcast(dtypes.uint32)[0]
      s0 = _amd_load(raw[base + 17 + subgroup * 2], 2, stream=True).bitcast(dtypes.uint32)[0]
    else:
      q0, s0 = _load_u32(raw, base, 2 + subgroup * 4), _load_u32(raw, base, 34 + subgroup * 4)
    qh, sc_b = _load_byte(raw, base, 66 + subgroup), _load_byte(raw, base, 74 + subgroup)
    acc = UOp.alloc((2,), dtypes.int32, addrspace=AddrSpace.REG)
    h0 = UOp.range(2, 11, AxisType.UPCAST)
    acc = acc.after(acc.after(dep)[h0].store(UOp.const(0, dtypes.int32)).end(h0))
    hl, pi = UOp.range(2, 12), UOp.range(2, 13)
    p = hl * 2 + pi
    idx = ((q0 >> _sh(8 * p)) & 255) | (((qh >> _sh(2 * p)) & 3) << 8)
    sgn = (s0 >> _sh(8 * p)) & 255
    a1 = _amd_dp4a(_iq_signed_word(grid[idx * 2].load(), sgn), x_reg[2 * p].load(), acc.after(hl, pi)[hl].load(), cdna)
    a2 = _amd_dp4a(_iq_signed_word(grid[idx * 2 + 1].load(), sgn >> 4), x_reg[2 * p + 1].load(), a1, cdna)
    acc = acc.after(acc[hl].store(a2).end(pi, hl))
    d = _half(raw[base]) * 0.25
    sf0, sf1 = ((sc_b & 15).float() + 0.5) * d, (((sc_b >> 4) & 15).float() + 0.5) * d
    return _fma(acc[0].load().float(), sf0, acc[1].load().float() * sf1) * xd_val
  def byte(offset): return _load_byte(raw, base, offset)
  def word(offset): return _load_u32(raw, base, offset)
  # One accumulator per scale group: 32 weights for Q4/Q5/IQ4_XS, two groups of 16 otherwise.
  if ggml_type in (Q4_K, Q5_K, IQ4_XS):
    dots = [_dp4a_loop(8, dep, lambda i: _quant_word(raw, base, subgroup, i, ggml_type, grid), x_reg, cdna)]
  else:
    d0, d1 = _dp4a_2loop(dep, lambda h, i: _quant_word(raw, base, subgroup, h * 4 + i, ggml_type, grid), x_reg, cdna)
    dots = [d0, d1]
  if ggml_type in (Q4_K, Q5_K):
    d, dmin, scale, minimum = _q5_scales(raw, base, subgroup)
    ds, dm = d * scale, dmin * minimum
    total = dots[0].float() * ds - (xs_fn(0) + xs_fn(1)) * dm
  elif ggml_type == IQ4_XS:
    total = dots[0].float() * _iq4_scale(raw, base, subgroup)
  elif ggml_type == Q6_K:
    # Subtract the quant offset via the activation sums instead of unpacking signed bytes.
    scales = [(raw[base+96+subgroup] >> (h*8)).cast(dtypes.uint8).bitcast(dtypes.int8).float() for h in range(2)]
    total = sum((dots[h].float()-32*xs_fn(h))*scales[h] for h in range(2))
    return total * xd_val * _half(raw[base+104])
  elif ggml_type in (Q2_K, Q3_K):
    d_reg = UOp.alloc((2,), dtypes.int32, addrspace=AddrSpace.REG)
    d_reg = d_reg.after(d_reg[0].store(dots[0]), d_reg[1].store(dots[1]))
    xs_reg = UOp.alloc((2,), dtypes.float32, addrspace=AddrSpace.REG)
    xs_reg = xs_reg.after(xs_reg[0].store(xs_fn(0)), xs_reg[1].store(xs_fn(1)))
    hu = UOp.range(2, 14, AxisType.UPCAST)
    j = subgroup * 2 + hu
    if ggml_type == Q2_K:
      sc = byte(j)
      term = d_reg[hu].load().float() * (sc & 15).float() * _half(raw[base + 40]) - xs_reg[hu].load() * (sc >> 4).float() * _half(raw[base + 41])
    else:
      sc = ((byte(96 + j % 8) >> _sh((j // 8) * 4)) & 15) | (((byte(104 + j % 4) >> _sh((j // 4) * 2)) & 3) << 4)
      term = (d_reg[hu].load().float() - 4 * xs_reg[hu].load()) * (sc.cast(dtypes.int32) - 32).float() * _half(raw[base + 54])
    total = term.reduce(hu, arg=(Ops.ADD, 0))
  elif ggml_type in (IQ2_XS, IQ2_S):
    scales = byte((66 if ggml_type == IQ2_XS else 74)+subgroup)
    total = sum(dots[h].float() * (((scales >> (h*4)) & 15).float()+0.5) for h in range(2)) * 0.25 * _half(raw[base])
  elif ggml_type == IQ1_S:
    scale, zero = _iq1_scale(raw, base, subgroup)
    total = ((dots[0]+dots[1]).float() - (xs_fn(0)+xs_fn(1))*zero) * scale
  elif ggml_type == NVFP4:
    offset = (subgroup//2)*36 + (subgroup%2)*2
    total = dots[0].float()*_ue4m3_halved(byte(offset)) + dots[1].float()*_ue4m3_halved(byte(offset+1))
  else:
    total = (dots[0]+dots[1]).float() * (_e8m0_halved(raw[base]) if ggml_type == MXFP4 else _half(raw[base]))
    if ggml_type == IQ3_S: total *= (1+2*((byte(106+subgroup//2) >> _sh((subgroup%2)*4)) & 15)).float()
    if ggml_type == IQ3_XXS: total *= ((word(66+subgroup*4) >> 28).float()+0.5)*0.5
    if ggml_type == IQ2_XXS: total *= ((word(6+subgroup*8) >> 28).float()+0.5)*0.25
  return total * xd_val

def _quant_decode_body(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, grids:tuple[UOp, ...], out_features:int|UOp, in_features:int|UOp, ggml_type:int,
                       cdna:bool, gate:Callable|None=None, xtoken:Callable|None=None, name:str="linear_") -> UOp:
  def group_dot(token:UOp, output:UOp, group:UOp, dep:UOp) -> UOp:
    base = _quant_base(output, group, in_features, ggml_type, raw.dtype.itemsize)
    if gate is not None: base = gate(token, base)
    if xtoken is not None: token = xtoken(token)
    x_reg = UOp.alloc((8,), dtypes.uint32, addrspace=AddrSpace.REG)
    x_reg = x_reg.after(x_reg.store(_amd_load(xq[token, group, 0], 8)))
    return _quant_group_dot(raw, base, group%8, x_reg, xd[token, group], lambda h: xs[token, group, h].load(),
                            ggml_type, cdna, grids[0] if grids else None, dep)
  return _decode_linear(out, out_features, in_features//32, group_dot, name+QUANT_NAMES[ggml_type])

@functools.cache
def _moe_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, sel:UOp, *grids:UOp,
                       out_features:UOp, in_features:UOp, ggml_type:int, x_share:UOp, cdna:bool=False, ndev:UOp) -> UOp:
  expert_words = out_features * in_features * QUANT_SIZES[ggml_type] // 256 // raw.dtype.itemsize
  local_experts = raw.shape[0]//expert_words
  def expert(row:UOp) -> UOp: return sel[row].cast(dtypes.weakint) - UOp.range(ndev, 4, AxisType.DEVICE)*local_experts
  def gate(row:UOp, base:UOp) -> UOp:
    return (base + expert(row).maximum(0).minimum(local_experts-1) * expert_words).valid((expert(row) >= 0) & (expert(row) < local_experts))
  return _quant_decode_body(out, raw, xq, xd, xs, grids, out_features, in_features, ggml_type, cdna, gate, lambda row: row // x_share, "moe_")

def _iq_half_elem_f32(table:UOp, base_idx:UOp, s4:UOp, sc_f:UOp, b_idx:UOp) -> UOp:
  h_pair = table[base_idx + b_idx // 2].load()
  sign = ((s4 >> _sh(b_idx)) & 1) << 15
  h16 = (((h_pair >> _sh((b_idx & 1) * 16)) & 0xffff) ^ sign).cast(dtypes.uint16).bitcast(dtypes.half)
  return (h16.float() * sc_f).cast(dtypes.half)

# kgroups: groups of 16 lanes that split k in a 16x16x16 fragment, RDNA3 copies a and b in both halves
def _wmma_words(lane_group:UOp, kgroups:int) -> tuple[UOp|int, ...]:
  return tuple(range(4)) if kgroups == 1 else tuple(lane_group + kgroups * j for j in range(4 // kgroups))

def _make_wmma_dequant(raw:UOp, ggml_type:int, lane_group:UOp, kgroups:int, k_u:UOp,
                       table:UOp|None) -> Callable[[UOp, UOp, UOp], UOp|tuple[UOp, ...]]:
  w_u, b_u = k_u // 4, k_u % 4
  w_idx = w_u if kgroups == 1 else (lane_group + kgroups * w_u)
  word_indices = _wmma_words(lane_group, kgroups)
  if ggml_type in (Q4_K, Q5_K):
    def dequant_q5(base:UOp, subgroup:UOp, half:UOp) -> UOp:
      d, dmin, scale, minimum = _q5_scales(raw, base, subgroup, stream=False)
      ds, dm = d * scale, dmin * minimum
      qs_base = base + (4 if ggml_type == Q4_K else 12) + (subgroup // 2) * 8 + half * 4
      word = (raw[qs_base + w_idx].load() >> ((subgroup & 1) * 4).cast(dtypes.uint32)) & 0x0f0f0f0f
      if ggml_type == Q5_K: word |= ((raw[base + 4 + half * 4 + w_idx].load() >> subgroup.cast(dtypes.uint32)) & 0x01010101) << 4
      return (((word >> _sh(b_u * 8)) & 255).float() * ds - dm).cast(dtypes.float16)
    return dequant_q5
  if ggml_type == IQ4_XS:
    assert table is not None
    def dequant_iq4(base:UOp, subgroup:UOp, half:UOp) -> UOp:
      scale = _iq4_scale(raw, base, subgroup)
      pair = table[((raw[base + 2 + subgroup * 4 + w_idx].load() >> _sh(b_u * 8)) & 255).cast(dtypes.weakint)].load()
      return (_half((pair >> _sh(half * 16)) & 0xffff) * scale).cast(dtypes.float16)
    return dequant_iq4
  if ggml_type == MXFP4:
    def dequant_mxfp4(base:UOp, subgroup:UOp, half:UOp) -> tuple[UOp, ...]:
      b_base = base + subgroup * 17
      e = raw[b_base].cast(dtypes.uint32)
      small = e < 116
      # Shift small exponents up by 2^4 (2^-6 scaled e2m1) so subnormal fp16 results stay bit-exact down to e=100.
      c_lo = small.where(UOp.const(0x26242000, dtypes.uint32), UOp.const(0x36343000, dtypes.uint32))
      c_hi = small.where(UOp.const(0x2e2c2a28, dtypes.uint32), UOp.const(0x3e3c3a38, dtypes.uint32))
      sc2 = _e8m0_halved(e + small.where(UOp.const(7, dtypes.uint32), UOp.const(3, dtypes.uint32))).cast(dtypes.half).reshape((1,)).expand((2,))
      zero_u, sel01, sel23 = UOp.const(0, dtypes.uint32), UOp.const(0x05000400, dtypes.uint32), UOp.const(0x07000600, dtypes.uint32)
      sh = _sh(half * 4)
      out_vals: list[UOp] = []
      for wi in word_indices:
        pw = _load_u32(raw, b_base, 1 + wi * 4) >> sh
        fp_bytes = _amd_byte_perm(c_hi, c_lo, pw & 0x07070707) | ((pw & 0x08080808) << 4)
        v01 = _amd_byte_perm(fp_bytes, zero_u, sel01).reshape((1,)).bitcast(dtypes.half) * sc2
        v23 = _amd_byte_perm(fp_bytes, zero_u, sel23).reshape((1,)).bitcast(dtypes.half) * sc2
        out_vals.extend((v01[0], v01[1], v23[0], v23[1]))
      return tuple(out_vals)
    return dequant_mxfp4
  def dequant(base:UOp, subgroup:UOp, half:UOp) -> UOp|tuple[UOp, ...]:
    if ggml_type in SMALL_BLOCK_QUANTS: base += subgroup * SMALL_BLOCK_QUANTS[ggml_type]  # eight independent blocks per 256 weights
    if ggml_type == Q8_0:
      scale2 = raw[base].load().bitcast(dtypes.half).reshape((1,)).expand((2,))
      neg1152_2 = UOp.const(-1152.0, dtypes.half).reshape((1,)).expand((2,))
      c6400 = UOp.const(0x64006400, dtypes.uint32)
      if word_indices == (0, 1, 2, 3):
        c8080, sel_lo, sel_hi = UOp.const(0x80808080, dtypes.uint32), UOp.const(0x05010500, dtypes.uint32), UOp.const(0x05030502, dtypes.uint32)
        hv = _amd_load(raw[base + 1 + half * 8], 8).bitcast(dtypes.uint32)
        out_vals: list[UOp] = []
        for qi in range(4):
          ux = hv[qi] ^ c8080
          v01 = (_amd_byte_perm(c6400, ux, sel_lo).reshape((1,)).bitcast(dtypes.half) + neg1152_2) * scale2
          v23 = (_amd_byte_perm(c6400, ux, sel_hi).reshape((1,)).bitcast(dtypes.half) + neg1152_2) * scale2
          out_vals.extend((v01[0], v01[1], v23[0], v23[1]))
        return tuple(out_vals)
      c8080, csel = UOp.const(0x8080, dtypes.uint32), UOp.const(0x05010500, dtypes.uint32)
      out_vals = []
      for w in word_indices:
        for p in range(2):
          ux = raw[base + 1 + half * 8 + w * 2 + p].load().cast(dtypes.uint32) ^ c8080
          v01 = (_amd_byte_perm(c6400, ux, csel).reshape((1,)).bitcast(dtypes.half) + neg1152_2) * scale2
          out_vals.extend((v01[0], v01[1]))
      return tuple(out_vals)
    def byte(offset): return _load_byte(raw, base, offset)
    zero: int|UOp = 0
    minimum = UOp.const(0, dtypes.float)
    if ggml_type == Q2_K:
      sc = byte(subgroup * 2 + half)
      scale, minimum = _half(raw[base + 40]) * (sc & 15).float(), _half(raw[base + 41]) * (sc >> 4).float()
    elif ggml_type == Q3_K:
      j = subgroup * 2 + half
      sc = ((byte(96 + j % 8) >> _sh((j // 8) * 4)) & 15) | (((byte(104 + j % 4) >> _sh((j // 4) * 2)) & 3) << 4)
      scale, zero = _half(raw[base + 54]) * (sc.cast(dtypes.int32) - 32).float(), 4
    elif ggml_type == Q6_K:
      scale, zero = _half(raw[base + 104]) * byte(192 + subgroup * 2 + half).cast(dtypes.uint8).bitcast(dtypes.int8).float(), 32
    elif ggml_type == IQ1_S: scale, zero = _iq1_scale(raw, base, subgroup)
    elif ggml_type == NVFP4:
      e = byte((subgroup // 2) * 36 + (subgroup % 2) * 2 + half).cast(dtypes.uint32)
      if word_indices == (0, 1, 2, 3):
        sc_u16 = e.eq(0x7f).where(UOp.const(0, dtypes.uint32), (e & 0x7f) << 7).cast(dtypes.uint16)
        sc2 = sc_u16.bitcast(dtypes.half).reshape((1,)).expand((2,))
        # Exact FP16 high bytes for 256x-scaled e2m1 magnitudes [0, 128, 256, 384, 512, 768, 1024, 1536] (low bytes 0x00)
        c_lo, c_hi = UOp.const(0x5e5c5800, dtypes.uint32), UOp.const(0x66646260, dtypes.uint32)
        zero_u = UOp.const(0, dtypes.uint32)
        sel01, sel23 = UOp.const(0x05000400, dtypes.uint32), UOp.const(0x07000600, dtypes.uint32)
        subblock = (subgroup % 2) * 2 + half
        pv = _amd_load(raw[base + (subgroup // 2) * 9 + 1 + subblock * 2], 2)
        vals: list[UOp|None] = [None] * 16
        for wi in range(2):
          pw = pv[wi]
          lo_fp = _amd_byte_perm(c_hi, c_lo, pw & 0x07070707) | ((pw & 0x08080808) << 4)
          hi_fp = _amd_byte_perm(c_hi, c_lo, (pw >> 4) & 0x07070707) | (pw & 0x80808080)
          for bi, fp_bytes in ((0, lo_fp), (8, hi_fp)):
            v01 = _amd_byte_perm(fp_bytes, zero_u, sel01).reshape((1,)).bitcast(dtypes.half) * sc2
            v23 = _amd_byte_perm(fp_bytes, zero_u, sel23).reshape((1,)).bitcast(dtypes.half) * sc2
            idx = bi + wi * 4
            vals[idx], vals[idx + 1], vals[idx + 2], vals[idx + 3] = v01[0], v01[1], v23[0], v23[1]
        return tuple(unwrap(v) for v in vals)
      scale = _ue4m3_halved(e)
    else:
      scale = _e8m0_halved(raw[base]) if ggml_type == MXFP4 else _half(raw[base])
      if ggml_type in (IQ2_XS, IQ2_S):
        sc = byte((66 if ggml_type == IQ2_XS else 74) + subgroup)
        scale *= (((sc >> _sh(half * 4)) & 15).float() + 0.5) * 0.25
      elif ggml_type == IQ3_S: scale *= (1 + 2 * ((byte(106 + subgroup // 2) >> _sh((subgroup % 2) * 4)) & 15)).float()
      elif ggml_type == IQ3_XXS:
        assert table is not None
        sc_w = _load_u32(raw, base, 66 + subgroup * 4)
        sc_f = scale * ((sc_w >> 28).float() + 0.5) * 32.0
        idx_i = half * 4 + w_idx
        sgn8 = _iq_even_signs((sc_w >> _sh(7 * (idx_i // 2))) & 127)
        return _iq_half_elem_f32(table, byte(2 + subgroup * 8 + idx_i) * 2, sgn8 >> _sh(4 * (idx_i & 1)), sc_f, b_u)
      elif ggml_type == IQ2_XXS: scale *= ((_load_u32(raw, base, 6 + subgroup * 8) >> 28).float() + 0.5) * 0.25
    if ggml_type == IQ2_S:
      assert table is not None
      sc_f, qh = scale * 64.0, byte(66 + subgroup)
      idx_i = half * 4 + w_idx
      pair = idx_i // 2
      g_idx = byte(2 + subgroup * 4 + pair) | (((qh >> _sh(2 * pair)) & 3) << 8)
      s4 = byte(34 + subgroup * 4 + pair) >> _sh(4 * (idx_i & 1))
      return _iq_half_elem_f32(table, g_idx * 4 + (idx_i & 1) * 2, s4, sc_f, b_u)
    if ggml_type in (IQ4_NL, MXFP4, NVFP4):
      words = tuple(_quant_word(raw, base, subgroup, half * 4 + wi, ggml_type, table, stream=False) for wi in word_indices)
      w_reg = UOp.alloc((len(words),), dtypes.uint32, addrspace=AddrSpace.REG)
      word = w_reg.after(w_reg.store(UOp.stack(*words)))[w_u].load()
    else:
      word = _quant_word(raw, base, subgroup, half * 4 + w_idx, ggml_type, table, stream=False)
    return (((word >> _sh(b_u * 8)).cast(dtypes.uint8).bitcast(dtypes.int8).float() - zero) * scale - minimum).cast(dtypes.half)
  return dequant

@functools.cache
def _quant_linear_f16_wmma_kernel(out:UOp, raw:UOp, x:UOp, *grids:UOp,
                                  m_blks:UOp, m_tiles:UOp, out_features:UOp, in_features:UOp,
                                  ggml_type:int, kgroups:int, lanes:int) -> UOp:
  output_block, token_block = UOp.range(out_features // 16, 0, AxisType.GLOBAL), UOp.range(m_blks, 1, AxisType.GLOBAL)
  lane = UOp.range(lanes, -1, axis_type=AxisType.WARP)
  col, lane_group, rows, frag_k = lane % 16, lane // 16, 256 // lanes, 16 // kgroups
  output = output_block * 16 + col
  grid = None
  if grids:
    grid = UOp.alloc((int(grids[0].numel()),), dtypes.uint32, addrspace=AddrSpace.LOCAL)
    g_i, g4 = UOp.range(int(grid.numel()) // (lanes * 4), 20), UOp.range(4, 21, AxisType.UPCAST)
    g_off = (lane + g_i * lanes) * 4 + g4
    grid = grid.after(grid[g_off].store(grids[0][g_off].load()).end(g4, g_i))
  type_words = QUANT_SIZES[ggml_type] // raw.dtype.itemsize
  output_words = in_features * type_words // GGML_BLOCK_SIZE
  group, half = UOp.range(in_features // Q8_GROUP_SIZE, 4), UOp.range(2, 5)
  block, subgroup, half_group = group // 8, group % 8, group * 2 + half
  k_u, m_u, mt_u = UOp.range(frag_k, 10, AxisType.UPCAST), UOp.range(rows, 11, AxisType.UPCAST), UOp.range(m_tiles, 12, AxisType.UPCAST)
  w_idx = (k_u // 4) if kgroups == 1 else (lane_group + kgroups * (k_u // 4))
  a_elem = x[(token_block * m_tiles + mt_u) * 16 + col, half_group * 16 + w_idx * 4 + (k_u % 4)].load()
  b_raw = _make_wmma_dequant(raw, ggml_type, lane_group, kgroups, k_u, grid)(output * output_words + block * type_words, subgroup, half)
  if isinstance(b_raw, tuple):
    b_reg = UOp.alloc((frag_k,), dtypes.float16, addrspace=AddrSpace.REG)
    b_elem = b_reg.after(b_reg.after(group, half).store(UOp.stack(*b_raw)))[k_u].load()
  else: b_elem = b_raw
  tc_axes = (((k_u.arg, frag_k), (m_u.arg, 1)), ((k_u.arg, frag_k), (m_u.arg, 1)), ((k_u.arg, 1), (m_u.arg, rows)))
  wmma = UOp.wmma(a_elem, b_elem, UOp.const((0.0,) * rows, dtypes.float32), WMMA_ARG[0], lanes, tc_upcast_axes=tc_axes)
  res = wmma.reduce(group, half, arg=(Ops.ADD, 0))
  row_in_tile = lane_group * rows + m_u if kgroups > 1 else m_u * 2 + lane_group
  st = UOp(Ops.STORE, src=(out[(token_block * m_tiles + mt_u) * 16 + row_in_tile, output], res.cast(out.dtype)))
  return st.end(k_u, m_u, mt_u, token_block, output_block, lane).sink(
    arg=KernelInfo(name=f"linear_{QUANT_NAMES[ggml_type]}_f16_wmma", optimized=True))

