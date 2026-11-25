from __future__ import annotations

import functools
from collections.abc import Callable

from tinygrad import UOp
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

from slopkernels.src.amd import WMMA_ARG, _amd_byte_perm, _amd_fdot2, _amd_load, _cvt_pk_f16, warp_reduce

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
DIRECT_GEMV_QUANTS = (Q8_0, Q6_K, Q5_K, Q4_K, Q2_K, IQ4_XS, IQ3_S, IQ1_S, MXFP4)
DP4A_M_QUANTS = (Q8_0, Q4_K, IQ4_XS, MXFP4)

def _amd_dp4a(a:UOp, b:UOp, c:UOp, cdna:bool) -> UOp:
  # CDNA has no sudot4 (dot8-insts)
  return UOp.native("__builtin_amdgcn_sdot4({}, {}, {}, false)" if cdna else "__builtin_amdgcn_sudot4(true, {}, true, {}, {}, false)",
                    shape=a.shape, dtype=dtypes.int32, fmt=True).call(a, b, c, inline=True)

def _perm_alignbyte(hi:UOp, lo:UOp, shift_bytes:int|UOp) -> UOp:
  # Extract 4 contiguous bytes starting at shift_bytes from (hi:lo) using existing v_perm_b32 (zero extra native).
  sel = (UOp.const(0x03020100 + (shift_bytes & 3) * 0x01010101, dtypes.uint32) if isinstance(shift_bytes, int)
         else UOp.const(0x03020100, dtypes.uint32) + (shift_bytes.cast(dtypes.uint32) & 3) * UOp.const(0x01010101, dtypes.uint32))
  return _amd_byte_perm(hi, lo, sel)

def _splat_h2(h:UOp) -> UOp:
  u = h.bitcast(dtypes.uint16).cast(dtypes.uint32)
  return _amd_byte_perm(u, u, UOp.const(0x01000100, dtypes.uint32)).reshape((1,)).bitcast(dtypes.half)

def _load_byte(raw:UOp, base:UOp, offset:int|UOp) -> UOp:
  size = raw.dtype.itemsize
  sh = (offset % size) * 8
  return (raw[base + offset // size].cast(dtypes.uint32) >> (sh.cast(dtypes.uint32) if isinstance(sh, UOp) else sh)) & 255
def _load_u32(raw:UOp, base:UOp, offset:int|UOp) -> UOp:
  # Halfword-aligned (mxfp4 byte-aligned) formats cannot load u32 directly.
  size = raw.dtype.itemsize
  return functools.reduce(lambda x,y: x|y, (raw[base+offset//size+i].cast(dtypes.uint32) << (8*size*i) for i in range(4//size)))

def _half(value:UOp) -> UOp: return value.cast(dtypes.uint16).bitcast(dtypes.float16).float()

IQ4NL_LUT = (0xbfad9881, 0xf6eaddcf, 0x26190d01, 0x71594535)
IQ4NL_U8_LUT = tuple(x ^ 0x80808080 for x in IQ4NL_LUT)
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
  if stream:
    hdr = _amd_load(raw[base], 4, stream=True)
    w0, w1, w2, w3 = hdr[0], hdr[1], hdr[2], hdr[3]
  else: w0, w1, w2, w3 = raw[base].load(), raw[base+1].load(), raw[base+2].load(), raw[base+3].load()
  sb = (subgroup & 3) * 8  # byte within word
  byte1, byte2, byte3 = (w1 >> sb) & 255, (w2 >> sb) & 255, (w3 >> sb) & 255
  scale = (subgroup < 4).where(byte1 & 63, (byte3 & 15) | ((byte1 >> 6) << 4))
  minimum = (subgroup < 4).where(byte2 & 63, (byte3 >> 4) | ((byte2 >> 6) << 4))
  return _half((w0 & 0xffff).cast(dtypes.uint16)), _half((w0 >> 16).cast(dtypes.uint16)), scale.float(), minimum.float()

def _q4k_pair_scales(hdr:list[UOp], sub_pair:UOp) -> tuple[UOp, UOp, UOp, UOp]:
  w0, w1, w2, w3 = hdr[0], hdr[1], hdr[2], hdr[3]
  d, dmin = _half((w0 & 0xffff).cast(dtypes.uint16)), _half((w0 >> 16).cast(dtypes.uint16))
  sb0 = ((sub_pair & 1) * 16).cast(dtypes.uint32)
  sb1, hi = sb0 + UOp.const(8, dtypes.uint32), sub_pair >= 2
  b10, b20, b30 = (w1 >> sb0) & 255, (w2 >> sb0) & 255, (w3 >> sb0) & 255
  b11, b21, b31 = (w1 >> sb1) & 255, (w2 >> sb1) & 255, (w3 >> sb1) & 255
  sc0, mn0 = hi.where((b30 & 15) | ((b10 >> 6) << 4), b10 & 63), hi.where((b30 >> 4) | ((b20 >> 6) << 4), b20 & 63)
  sc1, mn1 = hi.where((b31 & 15) | ((b11 >> 6) << 4), b11 & 63), hi.where((b31 >> 4) | ((b21 >> 6) << 4), b21 & 63)
  return d * sc0.float(), dmin * mn0.float(), d * sc1.float(), dmin * mn1.float()

def _iq4_scale_hdr(header:UOp, low:UOp, subgroup:UOp) -> UOp:
  scale = ((low >> (4 * subgroup).cast(dtypes.uint32)) & 15) | (((header >> (16 + 2 * subgroup).cast(dtypes.uint32)) & 3) << 4)
  return _half(header) * (scale.cast(dtypes.int32) - 32).float()

def _iq4_scale(raw:UOp, base:UOp, subgroup:UOp, stream:bool=False) -> UOp:
  if stream:
    hdr = _amd_load(raw[base], 2, stream=True)
    return _iq4_scale_hdr(hdr[0], hdr[1], subgroup)
  return _iq4_scale_hdr(raw[base].load(), raw[base+1].load(), subgroup)

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
  d = warp_max.alu(Ops.FDIV, warp_max.const_like(127.0)).maximum(1e-8)
  # IEEE round-to-nearest-even for |x| <= 127 without native nearbyintf: adding 1.5*2^23 aligns the ULP to 1.
  quant = ((value.alu(Ops.FDIV, d).clip(-127.0, 127.0) + 12582912.0).bitcast(dtypes.int32) - 0x4b400000).cast(dtypes.int8)
  word = quant.cast(dtypes.uint8).cast(dtypes.uint32) << ((lane%4)*8).cast(dtypes.uint32)
  word = warp_reduce(word, Ops.OR, masks=(1, 2))
  stores = (q[token, group, (lane//4).valid((lane%4).eq(0))].store(word),
            scale[token, group.valid(lane.eq(0))].store(d),
            xsum[token, group, (lane//16).valid((lane%16).eq(0))].store(warp_reduce(quant.float())))
  return UOp.group(*stores).end(token_group, lane).sink(arg=KernelInfo(name="q8_quantize", opts_to_apply=()))

def _decode_linear(out:UOp, out_features:int|UOp, group_count:int|UOp, waves:UOp, group_dot, name:str, share_weights:bool=True) -> UOp:
  row = UOp.range(out.shape[0] * out_features, 0, AxisType.GLOBAL)
  # lane has the smaller local axis id so it is always threadIdx.x (contiguous 32-lane half-wave on both RDNA and CDNA)
  lane, wave = UOp.range(32, 1, AxisType.LOCAL), UOp.range(waves, 2, AxisType.LOCAL)
  token, output = (row % out.shape[0], row // out.shape[0]) if share_weights else (row // out_features, row % out_features)
  steps = (group_count + waves * 32 - 1) // (waves * 32)
  acc = UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG)
  acc = acc.after(acc[0].store(UOp.const(0, dtypes.float32)))
  s = UOp.range(steps, 4)
  group = lane + (wave + s * waves) * 32
  upd = acc[0].store(acc.after(s)[0].load() +
                     (group < group_count).where(group_dot(token, output, group.minimum(group_count - 1), s), UOp.const(0, dtypes.float32))).end(s)
  val = warp_reduce(acc.after(upd)[0].load(), full_wave=True)
  lds = UOp.alloc((waves,), dtypes.float32, addrspace=AddrSpace.LOCAL)
  lds = lds.after(lds[wave.valid(lane.eq(0))].store(val).barrier())
  w_r = UOp.range(waves, 5)
  total = waves.eq(1).where(val, lds[w_r].load().reduce(w_r, arg=(Ops.ADD, 0)))
  return out[token, output.valid(wave.eq(0) & lane.eq(0))].store(total.cast(out.dtype)).end(row, wave, lane).sink(
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

def _quant_word(raw:UOp, base:UOp, subgroup:UOp, i:int|UOp, ggml_type:int, grid:UOp|None) -> UOp:
  # Four packed weight bytes, shared by integer-dot decode and FP16 WMMA prefill.
  def byte(offset): return _load_byte(raw, base, offset)
  def word(offset): return _load_u32(raw, base, offset)
  if ggml_type in (Q2_K, Q3_K):
    offset = (16 if ggml_type == Q2_K else 32) + (subgroup//4)*32 + i*4
    weights = (word(offset) >> ((subgroup%4)*2).cast(dtypes.uint32)) & 0x03030303
    if ggml_type == Q3_K: weights |= ((word(i*4) >> subgroup.cast(dtypes.uint32)) & 0x01010101) << 2
    return weights
  if ggml_type == IQ4_NL:
    return _nibble_lut(word(2 + (i%4)*4), _sh(4*(i//4)), IQ4NL_LUT)
  assert grid is not None
  grid_words = 2 if ggml_type in (IQ2_XXS, IQ2_XS, IQ1_S) else 1
  if ggml_type in (IQ3_S, IQ2_XXS):
    index = byte(2+subgroup*8+i//grid_words)
    if ggml_type == IQ3_S:
      index |= ((byte(66+subgroup) >> _sh(i)) & 1) << 8
      signs = byte(74+subgroup*4+i//2)
    else: signs = _iq_even_signs((word(6+subgroup*8) >> _sh(7*(i//2))) & 127)
  elif ggml_type == IQ2_XS:
    packed = raw[base+1+subgroup*4+i//2].cast(dtypes.uint32)
    index, signs = packed & 511, _iq_even_signs(packed >> 9)
  else:  # IQ1_S
    index = byte(2+subgroup*4+i//2) | (((raw[base+17+subgroup].cast(dtypes.uint32) >> _sh(3*(i//2))) & 7) << 8)
    signs = index.const_like(0)
  return _iq_signed_word(grid[index*grid_words+i%grid_words].load(), signs >> _sh(4*(i%2)))

@functools.cache
def _quant_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, *grids:UOp, out_features:UOp, in_features:UOp, waves:UOp, ggml_type:int,
                         cdna:bool=False, group_rows:UOp) -> UOp:
  words = out_features * in_features * QUANT_SIZES[ggml_type] // GGML_BLOCK_SIZE // raw.dtype.itemsize
  return _quant_decode_body(out, raw, xq, xd, xs, grids, out_features, in_features, waves, ggml_type, cdna,
                            lambda token, base: base + (group_rows > 0).where(token // group_rows.maximum(1) * words, 0))

def _quant_base(output:UOp, group:UOp, in_features:int|UOp, ggml_type:int, itemsize:int) -> UOp:
  if ggml_type == NVFP4: return (output*in_features//64 + (group//8)*4) * (36//itemsize)
  if ggml_type in SMALL_BLOCK_QUANTS: return (output*in_features//32 + group) * SMALL_BLOCK_QUANTS[ggml_type]
  return (output*in_features//256 + group//8) * (QUANT_SIZES[ggml_type]//itemsize)

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
  if ggml_type == Q8_0:
    w_reg = UOp.alloc((8,), dtypes.uint32, addrspace=AddrSpace.REG)
    w_reg = w_reg.after(w_reg[:4].store(_amd_load(raw[base + 1], 8, stream=True).bitcast(dtypes.uint32)),
                        w_reg[4:].store(_amd_load(raw[base + 9], 8, stream=True).bitcast(dtypes.uint32)))
    return (_dp4a_loop(8, dep, lambda i: w_reg[i].load(), x_reg, cdna).float() * _half(_amd_load(raw[base]))) * xd_val
  if ggml_type in (IQ4_NL, IQ4_XS, MXFP4):
    p_reg = UOp.alloc((4,), dtypes.uint32, addrspace=AddrSpace.REG)
    p_load = (_amd_load(raw[base + 2 + subgroup * 4], 4, stream=True) if ggml_type == IQ4_XS
              else _amd_load(raw[base + 1], 16 if ggml_type == MXFP4 else 8, stream=True).bitcast(dtypes.uint32))
    p_reg = p_reg.after(p_reg.store(p_load))
    lut = MXFP4_LUT if ggml_type == MXFP4 else IQ4NL_LUT
    acc = UOp.alloc((1,), dtypes.int32, addrspace=AddrSpace.REG)
    acc = acc.after(acc.after(dep)[0].store(UOp.const(0, dtypes.int32)))
    i = UOp.range(4, 11)
    pw = p_reg[i].load()
    a1 = _amd_dp4a(_nibble_lut(pw, 0, lut), x_reg[i].load(), acc.after(i)[0].load(), cdna)
    a2 = _amd_dp4a(_nibble_lut(pw, 4, lut), x_reg[4 + i].load(), a1, cdna)
    acc_i = acc.after(acc[0].store(a2).end(i))[0].load()
    sc = (_e8m0_halved(_amd_load(raw[base])) if ggml_type == MXFP4 else
          _half(_amd_load(raw[base])) if ggml_type == IQ4_NL else _iq4_scale(raw, base, subgroup, stream=True))
    return (acc_i.float() * sc) * xd_val
  if ggml_type == Q6_K:
    ql_base, qh_base = base + (subgroup // 4) * 32 + (subgroup % 2) * 16, base + 64 + (subgroup // 4) * 16
    ql_reg, qh_reg = UOp.alloc((8,), dtypes.uint32, addrspace=AddrSpace.REG), UOp.alloc((8,), dtypes.uint32, addrspace=AddrSpace.REG)
    ql_reg = ql_reg.after(ql_reg[:4].store(_amd_load(raw[ql_base], 8, stream=True).bitcast(dtypes.uint32)),
                          ql_reg[4:].store(_amd_load(raw[ql_base + 8], 8, stream=True).bitcast(dtypes.uint32)))
    qh_reg = qh_reg.after(qh_reg[:4].store(_amd_load(raw[qh_base], 8, stream=True).bitcast(dtypes.uint32)),
                          qh_reg[4:].store(_amd_load(raw[qh_base + 8], 8, stream=True).bitcast(dtypes.uint32)))
    ql_sh, qh_sh = ((subgroup % 4 // 2) * 4).cast(dtypes.uint32), ((subgroup % 4) * 2).cast(dtypes.uint32)
    dots = _dp4a_2loop(dep, lambda h, i: ((ql_reg[h * 4 + i].load() >> ql_sh) & 0x0f0f0f0f) |
                       (((qh_reg[h * 4 + i].load() >> qh_sh) & 0x03030303) << 4), x_reg, cdna)
    sc_w = _amd_load(raw[base + 96 + subgroup])
    scales = [(sc_w >> (h * 8)).cast(dtypes.uint8).bitcast(dtypes.int8).float() for h in range(2)]
    return sum((dots[h].float() - 32 * xs_fn(h)) * scales[h] for h in range(2)) * xd_val * _half(_amd_load(raw[base + 104]))
  if ggml_type in (Q4_K, Q5_K):
    q_reg = UOp.alloc((8,), dtypes.uint32, addrspace=AddrSpace.REG)
    q_reg = q_reg.after(q_reg.store(_amd_load(raw[base + (4 if ggml_type == Q4_K else 12) + (subgroup // 2) * 8], 8, stream=True)))
    h_reg = UOp.alloc((8,), dtypes.uint32, addrspace=AddrSpace.REG) if ggml_type == Q5_K else None
    if h_reg is not None: h_reg = h_reg.after(h_reg.store(_amd_load(raw[base + 4], 8, stream=True)))
    q5_sh = ((subgroup & 1) * 4).cast(dtypes.uint32)
    def q5_w(i:UOp) -> UOp:
      w = (q_reg[i].load() >> q5_sh) & 0x0f0f0f0f
      return w if h_reg is None else (w | (((h_reg[i].load() >> subgroup.cast(dtypes.uint32)) & 0x01010101) << 4))
    d, dmin, scale, minimum = _q5_scales(raw, base, subgroup)
    return (_dp4a_loop(8, dep, q5_w, x_reg, cdna).float() * (d * scale) - (xs_fn(0) + xs_fn(1)) * (dmin * minimum)) * xd_val
  if ggml_type == NVFP4:
    b64, h64 = subgroup // 2, subgroup % 2
    sc_w = raw[base + b64 * 9].load()
    p_reg = UOp.alloc((4,), dtypes.uint32, addrspace=AddrSpace.REG)
    p_reg = p_reg.after(p_reg.store(_amd_load(raw[base + b64 * 9 + 1 + h64 * 4], 4, stream=True)))
    d0, d1 = _dp4a_2loop(dep, lambda h, i: _nibble_lut(p_reg[h * 2 + i % 2].load(), _sh((i // 2) * 4), MXFP4_LUT), x_reg, cdna)
    return (d0.float() * _ue4m3_halved((sc_w >> (h64 * 16).cast(dtypes.uint32)) & 255) +
            d1.float() * _ue4m3_halved((sc_w >> (h64 * 16 + 8).cast(dtypes.uint32)) & 255)) * xd_val
  if ggml_type == IQ1_S:
    assert grid is not None
    qh_u = _amd_load(raw[base + 17 + subgroup]).cast(dtypes.uint32)
    qs_u = _amd_load(raw[base + 1 + subgroup * 2], 2, stream=True).bitcast(dtypes.uint32)[0]
    acc = UOp.alloc((1,), dtypes.int32, addrspace=AddrSpace.REG)
    acc = acc.after(acc.after(dep)[0].store(UOp.const(0, dtypes.int32)))
    p = UOp.range(4, 11)
    idx = ((qs_u >> _sh(8 * p)) & 255) | (((qh_u >> _sh(3 * p)) & 7) << 8)
    g01 = _amd_load(grid[idx * 2], 2)
    a1 = _amd_dp4a(g01[0], x_reg[2 * p].load(), acc.after(p)[0].load(), cdna)
    a2 = _amd_dp4a(g01[1], x_reg[2 * p + 1].load(), a1, cdna)
    scale, zero = (((qh_u >> 12) & 7) * 2 + 1).float() * _half(_amd_load(raw[base])), (qh_u >> 15).eq(0).where(-0.125, 0.125)
    return (acc.after(acc[0].store(a2).end(p))[0].load().float() - (xs_fn(0) + xs_fn(1)) * zero) * scale * xd_val
  if ggml_type == IQ3_XXS:
    assert grid is not None
    q_reg = UOp.alloc((2,), dtypes.uint32, addrspace=AddrSpace.REG)
    sc_w = _amd_load(raw[base + 33 + subgroup * 2], 2, stream=True).bitcast(dtypes.uint32)[0]
    q_reg = q_reg.after(q_reg.store(_amd_load(raw[base + 1 + subgroup * 4], 4, stream=True).bitcast(dtypes.uint32)))
    acc = UOp.alloc((1,), dtypes.int32, addrspace=AddrSpace.REG)
    acc = acc.after(acc.after(dep)[0].store(UOp.const(0, dtypes.int32)))
    p = UOp.range(4, 11)
    sgn, qw, sh = _iq_even_signs((sc_w >> _sh(7 * p)) & 127), q_reg[p // 2].load(), (p & 1) * 16
    a1 = _amd_dp4a(_iq_signed_word(grid[(qw >> _sh(sh)) & 255].load(), sgn), x_reg[2 * p].load(), acc.after(p)[0].load(), cdna)
    a2 = _amd_dp4a(_iq_signed_word(grid[(qw >> _sh(sh + 8)) & 255].load(), sgn >> 4), x_reg[2 * p + 1].load(), a1, cdna)
    return acc.after(acc[0].store(a2).end(p))[0].load().float() * (_half(raw[base]) * ((sc_w >> 28).float() + 0.5) * 0.5) * xd_val
  if ggml_type == IQ2_S:
    assert grid is not None
    q0 = _amd_load(raw[base + 1 + subgroup * 2], 2, stream=True).bitcast(dtypes.uint32)[0]
    s0 = _amd_load(raw[base + 17 + subgroup * 2], 2, stream=True).bitcast(dtypes.uint32)[0]
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
    return (acc[0].load().float() * sf0 + acc[1].load().float() * sf1) * xd_val
  def byte(offset): return _load_byte(raw, base, offset)
  def word(offset): return _load_u32(raw, base, offset)
  dots = _dp4a_2loop(dep, lambda h, i: _quant_word(raw, base, subgroup, h * 4 + i, ggml_type, grid), x_reg, cdna)
  if ggml_type in (Q2_K, Q3_K):
    d_reg, xs_reg = UOp.alloc((2,), dtypes.int32, addrspace=AddrSpace.REG), UOp.alloc((2,), dtypes.float32, addrspace=AddrSpace.REG)
    d_reg, xs_reg = d_reg.after(d_reg[0].store(dots[0]), d_reg[1].store(dots[1])), xs_reg.after(xs_reg[0].store(xs_fn(0)), xs_reg[1].store(xs_fn(1)))
    hu = UOp.range(2, 14, AxisType.UPCAST)
    j = subgroup * 2 + hu
    if ggml_type == Q2_K:
      sc = byte(j)
      term = d_reg[hu].load().float() * (sc & 15).float() * _half(raw[base + 40]) - xs_reg[hu].load() * (sc >> 4).float() * _half(raw[base + 41])
    else:
      sc = ((byte(96 + j % 8) >> _sh((j // 8) * 4)) & 15) | (((byte(104 + j % 4) >> _sh((j // 4) * 2)) & 3) << 4)
      term = (d_reg[hu].load().float() - 4 * xs_reg[hu].load()) * (sc.cast(dtypes.int32) - 32).float() * _half(raw[base + 54])
    return term.reduce(hu, arg=(Ops.ADD, 0)) * xd_val
  if ggml_type == IQ2_XS:
    scales = byte(66 + subgroup)
    return sum(dots[h].float() * (((scales >> (h*4)) & 15).float()+0.5) for h in range(2)) * 0.25 * _half(raw[base]) * xd_val
  total = (dots[0]+dots[1]).float() * _half(raw[base])
  if ggml_type == IQ3_S: total *= (1+2*((byte(106+subgroup//2) >> _sh((subgroup%2)*4)) & 15)).float()
  if ggml_type == IQ2_XXS: total *= ((word(6+subgroup*8) >> 28).float()+0.5)*0.25
  return total * xd_val

def _quant_decode_body(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, grids:tuple[UOp, ...], out_features:int|UOp, in_features:int|UOp,
                       waves:UOp, ggml_type:int, cdna:bool, gate:Callable|None=None, xtoken:Callable|None=None, name:str="linear_") -> UOp:
  def group_dot(token:UOp, output:UOp, group:UOp, dep:UOp) -> UOp:
    base = _quant_base(output, group, in_features, ggml_type, raw.dtype.itemsize)
    if gate is not None: base = gate(token, base)
    if xtoken is not None: token = xtoken(token)
    x_reg = UOp.alloc((8,), dtypes.uint32, addrspace=AddrSpace.REG)
    x_reg = x_reg.after(x_reg.store(_amd_load(xq[token, group, 0], 8)))
    return _quant_group_dot(raw, base, group%8, x_reg, xd[token, group], lambda h: xs[token, group, h].load(),
                            ggml_type, cdna, grids[0] if grids else None, dep)
  return _decode_linear(out, out_features, in_features//32, waves, group_dot, name+QUANT_NAMES[ggml_type], share_weights=name == "linear_")

@functools.cache
def _moe_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, sel:UOp, *grids:UOp,
                       out_features:UOp, in_features:UOp, waves:UOp, ggml_type:int, x_share:UOp, cdna:bool=False, ndev:UOp) -> UOp:
  expert_words = out_features * in_features * QUANT_SIZES[ggml_type] // 256 // raw.dtype.itemsize
  local_experts = raw.shape[0]//expert_words
  def expert(row:UOp) -> UOp: return sel[row].cast(dtypes.weakint) - UOp.range(ndev, 4, AxisType.DEVICE)*local_experts
  def gate(row:UOp, base:UOp) -> UOp:
    return (base + expert(row).maximum(0).minimum(local_experts-1) * expert_words).valid((expert(row) >= 0) & (expert(row) < local_experts))
  return _quant_decode_body(out, raw, xq, xd, xs, grids, out_features, in_features, waves, ggml_type, cdna, gate, lambda row: row // x_share, "moe_")

# ******** direct zero-bias-sub v_dot2_f32_f16 GEMV & MoE (eliminates q8_quantize launch) ********

def _direct_gemv_body(out:UOp, raw:UOp, x:UOp, grids:tuple[UOp, ...], out_features:UOp, in_features:UOp,
                      waves:UOp, rows_per_wg:UOp, ggml_type:int, half_mode:bool,
                      gate:Callable|None=None, xtoken:Callable|None=None, name:str="direct_") -> UOp:
  n_rows_outer = out.shape[0]
  row_blks = out_features // rows_per_wg
  rb = UOp.range(n_rows_outer * row_blks, 0, AxisType.GLOBAL)
  tk, row0 = rb // row_blks, (rb % row_blks) * rows_per_wg
  lane, wave = UOp.range(32, 1, AxisType.LOCAL), UOp.range(waves, 2, AxisType.LOCAL)
  tid = wave * 32 + lane
  group_count = in_features // 32
  total_units = group_count * (2 if half_mode else 1)
  steps = (total_units + waves * 32 - 1) // (waves * 32)
  x_tok = xtoken(tk) if xtoken is not None else tk
  grid = grids[0] if grids else None
  c6400, c8080 = UOp.const(0x64006400, dtypes.uint32), UOp.const(0x80808080, dtypes.uint32)
  sel_lo, sel_hi = UOp.const(0x05010500, dtypes.uint32), UOp.const(0x05030502, dtypes.uint32)
  ones_h2, z = UOp.const(0x3c003c00, dtypes.uint32), UOp.const(0.0, dtypes.float32)
  acc = UOp.alloc((rows_per_wg,), dtypes.float32, addrspace=AddrSpace.REG)
  acc = acc.after(acc.store(acc.const_like(0.0)))
  s = UOp.range(steps, 4)
  unit = tid + s * (waves * 32)
  valid_u = unit < total_units
  u_safe = unit.minimum(total_units - 1)
  grp = (u_safe // 2) if half_mode else u_safe
  h_idx = (u_safe % 2) if half_mode else UOp.const(0, dtypes.weakint)
  sub = grp % 8
  n_chunks, x_base = (4, grp * 32 + h_idx * 16) if half_mode else (8, grp * 32)
  x_h2: list[tuple[UOp, UOp]] = []
  for j in range(n_chunks):
    xv = _amd_load(x[x_tok, x_base + j * 4], 4)
    if x.dtype == dtypes.float16:
      xh = xv.bitcast(dtypes.uint32)
      x_h2.append((xh[0], xh[1]))
    else:
      x_h2.append((_cvt_pk_f16(xv[0], xv[1]), _cvt_pk_f16(xv[2], xv[3])))
  xs0 = z
  for j in range(4): xs0 = _amd_fdot2(ones_h2, x_h2[j][1], _amd_fdot2(ones_h2, x_h2[j][0], xs0))
  xs1 = z
  if not half_mode:
    for j in range(4, 8): xs1 = _amd_fdot2(ones_h2, x_h2[j][1], _amd_fdot2(ones_h2, x_h2[j][0], xs1))
  xs_all = xs0 + xs1

  def dot4_nb(wu:UOp, j_idx:int, cur_d:UOp, signed:bool=False) -> UOp:
    ux = (wu ^ c8080) if signed else wu
    p0, p1 = _amd_byte_perm(c6400, ux, sel_lo), _amd_byte_perm(c6400, ux, sel_hi)
    return _amd_fdot2(p1, x_h2[j_idx][1], _amd_fdot2(p0, x_h2[j_idx][0], cur_d))

  r = UOp.range(rows_per_wg, 6)
  base = _quant_base(row0 + r, grp, in_features, ggml_type, raw.dtype.itemsize)
  if gate is not None: base = gate(tk, base)
  if ggml_type == Q8_0:
    if half_mode:
      wv = _amd_load(raw[base + 1 + h_idx * 8], 8, stream=True).bitcast(dtypes.uint32)
      d_acc = z
      for j in range(4): d_acc = dot4_nb(wv[j], j, d_acc, True)
      d_acc = d_acc - UOp.const(1152.0, dtypes.float32) * xs0
    else:
      w0 = _amd_load(raw[base + 1], 8, stream=True).bitcast(dtypes.uint32)
      w1 = _amd_load(raw[base + 9], 8, stream=True).bitcast(dtypes.uint32)
      d_acc = z
      for j in range(4): d_acc = dot4_nb(w0[j], j, d_acc, True)
      for j in range(4): d_acc = dot4_nb(w1[j], 4 + j, d_acc, True)
      d_acc = d_acc - UOp.const(1152.0, dtypes.float32) * xs_all
    gdot = d_acc * _half(_amd_load(raw[base]))
  elif ggml_type in (Q4_K, Q5_K):
    sh = ((sub & 1) * 4).cast(dtypes.uint32)
    d, dmin, sc, mn = _q5_scales(raw, base, sub, stream=True)
    ds, dm = d * sc, dmin * mn + UOp.const(1024.0, dtypes.float32) * (d * sc)
    if half_mode:
      qv = _amd_load(raw[base + (4 if ggml_type == Q4_K else 12) + (sub // 2) * 8 + h_idx * 4], 4, stream=True)
      hv = _amd_load(raw[base + 4 + h_idx * 4], 4, stream=True) if ggml_type == Q5_K else None
      d_acc = z
      for j in range(4):
        w = (qv[j] >> sh) & 0x0f0f0f0f
        if hv is not None: w = w | (((hv[j] >> sub.cast(dtypes.uint32)) & 0x01010101) << 4)
        d_acc = dot4_nb(w, j, d_acc, False)
      gdot = d_acc * ds - xs0 * dm
    else:
      qv = _amd_load(raw[base + (4 if ggml_type == Q4_K else 12) + (sub // 2) * 8], 8, stream=True)
      hv = _amd_load(raw[base + 4], 8, stream=True) if ggml_type == Q5_K else None
      d_acc = z
      for j in range(8):
        w = (qv[j] >> sh) & 0x0f0f0f0f
        if hv is not None: w = w | (((hv[j] >> sub.cast(dtypes.uint32)) & 0x01010101) << 4)
        d_acc = dot4_nb(w, j, d_acc, False)
      gdot = d_acc * ds - xs_all * dm
  elif ggml_type == Q6_K:
    ql_b, qh_b = base + (sub // 4) * 32 + (sub % 2) * 16, base + 64 + (sub // 4) * 16
    ql_sh, qh_sh = ((sub % 4 // 2) * 4).cast(dtypes.uint32), ((sub % 4) * 2).cast(dtypes.uint32)
    sc_w, d_sc = _amd_load(raw[base + 96 + sub]), _half(_amd_load(raw[base + 104]))
    if half_mode:
      ql = _amd_load(raw[ql_b + h_idx * 8], 8, stream=True).bitcast(dtypes.uint32)
      qh = _amd_load(raw[qh_b + h_idx * 8], 8, stream=True).bitcast(dtypes.uint32)
      s_h = ((sc_w >> (h_idx * 8).cast(dtypes.uint32)) & 255).cast(dtypes.uint8).bitcast(dtypes.int8).float()
      d_acc = z
      for j in range(4):
        d_acc = dot4_nb(((ql[j] >> ql_sh) & 0x0f0f0f0f) | (((qh[j] >> qh_sh) & 0x03030303) << 4), j, d_acc, False)
      gdot = (d_acc - UOp.const(1056.0, dtypes.float32) * xs0) * s_h * d_sc
    else:
      ql0, ql1 = _amd_load(raw[ql_b], 8, stream=True).bitcast(dtypes.uint32), _amd_load(raw[ql_b + 8], 8, stream=True).bitcast(dtypes.uint32)
      qh0, qh1 = _amd_load(raw[qh_b], 8, stream=True).bitcast(dtypes.uint32), _amd_load(raw[qh_b + 8], 8, stream=True).bitcast(dtypes.uint32)
      s0, s1 = (sc_w & 255).cast(dtypes.uint8).bitcast(dtypes.int8).float(), ((sc_w >> 8) & 255).cast(dtypes.uint8).bitcast(dtypes.int8).float()
      d0, d1 = z, z
      for j in range(4): d0 = dot4_nb(((ql0[j] >> ql_sh) & 0x0f0f0f0f) | (((qh0[j] >> qh_sh) & 0x03030303) << 4), j, d0, False)
      for j in range(4): d1 = dot4_nb(((ql1[j] >> ql_sh) & 0x0f0f0f0f) | (((qh1[j] >> qh_sh) & 0x03030303) << 4), 4 + j, d1, False)
      gdot = ((d0 - UOp.const(1056.0, dtypes.float32) * xs0) * s0 + (d1 - UOp.const(1056.0, dtypes.float32) * xs1) * s1) * d_sc
  elif ggml_type == Q2_K:
    qs_b, sh = base + 8 + (sub // 4) * 16, ((sub % 4) * 2).cast(dtypes.uint32)
    sc_u, dm = _amd_load(raw[base + sub]), _amd_load(raw[base + 40], 2, stream=True)
    dh, dmh = _half(dm[0]), _half(dm[1])
    if half_mode:
      qs = _amd_load(raw[qs_b + h_idx * 8], 8, stream=True).bitcast(dtypes.uint32)
      sc_b = ((sc_u >> (h_idx * 8).cast(dtypes.uint32)) & 255).cast(dtypes.uint32)
      d_acc = z
      for j in range(4): d_acc = dot4_nb((qs[j] >> sh) & 0x03030303, j, d_acc, False)
      ds = (sc_b & 15).float() * dh
      gdot = d_acc * ds - xs0 * ((sc_b >> 4).float() * dmh + UOp.const(1024.0, dtypes.float32) * ds)
    else:
      qs0, qs1 = _amd_load(raw[qs_b], 8, stream=True).bitcast(dtypes.uint32), _amd_load(raw[qs_b + 8], 8, stream=True).bitcast(dtypes.uint32)
      sc0, sc1 = (sc_u & 255).cast(dtypes.uint32), ((sc_u >> 8) & 255).cast(dtypes.uint32)
      d0, d1 = z, z
      for j in range(4): d0 = dot4_nb((qs0[j] >> sh) & 0x03030303, j, d0, False)
      for j in range(4): d1 = dot4_nb((qs1[j] >> sh) & 0x03030303, 4 + j, d1, False)
      ds0, ds1 = (sc0 & 15).float() * dh, (sc1 & 15).float() * dh
      gdot = ((d0 * ds0 - xs0 * ((sc0 >> 4).float() * dmh + UOp.const(1024.0, dtypes.float32) * ds0)) +
              (d1 * ds1 - xs1 * ((sc1 >> 4).float() * dmh + UOp.const(1024.0, dtypes.float32) * ds1)))
  elif ggml_type == IQ4_XS:
    pv = _amd_load(raw[base + 2 + sub * 4], 4, stream=True)
    d_acc = z
    if half_mode:
      sh_n = (h_idx * 4).cast(dtypes.uint32)
      for j in range(4): d_acc = dot4_nb(_nibble_lut(pv[j], sh_n, IQ4NL_U8_LUT), j, d_acc, False)
      d_acc = d_acc - UOp.const(1152.0, dtypes.float32) * xs0
    else:
      for j in range(4): d_acc = dot4_nb(_nibble_lut(pv[j], 0, IQ4NL_U8_LUT), j, d_acc, False)
      for j in range(4): d_acc = dot4_nb(_nibble_lut(pv[j], 4, IQ4NL_U8_LUT), 4 + j, d_acc, False)
      d_acc = d_acc - UOp.const(1152.0, dtypes.float32) * xs_all
    gdot = d_acc * _iq4_scale(raw, base, sub, stream=True)
  elif ggml_type == MXFP4:
    raw_u32 = raw.bitcast(dtypes.uint32)
    w_idx, b_sh = base // 4, (base % 4).cast(dtypes.uint32)
    w03, w4 = _amd_load(raw_u32[w_idx], 4, stream=True), _amd_load(raw_u32[w_idx + 4])
    ws = [w03[0], w03[1], w03[2], w03[3], w4]
    sc, s1, is3 = _e8m0_halved((ws[0] >> (b_sh * 8)) & 255), (b_sh + 1) & 3, b_sh.eq(3)
    pws = [is3.where(ws[j + 1], _perm_alignbyte(ws[j + 1], ws[j], s1)) for j in range(4)]
    d_acc = z
    if half_mode:
      sh_n = (h_idx * 4).cast(dtypes.uint32)
      for j in range(4): d_acc = dot4_nb(_nibble_lut(pws[j], sh_n, MXFP4_LUT), j, d_acc, True)
      d_acc = d_acc - UOp.const(1152.0, dtypes.float32) * xs0
    else:
      for j in range(4): d_acc = dot4_nb(_nibble_lut(pws[j], 0, MXFP4_LUT), j, d_acc, True)
      for j in range(4): d_acc = dot4_nb(_nibble_lut(pws[j], 4, MXFP4_LUT), 4 + j, d_acc, True)
      d_acc = d_acc - UOp.const(1152.0, dtypes.float32) * xs_all
    gdot = d_acc * sc
  elif ggml_type == IQ3_S:
    assert grid is not None
    qh_b = (_amd_load(raw[base + 33 + sub // 2]).cast(dtypes.uint32) >> ((sub & 1) * 8).cast(dtypes.uint32)) & 255
    sc_n = (_amd_load(raw[base + 53 + sub // 4]).cast(dtypes.uint32) >> ((sub & 3) * 4).cast(dtypes.uint32)) & 15
    d_sc, d_acc = _half(_amd_load(raw[base])) * (1 + 2 * sc_n).float(), z
    if half_mode:
      qs_w = _amd_load(raw[base + 1 + sub * 4 + h_idx * 2], 2, stream=True).bitcast(dtypes.uint32)[0]
      sg_w = _amd_load(raw[base + 37 + sub * 2 + h_idx]).cast(dtypes.uint32)
      for j in range(4):
        idx = ((qs_w >> (j * 8)) & 255) | (((qh_b >> (h_idx * 4 + j).cast(dtypes.uint32)) & 1) << 8)
        d_acc = dot4_nb(_iq_signed_word(grid[idx].load(), (sg_w >> (j * 4)) & 15), j, d_acc, True)
      d_acc = d_acc - UOp.const(1152.0, dtypes.float32) * xs0
    else:
      qs_u = _amd_load(raw[base + 1 + sub * 4], 4, stream=True).bitcast(dtypes.uint32)
      sg_u = _amd_load(raw[base + 37 + sub * 2], 2, stream=True).bitcast(dtypes.uint32)[0]
      for j in range(8):
        idx = ((qs_u[j // 4] >> ((j % 4) * 8)) & 255) | (((qh_b >> j) & 1) << 8)
        d_acc = dot4_nb(_iq_signed_word(grid[idx].load(), (sg_u >> ((j // 2) * 8 + (j % 2) * 4)) & 15), j, d_acc, True)
      d_acc = d_acc - UOp.const(1152.0, dtypes.float32) * xs_all
    gdot = d_acc * d_sc
  else:  # IQ1_S
    assert grid is not None
    qh_u = _amd_load(raw[base + 17 + sub]).cast(dtypes.uint32)
    sc = (((qh_u >> 12) & 7) * 2 + 1).float() * _half(_amd_load(raw[base]))
    zero = (qh_u >> 15).eq(0).where(-0.125, 0.125) + UOp.const(1152.0, dtypes.float32)
    d_acc = z
    if half_mode:
      qs_h = _amd_load(raw[base + 1 + sub * 2 + h_idx]).cast(dtypes.uint32)
      for p in range(2):
        idx = ((qs_h >> (p * 8)) & 255) | (((qh_u >> (3 * (h_idx * 2 + p)).cast(dtypes.uint32)) & 7) << 8)
        g01 = _amd_load(grid[idx * 2], 2)
        d_acc = dot4_nb(g01[1], p * 2 + 1, dot4_nb(g01[0], p * 2, d_acc, True), True)
      gdot = (d_acc - xs0 * zero) * sc
    else:
      qs_u = _amd_load(raw[base + 1 + sub * 2], 2, stream=True).bitcast(dtypes.uint32)[0]
      for p in range(4):
        idx = ((qs_u >> (p * 8)) & 255) | (((qh_u >> (3 * p)) & 7) << 8)
        g01 = _amd_load(grid[idx * 2], 2)
        d_acc = dot4_nb(g01[1], p * 2 + 1, dot4_nb(g01[0], p * 2, d_acc, True), True)
      gdot = (d_acc - xs_all * zero) * sc
  upd = acc[r].store(acc.after(s)[r].load() + valid_u.where(gdot, z)).end(r, s)
  lds = UOp.alloc((rows_per_wg, waves), dtypes.float32, addrspace=AddrSpace.LOCAL)
  r_w = UOp.range(rows_per_wg, 7)
  lds_r = lds.after(lds[r_w, wave.valid(lane.eq(0))].store(warp_reduce(acc.after(upd)[r_w].load(), full_wave=True)).end(r_w).barrier())
  r_o, wr = UOp.range(rows_per_wg, 8), UOp.range(waves, 9)
  total = waves.eq(1).where(lds_r[r_o, 0].load(), lds_r[r_o, wr].load().reduce(wr, arg=(Ops.ADD, 0)))
  return out[tk, (row0 + r_o).valid(wave.eq(0) & lane.eq(0))].store(total.cast(out.dtype)).end(r_o, rb, wave, lane).sink(
    arg=KernelInfo(name=name + QUANT_NAMES[ggml_type], opts_to_apply=()))

@functools.cache
def _quant_direct_gemv_kernel(out:UOp, raw:UOp, x:UOp, *grids:UOp, out_features:UOp, in_features:UOp, waves:UOp,
                              rows_per_wg:UOp, ggml_type:int, half_mode:bool=False) -> UOp:
  return _direct_gemv_body(out, raw, x, grids, out_features, in_features, waves, rows_per_wg, ggml_type, half_mode, name="direct_linear_")

@functools.cache
def _moe_direct_kernel(out:UOp, raw:UOp, x:UOp, sel:UOp, *grids:UOp, out_features:UOp, in_features:UOp, waves:UOp,
                       rows_per_wg:UOp, ggml_type:int, x_share:UOp, ndev:UOp, half_mode:bool=False) -> UOp:
  expert_words = out_features * in_features * QUANT_SIZES[ggml_type] // 256 // raw.dtype.itemsize
  local_experts = raw.shape[0] // expert_words
  def expert(row:UOp) -> UOp: return sel[row].cast(dtypes.weakint) - UOp.range(ndev, 10, AxisType.DEVICE) * local_experts
  def gate(row:UOp, base:UOp) -> UOp:
    return (base + expert(row).maximum(0).minimum(local_experts - 1) * expert_words).valid((expert(row) >= 0) & (expert(row) < local_experts))
  return _direct_gemv_body(out, raw, x, grids, out_features, in_features, waves, rows_per_wg, ggml_type, half_mode,
                           gate, lambda row: row // x_share, "direct_moe_")

# ******** weight-reusing multi-token DP4A GEMV (2 <= tokens <= 8) ********

@functools.cache
def _quant_decode_m_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, *, tokens:UOp, out_features:UOp, in_features:UOp,
                           waves:UOp, ggml_type:int, cdna:bool=False) -> UOp:
  # One WG per output row reads quantized weights ONCE and reuses them across all M tokens against L1-resident xq.
  row = UOp.range(out_features, 0, AxisType.GLOBAL)
  lane, wave = UOp.range(32, 1, AxisType.LOCAL), UOp.range(waves, 2, AxisType.LOCAL)
  group_count = in_features // 32
  steps = (group_count + waves * 32 - 1) // (waves * 32)
  acc = UOp.alloc((tokens,), dtypes.float32, addrspace=AddrSpace.REG)
  acc = acc.after(acc.store(acc.const_like(0.0)))
  s = UOp.range(steps, 4)
  grp_raw = lane + (wave + s * waves) * 32
  valid_g, grp = grp_raw < group_count, grp_raw.minimum(group_count - 1)
  sub, base = grp % 8, _quant_base(row, grp, in_features, ggml_type, raw.dtype.itemsize)
  z_i, z_f = UOp.const(0, dtypes.int32), UOp.const(0.0, dtypes.float32)
  if ggml_type == Q4_K:
    qv = _amd_load(raw[base + 4 + (sub // 2) * 8], 8, stream=True)
    sh = ((sub & 1) * 4).cast(dtypes.uint32)
    d, dmin, sc, mn = _q5_scales(raw, base, sub, stream=True)
    ds, dm = d * sc, dmin * mn
    ws = [(qv[j] >> sh) & 0x0f0f0f0f for j in range(8)]
  elif ggml_type == Q8_0:
    w0, w1 = _amd_load(raw[base + 1], 8, stream=True).bitcast(dtypes.uint32), _amd_load(raw[base + 9], 8, stream=True).bitcast(dtypes.uint32)
    sc, ws = _half(_amd_load(raw[base])), [w0[0], w0[1], w0[2], w0[3], w1[0], w1[1], w1[2], w1[3]]
  elif ggml_type == IQ4_XS:
    pv, sc = _amd_load(raw[base + 2 + sub * 4], 4, stream=True), _iq4_scale(raw, base, sub, stream=True)
    ws = [_nibble_lut(pv[j], 0, IQ4NL_LUT) for j in range(4)] + [_nibble_lut(pv[j], 4, IQ4NL_LUT) for j in range(4)]
  else:  # MXFP4
    raw_u32 = raw.bitcast(dtypes.uint32)
    w_idx, b_sh = base // 4, (base % 4).cast(dtypes.uint32)
    w03, w4 = _amd_load(raw_u32[w_idx], 4, stream=True), _amd_load(raw_u32[w_idx + 4])
    rws = [w03[0], w03[1], w03[2], w03[3], w4]
    sc, s1, is3 = _e8m0_halved((rws[0] >> (b_sh * 8)) & 255), (b_sh + 1) & 3, b_sh.eq(3)
    pws = [is3.where(rws[j + 1], _perm_alignbyte(rws[j + 1], rws[j], s1)) for j in range(4)]
    ws = [_nibble_lut(pws[j], 0, MXFP4_LUT) for j in range(4)] + [_nibble_lut(pws[j], 4, MXFP4_LUT) for j in range(4)]
  t = UOp.range(tokens, 5)
  xh0, xh1 = _amd_load(xq[t, grp, 0], 4), _amd_load(xq[t, grp, 4], 4)
  idot = z_i
  for j in range(4): idot = _amd_dp4a(ws[j], xh0[j], idot, cdna)
  for j in range(4): idot = _amd_dp4a(ws[4 + j], xh1[j], idot, cdna)
  if ggml_type == Q4_K:
    gdot = (idot.float() * ds - (xs[t, grp, 0].load() + xs[t, grp, 1].load()) * dm) * xd[t, grp].load()
  else:
    gdot = (idot.float() * sc) * xd[t, grp].load()
  upd = acc[t].store(acc.after(s)[t].load() + valid_g.where(gdot, z_f)).end(t, s)
  lds = UOp.alloc((tokens, waves), dtypes.float32, addrspace=AddrSpace.LOCAL)
  t_w = UOp.range(tokens, 7)
  lds_r = lds.after(lds[t_w, wave.valid(lane.eq(0))].store(warp_reduce(acc.after(upd)[t_w].load(), full_wave=True)).end(t_w).barrier())
  t_o, wr = UOp.range(tokens, 8), UOp.range(waves, 9)
  total = waves.eq(1).where(lds_r[t_o, 0].load(), lds_r[t_o, wr].load().reduce(wr, arg=(Ops.ADD, 0)))
  return out[t_o, row.valid(wave.eq(0) & lane.eq(0))].store(total.cast(out.dtype)).end(t_o, row, wave, lane).sink(
    arg=KernelInfo(name="linear_m_" + QUANT_NAMES[ggml_type], opts_to_apply=()))

# ******** FP16 WMMA quantized GEMM ********

def _iq_half_elem_f32(table:UOp, base_idx:UOp, s4:UOp, sc_f:UOp, b_idx:UOp) -> UOp:
  h_pair = table[base_idx + b_idx // 2].load()
  sign = ((s4 >> _sh(b_idx)) & 1) << 15
  h16 = (((h_pair >> _sh((b_idx & 1) * 16)) & 0xffff) ^ sign).cast(dtypes.uint16).bitcast(dtypes.half)
  return (h16.float() * sc_f).cast(dtypes.half)

# kgroups: groups of 16 lanes that split k in a 16x16x16 fragment, RDNA3 copies a and b in both halves
def _wmma_words(lane_group:UOp, kgroups:int) -> tuple[UOp|int, ...]:
  return tuple(range(4)) if kgroups == 1 else tuple(lane_group + kgroups * j for j in range(4 // kgroups))

def _e2m1_h2_pair(pw:UOp, c_lo:UOp, c_hi:UOp, sc2:UOp) -> tuple[UOp, UOp]:
  # Unpack 4 low-nibble e2m1 values into 2 scaled half2 vectors using 3 byte_perms and 2 half2 multiplies.
  fp_bytes = _amd_byte_perm(c_hi, c_lo, pw & 0x07070707) | ((pw & 0x08080808) << 4)
  zero_u = UOp.const(0, dtypes.uint32)
  v01 = _amd_byte_perm(fp_bytes, zero_u, UOp.const(0x05000400, dtypes.uint32)).reshape((1,)).bitcast(dtypes.half) * sc2
  v23 = _amd_byte_perm(fp_bytes, zero_u, UOp.const(0x07000600, dtypes.uint32)).reshape((1,)).bitcast(dtypes.half) * sc2
  return v01, v23

def _e2m1_half4(pw:UOp, c_lo:UOp, c_hi:UOp, sc2:UOp) -> tuple[UOp, UOp, UOp, UOp]:
  v01, v23 = _e2m1_h2_pair(pw, c_lo, c_hi, sc2)
  return v01[0], v01[1], v23[0], v23[1]

def _make_wmma_dequant(raw:UOp, ggml_type:int, lane_group:UOp, kgroups:int, k_u:UOp,
                       table:UOp|None) -> Callable[[UOp, UOp, UOp], UOp|tuple[UOp, ...]]:
  w_u, b_u = k_u // 4, k_u % 4
  w_idx = w_u if kgroups == 1 else (lane_group + kgroups * w_u)
  word_indices = _wmma_words(lane_group, kgroups)
  if ggml_type in (Q4_K, Q5_K):
    def dequant_q5(base:UOp, subgroup:UOp, half:UOp) -> tuple[UOp, ...]:
      d, dmin, scale, minimum = _q5_scales(raw, base, subgroup, stream=False)
      ds, dm = d * scale, dmin * minimum
      qs_base = base + (4 if ggml_type == Q4_K else 12) + (subgroup // 2) * 8 + half * 4
      sh_s = ((subgroup & 1) * 4).cast(dtypes.uint32)
      qs_vec = _amd_load(raw[qs_base], 4) if word_indices == (0, 1, 2, 3) else None
      qh_vec = _amd_load(raw[base + 4 + half * 4], 4) if ggml_type == Q5_K and word_indices == (0, 1, 2, 3) else None
      out_vals: list[UOp] = []
      for qi, wi in enumerate(word_indices):
        word = ((qs_vec[qi] if qs_vec is not None else raw[qs_base + wi].load()) >> sh_s) & 0x0f0f0f0f
        if ggml_type == Q5_K:
          word |= (((qh_vec[qi] if qh_vec is not None else raw[base + 4 + half * 4 + wi].load()) >> subgroup.cast(dtypes.uint32)) & 0x01010101) << 4
        out_vals.extend(((word >> (b * 8)) & 255).float().alu(Ops.MULACC, ds, -dm).cast(dtypes.half) for b in range(4))
      return tuple(out_vals)
    return dequant_q5
  if ggml_type == Q6_K:
    def dequant_q6(base:UOp, subgroup:UOp, half:UOp) -> tuple[UOp, ...]:
      scale = _half(raw[base + 104]) * _load_byte(raw, base, 192 + subgroup * 2 + half).cast(dtypes.uint8).bitcast(dtypes.int8).float()
      ql_base, qh_base = base + (subgroup // 4) * 32 + (subgroup % 2) * 16 + half * 8, base + 64 + (subgroup // 4) * 16 + half * 8
      ql_sh, qh_sh = ((subgroup % 4 // 2) * 4).cast(dtypes.uint32), ((subgroup % 4) * 2).cast(dtypes.uint32)
      out_vals: list[UOp] = []
      for wi in word_indices:
        ql_w = _amd_load(raw[ql_base + wi * 2], 2).bitcast(dtypes.uint32)[0]
        qh_w = _amd_load(raw[qh_base + wi * 2], 2).bitcast(dtypes.uint32)[0]
        w = ((ql_w >> ql_sh) & 0x0f0f0f0f) | (((qh_w >> qh_sh) & 0x03030303) << 4)
        out_vals.extend(((w >> (b * 8)) & 63).float().alu(Ops.MULACC, scale, -32.0 * scale).cast(dtypes.half) for b in range(4))
      return tuple(out_vals)
    return dequant_q6
  if ggml_type == IQ4_XS:
    def dequant_iq4(base:UOp, subgroup:UOp, half:UOp) -> tuple[UOp, ...]:
      scale, sh = _iq4_scale(raw, base, subgroup), _sh(half * 4)
      qs_base = base + 2 + subgroup * 4
      qs_vec = _amd_load(raw[qs_base], 4) if word_indices == (0, 1, 2, 3) else None
      out_vals: list[UOp] = []
      for qi, wi in enumerate(word_indices):
        w = _nibble_lut(qs_vec[qi] if qs_vec is not None else raw[qs_base + wi].load(), sh, IQ4NL_LUT)
        out_vals.extend(((w >> (b * 8)).cast(dtypes.uint8).bitcast(dtypes.int8).float() * scale).cast(dtypes.half) for b in range(4))
      return tuple(out_vals)
    return dequant_iq4
  if ggml_type == MXFP4:
    def dequant_mxfp4(base:UOp, subgroup:UOp, half:UOp) -> tuple[UOp, ...]:
      b_base = base + subgroup * 17
      e = raw[b_base].cast(dtypes.uint32)
      small = e < 116
      # Shift small exponents up by 2^4 (2^-6 scaled e2m1) so subnormal fp16 results stay bit-exact down to e=100.
      c_lo = small.where(UOp.const(0x26242000, dtypes.uint32), UOp.const(0x36343000, dtypes.uint32))
      c_hi = small.where(UOp.const(0x2e2c2a28, dtypes.uint32), UOp.const(0x3e3c3a38, dtypes.uint32))
      sc2 = _splat_h2(_e8m0_halved(e + small.where(UOp.const(7, dtypes.uint32), UOp.const(3, dtypes.uint32))).cast(dtypes.half))
      sh = _sh(half * 4)
      return tuple(v for wi in word_indices for v in _e2m1_half4(_load_u32(raw, b_base, 1 + wi * 4) >> sh, c_lo, c_hi, sc2))
    return dequant_mxfp4
  def dequant(base:UOp, subgroup:UOp, half:UOp) -> UOp|tuple[UOp, ...]:
    if ggml_type in SMALL_BLOCK_QUANTS: base += subgroup * SMALL_BLOCK_QUANTS[ggml_type]  # eight independent blocks per 256 weights
    if ggml_type == Q8_0:
      scale2 = _splat_h2(raw[base].load().bitcast(dtypes.half))
      neg1152_2 = UOp.const(0xe480e480, dtypes.uint32).reshape((1,)).bitcast(dtypes.half)
      c6400, c8080 = UOp.const(0x64006400, dtypes.uint32), UOp.const(0x80808080, dtypes.uint32)
      sel_lo, sel_hi = UOp.const(0x05010500, dtypes.uint32), UOp.const(0x05030502, dtypes.uint32)
      hv = _amd_load(raw[base + 1 + half * 8], 8).bitcast(dtypes.uint32) if word_indices == (0, 1, 2, 3) else None
      out_vals: list[UOp] = []
      for qi, wi in enumerate(word_indices):
        ux = (hv[qi] if hv is not None else _amd_load(raw[base + 1 + half * 8 + wi * 2], 2).bitcast(dtypes.uint32)[0]) ^ c8080
        v01 = (_amd_byte_perm(c6400, ux, sel_lo).reshape((1,)).bitcast(dtypes.half) + neg1152_2) * scale2
        v23 = (_amd_byte_perm(c6400, ux, sel_hi).reshape((1,)).bitcast(dtypes.half) + neg1152_2) * scale2
        out_vals.extend((v01[0], v01[1], v23[0], v23[1]))
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
    elif ggml_type == IQ1_S: scale, zero = _iq1_scale(raw, base, subgroup)
    elif ggml_type == NVFP4:
      e = byte((subgroup // 2) * 36 + (subgroup % 2) * 2 + half).cast(dtypes.uint32)
      sc2 = _splat_h2(e.eq(0x7f).where(UOp.const(0, dtypes.uint32), (e & 0x7f) << 7).cast(dtypes.uint16).bitcast(dtypes.half))
      # Exact FP16 high bytes for 256x-scaled e2m1 magnitudes [0, 128, 256, 384, 512, 768, 1024, 1536] (low bytes 0x00)
      c_lo, c_hi = UOp.const(0x5e5c5800, dtypes.uint32), UOp.const(0x66646260, dtypes.uint32)
      p_base = base + (subgroup // 2) * 9 + 1 + ((subgroup % 2) * 2 + half) * 2
      return tuple(v for wi in word_indices for v in _e2m1_half4(raw[p_base + wi % 2].load() >> _sh(4 * (wi // 2)), c_lo, c_hi, sc2))
    else:
      scale = _half(raw[base])
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
    if ggml_type == IQ4_NL:
      words = tuple(_quant_word(raw, base, subgroup, half * 4 + wi, ggml_type, table) for wi in word_indices)
      w_reg = UOp.alloc((len(words),), dtypes.uint32, addrspace=AddrSpace.REG)
      word = w_reg.after(w_reg.store(UOp.stack(*words)))[w_u].load()
    else:
      word = _quant_word(raw, base, subgroup, half * 4 + w_idx, ggml_type, table)
    return (((word >> _sh(b_u * 8)).cast(dtypes.uint8).bitcast(dtypes.int8).float() - zero).alu(Ops.MULACC, scale, -minimum)).cast(dtypes.half)
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
  if ggml_type in (Q4_K, Q5_K, Q2_K, Q3_K, Q6_K):
    block, subgroup, half = UOp.range(in_features // GGML_BLOCK_SIZE, 4), UOp.range(8, 5), UOp.range(2, 6)
    red_ranges: tuple[UOp, ...] = (block, subgroup, half)
  else:
    group, half = UOp.range(in_features // Q8_GROUP_SIZE, 4), UOp.range(2, 5)
    block, subgroup = group // 8, group % 8
    red_ranges = (group, half)
  half_group = (block * 8 + subgroup) * 2 + half
  k_u, m_u, mt_u = UOp.range(frag_k, 10, AxisType.UPCAST), UOp.range(rows, 11, AxisType.UPCAST), UOp.range(m_tiles, 12, AxisType.UPCAST)
  w_idx = (k_u // 4) if kgroups == 1 else (lane_group + kgroups * (k_u // 4))
  a_elem = x[(token_block * m_tiles + mt_u) * 16 + col, half_group * 16 + w_idx * 4 + (k_u % 4)].load()
  b_raw = _make_wmma_dequant(raw, ggml_type, lane_group, kgroups, k_u, grid)(output * output_words + block * type_words, subgroup, half)
  if isinstance(b_raw, tuple):
    b_reg = UOp.alloc((frag_k,), dtypes.float16, addrspace=AddrSpace.REG)
    b_elem = b_reg.after(b_reg.after(*red_ranges).store(UOp.stack(*b_raw)))[k_u].load()
  else: b_elem = b_raw
  tc_axes = (((k_u.arg, frag_k), (m_u.arg, 1)), ((k_u.arg, frag_k), (m_u.arg, 1)), ((k_u.arg, 1), (m_u.arg, rows)))
  wmma = UOp.wmma(a_elem, b_elem, UOp.const((0.0,) * rows, dtypes.float32), WMMA_ARG[0], lanes, tc_upcast_axes=tc_axes)
  res = wmma.reduce(*red_ranges, arg=(Ops.ADD, 0))
  row_in_tile = lane_group * rows + m_u if kgroups > 1 else m_u * 2 + lane_group
  st = UOp(Ops.STORE, src=(out[(token_block * m_tiles + mt_u) * 16 + row_in_tile, output], res.cast(out.dtype)))
  return st.end(k_u, m_u, mt_u, token_block, output_block, lane).sink(
    arg=KernelInfo(name=f"linear_{QUANT_NAMES[ggml_type]}_f16_wmma", optimized=True))

@functools.cache
def _quant_linear_2d_wmma_kernel(out:UOp, raw:UOp, x:UOp, *grids:UOp,
                                 m_blks:UOp, n_blks:UOp, wm:UOp, wn:UOp, tm:UOp, tn:UOp, in_features:UOp,
                                 split_k:UOp, ggml_type:int, rdna4:bool=True) -> UOp:
  # Pipelined 2D LDS-tiled FP16 WMMA (BK=64) with 16B XOR-swizzled bank-free ds_load_b128 on BOTH RDNA4 and RDNA3,
  # register prefetching (2 barriers/tile), half-qs Q4_K, packed half2 dequant, aligned u32 + byte_perm, and uneven split-K.
  bm, bn, waves, threads = wm * tm * 16, wn * tn * 16, wm * wn, wm * wn * 32
  tok = m_blks * bm
  gid = UOp.range(m_blks * n_blks, 0, AxisType.GLOBAL)
  swz = (m_blks % 4).eq(0).where(4, (m_blks % 2).eq(0).where(2, 1))
  grp, in_grp = gid // (swz * n_blks), gid % (swz * n_blks)
  block_m, block_n = grp * swz + in_grp % swz, in_grp // swz
  sk_r = UOp.range(split_k, 3, AxisType.GLOBAL)
  total_kt = in_features // 64
  base_kt, rem_kt = total_kt // split_k, total_kt % split_k
  k_tile_base = sk_r * base_kt + sk_r.minimum(rem_kt)
  k_tiles_me = base_kt + (sk_r < rem_kt).where(UOp.const(1, dtypes.weakint), UOp.const(0, dtypes.weakint))
  lane, wave = UOp.range(32, -1, AxisType.WARP), UOp.range(waves, 2, AxisType.LOCAL)
  tid, wave_m, wave_n, col, lane_group = wave * 32 + lane, wave // wn, wave % wn, lane % 16, lane // 16
  grid = None
  if grids:
    g_len = int(grids[0].numel())
    grid = UOp.alloc((g_len,), dtypes.uint32, addrspace=AddrSpace.LOCAL)
    g_i = UOp.range((g_len + threads - 1) // threads, 20)
    g_off = tid + g_i * threads
    grid = grid.after(grid[g_off.valid(g_off < g_len)].store(grids[0][g_off.valid(g_off < g_len)].load()).end(g_i).barrier())
  # 16B XOR-swizzled LDS (bm x 8 x 8 half) for BOTH RDNA4 & RDNA3: zero bank conflicts, exact 32KB (128x128) or 48KB (256x128).
  A_lds = UOp.alloc((bm, 8, 8), dtypes.half, addrspace=AddrSpace.LOCAL)
  B_lds = UOp.alloc((bn, 8, 8), dtypes.half, addrspace=AddrSpace.LOCAL)
  A_u32 = A_lds.flatten().bitcast(dtypes.uint32).reshape(bm, 8, 4)
  B_h2 = B_lds.reshape(bn, 8, 4, 2)
  x_u32 = x.flatten().bitcast(dtypes.uint32)
  acc = UOp.alloc((tm, tn, 8), dtypes.float32, addrspace=AddrSpace.REG)
  acc = acc.after(acc.store(acc.const_like(0.0)))
  a_iters, b_iters = (2 * tm) // wn, tn // wm
  fast_b = ggml_type in (Q8_0, Q4_K, IQ4_XS, MXFP4)
  raw_u32 = raw.bitcast(dtypes.uint32) if fast_b else raw
  type_words = QUANT_SIZES[ggml_type] // raw.dtype.itemsize
  output_words = in_features * type_words // GGML_BLOCK_SIZE

  def load_a_reg(kt:UOp, dep:tuple[UOp, ...], rid:int) -> UOp:
    buf = UOp.alloc((a_iters, 8), dtypes.uint32, addrspace=AddrSpace.REG)
    ai = UOp.range(a_iters, rid)
    bid = tid + ai * threads
    am, ak = bid // 4, bid % 4
    a_base = (block_m * bm + am) * (in_features // 2) + kt * 32 + ak * 8
    v0, v1 = _amd_load(x_u32[a_base], 4), _amd_load(x_u32[a_base + 4], 4)
    sts = [buf.after(*dep)[ai, j].store(v0[j]) for j in range(4)] + [buf.after(*dep)[ai, 4 + j].store(v1[j]) for j in range(4)]
    return buf.after(UOp.group(*sts).end(ai))

  def store_a_lds(areg:UOp, dep:tuple[UOp, ...], rid:int) -> UOp:
    dst = A_u32.after(*dep)
    ai = UOp.range(a_iters, rid, AxisType.UPCAST)
    bid = tid + ai * threads
    am, ak = bid // 4, bid % 4
    c0, c1 = (ak * 2) ^ (am & 7), (ak * 2 + 1) ^ (am & 7)
    s0, s1 = ((0, 1, 4, 5), (2, 3, 6, 7)) if rdna4 else ((0, 1, 2, 3), (4, 5, 6, 7))
    sts = [dst[am, c0, j].store(areg[ai, s].load()) for j, s in enumerate(s0)] + [dst[am, c1, j].store(areg[ai, s].load()) for j, s in enumerate(s1)]
    return UOp.group(*sts).end(ai)

  def load_b_reg(kt:UOp, dep:tuple[UOp, ...], rid:int) -> tuple[UOp, ...]:
    bi = UOp.range(b_iters, rid)
    bp = tid + bi * threads
    bn_i, bg = bp // 2, bp % 2
    gn, gbg = block_n * bn + bn_i, kt * 2 + bg
    if ggml_type == Q8_0:
      wbuf = UOp.alloc((b_iters, 9), dtypes.uint32, addrspace=AddrSpace.REG)
      w_off = ((gn * (in_features // 32) + gbg) * 34) // 4
      v0, v1, w8 = _amd_load(raw_u32[w_off], 4), _amd_load(raw_u32[w_off + 4], 4), raw_u32[w_off + 8].load()
      sts = ([wbuf.after(*dep)[bi, j].store(v0[j]) for j in range(4)] +
             [wbuf.after(*dep)[bi, 4 + j].store(v1[j]) for j in range(4)] + [wbuf.after(*dep)[bi, 8].store(w8)])
      return (wbuf.after(UOp.group(*sts).end(bi)),)
    if ggml_type == Q4_K:
      qbuf = UOp.alloc((b_iters, 4), dtypes.uint32, addrspace=AddrSpace.REG)
      sbuf = UOp.alloc((b_iters, 2), dtypes.uint32, addrspace=AddrSpace.REG)
      base = (gn * (in_features // 256) + kt // 4) * 36
      hdr, qv = _amd_load(raw_u32[base], 4), _amd_load(raw_u32[base + 4 + (kt % 4) * 8 + bg * 4], 4)
      ds0, dm0, ds1, dm1 = _q4k_pair_scales([hdr[0], hdr[1], hdr[2], hdr[3]], kt % 4)
      p0 = UOp.stack(ds0.cast(dtypes.half), (-dm0).cast(dtypes.half)).bitcast(dtypes.uint32)
      p1 = UOp.stack(ds1.cast(dtypes.half), (-dm1).cast(dtypes.half)).bitcast(dtypes.uint32)
      sts = [qbuf.after(*dep)[bi, j].store(qv[j]) for j in range(4)] + [sbuf.after(*dep)[bi, 0].store(p0), sbuf.after(*dep)[bi, 1].store(p1)]
      g = UOp.group(*sts).end(bi)
      return qbuf.after(g), sbuf.after(g)
    if ggml_type == MXFP4:
      wbuf = UOp.alloc((b_iters, 5), dtypes.uint32, addrspace=AddrSpace.REG)
      w_off = ((gn * (in_features // 32) + gbg) * 17) // 4
      v0, w4 = _amd_load(raw_u32[w_off], 4), raw_u32[w_off + 4].load()
      sts = [wbuf.after(*dep)[bi, j].store(v0[j]) for j in range(4)] + [wbuf.after(*dep)[bi, 4].store(w4)]
      return (wbuf.after(UOp.group(*sts).end(bi)),)
    # IQ4_XS
    qbuf = UOp.alloc((b_iters, 4), dtypes.uint32, addrspace=AddrSpace.REG)
    sbuf = UOp.alloc((b_iters,), dtypes.float32, addrspace=AddrSpace.REG)
    sub = gbg % 8
    base = (gn * (in_features // 256) + gbg // 8) * 34
    hdr, qv = _amd_load(raw_u32[base], 2), _amd_load(raw_u32[base + 2 + sub * 4], 4)
    sc = _iq4_scale_hdr(hdr[0], hdr[1], sub)
    sts = [qbuf.after(*dep)[bi, j].store(qv[j]) for j in range(4)] + [sbuf.after(*dep)[bi].store(sc)]
    g = UOp.group(*sts).end(bi)
    return qbuf.after(g), sbuf.after(g)

  def store_b_lds(bregs:tuple[UOp, ...], kt:UOp, dep:tuple[UOp, ...], rid:int) -> UOp:
    bi = UOp.range(b_iters, rid)
    bp = tid + bi * threads
    bn_i, bg = bp // 2, bp % 2
    gn, gbg = block_n * bn + bn_i, kt * 2 + bg
    if fast_b:
      dst = B_h2.after(*dep)
      sts: list[UOp] = []
      c6400, c8080 = UOp.const(0x64006400, dtypes.uint32), UOp.const(0x80808080, dtypes.uint32)
      sel_lo, sel_hi = UOp.const(0x05010500, dtypes.uint32), UOp.const(0x05030502, dtypes.uint32)
      neg1152_2 = UOp.const(0xe480e480, dtypes.uint32).reshape((1,)).bitcast(dtypes.half)
      neg1024_2 = UOp.const(0xe400e400, dtypes.uint32).reshape((1,)).bitcast(dtypes.half)
      def store_k16(bk_v:int|UOp, p0:tuple[UOp, UOp], p1:tuple[UOp, UOp], p2:tuple[UOp, UOp], p3:tuple[UOp, UOp]):
        c0, c1 = (bk_v * 2) ^ (bn_i & 7), (bk_v * 2 + 1) ^ (bn_i & 7)
        q0, q1 = ((p0[0], p0[1], p2[0], p2[1]), (p1[0], p1[1], p3[0], p3[1])) if rdna4 else (
          (p0[0], p0[1], p1[0], p1[1]), (p2[0], p2[1], p3[0], p3[1]))
        for j, h2 in enumerate(q0): sts.append(dst[bn_i, c0, j].store(h2))
        for j, h2 in enumerate(q1): sts.append(dst[bn_i, c1, j].store(h2))
      def dec4_u8(ux:UOp, sc2:UOp) -> tuple[UOp, UOp]:
        return ((_amd_byte_perm(c6400, ux, sel_lo).reshape((1,)).bitcast(dtypes.half) + neg1152_2) * sc2,
                (_amd_byte_perm(c6400, ux, sel_hi).reshape((1,)).bitcast(dtypes.half) + neg1152_2) * sc2)
      def dec4_q4k(qw:UOp, ds2:UOp, dm2:UOp) -> tuple[UOp, UOp]:
        return ((_amd_byte_perm(c6400, qw, sel_lo).reshape((1,)).bitcast(dtypes.half) + neg1024_2) * ds2 + dm2,
                (_amd_byte_perm(c6400, qw, sel_hi).reshape((1,)).bitcast(dtypes.half) + neg1024_2) * ds2 + dm2)
      if ggml_type == Q8_0:
        ws = [bregs[0][bi, j].load() for j in range(9)]
        odd = (gbg & 1).ne(0)
        sc2 = _amd_byte_perm(ws[0], ws[0], odd.where(UOp.const(0x03020302, dtypes.uint32),
                                                     UOp.const(0x01000100, dtypes.uint32))).reshape((1,)).bitcast(dtypes.half)
        w8s = [odd.where(ws[j + 1], _perm_alignbyte(ws[j + 1], ws[j], 2)) ^ c8080 for j in range(8)]
        store_k16(bg * 2, *(dec4_u8(w8s[j], sc2) for j in range(4)))
        store_k16(bg * 2 + 1, *(dec4_u8(w8s[4 + j], sc2) for j in range(4)))
      elif ggml_type == Q4_K:
        qs = [bregs[0][bi, j].load() for j in range(4)]
        sp0, sp1 = bregs[1][bi, 0].load(), bregs[1][bi, 1].load()
        ds0_2 = _amd_byte_perm(sp0, sp0, UOp.const(0x01000100, dtypes.uint32)).reshape((1,)).bitcast(dtypes.half)
        dm0_2 = _amd_byte_perm(sp0, sp0, UOp.const(0x03020302, dtypes.uint32)).reshape((1,)).bitcast(dtypes.half)
        ds1_2 = _amd_byte_perm(sp1, sp1, UOp.const(0x01000100, dtypes.uint32)).reshape((1,)).bitcast(dtypes.half)
        dm1_2 = _amd_byte_perm(sp1, sp1, UOp.const(0x03020302, dtypes.uint32)).reshape((1,)).bitcast(dtypes.half)
        store_k16(bg, *(dec4_q4k(qs[j] & 0x0f0f0f0f, ds0_2, dm0_2) for j in range(4)))
        store_k16(2 + bg, *(dec4_q4k((qs[j] >> 4) & 0x0f0f0f0f, ds1_2, dm1_2) for j in range(4)))
      elif ggml_type == MXFP4:
        ws = [bregs[0][bi, j].load() for j in range(5)]
        rem = ((gn * (in_features // 32) + gbg) & 3).cast(dtypes.uint32)
        e, is_al, sh_b = (ws[0] >> (rem * 8)) & 255, rem.eq(3), (rem + 1) & 3
        ps = [is_al.where(ws[j + 1], _perm_alignbyte(ws[j + 1], ws[j], sh_b)) for j in range(4)]
        small = e < 116
        c_lo = small.where(UOp.const(0x26242000, dtypes.uint32), UOp.const(0x36343000, dtypes.uint32))
        c_hi = small.where(UOp.const(0x2e2c2a28, dtypes.uint32), UOp.const(0x3e3c3a38, dtypes.uint32))
        sc2 = _splat_h2(_e8m0_halved(e + small.where(UOp.const(7, dtypes.uint32), UOp.const(3, dtypes.uint32))).cast(dtypes.half))
        store_k16(bg * 2, *(_e2m1_h2_pair(ps[j], c_lo, c_hi, sc2) for j in range(4)))
        store_k16(bg * 2 + 1, *(_e2m1_h2_pair(ps[j] >> 4, c_lo, c_hi, sc2) for j in range(4)))
      else:  # IQ4_XS
        sc_f, ps = bregs[1][bi].load(), [bregs[0][bi, j].load() for j in range(4)]
        def dec4_iq4(ux:UOp) -> tuple[UOp, UOp]:
          q01 = _amd_byte_perm(c6400, ux, sel_lo).reshape((1,)).bitcast(dtypes.half) + neg1152_2
          q23 = _amd_byte_perm(c6400, ux, sel_hi).reshape((1,)).bitcast(dtypes.half) + neg1152_2
          return (UOp.stack((q01[0].float() * sc_f).cast(dtypes.half), (q01[1].float() * sc_f).cast(dtypes.half)),
                  UOp.stack((q23[0].float() * sc_f).cast(dtypes.half), (q23[1].float() * sc_f).cast(dtypes.half)))
        store_k16(bg * 2, *(dec4_iq4(_nibble_lut(ps[j], 0, IQ4NL_U8_LUT)) for j in range(4)))
        store_k16(bg * 2 + 1, *(dec4_iq4(_nibble_lut(ps[j], 4, IQ4NL_U8_LUT)) for j in range(4)))
      return UOp.group(*sts).end(bi)
    # Generic path for all remaining GGUF formats (Q6_K, Q5_K, Q3_K, Q2_K, IQ*, NVFP4)
    base, subgroup = gn * output_words + (gbg // 8) * type_words, gbg % 8
    B_dst = B_lds.after(*dep)
    b_sts: list[UOp] = []
    for k16 in range(2):
      k_u = UOp.range(16, rid + 1 + k16, AxisType.UPCAST)
      vals = _make_wmma_dequant(raw, ggml_type, UOp.const(0, dtypes.weakint), 1, k_u, grid)(base, subgroup, UOp.const(k16, dtypes.weakint))
      bc0, bc1 = (bg * 4 + k16 * 2) ^ (bn_i & 7), (bg * 4 + k16 * 2 + 1) ^ (bn_i & 7)
      if isinstance(vals, tuple):
        i0, i1 = ((0, 1, 2, 3, 8, 9, 10, 11), (4, 5, 6, 7, 12, 13, 14, 15)) if rdna4 else (tuple(range(8)), tuple(range(8, 16)))
        for q, idx in enumerate(i0): b_sts.append(B_dst[bn_i, bc0, q].store(vals[idx]))
        for q, idx in enumerate(i1): b_sts.append(B_dst[bn_i, bc1, q].store(vals[idx]))
      else:
        bc = (bg * 4 + k16 * 2 + ((k_u // 4) % 2 if rdna4 else k_u // 8)) ^ (bn_i & 7)
        bq = ((k_u // 8) * 4 + (k_u % 4)) if rdna4 else (k_u % 8)
        b_sts.append(B_dst[bn_i, bc, bq].store(vals).end(k_u))
    return UOp.group(*b_sts).end(bi)

  def compute_wmma(A_r:UOp, B_r:UOp, acc_in:UOp, rid:int) -> UOp:
    kk, tm_r, tn_r = UOp.range(4, rid), UOp.range(tm, rid + 1), UOp.range(tn, rid + 2)
    mr, nr = (wave_m * tm + tm_r) * 16 + col, (wave_n * tn + tn_r) * 16 + col
    if rdna4:
      a_f, b_f = A_r[mr, (kk * 2 + lane_group) ^ (mr & 7)], B_r[nr, (kk * 2 + lane_group) ^ (nr & 7)]
    else:
      a_f = A_r[mr, (kk * 2) ^ (mr & 7)].cat(A_r[mr, (kk * 2 + 1) ^ (mr & 7)])
      b_f = B_r[nr, (kk * 2) ^ (nr & 7)].cat(B_r[nr, (kk * 2 + 1) ^ (nr & 7)])
    return acc[tm_r, tn_r].store(UOp.wmma(a_f, b_f, acc_in.after(kk)[tm_r, tn_r], *WMMA_ARG)).end(tn_r, tm_r, kk)

  ar0 = load_a_reg(k_tile_base, (), 22)
  br0 = load_b_reg(k_tile_base, (), 24) if fast_b else ()
  st0 = UOp.group(store_a_lds(ar0, (), 26), store_b_lds(br0, k_tile_base, (), 28)).barrier()
  kt = UOp.range(k_tiles_me - 1, 10)
  nxt = k_tile_base + kt + 1
  ar_n = load_a_reg(nxt, (st0, kt), 32)
  br_n = load_b_reg(nxt, (st0, kt), 34) if fast_b else ()
  comp = compute_wmma(A_lds.after(st0, kt, ar_n, *br_n), B_lds.after(st0, kt, ar_n, *br_n), acc.after(st0, kt), 100)
  mid = UOp.group(ar_n, *br_n, comp).barrier()
  st_n = UOp.group(store_a_lds(ar_n, (mid,), 46), store_b_lds(br_n, nxt, (mid,), 48)).barrier().end(kt)
  done = compute_wmma(A_lds.after(st0, st_n), B_lds.after(st0, st_n), acc.after(st0, st_n), 140)
  acc_f = acc.after(done)
  tm_o, tn_o, e_o = UOp.range(tm, 70), UOp.range(tn, 71), UOp.range(8, 72, AxisType.UPCAST)
  out_m = block_m * bm + (wave_m * tm + tm_o) * 16 + (lane_group * 8 + e_o if rdna4 else e_o * 2 + lane_group)
  out_n = block_n * bn + (wave_n * tn + tn_o) * 16 + col
  return out[sk_r * tok + out_m, out_n].store(acc_f[tm_o, tn_o, e_o].load().cast(out.dtype)).end(e_o, tn_o, tm_o, wave, lane, sk_r, gid).sink(
    arg=KernelInfo(name=f"linear_{QUANT_NAMES[ggml_type]}_2d_wmma", opts_to_apply=()))
