from __future__ import annotations

import functools
import math
from collections.abc import Callable
from typing import cast

from tinygrad import UOp
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import unwrap
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

from slopkernels.src.amd import LOG2E, WMMA_ARG, _amd_load, _fast_tanh, _unbind, warp_reduce

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
  return (raw[base + offset//size].cast(dtypes.uint32) >> ((offset%size)*8)) & 255
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

def _iq1_scale(raw:UOp, base:UOp, subgroup:UOp) -> tuple[UOp, UOp]:
  qh = raw[base+17+subgroup].cast(dtypes.uint32)
  return (((qh >> 12) & 7)*2+1).float() * _half(raw[base]), (qh >> 15).eq(0).where(-0.125, 0.125)

@functools.cache
def _q8_quantize_kernel(q:UOp, scale:UOp, xsum:UOp, x:UOp, tokens:int, in_features:int) -> UOp:
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

def _decode_linear(out:UOp, out_features:int, group_count:int, group_dot, name:str, token_tile:int=1, post:Callable|None=None,
                   group_dots:Callable|None=None) -> UOp:
  chunks = out.shape[2] if len(out.shape) == 3 else 1
  # One wave per output/chunk; group neighboring rows to amortize workgroup scheduling.
  rows = math.gcd(out_features, 4)
  row = UOp.range(out.shape[0]//token_tile*out_features//rows, 0, AxisType.GLOBAL)
  wave = UOp.range(rows, 3, AxisType.LOCAL)
  token_output = row*rows+wave
  chunk = UOp.range(chunks, 1, AxisType.GLOBAL) if chunks > 1 else UOp.const(0, dtypes.weakint)
  lane = UOp.range(32, 2, AxisType.LOCAL)
  token, output = token_output // out_features * token_tile, token_output % out_features
  steps = (group_count + chunks * 32 - 1) // (chunks * 32)
  if token_tile > 1 and group_dots is not None:
    zero = UOp.const(0, dtypes.float32)
    if steps == 1:
      group = lane + chunk * 32
      raw_vals = group_dots(token, output, group.minimum(group_count - 1))
      vals = [warp_reduce((group < group_count).where(v, zero), full_wave=True) for v in raw_vals]
    else:
      accs = tuple(UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG) for _ in range(token_tile))
      accs = tuple(acc.after(acc[0].store(zero)) for acc in accs)
      s = UOp.range(steps, 4)
      group = lane + (chunk + s * chunks) * 32
      raw_vals = group_dots(token, output, group.minimum(group_count - 1))
      upd = UOp.group(*(accs[t][0].store(accs[t].after(s)[0].load() + (group < group_count).where(raw_vals[t], zero))
                        for t in range(token_tile))).end(s)
      vals = [warp_reduce(accs[t].after(upd)[0].load(), full_wave=True) for t in range(token_tile)]
    if post is not None: vals = [post(token + t, vals[t]) for t in range(token_tile)]
    if len(out.shape) == 3:
      stores = [out[token+t, output, chunk.valid(lane.eq(0))].store(vals[t].cast(out.dtype)) for t in range(token_tile)]
      return UOp.group(*stores).end(row, wave, *([chunk] if chunks > 1 else []), lane).sink(arg=KernelInfo(name=name, opts_to_apply=()))
    stores = [out[token+t, output.valid(lane.eq(0))].store(vals[t].cast(out.dtype)) for t in range(token_tile)]
    return UOp.group(*stores).end(row, wave, lane).sink(arg=KernelInfo(name=name, opts_to_apply=()))
  def total(token:UOp) -> UOp:
    if steps == 1:
      group = lane + chunk * 32
      val = warp_reduce((group < group_count).where(group_dot(token, output, group.minimum(group_count-1)), UOp.const(0, dtypes.float32)),
                        full_wave=True)
    else:
      acc = UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG)
      acc = acc.after(acc[0].store(UOp.const(0, dtypes.float32)))
      s = UOp.range(steps, 4)
      group = lane + (chunk + s * chunks) * 32
      upd = acc[0].store(acc.after(s)[0].load() +
                         (group < group_count).where(group_dot(token, output, group.minimum(group_count-1)), UOp.const(0, dtypes.float32))).end(s)
      val = warp_reduce(acc.after(upd)[0].load(), full_wave=True)
    return post(token, val) if post is not None else val
  if len(out.shape) == 3:
    stores = [out[token+t, output, chunk.valid(lane.eq(0))].store(total(token+t).cast(out.dtype)) for t in range(token_tile)]
    return UOp.group(*stores).end(row, wave, *([chunk] if chunks > 1 else []), lane).sink(arg=KernelInfo(name=name, opts_to_apply=()))
  stores = [out[token+t, output.valid(lane.eq(0))].store(total(token+t).cast(out.dtype)) for t in range(token_tile)]
  return UOp.group(*stores).end(row, wave, lane).sink(arg=KernelInfo(name=name, opts_to_apply=()))

def _iq_even_signs(signs:UOp) -> UOp:
  parity = signs ^ (signs >> 4)
  parity ^= parity >> 2
  parity ^= parity >> 1
  return signs | ((parity & 1) << 7)

def _iq_signed_word(word:UOp, signs:UOp) -> UOp:
  ones = ((signs.cast(dtypes.uint32) & 15) * 0x00204081) & 0x01010101
  # Grid magnitudes are nonzero and <128, so each byte can be negated without a carry into its neighbor.
  return (word ^ (ones * 255)) + ones

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
    weights = (word(offset) >> ((subgroup%4)*2)) & 0x03030303
    if ggml_type == Q3_K: weights |= ((word(i*4) >> subgroup) & 0x01010101) << 2
    return weights
  if ggml_type == NVFP4:
    block64, subblock = subgroup//2, (subgroup%2)*2 + i//4
    packed = word(block64*36 + 4 + subblock*8 + (i%2)*4)
    return _nibble_lut(packed, 4*((i%4)//2), MXFP4_LUT)
  if ggml_type in (IQ4_NL, IQ4_XS, MXFP4):
    packed = _amd_load(raw[base+2+subgroup*4+i%4]) if ggml_type == IQ4_XS else word((1 if ggml_type == MXFP4 else 2)+(i%4)*4)
    return _nibble_lut(packed, 4*(i//4), MXFP4_LUT if ggml_type == MXFP4 else IQ4NL_LUT)
  assert grid is not None
  grid_words = 2 if ggml_type in (IQ2_XXS, IQ2_XS, IQ1_S, IQ2_S) else 1
  if ggml_type in (IQ3_S, IQ3_XXS, IQ2_XXS):
    index = byte(2+subgroup*8+i//grid_words)
    if ggml_type == IQ3_S:
      index |= ((byte(66+subgroup) >> i) & 1) << 8
      signs = byte(74+subgroup*4+i//2)
    else: signs = _iq_even_signs((word(66+subgroup*4 if ggml_type == IQ3_XXS else 6+subgroup*8) >> (7*(i//2))) & 127)
  elif ggml_type == IQ2_XS:
    packed = raw[base+1+subgroup*4+i//2].cast(dtypes.uint32)
    index, signs = packed & 511, _iq_even_signs(packed >> 9)
  elif ggml_type == IQ1_S:
    index = byte(2+subgroup*4+i//2) | (((raw[base+17+subgroup].cast(dtypes.uint32) >> (3*(i//2))) & 7) << 8)
    signs = index.const_like(0)
  else:
    index = byte(2+subgroup*4+i//2) | (((byte(66+subgroup) >> (2*(i//2))) & 3) << 8)
    signs = byte(34+subgroup*4+i//2)
  return _iq_signed_word(grid[index*grid_words+i%grid_words].load(), signs >> (4*(i%2)))

@functools.cache
def _quant_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, *grids:UOp, out_features:int, in_features:int, ggml_type:int,
                         cdna:bool=False, group_rows:int=0) -> UOp:
  tokens = cast(int, out.shape[0])
  token_tile = tokens if tokens <= 12 else (4 if tokens <= 16 and tokens % 4 == 0 else 1)
  words = out_features * in_features * QUANT_SIZES[ggml_type] // GGML_BLOCK_SIZE // raw.dtype.itemsize
  gate = (lambda token, base: base + token // group_rows * words) if group_rows else None
  return _quant_decode_body(out, raw, xq, xd, xs, grids, out_features, in_features, ggml_type, cdna, gate, token_tile=token_tile)

def _situ_expr(g:UOp, u:UOp, situ:tuple[float, float]) -> UOp:
  b0, b1 = situ
  return _fast_tanh(g * (1.0 / b0)) * b0 * (1.0 / (1.0 + (g * (-LOG2E)).exp2())) * _fast_tanh(u * (1.0 / b1)) * b1

def _quant_base(output:UOp, group:UOp, in_features:int, ggml_type:int, itemsize:int) -> UOp:
  if ggml_type == NVFP4: return (output*in_features//64 + (group//8)*4) * (36//itemsize)
  if ggml_type in SMALL_BLOCK_QUANTS: return (output*in_features//32 + group) * SMALL_BLOCK_QUANTS[ggml_type]
  return (output*in_features//256 + group//8) * (QUANT_SIZES[ggml_type]//itemsize)

def _stage_iq_grid(grid:UOp|None, ggml_type:int, wave:UOp, lane:UOp, rows:int, active:UOp|None=None) -> UOp|None:
  if grid is None or ggml_type not in (IQ3_XXS, IQ2_S): return grid
  n, threads = int(grid.numel()), rows * 32
  if n % threads: return grid
  lds = UOp.alloc((n,), dtypes.uint32, addrspace=AddrSpace.LOCAL)
  tid = wave * 32 + lane
  if active is not None:
    li = UOp.range(active.where(UOp.const(n // threads, dtypes.weakint), UOp.const(0, dtypes.weakint)), 90)
    return lds.after(lds[tid + li * threads].store(grid[tid + li * threads].load()).end(li).barrier())
  return lds.after(UOp.group(*(lds[tid + i * threads].store(grid[tid + i * threads].load()) for i in range(n // threads))).barrier())

def _fma(a:UOp, b:UOp, c:UOp) -> UOp:
  args = tuple(x.load() if x.op is Ops.INDEX else x for x in (a, b, c))
  return UOp.native("__builtin_fmaf({0},{1},{2})", shape=args[0].shape, dtype=dtypes.float32, fmt=True).call(*args)

def _nvfp4_b64_dot(raw:UOp, base64:UOp, xw0:UOp, xw1:UOp, xd0:UOp, xd1:UOp, cdna:bool) -> UOp:
  sc_w = raw[base64].load()
  pv0 = _amd_load(raw[base64 + 1], 4, stream=True)
  pv1 = _amd_load(raw[base64 + 5], 4, stream=True)
  d0, d1, d2, d3 = (UOp.const(0, dtypes.int32) for _ in range(4))
  for i in range(4):
    d0 = _amd_dp4a(_nibble_lut(pv0[i % 2], 4 * (i // 2), MXFP4_LUT), xw0[i], d0, cdna)
    d1 = _amd_dp4a(_nibble_lut(pv0[2 + i % 2], 4 * (i // 2), MXFP4_LUT), xw0[4 + i], d1, cdna)
    d2 = _amd_dp4a(_nibble_lut(pv1[i % 2], 4 * (i // 2), MXFP4_LUT), xw1[i], d2, cdna)
    d3 = _amd_dp4a(_nibble_lut(pv1[2 + i % 2], 4 * (i // 2), MXFP4_LUT), xw1[4 + i], d3, cdna)
  s0, s1 = _ue4m3_halved(sc_w & 255), _ue4m3_halved((sc_w >> 8) & 255)
  s2, s3 = _ue4m3_halved((sc_w >> 16) & 255), _ue4m3_halved((sc_w >> 24) & 255)
  return _fma(_fma(d0.float(), s0, d1.float() * s1), xd0, _fma(d2.float(), s2, d3.float() * s3) * xd1)

def _quant_group_dot(raw:UOp, base:UOp, subgroup:UOp, xwords:UOp, xd_val:UOp, xs_fn:Callable[[int], UOp],
                     ggml_type:int, cdna:bool, grid:UOp|None) -> UOp:
  if ggml_type == Q8_0 and raw.dtype == dtypes.uint16:
    w0 = _amd_load(raw[base + 1], 8, stream=True).bitcast(dtypes.uint32)
    w1 = _amd_load(raw[base + 9], 8, stream=True).bitcast(dtypes.uint32)
    acc_i = UOp.const(0, dtypes.int32)
    for i in range(4): acc_i = _amd_dp4a(w0[i], xwords[i], acc_i, cdna)
    for i in range(4): acc_i = _amd_dp4a(w1[i], xwords[4 + i], acc_i, cdna)
    return (acc_i.float() * _half(raw[base])) * xd_val
  if ggml_type == IQ4_NL and raw.dtype == dtypes.uint16:
    pv = _amd_load(raw[base + 1], 8, stream=True).bitcast(dtypes.uint32)
    acc_i = UOp.const(0, dtypes.int32)
    for i in range(4): acc_i = _amd_dp4a(_nibble_lut(pv[i], 0, IQ4NL_LUT), xwords[i], acc_i, cdna)
    for i in range(4): acc_i = _amd_dp4a(_nibble_lut(pv[i], 4, IQ4NL_LUT), xwords[4 + i], acc_i, cdna)
    return (acc_i.float() * _half(raw[base])) * xd_val
  if ggml_type in (Q4_K, Q5_K) and raw.dtype == dtypes.uint32:
    qv = _amd_load(raw[base + (4 if ggml_type == Q4_K else 12) + (subgroup // 2) * 8], 8, stream=True)
    hv = _amd_load(raw[base + 4], 8, stream=True) if ggml_type == Q5_K else None
    q5_sh = ((subgroup & 1) * 4).cast(dtypes.uint32)
    acc_i = UOp.const(0, dtypes.int32)
    for i in range(8):
      w = (qv[i] >> q5_sh) & 0x0f0f0f0f
      if hv is not None: w |= ((hv[i] >> subgroup.cast(dtypes.uint32)) & 0x01010101) << 4
      acc_i = _amd_dp4a(w, xwords[i], acc_i, cdna)
    d, dmin, scale, minimum = _q5_scales(raw, base, subgroup)
    return (acc_i.float() * (d * scale) - (xs_fn(0) + xs_fn(1)) * (dmin * minimum)) * xd_val
  if ggml_type == NVFP4 and raw.dtype == dtypes.uint32:
    b64, h64 = subgroup // 2, subgroup % 2
    sc_w = raw[base + b64 * 9].load()
    pv = _amd_load(raw[base + b64 * 9 + 1 + h64 * 4], 4, stream=True)
    d0, d1 = UOp.const(0, dtypes.int32), UOp.const(0, dtypes.int32)
    for i in range(4): d0 = _amd_dp4a(_nibble_lut(pv[i % 2], 4 * (i // 2), MXFP4_LUT), xwords[i], d0, cdna)
    for i in range(4): d1 = _amd_dp4a(_nibble_lut(pv[2 + i % 2], 4 * (i // 2), MXFP4_LUT), xwords[4 + i], d1, cdna)
    ns0 = _ue4m3_halved((sc_w >> (h64 * 16).cast(dtypes.uint32)) & 255)
    ns1 = _ue4m3_halved((sc_w >> (h64 * 16 + 8).cast(dtypes.uint32)) & 255)
    return (d0.float() * ns0 + d1.float() * ns1) * xd_val
  if ggml_type == IQ3_XXS and grid is not None:
    if raw.dtype == dtypes.uint16:
      sc_w = _amd_load(raw[base + 33 + subgroup * 2], 2, stream=True).bitcast(dtypes.uint32)[0]
      qv = _amd_load(raw[base + 1 + subgroup * 4], 4, stream=True).bitcast(dtypes.uint32)
      q0, q1 = qv[0], qv[1]
    else:
      sc_w = _load_u32(raw, base, 66 + subgroup * 4)
      q0, q1 = _load_u32(raw, base, 2 + subgroup * 8), _load_u32(raw, base, 6 + subgroup * 8)
    acc_i = UOp.const(0, dtypes.int32)
    for p in range(4):
      sgn = _iq_even_signs((sc_w >> (7 * p)) & 127)
      qw = q0 if p < 2 else q1
      sh = 16 * (p & 1)
      acc_i = _amd_dp4a(_iq_signed_word(grid[(qw >> sh) & 255].load(), sgn), xwords[2 * p], acc_i, cdna)
      acc_i = _amd_dp4a(_iq_signed_word(grid[(qw >> (sh + 8)) & 255].load(), sgn >> 4), xwords[2 * p + 1], acc_i, cdna)
    return acc_i.float() * (_half(raw[base]) * ((sc_w >> 28).float() + 0.5) * 0.5) * xd_val
  if ggml_type == IQ2_S and grid is not None:
    if raw.dtype == dtypes.uint16:
      q0 = _amd_load(raw[base + 1 + subgroup * 2], 2, stream=True).bitcast(dtypes.uint32)[0]
      s0 = _amd_load(raw[base + 17 + subgroup * 2], 2, stream=True).bitcast(dtypes.uint32)[0]
    else:
      q0, s0 = _load_u32(raw, base, 2 + subgroup * 4), _load_u32(raw, base, 34 + subgroup * 4)
    qh, sc_b = _load_byte(raw, base, 66 + subgroup), _load_byte(raw, base, 74 + subgroup)
    d0, d1 = UOp.const(0, dtypes.int32), UOp.const(0, dtypes.int32)
    for p in range(4):
      idx = ((q0 >> (8 * p)) & 255) | (((qh >> (2 * p)) & 3) << 8)
      sgn = (s0 >> (8 * p)) & 255
      w0 = _iq_signed_word(grid[idx * 2].load(), sgn)
      w1 = _iq_signed_word(grid[idx * 2 + 1].load(), sgn >> 4)
      if p < 2: d0 = _amd_dp4a(w1, xwords[2 * p + 1], _amd_dp4a(w0, xwords[2 * p], d0, cdna), cdna)
      else: d1 = _amd_dp4a(w1, xwords[2 * p + 1], _amd_dp4a(w0, xwords[2 * p], d1, cdna), cdna)
    d = _half(raw[base]) * 0.25
    sf0, sf1 = ((sc_b & 15).float() + 0.5) * d, (((sc_b >> 4) & 15).float() + 0.5) * d
    return _fma(d0.float(), sf0, d1.float() * sf1) * xd_val
  def byte(offset): return _load_byte(raw, base, offset)
  def word(offset): return _load_u32(raw, base, offset)
  # One accumulator per scale group: 32 weights for Q4/Q5/IQ4_XS, two groups of 16 otherwise.
  dots = [UOp.const(0, dtypes.int32)] * (1 if ggml_type in (Q4_K, Q5_K, IQ4_XS) else 2)
  for i in range(8):
    weights = _quant_word(raw, base, subgroup, i, ggml_type, grid)
    acc = i//(8//len(dots))
    dots[acc] = _amd_dp4a(weights, xwords[i], dots[acc], cdna)
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
    total = UOp.const(0, dtypes.float32)
    for half in range(2):
      j = subgroup*2+half
      if ggml_type == Q2_K:
        scale = byte(j)
        total += dots[half].float() * (scale & 15).float() * _half(raw[base+40]) - \
                 xs_fn(half) * (scale >> 4).float() * _half(raw[base+41])
      else:
        scale = ((byte(96+j%8) >> ((j//8)*4)) & 15) | (((byte(104+j%4) >> ((j//4)*2)) & 3) << 4)
        total += (dots[half].float() - 4*xs_fn(half)) * (scale.cast(dtypes.int32)-32).float() * _half(raw[base+54])
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
    if ggml_type == IQ3_S: total *= (1+2*((byte(106+subgroup//2) >> ((subgroup%2)*4)) & 15)).float()
    if ggml_type == IQ3_XXS: total *= ((word(66+subgroup*4) >> 28).float()+0.5)*0.5
    if ggml_type == IQ2_XXS: total *= ((word(6+subgroup*8) >> 28).float()+0.5)*0.25
  return total * xd_val

def _quant_group_dots_multi(raw:UOp, base:UOp, subgroup:UOp, xwords_list:tuple[UOp, ...], xd_list:tuple[UOp, ...],
                            xs_fns:tuple[Callable[[int], UOp], ...], ggml_type:int, cdna:bool, grid:UOp|None) -> tuple[UOp, ...]:
  if ggml_type == Q8_0 and raw.dtype == dtypes.uint16:
    n_tok = len(xwords_list)
    w0 = _amd_load(raw[base + 1], 8, stream=True).bitcast(dtypes.uint32)
    w1 = _amd_load(raw[base + 9], 8, stream=True).bitcast(dtypes.uint32)
    acc_l = [UOp.const(0, dtypes.int32)] * n_tok
    for i in range(4):
      wi0, wi1 = w0[i], w1[i]
      for t in range(n_tok): acc_l[t] = _amd_dp4a(wi1, xwords_list[t][4 + i], _amd_dp4a(wi0, xwords_list[t][i], acc_l[t], cdna), cdna)
    sc = _half(raw[base])
    return tuple((acc_l[t].float() * sc) * xd_list[t] for t in range(n_tok))
  if ggml_type == IQ4_NL and raw.dtype == dtypes.uint16:
    n_tok = len(xwords_list)
    pv = _amd_load(raw[base + 1], 8, stream=True).bitcast(dtypes.uint32)
    acc_l = [UOp.const(0, dtypes.int32)] * n_tok
    for i in range(4):
      w0, w1 = _nibble_lut(pv[i], 0, IQ4NL_LUT), _nibble_lut(pv[i], 4, IQ4NL_LUT)
      for t in range(n_tok): acc_l[t] = _amd_dp4a(w1, xwords_list[t][4 + i], _amd_dp4a(w0, xwords_list[t][i], acc_l[t], cdna), cdna)
    sc = _half(raw[base])
    return tuple((acc_l[t].float() * sc) * xd_list[t] for t in range(n_tok))
  if ggml_type in (Q4_K, Q5_K) and raw.dtype == dtypes.uint32:
    n_tok = len(xwords_list)
    qv = _amd_load(raw[base + (4 if ggml_type == Q4_K else 12) + (subgroup // 2) * 8], 8, stream=True)
    hv = _amd_load(raw[base + 4], 8, stream=True) if ggml_type == Q5_K else None
    q5_sh = ((subgroup & 1) * 4).cast(dtypes.uint32)
    acc_l = [UOp.const(0, dtypes.int32)] * n_tok
    for i in range(8):
      w = (qv[i] >> q5_sh) & 0x0f0f0f0f
      if hv is not None: w |= ((hv[i] >> subgroup.cast(dtypes.uint32)) & 0x01010101) << 4
      for t in range(n_tok): acc_l[t] = _amd_dp4a(w, xwords_list[t][i], acc_l[t], cdna)
    d, dmin, scale, minimum = _q5_scales(raw, base, subgroup)
    ds, dm = d * scale, dmin * minimum
    return tuple((acc_l[t].float() * ds - (xs_fns[t](0) + xs_fns[t](1)) * dm) * xd_list[t] for t in range(n_tok))
  if ggml_type == NVFP4 and raw.dtype == dtypes.uint32:
    n_tok = len(xwords_list)
    b64, h64 = subgroup // 2, subgroup % 2
    sc_w = raw[base + b64 * 9].load()
    pv = _amd_load(raw[base + b64 * 9 + 1 + h64 * 4], 4, stream=True)
    d0, d1 = [UOp.const(0, dtypes.int32)] * n_tok, [UOp.const(0, dtypes.int32)] * n_tok
    for i in range(4):
      w = _nibble_lut(pv[i % 2], 4 * (i // 2), MXFP4_LUT)
      for t in range(n_tok): d0[t] = _amd_dp4a(w, xwords_list[t][i], d0[t], cdna)
    for i in range(4):
      w = _nibble_lut(pv[2 + i % 2], 4 * (i // 2), MXFP4_LUT)
      for t in range(n_tok): d1[t] = _amd_dp4a(w, xwords_list[t][4 + i], d1[t], cdna)
    ns0 = _ue4m3_halved((sc_w >> (h64 * 16).cast(dtypes.uint32)) & 255)
    ns1 = _ue4m3_halved((sc_w >> (h64 * 16 + 8).cast(dtypes.uint32)) & 255)
    return tuple((d0[t].float() * ns0 + d1[t].float() * ns1) * xd_list[t] for t in range(n_tok))
  if ggml_type == IQ3_XXS and grid is not None:
    n_tok = len(xwords_list)
    if raw.dtype == dtypes.uint16:
      sc_w = _amd_load(raw[base + 33 + subgroup * 2], 2, stream=True).bitcast(dtypes.uint32)[0]
      qv = _amd_load(raw[base + 1 + subgroup * 4], 4, stream=True).bitcast(dtypes.uint32)
      q0, q1 = qv[0], qv[1]
    else:
      sc_w = _load_u32(raw, base, 66 + subgroup * 4)
      q0, q1 = _load_u32(raw, base, 2 + subgroup * 8), _load_u32(raw, base, 6 + subgroup * 8)
    acc_l = [UOp.const(0, dtypes.int32)] * n_tok
    for p in range(4):
      sgn = _iq_even_signs((sc_w >> (7 * p)) & 127)
      qw, sh = (q0 if p < 2 else q1), 16 * (p & 1)
      w0 = _iq_signed_word(grid[(qw >> sh) & 255].load(), sgn)
      w1 = _iq_signed_word(grid[(qw >> (sh + 8)) & 255].load(), sgn >> 4)
      for t in range(n_tok): acc_l[t] = _amd_dp4a(w1, xwords_list[t][2 * p + 1], _amd_dp4a(w0, xwords_list[t][2 * p], acc_l[t], cdna), cdna)
    sc = _half(raw[base]) * ((sc_w >> 28).float() + 0.5) * 0.5
    return tuple(acc_l[t].float() * sc * xd_list[t] for t in range(n_tok))
  if ggml_type == IQ2_S and grid is not None:
    n_tok = len(xwords_list)
    if raw.dtype == dtypes.uint16:
      iq_q0 = _amd_load(raw[base + 1 + subgroup * 2], 2, stream=True).bitcast(dtypes.uint32)[0]
      iq_s0 = _amd_load(raw[base + 17 + subgroup * 2], 2, stream=True).bitcast(dtypes.uint32)[0]
    else:
      iq_q0, iq_s0 = _load_u32(raw, base, 2 + subgroup * 4), _load_u32(raw, base, 34 + subgroup * 4)
    qh, sc_b = _load_byte(raw, base, 66 + subgroup), _load_byte(raw, base, 74 + subgroup)
    d0, d1 = [UOp.const(0, dtypes.int32)] * n_tok, [UOp.const(0, dtypes.int32)] * n_tok
    for p in range(4):
      idx = ((iq_q0 >> (8 * p)) & 255) | (((qh >> (2 * p)) & 3) << 8)
      sgn = (iq_s0 >> (8 * p)) & 255
      w0 = _iq_signed_word(grid[idx * 2].load(), sgn)
      w1 = _iq_signed_word(grid[idx * 2 + 1].load(), sgn >> 4)
      dst = d0 if p < 2 else d1
      for t in range(n_tok): dst[t] = _amd_dp4a(w1, xwords_list[t][2 * p + 1], _amd_dp4a(w0, xwords_list[t][2 * p], dst[t], cdna), cdna)
    d = _half(raw[base]) * 0.25
    sf0, sf1 = ((sc_b & 15).float() + 0.5) * d, (((sc_b >> 4) & 15).float() + 0.5) * d
    return tuple(_fma(d0[t].float(), sf0, d1[t].float() * sf1) * xd_list[t] for t in range(n_tok))
  def byte(offset): return _load_byte(raw, base, offset)
  def word(offset): return _load_u32(raw, base, offset)
  n_tok, n_dots = len(xwords_list), 1 if ggml_type in (Q4_K, Q5_K, IQ4_XS) else 2
  dots = [[UOp.const(0, dtypes.int32)] * n_dots for _ in range(n_tok)]
  for i in range(8):
    weights = _quant_word(raw, base, subgroup, i, ggml_type, grid)
    acc = i // (8 // n_dots)
    for t in range(n_tok): dots[t][acc] = _amd_dp4a(weights, xwords_list[t][i], dots[t][acc], cdna)
  if ggml_type in (Q4_K, Q5_K):
    d, dmin, scale, minimum = _q5_scales(raw, base, subgroup)
    ds, dm = d * scale, dmin * minimum
    return tuple((dots[t][0].float() * ds - (xs_fns[t](0) + xs_fns[t](1)) * dm) * xd_list[t] for t in range(n_tok))
  if ggml_type == IQ4_XS:
    sc = _iq4_scale(raw, base, subgroup)
    return tuple((dots[t][0].float() * sc) * xd_list[t] for t in range(n_tok))
  if ggml_type == Q6_K:
    scales = [(raw[base+96+subgroup] >> (h*8)).cast(dtypes.uint8).bitcast(dtypes.int8).float() for h in range(2)]
    d = _half(raw[base+104])
    return tuple((sum((dots[t][h].float() - 32 * xs_fns[t](h)) * scales[h] for h in range(2))) * xd_list[t] * d for t in range(n_tok))
  if ggml_type in (Q2_K, Q3_K):
    if ggml_type == Q2_K:
      d, dmin = _half(raw[base+40]), _half(raw[base+41])
      scs = [byte(subgroup * 2 + h) for h in range(2)]
      s0, m0 = [d * (s & 15).float() for s in scs], [dmin * (s >> 4).float() for s in scs]
      return tuple(sum((dots[t][h].float() * s0[h] - xs_fns[t](h) * m0[h] for h in range(2)), UOp.const(0, dtypes.float32)) * xd_list[t]
                   for t in range(n_tok))
    d = _half(raw[base+54])
    scs = [(((byte(96 + (j:=subgroup*2+h) % 8) >> ((j // 8) * 4)) & 15) |
            (((byte(104 + j % 4) >> ((j // 4) * 2)) & 3) << 4)).cast(dtypes.int32) - 32 for h in range(2)]
    sf = [s.float() * d for s in scs]
    return tuple(sum(((dots[t][h].float() - 4 * xs_fns[t](h)) * sf[h] for h in range(2)), UOp.const(0, dtypes.float32)) * xd_list[t]
                 for t in range(n_tok))
  if ggml_type in (IQ2_XS, IQ2_S):
    scales = byte((66 if ggml_type == IQ2_XS else 74) + subgroup)
    d = _half(raw[base]) * 0.25
    sf = [((((scales >> (h * 4)) & 15).float() + 0.5) * d) for h in range(2)]
    return tuple(sum(dots[t][h].float() * sf[h] for h in range(2)) * xd_list[t] for t in range(n_tok))
  if ggml_type == IQ1_S:
    scale, zero = _iq1_scale(raw, base, subgroup)
    return tuple((((dots[t][0] + dots[t][1]).float() - (xs_fns[t](0) + xs_fns[t](1)) * zero) * scale) * xd_list[t] for t in range(n_tok))
  if ggml_type == NVFP4:
    offset = (subgroup // 2) * 36 + (subgroup % 2) * 2
    ns0, ns1 = _ue4m3_halved(byte(offset)), _ue4m3_halved(byte(offset + 1))
    return tuple((dots[t][0].float() * ns0 + dots[t][1].float() * ns1) * xd_list[t] for t in range(n_tok))
  sc = _e8m0_halved(raw[base]) if ggml_type == MXFP4 else _half(raw[base])
  if ggml_type == IQ3_S: sc = sc * (1 + 2 * ((byte(106 + subgroup // 2) >> ((subgroup % 2) * 4)) & 15)).float()
  if ggml_type == IQ3_XXS: sc = sc * ((word(66 + subgroup * 4) >> 28).float() + 0.5) * 0.5
  if ggml_type == IQ2_XXS: sc = sc * ((word(6 + subgroup * 8) >> 28).float() + 0.5) * 0.25
  return tuple(((dots[t][0] + dots[t][1]).float() * sc) * xd_list[t] for t in range(n_tok))

def _quant_decode_body(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, grids:tuple[UOp, ...], out_features:int, in_features:int, ggml_type:int,
                       cdna:bool, gate:Callable|None=None, xtoken:Callable|None=None, name:str="linear_", token_tile:int=1,
                       post:Callable|None=None) -> UOp:
  # gate(token, base) -> base: moves the weight rows, xtoken(token) -> the activation row of an output row
  def group_dot(token:UOp, output:UOp, group:UOp) -> UOp:
    base = _quant_base(output, group, in_features, ggml_type, raw.dtype.itemsize)
    if gate is not None: base = gate(token, base)
    if xtoken is not None: token = xtoken(token)
    xwords = _amd_load(xq[token, group, 0], 8)
    return _quant_group_dot(raw, base, group%8, xwords, xd[token, group], lambda h: xs[token, group, h].load(),
                            ggml_type, cdna, grids[0] if grids else None)
  group_dots = None
  if token_tile > 1 and gate is None and xtoken is None:
    def group_dots(token:UOp, output:UOp, group:UOp) -> tuple[UOp, ...]:
      base = _quant_base(output, group, in_features, ggml_type, raw.dtype.itemsize)
      xwords_list = tuple(_amd_load(xq[token + t, group, 0], 8) for t in range(token_tile))
      xd_list = tuple(xd[token + t, group] for t in range(token_tile))
      xs_fns = tuple((lambda h, t=t: xs[token + t, group, h].load()) for t in range(token_tile))
      return _quant_group_dots_multi(raw, base, group % 8, xwords_list, xd_list, xs_fns, ggml_type, cdna, grids[0] if grids else None)
  return _decode_linear(out, out_features, in_features//32, group_dot, name+QUANT_NAMES[ggml_type], token_tile, post=post, group_dots=group_dots)

@functools.cache
def _moe_decode_situ_q8_kernel(out:UOp, q_out:UOp, sc_out:UOp, xs_out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, sel:UOp, *extra:UOp,
                               out_features:int, in_features:int, ggml_type:int, x_share:int, cdna:bool=False,
                               has_scale:bool=False, situ:tuple[float, float]=(1.0, 1.0)) -> UOp:
  # 8 waves x 4 outputs/wave = 32 consecutive SiTU outputs (= 1 Q8 group): stages IQ grid once per 32 outputs,
  # reuses xwords across 8 weight rows, and fuses q8_quantize directly into the epilogue.
  eidx = 0
  exp_scale = extra[eidx] if has_scale else None
  if has_scale: eidx += 1
  grid = extra[eidx] if eidx < len(extra) else None
  H = out_features // 2
  n_g, group_count = H // Q8_GROUP_SIZE, in_features // Q8_GROUP_SIZE
  steps = (group_count + 31) // 32
  n_rows = cast(int, out.shape[0])
  WAVES, RPW = 8, 4
  wg = UOp.range(n_rows * n_g, 0, AxisType.GLOBAL)
  wave = UOp.range(WAVES, 3, AxisType.LOCAL)
  lane = UOp.range(32, -1, AxisType.WARP)
  grid = _stage_iq_grid(grid, ggml_type, wave, lane, WAVES)
  tok_row, g_out = wg // n_g, wg % n_g
  out_base = g_out * Q8_GROUP_SIZE
  exp_id = sel[tok_row].load().maximum(0).cast(dtypes.weakint)
  expert_words = out_features * in_features * QUANT_SIZES[ggml_type] // GGML_BLOCK_SIZE // raw.dtype.itemsize
  exp_base = exp_id * expert_words
  x_row = tok_row // x_share
  zero = UOp.const(0, dtypes.float32)
  if steps == 1:
    g_safe = lane.minimum(group_count - 1)
    in_range, sub = lane < group_count, g_safe % 8
    xwords = _amd_load(xq[x_row, g_safe, 0], 8)
    xd_val, xs_fn = xd[x_row, g_safe], (lambda h: xs[x_row, g_safe, h].load())
    vg_l, vu_l = [], []
    for r in range(RPW):
      o_r = out_base + wave * RPW + r
      bg = exp_base + _quant_base(o_r, g_safe, in_features, ggml_type, raw.dtype.itemsize)
      bu = exp_base + _quant_base(o_r + H, g_safe, in_features, ggml_type, raw.dtype.itemsize)
      vg_l.append(warp_reduce(in_range.where(_quant_group_dot(raw, bg, sub, xwords, xd_val, xs_fn, ggml_type, cdna, grid), zero), full_wave=True))
      vu_l.append(warp_reduce(in_range.where(_quant_group_dot(raw, bu, sub, xwords, xd_val, xs_fn, ggml_type, cdna, grid), zero), full_wave=True))
  else:
    acc_g = tuple(UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG) for _ in range(RPW))
    acc_g = tuple(a.after(a[0].store(zero)) for a in acc_g)
    acc_u = tuple(UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG) for _ in range(RPW))
    acc_u = tuple(a.after(a[0].store(zero)) for a in acc_u)
    s = UOp.range(steps, 4)
    group = lane + s * 32
    g_safe = group.minimum(group_count - 1)
    in_range, sub = group < group_count, g_safe % 8
    xwords = _amd_load(xq[x_row, g_safe, 0], 8)
    xd_val, xs_fn = xd[x_row, g_safe], (lambda h: xs[x_row, g_safe, h].load())
    st: list[UOp] = []
    for r in range(RPW):
      o_r = out_base + wave * RPW + r
      vg = _quant_group_dot(raw, exp_base + _quant_base(o_r, g_safe, in_features, ggml_type, raw.dtype.itemsize),
                            sub, xwords, xd_val, xs_fn, ggml_type, cdna, grid)
      vu = _quant_group_dot(raw, exp_base + _quant_base(o_r + H, g_safe, in_features, ggml_type, raw.dtype.itemsize),
                            sub, xwords, xd_val, xs_fn, ggml_type, cdna, grid)
      st.append(acc_g[r][0].store(acc_g[r].after(s)[0].load() + in_range.where(vg, zero)))
      st.append(acc_u[r][0].store(acc_u[r].after(s)[0].load() + in_range.where(vu, zero)))
    upd = UOp.group(*st).end(s)
    vg_l = [warp_reduce(acc_g[r].after(upd)[0].load(), full_wave=True) for r in range(RPW)]
    vu_l = [warp_reduce(acc_u[r].after(upd)[0].load(), full_wave=True) for r in range(RPW)]
  scale_val = exp_scale[exp_id].load().float() if exp_scale is not None else None
  lds_act = UOp.alloc((Q8_GROUP_SIZE,), dtypes.float32, addrspace=AddrSpace.LOCAL)
  act_st: list[UOp] = []
  out_st: list[UOp] = []
  for r in range(RPW):
    gv, uv = (vg_l[r] * scale_val, vu_l[r] * scale_val) if scale_val is not None else (vg_l[r], vu_l[r])
    fv = _situ_expr(gv, uv, situ)
    act_st.append(lds_act[(wave * RPW + r).valid(lane.eq(0))].store(fv))
    out_st.append(out[tok_row, (out_base + wave * RPW + r).valid(lane.eq(0))].store(fv.cast(out.dtype)))
  lds_act_r = lds_act.after(UOp.group(*act_st).barrier())
  w0 = wave.eq(0)
  qv = lds_act_r[lane].load()
  wmax = warp_reduce(qv.abs(), Ops.MAX, full_wave=True)
  d = UOp.native("({0}/127.0f)", shape=wmax.shape, dtype=dtypes.float, fmt=True).call(wmax).maximum(1e-8)
  quant = UOp.native("__builtin_nearbyintf({0}/{1})", shape=qv.shape, dtype=dtypes.float, fmt=True).call(qv, d).clip(-127, 127).cast(dtypes.int8)
  word = warp_reduce(quant.cast(dtypes.uint8).cast(dtypes.uint32) << ((lane % 4) * 8).cast(dtypes.uint32), Ops.OR, masks=(1, 2))
  q_stores = [q_out[tok_row, g_out, (lane // 4).valid(w0 & (lane % 4).eq(0))].store(word),
              sc_out[tok_row, g_out.valid(w0 & lane.eq(0))].store(d),
              xs_out[tok_row, g_out, (lane // 16).valid(w0 & (lane % 16).eq(0))].store(warp_reduce(quant.float()))]
  return UOp.group(*out_st, *q_stores).end(wg, wave, lane).sink(
    arg=KernelInfo(name=f"moe_{QUANT_NAMES[ggml_type]}_situ_q8", opts_to_apply=()))

@functools.cache
def _moe_decode_down_sum_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, sel:UOp, probs_buf:UOp, *extra:UOp,
                                out_features:int, in_features:int, ggml_type:int, slots:int, cdna:bool=False,
                                has_scale:bool=False, has_add:bool=False) -> UOp:
  # Single-pass decode MoE down-projection + weighted slot reduction (+ optional shared-expert add).
  eidx = 0
  exp_scale = extra[eidx] if has_scale else None
  if has_scale: eidx += 1
  add_buf = extra[eidx] if has_add else None
  if has_add: eidx += 1
  grid = extra[eidx] if eidx < len(extra) else None
  rows = math.gcd(out_features, 4)
  out_blk = UOp.range(out_features // rows, 0, AxisType.GLOBAL)
  wave = UOp.range(rows, 3, AxisType.LOCAL)
  lane = UOp.range(32, -1, AxisType.WARP)
  grid = _stage_iq_grid(grid, ggml_type, wave, lane, rows)
  output = out_blk * rows + wave
  expert_words = out_features * in_features * QUANT_SIZES[ggml_type] // GGML_BLOCK_SIZE // raw.dtype.itemsize
  zero = UOp.const(0, dtypes.float32)
  acc = UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG)
  acc = acc.after(acc[0].store(zero))
  s_slot = UOp.range(slots, 4)
  exp_id = sel[s_slot].load().maximum(0).cast(dtypes.weakint)
  exp_base = exp_id * expert_words
  if ggml_type == NVFP4 and raw.dtype == dtypes.uint32 and in_features % 64 == 0:
    n_b64 = in_features // 64
    steps_b64 = (n_b64 + 31) // 32
    if steps_b64 == 1:
      b64_safe = lane.minimum(n_b64 - 1)
      in_range = lane < n_b64
      g0, g1 = b64_safe * 2, b64_safe * 2 + 1
      dot = _nvfp4_b64_dot(raw, exp_base + (output * n_b64 + b64_safe) * 9,
                           _amd_load(xq[s_slot, g0, 0], 8), _amd_load(xq[s_slot, g1, 0], 8),
                           xd[s_slot, g0], xd[s_slot, g1], cdna)
      slot_v = warp_reduce(in_range.where(dot, zero), full_wave=True)
    else:
      k_acc = UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG)
      k_acc = k_acc.after(k_acc.after(s_slot)[0].store(zero))
      s_k = UOp.range(steps_b64, 5)
      b64 = lane + s_k * 32
      b64_safe = b64.minimum(n_b64 - 1)
      in_range = b64 < n_b64
      g0, g1 = b64_safe * 2, b64_safe * 2 + 1
      dot = _nvfp4_b64_dot(raw, exp_base + (output * n_b64 + b64_safe) * 9,
                           _amd_load(xq[s_slot, g0, 0], 8), _amd_load(xq[s_slot, g1, 0], 8),
                           xd[s_slot, g0], xd[s_slot, g1], cdna)
      k_upd = k_acc[0].store(k_acc.after(s_k)[0].load() + in_range.where(dot, zero)).end(s_k)
      slot_v = warp_reduce(k_acc.after(k_upd)[0].load(), full_wave=True)
  else:
    group_count = in_features // Q8_GROUP_SIZE
    steps = (group_count + 31) // 32
    if steps == 1:
      g_safe = lane.minimum(group_count - 1)
      in_range = lane < group_count
      dot = _quant_group_dot(raw, exp_base + _quant_base(output, g_safe, in_features, ggml_type, raw.dtype.itemsize),
                             g_safe % 8, _amd_load(xq[s_slot, g_safe, 0], 8), xd[s_slot, g_safe],
                             lambda h: xs[s_slot, g_safe, h].load(), ggml_type, cdna, grid)
      slot_v = warp_reduce(in_range.where(dot, zero), full_wave=True)
    else:
      k_acc = UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG)
      k_acc = k_acc.after(k_acc.after(s_slot)[0].store(zero))
      s_k = UOp.range(steps, 5)
      group = lane + s_k * 32
      g_safe = group.minimum(group_count - 1)
      in_range = group < group_count
      dot = _quant_group_dot(raw, exp_base + _quant_base(output, g_safe, in_features, ggml_type, raw.dtype.itemsize),
                             g_safe % 8, _amd_load(xq[s_slot, g_safe, 0], 8), xd[s_slot, g_safe],
                             lambda h: xs[s_slot, g_safe, h].load(), ggml_type, cdna, grid)
      k_upd = k_acc[0].store(k_acc.after(s_k)[0].load() + in_range.where(dot, zero)).end(s_k)
      slot_v = warp_reduce(k_acc.after(k_upd)[0].load(), full_wave=True)
  if exp_scale is not None: slot_v = slot_v * exp_scale[exp_id].load().float()
  # Barrier FMA contraction on slot_v * prob (readfirstlane is 1 cycle on uniform warp-reduced value) so T=1 is bitwise identical to grouped verify.
  slot_v = (slot_v * probs_buf[s_slot].load().float()).bitcast(dtypes.int32)
  slot_v = UOp.native("__builtin_amdgcn_readfirstlane({0})", shape=slot_v.shape, dtype=dtypes.int32, fmt=True).call(slot_v).bitcast(dtypes.float32)
  upd = acc[0].store(acc.after(s_slot)[0].load() + slot_v).end(s_slot)
  total = acc.after(upd)[0].load()
  if add_buf is not None: total = total + add_buf[output].load().float()
  return out[0, output.valid(lane.eq(0))].store(total.cast(out.dtype)).end(out_blk, wave, lane).sink(
    arg=KernelInfo(name=f"moe_{QUANT_NAMES[ggml_type]}_down_sum", opts_to_apply=()))

@functools.cache
def _moe_decode_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, sel:UOp, *extra:UOp,
                       out_features:int, in_features:int, ggml_type:int, x_share:int, cdna:bool=False, ndev:int=1,
                       has_scale:bool=False) -> UOp:
  # out rows are (token, slot): the slot's expert reads its weight rows, x_share slots share an activation row
  exp_scale = extra[0] if has_scale else None
  grids = extra[1:] if has_scale else extra
  expert_words = out_features * in_features * QUANT_SIZES[ggml_type] // 256 // raw.dtype.itemsize
  local_experts = raw.shape[0]//expert_words
  def expert(row:UOp) -> UOp: return sel[row].cast(dtypes.weakint) - UOp.range(ndev, 4, AxisType.DEVICE)*local_experts
  def gate(row:UOp, base:UOp) -> UOp:
    base = base + expert(row).maximum(0).minimum(local_experts-1) * expert_words
    return base.valid((expert(row) >= 0) & (expert(row) < local_experts)) if ndev > 1 else base
  def xtoken(row:UOp) -> UOp: return row // x_share
  post = (lambda row, val: val * exp_scale[sel[row].maximum(0).cast(dtypes.weakint)].load().float()) if exp_scale is not None else None
  return _quant_decode_body(out, raw, xq, xd, xs, grids, out_features, in_features, ggml_type, cdna, gate, xtoken, "moe_", post=post)

def _load_afrag_f16(x:UOp, row:UOp, col0:UOp, ks:tuple[int|UOp, ...], kgroups:int) -> UOp:
  if x.dtype == dtypes.float16:
    if kgroups == 1:
      lo, hi = _amd_load(x[row, col0], 8), _amd_load(x[row, col0 + 8], 8)
      return UOp.stack(*(lo[i] for i in range(8)), *(hi[i] for i in range(8)))
    if kgroups == 2:
      lo, hi = _amd_load(x[row, col0 + ks[0]], 4), _amd_load(x[row, col0 + ks[4]], 4)
      return UOp.stack(*(lo[i] for i in range(4)), *(hi[i] for i in range(4)))
  return UOp.stack(*(x[row, col0 + i].cast(dtypes.float16) for i in ks))

def _iq_half_vals_f32(table:UOp, base_idx:UOp, s4:UOp, sc_f:UOp) -> tuple[UOp, UOp, UOp, UOp]:
  h01, h23 = table[base_idx].load(), table[base_idx + 1].load()
  x01 = ((s4 & 1) << 15) | ((s4 & 2) << 30)
  x23 = ((s4 & 4) << 13) | ((s4 & 8) << 28)
  v01 = (h01 ^ x01).reshape((1,)).bitcast(dtypes.half)
  v23 = (h23 ^ x23).reshape((1,)).bitcast(dtypes.half)
  return ((v01[0].float() * sc_f).cast(dtypes.half), (v01[1].float() * sc_f).cast(dtypes.half),
          (v23[0].float() * sc_f).cast(dtypes.half), (v23[1].float() * sc_f).cast(dtypes.half))

# kgroups: groups of 16 lanes that split k in a 16x16x16 fragment, RDNA3 copies a and b in both halves
def _wmma_words(lane_group:UOp, kgroups:int) -> tuple[UOp|int, ...]:
  return tuple(range(4)) if kgroups == 1 else tuple(lane_group + kgroups*j for j in range(4//kgroups))

def _wmma_layout(out:UOp, out_features:int, token_tile:int, output_tiles:int, lanes:int, valid_tokens:int|UOp|None=None,
                 max_waves:int=2):
  if out_features % (16*output_tiles): output_tiles = 1
  output_waves = 4 if max_waves >= 4 and out_features % (64*output_tiles) == 0 else 2 if out_features % (32*output_tiles) == 0 else 1
  n_tblocks = out.shape[0] // token_tile if valid_tokens is None else (_unbind(valid_tokens) + token_tile - 1) // token_tile
  token_block = UOp.range(n_tblocks, 0, AxisType.GLOBAL)
  output_block = UOp.range(out_features//(16*output_tiles*output_waves), 1, AxisType.GLOBAL)
  # lane is a hardware WARP range (like the flash kernel): the fragment math stays visible without being
  # range-split into nested loops, which would scramble the WMMA fragment layout
  lane, wave = UOp.range(lanes, -1, axis_type=AxisType.WARP), UOp.range(output_waves, 3, axis_type=AxisType.LOCAL)
  col, lane_group, rows = lane % 16, lane // 16, 256 // lanes
  outputs = tuple((output_block*output_waves+wave)*(16*output_tiles) + tile*16 + col for tile in range(output_tiles))
  inputs = tuple(token_block*token_tile + tile*16 + col for tile in range(token_tile//16))
  tokens = tuple(tuple(token_block*token_tile + tile*16 + lane_group*rows + i for i in range(rows)) for tile in range(token_tile//16))
  return output_waves, token_block, output_block, lane, wave, lane_group, outputs, inputs, tokens

def _wmma_stores(out, outputs, tokens, accs, update, lane, wave, output_waves, kgroups,
                 active:UOp|None=None, out_features:int=0, bias:UOp|None=None, scale:UOp|None=None,
                 situ:tuple[float, float]|None=None, probs:UOp|None=None,
                 sorted_rows:UOp|None=None, tile_base:UOp|None=None, n_sub:int=0):
  def store(token, output, value):
    if scale is not None: value = value * scale
    if bias is not None: value = value + bias[output.minimum(out_features - 1) if out_features else output].load().float()
    if probs is not None:
      cond = (active & (token >= 0)) if active is not None else (token >= 0)
      safe_row = token.maximum(0).cast(dtypes.weakint)
      value = value * probs[safe_row.valid(cond)].load().float()
      return out[safe_row.valid(cond), output].store(value.cast(out.dtype))
    if active is not None: return out[token.maximum(0).cast(dtypes.weakint).valid(active & (token >= 0)), output].store(value.cast(out.dtype))
    return out[token, output.valid(output < out_features) if out_features else output].store(value.cast(out.dtype))
  def store_situ(token, g_val, u_val):
    assert situ is not None
    if scale is not None: g_val, u_val = g_val * scale, u_val * scale
    cond = (active & (token >= 0)) if active is not None else (token >= 0)
    return out[token.maximum(0).cast(dtypes.weakint).valid(cond), outputs[0]].store(_situ_expr(g_val, u_val, situ).cast(out.dtype))
  tt = n_sub if sorted_rows is not None else len(tokens)
  lane_group = lane // 16
  # RDNA4 and CDNA (kgroups > 1) own consecutive rows per lane group, matching tokens directly.
  if kgroups > 1:
    rows = int(accs[0][0].shape[0])
    def _tok(ti:int, i:int) -> UOp:
      if sorted_rows is not None:
        assert tile_base is not None and active is not None
        return sorted_rows[(tile_base + ti * 16 + lane_group * rows + i).valid(active)].load()
      return tokens[ti][i]
    if situ is not None:
      return [store_situ(_tok(ti, i), accs[0][ti].after(update)[i].load(), accs[1][ti].after(update)[i].load())
              for ti in range(tt) for i in range(rows)]
    return [store(_tok(ti, i), output, accs[ot][ti].after(update)[i].load()) for ot, output in enumerate(outputs)
            for ti in range(tt) for i in range(rows)]
  # the accumulator fragment halves are exchanged between lane pairs (l, l^16) through LDS (chunked so LDS stays <= 16 KB)
  chunk_t = max(1, min(tt, (8 if output_waves == 4 else 16) // (output_waves * len(accs))))
  lds = UOp.alloc((output_waves, 32, chunk_t * len(accs) * 8), dtypes.float32, addrspace=AddrSpace.LOCAL)
  low = lane_group.eq(0)
  all_stores: list[UOp] = []
  prev_done: UOp|None = None
  for t0 in range(0, tt, chunk_t):
    t1 = min(t0 + chunk_t, tt)
    batch_accs = [accs[ot][t] for ot in range(len(accs)) for t in range(t0, t1)]
    lds_w = lds if prev_done is None else lds.after(prev_done)
    w_stores = [lds_w[wave, lane, a * 8 + i].store(acc.after(update)[i].load()) for a, acc in enumerate(batch_accs) for i in range(8)]
    lds_r = lds_w.after(*w_stores)
    def values(ai:int, _lds_r:UOp=lds_r) -> tuple[UOp, ...]:
      own = tuple(_lds_r[wave, lane, ai * 8 + i].load() for i in range(8))
      peer = tuple(_lds_r[wave, lane ^ 16, ai * 8 + i].load() for i in range(8))
      return tuple(low.where(own[i], peer[i + 4]) if j == 0 else low.where(peer[i], own[i + 4]) for i in range(4) for j in range(2))
    def row_tokens(ti:int) -> tuple[UOp, ...]:
      if sorted_rows is not None:
        assert tile_base is not None and active is not None
        return tuple(sorted_rows[(tile_base + ti * 16 + lane_group * 8 + i).valid(active)].load() for i in range(8))
      return tokens[ti]
    bt = t1 - t0
    if situ is not None:
      b_stores = [store_situ(token, gv, uv) for bi in range(bt)
                  for token, gv, uv in zip(row_tokens(t0 + bi), values(bi), values(bt + bi))]
    else:
      b_stores = [store(token, output, value) for ot, output in enumerate(outputs)
                  for bi in range(bt) for token, value in zip(row_tokens(t0 + bi), values(ot * bt + bi))]
    all_stores.extend(b_stores)
    if t1 < tt: prev_done = UOp.group(*b_stores).barrier()
  return all_stores

def _amd_f16_wmma_lds_kernel(out:UOp, w:UOp, x:UOp, bias:UOp|None, out_features:int, in_features:int, kgroups:int) -> UOp:
  # 128x128x32 software-pipelined LDS-tiled FP16 WMMA GEMM: 4 waves (2x2), 4x4=16 WMMA tiles/wave (32 WMMAs/K-tile).
  # Put m_blk on the fast-varying grid axis (gidx0) so concurrent WGs share the same 128xK weight slice in L2/L3.
  tokens = cast(int, out.shape[0])
  n_blk, m_blk = UOp.range(out_features // 128, 0, AxisType.GLOBAL), UOp.range(tokens // 128, 1, AxisType.GLOBAL)
  wave, lane = UOp.range(4, 3, AxisType.LOCAL), UOp.range(32, -1, AxisType.WARP)
  tid, wave_m, wave_n = wave * 32 + lane, wave // 2, wave % 2
  col, lane_group = lane % 16, lane // 16
  w, x = w.reshape(out_features, in_features), x.reshape(tokens, in_features)
  a_lds = UOp.alloc((128, 36), dtypes.float16, addrspace=AddrSpace.LOCAL)
  b_lds = UOp.alloc((128, 36), dtypes.float16, addrspace=AddrSpace.LOCAL)
  accs = tuple(tuple(UOp.alloc((8,), dtypes.float32, addrspace=AddrSpace.REG) for _ in range(4)) for _ in range(4))
  accs = tuple(tuple(acc.after(acc.store(acc.const_like(0))) for acc in o_accs) for o_accs in accs)
  ks_idx = tuple(word * 4 + b for word in _wmma_words(lane_group, kgroups) for b in range(4))
  ld_row, ld_col = tid // 4, (tid % 4) * 8

  def _load_tile(k_off:int|UOp) -> tuple[list[UOp], list[UOp]]:
    return ([_amd_load(x[m_blk * 128 + it * 32 + ld_row, k_off + ld_col], 8) for it in range(4)],
            [_amd_load(w[n_blk * 128 + it * 32 + ld_row, k_off + ld_col], 8) for it in range(4)])

  def _store_lds(a_buf:UOp, b_buf:UOp, vas:list[UOp], vbs:list[UOp]) -> UOp:
    st: list[UOp] = []
    for it in range(4):
      r = it * 32 + ld_row
      for j in range(8):
        st.append(a_buf[r, ld_col + j].store(vas[it][j]))
        st.append(b_buf[r, ld_col + j].store(vbs[it][j]))
    return UOp.group(*st).barrier()

  def _compute_tile(a_s:UOp, b_s:UOp, dep:tuple[UOp, ...], k_sub_id:int) -> UOp:
    k_sub = UOp.range(2, k_sub_id)
    k0 = k_sub * 16
    afrags = tuple(_load_afrag_f16(a_s, wave_m * 64 + wm * 16 + col, k0, ks_idx, kgroups) for wm in range(4))
    w_st: list[UOp] = []
    for wn in range(4):
      bfrag = _load_afrag_f16(b_s, wave_n * 64 + wn * 16 + col, k0, ks_idx, kgroups)
      for wm in range(4):
        w_st.append(accs[wn][wm].store(UOp.wmma(afrags[wm], bfrag, accs[wn][wm].after(*dep, k_sub), WMMA_ARG[0], 32)))
    return UOp.group(*w_st).end(k_sub)

  # Prologue: stage k_tile = 0 into LDS
  va0, vb0 = _load_tile(0)
  init_staged = _store_lds(a_lds, b_lds, va0, vb0)
  n_ktiles = in_features // 32
  if n_ktiles > 1:
    k_tile = UOp.range(n_ktiles - 1, 4)
    # Software pipeline: prefetch next tile from global memory while computing 32 WMMAs on current LDS tile
    va_n, vb_n = _load_tile((k_tile + 1) * 32)
    comp = _compute_tile(a_lds.after(init_staged, k_tile), b_lds.after(init_staged, k_tile), (k_tile,), 5).barrier()
    refill = _store_lds(a_lds.after(comp), b_lds.after(comp), va_n, vb_n)
    loop_done = UOp.group(comp, refill).end(k_tile)
    update = _compute_tile(a_lds.after(loop_done), b_lds.after(loop_done), (loop_done,), 6)
  else:
    update = _compute_tile(a_lds.after(init_staged), b_lds.after(init_staged), (init_staged,), 5)

  outputs = tuple(n_blk * 128 + wave_n * 64 + wn * 16 + col for wn in range(4))
  token_rows = tuple(tuple(m_blk * 128 + wave_m * 64 + wm * 16 + lane_group * 8 + i for i in range(8)) for wm in range(4))
  if kgroups > 1:
    stores = _wmma_stores(out, outputs, token_rows, accs, update, lane, wave, 4, kgroups, bias=bias)
  else:
    # Exchange one 16x16 tile per wave at a time (4 KB LDS -> total LDS 22 KB <= 32 KB).
    ep_lds = UOp.alloc((4, 32, 8), dtypes.float32, addrspace=AddrSpace.LOCAL)
    low, stores, prev_done = lane_group.eq(0), [], None
    for wn, output in enumerate(outputs):
      for wm in range(4):
        lds_w = ep_lds if prev_done is None else ep_lds.after(prev_done)
        done = UOp.group(*(lds_w[wave, lane, i].store(accs[wn][wm].after(update)[i].load()) for i in range(8))).barrier()
        lds_r = ep_lds.after(done)
        vals = [low.where(lds_r[wave, lane + (i % 2) * 16, i // 2].load(),
                          lds_r[wave, (lane - 16) + (i % 2) * 16, 4 + i // 2].load()) for i in range(8)]
        prev_done = UOp.group(*(out[token_rows[wm][i], output].store(
          (vals[i] + bias[output].load().float() if bias is not None else vals[i]).cast(out.dtype)) for i in range(8))).barrier()
        stores.append(prev_done)
  return UOp.group(*stores).end(n_blk, m_blk, wave, lane).sink(arg=KernelInfo(name="linear_f16_wmma_lds", opts_to_apply=()))

@functools.cache
def _amd_f16_wmma_kernel(out:UOp, w:UOp, x:UOp, *rest:UOp, out_features:int, in_features:int, kgroups:int, lanes:int,
                         valid_tokens:int|UOp|None=None) -> UOp:
  bias: UOp|None = rest[0] if rest else None
  tokens = cast(int, out.shape[0])
  if (valid_tokens is None and lanes == 32 and w.dtype == dtypes.float16 and x.dtype == dtypes.float16
      and tokens % 128 == 0 and out_features % 128 == 0 and in_features % 32 == 0):
    return _amd_f16_wmma_lds_kernel(out, w, x, bias, out_features, in_features, kgroups)
  output_tiles = 2 if out_features >= 128 and out_features % 32 == 0 else 1
  output_waves = 4 if out_features == 64 else 2 if out_features >= 128 and out_features % (32 * output_tiles) == 0 else 1
  token_tile = 64 if tokens % 64 == 0 and out_features >= 64 else 32 if tokens % 32 == 0 and out_features >= 32 else 16
  stride = 16 * output_tiles * output_waves
  out_blocks = (out_features + stride - 1) // stride
  n_tblocks = tokens // token_tile if valid_tokens is None else (_unbind(valid_tokens) + token_tile - 1) // token_tile
  token_block = UOp.range(n_tblocks, 0, AxisType.GLOBAL)
  output_block = UOp.range(out_blocks, 1, AxisType.GLOBAL)
  lane, wave = UOp.range(lanes, -1, axis_type=AxisType.WARP), UOp.range(output_waves, 3, AxisType.LOCAL)
  col, lane_group, rows = lane % 16, lane // 16, 256 // lanes
  outputs = tuple((output_block * output_waves + wave) * (16 * output_tiles) + tile * 16 + col for tile in range(output_tiles))
  input_tokens = tuple(token_block * token_tile + tile * 16 + col for tile in range(token_tile // 16))
  token_rows = tuple(tuple(token_block * token_tile + tile * 16 + lane_group * rows + i for i in range(rows)) for tile in range(token_tile // 16))
  w, x = w.reshape(out_features, in_features), x.reshape(tokens, in_features)
  accs = tuple(tuple(UOp.alloc((rows,), dtypes.float32, addrspace=AddrSpace.REG) for _ in range(token_tile // 16)) for _ in range(output_tiles))
  accs = tuple(tuple(acc.after(acc.store(acc.const_like(0))) for acc in output_accs) for output_accs in accs)
  ks = tuple(word * 4 + b for word in _wmma_words(lane_group, kgroups) for b in range(4))
  k_tile = UOp.range(in_features // 16, 4)
  wmma_accs = [list(output_accs) for output_accs in accs]
  afrags = tuple(_load_afrag_f16(x, input_token, k_tile * 16, ks, kgroups) for input_token in input_tokens)
  o_pad = out_features % stride != 0
  for output_tile, output in enumerate(outputs):
    safe_o = output.minimum(out_features - 1) if o_pad else output
    bfrag = _load_afrag_f16(w, safe_o, k_tile * 16, ks, kgroups)
    for tile, afrag in enumerate(afrags):
      wmma_accs[output_tile][tile] = UOp.wmma(afrag, bfrag, accs[output_tile][tile].after(k_tile), WMMA_ARG[0], lanes)
  update = UOp.group(*(acc.store(value) for output_accs, output_values in zip(accs, wmma_accs)
                       for acc, value in zip(output_accs, output_values))).end(k_tile)
  stores = _wmma_stores(out, outputs, token_rows, accs, update, lane, wave, output_waves, kgroups,
                        out_features=out_features if o_pad else 0, bias=bias)
  return UOp.group(*stores).end(token_block, output_block, lane, wave).sink(arg=KernelInfo(name="linear_f16_wmma", opts_to_apply=()))

def _quant_linear_wmma(out, x, out_features, in_features, type_words, layout, dequant, name, kgroups, lanes, scale:UOp|None=None):
  x = x.reshape(out.shape[0], in_features)
  output_waves, token_block, output_block, lane, wave, lane_group, outputs, input_tokens, tokens = layout
  token_tile, output_tiles = len(tokens)*16, len(outputs)
  # IQ4_NL blocks are 32 wide: a device can hold part of a 256 weight row group
  output_words = in_features * type_words // GGML_BLOCK_SIZE
  accs = tuple(tuple(UOp.alloc((len(tokens[0]),), dtypes.float32, addrspace=AddrSpace.REG)
                     for tile in range(token_tile // 16)) for ot in range(output_tiles))
  accs = tuple(tuple(acc.after(acc.store(acc.const_like(0))) for acc in output_accs) for output_accs in accs)
  ks = tuple(word*4 + b for word in _wmma_words(lane_group, kgroups) for b in range(4))
  group = UOp.range(in_features // Q8_GROUP_SIZE, 4)
  block, subgroup = group // 8, group % 8
  wmma_accs = [list(output_accs) for output_accs in accs]
  for half in range(2):
    afrags = tuple(_load_afrag_f16(x, input_token, group*32 + half*16, ks, kgroups) for input_token in input_tokens)
    for output_tile,output in enumerate(outputs):
      bfrag = UOp.stack(*dequant(output*output_words + block*type_words, subgroup, half))
      for tile,afrag in enumerate(afrags):
        previous = accs[output_tile][tile].after(group) if half == 0 else wmma_accs[output_tile][tile]
        wmma_accs[output_tile][tile] = UOp.wmma(afrag, bfrag, previous, WMMA_ARG[0], lanes)
  update = UOp.group(*(acc.store(value) for output_accs,output_values in zip(accs, wmma_accs)
                       for acc,value in zip(output_accs, output_values))).end(group)
  stores = _wmma_stores(out, outputs, tokens, accs, update, lane, wave, output_waves, kgroups, scale=scale)
  return UOp.group(*stores).end(token_block, output_block, lane, wave).sink(arg=KernelInfo(name=name, opts_to_apply=()))

@functools.cache
def _q5_linear_f16_wmma_kernel(out:UOp, raw:UOp, x:UOp, out_features:int, in_features:int, ggml_type:int, kgroups:int, lanes:int,
                               valid_tokens:int|UOp|None=None) -> UOp:
  token_tile, output_tiles = (64, 1 if out_features <= 1024 else 2) if out.shape[0] % 64 == 0 else \
    (32 if out.shape[0] % 32 == 0 else 16, 2)
  layout = _wmma_layout(out, out_features, token_tile, output_tiles, lanes, valid_tokens=valid_tokens)
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
def _iq4_linear_f16_wmma_kernel(out:UOp, raw:UOp, x:UOp, lut:UOp, out_features:int, in_features:int, kgroups:int, lanes:int,
                                valid_tokens:int|UOp|None=None) -> UOp:
  token_tile = 32 if out_features <= 1024 and out.shape[0] % 32 == 0 else 64 if out.shape[0] % 64 == 0 and \
    out_features <= 6144 else 128 if out.shape[0] % 128 == 0 else \
    32 if out.shape[0] % 32 == 0 else 16
  output_tiles = 1 if out_features <= 1024 else 2 if out_features <= 6144 else 1 if out_features < 8192 else 2
  layout = _wmma_layout(out, out_features, token_tile, output_tiles, lanes, valid_tokens=valid_tokens)
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

def _make_wmma_dequant(raw:UOp, ggml_type:int, word_indices:tuple[UOp|int, ...],
                       table:UOp|None) -> Callable[[UOp, UOp, int], tuple[UOp, ...]]:
  if ggml_type in (Q4_K, Q5_K):
    def dequant_q5(base:UOp, subgroup:UOp, half:int) -> tuple[UOp, ...]:
      d, dmin, scale, minimum = _q5_scales(raw, base, subgroup)
      qs_base = base + (4 if ggml_type == Q4_K else 12) + (subgroup // 2)*8 + half*4
      words = tuple((raw[qs_base+i] >> ((subgroup&1)*4).cast(dtypes.uint32) & 0x0f0f0f0f) |
        (((raw[base+4+half*4+i] >> subgroup.cast(dtypes.uint32) & 0x01010101) << 4) if ggml_type == Q5_K else 0) for i in word_indices)
      return tuple(((word >> (byte*8) & 255).float()*d*scale-dmin*minimum).cast(dtypes.float16) for word in words for byte in range(4))
    return dequant_q5
  if ggml_type == IQ4_XS:
    assert table is not None
    def dequant_iq4(base:UOp, subgroup:UOp, half:int) -> tuple[UOp, ...]:
      scale = _iq4_scale(raw, base, subgroup)
      pairs = tuple(table[((raw[base + 2 + subgroup*4 + word] >> (byte*8)) & 255).cast(dtypes.weakint)]
                    for word in word_indices for byte in range(4))
      return tuple((_half((pair >> (half*16)) & 0xffff)*scale).cast(dtypes.float16) for pair in pairs)
    return dequant_iq4
  def dequant(base:UOp, subgroup:UOp, half:int) -> tuple[UOp, ...]:
    if ggml_type in SMALL_BLOCK_QUANTS: base += subgroup*SMALL_BLOCK_QUANTS[ggml_type]  # eight independent blocks per 256 weights
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
    zero:int|UOp = 0
    minimum = UOp.const(0, dtypes.float)
    if ggml_type == Q2_K:
      sc = byte(subgroup*2+half)
      scale, minimum = _half(raw[base+40])*(sc & 15).float(), _half(raw[base+41])*(sc >> 4).float()
    elif ggml_type == Q3_K:
      j = subgroup*2+half
      sc = ((byte(96+j%8) >> ((j//8)*4)) & 15) | (((byte(104+j%4) >> ((j//4)*2)) & 3) << 4)
      scale, zero = _half(raw[base+54])*(sc.cast(dtypes.int32)-32).float(), 4
    elif ggml_type == Q6_K:
      scale, zero = _half(raw[base+104])*byte(192+subgroup*2+half).cast(dtypes.uint8).bitcast(dtypes.int8).float(), 32
    elif ggml_type == IQ1_S: scale, zero = _iq1_scale(raw, base, subgroup)
    elif ggml_type == NVFP4:
      e = byte((subgroup//2)*36 + (subgroup%2)*2 + half).cast(dtypes.uint32)
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
        sc = byte((66 if ggml_type == IQ2_XS else 74)+subgroup)
        scale *= (((sc >> (half*4)) & 15).float()+0.5)*0.25
      elif ggml_type == IQ3_S: scale *= (1+2*((byte(106+subgroup//2) >> ((subgroup%2)*4)) & 15)).float()
      elif ggml_type == IQ3_XXS:
        assert table is not None
        sc_w = _load_u32(raw, base, 66+subgroup*4)
        sc_f = scale * ((sc_w >> 28).float()+0.5)*32.0
        out_v: list[UOp] = []
        for w_idx in word_indices:
          idx_i = half*4 + w_idx
          sgn8 = _iq_even_signs((sc_w >> (7*(idx_i//2))) & 127)
          out_v.extend(_iq_half_vals_f32(table, byte(2+subgroup*8+idx_i)*2, sgn8 >> (4*(idx_i & 1)), sc_f))
        return tuple(out_v)
      elif ggml_type == IQ2_XXS: scale *= ((_load_u32(raw, base, 6+subgroup*8) >> 28).float()+0.5)*0.25
    if ggml_type == IQ2_S:
      assert table is not None
      sc_f = scale * 64.0
      qh = byte(66+subgroup)
      out_v = []
      for w_idx in word_indices:
        idx_i = half*4 + w_idx
        pair = idx_i // 2
        g_idx = byte(2+subgroup*4+pair) | (((qh >> (2*pair)) & 3) << 8)
        s4 = byte(34+subgroup*4+pair) >> (4*(idx_i & 1))
        out_v.extend(_iq_half_vals_f32(table, g_idx*4 + (idx_i & 1)*2, s4, sc_f))
      return tuple(out_v)
    words = tuple(_quant_word(raw, base, subgroup, half*4+i, ggml_type, table, stream=False) for i in word_indices)
    return tuple((((word >> (b*8)).cast(dtypes.uint8).bitcast(dtypes.int8).float()-zero)*scale-minimum).cast(dtypes.half)
                 for word in words for b in range(4))
  return dequant

@functools.cache
def _quant_linear_f16_wmma_kernel(out:UOp, raw:UOp, x:UOp, *grids:UOp,
                                  out_features:int, in_features:int, ggml_type:int, kgroups:int, lanes:int,
                                  valid_tokens:int|UOp|None=None) -> UOp:
  token_tile = 64 if out.shape[0] % 64 == 0 else 32 if out.shape[0] % 32 == 0 else 16
  layout = _wmma_layout(out, out_features, token_tile, 2, lanes, valid_tokens=valid_tokens)
  output_waves, _, _, lane, wave, lane_group, _, _, _ = layout
  word_indices = _wmma_words(lane_group, kgroups)
  grid = None
  if grids:
    grid = UOp.alloc((int(grids[0].numel()),), dtypes.uint32, addrspace=AddrSpace.LOCAL)
    tid, threads = wave*lanes+lane, output_waves*lanes
    grid = grid.after(*(grid[tid+i*threads].store(grids[0][tid+i*threads]) for i in range(int(grid.numel())//threads)))
  return _quant_linear_wmma(out, x, out_features, in_features, QUANT_SIZES[ggml_type]//raw.dtype.itemsize, layout,
                            _make_wmma_dequant(raw, ggml_type, word_indices, grid),
                            f"linear_{QUANT_NAMES[ggml_type]}_f16_wmma", kgroups, lanes)

@functools.cache
def _quant_decode_multi_proj_kernel(*args:UOp, out_features_tuple:tuple[int, ...], in_features:int, ggml_type:int,
                                    tokens:int, cdna:bool=False) -> UOp:
  # Fuses m quantized linear projections sharing x and ggml_type into one launch with uniform per-workgroup dispatch.
  m = len(out_features_tuple)
  outs, raws = args[:m], args[m:2*m]
  xq, xd, xs = args[2*m], args[2*m+1], args[2*m+2]
  grid = args[2*m+3] if len(args) > 2*m+3 else None
  rows = 4
  assert m > 0 and all(o > 0 for o in out_features_tuple)
  blks = tuple((o + rows - 1) // rows for o in out_features_tuple)
  offs = tuple(sum(blks[:i]) for i in range(m))
  tot_blks = sum(blks)
  group_count = in_features // Q8_GROUP_SIZE
  steps = (group_count + 31) // 32
  out_blk = UOp.range(tot_blks, 0, AxisType.GLOBAL)
  wave = UOp.range(rows, 3, AxisType.LOCAL)
  lane = UOp.range(32, -1, AxisType.WARP)
  grid = _stage_iq_grid(grid, ggml_type, wave, lane, rows)
  zero = UOp.const(0, dtypes.float32)
  accs = tuple(UOp.alloc((1,), dtypes.float32, addrspace=AddrSpace.REG) for _ in range(tokens))
  accs = tuple(a.after(a[0].store(zero)) for a in accs)
  prev_upd: UOp|None = None
  conds: list[UOp] = []
  local_outs: list[UOp] = []
  for k in range(m):
    cond_k = (out_blk >= offs[k]) & (out_blk < offs[k] + blks[k])
    loc_o = (out_blk - offs[k]).maximum(0).minimum(blks[k] - 1) * rows + wave
    active_k = cond_k & (loc_o < out_features_tuple[k])
    loc_o_safe = loc_o.minimum(out_features_tuple[k] - 1)
    conds.append(active_k)
    local_outs.append(loc_o_safe)
    s = UOp.range(active_k.where(UOp.const(steps, dtypes.weakint), UOp.const(0, dtypes.weakint)), 4 + k)
    group = lane if steps == 1 else (lane + s * 32)
    g_safe = group.minimum(group_count - 1)
    in_range, sub = group < group_count, g_safe % 8
    base = _quant_base(loc_o_safe, g_safe, in_features, ggml_type, raws[k].dtype.itemsize)
    if tokens == 1:
      xw = _amd_load(xq[0, g_safe, 0], 8)
      def _xs0(h:int, gs:UOp=g_safe) -> UOp: return xs[0, gs, h].load()
      vs: tuple[UOp, ...] = (_quant_group_dot(raws[k], base, sub, xw, xd[0, g_safe], _xs0, ggml_type, cdna, grid),)
    else:
      xws = tuple(_amd_load(xq[t, g_safe, 0], 8) for t in range(tokens))
      xds = tuple(xd[t, g_safe] for t in range(tokens))
      def _mk_xs(t_i:int, gs:UOp) -> Callable[[int], UOp]: return lambda h: xs[t_i, gs, h].load()
      xss = tuple(_mk_xs(t, g_safe) for t in range(tokens))
      vs = _quant_group_dots_multi(raws[k], base, sub, xws, xds, xss, ggml_type, cdna, grid)
    if steps == 1:
      st = [(accs[t].after(prev_upd, s) if prev_upd is not None else accs[t].after(s))[0].store(in_range.where(vs[t], zero)) for t in range(tokens)]
    else:
      st = [accs[t][0].store((accs[t].after(prev_upd, s) if prev_upd is not None else accs[t].after(s))[0].load() +
                             in_range.where(vs[t], zero)) for t in range(tokens)]
    prev_upd = UOp.group(*st).end(s)
  assert prev_upd is not None
  vals = [warp_reduce(accs[t].after(prev_upd)[0].load(), full_wave=True) for t in range(tokens)]
  stores = [outs[k][t, local_outs[k].valid(conds[k] & lane.eq(0))].store(vals[t].cast(outs[k].dtype))
            for k in range(m) for t in range(tokens)]
  return UOp.group(*stores).end(out_blk, wave, lane).sink(
    arg=KernelInfo(name=f"linear_{QUANT_NAMES[ggml_type]}_multi{m}", opts_to_apply=()))

