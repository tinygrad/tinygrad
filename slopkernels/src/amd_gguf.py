import functools
from collections.abc import Callable

from tinygrad import UOp
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo, Ops

from slopkernels.src.amd import (WMMA_ARG, _amd_byte_perm, _amd_fdot2, _amd_load, _cvt_pk_f16, _ds_bpermute,
                                 _fast_fmax, _lane_xor, _nat, warp_reduce)

GGML_BLOCK_SIZE, Q8_GROUP_SIZE = 256, 32
Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K = 8, 10, 11, 12, 13, 14
IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS, MXFP4, NVFP4 = 16, 17, 18, 19, 20, 21, 22, 23, 39, 40
QUANT_SIZES = {Q8_0: 272, Q2_K: 84, Q3_K: 110, Q4_K: 144, Q5_K: 176, Q6_K: 210, IQ2_XXS: 66, IQ2_XS: 74,
               IQ3_XXS: 98, IQ1_S: 50, IQ4_NL: 144, IQ3_S: 110, IQ2_S: 82, IQ4_XS: 136, MXFP4: 136, NVFP4: 144}
QUANT_NAMES = {Q8_0: "q8_0", Q2_K: "q2_k", Q3_K: "q3_k", Q4_K: "q4_k", Q5_K: "q5_k", Q6_K: "q6",
               IQ2_XXS: "iq2_xxs", IQ2_XS: "iq2_xs", IQ3_XXS: "iq3_xxs", IQ1_S: "iq1_s", IQ4_NL: "iq4_nl",
               IQ3_S: "iq3_s", IQ2_S: "iq2_s", IQ4_XS: "iq4_xs", MXFP4: "mxfp4", NVFP4: "nvfp4"}
SMALL_BLOCK_QUANTS = {Q8_0: 17, IQ4_NL: 9, MXFP4: 17}
HALFWORD_QUANTS = {Q8_0, Q2_K, Q3_K, Q6_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ4_NL, IQ3_S, IQ2_S}
OTF_FP8_QUANTS = {Q8_0, Q6_K, Q5_K, Q4_K, Q3_K, Q2_K, IQ4_XS, IQ3_S, IQ3_XXS, IQ2_S, MXFP4}
_RPW2_QUANTS = {IQ1_S, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_XXS, IQ3_S}
_ZERO_XS_QUANTS = {IQ3_XXS, Q3_K, IQ2_S, IQ2_XS, IQ2_XXS, IQ1_S, MXFP4, NVFP4}
_LDS_GRID_QUANTS = {IQ3_S, IQ3_XXS, IQ2_S, IQ2_XS, IQ2_XXS}
IQ4NL_LUT = (0xbfad9881, 0xf6eaddcf, 0x26190d01, 0x71594535)
IQ4NL_U8_LUT = tuple(x ^ 0x80808080 for x in IQ4NL_LUT)
_IQ4NL_VALS = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)
MXFP4_LUT = (0x03020100, 0x0c080604, 0xfdfeff00, 0xf4f8fafc)
MXFP4_U8_LUT = tuple(x ^ 0x80808080 for x in MXFP4_LUT)

def _sh(x:int|UOp) -> int|UOp: return x.cast(dtypes.uint32) if isinstance(x, UOp) else x
def _v4(v:UOp) -> list[UOp]: return [v[0], v[1], v[2], v[3]]
def _half(v:UOp) -> UOp: return v.cast(dtypes.uint16).bitcast(dtypes.float16).float()
def _fast_rcp(v:UOp) -> UOp: return _nat("__builtin_amdgcn_rcpf({})", dtypes.float32, v)
def _lane_bpermute(src_lane:UOp, v:UOp) -> UOp: return _ds_bpermute(src_lane.cast(dtypes.int32) << 2, v, inline=False)
def _pin_f32(v:UOp, *deps:UOp) -> UOp:
  res = _nat('({{ float _v = ({0}); __asm__ volatile("" : "+v"(_v) :: "memory"); _v; }})', dtypes.float32, v, inline=False)
  return res.after(*deps) if deps else res
def _perm_alignbyte(hi:UOp, lo:UOp, sh:int|UOp) -> UOp:
  return _nat("__builtin_amdgcn_alignbyte({}, {}, {})", dtypes.uint32, hi.cast(dtypes.uint32), lo.cast(dtypes.uint32),
              sh.cast(dtypes.uint32) if isinstance(sh, UOp) else UOp.const(sh, dtypes.uint32))
def _splat_h2(u:UOp) -> UOp: return _amd_byte_perm(u, u, UOp.const(0x01000100, dtypes.uint32)).reshape((1,)).bitcast(dtypes.half)
def _f32_to_h2(v:UOp) -> UOp: return _nat("__builtin_bit_cast(unsigned int, __builtin_amdgcn_cvt_pkrtz({0}, {0}))", dtypes.uint32, v)
def _splat_f_h2(f:UOp) -> UOp: return _splat_h2(f.cast(dtypes.half).bitcast(dtypes.uint16).cast(dtypes.uint32))
def _load_byte(raw:UOp, base:UOp, off:int|UOp) -> UOp:
  w = raw.dtype.itemsize
  return raw[base + off // w].cast(dtypes.uint32) >> _sh((off % w) * 8) & 255
def _load_u32(raw:UOp, base:UOp, off:int|UOp, stream:bool=False) -> UOp:
  if raw.dtype == dtypes.uint32: return _amd_load(raw[base + off // 4]) if stream else raw[base + off // 4]
  if raw.dtype == dtypes.uint16 and isinstance(off, int) and off % 2 == 0:
    return raw[base + off // 2].cast(dtypes.uint32) | (raw[base + off // 2 + 1].cast(dtypes.uint32) << 16)
  return (_load_byte(raw, base, off) | (_load_byte(raw, base, off + 1) << 8) |
          (_load_byte(raw, base, off + 2) << 16) | (_load_byte(raw, base, off + 3) << 24))
def _nibble_lut(packed:UOp, sh:int|UOp, lut:tuple[int|UOp, ...]) -> UOp:
  nib = (packed >> _sh(sh)) & 0x0f0f0f0f
  lo = _amd_byte_perm(UOp.const(lut[1], dtypes.uint32) if isinstance(lut[1], int) else lut[1],
                      UOp.const(lut[0], dtypes.uint32) if isinstance(lut[0], int) else lut[0], nib & 0x07070707)
  hi = _amd_byte_perm(UOp.const(lut[3], dtypes.uint32) if isinstance(lut[3], int) else lut[3],
                      UOp.const(lut[2], dtypes.uint32) if isinstance(lut[2], int) else lut[2], nib & 0x07070707)
  return _amd_byte_perm(hi, lo, UOp.const(0x03020100, dtypes.uint32) | ((nib & 0x08080808) >> 1))
def _e8m0_halved(e:UOp) -> UOp:
  e = e.cast(dtypes.uint32)
  return (e < 2).where((UOp.const(0x00200000, dtypes.uint32) << e).bitcast(dtypes.float), ((e - 1) << 23).bitcast(dtypes.float))
def _ue4m3_halved(e:UOp) -> UOp:
  e = e.cast(dtypes.uint32)
  exponent, mantissa = (e >> 3) & 15, e & 7
  normal = ((exponent + 119) << 23) | (mantissa << 20)
  return e.eq(0x7f).where(0.0, exponent.eq(0).where(mantissa.float() * 2**-10, normal.bitcast(dtypes.float32)))
def _iq_even_signs(s:UOp) -> UOp:
  p = s ^ (s >> 4)
  p = p ^ (p >> 2)
  p = p ^ (p >> 1)
  return s | ((p & 1) << 7)
def _sign_h2(s2:UOp) -> UOp: return ((s2 & 1) << 15) | ((s2 & 2) << 30)
def _iq_signed_word(w:UOp, s:UOp) -> UOp:
  ones = ((s.cast(dtypes.uint32) & 15) * 0x00204081) & 0x01010101
  return (w ^ (ones * 255)) + ones
def _q5_scales(raw:UOp, base:UOp, sub:UOp, stream:bool=True) -> tuple[UOp, UOp, UOp, UOp]:
  if stream: w0, w1, w2, w3 = _v4(_amd_load(raw[base], 4, stream=True))
  else: w0, w1, w2, w3 = raw[base], raw[base + 1], raw[base + 2], raw[base + 3]
  sc_w = (sub < 4).where((w1 >> _sh(sub * 8)) & 63, (((w3 >> _sh((sub - 4) * 8)) & 15) | (((w1 >> _sh((sub - 4) * 8 + 6)) & 3) << 4)))
  mn_w = (sub < 4).where((w2 >> _sh(sub * 8)) & 63, (((w3 >> _sh((sub - 4) * 8 + 4)) & 15) | (((w2 >> _sh((sub - 4) * 8 + 6)) & 3) << 4)))
  return _half(w0 & 0xffff), _half(w0 >> 16), sc_w.float(), mn_w.float()
def _q4k_pair_scales(hdr:list[UOp], sp:UOp) -> tuple[UOp, UOp, UOp, UOp]:
  w0, w1, w2, w3 = hdr
  d, dmin, hi = _half(w0 & 0xffff), _half(w0 >> 16), sp >= 2
  s_lo, m_lo = w1 >> _sh(sp * 16), w2 >> _sh(sp * 16)
  w3s, w1h, w2h = w3 >> _sh((sp - 2) * 16), w1 >> _sh((sp - 2) * 16), w2 >> _sh((sp - 2) * 16)
  sc0 = hi.where((w3s & 15) | (((w1h >> 6) & 3) << 4), s_lo & 63)
  mn0 = hi.where(((w3s >> 4) & 15) | (((w2h >> 6) & 3) << 4), m_lo & 63)
  sc1 = hi.where(((w3s >> 8) & 15) | (((w1h >> 14) & 3) << 4), (s_lo >> 8) & 63)
  mn1 = hi.where(((w3s >> 12) & 15) | (((w2h >> 14) & 3) << 4), (m_lo >> 8) & 63)
  return d * sc0.float(), dmin * mn0.float(), d * sc1.float(), dmin * mn1.float()
def _iq4_scale(raw:UOp, base:UOp, sub:UOp, stream:bool=False) -> UOp:
  h0, h1 = (_amd_load(raw[base], 2, stream=True)[0], _amd_load(raw[base], 2, stream=True)[1]) if stream else (raw[base], raw[base + 1])
  return _half(h0) * ((((h1 >> _sh(4 * sub)) & 15) | (((h0 >> _sh(16 + 2 * sub)) & 3) << 4)).cast(dtypes.int32) - 32).float()
def _iq1_scale(raw:UOp, base:UOp, sub:UOp) -> tuple[UOp, UOp]:
  qh = raw[base + 17 + sub].cast(dtypes.uint32)
  return (((qh >> 12) & 7) * 2 + 1).float() * _half(raw[base]), (qh >> 15).eq(0).where(-0.125, 0.125)
def _quant_base(out_i:UOp, grp:UOp, in_f:int|UOp, qt:int, itemsize:int) -> UOp:
  if qt in SMALL_BLOCK_QUANTS: return (out_i * (in_f // Q8_GROUP_SIZE) + grp) * SMALL_BLOCK_QUANTS[qt]
  return (out_i * (in_f // GGML_BLOCK_SIZE) + grp // 8) * (QUANT_SIZES[qt] // itemsize)

# ******** unified single-kernel direct fdot2 GEMV & MoE (RDNA3 / RDNA4 / CDNA) ********

def _direct_group_dot_fn(raw:UOp, raw_u32:UOp, base:UOp, sub:UOp, qt:int, grid:UOp|None,
                         aligned4:bool=True, stream_w:bool=True, lane:UOp|None=None) -> Callable[[list[UOp], UOp, UOp], UOp]:
  c6400, c8080 = UOp.const(0x64006400, dtypes.uint32), UOp.const(0x80808080, dtypes.uint32)
  sel_lo, sel_hi = UOp.const(0x05010500, dtypes.uint32), UOp.const(0x05030502, dtypes.uint32)
  sel01, sel23 = UOp.const(0x05010400, dtypes.uint32), UOp.const(0x07030602, dtypes.uint32)
  sel01z, sel23z = UOp.const(0x05000400, dtypes.uint32), UOp.const(0x07000600, dtypes.uint32)
  z, z32 = UOp.const(0.0, dtypes.float32), UOp.const(0, dtypes.uint32)
  def ld(p:UOp, lanes:int|None=None) -> UOp:
    return _amd_load(p, lanes, stream=stream_w) if lanes is not None else p.load(arg="nontemporal" if stream_w else None)
  def make_wh2_dot(wh2:list[UOp], comb:Callable[[UOp, UOp, UOp, UOp], UOp]) -> Callable[[list[UOp], UOp, UOp], UOp]:
    def fn(xh:list[UOp], xs0:UOp, xs1:UOp) -> UOp:
      d0, d1 = z, z
      for j in range(8): d0 = _amd_fdot2(wh2[j], xh[j], d0)
      for j in range(8, 16): d1 = _amd_fdot2(wh2[j], xh[j], d1)
      return comb(d0, d1, xs0, xs1)
    return fn
  def make_dot(ws:list[UOp], signed:bool, comb:Callable[[UOp, UOp, UOp, UOp], UOp]) -> Callable[[list[UOp], UOp, UOp], UOp]:
    wh2 = [h for w in ws for ux in ((w ^ c8080) if signed else w,)
           for h in (_amd_byte_perm(c6400, ux, sel_lo), _amd_byte_perm(c6400, ux, sel_hi))]
    return make_wh2_dot(wh2, comb)
  def iq2_wh2(gw:UOp, s4:UOp) -> tuple[UOp, UOp]:
    sel = (gw >> 4) & 0x03030303
    hi_b = _amd_byte_perm(z32, UOp.const(0x00514e48, dtypes.uint32), sel)
    lo_b = _amd_byte_perm(z32, UOp.const(0x00604000, dtypes.uint32), sel)
    return _amd_byte_perm(hi_b, lo_b, sel01) ^ _sign_h2(s4), _amd_byte_perm(hi_b, lo_b, sel23) ^ _sign_h2(s4 >> 2)
  def mxfp4_wh2(pw:UOp) -> tuple[UOp, UOp]:
    hi_b = (_amd_byte_perm(UOp.const(0x4a484644, dtypes.uint32), UOp.const(0x42403c00, dtypes.uint32), pw & 0x07070707)
            | ((pw & 0x08080808) << 4))
    return _amd_byte_perm(hi_b, z32, sel01z), _amd_byte_perm(hi_b, z32, sel23z)
  if qt == IQ3_XXS:
    assert grid is not None
    sc_w = ld(raw[base + 33 + sub * 2], 2).bitcast(dtypes.uint32)[0]
    qv = ld(raw[base + 1 + sub * 4], 4).bitcast(dtypes.uint32)
    sc = _half(ld(raw[base])) * ((sc_w >> 28).float() + 0.5) * 32.0
    wh2 = [g01[m] ^ _sign_h2(s4 >> (2 * m)) for p in range(4) for sgn in (_iq_even_signs((sc_w >> (7 * p)) & 127),)
           for k2 in (0, 1) for g01 in (_amd_load(grid[((qv[p // 2] >> ((p & 1) * 16 + k2 * 8)) & 255) * 2], 2),)
           for s4 in (sgn >> (k2 * 4),) for m in (0, 1)]
    return make_wh2_dot(wh2, lambda d0, d1, x0, x1: (d0 + d1) * sc)
  if qt == Q8_0:
    ws = _v4(ld(raw[base + 1], 8).bitcast(dtypes.uint32)) + _v4(ld(raw[base + 9], 8).bitcast(dtypes.uint32))
    sc = _half(ld(raw[base]))
    return make_dot(ws, True, lambda d0, d1, x0, x1: ((d0 + d1) - UOp.const(1152.0, dtypes.float32) * (x0 + x1)) * sc)
  if qt in (Q4_K, Q5_K):
    qv = ld(raw[base + (4 if qt == Q4_K else 12) + (sub // 2) * 8], 8)
    hv = ld(raw[base + 4], 8) if qt == Q5_K else None
    sh, (d, dmin, sc, mn) = ((sub & 1) * 4).cast(dtypes.uint32), _q5_scales(raw, base, sub, stream=stream_w)
    ds, dm = d * sc, dmin * mn
    ws = [((qv[j] >> sh) & 0x0f0f0f0f) | ((((hv[j] >> sub.cast(dtypes.uint32)) & 0x01010101) << 4) if hv is not None else 0)
          for j in range(8)]
    return make_dot(ws, False, lambda d0, d1, x0, x1: (d0 + d1) * ds - (x0 + x1) * (dm + UOp.const(1024.0, dtypes.float32) * ds))
  if qt == Q6_K:
    ql_b, qh_b = base + (sub // 4) * 32 + (sub % 2) * 16, base + 64 + (sub // 4) * 16
    ql_sh, qh_sh = ((sub % 4 // 2) * 4).cast(dtypes.uint32), ((sub % 4) * 2).cast(dtypes.uint32)
    ql = _v4(ld(raw[ql_b], 8).bitcast(dtypes.uint32)) + _v4(ld(raw[ql_b + 8], 8).bitcast(dtypes.uint32))
    qh = _v4(ld(raw[qh_b], 8).bitcast(dtypes.uint32)) + _v4(ld(raw[qh_b + 8], 8).bitcast(dtypes.uint32))
    sc_w, d_sc = ld(raw[base + 96 + sub]), _half(ld(raw[base + 104]))
    s0 = (sc_w & 255).cast(dtypes.uint8).bitcast(dtypes.int8).float() * d_sc
    s1 = ((sc_w >> 8) & 255).cast(dtypes.uint8).bitcast(dtypes.int8).float() * d_sc
    ws = [((ql[j] >> ql_sh) & 0x0f0f0f0f) | (((qh[j] >> qh_sh) & 0x03030303) << 4) for j in range(8)]
    return make_dot(ws, False, lambda d0, d1, x0, x1: (d0 - UOp.const(1056.0, dtypes.float32) * x0) * s0 +
                                                      (d1 - UOp.const(1056.0, dtypes.float32) * x1) * s1)
  if qt == Q3_K:
    assert lane is not None
    b0, b4 = (lane // 8) * 8, (lane // 4) * 4
    hm_m = ld(raw[base + sub * 2], 2).bitcast(dtypes.uint32)[0]
    hm = [_lane_bpermute(b0 + j, hm_m) for j in range(8)]
    qs_p = ld(raw[base + 16 + (sub // 4) * 16 + (sub % 4) * 4], 4).bitcast(dtypes.uint32)
    qs = [_lane_bpermute(b4 + j // 2, qs_p[j % 2]) for j in range(8)]
    sl16, sh16 = ld(raw[base + 48 + (sub % 4)]).cast(dtypes.uint32), ld(raw[base + 52 + (sub % 2)]).cast(dtypes.uint32)
    qs_sh, hm_sh, d_sc = ((sub % 4) * 2).cast(dtypes.uint32), sub.cast(dtypes.uint32), _half(ld(raw[base + 54]))
    s = [((((sl16 >> (h * 8 + (sub // 4) * 4).cast(dtypes.uint32)) & 15) |
           (((sh16 >> (h * 8 + (sub // 2) * 2).cast(dtypes.uint32)) & 3) << 4)).cast(dtypes.int32) - 32).float() * d_sc for h in (0, 1)]
    c_lo, c_hi = UOp.const(0xbcc0c2c4, dtypes.uint32), UOp.const(0x42403c00, dtypes.uint32)
    wh2 = [h for j in range(8)
           for hi_b in (_amd_byte_perm(c_hi, c_lo, ((qs[j] >> qs_sh) & 0x03030303) | (((hm[j] >> hm_sh) & 0x01010101) << 2)),)
           for h in (_amd_byte_perm(hi_b, z32, sel01z), _amd_byte_perm(hi_b, z32, sel23z))]
    return make_wh2_dot(wh2, lambda d0, d1, x0, x1: d0 * s[0] + d1 * s[1])
  if qt == Q2_K:
    qs = _v4(ld(raw[base + 8 + (sub // 4) * 16], 8).bitcast(dtypes.uint32)) + _v4(ld(raw[base + 16 + (sub // 4) * 16], 8).bitcast(dtypes.uint32))
    sh, dm, sc_w = ((sub % 4) * 2).cast(dtypes.uint32), ld(raw[base + 40], 2), ld(raw[base + sub]).cast(dtypes.uint32)
    dh, dmh, sc0, sc1 = _half(dm[0]), _half(dm[1]), sc_w & 255, (sc_w >> 8) & 255
    ds0, dm0, ds1, dm1 = (sc0 & 15).float() * dh, (sc0 >> 4).float() * dmh, (sc1 & 15).float() * dh, (sc1 >> 4).float() * dmh
    ws = [(qs[j] >> sh) & 0x03030303 for j in range(8)]
    return make_dot(ws, False, lambda d0, d1, x0, x1: d0 * ds0 - x0 * (dm0 + UOp.const(1024.0, dtypes.float32) * ds0) +
                                                      d1 * ds1 - x1 * (dm1 + UOp.const(1024.0, dtypes.float32) * ds1))
  if qt in (IQ4_XS, IQ4_NL):
    pv = ld(raw[base + 2 + sub * 4], 4) if qt == IQ4_XS else ld(raw[base + 1], 8).bitcast(dtypes.uint32)
    sc = _iq4_scale(raw, base, sub, stream=stream_w) if qt == IQ4_XS else _half(ld(raw[base]))
    ws = [_nibble_lut(pv[j], sh, IQ4NL_U8_LUT) for sh in (0, 4) for j in range(4)]
    return make_dot(ws, False, lambda d0, d1, x0, x1: ((d0 + d1) - UOp.const(1152.0, dtypes.float32) * (x0 + x1)) * sc)
  if qt == MXFP4:
    if aligned4:
      w_idx, b_sh = base // 4, (base % 4).cast(dtypes.uint32)
      w03, w4 = ld(raw_u32[w_idx], 4), ld(raw_u32[w_idx + 4])
      rws = [w03[0], w03[1], w03[2], w03[3], w4]
      sc, s1, is3 = _e8m0_halved((rws[0] >> (b_sh * 8)) & 255), (b_sh + 1) & 3, b_sh.eq(3)
      pws = [is3.where(rws[j + 1], _perm_alignbyte(rws[j + 1], rws[j], s1)) for j in range(4)]
    else:
      pv = ld(raw[base + 1], 16).bitcast(dtypes.uint32)
      pws, sc = [pv[j] for j in range(4)], _e8m0_halved(ld(raw[base]))
    wh2 = [h for sh in (0, 4) for j in range(4) for h in mxfp4_wh2((pws[j] >> sh) if sh else pws[j])]
    return make_wh2_dot(wh2, lambda d0, d1, x0, x1: (d0 + d1) * sc)
  if qt == NVFP4:
    b64, h64 = sub // 2, sub % 2
    sc_w, pv = ld(raw[base + b64 * 9]), ld(raw[base + b64 * 9 + 1 + h64 * 4], 4)
    wh2 = [h for h_i in range(2) for i in range(4) for h in mxfp4_wh2((pv[h_i * 2 + i % 2] >> ((i // 2) * 4)) if i >= 2 else pv[h_i * 2 + i % 2])]
    s0, s1 = _ue4m3_halved((sc_w >> _sh(h64 * 16)) & 255), _ue4m3_halved((sc_w >> _sh(h64 * 16 + 8)) & 255)
    return make_wh2_dot(wh2, lambda d0, d1, x0, x1: d0 * s0 + d1 * s1)
  assert grid is not None
  if qt == IQ3_S:
    qh_u = (ld(raw[base + 33 + sub // 2]).cast(dtypes.uint32) >> _sh((sub & 1) * 8)) & 255
    sc = _half(ld(raw[base])) * (1 + 2 * ((ld(raw[base + 53 + sub // 4]).cast(dtypes.uint32) >> _sh((sub & 3) * 4)) & 15)).float()
    qs_v, sg_w = ld(raw[base + 1 + sub * 4], 4).bitcast(dtypes.uint32), ld(raw[base + 37 + sub * 2], 2).bitcast(dtypes.uint32)[0]
    ws = [_iq_signed_word(grid[((qs_v[h] >> (j * 8)) & 255) | (((qh_u >> (h * 4 + j)) & 1) << 8)].load(), (sg_w >> (h * 16 + j * 4)) & 15)
          for h in range(2) for j in range(4)]
    return make_dot(ws, True, lambda d0, d1, x0, x1: ((d0 + d1) - UOp.const(1152.0, dtypes.float32) * (x0 + x1)) * sc)
  if qt == IQ1_S:
    qh_u, qs_w = ld(raw[base + 17 + sub]).cast(dtypes.uint32), ld(raw[base + 1 + sub * 2], 2).bitcast(dtypes.uint32)[0]
    sc, is_p = (((qh_u >> 12) & 7) * 2 + 1).float() * _half(ld(raw[base])), (qh_u >> 15).eq(0)
    hi_t = is_p.where(UOp.const(0xbb003c30, dtypes.uint32), UOp.const(0xbc003bb0, dtypes.uint32))
    lo_t = is_p.where(UOp.const(0x00008000, dtypes.uint32), UOp.const(0x80000000, dtypes.uint32))
    wh2 = [h for p in range(4) for g01 in (_amd_load(grid[(((qs_w >> (p * 8)) & 255) | (((qh_u >> (3 * p)) & 7) << 8)) * 2], 2),)
           for k in (0, 1) for sel in (g01[k] & 0x03030303,)
           for hi_b, lo_b in ((_amd_byte_perm(z32, hi_t, sel), _amd_byte_perm(z32, lo_t, sel)),)
           for h in (_amd_byte_perm(hi_b, lo_b, sel01), _amd_byte_perm(hi_b, lo_b, sel23))]
    return make_wh2_dot(wh2, lambda d0, d1, x0, x1: (d0 + d1) * sc)
  if qt == IQ2_S:
    qh_u = (ld(raw[base + 33 + sub // 2]).cast(dtypes.uint32) >> _sh((sub & 1) * 8)) & 255
    sc_b = (ld(raw[base + 37 + sub // 2]).cast(dtypes.uint32) >> _sh((sub & 1) * 8)) & 255
    q_w, s_w = ld(raw[base + 1 + sub * 2], 2).bitcast(dtypes.uint32)[0], ld(raw[base + 17 + sub * 2], 2).bitcast(dtypes.uint32)[0]
    d_b = _half(ld(raw[base])) * 0.25
    sf0, sf1 = ((sc_b & 15).float() + 0.5) * d_b, (((sc_b >> 4) & 15).float() + 0.5) * d_b
    wh2 = [h for p in range(4) for g01 in (_amd_load(grid[(((q_w >> (p * 8)) & 255) | (((qh_u >> (2 * p)) & 3) << 8)) * 2], 2),)
           for k in (0, 1) for h in iq2_wh2(g01[k], ((s_w >> (p * 8)) & 255) >> (k * 4))]
    return make_wh2_dot(wh2, lambda d0, d1, x0, x1: d0 * sf0 + d1 * sf1)
  if qt == IQ2_XS:
    sc_b, d_b = (ld(raw[base + 33 + sub // 2]).cast(dtypes.uint32) >> _sh((sub & 1) * 8)) & 255, _half(ld(raw[base])) * 0.25
    sf0, sf1 = ((sc_b & 15).float() + 0.5) * d_b, (((sc_b >> 4) & 15).float() + 0.5) * d_b
    wh2 = [h for p in range(4) for pk in (ld(raw[base + 1 + sub * 4 + p]).cast(dtypes.uint32),)
           for g01 in (_amd_load(grid[(pk & 511) * 2], 2),) for k in (0, 1) for h in iq2_wh2(g01[k], _iq_even_signs(pk >> 9) >> (k * 4))]
    return make_wh2_dot(wh2, lambda d0, d1, x0, x1: d0 * sf0 + d1 * sf1)
  qs_w, sg_w = ld(raw[base + 1 + sub * 4], 2).bitcast(dtypes.uint32)[0], ld(raw[base + 3 + sub * 4], 2).bitcast(dtypes.uint32)[0]
  sc = _half(ld(raw[base])) * ((sg_w >> 28).float() + 0.5) * 0.25
  wh2 = [h for p in range(4) for g01 in (_amd_load(grid[((qs_w >> (p * 8)) & 255) * 2], 2),)
         for k in (0, 1) for h in iq2_wh2(g01[k], _iq_even_signs((sg_w >> (7 * p)) & 127) >> (k * 4))]
  return make_wh2_dot(wh2, lambda d0, d1, x0, x1: (d0 + d1) * sc)

def _load_x_32(x:UOp, xt:UOp, grp:UOp, ones_h2:UOp, z:UOp, need_xs:bool, valid:UOp|None=None, rne:bool=False) -> tuple[list[UOp], UOp, UOp]:
  xh, x_row = [], xt.valid(valid) if valid is not None else xt
  for j in range(8):
    xv = _amd_load(x[x_row, grp * 32 + j * 4], 4)
    if x.dtype == dtypes.float16:
      u2 = xv.bitcast(dtypes.uint32)
      xh.extend([u2[0], u2[1]])
    elif rne:
      def _pk_rne(a:UOp, b:UOp) -> UOp:
        return a.cast(dtypes.half).bitcast(dtypes.uint16).cast(dtypes.uint32) | (b.cast(dtypes.half).bitcast(dtypes.uint16).cast(dtypes.uint32) << 16)
      xh.extend([_pk_rne(xv[0], xv[1]), _pk_rne(xv[2], xv[3])])
    else: xh.extend([_cvt_pk_f16(xv[0].float(), xv[1].float()), _cvt_pk_f16(xv[2].float(), xv[3].float())])
  if valid is not None: xh = [valid.where(h, UOp.const(0, dtypes.uint32)) for h in xh]
  if not need_xs: return xh, z, z
  xs0, xs1 = z, z
  for j in range(8): xs0 = _amd_fdot2(ones_h2, xh[j], xs0)
  for j in range(8, 16): xs1 = _amd_fdot2(ones_h2, xh[j], xs1)
  return xh, xs0, xs1

def _direct_gemv_body(out:UOp, raw:UOp, x:UOp, grids:tuple[UOp, ...], out_features:UOp, in_features:UOp, qt:int, x_lds:bool=False,
                      rpw:int|UOp=1, wk_cap:int=16, w1_only:bool=False, aligned4:bool=True, stream_w:bool=True, lds_grid:bool=False,
                      group_rows:UOp|None=None, gate:Callable|None=None, xtoken:Callable|None=None, name:str="linear_",
                      row_mask:Callable|None=None) -> UOp:
  tokens, group_count = out.shape[0], in_features // 32
  assert isinstance(tokens, UOp)
  wg_groups, t_per_wg = ((group_rows > 0).where(tokens // group_rows.maximum(1), 1),
                         (group_rows > 0).where(group_rows, tokens)) if group_rows is not None else (tokens, UOp.const(1, dtypes.weakint))
  if x_lds:
    w_rows = ((t_per_wg >= 8) & (out_features % 16).eq(0)).where(16, (out_features % 8).eq(0).where(8, (out_features % 4).eq(0).where(4, 1)))
    waves_k, lanes_k, opw = UOp.const(1, dtypes.weakint), UOp.const(32, dtypes.weakint), UOp.const(1, dtypes.weakint)
    t_inner, t_outer = t_per_wg, UOp.const(1, dtypes.weakint)
  else:
    if w1_only: waves_k = UOp.const(1, dtypes.weakint)
    else:
      k_waves = (group_count + 31) // 32
      k_steps = (k_waves + wk_cap - 1) // wk_cap
      waves_k = (k_waves + k_steps - 1) // k_steps
    lanes_k = (waves_k.eq(1) & (group_count <= 8) & (out_features % (4 * rpw)).eq(0)).where(
      8, (waves_k.eq(1) & (group_count <= 16) & (out_features % (2 * rpw)).eq(0)).where(16, 32))
    opw, w_rows, t_inner, t_outer = 32 // lanes_k, UOp.const(1, dtypes.weakint), UOp.const(1, dtypes.weakint), t_per_wg
  wg_rows, threads = w_rows * opw * rpw, waves_k * w_rows * 32
  row_blks, steps = out_features // wg_rows, (group_count + waves_k * lanes_k - 1) // (waves_k * lanes_k)
  rb = UOp.range(wg_groups * t_outer * row_blks, 0, AxisType.GLOBAL)
  lane, wave_k, w_row = UOp.range(32, -1, AxisType.WARP), UOp.range(waves_k, 1, AxisType.LOCAL), UOp.range(w_rows, 2, AxisType.LOCAL)
  tid, wg_g, rem = (w_row * waves_k + wave_k) * 32 + lane, rb // (t_outer * row_blks), rb % (t_outer * row_blks)
  t_o, rblk, sub_l, which = rem // row_blks, rem % row_blks, lane % lanes_k, lane // lanes_k
  row_base = ((rblk * w_rows + w_row) * opw + which) * rpw
  raw_u32, need_xs, grid = raw.bitcast(dtypes.uint32) if (qt == MXFP4 and aligned4) else raw, qt not in _ZERO_XS_QUANTS, grids[0] if grids else None
  if lds_grid and grid is not None and not w1_only and not x_lds:
    g_len = int(grid.numel())
    g_lds, g_i = UOp.alloc((g_len,), dtypes.uint32, addrspace=AddrSpace.LOCAL), UOp.range((g_len + threads - 1) // threads, 20)
    g_off = tid + g_i * threads
    grid = g_lds.after(g_lds[g_off.valid(g_off < g_len)].store(grid[g_off.valid(g_off < g_len)].load()).end(g_i).barrier())
  ones_h2, z = UOp.const(0x3c003c00, dtypes.uint32), UOp.const(0.0, dtypes.float32)
  acc = UOp.alloc((t_inner * rpw,), dtypes.float32, addrspace=AddrSpace.REG)
  acc = acc.after(acc.store(acc.const_like(0.0)))
  xh_lds = UOp.alloc((t_inner, 16, 32), dtypes.uint32, addrspace=AddrSpace.LOCAL) if x_lds else None
  xs_lds = UOp.alloc((t_inner, 2, 32), dtypes.float32, addrspace=AddrSpace.LOCAL) if (x_lds and need_xs) else None

  s = UOp.range(steps, 5)
  if x_lds:
    assert xh_lds is not None
    tot_xg, xi = t_inner * 32, UOp.range((t_inner * 32 + threads - 1) // threads, 22)
    xid = tid + xi * threads
    xok, xid_s = xid < tot_xg, xid.minimum(tot_xg - 1)
    xt_l, xg_l = xid_s // 32, xid_s % 32
    gg = s * 32 + xg_l
    g_ok, gg_s, xt_g = xok & (gg < group_count), gg.minimum(group_count - 1), wg_g * t_per_wg + xt_l
    xh_l, xs0_l, xs1_l = _load_x_32(x, xtoken(xt_g) if xtoken is not None else xt_g, gg_s, ones_h2, z, need_xs, valid=g_ok, rne=True)
    sts = [xh_lds.after(s)[xt_l.valid(xok), j, xg_l].store(xh_l[j]) for j in range(16)]
    if xs_lds is not None: sts += [xs_lds.after(s)[xt_l.valid(xok), 0, xg_l].store(xs0_l), xs_lds.after(s)[xt_l.valid(xok), 1, xg_l].store(xs1_l)]
    xb = UOp.group(*sts).end(xi).barrier()
    xh_r, xs_r = xh_lds.after(xb), xs_lds.after(xb) if xs_lds is not None else None

  grp_raw = sub_l + (wave_k + s * waves_k) * lanes_k
  valid_g, grp = grp_raw < group_count, grp_raw.minimum(group_count - 1)
  g_off = ((group_rows > 0).where(wg_g * (out_features * in_features * QUANT_SIZES[qt] // 256 // raw.dtype.itemsize), 0)
           if group_rows is not None else 0)
  def _row_dot(r:int|UOp):
    base = _quant_base(row_base + r, grp, in_features, qt, raw.dtype.itemsize) + g_off
    return _direct_group_dot_fn(raw, raw_u32, gate(wg_g, base) if gate is not None else base, grp % 8, qt, grid,
                                aligned4=aligned4, stream_w=stream_w, lane=lane)
  if x_lds:
    dot_fn, t_i = _row_dot(0), UOp.range(t_inner, 6)
    xh_v = [xh_r[t_i, j, lane].load() for j in range(16)]
    xs0_v, xs1_v = (xs_r[t_i, 0, lane].load(), xs_r[t_i, 1, lane].load()) if xs_r is not None else (z, z)
    upd = acc[t_i].store(acc.after(s)[t_i].load() + valid_g.where(dot_fn(xh_v, xs0_v, xs1_v), z)).end(t_i).barrier().end(s)
  else:
    tok = wg_g * t_per_wg + t_o
    xh_v, xs0_v, xs1_v = _load_x_32(x, xtoken(tok) if xtoken is not None else tok, grp, ones_h2, z, need_xs)
    r_i = UOp.range(rpw, 26)
    upd = acc[r_i].store(acc.after(s, r_i)[r_i].load() + valid_g.where(_row_dot(r_i)(xh_v, xs0_v, xs1_v), z)).end(r_i, s)

  acc_done = acc.after(upd)
  def _red_sub(v:UOp) -> UOp:
    v = warp_reduce(v, masks=(4, 2, 1))
    v = (lanes_k > 8).where(v + _lane_xor(v, 8), v)
    return (lanes_k > 16).where(v + _lane_xor(v, 16), v)
  def _mask_out(v:UOp) -> UOp: return (row_mask(wg_g).where(v, z) if row_mask is not None else v).cast(out.dtype)
  if x_lds:
    t_o2 = UOp.range(t_inner, 8)
    return out[wg_g * t_per_wg + t_o + t_o2, row_base.valid(lane.eq(0))].store(_mask_out(warp_reduce(acc_done[t_o2].load(), full_wave=True))).end(
      t_o2, lane, wave_k, w_row, rb).sink(arg=KernelInfo(name=name + QUANT_NAMES[qt], opts_to_apply=()))
  tok_o, r_o = wg_g * t_per_wg + t_o, UOp.range(rpw, 27)
  if w1_only:
    return out[tok_o, (row_base + r_o).valid(sub_l.eq(0))].store(_mask_out(_red_sub(acc_done[r_o].load()))).end(
      r_o, lane, wave_k, w_row, rb).sink(arg=KernelInfo(name=name + QUANT_NAMES[qt], opts_to_apply=()))
  w_lds, r_s = UOp.alloc((rpw, waves_k), dtypes.float32, addrspace=AddrSpace.LOCAL), UOp.range(rpw, 28)
  w_lds_r = w_lds.after(w_lds[r_s, wave_k.valid(sub_l.eq(0))].store(_red_sub(acc_done[r_s].load())).end(r_s).barrier())
  wk_r = UOp.range(waves_k, 12)
  return out[tok_o, (row_base + r_o).valid(wave_k.eq(0) & sub_l.eq(0))].store(
    _mask_out(w_lds_r[r_o, wk_r].load().reduce(wk_r, arg=(Ops.ADD, 0)))).end(
      r_o, lane, wave_k, w_row, rb).sink(arg=KernelInfo(name=name + QUANT_NAMES[qt], opts_to_apply=()))

def _select_direct_variants(build:Callable[..., UOp], inf:UOp, outf:UOp, qt:int) -> UOp:
  k_w1, k_reg = build(w1_only=True), build(lds_grid=(qt == IQ3_XXS))
  if qt in _RPW2_QUANTS:
    k_r2 = build(rpw=2, wk_cap=8, lds_grid=(qt == IQ3_XXS))
    if qt in _LDS_GRID_QUANTS and qt != IQ3_XXS: k_r2 = UOp(Ops.WHERE, src=(inf > 8192, build(rpw=2, wk_cap=8, lds_grid=True), k_r2))
    k_reg = UOp(Ops.WHERE, src=((outf % 2).eq(0), k_r2, k_reg))
  return UOp(Ops.WHERE, src=(inf <= 1024, k_w1, k_reg))

# ******** shared quant word & exact FP16 WMMA dequant ********

def _quant_word(raw:UOp, base:UOp, sub:UOp, i:int|UOp, qt:int, grid:UOp|None) -> UOp:
  if qt == Q2_K: return (_load_u32(raw, base, 16 + (sub // 4) * 32 + (i // 4) * 16 + (i % 4) * 4) >> _sh((sub % 4) * 2)) & 0x03030303
  if qt == Q3_K:
    qs = (_load_u32(raw, base, 32 + (sub // 4) * 32 + (i // 4) * 16 + (i % 4) * 4) >> _sh((sub % 4) * 2)) & 0x03030303
    return qs | (((_load_u32(raw, base, (i // 4) * 16 + (i % 4) * 4) >> _sh(sub)) & 0x01010101) << 2)
  if qt == Q6_K:
    ql = (_load_u32(raw, base, (sub // 4) * 64 + (sub % 2) * 32 + (i // 4) * 16 + (i % 4) * 4) >> _sh((sub % 4 // 2) * 4)) & 0x0f0f0f0f
    return ql | (((_load_u32(raw, base, 128 + (sub // 4) * 32 + (i // 4) * 16 + (i % 4) * 4) >> _sh((sub % 4) * 2)) & 0x03030303) << 4)
  if qt == IQ4_NL: return _nibble_lut(_load_u32(raw, base, 2 + (i % 4) * 4), (i // 4) * 4, IQ4NL_LUT)
  if qt == NVFP4:
    return _nibble_lut(_load_u32(raw, base, (sub // 2) * 36 + 4 + (sub % 2) * 16 + (i // 4) * 8 + (i % 2) * 4), ((i // 2) % 2) * 4, MXFP4_LUT)
  assert grid is not None
  if qt == IQ3_S:
    return _iq_signed_word(grid[_load_byte(raw, base, 2 + sub * 8 + i) | (((_load_byte(raw, base, 66 + sub) >> _sh(i)) & 1) << 8)].load(),
                           _load_byte(raw, base, 74 + sub * 4 + i // 2) >> _sh((i % 2) * 4))
  if qt == IQ3_XXS:
    return _iq_signed_word(grid[_load_byte(raw, base, 2 + sub * 8 + i)].load(),
                           _iq_even_signs((_load_u32(raw, base, 66 + sub * 4) >> _sh((i // 2) * 7)) & 127) >> _sh((i % 2) * 4))
  if qt == IQ2_XXS:
    return _iq_signed_word(grid[_load_byte(raw, base, 2 + sub * 8 + i // 2) * 2 + i % 2].load(),
                           _iq_even_signs((_load_u32(raw, base, 6 + sub * 8) >> _sh((i // 2) * 7)) & 127) >> _sh((i % 2) * 4))
  if qt == IQ2_XS:
    p = raw[base + 1 + sub * 4 + i // 2].cast(dtypes.uint32)
    return _iq_signed_word(grid[(p & 511) * 2 + i % 2].load(), _iq_even_signs(p >> 9) >> _sh((i % 2) * 4))
  if qt == IQ2_S:
    gi = (_load_byte(raw, base, 2 + sub * 4 + i // 2) | (((_load_byte(raw, base, 66 + sub) >> _sh((i // 2) * 2)) & 3) << 8)) * 2 + i % 2
    return _iq_signed_word(grid[gi].load(), _load_byte(raw, base, 34 + sub * 4 + i // 2) >> _sh((i % 2) * 4))
  gi = (_load_byte(raw, base, 2 + sub * 4 + i // 2) | (((raw[base + 17 + sub].cast(dtypes.uint32) >> _sh((i // 2) * 3)) & 7) << 8)) * 2 + i % 2
  return grid[gi].load()

def _make_wmma_dequant(raw:UOp, qt:int, lane_group:UOp, kgroups:int, k_u:UOp, table:UOp|None) -> Callable[[UOp, UOp, UOp], UOp|tuple[UOp, ...]]:
  w_u, b_u = k_u // 4, k_u % 4
  w_idx = w_u if kgroups == 1 else (lane_group + kgroups * w_u)
  word_indices = tuple(range(4)) if kgroups == 1 else tuple(lane_group + kgroups * j for j in range(4 // kgroups))
  def dequant(base:UOp, sub:UOp, half:UOp) -> UOp|tuple[UOp, ...]:
    if qt in (Q4_K, Q5_K):
      d, dmin, sc, mn = _q5_scales(raw, base, sub, stream=False)
      w = (raw[base + (4 if qt == Q4_K else 12) + (sub // 2) * 8 + half * 4 + w_idx].load() >> _sh((sub & 1) * 4)) & 0x0f0f0f0f
      if qt == Q5_K: w |= ((raw[base + 4 + half * 4 + w_idx].load() >> _sh(sub)) & 0x01010101) << 4
      return (((w >> _sh(b_u * 8)) & 255).float() * (d * sc) - dmin * mn).cast(dtypes.float16)
    if qt == IQ4_XS:
      assert table is not None
      pair = table[((raw[base + 2 + sub * 4 + w_idx].load() >> _sh(b_u * 8)) & 255).cast(dtypes.weakint)].load()
      return (_half((pair >> _sh(half * 16)) & 0xffff) * _iq4_scale(raw, base, sub)).cast(dtypes.float16)
    if qt in SMALL_BLOCK_QUANTS: base = base + sub * SMALL_BLOCK_QUANTS[qt]
    if qt == MXFP4:
      e = raw[base].cast(dtypes.uint32)
      small = e < 116
      c_lo = small.where(UOp.const(0x26242000, dtypes.uint32), UOp.const(0x36343000, dtypes.uint32))
      c_hi = small.where(UOp.const(0x2e2c2a28, dtypes.uint32), UOp.const(0x3e3c3a38, dtypes.uint32))
      sc2 = _e8m0_halved(e + small.where(UOp.const(7, dtypes.uint32), UOp.const(3, dtypes.uint32))).cast(dtypes.half).reshape((1,)).expand((2,))
      z_u, s01, s23, out_v = UOp.const(0, dtypes.uint32), UOp.const(0x05000400, dtypes.uint32), UOp.const(0x07000600, dtypes.uint32), list[UOp]()
      for wi in word_indices:
        pw = _load_u32(raw, base, 1 + wi * 4) >> _sh(half * 4)
        fpb = _amd_byte_perm(c_hi, c_lo, pw & 0x07070707) | ((pw & 0x08080808) << 4)
        v01 = _amd_byte_perm(fpb, z_u, s01).reshape((1,)).bitcast(dtypes.half) * sc2
        v23 = _amd_byte_perm(fpb, z_u, s23).reshape((1,)).bitcast(dtypes.half) * sc2
        out_v.extend((v01[0], v01[1], v23[0], v23[1]))
      return tuple(out_v)
    if qt == Q8_0:
      sc2 = raw[base].load().bitcast(dtypes.half).reshape((1,)).expand((2,))
      neg, c64 = UOp.const(-1152.0, dtypes.half).reshape((1,)).expand((2,)), UOp.const(0x64006400, dtypes.uint32)
      if word_indices == (0, 1, 2, 3):
        hv = _amd_load(raw[base + 1 + half * 8], 8).bitcast(dtypes.uint32)
        c80, slo, shi = UOp.const(0x80808080, dtypes.uint32), UOp.const(0x05010500, dtypes.uint32), UOp.const(0x05030502, dtypes.uint32)
        return tuple(v[k] for qi in range(4) for sl in (slo, shi)
                     for v in ((_amd_byte_perm(c64, hv[qi] ^ c80, sl).reshape((1,)).bitcast(dtypes.half) + neg) * sc2,) for k in (0, 1))
      return tuple(v[k] for w in word_indices for p in range(2)
                   for u in (raw[base + 1 + half * 8 + w * 2 + p].load().cast(dtypes.uint32) ^ UOp.const(0x8080, dtypes.uint32),)
                   for v in ((_amd_byte_perm(c64, u, UOp.const(0x05010500, dtypes.uint32)).reshape((1,)).bitcast(dtypes.half) + neg) * sc2,)
                   for k in (0, 1))
    def byte(o:int|UOp) -> UOp: return _load_byte(raw, base, o)
    zero: int|UOp = 0
    minimum = UOp.const(0, dtypes.float)
    if qt == Q2_K:
      sc = byte(sub * 2 + half)
      scale, minimum = _half(raw[base + 40]) * (sc & 15).float(), _half(raw[base + 41]) * (sc >> 4).float()
    elif qt == Q3_K:
      j = sub * 2 + half
      sc_q3 = (((byte(96 + j % 8) >> _sh((j // 8) * 4)) & 15) | (((byte(104 + j % 4) >> _sh((j // 4) * 2)) & 3) << 4)).cast(dtypes.int32) - 32
      scale, zero = _half(raw[base + 54]) * sc_q3.float(), 4
    elif qt == Q6_K: scale, zero = _half(raw[base + 104]) * byte(192 + sub * 2 + half).cast(dtypes.uint8).bitcast(dtypes.int8).float(), 32
    elif qt == IQ1_S: scale, zero = _iq1_scale(raw, base, sub)
    elif qt == NVFP4:
      e = byte((sub // 2) * 36 + (sub % 2) * 2 + half).cast(dtypes.uint32)
      if word_indices == (0, 1, 2, 3):
        sc2 = e.eq(0x7f).where(UOp.const(0, dtypes.uint32), (e & 0x7f) << 7).cast(dtypes.uint16).bitcast(dtypes.half).reshape((1,)).expand((2,))
        c_lo, c_hi, z_u = UOp.const(0x5e5c5800, dtypes.uint32), UOp.const(0x66646260, dtypes.uint32), UOp.const(0, dtypes.uint32)
        s01, s23 = UOp.const(0x05000400, dtypes.uint32), UOp.const(0x07000600, dtypes.uint32)
        pv, vals = _amd_load(raw[base + (sub // 2) * 9 + 1 + ((sub % 2) * 2 + half) * 2], 2), list[UOp|None]([None] * 16)
        for wi in range(2):
          pw = pv[wi]
          for bi, fpb in ((0, _amd_byte_perm(c_hi, c_lo, pw & 0x07070707) | ((pw & 0x08080808) << 4)),
                          (8, _amd_byte_perm(c_hi, c_lo, (pw >> 4) & 0x07070707) | (pw & 0x80808080))):
            v01 = _amd_byte_perm(fpb, z_u, s01).reshape((1,)).bitcast(dtypes.half) * sc2
            v23 = _amd_byte_perm(fpb, z_u, s23).reshape((1,)).bitcast(dtypes.half) * sc2
            vals[bi + wi * 4:bi + wi * 4 + 4] = [v01[0], v01[1], v23[0], v23[1]]
        return tuple(v for v in vals if v is not None)
      scale = _ue4m3_halved(e)
    else:
      scale = _half(raw[base])
      if qt in (IQ2_XS, IQ2_S): scale *= (((byte((66 if qt == IQ2_XS else 74) + sub) >> _sh(half * 4)) & 15).float() + 0.5) * 0.25
      elif qt == IQ3_S: scale *= (1 + 2 * ((byte(106 + sub // 2) >> _sh((sub % 2) * 4)) & 15)).float()
      elif qt == IQ3_XXS: scale *= ((_load_u32(raw, base, 66 + sub * 4) >> 28).float() + 0.5) * 0.5
      elif qt == IQ2_XXS: scale *= ((_load_u32(raw, base, 6 + sub * 8) >> 28).float() + 0.5) * 0.25
    if qt in (IQ4_NL, NVFP4):
      ws = tuple(_quant_word(raw, base, sub, half * 4 + wi, qt, table) for wi in word_indices)
      w_reg = UOp.alloc((len(ws),), dtypes.uint32, addrspace=AddrSpace.REG)
      w = w_reg.after(w_reg.store(UOp.stack(*ws)))[w_u].load()
    else: w = _quant_word(raw, base, sub, half * 4 + w_idx, qt, table)
    return (((w >> _sh(b_u * 8)).cast(dtypes.uint8).bitcast(dtypes.int8).float() - zero) * scale - minimum).cast(dtypes.half)
  return dequant

@functools.cache
def _quant_linear_f16_wmma_kernel(out:UOp, raw:UOp, x:UOp, *grids:UOp, m_blks:UOp, m_tiles:UOp, out_features:UOp, in_features:UOp,
                                  ggml_type:int, kgroups:int, lanes:int) -> UOp:
  ob, tb = UOp.range(out_features // 16, 0, AxisType.GLOBAL), UOp.range(m_blks, 1, AxisType.GLOBAL)
  lane = UOp.range(lanes, -1, axis_type=AxisType.WARP)
  col, lg, rows, frag_k = lane % 16, lane // 16, 256 // lanes, 16 // kgroups
  output, grid = ob * 16 + col, None
  if grids:
    g_n = int(grids[0].numel())
    grid = UOp.alloc((g_n,), dtypes.uint32, addrspace=AddrSpace.LOCAL)
    g_i, g4 = UOp.range(g_n // (lanes * 4), 20), UOp.range(4, 21, AxisType.UPCAST)
    g_off = (lane + g_i * lanes) * 4 + g4
    grid = grid.after(grid[g_off].store(grids[0][g_off].load()).end(g4, g_i))
  tw, ow = QUANT_SIZES[ggml_type] // raw.dtype.itemsize, in_features * (QUANT_SIZES[ggml_type] // raw.dtype.itemsize) // GGML_BLOCK_SIZE
  grp, half = UOp.range(in_features // Q8_GROUP_SIZE, 4), UOp.range(2, 5)
  k_u, m_u, mt_u = UOp.range(frag_k, 10, AxisType.UPCAST), UOp.range(rows, 11, AxisType.UPCAST), UOp.range(m_tiles, 12, AxisType.UPCAST)
  w_idx = (k_u // 4) if kgroups == 1 else (lg + kgroups * (k_u // 4))
  a_el = x[(tb * m_tiles + mt_u) * 16 + col, (grp * 2 + half) * 16 + w_idx * 4 + (k_u % 4)].load()
  b_raw = _make_wmma_dequant(raw, ggml_type, lg, kgroups, k_u, grid)(output * ow + (grp // 8) * tw, grp % 8, half)
  b_el = ((br := UOp.alloc((frag_k,), dtypes.float16, addrspace=AddrSpace.REG)).after(
    br.after(grp, half).store(UOp.stack(*b_raw)))[k_u].load() if isinstance(b_raw, tuple) else b_raw)
  tc_axes = (((k_u.arg, frag_k), (m_u.arg, 1)), ((k_u.arg, frag_k), (m_u.arg, 1)), ((k_u.arg, 1), (m_u.arg, rows)))
  res = UOp.wmma(a_el, b_el, UOp.const((0.0,) * rows, dtypes.float32), WMMA_ARG[0], lanes, tc_upcast_axes=tc_axes).reduce(grp, half, arg=(Ops.ADD, 0))
  return UOp(Ops.STORE, src=(out[(tb * m_tiles + mt_u) * 16 + (lg * rows + m_u if kgroups > 1 else m_u * 2 + lg), output], res.cast(out.dtype))).end(
    k_u, m_u, mt_u, tb, ob, lane).sink(arg=KernelInfo(name=f"linear_{QUANT_NAMES[ggml_type]}_f16_wmma", optimized=True))

# ******** RDNA3 Direct-A + Double-Buffered-B 2D FP16 WMMA (with in-WG K-split) ********

@functools.cache
def _rdna3_gemm_2d_kernel(out:UOp, raw:UOp, x:UOp, *grids:UOp, m_blks:UOp, n_blks:UOp, wm:UOp, wn:UOp, tm:UOp, tn:UOp,
                          bk:UOp, out_features:UOp, in_features:UOp, ggml_type:int, wk:int|UOp=1, wk_split:bool=False) -> UOp:
  bm, bn, waves = wm * tm * 16, wn * tn * 16, wm * wn * wk
  threads, k_groups, k8_cnt, k16_cnt = waves * 32, bk // 32, bk // 8, bk // 16
  k16_pw = k16_cnt // wk
  split_ks, b_total, total_kt = k16_pw // 2, bn * k_groups, (in_features // bk).maximum(1)
  b_iters, k_pairs, swz_m = (b_total + threads - 1) // threads, (total_kt + 1) // 2, (k8_cnt - 1).minimum(7)
  fast_b = ggml_type in (Q8_0, Q4_K, IQ4_XS, MXFP4, IQ3_XXS, IQ2_S)
  x_u32, tw = x.flatten().bitcast(dtypes.uint32), QUANT_SIZES[ggml_type] // raw.dtype.itemsize
  ow = in_features * tw // GGML_BLOCK_SIZE
  gid = UOp.range(m_blks * n_blks, 0, AxisType.GLOBAL)
  swz = (m_blks % 4).eq(0).where(4, (m_blks % 2).eq(0).where(2, 1))
  grp, in_grp = gid // (swz * n_blks), gid % (swz * n_blks)
  block_m, block_n = grp * swz + in_grp % swz, in_grp // swz
  lane, wave = UOp.range(32, -1, AxisType.WARP), UOp.range(waves, 2, AxisType.LOCAL)
  tid, col, lg, wave_k, wave_mn = wave * 32 + lane, lane % 16, lane // 16, wave % wk, wave // wk
  wave_m, wave_n = wave_mn // wn, wave_mn % wn
  grid = None
  if grids and (not fast_b or ggml_type in (IQ3_XXS, IQ2_S)):
    g_len = int(grids[0].numel())
    grid, g_i = UOp.alloc((g_len,), dtypes.uint32, addrspace=AddrSpace.LOCAL), UOp.range((g_len + threads - 1) // threads, 20)
    g_off = tid + g_i * threads
    grid = grid.after(grid[g_off.valid(g_off < g_len)].store(grids[0][g_off.valid(g_off < g_len)].load()).end(g_i).barrier())
  b_elems = bn * k8_cnt * 8
  red_f32_elems = (wm * wn * wk * tm * tn * 256) if wk_split else 0
  b0_halves = b_elems.maximum(2 * red_f32_elems) if wk_split else b_elems
  B_lds_raw = [UOp.alloc((b0_halves,), dtypes.half, addrspace=AddrSpace.LOCAL), UOp.alloc((b_elems,), dtypes.half, addrspace=AddrSpace.LOCAL)]
  B_lds = [B_lds_raw[0][:b_elems].reshape(bn, k8_cnt, 8) if wk_split else B_lds_raw[0].reshape(bn, k8_cnt, 8),
           B_lds_raw[1].reshape(bn, k8_cnt, 8)]
  B_h2, acc = [b.reshape(bn, k8_cnt, 4, 2) for b in B_lds], UOp.alloc((tm, tn, 8), dtypes.float32, addrspace=AddrSpace.REG)
  acc, exact_q = acc.after(acc.store(acc.const_like(0.0))), out_features < 32
  stride = {Q4_K: 6, IQ4_XS: 6, Q8_0: 9, MXFP4: 5, IQ3_XXS: 4, IQ2_S: 5}.get(ggml_type, 1)

  def load_b_reg(kt:UOp, dep:tuple[UOp, ...], rid:int) -> tuple[UOp, ...]:
    if not fast_b: return ()
    buf, bi = UOp.alloc((b_iters, stride), dtypes.uint32, addrspace=AddrSpace.REG), UOp.range(b_iters, rid)
    bp = (tid + bi * threads).minimum(b_total - 1)
    gn = (block_n * bn + bp // k_groups).minimum(out_features - 1)
    gbg = (kt * k_groups + bp % k_groups).minimum(in_features // 32 - 1)
    if ggml_type == Q4_K:
      base, sub = (gn * (in_features // 256) + gbg // 8) * 36, gbg % 8
      ds0, dm0, ds1, dm1 = _q4k_pair_scales(_v4(_amd_load(raw[base], 4)), sub // 2)
      vs = _v4(_amd_load(raw[base + 4 + (sub // 2) * 8 + (sub & 1) * 4], 4)) + [
        UOp.stack(ds0.cast(dtypes.half), (-dm0).cast(dtypes.half)).bitcast(dtypes.uint32),
        UOp.stack(ds1.cast(dtypes.half), (-dm1).cast(dtypes.half)).bitcast(dtypes.uint32)]
    elif ggml_type == IQ4_XS:
      sub, base = gbg % 8, (gn * (in_features // 256) + gbg // 8) * 34
      hdr = _amd_load(raw[base], 2)
      vs = _v4(_amd_load(raw[base + 2 + sub * 4], 4)) + [hdr[0], hdr[1]]
    elif ggml_type == Q8_0:
      wb = (gn * (in_features // 32) + gbg) * 17
      vs = (_v4(_amd_load(raw[wb + 1], 8).bitcast(dtypes.uint32)) +
            _v4(_amd_load(raw[wb + 9], 8).bitcast(dtypes.uint32)) + [raw[wb].load().cast(dtypes.uint32)])
    elif ggml_type == MXFP4:
      wb = (gn * (in_features // 32) + gbg) * 17
      vs = _v4(_amd_load(raw[wb + 1], 16, stream=True).bitcast(dtypes.uint32)) + [raw[wb].load().cast(dtypes.uint32)]
    elif ggml_type == IQ3_XXS:
      sub, base = gbg % 8, gn * ow + (gbg // 8) * tw
      qs = _amd_load(raw[base + 1 + sub * 4], 4).bitcast(dtypes.uint32)
      vs = [qs[0], qs[1], _amd_load(raw[base + 33 + sub * 2], 2).bitcast(dtypes.uint32)[0], raw[base].load().cast(dtypes.uint32)]
    else:  # IQ2_S
      sub, base = gbg % 8, gn * ow + (gbg // 8) * tw
      vs = [_amd_load(raw[base + 1 + sub * 2], 2).bitcast(dtypes.uint32)[0],
            _amd_load(raw[base + 17 + sub * 2], 2).bitcast(dtypes.uint32)[0],
            raw[base + 33 + sub // 2].load().cast(dtypes.uint32),
            raw[base + 37 + sub // 2].load().cast(dtypes.uint32), raw[base].load().cast(dtypes.uint32)]
    return (buf.after(UOp.group(*(buf.after(*dep)[bi, j].store(vs[j]) for j in range(stride))).end(bi)),)

  def store_b_lds(buf_i:int, bregs:tuple[UOp, ...], kt:UOp, dep:tuple[UOp, ...], rid:int) -> UOp:
    bi = UOp.range(b_iters, rid)
    bp_raw = tid + bi * threads
    ok, bp = bp_raw < b_total, bp_raw.minimum(b_total - 1)
    bn_i, bg = bp // k_groups, bp % k_groups
    bn_v, gn, gbg = bn_i.valid(ok), (block_n * bn + bn_i).minimum(out_features - 1), (kt * k_groups + bg).minimum(in_features // 32 - 1)
    if fast_b:
      reg, dst, sts = bregs[0], B_h2[buf_i].after(*dep), list[UOp]()
      c64, c80, slo, shi = (UOp.const(v, dtypes.uint32) for v in (0x64006400, 0x80808080, 0x05010500, 0x05030502))
      n1152 = UOp.const(0xe480e480, dtypes.uint32).reshape((1,)).bitcast(dtypes.half)
      n1024 = UOp.const(0xe400e400, dtypes.uint32).reshape((1,)).bitcast(dtypes.half)
      def store_k16(bk_v:int|UOp, *ps:tuple[UOp, UOp]):
        for m_i, q in enumerate(((ps[0][0], ps[0][1], ps[1][0], ps[1][1]), (ps[2][0], ps[2][1], ps[3][0], ps[3][1]))):
          sts.extend(dst[bn_v, (bk_v * 2 + m_i) ^ (bn_i & swz_m), j].store(h2) for j, h2 in enumerate(q))
      if ggml_type == Q4_K:
        qs, sps, sub_q = [reg[bi, j].load() for j in range(4)], [reg[bi, 4 + m].load() for m in (0, 1)], gbg % 8
        ds0_e, dm0_e, ds1_e, dm1_e = _q4k_pair_scales(_v4(_amd_load(raw[(gn * (in_features // 256) + gbg // 8) * 36], 4)), sub_q // 2)
        for m_i, (sp, (ds_e, ndm_e)) in enumerate(zip(sps, ((ds0_e, -dm0_e), (ds1_e, -dm1_e)))):
          ds2, dm2 = _splat_h2(sp), _amd_byte_perm(sp, sp, UOp.const(0x03020302, dtypes.uint32)).reshape((1,)).bitcast(dtypes.half)
          def dec4_q4k(u:UOp) -> tuple[UOp, UOp]:
            u4 = u & 0x0f0f0f0f
            f0, f1 = ((_amd_byte_perm(c64, u4, sl).reshape((1,)).bitcast(dtypes.half) + n1024) * ds2 + dm2 for sl in (slo, shi))
            he = [(((u >> (b * 8)) & 15).float() * ds_e + ndm_e).cast(dtypes.half) for b in range(4)]
            return exact_q.where(UOp.stack(he[0], he[1]), f0), exact_q.where(UOp.stack(he[2], he[3]), f1)
          store_k16((bg // 2) * 4 + m_i * 2 + (bg % 2), *(dec4_q4k(qs[j] >> (m_i * 4) if m_i else qs[j]) for j in range(4)))
      elif ggml_type == IQ4_XS:
        sub, ps, h0, h1 = gbg % 8, [reg[bi, j].load() for j in range(4)], reg[bi, 4].load(), reg[bi, 5].load()
        sc_f = _half(h0) * ((((h1 >> _sh(4 * sub)) & 15) | (((h0 >> _sh(16 + 2 * sub)) & 3) << 4)).cast(dtypes.int32) - 32).float()
        c4b, zf = UOp.const(0x4b000000, dtypes.uint32), UOp.const(-8388736.0, dtypes.float32)
        def dec4_iq4(ux_u:UOp) -> tuple[UOp, UOp]:
          hs = [((_amd_byte_perm(c4b, ux_u, UOp.const(0x07060500 + b, dtypes.uint32)).bitcast(dtypes.float32) + zf) * sc_f).cast(dtypes.half)
                for b in range(4)]
          return UOp.stack(hs[0], hs[1]), UOp.stack(hs[2], hs[3])
        for k_i in (0, 1): store_k16(bg * 2 + k_i, *(dec4_iq4(_nibble_lut(ps[j], k_i * 4, IQ4NL_U8_LUT)) for j in range(4)))
      elif ggml_type == Q8_0:
        ws = [reg[bi, j].load() for j in range(9)]
        sc2 = _splat_h2(ws[8])
        for k_i in (0, 1):
          ps_q8 = (tuple((_amd_byte_perm(c64, ws[k_i * 4 + j] ^ c80, sl).reshape((1,)).bitcast(dtypes.half) + n1152) * sc2
                         for sl in (slo, shi)) for j in range(4))
          store_k16(bg * 2 + k_i, *ps_q8)
      elif ggml_type == MXFP4:
        ps, e = [reg[bi, j].load() for j in range(4)], reg[bi, 4].load()
        small = e < 116
        c_lo = small.where(UOp.const(0x26242000, dtypes.uint32), UOp.const(0x36343000, dtypes.uint32))
        c_hi = small.where(UOp.const(0x2e2c2a28, dtypes.uint32), UOp.const(0x3e3c3a38, dtypes.uint32))
        sc2, z_u = _splat_f_h2(_e8m0_halved(e + small.where(UOp.const(7, dtypes.uint32), UOp.const(3, dtypes.uint32)))), UOp.const(0, dtypes.uint32)
        def e2m1_h2(pw:UOp) -> tuple[UOp, UOp]:
          fpb = _amd_byte_perm(c_hi, c_lo, pw & 0x07070707) | ((pw & 0x08080808) << 4)
          return (_amd_byte_perm(fpb, z_u, UOp.const(0x05000400, dtypes.uint32)).reshape((1,)).bitcast(dtypes.half) * sc2,
                  _amd_byte_perm(fpb, z_u, UOp.const(0x07000600, dtypes.uint32)).reshape((1,)).bitcast(dtypes.half) * sc2)
        for k_i in (0, 1): store_k16(bg * 2 + k_i, *(e2m1_h2(ps[j] >> (k_i * 4) if k_i else ps[j]) for j in range(4)))
      elif ggml_type == IQ3_XXS:
        assert grid is not None
        q0, q1, sc_w, d_u = (reg[bi, j].load() for j in range(4))
        sc_f = _half(d_u) * ((sc_w >> 28).float() + 0.5) * 32.0
        sc2, sgs = _splat_f_h2(sc_f), [_iq_even_signs((sc_w >> (7 * p)) & 127) for p in range(4)]
        def dec4_iq3(g_idx:UOp, s4:UOp) -> tuple[UOp, UOp]:
          g0, g1 = grid[g_idx * 2].load() ^ _sign_h2(s4), grid[g_idx * 2 + 1].load() ^ _sign_h2(s4 >> 2)
          h0, h1 = g0.reshape((1,)).bitcast(dtypes.half), g1.reshape((1,)).bitcast(dtypes.half)
          e0 = UOp.stack((h0[0].float() * sc_f).cast(dtypes.half), (h0[1].float() * sc_f).cast(dtypes.half))
          e1 = UOp.stack((h1[0].float() * sc_f).cast(dtypes.half), (h1[1].float() * sc_f).cast(dtypes.half))
          return exact_q.where(e0, h0 * sc2), exact_q.where(e1, h1 * sc2)
        for k_i, qw in ((0, q0), (1, q1)):
          store_k16(bg * 2 + k_i, dec4_iq3(qw & 255, sgs[k_i * 2]), dec4_iq3((qw >> 8) & 255, sgs[k_i * 2] >> 4),
                    dec4_iq3((qw >> 16) & 255, sgs[k_i * 2 + 1]), dec4_iq3((qw >> 24) & 255, sgs[k_i * 2 + 1] >> 4))
      else:  # IQ2_S (raw 2048-u32 grid!)
        assert grid is not None
        sub, (q_w, s_w, qh_u, sc_u, d_u) = gbg % 8, (reg[bi, j].load() for j in range(5))
        qh_b, sc_b, d_f = (qh_u >> _sh((sub & 1) * 8)) & 255, (sc_u >> _sh((sub & 1) * 8)) & 255, _half(d_u)
        sf0, sf1 = d_f * ((sc_b & 15).float() + 0.5) * 0.25, d_f * (((sc_b >> 4) & 15).float() + 0.5) * 0.25
        sc2_0, sc2_1 = _splat_f_h2(sf0), _splat_f_h2(sf1)
        def dec4_iq2(gw:UOp, s4:UOp, sf:UOp, sc2:UOp) -> tuple[UOp, UOp]:
          sg = _iq_signed_word(gw, s4)
          f0, f1 = ((_amd_byte_perm(c64, sg ^ c80, sl).reshape((1,)).bitcast(dtypes.half) + n1152) * sc2 for sl in (slo, shi))
          he = [(((sg >> (b * 8)).cast(dtypes.uint8).bitcast(dtypes.int8).float()) * sf).cast(dtypes.half) for b in range(4)]
          return exact_q.where(UOp.stack(he[0], he[1]), f0), exact_q.where(UOp.stack(he[2], he[3]), f1)
        idxs = [((q_w >> (p * 8)) & 255) | (((qh_b >> (2 * p)) & 3) << 8) for p in range(4)]
        for k_i, (sf, sc2) in enumerate(((sf0, sc2_0), (sf1, sc2_1))):
          p0, p1 = k_i * 2, k_i * 2 + 1
          store_k16(bg * 2 + k_i, dec4_iq2(grid[idxs[p0] * 2].load(), (s_w >> (p0 * 8)) & 15, sf, sc2),
                    dec4_iq2(grid[idxs[p0] * 2 + 1].load(), (s_w >> (p0 * 8 + 4)) & 15, sf, sc2),
                    dec4_iq2(grid[idxs[p1] * 2].load(), (s_w >> (p1 * 8)) & 15, sf, sc2),
                    dec4_iq2(grid[idxs[p1] * 2 + 1].load(), (s_w >> (p1 * 8 + 4)) & 15, sf, sc2))
      return UOp.group(*sts).end(bi)
    base, sub = (((gn * (in_features // 32) + gbg) * SMALL_BLOCK_QUANTS[ggml_type], UOp.const(0, dtypes.weakint))
                 if ggml_type in SMALL_BLOCK_QUANTS else (gn * ow + (gbg // 8) * tw, gbg % 8))
    B_dst, b_sts = B_lds[buf_i].after(*dep), list[UOp]()
    for k16 in range(2):
      k_u = UOp.range(16, rid + 1 + k16, AxisType.UPCAST)
      vals = _make_wmma_dequant(raw, ggml_type, UOp.const(0, dtypes.weakint), 1, k_u, grid)(base, sub, UOp.const(k16, dtypes.weakint))
      if isinstance(vals, tuple):
        b_sts.extend(B_dst[bn_v, (bg * 4 + k16 * 2 + m) ^ (bn_i & swz_m), q].store(vals[m * 8 + q]) for m in (0, 1) for q in range(8))
      else: b_sts.append(B_dst[bn_v, (bg * 4 + k16 * 2 + k_u // 8) ^ (bn_i & swz_m), k_u % 8].store(vals).end(k_u))
    return UOp.group(*b_sts).end(bi)

  def compute_slice(kt_cur:UOp, B_r:UOp, acc_in:UOp, kk_start:int|UOp, kk_cnt:UOp, rid:int, valid_k:UOp|None=None) -> UOp:
    n_sl = (kk_cnt // 2).maximum(1)
    sl_r, kk_r, tm_r, tn_r = UOp.range(n_sl, rid + 3), UOp.range(kk_cnt // n_sl, rid), UOp.range(tm, rid + 1), UOp.range(tn, rid + 2)
    kk_loc = (kk_start + sl_r * (kk_cnt // n_sl) + kk_r) if not (isinstance(kk_start, int) and kk_start == 0) else (sl_r * (kk_cnt // n_sl) + kk_r)
    kk = wave_k * k16_pw + kk_loc if wk_split else kk_loc
    mr, nr = (block_m * bm + (wave_m * tm + tm_r) * 16 + col).minimum(x.shape[0] - 1), (wave_n * tn + tn_r) * 16 + col
    a_off = (mr * (in_features // 2) + kt_cur * (bk // 2) + kk * 8).minimum(x.shape[0] * (in_features // 2) - 8)
    a_f = _amd_load(x_u32[a_off], 4).cat(_amd_load(x_u32[a_off + 4], 4)).bitcast(dtypes.half)
    b_f = B_r[nr, (kk * 2) ^ (nr & swz_m)].cat(B_r[nr, (kk * 2 + 1) ^ (nr & swz_m)])
    acc_old = acc_in.after(sl_r, kk_r)[tm_r, tn_r]
    new_acc = UOp.wmma(a_f, b_f, acc_old, *WMMA_ARG)
    w = acc[tm_r, tn_r].store(valid_k.where(new_acc, acc_old.load()) if valid_k is not None else new_acc).end(tn_r, tm_r, kk_r)
    ti, tj = UOp.range(tm, rid + 4), UOp.range(tn, rid + 5)
    return acc[ti, tj, 7].store(_pin_f32(acc.after(w)[ti, tj, 7].load(), w)).end(tj, ti, sl_r)

  zw = UOp.const(0, dtypes.weakint)
  st0, pair = store_b_lds(0, load_b_reg(zw, (), 22), zw, (), 24).barrier(), UOp.range(k_pairs, 10)
  k0, k1_raw, k2 = pair * 2, pair * 2 + 1, ((pair + 1) * 2).minimum(total_kt - 1)
  def step(bc:int, bn_i:int, kc:UOp, kn:UOp, deps:tuple[UOp, ...], rb:int, valid_k:UOp|None=None) -> UOp:
    br, pa = load_b_reg(kn, deps, rb), compute_slice(kc, B_lds[bc].after(*deps), acc.after(*deps), 0, split_ks, rb + 4, valid_k)
    return UOp.group(compute_slice(kc, B_lds[bc].after(*deps, pa), acc.after(pa), split_ks, k16_pw - split_ks, rb + 20, valid_k),
                     store_b_lds(bn_i, br, kn, (*deps, pa), rb + 14)).barrier()
  done = step(1, 0, k1_raw.minimum(total_kt - 1), k2,
              (step(0, 1, k0, k1_raw.minimum(total_kt - 1), (st0, pair), 30),), 70, k1_raw < total_kt).end(pair)
  tm_o, tn_o, e_o = UOp.range(tm, 190), UOp.range(tn, 191), UOp.range(8, 192, AxisType.UPCAST)
  om = (block_m * bm + (wave_m * tm + tm_o) * 16 + e_o * 2 + lg).minimum(out.shape[0] - 1)
  on = (block_n * bn + (wave_n * tn + tn_o) * 16 + col).minimum(out_features - 1)
  if not wk_split:
    return out[om, on].store(
      acc.after(done)[tm_o, tn_o, e_o].load().cast(out.dtype)).end(e_o, tn_o, tm_o, wave, lane, gid).sink(
      arg=KernelInfo(name=f"linear_{QUANT_NAMES[ggml_type]}_f16_wmma", opts_to_apply=()))
  red_lds = B_lds_raw[0].after(done).bitcast(dtypes.float32)[:red_f32_elems].reshape(wm * wn, wk, tm, tn, 32, 8)
  tm_s, tn_s, e_s = UOp.range(tm, 180), UOp.range(tn, 181), UOp.range(8, 182, AxisType.UPCAST)
  red_r = red_lds.after(red_lds[wave_mn, wave_k, tm_s, tn_s, lane, e_s].store(acc.after(done)[tm_s, tn_s, e_s].load()).end(e_s, tn_s, tm_s).barrier())
  k_w = UOp.range(wk, 185)
  tot = red_r[wave_mn, k_w, tm_o, tn_o, lane, e_o].load().reduce(k_w, arg=(Ops.ADD, 0))
  return out[om.valid(wave_k.eq(0)), on].store(tot.cast(out.dtype)).end(e_o, tn_o, tm_o, wave, lane, gid).sink(
    arg=KernelInfo(name=f"linear_{QUANT_NAMES[ggml_type]}_f16_wmma", opts_to_apply=()))

# ******** RDNA4 Direct-FP16 Zero-LDS Split-K (M<64 & N<2048) + OTF FP8 2D WMMA ********

def _cvt_pk_fp8(a:UOp, b:UOp, old:UOp, hi:bool) -> UOp:
  return _nat(f"__builtin_amdgcn_cvt_pk_fp8_f32({{0}}, {{1}}, {{2}}, {'true' if hi else 'false'})", dtypes.uint32, a, b, old)
def _pack4_fp8(*fs:UOp) -> UOp: return _cvt_pk_fp8(fs[2], fs[3], _cvt_pk_fp8(fs[0], fs[1], UOp.const(0, dtypes.uint32), False), True)
def _b4_to_fp8(u:UOp, sc:UOp, zero_off:float, add:UOp|float=0.0) -> UOp:
  c4b, zf = UOp.const(0x4b000000, dtypes.uint32), UOp.const(zero_off, dtypes.float32)
  return _pack4_fp8(*((_amd_byte_perm(c4b, u, UOp.const(0x07060500 + b, dtypes.uint32)).bitcast(dtypes.float32) + zf) * sc + add for b in range(4)))
def _pk_max_u16(a:UOp, b:UOp) -> UOp:
  return UOp.native(
    "({{ unsigned int _x = ({0}), _y = ({1}); __builtin_bit_cast(unsigned int, __builtin_elementwise_max("
    "__builtin_bit_cast(unsigned short __attribute__((ext_vector_type(2))), _x), "
    "__builtin_bit_cast(unsigned short __attribute__((ext_vector_type(2))), _y))); }})",
    shape=(), dtype=dtypes.uint32, fmt=True).call(a, b, inline=False)
def _h2x2_to_fp8_pk(u0:UOp, u1:UOp, inv_h2:UOp) -> UOp:
  h2t = "decltype(__builtin_amdgcn_cvt_pkrtz(0.0f, 0.0f))"
  return UOp.native(
    f"({{{{ unsigned int _u0 = ({{0}}), _u1 = ({{1}}), _s = ({{2}}); "
    f"{h2t} _a = __builtin_bit_cast({h2t}, _u0) * __builtin_bit_cast({h2t}, _s); "
    f"{h2t} _b = __builtin_bit_cast({h2t}, _u1) * __builtin_bit_cast({h2t}, _s); "
    f"unsigned int _r = __builtin_amdgcn_cvt_pk_fp8_f32((float)_a[0], (float)_a[1], 0u, false); "
    f"__builtin_amdgcn_cvt_pk_fp8_f32((float)_b[0], (float)_b[1], _r, true); }}}})",
    shape=(), dtype=dtypes.uint32, fmt=True).call(u0, u1, inv_h2, inline=True)

@functools.cache
def _rdna4_zlds_f16_kernel(out:UOp, raw:UOp, x:UOp, *grids:UOp, tok:UOp, out_features:UOp, in_features:UOp,
                           ggml_type:int, waves_k:int=4, tm:int=2, tn:int=2) -> UOp:
  bm, bn = tm * 16, tn * 16
  m_blks, n_blks, k_sb = (tok + bm - 1) // bm, (out_features + bn - 1) // bn, in_features // 256
  raw_u32 = raw.bitcast(dtypes.uint32) if raw.dtype != dtypes.uint32 else raw
  x_u32, grid, tw = x.flatten().bitcast(dtypes.uint32), grids[0] if grids else None, QUANT_SIZES[ggml_type] // raw.dtype.itemsize
  ow = in_features * tw // GGML_BLOCK_SIZE
  gid, lane, wk = UOp.range(m_blks * n_blks, 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP), UOp.range(waves_k, 2, AxisType.LOCAL)
  block_m, block_n, col, lg = gid // n_blks, gid % n_blks, lane % 16, lane // 16
  acc = UOp.alloc((tm, tn, 8), dtypes.float32, addrspace=AddrSpace.REG)
  acc = acc.after(acc.store(acc.const_like(0.0)))
  c4b, c80 = UOp.const(0x4b000000, dtypes.uint32), UOp.const(0x80808080, dtypes.uint32)
  c64, slo, shi = UOp.const(0x64006400, dtypes.uint32), UOp.const(0x05010500, dtypes.uint32), UOp.const(0x05030502, dtypes.uint32)
  n1024 = UOp.const(0xe400e400, dtypes.uint32).reshape((1,)).bitcast(dtypes.half)
  n1056 = UOp.const(0xe420e420, dtypes.uint32).reshape((1,)).bitcast(dtypes.half)
  z_off = -8388640.0 if ggml_type == Q6_K else -8388612.0 if ggml_type == Q3_K else -8388608.0
  def u8x8_to_f16(u0:UOp, u1:UOp, zf:float, sc:UOp, add:UOp|float=0.0) -> UOp:
    zc = UOp.const(zf, dtypes.float32)
    return UOp.stack(*(((_amd_byte_perm(c4b, u, UOp.const(0x07060500 + bi, dtypes.uint32)).bitcast(dtypes.float32) + zc) * sc + add).cast(dtypes.half)
                       for u in (u0, u1) for bi in range(4)))
  def u8x8_h2(u0:UOp, u1:UOp, neg:UOp, sc2:UOp, add2:UOp|None=None) -> UOp:
    return UOp.stack(*(v[k] for u in (u0, u1) for sl in (slo, shi)
                       for v in ((_amd_byte_perm(c64, u, sl).reshape((1,)).bitcast(dtypes.half) + neg) * sc2 +
                                 (add2 if add2 is not None else UOp.const(0.0, dtypes.half)),) for k in (0, 1)))
  sb_i = UOp.range(k_sb // waves_k, 10)
  sb, hdrs = wk + sb_i * waves_k, list[tuple[UOp, ...]]()
  for tj in range(tn):
    gn = (block_n * bn + tj * 16 + col).minimum(out_features - 1)
    sb_blk, base = gn * k_sb + sb, gn * ow + sb * tw
    if ggml_type in (Q4_K, Q5_K): hdrs.append((base, *_v4(_amd_load(raw_u32[base], 4))))
    elif ggml_type == IQ4_XS:
      h01 = _amd_load(raw_u32[base], 2)
      hdrs.append((base, h01[0], h01[1]))
    elif ggml_type == Q2_K:
      dm_w = raw_u32[sb_blk * 21 + 20].load()
      hdrs.append((sb_blk * 21, _half(dm_w & 0xffff), _half(dm_w >> 16)))
    elif ggml_type in (Q6_K, Q3_K, IQ3_XXS, IQ3_S, IQ2_S, IQ2_XS, IQ2_XXS, IQ1_S):
      hdrs.append((base, _half(raw[base + (104 if ggml_type == Q6_K else 54 if ggml_type == Q3_K else 0)].load().cast(dtypes.uint32))))
    else: hdrs.append((base,))
  def get_b_pair(tj:int, sp_r:UOp, m:int) -> tuple[UOp, UOp]:
    sub = sp_r * 2 + m
    if ggml_type in (Q4_K, Q5_K):
      b, h0_k, h1_k, h2_k, h3_k = hdrs[tj]
      ds0, dm0, ds1, dm1 = _q4k_pair_scales([h0_k, h1_k, h2_k, h3_k], sp_r)
      ds_f, dm_f, sh, off = (ds0 if m == 0 else ds1), (-dm0 if m == 0 else -dm1), m * 4, 4 if ggml_type == Q4_K else 12
      p0, p1 = _amd_load(raw_u32[b + off + sp_r * 8 + lg * 2], 2), _amd_load(raw_u32[b + off + sp_r * 8 + 4 + lg * 2], 2)
      if ggml_type == Q4_K:
        ds2, dm2 = _splat_f_h2(ds_f), _splat_f_h2(dm_f)
        return (u8x8_h2(((p0[0] >> sh) if sh else p0[0]) & 0x0f0f0f0f, ((p0[1] >> sh) if sh else p0[1]) & 0x0f0f0f0f, n1024, ds2, dm2),
                u8x8_h2(((p1[0] >> sh) if sh else p1[0]) & 0x0f0f0f0f, ((p1[1] >> sh) if sh else p1[1]) & 0x0f0f0f0f, n1024, ds2, dm2))
      h0, h1 = _amd_load(raw_u32[b + 4 + lg * 2], 2), _amd_load(raw_u32[b + 4 + 4 + lg * 2], 2)
      def mk_q5(p:UOp, hv:UOp) -> UOp:
        u0, u1 = ((((p[k] >> sh) if sh else p[k]) & 0x0f0f0f0f) | (((hv[k] >> _sh(sub)) & 0x01010101) << 4) for k in (0, 1))
        return u8x8_to_f16(u0, u1, -8388608.0, ds_f, dm_f)
      return mk_q5(p0, h0), mk_q5(p1, h1)
    if ggml_type == Q6_K:
      b, d_f = hdrs[tj]
      ql_b, qh_b = b + (sub // 4) * 32 + (sub % 2) * 16, b + 64 + (sub // 4) * 16
      ql_sh, qh_sh = ((sub % 4 // 2) * 4).cast(dtypes.uint32), ((sub % 4) * 2).cast(dtypes.uint32)
      ql0, ql1 = _amd_load(raw[ql_b + lg * 4], 4).bitcast(dtypes.uint32), _amd_load(raw[ql_b + 8 + lg * 4], 4).bitcast(dtypes.uint32)
      qh0, qh1 = _amd_load(raw[qh_b + lg * 4], 4).bitcast(dtypes.uint32), _amd_load(raw[qh_b + 8 + lg * 4], 4).bitcast(dtypes.uint32)
      sc_w = raw[b + 96 + sub].load().cast(dtypes.uint32)
      s0 = _splat_f_h2((sc_w & 255).cast(dtypes.uint8).bitcast(dtypes.int8).float() * d_f)
      s1 = _splat_f_h2(((sc_w >> 8) & 255).cast(dtypes.uint8).bitcast(dtypes.int8).float() * d_f)
      return (u8x8_h2(((ql0[0] >> ql_sh) & 0x0f0f0f0f) | (((qh0[0] >> qh_sh) & 0x03030303) << 4),
                      ((ql0[1] >> ql_sh) & 0x0f0f0f0f) | (((qh0[1] >> qh_sh) & 0x03030303) << 4), n1056, s0),
              u8x8_h2(((ql1[0] >> ql_sh) & 0x0f0f0f0f) | (((qh1[0] >> qh_sh) & 0x03030303) << 4),
                      ((ql1[1] >> ql_sh) & 0x0f0f0f0f) | (((qh1[1] >> qh_sh) & 0x03030303) << 4), n1056, s1))
    if ggml_type == IQ4_XS:
      b, h0, h1 = hdrs[tj]
      sc_f = _half(h0) * ((((h1 >> _sh(4 * sub)) & 15) | (((h0 >> _sh(16 + 2 * sub)) & 3) << 4)).cast(dtypes.int32) - 32).float()
      p = _amd_load(raw_u32[b + 2 + sub * 4 + lg * 2], 2)
      return (u8x8_to_f16(_nibble_lut(p[0], 0, IQ4NL_U8_LUT), _nibble_lut(p[1], 0, IQ4NL_U8_LUT), -8388736.0, sc_f),
              u8x8_to_f16(_nibble_lut(p[0], 4, IQ4NL_U8_LUT), _nibble_lut(p[1], 4, IQ4NL_U8_LUT), -8388736.0, sc_f))
    if ggml_type == Q8_0:
      wb = hdrs[tj][0] + sub * 17
      sc_f = _half(raw[wb].load().cast(dtypes.uint32))
      q0, q1 = _amd_load(raw[wb + 1 + lg * 4], 4).bitcast(dtypes.uint32), _amd_load(raw[wb + 9 + lg * 4], 4).bitcast(dtypes.uint32)
      return u8x8_to_f16(q0[0] ^ c80, q0[1] ^ c80, -8388736.0, sc_f), u8x8_to_f16(q1[0] ^ c80, q1[1] ^ c80, -8388736.0, sc_f)
    if ggml_type == MXFP4:
      boff, z_u = hdrs[tj][0] + sub * 17, UOp.const(0, dtypes.uint32)
      s01, s23 = UOp.const(0x05000400, dtypes.uint32), UOp.const(0x07000600, dtypes.uint32)
      e = raw[boff].load().cast(dtypes.uint32)
      small = e < 116
      c_lo = small.where(UOp.const(0x26242000, dtypes.uint32), UOp.const(0x36343000, dtypes.uint32))
      c_hi = small.where(UOp.const(0x2e2c2a28, dtypes.uint32), UOp.const(0x3e3c3a38, dtypes.uint32))
      sc2 = _splat_f_h2(_e8m0_halved(e + small.where(UOp.const(7, dtypes.uint32), UOp.const(3, dtypes.uint32))))
      p = _amd_load(raw[boff + 1 + lg * 8], 8).bitcast(dtypes.uint32)
      def mk_mx(sh_m:int) -> UOp:
        return UOp.stack(*(v[k] for j in (0, 1) for pw in ((p[j] >> sh_m) if sh_m else p[j],)
                           for fpb in (_amd_byte_perm(c_hi, c_lo, pw & 0x07070707) | ((pw & 0x08080808) << 4),)
                           for v in (_amd_byte_perm(fpb, z_u, s01).reshape((1,)).bitcast(dtypes.half) * sc2,
                                     _amd_byte_perm(fpb, z_u, s23).reshape((1,)).bitcast(dtypes.half) * sc2) for k in (0, 1)))
      return mk_mx(0), mk_mx(4)
    if ggml_type == Q2_K:
      b32, dr, dmr = hdrs[tj]
      sc16, sh_q2 = (raw_u32[b32 + sp_r].load() >> (m * 16)) & 0xffff, _sh((sub % 4) * 2)
      def mk_q2(h_i:int, sc_b:UOp) -> UOp:
        p = _amd_load(raw_u32[b32 + 4 + (sp_r // 2) * 8 + h_i * 4 + lg * 2], 2)
        return u8x8_to_f16((p[0] >> sh_q2) & 0x03030303, (p[1] >> sh_q2) & 0x03030303,
                           -8388608.0, dr * (sc_b & 15).float(), -dmr * (sc_b >> 4).float())
      return mk_q2(0, sc16 & 255), mk_q2(1, (sc16 >> 8) & 255)
    b, d_f = hdrs[tj] if len(hdrs[tj]) == 2 else (hdrs[tj][0], UOp.const(0.0, dtypes.float32))
    out2 = []
    for h_i in (0, 1):
      wb = b + sub * SMALL_BLOCK_QUANTS[ggml_type] if ggml_type in SMALL_BLOCK_QUANTS else b
      w0, w1 = _quant_word(raw, wb, sub, h_i * 4 + lg * 2, ggml_type, grid), _quant_word(raw, wb, sub, h_i * 4 + lg * 2 + 1, ggml_type, grid)
      if ggml_type == Q6_K:
        out2.append(u8x8_to_f16(w0, w1, z_off, _load_byte(raw, b, 192 + sub * 2 + h_i).cast(dtypes.uint8).bitcast(dtypes.int8).float() * d_f))
      elif ggml_type == Q3_K:
        j = sub * 2 + h_i
        sc_q3 = (((_load_byte(raw, b, 96 + j % 8) >> _sh((j // 8) * 4)) & 15) |
                 (((_load_byte(raw, b, 104 + j % 4) >> _sh((j // 4) * 2)) & 3) << 4)).cast(dtypes.int32) - 32
        out2.append(u8x8_to_f16(w0, w1, z_off, sc_q3.float() * d_f))
      elif ggml_type == IQ1_S:
        sc, zero = _iq1_scale(raw, b, sub)
        out2.append(u8x8_to_f16(w0 ^ c80, w1 ^ c80, -8388736.0, sc, -sc * zero))
      else:
        if ggml_type == IQ3_XXS: sc = d_f * ((_load_u32(raw, b, 66 + sub * 4) >> 28).float() + 0.5) * 0.5
        elif ggml_type == IQ3_S: sc = d_f * (1 + 2 * ((_load_byte(raw, b, 106 + sub // 2) >> _sh((sub % 2) * 4)) & 15)).float()
        elif ggml_type in (IQ2_XS, IQ2_S):
          sc = d_f * (((_load_byte(raw, b, (66 if ggml_type == IQ2_XS else 74) + sub) >> _sh(h_i * 4)) & 15).float() + 0.5) * 0.25
        elif ggml_type == IQ2_XXS: sc = d_f * ((_load_u32(raw, b, 6 + sub * 8) >> 28).float() + 0.5) * 0.25
        elif ggml_type == NVFP4: sc = _ue4m3_halved(_load_byte(raw, b, (sub // 2) * 36 + (sub % 2) * 2 + h_i))
        else: sc = _half(raw[wb].load().cast(dtypes.uint32))
        out2.append(u8x8_to_f16(w0 ^ c80, w1 ^ c80, -8388736.0, sc))
    return out2[0], out2[1]
  sp_r = UOp.range(4, 11)
  k64, prev = sb * 4 + sp_r, acc.after(sb_i, sp_r)
  for m in (0, 1):
    a_pair = [[_amd_load(x_u32[(block_m * bm + ti * 16 + col).minimum(tok - 1) * (in_features // 2) +
                               k64 * 32 + (m * 2 + k2) * 8 + lg * 4], 4).bitcast(dtypes.half)
               for k2 in (0, 1)] for ti in range(tm)]
    for tj in range(tn):
      b0, b1 = get_b_pair(tj, sp_r, m)
      for k2, bf in ((0, b0), (1, b1)):
        for ti in range(tm):
          w = acc[ti, tj].store(UOp.wmma(a_pair[ti][k2], bf, prev[ti, tj], *WMMA_ARG))
          prev = acc.after(w)
  acc_f = acc.after(w.end(sp_r, sb_i))
  red_lds = UOp.alloc((waves_k, tm, tn, 32, 8), dtypes.float32, addrspace=AddrSpace.LOCAL)
  ti_r, tj_r, e_r = UOp.range(tm, 180), UOp.range(tn, 181), UOp.range(8, 182, AxisType.UPCAST)
  r_lds = red_lds.after(red_lds[wk, ti_r, tj_r, lane, e_r].store(acc_f[ti_r, tj_r, e_r].load()).end(e_r, tj_r, ti_r).barrier())
  tm_o, tn_o, e_o = UOp.range(tm, 190), UOp.range(tn, 191), UOp.range(8, 192, AxisType.UPCAST)
  val = sum((r_lds[w_i, tm_o, tn_o, lane, e_o].load() for w_i in range(1, waves_k)), r_lds[0, tm_o, tn_o, lane, e_o].load())
  om, on = block_m * bm + tm_o * 16 + lg * 8 + e_o, block_n * bn + tn_o * 16 + col
  return out[om.valid((om < tok) & wk.eq(0)), on.valid(on < out_features)].store(val.cast(out.dtype)).end(
    e_o, tn_o, tm_o, wk, lane, gid).sink(arg=KernelInfo(name=f"linear_{QUANT_NAMES[ggml_type]}_zlds_f16", opts_to_apply=()))

@functools.cache
def _rdna4_gemm_fp8_kernel(out:UOp, raw:UOp, x:UOp, *grids:UOp, tok:UOp, out_features:UOp, in_features:UOp, ggml_type:int,
                           wm:int=4, wn:int=4, tm:int=2, tn:int=2, double_buf:bool=False, lanes_a:int=4) -> UOp:
  bm, bn, waves, nbuf = wm * tm * 16, wn * tn * 16, wm * wn, 2 if double_buf else 1
  threads, n_u32_a = waves * 32, 64 // lanes_a
  mask_sdst = not double_buf and bm == 128 and bn == 128 and ggml_type in (IQ4_XS, Q8_0, Q4_K, IQ3_XXS, Q3_K, Q2_K, IQ3_S, IQ2_S, MXFP4)
  a_iters, b_iters, m_blks, n_blks = (bm * lanes_a) // threads, (bn * 4) // threads, (tok + bm - 1) // bm, (out_features + bn - 1) // bn
  k_tiles, k_sb = in_features // 128, in_features // 256
  raw_u32 = raw.bitcast(dtypes.uint32) if raw.dtype != dtypes.uint32 else raw
  x_u32, grid = x.flatten().bitcast(dtypes.uint32), grids[0] if grids else None
  gid = UOp.range(m_blks * n_blks, 0, AxisType.GLOBAL)
  swz = (m_blks % 4).eq(0).where(4, (m_blks % 2).eq(0).where(2, 1))
  grp, in_grp = gid // (swz * n_blks), gid % (swz * n_blks)
  block_m, block_n = grp * swz + in_grp % swz, in_grp // swz
  lane, wave = UOp.range(32, -1, AxisType.WARP), UOp.range(waves, 2, AxisType.LOCAL)
  tid, wave_m, wave_n, col, lg = wave * 32 + lane, wave // wn, wave % wn, lane % 16, lane // 16
  A_lds = [UOp.alloc((bm, 8, 2, 8), dtypes.fp8e4m3, addrspace=AddrSpace.LOCAL) for _ in range(nbuf)]
  B_lds = [UOp.alloc((bn, 8, 2, 8), dtypes.fp8e4m3, addrspace=AddrSpace.LOCAL) for _ in range(nbuf)]
  A_u32 = [a.flatten().bitcast(dtypes.uint32).reshape(bm, 8, 4) for a in A_lds]
  B_u32 = [b.flatten().bitcast(dtypes.uint32).reshape(bn, 8, 4) for b in B_lds]
  Asc_lds = [UOp.alloc((bm,), dtypes.float32, addrspace=AddrSpace.LOCAL) for _ in range(nbuf)]
  Bsc_lds = [UOp.alloc((bn,), dtypes.float32, addrspace=AddrSpace.LOCAL) for _ in range(nbuf)]
  acc = UOp.alloc((tm, tn, 8), dtypes.float32, addrspace=AddrSpace.REG)
  sub_acc, eps = UOp.alloc((tm, tn, 8), dtypes.float32, addrspace=AddrSpace.REG), UOp.const(1e-30, dtypes.float32)
  acc = acc.after(acc.store(acc.const_like(0.0)))
  def lds_c(row:UOp, kg:int|UOp, lg_:int|UOp) -> tuple[UOp, UOp]: return row, (kg * 2 + lg_) ^ (row & 7)
  def pin_acc(a:UOp, dep:UOp) -> UOp:
    return UOp.group(*(a[i, j, 7].store(_pin_f32(a.after(dep)[i, j, 7].load(), dep)) for i in range(tm) for j in range(tn)))

  def load_a_reg(kt:UOp, dep:tuple[UOp, ...], rid:int) -> UOp:
    buf, x_s = UOp.alloc((a_iters, n_u32_a), dtypes.uint32, addrspace=AddrSpace.REG), x_u32.after(*dep) if dep else x_u32
    def one(idx:int|UOp) -> list[UOp]:
      ap = tid if (isinstance(idx, int) and idx == 0) else (tid + idx * threads)
      gm, ag = (block_m * bm + ap // lanes_a).minimum(tok - 1), ap % lanes_a
      if n_u32_a == 8:
        a_off = gm * (in_features // 2) + kt * 64 + (ag // 2) * 16 + (ag % 2) * 4
        vs = _v4(_amd_load(x_s[a_off], 4)) + _v4(_amd_load(x_s[a_off + 8], 4))
      else: vs = [v for c in range(4) for v in _v4(_amd_load(x_s[gm * (in_features // 2) + kt * 64 + ag * 16 + c * 4], 4))]
      return [buf.after(*dep)[idx, j].store(vs[j]) for j in range(n_u32_a)]
    return buf.after(UOp.group(*one(0))) if a_iters == 1 else buf.after(UOp.group(*one(ai := UOp.range(a_iters, rid))).end(ai))

  def quant_store_a(bi:int, areg:UOp, dep:tuple[UOp, ...], rid:int) -> UOp:
    dst, sdst = A_u32[bi].after(*dep), Asc_lds[bi].after(*dep)
    def one(idx:int|UOp) -> list[UOp]:
      ap = tid if (isinstance(idx, int) and idx == 0) else (tid + idx * threads)
      am, ag, us = ap // lanes_a, ap % lanes_a, [areg[idx, j].load() for j in range(n_u32_a)]
      lvl = [_pk_max_u16(us[2 * i] & 0x7fff7fff, us[2 * i + 1] & 0x7fff7fff) for i in range(n_u32_a // 2)]
      while len(lvl) > 1: lvl = [_pk_max_u16(lvl[2 * i], lvl[2 * i + 1]) for i in range(len(lvl) // 2)]
      rm = lvl[0]
      for sh in (1, 2, 4):
        if sh < lanes_a: rm = _pk_max_u16(rm, _lane_xor(rm, sh))
      asc = _fast_fmax(_fast_fmax(_half(rm & 0xffff), _half(rm >> 16)) * UOp.const(1.0 / 448.0, dtypes.float32),
                       UOp.const(1.0 / 32768.0, dtypes.float32))
      inv_h2, am_s = _f32_to_h2(_fast_rcp(asc)), am.valid(ag.eq(0)) if mask_sdst else am
      if n_u32_a == 8:
        return [dst[am, ag ^ (am & 7), j].store(_h2x2_to_fp8_pk(us[2 * j], us[2 * j + 1], inv_h2)) for j in range(4)] + [sdst[am_s].store(asc)]
      w8 = [_h2x2_to_fp8_pk(us[2 * j], us[2 * j + 1], inv_h2) for j in range(8)]
      return ([dst[am, (ag * 2 + lg_) ^ (am & 7), j].store(w8[s])
               for lg_, seq in ((0, (0, 1, 4, 5)), (1, (2, 3, 6, 7))) for j, s in enumerate(seq)] + [sdst[am_s].store(asc)])
    return UOp.group(*one(0)) if a_iters == 1 else UOp.group(*one(ai := UOp.range(a_iters, rid))).end(ai)

  def load_b_reg(kt:UOp, dep:tuple[UOp, ...], rid:int) -> UOp:
    ru, rd = raw_u32.after(*dep) if dep else raw_u32, raw.after(*dep) if dep else raw
    def one(bp:UOp) -> list[UOp]:
      gn, bg, gbg = (block_n * bn + bp // 4).minimum(out_features - 1), bp % 4, kt * 4 + bp % 4
      sub, sb = gbg % 8, gn * k_sb + gbg // 8
      if ggml_type == MXFP4: return _v4(_amd_load(ru[(b32 := ((gn * (in_features // 32) + kt * 4) * 17) // 4) + bg * 4], 4)) + [ru[b32 + 16].load()]
      if ggml_type == Q8_0:
        b32, off = (gn * (in_features // 64) + gbg // 2) * 17, (bg & 1) * 9
        return _v4(_amd_load(ru[b32 + off], 4)) + _v4(_amd_load(ru[b32 + off + 4], 4)) + [ru[b32 + 8].load()]
      if ggml_type == Q4_K: return _v4(_amd_load(ru[(b := sb * 36) + 4 + ((gbg // 2) % 4) * 8 + (gbg & 1) * 4], 4)) + [ru[b + bg].load()]
      if ggml_type == Q5_K:
        b, qh2 = sb * 44, _amd_load(ru[sb * 44 + 4 + bg * 2], 2)
        return _v4(_amd_load(ru[b + 12 + ((gbg // 2) % 4) * 8 + (gbg & 1) * 4], 4)) + [qh2[0], qh2[1], ru[b + bg].load()]
      if ggml_type == Q6_K:
        b, s4 = sb * 105, sub // 4
        qh = _amd_load(rd[b + 64 + s4 * 16 + bg * 4], 4).bitcast(dtypes.uint32)
        ql = _v4(_amd_load(rd[b + s4 * 32 + (bg & 1) * 16 + (bg // 2) * 8], 8).bitcast(dtypes.uint32))
        return ql + [qh[0], qh[1], rd[b + 96 + sub].load().cast(dtypes.uint32), rd[b + 104].load().cast(dtypes.uint32)]
      if ggml_type == Q3_K:
        b, s4 = sb * 55, sub // 4
        hm, qs = _amd_load(rd[b + bg * 4], 4).bitcast(dtypes.uint32), _amd_load(rd[b + 16 + s4 * 16 + bg * 4], 4).bitcast(dtypes.uint32)
        return [hm[0], hm[1], qs[0], qs[1], rd[b + 48 + bg].load().cast(dtypes.uint32),
                rd[b + 52 + (bg & 1)].load().cast(dtypes.uint32), rd[b + 54].load().cast(dtypes.uint32)]
      if ggml_type == IQ4_XS: return _v4(_amd_load(ru[(b := sb * 34) + 2 + sub * 4], 4)) + [ru[b + (bg & 1)].load()]
      if ggml_type == Q2_K:
        b32 = sb * 21
        return (_v4(_amd_load(ru[b32 + 4 + (sub // 4) * 8], 4)) + _v4(_amd_load(ru[b32 + 8 + (sub // 4) * 8], 4)) +
                [ru[b32 + sub // 2].load(), ru[b32 + 20].load()])
      if ggml_type == IQ3_S:
        b, q01 = sb * 55, _amd_load(rd[sb * 55 + 1 + sub * 4], 4).bitcast(dtypes.uint32)
        return [q01[0], q01[1], _amd_load(rd[b + 37 + sub * 2], 2).bitcast(dtypes.uint32)[0],
                rd[b + 33 + sub // 2].load().cast(dtypes.uint32), rd[b + 53 + sub // 4].load().cast(dtypes.uint32), rd[b].load().cast(dtypes.uint32)]
      if ggml_type == IQ3_XXS:
        b, q01 = sb * 49, _amd_load(rd[sb * 49 + 1 + sub * 4], 4).bitcast(dtypes.uint32)
        return [q01[0], q01[1], _amd_load(rd[b + 33 + sub * 2], 2).bitcast(dtypes.uint32)[0], rd[b].load().cast(dtypes.uint32)]
      b = sb * 41
      return [_amd_load(rd[b + 1 + sub * 2], 2).bitcast(dtypes.uint32)[0], _amd_load(rd[b + 17 + sub * 2], 2).bitcast(dtypes.uint32)[0],
              rd[b + 33 + sub // 2].load().cast(dtypes.uint32), rd[b + 37 + sub // 2].load().cast(dtypes.uint32), rd[b].load().cast(dtypes.uint32)]
    if b_iters == 1:
      uv = one(tid)
      ub = UOp.alloc((len(uv),), dtypes.uint32, addrspace=AddrSpace.REG)
      return ub.after(UOp.group(*(ub.after(*dep)[j].store(v) for j, v in enumerate(uv))))
    bi = UOp.range(b_iters, rid)
    uv = one(tid + bi * threads)
    ub = UOp.alloc((b_iters, len(uv)), dtypes.uint32, addrspace=AddrSpace.REG)
    return ub.after(UOp.group(*(ub.after(*dep)[bi, j].store(v) for j, v in enumerate(uv))).end(bi))

  def store_b_lds(buf_i:int, breg:UOp, kt:UOp, dep:tuple[UOp, ...], rid:int) -> UOp:
    dst, sdst = B_u32[buf_i].after(*dep), Bsc_lds[buf_i].after(*dep)
    def one(idx:int|UOp|None) -> list[UOp]:
      bp = tid if idx is None else (tid + idx * threads)
      bn_i, bg, base_l = bp // 4, bp % 4, (lane // 4) * 4
      bn_s, gbg, sub = (bn_i.valid(bg.eq(0)) if mask_sdst else bn_i), kt * 4 + bg, (kt * 4 + bg) % 8
      def get_u(j:int) -> UOp: return breg[j].load() if idx is None else breg[idx, j].load()
      def put(w8:list[UOp], sc:UOp) -> list[UOp]:
        return ([dst[bn_i, (bg * 2 + lg_) ^ (bn_i & 7), j].store(w8[s])
                 for lg_, seq in ((0, (0, 1, 4, 5)), (1, (2, 3, 6, 7))) for j, s in enumerate(seq)] + [sdst[bn_s].store(sc)])
      if ggml_type == MXFP4:
        ws03, rem = [get_u(j) for j in range(4)], bg.cast(dtypes.uint32)
        ws = ws03 + [bg.eq(3).where(get_u(4), _lane_bpermute(lane + 1, ws03[0]))]
        e = (ws[0] >> (rem * 8)) & 255
        em = e.maximum(_lane_xor(e, 1))
        em = em.maximum(_lane_xor(em, 2))
        de, c11 = em - e, UOp.const(11, dtypes.uint32)
        ok, d4 = de <= c11, ((c11 - de.minimum(c11)) << 3) * UOp.const(0x01010101, dtypes.uint32)
        l0 = ok.where(UOp.const(0x0c080000, dtypes.uint32) + (d4 & UOp.const(0xffffff00, dtypes.uint32)), 0)
        l1 = ok.where(UOp.const(0x1c181410, dtypes.uint32) + d4, 0)
        ps = [rem.eq(3).where(ws[j + 1], _perm_alignbyte(ws[j + 1], ws[j], (rem + 1) & 3)) for j in range(4)]
        return put([_amd_byte_perm(l1, l0, ((ps[j] >> sh) if sh else ps[j]) & 0x07070707) | ((((ps[j] >> sh) if sh else ps[j]) & 0x08080808) << 4)
                    for sh in (0, 4) for j in range(4)], _e8m0_halved(em) * UOp.const(1.0 / 16.0, dtypes.float32))
      if ggml_type == Q8_0:
        us, ext, is_e = [get_u(j) for j in range(8)], get_u(8), (bg & 1).eq(0)
        sc = _half(is_e.where(us[0] & 0xffff, ext >> 16))
        sm = _fast_fmax(sc.abs(), _lane_xor(sc.abs(), 1))
        tsc = _fast_fmax(_fast_fmax(sm, _lane_xor(sm, 2)), eps)
        sn, ue, c80 = sc * _fast_rcp(tsc), us + [ext], UOp.const(0x80808080, dtypes.uint32)
        sel_a = is_e.where(UOp.const(0x05040302, dtypes.uint32), UOp.const(0x03020100, dtypes.uint32))
        return put([_b4_to_fp8(_amd_byte_perm(ue[j + 1], ue[j], sel_a) ^ c80, sn, -8388736.0) for j in range(8)], tsc)
      if ggml_type in (Q4_K, Q5_K):
        qs, hdr = [get_u(j) for j in range(4)], [_lane_bpermute(base_l + k, get_u(4 if ggml_type == Q4_K else 6)) for k in range(4)]
        tsc = _fast_fmax(_fast_fmax(_half(hdr[0] & 0xffff).abs() * UOp.const(4.0 if ggml_type == Q4_K else 8.0, dtypes.float32),
                                    _half(hdr[0] >> 16).abs() * UOp.const(0.25, dtypes.float32)), eps)
        inv, is_lo, (ds0, dm0, ds1, dm1) = _fast_rcp(tsc), (gbg & 1).eq(0), _q4k_pair_scales(hdr, (gbg // 2) % 4)
        ds_m, dm_m, qs_o = is_lo.where(ds0, ds1) * inv, is_lo.where(-dm0, -dm1) * inv, [_lane_bpermute(lane ^ 1, qs[j]) for j in range(4)]
        p_all = [is_lo.where(qs[j], qs_o[j]) for j in range(4)] + [is_lo.where(qs_o[j], qs[j]) for j in range(4)]
        sh = is_lo.where(UOp.const(0, dtypes.uint32), UOp.const(4, dtypes.uint32))
        if ggml_type == Q4_K:
          lut = tuple(_pack4_fp8(*(UOp.const(float(k + i), dtypes.float32) * ds_m + dm_m for i in range(4))) for k in (0, 4, 8, 12))
          return put([_nibble_lut(p_all[j], sh, lut) for j in range(8)], tsc)
        qh = [v for k in range(4) for v in (_lane_bpermute(base_l + k, get_u(4)), _lane_bpermute(base_l + k, get_u(5)))]
        return put([_b4_to_fp8(((p_all[j] >> sh) & 0x0f0f0f0f) | (((qh[j] >> _sh(sub)) & 0x01010101) << 4), ds_m, -8388608.0, dm_m)
                    for j in range(8)], tsc)
      if ggml_type == Q6_K:
        ql_m, sc_w, is_l2 = [get_u(j) for j in range(4)], get_u(6), bg < 2
        ql_o = [_lane_bpermute(lane ^ 2, ql_m[j]) for j in range(4)]
        ql = [is_l2.where(ql_m[j], ql_o[j]) for j in range(4)] + [is_l2.where(ql_o[j], ql_m[j]) for j in range(4)]
        qh = [v for k in range(4) for v in (_lane_bpermute(base_l + k, get_u(4)), _lane_bpermute(base_l + k, get_u(5)))]
        s0 = (sc_w & 255).cast(dtypes.uint8).bitcast(dtypes.int8).float() * (1.0 / 16.0)
        s1 = ((sc_w >> 8) & 255).cast(dtypes.uint8).bitcast(dtypes.int8).float() * (1.0 / 16.0)
        ql_sh, qh_sh = _sh((sub % 4 // 2) * 4), _sh((sub % 4) * 2)
        return put([_b4_to_fp8(((ql[j] >> ql_sh) & 0x0f0f0f0f) | (((qh[j] >> qh_sh) & 0x03030303) << 4), s0 if j < 4 else s1, -8388640.0)
                    for j in range(8)], _half(get_u(7)) * 16.0)
      if ggml_type == Q3_K:
        hm = [v for k in range(4) for v in (_lane_bpermute(base_l + k, get_u(0)), _lane_bpermute(base_l + k, get_u(1)))]
        qs = [v for k in range(4) for v in (_lane_bpermute(base_l + k, get_u(2)), _lane_bpermute(base_l + k, get_u(3)))]
        sl16, sh16 = get_u(4), get_u(5)
        q3_scs = [((((sl16 >> _sh(h * 8 + (sub // 4) * 4)) & 15) |
                    (((sh16 >> _sh(h * 8 + (sub // 2) * 2)) & 3) << 4)).cast(dtypes.int32) - 32).float() * 2.0 for h in (0, 1)]
        luts = [(_pack4_fp8(*(float(v) * sc for v in (-4, -3, -2, -1))), _pack4_fp8(*(float(v) * sc for v in (0, 1, 2, 3)))) for sc in q3_scs]
        qs_sh, hm_sh = _sh((sub % 4) * 2), _sh(sub)
        return put([_amd_byte_perm(luts[j // 4][1], luts[j // 4][0], ((qs[j] >> qs_sh) & 0x03030303) | (((hm[j] >> hm_sh) & 0x01010101) << 2))
                    for j in range(8)], _half(get_u(6)) * 0.5)
      if ggml_type == IQ4_XS:
        ps, h0, h1 = [get_u(j) for j in range(4)], _lane_bpermute(base_l, get_u(4)), _lane_bpermute(base_l + 1, get_u(4))
        sc_i = (((h1 >> _sh(4 * sub)) & 15) | (((h0 >> _sh(16 + 2 * sub)) & 3) << 4)).cast(dtypes.int32) - 32
        sc = sc_i.float() * UOp.const(3.0 / 32.0, dtypes.float32)
        lut = tuple(_pack4_fp8(*(UOp.const(float(v), dtypes.float32) * sc for v in _IQ4NL_VALS[k:k + 4])) for k in (0, 4, 8, 12))
        return put([_nibble_lut(ps[j], sh, lut) for sh in (0, 4) for j in range(4)], _half(h0) * UOp.const(32.0 / 3.0, dtypes.float32))
      if ggml_type == Q2_K:
        qs, sc16, dm_w = [get_u(j) for j in range(8)], (get_u(8) >> _sh((sub & 1) * 16)) & 0xffff, get_u(9)
        dr, dmr = _half(dm_w & 0xffff), _half(dm_w >> 16)
        tsc = _fast_fmax(_fast_fmax(dr.abs(), dmr.abs()), eps) * 0.25
        inv, sh, z_u = _fast_rcp(tsc), _sh((sub % 4) * 2), UOp.const(0, dtypes.uint32)
        luts1 = [_pack4_fp8(*(float(i) * dr * inv * (sc & 15).float() - dmr * inv * (sc >> 4).float() for i in range(4)))
                 for sc in (sc16 & 255, (sc16 >> 8) & 255)]
        return put([_amd_byte_perm(z_u, luts1[j // 4], (qs[j] >> sh) & 0x03030303) for j in range(8)], tsc)
      assert grid is not None
      if ggml_type == IQ3_S:
        q0, q1, s0 = get_u(0), get_u(1), get_u(2)
        qh_s, sc_n = (get_u(3) >> _sh((sub & 1) * 8)) & 255, (get_u(4) >> _sh((sub & 3) * 4)) & 15
        sn = (1.0 + 2.0 * sc_n.float()) * (1.0 / 16.0)
        l0, l1 = _pack4_fp8(*(float(v) * sn for v in (1, 3, 5, 7))), _pack4_fp8(*(float(v) * sn for v in (9, 11, 13, 15)))
        w8 = [_amd_byte_perm(l1, l0, (grid[(((q0 if i < 4 else q1) >> ((i & 3) * 8)) & 255) | (((qh_s >> i) & 1) << 8)].load() >> 1) & 0x07070707)
              ^ (((((s0 >> (i * 4)) & 15) * 0x00204081) & 0x01010101) << 7) for i in range(8)]
        return put(w8, _half(get_u(5)) * 16.0)
      if ggml_type == IQ3_XXS:
        q0, q1, sc_w = get_u(0), get_u(1), get_u(2)
        sn = ((sc_w >> 28).float() + 0.5) * 0.25
        l0, l1 = _pack4_fp8(*(float(v) * sn for v in (4, 12, 20, 28))), _pack4_fp8(*(float(v) * sn for v in (36, 44, 52, 62)))
        w8 = [_amd_byte_perm(l1, l0, (grid[((q0 if p < 2 else q1) >> ((p & 1) * 16 + k * 8)) & 255].load() >> 3) & 0x07070707)
              ^ (((((_iq_even_signs((sc_w >> (7 * p)) & 127) >> (k * 4)) & 15) * 0x00204081) & 0x01010101) << 7) for p in range(4) for k in (0, 1)]
        return put(w8, _half(get_u(3)) * 2.0)
      q0, s0, bsh = get_u(0), get_u(1), _sh((sub & 1) * 8)
      qh_2, sc_b, z_u = (get_u(2) >> bsh) & 255, (get_u(3) >> bsh) & 255, UOp.const(0, dtypes.uint32)
      luts2 = [_pack4_fp8(8.0 * sf, 25.0 * sf, 43.0 * sf, UOp.const(0.0, dtypes.float32))
               for k in (0, 1) for sf in ((((sc_b >> (k * 4)) & 15).float() + 0.5) * 0.5,)]
      w8 = [_amd_byte_perm(z_u, luts2[p // 2], (g01[k] >> 4) & 0x03030303)
            ^ ((((((s0 >> (8 * p)) >> (k * 4)) & 15) * 0x00204081) & 0x01010101) << 7)
            for p in range(4) for g01 in (_amd_load(grid[(((q0 >> (8 * p)) & 255) | (((qh_2 >> (2 * p)) & 3) << 8)) * 2], 2),) for k in (0, 1)]
      return put(w8, _half(get_u(4)) * 0.5)
    return UOp.group(*one(None)) if b_iters == 1 else UOp.group(*one(bi := UOp.range(b_iters, rid))).end(bi)

  def compute_slice(A_r:UOp, B_r:UOp, sa_in:UOp, kg0:int, rid:int) -> UOp:
    kg_r = UOp.range(2, rid)
    kg, prev = (kg_r + kg0) if kg0 else kg_r, sa_in.after(kg_r)
    for k in (0, 1):
      tm_r, tn_r = UOp.range(tm, rid + 1 + 2 * k), UOp.range(tn, rid + 2 + 2 * k)
      w = sub_acc[tm_r, tn_r].store(UOp.wmma(A_r[*lds_c((wave_m * tm + tm_r) * 16 + col, kg, lg), k],
                                             B_r[*lds_c((wave_n * tn + tn_r) * 16 + col, kg, lg), k], prev[tm_r, tn_r], *WMMA_ARG)).end(tn_r, tm_r)
      prev = sub_acc.after(w)
    return w.end(kg_r)

  def flush_sub(Asc_r:UOp, Bsc_r:UOp, acc_in:UOp, sa_in:UOp, rid:int) -> UOp:
    tm_f, tn_f, e_f = UOp.range(tm, rid), UOp.range(tn, rid + 1), UOp.range(8, rid + 2, AxisType.UPCAST)
    return acc[tm_f, tn_f, e_f].store(acc_in[tm_f, tn_f, e_f].load() + sa_in[tm_f, tn_f, e_f].load() * (
      Asc_r[(wave_m * tm + tm_f) * 16 + lg * 8 + e_f].load() * Bsc_r[(wave_n * tn + tn_f) * 16 + col].load())).end(e_f, tn_f, tm_f)

  zw = UOp.const(0, dtypes.weakint)
  st0 = UOp.group(quant_store_a(0, load_a_reg(zw, (), 22), (), 26), store_b_lds(0, load_b_reg(zw, (), 24), zw, (), 28)).barrier()
  if not double_buf:
    kt = UOp.range(k_tiles - 1, 10)
    ar_n, br_n = load_a_reg(kt + 1, (st0, kt), 32), load_b_reg(kt + 1, (st0, kt), 34)
    sa0 = sub_acc.after(sub_acc.after(st0, kt, ar_n, br_n).store(sub_acc.const_like(0.0)))
    ca = compute_slice(A_lds[0].after(st0, kt, ar_n, br_n), B_lds[0].after(st0, kt, ar_n, br_n), sa0, 0, 100)
    pa = pin_acc(sub_acc, ca)
    comp = compute_slice(A_lds[0].after(st0, kt, pa), B_lds[0].after(st0, kt, pa), sub_acc.after(ca, pa), 2, 110)
    fl_k = flush_sub(Asc_lds[0].after(st0, kt, comp), Bsc_lds[0].after(st0, kt, comp), acc.after(st0, kt), sub_acc.after(comp), 120)
    mid = UOp.group(ar_n, br_n, pin_acc(acc, fl_k)).barrier()
    st_n = UOp.group(quant_store_a(0, ar_n, (mid,), 46), store_b_lds(0, br_n, kt + 1, (mid,), 48)).barrier().end(kt)
    sa_e = sub_acc.after(sub_acc.after(st0, st_n).store(sub_acc.const_like(0.0)))
    ca_e = compute_slice(A_lds[0].after(st0, st_n), B_lds[0].after(st0, st_n), sa_e, 0, 140)
    pa_e = pin_acc(sub_acc, ca_e)
    comp_e = compute_slice(A_lds[0].after(st0, st_n, pa_e), B_lds[0].after(st0, st_n, pa_e), sub_acc.after(ca_e, pa_e), 2, 150)
    fl_e = flush_sub(Asc_lds[0].after(st0, st_n, comp_e), Bsc_lds[0].after(st0, st_n, comp_e), acc.after(st0, st_n), sub_acc.after(comp_e), 160)
    acc_f = acc.after(fl_e)
  else:
    def step_h(bc:int, bn_i:int, kn:UOp, deps:tuple[UOp, ...], rb:int) -> UOp:
      ar, br = load_a_reg(kn, deps, rb), load_b_reg(kn, deps, rb + 2)
      sa0 = sub_acc.after(sub_acc.after(*deps, ar, br).store(sub_acc.const_like(0.0)))
      ca = compute_slice(A_lds[bc].after(*deps, ar, br), B_lds[bc].after(*deps, ar, br), sa0, 0, rb + 10)
      pa = pin_acc(sub_acc, ca)
      cb = compute_slice(A_lds[bc].after(*deps, pa), B_lds[bc].after(*deps, pa), sub_acc.after(ca, pa), 2, rb + 16)
      fl = flush_sub(Asc_lds[bc].after(*deps, cb), Bsc_lds[bc].after(*deps, cb), acc.after(*deps, cb), sub_acc.after(cb), rb + 30)
      return UOp.group(pin_acc(acc, fl), store_b_lds(bn_i, br, kn, (*deps, pa), rb + 6), quant_store_a(bn_i, ar, (*deps, pa), rb + 4)).barrier()
    pair = UOp.range(k_tiles // 2, 10)
    acc_f = acc.after(step_h(1, 0, (pair + 1).minimum(k_tiles // 2 - 1) * 2, (step_h(0, 1, pair * 2 + 1, (st0, pair), 30),), 70).end(pair))
  tm_o, tn_o, e_o = UOp.range(tm, 190), UOp.range(tn, 191), UOp.range(8, 192, AxisType.UPCAST)
  om, on = block_m * bm + (wave_m * tm + tm_o) * 16 + lg * 8 + e_o, block_n * bn + (wave_n * tn + tn_o) * 16 + col
  return out[om.valid(om < tok), on.valid(on < out_features)].store(acc_f[tm_o, tn_o, e_o].load().cast(out.dtype)).end(
    e_o, tn_o, tm_o, wave, lane, gid).sink(arg=KernelInfo(name=f"linear_{QUANT_NAMES[ggml_type]}_fp8_wmma", opts_to_apply=()))
