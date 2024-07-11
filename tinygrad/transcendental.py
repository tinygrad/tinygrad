import math
from typing import Tuple, List
from tinygrad.dtype import dtypes, DType
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps
from tinygrad.lazy import LazyBuffer

def is_dtype_transcendental_supported(d:DType):
  return d in [dtypes.float16, dtypes.float32, dtypes.float64]

def _lazy_map_numbers(x:LazyBuffer, inf:LazyBuffer, _inf:LazyBuffer, nan:LazyBuffer, ratio:LazyBuffer):
  """replace inf -> inf, -inf -> _inf, nan -> nan, otherwise -> ratio"""
  return x.e(BinaryOps.CMPNE, x.const(math.inf)).e(TernaryOps.WHERE, x.e(BinaryOps.CMPNE, x).e(TernaryOps.WHERE, nan, x.e(BinaryOps.CMPNE, x.const(-math.inf)).e(TernaryOps.WHERE, ratio, _inf)), inf) # noqa: E501
# *** helper functions for double/quad precision arithmetics ***
def dfadd2_f2_f2_f2(xx:LazyBuffer, xy:LazyBuffer, yx:LazyBuffer, yy:LazyBuffer) -> Tuple[LazyBuffer, LazyBuffer]:
  rx = xx.e(BinaryOps.ADD, yx)
  v = rx.e(BinaryOps.ADD, xx.e(UnaryOps.NEG))
  ry = xx.e(BinaryOps.ADD, rx.e(BinaryOps.ADD, v.e(UnaryOps.NEG)).e(UnaryOps.NEG)).e(BinaryOps.ADD, yx.e(BinaryOps.ADD, v.e(UnaryOps.NEG)))
  ry = xy.e(BinaryOps.ADD, yy)
  return rx, ry

def dfmul2_f2_f2_f2(xx:LazyBuffer, xy:LazyBuffer, yx:LazyBuffer, yy:LazyBuffer) -> Tuple[LazyBuffer, LazyBuffer]:
  rx = xx.e(BinaryOps.MUL, yx)
  ry = xx.e(BinaryOps.MUL, yx).e(BinaryOps.ADD, rx.e(UnaryOps.NEG)).e(BinaryOps.ADD, xx.e(BinaryOps.MUL, yy)).e(BinaryOps.ADD, xy.e(BinaryOps.MUL, yx)) # noqa: E501
  return rx, ry

def dfdiv2_f2_f2_f2(nx:LazyBuffer, ny:LazyBuffer, dx:LazyBuffer, dy:LazyBuffer) -> Tuple[LazyBuffer, LazyBuffer]:
  t = dx.e(UnaryOps.RECIP)
  qx = nx.e(BinaryOps.MUL, t)
  u = qx.e(UnaryOps.NEG).e(BinaryOps.ADD, nx.e(BinaryOps.MUL, t)).e(BinaryOps.ADD, qx.e(BinaryOps.MUL, dx.const(1).e(BinaryOps.ADD, dx.e(BinaryOps.MUL, t).e(UnaryOps.NEG)))) # noqa: E501
  qy = t.e(BinaryOps.MUL, ny.e(BinaryOps.ADD, qx.e(BinaryOps.MUL, dy).e(UnaryOps.NEG))).e(BinaryOps.ADD, u)
  return qx, qy
# *** helper functions for bit manipulation ***
def significand_bits(d:DType) -> int:
  assert is_dtype_transcendental_supported(d)
  return {dtypes.float64: 52, dtypes.float32: 23, dtypes.float16: 10}[d]

def exponent_bias(d:DType) -> int:
  assert is_dtype_transcendental_supported(d)
  return {dtypes.float64: 1022, dtypes.float32: 126, dtypes.float16: 14}[d]

def exponent_mask(d:DType) -> int:
  assert is_dtype_transcendental_supported(d)
  return {dtypes.float64: 0x7FF, dtypes.float32: 0xFF, dtypes.float16: 0x1F}[d]

def float_to_bits(d:LazyBuffer) -> LazyBuffer:
  assert is_dtype_transcendental_supported(d.dtype)
  cast_to = {dtypes.float64: dtypes.uint64, dtypes.float32: dtypes.uint32, dtypes.float16: dtypes.uint16}[d.dtype]
  return d.cast(cast_to, True, allow_buffer_view=False)

def bits_to_float(d:LazyBuffer, float_dtype:DType) -> LazyBuffer:
  assert d.dtype in [dtypes.uint64, dtypes.uint32, dtypes.uint16]
  cast_to = {dtypes.uint64: dtypes.float64, dtypes.uint32: dtypes.float32, dtypes.uint16: float_dtype}[d.dtype]
  return d.cast(cast_to, True, allow_buffer_view=False)
# **** utils ****
def shr(x:LazyBuffer, y:int) -> LazyBuffer: return x.e(BinaryOps.IDIV, x.const(2**y))
def shl(x:LazyBuffer, y:int) -> LazyBuffer: return x.e(BinaryOps.MUL, x.const(2**y))

def rintk(d:LazyBuffer) -> LazyBuffer:
  """ceiling(d:float) -> int"""
  assert is_dtype_transcendental_supported(d.dtype)
  return_t = {dtypes.float64: dtypes.int64, dtypes.float32: dtypes.int32, dtypes.float16: dtypes.int16}[d.dtype]
  return d.e(BinaryOps.ADD, d.e(BinaryOps.CMPLT, d.const(0.0)).e(TernaryOps.WHERE, d.const(-0.5), d.const(0.5))).cast(return_t)

def pow2if(q:LazyBuffer, float_dtype:DType):
  """cast(2^q, float_dtype) where q is any integer in the range of [-126, 127]"""
  assert q.dtype in (dtypes.int64, dtypes.int32, dtypes.int16, dtypes.uint32)
  final_dtype = {dtypes.int64: dtypes.float64, dtypes.int32: dtypes.float32, dtypes.int16: float_dtype, dtypes.uint32: dtypes.float32}[q.dtype]
  return shl(q.e(BinaryOps.ADD, q.const(exponent_bias(final_dtype)+1)), significand_bits(final_dtype)).cast(final_dtype, True, allow_buffer_view=False) # noqa: E501

def ilogb2k(d:LazyBuffer) -> LazyBuffer:
  """calculate the integer part of log2(d), where d is normalized fp value in the range of [0, +inf)."""
  assert is_dtype_transcendental_supported(d.dtype)
  dint = d.cast({dtypes.float64: dtypes.int64, dtypes.float32: dtypes.int32, dtypes.float16: dtypes.int16}[d.dtype], True, allow_buffer_view=False)
  # -1 <= ilog2bk(d) <= 128
  # ((float_to_bits(d) >> significand_bits(dtype)) & exponent_mask(dtype)) - exponent_bias(dtype)
  return shr(dint, significand_bits(d.dtype)).e(BinaryOps.AND, dint.const(exponent_mask(d.dtype))).e(BinaryOps.ADD, dint.const(-(exponent_bias(d.dtype)+1))) # noqa: E501

def ldexp3k(d:LazyBuffer, e:LazyBuffer) -> LazyBuffer:
  """d*2^e. e is a number obtained by casting an integer in the range [-127, 127] to a float. d is any float number."""
  assert is_dtype_transcendental_supported(d.dtype) and is_dtype_transcendental_supported(e.dtype)
  dtype = d.dtype
  cast_map = {dtypes.float64: dtypes.int64, dtypes.float32: dtypes.int32, dtypes.float16: dtypes.int16}
  e = e.cast(cast_map[d.dtype])
  m1 = d.cast(cast_map[d.dtype], True, allow_buffer_view=False)
  m2 = shl(e, significand_bits(d.dtype))
  return m1.e(BinaryOps.ADD, m2).cast(d.dtype, True, allow_buffer_view=False).cast(dtype)

def ldexp2k(d:LazyBuffer, e:LazyBuffer) -> LazyBuffer:
  """d*2^e. much faster than ldexp3k but risky. d > 0 and d is not denormal."""
  assert is_dtype_transcendental_supported(d.dtype) and e.dtype in (dtypes.int16, dtypes.int32, dtypes.int64)
  return d.e(BinaryOps.MUL, pow2if(shr(e, 1), d.dtype)).e(BinaryOps.MUL, pow2if(e.e(BinaryOps.ADD, shr(e, 1).e(UnaryOps.NEG)), d.dtype)) # noqa: E501

def frexp(v:LazyBuffer) -> Tuple[LazyBuffer, LazyBuffer]:
  """frexp(v) -> (mantissa, exponent)"""
  assert is_dtype_transcendental_supported(v.dtype)
  # m1 = masks for mantissa, m2 = masks to normalize the mantissa.
  m1 = {dtypes.float64: 0x800FFFFF, dtypes.float32: 0x807FFFFF, dtypes.float16: 0x83FF}[v.dtype]
  m2 = {dtypes.float64: 0x3FE0000000000000, dtypes.float32: 0x3F000000, dtypes.float16: 0x3C00}[v.dtype]
  bias = {dtypes.float64: 1022, dtypes.float32: 126, dtypes.float16: 15}[v.dtype]
  bits = float_to_bits(v)
  exponent = shr(bits, significand_bits(v.dtype)).e(BinaryOps.AND, bits.const(exponent_mask(v.dtype)))
  exponent_zero = exponent.e(BinaryOps.CMPNE, exponent.const(0.0))
  result_f = bits_to_float(bits.e(BinaryOps.AND, bits.const(m1)).e(BinaryOps.OR, bits.const(m2)), v.dtype)
  value = exponent_zero.e(TernaryOps.WHERE, result_f, v)
  exp = exponent.e(BinaryOps.ADD, exponent.const(-bias))
  exp = exponent_zero.e(TernaryOps.WHERE, exp, exp.const(0))
  if v.dtype == dtypes.float16:
    exp = exp.cast(dtypes.int16, True, allow_buffer_view=False)
  return value, exp

def mla(x:LazyBuffer, y:LazyBuffer, z:LazyBuffer) -> LazyBuffer:
  """x*y+z"""
  return x.e(BinaryOps.MUL, y).e(BinaryOps.ADD, z)

def polyN(u:LazyBuffer, s:LazyBuffer, coeffs:List[float]) -> LazyBuffer:
  for c in coeffs:
    u = mla(u, s, u.const(c))
  return u
# *** reduction algorithms for sine ***
def payne_hanek_reduction(d:LazyBuffer) -> Tuple[LazyBuffer, LazyBuffer]:
  """
  Performs Payne-Hanek Reduction: computes the remainder of `d` modulo pi/2 for the values `d` where
    39800.0 <= d <= +Inf
  Returns a tuple of `(r, q)`:
  - `r`[d.dtype] is the reminder value corresponding to `round_to_nearest(x % pi/2)`.
    ensuring that `r` is in the range of [0, pi/2).
  - `q`[int32] is an integer taking values 0,1,2 or 3, corresponding to the quadrant of the original angle `d`.
  """
  assert is_dtype_transcendental_supported(d.dtype)
  two_over_pi_f = [0x00000000,0x28be60db,0x9391054a,0x7f09d5f4,0x7d4d3770,0x36d8a566,0x4f10e410]

  input_dtype: DType = d.dtype
  dtype_via = dtypes.float32 if d.dtype == dtypes.float16 else d.dtype
  acc_dtype = dtypes.uint64

  f, e = frexp(d)
  ia = (k := f.cast(dtype_via)).e(BinaryOps.MUL, k.const(4.294967296e9)).cast(dtypes.uint64)
  i = shr(e.cast(dtypes.uint64), 5)
  e = (k := e.cast(dtypes.uint64)).e(BinaryOps.AND, k.const(31)).cast(dtypes.uint32)
  offset = e.const(32).e(BinaryOps.ADD, e.e(UnaryOps.NEG))

  def _eq(arr:LazyBuffer, eq_to:int) -> LazyBuffer: return arr.e(BinaryOps.CMPNE, arr.const(eq_to))
  def _take(an:LazyBuffer, offset:int, count:int=0) -> LazyBuffer:
    """an = two_over_pi_f[i+offset]"""
    if count+offset <= len(two_over_pi_f[0:-2]):
      an = _eq(i, count).e(TernaryOps.WHERE, _take(an, offset, count=count+1), an.const(two_over_pi_f[count+offset]))
    return an
  def _exact_pow2if(x): return pow2if(x, d.dtype).cast(acc_dtype)
  def _shl_lazy(x, y): return x.cast(acc_dtype).e(BinaryOps.MUL, _exact_pow2if(y)).cast(dtypes.uint32)
  def _shr_lazy(x, y): return x.cast(acc_dtype).e(BinaryOps.IDIV, _exact_pow2if(y)).cast(dtypes.uint32)
  # a_n = (two_over_pi_f[Int(i) + n] << e) | (two_over_pi_f[Int(i) + n+1] >> (nbits - e))
  a1 = _take(i.const(0).cast(dtypes.uint32), 0)
  a2 = _take(i.const(0).cast(dtypes.uint32), 1)
  a3 = _take(i.const(0).cast(dtypes.uint32), 2)
  a4 = _take(i.const(0).cast(dtypes.uint32), 3)
  # Note: e >= 1 for all numbers d >= 1.0. assume e != 0
  hi = _shl_lazy(a1, e).e(BinaryOps.OR, _shr_lazy(a2, offset))
  mi = _shl_lazy(a2, e).e(BinaryOps.OR, _shr_lazy(a3, offset))
  lo = _shl_lazy(a3, e).e(BinaryOps.OR, _shr_lazy(a4, offset))

  def _hp_mul(x:LazyBuffer, y:LazyBuffer) -> LazyBuffer: return x.cast(dtypes.uint64).e(BinaryOps.MUL, y.cast(dtypes.uint64))
  p = _hp_mul(ia, lo)
  p = _hp_mul(ia, mi).e(BinaryOps.ADD, shr(p, 32))
  p = shl(_hp_mul(ia, hi), 32).e(BinaryOps.ADD, p)

  q = shr(p, 62).cast(dtypes.int32)
  p = p.e(BinaryOps.AND, p.const(0x3fffffffffffffff))

  d = p.cast(dtype_via)
  d = d.e(BinaryOps.MUL, d.const(3.4061215800865545e-19))
  r = d.cast(input_dtype)

  fraction_map = f.e(BinaryOps.CMPLT, f.const(0.5))
  # if fraction >= 0.5, r -= pi/2, q += 1
  r = fraction_map.e(TernaryOps.WHERE, r, r.e(BinaryOps.ADD, r.const(-math.pi / 2)))
  q = fraction_map.e(TernaryOps.WHERE, q, q.e(BinaryOps.ADD, q.const(1)))
  return r, q

def cody_waite_reduction(d:LazyBuffer) -> Tuple[LazyBuffer, LazyBuffer]:
  """
  Performs Cody-Waite Reduction: computes the reminder of `d` modulo pi/2 for the values `d` where
      0 <= abs(d) <= 39800.0
  Returns a tuple of `(r, q)`, where the output format is the same as that of `payne_hanek_reduction`.
  """
  m_1_pi = 0.318309886183790671537767526745028724
  qdh = d.e(BinaryOps.MUL, d.const(m_1_pi / 16777216)).cast(dtypes.int64).cast(d.dtype).e(BinaryOps.MUL, d.const(16777216.0))
  def _quadrant(x:LazyBuffer) -> LazyBuffer:
    if x.dtype == dtypes.float64:
      return rintk(mla(d, d.const(m_1_pi), qdh.e(UnaryOps.NEG))).cast(x.dtype)
    return rintk(x.e(BinaryOps.MUL, d.const(m_1_pi))).cast(x.dtype)
  def _reduce_d(x:LazyBuffer, q:LazyBuffer):
    if x.dtype == dtypes.float64:
      d = mla(qdh, x.const(-3.1415926218032836914), x)
      d = mla(q, x.const(-3.1415926218032836914), d)
      d = mla(qdh, x.const(-3.1786509424591713469e-08), d)
      d = mla(q, x.const(-3.1786509424591713469e-08), d)
      d = mla(qdh, x.const(-1.2246467864107188502e-16), d)
      d = mla(q, x.const(-1.2246467864107188502e-16), d)
      d = mla(qdh.e(BinaryOps.ADD, q), x.const(-1.2736634327021899816e-24), d)
    elif x.dtype == dtypes.float16:
      # when reducing `d`, FP16 needs FP32 precision to achieve 1.0 ULP precision.
      d = _reduce_d(x.cast(dtypes.float32), q.cast(dtypes.float32)).cast(dtypes.float16)
    else:
      d = mla(q, x.const(-3.1414794921875), x)
      d = mla(q, x.const(-0.00011315941810607910156), d)
      d = mla(q, x.const(-1.9841872589410058936e-09), d)
      d = mla(q, x.const(-1.2154201256553420762e-10), d)
    return d
  return _reduce_d(d, (q := _quadrant(d))), q.cast(dtypes.int32)
# *** approximate sine on small angle. ***
def trig_poly(d:LazyBuffer, coeff32, coeff64):
  u = None
  s = d.e(BinaryOps.MUL, d)
  if d.dtype == dtypes.float64:
    s2 = s.e(BinaryOps.MUL, s)
    s4 = s2.e(BinaryOps.MUL, s2)
    def __poly4(x: LazyBuffer, x2: LazyBuffer, c3, c2, c1, c0) -> LazyBuffer: return mla(x2, mla(x, x.const(c3), x.const(c2)), mla(x, x.const(c1), x.const(c0))) # noqa: E501
    def __poly8(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0) -> LazyBuffer: return mla(x4, __poly4(x, x2, c7, c6, c5, c4), __poly4(x, x2, c3, c2, c1, c0)) # noqa: E501
    u = __poly8(s, s2, s4, *coeff64[:-1])
    u = mla(u, s, d.const(coeff64[-1]))
  else:
    u = polyN(s.const(coeff32[0]), s, coeff32[1:])
  return mla(s, u.e(BinaryOps.MUL, d), d)
# approximate sine on [-pi/2, pi/2]
def sin_poly(d:LazyBuffer) -> LazyBuffer: return trig_poly(d, [2.6083159809786593541503e-06, -0.0001981069071916863322258, 0.00833307858556509017944336, -0.166666597127914428710938], [-7.97255955009037868891952e-18, 2.81009972710863200091251e-15, -7.64712219118158833288484e-13, 1.60590430605664501629054e-10, -2.50521083763502045810755e-08, 2.75573192239198747630416e-06, -0.000198412698412696162806809, 0.00833333333333332974823815, -0.166666666666666657414808]) # noqa: E501

def sin_poly_small(d:LazyBuffer, q:LazyBuffer) -> LazyBuffer:
  def _ifand(n: int): return q.e(BinaryOps.AND, q.const(n)).e(BinaryOps.CMPNE, q.const(0))
  r = sin_poly(d)
  return r.e(BinaryOps.MUL, _ifand(1).e(TernaryOps.WHERE, r.const(-1), r.const(1)))

def sin_poly_large(d:LazyBuffer, q:LazyBuffer) -> LazyBuffer:
  def _ifand(n: int): return q.e(BinaryOps.AND, q.const(n)).e(BinaryOps.CMPNE, q.const(0))
  d = d.e(BinaryOps.ADD, _ifand(1).e(TernaryOps.WHERE, d.const(math.pi / 2), d.const(0)))
  r = sin_poly(d)
  return r.e(BinaryOps.MUL, _ifand(2).e(TernaryOps.WHERE, r.const(-1), r.const(1)))
# *** toplevel functions for xsin/xlog2/xexp2 ***
def xsin(d:LazyBuffer, fast:bool=False, switch_over:float=39800.0) -> LazyBuffer:
  """
  Implements a 1.0 ULP approximation for UnaryOps.SIN.
  - fast=True assumes x <= switch_over.
  - switch_over is the threshold for switching to payne_hanek_reduction.
  """
  assert is_dtype_transcendental_supported(d.dtype)
  if 0 in d.shape: return d
  if d.dtype == dtypes.float16:
    switch_over = 9500.0
  reduction_algo = cody_waite_reduction if fast else payne_hanek_reduction
  # mask +-inf/nan as zero
  x = _lazy_map_numbers(d, d.const(0.0), d.const(0.0), d.const(0.0), d)
  # x_sign = sign(x)
  x_sign = x.e(BinaryOps.CMPNE, d.const(0)).e(TernaryOps.WHERE, x.e(BinaryOps.CMPLT, x.const(0)).e(TernaryOps.WHERE, x.const(-1), x.const(1)), x.const(0)) # noqa: E501
  x_abs = x.e(BinaryOps.MUL, x_sign)
  r, q = reduction_algo(x_abs)
  if fast:
    result = sin_poly_small(r, q)
  else:
    # Payne Hanek Reduction assumes abs(x) >= pi/4, so for smaller values, use cody_waite_reduction.
    switch_over_map = x_abs.e(BinaryOps.CMPLT, x.const(switch_over))
    r_fast, q_fast = cody_waite_reduction(x_abs)
    r = switch_over_map.e(TernaryOps.WHERE, r_fast, r)
    q = switch_over_map.e(TernaryOps.WHERE, q_fast, q)
    result = switch_over_map.e(TernaryOps.WHERE, sin_poly_small(r, q), sin_poly_large(r, q))
  result = result.e(BinaryOps.MUL, x_sign) # adjusts the sign for abs(x).
  # sin(Inf) = NaN, sin(-Inf) = NaN, sin(NaN) = NaN
  return _lazy_map_numbers(d, d.const(math.nan), d.const(math.nan), d.const(math.nan), result)

def xexp2(x:LazyBuffer) -> LazyBuffer:
  """
  Implements a 1.0 ULP approximation for UnaryOps.EXP2
  - Paper: https://arxiv.org/pdf/2001.09258
  """
  assert is_dtype_transcendental_supported(x.dtype)
  if 0 in x.shape: return x
  fp64_p = x.dtype == dtypes.float64
  # mask +=inf/nan as zero.
  d = _lazy_map_numbers(x, x.const(0.0), x.const(0.0), x.const(0.0), x)
  q = rintk(d)
  # s = d - round(d)
  s = d.e(BinaryOps.ADD, q.cast(d.dtype).e(UnaryOps.NEG))
  # a polynomial approximation with 13 non-zero terms in the range of [−(log 2)/2,(log 2)/2].
  if fp64_p:
    u = polyN(s.const(0.4434359082926529454e-9), s, [0.7073164598085707425e-8, 0.1017819260921760451e-6, 0.1321543872511327615e-5, 0.1525273353517584730e-4, 0.1540353045101147808e-3, 0.1333355814670499073e-2, 0.9618129107597600536e-2, 0.5550410866482046596e-1, 0.2402265069591012214e+0, 0.6931471805599452862e+0, 0.1000000000000000000e+1]) # noqa: E501
  else:
    u = polyN(s.const(0.1535920892e-3), s, [0.1339262701e-2, 0.9618384764e-2, 0.5550347269e-1, 0.2402264476e+0, 0.6931471825e+0, 0.1000000000e+1])
  u = ldexp2k(u, q) # u*2^q
  upper = {dtypes.float64: 1024, dtypes.float32: 128, dtypes.float16: 23.0}[d.dtype]
  lower = {dtypes.float64: -2000, dtypes.float32: -150, dtypes.float16: -22}[d.dtype]
  # Replace x >= upper with +inf
  u = d.e(BinaryOps.CMPNE, d.const(upper)).e(TernaryOps.WHERE, u, d.const(math.inf))
  u = d.e(BinaryOps.CMPLT, d.const(upper)).e(TernaryOps.WHERE, u, d.const(math.inf))
  # Replace x <= lower with zero.
  u = d.e(BinaryOps.CMPLT, d.const(lower)).e(TernaryOps.WHERE, d.const(0.0), u)
  # x=NaN never satisfies x < Inf. (for fastmode)
  u = d.e(BinaryOps.CMPLT, d.const(math.inf)).e(TernaryOps.WHERE, u, u.const(math.nan))
  # exp2(Inf) = Inf, exp2(-Inf) = 0, exp2(NaN) = NaN
  return _lazy_map_numbers(x, x.const(math.inf), x.const(0.0), x.const(math.nan), u)

def xlog2(d:LazyBuffer) -> LazyBuffer:
  """
  Implements a 1.0 ULP approximation for UnaryOps.LOG2
  Paper: https://arxiv.org/pdf/2001.09258
  """
  assert is_dtype_transcendental_supported(d.dtype)
  if 0 in d.shape: return d
  fp64_p = d.dtype == dtypes.float64
  FLT_MIN = d.const(1e-6 if d.dtype == dtypes.float16 else 1e-4)
  Y_FLT_MIN = d.const(math.log2({dtypes.float64: 1e-228, dtypes.float32: 1e-38, dtypes.float16: 1e-6}[d.dtype]))
  d_orig = d
  denormal_map = d.e(BinaryOps.CMPLT, FLT_MIN)
  for _ in range(4):
    d = denormal_map.e(TernaryOps.WHERE, d.e(BinaryOps.MUL, d.const(2 ** 16)), d)

  e = ilogb2k(d.e(BinaryOps.MUL, d.const(1.0 / 0.75))).cast(d.dtype)
  m = ldexp3k(d, e.e(UnaryOps.NEG))
  e = denormal_map.e(TernaryOps.WHERE, e.e(BinaryOps.ADD, e.const(-64)), e)

  if fp64_p:
    x = m.e(BinaryOps.ADD, m.const(-1.0)).e(BinaryOps.MUL, m.e(BinaryOps.ADD, m.const(1.0)).e(UnaryOps.RECIP))
    x2 = x.e(BinaryOps.MUL, x)
    t = polyN(x.const(0.2211941750456081490e+0), x2, [0.2200768693152277689e+0, 0.2623708057488514656e+0, 0.3205977477944495502e+0, 0.4121985945485324709e+0, 0.5770780162997058982e+0, 0.96179669392608091449]) # noqa: E501
    s_hi, s_lo = dfadd2_f2_f2_f2(e, e.const(0), *dfmul2_f2_f2_f2(t.const(2.885390081777926774), t.const(0), x, x.const(0)))
    r = mla(t, x.e(BinaryOps.MUL, x2), s_hi.e(BinaryOps.ADD, s_lo))
  else:
    xx, xy = dfdiv2_f2_f2_f2(*dfadd2_f2_f2_f2(m.const(-1), m.const(0), m, m.const(0)), *dfadd2_f2_f2_f2(m.const(1), m.const(0), m, m.const(0)))
    x2 = xx.e(BinaryOps.MUL, xx)
    t = polyN(d.const(0.4374550283e+0), x2, [0.5764790177e+0, 0.9618012905120])
    sx, sy = dfadd2_f2_f2_f2(e, e.const(0), *dfmul2_f2_f2_f2(xx, xy, xx.const(2.8853900432586669922), xy.const(3.2734474483568488616e-08)))
    sx, sy = dfadd2_f2_f2_f2(sx, sy, x2.const(0), x2.e(BinaryOps.MUL, xx).e(BinaryOps.MUL, t))
    r = sx.e(BinaryOps.ADD, sy)
  # log2(Inf) = Inf
  r = d_orig.e(BinaryOps.CMPNE, d.const(math.inf)).e(TernaryOps.WHERE, r, r.const(math.inf))
  # log2(x=-0.01) = NaN. where x < 0
  r = d_orig.e(BinaryOps.CMPLT, d.const(-0.0)).e(TernaryOps.WHERE, r.const(math.nan), r)
  # log2(0) = -Inf
  r = d_orig.e(BinaryOps.CMPNE, d.const(0.0)).e(TernaryOps.WHERE, r, r.const(-math.inf))
  # y=log2(x) must be existing in the range of [log2(FLT_MIN), log2(Inf)]. otherwise the input was poisoned.
  # one exception is that x=0.0, it becomes -inf.
  r_inf_mapped = d_orig.e(BinaryOps.CMPNE, d_orig.const(0.0)).e(TernaryOps.WHERE, r.const(math.nan), r.const(-math.inf))
  r = r.e(BinaryOps.CMPLT, Y_FLT_MIN).e(TernaryOps.WHERE, r_inf_mapped, r)
  # log(NaN) = NaN, using for all real number x, either of x < Inf, x == Inf becomes True.
  r = d_orig.e(BinaryOps.CMPLT, d_orig.const(math.inf)).e(
    TernaryOps.WHERE, r, d_orig.e(BinaryOps.CMPNE, d_orig.const(math.inf)).e(
      TernaryOps.WHERE, d.const(math.nan), d))
  # log(-0.0) = -Inf. In certain devices like PTX, x == -0.0 won't be true. so making reciprocal.
  r = d_orig.e(UnaryOps.RECIP).e(BinaryOps.CMPNE, d_orig.const(-math.inf)).e(TernaryOps.WHERE, r, r.const(-math.inf))
  return r
