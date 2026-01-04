# DSL for RDNA3 pseudocode - makes pseudocode expressions work directly as Python
import struct, math
from extra.assembly.amd.dsl import MASK32, MASK64, _f32, _i32, _sext, _f16, _i16, _f64, _i64

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _div(a, b):
  try: return a / b
  except ZeroDivisionError:
    if a == 0.0 or math.isnan(a): return float("nan")
    return math.copysign(float("inf"), a * b) if b == 0.0 else float("inf") if a > 0 else float("-inf")
def _to_f16_bits(v): return v if isinstance(v, int) else _i16(v)
def _isnan(x):
  try: return math.isnan(float(x))
  except (TypeError, ValueError): return False
def _check_nan_type(x, quiet_bit_expected, default):
  """Check NaN type by examining quiet bit. Returns default if can't determine."""
  try:
    if not math.isnan(float(x)): return False
    if hasattr(x, '_reg') and hasattr(x, '_bits'):
      bits = x._reg._val & ((1 << x._bits) - 1)
      # NaN format: exponent all 1s, quiet bit, mantissa != 0
      # f16: exp[14:10]=31, quiet=bit9, mant[8:0]  |  f32: exp[30:23]=255, quiet=bit22, mant[22:0]  |  f64: exp[62:52]=2047, quiet=bit51, mant[51:0]
      exp_bits, quiet_pos, mant_mask = {16: (0x1f, 9, 0x3ff), 32: (0xff, 22, 0x7fffff), 64: (0x7ff, 51, 0xfffffffffffff)}.get(x._bits, (0,0,0))
      exp_shift = {16: 10, 32: 23, 64: 52}.get(x._bits, 0)
      if exp_bits and ((bits >> exp_shift) & exp_bits) == exp_bits and (bits & mant_mask) != 0:
        return ((bits >> quiet_pos) & 1) == quiet_bit_expected
    return default
  except (TypeError, ValueError): return False
def _isquietnan(x): return _check_nan_type(x, 1, True)  # quiet NaN has quiet bit = 1
def _issignalnan(x): return _check_nan_type(x, 0, False)  # signaling NaN has quiet bit = 0
def _gt_neg_zero(a, b): return (a > b) or (a == 0 and b == 0 and not math.copysign(1, a) < 0 and math.copysign(1, b) < 0)
def _lt_neg_zero(a, b): return (a < b) or (a == 0 and b == 0 and math.copysign(1, a) < 0 and not math.copysign(1, b) < 0)
def _fma(a, b, c): return a * b + c
def _signext(v): return v
def _fpop(fn):
  def wrapper(x):
    x = float(x)
    if math.isnan(x) or math.isinf(x): return x
    result = float(fn(x))
    # Preserve sign of zero (IEEE 754: ceil(-0.0) = -0.0, ceil(-0.1) = -0.0)
    if result == 0.0: return math.copysign(0.0, x)
    return result
  return wrapper
trunc, floor, ceil = _fpop(math.trunc), _fpop(math.floor), _fpop(math.ceil)
class _SafeFloat(float):
  """Float subclass that uses _div for division to handle 0/inf correctly."""
  def __truediv__(self, o): return _div(float(self), float(o))
  def __rtruediv__(self, o): return _div(float(o), float(self))
def sqrt(x): return _SafeFloat(math.sqrt(x)) if x >= 0 else _SafeFloat(float("nan"))
def log2(x): return math.log2(x) if x > 0 else (float("-inf") if x == 0 else float("nan"))
i32_to_f32 = u32_to_f32 = i32_to_f64 = u32_to_f64 = f32_to_f64 = f64_to_f32 = float
def _f_to_int(f, lo, hi): f = float(f); return 0 if math.isnan(f) else (hi if f >= hi else lo if f <= lo else int(f))
def f32_to_i32(f): return _f_to_int(f, -2147483648, 2147483647)
def f32_to_u32(f): return _f_to_int(f, 0, 4294967295)
f64_to_i32, f64_to_u32 = f32_to_i32, f32_to_u32
def f32_to_f16(f):
  f = float(f)
  if math.isnan(f): return 0x7e00  # f16 NaN
  if math.isinf(f): return 0x7c00 if f > 0 else 0xfc00  # f16 ±infinity
  try: return struct.unpack("<H", struct.pack("<e", f))[0]
  except OverflowError: return 0x7c00 if f > 0 else 0xfc00  # overflow -> ±infinity
def _f16_to_f32_bits(bits): return struct.unpack("<e", struct.pack("<H", int(bits) & 0xffff))[0]
def f16_to_f32(v): return v if isinstance(v, float) else _f16_to_f32_bits(v)
def i16_to_f16(v): return f32_to_f16(float(_sext(int(v) & 0xffff, 16)))
def u16_to_f16(v): return f32_to_f16(float(int(v) & 0xffff))
def f16_to_i16(bits): f = _f16_to_f32_bits(bits); return max(-32768, min(32767, int(f))) if not math.isnan(f) else 0
def f16_to_u16(bits): f = _f16_to_f32_bits(bits); return max(0, min(65535, int(f))) if not math.isnan(f) else 0
def u8_to_u32(v): return int(v) & 0xff
def u4_to_u32(v): return int(v) & 0xf
def _sign(f): return 1 if math.copysign(1.0, f) < 0 else 0
def _mantissa_f32(f): return struct.unpack("<I", struct.pack("<f", f))[0] & 0x7fffff if not (math.isinf(f) or math.isnan(f)) else 0
def _ldexp(m, e): return math.ldexp(m, e)
def isEven(x):
  x = float(x)
  if math.isinf(x) or math.isnan(x): return False
  return int(x) % 2 == 0
def fract(x): return x - math.floor(x)
PI = math.pi
def _trig(fn, x):
  # V_SIN/COS_F32: hardware does frac on input cycles before computing
  if math.isinf(x) or math.isnan(x): return float("nan")
  frac_cycles = fract(x / (2 * math.pi))
  result = fn(frac_cycles * 2 * math.pi)
  # Hardware returns exactly 0 for cos(π/2), sin(π), etc. due to lookup table
  # Round very small results (below f32 precision) to exactly 0
  if abs(result) < 1e-7: return 0.0
  return result
def sin(x): return _trig(math.sin, x)
def cos(x): return _trig(math.cos, x)
def pow(a, b):
  try: return a ** b
  except OverflowError: return float("inf") if b > 0 else 0.0
def _brev(v, bits): return int(bin(v & ((1 << bits) - 1))[2:].zfill(bits)[::-1], 2)
def _brev32(v): return _brev(v, 32)
def _brev64(v): return _brev(v, 64)
def _ctz(v, bits):
  v, n = int(v) & ((1 << bits) - 1), 0
  if v == 0: return bits
  while (v & 1) == 0: v >>= 1; n += 1
  return n
def _ctz32(v): return _ctz(v, 32)
def _ctz64(v): return _ctz(v, 64)
def _exponent(f):
  # Handle TypedView (f16/f32/f64) to get correct exponent for that type
  if hasattr(f, '_bits') and hasattr(f, '_float') and f._float:
    raw = f._val
    if f._bits == 16: return (raw >> 10) & 0x1f  # f16: 5-bit exponent
    if f._bits == 32: return (raw >> 23) & 0xff  # f32: 8-bit exponent
    if f._bits == 64: return (raw >> 52) & 0x7ff  # f64: 11-bit exponent
  # Fallback: convert to f32 and get exponent
  f = float(f)
  if math.isinf(f) or math.isnan(f): return 255
  if f == 0.0: return 0
  try: bits = struct.unpack("<I", struct.pack("<f", f))[0]; return (bits >> 23) & 0xff
  except: return 0
def _is_denorm_f32(f):
  if not isinstance(f, float): f = _f32(int(f) & 0xffffffff)
  if math.isinf(f) or math.isnan(f) or f == 0.0: return False
  bits = struct.unpack("<I", struct.pack("<f", float(f)))[0]
  return (bits >> 23) & 0xff == 0
def _is_denorm_f64(f):
  if not isinstance(f, float): f = _f64(int(f) & 0xffffffffffffffff)
  if math.isinf(f) or math.isnan(f) or f == 0.0: return False
  bits = struct.unpack("<Q", struct.pack("<d", float(f)))[0]
  return (bits >> 52) & 0x7ff == 0
def v_min_f32(a, b): return a if math.isnan(b) else b if math.isnan(a) else (a if _lt_neg_zero(a, b) else b)
def v_max_f32(a, b): return a if math.isnan(b) else b if math.isnan(a) else (a if _gt_neg_zero(a, b) else b)
v_min_f16, v_max_f16 = v_min_f32, v_max_f32
v_min_i32, v_max_i32 = min, max
v_min_i16, v_max_i16 = min, max
def v_min_u32(a, b): return min(a & MASK32, b & MASK32)
def v_max_u32(a, b): return max(a & MASK32, b & MASK32)
def v_min_u16(a, b): return min(a & 0xffff, b & 0xffff)
def v_max_u16(a, b): return max(a & 0xffff, b & 0xffff)
def v_min3_f32(a, b, c): return v_min_f32(v_min_f32(a, b), c)
def v_max3_f32(a, b, c): return v_max_f32(v_max_f32(a, b), c)
v_min3_f16, v_max3_f16 = v_min3_f32, v_max3_f32
v_min3_i32, v_max3_i32, v_min3_i16, v_max3_i16 = min, max, min, max
def v_min3_u32(a, b, c): return min(a & MASK32, b & MASK32, c & MASK32)
def v_max3_u32(a, b, c): return max(a & MASK32, b & MASK32, c & MASK32)
def v_min3_u16(a, b, c): return min(a & 0xffff, b & 0xffff, c & 0xffff)
def v_max3_u16(a, b, c): return max(a & 0xffff, b & 0xffff, c & 0xffff)
def ABSDIFF(a, b): return abs(int(a) - int(b))

# BF16 (bfloat16) conversion functions
def _bf16(i):
  """Convert bf16 bits to float. BF16 is just the top 16 bits of f32."""
  return struct.unpack("<f", struct.pack("<I", (i & 0xffff) << 16))[0]
def _ibf16(f):
  """Convert float to bf16 bits (truncate to top 16 bits of f32)."""
  if math.isnan(f): return 0x7fc0  # bf16 quiet NaN
  if math.isinf(f): return 0x7f80 if f > 0 else 0xff80  # bf16 ±infinity
  try: return (struct.unpack("<I", struct.pack("<f", float(f)))[0] >> 16) & 0xffff
  except (OverflowError, struct.error): return 0x7f80 if f > 0 else 0xff80
def bf16_to_f32(v): return _bf16(v) if isinstance(v, int) else float(v)
def f32_to_bf16(f): return _ibf16(f)

# BYTE_PERMUTE for V_PERM_B32 - select bytes from 64-bit data based on selector
def BYTE_PERMUTE(data, sel):
  """Select a byte from 64-bit data based on selector value.
  sel 0-7: select byte from data (S1 is bytes 0-3, S0 is bytes 4-7 in {S0,S1})
  sel 8-11: sign-extend from specific bytes (8->byte1, 9->byte3, 10->byte5, 11->byte7)
  sel 12: constant 0x00
  sel >= 13: constant 0xFF"""
  sel = int(sel) & 0xff
  if sel <= 7: return (int(data) >> (sel * 8)) & 0xff
  if sel == 8: return 0xff if ((int(data) >> 15) & 1) else 0x00  # sign of byte 1
  if sel == 9: return 0xff if ((int(data) >> 31) & 1) else 0x00  # sign of byte 3
  if sel == 10: return 0xff if ((int(data) >> 47) & 1) else 0x00  # sign of byte 5
  if sel == 11: return 0xff if ((int(data) >> 63) & 1) else 0x00  # sign of byte 7
  if sel == 12: return 0x00
  return 0xff  # sel >= 13

# v_sad_u8 helper for V_SAD instructions (sum of absolute differences of 4 bytes)
def v_sad_u8(s0, s1, s2):
  """V_SAD_U8: Sum of absolute differences of 4 byte pairs plus accumulator."""
  s0, s1, s2 = int(s0), int(s1), int(s2)
  result = s2
  for i in range(4):
    a = (s0 >> (i * 8)) & 0xff
    b = (s1 >> (i * 8)) & 0xff
    result += abs(a - b)
  return result & 0xffffffff

# v_msad_u8 helper (masked SAD - skip when reference byte is 0)
def v_msad_u8(s0, s1, s2):
  """V_MSAD_U8: Masked sum of absolute differences (skip if reference byte is 0)."""
  s0, s1, s2 = int(s0), int(s1), int(s2)
  result = s2
  for i in range(4):
    a = (s0 >> (i * 8)) & 0xff
    b = (s1 >> (i * 8)) & 0xff
    if b != 0:  # Only add diff if reference (s1) byte is non-zero
      result += abs(a - b)
  return result & 0xffffffff
def f16_to_snorm(f): return max(-32768, min(32767, int(round(max(-1.0, min(1.0, f)) * 32767))))
def f16_to_unorm(f): return max(0, min(65535, int(round(max(0.0, min(1.0, f)) * 65535))))
def f32_to_snorm(f): return max(-32768, min(32767, int(round(max(-1.0, min(1.0, f)) * 32767))))
def f32_to_unorm(f): return max(0, min(65535, int(round(max(0.0, min(1.0, f)) * 65535))))
def v_cvt_i16_f32(f): return max(-32768, min(32767, int(f))) if not math.isnan(f) else 0
def v_cvt_u16_f32(f): return max(0, min(65535, int(f))) if not math.isnan(f) else 0
def u32_to_u16(u): return int(u) & 0xffff
def i32_to_i16(i): return ((int(i) + 32768) & 0xffff) - 32768
def SAT8(v): return max(0, min(255, int(v)))
def f32_to_u8(f): return max(0, min(255, int(f))) if not math.isnan(f) else 0
def mantissa(f):
  if f == 0.0 or math.isinf(f) or math.isnan(f): return f
  m, _ = math.frexp(f)
  return m  # AMD V_FREXP_MANT returns mantissa in [0.5, 1.0) range
def signext_from_bit(val, bit):
  bit = int(bit)
  if bit == 0: return 0
  mask = (1 << bit) - 1
  val = int(val) & mask
  if val & (1 << (bit - 1)): return val - (1 << bit)
  return val

# Aliases used in pseudocode
s_ff1_i32_b32, s_ff1_i32_b64 = _ctz32, _ctz64
GT_NEG_ZERO, LT_NEG_ZERO = _gt_neg_zero, _lt_neg_zero
isNAN = _isnan
isQuietNAN = _isquietnan
isSignalNAN = _issignalnan
fma, ldexp, sign, exponent = _fma, _ldexp, _sign, _exponent
def F(x):
  """32'F(x) or 64'F(x) - interpret x as float. If x is int, treat as bit pattern."""
  if isinstance(x, int): return _f32(x)  # int -> interpret as f32 bits
  if isinstance(x, TypedView): return x  # preserve TypedView for bit-pattern checks
  return float(x)  # already a float or float-like
signext = lambda x: int(x)  # sign-extend to full width - already handled by Python's arbitrary precision ints
pack = lambda hi, lo: ((int(hi) & 0xffff) << 16) | (int(lo) & 0xffff)
pack32 = lambda hi, lo: ((int(hi) & 0xffffffff) << 32) | (int(lo) & 0xffffffff)
_pack, _pack32 = pack, pack32  # Aliases for internal use
WAVE32, WAVE64 = True, False

# Float overflow/underflow constants
OVERFLOW_F32 = float('inf')
UNDERFLOW_F32 = 0.0
OVERFLOW_F64 = float('inf')
UNDERFLOW_F64 = 0.0
MAX_FLOAT_F32 = 3.4028235e+38  # Largest finite float32

# INF object that supports .f16/.f32/.f64 access and comparison with floats
class _Inf:
  f16 = f32 = f64 = float('inf')
  def __neg__(self): return _NegInf()
  def __pos__(self): return self
  def __float__(self): return float('inf')
  def __eq__(self, other): return float(other) == float('inf') if not isinstance(other, _NegInf) else False
  def __req__(self, other): return self.__eq__(other)
class _NegInf:
  f16 = f32 = f64 = float('-inf')
  def __neg__(self): return _Inf()
  def __pos__(self): return self
  def __float__(self): return float('-inf')
  def __eq__(self, other): return float(other) == float('-inf') if not isinstance(other, _Inf) else False
  def __req__(self, other): return self.__eq__(other)
INF = _Inf()

# Rounding mode placeholder
class _RoundMode:
  NEAREST_EVEN = 0
ROUND_MODE = _RoundMode()

# Helper functions for pseudocode
def cvtToQuietNAN(x): return float('nan')
DST = None  # Placeholder, will be set in context

class _WaveMode:
  IEEE = False
WAVE_MODE = _WaveMode()

class _DenormChecker:
  """Comparator for denormalized floats. x == DENORM.f32 checks if x is denormalized."""
  def __init__(self, bits): self._bits = bits
  def _check(self, other):
    return _is_denorm_f64(float(other)) if self._bits == 64 else _is_denorm_f32(float(other))
  def __eq__(self, other): return self._check(other)
  def __req__(self, other): return self._check(other)
  def __ne__(self, other): return not self._check(other)

class _Denorm:
  f32 = _DenormChecker(32)
  f64 = _DenormChecker(64)
DENORM = _Denorm()

class TypedView:
  """View into a Reg with typed access. Used for both full-width (Reg.u32) and slices (Reg[31:16])."""
  __slots__ = ('_reg', '_high', '_low', '_signed', '_float', '_bf16')
  def __init__(self, reg, high, low=0, signed=False, is_float=False, is_bf16=False):
    # Handle reversed slices like [0:31] which means bit-reverse
    if high < low: high, low = low, high
    self._reg, self._high, self._low = reg, high, low
    self._signed, self._float, self._bf16 = signed, is_float, is_bf16

  def _nbits(self): return self._high - self._low + 1
  def _mask(self): return (1 << self._nbits()) - 1
  def _get(self): return (self._reg._val >> self._low) & self._mask()
  def _set(self, v): self._reg._val = (self._reg._val & ~(self._mask() << self._low)) | ((int(v) & self._mask()) << self._low)

  @property
  def _val(self): return self._get()
  @property
  def _bits(self): return self._nbits()

  # Type accessors for slices (e.g., D0[31:16].f16)
  u8 = property(lambda s: s._get() & 0xff)
  u16 = property(lambda s: s._get() & 0xffff, lambda s, v: s._set(v))
  u32 = property(lambda s: s._get() & MASK32, lambda s, v: s._set(v))
  i16 = property(lambda s: _sext(s._get() & 0xffff, 16), lambda s, v: s._set(v))
  i32 = property(lambda s: _sext(s._get() & MASK32, 32), lambda s, v: s._set(v))
  f16 = property(lambda s: _f16(s._get()), lambda s, v: s._set(v if isinstance(v, int) else _i16(float(v))))
  f32 = property(lambda s: _f32(s._get()), lambda s, v: s._set(_i32(float(v))))
  bf16 = property(lambda s: _bf16(s._get()), lambda s, v: s._set(v if isinstance(v, int) else _ibf16(float(v))))
  b16, b32 = u16, u32

  # Chained type access (e.g., jump_addr.i64 when jump_addr is already TypedView)
  @property
  def i64(s): return s if s._nbits() == 64 and s._signed else int(s)
  @property
  def u64(s): return s if s._nbits() == 64 and not s._signed else int(s) & MASK64

  def __getitem__(self, key):
    if isinstance(key, slice):
      high, low = int(key.start), int(key.stop)
      return TypedView(self._reg, high, low)
    return (self._get() >> int(key)) & 1

  def __setitem__(self, key, value):
    if isinstance(key, slice):
      high, low = int(key.start), int(key.stop)
      if high < low: high, low, value = low, high, _brev(int(value), low - high + 1)
      mask = (1 << (high - low + 1)) - 1
      self._reg._val = (self._reg._val & ~(mask << low)) | ((int(value) & mask) << low)
    elif value: self._reg._val |= (1 << int(key))
    else: self._reg._val &= ~(1 << int(key))

  def __int__(self): return _sext(self._get(), self._nbits()) if self._signed else self._get()
  def __index__(self): return int(self)
  def __trunc__(self): return int(float(self)) if self._float else int(self)
  def __float__(self):
    if self._float:
      if self._bf16: return _bf16(self._get())
      bits = self._nbits()
      return _f16(self._get()) if bits == 16 else _f32(self._get()) if bits == 32 else _f64(self._get())
    return float(int(self))
  def __bool__(s): return bool(int(s))

  # Arithmetic - floats use float(), ints use int()
  def __add__(s, o): return float(s) + float(o) if s._float else int(s) + int(o)
  def __radd__(s, o): return float(o) + float(s) if s._float else int(o) + int(s)
  def __sub__(s, o): return float(s) - float(o) if s._float else int(s) - int(o)
  def __rsub__(s, o): return float(o) - float(s) if s._float else int(o) - int(s)
  def __mul__(s, o): return float(s) * float(o) if s._float else int(s) * int(o)
  def __rmul__(s, o): return float(o) * float(s) if s._float else int(o) * int(s)
  def __truediv__(s, o): return _div(float(s), float(o)) if s._float else _div(int(s), int(o))
  def __rtruediv__(s, o): return _div(float(o), float(s)) if s._float else _div(int(o), int(s))
  def __pow__(s, o): return float(s) ** float(o) if s._float else int(s) ** int(o)
  def __rpow__(s, o): return float(o) ** float(s) if s._float else int(o) ** int(s)
  def __neg__(s): return -float(s) if s._float else -int(s)
  def __abs__(s): return abs(float(s)) if s._float else abs(int(s))

  # Bitwise - GPU shifts mask the shift amount to valid range
  def __and__(s, o): return int(s) & int(o)
  def __or__(s, o): return int(s) | int(o)
  def __xor__(s, o): return int(s) ^ int(o)
  def __invert__(s): return ~int(s)
  def __lshift__(s, o): n = int(o); return int(s) << n if 0 <= n < 64 else 0
  def __rshift__(s, o): n = int(o); return int(s) >> n if 0 <= n < 64 else 0
  def __rand__(s, o): return int(o) & int(s)
  def __ror__(s, o): return int(o) | int(s)
  def __rxor__(s, o): return int(o) ^ int(s)
  def __rlshift__(s, o): n = int(s); return int(o) << n if 0 <= n < 64 else 0
  def __rrshift__(s, o): n = int(s); return int(o) >> n if 0 <= n < 64 else 0

  # Comparison - handle _DenormChecker specially
  def __eq__(s, o):
    if isinstance(o, _DenormChecker): return o._check(s)
    return float(s) == float(o) if s._float else int(s) == int(o)
  def __ne__(s, o):
    if isinstance(o, _DenormChecker): return not o._check(s)
    return float(s) != float(o) if s._float else int(s) != int(o)
  def __lt__(s, o): return float(s) < float(o) if s._float else int(s) < int(o)
  def __le__(s, o): return float(s) <= float(o) if s._float else int(s) <= int(o)
  def __gt__(s, o): return float(s) > float(o) if s._float else int(s) > int(o)
  def __ge__(s, o): return float(s) >= float(o) if s._float else int(s) >= int(o)

SliceProxy = TypedView  # Alias for compatibility

class Reg:
  """GPU register: D0.f32 = S0.f32 + S1.f32 just works. Supports up to 128 bits for DS_LOAD_B128."""
  __slots__ = ('_val',)
  def __init__(self, val=0): self._val = int(val)

  # Typed views - TypedView(reg, high, signed, is_float, is_bf16)
  u64 = property(lambda s: TypedView(s, 63), lambda s, v: setattr(s, '_val', int(v) & MASK64))
  i64 = property(lambda s: TypedView(s, 63, signed=True), lambda s, v: setattr(s, '_val', int(v) & MASK64))
  b64 = property(lambda s: TypedView(s, 63), lambda s, v: setattr(s, '_val', int(v) & MASK64))
  f64 = property(lambda s: TypedView(s, 63, is_float=True), lambda s, v: setattr(s, '_val', v if isinstance(v, int) else _i64(float(v))))
  u32 = property(lambda s: TypedView(s, 31), lambda s, v: setattr(s, '_val', int(v) & MASK32))
  i32 = property(lambda s: TypedView(s, 31, signed=True), lambda s, v: setattr(s, '_val', int(v) & MASK32))
  b32 = property(lambda s: TypedView(s, 31), lambda s, v: setattr(s, '_val', int(v) & MASK32))
  f32 = property(lambda s: TypedView(s, 31, is_float=True), lambda s, v: setattr(s, '_val', _i32(float(v))))
  u24 = property(lambda s: TypedView(s, 23))
  i24 = property(lambda s: TypedView(s, 23, signed=True))
  u16 = property(lambda s: TypedView(s, 15), lambda s, v: setattr(s, '_val', (s._val & 0xffff0000) | (int(v) & 0xffff)))
  i16 = property(lambda s: TypedView(s, 15, signed=True), lambda s, v: setattr(s, '_val', (s._val & 0xffff0000) | (int(v) & 0xffff)))
  b16 = property(lambda s: TypedView(s, 15), lambda s, v: setattr(s, '_val', (s._val & 0xffff0000) | (int(v) & 0xffff)))
  f16 = property(lambda s: TypedView(s, 15, is_float=True), lambda s, v: setattr(s, '_val', (s._val & 0xffff0000) | ((v if isinstance(v, int) else _i16(float(v))) & 0xffff)))
  bf16 = property(lambda s: TypedView(s, 15, is_float=True, is_bf16=True), lambda s, v: setattr(s, '_val', (s._val & 0xffff0000) | ((v if isinstance(v, int) else _ibf16(float(v))) & 0xffff)))
  u8 = property(lambda s: TypedView(s, 7))
  i8 = property(lambda s: TypedView(s, 7, signed=True))
  u3 = property(lambda s: TypedView(s, 2))  # 3-bit for opsel fields
  u1 = property(lambda s: TypedView(s, 0))  # single bit

  def __getitem__(s, key):
    if isinstance(key, slice): return TypedView(s, int(key.start), int(key.stop))
    return (s._val >> int(key)) & 1

  def __setitem__(s, key, value):
    if isinstance(key, slice):
      high, low = int(key.start), int(key.stop)
      if high < low: high, low = low, high
      mask = (1 << (high - low + 1)) - 1
      s._val = (s._val & ~(mask << low)) | ((int(value) & mask) << low)
    elif value: s._val |= (1 << int(key))
    else: s._val &= ~(1 << int(key))

  def __int__(s): return s._val
  def __index__(s): return s._val
  def __bool__(s): return bool(s._val)

  # Arithmetic (for tmp = tmp + 1 patterns). Float operands trigger f32 interpretation.
  def __add__(s, o): return (_f32(s._val) + float(o)) if isinstance(o, float) else s._val + int(o)
  def __radd__(s, o): return (float(o) + _f32(s._val)) if isinstance(o, float) else int(o) + s._val
  def __sub__(s, o): return (_f32(s._val) - float(o)) if isinstance(o, float) else s._val - int(o)
  def __rsub__(s, o): return (float(o) - _f32(s._val)) if isinstance(o, float) else int(o) - s._val
  def __mul__(s, o): return (_f32(s._val) * float(o)) if isinstance(o, float) else s._val * int(o)
  def __rmul__(s, o): return (float(o) * _f32(s._val)) if isinstance(o, float) else int(o) * s._val
  def __and__(s, o): return s._val & int(o)
  def __rand__(s, o): return int(o) & s._val
  def __or__(s, o): return s._val | int(o)
  def __ror__(s, o): return int(o) | s._val
  def __xor__(s, o): return s._val ^ int(o)
  def __rxor__(s, o): return int(o) ^ s._val
  def __lshift__(s, o): n = int(o); return s._val << n if 0 <= n < 64 else 0
  def __rshift__(s, o): n = int(o); return s._val >> n if 0 <= n < 64 else 0
  def __invert__(s): return ~s._val

  # Comparison (for tmp >= 0x100000000 patterns)
  def __lt__(s, o): return s._val < int(o)
  def __le__(s, o): return s._val <= int(o)
  def __gt__(s, o): return s._val > int(o)
  def __ge__(s, o): return s._val >= int(o)
  def __eq__(s, o): return s._val == int(o)
  def __ne__(s, o): return s._val != int(o)

# 2/PI with 1201 bits of precision for V_TRIG_PREOP_F64
TWO_OVER_PI_1201 = Reg(0x0145f306dc9c882a53f84eafa3ea69bb81b6c52b3278872083fca2c757bd778ac36e48dc74849ba5c00c925dd413a32439fc3bd63962534e7dd1046bea5d768909d338e04d68befc827323ac7306a673e93908bf177bf250763ff12fffbc0b301fde5e2316b414da3eda6cfd9e4f96136e9e8c7ecd3cbfd45aea4f758fd7cbe2f67a0e73ef14a525d4d7f6bf623f1aba10ac06608df8f6)
