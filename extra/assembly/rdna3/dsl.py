# DSL for RDNA3 pseudocode - makes pseudocode expressions work directly as Python
import struct, math, re

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS (previously in helpers.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _f32(i): return struct.unpack("<f", struct.pack("<I", i & 0xffffffff))[0]
def _i32(f):
  if isinstance(f, int): f = float(f)
  if math.isnan(f): return 0xffc00000 if math.copysign(1.0, f) < 0 else 0x7fc00000
  if math.isinf(f): return 0x7f800000 if f > 0 else 0xff800000
  try: return struct.unpack("<I", struct.pack("<f", f))[0]
  except (OverflowError, struct.error): return 0x7f800000 if f > 0 else 0xff800000
def _div(a, b):
  try: return a / b
  except ZeroDivisionError:
    if a == 0.0 or math.isnan(a): return float("nan")
    return math.copysign(float("inf"), a * b) if b == 0.0 else float("inf") if a > 0 else float("-inf")
def _sext(v, b): return v - (1 << b) if v & (1 << (b - 1)) else v
def _f16(i): return struct.unpack("<e", struct.pack("<H", i & 0xffff))[0]
def _i16(f):
  if math.isnan(f): return 0x7e00
  if math.isinf(f): return 0x7c00 if f > 0 else 0xfc00
  try: return struct.unpack("<H", struct.pack("<e", f))[0]
  except (OverflowError, struct.error): return 0x7c00 if f > 0 else 0xfc00
def _to_f16_bits(v): return v if isinstance(v, int) else _i16(v)
def _f64(i): return struct.unpack("<d", struct.pack("<Q", i & 0xffffffffffffffff))[0]
def _i64(f):
  if math.isnan(f): return 0x7ff8000000000000
  if math.isinf(f): return 0x7ff0000000000000 if f > 0 else 0xfff0000000000000
  try: return struct.unpack("<Q", struct.pack("<d", f))[0]
  except (OverflowError, struct.error): return 0x7ff0000000000000 if f > 0 else 0xfff0000000000000
def _isnan(x):
  try: return math.isnan(float(x))
  except (TypeError, ValueError): return False
def _isquietnan(x):
  """Check if x is a quiet NaN. For f32: exponent=255, bit22=1, mantissa!=0"""
  try:
    if not math.isnan(float(x)): return False
    # Get raw bits from TypedView or similar object with _reg attribute
    if hasattr(x, '_reg') and hasattr(x, '_bits'):
      bits = x._reg._val & ((1 << x._bits) - 1)
      if x._bits == 32:
        return ((bits >> 23) & 0xff) == 255 and ((bits >> 22) & 1) == 1 and (bits & 0x7fffff) != 0
      if x._bits == 64:
        return ((bits >> 52) & 0x7ff) == 0x7ff and ((bits >> 51) & 1) == 1 and (bits & 0xfffffffffffff) != 0
    return True  # Default to quiet NaN if we can't determine bit pattern
  except (TypeError, ValueError): return False
def _issignalnan(x):
  """Check if x is a signaling NaN. For f32: exponent=255, bit22=0, mantissa!=0"""
  try:
    if not math.isnan(float(x)): return False
    # Get raw bits from TypedView or similar object with _reg attribute
    if hasattr(x, '_reg') and hasattr(x, '_bits'):
      bits = x._reg._val & ((1 << x._bits) - 1)
      if x._bits == 32:
        return ((bits >> 23) & 0xff) == 255 and ((bits >> 22) & 1) == 0 and (bits & 0x7fffff) != 0
      if x._bits == 64:
        return ((bits >> 52) & 0x7ff) == 0x7ff and ((bits >> 51) & 1) == 0 and (bits & 0xfffffffffffff) != 0
    return False  # Default to not signaling if we can't determine bit pattern
  except (TypeError, ValueError): return False
def _gt_neg_zero(a, b): return (a > b) or (a == 0 and b == 0 and not math.copysign(1, a) < 0 and math.copysign(1, b) < 0)
def _lt_neg_zero(a, b): return (a < b) or (a == 0 and b == 0 and math.copysign(1, a) < 0 and not math.copysign(1, b) < 0)
def _fma(a, b, c): return a * b + c
def _signext(v): return v
def trunc(x):
  x = float(x)
  return x if math.isnan(x) or math.isinf(x) else math.trunc(x)
def floor(x):
  x = float(x)
  return x if math.isnan(x) or math.isinf(x) else math.floor(x)
def ceil(x):
  x = float(x)
  return x if math.isnan(x) or math.isinf(x) else math.ceil(x)
def sqrt(x): return math.sqrt(x) if x >= 0 else float("nan")
def log2(x): return math.log2(x) if x > 0 else (float("-inf") if x == 0 else float("nan"))
i32_to_f32 = u32_to_f32 = i32_to_f64 = u32_to_f64 = f32_to_f64 = f64_to_f32 = float
def f32_to_i32(f):
  f = float(f)
  if math.isnan(f): return 0
  if f >= 2147483647: return 2147483647
  if f <= -2147483648: return -2147483648
  return int(f)
def f32_to_u32(f):
  f = float(f)
  if math.isnan(f): return 0
  if f >= 4294967295: return 4294967295
  if f <= 0: return 0
  return int(f)
f64_to_i32 = f32_to_i32
f64_to_u32 = f32_to_u32
def f32_to_f16(f): return struct.unpack("<H", struct.pack("<e", float(f)))[0]
def _f16_to_f32_bits(bits): return struct.unpack("<e", struct.pack("<H", int(bits) & 0xffff))[0]
def f16_to_f32(v): return v if isinstance(v, float) else _f16_to_f32_bits(v)
def i16_to_f16(v): return f32_to_f16(float(_sext(int(v) & 0xffff, 16)))
def u16_to_f16(v): return f32_to_f16(float(int(v) & 0xffff))
def f16_to_i16(bits): f = _f16_to_f32_bits(bits); return max(-32768, min(32767, int(f))) if not math.isnan(f) else 0
def f16_to_u16(bits): f = _f16_to_f32_bits(bits); return max(0, min(65535, int(f))) if not math.isnan(f) else 0
def _sign(f): return 1 if math.copysign(1.0, f) < 0 else 0
def _mantissa_f32(f): return struct.unpack("<I", struct.pack("<f", f))[0] & 0x7fffff if not (math.isinf(f) or math.isnan(f)) else 0
def _ldexp(m, e): return math.ldexp(m, e)
def isEven(x): return int(x) % 2 == 0
def fract(x): return x - math.floor(x)
PI = math.pi
def sin(x): return float("nan") if math.isinf(x) or math.isnan(x) else math.sin(x)
def cos(x): return float("nan") if math.isinf(x) or math.isnan(x) else math.cos(x)
def pow(a, b):
  try: return a ** b
  except OverflowError: return float("inf") if b > 0 else 0.0
def _brev32(v): return int(bin(v & 0xffffffff)[2:].zfill(32)[::-1], 2)
def _brev64(v): return int(bin(v & 0xffffffffffffffff)[2:].zfill(64)[::-1], 2)
def _ctz32(v):
  v = int(v) & 0xffffffff
  if v == 0: return 32
  n = 0
  while (v & 1) == 0: v >>= 1; n += 1
  return n
def _ctz64(v):
  v = int(v) & 0xffffffffffffffff
  if v == 0: return 64
  n = 0
  while (v & 1) == 0: v >>= 1; n += 1
  return n
def _exponent(f):
  if math.isinf(f) or math.isnan(f): return 255
  if f == 0.0: return 0
  try: bits = struct.unpack("<I", struct.pack("<f", float(f)))[0]; return (bits >> 23) & 0xff
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
def v_min_f32(a, b):
  if math.isnan(b): return a
  if math.isnan(a): return b
  return a if _lt_neg_zero(a, b) else b
def v_max_f32(a, b):
  if math.isnan(b): return a
  if math.isnan(a): return b
  return a if _gt_neg_zero(a, b) else b
def v_min_i32(a, b): return min(a, b)
def v_max_i32(a, b): return max(a, b)
def v_min_u32(a, b): return min(a & 0xffffffff, b & 0xffffffff)
def v_max_u32(a, b): return max(a & 0xffffffff, b & 0xffffffff)
v_min_f16 = v_min_f32
v_max_f16 = v_max_f32
v_min_i16 = v_min_i32
v_max_i16 = v_max_i32
def v_min_u16(a, b): return min(a & 0xffff, b & 0xffff)
def v_max_u16(a, b): return max(a & 0xffff, b & 0xffff)
def v_min3_f32(a, b, c): return v_min_f32(v_min_f32(a, b), c)
def v_max3_f32(a, b, c): return v_max_f32(v_max_f32(a, b), c)
def v_min3_i32(a, b, c): return min(a, b, c)
def v_max3_i32(a, b, c): return max(a, b, c)
def v_min3_u32(a, b, c): return min(a & 0xffffffff, b & 0xffffffff, c & 0xffffffff)
def v_max3_u32(a, b, c): return max(a & 0xffffffff, b & 0xffffffff, c & 0xffffffff)
v_min3_f16 = v_min3_f32
v_max3_f16 = v_max3_f32
v_min3_i16 = v_min3_i32
v_max3_i16 = v_max3_i32
def v_min3_u16(a, b, c): return min(a & 0xffff, b & 0xffff, c & 0xffff)
def v_max3_u16(a, b, c): return max(a & 0xffff, b & 0xffff, c & 0xffff)
def ABSDIFF(a, b): return abs(a - b)
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
  return math.copysign(m * 2.0, f)
def signext_from_bit(val, bit):
  bit = int(bit)
  if bit == 0: return 0
  mask = (1 << bit) - 1
  val = int(val) & mask
  if val & (1 << (bit - 1)): return val - (1 << bit)
  return val

# ═══════════════════════════════════════════════════════════════════════════════
# DSL EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
  # Classes
  'Reg', 'SliceProxy', 'TypedView', 'ExecContext', 'compile_pseudocode',
  # Pack functions
  '_pack', '_pack32', 'pack', 'pack32',
  # Constants
  'WAVE32', 'WAVE64', 'MASK32', 'MASK64', 'WAVE_MODE', 'DENORM', 'OVERFLOW_F32', 'UNDERFLOW_F32',
  'OVERFLOW_F64', 'UNDERFLOW_F64', 'MAX_FLOAT_F32', 'ROUND_MODE', 'cvtToQuietNAN', 'DST', 'INF', 'PI',
  # Aliases for pseudocode
  's_ff1_i32_b32', 's_ff1_i32_b64', 'GT_NEG_ZERO', 'LT_NEG_ZERO',
  'isNAN', 'isQuietNAN', 'isSignalNAN', 'fma', 'ldexp', 'sign', 'exponent', 'F', 'signext',
  # Conversion functions
  '_f32', '_i32', '_f16', '_i16', '_f64', '_i64', '_sext', '_to_f16_bits', '_f16_to_f32_bits',
  'i32_to_f32', 'u32_to_f32', 'i32_to_f64', 'u32_to_f64', 'f32_to_f64', 'f64_to_f32',
  'f32_to_i32', 'f32_to_u32', 'f64_to_i32', 'f64_to_u32', 'f32_to_f16', 'f16_to_f32',
  'i16_to_f16', 'u16_to_f16', 'f16_to_i16', 'f16_to_u16', 'u32_to_u16', 'i32_to_i16',
  'f16_to_snorm', 'f16_to_unorm', 'f32_to_snorm', 'f32_to_unorm', 'v_cvt_i16_f32', 'v_cvt_u16_f32',
  'SAT8', 'f32_to_u8',
  # Math functions
  'trunc', 'floor', 'ceil', 'sqrt', 'log2', 'sin', 'cos', 'pow', 'fract', 'isEven', 'mantissa',
  # Min/max functions
  'v_min_f32', 'v_max_f32', 'v_min_i32', 'v_max_i32', 'v_min_u32', 'v_max_u32',
  'v_min_f16', 'v_max_f16', 'v_min_i16', 'v_max_i16', 'v_min_u16', 'v_max_u16',
  'v_min3_f32', 'v_max3_f32', 'v_min3_i32', 'v_max3_i32', 'v_min3_u32', 'v_max3_u32',
  'v_min3_f16', 'v_max3_f16', 'v_min3_i16', 'v_max3_i16', 'v_min3_u16', 'v_max3_u16',
  'ABSDIFF',
  # Bit manipulation
  '_brev32', '_brev64', '_ctz32', '_ctz64', '_exponent', '_is_denorm_f32', '_is_denorm_f64',
  '_sign', '_mantissa_f32', '_div', '_isnan', '_isquietnan', '_issignalnan', '_gt_neg_zero', '_lt_neg_zero', '_fma', '_ldexp', '_signext',
  'signext_from_bit',
]

# Aliases used in pseudocode
s_ff1_i32_b32, s_ff1_i32_b64 = _ctz32, _ctz64
GT_NEG_ZERO, LT_NEG_ZERO = _gt_neg_zero, _lt_neg_zero
isNAN = _isnan
isQuietNAN = _isquietnan
isSignalNAN = _issignalnan
fma, ldexp, sign, exponent = _fma, _ldexp, _sign, _exponent
F = signext = lambda x: x
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

# INF object that supports .f16/.f32/.f64 access
class _Inf:
  f16 = f32 = f64 = float('inf')
  def __neg__(self): return _NegInf()
  def __pos__(self): return self
class _NegInf:
  f16 = f32 = f64 = float('-inf')
  def __neg__(self): return _Inf()
  def __pos__(self): return self
INF = _Inf()

# Rounding mode placeholder
class _RoundMode:
  NEAREST_EVEN = 0
ROUND_MODE = _RoundMode()

# Helper functions for pseudocode
def cvtToQuietNAN(x): return float('nan')
DST = None  # Placeholder, will be set in context

MASK32, MASK64 = 0xffffffff, 0xffffffffffffffff

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

def _brev(v, bits):
  """Bit-reverse a value."""
  result = 0
  for i in range(bits): result |= ((v >> i) & 1) << (bits - 1 - i)
  return result

class SliceProxy:
  """Proxy for D0[31:16] that supports .f16/.u16 etc getters and setters."""
  __slots__ = ('_reg', '_high', '_low', '_reversed')
  def __init__(self, reg, high, low):
    self._reg = reg
    # Handle reversed slices like [0:31] which means bit-reverse
    if high < low: self._high, self._low, self._reversed = low, high, True
    else: self._high, self._low, self._reversed = high, low, False
  def _nbits(self): return self._high - self._low + 1
  def _mask(self): return (1 << self._nbits()) - 1
  def _get(self):
    v = (self._reg._val >> self._low) & self._mask()
    return _brev(v, self._nbits()) if self._reversed else v
  def _set(self, v):
    v = int(v)
    if self._reversed: v = _brev(v, self._nbits())
    self._reg._val = (self._reg._val & ~(self._mask() << self._low)) | ((v & self._mask()) << self._low)

  u8 = property(lambda s: s._get() & 0xff)
  u16 = property(lambda s: s._get() & 0xffff, lambda s, v: s._set(v))
  u32 = property(lambda s: s._get() & MASK32, lambda s, v: s._set(v))
  i16 = property(lambda s: _sext(s._get() & 0xffff, 16), lambda s, v: s._set(v))
  i32 = property(lambda s: _sext(s._get() & MASK32, 32), lambda s, v: s._set(v))
  f16 = property(lambda s: _f16(s._get()), lambda s, v: s._set(v if isinstance(v, int) else _i16(float(v))))
  f32 = property(lambda s: _f32(s._get()), lambda s, v: s._set(_i32(float(v))))
  b16, b32 = u16, u32

  def __int__(self): return self._get()
  def __index__(self): return self._get()

class TypedView:
  """View for S0.u32 that supports [4:0] slicing and [bit] access."""
  __slots__ = ('_reg', '_bits', '_signed', '_float')
  def __init__(self, reg, bits, signed=False, is_float=False):
    self._reg, self._bits, self._signed, self._float = reg, bits, signed, is_float

  @property
  def _val(self):
    mask = MASK64 if self._bits == 64 else MASK32 if self._bits == 32 else (1 << self._bits) - 1
    return self._reg._val & mask

  def __getitem__(self, key):
    if isinstance(key, slice):
      high, low = int(key.start), int(key.stop)
      return SliceProxy(self._reg, high, low)
    return (self._val >> int(key)) & 1

  def __setitem__(self, key, value):
    if isinstance(key, slice):
      high, low = int(key.start), int(key.stop)
      if high < low: high, low, value = low, high, _brev(int(value), low - high + 1)
      mask = (1 << (high - low + 1)) - 1
      self._reg._val = (self._reg._val & ~(mask << low)) | ((int(value) & mask) << low)
    elif value: self._reg._val |= (1 << int(key))
    else: self._reg._val &= ~(1 << int(key))

  def __int__(self): return _sext(self._val, self._bits) if self._signed else self._val
  def __index__(self): return int(self)
  def __trunc__(self): return int(float(self)) if self._float else int(self)
  def __float__(self):
    if self._float:
      return _f16(self._val) if self._bits == 16 else _f32(self._val) if self._bits == 32 else _f64(self._val)
    return float(int(self))

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

  def __bool__(s): return bool(int(s))

class Reg:
  """GPU register: D0.f32 = S0.f32 + S1.f32 just works."""
  __slots__ = ('_val',)
  def __init__(self, val=0): self._val = int(val) & MASK64

  # Typed views
  u64 = property(lambda s: TypedView(s, 64), lambda s, v: setattr(s, '_val', int(v) & MASK64))
  i64 = property(lambda s: TypedView(s, 64, signed=True), lambda s, v: setattr(s, '_val', int(v) & MASK64))
  b64 = property(lambda s: TypedView(s, 64), lambda s, v: setattr(s, '_val', int(v) & MASK64))
  f64 = property(lambda s: TypedView(s, 64, is_float=True), lambda s, v: setattr(s, '_val', v if isinstance(v, int) else _i64(float(v))))
  u32 = property(lambda s: TypedView(s, 32), lambda s, v: setattr(s, '_val', int(v) & MASK32))
  i32 = property(lambda s: TypedView(s, 32, signed=True), lambda s, v: setattr(s, '_val', int(v) & MASK32))
  b32 = property(lambda s: TypedView(s, 32), lambda s, v: setattr(s, '_val', int(v) & MASK32))
  f32 = property(lambda s: TypedView(s, 32, is_float=True), lambda s, v: setattr(s, '_val', _i32(float(v))))
  u24 = property(lambda s: TypedView(s, 24))
  i24 = property(lambda s: TypedView(s, 24, signed=True))
  u16 = property(lambda s: TypedView(s, 16), lambda s, v: setattr(s, '_val', int(v) & 0xffff))
  i16 = property(lambda s: TypedView(s, 16, signed=True), lambda s, v: setattr(s, '_val', int(v) & 0xffff))
  b16 = property(lambda s: TypedView(s, 16), lambda s, v: setattr(s, '_val', int(v) & 0xffff))
  f16 = property(lambda s: TypedView(s, 16, is_float=True), lambda s, v: setattr(s, '_val', v if isinstance(v, int) else _i16(float(v))))
  u8 = property(lambda s: TypedView(s, 8))
  i8 = property(lambda s: TypedView(s, 8, signed=True))

  def __getitem__(s, key):
    if isinstance(key, slice): return SliceProxy(s, int(key.start), int(key.stop))
    return (s._val >> int(key)) & 1

  def __setitem__(s, key, value):
    if isinstance(key, slice):
      high, low = int(key.start), int(key.stop)
      mask = (1 << (high - low + 1)) - 1
      s._val = (s._val & ~(mask << low)) | ((int(value) & mask) << low)
    elif value: s._val |= (1 << int(key))
    else: s._val &= ~(1 << int(key))

  def __int__(s): return s._val
  def __index__(s): return s._val
  def __bool__(s): return bool(s._val)

  # Arithmetic (for tmp = tmp + 1 patterns)
  def __add__(s, o): return s._val + int(o)
  def __radd__(s, o): return int(o) + s._val
  def __sub__(s, o): return s._val - int(o)
  def __rsub__(s, o): return int(o) - s._val
  def __mul__(s, o): return s._val * int(o)
  def __rmul__(s, o): return int(o) * s._val
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

# ═══════════════════════════════════════════════════════════════════════════════
# COMPILER: pseudocode -> Python (minimal transforms)
# ═══════════════════════════════════════════════════════════════════════════════

def compile_pseudocode(pseudocode: str) -> str:
  """Compile pseudocode to Python. Transforms are minimal - most syntax just works."""
  # Join continuation lines (lines ending with || or && or open paren)
  raw_lines = pseudocode.strip().split('\n')
  joined_lines: list[str] = []
  for line in raw_lines:
    line = line.strip()
    if joined_lines and (joined_lines[-1].rstrip().endswith(('||', '&&', '(', ',')) or
                         (joined_lines[-1].count('(') > joined_lines[-1].count(')'))):
      joined_lines[-1] = joined_lines[-1].rstrip() + ' ' + line
    else:
      joined_lines.append(line)

  lines = []
  indent, need_pass = 0, False
  for line in joined_lines:
    line = line.strip()
    if not line or line.startswith('//'): continue

    # Control flow - only need pass before outdent (endif/endfor/else/elsif)
    if line.startswith('if '):
      lines.append('  ' * indent + f"if {_expr(line[3:].rstrip(' then'))}:")
      indent += 1
      need_pass = True
    elif line.startswith('elsif '):
      if need_pass: lines.append('  ' * indent + "pass")
      indent -= 1
      lines.append('  ' * indent + f"elif {_expr(line[6:].rstrip(' then'))}:")
      indent += 1
      need_pass = True
    elif line == 'else':
      if need_pass: lines.append('  ' * indent + "pass")
      indent -= 1
      lines.append('  ' * indent + "else:")
      indent += 1
      need_pass = True
    elif line.startswith('endif'):
      if need_pass: lines.append('  ' * indent + "pass")
      indent -= 1
      need_pass = False
    elif line.startswith('endfor'):
      if need_pass: lines.append('  ' * indent + "pass")
      indent -= 1
      need_pass = False
    elif line.startswith('declare '):
      pass
    elif m := re.match(r'for (\w+) in (.+?)\s*:\s*(.+?) do', line):
      start, end = _expr(m[2].strip()), _expr(m[3].strip())
      lines.append('  ' * indent + f"for {m[1]} in range({start}, int({end})+1):")
      indent += 1
      need_pass = True
    elif '=' in line and not line.startswith('=='):
      need_pass = False
      line = line.rstrip(';')
      # Handle tuple unpacking: { D1.u1, D0.u64 } = expr
      if m := re.match(r'\{\s*D1\.[ui]1\s*,\s*D0\.[ui]64\s*\}\s*=\s*(.+)', line):
        rhs = _expr(m[1])
        lines.append('  ' * indent + f"_full = {rhs}")
        lines.append('  ' * indent + f"D0.u64 = int(_full) & 0xffffffffffffffff")
        lines.append('  ' * indent + f"D1 = Reg((int(_full) >> 64) & 1)")
      # Compound assignment
      elif any(op in line for op in ('+=', '-=', '*=', '/=', '|=', '&=', '^=')):
        for op in ('+=', '-=', '*=', '/=', '|=', '&=', '^='):
          if op in line:
            lhs, rhs = line.split(op, 1)
            lines.append('  ' * indent + f"{lhs.strip()} {op} {_expr(rhs.strip())}")
            break
      else:
        lhs, rhs = line.split('=', 1)
        lines.append('  ' * indent + _assign(lhs.strip(), _expr(rhs.strip())))
  # If we ended with a control statement that needs a body, add pass
  if need_pass: lines.append('  ' * indent + "pass")
  return '\n'.join(lines)

def _assign(lhs: str, rhs: str) -> str:
  """Generate assignment. Bare tmp/SCC/etc get wrapped in Reg()."""
  if lhs in ('tmp', 'SCC', 'VCC', 'EXEC', 'D0', 'D1', 'saveexec'):
    return f"{lhs} = Reg({rhs})"
  return f"{lhs} = {rhs}"

def _expr(e: str) -> str:
  """Expression transform: minimal - just fix syntax differences."""
  e = e.strip()
  e = e.replace('&&', ' and ').replace('||', ' or ').replace('<>', ' != ')
  e = re.sub(r'!([^=])', r' not \1', e)

  # Pack: { hi, lo } -> _pack(hi, lo)
  e = re.sub(r'\{\s*(\w+\.u32)\s*,\s*(\w+\.u32)\s*\}', r'_pack32(\1, \2)', e)
  def pack(m):
    hi, lo = _expr(m[1].strip()), _expr(m[2].strip())
    return f'_pack({hi}, {lo})'
  e = re.sub(r'\{\s*([^,{}]+)\s*,\s*([^,{}]+)\s*\}', pack, e)

  # Literals: 1'0U -> 0, 32'I(x) -> (x), B(x) -> (x)
  e = re.sub(r"\d+'([0-9a-fA-Fx]+)[UuFf]*", r'\1', e)
  e = re.sub(r"\d+'[FIBU]\(", "(", e)
  e = re.sub(r'\bB\(', '(', e)  # Bare B( without digit prefix
  e = re.sub(r'([0-9a-fA-Fx])ULL\b', r'\1', e)
  e = re.sub(r'([0-9a-fA-Fx])LL\b', r'\1', e)
  e = re.sub(r'([0-9a-fA-Fx])U\b', r'\1', e)
  e = re.sub(r'(\d\.?\d*)F\b', r'\1', e)
  # Remove redundant type suffix after lane access: VCC.u64[laneId].u64 -> VCC.u64[laneId]
  e = re.sub(r'(\[laneId\])\.[uib]\d+', r'\1', e)

  # Constants - INF is defined as an object supporting .f32/.f64 access
  e = e.replace('+INF', 'INF').replace('-INF', '(-INF)')
  e = re.sub(r'NAN\.f\d+', 'float("nan")', e)

  # Recursively process bracket contents to handle nested ternaries like S1.u32[x ? a : b]
  def process_brackets(s):
    result, i = [], 0
    while i < len(s):
      if s[i] == '[':
        # Find matching ]
        depth, start = 1, i + 1
        j = start
        while j < len(s) and depth > 0:
          if s[j] == '[': depth += 1
          elif s[j] == ']': depth -= 1
          j += 1
        inner = _expr(s[start:j-1])  # Recursively process bracket content
        result.append('[' + inner + ']')
        i = j
      else:
        result.append(s[i])
        i += 1
    return ''.join(result)
  e = process_brackets(e)

  # Ternary: a ? b : c -> (b if a else c)
  while '?' in e:
    depth, bracket, q = 0, 0, -1
    for i, c in enumerate(e):
      if c == '(': depth += 1
      elif c == ')': depth -= 1
      elif c == '[': bracket += 1
      elif c == ']': bracket -= 1
      elif c == '?' and depth == 0 and bracket == 0: q = i; break
    if q < 0: break
    depth, bracket, col = 0, 0, -1
    for i in range(q + 1, len(e)):
      if e[i] == '(': depth += 1
      elif e[i] == ')': depth -= 1
      elif e[i] == '[': bracket += 1
      elif e[i] == ']': bracket -= 1
      elif e[i] == ':' and depth == 0 and bracket == 0: col = i; break
    if col < 0: break
    cond, t, f = e[:q].strip(), e[q+1:col].strip(), e[col+1:].strip()
    e = f'(({t}) if ({cond}) else ({f}))'
  return e

# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTION CONTEXT
# ═══════════════════════════════════════════════════════════════════════════════

class ExecContext:
  """Context for running compiled pseudocode."""
  def __init__(self, s0=0, s1=0, s2=0, d0=0, scc=0, vcc=0, lane=0, exec_mask=MASK32, literal=0, vgprs=None, src0_idx=0, vdst_idx=0):
    self.S0, self.S1, self.S2 = Reg(s0), Reg(s1), Reg(s2)
    self.D0, self.D1 = Reg(d0), Reg(0)
    self.SCC, self.VCC, self.EXEC = Reg(scc), Reg(vcc), Reg(exec_mask)
    self.tmp, self.saveexec = Reg(0), Reg(exec_mask)
    self.lane, self.laneId, self.literal = lane, lane, literal
    self.SIMM16, self.SIMM32 = Reg(literal), Reg(literal)
    self.VGPR = vgprs if vgprs is not None else {}
    self.SRC0, self.VDST = Reg(src0_idx), Reg(vdst_idx)

  def run(self, code: str):
    """Execute compiled code."""
    # Start with module globals (helpers, aliases), then add instance-specific bindings
    ns = dict(globals())
    ns.update({
      'S0': self.S0, 'S1': self.S1, 'S2': self.S2, 'D0': self.D0, 'D1': self.D1,
      'SCC': self.SCC, 'VCC': self.VCC, 'EXEC': self.EXEC,
      'EXEC_LO': SliceProxy(self.EXEC, 31, 0), 'EXEC_HI': SliceProxy(self.EXEC, 63, 32),
      'tmp': self.tmp, 'saveexec': self.saveexec,
      'lane': self.lane, 'laneId': self.laneId, 'literal': self.literal,
      'SIMM16': self.SIMM16, 'SIMM32': self.SIMM32,
      'VGPR': self.VGPR, 'SRC0': self.SRC0, 'VDST': self.VDST,
    })
    exec(code, ns)
    # Sync rebinds: if register was reassigned to new Reg or value, copy it back
    def _sync(ctx_reg, ns_val):
      if isinstance(ns_val, Reg): ctx_reg._val = ns_val._val
      else: ctx_reg._val = int(ns_val) & MASK64
    if ns.get('SCC') is not self.SCC: _sync(self.SCC, ns['SCC'])
    if ns.get('VCC') is not self.VCC: _sync(self.VCC, ns['VCC'])
    if ns.get('EXEC') is not self.EXEC: _sync(self.EXEC, ns['EXEC'])
    if ns.get('D0') is not self.D0: _sync(self.D0, ns['D0'])
    if ns.get('D1') is not self.D1: _sync(self.D1, ns['D1'])
    if ns.get('tmp') is not self.tmp: _sync(self.tmp, ns['tmp'])
    if ns.get('saveexec') is not self.saveexec: _sync(self.saveexec, ns['saveexec'])

  def result(self) -> dict:
    return {"d0": self.D0._val, "scc": self.SCC._val & 1}

# ═══════════════════════════════════════════════════════════════════════════════
# PDF EXTRACTION AND CODE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

PDF_URL = "https://docs.amd.com/api/khub/documents/UVVZM22UN7tMUeiW_4ShTQ/content"
INST_PATTERN = re.compile(r'^([SV]_[A-Z0-9_]+)\s+(\d+)\s*$', re.M)

# Patterns that can't be handled by the DSL (require special handling in emu.py)
UNSUPPORTED = ['SGPR[', 'V_SWAP', 'eval ', 'BYTE_PERMUTE', 'FATAL_HALT', 'HW_REGISTERS',
               'PC =', 'PC=', 'PC+', '= PC', 'v_sad', '+:', 'vscnt', 'vmcnt', 'expcnt', 'lgkmcnt',
               'CVT_OFF_TABLE', '.bf16', 'ThreadMask', 'u8_to_u32', 'u4_to_u32',
               'S1[i', 'C.i32', 'v_msad_u8', 'S[i]', 'in[', '2.0 / PI',
               'if n.', 'DST.u32', 'addrd = DST', 'addr = DST']  # Malformed pseudocode from PDF

def extract_pseudocode(text: str) -> str | None:
  """Extract pseudocode from an instruction description snippet."""
  lines, result, depth = text.split('\n'), [], 0
  for line in lines:
    s = line.strip()
    if not s: continue
    if re.match(r'^\d+ of \d+$', s): continue
    if re.match(r'^\d+\.\d+\..*Instructions', s): continue
    if s.startswith('"RDNA') or s.startswith('AMD '): continue
    if s.startswith('Notes') or s.startswith('Functional examples'): break
    if s.startswith('if '): depth += 1
    elif s.startswith('endif'): depth = max(0, depth - 1)
    if s.endswith('.') and not any(p in s for p in ['D0', 'D1', 'S0', 'S1', 'S2', 'SCC', 'VCC', 'tmp', '=']): continue
    if re.match(r'^[a-z].*\.$', s) and '=' not in s: continue
    is_code = (
      any(p in s for p in ['D0.', 'D1.', 'S0.', 'S1.', 'S2.', 'SCC =', 'SCC ?', 'VCC', 'EXEC', 'tmp =', 'tmp[', 'lane =']) or
      any(p in s for p in ['D0[', 'D1[', 'S0[', 'S1[', 'S2[']) or
      s.startswith(('if ', 'else', 'elsif', 'endif', 'declare ', 'for ', 'endfor', '//')) or
      re.match(r'^[a-z_]+\s*=', s) or re.match(r'^[a-z_]+\[', s) or (depth > 0 and '=' in s)
    )
    if is_code: result.append(s)
  return '\n'.join(result) if result else None

def parse_pseudocode_from_pdf(pdf_path: str | None = None) -> dict:
  """Parse pseudocode from PDF for all ops. Returns {enum_cls: {op: pseudocode}}."""
  import pdfplumber
  from tinygrad.helpers import fetch
  from extra.assembly.rdna3.autogen import SOP1Op, SOP2Op, SOPCOp, SOPKOp, SOPPOp, VOP1Op, VOP2Op, VOP3Op, VOP3SDOp, VOP3POp, VOPCOp

  OP_ENUMS = [SOP1Op, SOP2Op, SOPCOp, SOPKOp, SOPPOp, VOP1Op, VOP2Op, VOP3Op, VOP3SDOp, VOP3POp, VOPCOp]
  defined_ops = {}
  for enum_cls in OP_ENUMS:
    for op in enum_cls:
      if op.name.startswith(('S_', 'V_')): defined_ops[(op.name, op.value)] = (enum_cls, op)

  pdf = pdfplumber.open(fetch(PDF_URL) if pdf_path is None else pdf_path)
  all_text = '\n'.join(pdf.pages[i].extract_text() or '' for i in range(195, 560))
  matches = list(INST_PATTERN.finditer(all_text))
  instructions: dict = {cls: {} for cls in OP_ENUMS}

  for i, match in enumerate(matches):
    name, opcode = match.group(1), int(match.group(2))
    key = (name, opcode)
    if key not in defined_ops: continue
    enum_cls, enum_val = defined_ops[key]
    start = match.end()
    end = matches[i + 1].start() if i + 1 < len(matches) else start + 2000
    snippet = all_text[start:end].strip()
    if (pseudocode := extract_pseudocode(snippet)): instructions[enum_cls][enum_val] = pseudocode

  return instructions

def generate_dsl_pseudocode(output_path: str = "extra/assembly/rdna3/autogen/dsl_pseudocode.py"):
  """Generate dsl_pseudocode.py - a drop-in replacement for pseudocode.py using the DSL."""
  from pathlib import Path
  from extra.assembly.rdna3.autogen import SOP1Op, SOP2Op, SOPCOp, SOPKOp, SOPPOp, VOP1Op, VOP2Op, VOP3Op, VOP3SDOp, VOP3POp, VOPCOp

  OP_ENUMS = [SOP1Op, SOP2Op, SOPCOp, SOPKOp, SOPPOp, VOP1Op, VOP2Op, VOP3Op, VOP3SDOp, VOP3POp, VOPCOp]

  print("Parsing pseudocode from PDF...")
  by_cls = parse_pseudocode_from_pdf()

  total_found, total_ops = 0, 0
  for enum_cls in OP_ENUMS:
    total = sum(1 for op in enum_cls if op.name.startswith(('S_', 'V_')))
    found = len(by_cls.get(enum_cls, {}))
    total_found += found
    total_ops += total
    print(f"{enum_cls.__name__}: {found}/{total} ({100*found//total if total else 0}%)")
  print(f"Total: {total_found}/{total_ops} ({100*total_found//total_ops}%)")

  print("\nCompiling to DSL functions...")
  lines = ['''# autogenerated by dsl.py - do not edit
# to regenerate: python -m extra.assembly.rdna3.dsl
# ruff: noqa: E501,F405,F403
from extra.assembly.rdna3.autogen import SOP1Op, SOP2Op, SOPCOp, SOPKOp, SOPPOp, VOP1Op, VOP2Op, VOP3Op, VOP3SDOp, VOP3POp, VOPCOp
from extra.assembly.rdna3.dsl import *
''']

  compiled_count, skipped_count = 0, 0

  for enum_cls in OP_ENUMS:
    cls_name = enum_cls.__name__
    pseudocode_dict = by_cls.get(enum_cls, {})
    if not pseudocode_dict: continue

    fn_entries = []
    for op, pc in pseudocode_dict.items():
      if any(p in pc for p in UNSUPPORTED):
        skipped_count += 1
        continue

      try:
        code = compile_pseudocode(pc)
        # Detect flags for result handling
        is_64 = any(p in pc for p in ['D0.u64', 'D0.b64', 'D0.f64', 'D0.i64', 'D1.u64', 'D1.b64', 'D1.f64', 'D1.i64'])
        has_d1 = '{ D1' in pc
        if has_d1: is_64 = True
        is_cmp = cls_name == 'VOPCOp' and 'D0.u64[laneId]' in pc
        is_cmpx = cls_name == 'VOPCOp' and 'EXEC.u64[laneId]' in pc  # V_CMPX writes to EXEC per-lane
        # V_DIV_SCALE passes through S0 if no branch taken
        is_div_scale = 'DIV_SCALE' in op.name
        # VOP3SD instructions that write VCC per-lane (either via VCC.u64[laneId] or by setting VCC = 0/1)
        has_sdst = cls_name == 'VOP3SDOp' and ('VCC.u64[laneId]' in pc or is_div_scale)

        # Generate function with indented body
        fn_name = f"_{cls_name}_{op.name}"
        lines.append(f"def {fn_name}(s0, s1, s2, d0, scc, vcc, lane, exec_mask, literal, VGPR, _vars, src0_idx=0, vdst_idx=0):")
        # Add original pseudocode as comment
        for pc_line in pc.split('\n'):
          lines.append(f"  # {pc_line}")
        # V_DIV_SCALE: D0 defaults to S0 if no branch taken
        if is_div_scale:
          lines.append("  S0, S1, S2, D0, D1 = Reg(s0), Reg(s1), Reg(s2), Reg(s0), Reg(0)")
        else:
          lines.append("  S0, S1, S2, D0, D1 = Reg(s0), Reg(s1), Reg(s2), Reg(d0), Reg(0)")
        lines.append("  SCC, VCC, EXEC = Reg(scc), Reg(vcc), Reg(exec_mask)")
        lines.append("  EXEC_LO, EXEC_HI = SliceProxy(EXEC, 31, 0), SliceProxy(EXEC, 63, 32)")
        lines.append("  tmp, saveexec = Reg(0), Reg(exec_mask)")
        lines.append("  laneId = lane")
        lines.append("  SIMM16, SIMM32 = Reg(literal), Reg(literal)")
        lines.append("  SRC0, VDST = Reg(src0_idx), Reg(vdst_idx)")
        # Add compiled pseudocode with markers
        lines.append("  # --- compiled pseudocode ---")
        for line in code.split('\n'):
          lines.append(f"  {line}")
        lines.append("  # --- end pseudocode ---")
        # Generate result dict
        lines.append("  result = {'d0': D0._val, 'scc': SCC._val & 1}")
        if has_sdst:
          lines.append("  result['vcc_lane'] = (VCC._val >> lane) & 1")
        else:
          lines.append("  if VCC._val != vcc: result['vcc_lane'] = (VCC._val >> lane) & 1")
        if is_cmpx:
          lines.append("  result['exec_lane'] = (EXEC._val >> lane) & 1")
        else:
          lines.append("  if EXEC._val != exec_mask: result['exec'] = EXEC._val")
        if is_cmp:
          lines.append("  result['vcc_lane'] = (D0._val >> lane) & 1")
        if is_64:
          lines.append("  result['d0_64'] = True")
        if has_d1:
          lines.append("  result['d1'] = D1._val & 1")
        lines.append("  return result")
        lines.append("")

        fn_entries.append((op, fn_name))
        compiled_count += 1
      except Exception as e:
        print(f"  Warning: Failed to compile {op.name}: {e}")
        skipped_count += 1

    if fn_entries:
      lines.append(f'{cls_name}_FUNCTIONS = {{')
      for op, fn_name in fn_entries:
        lines.append(f"  {cls_name}.{op.name}: {fn_name},")
      lines.append('}')
      lines.append('')

  # Add manually implemented lane instructions
  lines.append('''
# Manually implemented lane instructions (require special vgpr_write handling)
def _VOP3Op_V_WRITELANE_B32(s0, s1, s2, d0, scc, vcc, lane, exec_mask, literal, VGPR, _vars, src0_idx=0, vdst_idx=0):
  # VGPR[lane][VDST] = S0.b32 - writes s0 to specified lane's VGPR
  wr_lane = s1 & 0x1f  # lane select (5 bits for wave32)
  return {'d0': d0, 'scc': scc, 'vgpr_write': (wr_lane, vdst_idx, s0 & 0xffffffff)}

def _VOP3Op_V_READLANE_B32(s0, s1, s2, d0, scc, vcc, lane, exec_mask, literal, VGPR, _vars, src0_idx=0, vdst_idx=0):
  # D0 = VGPR[lane][SRC0] - reads from specified lane's VGPR
  rd_lane = s1 & 0x1f  # lane select (5 bits for wave32)
  val = VGPR[rd_lane][src0_idx] if VGPR and rd_lane in VGPR and src0_idx in VGPR.get(rd_lane, {}) else s0
  return {'d0': val & 0xffffffff, 'scc': scc}

def _VOP1Op_V_READFIRSTLANE_B32(s0, s1, s2, d0, scc, vcc, lane, exec_mask, literal, VGPR, _vars, src0_idx=0, vdst_idx=0):
  # D0 = VGPR[first_active_lane][SRC0] - reads from first active lane
  first_lane = 0
  for i in range(32):
    if exec_mask & (1 << i):
      first_lane = i
      break
  val = VGPR[first_lane][src0_idx] if VGPR and first_lane in VGPR and src0_idx in VGPR.get(first_lane, {}) else s0
  return {'d0': val & 0xffffffff, 'scc': scc}
''')

  lines.append('COMPILED_FUNCTIONS = {')
  for enum_cls in OP_ENUMS:
    cls_name = enum_cls.__name__
    if by_cls.get(enum_cls): lines.append(f'  {cls_name}: {cls_name}_FUNCTIONS,')
  lines.append('}')
  lines.append('')
  lines.append("# Add lane instructions to their respective dicts")
  lines.append("VOP3Op_FUNCTIONS[VOP3Op.V_WRITELANE_B32] = _VOP3Op_V_WRITELANE_B32")
  lines.append("VOP3Op_FUNCTIONS[VOP3Op.V_READLANE_B32] = _VOP3Op_V_READLANE_B32")
  lines.append("VOP1Op_FUNCTIONS[VOP1Op.V_READFIRSTLANE_B32] = _VOP1Op_V_READFIRSTLANE_B32")
  lines.append('')
  lines.append('def get_compiled_functions(): return COMPILED_FUNCTIONS')

  Path(output_path).write_text('\n'.join(lines))
  print(f"\nGenerated {output_path}: {compiled_count} compiled, {skipped_count} skipped")

if __name__ == "__main__":
  generate_dsl_pseudocode()
