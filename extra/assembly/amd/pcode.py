# DSL for RDNA3 pseudocode - makes pseudocode expressions work directly as Python
import struct, math, re, functools

MASK32, MASK64 = 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF

# Float/int bit conversion functions
_struct_f, _struct_I = struct.Struct("<f"), struct.Struct("<I")
_struct_e, _struct_H = struct.Struct("<e"), struct.Struct("<H")
_struct_d, _struct_Q = struct.Struct("<d"), struct.Struct("<Q")
def _f32(i):
  i = i & MASK32
  # RDNA3 default mode: flush f32 denormals to zero (FTZ)
  # Denormal: exponent=0 (bits 23-30) and mantissa!=0 (bits 0-22)
  if (i & 0x7f800000) == 0 and (i & 0x007fffff) != 0: return 0.0
  return _struct_f.unpack(_struct_I.pack(i))[0]
def _i32(f):
  if isinstance(f, int): f = float(f)
  if math.isnan(f): return 0xffc00000 if math.copysign(1.0, f) < 0 else 0x7fc00000
  if math.isinf(f): return 0x7f800000 if f > 0 else 0xff800000
  try:
    bits = _struct_I.unpack(_struct_f.pack(f))[0]
    # RDNA3 default mode: flush f32 denormals to zero (FTZ)
    if (bits & 0x7f800000) == 0 and (bits & 0x007fffff) != 0: return 0x80000000 if bits & 0x80000000 else 0
    return bits
  except (OverflowError, struct.error): return 0x7f800000 if f > 0 else 0xff800000
def _sext(v, b): return v - (1 << b) if v & (1 << (b - 1)) else v
def _f16(i): return _struct_e.unpack(_struct_H.pack(i & 0xffff))[0]
def _i16(f):
  if math.isnan(f): return 0x7e00
  if math.isinf(f): return 0x7c00 if f > 0 else 0xfc00
  try: return _struct_H.unpack(_struct_e.pack(f))[0]
  except (OverflowError, struct.error): return 0x7c00 if f > 0 else 0xfc00
def _f64(i): return _struct_d.unpack(_struct_Q.pack(i & MASK64))[0]
def _i64(f):
  if math.isnan(f): return 0x7ff8000000000000
  if math.isinf(f): return 0x7ff0000000000000 if f > 0 else 0xfff0000000000000
  try: return _struct_Q.unpack(_struct_d.pack(f))[0]
  except (OverflowError, struct.error): return 0x7ff0000000000000 if f > 0 else 0xfff0000000000000

# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _div(a, b):
  try: return a / b
  except ZeroDivisionError:
    if a == 0.0 or math.isnan(a): return float("nan")
    return math.copysign(float("inf"), a * b) if b == 0.0 else float("inf") if a > 0 else float("-inf")
def _check_nan_type(x, quiet_bit_expected, default):
  try:
    if not math.isnan(float(x)): return False
    if hasattr(x, '_reg') and hasattr(x, '_bits'):
      bits = x._reg._val & ((1 << x._bits) - 1)
      exp_bits, quiet_pos, mant_mask = {16: (0x1f, 9, 0x3ff), 32: (0xff, 22, 0x7fffff), 64: (0x7ff, 51, 0xfffffffffffff)}.get(x._bits, (0,0,0))
      exp_shift = {16: 10, 32: 23, 64: 52}.get(x._bits, 0)
      if exp_bits and ((bits >> exp_shift) & exp_bits) == exp_bits and (bits & mant_mask) != 0:
        return ((bits >> quiet_pos) & 1) == quiet_bit_expected
    return default
  except (TypeError, ValueError): return False
def _gt_neg_zero(a, b): return (a > b) or (a == 0 and b == 0 and not math.copysign(1, a) < 0 and math.copysign(1, b) < 0)
def _lt_neg_zero(a, b): return (a < b) or (a == 0 and b == 0 and math.copysign(1, a) < 0 and not math.copysign(1, b) < 0)
def _fpop(fn):
  def wrapper(x):
    x = float(x)
    if math.isnan(x) or math.isinf(x): return x
    result = float(fn(x))
    return math.copysign(0.0, x) if result == 0.0 else result
  return wrapper
def _f_to_int(f, lo, hi): f = float(f); return 0 if math.isnan(f) else (hi if f >= hi else lo if f <= lo else int(f))
def _f16_to_f32_bits(bits): return struct.unpack("<e", struct.pack("<H", int(bits) & 0xffff))[0]
def _brev(v, bits): return int(bin(v & ((1 << bits) - 1))[2:].zfill(bits)[::-1], 2)
def _ctz(v, bits):
  v, n = int(v) & ((1 << bits) - 1), 0
  if v == 0: return bits
  while (v & 1) == 0: v >>= 1; n += 1
  return n

def _bf16(i):
  """Convert bf16 bits to float. BF16 is just the top 16 bits of f32."""
  return struct.unpack("<f", struct.pack("<I", (i & 0xffff) << 16))[0]
def _ibf16(f):
  """Convert float to bf16 bits (truncate to top 16 bits of f32)."""
  if math.isnan(f): return 0x7fc0  # bf16 quiet NaN
  if math.isinf(f): return 0x7f80 if f > 0 else 0xff80  # bf16 ±infinity
  try: return (struct.unpack("<I", struct.pack("<f", float(f)))[0] >> 16) & 0xffff
  except (OverflowError, struct.error): return 0x7f80 if f > 0 else 0xff80
def _trig(fn, x):
  # V_SIN/COS_F32: hardware does frac on input cycles before computing
  if math.isinf(x) or math.isnan(x): return float("nan")
  frac_cycles = fract(x / (2 * math.pi))
  result = fn(frac_cycles * 2 * math.pi)
  # Hardware returns exactly 0 for cos(π/2), sin(π), etc. due to lookup table
  # Round very small results (below f32 precision) to exactly 0
  if abs(result) < 1e-7: return 0.0
  return result

class _SafeFloat(float):
  """Float subclass that uses _div for division to handle 0/inf correctly."""
  def __truediv__(self, o): return _div(float(self), float(o))
  def __rtruediv__(self, o): return _div(float(o), float(self))

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

class _RoundMode:
  NEAREST_EVEN = 0

class _WaveMode:
  IEEE = False

class _DenormChecker:
  """Comparator for denormalized floats. x == DENORM.f32 checks if x is denormalized."""
  def __init__(self, bits): self._bits = bits
  def _check(self, other):
    f = float(other)
    if math.isinf(f) or math.isnan(f) or f == 0.0: return False
    if self._bits == 64:
      bits = struct.unpack("<Q", struct.pack("<d", f))[0]
      return (bits >> 52) & 0x7ff == 0
    bits = struct.unpack("<I", struct.pack("<f", f))[0]
    return (bits >> 23) & 0xff == 0
  def __eq__(self, other): return self._check(other)
  def __req__(self, other): return self._check(other)
  def __ne__(self, other): return not self._check(other)

class _Denorm:
  f32 = _DenormChecker(32)
  f64 = _DenormChecker(64)

_pack = lambda hi, lo: ((int(hi) & 0xffff) << 16) | (int(lo) & 0xffff)
_pack32 = lambda hi, lo: ((int(hi) & 0xffffffff) << 32) | (int(lo) & 0xffffffff)

class TypedView:
  """View into a Reg with typed access. Used for both full-width (Reg.u32) and slices (Reg[31:16])."""
  __slots__ = ('_reg', '_high', '_low', '_signed', '_float', '_bf16', '_reversed')
  def __init__(self, reg, high, low=0, signed=False, is_float=False, is_bf16=False):
    # Handle reversed slices like [0:31] which means bit-reverse
    if high < low: high, low, reversed = low, high, True
    else: reversed = False
    self._reg, self._high, self._low, self._reversed = reg, high, low, reversed
    self._signed, self._float, self._bf16 = signed, is_float, is_bf16

  def _nbits(self): return self._high - self._low + 1
  def _mask(self): return (1 << self._nbits()) - 1
  def _get(self):
    v = (self._reg._val >> self._low) & self._mask()
    return _brev(v, self._nbits()) if self._reversed else v
  def _set(self, v):
    v = int(v)
    if self._reversed: v = _brev(v, self._nbits())
    self._reg._val = (self._reg._val & ~(self._mask() << self._low)) | ((v & self._mask()) << self._low)

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
  def __lshift__(s, o): n = int(o); return int(s) << n if 0 <= n < 64 or s._nbits() > 64 else 0
  def __rshift__(s, o): n = int(o); return int(s) >> n if 0 <= n < 64 or s._nbits() > 64 else 0
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

# ═══════════════════════════════════════════════════════════════════════════════
# PSEUDOCODE API - Functions and constants from AMD ISA pseudocode
# ═══════════════════════════════════════════════════════════════════════════════

# Rounding and float operations
trunc, floor, ceil = _fpop(math.trunc), _fpop(math.floor), _fpop(math.ceil)
def sqrt(x): return _SafeFloat(math.sqrt(x)) if x >= 0 else _SafeFloat(float("nan"))
def log2(x): return math.log2(x) if x > 0 else (float("-inf") if x == 0 else float("nan"))
def fract(x): return x - math.floor(x)
def sin(x): return _trig(math.sin, x)
def cos(x): return _trig(math.cos, x)
def pow(a, b):
  try: return a ** b
  except OverflowError: return float("inf") if b > 0 else 0.0
def isEven(x):
  x = float(x)
  if math.isinf(x) or math.isnan(x): return False
  return int(x) % 2 == 0
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

# Type conversions
i32_to_f32 = u32_to_f32 = i32_to_f64 = u32_to_f64 = f32_to_f64 = f64_to_f32 = float
def f32_to_i32(f): return _f_to_int(f, -2147483648, 2147483647)
def f32_to_u32(f): return _f_to_int(f, 0, 4294967295)
f64_to_i32, f64_to_u32 = f32_to_i32, f32_to_u32
def f32_to_f16(f):
  f = float(f)
  if math.isnan(f): return 0x7e00  # f16 NaN
  if math.isinf(f): return 0x7c00 if f > 0 else 0xfc00  # f16 ±infinity
  try: return struct.unpack("<H", struct.pack("<e", f))[0]
  except OverflowError: return 0x7c00 if f > 0 else 0xfc00  # overflow -> ±infinity
def f16_to_f32(v): return v if isinstance(v, float) else _f16_to_f32_bits(v)
def i16_to_f16(v): return f32_to_f16(float(_sext(int(v) & 0xffff, 16)))
def u16_to_f16(v): return f32_to_f16(float(int(v) & 0xffff))
def f16_to_i16(bits): f = _f16_to_f32_bits(bits); return max(-32768, min(32767, int(f))) if not math.isnan(f) else 0
def f16_to_u16(bits): f = _f16_to_f32_bits(bits); return max(0, min(65535, int(f))) if not math.isnan(f) else 0
def bf16_to_f32(v): return _bf16(v) if isinstance(v, int) else float(v)
def f32_to_bf16(f): return _ibf16(f)
def u8_to_u32(v): return int(v) & 0xff
def u4_to_u32(v): return int(v) & 0xf
def u32_to_u16(u): return int(u) & 0xffff
def i32_to_i16(i): return ((int(i) + 32768) & 0xffff) - 32768
def f16_to_snorm(f): return max(-32768, min(32767, int(round(max(-1.0, min(1.0, f)) * 32767))))
def f16_to_unorm(f): return max(0, min(65535, int(round(max(0.0, min(1.0, f)) * 65535))))
def f32_to_snorm(f): return max(-32768, min(32767, int(round(max(-1.0, min(1.0, f)) * 32767))))
def f32_to_unorm(f): return max(0, min(65535, int(round(max(0.0, min(1.0, f)) * 65535))))
def v_cvt_i16_f32(f): return max(-32768, min(32767, int(f))) if not math.isnan(f) else 0
def v_cvt_u16_f32(f): return max(0, min(65535, int(f))) if not math.isnan(f) else 0
def SAT8(v): return max(0, min(255, int(v)))
def f32_to_u8(f): return max(0, min(255, int(f))) if not math.isnan(f) else 0

# Min/max operations
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

# SAD/MSAD operations
def ABSDIFF(a, b): return abs(int(a) - int(b))
def v_sad_u8(s0, s1, s2):
  """V_SAD_U8: Sum of absolute differences of 4 byte pairs plus accumulator."""
  s0, s1, s2 = int(s0), int(s1), int(s2)
  result = s2
  for i in range(4):
    a = (s0 >> (i * 8)) & 0xff
    b = (s1 >> (i * 8)) & 0xff
    result += abs(a - b)
  return result & 0xffffffff
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

def BYTE_PERMUTE(data, sel):
  """Select a byte from 64-bit data based on selector value."""
  sel = int(sel) & 0xff
  if sel <= 7: return (int(data) >> (sel * 8)) & 0xff
  if sel == 8: return 0xff if ((int(data) >> 15) & 1) else 0x00
  if sel == 9: return 0xff if ((int(data) >> 31) & 1) else 0x00
  if sel == 10: return 0xff if ((int(data) >> 47) & 1) else 0x00
  if sel == 11: return 0xff if ((int(data) >> 63) & 1) else 0x00
  if sel == 12: return 0x00
  return 0xff

# Pseudocode functions
def s_ff1_i32_b32(v): return _ctz(v, 32)
def s_ff1_i32_b64(v): return _ctz(v, 64)
GT_NEG_ZERO, LT_NEG_ZERO = _gt_neg_zero, _lt_neg_zero
def isNAN(x):
  try: return math.isnan(float(x))
  except (TypeError, ValueError): return False
def isQuietNAN(x): return _check_nan_type(x, 1, True)
def isSignalNAN(x): return _check_nan_type(x, 0, False)
def fma(a, b, c):
  try: return math.fma(a, b, c)
  except ValueError: return float('nan')
def ldexp(m, e): return math.ldexp(m, e)
def sign(f): return 1 if math.copysign(1.0, f) < 0 else 0
def exponent(f):
  if hasattr(f, '_bits') and hasattr(f, '_float') and f._float:
    raw = f._val
    if f._bits == 16: return (raw >> 10) & 0x1f
    if f._bits == 32: return (raw >> 23) & 0xff
    if f._bits == 64: return (raw >> 52) & 0x7ff
  f = float(f)
  if math.isinf(f) or math.isnan(f): return 255
  if f == 0.0: return 0
  try: bits = struct.unpack("<I", struct.pack("<f", f))[0]; return (bits >> 23) & 0xff
  except: return 0
def signext(x): return int(x)
def cvtToQuietNAN(x): return float('nan')

def F(x):
  """32'F(x) or 64'F(x) - interpret x as float. If x is int, treat as bit pattern."""
  if isinstance(x, int): return _f32(x)
  if isinstance(x, TypedView): return x
  return float(x)

# Constants
PI = math.pi
WAVE32, WAVE64 = True, False
OVERFLOW_F32, UNDERFLOW_F32 = float('inf'), 0.0
OVERFLOW_F64, UNDERFLOW_F64 = float('inf'), 0.0
MAX_FLOAT_F32 = 3.4028235e+38
INF = _Inf()
ROUND_MODE = _RoundMode()
WAVE_MODE = _WaveMode()
DENORM = _Denorm()

# 2/PI with 1201 bits of precision for V_TRIG_PREOP_F64
TWO_OVER_PI_1201 = Reg(0x0145f306dc9c882a53f84eafa3ea69bb81b6c52b3278872083fca2c757bd778ac36e48dc74849ba5c00c925dd413a32439fc3bd63962534e7dd1046bea5d768909d338e04d68befc827323ac7306a673e93908bf177bf250763ff12fffbc0b301fde5e2316b414da3eda6cfd9e4f96136e9e8c7ecd3cbfd45aea4f758fd7cbe2f67a0e73ef14a525d4d7f6bf623f1aba10ac06608df8f6)

# ═══════════════════════════════════════════════════════════════════════════════
# COMPILER: pseudocode -> Python (minimal transforms)
# ═══════════════════════════════════════════════════════════════════════════════

def _filter_pseudocode(pseudocode: str) -> str:
  """Filter raw PDF pseudocode to only include actual code lines."""
  pcode_lines, in_lambda, depth = [], 0, 0
  for line in pseudocode.split('\n'):
    s = line.strip()
    if not s: continue
    if '=>' in s or re.match(r'^[A-Z_]+\(', s): continue  # Skip example lines
    if '= lambda(' in s: in_lambda += 1; continue  # Skip lambda definitions
    if in_lambda > 0:
      if s.endswith(');'): in_lambda -= 1
      continue
    # Only include lines that look like pseudocode
    is_code = (any(p in s for p in ['D0.', 'D1.', 'S0.', 'S1.', 'S2.', 'SCC =', 'SCC ?', 'VCC', 'EXEC', 'tmp =', 'tmp[', 'lane =', 'PC =',
                                    'D0[', 'D1[', 'S0[', 'S1[', 'S2[', 'MEM[', 'RETURN_DATA', 'VADDR', 'VDATA', 'VDST', 'SADDR', 'OFFSET']) or
               s.startswith(('if ', 'else', 'elsif', 'endif', 'declare ', 'for ', 'endfor', '//')) or
               re.match(r'^[a-z_]+\s*=', s) or re.match(r'^[a-z_]+\[', s) or (depth > 0 and '=' in s))
    if s.startswith('if '): depth += 1
    elif s.startswith('endif'): depth = max(0, depth - 1)
    if is_code: pcode_lines.append(s)
  return '\n'.join(pcode_lines)

def _compile_pseudocode(pseudocode: str) -> str:
  """Compile pseudocode to Python. Transforms are minimal - most syntax just works."""
  pseudocode = re.sub(r'\bpass\b', 'pass_', pseudocode)  # 'pass' is Python keyword
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
  indent, need_pass, in_first_match_loop = 0, False, False
  for line in joined_lines:
    line = line.split('//')[0].strip()  # Strip C-style comments
    if not line: continue
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
      need_pass, in_first_match_loop = False, False
    elif line.startswith('declare '):
      pass
    elif m := re.match(r'for (\w+) in (.+?)\s*:\s*(.+?) do', line):
      start, end = _expr(m[2].strip()), _expr(m[3].strip())
      lines.append('  ' * indent + f"for {m[1]} in range({start}, int({end})+1):")
      indent += 1
      need_pass, in_first_match_loop = True, True
    elif '=' in line and not line.startswith('=='):
      need_pass = False
      line = line.rstrip(';')
      if m := re.match(r'\{\s*D1\.[ui]1\s*,\s*D0\.[ui]64\s*\}\s*=\s*(.+)', line):
        rhs = _expr(m[1])
        lines.append('  ' * indent + f"_full = {rhs}")
        lines.append('  ' * indent + f"D0.u64 = int(_full) & 0xffffffffffffffff")
        lines.append('  ' * indent + f"D1 = Reg((int(_full) >> 64) & 1)")
      elif any(op in line for op in ('+=', '-=', '*=', '/=', '|=', '&=', '^=')):
        for op in ('+=', '-=', '*=', '/=', '|=', '&=', '^='):
          if op in line:
            lhs, rhs = line.split(op, 1)
            lines.append('  ' * indent + f"{lhs.strip()} {op} {_expr(rhs.strip())}")
            break
      else:
        lhs, rhs = line.split('=', 1)
        lhs_s, rhs_s = _expr(lhs.strip()), rhs.strip()
        stmt = _assign(lhs_s, _expr(rhs_s))
        if in_first_match_loop and rhs_s == 'i' and (lhs_s == 'tmp' or lhs_s == 'D0.i32'):
          stmt += "; break"
        lines.append('  ' * indent + stmt)
  if need_pass: lines.append('  ' * indent + "pass")
  return '\n'.join(lines)

def _assign(lhs: str, rhs: str) -> str:
  if lhs in ('tmp', 'SCC', 'VCC', 'EXEC', 'D0', 'D1', 'saveexec', 'PC'):
    return f"{lhs} = Reg({rhs})"
  return f"{lhs} = {rhs}"

def _expr(e: str) -> str:
  e = e.strip()
  e = e.replace('&&', ' and ').replace('||', ' or ').replace('<>', ' != ')
  e = re.sub(r'!([^=])', r' not \1', e)
  e = re.sub(r'\{\s*(\w+\.u32)\s*,\s*(\w+\.u32)\s*\}', r'_pack32(\1, \2)', e)
  def pack(m):
    hi, lo = _expr(m[1].strip()), _expr(m[2].strip())
    return f'_pack({hi}, {lo})'
  e = re.sub(r'\{\s*([^,{}]+)\s*,\s*([^,{}]+)\s*\}', pack, e)
  e = re.sub(r"1201'B\(2\.0\s*/\s*PI\)", "TWO_OVER_PI_1201", e)
  e = re.sub(r"\d+'([0-9a-fA-Fx]+)[UuFf]*", r'\1', e)
  e = re.sub(r"\d+'[FIBU]\(", "(", e)
  e = re.sub(r'\bB\(', '(', e)
  e = re.sub(r'([0-9a-fA-Fx])ULL\b', r'\1', e)
  e = re.sub(r'([0-9a-fA-Fx])LL\b', r'\1', e)
  e = re.sub(r'([0-9a-fA-Fx])U\b', r'\1', e)
  e = re.sub(r'(\d\.?\d*)F\b', r'\1', e)
  e = re.sub(r'(\[laneId\])\.[uib]\d+', r'\1', e)
  e = e.replace('+INF', 'INF').replace('-INF', '(-INF)')
  e = re.sub(r'NAN\.f\d+', 'float("nan")', e)
  def convert_verilog_slice(m):
    start, width = m.group(1).strip(), m.group(2).strip()
    return f'[({start}) + ({width}) - 1 : ({start})]'
  e = re.sub(r'\[([^:\[\]]+)\s*\+:\s*([^:\[\]]+)\]', convert_verilog_slice, e)
  def process_brackets(s):
    result, i = [], 0
    while i < len(s):
      if s[i] == '[':
        depth, start = 1, i + 1
        j = start
        while j < len(s) and depth > 0:
          if s[j] == '[': depth += 1
          elif s[j] == ']': depth -= 1
          j += 1
        inner = _expr(s[start:j-1])
        result.append('[' + inner + ']')
        i = j
      else:
        result.append(s[i])
        i += 1
    return ''.join(result)
  e = process_brackets(e)
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

def _apply_pseudocode_fixes(op_name: str, code: str) -> str:
  """Apply known fixes for PDF pseudocode bugs."""
  if op_name == 'V_DIV_FMAS_F32':
    code = code.replace('D0.f32 = 2.0 ** 32 * fma(S0.f32, S1.f32, S2.f32)',
                        'D0.f32 = (2.0 ** 64 if exponent(S2.f32) > 127 else 2.0 ** -64) * fma(S0.f32, S1.f32, S2.f32)')
  if op_name == 'V_DIV_FMAS_F64':
    code = code.replace('D0.f64 = 2.0 ** 64 * fma(S0.f64, S1.f64, S2.f64)',
                        'D0.f64 = (2.0 ** 128 if exponent(S2.f64) > 1023 else 2.0 ** -128) * fma(S0.f64, S1.f64, S2.f64)')
  if op_name == 'V_DIV_SCALE_F32':
    code = code.replace('D0.f32 = float("nan")', 'VCC = Reg(1 << laneId); D0.f32 = float("nan")')
    code = code.replace('elif S1.f32 == DENORM.f32:\n  D0.f32 = ldexp(S0.f32, 64)', 'elif False:\n  pass')
    code += '\nif S1.f32 == DENORM.f32:\n  D0.f32 = float("nan")'
    code = code.replace('elif exponent(S2.f32) <= 23:\n  D0.f32 = ldexp(S0.f32, 64)', 'elif exponent(S2.f32) <= 23:\n  VCC = Reg(1 << laneId); D0.f32 = ldexp(S0.f32, 64)')
    code = code.replace('elif S2.f32 / S1.f32 == DENORM.f32:\n  VCC = Reg(0x1)\n  if S0.f32 == S2.f32:\n    D0.f32 = ldexp(S0.f32, 64)', 'elif S2.f32 / S1.f32 == DENORM.f32:\n  VCC = Reg(1 << laneId)')
  if op_name == 'V_DIV_SCALE_F64':
    code = code.replace('D0.f64 = float("nan")', 'VCC = Reg(1 << laneId); D0.f64 = float("nan")')
    code = code.replace('elif S1.f64 == DENORM.f64:\n  D0.f64 = ldexp(S0.f64, 128)', 'elif False:\n  pass')
    code += '\nif S1.f64 == DENORM.f64:\n  D0.f64 = float("nan")'
    code = code.replace('elif exponent(S2.f64) <= 52:\n  D0.f64 = ldexp(S0.f64, 128)', 'elif exponent(S2.f64) <= 52:\n  VCC = Reg(1 << laneId); D0.f64 = ldexp(S0.f64, 128)')
    code = code.replace('elif S2.f64 / S1.f64 == DENORM.f64:\n  VCC = Reg(0x1)\n  if S0.f64 == S2.f64:\n    D0.f64 = ldexp(S0.f64, 128)', 'elif S2.f64 / S1.f64 == DENORM.f64:\n  VCC = Reg(1 << laneId)')
  if op_name == 'V_DIV_FIXUP_F32':
    code = code.replace('D0.f32 = ((-abs(S0.f32)) if (sign_out) else (abs(S0.f32)))',
                        'D0.f32 = ((-OVERFLOW_F32) if (sign_out) else (OVERFLOW_F32)) if isNAN(S0.f32) else ((-abs(S0.f32)) if (sign_out) else (abs(S0.f32)))')
  if op_name == 'V_DIV_FIXUP_F64':
    code = code.replace('D0.f64 = ((-abs(S0.f64)) if (sign_out) else (abs(S0.f64)))',
                        'D0.f64 = ((-OVERFLOW_F64) if (sign_out) else (OVERFLOW_F64)) if isNAN(S0.f64) else ((-abs(S0.f64)) if (sign_out) else (abs(S0.f64)))')
  if op_name == 'V_TRIG_PREOP_F64':
    code = code.replace('result = F((TWO_OVER_PI_1201[1200 : 0] << shift.u32) & 0x1fffffffffffff)',
                        'result = float(((TWO_OVER_PI_1201[1200 : 0] << int(shift)) >> (1201 - 53)) & 0x1fffffffffffff)')
  return code

def _generate_function(cls_name: str, op_name: str, pc: str, code: str) -> str:
  """Generate a single compiled pseudocode function.
  Functions take int parameters and return dict of int values.
  Reg wrapping happens inside the function, only for registers actually used."""
  has_d1 = '{ D1' in pc
  is_cmpx = (cls_name in ('VOPCOp', 'VOP3Op')) and 'EXEC.u64[laneId]' in pc
  is_div_scale = 'DIV_SCALE' in op_name
  has_sdst = cls_name == 'VOP3SDOp' and ('VCC.u64[laneId]' in pc or is_div_scale)
  is_ds = cls_name == 'DSOp'
  is_flat = cls_name in ('FLATOp', 'GLOBALOp', 'SCRATCHOp')
  is_smem = cls_name == 'SMEMOp'
  has_s_array = 'S[i]' in pc  # FMA_MIX style: S[0], S[1], S[2] array access
  combined = code + pc

  fn_name = f"_{cls_name}_{op_name}"

  # Detect which registers are used/modified
  def needs_init(name): return name in combined and not re.search(rf'^\s*{name}\s*=\s*Reg\(', code, re.MULTILINE)
  modifies_d0 = is_div_scale or bool(re.search(r'\bD0\b[.\[]', combined))
  modifies_exec = is_cmpx or bool(re.search(r'EXEC\.(u32|u64|b32|b64)\s*=', combined))
  modifies_vcc = has_sdst or bool(re.search(r'VCC\.(u32|u64|b32|b64)\s*=|VCC\.u64\[laneId\]\s*=', combined))
  modifies_scc = bool(re.search(r'\bSCC\s*=', combined))
  modifies_pc = bool(re.search(r'\bPC\s*=', combined))

  # Build function signature and Reg init lines
  if is_smem:
    lines = [f"def {fn_name}(MEM, addr):"]
    reg_inits = ["ADDR=Reg(addr)", "SDATA=Reg(0)"]
    special_regs = []
  elif is_ds:
    lines = [f"def {fn_name}(MEM, addr, data0, data1, offset0, offset1):"]
    reg_inits = ["ADDR=Reg(addr)", "DATA0=Reg(data0)", "DATA1=Reg(data1)", "OFFSET0=Reg(offset0)", "OFFSET1=Reg(offset1)", "RETURN_DATA=Reg(0)"]
    special_regs = [('DATA', 'DATA0'), ('DATA2', 'DATA1'), ('OFFSET', 'OFFSET0'), ('ADDR_BASE', 'ADDR')]
  elif is_flat:
    lines = [f"def {fn_name}(MEM, addr, vdata, vdst):"]
    reg_inits = ["ADDR=addr", "VDATA=Reg(vdata)", "VDST=Reg(vdst)", "RETURN_DATA=Reg(0)"]
    special_regs = [('DATA', 'VDATA')]
  elif has_s_array:
    # FMA_MIX style: needs S[i] array, opsel, opsel_hi for source selection (neg/neg_hi applied in emu.py before call)
    lines = [f"def {fn_name}(s0, s1, s2, d0, scc, vcc, laneId, exec_mask, literal, VGPR, src0_idx=0, vdst_idx=0, pc=None, opsel=0, opsel_hi=0):"]
    reg_inits = ["S0=Reg(s0)", "S1=Reg(s1)", "S2=Reg(s2)", "S=[S0,S1,S2]", "D0=Reg(d0)", "OPSEL=Reg(opsel)", "OPSEL_HI=Reg(opsel_hi)"]
    special_regs = []
    # Detect array declarations like "declare in : 32'F[3]" and create them (rename 'in' to 'ins' since 'in' is a keyword)
    if "in[" in combined:
      reg_inits.append("ins=[Reg(0),Reg(0),Reg(0)]")
      code = code.replace("in[", "ins[")
  else:
    lines = [f"def {fn_name}(s0, s1, s2, d0, scc, vcc, laneId, exec_mask, literal, VGPR, src0_idx=0, vdst_idx=0, pc=None):"]
    # Only create Regs for registers actually used in the pseudocode
    reg_inits = []
    if 'S0' in combined: reg_inits.append("S0=Reg(s0)")
    if 'S1' in combined: reg_inits.append("S1=Reg(s1)")
    if 'S2' in combined: reg_inits.append("S2=Reg(s2)")
    if modifies_d0 or 'D0' in combined: reg_inits.append("D0=Reg(s0)" if is_div_scale else "D0=Reg(d0)")
    if modifies_scc or 'SCC' in combined: reg_inits.append("SCC=Reg(scc)")
    if modifies_vcc or 'VCC' in combined: reg_inits.append("VCC=Reg(vcc)")
    if modifies_exec or 'EXEC' in combined: reg_inits.append("EXEC=Reg(exec_mask)")
    if modifies_pc or 'PC' in combined: reg_inits.append("PC=Reg(pc) if pc is not None else None")
    special_regs = [('D1', 'Reg(0)'), ('SIMM16', 'Reg(literal)'), ('SIMM32', 'Reg(literal)'),
                    ('SRC0', 'Reg(src0_idx)'), ('VDST', 'Reg(vdst_idx)')]
    if needs_init('tmp'): special_regs.insert(0, ('tmp', 'Reg(0)'))
    if needs_init('saveexec'): special_regs.insert(0, ('saveexec', 'Reg(EXEC._val)'))

  # Build init code
  init_parts = reg_inits.copy()
  for name, init in special_regs:
    if name in combined: init_parts.append(f"{name}={init}")
  if 'EXEC_LO' in code: init_parts.append("EXEC_LO=TypedView(EXEC, 31, 0)")
  if 'EXEC_HI' in code: init_parts.append("EXEC_HI=TypedView(EXEC, 63, 32)")
  if 'VCCZ' in code and not re.search(r'^\s*VCCZ\s*=', code, re.MULTILINE): init_parts.append("VCCZ=Reg(1 if VCC._val == 0 else 0)")
  if 'EXECZ' in code and not re.search(r'^\s*EXECZ\s*=', code, re.MULTILINE): init_parts.append("EXECZ=Reg(1 if EXEC._val == 0 else 0)")

  # Add init line and separator
  if init_parts: lines.append(f"  {'; '.join(init_parts)}")

  # Add compiled pseudocode
  for line in code.split('\n'):
    if line.strip(): lines.append(f"  {line}")

  # Build result dict
  result_items = []
  if modifies_d0: result_items.append("'D0': D0._val")
  if modifies_scc: result_items.append("'SCC': SCC._val")
  if modifies_vcc: result_items.append("'VCC': VCC._val")
  if modifies_exec: result_items.append("'EXEC': EXEC._val")
  if has_d1: result_items.append("'D1': D1._val")
  if modifies_pc: result_items.append("'PC': PC._val")
  if is_smem and 'SDATA' in combined and re.search(r'^\s*SDATA[\.\[].*=', code, re.MULTILINE):
    result_items.append("'SDATA': SDATA._val")
  if is_ds and 'RETURN_DATA' in combined and re.search(r'^\s*RETURN_DATA[\.\[].*=', code, re.MULTILINE):
    result_items.append("'RETURN_DATA': RETURN_DATA._val")
  if is_flat:
    if 'RETURN_DATA' in combined and re.search(r'^\s*RETURN_DATA[\.\[].*=', code, re.MULTILINE):
      result_items.append("'RETURN_DATA': RETURN_DATA._val")
    if re.search(r'^\s*VDATA[\.\[].*=', code, re.MULTILINE):
      result_items.append("'VDATA': VDATA._val")
  lines.append(f"  return {{{', '.join(result_items)}}}")
  return '\n'.join(lines)

# Build the globals dict for exec() - includes all pcode symbols
_PCODE_GLOBALS = {
  'Reg': Reg, 'TypedView': TypedView, '_pack': _pack, '_pack32': _pack32,
  'ABSDIFF': ABSDIFF, 'BYTE_PERMUTE': BYTE_PERMUTE, 'DENORM': DENORM, 'F': F,
  'GT_NEG_ZERO': GT_NEG_ZERO, 'LT_NEG_ZERO': LT_NEG_ZERO, 'INF': INF,
  'MAX_FLOAT_F32': MAX_FLOAT_F32, 'OVERFLOW_F32': OVERFLOW_F32, 'OVERFLOW_F64': OVERFLOW_F64,
  'UNDERFLOW_F32': UNDERFLOW_F32, 'UNDERFLOW_F64': UNDERFLOW_F64,
  'PI': PI, 'ROUND_MODE': ROUND_MODE, 'WAVE_MODE': WAVE_MODE,
  'WAVE32': WAVE32, 'WAVE64': WAVE64, 'TWO_OVER_PI_1201': TWO_OVER_PI_1201,
  'SAT8': SAT8, 'trunc': trunc, 'floor': floor, 'ceil': ceil, 'sqrt': sqrt,
  'log2': log2, 'fract': fract, 'sin': sin, 'cos': cos, 'pow': pow,
  'isEven': isEven, 'mantissa': mantissa, 'signext_from_bit': signext_from_bit,
  'i32_to_f32': i32_to_f32, 'u32_to_f32': u32_to_f32, 'i32_to_f64': i32_to_f64,
  'u32_to_f64': u32_to_f64, 'f32_to_f64': f32_to_f64, 'f64_to_f32': f64_to_f32,
  'f32_to_i32': f32_to_i32, 'f32_to_u32': f32_to_u32, 'f64_to_i32': f64_to_i32,
  'f64_to_u32': f64_to_u32, 'f32_to_f16': f32_to_f16, 'f16_to_f32': f16_to_f32,
  'i16_to_f16': i16_to_f16, 'u16_to_f16': u16_to_f16, 'f16_to_i16': f16_to_i16,
  'f16_to_u16': f16_to_u16, 'bf16_to_f32': bf16_to_f32, 'f32_to_bf16': f32_to_bf16,
  'u8_to_u32': u8_to_u32, 'u4_to_u32': u4_to_u32, 'u32_to_u16': u32_to_u16,
  'i32_to_i16': i32_to_i16, 'f16_to_snorm': f16_to_snorm, 'f16_to_unorm': f16_to_unorm,
  'f32_to_snorm': f32_to_snorm, 'f32_to_unorm': f32_to_unorm,
  'v_cvt_i16_f32': v_cvt_i16_f32, 'v_cvt_u16_f32': v_cvt_u16_f32, 'f32_to_u8': f32_to_u8,
  'v_min_f32': v_min_f32, 'v_max_f32': v_max_f32, 'v_min_f16': v_min_f16, 'v_max_f16': v_max_f16,
  'v_min_i32': v_min_i32, 'v_max_i32': v_max_i32, 'v_min_i16': v_min_i16, 'v_max_i16': v_max_i16,
  'v_min_u32': v_min_u32, 'v_max_u32': v_max_u32, 'v_min_u16': v_min_u16, 'v_max_u16': v_max_u16,
  'v_min3_f32': v_min3_f32, 'v_max3_f32': v_max3_f32, 'v_min3_f16': v_min3_f16, 'v_max3_f16': v_max3_f16,
  'v_min3_i32': v_min3_i32, 'v_max3_i32': v_max3_i32, 'v_min3_i16': v_min3_i16, 'v_max3_i16': v_max3_i16,
  'v_min3_u32': v_min3_u32, 'v_max3_u32': v_max3_u32, 'v_min3_u16': v_min3_u16, 'v_max3_u16': v_max3_u16,
  'v_sad_u8': v_sad_u8, 'v_msad_u8': v_msad_u8,
  's_ff1_i32_b32': s_ff1_i32_b32, 's_ff1_i32_b64': s_ff1_i32_b64,
  'isNAN': isNAN, 'isQuietNAN': isQuietNAN, 'isSignalNAN': isSignalNAN,
  'fma': fma, 'ldexp': ldexp, 'sign': sign, 'exponent': exponent,
  'signext': signext, 'cvtToQuietNAN': cvtToQuietNAN,
}

@functools.cache
def compile_pseudocode(cls_name: str, op_name: str, pseudocode: str):
  """Compile pseudocode string to executable function. Cached for performance."""
  filtered = _filter_pseudocode(pseudocode)
  code = _compile_pseudocode(filtered)
  code = _apply_pseudocode_fixes(op_name, code)
  fn_code = _generate_function(cls_name, op_name, filtered, code)
  fn_name = f"_{cls_name}_{op_name}"
  local_ns = {}
  exec(fn_code, _PCODE_GLOBALS, local_ns)
  return local_ns[fn_name]
