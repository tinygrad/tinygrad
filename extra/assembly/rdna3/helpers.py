# Helper functions for RDNA3 pseudocode emulation
import struct, math

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
def _isnan(x): return math.isnan(x) if isinstance(x, float) else False
def _gt_neg_zero(a, b): return (a > b) or (a == 0 and b == 0 and not math.copysign(1, a) < 0 and math.copysign(1, b) < 0)
def _lt_neg_zero(a, b): return (a < b) or (a == 0 and b == 0 and math.copysign(1, a) < 0 and not math.copysign(1, b) < 0)
def _fma(a, b, c): return a * b + c
def _signext(v): return v
trunc, floor, ceil = math.trunc, math.floor, math.ceil
def sqrt(x): return math.sqrt(x) if x >= 0 else float("nan")
def log2(x): return math.log2(x) if x > 0 else (float("-inf") if x == 0 else float("nan"))
i32_to_f32 = u32_to_f32 = i32_to_f64 = u32_to_f64 = f32_to_f64 = f64_to_f32 = float
def f32_to_i32(f):
  if math.isnan(f): return 0
  if f >= 2147483647: return 2147483647
  if f <= -2147483648: return -2147483648
  return int(f)
def f32_to_u32(f):
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
