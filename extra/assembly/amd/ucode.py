# UOp-based pseudocode compiler for AMD GPU instruction emulation
# Transforms pseudocode -> UOps -> execution via simplify
# Designed for reversible transformation (UOps -> instruction selection)

import re, functools, struct
from tinygrad.uop.ops import UOp, Ops
from tinygrad.dtype import dtypes, DType

# ═══════════════════════════════════════════════════════════════════════════════
# TYPE MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

DTYPE_MAP = {
  'f32': dtypes.float32, 'f16': dtypes.float16, 'f64': dtypes.float64,
  'u32': dtypes.uint32, 'u16': dtypes.uint16, 'u64': dtypes.uint64,
  'i32': dtypes.int32, 'i16': dtypes.int16, 'i64': dtypes.int64,
  'u24': dtypes.uint24, 'i24': dtypes.int24,
  'b32': dtypes.uint32, 'b16': dtypes.uint16, 'b64': dtypes.uint64,
  'u8': dtypes.uint8, 'i8': dtypes.int8,
  'u': dtypes.uint32, 'i': dtypes.int32, 'f': dtypes.float32,  # shorthand types
  'u1': dtypes.uint32, 'i1': dtypes.int32,  # 1-bit as 32-bit
}

def _is_float(dtype: DType) -> bool: return dtype in (dtypes.float16, dtypes.float32, dtypes.float64)

# ═══════════════════════════════════════════════════════════════════════════════
# UOP GRAPH BUILDER (compile time)
# ═══════════════════════════════════════════════════════════════════════════════

class UOpBuilder:
  """Builds a UOp graph from pseudocode expressions at compile time."""
  
  def __init__(self):
    # Create DEFINE_VAR placeholders for inputs
    self.input_vars = {
      'S0': UOp(Ops.DEFINE_VAR, dtypes.uint32, (), ('S0', 0, 0xffffffff)),
      'S1': UOp(Ops.DEFINE_VAR, dtypes.uint32, (), ('S1', 0, 0xffffffff)),
      'S2': UOp(Ops.DEFINE_VAR, dtypes.uint32, (), ('S2', 0, 0xffffffff)),
      'D0': UOp(Ops.DEFINE_VAR, dtypes.uint32, (), ('D0', 0, 0xffffffff)),
      # 64-bit variants for ops that need them
      'S0_64': UOp(Ops.DEFINE_VAR, dtypes.uint64, (), ('S0_64', 0, 0xffffffffffffffff)),
      'S1_64': UOp(Ops.DEFINE_VAR, dtypes.uint64, (), ('S1_64', 0, 0xffffffffffffffff)),
      'S2_64': UOp(Ops.DEFINE_VAR, dtypes.uint64, (), ('S2_64', 0, 0xffffffffffffffff)),
      'D0_64': UOp(Ops.DEFINE_VAR, dtypes.uint64, (), ('D0_64', 0, 0xffffffffffffffff)),
      'SCC': UOp(Ops.DEFINE_VAR, dtypes.uint32, (), ('SCC', 0, 1)),
      'VCC': UOp(Ops.DEFINE_VAR, dtypes.uint64, (), ('VCC', 0, 0xffffffffffffffff)),
      'EXEC': UOp(Ops.DEFINE_VAR, dtypes.uint64, (), ('EXEC', 0, 0xffffffffffffffff)),
      'laneId': UOp(Ops.DEFINE_VAR, dtypes.uint32, (), ('laneId', 0, 31)),
      # Immediate constants for SOPK/literal instructions
      'SIMM16': UOp(Ops.DEFINE_VAR, dtypes.int32, (), ('SIMM16', -32768, 32767)),
      'SIMM32': UOp(Ops.DEFINE_VAR, dtypes.uint32, (), ('SIMM32', 0, 0xffffffff)),
      # Program counter for branch instructions
      'PC': UOp(Ops.DEFINE_VAR, dtypes.uint64, (), ('PC', 0, 0xffffffffffffffff)),
    }
    self.vars: dict[str, UOp] = dict(self.input_vars)
    self.outputs: list[tuple[str, UOp, DType]] = []  # (name, uop, dtype)
  
  def const(self, val, dtype: DType) -> UOp:
    return UOp(Ops.CONST, dtype, (), val)
  
  def cast(self, x: UOp, dtype: DType) -> UOp:
    if x.dtype == dtype: return x
    # BITCAST only works for same-size types, use CAST otherwise
    if dtype.itemsize == x.dtype.itemsize:
      return UOp(Ops.BITCAST, dtype, (x,))
    return UOp(Ops.CAST, dtype, (x,))
  
  def parse_type(self, s: str) -> tuple[str, DType]:
    if '.' in s:
      var, typ = s.rsplit('.', 1)
      return var.strip(), DTYPE_MAP.get(typ, dtypes.uint32)
    return s.strip(), dtypes.uint32
  
  def parse_expr(self, expr: str, dtype_hint: DType = None) -> tuple[UOp, DType]:
    expr = expr.strip()
    # Strip trailing punctuation (period used as sentence end in pseudocode)
    if expr.endswith('.') and not expr[-2:-1].isdigit():
      expr = expr[:-1]

    # Handle parentheses
    if expr.startswith('(') and expr.endswith(')'):
      depth = 0
      for i, c in enumerate(expr):
        if c == '(': depth += 1
        elif c == ')': depth -= 1
        if depth == 0 and i < len(expr) - 1: break
      else:
        return self.parse_expr(expr[1:-1], dtype_hint)
    
    # Handle type cast: 32'I(expr), 64'U(expr), 64'F(expr), 16'F(expr), 1'1U, 1'0U
    # Only match if the cast spans the whole expression
    if m := re.match(r"^(\d+)'([IUFB])\(", expr):
      bits, typ = int(m.group(1)), m.group(2)
      # Find matching closing paren
      start = m.end() - 1  # position of opening paren
      depth, end = 0, start
      for i, c in enumerate(expr[start:], start):
        if c == '(': depth += 1
        elif c == ')': depth -= 1
        if depth == 0: end = i; break
      # Only use this if cast spans entire expression
      if end == len(expr) - 1:
        inner = expr[start+1:end]
        dtype_map = {
          (16, 'I'): dtypes.int16, (16, 'U'): dtypes.uint16, (16, 'F'): dtypes.float16,
          (32, 'I'): dtypes.int32, (32, 'U'): dtypes.uint32, (32, 'F'): dtypes.float32, (32, 'B'): dtypes.uint32,
          (64, 'I'): dtypes.int64, (64, 'U'): dtypes.uint64, (64, 'F'): dtypes.float64, (64, 'B'): dtypes.uint64,
        }
        dtype = dtype_map.get((bits, typ), dtypes.uint32)
        inner_uop, inner_dt = self.parse_expr(inner, dtype)
        # For float casts, use CAST for value conversion
        if typ == 'F':
          return UOp(Ops.CAST, dtype, (inner_uop,)), dtype
        # If inner is already the right size integer, just use it (masking already done by .u24/.i24)
        if inner_dt in (dtypes.uint32, dtypes.int32) and bits == 32: return inner_uop, dtype
        if inner_dt in (dtypes.uint64, dtypes.int64) and bits == 64: return inner_uop, dtype
        return self.cast(inner_uop, dtype), dtype
    if m := re.match(r"^(\d+)'(\d+)([IU])$", expr):
      # Constant like 1'1U or 1'0U
      val = int(m.group(2))
      return self.const(val, dtypes.uint32), dtypes.uint32
    
    # Handle signext(expr) - sign extension
    # Only match if signext spans the whole expression
    if m := re.match(r"^signext\(", expr):
      # Find matching closing paren
      start = m.end() - 1
      depth, end = 0, start
      for i, c in enumerate(expr[start:], start):
        if c == '(': depth += 1
        elif c == ')': depth -= 1
        if depth == 0: end = i; break
      # Only use this if signext spans entire expression
      if end == len(expr) - 1:
        inner = expr[start+1:end]
        inner_uop, inner_dt = self.parse_expr(inner)
        # Sign extend to 64-bit for arithmetic
        return self.cast(inner_uop, dtypes.int64), dtypes.int64

    # Handle function calls: fma, trunc, floor, sqrt, isNAN, abs, etc.
    if m := re.match(r"^(\w+)\(", expr):
      fn_name = m.group(1)
      start = m.end() - 1
      depth, end = 0, start
      for i, c in enumerate(expr[start:], start):
        if c == '(': depth += 1
        elif c == ')': depth -= 1
        if depth == 0: end = i; break
      if end == len(expr) - 1:
        inner = expr[start+1:end]
        # Parse comma-separated arguments
        args = []
        depth = 0
        last = 0
        for i, c in enumerate(inner):
          if c in '([': depth += 1
          elif c in ')]': depth -= 1
          elif c == ',' and depth == 0:
            args.append(inner[last:i].strip())
            last = i + 1
        args.append(inner[last:].strip())
        
        if fn_name == 'fma' and len(args) == 3:
          a, _ = self.parse_expr(args[0], dtype_hint)
          b, _ = self.parse_expr(args[1], dtype_hint)
          c, dt = self.parse_expr(args[2], dtype_hint)
          return UOp(Ops.MULACC, dt, (a, b, c)), dt
        elif fn_name == 'trunc' and len(args) == 1:
          inner_uop, dt = self.parse_expr(args[0], dtype_hint)
          return UOp(Ops.TRUNC, dt, (inner_uop,)), dt
        elif fn_name == 'floor' and len(args) == 1:
          inner_uop, dt = self.parse_expr(args[0], dtype_hint)
          # floor(x) = trunc(x) - (x < 0 and x != trunc(x) ? 1 : 0)
          # For now, use a simpler approach - just trunc for positive, trunc-1 for negative non-integer
          # Actually, Python's math.floor works correctly, so we can use it via constant folding
          # But we need a FLOOR op - for now just use trunc (may be slightly wrong for negative)
          return UOp(Ops.TRUNC, dt, (inner_uop,)), dt  # TODO: proper floor
        elif fn_name == 'sqrt' and len(args) == 1:
          inner_uop, dt = self.parse_expr(args[0], dtype_hint)
          return UOp(Ops.SQRT, dt, (inner_uop,)), dt
        elif fn_name == 'abs' and len(args) == 1:
          inner_uop, dt = self.parse_expr(args[0], dtype_hint)
          # abs(x) = x < 0 ? -x : x
          zero = self.const(0, dt)
          neg = UOp(Ops.NEG, dt, (inner_uop,))
          cond = UOp(Ops.CMPLT, dtypes.bool, (inner_uop, zero))
          return UOp(Ops.WHERE, dt, (cond, neg, inner_uop)), dt
        elif fn_name == 'isNAN' and len(args) == 1:
          inner_uop, dt = self.parse_expr(args[0], dtype_hint)
          # isNAN(x) = x != x
          return UOp(Ops.CMPNE, dtypes.bool, (inner_uop, inner_uop)), dtypes.bool
        elif fn_name == 'isQuietNAN' and len(args) == 1:
          # For now, treat same as isNAN
          inner_uop, dt = self.parse_expr(args[0], dtype_hint)
          return UOp(Ops.CMPNE, dtypes.bool, (inner_uop, inner_uop)), dtypes.bool
        elif fn_name == 'isSignalNAN' and len(args) == 1:
          # For now, return false (signal NaN is rare)
          return self.const(0, dtypes.bool), dtypes.bool
        elif fn_name == 'cvtToQuietNAN' and len(args) == 1:
          # Convert signaling NaN to quiet NaN - for our purposes, just return input
          inner_uop, dt = self.parse_expr(args[0], dtype_hint)
          return inner_uop, dt
        elif fn_name == 'isINF' and len(args) == 1:
          # isINF(x) - check if infinite
          inner_uop, dt = self.parse_expr(args[0], dtype_hint)
          # x == inf or x == -inf, but we need the constants
          # For now, return a placeholder that will work for constant folding
          inf = self.const(float('inf'), dt)
          neg_inf = self.const(float('-inf'), dt)
          is_pos_inf = UOp(Ops.CMPEQ, dtypes.bool, (inner_uop, inf))
          is_neg_inf = UOp(Ops.CMPEQ, dtypes.bool, (inner_uop, neg_inf))
          return UOp(Ops.OR, dtypes.bool, (is_pos_inf, is_neg_inf)), dtypes.bool
        elif fn_name in ('u32_to_f32', 'i32_to_f32', 'f32_to_u32', 'f32_to_i32', 'f16_to_f32', 'f32_to_f16',
                         'f32_to_u8', 'f32_to_i8', 'f32_to_u16', 'f32_to_i16', 'v_cvt_i16_f32', 'v_cvt_u16_f32',
                         'f64_to_i32', 'f64_to_u32', 'i32_to_f64', 'u32_to_f64', 'f64_to_f32', 'f32_to_f64',
                         'f16_to_snorm', 'f16_to_unorm', 'u16_to_f16', 'i16_to_f16', 'f16_to_u16', 'f16_to_i16'):
          # These are VALUE conversions, not bit reinterpretations - always use CAST
          inner_uop, inner_dt = self.parse_expr(args[0])
          if fn_name == 'u32_to_f32':
            return UOp(Ops.CAST, dtypes.float32, (inner_uop,)), dtypes.float32
          elif fn_name == 'i32_to_f32':
            return UOp(Ops.CAST, dtypes.float32, (inner_uop,)), dtypes.float32
          elif fn_name == 'f32_to_u32':
            # Clamp negative to 0, then convert (AMD semantics)
            zero = self.const(0.0, dtypes.float32)
            clamped = UOp(Ops.WHERE, dtypes.float32, (UOp(Ops.CMPLT, dtypes.bool, (inner_uop, zero)), zero, inner_uop))
            return UOp(Ops.CAST, dtypes.uint32, (clamped,)), dtypes.uint32
          elif fn_name == 'f32_to_i32':
            return UOp(Ops.CAST, dtypes.int32, (inner_uop,)), dtypes.int32
          elif fn_name == 'f16_to_f32':
            # f16 -> f32 value conversion
            return UOp(Ops.CAST, dtypes.float32, (inner_uop,)), dtypes.float32
          elif fn_name == 'f32_to_f16':
            # f32 -> f16 value conversion
            return UOp(Ops.CAST, dtypes.float16, (inner_uop,)), dtypes.float16
          elif fn_name == 'f32_to_u8':
            return UOp(Ops.CAST, dtypes.uint8, (inner_uop,)), dtypes.uint8
          elif fn_name == 'f32_to_i8':
            return UOp(Ops.CAST, dtypes.int8, (inner_uop,)), dtypes.int8
          elif fn_name in ('f32_to_u16', 'v_cvt_u16_f32'):
            return UOp(Ops.CAST, dtypes.uint16, (inner_uop,)), dtypes.uint16
          elif fn_name in ('f32_to_i16', 'v_cvt_i16_f32'):
            return UOp(Ops.CAST, dtypes.int16, (inner_uop,)), dtypes.int16
          elif fn_name == 'f64_to_i32':
            return UOp(Ops.CAST, dtypes.int32, (inner_uop,)), dtypes.int32
          elif fn_name == 'f64_to_u32':
            # Clamp negative to 0
            zero = self.const(0.0, dtypes.float64)
            clamped = UOp(Ops.WHERE, dtypes.float64, (UOp(Ops.CMPLT, dtypes.bool, (inner_uop, zero)), zero, inner_uop))
            return UOp(Ops.CAST, dtypes.uint32, (clamped,)), dtypes.uint32
          elif fn_name == 'i32_to_f64':
            return UOp(Ops.CAST, dtypes.float64, (inner_uop,)), dtypes.float64
          elif fn_name == 'u32_to_f64':
            return UOp(Ops.CAST, dtypes.float64, (inner_uop,)), dtypes.float64
          elif fn_name == 'f64_to_f32':
            return UOp(Ops.CAST, dtypes.float32, (inner_uop,)), dtypes.float32
          elif fn_name == 'f32_to_f64':
            return UOp(Ops.CAST, dtypes.float64, (inner_uop,)), dtypes.float64
          elif fn_name == 'f16_to_snorm':
            # Convert f16 to signed normalized i16 (-1.0 to 1.0 -> -32768 to 32767)
            clamped = UOp(Ops.WHERE, inner_uop.dtype, (UOp(Ops.CMPLT, dtypes.bool, (inner_uop, self.const(-1.0, inner_uop.dtype))),
                          self.const(-1.0, inner_uop.dtype), inner_uop))
            clamped = UOp(Ops.WHERE, inner_uop.dtype, (UOp(Ops.CMPLT, dtypes.bool, (self.const(1.0, inner_uop.dtype), clamped)),
                          self.const(1.0, inner_uop.dtype), clamped))
            scaled = UOp(Ops.MUL, inner_uop.dtype, (clamped, self.const(32767.0, inner_uop.dtype)))
            return UOp(Ops.CAST, dtypes.int16, (scaled,)), dtypes.int16
          elif fn_name == 'f16_to_unorm':
            # Convert f16 to unsigned normalized u16 (0.0 to 1.0 -> 0 to 65535)
            clamped = UOp(Ops.WHERE, inner_uop.dtype, (UOp(Ops.CMPLT, dtypes.bool, (inner_uop, self.const(0.0, inner_uop.dtype))),
                          self.const(0.0, inner_uop.dtype), inner_uop))
            clamped = UOp(Ops.WHERE, inner_uop.dtype, (UOp(Ops.CMPLT, dtypes.bool, (self.const(1.0, inner_uop.dtype), clamped)),
                          self.const(1.0, inner_uop.dtype), clamped))
            scaled = UOp(Ops.MUL, inner_uop.dtype, (clamped, self.const(65535.0, inner_uop.dtype)))
            return UOp(Ops.CAST, dtypes.uint16, (scaled,)), dtypes.uint16
          elif fn_name == 'u16_to_f16':
            # Convert u16 integer to f16 float
            return UOp(Ops.CAST, dtypes.float16, (inner_uop,)), dtypes.float16
          elif fn_name == 'i16_to_f16':
            # Convert i16 integer to f16 float
            return UOp(Ops.CAST, dtypes.float16, (inner_uop,)), dtypes.float16
          elif fn_name == 'f16_to_u16':
            # Convert f16 float to u16 integer
            return UOp(Ops.CAST, dtypes.uint16, (inner_uop,)), dtypes.uint16
          elif fn_name == 'f16_to_i16':
            # Convert f16 float to i16 integer
            return UOp(Ops.CAST, dtypes.int16, (inner_uop,)), dtypes.int16
        elif fn_name == 'exp2' and len(args) == 1:
          inner_uop, dt = self.parse_expr(args[0], dtype_hint)
          return UOp(Ops.EXP2, dt, (inner_uop,)), dt
        elif fn_name == 'log2' and len(args) == 1:
          inner_uop, dt = self.parse_expr(args[0], dtype_hint)
          return UOp(Ops.LOG2, dt, (inner_uop,)), dt
        elif fn_name == 'sin' and len(args) == 1:
          inner_uop, dt = self.parse_expr(args[0], dtype_hint)
          return UOp(Ops.SIN, dt, (inner_uop,)), dt
        elif fn_name == 'cos' and len(args) == 1:
          inner_uop, dt = self.parse_expr(args[0], dtype_hint)
          # cos(x) = sin(x + pi/2)
          pi_2 = self.const(1.5707963267948966, dt)
          return UOp(Ops.SIN, dt, (UOp(Ops.ADD, dt, (inner_uop, pi_2)),)), dt
        elif fn_name == 'rcp' and len(args) == 1:
          inner_uop, dt = self.parse_expr(args[0], dtype_hint)
          return UOp(Ops.RECIPROCAL, dt, (inner_uop,)), dt
        elif fn_name == 'rsqrt' and len(args) == 1:
          inner_uop, dt = self.parse_expr(args[0], dtype_hint)
          sqrt_val = UOp(Ops.SQRT, dt, (inner_uop,))
          return UOp(Ops.RECIPROCAL, dt, (sqrt_val,)), dt
        elif fn_name == 'min' and len(args) == 2:
          a, dt = self.parse_expr(args[0], dtype_hint)
          b, _ = self.parse_expr(args[1], dtype_hint)
          # min(a,b) = a < b ? a : b
          cond = UOp(Ops.CMPLT, dtypes.bool, (a, b))
          return UOp(Ops.WHERE, dt, (cond, a, b)), dt
        elif fn_name == 'max' and len(args) == 2:
          a, dt = self.parse_expr(args[0], dtype_hint)
          b, _ = self.parse_expr(args[1], dtype_hint)
          # max(a,b) = a > b ? a : b
          cond = UOp(Ops.CMPLT, dtypes.bool, (b, a))
          return UOp(Ops.WHERE, dt, (cond, a, b)), dt
        elif fn_name == 'clamp' and len(args) == 3:
          x, dt = self.parse_expr(args[0], dtype_hint)
          lo, _ = self.parse_expr(args[1], dtype_hint)
          hi, _ = self.parse_expr(args[2], dtype_hint)
          # clamp(x, lo, hi) = min(max(x, lo), hi)
          cond_lo = UOp(Ops.CMPLT, dtypes.bool, (x, lo))
          max_val = UOp(Ops.WHERE, dt, (cond_lo, lo, x))
          cond_hi = UOp(Ops.CMPLT, dtypes.bool, (hi, max_val))
          return UOp(Ops.WHERE, dt, (cond_hi, hi, max_val)), dt
        elif fn_name == 'signext_from_bit' and len(args) == 2:
          # signext_from_bit(val, width) - sign extend val from width bits to full type
          val_uop, dt = self.parse_expr(args[0], dtype_hint)
          width_uop, _ = self.parse_expr(args[1])
          # Sign bit is at position (width - 1)
          # Sign extend: ((val ^ (1 << (width-1))) - (1 << (width-1)))
          # But we need to handle width=0 case: return 0
          one = self.const(1, dt)
          width_minus_1 = UOp(Ops.SUB, dt, (self.cast(width_uop, dt), one))
          sign_bit = UOp(Ops.SHL, dt, (one, width_minus_1))
          xored = UOp(Ops.XOR, dt, (val_uop, sign_bit))
          result = UOp(Ops.SUB, dt, (xored, sign_bit))
          # If width is 0, return 0
          width_is_zero = UOp(Ops.CMPEQ, dtypes.bool, (width_uop, self.const(0, width_uop.dtype)))
          return UOp(Ops.WHERE, dt, (width_is_zero, self.const(0, dt), result)), dt
        elif fn_name == 'ABSDIFF' and len(args) == 2:
          # ABSDIFF(a, b) = |a - b| for unsigned values
          a, _ = self.parse_expr(args[0])
          b, _ = self.parse_expr(args[1])
          # max(a,b) - min(a,b)
          a_gt_b = UOp(Ops.CMPLT, dtypes.bool, (b, a))
          max_val = UOp(Ops.WHERE, dtypes.uint32, (a_gt_b, self.cast(a, dtypes.uint32), self.cast(b, dtypes.uint32)))
          min_val = UOp(Ops.WHERE, dtypes.uint32, (a_gt_b, self.cast(b, dtypes.uint32), self.cast(a, dtypes.uint32)))
          return UOp(Ops.SUB, dtypes.uint32, (max_val, min_val)), dtypes.uint32
        elif fn_name == 'exponent' and len(args) == 1:
          # exponent(x) - extract IEEE exponent bits from float
          # f16: bits[14:10] (5 bits), f32: bits[30:23] (8 bits), f64: bits[62:52] (11 bits)
          inner_uop, dt = self.parse_expr(args[0], dtype_hint)
          if dt == dtypes.float64:
            bits = UOp(Ops.BITCAST, dtypes.uint64, (inner_uop,))
            exp = UOp(Ops.SHR, dtypes.uint64, (bits, self.const(52, dtypes.uint64)))
            exp = UOp(Ops.AND, dtypes.uint32, (self.cast(exp, dtypes.uint32), self.const(0x7ff, dtypes.uint32)))
          elif dt == dtypes.float16:
            bits = UOp(Ops.BITCAST, dtypes.uint16, (inner_uop,))
            exp = UOp(Ops.SHR, dtypes.uint16, (bits, self.const(10, dtypes.uint16)))
            exp = UOp(Ops.AND, dtypes.uint32, (self.cast(exp, dtypes.uint32), self.const(0x1f, dtypes.uint32)))
          else:  # f32
            bits = UOp(Ops.BITCAST, dtypes.uint32, (inner_uop,))
            exp = UOp(Ops.SHR, dtypes.uint32, (bits, self.const(23, dtypes.uint32)))
            exp = UOp(Ops.AND, dtypes.uint32, (exp, self.const(0xff, dtypes.uint32)))
          return exp, dtypes.uint32
        elif fn_name == 'isEven' and len(args) == 1:
          # isEven(x) - check if integer part is even
          inner_uop, dt = self.parse_expr(args[0], dtype_hint)
          # Cast to int64, check bit 0
          int_val = UOp(Ops.CAST, dtypes.int64, (inner_uop,))
          bit0 = UOp(Ops.AND, dtypes.int64, (int_val, self.const(1, dtypes.int64)))
          return UOp(Ops.CMPEQ, dtypes.bool, (bit0, self.const(0, dtypes.int64))), dtypes.bool
        elif fn_name == 'sign' and len(args) == 1:
          # sign(x) - return 1 if x is negative (sign bit set), 0 otherwise
          inner_uop, dt = self.parse_expr(args[0], dtype_hint)
          if dt == dtypes.float64:
            bits = UOp(Ops.BITCAST, dtypes.uint64, (inner_uop,))
            sign_bit = UOp(Ops.SHR, dtypes.uint64, (bits, self.const(63, dtypes.uint64)))
            return UOp(Ops.AND, dtypes.uint32, (self.cast(sign_bit, dtypes.uint32), self.const(1, dtypes.uint32))), dtypes.uint32
          elif dt == dtypes.float16:
            bits = UOp(Ops.BITCAST, dtypes.uint16, (inner_uop,))
            sign_bit = UOp(Ops.SHR, dtypes.uint16, (bits, self.const(15, dtypes.uint16)))
            return UOp(Ops.AND, dtypes.uint32, (self.cast(sign_bit, dtypes.uint32), self.const(1, dtypes.uint32))), dtypes.uint32
          else:  # f32
            bits = UOp(Ops.BITCAST, dtypes.uint32, (inner_uop,))
            sign_bit = UOp(Ops.SHR, dtypes.uint32, (bits, self.const(31, dtypes.uint32)))
            return UOp(Ops.AND, dtypes.uint32, (sign_bit, self.const(1, dtypes.uint32))), dtypes.uint32
        elif fn_name == 'fract' and len(args) == 1:
          # fract(x) = x - floor(x) = x - trunc(x) for positive, need proper floor for negative
          inner_uop, dt = self.parse_expr(args[0], dtype_hint)
          truncated = UOp(Ops.TRUNC, dt, (inner_uop,))
          return UOp(Ops.SUB, dt, (inner_uop, truncated)), dt
        elif fn_name == 'mantissa' and len(args) == 1:
          # mantissa(x) - extract IEEE mantissa bits from float
          # f16: bits[9:0] (10 bits), f32: bits[22:0] (23 bits), f64: bits[51:0] (52 bits)
          inner_uop, dt = self.parse_expr(args[0], dtype_hint)
          if dt == dtypes.float64:
            bits = UOp(Ops.BITCAST, dtypes.uint64, (inner_uop,))
            mant = UOp(Ops.AND, dtypes.uint64, (bits, self.const(0xfffffffffffff, dtypes.uint64)))
            return mant, dtypes.uint64
          elif dt == dtypes.float16:
            bits = UOp(Ops.BITCAST, dtypes.uint16, (inner_uop,))
            mant = UOp(Ops.AND, dtypes.uint32, (self.cast(bits, dtypes.uint32), self.const(0x3ff, dtypes.uint32)))
            return mant, dtypes.uint32
          else:  # f32
            bits = UOp(Ops.BITCAST, dtypes.uint32, (inner_uop,))
            mant = UOp(Ops.AND, dtypes.uint32, (bits, self.const(0x7fffff, dtypes.uint32)))
            return mant, dtypes.uint32
        elif fn_name == 'pow' and len(args) == 2:
          # pow(base, exp) - when base is 2.0, use exp2
          base, base_dt = self.parse_expr(args[0], dtype_hint)
          exp, _ = self.parse_expr(args[1], dtype_hint)
          result_dt = base_dt if _is_float(base_dt) else dtype_hint or dtypes.float32
          # Check if base is 2.0
          if base.op == Ops.CONST and base.arg == 2.0:
            # For exponent, use CAST (value conversion), not BITCAST
            exp_uop = UOp(Ops.CAST, result_dt, (exp,)) if exp.dtype != result_dt else exp
            return UOp(Ops.EXP2, result_dt, (exp_uop,)), result_dt
          # General case: pow(a, b) = exp2(b * log2(a))
          base_cast = UOp(Ops.CAST, result_dt, (base,)) if base.dtype != result_dt else base
          exp_cast = UOp(Ops.CAST, result_dt, (exp,)) if exp.dtype != result_dt else exp
          log_a = UOp(Ops.LOG2, result_dt, (base_cast,))
          product = UOp(Ops.MUL, result_dt, (exp_cast, log_a))
          return UOp(Ops.EXP2, result_dt, (product,)), result_dt
        elif fn_name == 'LT_NEG_ZERO' and len(args) == 2:
          # LT_NEG_ZERO(a, b) - less than comparison where -0 < +0
          # This differs from IEEE where -0 == +0
          a, dt = self.parse_expr(args[0], dtype_hint)
          b, _ = self.parse_expr(args[1], dtype_hint)
          # Compare as signed integers to make -0 < +0
          if dt == dtypes.float64:
            a_bits = UOp(Ops.BITCAST, dtypes.int64, (a,))
            b_bits = UOp(Ops.BITCAST, dtypes.int64, (b,))
          elif dt == dtypes.float16:
            a_bits = UOp(Ops.BITCAST, dtypes.int16, (a,))
            b_bits = UOp(Ops.BITCAST, dtypes.int16, (b,))
          else:  # f32
            a_bits = UOp(Ops.BITCAST, dtypes.int32, (a,))
            b_bits = UOp(Ops.BITCAST, dtypes.int32, (b,))
          return UOp(Ops.CMPLT, dtypes.bool, (a_bits, b_bits)), dtypes.bool
        elif fn_name == 'GT_NEG_ZERO' and len(args) == 2:
          # GT_NEG_ZERO(a, b) - greater than comparison where -0 < +0
          a, dt = self.parse_expr(args[0], dtype_hint)
          b, _ = self.parse_expr(args[1], dtype_hint)
          # Compare as signed integers
          if dt == dtypes.float64:
            a_bits = UOp(Ops.BITCAST, dtypes.int64, (a,))
            b_bits = UOp(Ops.BITCAST, dtypes.int64, (b,))
          elif dt == dtypes.float16:
            a_bits = UOp(Ops.BITCAST, dtypes.int16, (a,))
            b_bits = UOp(Ops.BITCAST, dtypes.int16, (b,))
          else:  # f32
            a_bits = UOp(Ops.BITCAST, dtypes.int32, (a,))
            b_bits = UOp(Ops.BITCAST, dtypes.int32, (b,))
          return UOp(Ops.CMPLT, dtypes.bool, (b_bits, a_bits)), dtypes.bool
        elif fn_name == 'SAT8' and len(args) == 1:
          # SAT8(x) - saturate to signed 8-bit range [-128, 127]
          inner_uop, dt = self.parse_expr(args[0], dtype_hint)
          # Clamp to [-128, 127]
          lo = self.const(-128, dt)
          hi = self.const(127, dt)
          clamped_lo = UOp(Ops.WHERE, dt, (UOp(Ops.CMPLT, dtypes.bool, (inner_uop, lo)), lo, inner_uop))
          clamped = UOp(Ops.WHERE, dt, (UOp(Ops.CMPLT, dtypes.bool, (hi, clamped_lo)), hi, clamped_lo))
          return clamped, dt
        # v_min/v_max functions - just forward to min/max
        elif fn_name in ('v_min_f32', 'v_min_f16', 'v_min_f64', 'v_min_i32', 'v_min_i16', 'v_min_u32', 'v_min_u16') and len(args) == 2:
          a, dt = self.parse_expr(args[0], dtype_hint)
          b, _ = self.parse_expr(args[1], dtype_hint)
          cond = UOp(Ops.CMPLT, dtypes.bool, (a, b))
          return UOp(Ops.WHERE, dt, (cond, a, b)), dt
        elif fn_name in ('v_max_f32', 'v_max_f16', 'v_max_f64', 'v_max_i32', 'v_max_i16', 'v_max_u32', 'v_max_u16') and len(args) == 2:
          a, dt = self.parse_expr(args[0], dtype_hint)
          b, _ = self.parse_expr(args[1], dtype_hint)
          cond = UOp(Ops.CMPLT, dtypes.bool, (b, a))
          return UOp(Ops.WHERE, dt, (cond, a, b)), dt

    # Handle ternary: cond ? true_val : false_val
    depth = bracket = 0
    q_pos = c_pos = -1
    for i, c in enumerate(expr):
      if c == '(': depth += 1
      elif c == ')': depth -= 1
      elif c == '[': bracket += 1
      elif c == ']': bracket -= 1
      elif c == '?' and depth == 0 and bracket == 0 and q_pos < 0: q_pos = i
      elif c == ':' and depth == 0 and bracket == 0 and q_pos >= 0: c_pos = i; break
    if q_pos > 0 and c_pos > q_pos:
      cond_uop, _ = self.parse_expr(expr[:q_pos].strip())
      true_uop, true_dt = self.parse_expr(expr[q_pos+1:c_pos].strip(), dtype_hint)
      false_uop, false_dt = self.parse_expr(expr[c_pos+1:].strip(), dtype_hint)
      return UOp(Ops.WHERE, true_dt, (cond_uop, true_uop, false_uop)), true_dt
    
    binop_map = {
      '+': Ops.ADD, '-': Ops.SUB, '*': Ops.MUL, '/': Ops.FDIV,
      '&': Ops.AND, '|': Ops.OR, '^': Ops.XOR,
      '<<': Ops.SHL, '>>': Ops.SHR,
      '<': Ops.CMPLT, '==': Ops.CMPEQ, '!=': Ops.CMPNE,
    }
    
    # Handle binary operators (lowest precedence first)
    # Order matters: check longer ops before shorter ones to avoid << matching <
    # ** (exponentiation) is highest precedence among binary ops
    for ops in [('||',), ('&&',), ('==', '!=', '<>', '<=', '>=', '<', '>'), ('|',), ('^',), ('&',), ('<<', '>>'), ('+', '-'), ('*', '/'), ('**',)]:
      depth = bracket = 0
      for i in range(len(expr) - 1, -1, -1):
        c = expr[i]
        if c == ')': depth += 1
        elif c == '(': depth -= 1
        elif c == ']': bracket += 1
        elif c == '[': bracket -= 1
        elif depth == 0 and bracket == 0:
          for op in sorted(ops, key=len, reverse=True):  # longest first
            if expr[i:i+len(op)] == op:
              # Check we're not matching < when we should match << or <=
              if op in ('<', '>') and i + 1 < len(expr) and expr[i+1] in '<>=':
                continue
              if op in ('<', '>') and i > 0 and expr[i-1] in '<>=':
                continue
              # Check we're not matching * when it's part of **
              if op == '*' and i + 1 < len(expr) and expr[i+1] == '*':
                continue
              if op == '*' and i > 0 and expr[i-1] == '*':
                continue
              left_expr = expr[:i].strip()
              right_expr = expr[i+len(op):].strip()
              if not left_expr: continue
              # Skip if this looks like unary - after another operator
              if op == '-' and left_expr and left_expr[-1] in '+-*/(<>=&|^': continue
              left_uop, left_dt = self.parse_expr(left_expr)
              right_uop, right_dt = self.parse_expr(right_expr, left_dt)
              result_dt = left_dt if _is_float(left_dt) else right_dt if _is_float(right_dt) else left_dt
              
              if op == '||':
                one, zero = self.const(1, dtypes.uint32), self.const(0, dtypes.uint32)
                inner = UOp(Ops.WHERE, dtypes.uint32, (right_uop, one, zero))
                return UOp(Ops.WHERE, dtypes.uint32, (left_uop, one, inner)), dtypes.uint32
              if op == '&&':
                one, zero = self.const(1, dtypes.uint32), self.const(0, dtypes.uint32)
                inner = UOp(Ops.WHERE, dtypes.uint32, (right_uop, one, zero))
                return UOp(Ops.WHERE, dtypes.uint32, (left_uop, inner, zero)), dtypes.uint32
              if op == '<>': op = '!='
              
              # Handle comparison ops that don't have direct UOp equivalents
              if op == '>':
                return UOp(Ops.CMPLT, dtypes.bool, (right_uop, left_uop)), dtypes.bool
              if op == '>=':
                lt = UOp(Ops.CMPLT, dtypes.bool, (left_uop, right_uop))
                return UOp(Ops.XOR, dtypes.bool, (lt, self.const(True, dtypes.bool))), dtypes.bool
              if op == '<=':
                lt = UOp(Ops.CMPLT, dtypes.bool, (right_uop, left_uop))
                return UOp(Ops.XOR, dtypes.bool, (lt, self.const(True, dtypes.bool))), dtypes.bool
              
              # Handle ** (exponentiation) - when base is 2.0, use exp2
              if op == '**':
                # 2.0 ** x = exp2(x), 2.0F ** x = exp2(x)
                # Check if left side is 2.0 constant
                if left_uop.op == Ops.CONST and left_uop.arg == 2.0:
                  # For exponent, always use CAST (value conversion), not BITCAST
                  exp_uop = UOp(Ops.CAST, result_dt, (right_uop,)) if right_uop.dtype != result_dt else right_uop
                  return UOp(Ops.EXP2, result_dt, (exp_uop,)), result_dt
                # General case: a ** b = exp2(b * log2(a))
                log_a = UOp(Ops.LOG2, result_dt, (left_uop,))
                exp_uop = UOp(Ops.CAST, result_dt, (right_uop,)) if right_uop.dtype != result_dt else right_uop
                product = UOp(Ops.MUL, result_dt, (exp_uop, log_a))
                return UOp(Ops.EXP2, result_dt, (product,)), result_dt

              uop_op = binop_map.get(op)
              if uop_op is None: raise ValueError(f"Unknown operator: {op}")
              out_dt = dtypes.bool if uop_op in (Ops.CMPLT, Ops.CMPEQ, Ops.CMPNE) else result_dt
              return UOp(uop_op, out_dt, (left_uop, right_uop)), out_dt
    
    # Unary operators
    if expr.startswith('-'):
      val_uop, dt = self.parse_expr(expr[1:])
      return UOp(Ops.NEG, dt, (val_uop,)), dt
    if expr.startswith('~'):
      val_uop, dt = self.parse_expr(expr[1:])
      return UOp(Ops.XOR, dt, (val_uop, self.const(-1, dt))), dt
    if expr.startswith('!'):
      val_uop, dt = self.parse_expr(expr[1:])
      return UOp(Ops.CMPEQ, dtypes.bool, (val_uop, self.const(0, dt))), dtypes.bool
    
    # Bit slice with type suffix: S0[4:0].u32, S0[15:0].f16
    if m := re.match(r'^([A-Za-z][A-Za-z0-9]*)\s*\[\s*(\d+)\s*:\s*(\d+)\s*\]\.([a-z]\d+)$', expr):
      var, high, low, typ = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
      dtype = DTYPE_MAP.get(typ, dtypes.uint32)
      if high < low: high, low = low, high
      base_uop = self.vars.get(var)
      if base_uop is None: raise ValueError(f"Unknown variable: {var}")
      mask = (1 << (high - low + 1)) - 1
      shifted = UOp(Ops.SHR, base_uop.dtype, (base_uop, self.const(low, base_uop.dtype))) if low > 0 else base_uop
      masked = UOp(Ops.AND, dtypes.uint32, (self.cast(shifted, dtypes.uint32), self.const(mask, dtypes.uint32)))
      # For float types, bitcast from integer bits
      if _is_float(dtype):
        if dtype == dtypes.float16:
          return UOp(Ops.BITCAST, dtypes.float16, (self.cast(masked, dtypes.uint16),)), dtype
        return UOp(Ops.BITCAST, dtype, (masked,)), dtype
      return self.cast(masked, dtype), dtype

    # Bit slice with type prefix: S0.u32[31:24], S0.u[5:0]
    if m := re.match(r'^([A-Za-z][A-Za-z0-9]*)\.([a-z]\d*)\s*\[\s*(\d+)\s*:\s*(\d+)\s*\]$', expr):
      var, typ, high, low = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
      if high < low: high, low = low, high
      base_uop = self.vars.get(var)
      if base_uop is None: raise ValueError(f"Unknown variable: {var}")
      mask = (1 << (high - low + 1)) - 1
      shifted = UOp(Ops.SHR, base_uop.dtype, (base_uop, self.const(low, base_uop.dtype))) if low > 0 else base_uop
      masked = UOp(Ops.AND, dtypes.uint32, (self.cast(shifted, dtypes.uint32), self.const(mask, dtypes.uint32)))
      return masked, dtypes.uint32

    # Bit slice with both type prefix and suffix: S0.u32[31:24].u32
    if m := re.match(r'^([A-Za-z][A-Za-z0-9]*)\.([a-z]\d+)\s*\[\s*(\d+)\s*:\s*(\d+)\s*\]\.([a-z]\d+)$', expr):
      var, var_typ, high, low, result_typ = m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), m.group(5)
      if high < low: high, low = low, high
      dtype = DTYPE_MAP.get(result_typ, dtypes.uint32)
      base_uop = self.vars.get(var)
      if base_uop is None: raise ValueError(f"Unknown variable: {var}")
      mask = (1 << (high - low + 1)) - 1
      shifted = UOp(Ops.SHR, base_uop.dtype, (base_uop, self.const(low, base_uop.dtype))) if low > 0 else base_uop
      masked = UOp(Ops.AND, dtypes.uint32, (self.cast(shifted, dtypes.uint32), self.const(mask, dtypes.uint32)))
      return self.cast(masked, dtype), dtype
    
    # Bit slice without type: S0[4:0]
    if m := re.match(r'^([A-Za-z][A-Za-z0-9]*)\s*\[\s*(\d+)\s*:\s*(\d+)\s*\]$', expr):
      var, high, low = m.group(1), int(m.group(2)), int(m.group(3))
      if high < low: high, low = low, high
      base_uop = self.vars.get(var)
      if base_uop is None: raise ValueError(f"Unknown variable: {var}")
      mask = (1 << (high - low + 1)) - 1
      shifted = UOp(Ops.SHR, base_uop.dtype, (base_uop, self.const(low, base_uop.dtype))) if low > 0 else base_uop
      masked = UOp(Ops.AND, dtypes.uint32, (self.cast(shifted, dtypes.uint32), self.const(mask, dtypes.uint32)))
      return masked, dtype_hint or dtypes.uint32
    
    # Bit index with expression: S1.u32[expr] - extract single bit at computed index
    # Handle complex expressions like S1.u32[sign(S0.f32) ? 5 : 6]
    if m := re.match(r'^([A-Za-z][A-Za-z0-9]*)\.([a-z]\d+)\[(.+)\]$', expr):
      var, typ, idx_expr = m.group(1), m.group(2), m.group(3)
      # Check it's not a bit range (digit:digit pattern without ?)
      # Allow expressions containing : if they also have ? (ternary)
      is_bit_range = ':' in idx_expr and '?' not in idx_expr and re.match(r'^\s*\d+\s*:\s*\d+\s*$', idx_expr)
      if not is_bit_range:
        base_uop = self.vars.get(var)
        if base_uop is None: raise ValueError(f"Unknown variable: {var}")
        # Parse idx_expr as an expression
        if idx_expr in self.vars:
          idx_uop = self.vars[idx_expr]
        elif idx_expr.isdigit():
          idx_uop = self.const(int(idx_expr), dtypes.uint32)
        else:
          idx_uop, _ = self.parse_expr(idx_expr)
        # (base >> idx) & 1
        shifted = UOp(Ops.SHR, base_uop.dtype, (base_uop, self.cast(idx_uop, base_uop.dtype)))
        masked = UOp(Ops.AND, dtypes.uint32, (self.cast(shifted, dtypes.uint32), self.const(1, dtypes.uint32)))
        return masked, dtypes.uint32

    # Bit index with variable index AND result type: VCC.u64[laneId].u32 - extract single bit with result type
    if m := re.match(r'^([A-Za-z][A-Za-z0-9]*)\.([a-z]\d+)\[(\w+)\]\.([a-z]\d+)$', expr):
      var, var_typ, idx_expr, result_typ = m.group(1), m.group(2), m.group(3), m.group(4)
      dtype = DTYPE_MAP.get(result_typ, dtypes.uint32)
      base_uop = self.vars.get(var)
      if base_uop is None: raise ValueError(f"Unknown variable: {var}")
      if idx_expr in self.vars:
        idx_uop = self.vars[idx_expr]
      else:
        idx_uop = self.const(int(idx_expr), dtypes.uint32)
      shifted = UOp(Ops.SHR, base_uop.dtype, (base_uop, self.cast(idx_uop, base_uop.dtype)))
      masked = UOp(Ops.AND, dtypes.uint32, (self.cast(shifted, dtypes.uint32), self.const(1, dtypes.uint32)))
      return self.cast(masked, dtype), dtype

    # Bit index with result type: S2.u32[24].u8 - extract single bit with type
    if m := re.match(r'^([A-Za-z][A-Za-z0-9]*)\.([a-z]\d+)\[(\d+)\]\.([a-z]\d+)$', expr):
      var, var_typ, bit_idx, result_typ = m.group(1), m.group(2), int(m.group(3)), m.group(4)
      dtype = DTYPE_MAP.get(result_typ, dtypes.uint32)
      base_uop = self.vars.get(var)
      if base_uop is None: raise ValueError(f"Unknown variable: {var}")
      # (base >> bit_idx) & 1
      shifted = UOp(Ops.SHR, base_uop.dtype, (base_uop, self.const(bit_idx, base_uop.dtype)))
      masked = UOp(Ops.AND, dtypes.uint32, (self.cast(shifted, dtypes.uint32), self.const(1, dtypes.uint32)))
      return self.cast(masked, dtype), dtype

    # Bit index without type: tmp[31], SIMM16.i16[15] - extract single bit
    if m := re.match(r'^([A-Za-z][A-Za-z0-9]*)(?:\.([a-z]\d+))?\[(\d+)\]$', expr):
      var, typ, bit_idx = m.group(1), m.group(2), int(m.group(3))
      base_uop = self.vars.get(var)
      if base_uop is None: raise ValueError(f"Unknown variable: {var}")
      # (base >> bit_idx) & 1
      shifted = UOp(Ops.SHR, base_uop.dtype, (base_uop, self.const(bit_idx, base_uop.dtype)))
      masked = UOp(Ops.AND, dtypes.uint32, (self.cast(shifted, dtypes.uint32), self.const(1, dtypes.uint32)))
      return masked, dtypes.uint32
    
    # Typed variable: S0.f32, S0.u24, S0.i24, S0.f64, S0.f16, EXEC.u64, SIMM16.i16, tmp.u32, etc.
    if m := re.match(r'^([A-Za-z][A-Za-z0-9]*)\.([a-z]\d+)$', expr):
      var, typ = m.group(1), m.group(2)
      dtype = DTYPE_MAP.get(typ, dtypes.uint32)
      # Handle VCCZ and EXECZ specially - they're computed from VCC/EXEC
      if var == 'VCCZ':
        vcc = self.vars.get('VCC')
        return UOp(Ops.CMPEQ, dtypes.bool, (vcc, self.const(0, dtypes.uint64))), dtypes.bool
      if var == 'EXECZ':
        exec_mask = self.vars.get('EXEC')
        return UOp(Ops.CMPEQ, dtypes.bool, (exec_mask, self.const(0, dtypes.uint64))), dtypes.bool
      # For 64-bit types, use the _64 variant of the variable (for input vars only)
      if typ in ('f64', 'u64', 'i64', 'b64') and var.isupper():
        base_uop = self.vars.get(var + '_64')
        if base_uop is None: base_uop = self.vars.get(var)  # fallback
      else:
        base_uop = self.vars.get(var)
      if base_uop is None: raise ValueError(f"Unknown variable: {var}")
      # For 24-bit types, mask to 24 bits
      if typ == 'u24':
        masked = UOp(Ops.AND, dtypes.uint32, (base_uop, self.const(0xffffff, dtypes.uint32)))
        return masked, dtypes.uint32
      if typ == 'i24':
        # Sign-extend from 24 bits: ((x & 0xffffff) ^ 0x800000) - 0x800000
        masked = UOp(Ops.AND, dtypes.uint32, (base_uop, self.const(0xffffff, dtypes.uint32)))
        xored = UOp(Ops.XOR, dtypes.int32, (masked, self.const(0x800000, dtypes.int32)))
        sext = UOp(Ops.SUB, dtypes.int32, (xored, self.const(0x800000, dtypes.int32)))
        return sext, dtypes.int32
      # For float types, bitcast from the integer representation
      if typ in ('f32', 'f64'):
        return UOp(Ops.BITCAST, dtype, (base_uop,)), dtype
      if typ == 'f16':
        # Mask to 16 bits and bitcast to f16
        masked = UOp(Ops.AND, dtypes.uint16, (self.cast(base_uop, dtypes.uint16), self.const(0xffff, dtypes.uint16)))
        return UOp(Ops.BITCAST, dtypes.float16, (masked,)), dtypes.float16
      return self.cast(base_uop, dtype), dtype
    
    # Plain variable
    if expr in self.vars:
      uop = self.vars[expr]
      dtype = dtype_hint or uop.dtype
      return self.cast(uop, dtype), dtype
    
    # Special constants
    import math
    if expr == 'PI': return self.const(math.pi, dtype_hint or dtypes.float64), dtype_hint or dtypes.float64
    if expr in ('INF', '+INF'): return self.const(float('inf'), dtype_hint or dtypes.float64), dtype_hint or dtypes.float64
    if expr == '-INF': return self.const(float('-inf'), dtype_hint or dtypes.float64), dtype_hint or dtypes.float64
    # Mode constants - fixed at compile time for RDNA3
    if expr == 'WAVE_MODE.IEEE': return self.const(1, dtypes.uint32), dtypes.uint32  # IEEE mode enabled
    if expr == 'WAVE32': return self.const(1, dtypes.uint32), dtypes.uint32  # 32-lane wavefront
    if expr == 'WAVE64': return self.const(0, dtypes.uint32), dtypes.uint32  # not 64-lane wavefront
    if expr == 'ROUND_MODE': return self.const(0, dtypes.uint32), dtypes.uint32  # round to nearest even
    # PC is passed as input
    if expr == 'PC': return self.vars.get('PC', self.const(0, dtypes.uint64)), dtypes.uint64
    # VCCZ and EXECZ - zero flags (VCC==0 and EXEC==0)
    if expr == 'VCCZ':
      vcc = self.vars.get('VCC')
      return UOp(Ops.CMPEQ, dtypes.bool, (vcc, self.const(0, dtypes.uint64))), dtypes.bool
    if expr == 'EXECZ':
      exec_mask = self.vars.get('EXEC')
      return UOp(Ops.CMPEQ, dtypes.bool, (exec_mask, self.const(0, dtypes.uint64))), dtypes.bool

    # Width-prefixed constants: 16'4 means 4 as 16-bit, 64'0 means 0 as 64-bit
    if m := re.match(r"^(\d+)'(-?\d+)$", expr):
      bits, val = int(m.group(1)), int(m.group(2))
      dtype_map = {8: dtypes.uint8, 16: dtypes.int16, 32: dtypes.int32, 64: dtypes.int64}
      dtype = dtype_map.get(bits, dtypes.int32)
      return self.const(val, dtype), dtype

    # Numeric literals
    expr_clean = re.sub(r"(\d+)'([0-9a-fA-Fx]+)[UuLlFf]*", r'\2', expr)
    expr_clean = re.sub(r'([0-9a-fA-Fx]+)[UuLlFf]+$', r'\1', expr_clean)
    try:
      if expr_clean.startswith('0x') or expr_clean.startswith('0X'):
        return self.const(int(expr_clean, 16), dtype_hint or dtypes.uint32), dtype_hint or dtypes.uint32
      if '.' in expr_clean or 'e' in expr_clean.lower():
        return self.const(float(expr_clean), dtype_hint or dtypes.float32), dtype_hint or dtypes.float32
      return self.const(int(expr_clean), dtype_hint or dtypes.uint32), dtype_hint or dtypes.uint32
    except ValueError:
      pass
    
    # Handle pack syntax: { hi, lo } -> (hi << N) | lo
    # For .u32/.u16 concatenation: { S0.u32, S1.u32 } -> 64-bit, { S0.u16, S1.u16 } -> 32-bit
    if expr.startswith('{') and expr.endswith('}'):
      inner = expr[1:-1].strip()
      # Find the comma that separates hi and lo
      depth = 0
      comma_pos = -1
      for i, c in enumerate(inner):
        if c in '([{': depth += 1
        elif c in ')]}': depth -= 1
        elif c == ',' and depth == 0:
          comma_pos = i
          break
      if comma_pos > 0:
        hi_expr = inner[:comma_pos].strip()
        lo_expr = inner[comma_pos+1:].strip()
        hi_uop, hi_dt = self.parse_expr(hi_expr)
        lo_uop, lo_dt = self.parse_expr(lo_expr)
        # Determine shift amount based on lo size
        if lo_dt.itemsize >= 4:
          # 32-bit elements -> 64-bit result
          hi_ext = self.cast(hi_uop, dtypes.uint64)
          lo_ext = self.cast(lo_uop, dtypes.uint64)
          hi_shifted = UOp(Ops.SHL, dtypes.uint64, (hi_ext, self.const(32, dtypes.uint64)))
          packed = UOp(Ops.OR, dtypes.uint64, (hi_shifted, lo_ext))
          return packed, dtypes.uint64
        else:
          # 16-bit elements -> 32-bit result
          hi_shifted = UOp(Ops.SHL, dtypes.uint32, (self.cast(hi_uop, dtypes.uint32), self.const(16, dtypes.uint32)))
          lo_masked = UOp(Ops.AND, dtypes.uint32, (self.cast(lo_uop, dtypes.uint32), self.const(0xffff, dtypes.uint32)))
          packed = UOp(Ops.OR, dtypes.uint32, (hi_shifted, lo_masked))
          return packed, dtypes.uint32
    
    raise ValueError(f"Cannot parse expression: {expr}")
  
  def parse_stmt(self, line: str):
    if '=' not in line or any(line.startswith(k) for k in ('if ', 'elsif ', 'for ', 'Set ')):
      return

    # Handle += and -= operators
    if '+=' in line or '-=' in line:
      is_sub = '-=' in line
      lhs, rhs = line.split('-=' if is_sub else '+=', 1)
      lhs, rhs = lhs.strip(), rhs.strip()
      var, dtype = self.parse_type(lhs)
      curr_val = self.vars.get(var)
      if curr_val is None:
        curr_val = self.const(0, dtype)
      inc_uop, _ = self.parse_expr(rhs, dtype)
      # For float types, need to bitcast curr_val before arithmetic
      if _is_float(dtype) and curr_val.dtype != dtype:
        curr_val = UOp(Ops.BITCAST, dtype, (curr_val,))
      op = Ops.SUB if is_sub else Ops.ADD
      result = UOp(op, dtype, (self.cast(curr_val, dtype), self.cast(inc_uop, dtype)))
      # Store back as bits for floats
      if _is_float(dtype):
        result_bits = UOp(Ops.BITCAST, dtypes.uint32 if dtype == dtypes.float32 else dtypes.uint64, (result,))
        self.vars[var] = result_bits
        if var in ('D0', 'D1', 'SCC', 'VCC', 'EXEC', 'PC'):
          self.outputs = [(n, u, d) for n, u, d in self.outputs if n != var]
          self.outputs.append((var, result_bits, dtypes.uint32 if dtype == dtypes.float32 else dtypes.uint64))
      else:
        self.vars[var] = result
        if var in ('D0', 'D1', 'SCC', 'VCC', 'EXEC', 'PC'):
          self.outputs = [(n, u, d) for n, u, d in self.outputs if n != var]
          self.outputs.append((var, result, dtype))
      return

    lhs, rhs = line.split('=', 1)
    lhs = lhs.strip()
    
    # Handle bit set: D0.u64[laneId] = expr or EXEC.u64[laneId] = expr (sets single bit based on condition)
    if m := re.match(r'^([A-Z][A-Z0-9]*)\.([a-z]\d+)\[(\w+)\]$', lhs):
      var, typ, idx_var = m.group(1), m.group(2), m.group(3)
      dtype = DTYPE_MAP.get(typ, dtypes.uint64)
      base_uop = self.vars.get(var)
      idx_uop = self.vars.get(idx_var)
      if base_uop is None: raise ValueError(f"Unknown variable: {var}")
      if idx_uop is None: raise ValueError(f"Unknown index variable: {idx_var}")
      # Parse RHS as condition
      cond_uop, _ = self.parse_expr(rhs.strip())
      # Set bit: (base & ~(1 << idx)) | (cond << idx)
      one = self.const(1, dtype)
      bit_mask = UOp(Ops.SHL, dtype, (one, self.cast(idx_uop, dtype)))
      inv_mask = UOp(Ops.XOR, dtype, (bit_mask, self.const(-1, dtype)))
      cleared = UOp(Ops.AND, dtype, (base_uop, inv_mask))
      cond_ext = self.cast(cond_uop, dtype)
      cond_bit = UOp(Ops.SHL, dtype, (UOp(Ops.AND, dtype, (cond_ext, one)), self.cast(idx_uop, dtype)))
      result = UOp(Ops.OR, dtype, (cleared, cond_bit))
      self.vars[var] = result
      if var in ('D0', 'D1', 'SCC', 'VCC', 'EXEC', 'PC'):
        self.outputs.append((var, result, dtype))
      return
    
    # Handle bit range assignment: D0[31:16].f16 = expr, D0[15:0].f16 = expr, tmp[31:16].i16 = expr
    if m := re.match(r'^([A-Za-z][A-Za-z0-9]*)\s*\[\s*(\d+)\s*:\s*(\d+)\s*\]\.([a-z]\d+)$', lhs):
      var, high, low, typ = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
      if high < low: high, low = low, high
      dtype = DTYPE_MAP.get(typ, dtypes.uint32)
      base_uop = self.vars.get(var)
      if base_uop is None: base_uop = self.const(0, dtypes.uint32)
      rhs_uop, _ = self.parse_expr(rhs.strip(), dtype)
      # For float types, convert to bits
      if _is_float(dtype):
        if dtype == dtypes.float16:
          rhs_bits = UOp(Ops.BITCAST, dtypes.uint16, (rhs_uop,))
          rhs_bits = self.cast(rhs_bits, dtypes.uint32)
        else:
          rhs_bits = UOp(Ops.BITCAST, dtypes.uint32 if dtype == dtypes.float32 else dtypes.uint64, (rhs_uop,))
      else:
        rhs_bits = self.cast(rhs_uop, dtypes.uint32)
      # Create mask and insert bits
      width = high - low + 1
      mask = (1 << width) - 1
      shifted_val = UOp(Ops.SHL, dtypes.uint32, (UOp(Ops.AND, dtypes.uint32, (rhs_bits, self.const(mask, dtypes.uint32))), self.const(low, dtypes.uint32)))
      inv_mask = ~(mask << low) & 0xffffffff
      cleared = UOp(Ops.AND, dtypes.uint32, (self.cast(base_uop, dtypes.uint32), self.const(inv_mask, dtypes.uint32)))
      result = UOp(Ops.OR, dtypes.uint32, (cleared, shifted_val))
      self.vars[var] = result
      if var in ('D0', 'D1', 'SCC', 'VCC'):
        # Only add to outputs if not already there, or replace existing
        self.outputs = [(n, u, d) for n, u, d in self.outputs if n != var]
        self.outputs.append((var, result, dtypes.uint32))
      return

    var, dtype = self.parse_type(lhs)
    rhs_uop, _ = self.parse_expr(rhs.strip(), dtype)
    self.vars[var] = rhs_uop
    # For 64-bit outputs, also update the _64 variant so subsequent reads find the computed value
    if dtype.itemsize == 8 and var in ('D0', 'D1', 'S0', 'S1'):
      self.vars[var + '_64'] = rhs_uop
    if var in ('D0', 'D1', 'SCC', 'VCC', 'EXEC', 'PC'):
      self.outputs.append((var, rhs_uop, dtype))
  
  def build_sink(self) -> UOp:
    """Build a SINK UOp containing all outputs."""
    if not self.outputs:
      return UOp(Ops.SINK, dtypes.void, ())
    return UOp(Ops.SINK, dtypes.void, tuple(uop for _, uop, _ in self.outputs))

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _float_to_bits(val: float, dtype: DType) -> int:
  import math
  if dtype == dtypes.float32:
    return struct.unpack('<I', struct.pack('<f', val))[0]
  elif dtype == dtypes.float16:
    # Handle overflow/underflow for f16
    if math.isnan(val): return 0x7e00  # f16 NaN
    if math.isinf(val): return 0x7c00 if val > 0 else 0xfc00  # f16 +/-inf
    if abs(val) > 65504.0: return 0x7c00 if val > 0 else 0xfc00  # overflow to inf
    if abs(val) < 6.103515625e-05 and val != 0: return 0x0000 if val > 0 else 0x8000  # underflow to zero
    return struct.unpack('<H', struct.pack('<e', val))[0]
  elif dtype == dtypes.float64:
    return struct.unpack('<Q', struct.pack('<d', val))[0]
  return int(val)

# ═══════════════════════════════════════════════════════════════════════════════
# COMPILED FUNCTION GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def _compile_pseudocode(pseudocode: str) -> tuple[UOp, list[tuple[str, DType]], dict[str, UOp]]:
  """Compile pseudocode to UOp graph. Returns (sink, output_info, input_vars)."""
  builder = UOpBuilder()
  lines = [line.split('//')[0].strip().rstrip(';') for line in pseudocode.strip().split('\n')]
  lines = [l for l in lines if l]

  i = 0
  while i < len(lines):
    line = lines[i]

    # Skip declare statements
    if line.startswith('declare '):
      i += 1
      continue

    # Handle for loops: for i in START : END do ... endfor
    if line.startswith('for ') and ' do' in line:
      # Parse: for VAR in START : END do
      m = re.match(r'for\s+(\w+)\s+in\s+(.+?)\s*:\s*(.+?)\s+do', line)
      if m:
        loop_var, start_expr, end_expr = m.group(1), m.group(2).strip(), m.group(3).strip()
        # Parse start and end as constants
        start_val = int(start_expr.replace("'", "").rstrip('U'))
        end_val = int(end_expr.replace("'", "").rstrip('U'))

        # Collect loop body until endfor
        i += 1
        loop_body = []
        depth = 1
        while i < len(lines) and depth > 0:
          if lines[i].startswith('for ') and ' do' in lines[i]:
            depth += 1
          elif lines[i] == 'endfor':
            depth -= 1
          if depth > 0:
            loop_body.append(lines[i])
          i += 1

        # Unroll the loop - process loop body for each iteration
        def expand_line(line, var, val):
          """Substitute loop variable and evaluate bracket expressions."""
          expanded = re.sub(rf'\b{var}\b', str(val), line)
          def eval_bracket(m):
            try: return '[' + str(eval(m.group(1))) + ']'
            except: return m.group(0)
          return re.sub(r'\[([^\]]+)\]', eval_bracket, expanded)

        for loop_val in range(start_val, end_val + 1):
          # Process loop body with index tracking
          body_idx = 0
          while body_idx < len(loop_body):
            body_line = loop_body[body_idx]
            expanded = expand_line(body_line, loop_var, loop_val)

            if expanded.startswith('if ') and ' then' in expanded:
              # Handle if inside loop
              cond_str = expanded[3:expanded.index(' then')].strip()
              cond_uop, _ = builder.parse_expr(cond_str)
              # Collect if body until endif
              body_idx += 1
              if_stmts = []
              while body_idx < len(loop_body) and loop_body[body_idx] != 'endif':
                stmt = expand_line(loop_body[body_idx], loop_var, loop_val)
                if_stmts.append(stmt)
                body_idx += 1
              body_idx += 1  # Skip endif
              # Process if body with condition
              for stmt in if_stmts:
                if '=' not in stmt:
                  continue
                lhs, rhs = stmt.split('=', 1)
                lhs, rhs = lhs.strip(), rhs.strip()
                var, dtype = builder.parse_type(lhs)
                curr_val = builder.vars.get(var)
                if curr_val is None:
                  curr_val = builder.const(0, dtype)
                new_val, _ = builder.parse_expr(rhs, dtype)
                if new_val.dtype != curr_val.dtype:
                  curr_val = builder.cast(curr_val, new_val.dtype)
                result = UOp(Ops.WHERE, new_val.dtype, (cond_uop, new_val, curr_val))
                builder.vars[var] = result
                if var in ('D0', 'D1', 'SCC', 'VCC', 'EXEC', 'PC'):
                  builder.outputs = [(n, u, d) for n, u, d in builder.outputs if n != var]
                  builder.outputs.append((var, result, dtype))
            elif '=' in expanded and not expanded.startswith('if '):
              builder.parse_stmt(expanded)
              body_idx += 1
            else:
              body_idx += 1
        continue

      i += 1
      continue

    # Handle if/elsif/else/endif blocks
    if line.startswith('if ') and ' then' in line:
      # Collect all branches: [(condition_str, body_stmts), ...]
      # Last entry may have condition_str=None for 'else' branch
      branches = []
      cond_str = line[3:line.index(' then')].strip()
      i += 1
      current_body = []

      while i < len(lines) and lines[i] != 'endif':
        if lines[i].startswith('elsif ') and ' then' in lines[i]:
          # Save current branch
          branches.append((cond_str, current_body))
          # Start new elsif branch
          cond_str = lines[i][6:lines[i].index(' then')].strip()
          current_body = []
          i += 1
        elif lines[i] == 'else':
          # Save current branch
          branches.append((cond_str, current_body))
          cond_str = None  # else has no condition
          current_body = []
          i += 1
        else:
          current_body.append(lines[i])
          i += 1

      # Save final branch
      branches.append((cond_str, current_body))

      # Parse all conditions
      parsed_branches = []
      for cond, body in branches:
        cond_uop = builder.parse_expr(cond)[0] if cond else None
        parsed_branches.append((cond_uop, body))

      # Helper to extract assignment info from a statement
      def parse_assignment(stmt):
        if '=' not in stmt or stmt.startswith('if ') or stmt.startswith('for ') or stmt.startswith('elsif '):
          return None
        if '+=' in stmt or '-=' in stmt:
          is_sub = '-=' in stmt
          lhs, rhs = stmt.split('-=' if is_sub else '+=', 1)
          return ('compound', lhs.strip(), rhs.strip(), is_sub)
        lhs, rhs = stmt.split('=', 1)
        return ('simple', lhs.strip(), rhs.strip(), None)

      # Collect all variables that are assigned in any branch
      assigned_vars = set()
      for _, body in parsed_branches:
        for stmt in body:
          info = parse_assignment(stmt)
          if info:
            var, _ = builder.parse_type(info[1])
            assigned_vars.add(var)

      # For each assigned variable, build a nested WHERE chain
      for var in assigned_vars:
        # Get current value as the default (used if no branch assigns)
        var_name, dtype = builder.parse_type(var)
        curr_val = builder.vars.get(var_name)
        if curr_val is None:
          curr_val = builder.const(0, dtype)

        # Build nested WHERE from last branch to first
        # result = cond1 ? val1 : (cond2 ? val2 : (cond3 ? val3 : else_val))
        result = curr_val  # default if no else branch

        # Process branches in reverse order
        for cond_uop, body in reversed(parsed_branches):
          # Find assignment to this var in this branch
          branch_val = None
          for stmt in body:
            info = parse_assignment(stmt)
            if info:
              stmt_var, stmt_dtype = builder.parse_type(info[1])
              if stmt_var == var_name:
                if info[0] == 'compound':
                  # += or -=
                  is_sub = info[3]
                  inc_uop, _ = builder.parse_expr(info[2], stmt_dtype)
                  base = builder.vars.get(var_name, builder.const(0, stmt_dtype))
                  if _is_float(stmt_dtype) and base.dtype != stmt_dtype:
                    base = UOp(Ops.BITCAST, stmt_dtype, (base,))
                  op = Ops.SUB if is_sub else Ops.ADD
                  branch_val = UOp(op, stmt_dtype, (base, inc_uop))
                else:
                  # simple assignment
                  branch_val, _ = builder.parse_expr(info[2], stmt_dtype)
                dtype = stmt_dtype
                break

          if branch_val is not None:
            if cond_uop is None:
              # This is the else branch - it becomes the new default
              result = branch_val
            else:
              # Conditional branch - wrap in WHERE
              if result.dtype != branch_val.dtype:
                result = builder.cast(result, branch_val.dtype)
              result = UOp(Ops.WHERE, branch_val.dtype, (cond_uop, branch_val, result))

        # Store result
        if _is_float(dtype):
          result_bits = UOp(Ops.BITCAST, dtypes.uint32 if dtype == dtypes.float32 else dtypes.uint64, (result,))
          builder.vars[var_name] = result_bits
          if var_name in ('D0', 'D1', 'SCC', 'VCC', 'EXEC', 'PC'):
            builder.outputs = [(n, u, d) for n, u, d in builder.outputs if n != var_name]
            builder.outputs.append((var_name, result_bits, dtypes.uint32 if dtype == dtypes.float32 else dtypes.uint64))
        else:
          builder.vars[var_name] = result
          if var_name in ('D0', 'D1', 'SCC', 'VCC', 'EXEC', 'PC'):
            builder.outputs = [(n, u, d) for n, u, d in builder.outputs if n != var_name]
            builder.outputs.append((var_name, result, dtype))

      i += 1  # Skip endif
      continue

    # Regular statement
    builder.parse_stmt(line)
    i += 1

  sink = builder.build_sink()
  output_info = [(name, dtype) for name, _, dtype in builder.outputs]
  return sink, output_info, builder.input_vars

def _make_uop_fn(sink: UOp, output_info: list[tuple[str, DType]], input_vars: dict[str, UOp]):
  """Create a runtime function that evaluates the UOp graph via simplify."""
  def fn(s0, s1, s2, d0, scc, vcc, laneId, exec_mask, literal, VGPR, src0_idx=0, vdst_idx=0, pc=None):
    # Build substitution map: DEFINE_VAR -> CONST
    # SIMM16 is passed via literal for SOPK instructions - may be unsigned 16-bit, convert to signed
    if literal is not None:
      simm16 = literal if -32768 <= literal <= 32767 else (literal - 65536 if literal < 65536 else 0)
    else:
      simm16 = 0
    dvars = {
      input_vars['S0']: UOp.const(dtypes.uint32, s0 & 0xffffffff),
      input_vars['S1']: UOp.const(dtypes.uint32, s1 & 0xffffffff),
      input_vars['S2']: UOp.const(dtypes.uint32, s2 & 0xffffffff),
      input_vars['D0']: UOp.const(dtypes.uint32, d0 & 0xffffffff),
      input_vars['S0_64']: UOp.const(dtypes.uint64, s0),
      input_vars['S1_64']: UOp.const(dtypes.uint64, s1),
      input_vars['S2_64']: UOp.const(dtypes.uint64, s2),
      input_vars['D0_64']: UOp.const(dtypes.uint64, d0),
      input_vars['SCC']: UOp.const(dtypes.uint32, scc),
      input_vars['VCC']: UOp.const(dtypes.uint64, vcc),
      input_vars['EXEC']: UOp.const(dtypes.uint64, exec_mask),
      input_vars['laneId']: UOp.const(dtypes.uint32, laneId),
      input_vars['SIMM16']: UOp.const(dtypes.int32, simm16),
      input_vars['SIMM32']: UOp.const(dtypes.uint32, literal if literal is not None else 0),
      input_vars['PC']: UOp.const(dtypes.uint64, pc if pc is not None else 0),
    }
    
    # Substitute and simplify all at once
    simplified_sink = sink.substitute(dvars).simplify()
    assert simplified_sink.op == Ops.SINK, f"expected SINK, got {simplified_sink.op}"
    
    result = {}
    for i, (name, dtype) in enumerate(output_info):
      out_uop = simplified_sink.src[i]
      assert out_uop.op == Ops.CONST, f"simplify did not produce CONST for {name}, got {out_uop.op}"
      val = out_uop.arg
      # Convert to bits
      if _is_float(dtype):
        bits = _float_to_bits(val, dtype)
      else:
        bits = int(val) & (0xffffffff if dtype in (dtypes.uint32, dtypes.int32) else 0xffffffffffffffff)
      result[name] = bits
    
    return result
  return fn

# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

# Ops that ucode.py supports - only include ops that compile successfully
# NOTE: Float comparisons using <=, >=, or NOT are excluded due to NaN handling issues
SUPPORTED_OPS: set[str] = {
  # VOP (153 ops)
  'V_ADD3_U32', 'V_ADD_CO_CI_U32', 'V_ADD_CO_U32', 'V_ADD_F16', 'V_ADD_F32', 'V_ADD_F64', 'V_ADD_LSHL_U32', 'V_ADD_NC_I16', 'V_ADD_NC_I32', 'V_ADD_NC_U16', 'V_ADD_NC_U32',
  'V_ALIGNBIT_B32', 'V_ALIGNBYTE_B32', 'V_AND_B16', 'V_AND_B32', 'V_AND_OR_B32', 'V_ASHRREV_I16', 'V_ASHRREV_I32', 'V_ASHRREV_I64',
  'V_BFE_I32', 'V_BFE_U32', 'V_BFI_B32', 'V_BFM_B32',

  'V_CNDMASK_B16', 'V_CNDMASK_B32', 'V_COS_F16', 'V_COS_F32', 'V_CUBEID_F32', 'V_CUBESC_F32',
  'V_CVT_F16_F32', 'V_CVT_F32_F16', 'V_CVT_F32_I32', 'V_CVT_F32_U32',
  'V_CVT_F32_UBYTE0', 'V_CVT_F32_UBYTE1', 'V_CVT_F32_UBYTE2', 'V_CVT_F32_UBYTE3', 'V_CVT_FLOOR_I32_F32',
  'V_CVT_I32_F32', 'V_CVT_I32_I16', 'V_CVT_NEAREST_I32_F32', 'V_CVT_PK_I16_F32', 'V_CVT_PK_U16_F32',
  'V_CVT_PK_U8_F32', 'V_CVT_U32_F32', 'V_CVT_U32_U16',
  'V_DOT2_F16_F16', 'V_DOT2_F32_F16', 'V_DOT2ACC_F32_F16',

  'V_FMA_DX9_ZERO_F32', 'V_FMA_F16', 'V_FMA_F32', 'V_FMA_F64', 'V_FMAAK_F16', 'V_FMAAK_F32',
  'V_FMAC_DX9_ZERO_F32', 'V_FMAC_F16', 'V_FMAC_F32', 'V_FMAMK_F16', 'V_FMAMK_F32',
  'V_FREXP_EXP_I16_F16', 'V_FREXP_EXP_I32_F32', 'V_FREXP_EXP_I32_F64',
  'V_LERP_U8', 'V_LOG_F16', 'V_LOG_F32',
  'V_LSHL_ADD_U32', 'V_LSHL_OR_B32', 'V_LSHLREV_B16', 'V_LSHLREV_B32', 'V_LSHLREV_B64', 'V_LSHRREV_B16', 'V_LSHRREV_B32', 'V_LSHRREV_B64',
  'V_MAD_I16', 'V_MAD_I32_I16', 'V_MAD_I32_I24', 'V_MAD_U16', 'V_MAD_U32_U16', 'V_MAD_U32_U24',
  'V_MAX_I16', 'V_MAX_I32', 'V_MAX_U16', 'V_MAX_U32', 'V_MIN_I16', 'V_MIN_I32', 'V_MIN_U16', 'V_MIN_U32',
  'V_MOV_B16', 'V_MOV_B32', 'V_MSAD_U8', 'V_MUL_DX9_ZERO_F32', 'V_MUL_F16', 'V_MUL_F32', 'V_MUL_F64',
  'V_MUL_HI_I32', 'V_MUL_HI_I32_I24', 'V_MUL_HI_U32', 'V_MUL_HI_U32_U24', 'V_MUL_I32_I24', 'V_MUL_LO_U16', 'V_MUL_LO_U32', 'V_MUL_U32_U24',
  'V_NOT_B16', 'V_NOT_B32', 'V_OR3_B32', 'V_OR_B16', 'V_OR_B32', 'V_PACK_B32_F16', 'V_PK_FMAC_F16', 'V_RCP_F16',
  'V_RCP_F32', 'V_RCP_F64', 'V_RCP_IFLAG_F32', 'V_RSQ_F16', 'V_RSQ_F32', 'V_RSQ_F64',
  'V_PK_ADD_F16', 'V_PK_ADD_I16', 'V_PK_ADD_U16', 'V_PK_ASHRREV_I16', 'V_PK_FMA_F16',
  'V_PK_LSHLREV_B16', 'V_PK_LSHRREV_B16', 'V_PK_MAD_I16', 'V_PK_MAD_U16',
  'V_PK_MAX_I16', 'V_PK_MAX_U16', 'V_PK_MIN_I16', 'V_PK_MIN_U16', 'V_PK_MUL_F16', 'V_PK_MUL_LO_U16',
  'V_PK_SUB_I16', 'V_PK_SUB_U16',
  'V_RNDNE_F16', 'V_RNDNE_F32', 'V_RNDNE_F64',
  'V_SAD_U8', 'V_SAD_U16', 'V_SAD_U32', 'V_SIN_F16', 'V_SIN_F32', 'V_SQRT_F16', 'V_SQRT_F32', 'V_SQRT_F64',
  # Conversions
  'V_CVT_F32_F64', 'V_CVT_F64_F32', 'V_CVT_F64_I32', 'V_CVT_F64_U32', 'V_CVT_I32_F64', 'V_CVT_U32_F64',
  'V_CVT_NORM_I16_F16', 'V_CVT_NORM_U16_F16', 'V_CVT_PK_NORM_I16_F16', 'V_CVT_PK_NORM_U16_F16', 'V_CVT_PK_RTZ_F16_F32',
  'V_SUB_CO_CI_U32', 'V_SUB_CO_U32', 'V_SUB_F16', 'V_SUB_F32', 'V_SUB_NC_I16', 'V_SUB_NC_I32', 'V_SUB_NC_U16', 'V_SUB_NC_U32',
  'V_SUBREV_CO_CI_U32', 'V_SUBREV_CO_U32', 'V_SUBREV_F16', 'V_SUBREV_F32', 'V_SUBREV_NC_U32', 'V_SWAP_B16', 'V_SWAP_B32',
  'V_TRUNC_F16', 'V_TRUNC_F32', 'V_TRUNC_F64', 'V_WRITELANE_B32', 'V_XAD_U32', 'V_XNOR_B32', 'V_XOR3_B32', 'V_XOR_B16', 'V_XOR_B32',
  # Additional VOP ops (newly supported)
  'V_CVT_F16_I16', 'V_CVT_F16_U16', 'V_CVT_I16_F16', 'V_CVT_U16_F16',
  'V_EXP_F16', 'V_EXP_F32',
  'V_LDEXP_F16', 'V_LDEXP_F32', 'V_LDEXP_F64',
  'V_CUBEMA_F32', 'V_CUBETC_F32',
  'V_SAT_PK_U8_I16',
  # min3/max3/minmax/maxmin ops
  'V_MAX3_I16', 'V_MAX3_I32', 'V_MAX3_U16', 'V_MAX3_U32',
  'V_MIN3_I16', 'V_MIN3_I32', 'V_MIN3_U16', 'V_MIN3_U32',
  'V_MAXMIN_I32', 'V_MAXMIN_U32', 'V_MINMAX_I32', 'V_MINMAX_U32',
  # VOPC - integer and float comparisons (112 ops)
  'V_CMP_EQ_F16', 'V_CMP_EQ_F32', 'V_CMP_EQ_F64', 'V_CMP_EQ_I16', 'V_CMP_EQ_I32', 'V_CMP_EQ_I64', 'V_CMP_EQ_U16', 'V_CMP_EQ_U32',
  'V_CMP_EQ_U64', 'V_CMP_F_F16', 'V_CMP_F_F32', 'V_CMP_F_F64', 'V_CMP_F_I32', 'V_CMP_F_I64', 'V_CMP_F_U32', 'V_CMP_F_U64',
  'V_CMP_GE_F16', 'V_CMP_GE_F32', 'V_CMP_GE_F64', 'V_CMP_GE_I16', 'V_CMP_GE_I32', 'V_CMP_GE_I64', 'V_CMP_GE_U16', 'V_CMP_GE_U32', 'V_CMP_GE_U64',
  'V_CMP_GT_F16', 'V_CMP_GT_F32', 'V_CMP_GT_F64', 'V_CMP_GT_I16', 'V_CMP_GT_I32', 'V_CMP_GT_I64', 'V_CMP_GT_U16', 'V_CMP_GT_U32', 'V_CMP_GT_U64',
  'V_CMP_LE_F16', 'V_CMP_LE_F32', 'V_CMP_LE_F64', 'V_CMP_LE_I16', 'V_CMP_LE_I32', 'V_CMP_LE_I64', 'V_CMP_LE_U16', 'V_CMP_LE_U32', 'V_CMP_LE_U64',
  'V_CMP_LG_F16', 'V_CMP_LG_F32', 'V_CMP_LG_F64',
  'V_CMP_LT_F16', 'V_CMP_LT_F32', 'V_CMP_LT_F64', 'V_CMP_LT_I16', 'V_CMP_LT_I32', 'V_CMP_LT_I64', 'V_CMP_LT_U16', 'V_CMP_LT_U32', 'V_CMP_LT_U64',
  'V_CMP_NE_I16', 'V_CMP_NE_I32', 'V_CMP_NE_I64', 'V_CMP_NE_U16', 'V_CMP_NE_U32', 'V_CMP_NE_U64',
  'V_CMP_NEQ_F16', 'V_CMP_NEQ_F32', 'V_CMP_NEQ_F64',
  'V_CMP_NGE_F16', 'V_CMP_NGE_F32', 'V_CMP_NGE_F64', 'V_CMP_NGT_F16', 'V_CMP_NGT_F32', 'V_CMP_NGT_F64',
  'V_CMP_NLE_F16', 'V_CMP_NLE_F32', 'V_CMP_NLE_F64', 'V_CMP_NLG_F16', 'V_CMP_NLG_F32', 'V_CMP_NLG_F64',
  'V_CMP_NLT_F16', 'V_CMP_NLT_F32', 'V_CMP_NLT_F64',
  'V_CMP_O_F16', 'V_CMP_O_F32', 'V_CMP_O_F64', 'V_CMP_T_F16', 'V_CMP_T_F32', 'V_CMP_T_F64',
  'V_CMP_T_I32', 'V_CMP_T_I64', 'V_CMP_T_U32', 'V_CMP_T_U64', 'V_CMP_U_F16', 'V_CMP_U_F32', 'V_CMP_U_F64',
  # VOPCX - compare and write exec (112 ops)
  'V_CMPX_EQ_F16', 'V_CMPX_EQ_F32', 'V_CMPX_EQ_F64', 'V_CMPX_EQ_I16', 'V_CMPX_EQ_I32', 'V_CMPX_EQ_I64', 'V_CMPX_EQ_U16', 'V_CMPX_EQ_U32',
  'V_CMPX_EQ_U64', 'V_CMPX_F_F16', 'V_CMPX_F_F32', 'V_CMPX_F_F64', 'V_CMPX_F_I32', 'V_CMPX_F_I64', 'V_CMPX_F_U32', 'V_CMPX_F_U64',
  'V_CMPX_GE_F16', 'V_CMPX_GE_F32', 'V_CMPX_GE_F64', 'V_CMPX_GE_I16', 'V_CMPX_GE_I32', 'V_CMPX_GE_I64', 'V_CMPX_GE_U16', 'V_CMPX_GE_U32', 'V_CMPX_GE_U64',
  'V_CMPX_GT_F16', 'V_CMPX_GT_F32', 'V_CMPX_GT_F64', 'V_CMPX_GT_I16', 'V_CMPX_GT_I32', 'V_CMPX_GT_I64', 'V_CMPX_GT_U16', 'V_CMPX_GT_U32', 'V_CMPX_GT_U64',
  'V_CMPX_LE_F16', 'V_CMPX_LE_F32', 'V_CMPX_LE_F64', 'V_CMPX_LE_I16', 'V_CMPX_LE_I32', 'V_CMPX_LE_I64', 'V_CMPX_LE_U16', 'V_CMPX_LE_U32', 'V_CMPX_LE_U64',
  'V_CMPX_LG_F16', 'V_CMPX_LG_F32', 'V_CMPX_LG_F64',
  'V_CMPX_LT_F16', 'V_CMPX_LT_F32', 'V_CMPX_LT_F64', 'V_CMPX_LT_I16', 'V_CMPX_LT_I32', 'V_CMPX_LT_I64', 'V_CMPX_LT_U16', 'V_CMPX_LT_U32', 'V_CMPX_LT_U64',
  'V_CMPX_NE_I16', 'V_CMPX_NE_I32', 'V_CMPX_NE_I64', 'V_CMPX_NE_U16', 'V_CMPX_NE_U32', 'V_CMPX_NE_U64',
  'V_CMPX_NEQ_F16', 'V_CMPX_NEQ_F32', 'V_CMPX_NEQ_F64',
  'V_CMPX_NGE_F16', 'V_CMPX_NGE_F32', 'V_CMPX_NGE_F64', 'V_CMPX_NGT_F16', 'V_CMPX_NGT_F32', 'V_CMPX_NGT_F64',
  'V_CMPX_NLE_F16', 'V_CMPX_NLE_F32', 'V_CMPX_NLE_F64', 'V_CMPX_NLG_F16', 'V_CMPX_NLG_F32', 'V_CMPX_NLG_F64',
  'V_CMPX_NLT_F16', 'V_CMPX_NLT_F32', 'V_CMPX_NLT_F64',
  'V_CMPX_O_F16', 'V_CMPX_O_F32', 'V_CMPX_O_F64', 'V_CMPX_T_F16', 'V_CMPX_T_F32', 'V_CMPX_T_F64',
  'V_CMPX_T_I32', 'V_CMPX_T_I64', 'V_CMPX_T_U32', 'V_CMPX_T_U64', 'V_CMPX_U_F16', 'V_CMPX_U_F32', 'V_CMPX_U_F64',
  # SOP (134 ops)
  'S_ABSDIFF_I32', 'S_ABS_I32', 'S_ADD_F16', 'S_ADD_F32', 'S_ADD_I32', 'S_ADD_U32', 'S_ADDC_U32', 'S_ADDK_I32',
  'S_AND_B32', 'S_AND_B64', 'S_AND_NOT0_SAVEEXEC_B32', 'S_AND_NOT0_SAVEEXEC_B64', 'S_AND_NOT0_WREXEC_B32', 'S_AND_NOT0_WREXEC_B64',
  'S_AND_NOT1_B32', 'S_AND_NOT1_B64', 'S_AND_NOT1_SAVEEXEC_B32', 'S_AND_NOT1_SAVEEXEC_B64', 'S_AND_NOT1_WREXEC_B32', 'S_AND_NOT1_WREXEC_B64',
  'S_AND_SAVEEXEC_B32', 'S_AND_SAVEEXEC_B64', 'S_ASHR_I32', 'S_ASHR_I64',
  'S_BCNT0_I32_B32', 'S_BCNT0_I32_B64', 'S_BCNT1_I32_B32', 'S_BCNT1_I32_B64',
  'S_BFE_I32', 'S_BFE_I64', 'S_BFE_U32', 'S_BFE_U64',
  'S_BFM_B32', 'S_BFM_B64', 'S_BITSET0_B32', 'S_BITSET0_B64', 'S_BITSET1_B32', 'S_BITSET1_B64',
  'S_CMOVK_I32', 'S_CMOV_B32', 'S_CMOV_B64', 'S_CSELECT_B32', 'S_CSELECT_B64',
  'S_CVT_F16_F32', 'S_CVT_F32_F16', 'S_CVT_F32_I32', 'S_CVT_F32_U32', 'S_CVT_HI_F32_F16', 'S_CVT_I32_F32', 'S_CVT_PK_RTZ_F16_F32', 'S_CVT_U32_F32',
  'S_DELAY_ALU', 'S_FMAAK_F32', 'S_FMAC_F16', 'S_FMAC_F32', 'S_FMAMK_F32',
  'S_LSHL_B32', 'S_LSHL_B64', 'S_LSHL1_ADD_U32', 'S_LSHL2_ADD_U32', 'S_LSHL3_ADD_U32', 'S_LSHL4_ADD_U32',
  'S_LSHR_B32', 'S_LSHR_B64', 'S_MAX_I32', 'S_MAX_U32', 'S_MIN_I32', 'S_MIN_U32', 'S_MOVK_I32', 'S_MOV_B32',
  'S_MOV_B64', 'S_MULK_I32', 'S_MUL_F16', 'S_MUL_F32', 'S_MUL_HI_I32', 'S_MUL_HI_U32', 'S_MUL_I32',
  'S_NAND_B32', 'S_NAND_B64', 'S_NAND_SAVEEXEC_B32', 'S_NAND_SAVEEXEC_B64',
  'S_NOP', 'S_NOR_B32', 'S_NOR_B64', 'S_NOR_SAVEEXEC_B32', 'S_NOR_SAVEEXEC_B64',
  'S_NOT_B32', 'S_NOT_B64', 'S_OR_B32', 'S_OR_B64',
  'S_OR_NOT0_SAVEEXEC_B32', 'S_OR_NOT0_SAVEEXEC_B64', 'S_OR_NOT1_B32', 'S_OR_NOT1_B64', 'S_OR_NOT1_SAVEEXEC_B32', 'S_OR_NOT1_SAVEEXEC_B64',
  'S_OR_SAVEEXEC_B32', 'S_OR_SAVEEXEC_B64',
  'S_PACK_HH_B32_B16', 'S_PACK_HL_B32_B16', 'S_PACK_LH_B32_B16', 'S_PACK_LL_B32_B16', 'S_RFE_B64', 'S_RNDNE_F16', 'S_RNDNE_F32',
  'S_SENDMSG_RTN_B32', 'S_SENDMSG_RTN_B64', 'S_SETPC_B64', 'S_SEXT_I32_I16', 'S_SEXT_I32_I8',
  'S_SUB_F16', 'S_SUB_F32', 'S_SUB_I32', 'S_SUB_U32', 'S_SUBB_U32',
  'S_TRUNC_F16', 'S_TRUNC_F32', 'S_VERSION',
  # Additional SOP ops (newly supported)
  'S_BITCMP0_B32', 'S_BITCMP0_B64', 'S_BITCMP1_B32', 'S_BITCMP1_B64',
  'S_MAX_F16', 'S_MAX_F32', 'S_MIN_F16', 'S_MIN_F32',
  'S_WAITCNT_EXPCNT', 'S_WAITCNT_LGKMCNT', 'S_WAITCNT_VMCNT', 'S_WAITCNT_VSCNT',
  # Branch/control flow ops
  'S_BRANCH', 'S_CALL_B64', 'S_CBRANCH_EXECNZ', 'S_CBRANCH_EXECZ', 'S_CBRANCH_SCC0', 'S_CBRANCH_SCC1',
  'S_CBRANCH_VCCNZ', 'S_CBRANCH_VCCZ', 'S_GETPC_B64',
  'S_XNOR_B32', 'S_XNOR_B64', 'S_XNOR_SAVEEXEC_B32', 'S_XNOR_SAVEEXEC_B64',
  'S_XOR_B32', 'S_XOR_B64', 'S_XOR_SAVEEXEC_B32', 'S_XOR_SAVEEXEC_B64',
  # SOPC (54 ops)
  'S_CMPK_EQ_I32', 'S_CMPK_EQ_U32', 'S_CMPK_GE_I32', 'S_CMPK_GE_U32', 'S_CMPK_GT_I32', 'S_CMPK_GT_U32', 'S_CMPK_LE_I32', 'S_CMPK_LE_U32',
  'S_CMPK_LG_I32', 'S_CMPK_LG_U32', 'S_CMPK_LT_I32', 'S_CMPK_LT_U32',
  'S_CMP_EQ_F16', 'S_CMP_EQ_F32', 'S_CMP_EQ_I32', 'S_CMP_EQ_U32', 'S_CMP_EQ_U64',
  'S_CMP_GE_F16', 'S_CMP_GE_F32', 'S_CMP_GE_I32', 'S_CMP_GE_U32',
  'S_CMP_GT_F16', 'S_CMP_GT_F32', 'S_CMP_GT_I32', 'S_CMP_GT_U32',
  'S_CMP_LE_F16', 'S_CMP_LE_F32', 'S_CMP_LE_I32', 'S_CMP_LE_U32',
  'S_CMP_LG_F16', 'S_CMP_LG_F32', 'S_CMP_LG_I32', 'S_CMP_LG_U32', 'S_CMP_LG_U64',
  'S_CMP_LT_F16', 'S_CMP_LT_F32', 'S_CMP_LT_I32', 'S_CMP_LT_U32',
  'S_CMP_NEQ_F16', 'S_CMP_NEQ_F32', 'S_CMP_NGE_F16', 'S_CMP_NGE_F32', 'S_CMP_NGT_F16', 'S_CMP_NGT_F32',
  'S_CMP_NLE_F16', 'S_CMP_NLE_F32', 'S_CMP_NLG_F16', 'S_CMP_NLG_F32', 'S_CMP_NLT_F16', 'S_CMP_NLT_F32',
  'S_CMP_O_F16', 'S_CMP_O_F32', 'S_CMP_U_F16', 'S_CMP_U_F32',
}

@functools.cache
def compile_uop(cls_name: str, op_name: str, pseudocode: str):
  """Compile pseudocode to UOp-based function. Returns None if unsupported."""
  if op_name not in SUPPORTED_OPS:
    return None
  sink, output_info, input_vars = _compile_pseudocode(pseudocode)
  return _make_uop_fn(sink, output_info, input_vars)
