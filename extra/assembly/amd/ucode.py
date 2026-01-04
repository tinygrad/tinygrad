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
  'b32': dtypes.uint32, 'b16': dtypes.uint16, 'b64': dtypes.uint64,
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
      'SCC': UOp(Ops.DEFINE_VAR, dtypes.uint32, (), ('SCC', 0, 1)),
      'VCC': UOp(Ops.DEFINE_VAR, dtypes.uint64, (), ('VCC', 0, 0xffffffffffffffff)),
      'EXEC': UOp(Ops.DEFINE_VAR, dtypes.uint64, (), ('EXEC', 0, 0xffffffffffffffff)),
    }
    self.vars: dict[str, UOp] = dict(self.input_vars)
    self.outputs: list[tuple[str, UOp, DType]] = []  # (name, uop, dtype)
  
  def const(self, val, dtype: DType) -> UOp:
    return UOp(Ops.CONST, dtype, (), val)
  
  def cast(self, x: UOp, dtype: DType) -> UOp:
    if x.dtype == dtype: return x
    return UOp(Ops.BITCAST, dtype, (x,))
  
  def parse_type(self, s: str) -> tuple[str, DType]:
    if '.' in s:
      var, typ = s.rsplit('.', 1)
      return var.strip(), DTYPE_MAP.get(typ, dtypes.uint32)
    return s.strip(), dtypes.uint32
  
  def parse_expr(self, expr: str, dtype_hint: DType = None) -> tuple[UOp, DType]:
    expr = expr.strip()
    
    # Handle parentheses
    if expr.startswith('(') and expr.endswith(')'):
      depth = 0
      for i, c in enumerate(expr):
        if c == '(': depth += 1
        elif c == ')': depth -= 1
        if depth == 0 and i < len(expr) - 1: break
      else:
        return self.parse_expr(expr[1:-1], dtype_hint)
    
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
    for ops in [('||',), ('&&',), ('==', '!=', '<>', '<=', '>=', '<', '>'), ('|',), ('^',), ('&',), ('<<', '>>'), ('+', '-'), ('*', '/')]:
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
              left_expr = expr[:i].strip()
              right_expr = expr[i+len(op):].strip()
              if not left_expr: continue
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
    
    # Bit slice with type: S0[4:0].u32
    if m := re.match(r'^([A-Z]\d?)\s*\[\s*(\d+)\s*:\s*(\d+)\s*\]\.([a-z]\d+)$', expr):
      var, high, low, typ = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
      dtype = DTYPE_MAP.get(typ, dtypes.uint32)
      if high < low: high, low = low, high
      base_uop = self.vars.get(var)
      if base_uop is None: raise ValueError(f"Unknown variable: {var}")
      mask = (1 << (high - low + 1)) - 1
      shifted = UOp(Ops.SHR, dtypes.uint32, (base_uop, self.const(low, dtypes.uint32))) if low > 0 else base_uop
      masked = UOp(Ops.AND, dtypes.uint32, (shifted, self.const(mask, dtypes.uint32)))
      return self.cast(masked, dtype), dtype
    
    # Bit slice without type: S0[4:0]
    if m := re.match(r'^([A-Z]\d?)\s*\[\s*(\d+)\s*:\s*(\d+)\s*\]$', expr):
      var, high, low = m.group(1), int(m.group(2)), int(m.group(3))
      if high < low: high, low = low, high
      base_uop = self.vars.get(var)
      if base_uop is None: raise ValueError(f"Unknown variable: {var}")
      mask = (1 << (high - low + 1)) - 1
      shifted = UOp(Ops.SHR, dtypes.uint32, (base_uop, self.const(low, dtypes.uint32))) if low > 0 else base_uop
      masked = UOp(Ops.AND, dtypes.uint32, (shifted, self.const(mask, dtypes.uint32)))
      return masked, dtype_hint or dtypes.uint32
    
    # Typed variable: S0.f32
    if m := re.match(r'^([A-Z]\d?)\.([a-z]\d+)$', expr):
      var, typ = m.group(1), m.group(2)
      dtype = DTYPE_MAP.get(typ, dtypes.uint32)
      base_uop = self.vars.get(var)
      if base_uop is None: raise ValueError(f"Unknown variable: {var}")
      return self.cast(base_uop, dtype), dtype
    
    # Plain variable
    if expr in self.vars:
      uop = self.vars[expr]
      dtype = dtype_hint or uop.dtype
      return self.cast(uop, dtype), dtype
    
    # Numeric literals
    expr_clean = re.sub(r"(\d)'([0-9a-fA-Fx]+)[UuLlFf]*", r'\2', expr)
    expr_clean = re.sub(r'([0-9a-fA-Fx]+)[UuLlFf]+$', r'\1', expr_clean)
    try:
      if expr_clean.startswith('0x') or expr_clean.startswith('0X'):
        return self.const(int(expr_clean, 16), dtype_hint or dtypes.uint32), dtype_hint or dtypes.uint32
      if '.' in expr_clean or 'e' in expr_clean.lower():
        return self.const(float(expr_clean), dtype_hint or dtypes.float32), dtype_hint or dtypes.float32
      return self.const(int(expr_clean), dtype_hint or dtypes.uint32), dtype_hint or dtypes.uint32
    except ValueError:
      pass
    
    raise ValueError(f"Cannot parse expression: {expr}")
  
  def parse_stmt(self, line: str):
    if '=' not in line or any(line.startswith(k) for k in ('if ', 'elsif ', 'for ')):
      return
    lhs, rhs = line.split('=', 1)
    var, dtype = self.parse_type(lhs.strip())
    rhs_uop, _ = self.parse_expr(rhs.strip(), dtype)
    self.vars[var] = rhs_uop
    if var in ('D0', 'D1', 'SCC', 'VCC'):
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
  if dtype == dtypes.float32:
    return struct.unpack('<I', struct.pack('<f', val))[0]
  elif dtype == dtypes.float16:
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
  for line in pseudocode.strip().split('\n'):
    line = line.split('//')[0].strip()
    if line:
      builder.parse_stmt(line)
  sink = builder.build_sink()
  output_info = [(name, dtype) for name, _, dtype in builder.outputs]
  return sink, output_info, builder.input_vars

def _make_uop_fn(sink: UOp, output_info: list[tuple[str, DType]], input_vars: dict[str, UOp]):
  """Create a runtime function that evaluates the UOp graph via simplify."""
  def fn(s0, s1, s2, d0, scc, vcc, laneId, exec_mask, literal, VGPR, src0_idx=0, vdst_idx=0, pc=None):
    # Build substitution map: DEFINE_VAR -> CONST
    dvars = {
      input_vars['S0']: UOp.const(dtypes.uint32, s0),
      input_vars['S1']: UOp.const(dtypes.uint32, s1),
      input_vars['S2']: UOp.const(dtypes.uint32, s2),
      input_vars['D0']: UOp.const(dtypes.uint32, d0),
      input_vars['SCC']: UOp.const(dtypes.uint32, scc),
      input_vars['VCC']: UOp.const(dtypes.uint64, vcc),
      input_vars['EXEC']: UOp.const(dtypes.uint64, exec_mask),
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

# Ops that ucode.py supports
# NOTE: float ops need FTZ (flush-to-zero) which UOp simplify doesn't handle
SUPPORTED_OPS: set[str] = {
  # VOP1
  'V_MOV_B32', 'V_NOT_B32',
  # Integer arithmetic
  'V_ADD_NC_U32', 'V_SUB_NC_U32', 'V_SUBREV_NC_U32',
  # Bitwise
  'V_AND_B32', 'V_OR_B32', 'V_XOR_B32', 'V_XNOR_B32',
  # Shifts
  'V_LSHLREV_B32', 'V_LSHRREV_B32', 'V_ASHRREV_I32',
  # Min/Max
  'V_MIN_U32', 'V_MAX_U32', 'V_MIN_I32', 'V_MAX_I32',
}

@functools.cache
def compile_uop(cls_name: str, op_name: str, pseudocode: str):
  """Compile pseudocode to UOp-based function. Returns None if unsupported."""
  if op_name not in SUPPORTED_OPS:
    return None
  sink, output_info, input_vars = _compile_pseudocode(pseudocode)
  return _make_uop_fn(sink, output_info, input_vars)
