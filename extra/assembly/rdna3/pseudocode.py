#!/usr/bin/env python3
"""Extract and parse pseudocode from AMD RDNA3.5 ISA PDF for emulation."""
import re, pathlib, struct, math
from typing import Any

# Pseudocode interpreter helpers
def _f32(i: int) -> float:
  """Convert u32 bits to f32."""
  return struct.unpack('<f', struct.pack('<I', i & 0xffffffff))[0]

def _i32(f: float) -> int:
  """Convert f32 to u32 bits."""
  if math.isnan(f): return 0x7fc00000
  if math.isinf(f): return 0x7f800000 if f > 0 else 0xff800000
  try: return struct.unpack('<I', struct.pack('<f', f))[0]
  except (OverflowError, struct.error): return 0x7f800000 if f > 0 else 0xff800000

def _sext(v: int, b: int) -> int:
  """Sign extend b-bit value to Python int."""
  return v - (1 << b) if v & (1 << (b - 1)) else v

class PseudocodeInterpreter:
  """Interpreter for RDNA3 pseudocode from AMD ISA PDF."""

  def __init__(self):
    self.vars: dict[str, Any] = {}
    self.ctx: dict[str, Any] = {}

  def eval_expr(self, expr: str) -> Any:
    """Evaluate a pseudocode expression using current context."""
    expr = expr.strip()
    s0, s1, s2, d0 = self.ctx['s0'], self.ctx['s1'], self.ctx['s2'], self.ctx['d0']
    scc, vcc, lane, exec_mask = self.ctx['scc'], self.ctx['vcc'], self.ctx['lane'], self.ctx['exec_mask']

    # FIRST: Convert single-bit indexing VAR[expr] to _getbit(VAR, expr) to avoid bracket issues
    # But NOT bit ranges like VAR[4:0] - those are handled separately
    def convert_bit_index(e):
      result, i = [], 0
      while i < len(e):
        # Match S0/S1/S2/D0 or var, optionally with .suffix, followed by [
        m = re.match(r'(S[012]|D0|\w+)(\.[ubi]\d+)?\[', e[i:])
        if m:
          var, suffix = m.group(1), m.group(2)
          start = i + m.end()
          depth, end = 1, start
          while end < len(e) and depth > 0:
            if e[end] == '[': depth += 1
            elif e[end] == ']': depth -= 1
            end += 1
          idx_expr = e[start:end-1]
          # Check if this is a bit range (contains : with simple numbers) - don't convert those
          if re.match(r'^\s*\d+\s*:\s*\d+\s*$', idx_expr):
            # Bit range - leave as-is for later handling
            result.append(e[i:end])
          elif var in ('S0', 'S1', 'S2', 'D0'):
            # Single bit access - convert to function call
            result.append(f'_getbit_{var.lower()}({idx_expr})')
          elif var in self.vars:
            base_val = int(self.vars[var])
            result.append(f'(({base_val} >> ({idx_expr})) & 1)')
          else:
            result.append(e[i:end])
          i = end
        else:
          result.append(e[i]); i += 1
      return ''.join(result)
    expr = convert_bit_index(expr)

    # Handle bit range like S1[4:0].u32 or S1[4:0]
    def replace_bit_range(m):
      var, hi, lo = m.group(1), int(m.group(2)), int(m.group(3))
      val = {'S0': s0, 'S1': s1, 'S2': s2, 'D0': d0}.get(var)
      if val is None: return m.group(0)
      return str((val >> lo) & ((1 << (hi - lo + 1)) - 1))
    # Handle S0.u32[4:0] style (with type suffix before bit range)
    expr = re.sub(r'(S[012]|D0)\.u32\[(\d+)\s*:\s*(\d+)\](?:\.u32)?', replace_bit_range, expr)
    # Handle S0[4:0] style (without type suffix)
    expr = re.sub(r'(S[012]|D0)\[(\d+)\s*:\s*(\d+)\](?:\.u32)?', replace_bit_range, expr)

    # Handle signext function
    expr = re.sub(r'signext\(S0\.i32\)', str(_sext(s0 & 0xffffffff, 32)), expr)
    expr = re.sub(r'signext\(S1\.i32\)', str(_sext(s1 & 0xffffffff, 32)), expr)
    expr = re.sub(r'signext\(([^)]+)\)', r'_sext(\1, 32)', expr)

    # Replace source/dest field access - 32-bit
    expr = re.sub(r'S0\.u32', str(s0 & 0xffffffff), expr)
    expr = re.sub(r'S0\.i32', str(_sext(s0 & 0xffffffff, 32)), expr)
    expr = re.sub(r'S0\.f32', f'_f32({s0 & 0xffffffff})', expr)
    expr = re.sub(r'S0\.b32', str(s0 & 0xffffffff), expr)
    expr = re.sub(r'S0\.u24', str(s0 & 0xffffff), expr)  # 24-bit unsigned
    expr = re.sub(r'S1\.u32', str(s1 & 0xffffffff), expr)
    expr = re.sub(r'S1\.i32', str(_sext(s1 & 0xffffffff, 32)), expr)
    expr = re.sub(r'S1\.f32', f'_f32({s1 & 0xffffffff})', expr)
    expr = re.sub(r'S1\.b32', str(s1 & 0xffffffff), expr)
    expr = re.sub(r'S1\.u24', str(s1 & 0xffffff), expr)  # 24-bit unsigned

    # SIMM16 for SOPK instructions (s1 contains the 16-bit immediate)
    # Handle bit access first: SIMM16.i16[bit]
    def replace_simm16_bit(m):
      bit = int(m.group(1))
      return str(((_sext(s1 & 0xffff, 16)) >> bit) & 1)
    expr = re.sub(r'SIMM16\.i16\[(\d+)\]', replace_simm16_bit, expr)
    expr = re.sub(r'SIMM16\.u16\[(\d+)\]', lambda m: str(((s1 & 0xffff) >> int(m.group(1))) & 1), expr)
    expr = re.sub(r'SIMM16\.i16', str(_sext(s1 & 0xffff, 16)), expr)
    expr = re.sub(r'SIMM16\.u16', str(s1 & 0xffff), expr)
    # SIMM32 for literal immediate (passed as 'literal' in context)
    literal = self.ctx.get('literal', 0)
    expr = re.sub(r'SIMM32\.f32', f'_f32({literal & 0xffffffff})', expr)
    expr = re.sub(r'SIMM32\.u32', str(literal & 0xffffffff), expr)
    expr = re.sub(r'SIMM32\.i32', str(_sext(literal & 0xffffffff, 32)), expr)
    # Special constants
    expr = re.sub(r'DENORM\.f32', '1e-38', expr)  # Smallest positive denormal f32
    expr = re.sub(r'DENORM\.f64', '5e-324', expr)  # Smallest positive denormal f64
    expr = re.sub(r'INF\.f32', 'INF', expr)
    expr = re.sub(r'INF\.f16', 'INF', expr)
    expr = re.sub(r'NAN\.f32', 'NAN', expr)
    expr = re.sub(r'NAN\.f64', 'NAN', expr)
    expr = re.sub(r'signext\(([^)]+)\)', r'_sext(\1, 32)', expr)
    expr = re.sub(r'S2\.u32', str(s2 & 0xffffffff), expr)
    expr = re.sub(r'S2\.i32', str(_sext(s2 & 0xffffffff, 32)), expr)
    expr = re.sub(r'S2\.f32', f'_f32({s2 & 0xffffffff})', expr)
    expr = re.sub(r'D0\.u32', str(d0 & 0xffffffff), expr)
    expr = re.sub(r'D0\.i32', str(_sext(d0 & 0xffffffff, 32)), expr)
    expr = re.sub(r'D0\.f32', f'_f32({d0 & 0xffffffff})', expr)
    expr = re.sub(r'D0\.b32', str(d0 & 0xffffffff), expr)

    # 64-bit access
    expr = re.sub(r'S0\.u64', str(s0 & 0xffffffffffffffff), expr)
    expr = re.sub(r'S0\.i64', str(_sext(s0 & 0xffffffffffffffff, 64)), expr)
    expr = re.sub(r'S0\.b64', str(s0 & 0xffffffffffffffff), expr)
    expr = re.sub(r'S1\.u64', str(s1 & 0xffffffffffffffff), expr)
    expr = re.sub(r'S1\.i64', str(_sext(s1 & 0xffffffffffffffff, 64)), expr)
    expr = re.sub(r'S2\.u64', str(s2 & 0xffffffffffffffff), expr)
    expr = re.sub(r'S2\.i64', str(_sext(s2 & 0xffffffffffffffff, 64)), expr)
    expr = re.sub(r'D0\.u64', str(d0 & 0xffffffffffffffff), expr)
    expr = re.sub(r'D0\.i64', str(_sext(d0 & 0xffffffffffffffff, 64)), expr)
    expr = re.sub(r'D0\.b64', str(d0 & 0xffffffffffffffff), expr)

    # VCC and EXEC access
    expr = re.sub(r'VCC\.u64\[laneId\]\.u64', str((vcc >> lane) & 1), expr)  # VCC bit as u64
    expr = re.sub(r'VCC\.u64\[laneId\]\.u32', str((vcc >> lane) & 1), expr)  # VCC bit as u32
    expr = re.sub(r'VCC\.u64\[laneId\]', str((vcc >> lane) & 1), expr)
    expr = re.sub(r'VCC\.u64', str(vcc & 0xffffffffffffffff), expr)
    expr = re.sub(r'EXEC\.u32', str(exec_mask & 0xffffffff), expr)
    expr = re.sub(r'EXEC\.u64', str(exec_mask & 0xffffffffffffffff), expr)

    # SCC access
    expr = re.sub(r'SCC\.u32', str(scc & 1), expr)
    expr = re.sub(r'SCC\.u64', str(scc & 1), expr)
    expr = re.sub(r'\bSCC\b', str(scc), expr)

    # laneId
    expr = re.sub(r'\blaneId\b', str(lane), expr)
    # Wave mode - assume Wave32 for tinygrad
    expr = re.sub(r'\bWAVE64\b', 'False', expr)
    expr = re.sub(r'\bWAVE32\b', 'True', expr)

    # Handle variables from context
    for name, val in self.vars.items():
      expr = re.sub(rf'\b{name}\.u32\b', str(int(val) & 0xffffffff), expr)
      expr = re.sub(rf'\b{name}\.i32\b', str(_sext(int(val) & 0xffffffff, 32)), expr)
      expr = re.sub(rf'\b{name}\.u64\b', str(int(val) & 0xffffffffffffffff), expr)
      expr = re.sub(rf'\b{name}\b', str(int(val) if not isinstance(val, float) else val), expr)

    # Handle type casts - use balanced paren matching
    def replace_cast(e, pattern, replacement):
      result, i = [], 0
      while i < len(e):
        m = re.match(pattern + r'\(', e[i:])
        if m:
          start = i + m.end()
          depth, end = 1, start
          while end < len(e) and depth > 0:
            if e[end] == '(': depth += 1
            elif e[end] == ')': depth -= 1
            end += 1
          result.append(replacement(e[start:end-1]))
          i = end
        else:
          result.append(e[i]); i += 1
      return ''.join(result)
    # Handle arbitrary bit-width casts: N'U (unsigned), N'I (signed), N'B (bits), N'F (float)
    def replace_bitwidth_cast(e):
      result, i = [], 0
      while i < len(e):
        m = re.match(r"(\d+)'([UIBF])\(", e[i:])
        if m:
          bits, cast_type = int(m.group(1)), m.group(2)
          start = i + m.end()
          depth, end = 1, start
          while end < len(e) and depth > 0:
            if e[end] == '(': depth += 1
            elif e[end] == ')': depth -= 1
            end += 1
          inner = e[start:end-1]
          mask = (1 << bits) - 1
          if cast_type == 'U': result.append(f'(({inner}) & {hex(mask)})')
          elif cast_type == 'I': result.append(f'(_sext(({inner}) & {hex(mask)}, {bits}))')
          elif cast_type == 'B': result.append(f'(({inner}) & {hex(mask)})')
          elif cast_type == 'F': result.append(f'float({inner})')
          i = end
        else:
          result.append(e[i]); i += 1
      return ''.join(result)
    # Iteratively process until no more casts remain (handles nested casts)
    while re.search(r"\d+'[UIBF]\(", expr):
      expr = replace_bitwidth_cast(expr)
    expr = re.sub(r"1'1U", '1', expr)
    expr = re.sub(r"1'0U", '0', expr)
    # Handle special functions - cvtToQuietNAN just passes through (returns NaN if input is NaN)
    expr = re.sub(r'cvtToQuietNAN\(', '(', expr)  # Just strip the function name

    # Handle conversion/math functions
    expr = re.sub(r'i32_to_f32\(([^)]+)\)', r'float(\1)', expr)
    expr = re.sub(r'u32_to_f32\(([^)]+)\)', r'float(\1)', expr)
    expr = re.sub(r'f32_to_i32\(([^)]+)\)', r'int(\1)', expr)
    expr = re.sub(r'f32_to_u32\(([^)]+)\)', r'max(0, int(\1))', expr)
    expr = re.sub(r'\bfma\(([^,]+),\s*([^,]+),\s*([^)]+)\)', r'((\1) * (\2) + (\3))', expr)
    expr = re.sub(r'\bmin\(', 'min(', expr)
    expr = re.sub(r'\bmax\(', 'max(', expr)
    # NaN checking functions - for simplicity, treat signal/quiet NaN the same
    expr = re.sub(r'isSignalNAN\(([^)]+)\)', r'_isnan(\1)', expr)
    expr = re.sub(r'isQuietNAN\(([^)]+)\)', r'_isnan(\1)', expr)
    # Handle nested parentheses correctly for these functions
    def replace_func_with_balanced_parens(e, func_name, replacement_fn):
      result, i = [], 0
      while i < len(e):
        if e[i:].startswith(func_name + '('):
          start = i + len(func_name) + 1
          depth, end = 1, start
          while end < len(e) and depth > 0:
            if e[end] == '(': depth += 1
            elif e[end] == ')': depth -= 1
            end += 1
          inner = e[start:end-1]
          result.append(replacement_fn(inner))
          i = end
        else:
          result.append(e[i]); i += 1
      return ''.join(result)
    expr = replace_func_with_balanced_parens(expr, 'isNAN', lambda x: f'_isnan({x})')
    expr = replace_func_with_balanced_parens(expr, 'isEven', lambda x: f'(int({x}) % 2 == 0)')
    expr = replace_func_with_balanced_parens(expr, 'fract', lambda x: f'(({x}) - floor({x}))')
    # Float manipulation functions
    expr = re.sub(r'exponent\(([^)]+)\)', r'_exponent(\1)', expr)
    expr = re.sub(r'mantissa\(([^)]+)\)', r'_mantissa(\1)', expr)
    expr = re.sub(r'sign\(([^)]+)\)', r'_sign(\1)', expr)
    expr = re.sub(r'ldexp\(([^,]+),\s*([^)]+)\)', r'_ldexp(\1, \2)', expr)
    expr = re.sub(r'signext_from_bit\(([^,]+),\s*([^)]+)\)', r'_sext(\1, \2)', expr)
    expr = re.sub(r'\bsin\(', '_sin(', expr)
    expr = re.sub(r'\bcos\(', '_cos(', expr)
    expr = re.sub(r'\bpow\(', '_pow(', expr)

    # Handle { a, b } concatenation (pack two values into one: (a << 32) | b)
    def replace_concat(m):
      a, b = m.group(1).strip(), m.group(2).strip()
      return f'((({a}) << 32) | ({b}))'
    expr = re.sub(r'\{\s*([^,]+)\s*,\s*([^}]+)\s*\}', replace_concat, expr)

    # Handle hex/int/float literals with suffixes
    expr = re.sub(r'0x([0-9a-fA-F]+)ULL', r'0x\1', expr)
    expr = re.sub(r'0x([0-9a-fA-F]+)U', r'0x\1', expr)
    expr = re.sub(r'0x([0-9a-fA-F]+)LL', r'0x\1', expr)
    expr = re.sub(r'(\d+)ULL\b', r'\1', expr)
    expr = re.sub(r'(\d+)LL\b', r'\1', expr)
    expr = re.sub(r'(\d+)U\b', r'\1', expr)
    expr = re.sub(r'(\d+\.?\d*)F\b', r'\1', expr)  # Float suffix

    # Handle ternary operator - convert C-style `cond ? true : false` to Python `true if cond else false`
    def convert_ternary_recursive(e):
      # First recursively process parenthesized subexpressions
      result, i = [], 0
      while i < len(e):
        if e[i] == '(':
          # Find matching close paren
          depth, end = 1, i + 1
          while end < len(e) and depth > 0:
            if e[end] == '(': depth += 1
            elif e[end] == ')': depth -= 1
            end += 1
          inner = convert_ternary_recursive(e[i+1:end-1])
          result.append(f'({inner})')
          i = end
        else:
          result.append(e[i]); i += 1
      e = ''.join(result)
      # Now convert ternary at current level
      i, q_pos, c_pos = 0, -1, -1
      while i < len(e):
        c = e[i]
        if c == '(' :
          # Skip parenthesized content (already processed)
          depth, end = 1, i + 1
          while end < len(e) and depth > 0:
            if e[end] == '(': depth += 1
            elif e[end] == ')': depth -= 1
            end += 1
          i = end
        elif c == '?' and q_pos < 0: q_pos = i; i += 1
        elif c == ':' and q_pos >= 0 and c_pos < 0: c_pos = i; i += 1
        else: i += 1
      if q_pos > 0 and c_pos > q_pos:
        cond = e[:q_pos].strip()
        true_val = e[q_pos+1:c_pos].strip()
        false_val = e[c_pos+1:].strip()
        return f'(({true_val}) if ({cond}) else ({false_val}))'
      return e
    if '?' in expr: expr = convert_ternary_recursive(expr)

    # Handle comparisons and logical operators
    expr = re.sub(r'<>', ' != ', expr)  # <> is "not equal" in pseudocode
    expr = re.sub(r'!=', ' != ', expr)
    expr = re.sub(r'==', ' == ', expr)
    expr = re.sub(r'&&', ' and ', expr)
    expr = re.sub(r'\|\|', ' or ', expr)
    expr = re.sub(r'!([^=])', r' not \1', expr)  # ! but not !=
    expr = re.sub(r'>=', ' >= ', expr)
    expr = re.sub(r'<=', ' <= ', expr)

    def _exponent(x):
      if not isinstance(x, float) or math.isnan(x) or math.isinf(x) or x == 0: return 0 if x == 0 else 255
      return ((struct.unpack('<I', struct.pack('<f', x))[0] >> 23) & 0xff)
    def _mantissa(x):
      if not isinstance(x, float): return 0
      return struct.unpack('<I', struct.pack('<f', x))[0] & 0x7fffff
    def _sign(x):
      if isinstance(x, float): return 1 if x < 0 or (x == 0 and struct.unpack('<I', struct.pack('<f', x))[0] >> 31) else 0
      return 1 if x < 0 else 0
    def _ldexp(x, exp): return math.ldexp(x, int(exp)) if isinstance(x, float) else x
    def _sin(x): return math.sin(x * 2 * math.pi) if isinstance(x, float) else 0  # RDNA sin takes input in range [0,1] representing [0, 2pi]
    def _cos(x): return math.cos(x * 2 * math.pi) if isinstance(x, float) else 0
    def _pow(x, y): return math.pow(x, y) if isinstance(x, float) and isinstance(y, (int, float)) else 0
    # Bit access functions for S0/S1/S2/D0 - deferred until all substitutions are done
    _getbit_s0 = lambda idx: (s0 >> int(idx)) & 1
    _getbit_s1 = lambda idx: (s1 >> int(idx)) & 1
    _getbit_s2 = lambda idx: (s2 >> int(idx)) & 1
    _getbit_d0 = lambda idx: (d0 >> int(idx)) & 1
    # Helper functions from AMD pseudocode (used by v_max3/v_min3/v_med3)
    def v_max_i32(a, b): return a if _sext(a & 0xffffffff, 32) > _sext(b & 0xffffffff, 32) else b
    def v_min_i32(a, b): return a if _sext(a & 0xffffffff, 32) < _sext(b & 0xffffffff, 32) else b
    def v_max_u32(a, b): return a if (a & 0xffffffff) > (b & 0xffffffff) else b
    def v_min_u32(a, b): return a if (a & 0xffffffff) < (b & 0xffffffff) else b
    def v_max_f32(a, b): return _i32(max(_f32(a), _f32(b)))
    def v_min_f32(a, b): return _i32(min(_f32(a), _f32(b)))
    def v_max3_i32(a, b, c): return v_max_i32(v_max_i32(a, b), c)
    def v_max3_u32(a, b, c): return v_max_u32(v_max_u32(a, b), c)
    def v_max3_f32(a, b, c): return v_max_f32(v_max_f32(a, b), c)
    def _sqrt(x): return math.sqrt(x) if x >= 0 else float('nan')
    def _log2(x): return math.log2(x) if x > 0 else (float('-inf') if x == 0 else float('nan'))
    try:
      return eval(expr, {'_f32': _f32, '_i32': _i32, '_sext': _sext, 'abs': abs, 'min': min, 'max': max, 'math': math,
                         'log2': _log2, 'sqrt': _sqrt, 'trunc': math.trunc, 'floor': math.floor, 'ceil': math.ceil,
                         'NAN': float('nan'), 'INF': float('inf'), '_isnan': lambda x: math.isnan(x) if isinstance(x, float) else False,
                         '_exponent': _exponent, '_mantissa': _mantissa, '_sign': _sign, '_ldexp': _ldexp,
                         '_sin': _sin, '_cos': _cos, '_pow': _pow, 'struct': struct,
                         '_getbit_s0': _getbit_s0, '_getbit_s1': _getbit_s1, '_getbit_s2': _getbit_s2, '_getbit_d0': _getbit_d0,
                         'v_max_i32': v_max_i32, 'v_min_i32': v_min_i32, 'v_max_u32': v_max_u32, 'v_min_u32': v_min_u32,
                         'v_max_f32': v_max_f32, 'v_min_f32': v_min_f32, 'v_max3_i32': v_max3_i32, 'v_max3_u32': v_max3_u32, 'v_max3_f32': v_max3_f32})
    except ZeroDivisionError:
      return float('inf')  # Division by zero returns infinity
    except Exception as e:
      raise ValueError(f"Failed to evaluate '{expr}': {e}")

  def _exec_block(self, lines: list[str], start: int, end: int) -> None:
    """Execute a block of lines from start to end (exclusive)."""
    i = start
    while i < end:
      line = lines[i].strip()
      i += 1
      if not line or line.startswith('//') or line.startswith('declare '): continue
      # Skip prose descriptions
      if line.endswith('.') and not any(p in line for p in ['D0', 'D1', 'S0', 'S1', 'S2', 'SCC', 'VCC', 'EXEC', 'tmp', '=', ';']): continue

      # Handle for loops: "for i in <start> : <end> do"
      if line.startswith('for '):
        m = re.match(r'for\s+(\w+)\s+in\s+(.+?)\s*:\s*(.+?)\s+do', line)
        if m:
          var_name, loop_start, loop_end = m.group(1), self.eval_expr(m.group(2)), self.eval_expr(m.group(3))
          # Find matching endfor or end of block
          body_start, depth = i, 1
          while i < end and depth > 0:
            l = lines[i].strip()
            if l.startswith('for '): depth += 1
            elif l == 'endfor' or l.startswith('endfor'): depth -= 1
            i += 1
          body_end = i - 1 if depth == 0 else end
          # Execute loop body for each iteration
          for loop_val in range(int(loop_start), int(loop_end) + 1):
            self.vars[var_name] = loop_val
            self._exec_block(lines, body_start, body_end)
        continue

      # Handle if/elsif/else/endif
      if line.startswith('if ') or line.startswith('elsif '):
        # Extract condition (remove "if ", " then", "elsif ")
        cond_str = line[3:] if line.startswith('if ') else line[6:]
        cond_str = cond_str.replace(' then', '').strip()
        cond = bool(self.eval_expr(cond_str))
        # Find the matching endif and any else/elsif branches
        body_start, depth, branches = i, 1, []
        while i < end and depth > 0:
          l = lines[i].strip()
          if l.startswith('if '): depth += 1
          elif l.startswith('endif'): depth -= 1
          elif depth == 1 and (l.startswith('elsif ') or l.startswith('else')):
            branches.append((i, l))
          i += 1
        # Execute the appropriate branch
        if cond:
          branch_end = branches[0][0] if branches else (i - 1 if depth == 0 else end)
          self._exec_block(lines, body_start, branch_end)
        else:
          # Try elsif/else branches
          for bi, (branch_line, branch_text) in enumerate(branches):
            next_end = branches[bi + 1][0] if bi + 1 < len(branches) else (i - 1 if depth == 0 else end)
            if branch_text.startswith('else'):
              self._exec_block(lines, branch_line + 1, next_end)
              break
            elif branch_text.startswith('elsif '):
              elsif_cond = branch_text[6:].replace(' then', '').strip()
              if bool(self.eval_expr(elsif_cond)):
                self._exec_block(lines, branch_line + 1, next_end)
                break
        continue

      if line.startswith(('endif', 'else', 'endfor')): continue  # Skip control flow markers

      # Execute assignment
      if '=' in line and not line.startswith('=='):
        parts = line.rstrip(';').split('=', 1)
        if len(parts) == 2:
          lhs, rhs = parts[0].strip(), parts[1].strip()
          val = self.eval_expr(rhs)
          self._assign(lhs, val)
          # Update context so subsequent lines can reference updated values
          if 'd0' in self._result: self.ctx['d0'] = self._result['d0']
          if 'scc' in self._result: self.ctx['scc'] = self._result['scc']
          if 'exec' in self._result: self.ctx['exec_mask'] = self._result['exec']

  def _assign(self, lhs: str, val: Any) -> None:
    """Assign a value to a destination."""
    # Handle concatenation assignment: { D1.u1, D0.u64 } = val
    if lhs.startswith('{') and lhs.endswith('}'):
      # Parse { D1.type, D0.type } pattern
      inner = lhs[1:-1].strip()
      parts = [p.strip() for p in inner.split(',')]
      if len(parts) == 2:
        d1_part, d0_part = parts[0], parts[1]
        # D0 gets the low bits (usually 64 bits)
        if 'D0.u64' in d0_part or 'D0.i64' in d0_part:
          self._result['d0'] = int(val) & 0xffffffffffffffff
          self._result['d0_64'] = True
        elif 'D0.u32' in d0_part or 'D0.i32' in d0_part:
          self._result['d0'] = int(val) & 0xffffffff
        # D1 gets the high bit (carry/overflow)
        if 'D1.u1' in d1_part or 'D1.i1' in d1_part:
          # For 65-bit result, bit 64 is the carry
          self._result['vcc_lane'] = (int(val) >> 64) & 1
      return
    if lhs == 'D0.u64[laneId]':
      self._result['vcc_lane'] = int(bool(val))
    elif lhs.startswith('D0.'):
      if '.f32' in lhs:
        self._result['d0'] = _i32(float(val)) if isinstance(val, (int, float)) else int(val) & 0xffffffff
      elif '.u64' in lhs or '.b64' in lhs or '.i64' in lhs:
        self._result['d0'] = int(val) & 0xffffffffffffffff
        self._result['d0_64'] = True
      else:
        self._result['d0'] = int(val) & 0xffffffff
    elif lhs.startswith('SCC'):
      self._result['scc'] = int(bool(val))
    elif lhs == 'EXEC.u64[laneId]':
      self._result['exec_lane'] = int(bool(val))
    elif lhs.startswith('EXEC'):
      self._result['exec'] = int(val) & (0xffffffff if '.u32' in lhs else 0xffffffffffffffff)
    elif lhs == 'VCC.u64[laneId]':
      self._result['vcc_lane'] = int(bool(val))
    elif lhs.startswith('VCC'):
      pass  # Other VCC writes not handled
    elif lhs == 'PC':
      self._result['pc_delta'] = int(val)
    elif '[' in lhs:
      # Bit assignment like tmp[i] = val or D0.u64[i * 2] = val
      m = re.match(r'(\w+)(?:\.\w+)?\[(.+)\]', lhs)
      if m:
        var_name, idx_expr = m.group(1), m.group(2)
        idx = int(self.eval_expr(idx_expr))
        if var_name in ('D0',):
          # Bit assignment to D0
          if val: self._result['d0'] = self._result.get('d0', 0) | (1 << idx)
          else: self._result['d0'] = self._result.get('d0', 0) & ~(1 << idx)
        elif var_name in self.vars:
          cur = int(self.vars[var_name])
          if val: self.vars[var_name] = cur | (1 << idx)
          else: self.vars[var_name] = cur & ~(1 << idx)
    else:
      self.vars[lhs] = int(val) if not isinstance(val, float) else val

  def execute(self, pseudocode: str | list[str], s0: int, s1: int, s2: int = 0, scc: int = 0, d0: int = 0,
              vcc: int = 0, lane: int = 0, exec_mask: int = 0xffffffff, literal: int = 0) -> dict[str, Any]:
    """Execute pseudocode and return dict with results."""
    self.vars = {}
    self.ctx = {'s0': s0, 's1': s1, 's2': s2, 'd0': d0, 'scc': scc, 'vcc': vcc, 'lane': lane, 'exec_mask': exec_mask, 'literal': literal}
    self._result: dict[str, Any] = {'d0': d0, 'scc': scc}

    lines = pseudocode.split('\n') if isinstance(pseudocode, str) else pseudocode
    self._exec_block(lines, 0, len(lines))

    return self._result


# ═══════════════════════════════════════════════════════════════════════════════
# PDF PARSING (only used by generate(), not at runtime)
# ═══════════════════════════════════════════════════════════════════════════════

PDF_URL = "https://docs.amd.com/api/khub/documents/UVVZM22UN7tMUeiW_4ShTQ/content"
INST_PATTERN = re.compile(r'^([SV]_[A-Z0-9_]+)\s+(\d+)\s*$', re.M)

# Op enum classes that have pseudocode in the PDF
from extra.assembly.rdna3.autogen import SOP1Op, SOP2Op, SOPCOp, SOPKOp, SOPPOp, VOP1Op, VOP2Op, VOP3Op, VOP3SDOp, VOPCOp
_OP_ENUMS = [SOP1Op, SOP2Op, SOPCOp, SOPKOp, SOPPOp, VOP1Op, VOP2Op, VOP3Op, VOP3SDOp, VOPCOp]

def _get_defined_ops() -> dict[tuple[str, int], tuple[type, Any]]:
  """Get all ops defined in autogen/__init__.py as {(name, opcode): (enum_cls, enum_val)}."""
  ops = {}
  for enum_cls in _OP_ENUMS:
    for op in enum_cls:
      if op.name.startswith(('S_', 'V_')): ops[(op.name, op.value)] = (enum_cls, op)
  return ops

def extract_pseudocode(text: str) -> str | None:
  """Extract pseudocode from an instruction description snippet. Returns single string or None."""
  lines, result, depth = text.split('\n'), [], 0
  for line in lines:
    s = line.strip()
    if not s: continue
    # Skip page headers/footers
    if re.match(r'^\d+ of \d+$', s): continue
    if re.match(r'^\d+\.\d+\..*Instructions', s): continue
    if s.startswith('"RDNA') or s.startswith('AMD '): continue
    # Stop at notes/examples sections
    if s.startswith('Notes') or s.startswith('Functional examples'): break
    # Track control flow depth
    if s.startswith('if '): depth += 1
    elif s.startswith('endif'): depth = max(0, depth - 1)
    # Skip prose sentences
    if s.endswith('.') and not any(p in s for p in ['D0', 'D1', 'S0', 'S1', 'S2', 'SCC', 'VCC', 'tmp', '=']): continue
    if re.match(r'^[a-z].*\.$', s) and '=' not in s: continue
    # Detect code lines
    is_code = (
      any(p in s for p in ['D0.', 'D1.', 'S0.', 'S1.', 'S2.', 'SCC =', 'SCC ?', 'VCC', 'EXEC', 'tmp =', 'lane =']) or
      s.startswith(('if ', 'else', 'elsif', 'endif', 'declare ', 'for ', '//')) or
      re.match(r'^[a-z_]+\s*=', s) or (depth > 0 and '=' in s)
    )
    if is_code: result.append(s)
  if not result: return None
  # Post-process: add default else clause for if/elsif chains that set D0 but have no else
  # This fixes incomplete pseudocode in the PDF (e.g., V_DIV_SCALE_F32)
  code = '\n'.join(result)
  if 'elsif' in code and 'else\n' not in code and '\nelse' not in code:
    # Check if this is an if/elsif chain that sets D0 in branches
    if re.search(r'elsif.*\n.*D0\.', code) and code.rstrip().endswith('endif'):
      # Determine the type suffix from existing D0 assignments
      m = re.search(r'D0\.(f32|f64|u32|i32|b32|u64|b64)', code)
      suffix = m.group(1) if m else 'f32'
      # Insert else clause before final endif
      code = code.rstrip()
      if code.endswith('endif'):
        code = code[:-5] + f'else\nD0.{suffix} = S0.{suffix}\nendif'
  return code

def parse_pseudocode(pdf_path: str | None = None) -> dict[type, dict[Any, str]]:
  """Parse pseudocode from PDF for all ops defined in autogen/__init__.py. Returns {enum_cls: {op: pseudocode}}."""
  import pdfplumber
  from tinygrad.helpers import fetch

  defined_ops = _get_defined_ops()
  pdf = pdfplumber.open(fetch(PDF_URL) if pdf_path is None else pdf_path)

  # Concatenate all instruction pages into one text blob
  all_text = '\n'.join(pdf.pages[i].extract_text() or '' for i in range(195, 560))

  # Find all instruction headers and their positions
  matches = list(INST_PATTERN.finditer(all_text))
  # Use separate dicts per enum class to avoid IntEnum hash collisions
  instructions: dict[type, dict[Any, str]] = {cls: {} for cls in _OP_ENUMS}

  for i, match in enumerate(matches):
    name, opcode = match.group(1), int(match.group(2))
    # Only process if this op is defined in __init__.py with matching opcode
    key = (name, opcode)
    if key not in defined_ops: continue
    enum_cls, enum_val = defined_ops[key]

    # Extract text until next instruction header
    start = match.end()
    end = matches[i + 1].start() if i + 1 < len(matches) else start + 2000
    snippet = all_text[start:end].strip()
    if (pseudocode := extract_pseudocode(snippet)): instructions[enum_cls][enum_val] = pseudocode

  return instructions

def generate(output_path: pathlib.Path | str | None = None) -> dict[type, dict[Any, str]]:
  """Generate pseudocode data file from PDF. Returns {enum_cls: {op: pseudocode}}."""
  by_cls = parse_pseudocode()

  # Print coverage stats
  total_found, total_ops = 0, 0
  for enum_cls in _OP_ENUMS:
    total = sum(1 for op in enum_cls if op.name.startswith(('S_', 'V_')))
    found = len(by_cls.get(enum_cls, {}))
    total_found += found
    total_ops += total
    print(f"{enum_cls.__name__}: {found}/{total} ({100*found//total if total else 0}%)")
  print(f"Total: {total_found}/{total_ops} ({100*total_found//total_ops}%)")

  if output_path is not None:
    lines = [
      "# autogenerated from AMD RDNA3.5 ISA PDF by pseudocode.py - do not edit",
      "# to regenerate: python -m extra.assembly.rdna3.pseudocode generate",
      "from extra.assembly.rdna3.autogen import SOP1Op, SOP2Op, SOPCOp, SOPKOp, SOPPOp, VOP1Op, VOP2Op, VOP3Op, VOP3SDOp, VOPCOp",
      "",
    ]
    for enum_cls in _OP_ENUMS:
      if not by_cls.get(enum_cls): continue
      lines.append(f"{enum_cls.__name__}_PSEUDOCODE = {{")
      for op in sorted(by_cls[enum_cls].keys(), key=lambda x: x.value):
        lines.append(f"  {enum_cls.__name__}.{op.name}: {by_cls[enum_cls][op]!r},")
      lines.append("}")
      lines.append("")
    pathlib.Path(output_path).write_text('\n'.join(lines) + '\n')

  return by_cls

# ═══════════════════════════════════════════════════════════════════════════════
# RUNTIME: Load pre-generated pseudocode
# ═══════════════════════════════════════════════════════════════════════════════

def get_pseudocode() -> dict[type, dict[Any, str]]:
  """Get pseudocode as {enum_cls: {op: pseudocode_str}}, loading from generated file if available."""
  try:
    from extra.assembly.rdna3.autogen import pseudocode_data as pd
    return {cls: getattr(pd, f"{cls.__name__}_PSEUDOCODE", {}) for cls in _OP_ENUMS}
  except ImportError:
    return generate()


if __name__ == "__main__":
  import sys
  if len(sys.argv) > 1 and sys.argv[1] == "generate":
    output = "extra/assembly/rdna3/autogen/pseudocode_data.py"
    result = generate(output)
    total = sum(len(d) for d in result.values())
    print(f"Generated {output} with {total} instructions")
  else:
    by_cls = get_pseudocode()
    total = sum(len(d) for d in by_cls.values())
    print(f"Loaded {total} instructions")
