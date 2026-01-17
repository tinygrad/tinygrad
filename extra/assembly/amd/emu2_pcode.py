# AMD pcode parser - converts pseudocode to UOps
import re
from tinygrad.dtype import dtypes, DType
from tinygrad.uop.ops import Ops, UOp

DTYPES = {'u32': dtypes.uint32, 'i32': dtypes.int, 'f32': dtypes.float32, 'b32': dtypes.uint32,
          'u64': dtypes.uint64, 'i64': dtypes.int64, 'f64': dtypes.float64, 'b64': dtypes.uint64,
          'u16': dtypes.uint16, 'i16': dtypes.short, 'f16': dtypes.half, 'b16': dtypes.uint16,
          'u1': dtypes.uint32}  # 1-bit treated as uint32 for comparisons

# Binary operators: op_type -> lambda (l, r) -> result
_BINOPS = {
  '|': lambda l, r: l | r, '^': lambda l, r: l ^ r, '&': lambda l, r: l & r,
  '>=': lambda l, r: l >= r, '<=': lambda l, r: l <= r, '==': lambda l, r: l.eq(r), '!=': lambda l, r: l.ne(r),
  '>>': lambda l, r: l >> r, '<<': lambda l, r: l << r, '>': lambda l, r: l > r, '<': lambda l, r: l < r,
  '+': lambda l, r: l + r, '*': lambda l, r: l * r, '/': lambda l, r: l / r,
  '-': lambda l, r: UOp.const(l.dtype, l.arg - r.arg) if l.op == Ops.CONST and r.op == Ops.CONST else l - r,
  '**': lambda l, r: UOp(Ops.EXP2, dtypes.float32, (r.cast(dtypes.float32),)) if l.op == Ops.CONST and l.arg == 2.0 else l,
}

def _apply_pseudocode_fixes(op_name: str, pcode: str) -> str:
  """Apply known fixes for PDF pseudocode bugs."""
  if op_name == 'V_DIV_FMAS_F32':
    pcode = pcode.replace('D0.f32 = 2.0F ** 32 * fma(S0.f32, S1.f32, S2.f32)',
      'D0.f32 = (exponent(S2.f32) > 127) ? (2.0F ** 64 * fma(S0.f32, S1.f32, S2.f32)) : (2.0F ** -64 * fma(S0.f32, S1.f32, S2.f32))')
  if op_name == 'V_DIV_FMAS_F64':
    pcode = pcode.replace('D0.f64 = 2.0 ** 64 * fma(S0.f64, S1.f64, S2.f64)',
      'D0.f64 = (exponent(S2.f64) > 1023) ? (2.0 ** 128 * fma(S0.f64, S1.f64, S2.f64)) : (2.0 ** -128 * fma(S0.f64, S1.f64, S2.f64))')
  if op_name == 'V_DIV_FIXUP_F32':
    pcode = pcode.replace('D0.f32 = sign_out ? -abs(S0.f32) : abs(S0.f32)',
      'D0.f32 = isNAN(S0.f32) ? (sign_out ? -OVERFLOW_F32 : OVERFLOW_F32) : (sign_out ? -abs(S0.f32) : abs(S0.f32))')
  if op_name == 'V_DIV_FIXUP_F64':
    pcode = pcode.replace('D0.f64 = sign_out ? -abs(S0.f64) : abs(S0.f64)',
      'D0.f64 = isNAN(S0.f64) ? (sign_out ? -OVERFLOW_F64 : OVERFLOW_F64) : (sign_out ? -abs(S0.f64) : abs(S0.f64))')
  if 'V_DIV_SCALE' in op_name:
    dt = 'f32' if 'F32' in op_name else 'f64'
    exp_lim, ldexp_val = ('23', '64') if dt == 'f32' else ('52', '128')
    pcode = pcode.replace(f'S2.{dt} / S1.{dt} == DENORM.{dt}', f'isDENORM(S2.{dt} / S1.{dt})')
    pcode = pcode.replace(f"1.0 / 64'F(S1.{dt}) == DENORM.f64", f"isDENORM(1.0 / 64'F(S1.{dt}))")
    pcode = pcode.replace(f'1.0 / S1.{dt} == DENORM.{dt}', f'isDENORM(1.0 / S1.{dt})')
    pcode = pcode.replace(f'S1.{dt} == DENORM.{dt}', f'isDENORM(S1.{dt})')
    pcode = pcode.replace(f'D0.{dt} = NAN.{dt}', f'VCC = 0x1LL;\nD0.{dt} = NAN.{dt}')
    pcode = pcode.replace(f'elsif isDENORM(S1.{dt}) then\nD0.{dt} = ldexp(S0.{dt}, {ldexp_val})', f'elsif 1 == 0 then\nD0.{dt} = S0.{dt}')
    pcode = pcode.replace(f'elsif exponent(S2.{dt}) <= {exp_lim} then\n// Numerator is tiny\nD0.{dt} = ldexp(S0.{dt}, {ldexp_val})',
                          f'elsif exponent(S2.{dt}) <= {exp_lim} then\nVCC = 0x1LL;\nD0.{dt} = ldexp(S0.{dt}, {ldexp_val})')
    pcode = pcode.replace(f'elsif isDENORM(S2.{dt} / S1.{dt}) then\nVCC = 0x1LL;\nif S0.{dt} == S2.{dt} then\n// Only scale the numerator\nD0.{dt} = ldexp(S0.{dt}, {ldexp_val})\nendif',
                          f'elsif isDENORM(S2.{dt} / S1.{dt}) then\nVCC = 0x1LL;\nD0.{dt} = S0.{dt}')
    pcode = pcode.replace(f'D0.{dt} = ldexp(S0.{dt}, {ldexp_val})\nendif\nelsif', f'D0.{dt} = ldexp(S0.{dt}, {ldexp_val})\nelse\nD0.{dt} = S0.{dt}\nendif\nelsif')
    lines = pcode.rstrip().split('\n')
    for i in range(len(lines) - 1, -1, -1):
      if lines[i].strip() == 'endif':
        lines.insert(i, f'else\nD0.{dt} = S0.{dt}')
        break
    pcode = '\n'.join(lines) + f';\nif isDENORM(S1.{dt}) then\nD0.{dt} = NAN.{dt}\nendif'
  if op_name == 'V_TRIG_PREOP_F64':
    pcode = pcode.replace("result = 64'F((1201'B(2.0 / PI)[1200 : 0] << shift.u32) & 1201'0x1fffffffffffff)", "result = trig_preop_result(shift)")
  return pcode

def parse_pcode(pcode: str, srcs: dict[str, UOp] | None = None, lane: UOp | None = None, op_name: str | None = None) -> tuple[dict[str, UOp], list[tuple[str, UOp]]]:
  """Parse pcode into UOps. Returns (vars, assigns) where assigns are (dest, value) tuples."""
  if op_name: pcode = _apply_pseudocode_fixes(op_name, pcode)
  vars: dict[str, UOp] = {n: UOp(Ops.DEFINE_VAR, dtypes.uint32, (), (n, UOp.const(dtypes.uint32, 0), UOp.const(dtypes.uint32, 0xFFFFFFFF)))
                          for n in ['S0', 'S1', 'S2', 'D0', 'VCC', 'EXEC', 'SCC', 'SIMM32']}
  if srcs: vars.update(srcs)
  vars['laneId'] = lane if lane is not None else UOp.const(dtypes.uint32, 0)
  assigns: list[tuple[str, UOp]] = []

  def parse_block(lines: list[str], start: int = 0) -> tuple[int, dict[str, UOp]]:
    """Parse a block of statements, returns (next_line_idx, block_assigns)."""
    block_assigns: dict[str, UOp] = {}
    i = start
    while i < len(lines):
      line = lines[i]
      # End of block markers
      if re.match(r'(elsif|else|endif|endfor)\b', line, re.IGNORECASE): break
      # For loop: unroll into nested WHERE
      if (m := re.match(r'for\s+(\w+)\s+in\s+(\d+)\s*:\s*(\d+)\s+do', line, re.IGNORECASE)):
        loop_var, start_val, end_val = m.group(1), int(m.group(2)), int(m.group(3))
        i += 1
        # Collect loop body
        body_lines, depth = [], 1
        while i < len(lines) and depth > 0:
          if re.match(r'for\s+', lines[i], re.IGNORECASE): depth += 1
          elif re.match(r'endfor\b', lines[i], re.IGNORECASE): depth -= 1
          if depth > 0: body_lines.append(lines[i])
          i += 1
        # Find condition and assignment in body (handles CLZ/CTZ pattern)
        cond_expr, assign_var, assign_expr = None, None, None
        for bl in body_lines:
          if (m2 := re.match(r'if\s+(.+?)\s+then', bl, re.IGNORECASE)): cond_expr = m2.group(1)
          elif (m2 := re.match(r'(\w+)(?:\.\w+)?\s*=\s*(.+)', bl)) and 'break' not in bl.lower():
            assign_var, assign_expr = m2.group(1), m2.group(2).rstrip(';').strip()
        if cond_expr and assign_var and assign_expr:
          # Unroll loop backwards to build nested WHERE
          result = block_assigns.get(assign_var, vars.get(assign_var, UOp.const(dtypes.int, -1)))
          for loop_i in range(end_val, start_val - 1, -1):
            loop_vars = vars.copy(); loop_vars.update(block_assigns)
            loop_vars[loop_var] = UOp.const(dtypes.uint32, loop_i)
            cond = parse_expr(cond_expr, loop_vars)
            if cond.dtype != dtypes.bool: cond = cond.ne(UOp.const(cond.dtype, 0))
            assign_val = parse_expr(assign_expr, loop_vars)
            if assign_val.dtype != result.dtype: assign_val = assign_val.cast(result.dtype)
            result = cond.where(assign_val, result)
          block_assigns[assign_var] = result
          vars[assign_var] = result
        continue
      # If/elsif/else block: build nested WHERE
      if (m := re.match(r'if\s+(.+?)\s+then$', line, re.IGNORECASE)):
        conditions: list[tuple[UOp, dict[str, UOp]]] = []  # [(cond, {var: val}), ...]
        else_assigns: dict[str, UOp] = {}
        cond = parse_expr(m.group(1), {**vars, **block_assigns})
        if cond.dtype != dtypes.bool: cond = cond.ne(UOp.const(cond.dtype, 0))
        i += 1
        i, branch_assigns = parse_block(lines, i)
        conditions.append((cond, branch_assigns))
        # Handle elsif/else chains
        while i < len(lines):
          if (m := re.match(r'elsif\s+(.+?)\s+then$', lines[i], re.IGNORECASE)):
            cond = parse_expr(m.group(1), {**vars, **block_assigns})
            if cond.dtype != dtypes.bool: cond = cond.ne(UOp.const(cond.dtype, 0))
            i += 1
            i, branch_assigns = parse_block(lines, i)
            conditions.append((cond, branch_assigns))
          elif re.match(r'else$', lines[i], re.IGNORECASE):
            i += 1
            i, else_assigns = parse_block(lines, i)
          elif re.match(r'endif\b', lines[i], re.IGNORECASE):
            i += 1; break
          else: break
        # Build nested WHERE for each variable
        all_vars = set()
        for _, ba in conditions: all_vars.update(ba.keys())
        all_vars.update(else_assigns.keys())
        for var in all_vars:
          result = else_assigns.get(var, block_assigns.get(var, vars.get(var, UOp.const(dtypes.uint32, 0))))
          for cond, ba in reversed(conditions):
            if var in ba: result = cond.where(ba[var], result)
          block_assigns[var] = result
          vars[var] = result
        continue
      # MEM[addr].type = value
      if (m := re.match(r'MEM\[(.+)\]\.(\w+)\s*=\s*(.+)', line)):
        addr = parse_expr(m.group(1), {**vars, **block_assigns})
        val = parse_expr(m.group(3), {**vars, **block_assigns})
        assigns.append((f'MEM[{m.group(1)}].{m.group(2)}', (addr, val)))
        i += 1; continue
      # MEM[addr].type += value
      if (m := re.match(r'MEM\[(.+)\]\.(\w+)\s*\+=\s*(.+)', line)):
        addr = parse_expr(m.group(1), {**vars, **block_assigns})
        lds = vars.get('_lds')
        if lds is not None:
          idx = (addr >> UOp.const(dtypes.uint32, 2)).cast(dtypes.index)
          new_val = lds.index(idx) + parse_expr(m.group(3), {**vars, **block_assigns})
          assigns.append((f'MEM[{m.group(1)}].{m.group(2)}', (addr, new_val)))
        i += 1; continue
      # VAR[high:low] = value
      if (m := re.match(r'(\w+)\[(\d+)\s*:\s*(\d+)\]\s*=\s*(.+)', line)):
        var_name, high_bit, low_bit = m.group(1), int(m.group(2)), int(m.group(3))
        val = parse_expr(m.group(4), {**vars, **block_assigns})
        assigns.append((f'{var_name}[{high_bit}:{low_bit}]', val))
        if var_name not in vars: vars[var_name] = UOp.const(dtypes.uint64 if high_bit >= 32 else dtypes.uint32, 0)
        i += 1; continue
      # Compound assignment: VAR += value, VAR -= value
      if (m := re.match(r'(\w+(?:\.\w+)?)\s*\+=\s*(.+)', line)):
        var_name = m.group(1).split('.')[0]
        old_val = block_assigns.get(var_name, vars.get(var_name, UOp.const(dtypes.uint32, 0)))
        block_assigns[var_name] = old_val + parse_expr(m.group(2), {**vars, **block_assigns})
        i += 1; continue
      if (m := re.match(r'(\w+(?:\.\w+)?)\s*-=\s*(.+)', line)):
        var_name = m.group(1).split('.')[0]
        old_val = block_assigns.get(var_name, vars.get(var_name, UOp.const(dtypes.uint32, 0)))
        block_assigns[var_name] = old_val - parse_expr(m.group(2), {**vars, **block_assigns})
        i += 1; continue
      # Regular assignment: VAR = value
      if (m := re.match(r'(\w+(?:\.\w+)?(?:\[\w+\])?)\s*=\s*(.+)', line)) and not re.search(r'[<>=!]=', line[:line.find('=')]):
        lhs, val = m.group(1), parse_expr(m.group(2), {**vars, **block_assigns})
        base = re.match(r'(\w+)', lhs).group(1)
        block_assigns[base] = val
        vars[base] = val
        i += 1; continue
      # Declaration: declare VAR : TYPE
      if (m := re.match(r'declare\s+(\w+)', line)):
        vars[m.group(1)] = UOp.const(dtypes.uint32, 0)
        i += 1; continue
      i += 1
    return i, block_assigns

  lines = [l.strip() for l in pcode.split('\n') if l.strip() and not l.strip().startswith('//')]
  _, final_assigns = parse_block(lines)

  # Build assigns from final values
  for var, val in final_assigns.items():
    if var in ['D0', 'SCC', 'VCC', 'EXEC', 'PC']:
      # Find the type suffix if any from the original pcode
      for line in lines:
        if (m := re.match(rf'{var}\.(\w+)', line)):
          assigns.append((f'{var}.{m.group(1)}', val)); break
      else:
        assigns.append((var, val))
  return vars, assigns

def parse_expr(expr: str, vars: dict[str, UOp]) -> UOp:
  """Parse an expression into a UOp."""
  expr = expr.strip()
  # Balanced parentheses
  if expr.startswith('(') and expr.endswith(')'):
    depth = 0
    for i, c in enumerate(expr):
      depth += (c == '(') - (c == ')')
      if depth == 0 and i < len(expr) - 1: break
    else: return parse_expr(expr[1:-1], vars)
  # Ternary - find ? and : at depth 0
  if '?' in expr:
    depth_paren, depth_bracket, q_pos, c_pos = 0, 0, -1, -1
    for i, ch in enumerate(expr):
      if ch == '(': depth_paren += 1
      elif ch == ')': depth_paren -= 1
      elif ch == '[': depth_bracket += 1
      elif ch == ']': depth_bracket -= 1
      elif ch == '?' and depth_paren == 0 and depth_bracket == 0: q_pos = i
      elif ch == ':' and depth_paren == 0 and depth_bracket == 0 and q_pos >= 0: c_pos = i; break
    if q_pos >= 0 and c_pos >= 0:
      cond = parse_expr(expr[:q_pos].strip(), vars)
      if cond.dtype != dtypes.bool: cond = cond.ne(UOp.const(cond.dtype, 0))
      return cond.where(parse_expr(expr[q_pos+1:c_pos].strip(), vars), parse_expr(expr[c_pos+1:].strip(), vars))
  # Binary ops (low to high precedence) - search right-to-left
  ops = [('||', '|'), ('&&', '&'), ('|', '|'), ('^', '^'), ('&', '&'), ('>=', '>='), ('<=', '<='), ('==', '=='),
         ('!=', '!='), ('<>', '!='), ('>>', '>>'), ('<<', '<<'), ('>', '>'), ('<', '<'), ('+', '+'), ('-', '-'), ('*', '*'), ('/', '/'), ('**', '**')]
  for op, op_type in ops:
    depths, bdepths = [0] * (len(expr) + 1), [0] * (len(expr) + 1)
    for i in range(len(expr) - 1, -1, -1):
      depths[i], bdepths[i] = depths[i+1], bdepths[i+1]
      if expr[i] == ')': depths[i] += 1
      elif expr[i] == '(': depths[i] -= 1
      elif expr[i] == ']': bdepths[i] += 1
      elif expr[i] == '[': bdepths[i] -= 1
    for i in range(len(expr)-len(op), -1, -1):
      if depths[i] == 0 and bdepths[i] == 0 and expr[i:i+len(op)] == op:
        if len(op) == 1 and i+1 < len(expr) and expr[i+1] in '=<>&|*': continue
        if len(op) == 1 and i > 0 and expr[i-1] in '=<>&|*': continue
        lhs, rhs = expr[:i].strip(), expr[i+len(op):].strip()
        if lhs and rhs:
          l, r = parse_expr(lhs, vars), parse_expr(rhs, vars)
          if op_type in ('>>', '<<', '>=', '<=', '==', '!=', '>', '<') and l.dtype != r.dtype: r = r.cast(l.dtype)
          if op_type in ('|', '^', '&') and l.dtype != r.dtype:
            if l.dtype.itemsize == r.dtype.itemsize:
              target = dtypes.uint32 if l.dtype.itemsize == 4 else dtypes.uint64 if l.dtype.itemsize == 8 else l.dtype
              l, r = l.bitcast(target), r.bitcast(target)
            else: r = r.cast(l.dtype)
          return _BINOPS[op_type](l, r)
  # Type cast: 64'U(...)
  if (m := re.match(r"(\d+)'([UIFB])\((.+)\)", expr)):
    dt = {('U',32): dtypes.uint32, ('U',64): dtypes.uint64, ('I',32): dtypes.int, ('I',64): dtypes.int64,
          ('F',32): dtypes.float32, ('F',64): dtypes.float64}.get((m.group(2), int(m.group(1))), dtypes.uint32)
    return parse_expr(m.group(3), vars).cast(dt)
  # Lane-indexed: VCC.u64[laneId]
  if (m := re.match(r'([a-zA-Z_]\w*)\.u64\[laneId\](?:\.(\w+))?$', expr)):
    v = vars.get(m.group(1), UOp.const(dtypes.uint32, 0))
    lane = vars['laneId'].cast(dtypes.uint32) if vars['laneId'].dtype != dtypes.uint32 else vars['laneId']
    result = (v >> lane) & UOp.const(dtypes.uint32, 1)
    if m.group(2): result = result.cast(DTYPES.get(m.group(2), dtypes.uint32))
    return result
  # Variable with type: S0.u32
  if (m := re.match(r'([a-zA-Z_]\w*)\.(\w+)$', expr)):
    v, dt = vars.get(m.group(1), UOp.const(dtypes.uint32, 0)), DTYPES.get(m.group(2), dtypes.uint32)
    if dt == v.dtype: return v
    if dt.itemsize == 2 and v.dtype.itemsize == 4:
      v16 = (v & UOp.const(v.dtype, 0xFFFF)).cast(dtypes.uint16)
      return v16 if dt == dtypes.uint16 else v16.bitcast(dt)
    return v.cast(dt) if dt.itemsize != v.dtype.itemsize else v.bitcast(dt)
  # Bit slice: S0[4:0] or S0[4:0].u32
  if (m := re.match(r'([a-zA-Z_]\w*)\[(\d+)\s*:\s*(\d+)\](?:\.(\w+))?$', expr)):
    hi, lo = int(m.group(2)), int(m.group(3))
    return (vars.get(m.group(1), UOp.const(dtypes.uint32, 0)) >> UOp.const(dtypes.uint32, lo)) & UOp.const(dtypes.uint32, (1<<(hi-lo+1))-1)
  # Bit slice with type prefix: S1.u32[4:0].u32
  if (m := re.match(r'([a-zA-Z_]\w*)\.(\w+)\[(\d+)\s*:\s*(\d+)\](?:\.(\w+))?$', expr)):
    hi, lo = int(m.group(3)), int(m.group(4))
    return (vars.get(m.group(1), UOp.const(dtypes.uint32, 0)) >> UOp.const(dtypes.uint32, lo)) & UOp.const(dtypes.uint32, (1<<(hi-lo+1))-1)
  # Single bit access: tmp.u32[31]
  if (m := re.match(r'([a-zA-Z_]\w*)\.(\w+)\[(.+)\]$', expr)):
    v = vars.get(m.group(1), UOp.const(dtypes.uint32, 0))
    dt = DTYPES.get(m.group(2), dtypes.uint32)
    if v.dtype != dt: v = v.cast(dt) if dt.itemsize != v.dtype.itemsize else v.bitcast(dt)
    if v.dtype != dtypes.uint32: v = v.bitcast(dtypes.uint32) if v.dtype.itemsize == 4 else v.cast(dtypes.uint32)
    bit = UOp.const(dtypes.uint32, int(m.group(3))) if m.group(3).isdigit() else parse_expr(m.group(3), vars).cast(dtypes.uint32)
    return (v >> bit) & UOp.const(dtypes.uint32, 1)
  # Literals
  if (m := re.match(r'0x([0-9a-fA-F]+)', expr)): return UOp.const(dtypes.uint64, int(m.group(1), 16))
  # Typed constants: 16'4, 1'0U, 1'1U, 64'0x... etc
  if (m := re.match(r"(\d+)'(\d+)U?", expr)):
    bits, val = int(m.group(1)), int(m.group(2))
    dt = {1: dtypes.uint32, 8: dtypes.uint8, 16: dtypes.int16 if 'U' not in expr else dtypes.uint16,
          32: dtypes.int if 'U' not in expr else dtypes.uint32, 64: dtypes.int64 if 'U' not in expr else dtypes.uint64}.get(bits, dtypes.uint32)
    return UOp.const(dt, val)
  if (m := re.match(r'(-?\d+)[UL]*$', expr)):
    val = int(m.group(1))
    return UOp.const(dtypes.int if val < 0 else dtypes.uint32, val)
  if (m := re.match(r'-?(\d+\.\d+)F?$', expr)): return UOp.const(dtypes.float32, float(expr.rstrip('F')))
  if (m := re.match(r'-?(\d+)F$', expr)): return UOp.const(dtypes.float32, float(expr.rstrip('F')))
  # Unary NOT
  if expr.startswith('~'):
    inner = parse_expr(expr[1:], vars)
    return inner ^ UOp.const(inner.dtype, (1 << (inner.dtype.itemsize * 8)) - 1)
  # Unary minus
  if expr.startswith('-') and len(expr) > 1 and expr[1] not in '0123456789':
    return parse_expr(expr[1:], vars).neg()
  # Variable
  if expr in vars: return vars[expr]
  if expr == 'PI': return UOp.const(dtypes.float32, 3.141592653589793)
  # Brace concatenation: { hi, lo }
  if (m := re.match(r'\{\s*(.+?)\s*,\s*(.+?)\s*\}', expr)):
    high = parse_expr(m.group(1), vars).cast(dtypes.uint64)
    low = parse_expr(m.group(2), vars).cast(dtypes.uint64)
    return (high << UOp.const(dtypes.uint64, 32)) | low
  # MEM[addr].type
  if (m := re.match(r'MEM\[(.+)\]\.(\w+)', expr)):
    addr = parse_expr(m.group(1), vars)
    if addr.dtype != dtypes.uint32: addr = addr.cast(dtypes.uint32)
    dt = DTYPES.get(m.group(2), dtypes.uint32)
    lds = vars.get('_lds')
    if lds is None: return UOp.const(dt, 0)
    idx = (addr >> UOp.const(dtypes.uint32, 2)).cast(dtypes.index)
    val = lds.index(idx)  # Don't call .load() - pm_add_loads will add it
    if dt in (dtypes.uint64, dtypes.int64):
      hi_idx = ((addr + UOp.const(dtypes.uint32, 4)) >> UOp.const(dtypes.uint32, 2)).cast(dtypes.index)
      val = val.cast(dtypes.uint64) | (lds.index(hi_idx).cast(dtypes.uint64) << UOp.const(dtypes.uint64, 32))
    return val
  # VAR[high:low]
  if (m := re.match(r'(\w+)\[(\d+)\s*:\s*(\d+)\]', expr)):
    var_name, high_bit, low_bit = m.group(1), int(m.group(2)), int(m.group(3))
    if var_name in vars:
      val = vars[var_name]
      width = high_bit - low_bit + 1
      # Cast shift amount to match value dtype to preserve result dtype
      shift_dt = dtypes.uint64 if val.dtype in (dtypes.uint64, dtypes.int64) else dtypes.uint32
      shifted = val >> UOp.const(shift_dt, low_bit) if low_bit > 0 else val
      return shifted & UOp.const(shifted.dtype, (1 << width) - 1)
    return UOp.const(dtypes.uint32, 0)
  # Function call
  if (result := _parse_func(expr, vars)) is not None: return result
  return UOp.const(dtypes.uint32, 0)

# Function implementations
def _floor(x: UOp) -> UOp:
  truncated = UOp(Ops.TRUNC, x.dtype, (x,))
  needs_adjust = (x < UOp.const(x.dtype, 0)) & x.ne(truncated)
  return needs_adjust.where(truncated - UOp.const(x.dtype, 1), truncated)

def _f16_extract(v: UOp) -> UOp:
  return (v & UOp.const(dtypes.uint32, 0xFFFF)).cast(dtypes.uint16).bitcast(dtypes.half) if v.dtype == dtypes.uint32 else v

def _isnan_f32(v: UOp) -> UOp:
  v32 = v.bitcast(dtypes.uint32) if v.dtype == dtypes.float32 else v
  return (v32 & UOp.const(dtypes.uint32, 0x7F800000)).eq(UOp.const(dtypes.uint32, 0x7F800000)) & \
         (v32 & UOp.const(dtypes.uint32, 0x007FFFFF)).ne(UOp.const(dtypes.uint32, 0))

def _check_nan(inner: str, vars: dict, quiet: bool) -> UOp:
  if (m := re.match(r"64'F\((.+)\)", inner)): inner = m.group(1)
  v = parse_expr(inner, vars)
  is_f16 = '.f16' in inner or v.dtype == dtypes.half
  if is_f16:
    v16 = (v & UOp.const(dtypes.uint32, 0xFFFF)) if v.dtype == dtypes.uint32 else v.bitcast(dtypes.uint16).cast(dtypes.uint32)
    exp_mask, mant_mask, quiet_bit = UOp.const(dtypes.uint32, 0x7C00), UOp.const(dtypes.uint32, 0x03FF), UOp.const(dtypes.uint32, 0x0200)
    is_nan_exp, has_mant, is_quiet = (v16 & exp_mask).eq(exp_mask), (v16 & mant_mask).ne(UOp.const(dtypes.uint32, 0)), (v16 & quiet_bit).ne(UOp.const(dtypes.uint32, 0))
  else:
    v32 = v.bitcast(dtypes.uint32) if v.dtype == dtypes.float32 else v
    exp_mask, mant_mask, quiet_bit = UOp.const(dtypes.uint32, 0x7F800000), UOp.const(dtypes.uint32, 0x007FFFFF), UOp.const(dtypes.uint32, 0x00400000)
    is_nan_exp, has_mant, is_quiet = (v32 & exp_mask).eq(exp_mask), (v32 & mant_mask).ne(UOp.const(dtypes.uint32, 0)), (v32 & quiet_bit).ne(UOp.const(dtypes.uint32, 0))
  return (is_nan_exp & is_quiet) if quiet else (is_nan_exp & has_mant & is_quiet.logical_not())

def _minmax_reduce(is_max: bool, dt: DType, args: list[UOp]) -> UOp:
  def cast(v): return v.bitcast(dt) if dt == dtypes.float32 and v.dtype == dtypes.uint32 else v.cast(dt)
  result = cast(args[0])
  for a in args[1:]: result = result.maximum(cast(a)) if is_max else result.minimum(cast(a))
  return result

# Table of function parsers: (regex_pattern, num_args, handler)
_FUNC_TABLE: list[tuple[str, int, callable]] = []

def _register_funcs():
  global _FUNC_TABLE
  # Unary math functions
  for name, op in [('sqrt', Ops.SQRT), ('trunc', Ops.TRUNC), ('log2', Ops.LOG2), ('sin', Ops.SIN)]:
    _FUNC_TABLE.append((rf'{name}\((.+)\)', 1, lambda a, v, m, op=op: UOp(op, a[0].dtype, (a[0],))))
  _FUNC_TABLE.append((r'cos\((.+)\)', 1, lambda a, v, m: UOp(Ops.SIN, a[0].dtype, (a[0] + UOp.const(a[0].dtype, 1.5707963267948966),))))
  _FUNC_TABLE.append((r'floor\((.+)\)', 1, lambda a, v, m: _floor(a[0])))
  _FUNC_TABLE.append((r'fract\((.+)\)', 1, lambda a, v, m: a[0] - _floor(a[0])))
  def _signext(a, v, m):
    val = a[0]
    # Sign extend to 64-bit for PC calculations
    if val.dtype in (dtypes.int16, dtypes.short):
      return val.cast(dtypes.int64)
    if val.dtype in (dtypes.int, dtypes.int32):
      return val.cast(dtypes.int64)
    return val
  _FUNC_TABLE.append((r'signext\((.+)\)', 1, _signext))
  _FUNC_TABLE.append((r'isEven\((.+)\)', 1, lambda a, v, m: (UOp(Ops.TRUNC, a[0].dtype, (a[0],)).cast(dtypes.int) & UOp.const(dtypes.int, 1)).eq(UOp.const(dtypes.int, 0))))
  _FUNC_TABLE.append((r'abs\((.+)\)', 1, lambda a, v, m: (a[0].bitcast(dtypes.uint32) & UOp.const(dtypes.uint32, 0x7FFFFFFF)).bitcast(dtypes.float32) if a[0].dtype == dtypes.float32 else
                                                         (a[0].cast(dtypes.uint64) & UOp.const(dtypes.uint64, 0x7FFFFFFFFFFFFFFF)).bitcast(dtypes.float64) if a[0].dtype == dtypes.float64 else
                                                         (a[0].bitcast(dtypes.uint16) & UOp.const(dtypes.uint16, 0x7FFF)).bitcast(dtypes.half) if a[0].dtype == dtypes.half else a[0]))
  # Binary math functions
  _FUNC_TABLE.append((r'max\((.+),\s*(.+)\)', 2, lambda a, v, m: UOp(Ops.MAX, a[0].dtype, (a[0], a[1]))))
  _FUNC_TABLE.append((r'min\((.+),\s*(.+)\)', 2, lambda a, v, m: UOp(Ops.MAX, a[0].dtype, (a[0].neg(), a[1].neg())).neg()))
  _FUNC_TABLE.append((r'pow\((.+),\s*(.+)\)', 2, lambda a, v, m: UOp(Ops.EXP2, dtypes.float32, (a[1].bitcast(dtypes.float32),)) if '2.0' in m.group(1) else a[0]))
  # Ternary math functions
  _FUNC_TABLE.append((r'fma\((.+),\s*(.+),\s*(.+)\)', 3, lambda a, v, m: a[0] * a[1] + a[2]))
  # Type conversions
  for src, dst in [('i32', dtypes.float32), ('u32', dtypes.float32)]:
    _FUNC_TABLE.append((rf'{src}_to_f32\((.+)\)', 1, lambda a, v, m, d=dst: a[0].cast(dtypes.int if 'i32' in m.group(0) else dtypes.uint32).cast(d)))
  for src, dst in [('f32', dtypes.int), ('f32', dtypes.uint32)]:
    _FUNC_TABLE.append((rf'f32_to_{dst.name.replace("dtypes.", "")}\((.+)\)', 1, lambda a, v, m, d=dst: UOp(Ops.TRUNC, a[0].dtype, (a[0],)).cast(d)))
  _FUNC_TABLE.append((r'f16_to_f32\((.+)\)', 1, lambda a, v, m: _f16_extract(a[0]).cast(dtypes.float32)))
  _FUNC_TABLE.append((r'f32_to_f16\((.+)\)', 1, lambda a, v, m: a[0].cast(dtypes.half)))
  _FUNC_TABLE.append((r'f16_to_i16\((.+)\)', 1, lambda a, v, m: UOp(Ops.TRUNC, _f16_extract(a[0]).dtype, (_f16_extract(a[0]),)).cast(dtypes.int16)))
  _FUNC_TABLE.append((r'f16_to_u16\((.+)\)', 1, lambda a, v, m: UOp(Ops.TRUNC, _f16_extract(a[0]).dtype, (_f16_extract(a[0]),)).cast(dtypes.uint16)))
  # Float classification
  _FUNC_TABLE.append((r'isNAN\((.+)\)', 1, lambda a, v, m: _isnan_f32(a[0])))
  _FUNC_TABLE.append((r'isSignalNAN\((.+)\)', 0, lambda a, v, m: _check_nan(m.group(1), v, quiet=False)))
  _FUNC_TABLE.append((r'isQuietNAN\((.+)\)', 0, lambda a, v, m: _check_nan(m.group(1), v, quiet=True)))
  _FUNC_TABLE.append((r'exponent\((.+)\)', 1, lambda a, v, m: ((a[0].bitcast(dtypes.uint16) if a[0].dtype == dtypes.half else (a[0] & UOp.const(dtypes.uint32, 0xFFFF)).cast(dtypes.uint16)).cast(dtypes.uint32) >> UOp.const(dtypes.uint32, 10)) & UOp.const(dtypes.uint32, 0x1F) if '.f16' in m.group(1) or a[0].dtype == dtypes.half else
                                                              ((a[0].bitcast(dtypes.uint32) if a[0].dtype == dtypes.float32 else a[0]) >> UOp.const(dtypes.uint32, 23)) & UOp.const(dtypes.uint32, 0xFF)))
  _FUNC_TABLE.append((r'sign\((.+)\)', 1, lambda a, v, m: ((a[0].bitcast(dtypes.uint16) if a[0].dtype == dtypes.half else (a[0] & UOp.const(dtypes.uint32, 0xFFFF)).cast(dtypes.uint16)).cast(dtypes.uint32) >> UOp.const(dtypes.uint32, 15)) & UOp.const(dtypes.uint32, 1) if '.f16' in m.group(1) or a[0].dtype == dtypes.half else
                                                          ((a[0].bitcast(dtypes.uint32) if a[0].dtype == dtypes.float32 else a[0]) >> UOp.const(dtypes.uint32, 31)) & UOp.const(dtypes.uint32, 1)))
  # signext_from_bit
  def _signext_from_bit(a, v, m):
    val, width = a[0].cast(dtypes.uint32), a[1].cast(dtypes.uint32)
    sign_bit = (val >> (width - UOp.const(dtypes.uint32, 1))) & UOp.const(dtypes.uint32, 1)
    mask = (UOp.const(dtypes.uint32, 1) << width) - UOp.const(dtypes.uint32, 1)
    return sign_bit.ne(UOp.const(dtypes.uint32, 0)).where(val | (mask ^ UOp.const(dtypes.uint32, 0xFFFFFFFF)), val)
  _FUNC_TABLE.append((r'signext_from_bit\((.+),\s*(.+)\)', 2, _signext_from_bit))
  # AMD v_min/v_max functions
  for is_max, name in [(False, 'min'), (True, 'max')]:
    for dt, suffix in [(dtypes.float32, 'f32'), (dtypes.int, 'i32'), (dtypes.uint32, 'u32')]:
      _FUNC_TABLE.append((rf'v_{name}_{suffix}\((.+),\s*(.+)\)', 2, lambda a, v, m, is_max=is_max, dt=dt: _minmax_reduce(is_max, dt, a)))
      _FUNC_TABLE.append((rf'v_{name}3_{suffix}\((.+),\s*(.+),\s*(.+)\)', 3, lambda a, v, m, is_max=is_max, dt=dt: _minmax_reduce(is_max, dt, a)))

_register_funcs()

def _parse_func(expr: str, vars: dict[str, UOp]) -> UOp | None:
  for pattern, nargs, handler in _FUNC_TABLE:
    if (m := re.match(pattern, expr)):
      args = [parse_expr(m.group(i+1), vars) for i in range(nargs)] if nargs > 0 else []
      return handler(args, vars, m)
  return None
