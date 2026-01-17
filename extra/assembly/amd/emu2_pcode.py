# AMD pcode parser - converts pseudocode to UOps
import re
from tinygrad.dtype import dtypes, DType
from tinygrad.uop.ops import Ops, UOp

DTYPES = {'u32': dtypes.uint32, 'i32': dtypes.int, 'f32': dtypes.float32, 'b32': dtypes.uint32,
          'u64': dtypes.uint64, 'i64': dtypes.int64, 'f64': dtypes.float64, 'b64': dtypes.uint64,
          'u16': dtypes.uint16, 'i16': dtypes.short, 'f16': dtypes.half, 'b16': dtypes.uint16,
          'u8': dtypes.uint8, 'i8': dtypes.int8, 'b8': dtypes.uint8,
          'u1': dtypes.uint32}  # 1-bit treated as uint32 for comparisons

# Binary operators: op_type -> lambda (l, r) -> result
_BINOPS = {
  '|': lambda l, r: l | r, '^': lambda l, r: l ^ r, '&': lambda l, r: l & r,
  '>=': lambda l, r: l >= r, '<=': lambda l, r: l <= r, '==': lambda l, r: l.eq(r), '!=': lambda l, r: l.ne(r),
  '>>': lambda l, r: l >> r, '<<': lambda l, r: l << r, '>': lambda l, r: l > r, '<': lambda l, r: l < r,
  '+': lambda l, r: l + (r.cast(l.dtype) if l.dtype.itemsize > r.dtype.itemsize else r),
  '*': lambda l, r: l * (r.cast(l.dtype) if l.dtype != r.dtype and l.dtype in (dtypes.float32, dtypes.float64) else r),
  '/': lambda l, r: l / r,
  '-': lambda l, r: UOp.const(l.dtype, l.arg - r.arg) if l.op == Ops.CONST and r.op == Ops.CONST else l - r,
  '**': lambda l, r: UOp(Ops.EXP2, l.dtype, (r.cast(l.dtype),)) if l.op == Ops.CONST and l.arg == 2.0 else l,
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
      'D0.f32 = isNAN(S0.f32) ? (sign_out ? -INF.f32 : +INF.f32) : (sign_out ? -abs(S0.f32) : abs(S0.f32))')
  if op_name == 'V_DIV_FIXUP_F64':
    pcode = pcode.replace('D0.f64 = sign_out ? -abs(S0.f64) : abs(S0.f64)',
      'D0.f64 = isNAN(S0.f64) ? (sign_out ? -INF : +INF) : (sign_out ? -abs(S0.f64) : abs(S0.f64))')
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
  vars['WAVE_MODE'] = {'IEEE': UOp.const(dtypes.uint32, 1)}  # IEEE mode is the default
  assigns: list[tuple[str, UOp]] = []

  def parse_block(lines: list[str], start: int = 0) -> tuple[int, dict[str, UOp]]:
    """Parse a block of statements, returns (next_line_idx, block_assigns)."""
    block_assigns: dict[str, UOp] = {}
    i = start
    while i < len(lines):
      line = lines[i]
      # End of block markers
      if re.match(r'(elsif|else|endif|endfor)\b', line, re.IGNORECASE): break
      # For loop: unroll and execute body for each iteration
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
        # Unroll: execute body for each iteration value
        for loop_i in range(start_val, end_val + 1):
          # Substitute loop variable in body lines
          subst_lines = []
          for bl in body_lines:
            # Replace .type[loop_var] patterns FIRST: OPSEL.u3[i] -> OPSEL.u3[2]
            subst = re.sub(rf'\.(\w+)\[{loop_var}\]', rf'.\g<1>[{loop_i}]', bl)
            # Replace array indexing patterns: VAR[loop_var] -> VAR{loop_i} (but not .type[i])
            subst = re.sub(rf'(?<!\.)\b(\w+)\[{loop_var}\]', rf'\g<1>{{{loop_i}}}', subst)
            subst_lines.append(subst)
          # Parse substituted body as a nested block
          _, iter_assigns = parse_block(subst_lines, 0)
          block_assigns.update(iter_assigns)
          vars.update(iter_assigns)
        continue
      # If/elsif/else block: build nested WHERE
      if (m := re.match(r'if\s+(.+?)\s+then$', line, re.IGNORECASE)):
        conditions: list[tuple[UOp, dict[str, UOp]]] = []  # [(cond, {var: val}), ...]
        else_assigns: dict[str, UOp] = {}
        cond = parse_expr(m.group(1), {**vars, **block_assigns})
        if cond.dtype != dtypes.bool: cond = cond.ne(UOp.const(cond.dtype, 0))
        i += 1
        # Snapshot vars before parsing branches - branches shouldn't see each other's assignments
        vars_snapshot = dict(vars)
        i, branch_assigns = parse_block(lines, i)
        conditions.append((cond, branch_assigns))
        # Restore vars for next branch
        vars.clear(); vars.update(vars_snapshot)
        # Handle elsif/else chains
        while i < len(lines):
          if (m := re.match(r'elsif\s+(.+?)\s+then$', lines[i], re.IGNORECASE)):
            cond = parse_expr(m.group(1), {**vars, **block_assigns})
            if cond.dtype != dtypes.bool: cond = cond.ne(UOp.const(cond.dtype, 0))
            i += 1
            i, branch_assigns = parse_block(lines, i)
            conditions.append((cond, branch_assigns))
            # Restore vars for next branch
            vars.clear(); vars.update(vars_snapshot)
          elif re.match(r'else$', lines[i], re.IGNORECASE):
            i += 1
            i, else_assigns = parse_block(lines, i)
            # Restore vars after else branch
            vars.clear(); vars.update(vars_snapshot)
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
      # MEM[addr].type = value or MEM[addr].type += value
      if (m := re.match(r'MEM\[(.+)\]\.(\w+)\s*(\+)?=\s*(.+)', line)):
        ctx = {**vars, **block_assigns}
        addr = parse_expr(m.group(1), ctx)
        rhs = parse_expr(m.group(4), ctx)
        dt = DTYPES.get(m.group(2), dtypes.uint32)
        if m.group(3) == '+':  # compound assignment: read old value, add rhs (pm_add_loads will add the load)
          mem = vars.get('_vmem') if '_vmem' in vars else vars.get('_lds')  # Use vmem for FLAT/GLOBAL, lds for DS
          if mem is not None:
            shift_amt = UOp.const(dtypes.uint64 if addr.dtype == dtypes.uint64 else dtypes.uint32, 2)
            idx = (addr >> shift_amt).cast(dtypes.index)
            old_val = mem.index(idx)
            # For 64-bit types, read both dwords and combine
            if dt in (dtypes.uint64, dtypes.int64, dtypes.float64):
              four = UOp.const(dtypes.uint64 if addr.dtype == dtypes.uint64 else dtypes.uint32, 4)
              hi_idx = ((addr + four) >> shift_amt).cast(dtypes.index)
              old_val = old_val.cast(dtypes.uint64) | (mem.index(hi_idx).cast(dtypes.uint64) << UOp.const(dtypes.uint64, 32))
            rhs = old_val + rhs
        assigns.append((f'MEM[{m.group(1)}].{m.group(2)}', (addr, rhs)))
        i += 1; continue
      # Lambda definition: NAME = lambda(args) (body) - must check FIRST
      if (m := re.match(r'(\w+)\s*=\s*lambda\(([^)]*)\)\s*\(', line)):
        lambda_name, lambda_args = m.group(1), [a.strip() for a in m.group(2).split(',')]
        # Collect multi-line lambda body until matching close paren
        body_start = line[m.end():]  # Start after opening paren (skip it)
        depth = 1
        # Count parens in body_start to handle single-line lambdas
        close_pos = -1
        for j, ch in enumerate(body_start):
          if ch == '(': depth += 1
          elif ch == ')':
            depth -= 1
            if depth == 0: close_pos = j; break
        if close_pos >= 0:
          # Lambda body is complete on this line - take content up to closing paren
          body = body_start[:close_pos].strip()
          i += 1  # Move past this line
        else:
          body_lines = [body_start] if body_start.strip() else []
          i += 1
          while i < len(lines) and depth > 0:
            # Find position where depth becomes 0
            close_pos = -1
            for j, ch in enumerate(lines[i]):
              if ch == '(': depth += 1
              elif ch == ')':
                depth -= 1
                if depth == 0: close_pos = j; break
            if close_pos >= 0:
              # Take content up to the closing paren
              body_lines.append(lines[i][:close_pos])
            else:
              body_lines.append(lines[i])
            i += 1
          body = ' '.join(body_lines).strip()
          # Don't increment i here - while loop already moved it past the lambda body
        vars[lambda_name] = ('lambda', lambda_args, body)
        continue
      # VAR[high:low] = value or VAR[high:low].type = value
      if (m := re.match(r'(\w+)\[(\d+)\s*:\s*(\d+)\](?:\.(\w+))?\s*=\s*(.+)', line)):
        var_name, high_bit, low_bit, type_suffix = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        val = parse_expr(m.group(5), {**vars, **block_assigns})
        assigns.append((f'{var_name}[{high_bit}:{low_bit}]' + (f'.{type_suffix}' if type_suffix else ''), val))
        # Also update vars to accumulate the bit slice (needed for later references like D0 = tmp)
        if var_name not in vars: vars[var_name] = UOp.const(dtypes.uint64 if high_bit >= 32 else dtypes.uint32, 0)
        old_val = block_assigns.get(var_name, vars.get(var_name))
        # Convert val to uint bits - use bitcast for floats to preserve bit pattern
        if val.dtype == dtypes.half:
          val_bits = val.bitcast(dtypes.uint16).cast(dtypes.uint32)
        elif val.dtype == dtypes.float32:
          val_bits = val.bitcast(dtypes.uint32)
        elif val.dtype == dtypes.float64:
          val_bits = val.bitcast(dtypes.uint64)
        elif val.dtype != dtypes.uint32:
          val_bits = val.cast(dtypes.uint32)
        else:
          val_bits = val
        mask = UOp.const(dtypes.uint32, ((1 << (high_bit - low_bit + 1)) - 1) << low_bit)
        new_val = (old_val & (mask ^ UOp.const(dtypes.uint32, 0xFFFFFFFF))) | (val_bits << UOp.const(dtypes.uint32, low_bit))
        block_assigns[var_name] = new_val
        vars[var_name] = new_val
        i += 1; continue
      # Compound assignment: VAR += value, VAR -= value
      if (m := re.match(r'(\w+(?:\.\w+)?)\s*\+=\s*(.+)', line)):
        var_name = m.group(1).split('.')[0]
        old_val = block_assigns.get(var_name, vars.get(var_name, UOp.const(dtypes.uint32, 0)))
        rhs = parse_expr(m.group(2), {**vars, **block_assigns})
        if rhs.dtype != old_val.dtype: rhs = rhs.cast(old_val.dtype)
        new_val = old_val + rhs
        block_assigns[var_name] = new_val
        vars[var_name] = new_val
        i += 1; continue
      if (m := re.match(r'(\w+(?:\.\w+)?)\s*-=\s*(.+)', line)):
        var_name = m.group(1).split('.')[0]
        old_val = block_assigns.get(var_name, vars.get(var_name, UOp.const(dtypes.uint32, 0)))
        rhs = parse_expr(m.group(2), {**vars, **block_assigns})
        if rhs.dtype != old_val.dtype: rhs = rhs.cast(old_val.dtype)
        new_val = old_val - rhs
        block_assigns[var_name] = new_val
        vars[var_name] = new_val
        i += 1; continue
      # Array-indexed assignment: VAR{idx} = value (from loop unrolling)
      if (m := re.match(r'(\w+)\{(\d+)\}\s*=\s*(.+)', line)):
        var_name, idx = m.group(1), int(m.group(2))
        val = parse_expr(m.group(3), {**vars, **block_assigns})
        actual_var = f'{var_name}{idx}'
        block_assigns[actual_var] = val
        vars[actual_var] = val
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

  lines = [l.strip().rstrip(';') for l in pcode.split('\n') if l.strip() and not l.strip().startswith('//')]
  _, final_assigns = parse_block(lines)

  # Build assigns from final values
  # Check which vars already have bit slice assigns
  vars_with_slices = set(dest.split('[')[0] for dest, _ in assigns if '[' in dest)
  for var, val in final_assigns.items():
    if var in ['D0', 'SCC', 'VCC', 'EXEC', 'PC', 'RETURN_DATA', 'VDATA']:
      # Skip if this var was updated via bit slices only (not a direct assignment)
      if var in vars_with_slices:
        # Check if there's also a direct assignment like "D0.b32 = tmp"
        has_direct_assign = any(re.match(rf'{var}\.\w+\s*=', line) for line in lines)
        if not has_direct_assign:
          continue  # Skip - bit slices already handle this
      # Find the full destination including type suffix and indexing from the original pcode
      for line in lines:
        if (m := re.match(rf'{var}\.(\w+(?:\[\w+\])?)', line)):
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
        # For - and +, skip if lhs ends with an operator (unary case)
        if op in ('-', '+') and lhs and lhs[-1] in '*/^|&<>=!': continue
        if lhs and rhs:
          l, r = parse_expr(lhs, vars), parse_expr(rhs, vars)
          if op_type in ('>>', '<<', '>=', '<=', '==', '!=', '>', '<') and l.dtype != r.dtype:
            # For comparisons with negative numbers, use signed comparison
            if r.dtype == dtypes.int and r.op == Ops.CONST and r.arg < 0:
              l = l.cast(dtypes.int)  # Convert left to signed for proper signed comparison
            else:
              r = r.cast(l.dtype)
          if op_type in ('|', '^', '&') and l.dtype != r.dtype:
            if l.dtype.itemsize == r.dtype.itemsize:
              target = dtypes.uint32 if l.dtype.itemsize == 4 else dtypes.uint64 if l.dtype.itemsize == 8 else l.dtype
              l, r = l.bitcast(target), r.bitcast(target)
            else: r = r.cast(l.dtype)
          return _BINOPS[op_type](l, r)
  # Type cast: 64'U(...) or 32'F(...) etc. - for F types, use bitcast to preserve bit patterns
  if (m := re.match(r"(\d+)'([UIFB])\((.+)\)", expr)):
    dt = {('U',32): dtypes.uint32, ('U',64): dtypes.uint64, ('I',32): dtypes.int, ('I',64): dtypes.int64,
          ('F',32): dtypes.float32, ('F',64): dtypes.float64}.get((m.group(2), int(m.group(1))), dtypes.uint32)
    inner = parse_expr(m.group(3), vars)
    # For float types with integer source (like 32'F(0xffc00000)), use bitcast to preserve bit patterns
    if m.group(2) == 'F' and inner.dtype in (dtypes.uint32, dtypes.uint64, dtypes.ulong, dtypes.int, dtypes.int64):
      # Ensure size matches before bitcast
      if inner.dtype.itemsize != dt.itemsize:
        inner = inner.cast(dtypes.uint32 if dt.itemsize == 4 else dtypes.uint64)
      return inner.bitcast(dt)
    return inner.cast(dt)
  # Lane-indexed: VCC.u64[laneId]
  if (m := re.match(r'([a-zA-Z_]\w*)\.u64\[laneId\](?:\.(\w+))?$', expr)):
    v = vars.get(m.group(1), UOp.const(dtypes.uint32, 0))
    lane = vars['laneId'].cast(dtypes.uint32) if vars['laneId'].dtype != dtypes.uint32 else vars['laneId']
    result = (v >> lane) & UOp.const(dtypes.uint32, 1)
    if m.group(2): result = result.cast(DTYPES.get(m.group(2), dtypes.uint32))
    return result
  # Variable with type: S0.u32 or dict access: WAVE_MODE.IEEE
  # (Skip special constants like INF.f32, NAN.f32 which are handled later)
  if (m := re.match(r'([a-zA-Z_]\w*)\.(\w+)$', expr)) and m.group(1) not in ('INF', 'UNDERFLOW', 'OVERFLOW', 'NAN'):
    v = vars.get(m.group(1), UOp.const(dtypes.uint32, 0))
    # Handle dict access (e.g., WAVE_MODE.IEEE)
    if isinstance(v, dict): return v.get(m.group(2), UOp.const(dtypes.uint32, 0))
    dt = DTYPES.get(m.group(2), dtypes.uint32)
    if dt == v.dtype: return v
    if dt.itemsize == 2 and v.dtype.itemsize == 4:
      v16 = (v & UOp.const(v.dtype, 0xFFFF)).cast(dtypes.uint16)
      return v16 if dt == dtypes.uint16 else v16.bitcast(dt)
    return v.cast(dt) if dt.itemsize != v.dtype.itemsize else v.bitcast(dt)
  # Bit slice: S0[4:0] or S0[4:0].u32 or S0[15:0].f16
  if (m := re.match(r'([a-zA-Z_]\w*)\[(\d+)\s*:\s*(\d+)\](?:\.(\w+))?$', expr)):
    var_name, hi, lo, type_suffix = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
    val = vars.get(var_name, UOp.const(dtypes.uint32, 0))
    # If extracting bits beyond value's bit width, look for indexed dword variable (e.g., VDATA2 for bits 95:64)
    val_bits = val.dtype.itemsize * 8
    if lo >= val_bits:
      dword_idx = lo // 32
      indexed_var = f'{var_name}{dword_idx}'
      if indexed_var in vars:
        val = vars[indexed_var]
        lo = lo % 32
        hi = (hi % 32) + lo  # Adjust hi relative to the dword
    shift_dt = dtypes.uint64 if val.dtype in (dtypes.uint64, dtypes.int64) else dtypes.uint32
    shifted = val >> UOp.const(shift_dt, lo) if lo > 0 else val
    result = shifted & UOp.const(shifted.dtype, (1<<(hi-lo+1))-1)
    # If type suffix specified, convert bits to that type
    if type_suffix:
      dt = DTYPES.get(type_suffix, dtypes.uint32)
      if dt == dtypes.half:
        result = result.cast(dtypes.uint16).bitcast(dtypes.half)
      elif dt != result.dtype:
        result = result.cast(dt) if dt.itemsize != result.dtype.itemsize else result.bitcast(dt)
    return result
  # Array-indexed bit slice: S{2}[15:0] or S{2}[15:0].f16 (from loop unrolling)
  if (m := re.match(r'([a-zA-Z_]\w*)\{(\d+)\}\[(\d+)\s*:\s*(\d+)\](?:\.(\w+))?$', expr)):
    var_name, idx, hi, lo, type_suffix = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), m.group(5)
    actual_var = f'{var_name}{idx}'
    val = vars.get(actual_var, UOp.const(dtypes.uint32, 0))
    shift_dt = dtypes.uint64 if val.dtype in (dtypes.uint64, dtypes.int64) else dtypes.uint32
    shifted = val >> UOp.const(shift_dt, lo) if lo > 0 else val
    result = shifted & UOp.const(shifted.dtype, (1<<(hi-lo+1))-1)
    if type_suffix:
      dt = DTYPES.get(type_suffix, dtypes.uint32)
      if dt == dtypes.half:
        result = result.cast(dtypes.uint16).bitcast(dtypes.half)
      elif dt != result.dtype:
        result = result.cast(dt) if dt.itemsize != result.dtype.itemsize else result.bitcast(dt)
    return result
  # Bit slice with type prefix: S1.u32[4:0].u32
  if (m := re.match(r'([a-zA-Z_]\w*)\.(\w+)\[(\d+)\s*:\s*(\d+)\](?:\.(\w+))?$', expr)):
    hi, lo = int(m.group(3)), int(m.group(4))
    val = vars.get(m.group(1), UOp.const(dtypes.uint32, 0))
    shift_dt = dtypes.uint64 if val.dtype in (dtypes.uint64, dtypes.int64) else dtypes.uint32
    shifted = val >> UOp.const(shift_dt, lo) if lo > 0 else val
    return shifted & UOp.const(shifted.dtype, (1<<(hi-lo+1))-1)
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
  if (m := re.match(r'(-?\d+)(ULL|LL|UL|L|U)?$', expr)):
    val = int(m.group(1))
    suffix = m.group(2) or ''
    if 'LL' in suffix: dt = dtypes.uint64 if 'U' in suffix else dtypes.int64
    elif 'L' in suffix: dt = dtypes.uint64 if 'U' in suffix else dtypes.int64  # Treat L as 64-bit too
    elif 'U' in suffix: dt = dtypes.uint32
    else: dt = dtypes.int if val < 0 else dtypes.uint32
    return UOp.const(dt, val)
  # Float literals: 2.0F or 2.0f -> float32, 2.0 (no suffix) -> float64
  if (m := re.match(r'-?(\d+\.\d+)[Ff]$', expr)): return UOp.const(dtypes.float32, float(expr.rstrip('Ff')))
  if (m := re.match(r'-?(\d+)[Ff]$', expr)): return UOp.const(dtypes.float32, float(expr.rstrip('Ff')))
  if (m := re.match(r'-?(\d+\.\d+)$', expr)): return UOp.const(dtypes.float64, float(expr))
  # Unary bitwise NOT
  if expr.startswith('~'):
    inner = parse_expr(expr[1:], vars)
    return inner ^ UOp.const(inner.dtype, (1 << (inner.dtype.itemsize * 8)) - 1)
  # Unary logical NOT: !x means x == 0
  if expr.startswith('!'):
    inner = parse_expr(expr[1:], vars)
    return inner.eq(UOp.const(inner.dtype, 0))
  # Unary minus
  if expr.startswith('-') and len(expr) > 1 and expr[1] not in '0123456789':
    return parse_expr(expr[1:], vars).neg()
  # Variable
  if expr in vars: return vars[expr]
  if expr == 'PI': return UOp.const(dtypes.float32, 3.141592653589793)
  # IEEE 754 special constants (handle both +INF and INF forms since unary minus is handled separately)
  if expr in ('+INF', 'INF'): return UOp.const(dtypes.float64, float('inf'))
  if expr == '-INF': return UOp.const(dtypes.float64, float('-inf'))
  if expr in ('+INF.f32', 'INF.f32'): return UOp.const(dtypes.float32, float('inf'))
  if expr == '-INF.f32': return UOp.const(dtypes.float32, float('-inf'))
  if expr in ('+INF.f16', 'INF.f16'): return UOp.const(dtypes.half, float('inf'))
  if expr == '-INF.f16': return UOp.const(dtypes.half, float('-inf'))
  # NaN constants
  if expr in ('NAN.f32', 'NAN'): return UOp.const(dtypes.uint32, 0x7FC00000).bitcast(dtypes.float32)  # Quiet NaN
  if expr == 'NAN.f64': return UOp.const(dtypes.uint64, 0x7FF8000000000000).bitcast(dtypes.float64)
  if expr == 'NAN.f16': return UOp.const(dtypes.uint16, 0x7E00).bitcast(dtypes.half)
  # Overflow/underflow: smallest denormal (underflow) and largest finite (overflow)
  if expr == 'UNDERFLOW_F32': return UOp.const(dtypes.uint32, 0x00000001).bitcast(dtypes.float32)  # ~1.4e-45
  if expr == 'OVERFLOW_F32': return UOp.const(dtypes.uint32, 0x7F7FFFFF).bitcast(dtypes.float32)  # ~3.4e38
  if expr == 'UNDERFLOW_F64': return UOp.const(dtypes.uint64, 0x0000000000000001).bitcast(dtypes.float64)
  if expr == 'OVERFLOW_F64': return UOp.const(dtypes.uint64, 0x7FEFFFFFFFFFFFFF).bitcast(dtypes.float64)
  # Array-indexed variable with type: S{0}.f32, in{1}.f32 (curly braces from loop unrolling)
  if (m := re.match(r'([a-zA-Z_]\w*)\{(\d+)\}\.(\w+)$', expr)):
    var_name, idx, type_suffix = m.group(1), int(m.group(2)), m.group(3)
    # Map array index to actual variable: S{0} -> S0, in{0} -> in0
    actual_var = f'{var_name}{idx}'
    v = vars.get(actual_var, UOp.const(dtypes.uint32, 0))
    dt = DTYPES.get(type_suffix, dtypes.uint32)
    if dt == v.dtype: return v
    return v.cast(dt) if dt.itemsize != v.dtype.itemsize else v.bitcast(dt)
  # Array-indexed variable: S{0}, in{1} (curly braces from loop unrolling)
  if (m := re.match(r'([a-zA-Z_]\w*)\{(\d+)\}$', expr)):
    var_name, idx = m.group(1), int(m.group(2))
    actual_var = f'{var_name}{idx}'
    return vars.get(actual_var, UOp.const(dtypes.uint32, 0))
  # Brace concatenation: { hi, lo }
  if (m := re.match(r'\{\s*(.+?)\s*,\s*(.+?)\s*\}', expr)):
    high = parse_expr(m.group(1), vars).cast(dtypes.uint64)
    low = parse_expr(m.group(2), vars).cast(dtypes.uint64)
    return (high << UOp.const(dtypes.uint64, 32)) | low
  # MEM[addr].type
  if (m := re.match(r'MEM\[(.+)\]\.(\w+)', expr)):
    addr = parse_expr(m.group(1), vars)
    dt = DTYPES.get(m.group(2), dtypes.uint32)
    mem = vars.get('_vmem') if '_vmem' in vars else vars.get('_lds')  # Use vmem for FLAT/GLOBAL, lds for DS
    if mem is None: return UOp.const(dt, 0)
    # Use appropriate shift based on address dtype (64-bit for vmem, 32-bit for lds)
    addr_dt = dtypes.uint64 if addr.dtype == dtypes.uint64 else dtypes.uint32
    shift_amt = UOp.const(addr_dt, 2)
    idx = (addr >> shift_amt).cast(dtypes.index)
    val = mem.index(idx)  # Don't call .load() - pm_add_loads will add it
    if dt in (dtypes.uint64, dtypes.int64, dtypes.float64):
      four = UOp.const(addr_dt, 4)
      hi_idx = ((addr + four) >> shift_amt).cast(dtypes.index)
      val = val.cast(dtypes.uint64) | (mem.index(hi_idx).cast(dtypes.uint64) << UOp.const(dtypes.uint64, 32))
    elif dt in (dtypes.uint8, dtypes.int8):
      # Extract byte: shift by (addr & 3) * 8, mask to 8 bits
      byte_offset = (addr & UOp.const(addr_dt, 3)).cast(dtypes.uint32)
      val = (val >> (byte_offset * UOp.const(dtypes.uint32, 8))) & UOp.const(dtypes.uint32, 0xFF)
    elif dt in (dtypes.uint16, dtypes.int16):
      # Extract halfword: shift by ((addr >> 1) & 1) * 16, mask to 16 bits
      half_offset = ((addr >> UOp.const(addr_dt, 1)) & UOp.const(addr_dt, 1)).cast(dtypes.uint32)
      val = (val >> (half_offset * UOp.const(dtypes.uint32, 16))) & UOp.const(dtypes.uint32, 0xFFFF)
    return val
  # Array element access: VAR[index] where index is a constant - maps to VAR{index} variable
  if (m := re.match(r'([a-zA-Z_]\w*)\[(\d+)\]$', expr)):
    var_name, idx = m.group(1), int(m.group(2))
    actual_var = f'{var_name}{idx}'
    if actual_var in vars: return vars[actual_var]
    # Fall through to bit slice handling if not found as array element
  # VAR[high:low] - handled above in "Bit slice" pattern
  # Lambda call: NAME(arg1, arg2, ...)
  if (m := re.match(r'(\w+)\((.+)\)$', expr)):
    func_name, args_str = m.group(1), m.group(2)
    if func_name in vars and isinstance(vars[func_name], tuple) and vars[func_name][0] == 'lambda':
      _, param_names, body = vars[func_name]
      # Parse arguments (handling nested parens)
      args = []
      depth, start = 0, 0
      for i, ch in enumerate(args_str):
        if ch == '(': depth += 1
        elif ch == ')': depth -= 1
        elif ch == ',' and depth == 0:
          args.append(args_str[start:i].strip())
          start = i + 1
      args.append(args_str[start:].strip())
      # Build new vars with lambda parameters bound to arguments
      lambda_vars = vars.copy()
      for param, arg in zip(param_names, args):
        lambda_vars[param] = parse_expr(arg, vars)
      return parse_expr(body, lambda_vars)
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

def _isnan(v: UOp) -> UOp:
  if v.dtype == dtypes.float64:
    v64 = v.bitcast(dtypes.uint64)
    exp_mask, mant_mask = UOp.const(dtypes.uint64, 0x7FF0000000000000), UOp.const(dtypes.uint64, 0x000FFFFFFFFFFFFF)
    return (v64 & exp_mask).eq(exp_mask) & (v64 & mant_mask).ne(UOp.const(dtypes.uint64, 0))
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
  for a in args[1:]:
    b = cast(a)
    if dt == dtypes.float32:
      # AMD NaN handling: v_min_f32(nan, x) = x, v_max_f32(nan, x) = x
      a_nan = _isnan(result)
      b_nan = _isnan(b)
      cmp = result.maximum(b) if is_max else result.minimum(b)
      # If a is NaN, return b; if b is NaN, return a; otherwise return cmp result
      result = a_nan.where(b, b_nan.where(result, cmp))
    else:
      result = result.maximum(b) if is_max else result.minimum(b)
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
    # Detect if value is an 8-bit or 16-bit masked value (from MEM[ADDR].i8 or MEM[ADDR].i16)
    # Look for AND with 0xFF or 0xFFFF mask
    is_8bit = val.op == Ops.AND and len(val.src) == 2 and val.src[1].op == Ops.CONST and val.src[1].arg == 0xFF
    is_16bit = val.op == Ops.AND and len(val.src) == 2 and val.src[1].op == Ops.CONST and val.src[1].arg == 0xFFFF
    if is_8bit or val.dtype in (dtypes.int8, dtypes.uint8):
      # Sign extend 8-bit to 32-bit: if bit 7 is set, OR with 0xFFFFFF00
      v32 = val.cast(dtypes.uint32) if val.dtype != dtypes.uint32 else val
      sign_bit = (v32 >> UOp.const(dtypes.uint32, 7)) & UOp.const(dtypes.uint32, 1)
      return sign_bit.ne(UOp.const(dtypes.uint32, 0)).where(
        v32 | UOp.const(dtypes.uint32, 0xFFFFFF00), v32).cast(dtypes.int)
    if is_16bit or val.dtype in (dtypes.int16, dtypes.short, dtypes.uint16):
      # Sign extend 16-bit to 32-bit: if bit 15 is set, OR with 0xFFFF0000
      v32 = val.cast(dtypes.uint32) if val.dtype != dtypes.uint32 else val
      sign_bit = (v32 >> UOp.const(dtypes.uint32, 15)) & UOp.const(dtypes.uint32, 1)
      return sign_bit.ne(UOp.const(dtypes.uint32, 0)).where(
        v32 | UOp.const(dtypes.uint32, 0xFFFF0000), v32).cast(dtypes.int)
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
  _FUNC_TABLE.append((r'f32_to_i32\((.+)\)', 1, lambda a, v, m: UOp(Ops.TRUNC, a[0].bitcast(dtypes.float32).dtype, (a[0].bitcast(dtypes.float32),)).cast(dtypes.int)))
  # f32_to_u32: clamp negative to 0 before converting (GPU behavior)
  def _f32_to_u32(a, v, m):
    f = a[0].bitcast(dtypes.float32)
    is_neg = f < UOp.const(dtypes.float32, 0.0)
    clamped = is_neg.where(UOp.const(dtypes.float32, 0.0), f)
    return UOp(Ops.TRUNC, clamped.dtype, (clamped,)).cast(dtypes.uint32)
  _FUNC_TABLE.append((r'f32_to_u32\((.+)\)', 1, _f32_to_u32))
  _FUNC_TABLE.append((r'f64_to_i32\((.+)\)', 1, lambda a, v, m: UOp(Ops.TRUNC, a[0].bitcast(dtypes.float64).dtype, (a[0].bitcast(dtypes.float64),)).cast(dtypes.int)))
  # f64_to_u32: clamp negative to 0 before converting (GPU behavior)
  def _f64_to_u32(a, v, m):
    f = a[0].bitcast(dtypes.float64)
    is_neg = f < UOp.const(dtypes.float64, 0.0)
    clamped = is_neg.where(UOp.const(dtypes.float64, 0.0), f)
    return UOp(Ops.TRUNC, clamped.dtype, (clamped,)).cast(dtypes.uint32)
  _FUNC_TABLE.append((r'f64_to_u32\((.+)\)', 1, _f64_to_u32))
  _FUNC_TABLE.append((r'f16_to_f32\((.+)\)', 1, lambda a, v, m: _f16_extract(a[0]).cast(dtypes.float32)))
  _FUNC_TABLE.append((r'f32_to_f16\((.+)\)', 1, lambda a, v, m: a[0].cast(dtypes.half)))
  _FUNC_TABLE.append((r'f16_to_i16\((.+)\)', 1, lambda a, v, m: UOp(Ops.TRUNC, _f16_extract(a[0]).dtype, (_f16_extract(a[0]),)).cast(dtypes.int16)))
  _FUNC_TABLE.append((r'f16_to_u16\((.+)\)', 1, lambda a, v, m: UOp(Ops.TRUNC, _f16_extract(a[0]).dtype, (_f16_extract(a[0]),)).cast(dtypes.uint16)))
  # Float classification
  _FUNC_TABLE.append((r'isNAN\((.+)\)', 1, lambda a, v, m: _isnan(a[0])))
  _FUNC_TABLE.append((r'isSignalNAN\((.+)\)', 0, lambda a, v, m: _check_nan(m.group(1), v, quiet=False)))
  _FUNC_TABLE.append((r'isQuietNAN\((.+)\)', 0, lambda a, v, m: _check_nan(m.group(1), v, quiet=True)))
  # cvtToQuietNAN: convert signaling NaN to quiet NaN by setting the quiet bit
  def _cvt_to_quiet_nan(a, v, m):
    val = a[0]
    if val.dtype == dtypes.float64:
      return (val.bitcast(dtypes.uint64) | UOp.const(dtypes.uint64, 0x0008000000000000)).bitcast(dtypes.float64)  # Set bit 51
    elif val.dtype == dtypes.half:
      return (val.bitcast(dtypes.uint16) | UOp.const(dtypes.uint16, 0x0200)).bitcast(dtypes.half)  # Set bit 9
    else:  # f32
      v32 = val.bitcast(dtypes.uint32) if val.dtype == dtypes.float32 else val
      return (v32 | UOp.const(dtypes.uint32, 0x00400000)).bitcast(dtypes.float32)  # Set bit 22
  _FUNC_TABLE.append((r'cvtToQuietNAN\((.+)\)', 1, _cvt_to_quiet_nan))
  def _exponent(a, v, m):
    val = a[0]
    if '.f16' in m.group(1) or val.dtype == dtypes.half:
      bits = val.bitcast(dtypes.uint16) if val.dtype == dtypes.half else (val & UOp.const(dtypes.uint32, 0xFFFF)).cast(dtypes.uint16)
      return (bits.cast(dtypes.uint32) >> UOp.const(dtypes.uint32, 10)) & UOp.const(dtypes.uint32, 0x1F)
    elif '.f64' in m.group(1) or val.dtype == dtypes.float64:
      bits = val.bitcast(dtypes.uint64) if val.dtype == dtypes.float64 else val
      return ((bits >> UOp.const(dtypes.uint64, 52)) & UOp.const(dtypes.uint64, 0x7FF)).cast(dtypes.uint32)
    else:  # f32
      bits = val.bitcast(dtypes.uint32) if val.dtype == dtypes.float32 else val
      return (bits >> UOp.const(dtypes.uint32, 23)) & UOp.const(dtypes.uint32, 0xFF)
  _FUNC_TABLE.append((r'exponent\((.+)\)', 1, _exponent))
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

  # ldexp: multiply float by 2^exp (scale by power of 2)
  def _ldexp(a, v, m):
    val, exp = a[0], a[1]
    # Convert val to float if it's uint bits
    if val.dtype == dtypes.uint32:
      val = val.bitcast(dtypes.float32)
    elif val.dtype == dtypes.uint64:
      val = val.bitcast(dtypes.float64)
    # exp should be a signed integer
    if exp.dtype in (dtypes.uint32, dtypes.uint64):
      exp = exp.cast(dtypes.int if exp.dtype == dtypes.uint32 else dtypes.int64)
    # ldexp(x, n) = x * 2^n
    # Use EXP2 to compute 2^n, then multiply
    two_pow_exp = UOp(Ops.EXP2, val.dtype, (exp.cast(val.dtype),))
    return val * two_pow_exp
  _FUNC_TABLE.append((r'ldexp\((.+),\s*(.+)\)', 2, _ldexp))

  # frexp_mant: extract mantissa (normalized to [0.5, 1.0))
  def _frexp_mant(a, v, m):
    val = a[0]
    if val.dtype == dtypes.uint32:
      val = val.bitcast(dtypes.float32)
    elif val.dtype == dtypes.uint64:
      val = val.bitcast(dtypes.float64)
    # For f32: extract bits, set exponent to 126 (bias-1), keep mantissa
    # frexp returns mantissa in [0.5, 1.0)
    if val.dtype == dtypes.float32:
      bits = val.bitcast(dtypes.uint32)
      # Clear exponent, set to 0x3f000000 (exponent = 126, mantissa unchanged -> [0.5, 1.0))
      mantissa_bits = (bits & UOp.const(dtypes.uint32, 0x807FFFFF)) | UOp.const(dtypes.uint32, 0x3f000000)
      return mantissa_bits.bitcast(dtypes.float32)
    else:  # f64
      bits = val.bitcast(dtypes.uint64)
      # Clear exponent, set to 0x3fe (exponent = 1022)
      mantissa_bits = (bits & UOp.const(dtypes.uint64, 0x800FFFFFFFFFFFFF)) | UOp.const(dtypes.uint64, 0x3fe0000000000000)
      return mantissa_bits.bitcast(dtypes.float64)
  _FUNC_TABLE.append((r'frexp_mant\((.+)\)', 1, _frexp_mant))
  _FUNC_TABLE.append((r'mantissa\((.+)\)', 1, _frexp_mant))  # Alias for pcode

  # frexp_exp: extract exponent (unbiased)
  def _frexp_exp(a, v, m):
    val = a[0]
    if val.dtype == dtypes.uint32:
      val = val.bitcast(dtypes.float32)
    elif val.dtype == dtypes.uint64:
      val = val.bitcast(dtypes.float64)
    if val.dtype == dtypes.float32:
      bits = val.bitcast(dtypes.uint32)
      exp = ((bits >> UOp.const(dtypes.uint32, 23)) & UOp.const(dtypes.uint32, 0xFF)).cast(dtypes.int)
      return exp - UOp.const(dtypes.int, 126)  # Unbias and adjust for frexp convention
    else:  # f64
      bits = val.bitcast(dtypes.uint64)
      exp = ((bits >> UOp.const(dtypes.uint64, 52)) & UOp.const(dtypes.uint64, 0x7FF)).cast(dtypes.int)
      return exp - UOp.const(dtypes.int, 1022)  # Unbias and adjust
  _FUNC_TABLE.append((r'frexp_exp\((.+)\)', 1, _frexp_exp))

_register_funcs()

def _parse_func(expr: str, vars: dict[str, UOp]) -> UOp | None:
  for pattern, nargs, handler in _FUNC_TABLE:
    if (m := re.match(pattern, expr)):
      args = [parse_expr(m.group(i+1), vars) for i in range(nargs)] if nargs > 0 else []
      return handler(args, vars, m)
  return None
