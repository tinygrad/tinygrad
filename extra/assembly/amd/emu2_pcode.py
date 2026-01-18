# AMD pcode parser - converts pseudocode to UOps
import re
from tinygrad.dtype import dtypes, DType
from tinygrad.uop.ops import Ops, UOp

DTYPES = {'u32': dtypes.uint32, 'i32': dtypes.int, 'f32': dtypes.float32, 'b32': dtypes.uint32,
          'u64': dtypes.uint64, 'i64': dtypes.int64, 'f64': dtypes.float64, 'b64': dtypes.uint64,
          'u16': dtypes.uint16, 'i16': dtypes.short, 'f16': dtypes.half, 'b16': dtypes.uint16,
          'u8': dtypes.uint8, 'i8': dtypes.int8, 'b8': dtypes.uint8,
          'u1': dtypes.uint32}  # 1-bit treated as uint32 for comparisons

def _ftz_f32(v: UOp) -> UOp:
  """Flush denormal f32 to zero (RDNA3 FTZ mode)."""
  if v.dtype != dtypes.float32: return v
  bits = v.bitcast(dtypes.uint32)
  exp = (bits >> UOp.const(dtypes.uint32, 23)) & UOp.const(dtypes.uint32, 0xFF)
  mantissa = bits & UOp.const(dtypes.uint32, 0x7FFFFF)
  is_denorm = exp.eq(UOp.const(dtypes.uint32, 0)) & mantissa.ne(UOp.const(dtypes.uint32, 0))
  sign = bits & UOp.const(dtypes.uint32, 0x80000000)
  return is_denorm.where(sign.bitcast(dtypes.float32), v)

def _isnan(v: UOp) -> UOp:
  """Check if float value is NaN."""
  if v.dtype == dtypes.float64:
    bits = v.bitcast(dtypes.uint64)
    return ((bits >> UOp.const(dtypes.uint64, 52)) & UOp.const(dtypes.uint64, 0x7FF)).eq(UOp.const(dtypes.uint64, 0x7FF)) & \
           (bits & UOp.const(dtypes.uint64, 0xFFFFFFFFFFFFF)).ne(UOp.const(dtypes.uint64, 0))
  v32 = v.cast(dtypes.float32) if v.dtype == dtypes.half else v
  bits = v32.bitcast(dtypes.uint32)
  return ((bits >> UOp.const(dtypes.uint32, 23)) & UOp.const(dtypes.uint32, 0xFF)).eq(UOp.const(dtypes.uint32, 0xFF)) & \
         (bits & UOp.const(dtypes.uint32, 0x7FFFFF)).ne(UOp.const(dtypes.uint32, 0))

def _cmp_with_nan(l: UOp, r: UOp, cmp_fn) -> UOp:
  """IEEE 754 compliant comparison - returns FALSE if either operand is NaN."""
  result = cmp_fn(l, r)
  if l.dtype not in (dtypes.float32, dtypes.float64, dtypes.half): return result
  return result & _isnan(l).logical_not() & _isnan(r).logical_not()

def _bitreverse(v: UOp, bits: int) -> UOp:
  """Reverse bits of a value."""
  dt = dtypes.uint64 if bits == 64 else dtypes.uint32
  v = v.cast(dt) if v.dtype != dt else v
  masks = [(0x5555555555555555, 1), (0x3333333333333333, 2), (0x0F0F0F0F0F0F0F0F, 4),
           (0x00FF00FF00FF00FF, 8), (0x0000FFFF0000FFFF, 16)] if bits == 64 else \
          [(0x55555555, 1), (0x33333333, 2), (0x0F0F0F0F, 4), (0x00FF00FF, 8)]
  for mask, shift in masks:
    m = UOp.const(dt, mask if bits == 64 else mask & 0xFFFFFFFF)
    v = ((v >> UOp.const(dt, shift)) & m) | ((v & m) << UOp.const(dt, shift))
  final_shift = 32 if bits == 64 else 16
  return (v >> UOp.const(dt, final_shift)) | (v << UOp.const(dt, final_shift))

def _extract_bits(val: UOp, hi: int, lo: int) -> UOp:
  """Extract bits [hi:lo] from value."""
  shift_dt = dtypes.uint64 if val.dtype in (dtypes.uint64, dtypes.int64) else dtypes.uint32
  shifted = val >> UOp.const(shift_dt, lo) if lo > 0 else val
  return shifted & UOp.const(shifted.dtype, (1 << (hi - lo + 1)) - 1)

def _try_eval(expr: str) -> str:
  """Try to evaluate a simple arithmetic expression."""
  try:
    if re.match(r'^[\d\s\+\-\*\/\(\)\&\|]+$', expr.strip()): return str(eval(expr))
  except: pass
  return expr

# Binary operators
_BINOPS = {
  '|': lambda l, r: l | r, '^': lambda l, r: l ^ r, '&': lambda l, r: l & r,
  '>=': lambda l, r: _cmp_with_nan(l, r, lambda a, b: a >= b),
  '<=': lambda l, r: _cmp_with_nan(l, r, lambda a, b: a <= b),
  '>': lambda l, r: _cmp_with_nan(l, r, lambda a, b: a > b),
  '<': lambda l, r: _cmp_with_nan(l, r, lambda a, b: a < b),
  '==': lambda l, r: l.eq(r), '!=': lambda l, r: l.ne(r),
  '>>': lambda l, r: l >> r, '<<': lambda l, r: l << r,
  '+': lambda l, r: l + (r.cast(l.dtype) if l.dtype != r.dtype else r),
  '-': lambda l, r: l - (r.cast(l.dtype) if l.dtype != r.dtype and l.dtype.itemsize == r.dtype.itemsize else r) if not (l.op == Ops.CONST and r.op == Ops.CONST) else UOp.const(l.dtype, l.arg - r.arg),
  '*': lambda l, r: l * (r.cast(l.dtype) if l.dtype != r.dtype else r),
  '/': lambda l, r: l / r,
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
    for old, new in [(f'S2.{dt} / S1.{dt} == DENORM.{dt}', f'isDENORM(S2.{dt} / S1.{dt})'),
                     (f"1.0 / 64'F(S1.{dt}) == DENORM.f64", f"isDENORM(1.0 / 64'F(S1.{dt}))"),
                     (f'1.0 / S1.{dt} == DENORM.{dt}', f'isDENORM(1.0 / S1.{dt})'),
                     (f'S1.{dt} == DENORM.{dt}', f'isDENORM(S1.{dt})'),
                     (f'D0.{dt} = NAN.{dt}', f'VCC = 0x1LL;\nD0.{dt} = NAN.{dt}'),
                     (f'elsif isDENORM(S1.{dt}) then\nD0.{dt} = ldexp(S0.{dt}, {ldexp_val})', f'elsif 1 == 0 then\nD0.{dt} = S0.{dt}'),
                     (f'elsif exponent(S2.{dt}) <= {exp_lim} then\n// Numerator is tiny\nD0.{dt} = ldexp(S0.{dt}, {ldexp_val})',
                      f'elsif exponent(S2.{dt}) <= {exp_lim} then\nVCC = 0x1LL;\nD0.{dt} = ldexp(S0.{dt}, {ldexp_val})'),
                     (f'elsif isDENORM(S2.{dt} / S1.{dt}) then\nVCC = 0x1LL;\nif S0.{dt} == S2.{dt} then\n// Only scale the numerator\nD0.{dt} = ldexp(S0.{dt}, {ldexp_val})\nendif',
                      f'elsif isDENORM(S2.{dt} / S1.{dt}) then\nVCC = 0x1LL;\nD0.{dt} = S0.{dt}'),
                     (f'D0.{dt} = ldexp(S0.{dt}, {ldexp_val})\nendif\nelsif', f'D0.{dt} = ldexp(S0.{dt}, {ldexp_val})\nelse\nD0.{dt} = S0.{dt}\nendif\nelsif')]:
      pcode = pcode.replace(old, new)
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
  vars['WAVE_MODE'] = {'IEEE': UOp.const(dtypes.uint32, 1)}
  vars['WAVE32'], vars['WAVE64'] = UOp.const(dtypes.bool, True), UOp.const(dtypes.bool, False)
  assigns: list[tuple[str, UOp]] = []

  def parse_block(lines: list[str], start: int = 0) -> tuple[int, dict[str, UOp]]:
    """Parse a block of statements."""
    block_assigns: dict[str, UOp] = {}
    i = start
    while i < len(lines):
      line = lines[i]
      if re.match(r'(elsif|else|endif|endfor)\b', line, re.IGNORECASE): break

      # For loop
      if (m := re.match(r"for\s+(\w+)\s+in\s+(?:\d+')?(\d+)U?\s*:\s*(?:\d+')?(\d+)U?\s+do", line, re.IGNORECASE)):
        loop_var, start_val, end_val = m.group(1), int(m.group(2)), int(m.group(3))
        i += 1
        body_lines, depth = [], 1
        while i < len(lines) and depth > 0:
          if re.match(r'for\s+', lines[i], re.IGNORECASE): depth += 1
          elif re.match(r'endfor\b', lines[i], re.IGNORECASE): depth -= 1
          if depth > 0: body_lines.append(lines[i])
          i += 1
        has_break = any('break' in bl.lower() for bl in body_lines)
        if has_break:
          found_var = f'_loop_found_{id(body_lines)}'
          vars[found_var] = block_assigns[found_var] = UOp.const(dtypes.bool, False)
        for loop_i in range(start_val, end_val + 1):
          subst_lines = []
          for bl in body_lines:
            if re.match(r'break\b', bl.strip(), re.IGNORECASE): continue
            subst = re.sub(rf'\.(\w+)\[{loop_var}\]', rf'.\g<1>[{loop_i}]', bl)
            subst = re.sub(rf'(?<!\.)\b(\w+)\[{loop_var}\]', rf'\g<1>{{{loop_i}}}', subst)
            subst = re.sub(rf'\b{loop_var}\b', str(loop_i), subst)
            subst = re.sub(r'\[([^\]\[]+?)\s*:\s*([^\]\[]+?)\]', lambda m: f'[{_try_eval(m.group(1))} : {_try_eval(m.group(2))}]', subst)
            subst_lines.append(subst)
          _, iter_assigns = parse_block(subst_lines, 0)
          if has_break:
            found = block_assigns.get(found_var, vars.get(found_var))
            not_found = found.eq(UOp.const(dtypes.bool, False))
            for var, val in iter_assigns.items():
              if var == found_var: continue
              old_val = block_assigns.get(var, vars.get(var, UOp.const(dtypes.uint32, 0)))
              if val.dtype != old_val.dtype and val.dtype.itemsize == old_val.dtype.itemsize: old_val = old_val.cast(val.dtype)
              block_assigns[var] = vars[var] = not_found.where(val, old_val)
            for j, bl in enumerate(body_lines):
              if (cm := re.match(r'if\s+(.+?)\s+then$', bl.strip(), re.IGNORECASE)):
                for k in range(j+1, len(body_lines)):
                  if re.match(r'break\b', body_lines[k].strip(), re.IGNORECASE):
                    cond_str = re.sub(rf'\b{loop_var}\b', str(loop_i), cm.group(1))
                    cond = parse_expr(cond_str, {**vars, **block_assigns})
                    if cond.dtype != dtypes.bool: cond = cond.ne(UOp.const(cond.dtype, 0))
                    block_assigns[found_var] = vars[found_var] = not_found.where(cond, found)
                    break
                  elif re.match(r'(endif|else|elsif)\b', body_lines[k].strip(), re.IGNORECASE): break
                break
          else:
            block_assigns.update(iter_assigns)
            vars.update(iter_assigns)
        continue

      # If/elsif/else block
      if (m := re.match(r'if\s+(.+?)\s+then$', line, re.IGNORECASE)):
        conditions: list[tuple[UOp, dict[str, UOp]]] = []
        else_assigns: dict[str, UOp] = {}
        cond = parse_expr(m.group(1), {**vars, **block_assigns})
        if cond.dtype != dtypes.bool: cond = cond.ne(UOp.const(cond.dtype, 0))
        i += 1
        vars_snapshot = dict(vars)
        i, branch_assigns = parse_block(lines, i)
        conditions.append((cond, branch_assigns))
        vars.clear(); vars.update(vars_snapshot)
        while i < len(lines):
          if (m := re.match(r'elsif\s+(.+?)\s+then$', lines[i], re.IGNORECASE)):
            cond = parse_expr(m.group(1), {**vars, **block_assigns})
            if cond.dtype != dtypes.bool: cond = cond.ne(UOp.const(cond.dtype, 0))
            i += 1
            i, branch_assigns = parse_block(lines, i)
            conditions.append((cond, branch_assigns))
            vars.clear(); vars.update(vars_snapshot)
          elif re.match(r'else$', lines[i], re.IGNORECASE):
            i += 1
            i, else_assigns = parse_block(lines, i)
            vars.clear(); vars.update(vars_snapshot)
          elif re.match(r'endif\b', lines[i], re.IGNORECASE):
            i += 1; break
          else: break
        all_vars = set()
        for _, ba in conditions: all_vars.update(ba.keys())
        all_vars.update(else_assigns.keys())
        for var in all_vars:
          result = else_assigns.get(var, block_assigns.get(var, vars.get(var, UOp.const(dtypes.uint32, 0))))
          for cond, ba in reversed(conditions):
            if var in ba:
              true_val = ba[var]
              if true_val.dtype != result.dtype and true_val.dtype.itemsize == result.dtype.itemsize: result = result.cast(true_val.dtype)
              result = cond.where(true_val, result)
          block_assigns[var] = vars[var] = result
        continue

      # MEM[addr].type = value or MEM[addr].type += value
      if (m := re.match(r'MEM\[(.+)\]\.(\w+)\s*(\+)?=\s*(.+)', line)):
        ctx = {**vars, **block_assigns}
        addr, rhs, dt = parse_expr(m.group(1), ctx), parse_expr(m.group(4), ctx), DTYPES.get(m.group(2), dtypes.uint32)
        if m.group(3) == '+':
          mem = vars.get('_vmem') if '_vmem' in vars else vars.get('_lds')
          if mem is not None:
            shift_amt = UOp.const(dtypes.uint64 if addr.dtype == dtypes.uint64 else dtypes.uint32, 2)
            idx = (addr >> shift_amt).cast(dtypes.index)
            old_val = mem.index(idx)
            if dt in (dtypes.uint64, dtypes.int64, dtypes.float64):
              four = UOp.const(addr.dtype, 4)
              old_val = old_val.cast(dtypes.uint64) | (mem.index(((addr + four) >> shift_amt).cast(dtypes.index)).cast(dtypes.uint64) << UOp.const(dtypes.uint64, 32))
            rhs = old_val + rhs
        assigns.append((f'MEM[{m.group(1)}].{m.group(2)}', (addr, rhs)))
        i += 1; continue

      # VGPR[lane][reg] = value
      if (m := re.match(r'VGPR\[([^\]]+)\]\[([^\]]+)\]\s*=\s*(.+)', line)):
        ctx = {**vars, **block_assigns}
        lane, reg, val = parse_expr(m.group(1), ctx), parse_expr(m.group(2), ctx), parse_expr(m.group(3), ctx)
        if lane.dtype != dtypes.uint32: lane = lane.cast(dtypes.uint32)
        if reg.dtype != dtypes.uint32: reg = reg.cast(dtypes.uint32)
        assigns.append((f'VGPR[{m.group(1)}][{m.group(2)}]', (reg * UOp.const(dtypes.uint32, 32) + lane, val)))
        i += 1; continue

      # Lambda definition
      if (m := re.match(r'(\w+)\s*=\s*lambda\(([^)]*)\)\s*\(', line)):
        lambda_name, lambda_args = m.group(1), [a.strip() for a in m.group(2).split(',')]
        body_start, depth = line[m.end():], 1
        close_pos = -1
        for j, ch in enumerate(body_start):
          if ch == '(': depth += 1
          elif ch == ')':
            depth -= 1
            if depth == 0: close_pos = j; break
        if close_pos >= 0:
          body = body_start[:close_pos].strip()
          i += 1
        else:
          body_lines = [body_start] if body_start.strip() else []
          i += 1
          while i < len(lines) and depth > 0:
            close_pos = -1
            for j, ch in enumerate(lines[i]):
              if ch == '(': depth += 1
              elif ch == ')':
                depth -= 1
                if depth == 0: close_pos = j; break
            body_lines.append(lines[i][:close_pos] if close_pos >= 0 else lines[i])
            i += 1
          body = '\n'.join(body_lines).strip()
        vars[lambda_name] = ('lambda', lambda_args, body)
        continue

      # VAR[high:low] = value or VAR[high:low].type = value or VAR.type[high:low] = value
      if (m := re.match(r'(\w+)(?:\.(\w+))?\[(\d+)\s*:\s*(\d+)\](?:\.(\w+))?\s*=\s*(.+)', line)):
        var_name, type_prefix, first_bit, second_bit, type_suffix = m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), m.group(5)
        val = parse_expr(m.group(6), {**vars, **block_assigns})
        high_bit, low_bit = max(first_bit, second_bit), min(first_bit, second_bit)
        type_info = type_prefix or type_suffix
        assigns.append((f'{var_name}[{high_bit}:{low_bit}]' + (f'.{type_info}' if type_info else ''), val))
        if var_name not in vars: vars[var_name] = UOp.const(dtypes.uint64 if high_bit >= 32 else dtypes.uint32, 0)
        old_val = block_assigns.get(var_name, vars.get(var_name))
        val_bits = val.bitcast(dtypes.uint16).cast(dtypes.uint32) if val.dtype == dtypes.half else \
                   val.bitcast(dtypes.uint32) if val.dtype == dtypes.float32 else \
                   val.bitcast(dtypes.uint64) if val.dtype == dtypes.float64 else \
                   val.cast(dtypes.uint32) if val.dtype != dtypes.uint32 else val
        mask = UOp.const(dtypes.uint32, ((1 << (high_bit - low_bit + 1)) - 1) << low_bit)
        new_val = (old_val & (mask ^ UOp.const(dtypes.uint32, 0xFFFFFFFF))) | (val_bits << UOp.const(dtypes.uint32, low_bit))
        block_assigns[var_name] = vars[var_name] = new_val
        i += 1; continue

      # Compound assignment: VAR += value, VAR -= value
      if (m := re.match(r'(\w+(?:\.\w+)?)\s*([+-])=\s*(.+)', line)):
        var_name = m.group(1).split('.')[0]
        old_val = block_assigns.get(var_name, vars.get(var_name, UOp.const(dtypes.uint32, 0)))
        rhs = parse_expr(m.group(3), {**vars, **block_assigns})
        if rhs.dtype != old_val.dtype: rhs = rhs.cast(old_val.dtype)
        new_val = (old_val + rhs) if m.group(2) == '+' else (old_val - rhs)
        block_assigns[var_name] = vars[var_name] = new_val
        i += 1; continue

      # Array-indexed assignment: VAR{idx} = value (from loop unrolling)
      if (m := re.match(r'(\w+)\{(\d+)\}\s*=\s*(.+)', line)):
        var_name, idx = m.group(1), int(m.group(2))
        val = parse_expr(m.group(3), {**vars, **block_assigns})
        existing_var = block_assigns.get(var_name, vars.get(var_name))
        if existing_var is not None and isinstance(existing_var, UOp):
          old_val, bit_pos = existing_var, UOp.const(dtypes.uint32, idx)
          bit_mask = UOp.const(dtypes.uint32, 1) << bit_pos
          val_bit = val.cast(dtypes.uint32) & UOp.const(dtypes.uint32, 1)
          new_val = (old_val & (bit_mask ^ UOp.const(dtypes.uint32, 0xFFFFFFFF))) | (val_bit << bit_pos)
          block_assigns[var_name] = vars[var_name] = new_val
        else:
          actual_var = f'{var_name}{idx}'
          block_assigns[actual_var] = vars[actual_var] = val
        i += 1; continue

      # Simple bit assignment: VAR[expr] = value
      if (m := re.match(r'(\w+)\[([^\]]+)\]\s*=\s*(.+)', line)):
        var_name, bit_expr, val_expr = m.group(1), m.group(2), m.group(3)
        if ':' not in bit_expr:
          existing_var = block_assigns.get(var_name, vars.get(var_name))
          is_array = any(f'{var_name}{j}' in vars or f'{var_name}{j}' in block_assigns for j in range(8))
          if existing_var is not None and isinstance(existing_var, UOp) and not is_array:
            ctx = {**vars, **block_assigns}
            bit_pos, val = parse_expr(bit_expr, ctx), parse_expr(val_expr, ctx)
            if bit_pos.dtype != dtypes.uint32: bit_pos = bit_pos.cast(dtypes.uint32)
            bit_mask = UOp.const(dtypes.uint32, 1) << bit_pos
            val_bit = val.cast(dtypes.uint32) & UOp.const(dtypes.uint32, 1)
            new_val = (existing_var & (bit_mask ^ UOp.const(dtypes.uint32, 0xFFFFFFFF))) | (val_bit << bit_pos)
            block_assigns[var_name] = vars[var_name] = new_val
            i += 1; continue

      # Typed element assignment: D.u8[0] = value
      if (m := re.match(r'(\w+)\.(\w+)\[(\d+)\]\s*=\s*(.+)', line)):
        var_name, type_str, idx, val_expr = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        dt = DTYPES.get(type_str, dtypes.uint32)
        val = parse_expr(val_expr, {**vars, **block_assigns})
        old_val = block_assigns.get(var_name, vars.get(var_name, UOp.const(dtypes.uint32, 0)))
        bit_width, low_bit = dt.itemsize * 8, idx * dt.itemsize * 8
        mask = UOp.const(dtypes.uint32, ((1 << bit_width) - 1) << low_bit)
        val_bits = ((val.cast(dtypes.uint32) if val.dtype != dtypes.uint32 else val) & UOp.const(dtypes.uint32, (1 << bit_width) - 1)) << UOp.const(dtypes.uint32, low_bit)
        new_val = (old_val & (mask ^ UOp.const(dtypes.uint32, 0xFFFFFFFF))) | val_bits
        block_assigns[var_name] = vars[var_name] = new_val
        assigns.append((f'{var_name}.{type_str}[{idx}]', val))
        i += 1; continue

      # Dynamic single-bit assignment: D0.u32[S0.u32[4:0]] = 1'1U
      if (m := re.match(r'(\w+)\.(\w+)\[(.*\[.*\].*)\]\s*=\s*(.+)', line)):
        var_name, type_suffix, bit_expr, val_expr = m.group(1), m.group(2), m.group(3), m.group(4)
        ctx = {**vars, **block_assigns}
        bit_pos, val = parse_expr(bit_expr, ctx), parse_expr(val_expr, ctx)
        old_val = block_assigns.get(var_name, vars.get(var_name, UOp.const(dtypes.uint32, 0)))
        if bit_pos.dtype != dtypes.uint32: bit_pos = bit_pos.cast(dtypes.uint32)
        bit_mask = UOp.const(dtypes.uint32, 1) << bit_pos
        if val.op == Ops.CONST and val.arg == 1: new_val = old_val | bit_mask
        elif val.op == Ops.CONST and val.arg == 0: new_val = old_val & (bit_mask ^ UOp.const(dtypes.uint32, 0xFFFFFFFF))
        else:
          val_bit = val.cast(dtypes.uint32) & UOp.const(dtypes.uint32, 1)
          new_val = (old_val & (bit_mask ^ UOp.const(dtypes.uint32, 0xFFFFFFFF))) | (val_bit << bit_pos)
        block_assigns[var_name] = vars[var_name] = new_val
        i += 1; continue

      # Compound destination: { D1.u1, D0.u64 } = value
      if (m := re.match(r'\{\s*(\w+)\.(\w+)\s*,\s*(\w+)\.(\w+)\s*\}\s*=\s*(.+)', line)):
        hi_var, hi_type, lo_var, lo_type, rhs = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        val = parse_expr(rhs, {**vars, **block_assigns})
        lo_dt, hi_dt = DTYPES.get(lo_type, dtypes.uint64), DTYPES.get(hi_type, dtypes.uint32)
        lo_bits = 64 if lo_dt in (dtypes.uint64, dtypes.int64) else 32
        lo_val = val.cast(lo_dt) if val.dtype.itemsize * 8 <= lo_bits else (val & UOp.const(val.dtype, (1 << lo_bits) - 1)).cast(lo_dt)
        hi_val = (val >> UOp.const(val.dtype, lo_bits)).cast(hi_dt)
        block_assigns[lo_var] = vars[lo_var] = lo_val
        block_assigns[hi_var] = vars[hi_var] = hi_val
        assigns.append((f'{lo_var}.{lo_type}', lo_val))
        assigns.append((f'{hi_var}.{hi_type}', hi_val))
        i += 1; continue

      # Regular assignment: VAR = value
      if (m := re.match(r'(\w+(?:\.\w+)?(?:\[\w+\])?)\s*=\s*(.+)', line)) and not re.search(r'[<>=!]=', line[:line.find('=')]):
        lhs, val = m.group(1), parse_expr(m.group(2), {**vars, **block_assigns})
        base = re.match(r'(\w+)', lhs).group(1)
        block_assigns[base] = vars[base] = val
        i += 1; continue

      # Declaration: declare VAR : TYPE
      if (m := re.match(r'declare\s+(\w+)', line)):
        vars[m.group(1)] = UOp.const(dtypes.uint32, 0)
        i += 1; continue
      i += 1
    return i, block_assigns

  lines = [l.strip().rstrip(';') for l in pcode.split('\n') if l.strip() and not l.strip().startswith('//')]
  _, final_assigns = parse_block(lines)

  vars_with_slices = set(dest.split('[')[0] for dest, _ in assigns if '[' in dest)
  for var, val in final_assigns.items():
    if var in ['D0', 'SCC', 'VCC', 'EXEC', 'PC', 'RETURN_DATA', 'VDATA']:
      if var in vars_with_slices:
        has_direct_assign = any(re.match(rf'{var}\.\w+\s*=', line) for line in lines)
        if not has_direct_assign: continue
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

  # Ternary
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

  # Binary ops (low to high precedence)
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
        if op in ('-', '+') and lhs and lhs[-1] in '*/^|&<>=!': continue
        if lhs and rhs:
          l, r = parse_expr(lhs, vars), parse_expr(rhs, vars)
          if op_type in ('>>', '<<', '>=', '<=', '==', '!=', '>', '<') and l.dtype != r.dtype:
            if r.dtype == dtypes.int and r.op == Ops.CONST and r.arg < 0: l = l.cast(dtypes.int)
            else: r = r.cast(l.dtype)
          if op_type in ('|', '^', '&') and l.dtype != r.dtype:
            if l.dtype.itemsize == r.dtype.itemsize:
              target = dtypes.uint32 if l.dtype.itemsize == 4 else dtypes.uint64 if l.dtype.itemsize == 8 else l.dtype
              l, r = l.bitcast(target), r.bitcast(target)
            else: r = r.cast(l.dtype)
          return _BINOPS[op_type](l, r)

  # Type cast: 64'U(...) or 32'F(...)
  if (m := re.match(r"(\d+)'([UIFB])\((.+)\)", expr)):
    bits = int(m.group(1))
    dt = {('U',32): dtypes.uint32, ('U',64): dtypes.uint64, ('I',32): dtypes.int, ('I',64): dtypes.int64,
          ('F',32): dtypes.float32, ('F',64): dtypes.float64}.get((m.group(2), bits), dtypes.uint64 if bits > 32 else dtypes.uint32)
    inner = parse_expr(m.group(3), vars)
    if m.group(2) == 'F' and inner.dtype in (dtypes.uint32, dtypes.uint64, dtypes.ulong, dtypes.int, dtypes.int64):
      if inner.dtype.itemsize != dt.itemsize: inner = inner.cast(dtypes.uint32 if dt.itemsize == 4 else dtypes.uint64)
      result = inner.bitcast(dt)
      return _ftz_f32(result) if dt == dtypes.float32 else result
    return inner.cast(dt)

  # Lane-indexed: VCC.u64[laneId]
  if (m := re.match(r'([a-zA-Z_]\w*)\.u64\[laneId\](?:\.(\w+))?$', expr)):
    v = vars.get(m.group(1), UOp.const(dtypes.uint32, 0))
    lane = vars['laneId'].cast(dtypes.uint32) if vars['laneId'].dtype != dtypes.uint32 else vars['laneId']
    result = (v >> lane) & UOp.const(dtypes.uint32, 1)
    return result.cast(DTYPES.get(m.group(2), dtypes.uint32)) if m.group(2) else result

  # Variable with type: S0.u32 or dict access: WAVE_MODE.IEEE
  if (m := re.match(r'([a-zA-Z_]\w*)\.(\w+)$', expr)) and m.group(1) not in ('INF', 'UNDERFLOW', 'OVERFLOW', 'NAN'):
    v = vars.get(m.group(1), UOp.const(dtypes.uint32, 0))
    if isinstance(v, dict): return v.get(m.group(2), UOp.const(dtypes.uint32, 0))
    dt = DTYPES.get(m.group(2), dtypes.uint32)
    if dt == v.dtype: return _ftz_f32(v) if dt == dtypes.float32 else v
    if dt.itemsize == 2 and v.dtype.itemsize == 4:
      v16 = (v & UOp.const(v.dtype, 0xFFFF)).cast(dtypes.uint16)
      return v16 if dt == dtypes.uint16 else v16.bitcast(dt)
    result = v.cast(dt) if dt.itemsize != v.dtype.itemsize else v.bitcast(dt)
    return _ftz_f32(result) if dt == dtypes.float32 else result

  # Bit slice: S0[4:0] or S0[4:0].u32 or S0[0:31] (reversed for bit reverse)
  if (m := re.match(r'([a-zA-Z_]\w*)\[(\d+)\s*:\s*(\d+)\](?:\.(\w+))?$', expr)):
    var_name, first, second, type_suffix = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
    val = vars.get(var_name, UOp.const(dtypes.uint32, 0))
    if first < second:  # Bit reverse
      result = _bitreverse(val, second - first + 1)
      if type_suffix:
        dt = DTYPES.get(type_suffix, dtypes.uint32)
        if dt != result.dtype: result = result.cast(dt) if dt.itemsize != result.dtype.itemsize else result.bitcast(dt)
      return result
    hi, lo = first, second
    val_bits = val.dtype.itemsize * 8
    if lo >= val_bits:
      dword_idx = lo // 32
      indexed_var = f'{var_name}{dword_idx}'
      if indexed_var in vars:
        val, lo, hi = vars[indexed_var], lo % 32, (hi % 32) + (lo % 32)
    result = _extract_bits(val, hi, lo)
    if type_suffix:
      dt = DTYPES.get(type_suffix, dtypes.uint32)
      if dt == dtypes.half: result = result.cast(dtypes.uint16).bitcast(dtypes.half)
      elif dt != result.dtype: result = result.cast(dt) if dt.itemsize != result.dtype.itemsize else result.bitcast(dt)
    return result

  # Array-indexed bit slice: S{2}[15:0] or S{2}[15:0].f16
  if (m := re.match(r'([a-zA-Z_]\w*)\{(\d+)\}\[(\d+)\s*:\s*(\d+)\](?:\.(\w+))?$', expr)):
    var_name, idx, hi, lo, type_suffix = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), m.group(5)
    val = vars.get(f'{var_name}{idx}', UOp.const(dtypes.uint32, 0))
    result = _extract_bits(val, hi, lo)
    if type_suffix:
      dt = DTYPES.get(type_suffix, dtypes.uint32)
      if dt == dtypes.half: result = result.cast(dtypes.uint16).bitcast(dtypes.half)
      elif dt != result.dtype: result = result.cast(dt) if dt.itemsize != result.dtype.itemsize else result.bitcast(dt)
    return result

  # Verilog-style: VAR.type[start +: width]
  if (m := re.match(r"([a-zA-Z_]\w*)\.(\w+)\[(.+?)\s*\+:\s*(?:\d+')?(\d+)U?\]", expr)):
    val = vars.get(m.group(1), UOp.const(dtypes.uint32, 0))
    start = parse_expr(m.group(3), vars)
    if start.dtype != dtypes.uint32: start = start.cast(dtypes.uint32)
    return (val >> start) & UOp.const(val.dtype, (1 << int(m.group(4))) - 1)

  # Bit slice with type prefix: S1.u32[4:0] or S0.u32[0:31]
  if (m := re.match(r'([a-zA-Z_]\w*)\.(\w+)\[(\d+)\s*:\s*(\d+)\](?:\.(\w+))?$', expr)):
    first, second = int(m.group(3)), int(m.group(4))
    val = vars.get(m.group(1), UOp.const(dtypes.uint32, 0))
    if first < second: return _bitreverse(val, second - first + 1)
    return _extract_bits(val, first, second)

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
  if (m := re.match(r"(\d+)'[dD](\d+)", expr)):
    dt = {8: dtypes.uint8, 16: dtypes.uint16, 32: dtypes.uint32, 64: dtypes.uint64}.get(int(m.group(1)), dtypes.uint32)
    return UOp.const(dt, int(m.group(2)))
  if (m := re.match(r"(\d+)'[hH]([0-9a-fA-F]+)", expr)):
    dt = {8: dtypes.uint8, 16: dtypes.uint16, 32: dtypes.uint32, 64: dtypes.uint64}.get(int(m.group(1)), dtypes.uint32)
    return UOp.const(dt, int(m.group(2), 16))
  if (m := re.match(r"(\d+)'[bB]([01]+)", expr)):
    dt = {8: dtypes.uint8, 16: dtypes.uint16, 32: dtypes.uint32, 64: dtypes.uint64}.get(int(m.group(1)), dtypes.uint32)
    return UOp.const(dt, int(m.group(2), 2))
  if (m := re.match(r"(\d+)'0x([0-9a-fA-F]+)", expr)):
    dt = {8: dtypes.uint8, 16: dtypes.uint16, 32: dtypes.uint32, 64: dtypes.uint64}.get(int(m.group(1)), dtypes.uint32)
    return UOp.const(dt, int(m.group(2), 16))
  if (m := re.match(r"(\d+)'(\d+)U?", expr)):
    bits, val = int(m.group(1)), int(m.group(2))
    dt = {1: dtypes.uint32, 8: dtypes.uint8, 16: dtypes.int16 if 'U' not in expr else dtypes.uint16,
          32: dtypes.int if 'U' not in expr else dtypes.uint32, 64: dtypes.int64 if 'U' not in expr else dtypes.uint64}.get(bits, dtypes.uint32)
    return UOp.const(dt, val)
  if (m := re.match(r'(-?\d+)(ULL|LL|UL|L|U)?$', expr)):
    val, suffix = int(m.group(1)), m.group(2) or ''
    dt = dtypes.uint64 if 'LL' in suffix or 'L' in suffix else dtypes.uint32 if 'U' in suffix else dtypes.int if val < 0 else dtypes.uint32
    if 'U' in suffix and ('LL' in suffix or 'L' in suffix): dt = dtypes.uint64
    return UOp.const(dt, val)
  if (m := re.match(r'-?(\d+\.\d+)[Ff]$', expr)): return UOp.const(dtypes.float32, float(expr.rstrip('Ff')))
  if (m := re.match(r'-?(\d+)[Ff]$', expr)): return UOp.const(dtypes.float32, float(expr.rstrip('Ff')))
  if (m := re.match(r'-?(\d+\.\d+)$', expr)): return UOp.const(dtypes.float64, float(expr))

  # Unary operators
  if expr.startswith('~'):
    inner = parse_expr(expr[1:], vars)
    return inner ^ UOp.const(inner.dtype, (1 << (inner.dtype.itemsize * 8)) - 1)
  if expr.startswith('!'):
    inner = parse_expr(expr[1:], vars)
    return inner.eq(UOp.const(inner.dtype, 0))
  if expr.startswith('-') and len(expr) > 1 and expr[1] not in '0123456789':
    return parse_expr(expr[1:], vars).neg()

  # Variable
  if expr in vars: return vars[expr]
  if expr == 'PI': return UOp.const(dtypes.float32, 3.141592653589793)

  # IEEE 754 special constants
  for pattern, val in [('+INF', float('inf')), ('INF', float('inf')), ('-INF', float('-inf'))]:
    if expr == pattern: return UOp.const(dtypes.float64, val)
    if expr == f'{pattern}.f32': return UOp.const(dtypes.float32, val)
    if expr == f'{pattern}.f16': return UOp.const(dtypes.half, val)
  if expr in ('NAN.f32', 'NAN'): return UOp.const(dtypes.uint32, 0x7FC00000).bitcast(dtypes.float32)
  if expr == 'NAN.f64': return UOp.const(dtypes.uint64, 0x7FF8000000000000).bitcast(dtypes.float64)
  if expr == 'NAN.f16': return UOp.const(dtypes.uint16, 0x7E00).bitcast(dtypes.half)
  if expr == 'UNDERFLOW_F32': return UOp.const(dtypes.uint32, 0x00000001).bitcast(dtypes.float32)
  if expr == 'OVERFLOW_F32': return UOp.const(dtypes.uint32, 0x7F7FFFFF).bitcast(dtypes.float32)
  if expr == 'UNDERFLOW_F64': return UOp.const(dtypes.uint64, 0x0000000000000001).bitcast(dtypes.float64)
  if expr == 'OVERFLOW_F64': return UOp.const(dtypes.uint64, 0x7FEFFFFFFFFFFFFF).bitcast(dtypes.float64)

  # Array-indexed variable with type: S{0}.f32
  if (m := re.match(r'([a-zA-Z_]\w*)\{(\d+)\}\.(\w+)$', expr)):
    v = vars.get(f'{m.group(1)}{m.group(2)}', UOp.const(dtypes.uint32, 0))
    dt = DTYPES.get(m.group(3), dtypes.uint32)
    return v if dt == v.dtype else v.cast(dt) if dt.itemsize != v.dtype.itemsize else v.bitcast(dt)

  # Array-indexed variable: S{0}
  if (m := re.match(r'([a-zA-Z_]\w*)\{(\d+)\}$', expr)):
    return vars.get(f'{m.group(1)}{m.group(2)}', UOp.const(dtypes.uint32, 0))

  # Brace concatenation: { hi, lo }
  if (m := re.match(r'\{\s*(.+?)\s*,\s*(.+?)\s*\}', expr)):
    high = parse_expr(m.group(1), vars).cast(dtypes.uint64)
    low = parse_expr(m.group(2), vars).cast(dtypes.uint64)
    return (high << UOp.const(dtypes.uint64, 32)) | low

  # VGPR[lane][reg]
  if (m := re.match(r'VGPR\[([^\]]+)\]\[([^\]]+)\]', expr)):
    lane, reg = parse_expr(m.group(1), vars), parse_expr(m.group(2), vars)
    if lane.dtype != dtypes.uint32: lane = lane.cast(dtypes.uint32)
    if reg.dtype != dtypes.uint32: reg = reg.cast(dtypes.uint32)
    vgpr = vars.get('_vgpr')
    if vgpr is None: return UOp.const(dtypes.uint32, 0)
    return vgpr.index((reg * UOp.const(dtypes.uint32, 32) + lane).cast(dtypes.index))

  # MEM[addr].type
  if (m := re.match(r'MEM\[(.+)\]\.(\w+)', expr)):
    addr, dt = parse_expr(m.group(1), vars), DTYPES.get(m.group(2), dtypes.uint32)
    mem = vars.get('_vmem') if '_vmem' in vars else vars.get('_lds')
    if mem is None: return UOp.const(dt, 0)
    addr_dt = dtypes.uint64 if addr.dtype == dtypes.uint64 else dtypes.uint32
    shift_amt = UOp.const(addr_dt, 2)
    idx = (addr >> shift_amt).cast(dtypes.index)
    val = mem.index(idx)
    if dt in (dtypes.uint64, dtypes.int64, dtypes.float64):
      val = val.cast(dtypes.uint64) | (mem.index(((addr + UOp.const(addr_dt, 4)) >> shift_amt).cast(dtypes.index)).cast(dtypes.uint64) << UOp.const(dtypes.uint64, 32))
    elif dt in (dtypes.uint8, dtypes.int8):
      byte_offset = (addr & UOp.const(addr_dt, 3)).cast(dtypes.uint32)
      val = (val >> (byte_offset * UOp.const(dtypes.uint32, 8))) & UOp.const(dtypes.uint32, 0xFF)
    elif dt in (dtypes.uint16, dtypes.int16):
      half_offset = ((addr >> UOp.const(addr_dt, 1)) & UOp.const(addr_dt, 1)).cast(dtypes.uint32)
      val = (val >> (half_offset * UOp.const(dtypes.uint32, 16))) & UOp.const(dtypes.uint32, 0xFFFF)
    return val

  # Array element access with type: VAR[index].type
  if (m := re.match(r'([a-zA-Z_]\w*)\[(\d+)\]\.(\w+)$', expr)):
    var_name, idx, type_suffix = m.group(1), int(m.group(2)), m.group(3)
    dt = DTYPES.get(type_suffix, dtypes.uint32)
    actual_var = f'{var_name}{idx}'
    if actual_var in vars:
      v = vars[actual_var]
      return v.cast(dt) if dt.itemsize != v.dtype.itemsize else v.bitcast(dt)
    if var_name in vars:
      v = vars[var_name]
      if v.dtype != dtypes.uint32: v = v.bitcast(dtypes.uint32) if v.dtype.itemsize == 4 else v.cast(dtypes.uint32)
      return ((v >> UOp.const(dtypes.uint32, idx)) & UOp.const(dtypes.uint32, 1)).cast(dt) if dt.itemsize != 4 else \
             ((v >> UOp.const(dtypes.uint32, idx)) & UOp.const(dtypes.uint32, 1)).bitcast(dt)

  # Array element access: VAR[index]
  if (m := re.match(r'([a-zA-Z_]\w*)\[(\d+)\]$', expr)):
    var_name, idx = m.group(1), int(m.group(2))
    actual_var = f'{var_name}{idx}'
    if actual_var in vars: return vars[actual_var]
    if var_name in vars:
      v = vars[var_name]
      if v.dtype != dtypes.uint32: v = v.bitcast(dtypes.uint32) if v.dtype.itemsize == 4 else v.cast(dtypes.uint32)
      return (v >> UOp.const(dtypes.uint32, idx)) & UOp.const(dtypes.uint32, 1)

  # Lambda call
  if (m := re.match(r'(\w+)\((.+)\)$', expr)):
    func_name, args_str = m.group(1), m.group(2)
    if func_name in vars and isinstance(vars[func_name], tuple) and vars[func_name][0] == 'lambda':
      _, param_names, body = vars[func_name]
      args, depth, start = [], 0, 0
      for i, ch in enumerate(args_str):
        if ch in '({': depth += 1
        elif ch in ')}': depth -= 1
        elif ch == ',' and depth == 0:
          args.append(args_str[start:i].strip())
          start = i + 1
      args.append(args_str[start:].strip())
      lambda_vars = vars.copy()
      for param, arg in zip(param_names, args): lambda_vars[param] = parse_expr(arg, vars)
      if ';' in body or '\n' in body or 'return' in body.lower():
        return _parse_lambda_body(body, lambda_vars)
      return parse_expr(body, lambda_vars)

  # Dynamic array element access
  if (m := re.match(r'([a-zA-Z_]\w*)\[([^\]]+)\]$', expr)):
    var_name, idx_expr = m.group(1), m.group(2)
    if ':' not in idx_expr:
      idx = parse_expr(idx_expr, vars)
      if idx.dtype != dtypes.uint32: idx = idx.cast(dtypes.uint32)
      elements = [(i, vars[f'{var_name}{i}']) for i in range(256) if f'{var_name}{i}' in vars]
      if elements:
        result = elements[-1][1]
        for elem_idx, elem_val in reversed(elements[:-1]):
          cond = idx.eq(UOp.const(dtypes.uint32, elem_idx))
          if elem_val.dtype != result.dtype:
            if elem_val.dtype.itemsize == result.dtype.itemsize: result = result.cast(elem_val.dtype)
            else: elem_val = elem_val.cast(result.dtype)
          result = cond.where(elem_val, result)
        return result

  # General suffix indexing
  if (m := re.match(r'(.+)\[(\d+)\]$', expr)):
    prefix, idx = m.group(1), int(m.group(2))
    if not re.match(r'^[a-zA-Z_]\w*$', prefix):
      base = parse_expr(prefix, vars)
      if base.dtype != dtypes.uint32: base = base.bitcast(dtypes.uint32) if base.dtype.itemsize == 4 else base.cast(dtypes.uint32)
      return (base >> UOp.const(dtypes.uint32, idx)) & UOp.const(dtypes.uint32, 1)

  if (m := re.match(r'(.+)\[(\d+)\]\.(\w+)$', expr)):
    prefix, idx, type_suffix = m.group(1), int(m.group(2)), m.group(3)
    dt = DTYPES.get(type_suffix, dtypes.uint32)
    if not re.match(r'^[a-zA-Z_]\w*$', prefix):
      base = parse_expr(prefix, vars)
      if base.dtype != dtypes.uint32: base = base.bitcast(dtypes.uint32) if base.dtype.itemsize == 4 else base.cast(dtypes.uint32)
      bit_val = (base >> UOp.const(dtypes.uint32, idx)) & UOp.const(dtypes.uint32, 1)
      return bit_val.cast(dt) if dt.itemsize != 4 else bit_val.bitcast(dt)

  # Function call
  if (result := _parse_func(expr, vars)) is not None: return result
  return UOp.const(dtypes.uint32, 0)

def _parse_lambda_body(body: str, lambda_vars: dict[str, UOp]) -> UOp:
  """Parse a multi-line lambda body with return statements."""
  body = body.replace(';', '\n')
  lines = [l.strip() for l in body.split('\n') if l.strip() and not l.strip().startswith('//')]
  return _parse_lambda_block(lines, 0, lambda_vars)[1]

def _parse_lambda_block(lines: list[str], start: int, vars: dict[str, UOp]) -> tuple[int, UOp]:
  """Parse a block in a lambda, returns (next_line_idx, return_value)."""
  i = start
  while i < len(lines):
    line = lines[i]
    if re.match(r'(elsif|else|endif|endfor)\b', line, re.IGNORECASE): break
    if (m := re.match(r'return\s+(.+)$', line, re.IGNORECASE)):
      return i + 1, parse_expr(m.group(1), vars)
    if (m := re.match(r'for\s+(\w+)\s+in\s+(\d+)\s*:\s*(\d+)\s+do', line, re.IGNORECASE)):
      loop_var, start_val, end_val = m.group(1), int(m.group(2)), int(m.group(3))
      i += 1
      body_lines, depth = [], 1
      while i < len(lines) and depth > 0:
        if re.match(r'for\s+', lines[i], re.IGNORECASE): depth += 1
        elif re.match(r'endfor\b', lines[i], re.IGNORECASE): depth -= 1
        if depth > 0: body_lines.append(lines[i])
        i += 1
      for loop_i in range(start_val, end_val + 1):
        for bl in body_lines:
          subst = re.sub(rf'\b{loop_var}\b', str(loop_i), bl)
          subst = re.sub(r'\[([^\]\[]+?)\s*:\s*([^\]\[]+?)\]', lambda m: f'[{_try_eval(m.group(1))} : {_try_eval(m.group(2))}]', subst)
          if (am := re.match(r'(\w+)\[(\d+)\]\s*=\s*(.+)', subst)):
            vars[f'{am.group(1)}{am.group(2)}'] = parse_expr(am.group(3), vars)
      continue
    if re.match(r'declare\s+', line, re.IGNORECASE):
      i += 1; continue
    if (m := re.match(r'if\s+(.+?)\s+then$', line, re.IGNORECASE)):
      conditions: list[tuple[UOp, UOp | None]] = []
      cond = parse_expr(m.group(1), vars)
      if cond.dtype != dtypes.bool: cond = cond.ne(UOp.const(cond.dtype, 0))
      i += 1
      i, ret_val = _parse_lambda_block(lines, i, vars)
      conditions.append((cond, ret_val))
      while i < len(lines):
        if (m := re.match(r'elsif\s+(.+?)\s+then$', lines[i], re.IGNORECASE)):
          cond = parse_expr(m.group(1), vars)
          if cond.dtype != dtypes.bool: cond = cond.ne(UOp.const(cond.dtype, 0))
          i += 1
          i, ret_val = _parse_lambda_block(lines, i, vars)
          conditions.append((cond, ret_val))
        elif re.match(r'else$', lines[i], re.IGNORECASE):
          i += 1
          i, else_ret = _parse_lambda_block(lines, i, vars)
          result = else_ret
          for cond, ret_val in reversed(conditions):
            if ret_val is not None:
              if ret_val.dtype != result.dtype and ret_val.dtype.itemsize == result.dtype.itemsize: result = result.cast(ret_val.dtype)
              result = cond.where(ret_val, result)
          return i, result
        elif re.match(r'endif\b', lines[i], re.IGNORECASE):
          i += 1; break
        else: break
      continue
    i += 1
  return i, UOp.const(dtypes.uint32, 0)

# Function implementations
def _floor(x: UOp) -> UOp:
  truncated = UOp(Ops.TRUNC, x.dtype, (x,))
  needs_adjust = (x < UOp.const(x.dtype, 0)) & x.ne(truncated)
  return needs_adjust.where(truncated - UOp.const(x.dtype, 1), truncated)

def _f16_extract(v: UOp) -> UOp:
  return (v & UOp.const(dtypes.uint32, 0xFFFF)).cast(dtypes.uint16).bitcast(dtypes.half) if v.dtype == dtypes.uint32 else v

def _check_nan(inner: str, vars: dict, quiet: bool) -> UOp:
  if (m := re.match(r"64'F\((.+)\)", inner)): inner = m.group(1)
  v = parse_expr(inner, vars)
  is_f16, is_f64 = '.f16' in inner or v.dtype == dtypes.half, '.f64' in inner or v.dtype == dtypes.float64
  if is_f16:
    v16 = (v & UOp.const(dtypes.uint32, 0xFFFF)) if v.dtype == dtypes.uint32 else v.bitcast(dtypes.uint16).cast(dtypes.uint32)
    exp_mask, mant_mask, quiet_bit = UOp.const(dtypes.uint32, 0x7C00), UOp.const(dtypes.uint32, 0x03FF), UOp.const(dtypes.uint32, 0x0200)
  elif is_f64:
    v16 = v.bitcast(dtypes.uint64) if v.dtype == dtypes.float64 else v.cast(dtypes.uint64)
    exp_mask, mant_mask, quiet_bit = UOp.const(dtypes.uint64, 0x7FF0000000000000), UOp.const(dtypes.uint64, 0x000FFFFFFFFFFFFF), UOp.const(dtypes.uint64, 0x0008000000000000)
  else:
    v16 = v.bitcast(dtypes.uint32) if v.dtype == dtypes.float32 else v
    exp_mask, mant_mask, quiet_bit = UOp.const(dtypes.uint32, 0x7F800000), UOp.const(dtypes.uint32, 0x007FFFFF), UOp.const(dtypes.uint32, 0x00400000)
  is_nan_exp = (v16 & exp_mask).eq(exp_mask)
  has_mant = (v16 & mant_mask).ne(UOp.const(v16.dtype, 0))
  is_quiet = (v16 & quiet_bit).ne(UOp.const(v16.dtype, 0))
  return (is_nan_exp & is_quiet) if quiet else (is_nan_exp & has_mant & is_quiet.logical_not())

def _minmax_reduce(is_max: bool, dt: DType, args: list[UOp]) -> UOp:
  def cast(v): return v.bitcast(dt) if dt == dtypes.float32 and v.dtype == dtypes.uint32 else v.cast(dt)
  result = cast(args[0])
  for a in args[1:]:
    b = cast(a)
    if dt == dtypes.float32:
      a_nan, b_nan = _isnan(result), _isnan(b)
      cmp = result.maximum(b) if is_max else result.minimum(b)
      result = a_nan.where(b, b_nan.where(result, cmp))
    else:
      result = result.maximum(b) if is_max else result.minimum(b)
  return result

# Function table
_FUNC_TABLE: list[tuple[str, int, callable]] = []

def _register_funcs():
  global _FUNC_TABLE

  def _find_two_pi_mul(x: UOp) -> tuple[UOp, float] | None:
    if x.op != Ops.MUL or len(x.src) != 2: return None
    for i, src in enumerate(x.src):
      if src.op == Ops.CONST and abs(src.arg - 6.283185307179586) < 1e-5:
        return (x.src[1 - i], 6.283185307179586)
      if src.op == Ops.MUL and len(src.src) == 2:
        vals = [s.arg for s in src.src if s.op == Ops.CONST] + [s.src[0].arg for s in src.src if s.op == Ops.CAST and s.src[0].op == Ops.CONST]
        if len(vals) == 2 and abs(vals[0] * vals[1] - 6.283185307179586) < 1e-5:
          return (x.src[1 - i], vals[0] * vals[1])
    return None

  def _trig_range_reduce(x: UOp, phase_offset: float = 0.0) -> UOp:
    match = _find_two_pi_mul(x)
    if match is not None:
      turns, two_pi_val = match
      if phase_offset != 0: turns = turns + UOp.const(turns.dtype, phase_offset)
      n = _floor(turns + UOp.const(turns.dtype, 0.5))
      return UOp(Ops.SIN, turns.dtype, ((turns - n) * UOp.const(turns.dtype, two_pi_val),))
    if phase_offset != 0: x = x + UOp.const(x.dtype, phase_offset * 6.283185307179586)
    two_pi, inv_two_pi = UOp.const(x.dtype, 6.283185307179586), UOp.const(x.dtype, 0.15915494309189535)
    n = _floor(x * inv_two_pi + UOp.const(x.dtype, 0.5))
    return UOp(Ops.SIN, x.dtype, (x - n * two_pi,))

  # Unary math
  for name, op in [('sqrt', Ops.SQRT), ('trunc', Ops.TRUNC), ('log2', Ops.LOG2)]:
    _FUNC_TABLE.append((rf'{name}\((.+)\)', 1, lambda a, v, m, op=op: UOp(op, a[0].dtype, (a[0],))))
  _FUNC_TABLE.append((r'sin\((.+)\)', 1, lambda a, v, m: _trig_range_reduce(a[0])))
  _FUNC_TABLE.append((r'cos\((.+)\)', 1, lambda a, v, m: _trig_range_reduce(a[0], 0.25)))
  _FUNC_TABLE.append((r'floor\((.+)\)', 1, lambda a, v, m: _floor(a[0])))
  _FUNC_TABLE.append((r'fract\((.+)\)', 1, lambda a, v, m: a[0] - _floor(a[0])))

  def _signext(a, v, m):
    val = a[0]
    is_8bit = val.op == Ops.AND and len(val.src) == 2 and val.src[1].op == Ops.CONST and val.src[1].arg == 0xFF
    is_16bit = val.op == Ops.AND and len(val.src) == 2 and val.src[1].op == Ops.CONST and val.src[1].arg == 0xFFFF
    if is_8bit or val.dtype in (dtypes.int8, dtypes.uint8):
      v32 = val.cast(dtypes.uint32) if val.dtype != dtypes.uint32 else val
      sign_bit = (v32 >> UOp.const(dtypes.uint32, 7)) & UOp.const(dtypes.uint32, 1)
      return sign_bit.ne(UOp.const(dtypes.uint32, 0)).where(v32 | UOp.const(dtypes.uint32, 0xFFFFFF00), v32).cast(dtypes.int)
    if is_16bit or val.dtype in (dtypes.int16, dtypes.short, dtypes.uint16):
      v32 = val.cast(dtypes.uint32) if val.dtype != dtypes.uint32 else val
      sign_bit = (v32 >> UOp.const(dtypes.uint32, 15)) & UOp.const(dtypes.uint32, 1)
      return sign_bit.ne(UOp.const(dtypes.uint32, 0)).where(v32 | UOp.const(dtypes.uint32, 0xFFFF0000), v32).cast(dtypes.int)
    return val.cast(dtypes.int64) if val.dtype in (dtypes.int, dtypes.int32) else val
  _FUNC_TABLE.append((r'signext\((.+)\)', 1, _signext))

  _FUNC_TABLE.append((r'isEven\((.+)\)', 1, lambda a, v, m: (UOp(Ops.TRUNC, a[0].dtype, (a[0],)).cast(dtypes.int) & UOp.const(dtypes.int, 1)).eq(UOp.const(dtypes.int, 0))))
  _FUNC_TABLE.append((r'abs\((.+)\)', 1, lambda a, v, m:
    (a[0].bitcast(dtypes.uint32) & UOp.const(dtypes.uint32, 0x7FFFFFFF)).bitcast(dtypes.float32) if a[0].dtype == dtypes.float32 else
    (a[0].cast(dtypes.uint64) & UOp.const(dtypes.uint64, 0x7FFFFFFFFFFFFFFF)).bitcast(dtypes.float64) if a[0].dtype == dtypes.float64 else
    (a[0].bitcast(dtypes.uint16) & UOp.const(dtypes.uint16, 0x7FFF)).bitcast(dtypes.half) if a[0].dtype == dtypes.half else a[0]))

  # Binary/ternary math
  _FUNC_TABLE.append((r'max\((.+),\s*(.+)\)', 2, lambda a, v, m: UOp(Ops.MAX, a[0].dtype, (a[0], a[1]))))
  _FUNC_TABLE.append((r'min\((.+),\s*(.+)\)', 2, lambda a, v, m: UOp(Ops.MAX, a[0].dtype, (a[0].neg(), a[1].neg())).neg()))
  _FUNC_TABLE.append((r'pow\((.+),\s*(.+)\)', 2, lambda a, v, m: UOp(Ops.EXP2, dtypes.float32, (a[1].bitcast(dtypes.float32),)) if '2.0' in m.group(1) else a[0]))
  _FUNC_TABLE.append((r'fma\((.+),\s*(.+),\s*(.+)\)', 3, lambda a, v, m: a[0] * a[1] + a[2]))

  # Type conversions
  for src, dst in [('i32', dtypes.float32), ('u32', dtypes.float32)]:
    _FUNC_TABLE.append((rf'{src}_to_f32\((.+)\)', 1, lambda a, v, m, d=dst: a[0].cast(dtypes.int if 'i32' in m.group(0) else dtypes.uint32).cast(d)))
  _FUNC_TABLE.append((r'f32_to_i32\((.+)\)', 1, lambda a, v, m: UOp(Ops.TRUNC, a[0].bitcast(dtypes.float32).dtype, (a[0].bitcast(dtypes.float32),)).cast(dtypes.int)))

  def _f_to_u(f, dt):
    is_neg = f < UOp.const(f.dtype, 0.0)
    return UOp(Ops.TRUNC, f.dtype, (is_neg.where(UOp.const(f.dtype, 0.0), f),)).cast(dt)
  _FUNC_TABLE.append((r'f32_to_u32\((.+)\)', 1, lambda a, v, m: _f_to_u(a[0].bitcast(dtypes.float32), dtypes.uint32)))
  _FUNC_TABLE.append((r'f64_to_i32\((.+)\)', 1, lambda a, v, m: UOp(Ops.TRUNC, a[0].bitcast(dtypes.float64).dtype, (a[0].bitcast(dtypes.float64),)).cast(dtypes.int)))
  _FUNC_TABLE.append((r'f64_to_u32\((.+)\)', 1, lambda a, v, m: _f_to_u(a[0].bitcast(dtypes.float64), dtypes.uint32)))
  _FUNC_TABLE.append((r'f16_to_f32\((.+)\)', 1, lambda a, v, m: _f16_extract(a[0]).cast(dtypes.float32)))
  _FUNC_TABLE.append((r'f32_to_f16\((.+)\)', 1, lambda a, v, m: a[0].cast(dtypes.half)))
  _FUNC_TABLE.append((r'f16_to_i16\((.+)\)', 1, lambda a, v, m: UOp(Ops.TRUNC, _f16_extract(a[0]).dtype, (_f16_extract(a[0]),)).cast(dtypes.int16)))
  _FUNC_TABLE.append((r'f16_to_u16\((.+)\)', 1, lambda a, v, m: UOp(Ops.TRUNC, _f16_extract(a[0]).dtype, (_f16_extract(a[0]),)).cast(dtypes.uint16)))

  def _bf16_to_f32(a, v, m):
    bits = (a[0].cast(dtypes.uint32) if a[0].dtype != dtypes.uint32 else a[0]) & UOp.const(dtypes.uint32, 0xFFFF)
    return (bits << UOp.const(dtypes.uint32, 16)).bitcast(dtypes.float32)
  _FUNC_TABLE.append((r'bf16_to_f32\((.+)\)', 1, _bf16_to_f32))

  # Float classification
  _FUNC_TABLE.append((r'isNAN\((.+)\)', 1, lambda a, v, m: _isnan(a[0])))
  _FUNC_TABLE.append((r'isSignalNAN\((.+)\)', 0, lambda a, v, m: _check_nan(m.group(1), v, quiet=False)))
  _FUNC_TABLE.append((r'isQuietNAN\((.+)\)', 0, lambda a, v, m: _check_nan(m.group(1), v, quiet=True)))

  def _cvt_to_quiet_nan(a, v, m):
    val = a[0]
    if val.dtype == dtypes.float64: return (val.bitcast(dtypes.uint64) | UOp.const(dtypes.uint64, 0x0008000000000000)).bitcast(dtypes.float64)
    elif val.dtype == dtypes.half: return (val.bitcast(dtypes.uint16) | UOp.const(dtypes.uint16, 0x0200)).bitcast(dtypes.half)
    return ((val.bitcast(dtypes.uint32) if val.dtype == dtypes.float32 else val) | UOp.const(dtypes.uint32, 0x00400000)).bitcast(dtypes.float32)
  _FUNC_TABLE.append((r'cvtToQuietNAN\((.+)\)', 1, _cvt_to_quiet_nan))

  def _is_denorm(a, v, m):
    val = a[0]
    if val.dtype == dtypes.float64 or '.f64' in m.group(1):
      bits = val.bitcast(dtypes.uint64) if val.dtype == dtypes.float64 else val.cast(dtypes.uint64)
      return (bits & UOp.const(dtypes.uint64, 0x7FF0000000000000)).eq(UOp.const(dtypes.uint64, 0)) & \
             (bits & UOp.const(dtypes.uint64, 0x000FFFFFFFFFFFFF)).ne(UOp.const(dtypes.uint64, 0))
    elif val.dtype == dtypes.half or '.f16' in m.group(1):
      bits = (val.bitcast(dtypes.uint16) if val.dtype == dtypes.half else (val & UOp.const(dtypes.uint32, 0xFFFF)).cast(dtypes.uint16)).cast(dtypes.uint32)
      return (bits & UOp.const(dtypes.uint32, 0x7C00)).eq(UOp.const(dtypes.uint32, 0)) & \
             (bits & UOp.const(dtypes.uint32, 0x03FF)).ne(UOp.const(dtypes.uint32, 0))
    bits = val.bitcast(dtypes.uint32) if val.dtype == dtypes.float32 else val
    return (bits & UOp.const(dtypes.uint32, 0x7F800000)).eq(UOp.const(dtypes.uint32, 0)) & \
           (bits & UOp.const(dtypes.uint32, 0x007FFFFF)).ne(UOp.const(dtypes.uint32, 0))
  _FUNC_TABLE.append((r'isDENORM\((.+)\)', 1, _is_denorm))

  def _exponent(a, v, m):
    val = a[0]
    if '.f16' in m.group(1) or val.dtype == dtypes.half:
      bits = val.bitcast(dtypes.uint16) if val.dtype == dtypes.half else (val & UOp.const(dtypes.uint32, 0xFFFF)).cast(dtypes.uint16)
      return ((bits.cast(dtypes.uint32) >> UOp.const(dtypes.uint32, 10)) & UOp.const(dtypes.uint32, 0x1F)).cast(dtypes.int)
    elif '.f64' in m.group(1) or val.dtype == dtypes.float64:
      bits = val.bitcast(dtypes.uint64) if val.dtype == dtypes.float64 else val
      return ((bits >> UOp.const(dtypes.uint64, 52)) & UOp.const(dtypes.uint64, 0x7FF)).cast(dtypes.int)
    bits = val.bitcast(dtypes.uint32) if val.dtype == dtypes.float32 else val
    return ((bits >> UOp.const(dtypes.uint32, 23)) & UOp.const(dtypes.uint32, 0xFF)).cast(dtypes.int)
  _FUNC_TABLE.append((r'exponent\((.+)\)', 1, _exponent))

  _FUNC_TABLE.append((r'sign\((.+)\)', 1, lambda a, v, m:
    ((a[0].bitcast(dtypes.uint16) if a[0].dtype == dtypes.half else (a[0] & UOp.const(dtypes.uint32, 0xFFFF)).cast(dtypes.uint16)).cast(dtypes.uint32) >> UOp.const(dtypes.uint32, 15)) & UOp.const(dtypes.uint32, 1) if '.f16' in m.group(1) or a[0].dtype == dtypes.half else
    ((a[0].bitcast(dtypes.uint32) if a[0].dtype == dtypes.float32 else a[0]) >> UOp.const(dtypes.uint32, 31)) & UOp.const(dtypes.uint32, 1)))

  def _signext_from_bit(a, v, m):
    val, width = a[0].cast(dtypes.uint32), a[1].cast(dtypes.uint32)
    sign_bit = (val >> (width - UOp.const(dtypes.uint32, 1))) & UOp.const(dtypes.uint32, 1)
    mask = (UOp.const(dtypes.uint32, 1) << width) - UOp.const(dtypes.uint32, 1)
    return sign_bit.ne(UOp.const(dtypes.uint32, 0)).where(val | (mask ^ UOp.const(dtypes.uint32, 0xFFFFFFFF)), val)
  _FUNC_TABLE.append((r'signext_from_bit\((.+),\s*(.+)\)', 2, _signext_from_bit))

  # AMD v_min/v_max
  for is_max, name in [(False, 'min'), (True, 'max')]:
    for dt, suffix in [(dtypes.float32, 'f32'), (dtypes.int, 'i32'), (dtypes.uint32, 'u32')]:
      _FUNC_TABLE.append((rf'v_{name}_{suffix}\((.+),\s*(.+)\)', 2, lambda a, v, m, is_max=is_max, dt=dt: _minmax_reduce(is_max, dt, a)))
      _FUNC_TABLE.append((rf'v_{name}3_{suffix}\((.+),\s*(.+),\s*(.+)\)', 3, lambda a, v, m, is_max=is_max, dt=dt: _minmax_reduce(is_max, dt, a)))

  def _ldexp(a, v, m):
    val, exp = a[0], a[1]
    if val.dtype == dtypes.uint32: val = val.bitcast(dtypes.float32)
    elif val.dtype == dtypes.uint64: val = val.bitcast(dtypes.float64)
    if exp.dtype in (dtypes.uint32, dtypes.uint64): exp = exp.cast(dtypes.int if exp.dtype == dtypes.uint32 else dtypes.int64)
    return val * UOp(Ops.EXP2, val.dtype, (exp.cast(val.dtype),))
  _FUNC_TABLE.append((r'ldexp\((.+),\s*(.+)\)', 2, _ldexp))

  def _frexp_mant(a, v, m):
    val = a[0]
    if val.dtype == dtypes.uint32: val = val.bitcast(dtypes.float32)
    elif val.dtype == dtypes.uint64: val = val.bitcast(dtypes.float64)
    if val.dtype == dtypes.float32:
      bits = val.bitcast(dtypes.uint32)
      return ((bits & UOp.const(dtypes.uint32, 0x807FFFFF)) | UOp.const(dtypes.uint32, 0x3f000000)).bitcast(dtypes.float32)
    bits = val.bitcast(dtypes.uint64)
    return ((bits & UOp.const(dtypes.uint64, 0x800FFFFFFFFFFFFF)) | UOp.const(dtypes.uint64, 0x3fe0000000000000)).bitcast(dtypes.float64)
  _FUNC_TABLE.append((r'frexp_mant\((.+)\)', 1, _frexp_mant))
  _FUNC_TABLE.append((r'mantissa\((.+)\)', 1, _frexp_mant))

  def _frexp_exp(a, v, m):
    val = a[0]
    if val.dtype == dtypes.uint32: val = val.bitcast(dtypes.float32)
    elif val.dtype == dtypes.uint64: val = val.bitcast(dtypes.float64)
    if val.dtype == dtypes.float32:
      exp = ((val.bitcast(dtypes.uint32) >> UOp.const(dtypes.uint32, 23)) & UOp.const(dtypes.uint32, 0xFF)).cast(dtypes.int)
      return exp - UOp.const(dtypes.int, 126)
    exp = ((val.bitcast(dtypes.uint64) >> UOp.const(dtypes.uint64, 52)) & UOp.const(dtypes.uint64, 0x7FF)).cast(dtypes.int)
    return exp - UOp.const(dtypes.int, 1022)
  _FUNC_TABLE.append((r'frexp_exp\((.+)\)', 1, _frexp_exp))

  # Trig preop
  TWO_OVER_PI_1201 = 0x0145f306dc9c882a53f84eafa3ea69bb81b6c52b3278872083fca2c757bd778ac36e48dc74849ba5c00c925dd413a32439fc3bd63962534e7dd1046bea5d768909d338e04d68befc827323ac7306a673e93908bf177bf250763ff12fffbc0b301fde5e2316b414da3eda6cfd9e4f96136e9e8c7ecd3cbfd45aea4f758fd7cbe2f67a0e73ef14a525d4d7f6bf623f1aba10ac06608df8f6
  _TRIG_PREOP_TABLE = {shift: float(((TWO_OVER_PI_1201 << shift) >> (1201 - 53)) & 0x1fffffffffffff) for shift in range(1149)}
  def _trig_preop_result(a, v, m):
    shift = a[0]
    if shift.op == Ops.CONST:
      shift_val = int(shift.arg)
      return UOp.const(dtypes.float64, _TRIG_PREOP_TABLE.get(shift_val, float(((TWO_OVER_PI_1201 << shift_val) >> (1201 - 53)) & 0x1fffffffffffff)))
    result = UOp.const(dtypes.float64, _TRIG_PREOP_TABLE[0])
    for shift_val in range(1148, -1, -1):
      result = shift.eq(UOp.const(shift.dtype, shift_val)).where(UOp.const(dtypes.float64, _TRIG_PREOP_TABLE[shift_val]), result)
    return result
  _FUNC_TABLE.append((r'trig_preop_result\((.+)\)', 1, _trig_preop_result))

  # Find first set bit
  def _ff1_b32(a, v, m):
    val = a[0].cast(dtypes.uint32) if a[0].dtype != dtypes.uint32 else a[0]
    result = UOp.const(dtypes.int, -1)
    for i in range(32):
      bit_set = (val >> UOp.const(dtypes.uint32, i)) & UOp.const(dtypes.uint32, 1)
      cond = bit_set.ne(UOp.const(dtypes.uint32, 0)) & result.eq(UOp.const(dtypes.int, -1))
      result = cond.where(UOp.const(dtypes.int, i), result)
    return result
  _FUNC_TABLE.append((r's_ff1_i32_b32\((.+)\)', 1, _ff1_b32))

  def _ff1_b64(a, v, m):
    val = a[0].cast(dtypes.uint64) if a[0].dtype != dtypes.uint64 else a[0]
    result = UOp.const(dtypes.int, -1)
    for i in range(64):
      bit_set = (val >> UOp.const(dtypes.uint64, i)) & UOp.const(dtypes.uint64, 1)
      cond = bit_set.ne(UOp.const(dtypes.uint64, 0)) & result.eq(UOp.const(dtypes.int, -1))
      result = cond.where(UOp.const(dtypes.int, i), result)
    return result
  _FUNC_TABLE.append((r's_ff1_i32_b64\((.+)\)', 1, _ff1_b64))

_register_funcs()

def _parse_func(expr: str, vars: dict[str, UOp]) -> UOp | None:
  for pattern, nargs, handler in _FUNC_TABLE:
    if (m := re.match(pattern, expr)):
      args = [parse_expr(m.group(i+1), vars) for i in range(nargs)] if nargs > 0 else []
      return handler(args, vars, m)
  return None
