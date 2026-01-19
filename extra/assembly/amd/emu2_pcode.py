# AMD pcode parser - converts pseudocode to UOps
import re
from tinygrad.dtype import dtypes, DType
from tinygrad.uop.ops import Ops, UOp

DTYPES = {'u32': dtypes.uint32, 'i32': dtypes.int, 'f32': dtypes.float32, 'b32': dtypes.uint32, 'u64': dtypes.uint64, 'i64': dtypes.int64,
          'f64': dtypes.float64, 'b64': dtypes.uint64, 'u16': dtypes.uint16, 'i16': dtypes.short, 'f16': dtypes.half, 'b16': dtypes.uint16,
          'u8': dtypes.uint8, 'i8': dtypes.int8, 'b8': dtypes.uint8, 'u1': dtypes.uint32}
_BITS_DT = {8: dtypes.uint8, 16: dtypes.uint16, 32: dtypes.uint32, 64: dtypes.uint64}

def _const(dt, v): return UOp.const(dt, v)
def _u32(v): return _const(dtypes.uint32, v)
def _u64(v): return _const(dtypes.uint64, v)
def _to_u32(v): return v if v.dtype == dtypes.uint32 else v.bitcast(dtypes.uint32) if v.dtype.itemsize == 4 else v.cast(dtypes.uint32)
def _to_bool(v): return v if v.dtype == dtypes.bool else v.ne(_const(v.dtype, 0))
def _cast_to(v, dt): return v if v.dtype == dt else v.cast(dt) if dt.itemsize != v.dtype.itemsize else v.bitcast(dt)
def _try_eval(expr):
  try: return str(eval(expr)) if re.match(r'^[\d\s\+\-\*\/\(\)\&\|]+$', expr.strip()) else expr
  except: return expr

# Float bit extraction - returns (bits, exp_mask, mant_mask, quiet_bit, exp_shift) based on float type
def _float_info(v: UOp, hint: str = "") -> tuple[UOp, UOp, UOp, UOp, int]:
  is_f64, is_f16 = v.dtype in (dtypes.float64, dtypes.uint64) or '.f64' in hint, v.dtype == dtypes.half or '.f16' in hint
  if is_f64:
    bits = v.bitcast(dtypes.uint64) if v.dtype == dtypes.float64 else v.cast(dtypes.uint64)
    return bits, _u64(0x7FF0000000000000), _u64(0x000FFFFFFFFFFFFF), _u64(0x0008000000000000), 52
  if is_f16:
    bits = (v.bitcast(dtypes.uint16) if v.dtype == dtypes.half else (v & _u32(0xFFFF)).cast(dtypes.uint16)).cast(dtypes.uint32)
    return bits, _u32(0x7C00), _u32(0x03FF), _u32(0x0200), 10
  bits = v.bitcast(dtypes.uint32) if v.dtype == dtypes.float32 else v.cast(dtypes.uint32)
  return bits, _u32(0x7F800000), _u32(0x007FFFFF), _u32(0x00400000), 23

def _isnan(v: UOp) -> UOp:
  bits, exp_m, mant_m, _, _ = _float_info(v.cast(dtypes.float32) if v.dtype == dtypes.half else v)
  return (bits & exp_m).eq(exp_m) & (bits & mant_m).ne(_const(bits.dtype, 0))

def _cmp_nan(l, r, fn):
  result = fn(l, r)
  return result & _isnan(l).logical_not() & _isnan(r).logical_not() if l.dtype in (dtypes.float32, dtypes.float64, dtypes.half) else result

def _bitreverse(v: UOp, bits: int) -> UOp:
  dt, masks = (dtypes.uint64, [(0x5555555555555555,1),(0x3333333333333333,2),(0x0F0F0F0F0F0F0F0F,4),(0x00FF00FF00FF00FF,8),(0x0000FFFF0000FFFF,16)]) \
    if bits == 64 else (dtypes.uint32, [(0x55555555,1),(0x33333333,2),(0x0F0F0F0F,4),(0x00FF00FF,8)])
  v = v.cast(dt) if v.dtype != dt else v
  for m, s in masks: v = ((v >> _const(dt, s)) & _const(dt, m)) | ((v & _const(dt, m)) << _const(dt, s))
  return (v >> _const(dt, 32 if bits == 64 else 16)) | (v << _const(dt, 32 if bits == 64 else 16))

def _extract_bits(val: UOp, hi: int, lo: int) -> UOp:
  dt = dtypes.uint64 if val.dtype in (dtypes.uint64, dtypes.int64) else dtypes.uint32
  return ((val >> _const(dt, lo)) if lo > 0 else val) & _const(val.dtype, (1 << (hi - lo + 1)) - 1)

def _set_bit(old, pos, val):
  mask = _u32(1) << pos
  return (old & (mask ^ _u32(0xFFFFFFFF))) | ((val.cast(dtypes.uint32) & _u32(1)) << pos)

def _val_to_bits(val):
  if val.dtype == dtypes.half: return val.bitcast(dtypes.uint16).cast(dtypes.uint32)
  if val.dtype == dtypes.float32: return val.bitcast(dtypes.uint32)
  if val.dtype == dtypes.float64: return val.bitcast(dtypes.uint64)
  return val if val.dtype == dtypes.uint32 else val.cast(dtypes.uint32)

_BINOPS = {
  '|': lambda l, r: l | r, '^': lambda l, r: l ^ r, '&': lambda l, r: l & r,
  '>=': lambda l, r: _cmp_nan(l, r, lambda a, b: a >= b), '<=': lambda l, r: _cmp_nan(l, r, lambda a, b: a <= b),
  '>': lambda l, r: _cmp_nan(l, r, lambda a, b: a > b), '<': lambda l, r: _cmp_nan(l, r, lambda a, b: a < b),
  '==': lambda l, r: l.eq(r), '!=': lambda l, r: l.ne(r), '>>': lambda l, r: l >> r, '<<': lambda l, r: l << r,
  '+': lambda l, r: l + (r.cast(l.dtype) if l.dtype != r.dtype else r),
  '-': lambda l, r: _const(l.dtype, l.arg - r.arg) if l.op == Ops.CONST and r.op == Ops.CONST else l - (r.cast(l.dtype) if l.dtype != r.dtype and l.dtype.itemsize == r.dtype.itemsize else r),
  '*': lambda l, r: l * (r.cast(l.dtype) if l.dtype != r.dtype else r), '/': lambda l, r: l / r,
  '**': lambda l, r: UOp(Ops.EXP2, l.dtype, (r.cast(l.dtype),)) if l.op == Ops.CONST and l.arg == 2.0 else l,
}

def _apply_pseudocode_fixes(op_name: str, pcode: str) -> str:
  fixes = {
    'V_DIV_FMAS_F32': ('D0.f32 = 2.0F ** 32 * fma(S0.f32, S1.f32, S2.f32)',
      'D0.f32 = (exponent(S2.f32) > 127) ? (2.0F ** 64 * fma(S0.f32, S1.f32, S2.f32)) : (2.0F ** -64 * fma(S0.f32, S1.f32, S2.f32))'),
    'V_DIV_FMAS_F64': ('D0.f64 = 2.0 ** 64 * fma(S0.f64, S1.f64, S2.f64)',
      'D0.f64 = (exponent(S2.f64) > 1023) ? (2.0 ** 128 * fma(S0.f64, S1.f64, S2.f64)) : (2.0 ** -128 * fma(S0.f64, S1.f64, S2.f64))'),
    'V_DIV_FIXUP_F32': ('D0.f32 = sign_out ? -abs(S0.f32) : abs(S0.f32)',
      'D0.f32 = isNAN(S0.f32) ? (sign_out ? -INF.f32 : +INF.f32) : (sign_out ? -abs(S0.f32) : abs(S0.f32))'),
    'V_DIV_FIXUP_F64': ('D0.f64 = sign_out ? -abs(S0.f64) : abs(S0.f64)',
      'D0.f64 = isNAN(S0.f64) ? (sign_out ? -INF : +INF) : (sign_out ? -abs(S0.f64) : abs(S0.f64))'),
    'V_TRIG_PREOP_F64': ("result = 64'F((1201'B(2.0 / PI)[1200 : 0] << shift.u32) & 1201'0x1fffffffffffff)", "result = trig_preop_result(shift)"),
  }
  if op_name in fixes: pcode = pcode.replace(fixes[op_name][0], fixes[op_name][1])
  if 'V_DIV_SCALE' in op_name:
    dt, exp_lim, ldexp_val = ('f32', '23', '64') if 'F32' in op_name else ('f64', '52', '128')
    # Use exponent-based denorm prediction to avoid FTZ issues: divWouldBeDenorm(a,b) checks if a/b would underflow
    for old, new in [(f'S2.{dt} / S1.{dt} == DENORM.{dt}', f'divWouldBeDenorm(S2.{dt}, S1.{dt})'), (f"1.0 / 64'F(S1.{dt}) == DENORM.f64", '0'),
                     (f'1.0 / S1.{dt} == DENORM.{dt}', '0'), (f'S1.{dt} == DENORM.{dt}', f'isDENORM(S1.{dt})'),
                     (f'D0.{dt} = NAN.{dt}', f'VCC = 0x1LL;\nD0.{dt} = NAN.{dt}'),
                     (f'elsif isDENORM(S1.{dt}) then\nD0.{dt} = ldexp(S0.{dt}, {ldexp_val})', f'elsif 1 == 0 then\nD0.{dt} = S0.{dt}'),
                     (f'elsif exponent(S2.{dt}) <= {exp_lim} then\n// Numerator is tiny\nD0.{dt} = ldexp(S0.{dt}, {ldexp_val})',
                      f'elsif exponent(S2.{dt}) <= {exp_lim} then\nVCC = 0x1LL;\nD0.{dt} = ldexp(S0.{dt}, {ldexp_val})'),
                     (f'elsif divWouldBeDenorm(S2.{dt}, S1.{dt}) then\nVCC = 0x1LL;\nif S0.{dt} == S2.{dt} then\n// Only scale the numerator\nD0.{dt} = ldexp(S0.{dt}, {ldexp_val})\nendif',
                      f'elsif divWouldBeDenorm(S2.{dt}, S1.{dt}) then\nVCC = 0x1LL;\nD0.{dt} = S0.{dt}'),
                     (f'D0.{dt} = ldexp(S0.{dt}, {ldexp_val})\nendif\nelsif', f'D0.{dt} = ldexp(S0.{dt}, {ldexp_val})\nelse\nD0.{dt} = S0.{dt}\nendif\nelsif')]:
      pcode = pcode.replace(old, new)
    lines = pcode.rstrip().split('\n')
    for i in range(len(lines) - 1, -1, -1):
      if lines[i].strip() == 'endif': lines.insert(i, f'else\nD0.{dt} = S0.{dt}'); break
    pcode = '\n'.join(lines) + f';\nif isDENORM(S1.{dt}) then\nD0.{dt} = NAN.{dt}\nendif'
  return pcode

def parse_pcode(pcode: str, srcs: dict[str, UOp] | None = None, lane: UOp | None = None, op_name: str | None = None) -> tuple[dict[str, UOp], list[tuple[str, UOp]]]:
  if op_name: pcode = _apply_pseudocode_fixes(op_name, pcode)
  vars: dict[str, UOp] = {n: UOp(Ops.DEFINE_VAR, dtypes.uint32, (), (n, _u32(0), _u32(0xFFFFFFFF))) for n in ['S0', 'S1', 'S2', 'D0', 'VCC', 'EXEC', 'SCC', 'SIMM32']}
  if srcs: vars.update(srcs)
  vars.update({'laneId': lane if lane is not None else _u32(0), 'WAVE_MODE': {'IEEE': _u32(1)}, 'WAVE32': _const(dtypes.bool, True), 'WAVE64': _const(dtypes.bool, False)})
  assigns: list[tuple[str, UOp]] = []

  def parse_block(lines: list[str], start: int = 0) -> tuple[int, dict[str, UOp]]:
    block_assigns: dict[str, UOp] = {}
    i = start
    while i < len(lines):
      line = lines[i]
      if re.match(r'(elsif|else|endif|endfor)\b', line, re.IGNORECASE): break
      ctx = {**vars, **block_assigns}

      # For loop
      if (m := re.match(r"for\s+(\w+)\s+in\s+(?:\d+')?(\d+)U?\s*:\s*(?:\d+')?(\d+)U?\s+do", line, re.IGNORECASE)):
        loop_var, start_val, end_val = m.group(1), int(m.group(2)), int(m.group(3))
        i += 1
        body_lines, depth = [], 1
        while i < len(lines) and depth > 0:
          depth += 1 if re.match(r'for\s+', lines[i], re.IGNORECASE) else -1 if re.match(r'endfor\b', lines[i], re.IGNORECASE) else 0
          if depth > 0: body_lines.append(lines[i])
          i += 1
        has_break = any('break' in bl.lower() for bl in body_lines)
        found_var = f'_found_{id(body_lines)}' if has_break else None
        if found_var: vars[found_var] = block_assigns[found_var] = _const(dtypes.bool, False)
        for loop_i in range(start_val, end_val + 1):
          # Substitute loop variable, but avoid transforming Verilog +: or -: slice syntax
          subst_lines = [re.sub(r'\[([^\]\[]+?)(?<![+-])\s*:\s*([^\]\[]+?)\]', lambda m: f'[{_try_eval(m.group(1))} : {_try_eval(m.group(2))}]',
                         re.sub(rf'\b{loop_var}\b', str(loop_i), re.sub(rf'(?<!\.)\b(\w+)\[{loop_var}\]', rf'\g<1>{{{loop_i}}}',
                         re.sub(rf'\.(\w+)\[{loop_var}\]', rf'.\g<1>[{loop_i}]', bl)))) for bl in body_lines if not re.match(r'break\b', bl.strip(), re.IGNORECASE)]
          _, iter_assigns = parse_block(subst_lines, 0)
          if has_break:
            found = block_assigns.get(found_var, vars.get(found_var))
            not_found = found.eq(_const(dtypes.bool, False))
            for var, val in iter_assigns.items():
              if var != found_var:
                old = block_assigns.get(var, vars.get(var, _u32(0)))
                block_assigns[var] = vars[var] = not_found.where(val, old.cast(val.dtype) if val.dtype != old.dtype and val.dtype.itemsize == old.dtype.itemsize else old)
            for j, bl in enumerate(body_lines):
              if (cm := re.match(r'if\s+(.+?)\s+then$', bl.strip(), re.IGNORECASE)):
                if any(re.match(r'break\b', body_lines[k].strip(), re.IGNORECASE) for k in range(j+1, len(body_lines))):
                  cond = _to_bool(parse_expr(re.sub(rf'\b{loop_var}\b', str(loop_i), cm.group(1)), {**vars, **block_assigns}))
                  block_assigns[found_var] = vars[found_var] = not_found.where(cond, found)
                  break
          else: block_assigns.update(iter_assigns); vars.update(iter_assigns)
        continue

      # If/elsif/else - optimize away branches with constant conditions (e.g., WAVE32/WAVE64)
      if (m := re.match(r'if\s+(.+?)\s+then$', line, re.IGNORECASE)):
        conditions, else_assigns, vars_snap = [], {}, dict(vars)
        cond = _to_bool(parse_expr(m.group(1), ctx))
        cond_const = cond.arg if cond.op == Ops.CONST else None  # Check if condition is constant
        i += 1
        i, branch = parse_block(lines, i)
        if cond_const is not False:  # Skip branch if statically False
          conditions.append((cond, branch, cond_const))
        vars.clear(); vars.update(vars_snap)
        while i < len(lines):
          if (m := re.match(r'elsif\s+(.+?)\s+then$', lines[i], re.IGNORECASE)):
            cond = _to_bool(parse_expr(m.group(1), {**vars, **block_assigns}))
            cond_const = cond.arg if cond.op == Ops.CONST else None
            i += 1; i, branch = parse_block(lines, i)
            if cond_const is not False: conditions.append((cond, branch, cond_const))
            vars.clear(); vars.update(vars_snap)
          elif re.match(r'else$', lines[i], re.IGNORECASE): i += 1; i, else_assigns = parse_block(lines, i); vars.clear(); vars.update(vars_snap)
          elif re.match(r'endif\b', lines[i], re.IGNORECASE): i += 1; break
          else: break
        # Check if any condition is statically True - use that branch directly
        static_true_idx = next((j for j, (_, _, cc) in enumerate(conditions) if cc is True), None)
        if static_true_idx is not None:
          _, branch, _ = conditions[static_true_idx]
          block_assigns.update(branch); vars.update(branch)
        elif not conditions:
          # All conditions were statically False - use else branch
          block_assigns.update(else_assigns); vars.update(else_assigns)
        else:
          # Dynamic conditions - merge with WHERE
          all_vars = set().union(*[ba.keys() for _, ba, _ in conditions], else_assigns.keys())
          for var in all_vars:
            result = else_assigns.get(var, block_assigns.get(var, vars.get(var, _u32(0))))
            for cond, ba, _ in reversed(conditions):
              if var in ba:
                tv = ba[var]
                result = cond.where(tv, result.cast(tv.dtype) if tv.dtype != result.dtype and tv.dtype.itemsize == result.dtype.itemsize else result)
            block_assigns[var] = vars[var] = result
        continue

      # MEM assignment
      if (m := re.match(r'MEM\[(.+)\]\.(\w+)\s*(\+)?=\s*(.+)', line)):
        addr, rhs, dt = parse_expr(m.group(1), ctx), parse_expr(m.group(4), ctx), DTYPES.get(m.group(2), dtypes.uint32)
        if m.group(3) == '+':
          mem = vars.get('_vmem') if '_vmem' in vars else vars.get('_lds')
          if mem is not None:
            adt = dtypes.uint64 if addr.dtype == dtypes.uint64 else dtypes.uint32
            idx = (addr >> _const(adt, 2)).cast(dtypes.index)
            old = mem.index(idx)
            if dt in (dtypes.uint64, dtypes.int64, dtypes.float64):
              old = old.cast(dtypes.uint64) | (mem.index(((addr + _const(adt, 4)) >> _const(adt, 2)).cast(dtypes.index)).cast(dtypes.uint64) << _const(dtypes.uint64, 32))
            rhs = old + rhs
        assigns.append((f'MEM[{m.group(1)}].{m.group(2)}', (addr, rhs))); i += 1; continue

      # VGPR assignment
      if (m := re.match(r'VGPR\[([^\]]+)\]\[([^\]]+)\]\s*=\s*(.+)', line)):
        ln, rg, val = parse_expr(m.group(1), ctx), parse_expr(m.group(2), ctx), parse_expr(m.group(3), ctx)
        assigns.append((f'VGPR[{m.group(1)}][{m.group(2)}]', (_to_u32(rg) * _u32(32) + _to_u32(ln), val))); i += 1; continue

      # Lambda definition
      if (m := re.match(r'(\w+)\s*=\s*lambda\(([^)]*)\)\s*\(', line)):
        body_start, depth = line[m.end():], 1
        for j, ch in enumerate(body_start):
          depth += 1 if ch == '(' else -1 if ch == ')' else 0
          if depth == 0: body = body_start[:j].strip(); i += 1; break
        else:
          body_lines = [body_start] if body_start.strip() else []
          i += 1
          while i < len(lines) and depth > 0:
            for j, ch in enumerate(lines[i]):
              depth += 1 if ch == '(' else -1 if ch == ')' else 0
              if depth == 0: body_lines.append(lines[i][:j]); break
            else: body_lines.append(lines[i])
            i += 1
          body = '\n'.join(body_lines).strip()
        vars[m.group(1)] = ('lambda', [a.strip() for a in m.group(2).split(',')], body); continue

      # Bit slice assignment
      if (m := re.match(r'(\w+)(?:\.(\w+))?\[(\d+)\s*:\s*(\d+)\](?:\.(\w+))?\s*=\s*(.+)', line)):
        var, hi, lo = m.group(1), max(int(m.group(3)), int(m.group(4))), min(int(m.group(3)), int(m.group(4)))
        val = parse_expr(m.group(6), ctx)
        assigns.append((f'{var}[{hi}:{lo}]' + (f'.{m.group(2) or m.group(5)}' if m.group(2) or m.group(5) else ''), val))
        if var not in vars: vars[var] = _const(dtypes.uint64 if hi >= 32 else dtypes.uint32, 0)
        old = block_assigns.get(var, vars.get(var))
        mask = _u32(((1 << (hi - lo + 1)) - 1) << lo)
        block_assigns[var] = vars[var] = (old & (mask ^ _u32(0xFFFFFFFF))) | (_val_to_bits(val) << _u32(lo))
        i += 1; continue

      # Compound assignment
      if (m := re.match(r'(\w+(?:\.\w+)?)\s*([+-])=\s*(.+)', line)):
        var = m.group(1).split('.')[0]
        old = block_assigns.get(var, vars.get(var, _u32(0)))
        rhs = parse_expr(m.group(3), ctx)
        if rhs.dtype != old.dtype: rhs = rhs.cast(old.dtype)
        block_assigns[var] = vars[var] = (old + rhs) if m.group(2) == '+' else (old - rhs)
        i += 1; continue

      # Array/bit indexed assignment
      if (m := re.match(r'(\w+)\{(\d+)\}\s*=\s*(.+)', line)):
        var, idx, val = m.group(1), int(m.group(2)), parse_expr(m.group(3), ctx)
        existing = block_assigns.get(var, vars.get(var))
        if existing is not None and isinstance(existing, UOp):  # bit assignment to scalar
          block_assigns[var] = vars[var] = _set_bit(existing, _u32(idx), val)
        else:  # array element or new variable
          block_assigns[f'{var}{idx}'] = vars[f'{var}{idx}'] = val
        i += 1; continue

      if (m := re.match(r'(\w+)\[([^\]]+)\]\s*=\s*(.+)', line)) and ':' not in m.group(2):
        var, bit_expr, val_expr = m.group(1), m.group(2), m.group(3)
        existing = block_assigns.get(var, vars.get(var))
        if existing is not None and isinstance(existing, UOp) and not any(f'{var}{j}' in vars or f'{var}{j}' in block_assigns for j in range(8)):
          block_assigns[var] = vars[var] = _set_bit(existing, _to_u32(parse_expr(bit_expr, ctx)), parse_expr(val_expr, ctx))
          i += 1; continue

      # Typed element assignment
      if (m := re.match(r'(\w+)\.(\w+)\[(\d+)\]\s*=\s*(.+)', line)):
        var, dt, idx = m.group(1), DTYPES.get(m.group(2), dtypes.uint32), int(m.group(3))
        val, old = parse_expr(m.group(4), ctx), block_assigns.get(var, vars.get(var, _u32(0)))
        bw, lo = dt.itemsize * 8, idx * dt.itemsize * 8
        mask = _u32(((1 << bw) - 1) << lo)
        block_assigns[var] = vars[var] = (old & (mask ^ _u32(0xFFFFFFFF))) | (((val.cast(dtypes.uint32) if val.dtype != dtypes.uint32 else val) & _u32((1 << bw) - 1)) << _u32(lo))
        assigns.append((f'{var}.{m.group(2)}[{idx}]', val)); i += 1; continue

      # Dynamic bit assignment
      if (m := re.match(r'(\w+)\.(\w+)\[(.*\[.*\].*)\]\s*=\s*(.+)', line)):
        var, bit_pos, val = m.group(1), _to_u32(parse_expr(m.group(3), ctx)), parse_expr(m.group(4), ctx)
        old, mask = block_assigns.get(var, vars.get(var, _u32(0))), _u32(1) << bit_pos
        block_assigns[var] = vars[var] = (old | mask) if val.op == Ops.CONST and val.arg == 1 else \
                                          (old & (mask ^ _u32(0xFFFFFFFF))) if val.op == Ops.CONST and val.arg == 0 else _set_bit(old, bit_pos, val)
        i += 1; continue

      # Compound destination
      if (m := re.match(r'\{\s*(\w+)\.(\w+)\s*,\s*(\w+)\.(\w+)\s*\}\s*=\s*(.+)', line)):
        val = parse_expr(m.group(5), ctx)
        lo_dt, hi_dt = DTYPES.get(m.group(4), dtypes.uint64), DTYPES.get(m.group(2), dtypes.uint32)
        lo_bits = 64 if lo_dt in (dtypes.uint64, dtypes.int64) else 32
        lo_val = val.cast(lo_dt) if val.dtype.itemsize * 8 <= lo_bits else (val & _const(val.dtype, (1 << lo_bits) - 1)).cast(lo_dt)
        hi_val = (val >> _const(val.dtype, lo_bits)).cast(hi_dt)
        block_assigns[m.group(3)] = vars[m.group(3)] = lo_val; block_assigns[m.group(1)] = vars[m.group(1)] = hi_val
        assigns.extend([(f'{m.group(3)}.{m.group(4)}', lo_val), (f'{m.group(1)}.{m.group(2)}', hi_val)]); i += 1; continue

      # Regular assignment
      if (m := re.match(r'(\w+(?:\.\w+)?(?:\[\w+\])?)\s*=\s*(.+)', line)) and not re.search(r'[<>=!]=', line[:line.find('=')]):
        base = re.match(r'(\w+)', m.group(1)).group(1)
        block_assigns[base] = vars[base] = parse_expr(m.group(2), ctx); i += 1; continue

      if (m := re.match(r'declare\s+(\w+)', line)):
        # Arrays (like in[3]) need separate element storage, scalars get initialized to 0
        if '[' in line: pass  # Don't create any variable - elements will be created on assignment
        else: vars[m.group(1)] = _u32(0)
        i += 1; continue
      i += 1
    return i, block_assigns

  lines = [l.strip().rstrip(';') for l in pcode.split('\n') if l.strip() and not l.strip().startswith('//')]
  _, final = parse_block(lines)

  sliced = set(d.split('[')[0] for d, _ in assigns if '[' in d)
  for var, val in final.items():
    if var in ['D0', 'SCC', 'VCC', 'EXEC', 'PC', 'RETURN_DATA', 'VDATA']:
      if var in sliced and not any(re.match(rf'{var}\.\w+\s*=', l) for l in lines): continue
      for l in lines:
        if (m := re.match(rf'{var}\.(\w+(?:\[\w+\])?)', l)): assigns.append((f'{var}.{m.group(1)}', val)); break
      else: assigns.append((var, val))
  return vars, assigns

def parse_expr(expr: str, vars: dict[str, UOp]) -> UOp:
  expr = expr.strip()
  if expr.startswith('(') and expr.endswith(')'):
    depth = 0
    for i, c in enumerate(expr):
      depth += (c == '(') - (c == ')')
      if depth == 0 and i < len(expr) - 1: break
    else: return parse_expr(expr[1:-1], vars)

  # Ternary
  if '?' in expr:
    dp, db, qp, cp = 0, 0, -1, -1
    for i, ch in enumerate(expr):
      dp += (ch == '(') - (ch == ')'); db += (ch == '[') - (ch == ']')
      if ch == '?' and dp == 0 and db == 0: qp = i
      elif ch == ':' and dp == 0 and db == 0 and qp >= 0: cp = i; break
    if qp >= 0 and cp >= 0:
      return _to_bool(parse_expr(expr[:qp].strip(), vars)).where(parse_expr(expr[qp+1:cp].strip(), vars), parse_expr(expr[cp+1:].strip(), vars))

  # Binary ops
  for op, op_type in [('||','|'),('&&','&'),('|','|'),('^','^'),('&','&'),('>=','>='),('<=','<='),('==','=='),('!=','!='),('<>','!='),
                      ('>>','>>'),('<<','<<'),('>','>'),('<','<'),('+','+'),('-','-'),('*','*'),('/','/'),('**','**')]:
    depths, bdepths = [0]*(len(expr)+1), [0]*(len(expr)+1)
    for i in range(len(expr)-1, -1, -1):
      depths[i], bdepths[i] = depths[i+1] + (1 if expr[i] == ')' else -1 if expr[i] == '(' else 0), bdepths[i+1] + (1 if expr[i] == ']' else -1 if expr[i] == '[' else 0)
    for i in range(len(expr)-len(op), -1, -1):
      if depths[i] == 0 and bdepths[i] == 0 and expr[i:i+len(op)] == op:
        if len(op) == 1 and ((i+1 < len(expr) and expr[i+1] in '=<>&|*') or (i > 0 and expr[i-1] in '=<>&|*')): continue
        lhs, rhs = expr[:i].strip(), expr[i+len(op):].strip()
        if op in ('-', '+') and lhs and lhs[-1] in '*/^|&<>=!': continue
        if lhs and rhs:
          l, r = parse_expr(lhs, vars), parse_expr(rhs, vars)
          if op_type in ('>>', '<<', '>=', '<=', '==', '!=', '>', '<') and l.dtype != r.dtype:
            if r.dtype == dtypes.int and r.op == Ops.CONST and r.arg < 0: l = l.cast(dtypes.int)
            else: r = r.cast(l.dtype)
          if op_type in ('|', '^', '&') and l.dtype != r.dtype:
            if l.dtype.itemsize == r.dtype.itemsize:
              t = dtypes.uint32 if l.dtype.itemsize == 4 else dtypes.uint64 if l.dtype.itemsize == 8 else l.dtype
              l, r = l.bitcast(t), r.bitcast(t)
            else: r = r.cast(l.dtype)
          return _BINOPS[op_type](l, r)

  # Type cast
  if (m := re.match(r"(\d+)'([UIFB])\((.+)\)", expr)):
    bits, inner = int(m.group(1)), parse_expr(m.group(3), vars)
    dt = {('U',32): dtypes.uint32, ('U',64): dtypes.uint64, ('I',32): dtypes.int, ('I',64): dtypes.int64, ('F',32): dtypes.float32, ('F',64): dtypes.float64}.get((m.group(2), bits), dtypes.uint64 if bits > 32 else dtypes.uint32)
    if m.group(2) == 'F' and inner.dtype in (dtypes.uint32, dtypes.uint64, dtypes.ulong, dtypes.int, dtypes.int64):
      if inner.dtype.itemsize != dt.itemsize: inner = inner.cast(dtypes.uint32 if dt.itemsize == 4 else dtypes.uint64)
      return inner.bitcast(dt)
    return inner.cast(dt)

  # Lane-indexed
  if (m := re.match(r'([a-zA-Z_]\w*)\.u64\[laneId\](?:\.(\w+))?$', expr)):
    result = (vars.get(m.group(1), _u32(0)) >> _to_u32(vars['laneId'])) & _u32(1)
    return result.cast(DTYPES.get(m.group(2), dtypes.uint32)) if m.group(2) else result

  # Variable with type
  if (m := re.match(r'([a-zA-Z_]\w*)\.(\w+)$', expr)) and m.group(1) not in ('INF', 'UNDERFLOW', 'OVERFLOW', 'NAN'):
    v = vars.get(m.group(1), _u32(0))
    if isinstance(v, dict): return v.get(m.group(2), _u32(0))
    dt = DTYPES.get(m.group(2), dtypes.uint32)
    if dt == v.dtype: return v
    if dt.itemsize == 2 and v.dtype.itemsize == 4: return (v & _const(v.dtype, 0xFFFF)).cast(dtypes.uint16) if dt == dtypes.uint16 else (v & _const(v.dtype, 0xFFFF)).cast(dtypes.uint16).bitcast(dt)
    return _cast_to(v, dt)

  # Bit slice
  if (m := re.match(r'([a-zA-Z_]\w*)\[(\d+)\s*:\s*(\d+)\](?:\.(\w+))?$', expr)):
    var, first, second, ts = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
    val = vars.get(var, _u32(0))
    if first < second:
      result = _bitreverse(val, second - first + 1)
      return _cast_to(result, DTYPES.get(ts, dtypes.uint32)) if ts else result
    hi, lo = first, second
    if lo >= val.dtype.itemsize * 8 and f'{var}{lo // 32}' in vars: val, lo, hi = vars[f'{var}{lo // 32}'], lo % 32, (hi % 32) + (lo % 32)
    result = _extract_bits(val, hi, lo)
    if ts:
      dt = DTYPES.get(ts, dtypes.uint32)
      return result.cast(dtypes.uint16).bitcast(dtypes.half) if dt == dtypes.half else _cast_to(result, dt)
    return result

  # Array bit slice
  if (m := re.match(r'([a-zA-Z_]\w*)\{(\d+)\}\[(\d+)\s*:\s*(\d+)\](?:\.(\w+))?$', expr)):
    result = _extract_bits(vars.get(f'{m.group(1)}{m.group(2)}', _u32(0)), int(m.group(3)), int(m.group(4)))
    if m.group(5):
      dt = DTYPES.get(m.group(5), dtypes.uint32)
      return result.cast(dtypes.uint16).bitcast(dtypes.half) if dt == dtypes.half else _cast_to(result, dt)
    return result

  # Verilog-style
  if (m := re.match(r"([a-zA-Z_]\w*)\.(\w+)\[(.+?)\s*\+:\s*(?:\d+')?(\d+)U?\]", expr)):
    return (vars.get(m.group(1), _u32(0)) >> _to_u32(parse_expr(m.group(3), vars))) & _const(vars.get(m.group(1), _u32(0)).dtype, (1 << int(m.group(4))) - 1)

  # Bit slice with type prefix
  if (m := re.match(r'([a-zA-Z_]\w*)\.(\w+)\[(\d+)\s*:\s*(\d+)\](?:\.(\w+))?$', expr)):
    first, second, val = int(m.group(3)), int(m.group(4)), vars.get(m.group(1), _u32(0))
    return _bitreverse(val, second - first + 1) if first < second else _extract_bits(val, first, second)

  # Single bit
  if (m := re.match(r'([a-zA-Z_]\w*)\.(\w+)\[(.+)\]$', expr)):
    v = _cast_to(vars.get(m.group(1), _u32(0)), DTYPES.get(m.group(2), dtypes.uint32))
    v = _to_u32(v)
    return (v >> (_u32(int(m.group(3))) if m.group(3).isdigit() else _to_u32(parse_expr(m.group(3), vars)))) & _u32(1)

  # Literals
  if (m := re.match(r'0x([0-9a-fA-F]+)', expr)): return _const(dtypes.uint64, int(m.group(1), 16))
  if (m := re.match(r"(\d+)'[dD](\d+)", expr)): return _const(_BITS_DT.get(int(m.group(1)), dtypes.uint32), int(m.group(2)))
  if (m := re.match(r"(\d+)'[hH]([0-9a-fA-F]+)", expr)): return _const(_BITS_DT.get(int(m.group(1)), dtypes.uint32), int(m.group(2), 16))
  if (m := re.match(r"(\d+)'[bB]([01]+)", expr)): return _const(_BITS_DT.get(int(m.group(1)), dtypes.uint32), int(m.group(2), 2))
  if (m := re.match(r"(\d+)'0x([0-9a-fA-F]+)", expr)): return _const(_BITS_DT.get(int(m.group(1)), dtypes.uint32), int(m.group(2), 16))
  if (m := re.match(r"(\d+)'-?(\d+\.\d+)", expr)):  # sized float literal e.g. 16'1.0
    bits = int(m.group(1))
    return _const({16: dtypes.half, 32: dtypes.float32, 64: dtypes.float64}.get(bits, dtypes.float32), float(m.group(2)))
  if (m := re.match(r"(\d+)'(\d+)U?", expr)):
    bits = int(m.group(1))
    return _const({1: dtypes.uint32, 8: dtypes.uint8, 16: dtypes.int16 if 'U' not in expr else dtypes.uint16, 32: dtypes.int if 'U' not in expr else dtypes.uint32, 64: dtypes.int64 if 'U' not in expr else dtypes.uint64}.get(bits, dtypes.uint32), int(m.group(2)))
  if (m := re.match(r'(-?\d+)(ULL|LL|UL|L|U)?$', expr)):
    val, sfx = int(m.group(1)), m.group(2) or ''
    return _const(dtypes.uint64 if 'U' in sfx and ('LL' in sfx or 'L' in sfx) else dtypes.uint64 if 'LL' in sfx or 'L' in sfx else dtypes.uint32 if 'U' in sfx else dtypes.int if val < 0 else dtypes.uint32, val)
  if re.match(r'-?\d+\.\d+[Ff]$', expr): return _const(dtypes.float32, float(expr.rstrip('Ff')))
  if re.match(r'-?\d+[Ff]$', expr): return _const(dtypes.float32, float(expr.rstrip('Ff')))
  if re.match(r'-?\d+\.\d+$', expr): return _const(dtypes.float64, float(expr))

  # Unary
  if expr.startswith('~'): inner = parse_expr(expr[1:], vars); return inner ^ _const(inner.dtype, (1 << (inner.dtype.itemsize * 8)) - 1)
  if expr.startswith('!'): return parse_expr(expr[1:], vars).eq(_const(parse_expr(expr[1:], vars).dtype, 0))
  if expr.startswith('-') and len(expr) > 1 and expr[1] not in '0123456789': return parse_expr(expr[1:], vars).neg()

  if expr in vars: return vars[expr]
  if expr == 'PI': return _const(dtypes.float32, 3.141592653589793)

  # Special float constants (INF, NAN, UNDERFLOW, OVERFLOW)
  _INF = {'+INF': float('inf'), 'INF': float('inf'), '-INF': float('-inf')}
  for p, v in _INF.items():
    if expr == p: return _const(dtypes.float64, v)
    if expr == f'{p}.f32': return _const(dtypes.float32, v)
    if expr == f'{p}.f16': return _const(dtypes.half, v)
  _SPECIAL = {'NAN': (0x7FC00000, dtypes.uint32, dtypes.float32), 'NAN.f32': (0x7FC00000, dtypes.uint32, dtypes.float32),
    'NAN.f64': (0x7FF8000000000000, dtypes.uint64, dtypes.float64), 'NAN.f16': (0x7E00, dtypes.uint16, dtypes.half),
    'UNDERFLOW_F32': (1, dtypes.uint32, dtypes.float32), 'OVERFLOW_F32': (0x7F7FFFFF, dtypes.uint32, dtypes.float32),
    'UNDERFLOW_F64': (1, dtypes.uint64, dtypes.float64), 'OVERFLOW_F64': (0x7FEFFFFFFFFFFFFF, dtypes.uint64, dtypes.float64)}
  if expr in _SPECIAL: bits, bt, ft = _SPECIAL[expr]; return _const(bt, bits).bitcast(ft)

  # Array variable
  if (m := re.match(r'([a-zA-Z_]\w*)\{(\d+)\}\.(\w+)$', expr)): return _cast_to(vars.get(f'{m.group(1)}{m.group(2)}', _u32(0)), DTYPES.get(m.group(3), dtypes.uint32))
  if (m := re.match(r'([a-zA-Z_]\w*)\{(\d+)\}$', expr)): return vars.get(f'{m.group(1)}{m.group(2)}', _u32(0))

  # Brace concat
  if (m := re.match(r'\{\s*(.+?)\s*,\s*(.+?)\s*\}', expr)):
    return (parse_expr(m.group(1), vars).cast(dtypes.uint64) << _const(dtypes.uint64, 32)) | parse_expr(m.group(2), vars).cast(dtypes.uint64)

  # VGPR/MEM
  if (m := re.match(r'VGPR\[([^\]]+)\]\[([^\]]+)\]', expr)):
    vgpr = vars.get('_vgpr')
    if vgpr is None: return _u32(0)
    return vgpr.index((_to_u32(parse_expr(m.group(2), vars)) * _u32(32) + _to_u32(parse_expr(m.group(1), vars))).cast(dtypes.index), ptr=True).load()
  if (m := re.match(r'MEM\[(.+)\]\.(\w+)', expr)):
    addr, dt = parse_expr(m.group(1), vars), DTYPES.get(m.group(2), dtypes.uint32)
    mem = vars.get('_vmem') if '_vmem' in vars else vars.get('_lds')
    if mem is None: return _const(dt, 0)
    adt, idx = dtypes.uint64 if addr.dtype == dtypes.uint64 else dtypes.uint32, (addr >> _const(addr.dtype, 2)).cast(dtypes.index)
    val = mem.index(idx)
    if dt in (dtypes.uint64, dtypes.int64, dtypes.float64):
      val = val.cast(dtypes.uint64) | (mem.index(((addr + _const(adt, 4)) >> _const(adt, 2)).cast(dtypes.index)).cast(dtypes.uint64) << _const(dtypes.uint64, 32))
    elif dt in (dtypes.uint8, dtypes.int8): val = (val >> ((addr & _const(adt, 3)).cast(dtypes.uint32) * _u32(8))) & _u32(0xFF)
    elif dt in (dtypes.uint16, dtypes.int16): val = (val >> (((addr >> _const(adt, 1)) & _const(adt, 1)).cast(dtypes.uint32) * _u32(16))) & _u32(0xFFFF)
    return val

  # Array element
  if (m := re.match(r'([a-zA-Z_]\w*)\[(\d+)\]\.(\w+)$', expr)):
    v = vars.get(f'{m.group(1)}{m.group(2)}') or vars.get(m.group(1))
    if v:
      dt = DTYPES.get(m.group(3), dtypes.uint32)
      if f'{m.group(1)}{m.group(2)}' in vars: return _cast_to(v, dt)
      return _cast_to((_to_u32(v) >> _u32(int(m.group(2)))) & _u32(1), dt)
  if (m := re.match(r'([a-zA-Z_]\w*)\[(\d+)\]$', expr)):
    if f'{m.group(1)}{m.group(2)}' in vars: return vars[f'{m.group(1)}{m.group(2)}']
    if m.group(1) in vars: return (_to_u32(vars[m.group(1)]) >> _u32(int(m.group(2)))) & _u32(1)

  # Lambda call
  if (m := re.match(r'(\w+)\((.+)\)$', expr)) and m.group(1) in vars and isinstance(vars[m.group(1)], tuple) and vars[m.group(1)][0] == 'lambda':
    _, params, body = vars[m.group(1)]
    args, depth, start = [], 0, 0
    for i, ch in enumerate(m.group(2)):
      depth += 1 if ch in '({' else -1 if ch in ')}' else 0
      if ch == ',' and depth == 0: args.append(m.group(2)[start:i].strip()); start = i + 1
    args.append(m.group(2)[start:].strip())
    lv = {**vars, **{p: parse_expr(a, vars) for p, a in zip(params, args)}}
    return _parse_lambda_body(body, lv) if ';' in body or '\n' in body or 'return' in body.lower() else parse_expr(body, lv)

  # Dynamic array
  if (m := re.match(r'([a-zA-Z_]\w*)\[([^\]]+)\]$', expr)) and ':' not in m.group(2):
    idx = _to_u32(parse_expr(m.group(2), vars))
    elems = [(i, vars[f'{m.group(1)}{i}']) for i in range(256) if f'{m.group(1)}{i}' in vars]
    if elems:
      result = elems[-1][1]
      for ei, ev in reversed(elems[:-1]):
        if ev.dtype != result.dtype and ev.dtype.itemsize == result.dtype.itemsize: result = result.cast(ev.dtype)
        elif ev.dtype != result.dtype: ev = ev.cast(result.dtype)
        result = idx.eq(_u32(ei)).where(ev, result)
      return result

  # General suffix
  if (m := re.match(r'(.+)\[(\d+)\]$', expr)) and not re.match(r'^[a-zA-Z_]\w*$', m.group(1)):
    return (_to_u32(parse_expr(m.group(1), vars)) >> _u32(int(m.group(2)))) & _u32(1)
  if (m := re.match(r'(.+)\[(\d+)\]\.(\w+)$', expr)) and not re.match(r'^[a-zA-Z_]\w*$', m.group(1)):
    return _cast_to((_to_u32(parse_expr(m.group(1), vars)) >> _u32(int(m.group(2)))) & _u32(1), DTYPES.get(m.group(3), dtypes.uint32))

  if (result := _parse_func(expr, vars)) is not None: return result
  # Check if this looks like a function call that we failed to handle
  if re.match(r'\w+\s*\(.+\)', expr): raise RuntimeError(f"[pcode] unhandled function call: {expr}")
  raise RuntimeError(f"[pcode] unhandled expression: {expr}")

def _parse_lambda_body(body: str, vars: dict[str, UOp]) -> UOp:
  return _parse_lambda_block([l.strip() for l in body.replace(';', '\n').split('\n') if l.strip() and not l.strip().startswith('//')], 0, vars)[1]

def _parse_lambda_block(lines: list[str], start: int, vars: dict[str, UOp]) -> tuple[int, UOp]:
  i = start
  while i < len(lines):
    line = lines[i]
    if re.match(r'(elsif|else|endif|endfor)\b', line, re.IGNORECASE): break
    if (m := re.match(r'return\s+(.+)$', line, re.IGNORECASE)): return i + 1, parse_expr(m.group(1), vars)
    if (m := re.match(r'for\s+(\w+)\s+in\s+(\d+)\s*:\s*(\d+)\s+do', line, re.IGNORECASE)):
      lv, sv, ev = m.group(1), int(m.group(2)), int(m.group(3))
      i += 1; body, depth = [], 1
      while i < len(lines) and depth > 0:
        depth += 1 if re.match(r'for\s+', lines[i], re.IGNORECASE) else -1 if re.match(r'endfor\b', lines[i], re.IGNORECASE) else 0
        if depth > 0: body.append(lines[i]); i += 1
        else: i += 1
      for li in range(sv, ev + 1):
        for bl in body:
          subst = re.sub(r'\[([^\]\[]+?)\s*:\s*([^\]\[]+?)\]', lambda m: f'[{_try_eval(m.group(1))} : {_try_eval(m.group(2))}]', re.sub(rf'\b{lv}\b', str(li), bl))
          if (am := re.match(r'(\w+)\[(\d+)\]\s*=\s*(.+)', subst)): vars[f'{am.group(1)}{am.group(2)}'] = parse_expr(am.group(3), vars)
      continue
    if re.match(r'declare\s+', line, re.IGNORECASE): i += 1; continue
    if (m := re.match(r'if\s+(.+?)\s+then$', line, re.IGNORECASE)):
      conds = [(_to_bool(parse_expr(m.group(1), vars)), None)]
      i += 1; i, rv = _parse_lambda_block(lines, i, vars); conds[0] = (conds[0][0], rv)
      while i < len(lines):
        if (m := re.match(r'elsif\s+(.+?)\s+then$', lines[i], re.IGNORECASE)):
          i += 1; i, rv = _parse_lambda_block(lines, i, vars); conds.append((_to_bool(parse_expr(m.group(1), vars)), rv))
        elif re.match(r'else$', lines[i], re.IGNORECASE):
          i += 1; i, er = _parse_lambda_block(lines, i, vars)
          result = er
          for c, rv in reversed(conds):
            if rv is not None:
              if rv.dtype != result.dtype and rv.dtype.itemsize == result.dtype.itemsize: result = result.cast(rv.dtype)
              result = c.where(rv, result)
          return i, result
        elif re.match(r'endif\b', lines[i], re.IGNORECASE): i += 1; break
        else: break
      continue
    i += 1
  return i, _u32(0)

def _floor(x): t = UOp(Ops.TRUNC, x.dtype, (x,)); return ((x < _const(x.dtype, 0)) & x.ne(t)).where(t - _const(x.dtype, 1), t)
def _f16_extract(v): return (v & _u32(0xFFFF)).cast(dtypes.uint16).bitcast(dtypes.half) if v.dtype == dtypes.uint32 else v

def _check_nan(inner: str, vars: dict, quiet: bool) -> UOp:
  if (m := re.match(r"64'F\((.+)\)", inner)): inner = m.group(1)
  v = parse_expr(inner, vars)
  bits, exp_m, mant_m, qb, _ = _float_info(v, inner)
  is_nan_exp, has_mant, is_q = (bits & exp_m).eq(exp_m), (bits & mant_m).ne(_const(bits.dtype, 0)), (bits & qb).ne(_const(bits.dtype, 0))
  return (is_nan_exp & is_q) if quiet else (is_nan_exp & has_mant & is_q.logical_not())

def _minmax_reduce(is_max, dt, args):
  def cast(v): return v.bitcast(dt) if dt == dtypes.float32 and v.dtype == dtypes.uint32 else v.cast(dt)
  result = cast(args[0])
  for a in args[1:]:
    b = cast(a)
    if dt == dtypes.float32: result = _isnan(result).where(b, _isnan(b).where(result, result.maximum(b) if is_max else result.minimum(b)))
    else: result = result.maximum(b) if is_max else result.minimum(b)
  return result

_FUNC_TABLE: list[tuple[str, int, callable]] = []

def _register_funcs():
  global _FUNC_TABLE
  def _find_two_pi_mul(x):
    if x.op != Ops.MUL or len(x.src) != 2: return None
    for i, s in enumerate(x.src):
      if s.op == Ops.CONST and abs(s.arg - 6.283185307179586) < 1e-5: return (x.src[1-i], 6.283185307179586)
      if s.op == Ops.MUL and len(s.src) == 2:
        vals = [ss.arg for ss in s.src if ss.op == Ops.CONST] + [ss.src[0].arg for ss in s.src if ss.op == Ops.CAST and ss.src[0].op == Ops.CONST]
        if len(vals) == 2 and abs(vals[0] * vals[1] - 6.283185307179586) < 1e-5: return (x.src[1-i], vals[0] * vals[1])
    return None

  def _trig_reduce(x, phase=0.0):
    match = _find_two_pi_mul(x)
    if match is not None:
      turns, two_pi = match
      if phase: turns = turns + _const(turns.dtype, phase)
      n = _floor(turns + _const(turns.dtype, 0.5))
      return UOp(Ops.SIN, turns.dtype, ((turns - n) * _const(turns.dtype, two_pi),))
    if phase: x = x + _const(x.dtype, phase * 6.283185307179586)
    n = _floor(x * _const(x.dtype, 0.15915494309189535) + _const(x.dtype, 0.5))
    return UOp(Ops.SIN, x.dtype, (x - n * _const(x.dtype, 6.283185307179586),))

  for name, op in [('sqrt', Ops.SQRT), ('trunc', Ops.TRUNC), ('log2', Ops.LOG2)]:
    _FUNC_TABLE.append((rf'{name}\((.+)\)', 1, lambda a, v, m, op=op: UOp(op, a[0].dtype, (a[0],))))
  _FUNC_TABLE.append((r'sin\((.+)\)', 1, lambda a, v, m: _trig_reduce(a[0])))
  _FUNC_TABLE.append((r'cos\((.+)\)', 1, lambda a, v, m: _trig_reduce(a[0], 0.25)))
  _FUNC_TABLE.append((r'floor\((.+)\)', 1, lambda a, v, m: _floor(a[0])))
  _FUNC_TABLE.append((r'fract\((.+)\)', 1, lambda a, v, m: a[0] - _floor(a[0])))

  def _signext(a, v, m):
    val = a[0]
    for bits, mask, ext in [(8, 0xFF, 0xFFFFFF00), (16, 0xFFFF, 0xFFFF0000)]:
      if (val.op == Ops.AND and len(val.src) == 2 and val.src[1].op == Ops.CONST and val.src[1].arg == mask) or val.dtype.itemsize == bits // 8:
        v32 = val.cast(dtypes.uint32) if val.dtype != dtypes.uint32 else val
        sb = (v32 >> _u32(bits - 1)) & _u32(1)
        return sb.ne(_u32(0)).where(v32 | _u32(ext), v32).cast(dtypes.int)
    return val.cast(dtypes.int64) if val.dtype in (dtypes.int, dtypes.int32) else val
  _FUNC_TABLE.append((r'signext\((.+)\)', 1, _signext))
  _FUNC_TABLE.append((r'isEven\((.+)\)', 1, lambda a, v, m: (UOp(Ops.TRUNC, a[0].dtype, (a[0],)).cast(dtypes.int) & _const(dtypes.int, 1)).eq(_const(dtypes.int, 0))))
  def _abs(a, v, m):
    if a[0].dtype not in (dtypes.float32, dtypes.float64, dtypes.half): return a[0]
    _, _, _, _, shift = _float_info(a[0])
    sign_mask = {10: 0x7FFF, 23: 0x7FFFFFFF, 52: 0x7FFFFFFFFFFFFFFF}[shift]
    bt, ft = {10: (dtypes.uint16, dtypes.half), 23: (dtypes.uint32, dtypes.float32), 52: (dtypes.uint64, dtypes.float64)}[shift]
    return (a[0].bitcast(bt) & _const(bt, sign_mask)).bitcast(ft)
  _FUNC_TABLE.append((r'abs\((.+)\)', 1, _abs))

  _FUNC_TABLE.append((r'max\((.+),\s*(.+)\)', 2, lambda a, v, m: UOp(Ops.MAX, a[0].dtype, (a[0], a[1]))))
  _FUNC_TABLE.append((r'min\((.+),\s*(.+)\)', 2, lambda a, v, m: UOp(Ops.MAX, a[0].dtype, (a[0].neg(), a[1].neg())).neg()))
  _FUNC_TABLE.append((r'pow\((.+),\s*(.+)\)', 2, lambda a, v, m: UOp(Ops.EXP2, dtypes.float32, (a[1].bitcast(dtypes.float32),)) if '2.0' in m.group(1) else a[0]))
  _FUNC_TABLE.append((r'fma\((.+),\s*(.+),\s*(.+)\)', 3, lambda a, v, m: a[0] * a[1] + a[2]))

  for src in ['i32', 'u32']: _FUNC_TABLE.append((rf'{src}_to_f32\((.+)\)', 1, lambda a, v, m: a[0].cast(dtypes.int if 'i32' in m.group(0) else dtypes.uint32).cast(dtypes.float32)))
  _FUNC_TABLE.append((r'f32_to_i32\((.+)\)', 1, lambda a, v, m: UOp(Ops.TRUNC, dtypes.float32, (a[0].bitcast(dtypes.float32),)).cast(dtypes.int)))
  def _f_to_u(f, dt): return UOp(Ops.TRUNC, f.dtype, ((f < _const(f.dtype, 0.0)).where(_const(f.dtype, 0.0), f),)).cast(dt)
  _FUNC_TABLE.append((r'f32_to_u32\((.+)\)', 1, lambda a, v, m: _f_to_u(a[0].bitcast(dtypes.float32), dtypes.uint32)))
  _FUNC_TABLE.append((r'f64_to_i32\((.+)\)', 1, lambda a, v, m: UOp(Ops.TRUNC, dtypes.float64, (a[0].bitcast(dtypes.float64),)).cast(dtypes.int)))
  _FUNC_TABLE.append((r'f64_to_u32\((.+)\)', 1, lambda a, v, m: _f_to_u(a[0].bitcast(dtypes.float64), dtypes.uint32)))
  _FUNC_TABLE.append((r'f16_to_f32\((.+)\)', 1, lambda a, v, m: _f16_extract(a[0]).cast(dtypes.float32)))
  _FUNC_TABLE.append((r'f32_to_f16\((.+)\)', 1, lambda a, v, m: a[0].cast(dtypes.half)))
  _FUNC_TABLE.append((r'f32_to_f64\((.+)\)', 1, lambda a, v, m: a[0].bitcast(dtypes.float32).cast(dtypes.float64)))
  _FUNC_TABLE.append((r'f64_to_f32\((.+)\)', 1, lambda a, v, m: a[0].bitcast(dtypes.float64).cast(dtypes.float32)))
  _FUNC_TABLE.append((r'i32_to_f64\((.+)\)', 1, lambda a, v, m: a[0].cast(dtypes.int).cast(dtypes.float64)))
  _FUNC_TABLE.append((r'u32_to_f64\((.+)\)', 1, lambda a, v, m: a[0].cast(dtypes.uint32).cast(dtypes.float64)))
  _FUNC_TABLE.append((r'f16_to_i16\((.+)\)', 1, lambda a, v, m: UOp(Ops.TRUNC, dtypes.half, (_f16_extract(a[0]),)).cast(dtypes.int16)))
  _FUNC_TABLE.append((r'f16_to_u16\((.+)\)', 1, lambda a, v, m: UOp(Ops.TRUNC, dtypes.half, (_f16_extract(a[0]),)).cast(dtypes.uint16)))
  _FUNC_TABLE.append((r'i16_to_f16\((.+)\)', 1, lambda a, v, m: a[0].cast(dtypes.int16).cast(dtypes.half)))
  _FUNC_TABLE.append((r'u16_to_f16\((.+)\)', 1, lambda a, v, m: a[0].cast(dtypes.uint16).cast(dtypes.half)))
  _FUNC_TABLE.append((r'bf16_to_f32\((.+)\)', 1, lambda a, v, m: (((a[0].cast(dtypes.uint32) if a[0].dtype != dtypes.uint32 else a[0]) & _u32(0xFFFF)) << _u32(16)).bitcast(dtypes.float32)))

  _FUNC_TABLE.append((r'isNAN\((.+)\)', 1, lambda a, v, m: _isnan(a[0])))
  _FUNC_TABLE.append((r'isSignalNAN\((.+)\)', 0, lambda a, v, m: _check_nan(m.group(1), v, False)))
  _FUNC_TABLE.append((r'isQuietNAN\((.+)\)', 0, lambda a, v, m: _check_nan(m.group(1), v, True)))
  def _cvt_quiet(a, v, m):
    bits, _, _, qb, _ = _float_info(a[0])
    bt, ft = (dtypes.uint64, dtypes.float64) if a[0].dtype == dtypes.float64 else (dtypes.uint16, dtypes.half) if a[0].dtype == dtypes.half else (dtypes.uint32, dtypes.float32)
    return (a[0].bitcast(bt) | qb).bitcast(ft)
  _FUNC_TABLE.append((r'cvtToQuietNAN\((.+)\)', 1, _cvt_quiet))

  def _is_denorm(a, v, m):
    bits, exp_m, mant_m, _, _ = _float_info(a[0], m.group(1))
    return (bits & exp_m).eq(_const(bits.dtype, 0)) & (bits & mant_m).ne(_const(bits.dtype, 0))
  _FUNC_TABLE.append((r'isDENORM\((.+)\)', 1, _is_denorm))

  def _exponent(a, v, m):
    bits, _, _, _, shift = _float_info(a[0], m.group(1))
    exp_bits = {10: 0x1F, 23: 0xFF, 52: 0x7FF}[shift]
    return ((bits >> _const(bits.dtype, shift)) & _const(bits.dtype, exp_bits)).cast(dtypes.int)
  _FUNC_TABLE.append((r'exponent\((.+)\)', 1, _exponent))

  def _div_would_be_denorm(a, v, m):
    bits_n, _, _, _, shift = _float_info(a[0], m.group(0))
    bits_d, _, _, _, _ = _float_info(a[1], m.group(0))
    exp_bits, min_exp = {10: (0x1F, -14), 23: (0xFF, -126), 52: (0x7FF, -1022)}[shift]
    exp_n = ((bits_n >> _const(bits_n.dtype, shift)) & _const(bits_n.dtype, exp_bits)).cast(dtypes.int)
    exp_d = ((bits_d >> _const(bits_d.dtype, shift)) & _const(bits_d.dtype, exp_bits)).cast(dtypes.int)
    return (exp_n - exp_d) < _const(dtypes.int, min_exp)
  _FUNC_TABLE.append((r'divWouldBeDenorm\((.+),\s*(.+)\)', 2, _div_would_be_denorm))

  def _sign(a, v, m):
    bits, _, _, _, shift = _float_info(a[0], m.group(1))
    sign_shift = {10: 15, 23: 31, 52: 63}[shift]
    return ((bits >> _const(bits.dtype, sign_shift)) & _const(bits.dtype, 1)).cast(dtypes.uint32)
  _FUNC_TABLE.append((r'sign\((.+)\)', 1, _sign))

  _FUNC_TABLE.append((r'signext_from_bit\((.+),\s*(.+)\)', 2, lambda a, v, m: (lambda val, w: ((val >> (w - _u32(1))) & _u32(1)).ne(_u32(0)).where(val | (((_u32(1) << w) - _u32(1)) ^ _u32(0xFFFFFFFF)), val))(_to_u32(a[0]), _to_u32(a[1]))))

  for is_max, name in [(False, 'min'), (True, 'max')]:
    for dt, sfx in [(dtypes.float32, 'f32'), (dtypes.int, 'i32'), (dtypes.uint32, 'u32'), (dtypes.int16, 'i16'), (dtypes.uint16, 'u16')]:
      _FUNC_TABLE.append((rf'v_{name}_{sfx}\((.+),\s*(.+)\)', 2, lambda a, v, m, im=is_max, d=dt: _minmax_reduce(im, d, a)))
      _FUNC_TABLE.append((rf'v_{name}3_{sfx}\((.+),\s*(.+),\s*(.+)\)', 3, lambda a, v, m, im=is_max, d=dt: _minmax_reduce(im, d, a)))

  def _ldexp(a, v, m):
    val, exp = a[0], a[1]
    if val.dtype == dtypes.uint32: val = val.bitcast(dtypes.float32)
    elif val.dtype == dtypes.uint64: val = val.bitcast(dtypes.float64)
    if exp.dtype in (dtypes.uint32, dtypes.uint64): exp = exp.cast(dtypes.int if exp.dtype == dtypes.uint32 else dtypes.int64)
    return val * UOp(Ops.EXP2, val.dtype, (exp.cast(val.dtype),))
  _FUNC_TABLE.append((r'ldexp\((.+),\s*(.+)\)', 2, _ldexp))

  def _frexp_mant(a, v, m):
    val = a[0].bitcast(dtypes.float32) if a[0].dtype == dtypes.uint32 else a[0].bitcast(dtypes.float64) if a[0].dtype == dtypes.uint64 else a[0]
    if val.dtype == dtypes.float32: return ((val.bitcast(dtypes.uint32) & _u32(0x807FFFFF)) | _u32(0x3f000000)).bitcast(dtypes.float32)
    return ((val.bitcast(dtypes.uint64) & _const(dtypes.uint64, 0x800FFFFFFFFFFFFF)) | _const(dtypes.uint64, 0x3fe0000000000000)).bitcast(dtypes.float64)
  _FUNC_TABLE.append((r'frexp_mant\((.+)\)', 1, _frexp_mant)); _FUNC_TABLE.append((r'mantissa\((.+)\)', 1, _frexp_mant))

  def _frexp_exp(a, v, m):
    val = a[0].bitcast(dtypes.float32) if a[0].dtype == dtypes.uint32 else a[0].bitcast(dtypes.float64) if a[0].dtype == dtypes.uint64 else a[0]
    if val.dtype == dtypes.float32: return ((val.bitcast(dtypes.uint32) >> _u32(23)) & _u32(0xFF)).cast(dtypes.int) - _const(dtypes.int, 126)
    return ((val.bitcast(dtypes.uint64) >> _const(dtypes.uint64, 52)) & _const(dtypes.uint64, 0x7FF)).cast(dtypes.int) - _const(dtypes.int, 1022)
  _FUNC_TABLE.append((r'frexp_exp\((.+)\)', 1, _frexp_exp))

  TWO_OVER_PI = 0x0145f306dc9c882a53f84eafa3ea69bb81b6c52b3278872083fca2c757bd778ac36e48dc74849ba5c00c925dd413a32439fc3bd63962534e7dd1046bea5d768909d338e04d68befc827323ac7306a673e93908bf177bf250763ff12fffbc0b301fde5e2316b414da3eda6cfd9e4f96136e9e8c7ecd3cbfd45aea4f758fd7cbe2f67a0e73ef14a525d4d7f6bf623f1aba10ac06608df8f6
  _PREOP = {s: float(((TWO_OVER_PI << s) >> (1201 - 53)) & 0x1fffffffffffff) for s in range(1149)}
  def _trig_preop(a, v, m):
    if a[0].op == Ops.CONST: return _const(dtypes.float64, _PREOP.get(int(a[0].arg), float(((TWO_OVER_PI << int(a[0].arg)) >> (1201 - 53)) & 0x1fffffffffffff)))
    result = _const(dtypes.float64, _PREOP[0])
    for s in range(1148, -1, -1): result = a[0].eq(_const(a[0].dtype, s)).where(_const(dtypes.float64, _PREOP[s]), result)
    return result
  _FUNC_TABLE.append((r'trig_preop_result\((.+)\)', 1, _trig_preop))

  def _ff1(a, v, m, bits):
    dt = dtypes.uint64 if bits == 64 else dtypes.uint32
    val = a[0].cast(dt) if a[0].dtype != dt else a[0]
    result = _const(dtypes.int, -1)
    for i in range(bits):
      cond = ((val >> _const(dt, i)) & _const(dt, 1)).ne(_const(dt, 0)) & result.eq(_const(dtypes.int, -1))
      result = cond.where(_const(dtypes.int, i), result)
    return result
  _FUNC_TABLE.append((r's_ff1_i32_b32\((.+)\)', 1, lambda a, v, m: _ff1(a, v, m, 32)))
  _FUNC_TABLE.append((r's_ff1_i32_b64\((.+)\)', 1, lambda a, v, m: _ff1(a, v, m, 64)))

_register_funcs()

def _parse_func(expr: str, vars: dict[str, UOp]) -> UOp | None:
  for pattern, nargs, handler in _FUNC_TABLE:
    if (m := re.match(pattern, expr)):
      return handler([parse_expr(m.group(i+1), vars) for i in range(nargs)] if nargs > 0 else [], vars, m)
  return None
