# AMD pcode parser - converts pseudocode to UOps
import re
from tinygrad.dtype import dtypes, DType
from tinygrad.uop.ops import Ops, UOp
from extra.assembly.amd.expr_parser import parse_expr as _tokenized_parse_expr, _FUNCS, _set_bit, _val_to_bits, _to_bool, _try_eval

DTYPES = {'u32': dtypes.uint32, 'i32': dtypes.int, 'f32': dtypes.float32, 'b32': dtypes.uint32, 'u64': dtypes.uint64, 'i64': dtypes.int64,
          'f64': dtypes.float64, 'b64': dtypes.uint64, 'u16': dtypes.uint16, 'i16': dtypes.short, 'f16': dtypes.half, 'b16': dtypes.uint16,
          'u8': dtypes.uint8, 'i8': dtypes.int8, 'b8': dtypes.uint8, 'u1': dtypes.uint32}

def _const(dt, v): return UOp.const(dt, v)
def _u32(v): return _const(dtypes.uint32, v)
def _u64(v): return _const(dtypes.uint64, v)
def _to_u32(v): return v if v.dtype == dtypes.uint32 else v.bitcast(dtypes.uint32) if v.dtype.itemsize == 4 else v.cast(dtypes.uint32)

def _apply_pseudocode_fixes(op_name: str, pcode: str) -> str:
  fixes = {
    # V_DIV_FMAS: scale direction depends on S2's exponent (compensates for v_div_scale's scaling)
    # For f32: exponent > 127 means S2 >= 2.0, scale UP; else scale DOWN
    # For f64: exponent > 1023 means S2 >= 2.0, scale UP; else scale DOWN
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
    # Convert whole-VCC assignments to per-lane assignments (VCC = 0x0LL -> VCC.u64[laneId] = 0)
    pcode = pcode.replace('VCC = 0x0LL', 'VCC.u64[laneId] = 0').replace('VCC = 0x1LL', 'VCC.u64[laneId] = 1')
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
  return _tokenized_parse_expr(expr, vars, _FUNCS)
