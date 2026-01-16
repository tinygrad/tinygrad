# AMD pcode parser - converts pseudocode to UOps
import re
from tinygrad.dtype import dtypes, DType
from tinygrad.uop.ops import Ops, UOp

DTYPES = {'u32': dtypes.uint32, 'i32': dtypes.int, 'f32': dtypes.float32, 'b32': dtypes.uint32,
          'u64': dtypes.uint64, 'i64': dtypes.int64, 'f64': dtypes.float64, 'b64': dtypes.uint64,
          'u16': dtypes.uint16, 'i16': dtypes.short, 'f16': dtypes.half, 'b16': dtypes.uint16}

def parse_pcode(pcode: str, srcs: dict[str, UOp] | None = None, lane: UOp | None = None) -> tuple[dict[str, UOp], list[tuple[str, UOp]]]:
  """Parse pcode into UOps. srcs can provide actual UOps for S0, S1, S2, D0, VCC, EXEC, SCC, SIMM32.
  
  For DS instructions, srcs should include: ADDR, OFFSET0, OFFSET1, DATA, DATA2, _lds, _wlds_fn, _exec_mask
  Returns assigns with special keys: 'MEM[...]' for LDS writes, 'RETURN_DATA[h:l]' for bit slice returns
  """
  vars: dict[str, UOp] = {n: UOp(Ops.DEFINE_VAR, dtypes.uint32, (), (n, UOp.const(dtypes.uint32, 0), UOp.const(dtypes.uint32, 0xFFFFFFFF)))
                          for n in ['S0', 'S1', 'S2', 'D0', 'VCC', 'EXEC', 'SCC', 'SIMM32']}
  if srcs: vars.update(srcs)
  vars['laneId'] = lane if lane is not None else UOp.const(dtypes.uint32, 0)
  assigns: list[tuple[str, UOp]] = []

  # Check for for loops with break (CLZ, CTZ patterns)
  if 'for ' in pcode.lower() and ' do' in pcode.lower() and 'break' in pcode.lower():
    return parse_pcode_with_for_loop(pcode, vars, assigns)

  # Check for if/elsif/else blocks and handle them specially
  if 'if ' in pcode.lower() and 'then' in pcode.lower():
    return parse_pcode_with_conditionals(pcode, vars, assigns)

  for stmt in re.split(r'[;\n]', pcode):
    stmt = stmt.strip()
    if not stmt or stmt.startswith('//'): continue

    # MEM[addr].type = value - LDS memory write
    if (m := re.match(r'MEM\[(.+)\]\.(\w+)\s*=\s*(.+)', stmt)):
      addr_expr, dtype_str, val_expr = m.group(1), m.group(2), m.group(3)
      addr = parse_expr(addr_expr, vars)
      val = parse_expr(val_expr, vars)
      assigns.append((f'MEM[{addr_expr}].{dtype_str}', (addr, val)))
      continue

    # MEM[addr].type += value - LDS compound assignment
    if (m := re.match(r'MEM\[(.+)\]\.(\w+)\s*\+=\s*(.+)', stmt)):
      addr_expr, dtype_str, val_expr = m.group(1), m.group(2), m.group(3)
      addr = parse_expr(addr_expr, vars)
      # Read current value, add, then write back
      lds = vars.get('_lds')
      if lds is not None:
        idx = (addr >> UOp.const(dtypes.uint32, 2)).cast(dtypes.index)
        old_val = lds.index(idx)
        new_val = old_val + parse_expr(val_expr, vars)
        assigns.append((f'MEM[{addr_expr}].{dtype_str}', (addr, new_val)))
      continue

    # VAR[high : low] = value - bit slice assignment (for RETURN_DATA)
    if (m := re.match(r'(\w+)\[(\d+)\s*:\s*(\d+)\]\s*=\s*(.+)', stmt)):
      var_name, high_bit, low_bit, val_expr = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
      val = parse_expr(val_expr, vars)
      assigns.append((f'{var_name}[{high_bit}:{low_bit}]', val))
      # Also update vars for subsequent reads
      if var_name not in vars:
        vars[var_name] = UOp.const(dtypes.uint64 if high_bit >= 32 else dtypes.uint32, 0)
      continue

    # Regular assignment: VAR = value or VAR.type = value
    if (m := re.match(r'(\w+(?:\.\w+)?(?:\[\w+\])?)\s*=\s*(.+)', stmt)) and not re.search(r'[<>=!]=', stmt[:stmt.find('=')]):
      lhs, val = m.group(1), parse_expr(m.group(2), vars)
      base = re.match(r'(\w+)', lhs).group(1)
      if base in ['D0', 'SCC', 'VCC', 'EXEC', 'RETURN_DATA']: assigns.append((lhs, val))
      vars[base] = val
  return vars, assigns

def parse_pcode_with_for_loop(pcode: str, vars: dict[str, UOp], assigns: list[tuple[str, UOp]]) -> tuple[dict[str, UOp], list[tuple[str, UOp]]]:
  """Parse pcode with for loop and break (like CLZ/CTZ patterns).

  Pattern: for i in START : END do if COND then ASSIGNMENT; break endif endfor
  This is converted to nested WHERE: for each i from END-1 down to START,
  if COND[i] then result = i, else result = previous_result
  """
  lines = [l.strip() for l in pcode.split('\n') if l.strip() and not l.strip().startswith('//')]

  # Parse pre-loop assignments (e.g., D0.i32 = -1)
  pre_loop_assigns = {}
  i = 0
  while i < len(lines):
    line = lines[i]
    if 'for ' in line.lower():
      break
    if (m := re.match(r'(\w+)(?:\.\w+)?\s*=\s*(.+)', line)):
      var_name = m.group(1)
      pre_loop_assigns[var_name] = parse_expr(m.group(2).rstrip(';').strip(), vars)  # Strip trailing semicolon
      vars[var_name] = pre_loop_assigns[var_name]
    i += 1

  # Parse for loop: for i in START : END do
  for_line = lines[i] if i < len(lines) else ""
  m = re.match(r'for\s+(\w+)\s+in\s+(\d+)\s*:\s*(\d+)\s+do', for_line, re.IGNORECASE)
  if not m:
    return vars, assigns
  loop_var = m.group(1)
  start_val = int(m.group(2))
  end_val = int(m.group(3))
  i += 1

  # Parse loop body: if COND then ASSIGNMENT; break endif
  cond_expr = None
  loop_assign_var = None
  loop_assign_expr = None
  while i < len(lines) and 'endfor' not in lines[i].lower():
    line = lines[i]
    if (m := re.match(r'if\s+(.+?)\s+then', line, re.IGNORECASE)):
      cond_expr = m.group(1)
    elif (m := re.match(r'(\w+)(?:\.\w+)?\s*=\s*(.+)', line)) and 'break' not in line.lower():
      loop_assign_var = m.group(1)
      loop_assign_expr = m.group(2).rstrip(';').strip()  # Strip trailing semicolon
    i += 1

  if cond_expr is None or loop_assign_var is None:
    return vars, assigns

  # Build nested WHERE by unrolling the loop backwards
  # Start with the default value (from pre-loop assignment)
  result = pre_loop_assigns.get(loop_assign_var, UOp.const(dtypes.int, -1))

  # Unroll from end to start (so first match wins with nested WHERE)
  for loop_i in range(end_val, start_val - 1, -1):
    # Substitute loop variable with current value
    loop_vars = vars.copy()
    loop_vars[loop_var] = UOp.const(dtypes.uint32, loop_i)

    # Parse condition with loop variable substituted
    cond = parse_expr(cond_expr, loop_vars)
    if cond.dtype != dtypes.bool:
      cond = cond.ne(UOp.const(cond.dtype, 0))

    # Parse assignment value (usually just the loop variable)
    assign_val = parse_expr(loop_assign_expr, loop_vars)
    # Ensure consistent dtype with result for WHERE
    if assign_val.dtype != result.dtype:
      assign_val = assign_val.cast(result.dtype)

    # Build WHERE: if cond then assign_val else previous_result
    result = cond.where(assign_val, result)

  vars[loop_assign_var] = result

  # Build final assigns
  final_assigns = []
  for var_name, val in pre_loop_assigns.items():
    if var_name in ['D0']:
      # Use the loop result, not the pre-loop value
      final_assigns.append((f'{var_name}.i32', vars[var_name]))

  return vars, final_assigns if final_assigns else assigns

def parse_pcode_with_conditionals(pcode: str, vars: dict[str, UOp], assigns: list[tuple[str, UOp]]) -> tuple[dict[str, UOp], list[tuple[str, UOp]]]:
  """Parse pcode with if/elsif/else blocks into nested WHERE expressions."""
  lines = [l.strip() for l in pcode.split('\n') if l.strip() and not l.strip().startswith('//')]

  # Extract conditions and their corresponding assignments
  # Format: if COND then STMT elsif COND then STMT ... else STMT endif
  conditions = []  # [(condition_uop, {var: value}), ...]
  else_assigns = {}

  i = 0
  while i < len(lines):
    line = lines[i]

    # Skip declarations like "declare result : 1'U;"
    if line.startswith('declare'):
      # Initialize declared variable to 0
      if (m := re.match(r'declare\s+(\w+)', line)):
        vars[m.group(1)] = UOp.const(dtypes.uint32, 0)
      i += 1
      continue

    # if COND then
    if (m := re.match(r'if\s+(.+?)\s+then$', line, re.IGNORECASE)):
      cond = parse_expr(m.group(1), vars)
      if cond.dtype != dtypes.bool: cond = cond.ne(UOp.const(cond.dtype, 0))
      i += 1
      block_assigns = {}
      while i < len(lines) and not re.match(r'(elsif|else|endif)', lines[i], re.IGNORECASE):
        stmt = lines[i]
        if (m2 := re.match(r'(\w+(?:\.\w+)?)\s*\+=\s*(.+)', stmt)):
          # Handle += operator
          var_name = m2.group(1).split('.')[0]
          old_val = vars.get(var_name, UOp.const(dtypes.uint32, 0))
          block_assigns[var_name] = old_val + parse_expr(m2.group(2), vars)
        elif (m2 := re.match(r'(\w+(?:\.\w+)?)\s*-=\s*(.+)', stmt)):
          # Handle -= operator
          var_name = m2.group(1).split('.')[0]
          old_val = vars.get(var_name, UOp.const(dtypes.uint32, 0))
          block_assigns[var_name] = old_val - parse_expr(m2.group(2), vars)
        elif (m2 := re.match(r'(\w+)\s*=\s*(.+)', stmt)):
          block_assigns[m2.group(1)] = parse_expr(m2.group(2), vars)
        i += 1
      conditions.append((cond, block_assigns))
      continue

    # elsif COND then
    if (m := re.match(r'elsif\s+(.+?)\s+then$', line, re.IGNORECASE)):
      cond = parse_expr(m.group(1), vars)
      if cond.dtype != dtypes.bool: cond = cond.ne(UOp.const(cond.dtype, 0))
      i += 1
      block_assigns = {}
      while i < len(lines) and not re.match(r'(elsif|else|endif)', lines[i], re.IGNORECASE):
        stmt = lines[i]
        if (m2 := re.match(r'(\w+)\s*=\s*(.+)', stmt)):
          block_assigns[m2.group(1)] = parse_expr(m2.group(2), vars)
        i += 1
      conditions.append((cond, block_assigns))
      continue

    # else
    if re.match(r'else$', line, re.IGNORECASE):
      i += 1
      while i < len(lines) and not re.match(r'endif', lines[i], re.IGNORECASE):
        stmt = lines[i]
        if (m2 := re.match(r'(\w+)\s*=\s*(.+)', stmt)):
          else_assigns[m2.group(1)] = parse_expr(m2.group(2), vars)
        i += 1
      continue

    # endif or simple assignment outside conditionals
    if re.match(r'endif', line, re.IGNORECASE):
      i += 1
      continue

    # Simple assignment
    if (m := re.match(r'(\w+(?:\.\w+)?(?:\[\w+\])?)\s*=\s*(.+)', line)) and not re.search(r'[<>=!]=', line[:line.find('=')]):
      lhs, val = m.group(1), parse_expr(m.group(2), vars)
      base = re.match(r'(\w+)', lhs).group(1)
      if base in ['D0', 'SCC', 'VCC', 'EXEC']:
        assigns.append((lhs, val))
      vars[base] = val

    i += 1

  # Build nested WHERE for each variable that was assigned in conditionals
  all_vars = set()
  for _, block in conditions:
    all_vars.update(block.keys())
  all_vars.update(else_assigns.keys())

  for var in all_vars:
    # Start with else value (or current value of var if no else)
    result = else_assigns.get(var, vars.get(var, UOp.const(dtypes.uint32, 0)))
    # Build nested WHERE from last condition to first
    for cond, block in reversed(conditions):
      if var in block:
        result = cond.where(block[var], result)
    vars[var] = result

  # Build final assigns:
  # 1. Look for explicit final assignments like D0.u64[laneId] = result
  final_assigns = []
  seen_dests = set()
  for line in lines:
    if (m := re.match(r'(D0\.\w+\[\w+\])\s*=\s*(\w+)', line)):
      lhs, rhs = m.group(1), m.group(2)
      if rhs in vars and lhs not in seen_dests:
        final_assigns.append((lhs, vars[rhs]))
        seen_dests.add(lhs)

  # 2. If no explicit final assignments, update the original D0 assigns with conditional values
  if not final_assigns:
    for dest, val in assigns:
      base = dest.split('.')[0] if '.' in dest else dest.split('[')[0]
      if base in all_vars:
        final_assigns.append((dest, vars[base]))
      else:
        final_assigns.append((dest, val))

  return vars, final_assigns if final_assigns else assigns

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

  # Ternary - but only if the ? and : are not inside brackets
  if '?' in expr:
    # Find the ? that's at depth 0
    depth_paren, depth_bracket = 0, 0
    q_pos, c_pos = -1, -1
    for i, ch in enumerate(expr):
      if ch == '(': depth_paren += 1
      elif ch == ')': depth_paren -= 1
      elif ch == '[': depth_bracket += 1
      elif ch == ']': depth_bracket -= 1
      elif ch == '?' and depth_paren == 0 and depth_bracket == 0: q_pos = i
      elif ch == ':' and depth_paren == 0 and depth_bracket == 0 and q_pos >= 0: c_pos = i; break
    if q_pos >= 0 and c_pos >= 0:
      cond_str = expr[:q_pos].strip()
      true_str = expr[q_pos+1:c_pos].strip()
      false_str = expr[c_pos+1:].strip()
      cond = parse_expr(cond_str, vars)
      if cond.dtype != dtypes.bool: cond = cond.ne(UOp.const(cond.dtype, 0))  # Convert to bool
      return cond.where(parse_expr(true_str, vars), parse_expr(false_str, vars))

  # Binary ops (low to high precedence) - search right-to-left for correct associativity
  ops = [('||', '|'), ('&&', '&'), ('|', '|'), ('^', '^'), ('&', '&'), ('>=', '>='), ('<=', '<='), ('==', '=='),
         ('!=', '!='), ('<>', '!='), ('>>', '>>'), ('<<', '<<'), ('>', '>'), ('<', '<'), ('+', '+'), ('-', '-'), ('*', '*'), ('/', '/'), ('**', '**')]
  for op, op_type in ops:
    # Pre-compute paren/bracket depth at each position (from right to left)
    depths = [0] * (len(expr) + 1)
    bdepths = [0] * (len(expr) + 1)
    for i in range(len(expr) - 1, -1, -1):
      depths[i], bdepths[i] = depths[i+1], bdepths[i+1]
      if expr[i] == ')': depths[i] += 1
      elif expr[i] == '(': depths[i] -= 1
      elif expr[i] == ']': bdepths[i] += 1
      elif expr[i] == '[': bdepths[i] -= 1
    for i in range(len(expr)-len(op), -1, -1):
      if depths[i] == 0 and bdepths[i] == 0 and expr[i:i+len(op)] == op:
        # Make sure we're not matching part of a longer operator
        if len(op) == 1 and i+1 < len(expr) and expr[i+1] in '=<>&|*': continue
        if len(op) == 1 and i > 0 and expr[i-1] in '=<>&|*': continue
        lhs, rhs = expr[:i].strip(), expr[i+len(op):].strip()
        if lhs and rhs:
          l, r = parse_expr(lhs, vars), parse_expr(rhs, vars)
          # For shifts, cast RHS (shift amount) to match LHS type for proper UOp generation
          if op_type in ('>>', '<<') and l.dtype != r.dtype: r = r.cast(l.dtype)
          if op_type in ('>=', '<=', '==', '!=', '>', '<') and l.dtype != r.dtype: r = r.cast(l.dtype)
          # For bitwise ops, ensure matching types (prefer unsigned)
          if op_type in ('|', '^', '&') and l.dtype != r.dtype:
            if l.dtype.itemsize == r.dtype.itemsize:
              # Same size, prefer unsigned
              target = dtypes.uint32 if l.dtype.itemsize == 4 else dtypes.uint64 if l.dtype.itemsize == 8 else l.dtype
              l, r = l.bitcast(target), r.bitcast(target)
            else:
              r = r.cast(l.dtype)
          if op_type == '|': return l | r
          if op_type == '^': return l ^ r
          if op_type == '&': return l & r
          if op_type == '>=': return l >= r
          if op_type == '<=': return l <= r
          if op_type == '==': return l.eq(r)
          if op_type == '!=': return l.ne(r)
          if op_type == '>>': return l >> r
          if op_type == '<<': return l << r
          if op_type == '>': return l > r
          if op_type == '<': return l < r
          if op_type == '+': return l + r
          if op_type == '-':
            # Constant fold subtraction to avoid problematic a + b * -1 pattern
            if l.op == Ops.CONST and r.op == Ops.CONST:
              return UOp.const(l.dtype, l.arg - r.arg)
            return l - r
          if op_type == '*': return l * r
          if op_type == '/': return l / r
          if op_type == '**':
            # Exponentiation: 2.0 ** x = exp2(x)
            if l.op == Ops.CONST and l.arg == 2.0:
              return UOp(Ops.EXP2, dtypes.float32, (r.cast(dtypes.float32),))
            return l  # Unsupported base

  # Type cast: 64'U(...)
  if (m := re.match(r"(\d+)'([UIFB])\((.+)\)", expr)):
    dt = {('U',32): dtypes.uint32, ('U',64): dtypes.uint64, ('I',32): dtypes.int, ('I',64): dtypes.int64,
          ('F',32): dtypes.float32, ('F',64): dtypes.float64}.get((m.group(2), int(m.group(1))), dtypes.uint32)
    return parse_expr(m.group(3), vars).cast(dt)

  # Lane-indexed: VCC.u64[laneId] or VCC.u64[laneId].u64 - must check BEFORE var.type
  if (m := re.match(r'([a-zA-Z_]\w*)\.u64\[laneId\](?:\.(\w+))?$', expr)):
    v = vars.get(m.group(1), UOp.const(dtypes.uint32, 0))
    lane = vars['laneId'].cast(dtypes.uint32) if vars['laneId'].dtype != dtypes.uint32 else vars['laneId']
    result = (v >> lane) & UOp.const(dtypes.uint32, 1)
    # If there's a type suffix like .u64, cast to that type
    if m.group(2):
      dt = DTYPES.get(m.group(2), dtypes.uint32)
      result = result.cast(dt)
    return result

  # Variable with type: S0.u32 (variable must start with a letter)
  if (m := re.match(r'([a-zA-Z_]\w*)\.(\w+)$', expr)):  # Added $ to require full match
    v, dt = vars.get(m.group(1), UOp.const(dtypes.uint32, 0)), DTYPES.get(m.group(2), dtypes.uint32)
    if dt == v.dtype: return v
    # Special handling for 16-bit types from 32-bit sources (truncate then bitcast)
    if dt.itemsize == 2 and v.dtype.itemsize == 4:
      v16 = (v & UOp.const(v.dtype, 0xFFFF)).cast(dtypes.uint16)
      return v16 if dt == dtypes.uint16 else v16.bitcast(dt)
    # Use cast for size changes, bitcast for same-size type reinterpret
    if dt.itemsize != v.dtype.itemsize: return v.cast(dt)
    return v.bitcast(dt)

  # Bit slice: S0[4:0] or S0[4 : 0].u32 (variable must start with a letter)
  if (m := re.match(r'([a-zA-Z_]\w*)\[(\d+)\s*:\s*(\d+)\](?:\.(\w+))?$', expr)):
    hi, lo = int(m.group(2)), int(m.group(3))
    return (vars.get(m.group(1), UOp.const(dtypes.uint32, 0)) >> UOp.const(dtypes.uint32, lo)) & UOp.const(dtypes.uint32, (1<<(hi-lo+1))-1)

  # Bit slice with type prefix: S1.u32[4:0].u32 (common in pcode)
  if (m := re.match(r'([a-zA-Z_]\w*)\.(\w+)\[(\d+)\s*:\s*(\d+)\](?:\.(\w+))?$', expr)):
    hi, lo = int(m.group(3)), int(m.group(4))
    v = vars.get(m.group(1), UOp.const(dtypes.uint32, 0))
    return (v >> UOp.const(dtypes.uint32, lo)) & UOp.const(dtypes.uint32, (1<<(hi-lo+1))-1)

  # Single bit access with type prefix: tmp.u32[31] or S1.u32[expr] (common in pcode for sign bit checks)
  if (m := re.match(r'([a-zA-Z_]\w*)\.(\w+)\[(.+)\]$', expr)):
    bit_expr = m.group(3)
    v = vars.get(m.group(1), UOp.const(dtypes.uint32, 0))
    dt = DTYPES.get(m.group(2), dtypes.uint32)
    if v.dtype != dt:
      if dt.itemsize != v.dtype.itemsize: v = v.cast(dt)
      else: v = v.bitcast(dt)
    # Cast to uint32 for bit manipulation
    if v.dtype != dtypes.uint32: v = v.bitcast(dtypes.uint32) if v.dtype.itemsize == 4 else v.cast(dtypes.uint32)
    # Parse the bit index (could be a number or an expression like "sign(x) ? 2 : 9")
    if bit_expr.isdigit():
      bit = UOp.const(dtypes.uint32, int(bit_expr))
    else:
      bit = parse_expr(bit_expr, vars)
      if bit.dtype != dtypes.uint32: bit = bit.cast(dtypes.uint32)
    return (v >> bit) & UOp.const(dtypes.uint32, 1)

  # Literals - check integers BEFORE floats to avoid "2" being parsed as 2.0
  if (m := re.match(r'0x([0-9a-fA-F]+)', expr)): return UOp.const(dtypes.uint64, int(m.group(1), 16))
  if (m := re.match(r"(\d+)'(\d+)U", expr)): return UOp.const(dtypes.uint32, int(m.group(2)))  # 1'1U -> 1
  if (m := re.match(r'(-?\d+)[UL]*$', expr)):
    val = int(m.group(1))
    return UOp.const(dtypes.int if val < 0 else dtypes.uint32, val)  # negative -> int, positive -> uint32
  if (m := re.match(r'-?(\d+\.\d+)F?$', expr)): return UOp.const(dtypes.float32, float(expr.rstrip('F')))  # -1.0F or 1.0F
  if (m := re.match(r'-?(\d+)F$', expr)): return UOp.const(dtypes.float32, float(expr.rstrip('F')))  # -2F or 2F

  # Unary NOT: ~S0.u32
  if expr.startswith('~'):
    inner = parse_expr(expr[1:], vars)
    return inner ^ UOp.const(inner.dtype, (1 << (inner.dtype.itemsize * 8)) - 1)

  # Unary minus: -S0.f32 or -floor(...)
  if expr.startswith('-') and len(expr) > 1 and expr[1] not in '0123456789':
    inner = parse_expr(expr[1:], vars)
    return inner.neg()

  # Variable
  if expr in vars: return vars[expr]
  # Constants
  if expr == 'PI': return UOp.const(dtypes.float32, 3.141592653589793)

  # Brace concatenation: { S0.u32, S1.u32 } -> 64-bit value with S0 as high 32 bits, S1 as low 32 bits
  if (m := re.match(r'\{\s*(.+?)\s*,\s*(.+?)\s*\}', expr)):
    high = parse_expr(m.group(1), vars).cast(dtypes.uint64)
    low = parse_expr(m.group(2), vars).cast(dtypes.uint64)
    return (high << UOp.const(dtypes.uint64, 32)) | low

  # MEM[addr].type - LDS memory read (for DS instructions)
  if (m := re.match(r'MEM\[(.+)\]\.(\w+)', expr)):
    addr_expr, dtype_str = m.group(1), m.group(2)
    addr = parse_expr(addr_expr, vars)
    if addr.dtype != dtypes.uint32: addr = addr.cast(dtypes.uint32)
    dt = DTYPES.get(dtype_str, dtypes.uint32)
    lds = vars.get('_lds')  # LDS buffer must be provided
    if lds is None: return UOp.const(dt, 0)  # No LDS buffer, return 0
    # LDS is uint32-indexed, so divide byte address by 4
    idx = (addr >> UOp.const(dtypes.uint32, 2)).cast(dtypes.index)
    val = lds.index(idx)
    # For b64/u64, read two consecutive dwords
    if dt in (dtypes.uint64, dtypes.int64):
      hi_idx = ((addr + UOp.const(dtypes.uint32, 4)) >> UOp.const(dtypes.uint32, 2)).cast(dtypes.index)
      hi = lds.index(hi_idx)
      val = val.cast(dtypes.uint64) | (hi.cast(dtypes.uint64) << UOp.const(dtypes.uint64, 32))
    return val

  # VAR[high : low] - bit slice read (for DS RETURN_DATA)
  if (m := re.match(r'(\w+)\[(\d+)\s*:\s*(\d+)\]', expr)):
    var_name, high_bit, low_bit = m.group(1), int(m.group(2)), int(m.group(3))
    if var_name in vars:
      val = vars[var_name]
      # Extract bits [high:low] - shift right by low, then mask
      width = high_bit - low_bit + 1
      shifted = val >> UOp.const(val.dtype, low_bit) if low_bit > 0 else val
      mask = UOp.const(val.dtype, (1 << width) - 1)
      return shifted & mask
    return UOp.const(dtypes.uint32, 0)

  # Functions: fma, pow, signext, sqrt, floor, trunc, log2, min, max
  if (m := re.match(r'fma\((.+),\s*(.+),\s*(.+)\)', expr)):
    return parse_expr(m.group(1), vars) * parse_expr(m.group(2), vars) + parse_expr(m.group(3), vars)
  if (m := re.match(r'pow\((.+),\s*(.+)\)', expr)) and '2.0' in m.group(1):
    return UOp(Ops.EXP2, dtypes.float32, (parse_expr(m.group(2), vars).bitcast(dtypes.float32),))
  if (m := re.match(r'signext\((.+)\)', expr)):
    # signext is a no-op for signed types - just ensure signed interpretation
    return parse_expr(m.group(1), vars)
  if (m := re.match(r'signext_from_bit\((.+),\s*(.+)\)', expr)):
    # Sign-extend from bit position: if bit (width-1) is set, fill upper bits with 1s
    val = parse_expr(m.group(1), vars)
    width = parse_expr(m.group(2), vars).cast(dtypes.uint32)
    # Cast val to uint32 for bit manipulation
    val_u = val.cast(dtypes.uint32) if val.dtype != dtypes.uint32 else val
    # sign_bit = (val >> (width - 1)) & 1
    sign_bit = (val_u >> (width - UOp.const(dtypes.uint32, 1))) & UOp.const(dtypes.uint32, 1)
    # mask = (1 << width) - 1 for the extracted bits
    mask = (UOp.const(dtypes.uint32, 1) << width) - UOp.const(dtypes.uint32, 1)
    # If sign bit is set, fill upper bits with 1s: result = val | ~mask
    sign_extend_mask = mask ^ UOp.const(dtypes.uint32, 0xFFFFFFFF)  # ~mask
    result = sign_bit.ne(UOp.const(dtypes.uint32, 0)).where(val_u | sign_extend_mask, val_u)
    # Cast back to original type if needed
    return result.bitcast(val.dtype) if val.dtype != dtypes.uint32 else result
  if (m := re.match(r'sqrt\((.+)\)', expr)):
    inner = parse_expr(m.group(1), vars)
    return UOp(Ops.SQRT, inner.dtype, (inner,))
  if (m := re.match(r'floor\((.+)\)', expr)):
    inner = parse_expr(m.group(1), vars)
    # floor(x) = trunc(x) if x >= 0 or x == trunc(x), else trunc(x) - 1
    truncated = UOp(Ops.TRUNC, inner.dtype, (inner,))
    is_negative = inner < UOp.const(inner.dtype, 0)
    is_not_whole = inner.ne(truncated)
    needs_adjust = is_negative & is_not_whole
    return needs_adjust.where(truncated - UOp.const(inner.dtype, 1), truncated)
  if (m := re.match(r'trunc\((.+)\)', expr)):
    inner = parse_expr(m.group(1), vars)
    return UOp(Ops.TRUNC, inner.dtype, (inner,))
  if (m := re.match(r'log2\((.+)\)', expr)):
    inner = parse_expr(m.group(1), vars)
    return UOp(Ops.LOG2, inner.dtype, (inner,))
  if (m := re.match(r'min\((.+),\s*(.+)\)', expr)):
    a, b = parse_expr(m.group(1), vars), parse_expr(m.group(2), vars)
    return UOp(Ops.MAX, a.dtype, (a.neg(), b.neg())).neg()  # min(a,b) = -max(-a,-b)
  if (m := re.match(r'max\((.+),\s*(.+)\)', expr)):
    a, b = parse_expr(m.group(1), vars), parse_expr(m.group(2), vars)
    return UOp(Ops.MAX, a.dtype, (a, b))
  if (m := re.match(r'sin\((.+)\)', expr)):
    inner = parse_expr(m.group(1), vars)
    return UOp(Ops.SIN, inner.dtype, (inner,))
  if (m := re.match(r'cos\((.+)\)', expr)):
    inner = parse_expr(m.group(1), vars)
    # cos(x) = sin(x + pi/2)
    return UOp(Ops.SIN, inner.dtype, (inner + UOp.const(inner.dtype, 1.5707963267948966),))
  if (m := re.match(r'fract\((.+)\)', expr)):
    inner = parse_expr(m.group(1), vars)
    # fract(x) = x - floor(x)
    truncated = UOp(Ops.TRUNC, inner.dtype, (inner,))
    is_negative = inner < UOp.const(inner.dtype, 0)
    is_not_whole = inner.ne(truncated)
    needs_adjust = is_negative & is_not_whole
    floor_val = needs_adjust.where(truncated - UOp.const(inner.dtype, 1), truncated)
    return inner - floor_val
  if (m := re.match(r'isEven\((.+)\)', expr)):
    inner = parse_expr(m.group(1), vars)
    # isEven: check if integer part is even (bit 0 = 0)
    int_val = UOp(Ops.TRUNC, inner.dtype, (inner,)).cast(dtypes.int)
    return (int_val & UOp.const(dtypes.int, 1)).eq(UOp.const(dtypes.int, 0))
  # Type conversion functions
  if (m := re.match(r'i32_to_f32\((.+)\)', expr)):
    return parse_expr(m.group(1), vars).cast(dtypes.int).cast(dtypes.float32)
  if (m := re.match(r'u32_to_f32\((.+)\)', expr)):
    return parse_expr(m.group(1), vars).cast(dtypes.uint32).cast(dtypes.float32)
  if (m := re.match(r'f32_to_i32\((.+)\)', expr)):
    inner = parse_expr(m.group(1), vars)
    return UOp(Ops.TRUNC, inner.dtype, (inner,)).cast(dtypes.int)
  if (m := re.match(r'f32_to_u32\((.+)\)', expr)):
    inner = parse_expr(m.group(1), vars)
    return UOp(Ops.TRUNC, inner.dtype, (inner,)).cast(dtypes.uint32)
  if (m := re.match(r'f16_to_f32\((.+)\)', expr)):
    inner = parse_expr(m.group(1), vars)
    # f16 is stored in low 16 bits of uint32 - extract and convert
    if inner.dtype == dtypes.uint32:
      inner = (inner & UOp.const(dtypes.uint32, 0xFFFF)).cast(dtypes.uint16).bitcast(dtypes.half)
    return inner.cast(dtypes.float32)
  if (m := re.match(r'f32_to_f16\((.+)\)', expr)):
    inner = parse_expr(m.group(1), vars)
    return inner.cast(dtypes.half)
  if (m := re.match(r'f16_to_i16\((.+)\)', expr)):
    inner = parse_expr(m.group(1), vars)
    # f16 is stored in low 16 bits of uint32 - extract and convert to int16
    if inner.dtype == dtypes.uint32:
      inner = (inner & UOp.const(dtypes.uint32, 0xFFFF)).cast(dtypes.uint16).bitcast(dtypes.half)
    return UOp(Ops.TRUNC, inner.dtype, (inner,)).cast(dtypes.int16)
  if (m := re.match(r'f16_to_u16\((.+)\)', expr)):
    inner = parse_expr(m.group(1), vars)
    # f16 is stored in low 16 bits of uint32 - extract and convert to uint16
    if inner.dtype == dtypes.uint32:
      inner = (inner & UOp.const(dtypes.uint32, 0xFFFF)).cast(dtypes.uint16).bitcast(dtypes.half)
    return UOp(Ops.TRUNC, inner.dtype, (inner,)).cast(dtypes.uint16)

  # Float classification functions for CMP_CLASS
  # Note: These work on the original float bits, NOT on f64 conversion (the pcode 64'F conversion is for precision, not bit checking)
  if (m := re.match(r'isSignalNAN\((.+)\)', expr)):
    # SignalNAN: exponent all 1s, mantissa != 0, top mantissa bit = 0
    inner = m.group(1)
    # Strip 64'F() wrapper if present - we want the original float bits
    if (m2 := re.match(r"64'F\((.+)\)", inner)): inner = m2.group(1)
    v = parse_expr(inner, vars)
    # Detect f16 vs f32 based on source type suffix
    is_f16 = '.f16' in inner or v.dtype == dtypes.half
    if is_f16:
      # f16: convert to uint16 for bit ops (vgprs store f16 in low 16 bits as uint32)
      v16 = (v & UOp.const(dtypes.uint32, 0xFFFF)) if v.dtype == dtypes.uint32 else v.bitcast(dtypes.uint16).cast(dtypes.uint32)
      exp_mask = UOp.const(dtypes.uint32, 0x7C00)  # f16 exponent mask
      mant_mask = UOp.const(dtypes.uint32, 0x03FF)  # f16 mantissa mask
      quiet_bit = UOp.const(dtypes.uint32, 0x0200)  # f16 quiet bit
      is_nan_exp = (v16 & exp_mask).eq(exp_mask)
      has_mantissa = (v16 & mant_mask).ne(UOp.const(dtypes.uint32, 0))
      is_quiet = (v16 & quiet_bit).ne(UOp.const(dtypes.uint32, 0))
    else:
      # f32: work on uint32 representation
      v32 = v.bitcast(dtypes.uint32) if v.dtype == dtypes.float32 else v
      exp_mask = UOp.const(dtypes.uint32, 0x7F800000)
      mant_mask = UOp.const(dtypes.uint32, 0x007FFFFF)
      quiet_bit = UOp.const(dtypes.uint32, 0x00400000)
      is_nan_exp = (v32 & exp_mask).eq(exp_mask)
      has_mantissa = (v32 & mant_mask).ne(UOp.const(dtypes.uint32, 0))
      is_quiet = (v32 & quiet_bit).ne(UOp.const(dtypes.uint32, 0))
    return is_nan_exp & has_mantissa & is_quiet.logical_not()

  if (m := re.match(r'isQuietNAN\((.+)\)', expr)):
    # QuietNAN: exponent all 1s, top mantissa bit = 1
    inner = m.group(1)
    # Strip 64'F() wrapper if present - we want the original float bits
    if (m2 := re.match(r"64'F\((.+)\)", inner)): inner = m2.group(1)
    v = parse_expr(inner, vars)
    # Detect f16 vs f32 based on source type suffix
    is_f16 = '.f16' in inner or v.dtype == dtypes.half
    if is_f16:
      # f16: convert to uint32 for bit ops
      v16 = (v & UOp.const(dtypes.uint32, 0xFFFF)) if v.dtype == dtypes.uint32 else v.bitcast(dtypes.uint16).cast(dtypes.uint32)
      exp_mask = UOp.const(dtypes.uint32, 0x7C00)  # f16 exponent mask
      quiet_bit = UOp.const(dtypes.uint32, 0x0200)  # f16 quiet bit
      is_nan_exp = (v16 & exp_mask).eq(exp_mask)
      is_quiet = (v16 & quiet_bit).ne(UOp.const(dtypes.uint32, 0))
    else:
      # f32: work on uint32 representation
      v32 = v.bitcast(dtypes.uint32) if v.dtype == dtypes.float32 else v
      exp_mask = UOp.const(dtypes.uint32, 0x7F800000)
      quiet_bit = UOp.const(dtypes.uint32, 0x00400000)
      is_nan_exp = (v32 & exp_mask).eq(exp_mask)
      is_quiet = (v32 & quiet_bit).ne(UOp.const(dtypes.uint32, 0))
    return is_nan_exp & is_quiet

  if (m := re.match(r'exponent\((.+)\)', expr)):
    # Extract exponent bits
    v = parse_expr(m.group(1), vars)
    if '.f16' in m.group(1) or v.dtype == dtypes.half:
      v32 = v.bitcast(dtypes.uint16) if v.dtype == dtypes.half else (v & UOp.const(dtypes.uint32, 0xFFFF)).cast(dtypes.uint16)
      return ((v32.cast(dtypes.uint32) >> UOp.const(dtypes.uint32, 10)) & UOp.const(dtypes.uint32, 0x1F))
    else:  # f32
      v32 = v.bitcast(dtypes.uint32) if v.dtype == dtypes.float32 else v
      return (v32 >> UOp.const(dtypes.uint32, 23)) & UOp.const(dtypes.uint32, 0xFF)

  if (m := re.match(r'sign\((.+)\)', expr)):
    # Extract sign bit (returns 1 for negative, 0 for positive)
    v = parse_expr(m.group(1), vars)
    if '.f16' in m.group(1) or v.dtype == dtypes.half:
      v32 = v.bitcast(dtypes.uint16) if v.dtype == dtypes.half else (v & UOp.const(dtypes.uint32, 0xFFFF)).cast(dtypes.uint16)
      return (v32.cast(dtypes.uint32) >> UOp.const(dtypes.uint32, 15)) & UOp.const(dtypes.uint32, 1)
    else:  # f32
      v32 = v.bitcast(dtypes.uint32) if v.dtype == dtypes.float32 else v
      return (v32 >> UOp.const(dtypes.uint32, 31)) & UOp.const(dtypes.uint32, 1)

  if (m := re.match(r'abs\((.+)\)', expr)):
    # Absolute value - clear sign bit
    v = parse_expr(m.group(1), vars)
    if v.dtype == dtypes.float64:
      return (v.cast(dtypes.uint64) & UOp.const(dtypes.uint64, 0x7FFFFFFFFFFFFFFF)).bitcast(dtypes.float64)
    elif v.dtype == dtypes.float32:
      return (v.bitcast(dtypes.uint32) & UOp.const(dtypes.uint32, 0x7FFFFFFF)).bitcast(dtypes.float32)
    elif v.dtype == dtypes.half:
      return (v.bitcast(dtypes.uint16) & UOp.const(dtypes.uint16, 0x7FFF)).bitcast(dtypes.half)
    return v  # For int types, would need different handling

  return UOp.const(dtypes.uint32, 0)
