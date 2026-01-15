# AMD pcode parser - converts pseudocode to UOps
import re
from tinygrad.dtype import dtypes, DType
from tinygrad.uop.ops import Ops, UOp

DTYPES = {'u32': dtypes.uint32, 'i32': dtypes.int, 'f32': dtypes.float32, 'b32': dtypes.uint32,
          'u64': dtypes.uint64, 'i64': dtypes.int64, 'f64': dtypes.float64, 'b64': dtypes.uint64,
          'u16': dtypes.uint16, 'i16': dtypes.short, 'f16': dtypes.half, 'b16': dtypes.uint16}

def parse_pcode(pcode: str, srcs: dict[str, UOp] | None = None, lane: UOp | None = None) -> tuple[dict[str, UOp], list[tuple[str, UOp]]]:
  """Parse pcode into UOps. srcs can provide actual UOps for S0, S1, S2, D0, VCC, EXEC, SCC."""
  vars: dict[str, UOp] = {n: UOp(Ops.DEFINE_VAR, dtypes.uint32, (), (n, UOp.const(dtypes.uint32, 0), UOp.const(dtypes.uint32, 0xFFFFFFFF)))
                          for n in ['S0', 'S1', 'S2', 'D0', 'VCC', 'EXEC', 'SCC']}
  if srcs: vars.update(srcs)
  vars['laneId'] = lane if lane is not None else UOp.const(dtypes.uint32, 0)
  assigns: list[tuple[str, UOp]] = []

  for stmt in re.split(r'[;\n]', pcode):
    stmt = stmt.strip()
    if not stmt or stmt.startswith('//'): continue
    if (m := re.match(r'(\w+(?:\.\w+)?(?:\[\w+\])?)\s*=\s*(.+)', stmt)) and not re.search(r'[<>=!]=', stmt[:stmt.find('=')]):
      lhs, val = m.group(1), parse_expr(m.group(2), vars)
      base = re.match(r'(\w+)', lhs).group(1)
      if base in ['D0', 'SCC', 'VCC', 'EXEC']: assigns.append((lhs, val))
      vars[base] = val
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
  if '?' in expr and (m := re.match(r'(.+)\s*\?\s*(.+)\s*:\s*(.+)', expr)):
    cond = parse_expr(m.group(1), vars)
    if cond.dtype != dtypes.bool: cond = cond.ne(UOp.const(cond.dtype, 0))  # Convert to bool
    return cond.where(parse_expr(m.group(2), vars), parse_expr(m.group(3), vars))

  # Binary ops (low to high precedence) - search right-to-left for correct associativity
  ops = [('||', '|'), ('&&', '&'), ('|', '|'), ('^', '^'), ('&', '&'), ('>=', '>='), ('<=', '<='), ('==', '=='),
         ('!=', '!='), ('<>', '!='), ('>>', '>>'), ('<<', '<<'), ('>', '>'), ('<', '<'), ('+', '+'), ('-', '-'), ('*', '*'), ('**', '**')]
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
          # For shifts and comparisons, cast RHS to match LHS type
          if op_type in ('>>', '<<', '>=', '<=', '==', '!=', '>', '<') and l.dtype != r.dtype: r = r.cast(l.dtype)
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
          if op_type == '-': return l - r
          if op_type == '*': return l * r
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

  # Lane-indexed: VCC.u64[laneId] - must check BEFORE var.type
  if (m := re.match(r'([a-zA-Z_]\w*)\.u64\[laneId\]', expr)):
    v = vars.get(m.group(1), UOp.const(dtypes.uint32, 0))
    lane = vars['laneId'].cast(dtypes.uint32) if vars['laneId'].dtype != dtypes.uint32 else vars['laneId']
    return (v >> lane) & UOp.const(dtypes.uint32, 1)

  # Variable with type: S0.u32 (variable must start with a letter)
  if (m := re.match(r'([a-zA-Z_]\w*)\.(\w+)$', expr)):  # Added $ to require full match
    v, dt = vars.get(m.group(1), UOp.const(dtypes.uint32, 0)), DTYPES.get(m.group(2), dtypes.uint32)
    if dt == v.dtype: return v
    # Use cast for size changes, bitcast for same-size type reinterpret
    if dt.itemsize != v.dtype.itemsize: return v.cast(dt)
    return v.bitcast(dt)

  # Bit slice: S0[4:0] or S0[4 : 0].u32 (variable must start with a letter)
  if (m := re.match(r'([a-zA-Z_]\w*)\[(\d+)\s*:\s*(\d+)\](?:\.(\w+))?$', expr)):  # Added $
    hi, lo = int(m.group(2)), int(m.group(3))
    return (vars.get(m.group(1), UOp.const(dtypes.uint32, 0)) >> UOp.const(dtypes.uint32, lo)) & UOp.const(dtypes.uint32, (1<<(hi-lo+1))-1)

  # Literals
  if (m := re.match(r'0x([0-9a-fA-F]+)', expr)): return UOp.const(dtypes.uint64, int(m.group(1), 16))
  if (m := re.match(r"(\d+)'(\d+)U", expr)): return UOp.const(dtypes.uint32, int(m.group(2)))  # 1'1U -> 1
  if (m := re.match(r'(\d+\.?\d*)F?$', expr)): return UOp.const(dtypes.float32, float(m.group(1)))  # 2.0F or 2.0
  if (m := re.match(r'(\d+)[UL]*$', expr)): return UOp.const(dtypes.uint32, int(m.group(1)))

  # Variable
  if expr in vars: return vars[expr]

  # Functions: fma, pow, signext
  if (m := re.match(r'fma\((.+),\s*(.+),\s*(.+)\)', expr)):
    return parse_expr(m.group(1), vars) * parse_expr(m.group(2), vars) + parse_expr(m.group(3), vars)
  if (m := re.match(r'pow\((.+),\s*(.+)\)', expr)) and '2.0' in m.group(1):
    return UOp(Ops.EXP2, dtypes.float32, (parse_expr(m.group(2), vars).bitcast(dtypes.float32),))
  if (m := re.match(r'signext\((.+)\)', expr)):
    # signext is a no-op for signed types - just ensure signed interpretation
    return parse_expr(m.group(1), vars)

  return UOp.const(dtypes.uint32, 0)

if __name__ == "__main__":
  from extra.assembly.amd.autogen.rdna3.str_pcode import PCODE
  from extra.assembly.amd.autogen.rdna3.enum import VOP2Op, SOP2Op, VOP3Op

  # Test basic parsing
  for name, op in [("V_ADD_F32", VOP2Op.V_ADD_F32_E32), ("V_LSHLREV_B32", VOP2Op.V_LSHLREV_B32_E32),
                   ("S_CSELECT_B32", SOP2Op.S_CSELECT_B32), ("V_ADD_CO_CI_U32", VOP2Op.V_ADD_CO_CI_U32_E32)]:
    print(f"\n{name}: {PCODE[op]}")
    for dest, val in parse_pcode(PCODE[op])[1]: print(f"  {dest} = {val}")

  # Test with actual source operands
  print("\n\nWith actual sources:")
  s0 = UOp.const(dtypes.uint32, 0x3f800000)  # 1.0f
  s1 = UOp.const(dtypes.uint32, 0x40000000)  # 2.0f
  _, assigns = parse_pcode(PCODE[VOP2Op.V_ADD_F32_E32], {'S0': s0, 'S1': s1})
  for dest, val in assigns: print(f"  {dest} = {val}")
