#!/usr/bin/env python3
"""Compile pseudocode from pseudocode_data.py to Python functions in pseudocode_functions.py."""
import re
from pathlib import Path

# Unsupported patterns that require interpreter fallback
UNSUPPORTED = ['SGPR[',
               'V_SWAP', 'eval ', 'BYTE_PERMUTE', 'FATAL_HALT', 'HW_REGISTERS',
               # Special registers (PC used)
               'PC =', 'PC=', 'PC+', '= PC',
               # Constants/functions not worth compiling
               'v_sad',
               # Complex bit manipulation patterns
               '+:',
               # Wait counter instructions with incomplete pseudocode
               'vscnt', 'vmcnt', 'expcnt', 'lgkmcnt',
               # Complex indexing patterns
               # Missing helper functions
               'CVT_OFF_TABLE',
               # bf16 and other unsupported types/functions
               '.bf16', 'ThreadMask', 'u8_to_u32', 'u4_to_u32',
               # Complex patterns with dynamic indexing or bare S0[...] bit ranges
               'S1[i', 'C.i32', 'v_msad_u8', 'S[i]', 'in[',

               # Complex trig preop with large constant
               '2.0 / PI']

def expr_to_python(expr: str) -> str:
  """Convert a pseudocode expression to Python."""
  e = expr.strip()
  # Handle DENORM comparisons FIRST before other transformations
  # Pattern: expr == DENORM.f32 or expr == DENORM.f64 means "is expr a denormal?"
  # Find balanced expressions before == DENORM by tracking parens
  def replace_denorm(e, ftype):
    marker = f'== DENORM.{ftype}'
    while marker in e:
      idx = e.find(marker)
      # Walk backwards to find the start of the expression (handle parens, operators)
      depth = 0
      start = idx - 1
      while start >= 0 and e[start] in ' \t': start -= 1  # skip whitespace
      # Find the start of the expression - stop at && || or start of string or opening (
      while start >= 0:
        c = e[start]
        if c == ')': depth += 1
        elif c == '(':
          if depth == 0: break
          depth -= 1
        elif c in '&|' and depth == 0:
          if start > 0 and e[start-1] == c: start += 1; break  # && or ||
        start -= 1
      start = max(0, start + 1) if start < idx else 0
      expr_part = e[start:idx].strip()
      if expr_part.startswith('(') and expr_part.endswith(')'):
        expr_part = expr_part[1:-1]  # strip outer parens if balanced
      # Convert the expression recursively (but mark DENORM as processed)
      expr_py = expr_to_python(expr_part.replace('DENORM', '_DENORM_MARKER_'))
      e = e[:start] + f'_is_denorm_{ftype}({expr_py})' + e[idx+len(marker):]
    return e
  e = replace_denorm(e, 'f32')
  e = replace_denorm(e, 'f64')
  # Handle bit concatenation: { S0.u32, S1.u32 } -> ((s0 << 32) | s1) (high in upper bits, low in lower)
  e = re.sub(r'\{\s*S(\d)\.(u|i|b)32\s*,\s*S(\d)\.(u|i|b)32\s*\}', lambda m: f'(((s{m.group(1)}&0xffffffff)<<32)|(s{m.group(3)}&0xffffffff))', e)
  # Handle 8-bit packing: { high_8bit, low_8bit } -> ((high & 0xff) << 8) | (low & 0xff)
  def convert_8bit_pack(m):
    high, low = m.group(1).strip(), m.group(2).strip()
    high_py = expr_to_python(high)
    low_py = expr_to_python(low)
    return f'((({high_py})&0xff)<<8)|(({low_py})&0xff)'
  e = re.sub(r'\{\s*([^,{}]+)\s*,\s*([^,{}]+)\s*\}', convert_8bit_pack, e)
  # Handle bit range extraction: S1.u32[4 : 0].u32 -> ((s1 >> 0) & 0x1f)
  # Pattern: SN.type[high : low].type (extracts bits high:low)
  e = re.sub(r'S(\d)\.(u|i|b)(32|64)\[(\d+)\s*:\s*(\d+)\]\.u\d+',
             lambda m: f'((s{m.group(1)} >> {m.group(5)}) & ((1 << ({m.group(4)} - {m.group(5)} + 1)) - 1))', e)
  # Handle bit range extraction without trailing type: S0.u32[31 : 24] -> ((s0 >> 24) & 0xff)
  e = re.sub(r'S(\d)\.(u|i|b)(32|64)\[(\d+)\s*:\s*(\d+)\](?!\.)',
             lambda m: f'((s{m.group(1)} >> {m.group(5)}) & ((1 << ({m.group(4)} - {m.group(5)} + 1)) - 1))', e)
  # Handle S0[31 : 16].f16 format - extract bits and convert to f16
  e = re.sub(r'S(\d)\[(\d+)\s*:\s*(\d+)\]\.f16',
             lambda m: f'_f16((s{m.group(1)} >> {m.group(3)}) & ((1 << ({m.group(2)} - {m.group(3)} + 1)) - 1))', e)
  # Handle S0[31 : 16].i16 format - extract bits and sign extend
  e = re.sub(r'S(\d)\[(\d+)\s*:\s*(\d+)\]\.i16',
             lambda m: f'_sext((s{m.group(1)} >> {m.group(3)}) & ((1 << ({m.group(2)} - {m.group(3)} + 1)) - 1), 16)', e)
  # Also handle S0[4 : 0].u32 format (without the first type)
  e = re.sub(r'S(\d)\[(\d+)\s*:\s*(\d+)\]\.u\d+',
             lambda m: f'((s{m.group(1)} >> {m.group(3)}) & ((1 << ({m.group(2)} - {m.group(3)} + 1)) - 1))', e)
  # Handle single bit extraction: .u32[bit], .u64[bit], .i32[bit], etc -> ((val >> bit) & 1)
  # Also handle dynamic bit access like S0.u32[i] or S0.u32[31 - i]
  e = re.sub(r'S(\d)\.(u|i|b)(32|64)\[(\d+)\s*-\s*(\w+)\]', lambda m: f'((s{m.group(1)} >> ({m.group(4)} - {m.group(5)})) & 1)', e)
  # Handle S0[i].u32 - dynamic bit access with trailing type (used in loops)
  e = re.sub(r'S(\d)\[(\w+)\]\.(u|i|b)\d+', lambda m: f'((s{m.group(1)} >> {m.group(2)}) & 1)', e)
  # Handle nested bit range in index like S0.u32[S1.u32[4 : 0]] - convert inner range first
  e = re.sub(r'S(\d)\.(u|i|b)(32|64)\[S(\d)\.(u|i|b)(32|64)\[(\d+)\s*:\s*(\d+)\]\]',
             lambda m: f'((s{m.group(1)} >> ((s{m.group(4)} >> {m.group(8)}) & ((1 << ({m.group(7)} - {m.group(8)} + 1)) - 1))) & 1)', e)
  # Handle complex expressions in brackets like S1.u32[sign(S0.f16) ? 2 : 9]
  def convert_complex_bit_access(m):
    src, typ, bits, inner = m.group(1), m.group(2), m.group(3), m.group(4)
    # Skip if it looks like a bit range - pattern is just "high : low" at the start
    if re.match(r'^\d+\s*:\s*\d+$', inner.strip()):
      return m.group(0)  # Return unchanged, let bit range regex handle it
    # Check if it looks like a simple variable or number - if not, recursively convert
    if re.match(r'^[\w\d]+$', inner.strip()):
      return f'((s{src} >> {inner}) & 1)'
    inner_py = expr_to_python(inner)
    return f'((s{src} >> ({inner_py})) & 1)'
  e = re.sub(r'S(\d)\.(u|i|b)(32|64)\[([^\]]+)\]', convert_complex_bit_access, e)
  e = re.sub(r'S(\d)\.(u|i|b)(32|64)\[(\d+)\]', lambda m: f'((s{m.group(1)} >> {m.group(4)}) & 1)', e)
  e = re.sub(r'D0\.(u|i|b)(32|64)\[(\d+)\]', lambda m: f'((_d0 >> {m.group(3)}) & 1)', e)
  # Handle D0[31 : 16].f16 - extract bits and convert to f16
  e = re.sub(r'D0\[(\d+)\s*:\s*(\d+)\]\.f16',
             lambda m: f'_f16((_d0 >> {m.group(2)}) & ((1 << ({m.group(1)} - {m.group(2)} + 1)) - 1))', e)
  # Handle SIMM16.i16[bit] - extract bit from sign-extended immediate
  e = re.sub(r'SIMM16\.i16\[(\d+)\]', lambda m: f'((literal >> {m.group(1)}) & 1)', e)
  # Handle SIMM16.u16[high : low] - extract bit range from immediate
  e = re.sub(r'SIMM16\.u16\[(\d+)\s*:\s*(\d+)\](?:\.u\d+)?',
             lambda m: f'((literal >> {m.group(2)}) & ((1 << ({m.group(1)} - {m.group(2)} + 1)) - 1))', e)
  # Handle tmp.type[bit] for temporary variables
  e = re.sub(r'tmp\.(u|i|b)(32|64)\[(\d+)\]', lambda m: f'((_tmp >> {m.group(3)}) & 1)', e)
  # Handle bare tmp[bit] (after we've converted tmp to _tmp, so match _tmp[bit] or tmp[bit])
  e = re.sub(r'_tmp\[(\d+)\]', lambda m: f'((_tmp >> {m.group(1)}) & 1)', e)
  e = re.sub(r'\btmp\[(\d+)\]', lambda m: f'((_tmp >> {m.group(1)}) & 1)', e)
  # Handle bare tmp variable first (before type conversions so we match tmp but not tmp.u32 etc)
  # Match tmp followed by word boundary but NOT by a dot (to avoid tmp.u32)
  e = re.sub(r'\btmp\b(?!\.)', '_tmp', e)
  # Handle sign_out and result variables
  e = re.sub(r'\bsign_out\b', '_vars.get("sign_out",0)', e)
  e = re.sub(r'\bresult\b(?!\.)', '_vars.get("result",0)', e)
  for src, py in [
    # 64-bit types
    ('S0.f64', '_f64(s0)'), ('S1.f64', '_f64(s1)'), ('S2.f64', '_f64(s2)'),
    ('S0.u64', 's0'), ('S1.u64', 's1'), ('S2.u64', 's2'),
    ('S0.i64', '_sext(s0,64)'), ('S1.i64', '_sext(s1,64)'), ('S2.i64', '_sext(s2,64)'),
    ('S0.b64', 's0'), ('S1.b64', 's1'), ('S2.b64', 's2'),
    ('D0.f64', '_f64(_d0)'), ('D0.u64', '_d0'), ('D0.i64', '_sext(_d0,64)'), ('D0.b64', '_d0'),
    # tmp (temporary variable) - note: bare tmp already converted to _tmp above
    ('tmp.u64', '_tmp'), ('tmp.i64', '_sext(_tmp,64)'), ('tmp.b64', '_tmp'),
    ('tmp.u32', '(_tmp&0xffffffff)'), ('tmp.i32', '_sext(_tmp&0xffffffff,32)'), ('tmp.b32', '(_tmp&0xffffffff)'),
    # SCC type conversions
    ('SCC.u64', '_scc'), ('SCC.u32', '_scc'),
    # EXEC type conversions
    ('EXEC.u64', '_exec'), ('EXEC.u32', '(_exec&0xffffffff)'),
    # saveexec variable
    ('saveexec.u64', '_saveexec'), ('saveexec.u32', '(_saveexec&0xffffffff)'),
    # SIMM16 immediate
    ('SIMM16.i16', '_sext(literal&0xffff,16)'), ('SIMM16.u16', '(literal&0xffff)'),
    # SIMM32 immediate (for VOP3 with 32-bit literal)
    ('SIMM32.f16', '_f16(literal&0xffff)'), ('SIMM32.f32', '_f32(literal)'),
    ('SIMM32.u32', '(literal&0xffffffff)'), ('SIMM32.i32', '_sext(literal&0xffffffff,32)'),
    # VCC lane access
    ('VCC.u64[laneId].u64', '((vcc>>lane)&1)'), ('VCC.u64[laneId].u32', '((vcc>>lane)&1)'),
    ('VCC.u64[laneId]', '((vcc>>lane)&1)'),
    # 32-bit types
    ('S0.f32', '_f32(s0)'), ('S1.f32', '_f32(s1)'), ('S2.f32', '_f32(s2)'),
    ('S0.u32', '(s0&0xffffffff)'), ('S1.u32', '(s1&0xffffffff)'), ('S2.u32', '(s2&0xffffffff)'),
    ('S0.i32', '_sext(s0&0xffffffff,32)'), ('S1.i32', '_sext(s1&0xffffffff,32)'), ('S2.i32', '_sext(s2&0xffffffff,32)'),
    ('S0.b32', '(s0&0xffffffff)'), ('S1.b32', '(s1&0xffffffff)'), ('S2.b32', '(s2&0xffffffff)'),
    ('D0.f32', '_f32(_d0)'), ('D0.u32', '(_d0&0xffffffff)'), ('D0.i32', '_sext(_d0&0xffffffff,32)'), ('D0.b32', '(_d0&0xffffffff)'),
    # 24-bit types
    ('S0.u24', '(s0&0xffffff)'), ('S1.u24', '(s1&0xffffff)'),
    ('S0.i24', '_sext(s0&0xffffff,24)'), ('S1.i24', '_sext(s1&0xffffff,24)'),
    # 16-bit types
    ('S0.u16', '(s0&0xffff)'), ('S1.u16', '(s1&0xffff)'), ('S2.u16', '(s2&0xffff)'),
    ('S0.i16', '_sext(s0&0xffff,16)'), ('S1.i16', '_sext(s1&0xffff,16)'), ('S2.i16', '_sext(s2&0xffff,16)'),
    ('S0.f16', '_f16(s0&0xffff)'), ('S1.f16', '_f16(s1&0xffff)'), ('S2.f16', '_f16(s2&0xffff)'),
    ('D0.f16', '_f16(_d0&0xffff)'),
    ('S0.b16', '(s0&0xffff)'), ('S1.b16', '(s1&0xffff)'), ('S2.b16', '(s2&0xffff)'),
    ('D0.b16', '(d0&0xffff)'),
    # 8-bit types for signext
    ('S0.i8', '_sext(s0&0xff,8)'),
  ]: e = e.replace(src, py)
  e = e.replace('WAVE_MODE.IEEE', 'False')
  e = e.replace('ROUND_MODE', '0')  # ROUND_MODE saved to unused variable
  # VGPR access: VGPR[lane][SRC0.u32] -> vgprs[_vars.get("lane",0)][src0_idx]
  e = re.sub(r'VGPR\[(\w+)\]\[SRC0\.u32\]', r'vgprs[_vars.get("\1",0)][_src0_idx]', e)
  e = re.sub(r'VGPR\[(\w+)\]\[VDST\.u32\]', r'vgprs[_vars.get("\1",0)][_vdst_idx]', e)
  # SRC0/VDST register index references
  e = e.replace('SRC0.u32', '_src0_idx')
  e = e.replace('VDST.u32', '_vdst_idx')
  # WAVE32/WAVE64 constants - we use WAVE32
  e = e.replace('WAVE32', 'True')
  e = e.replace('WAVE64', 'False')
  # EXEC_LO for wave32 exec mask
  e = e.replace('EXEC_LO.i32', '(_exec&0xffffffff)')
  e = e.replace('EXEC_LO', '(_exec&0xffffffff)')
  # s_ff1 - find first one (count trailing zeros)
  e = e.replace('s_ff1_i32_b64(EXEC)', '_ctz64(_exec)')
  e = e.replace('s_ff1_i32_b32(', '_ctz32(')
  # Handle INF and NAN with optional type suffixes
  e = re.sub(r'\+INF\.f\d+', 'float("inf")', e)
  e = re.sub(r'-INF\.f\d+', 'float("-inf")', e)
  e = e.replace('+INF', 'float("inf")').replace('-INF', 'float("-inf")')
  e = re.sub(r'NAN\.f\d+', 'float("nan")', e)
  # MAX_FLOAT constants - max finite float values
  e = e.replace('MAX_FLOAT_F32', '3.4028235e+38').replace('MAX_FLOAT_F16', '65504.0').replace('MAX_FLOAT_F64', '1.7976931348623157e+308')
  e = e.replace('-MAX_FLOAT_F32', '-3.4028235e+38').replace('-MAX_FLOAT_F16', '-65504.0').replace('-MAX_FLOAT_F64', '-1.7976931348623157e+308')
  # OVERFLOW/UNDERFLOW constants - these are the max/min representable values (same as MAX_FLOAT/min denorm)
  e = e.replace('OVERFLOW_F32', '3.4028235e+38').replace('OVERFLOW_F64', '1.7976931348623157e+308')
  e = e.replace('UNDERFLOW_F32', '1.4e-45').replace('UNDERFLOW_F64', '5e-324')  # smallest positive denormals
  e = re.sub(r"\d+'[FIBUH]\(", "(", e)
  e = e.replace('cvtToQuietNAN(', '(')
  e = re.sub(r'\bSCC\b', '_scc', e)
  for old, new in [('isSignalNAN', '_isnan'), ('isQuietNAN', '_isnan'), ('isNAN', '_isnan'),
                   ('GT_NEG_ZERO', '_gt_neg_zero'), ('LT_NEG_ZERO', '_lt_neg_zero'), ('fma', '_fma'),
                   ('signext', '_signext'), ('exponent', '_exponent'),
                   ('ldexp', '_ldexp')]:
    e = e.replace(old + '(', new + '(')
  # sign() function - but not signext
  e = re.sub(r'\bsign\(', '_sign(', e)
  while "'" in e:
    idx = e.find("'")
    if idx > 0 and e[idx-1].isdigit():
      start = idx - 1
      while start > 0 and e[start-1].isdigit(): start -= 1
      if idx + 1 < len(e) and e[idx+1] in 'FIBु' and idx + 2 < len(e) and e[idx+2] == '(':
        e = e[:start] + e[idx+3:]
        continue
    break
  # Strip numeric suffixes: 0U, 0ULL, 123LL, 1.0F
  e = re.sub(r'([\da-fA-Fx])ULL\b', r'\1', e)
  e = re.sub(r'([\da-fA-Fx])LL\b', r'\1', e)
  e = re.sub(r'(\d)U\b', r'\1', e)
  e = re.sub(r'(\d\.?\d*)F\b', r'\1', e)
  # Handle sized literals like 16'0.0 (16-bit float literal), 32'1 (32-bit int literal)
  e = re.sub(r"\d+'(-?\d+\.?\d*)", r'\1', e)
  # Convert C-style logical and comparison operators
  e = e.replace('&&', ' and ')
  e = e.replace('||', ' or ')
  e = e.replace('<>', ' != ')  # Pascal/C style not-equal
  e = re.sub(r'!([^=])', r' not \1', e)  # ! but not !=
  # Convert C-style ternary operator: cond ? true : false -> (true if cond else false)
  # Simple approach: find ? and : at depth 0, split into cond/true/false
  def convert_ternary(e):
    while '?' in e:
      # Find the ? at depth 0
      depth, q_pos = 0, -1
      for i, c in enumerate(e):
        if c == '(': depth += 1
        elif c == ')': depth -= 1
        elif c == '?' and depth == 0:
          q_pos = i
          break
      if q_pos < 0: break
      # Find the matching : at depth 0
      depth, c_pos = 0, -1
      for i in range(q_pos + 1, len(e)):
        c = e[i]
        if c == '(': depth += 1
        elif c == ')': depth -= 1
        elif c == ':' and depth == 0:
          c_pos = i
          break
      if c_pos < 0: break
      # Everything before ? is cond, between ? and : is true, after : is false
      cond = e[:q_pos].strip()
      true_val = e[q_pos+1:c_pos].strip()
      false_val = e[c_pos+1:].strip()
      e = f'(({true_val}) if ({cond}) else ({false_val}))'
    return e
  e = convert_ternary(e)
  # Use safe division for floating point division to handle division by zero (returns inf/-inf)
  # Pattern: number_or_expr / _f32(...) or _f64(...) or _f16(...)
  e = re.sub(r'(\d+\.?\d*|_f(?:16|32|64)\([^)]+\))\s*/\s*(_f(?:16|32|64)\([^)]+\))', r'_div(\1, \2)', e)
  return e

def compile_pseudocode(pseudocode: str) -> tuple[str, bool] | None:
  """Compile pseudocode to Python function body. Returns (code, is_64bit) or None if unsupported."""
  if any(p in pseudocode for p in UNSUPPORTED): return None

  # Strip malformed SAT8 pseudocode prefix (empty if/elsif/else with stray );)
  if pseudocode.startswith('if n.i32 <='):
    # Find the actual code after "endif);"
    if 'endif);' in pseudocode:
      pseudocode = pseudocode.split('endif);')[1].strip()

  # Join multi-line conditions (lines ending with || or && before 'then')
  raw_lines = pseudocode.split('\n')
  lines = []
  i = 0
  while i < len(raw_lines):
    line = raw_lines[i].strip()
    # Join lines that are part of a multi-line if condition
    while line.endswith('||') or line.endswith('&&'):
      i += 1
      if i < len(raw_lines):
        line = line + ' ' + raw_lines[i].strip()
    lines.append(line)
    i += 1

  # Check if we need VGPR access or special indices
  needs_vgpr = 'VGPR[' in pseudocode
  needs_src0_idx = 'SRC0' in pseudocode
  needs_vdst_idx = 'VDST' in pseudocode

  py_lines = ['def _fn(s0,s1,s2,d0,scc,vcc,lane,exec_mask,literal,vgprs,_vars,src0_idx=0,vdst_idx=0):']
  py_lines.append('  _d0,_scc,_vcc_lane,_exec_lane,_tmp,_exec,_saveexec,_d1=d0,scc,None,None,0,exec_mask,exec_mask,None')
  if needs_src0_idx: py_lines.append('  _src0_idx=src0_idx')
  if needs_vdst_idx: py_lines.append('  _vdst_idx=vdst_idx')
  indent = 1
  has_vcc_lane, has_exec_lane, has_d0_64, has_exec_mod, has_d1, has_vgpr_write = False, False, False, False, False, False

  last_was_block_start = False  # Track if we need to add 'pass' for empty blocks

  for line in lines:
    line = line.strip()
    if not line or line.startswith('//'): continue

    # Check if we're about to close a block that was just opened (empty block)
    needs_pass = last_was_block_start and line in ('else', 'endif', 'endfor') or \
                 (last_was_block_start and line.startswith('elsif '))
    if needs_pass:
      py_lines.append('  ' * indent + 'pass')

    last_was_block_start = False

    if line.startswith('if '):
      cond = expr_to_python(line[3:].rstrip(' then'))
      py_lines.append('  ' * indent + f'if {cond}:')
      indent += 1
      last_was_block_start = True
    elif line.startswith('elsif '):
      indent -= 1
      cond = expr_to_python(line[6:].rstrip(' then'))
      py_lines.append('  ' * indent + f'elif {cond}:')
      indent += 1
      last_was_block_start = True
    elif line == 'else':
      indent -= 1
      py_lines.append('  ' * indent + 'else:')
      indent += 1
      last_was_block_start = True
    elif line.startswith('endif'):
      indent -= 1
    elif line.startswith('endfor'):
      indent -= 1
    elif line.startswith('declare '):
      pass
    elif line.startswith('for '):
      # Parse: for i in 0 : 31 do
      match = re.match(r'for (\w+) in (\d+)\s*:\s*(\d+) do', line)
      if match:
        var, start, end = match.groups()
        py_lines.append('  ' * indent + f'for {var} in range({start}, {int(end)+1}):')
        indent += 1
    elif '=' in line and not line.startswith('=='):
      line = line.rstrip(';')
      for op in ['+=', '-=', '*=', '/=', '|=', '&=', '^=']:
        if op in line:
          lhs, rhs = line.split(op, 1)
          lhs, rhs = lhs.strip(), rhs.strip()
          rhs_py = expr_to_python(rhs)
          if lhs.startswith('D0.'):
            if 'f32' in lhs: py_lines.append('  ' * indent + f'_d0=_i32(_f32(_d0){op[0]}{rhs_py})')
            elif 'f16' in lhs: py_lines.append('  ' * indent + f'_d0=_i16(_f16(_d0){op[0]}{rhs_py})&0xffff')
            elif '64' in lhs: py_lines.append('  ' * indent + f'_d0=int(_d0{op[0]}{rhs_py})&0xffffffffffffffff')
            else: py_lines.append('  ' * indent + f'_d0=int(_d0{op[0]}{rhs_py})&0xffffffff')
          elif lhs == 'SCC':
            py_lines.append('  ' * indent + f'_scc=int(bool(_scc{op[0]}{rhs_py}))')
          else:
            py_lines.append('  ' * indent + f'_vars["{lhs}"]=_vars.get("{lhs}",0){op[0]}{rhs_py}')
          break
      else:
        lhs, rhs = line.split('=', 1)
        lhs, rhs = lhs.strip(), rhs.strip()
        rhs_py = expr_to_python(rhs)
        if lhs == 'D0.u64[laneId]' or lhs == 'VCC.u64[laneId]':
          py_lines.append('  ' * indent + f'_vcc_lane=int(bool({rhs_py}))')
          has_vcc_lane = True
        elif lhs == 'EXEC.u64[laneId]':
          py_lines.append('  ' * indent + f'_exec_lane=int(bool({rhs_py}))')
          has_exec_lane = True
        # Handle bit reversal: D0.u32[31 : 0] = S0.u32[0 : 31]
        elif (m32 := re.match(r'D0\.(u|b)32\[31\s*:\s*0\]', lhs)) and re.match(r'S(\d)\.(u|b)32\[0\s*:\s*31\]', rhs):
          src_match = re.match(r'S(\d)', rhs)
          if src_match: py_lines.append('  ' * indent + f'_d0=_brev32(s{src_match.group(1)})')
        elif (m64 := re.match(r'D0\.(u|b)64\[63\s*:\s*0\]', lhs)) and re.match(r'S(\d)\.(u|b)64\[0\s*:\s*63\]', rhs):
          src_match = re.match(r'S(\d)', rhs)
          if src_match: py_lines.append('  ' * indent + f'_d0=_brev64(s{src_match.group(1)})'); has_d0_64 = True
        # Handle { D1.u1, D0.u64 } = expr (65-bit result with carry)
        elif (d1d0_match := re.match(r'\{\s*D1\.(u|i)1\s*,\s*D0\.(u|i)64\s*\}', lhs)):
          # The result is 65 bits: D1 gets bit 64 (carry), D0 gets bits 0-63
          py_lines.append('  ' * indent + f'_full={rhs_py}')
          py_lines.append('  ' * indent + f'_d0=int(_full)&0xffffffffffffffff')
          py_lines.append('  ' * indent + f'_d1=(int(_full)>>64)&1')
          has_d0_64, has_d1 = True, True
        # Handle single-bit assignment to D0: D0.u32[bit_expr] = value or D0.u64[bit_expr] = value
        # Need to match balanced brackets for nested expressions like D0.u32[S0.u32[4 : 0]]
        elif (d0_bit_prefix := re.match(r'D0\.(u|b)(32|64)\[', lhs)):
          typ, bits = d0_bit_prefix.groups()
          # Extract content between outermost brackets (handle nested brackets)
          bracket_start = d0_bit_prefix.end() - 1  # position of '['
          depth, i = 1, bracket_start + 1
          while i < len(lhs) and depth > 0:
            if lhs[i] == '[': depth += 1
            elif lhs[i] == ']': depth -= 1
            i += 1
          bit_expr = lhs[bracket_start + 1:i - 1]
          # Check if it's a bit range (contains ':' at depth 0) vs single bit index
          is_range = False
          depth = 0
          for c in bit_expr:
            if c == '[': depth += 1
            elif c == ']': depth -= 1
            elif c == ':' and depth == 0: is_range = True; break
          if not is_range:
            bit_py = expr_to_python(bit_expr)
            mask = '0xffffffffffffffff' if bits == '64' else '0xffffffff'
            # Set or clear the bit at position bit_py based on rhs
            py_lines.append('  ' * indent + f'_d0=((_d0&~(1<<({bit_py})))|(int(bool({rhs_py}))<<({bit_py})))&{mask}')
            if bits == '64': has_d0_64 = True
          else:
            # Bit range - handled elsewhere, fall through
            if 'f32' in lhs: py_lines.append('  ' * indent + f'_d0=_i32({rhs_py})')
            elif 'f16' in lhs: py_lines.append('  ' * indent + f'_d0=_to_f16_bits({rhs_py})&0xffff')
            elif 'f64' in lhs: py_lines.append('  ' * indent + f'_d0=_i64({rhs_py})'); has_d0_64 = True
            elif '64' in lhs: py_lines.append('  ' * indent + f'_d0=int({rhs_py})&0xffffffffffffffff'); has_d0_64 = True
            else: py_lines.append('  ' * indent + f'_d0=int({rhs_py})&0xffffffff')
        elif lhs.startswith('D0.'):
          if 'f32' in lhs: py_lines.append('  ' * indent + f'_d0=_i32({rhs_py})')
          elif 'f16' in lhs: py_lines.append('  ' * indent + f'_d0=_to_f16_bits({rhs_py})&0xffff')
          elif 'f64' in lhs: py_lines.append('  ' * indent + f'_d0=_i64({rhs_py})'); has_d0_64 = True
          elif '64' in lhs: py_lines.append('  ' * indent + f'_d0=int({rhs_py})&0xffffffffffffffff'); has_d0_64 = True
          else: py_lines.append('  ' * indent + f'_d0=int({rhs_py})&0xffffffff')
        elif lhs == 'SCC' or lhs.startswith('SCC'):
          py_lines.append('  ' * indent + f'_scc=int(bool({rhs_py}))')
        elif lhs == 'tmp' or lhs.startswith('tmp.'):
          py_lines.append('  ' * indent + f'_tmp={rhs_py}')
        # Handle D0[high : low].type = expr (bit range assignment to destination)
        elif (d0_match := re.match(r'D0\[(\d+)\s*:\s*(\d+)\](?:\.(f16|i16|u16|f32|i32|u32|b16|b32))?', lhs)):
          high, low = int(d0_match.group(1)), int(d0_match.group(2))
          typ = d0_match.group(3)
          width = high - low + 1
          mask = (1 << width) - 1
          if typ == 'f16': rhs_bits = f'_i16({rhs_py})'
          elif typ == 'f32': rhs_bits = f'_i32({rhs_py})'
          else: rhs_bits = f'(int({rhs_py})&{hex(mask)})'
          py_lines.append('  ' * indent + f'_d0=(_d0&~{hex(mask << low)})|({rhs_bits}<<{low})')
        # Handle tmp[high : low].type = expr or tmp[high : low] = expr (bit range assignment)
        elif (tmp_match := re.match(r'tmp\[(\d+)\s*:\s*(\d+)\](?:\.(f16|i16|u16|f32|i32|u32|b16|b32))?', lhs)):
          high, low = int(tmp_match.group(1)), int(tmp_match.group(2))
          typ = tmp_match.group(3)
          width = high - low + 1
          mask = (1 << width) - 1
          # Convert rhs to bits if it's a float type
          if typ == 'f16': rhs_bits = f'_i16({rhs_py})'
          elif typ == 'f32': rhs_bits = f'_i32({rhs_py})'
          else: rhs_bits = f'(int({rhs_py})&{hex(mask)})'
          py_lines.append('  ' * indent + f'_tmp=(_tmp&~{hex(mask << low)})|({rhs_bits}<<{low})')
        elif lhs == 'saveexec':
          py_lines.append('  ' * indent + f'_saveexec={rhs_py}')
        elif lhs.startswith('EXEC.'):
          has_exec_mod = True
          if '64' in lhs: py_lines.append('  ' * indent + f'_exec=int({rhs_py})&0xffffffffffffffff')
          else: py_lines.append('  ' * indent + f'_exec=int({rhs_py})&0xffffffff')
        elif lhs == 'VCC':
          # VCC = 0x0 or VCC = 0x1 in pseudocode means set/clear this lane's VCC bit
          has_vcc_lane = True
          py_lines.append('  ' * indent + f'_vcc_lane=int(bool({rhs_py}))')
        elif (vgpr_match := re.match(r'VGPR\[(\w+)\]\[VDST\.u32\]', lhs)):
          # VGPR write: VGPR[lane][VDST.u32] = value
          has_vgpr_write = True
          lane_var = vgpr_match.group(1)
          py_lines.append('  ' * indent + f'_vgpr_write=(_vars.get("{lane_var}",0),_vdst_idx,int({rhs_py})&0xffffffff)')
        else:
          py_lines.append('  ' * indent + f'_vars["{lhs}"]={rhs_py}')

  ret_parts = ['"d0":_d0', '"scc":_scc']
  if has_vcc_lane: ret_parts.append('"vcc_lane":_vcc_lane')
  if has_exec_lane: ret_parts.append('"exec_lane":_exec_lane')
  if has_d0_64: ret_parts.append('"d0_64":True')
  if has_exec_mod: ret_parts.append('"exec":_exec')
  if has_d1: ret_parts.append('"d1":_d1')
  if has_vgpr_write: ret_parts.append('"vgpr_write":_vgpr_write')
  py_lines.append('  return {' + ','.join(ret_parts) + '}')

  return '\n'.join(py_lines), has_d0_64

def generate_functions_file(output_path: str) -> tuple[int, int]:
  """Generate pseudocode_functions.py with all compiled functions."""
  from extra.assembly.rdna3.autogen import pseudocode_data as pd
  from extra.assembly.rdna3.autogen import SOP1Op, SOP2Op, SOPCOp, SOPKOp, SOPPOp, VOP1Op, VOP2Op, VOP3Op, VOP3SDOp, VOP3POp, VOPCOp
  OP_ENUMS = [SOP1Op, SOP2Op, SOPCOp, SOPKOp, SOPPOp, VOP1Op, VOP2Op, VOP3Op, VOP3SDOp, VOP3POp, VOPCOp]

  lines = [
    '# autogenerated by pseudocode_compile.py - do not edit',
    '# to regenerate: python -m extra.assembly.rdna3.pseudocode_compile',
    'import struct, math',
    'from extra.assembly.rdna3.autogen import SOP1Op, SOP2Op, SOPCOp, SOPKOp, SOPPOp, VOP1Op, VOP2Op, VOP3Op, VOP3SDOp, VOP3POp, VOPCOp',
    '',
    '# Helper functions',
    'def _f32(i): return struct.unpack("<f", struct.pack("<I", i & 0xffffffff))[0]',
    'def _i32(f):',
    '  if isinstance(f, int): f = float(f)',
    '  if math.isnan(f): return 0x7fc00000',
    '  if math.isinf(f): return 0x7f800000 if f > 0 else 0xff800000',
    '  try: return struct.unpack("<I", struct.pack("<f", f))[0]',
    '  except (OverflowError, struct.error): return 0x7f800000 if f > 0 else 0xff800000',
    '# Safe division - returns IEEE 754 inf/-inf for division by zero',
    'def _div(a, b):',
    '  try: return a / b',
    '  except ZeroDivisionError:',
    '    if a == 0.0 or math.isnan(a): return float("nan")',
    '    return math.copysign(float("inf"), a * b) if b == 0.0 else float("inf") if a > 0 else float("-inf")',
    'def _sext(v, b): return v - (1 << b) if v & (1 << (b - 1)) else v',
    'def _f16(i): return struct.unpack("<e", struct.pack("<H", i & 0xffff))[0]',
    'def _i16(f):',
    '  if math.isnan(f): return 0x7e00',
    '  if math.isinf(f): return 0x7c00 if f > 0 else 0xfc00',
    '  try: return struct.unpack("<H", struct.pack("<e", f))[0]',
    '  except (OverflowError, struct.error): return 0x7c00 if f > 0 else 0xfc00',
    'def _to_f16_bits(v): return v if isinstance(v, int) else _i16(v)  # pass through ints (already bits), convert floats',
    'def _f64(i): return struct.unpack("<d", struct.pack("<Q", i & 0xffffffffffffffff))[0]',
    'def _i64(f):',
    '  if math.isnan(f): return 0x7ff8000000000000',
    '  if math.isinf(f): return 0x7ff0000000000000 if f > 0 else 0xfff0000000000000',
    '  try: return struct.unpack("<Q", struct.pack("<d", f))[0]',
    '  except (OverflowError, struct.error): return 0x7ff0000000000000 if f > 0 else 0xfff0000000000000',
    'def _isnan(x): return math.isnan(x) if isinstance(x, float) else False',
    'def _gt_neg_zero(a, b): return (a > b) or (a == 0 and b == 0 and not math.copysign(1, a) < 0 and math.copysign(1, b) < 0)',
    'def _lt_neg_zero(a, b): return (a < b) or (a == 0 and b == 0 and math.copysign(1, a) < 0 and not math.copysign(1, b) < 0)',
    'def _fma(a, b, c): return a * b + c',
    'def _signext(v): return v  # sign extension passthrough for already-extended values',
    'trunc, floor, ceil, sqrt, log2 = math.trunc, math.floor, math.ceil, lambda x: math.sqrt(x) if x >= 0 else float("nan"), lambda x: math.log2(x) if x > 0 else (float("-inf") if x == 0 else float("nan"))',
    '# Conversion functions used by pseudocode',
    'i32_to_f32 = u32_to_f32 = i32_to_f64 = u32_to_f64 = f32_to_f64 = f64_to_f32 = float',
    'def f32_to_i32(f):',
    '  if math.isnan(f): return 0',
    '  if f >= 2147483647: return 2147483647',
    '  if f <= -2147483648: return -2147483648',
    '  return int(f)',
    'def f32_to_u32(f):',
    '  if math.isnan(f): return 0',
    '  if f >= 4294967295: return 4294967295',
    '  if f <= 0: return 0',
    '  return int(f)',
    'f64_to_i32 = f32_to_i32',
    'f64_to_u32 = f32_to_u32',
    'def f32_to_f16(f): return struct.unpack("<H", struct.pack("<e", float(f)))[0]',
    'def _f16_to_f32_bits(bits): return struct.unpack("<e", struct.pack("<H", int(bits) & 0xffff))[0]',
    'def f16_to_f32(v): return v if isinstance(v, float) else _f16_to_f32_bits(v)  # pass through floats, convert bits',
    'def i16_to_f16(v): return f32_to_f16(float(_sext(int(v) & 0xffff, 16)))',
    'def u16_to_f16(v): return f32_to_f16(float(int(v) & 0xffff))',
    'def f16_to_i16(bits): f = _f16_to_f32_bits(bits); return max(-32768, min(32767, int(f))) if not math.isnan(f) else 0',
    'def f16_to_u16(bits): f = _f16_to_f32_bits(bits); return max(0, min(65535, int(f))) if not math.isnan(f) else 0',
    '# Sign and mantissa extraction',
    'def _sign(f): return 1 if math.copysign(1.0, f) < 0 else 0',
    'def _mantissa_f32(f): return struct.unpack("<I", struct.pack("<f", f))[0] & 0x7fffff if not (math.isinf(f) or math.isnan(f)) else 0',
    'def _mantissa_f64(f): return struct.unpack("<Q", struct.pack("<d", f))[0] & 0xfffffffffffff if not (math.isinf(f) or math.isnan(f)) else 0',
    'def _ldexp(m, e): return math.ldexp(m, e)',
    '# Math helper functions',
    'def isEven(x): return int(x) % 2 == 0',
    'def fract(x): return x - math.floor(x)',
    'PI = math.pi',
    'def sin(x): return float("nan") if math.isinf(x) or math.isnan(x) else math.sin(x)',
    'def cos(x): return float("nan") if math.isinf(x) or math.isnan(x) else math.cos(x)',
    '# Bit reversal functions',
    'def _brev32(v): return int(bin(v & 0xffffffff)[2:].zfill(32)[::-1], 2)',
    'def _brev64(v): return int(bin(v & 0xffffffffffffffff)[2:].zfill(64)[::-1], 2)',
    '# Count trailing zeros (find first one) - returns bit position of lowest set bit',
    'def _ctz32(v):',
    '  v = int(v) & 0xffffffff',
    '  if v == 0: return 32',
    '  n = 0',
    '  while (v & 1) == 0: v >>= 1; n += 1',
    '  return n',
    'def _ctz64(v):',
    '  v = int(v) & 0xffffffffffffffff',
    '  if v == 0: return 64',
    '  n = 0',
    '  while (v & 1) == 0: v >>= 1; n += 1',
    '  return n',
    '# IEEE float component extraction - _exponent returns biased exponent',
    'def _exponent(f):',
    '  if math.isinf(f) or math.isnan(f): return 0',
    '  if f == 0.0: return 0',
    '  # Get the biased exponent from the float representation',
    '  try: bits = struct.unpack("<I", struct.pack("<f", float(f)))[0]; return (bits >> 23) & 0xff',
    '  except: return 0',
    '# Denormal detection - a number is denormal if exponent is 0 but mantissa is non-zero',
    'def _is_denorm_f32(f):',
    '  if not isinstance(f, float): f = _f32(int(f) & 0xffffffff)',
    '  if math.isinf(f) or math.isnan(f) or f == 0.0: return False',
    '  bits = struct.unpack("<I", struct.pack("<f", float(f)))[0]',
    '  exp = (bits >> 23) & 0xff',
    '  return exp == 0',
    'def _is_denorm_f64(f):',
    '  if not isinstance(f, float): f = _f64(int(f) & 0xffffffffffffffff)',
    '  if math.isinf(f) or math.isnan(f) or f == 0.0: return False',
    '  bits = struct.unpack("<Q", struct.pack("<d", float(f)))[0]',
    '  exp = (bits >> 52) & 0x7ff',
    '  return exp == 0',
    '# v_min/v_max helper functions for MIN3/MAX3 ops',
    'def v_min_f32(a, b):',
    '  if math.isnan(b): return a',
    '  if math.isnan(a): return b',
    '  return a if _lt_neg_zero(a, b) else b',
    'def v_max_f32(a, b):',
    '  if math.isnan(b): return a',
    '  if math.isnan(a): return b',
    '  return a if _gt_neg_zero(a, b) else b',
    'def v_min_i32(a, b): return min(a, b)',
    'def v_max_i32(a, b): return max(a, b)',
    'def v_min_u32(a, b): return min(a & 0xffffffff, b & 0xffffffff)',
    'def v_max_u32(a, b): return max(a & 0xffffffff, b & 0xffffffff)',
    '# v_min/max for 16-bit types',
    'def v_min_f16(a, b): return v_min_f32(a, b)',
    'def v_max_f16(a, b): return v_max_f32(a, b)',
    'def v_min_i16(a, b): return min(a, b)',
    'def v_max_i16(a, b): return max(a, b)',
    'def v_min_u16(a, b): return min(a & 0xffff, b & 0xffff)',
    'def v_max_u16(a, b): return max(a & 0xffff, b & 0xffff)',
    '# v_min3/max3 - three operand min/max',
    'def v_min3_f32(a, b, c): return v_min_f32(v_min_f32(a, b), c)',
    'def v_max3_f32(a, b, c): return v_max_f32(v_max_f32(a, b), c)',
    'def v_min3_i32(a, b, c): return min(a, b, c)',
    'def v_max3_i32(a, b, c): return max(a, b, c)',
    'def v_min3_u32(a, b, c): return min(a & 0xffffffff, b & 0xffffffff, c & 0xffffffff)',
    'def v_max3_u32(a, b, c): return max(a & 0xffffffff, b & 0xffffffff, c & 0xffffffff)',
    'def v_min3_f16(a, b, c): return v_min_f16(v_min_f16(a, b), c)',
    'def v_max3_f16(a, b, c): return v_max_f16(v_max_f16(a, b), c)',
    'def v_min3_i16(a, b, c): return min(a, b, c)',
    'def v_max3_i16(a, b, c): return max(a, b, c)',
    'def v_min3_u16(a, b, c): return min(a & 0xffff, b & 0xffff, c & 0xffff)',
    'def v_max3_u16(a, b, c): return max(a & 0xffff, b & 0xffff, c & 0xffff)',
    '# ABSDIFF - absolute difference',
    'def ABSDIFF(a, b): return abs(a - b)',
    '# f16/f32 to normalized integer conversions',
    'def f16_to_snorm(f): return max(-32768, min(32767, int(round(max(-1.0, min(1.0, f)) * 32767))))',
    'def f16_to_unorm(f): return max(0, min(65535, int(round(max(0.0, min(1.0, f)) * 65535))))',
    'def f32_to_snorm(f): return max(-32768, min(32767, int(round(max(-1.0, min(1.0, f)) * 32767))))',
    'def f32_to_unorm(f): return max(0, min(65535, int(round(max(0.0, min(1.0, f)) * 65535))))',
    '# Integer conversion functions with clamping/truncation',
    'def v_cvt_i16_f32(f): return max(-32768, min(32767, int(f))) if not math.isnan(f) else 0',
    'def v_cvt_u16_f32(f): return max(0, min(65535, int(f))) if not math.isnan(f) else 0',
    'def u32_to_u16(u): return int(u) & 0xffff',
    'def i32_to_i16(i): return ((int(i) + 32768) & 0xffff) - 32768',
    '# Saturation function for packing',
    'def SAT8(v): return max(0, min(255, int(v)))',
    'def f32_to_u8(f): return max(0, min(255, int(f))) if not math.isnan(f) else 0',
    '# mantissa extraction - returns significand in range [1, 2) for frexp_mant',
    'def mantissa(f):',
    '  if f == 0.0 or math.isinf(f) or math.isnan(f): return f',
    '  m, _ = math.frexp(f)',
    '  return math.copysign(m * 2.0, f)  # frexp returns [0.5, 1), we need [1, 2)',
    '# Sign extend from a specific bit position',
    'def signext_from_bit(val, bit):',
    '  bit = int(bit)',
    '  if bit == 0: return 0',
    '  mask = (1 << bit) - 1',
    '  val = int(val) & mask',
    '  if val & (1 << (bit - 1)): return val - (1 << bit)',
    '  return val',
    '',
    '# Compiled pseudocode functions',
  ]

  compiled_count, fallback_count = 0, 0

  for enum_cls in OP_ENUMS:
    cls_name = enum_cls.__name__
    pseudocode_dict = getattr(pd, f"{cls_name}_PSEUDOCODE", {})
    if not pseudocode_dict: continue

    fn_names: dict = {}
    for op, pc in pseudocode_dict.items():
      result = compile_pseudocode(pc)
      if result is not None:
        code, _ = result
        fn_name = f'_fn_{cls_name}_{op.name}'
        # Add the function
        lines.append(code.replace('def _fn(', f'def {fn_name}('))
        lines.append('')
        fn_names[op] = fn_name
        compiled_count += 1
      else:
        fallback_count += 1

    # Add the dictionary for this class
    lines.append(f'{cls_name}_FUNCTIONS = {{')
    for op, fn_name in fn_names.items():
      if fn_name is not None:
        lines.append(f'  {cls_name}.{op.name}: {fn_name},')
    lines.append('}')
    lines.append('')

  # Add the main getter function
  lines.append('COMPILED_FUNCTIONS = {')
  for enum_cls in OP_ENUMS:
    cls_name = enum_cls.__name__
    lines.append(f'  {cls_name}: {cls_name}_FUNCTIONS,')
  lines.append('}')
  lines.append('')
  lines.append('def get_compiled_functions(): return COMPILED_FUNCTIONS')

  Path(output_path).write_text('\n'.join(lines))
  return compiled_count, fallback_count

if __name__ == "__main__":
  output = "extra/assembly/rdna3/autogen/pseudocode_functions.py"
  compiled, fallback = generate_functions_file(output)
  print(f"Generated {output}: {compiled} compiled, {fallback} fallback (interpreter)")
