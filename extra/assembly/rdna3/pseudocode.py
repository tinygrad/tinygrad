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
  if math.isinf(f): return 0x7f800000 if f > 0 else 0xff800000
  try: return struct.unpack('<I', struct.pack('<f', f))[0]
  except (OverflowError, struct.error): return 0x7f800000 if f > 0 else 0xff800000

def _sext(v: int, b: int) -> int:
  """Sign extend b-bit value to Python int."""
  return v - (1 << b) if v & (1 << (b - 1)) else v

class PseudocodeInterpreter:
  """Interpreter for RDNA3 pseudocode."""

  def __init__(self):
    self.vars: dict[str, Any] = {}

  def eval_expr(self, expr: str, s0: int, s1: int, s2: int, scc: int, d0: int) -> Any:
    """Evaluate a pseudocode expression."""
    expr = expr.strip()

    # Handle bit indexing like S1[4 : 0].u32 -> extract bits [4:0] from S1 (BEFORE variable substitution)
    def replace_bit_range(m):
      var, hi, lo = m.group(1), int(m.group(2)), int(m.group(3))
      if var == 'S0': val = s0
      elif var == 'S1': val = s1
      elif var == 'S2': val = s2
      else: return m.group(0)
      return str((val >> lo) & ((1 << (hi - lo + 1)) - 1))
    expr = re.sub(r'(S[012])\[(\d+)\s*:\s*(\d+)\]\.u32', replace_bit_range, expr)
    expr = re.sub(r'(S[012])\[(\d+)\s*:\s*(\d+)\]', replace_bit_range, expr)

    # Handle signext function BEFORE substituting S0.i32 etc.
    expr = re.sub(r'signext\(S0\.i32\)', str(_sext(s0 & 0xffffffff, 32)), expr)
    expr = re.sub(r'signext\(S1\.i32\)', str(_sext(s1 & 0xffffffff, 32)), expr)
    expr = re.sub(r'signext\(([^)]+)\)', r'_sext(\1, 32)', expr)

    # Replace source/dest field access
    expr = re.sub(r'S0\.u32', str(s0 & 0xffffffff), expr)
    expr = re.sub(r'S0\.i32', str(_sext(s0 & 0xffffffff, 32)), expr)
    expr = re.sub(r'S0\.f32', f'_f32({s0 & 0xffffffff})', expr)
    expr = re.sub(r'S0\.b32', str(s0 & 0xffffffff), expr)
    expr = re.sub(r'S1\.u32', str(s1 & 0xffffffff), expr)
    expr = re.sub(r'S1\.i32', str(_sext(s1 & 0xffffffff, 32)), expr)
    expr = re.sub(r'S1\.f32', f'_f32({s1 & 0xffffffff})', expr)
    expr = re.sub(r'S1\.b32', str(s1 & 0xffffffff), expr)
    expr = re.sub(r'S2\.u32', str(s2 & 0xffffffff), expr)
    expr = re.sub(r'S2\.f32', f'_f32({s2 & 0xffffffff})', expr)
    expr = re.sub(r'D0\.u32', str(d0 & 0xffffffff), expr)
    expr = re.sub(r'D0\.i32', str(_sext(d0 & 0xffffffff, 32)), expr)
    expr = re.sub(r'D0\.f32', f'_f32({d0 & 0xffffffff})', expr)
    expr = re.sub(r'SCC\.u32', str(scc & 1), expr)
    expr = re.sub(r'SCC\.u64', str(scc & 1), expr)

    # Handle single bit indexing like tmp[31]
    expr = re.sub(r'(\w+)\[(\d+)\]', r'((\1 >> \2) & 1)', expr)

    # Handle tmp variable
    if 'tmp' in self.vars:
      expr = re.sub(r'\btmp\.u32\b', str(self.vars['tmp'] & 0xffffffff), expr)
      expr = re.sub(r'\btmp\.i32\b', str(_sext(self.vars['tmp'] & 0xffffffff, 32)), expr)
      expr = re.sub(r'\btmp\b', str(self.vars['tmp']), expr)

    # Handle type casts
    expr = re.sub(r"64'U\(([^)]+)\)", r'((\1) & 0xffffffffffffffff)', expr)
    expr = re.sub(r"32'U\(([^)]+)\)", r'((\1) & 0xffffffff)', expr)
    expr = re.sub(r"32'I\(([^)]+)\)", r'((_sext((\1), 32)) & 0xffffffff)', expr)
    expr = re.sub(r"1'1U", '1', expr)
    expr = re.sub(r"1'0U", '0', expr)

    # Handle conversion functions
    expr = re.sub(r'i32_to_f32\(([^)]+)\)', r'float(\1)', expr)
    expr = re.sub(r'u32_to_f32\(([^)]+)\)', r'float(\1)', expr)
    expr = re.sub(r'f32_to_i32\(([^)]+)\)', r'int(\1)', expr)
    expr = re.sub(r'f32_to_u32\(([^)]+)\)', r'max(0, int(\1))', expr)

    # Handle bare SCC reference
    expr = re.sub(r'\bSCC\b', str(scc), expr)

    # Handle hex literals
    expr = re.sub(r'0x([0-9a-fA-F]+)ULL', r'0x\1', expr)
    expr = re.sub(r'0x([0-9a-fA-F]+)U', r'0x\1', expr)
    expr = re.sub(r'(\d+)U\b', r'\1', expr)
    expr = re.sub(r'(\d+)ULL\b', r'\1', expr)

    # Handle ternary operator
    m = re.search(r'([^?]+)\s*\?\s*([^:]+)\s*:\s*(.+)', expr)
    if m:
      cond, true_val, false_val = m.groups()
      return self.eval_expr(true_val, s0, s1, s2, scc, d0) if self.eval_expr(cond, s0, s1, s2, scc, d0) else self.eval_expr(false_val, s0, s1, s2, scc, d0)

    try:
      return eval(expr, {'_f32': _f32, '_i32': _i32, '_sext': _sext, 'abs': abs, 'math': math})
    except Exception as e:
      raise ValueError(f"Failed to evaluate '{expr}': {e}")

  def execute(self, pseudocode: list[str], s0: int, s1: int, s2: int = 0, scc: int = 0, d0: int = 0) -> tuple[int, int]:
    """Execute pseudocode and return (result, scc)."""
    self.vars = {}
    result, new_scc = d0, scc

    for line in pseudocode:
      line = line.strip()
      if line.startswith('//'): continue
      if not line or line.startswith('if ') or line.startswith('else') or line.startswith('endif'): continue

      if '=' in line and not any(line.startswith(p) for p in ['if ', '==']):
        parts = line.rstrip(';').split('=', 1)
        if len(parts) == 2:
          lhs, rhs = parts[0].strip(), parts[1].strip()
          val = self.eval_expr(rhs, s0, s1, s2, new_scc, result)

          if lhs.startswith('D0.'):
            if '.f32' in lhs:
              result = _i32(float(val)) if isinstance(val, (int, float)) else int(val) & 0xffffffff
            else:
              result = int(val) & 0xffffffff
          elif lhs.startswith('SCC'):
            new_scc = int(bool(val))
          elif lhs == 'tmp':
            self.vars['tmp'] = int(val) if not isinstance(val, float) else val
          else:
            self.vars[lhs] = val

    return result, new_scc


# ═══════════════════════════════════════════════════════════════════════════════
# PDF PARSING (only used by generate(), not at runtime)
# ═══════════════════════════════════════════════════════════════════════════════

PDF_URL = "https://docs.amd.com/api/khub/documents/UVVZM22UN7tMUeiW_4ShTQ/content"
SECTIONS = {
  'SOP2': (198, 216), 'SOPK': (216, 226), 'SOP1': (226, 245), 'SOPC': (245, 256),
  'SOPP': (256, 270), 'SMEM': (270, 291), 'VOP1': (291, 320), 'VOPC': (320, 380),
  'VOP2_VOP3': (380, 550),
}
INST_PATTERN = re.compile(r'^([A-Z][A-Z0-9_]+)\s+(\d+)\s*$', re.M)

def extract_pseudocode_lines(text: str) -> list[str]:
  """Extract pseudocode lines from an instruction description snippet."""
  lines, result, depth = text.split('\n'), [], 0
  for line in lines:
    s = line.strip()
    if not s or re.match(r'^\d+ of \d+$', s): continue
    if s.startswith('Notes') or re.match(r'^\d+\.\d+\..*Instructions', s): break
    if s.startswith('Functional examples'): break
    if s.startswith('if '): depth += 1
    elif s.startswith('endif'): depth = max(0, depth - 1)
    if s.endswith('.') and not any(p in s for p in ['D0', 'D1', 'S0', 'S1', 'S2', 'SCC', 'VCC', 'tmp', '=']): continue
    if re.match(r'^[a-z].*\.$', s) and '=' not in s: continue
    is_code = (
      any(p in s for p in ['D0.', 'D1.', 'S0.', 'S1.', 'S2.', 'SCC =', 'SCC ?', 'VCC', 'EXEC', 'tmp =', 'lane =']) or
      s.startswith(('if ', 'else', 'elsif', 'endif', 'declare ', 'for ', '//')) or
      re.match(r'^[a-z_]+\s*=', s) or (depth > 0 and '=' in s)
    )
    if is_code: result.append(s)
  return result

def parse_pseudocode(pdf_path: str | None = None) -> dict[str, dict]:
  """Parse all instruction pseudocode from the PDF. Returns dict of name -> {opcode, section, pseudocode}."""
  import pdfplumber
  from tinygrad.helpers import fetch
  pdf = pdfplumber.open(fetch(PDF_URL) if pdf_path is None else pdf_path)
  page_texts = {i: pdf.pages[i].extract_text() or '' for i in range(195, 550)}

  instructions = {}
  for section_name, (start_page, end_page) in SECTIONS.items():
    section_text = '\n'.join(page_texts.get(i, '') for i in range(start_page, end_page))
    for match in INST_PATTERN.finditer(section_text):
      name, opcode = match.group(1), int(match.group(2))
      if not name.startswith(('S_', 'V_')): continue
      start = match.end()
      next_match = INST_PATTERN.search(section_text, start)
      end = next_match.start() if next_match else start + 2000
      snippet = section_text[start:end].strip()
      pseudocode = extract_pseudocode_lines(snippet)
      if pseudocode:
        instructions[name] = {'opcode': opcode, 'section': section_name, 'pseudocode': pseudocode}
  return instructions

def generate(output_path: pathlib.Path | str | None = None) -> dict[str, dict]:
  """Generate pseudocode data file from PDF. Returns parsed instructions."""
  instructions = parse_pseudocode()

  if output_path is not None:
    lines = [
      "# autogenerated from AMD RDNA3.5 ISA PDF by pseudocode.py - do not edit",
      "# instruction name -> {'opcode': int, 'section': str, 'pseudocode': list[str]}",
      "PSEUDOCODE: dict[str, dict] = {"
    ]
    for name, info in sorted(instructions.items()):
      pc_str = repr(info['pseudocode'])
      lines.append(f"  {name!r}: {{'opcode': {info['opcode']}, 'section': {info['section']!r}, 'pseudocode': {pc_str}}},")
    lines.append("}")
    pathlib.Path(output_path).write_text('\n'.join(lines))

  return instructions

# ═══════════════════════════════════════════════════════════════════════════════
# RUNTIME: Load pre-generated pseudocode
# ═══════════════════════════════════════════════════════════════════════════════

def get_pseudocode() -> dict[str, dict]:
  """Get pseudocode dict, loading from generated file if available."""
  try:
    from extra.assembly.rdna3.autogen.pseudocode_data import PSEUDOCODE
    return PSEUDOCODE
  except ImportError:
    # Fall back to parsing PDF (slow, but works if autogen not available)
    return parse_pseudocode()


if __name__ == "__main__":
  import sys
  if len(sys.argv) > 1 and sys.argv[1] == "generate":
    output = "extra/assembly/rdna3/autogen/pseudocode_data.py"
    result = generate(output)
    print(f"Generated {output} with {len(result)} instructions")
  else:
    # Test mode
    instructions = get_pseudocode()
    print(f"Loaded {len(instructions)} instructions")

    interp = PseudocodeInterpreter()
    if 'S_ADD_U32' in instructions:
      pc = instructions['S_ADD_U32']['pseudocode']
      print(f"\nS_ADD_U32 pseudocode: {pc}")
      result, scc = interp.execute(pc, s0=0xffffffff, s1=2)
      print(f"S_ADD_U32(0xffffffff, 2) = {result:#x}, scc={scc}")

    if 'V_ADD_F32' in instructions:
      pc = instructions['V_ADD_F32']['pseudocode']
      print(f"\nV_ADD_F32 pseudocode: {pc}")
      result, _ = interp.execute(pc, s0=_i32(1.5), s1=_i32(2.5))
      print(f"V_ADD_F32(1.5, 2.5) = {_f32(result)}")
