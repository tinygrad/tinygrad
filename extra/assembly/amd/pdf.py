# Generate AMD ISA autogen files from PDF documentation
# Combines format/enum generation (previously in dsl.py) and pseudocode compilation (previously in pcode.py)
# Usage: python -m extra.assembly.amd.pdf [--arch rdna3|rdna4|cdna|all]
import re, functools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

PDF_URLS = {
  "rdna3": "https://docs.amd.com/api/khub/documents/UVVZM22UN7tMUeiW_4ShTQ/content",
  "rdna4": "https://docs.amd.com/api/khub/documents/uQpkEvk3pv~kfAb2x~j4uw/content",
  "cdna": ["https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf",
           "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf"],
}

# Field type mappings and ordering
FIELD_TYPES = {'SSRC0': 'SSrc', 'SSRC1': 'SSrc', 'SOFFSET': 'SSrc', 'SADDR': 'SSrc', 'SRC0': 'Src', 'SRC1': 'Src', 'SRC2': 'Src',
  'SDST': 'SGPRField', 'SBASE': 'SGPRField', 'SDATA': 'SGPRField', 'SRSRC': 'SGPRField', 'VDST': 'VGPRField', 'VSRC1': 'VGPRField',
  'VDATA': 'VGPRField', 'VADDR': 'VGPRField', 'ADDR': 'VGPRField', 'DATA': 'VGPRField', 'DATA0': 'VGPRField', 'DATA1': 'VGPRField',
  'SIMM16': 'SImm', 'OFFSET': 'Imm', 'OPX': 'VOPDOp', 'OPY': 'VOPDOp', 'SRCX0': 'Src', 'SRCY0': 'Src',
  'VSRCX1': 'VGPRField', 'VSRCY1': 'VGPRField', 'VDSTX': 'VGPRField', 'VDSTY': 'VDSTYEnc'}
FIELD_ORDER = {
  'SOP2': ['op', 'sdst', 'ssrc0', 'ssrc1'], 'SOP1': ['op', 'sdst', 'ssrc0'], 'SOPC': ['op', 'ssrc0', 'ssrc1'],
  'SOPK': ['op', 'sdst', 'simm16'], 'SOPP': ['op', 'simm16'], 'VOP1': ['op', 'vdst', 'src0'], 'VOPC': ['op', 'src0', 'vsrc1'],
  'VOP2': ['op', 'vdst', 'src0', 'vsrc1'], 'VOP3SD': ['op', 'vdst', 'sdst', 'src0', 'src1', 'src2', 'clmp'],
  'SMEM': ['op', 'sdata', 'sbase', 'soffset', 'offset', 'glc', 'dlc'], 'DS': ['op', 'vdst', 'addr', 'data0', 'data1'],
  'VOP3': ['op', 'vdst', 'src0', 'src1', 'src2', 'omod', 'neg', 'abs', 'clmp', 'opsel'],
  'VOP3P': ['op', 'vdst', 'src0', 'src1', 'src2', 'neg', 'neg_hi', 'opsel', 'opsel_hi', 'clmp'],
  'FLAT': ['op', 'vdst', 'addr', 'data', 'saddr', 'offset', 'seg', 'dlc', 'glc', 'slc'],
  'MUBUF': ['op', 'vdata', 'vaddr', 'srsrc', 'soffset', 'offset', 'offen', 'idxen', 'glc', 'dlc', 'slc', 'tfe'],
  'MTBUF': ['op', 'vdata', 'vaddr', 'srsrc', 'soffset', 'offset', 'format', 'offen', 'idxen', 'glc', 'dlc', 'slc', 'tfe'],
  'MIMG': ['op', 'vdata', 'vaddr', 'srsrc', 'ssamp', 'dmask', 'dim', 'unrm', 'dlc', 'glc', 'slc'],
  'EXP': ['en', 'target', 'vsrc0', 'vsrc1', 'vsrc2', 'vsrc3', 'done', 'row'],
  'VINTERP': ['op', 'vdst', 'src0', 'src1', 'src2', 'waitexp', 'clmp', 'opsel', 'neg'],
  'VOPD': ['opx', 'opy', 'vdstx', 'vdsty', 'srcx0', 'vsrcx1', 'srcy0', 'vsrcy1'],
  'LDSDIR': ['op', 'vdst', 'attr', 'attr_chan', 'wait_va']}
SRC_EXTRAS = {233: 'DPP8', 234: 'DPP8FI', 250: 'DPP16', 251: 'VCCZ', 252: 'EXECZ', 254: 'LDS_DIRECT'}
FLOAT_MAP = {'0.5': 'POS_HALF', '-0.5': 'NEG_HALF', '1.0': 'POS_ONE', '-1.0': 'NEG_ONE', '2.0': 'POS_TWO', '-2.0': 'NEG_TWO',
  '4.0': 'POS_FOUR', '-4.0': 'NEG_FOUR', '1/(2*PI)': 'INV_2PI', '0': 'ZERO'}
INST_PATTERN = re.compile(r'^([SVD]S?_[A-Z0-9_]+|(?:FLAT|GLOBAL|SCRATCH)_[A-Z0-9_]+)\s+(\d+)\s*$', re.M)

# Patterns that can't be handled by the DSL (require special handling in emu.py)
UNSUPPORTED = ['SGPR[', 'V_SWAP', 'eval ', 'FATAL_HALT', 'HW_REGISTERS',
               'vscnt', 'vmcnt', 'expcnt', 'lgkmcnt',
               'CVT_OFF_TABLE', 'ThreadMask',
               'S1[i', 'C.i32', 'S[i]', 'in[',
               'if n.', 'DST.u32', 'addrd = DST', 'addr = DST',
               'BARRIER_STATE', 'ReallocVgprs',
               'GPR_IDX', 'VSKIP', 'specified in', 'TTBL',
               'fp6', 'bf6', 'GS_REGS', 'M0.base', 'DS_DATA', '= 0..', 'sign(src', 'if no LDS', 'gds_base', 'vector mask',
               'SGPR_ADDR', 'INST_OFFSET', 'laneID']  # FLAT ops with non-standard vars

# ═══════════════════════════════════════════════════════════════════════════════
# COMPILER: pseudocode -> Python (minimal transforms)
# ═══════════════════════════════════════════════════════════════════════════════

def compile_pseudocode(pseudocode: str) -> str:
  """Compile pseudocode to Python. Transforms are minimal - most syntax just works."""
  pseudocode = re.sub(r'\bpass\b', 'pass_', pseudocode)  # 'pass' is Python keyword
  raw_lines = pseudocode.strip().split('\n')
  joined_lines: list[str] = []
  for line in raw_lines:
    line = line.strip()
    if joined_lines and (joined_lines[-1].rstrip().endswith(('||', '&&', '(', ',')) or
                         (joined_lines[-1].count('(') > joined_lines[-1].count(')'))):
      joined_lines[-1] = joined_lines[-1].rstrip() + ' ' + line
    else:
      joined_lines.append(line)

  lines = []
  indent, need_pass, in_first_match_loop = 0, False, False
  for line in joined_lines:
    line = line.split('//')[0].strip()  # Strip C-style comments
    if not line: continue
    if line.startswith('if '):
      lines.append('  ' * indent + f"if {_expr(line[3:].rstrip(' then'))}:")
      indent += 1
      need_pass = True
    elif line.startswith('elsif '):
      if need_pass: lines.append('  ' * indent + "pass")
      indent -= 1
      lines.append('  ' * indent + f"elif {_expr(line[6:].rstrip(' then'))}:")
      indent += 1
      need_pass = True
    elif line == 'else':
      if need_pass: lines.append('  ' * indent + "pass")
      indent -= 1
      lines.append('  ' * indent + "else:")
      indent += 1
      need_pass = True
    elif line.startswith('endif'):
      if need_pass: lines.append('  ' * indent + "pass")
      indent -= 1
      need_pass = False
    elif line.startswith('endfor'):
      if need_pass: lines.append('  ' * indent + "pass")
      indent -= 1
      need_pass, in_first_match_loop = False, False
    elif line.startswith('declare '):
      pass
    elif m := re.match(r'for (\w+) in (.+?)\s*:\s*(.+?) do', line):
      start, end = _expr(m[2].strip()), _expr(m[3].strip())
      lines.append('  ' * indent + f"for {m[1]} in range({start}, int({end})+1):")
      indent += 1
      need_pass, in_first_match_loop = True, True
    elif '=' in line and not line.startswith('=='):
      need_pass = False
      line = line.rstrip(';')
      if m := re.match(r'\{\s*D1\.[ui]1\s*,\s*D0\.[ui]64\s*\}\s*=\s*(.+)', line):
        rhs = _expr(m[1])
        lines.append('  ' * indent + f"_full = {rhs}")
        lines.append('  ' * indent + f"D0.u64 = int(_full) & 0xffffffffffffffff")
        lines.append('  ' * indent + f"D1 = Reg((int(_full) >> 64) & 1)")
      elif any(op in line for op in ('+=', '-=', '*=', '/=', '|=', '&=', '^=')):
        for op in ('+=', '-=', '*=', '/=', '|=', '&=', '^='):
          if op in line:
            lhs, rhs = line.split(op, 1)
            lines.append('  ' * indent + f"{lhs.strip()} {op} {_expr(rhs.strip())}")
            break
      else:
        lhs, rhs = line.split('=', 1)
        lhs_s, rhs_s = _expr(lhs.strip()), rhs.strip()
        stmt = _assign(lhs_s, _expr(rhs_s))
        if in_first_match_loop and rhs_s == 'i' and (lhs_s == 'tmp' or lhs_s == 'D0.i32'):
          stmt += "; break"
        lines.append('  ' * indent + stmt)
  if need_pass: lines.append('  ' * indent + "pass")
  return '\n'.join(lines)

def _assign(lhs: str, rhs: str) -> str:
  if lhs in ('tmp', 'SCC', 'VCC', 'EXEC', 'D0', 'D1', 'saveexec', 'PC'):
    return f"{lhs} = Reg({rhs})"
  return f"{lhs} = {rhs}"

def _expr(e: str) -> str:
  e = e.strip()
  e = e.replace('&&', ' and ').replace('||', ' or ').replace('<>', ' != ')
  e = re.sub(r'!([^=])', r' not \1', e)
  e = re.sub(r'\{\s*(\w+\.u32)\s*,\s*(\w+\.u32)\s*\}', r'_pack32(\1, \2)', e)
  def pack(m):
    hi, lo = _expr(m[1].strip()), _expr(m[2].strip())
    return f'_pack({hi}, {lo})'
  e = re.sub(r'\{\s*([^,{}]+)\s*,\s*([^,{}]+)\s*\}', pack, e)
  e = re.sub(r"1201'B\(2\.0\s*/\s*PI\)", "TWO_OVER_PI_1201", e)
  e = re.sub(r"\d+'([0-9a-fA-Fx]+)[UuFf]*", r'\1', e)
  e = re.sub(r"\d+'[FIBU]\(", "(", e)
  e = re.sub(r'\bB\(', '(', e)
  e = re.sub(r'([0-9a-fA-Fx])ULL\b', r'\1', e)
  e = re.sub(r'([0-9a-fA-Fx])LL\b', r'\1', e)
  e = re.sub(r'([0-9a-fA-Fx])U\b', r'\1', e)
  e = re.sub(r'(\d\.?\d*)F\b', r'\1', e)
  e = re.sub(r'(\[laneId\])\.[uib]\d+', r'\1', e)
  e = e.replace('+INF', 'INF').replace('-INF', '(-INF)')
  e = re.sub(r'NAN\.f\d+', 'float("nan")', e)
  def convert_verilog_slice(m):
    start, width = m.group(1).strip(), m.group(2).strip()
    return f'[({start}) + ({width}) - 1 : ({start})]'
  e = re.sub(r'\[([^:\[\]]+)\s*\+:\s*([^:\[\]]+)\]', convert_verilog_slice, e)
  def process_brackets(s):
    result, i = [], 0
    while i < len(s):
      if s[i] == '[':
        depth, start = 1, i + 1
        j = start
        while j < len(s) and depth > 0:
          if s[j] == '[': depth += 1
          elif s[j] == ']': depth -= 1
          j += 1
        inner = _expr(s[start:j-1])
        result.append('[' + inner + ']')
        i = j
      else:
        result.append(s[i])
        i += 1
    return ''.join(result)
  e = process_brackets(e)
  while '?' in e:
    depth, bracket, q = 0, 0, -1
    for i, c in enumerate(e):
      if c == '(': depth += 1
      elif c == ')': depth -= 1
      elif c == '[': bracket += 1
      elif c == ']': bracket -= 1
      elif c == '?' and depth == 0 and bracket == 0: q = i; break
    if q < 0: break
    depth, bracket, col = 0, 0, -1
    for i in range(q + 1, len(e)):
      if e[i] == '(': depth += 1
      elif e[i] == ')': depth -= 1
      elif e[i] == '[': bracket += 1
      elif e[i] == ']': bracket -= 1
      elif e[i] == ':' and depth == 0 and bracket == 0: col = i; break
    if col < 0: break
    cond, t, f = e[:q].strip(), e[q+1:col].strip(), e[col+1:].strip()
    e = f'(({t}) if ({cond}) else ({f}))'
  return e

# ═══════════════════════════════════════════════════════════════════════════════
# PDF PARSING WITH PAGE CACHING
# ═══════════════════════════════════════════════════════════════════════════════

class CachedPDF:
  """PDF wrapper with page text/table caching for faster repeated access."""
  def __init__(self, pdf):
    self._pdf, self._text_cache, self._table_cache = pdf, {}, {}
  def __len__(self): return len(self._pdf.pages)
  def text(self, i):
    if i not in self._text_cache: self._text_cache[i] = self._pdf.pages[i].extract_text() or ''
    return self._text_cache[i]
  def tables(self, i):
    if i not in self._table_cache: self._table_cache[i] = [t.extract() for t in self._pdf.pages[i].find_tables()]
    return self._table_cache[i]

def _parse_bits(s: str) -> tuple[int, int] | None:
  return (int(m.group(1)), int(m.group(2) or m.group(1))) if (m := re.match(r'\[(\d+)(?::(\d+))?\]', s)) else None

def _parse_fields_table(table: list, fmt: str, enums: set[str]) -> list[tuple]:
  fields = []
  for row in table[1:]:
    if not row or not row[0]: continue
    name, bits_str = row[0].split('\n')[0].strip(), (row[1] or '').split('\n')[0].strip()
    if not (bits := _parse_bits(bits_str)): continue
    enc_val, hi, lo = None, bits[0], bits[1]
    if name == 'ENCODING' and row[2]:
      if m := re.search(r"(?:'b|Must be:\s*)([01_]+)", row[2]):
        enc_bits = m.group(1).replace('_', '')
        enc_val, declared_width, actual_width = int(enc_bits, 2), hi - lo + 1, len(enc_bits)
        if actual_width > declared_width: lo = hi - actual_width + 1
    ftype = f"{fmt}Op" if name == 'OP' and f"{fmt}Op" in enums else FIELD_TYPES.get(name.upper())
    fields.append((name, hi, lo, enc_val, ftype))
  return fields

def _parse_single_pdf(url: str):
  """Parse a single PDF and return (formats, enums, src_enum, doc_name, instructions)."""
  import pdfplumber
  from tinygrad.helpers import fetch

  pdf = CachedPDF(pdfplumber.open(fetch(url)))
  total_pages = len(pdf)

  # Auto-detect document type
  first_page = pdf.text(0)
  is_cdna4, is_cdna3 = 'CDNA4' in first_page or 'CDNA 4' in first_page, 'CDNA3' in first_page or 'MI300' in first_page
  is_cdna, is_rdna4 = is_cdna3 or is_cdna4, 'RDNA4' in first_page or 'RDNA 4' in first_page
  is_rdna35, is_rdna3 = 'RDNA3.5' in first_page or 'RDNA 3.5' in first_page, 'RDNA3' in first_page and 'RDNA3.5' not in first_page
  doc_name = "CDNA4" if is_cdna4 else "CDNA3" if is_cdna3 else "RDNA4" if is_rdna4 else "RDNA3.5" if is_rdna35 else "RDNA3" if is_rdna3 else "Unknown"

  # Find Microcode Formats section (for formats/enums)
  microcode_start = next((i for i in range(int(total_pages * 0.2), total_pages)
                          if re.search(r'\d+\.\d+\.\d+\.\s+SOP2\b|Chapter \d+\.\s+Microcode Formats', pdf.text(i))), int(total_pages * 0.9))
  # Find Instructions section (for pseudocode)
  instr_start = next((i for i in range(int(total_pages * 0.1), int(total_pages * 0.5))
                      if re.search(r'Chapter \d+\.\s+Instructions\b', pdf.text(i))), total_pages // 3)
  instr_end = next((i for start in [int(total_pages * 0.6), int(total_pages * 0.5), instr_start]
                    for i in range(start, min(start + 100, total_pages))
                    if re.search(r'Chapter \d+\.\s+Microcode Formats', pdf.text(i))), total_pages)

  # Parse src enum from SSRC encoding table
  src_enum = dict(SRC_EXTRAS)
  for i in range(microcode_start, min(microcode_start + 10, total_pages)):
    text = pdf.text(i)
    if 'SSRC0' in text and 'VCC_LO' in text:
      for m in re.finditer(r'^(\d+)\s+(\S+)', text, re.M):
        val, name = int(m.group(1)), m.group(2).rstrip('.:')
        if name in FLOAT_MAP: src_enum[val] = FLOAT_MAP[name]
        elif re.match(r'^[A-Z][A-Z0-9_]*$', name): src_enum[val] = name
      break

  # Parse opcode tables
  full_text = '\n'.join(pdf.text(i) for i in range(microcode_start, min(microcode_start + 50, total_pages)))
  enums: dict[str, dict[int, str]] = {}
  for m in re.finditer(r'Table \d+\. (\w+) Opcodes(.*?)(?=Table \d+\.|\n\d+\.\d+\.\d+\.\s+\w+\s*\nDescription|$)', full_text, re.S):
    if ops := {int(x.group(1)): x.group(2) for x in re.finditer(r'(\d+)\s+([A-Z][A-Z0-9_]+)', m.group(2))}:
      enums[m.group(1) + "Op"] = ops
  if vopd_m := re.search(r'Table \d+\. VOPD Y-Opcodes\n(.*?)(?=Table \d+\.|15\.\d)', full_text, re.S):
    if ops := {int(x.group(1)): x.group(2) for x in re.finditer(r'(\d+)\s+(V_DUAL_\w+)', vopd_m.group(1))}:
      enums["VOPDOp"] = ops
  enum_names = set(enums.keys())

  # Parse instruction formats
  def is_fields_table(t): return t and len(t) > 1 and t[0] and 'Field' in str(t[0][0] or '')
  def has_encoding(fields): return any(f[0] == 'ENCODING' for f in fields)
  def has_header_before_fields(text): return (pos := text.find('Field Name')) != -1 and bool(re.search(r'\d+\.\d+\.\d+\.\s+\w+\s*\n', text[:pos]))

  format_headers = []
  for i in range(50):
    if microcode_start + i >= total_pages: break
    text = pdf.text(microcode_start + i)
    for m in re.finditer(r'\d+\.\d+\.\d+\.\s+(\w+)\s*\n?Description', text): format_headers.append((m.group(1), i, m.start()))
    for m in re.finditer(r'\d+\.\d+\.\d+\.\s+(\w+)\s*\n', text):
      fmt_name = m.group(1)
      if is_cdna and fmt_name.isupper() and len(fmt_name) >= 2: format_headers.append((fmt_name, i, m.start()))
      elif m.start() > len(text) - 200 and 'Description' not in text[m.end():] and i + 1 < 50:
        next_text = pdf.text(microcode_start + i + 1).lstrip()
        if next_text.startswith('Description') or (next_text.startswith('"RDNA') and 'Description' in next_text[:200]):
          format_headers.append((fmt_name, i, m.start()))

  formats: dict[str, list] = {}
  for fmt_name, rel_idx, header_pos in format_headers:
    if fmt_name in formats: continue
    page_idx = microcode_start + rel_idx
    text = pdf.text(page_idx)
    field_pos = text.find('Field Name', header_pos)
    fields = None
    for offset in range(3):
      if page_idx + offset >= total_pages: break
      if offset > 0 and has_header_before_fields(pdf.text(page_idx + offset)): break
      for t in pdf.tables(page_idx + offset) if offset > 0 or field_pos > header_pos else []:
        if is_fields_table(t) and (f := _parse_fields_table(t, fmt_name, enum_names)) and has_encoding(f): fields = f; break
      if fields: break
    if not fields and field_pos > header_pos:
      for t in pdf.tables(page_idx):
        if is_fields_table(t) and (f := _parse_fields_table(t, fmt_name, enum_names)): fields = f; break
    if not fields: continue
    field_names = {f[0] for f in fields}
    for pg_offset in range(1, 3):
      if page_idx + pg_offset >= total_pages or has_header_before_fields(pdf.text(page_idx + pg_offset)): break
      for t in pdf.tables(page_idx + pg_offset):
        if is_fields_table(t) and (extra := _parse_fields_table(t, fmt_name, enum_names)) and not has_encoding(extra):
          for ef in extra:
            if ef[0] not in field_names: fields.append(ef); field_names.add(ef[0])
          break
    formats[fmt_name] = fields

  # Fix known PDF errors
  if 'SMEM' in formats:
    formats['SMEM'] = [(n, 13 if n == 'DLC' else 14 if n == 'GLC' else h, 13 if n == 'DLC' else 14 if n == 'GLC' else l, e, t)
                       for n, h, l, e, t in formats['SMEM']]
  if doc_name in ('RDNA3', 'RDNA3.5'):
    if 'SOPPOp' in enums: assert 8 not in enums['SOPPOp']; enums['SOPPOp'][8] = 'S_WAITCNT_DEPCTR'
    if 'DSOp' in enums:
      for k, v in {24: 'DS_GWS_SEMA_RELEASE_ALL', 25: 'DS_GWS_INIT', 26: 'DS_GWS_SEMA_V', 27: 'DS_GWS_SEMA_BR', 28: 'DS_GWS_SEMA_P', 29: 'DS_GWS_BARRIER'}.items():
        assert k not in enums['DSOp']; enums['DSOp'][k] = v
    if 'FLATOp' in enums:
      for k, v in {40: 'GLOBAL_LOAD_ADDTID_B32', 41: 'GLOBAL_STORE_ADDTID_B32', 55: 'FLAT_ATOMIC_CSUB_U32'}.items():
        assert k not in enums['FLATOp']; enums['FLATOp'][k] = v

  # Extract pseudocode for instructions
  all_text = '\n'.join(pdf.text(i) for i in range(instr_start, instr_end))
  matches = list(INST_PATTERN.finditer(all_text))
  raw_pseudocode: dict[tuple[str, int], str] = {}
  for i, match in enumerate(matches):
    name, opcode = match.group(1), int(match.group(2))
    start, end = match.end(), matches[i + 1].start() if i + 1 < len(matches) else match.end() + 2000
    snippet = all_text[start:end].strip()
    if pseudocode := _extract_pseudocode(snippet): raw_pseudocode[(name, opcode)] = pseudocode

  return {"formats": formats, "enums": enums, "src_enum": src_enum, "doc_name": doc_name, "pseudocode": raw_pseudocode, "is_cdna": is_cdna}

def _extract_pseudocode(text: str) -> str | None:
  """Extract pseudocode from an instruction description snippet."""
  lines, result, depth, in_lambda = text.split('\n'), [], 0, 0
  for line in lines:
    s = line.strip()
    if not s or re.match(r'^\d+ of \d+$', s) or re.match(r'^\d+\.\d+\..*Instructions', s): continue
    if s.startswith(('Notes', 'Functional examples', '•', '-')): break  # Stop at notes/bullets
    if s.startswith(('"RDNA', 'AMD ', 'CDNA')): continue
    if '•' in s or '–' in s: continue  # Skip lines with bullets/dashes
    if '= lambda(' in s: in_lambda += 1; continue
    if in_lambda > 0:
      if s.endswith(');'): in_lambda -= 1
      continue
    if s.startswith('if '): depth += 1
    elif s.startswith('endif'): depth = max(0, depth - 1)
    if s.endswith('.') and not any(p in s for p in ['D0', 'D1', 'S0', 'S1', 'S2', 'SCC', 'VCC', 'tmp', '=']): continue
    if re.match(r'^[a-z].*\.$', s) and '=' not in s: continue
    is_code = (any(p in s for p in ['D0.', 'D1.', 'S0.', 'S1.', 'S2.', 'SCC =', 'SCC ?', 'VCC', 'EXEC', 'tmp =', 'tmp[', 'lane =', 'PC =',
                                    'D0[', 'D1[', 'S0[', 'S1[', 'S2[', 'MEM[', 'RETURN_DATA',
                                    'VADDR', 'VDATA', 'VDST', 'SADDR', 'OFFSET']) or
               s.startswith(('if ', 'else', 'elsif', 'endif', 'declare ', 'for ', 'endfor', '//')) or
               re.match(r'^[a-z_]+\s*=', s) or re.match(r'^[a-z_]+\[', s) or (depth > 0 and '=' in s))
    if is_code: result.append(s)
  return '\n'.join(result) if result else None

def _merge_results(results: list[dict]) -> dict:
  """Merge multiple PDF parse results into a superset."""
  merged = {"formats": {}, "enums": {}, "src_enum": dict(SRC_EXTRAS), "doc_names": [], "pseudocode": {}, "is_cdna": False}
  for r in results:
    merged["doc_names"].append(r["doc_name"])
    merged["is_cdna"] = merged["is_cdna"] or r["is_cdna"]
    for val, name in r["src_enum"].items():
      if val in merged["src_enum"]: assert merged["src_enum"][val] == name
      else: merged["src_enum"][val] = name
    for enum_name, ops in r["enums"].items():
      if enum_name not in merged["enums"]: merged["enums"][enum_name] = {}
      for val, name in ops.items():
        if val in merged["enums"][enum_name]: assert merged["enums"][enum_name][val] == name
        else: merged["enums"][enum_name][val] = name
    for fmt_name, fields in r["formats"].items():
      if fmt_name not in merged["formats"]: merged["formats"][fmt_name] = list(fields)
      else:
        existing = {f[0]: (f[1], f[2]) for f in merged["formats"][fmt_name]}
        for f in fields:
          if f[0] in existing: assert existing[f[0]] == (f[1], f[2])
          else: merged["formats"][fmt_name].append(f)
    for key, pc in r["pseudocode"].items():
      if key not in merged["pseudocode"]: merged["pseudocode"][key] = pc
  return merged

# ═══════════════════════════════════════════════════════════════════════════════
# CODE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_enum_py(enums, src_enum, doc_name) -> str:
  """Generate enum.py content (just enums, no dsl.py dependency)."""
  def enum_lines(name, items): return [f"class {name}(IntEnum):"] + [f"  {n} = {v}" for v, n in sorted(items.items())] + [""]
  lines = [f"# autogenerated from AMD {doc_name} ISA PDF by pdf.py - do not edit", "from enum import IntEnum", ""]
  lines += enum_lines("SrcEnum", src_enum) + sum([enum_lines(n, ops) for n, ops in sorted(enums.items())], [])
  return '\n'.join(lines)

def _generate_ins_py(formats, enums, src_enum, doc_name) -> str:
  """Generate ins.py content (instruction formats and helpers, imports dsl.py and enum.py)."""
  def field_key(f, order): return order.index(f[0].lower()) if f[0].lower() in order else 1000
  lines = [f"# autogenerated from AMD {doc_name} ISA PDF by pdf.py - do not edit",
           "# ruff: noqa: F401,F403", "from typing import Annotated",
           "from extra.assembly.amd.dsl import bits, BitField, Inst32, Inst64, SGPR, VGPR, TTMP as TTMP, s as s, v as v, ttmp as ttmp, SSrc, Src, SImm, Imm, VDSTYEnc, SGPRField, VGPRField",
           "from extra.assembly.amd.autogen.{arch}.enum import *",
           "import functools", ""]
  format_defaults = {'VOP3P': {'opsel_hi': 3, 'opsel_hi2': 1}}
  lines.append("# instruction formats")
  for fmt_name, fields in sorted(formats.items()):
    base = "Inst64" if max(f[1] for f in fields) > 31 or fmt_name == 'VOP3SD' else "Inst32"
    order = FIELD_ORDER.get(fmt_name, [])
    lines.append(f"class {fmt_name}({base}):")
    if enc := next((f for f in fields if f[0] == 'ENCODING'), None):
      lines.append(f"  encoding = bits[{enc[1]}:{enc[2]}] == 0b{enc[3]:b}" if enc[1] != enc[2] else f"  encoding = bits[{enc[1]}] == {enc[3]}")
    if defaults := format_defaults.get(fmt_name): lines.append(f"  _defaults = {defaults}")
    for name, hi, lo, _, ftype in sorted([f for f in fields if f[0] != 'ENCODING'], key=lambda f: field_key(f, order)):
      ann = f":Annotated[BitField, {ftype}]" if ftype and ftype.endswith('Op') else f":{ftype}" if ftype else ""
      lines.append(f"  {name.lower()}{ann} = bits[{hi}]" if hi == lo else f"  {name.lower()}{ann} = bits[{hi}:{lo}]")
    lines.append("")
  lines.append("# instruction helpers")
  for cls_name, ops in sorted(enums.items()):
    fmt = cls_name[:-2]
    for op_val, name in sorted(ops.items()):
      seg = {"GLOBAL": ", seg=2", "SCRATCH": ", seg=1"}.get(fmt, "")
      tgt = {"GLOBAL": "FLAT, GLOBALOp", "SCRATCH": "FLAT, SCRATCHOp"}.get(fmt, f"{fmt}, {cls_name}")
      if fmt in formats or fmt in ("GLOBAL", "SCRATCH"):
        suffix = "_e32" if fmt in ("VOP1", "VOP2", "VOPC") else "_e64" if fmt == "VOP3" and op_val < 512 else ""
        if name in ('V_FMAMK_F32', 'V_FMAMK_F16'):
          lines.append(f"def {name.lower()}{suffix}(vdst, src0, K, vsrc1): return {fmt}({cls_name}.{name}, vdst, src0, vsrc1, literal=K)")
        elif name in ('V_FMAAK_F32', 'V_FMAAK_F16'):
          lines.append(f"def {name.lower()}{suffix}(vdst, src0, vsrc1, K): return {fmt}({cls_name}.{name}, vdst, src0, vsrc1, literal=K)")
        else: lines.append(f"{name.lower()}{suffix} = functools.partial({tgt}.{name}{seg})")
  src_names = {name for _, name in src_enum.items()}
  lines += [""] + [f"{name} = SrcEnum.{name}" for _, name in sorted(src_enum.items()) if name not in {'DPP8', 'DPP16'}]
  if "NULL" in src_names: lines.append("OFF = NULL\n")
  return '\n'.join(lines)

def _generate_gen_pcode_py(enums, pseudocode, arch) -> str:
  """Generate gen_pcode.py content (compiled pseudocode functions)."""
  # Get op enums for this arch (import from .ins which re-exports from .enum)
  import importlib
  autogen = importlib.import_module(f"extra.assembly.amd.autogen.{arch}.ins")
  OP_ENUMS = [getattr(autogen, name) for name in ['SOP1Op', 'SOP2Op', 'SOPCOp', 'SOPKOp', 'SOPPOp', 'VOP1Op', 'VOP2Op', 'VOP3Op', 'VOP3SDOp', 'VOP3POp', 'VOPCOp', 'VOP3AOp', 'VOP3BOp', 'DSOp', 'FLATOp', 'GLOBALOp', 'SCRATCHOp'] if hasattr(autogen, name)]

  # Build defined ops mapping
  defined_ops: dict[tuple, list] = {}
  for enum_cls in OP_ENUMS:
    for op in enum_cls:
      if op.name.startswith(('S_', 'V_', 'DS_', 'FLAT_', 'GLOBAL_', 'SCRATCH_')): defined_ops.setdefault((op.name, op.value), []).append((enum_cls, op))

  enum_names = [e.__name__ for e in OP_ENUMS]
  lines = [f'''# autogenerated by pdf.py - do not edit
# to regenerate: python -m extra.assembly.amd.pdf --arch {arch}
# ruff: noqa: E501,F405,F403
# mypy: ignore-errors
from extra.assembly.amd.autogen.{arch}.enum import {", ".join(enum_names)}
from extra.assembly.amd.pcode import *
''']

  instructions: dict = {cls: {} for cls in OP_ENUMS}
  for key, pc in pseudocode.items():
    if key in defined_ops:
      for enum_cls, enum_val in defined_ops[key]: instructions[enum_cls][enum_val] = pc

  for enum_cls in OP_ENUMS:
    cls_name = enum_cls.__name__
    if not instructions.get(enum_cls): continue
    fn_entries = []
    for op, pc in instructions[enum_cls].items():
      if any(p in pc for p in UNSUPPORTED): continue
      try:
        code = compile_pseudocode(pc)
        code = _apply_pseudocode_fixes(op, code)
        fn_name, fn_code = _generate_function(cls_name, op, pc, code)
        lines.append(fn_code)
        fn_entries.append((op, fn_name))
      except Exception as e: print(f"  Warning: Failed to compile {op.name}: {e}")
    if fn_entries:
      lines.append(f'{cls_name}_FUNCTIONS = {{')
      for op, fn_name in fn_entries: lines.append(f"  {cls_name}.{op.name}: {fn_name},")
      lines.append('}\n')

  # Add V_WRITELANE_B32 if VOP3Op exists
  if 'VOP3Op' in enum_names:
    lines.append('''
# V_WRITELANE_B32: Write scalar to specific lane's VGPR (not in PDF pseudocode)
def _VOP3Op_V_WRITELANE_B32(s0, s1, s2, d0, scc, vcc, lane, exec_mask, literal, VGPR, _vars, src0_idx=0, vdst_idx=0):
  wr_lane = s1 & 0x1f
  return {'d0': d0, 'scc': scc, 'vgpr_write': (wr_lane, vdst_idx, s0 & 0xffffffff)}
VOP3Op_FUNCTIONS[VOP3Op.V_WRITELANE_B32] = _VOP3Op_V_WRITELANE_B32
''')

  lines.append('COMPILED_FUNCTIONS = {')
  for enum_cls in OP_ENUMS:
    if instructions.get(enum_cls): lines.append(f'  {enum_cls.__name__}: {enum_cls.__name__}_FUNCTIONS,')
  lines.append('}\n\ndef get_compiled_functions(): return COMPILED_FUNCTIONS')
  return '\n'.join(lines)

def _apply_pseudocode_fixes(op, code: str) -> str:
  """Apply known fixes for PDF pseudocode bugs."""
  if op.name == 'V_DIV_FMAS_F32':
    code = code.replace('D0.f32 = 2.0 ** 32 * fma(S0.f32, S1.f32, S2.f32)',
                        'D0.f32 = (2.0 ** 64 if exponent(S2.f32) > 127 else 2.0 ** -64) * fma(S0.f32, S1.f32, S2.f32)')
  if op.name == 'V_DIV_FMAS_F64':
    code = code.replace('D0.f64 = 2.0 ** 64 * fma(S0.f64, S1.f64, S2.f64)',
                        'D0.f64 = (2.0 ** 128 if exponent(S2.f64) > 1023 else 2.0 ** -128) * fma(S0.f64, S1.f64, S2.f64)')
  if op.name == 'V_DIV_SCALE_F32':
    code = code.replace('D0.f32 = float("nan")', 'VCC = Reg(0x1); D0.f32 = float("nan")')
    code = code.replace('elif S1.f32 == DENORM.f32:\n  D0.f32 = ldexp(S0.f32, 64)', 'elif False:\n  pass')
    code += '\nif S1.f32 == DENORM.f32:\n  D0.f32 = float("nan")'
    code = code.replace('elif exponent(S2.f32) <= 23:\n  D0.f32 = ldexp(S0.f32, 64)', 'elif exponent(S2.f32) <= 23:\n  VCC = Reg(0x1); D0.f32 = ldexp(S0.f32, 64)')
    code = code.replace('elif S2.f32 / S1.f32 == DENORM.f32:\n  VCC = Reg(0x1)\n  if S0.f32 == S2.f32:\n    D0.f32 = ldexp(S0.f32, 64)', 'elif S2.f32 / S1.f32 == DENORM.f32:\n  VCC = Reg(0x1)')
  if op.name == 'V_DIV_SCALE_F64':
    code = code.replace('D0.f64 = float("nan")', 'VCC = Reg(0x1); D0.f64 = float("nan")')
    code = code.replace('elif S1.f64 == DENORM.f64:\n  D0.f64 = ldexp(S0.f64, 128)', 'elif False:\n  pass')
    code += '\nif S1.f64 == DENORM.f64:\n  D0.f64 = float("nan")'
    code = code.replace('elif exponent(S2.f64) <= 52:\n  D0.f64 = ldexp(S0.f64, 128)', 'elif exponent(S2.f64) <= 52:\n  VCC = Reg(0x1); D0.f64 = ldexp(S0.f64, 128)')
    code = code.replace('elif S2.f64 / S1.f64 == DENORM.f64:\n  VCC = Reg(0x1)\n  if S0.f64 == S2.f64:\n    D0.f64 = ldexp(S0.f64, 128)', 'elif S2.f64 / S1.f64 == DENORM.f64:\n  VCC = Reg(0x1)')
  if op.name == 'V_DIV_FIXUP_F32':
    code = code.replace('D0.f32 = ((-abs(S0.f32)) if (sign_out) else (abs(S0.f32)))',
                        'D0.f32 = ((-OVERFLOW_F32) if (sign_out) else (OVERFLOW_F32)) if isNAN(S0.f32) else ((-abs(S0.f32)) if (sign_out) else (abs(S0.f32)))')
  if op.name == 'V_DIV_FIXUP_F64':
    code = code.replace('D0.f64 = ((-abs(S0.f64)) if (sign_out) else (abs(S0.f64)))',
                        'D0.f64 = ((-OVERFLOW_F64) if (sign_out) else (OVERFLOW_F64)) if isNAN(S0.f64) else ((-abs(S0.f64)) if (sign_out) else (abs(S0.f64)))')
  if op.name == 'V_TRIG_PREOP_F64':
    code = code.replace('result = F((TWO_OVER_PI_1201[1200 : 0] << shift.u32) & 0x1fffffffffffff)',
                        'result = float(((TWO_OVER_PI_1201[1200 : 0] << int(shift)) >> (1201 - 53)) & 0x1fffffffffffff)')
  return code

def _generate_function(cls_name: str, op, pc: str, code: str) -> tuple[str, str]:
  """Generate a single compiled pseudocode function."""
  has_d1 = '{ D1' in pc
  is_cmpx = (cls_name in ('VOPCOp', 'VOP3Op')) and 'EXEC.u64[laneId]' in pc
  is_div_scale = 'DIV_SCALE' in op.name
  has_sdst = cls_name == 'VOP3SDOp' and ('VCC.u64[laneId]' in pc or is_div_scale)
  is_ds = cls_name == 'DSOp'
  is_flat = cls_name in ('FLATOp', 'GLOBALOp', 'SCRATCHOp')
  combined = code + pc

  fn_name = f"_{cls_name}_{op.name}"
  # Function accepts Reg objects directly (uppercase names), laneId is passed directly as int
  # DSOp functions get additional MEM and offset parameters
  # FLAT/GLOBAL ops get MEM, vaddr, vdata, saddr, offset parameters
  if is_ds:
    lines = [f"def {fn_name}(MEM, ADDR, DATA0, DATA1, OFFSET0, OFFSET1, RETURN_DATA):"]
  elif is_flat:
    lines = [f"def {fn_name}(MEM, ADDR, VDATA, VDST, RETURN_DATA):"]
  else:
    lines = [f"def {fn_name}(S0, S1, S2, D0, SCC, VCC, laneId, EXEC, literal, VGPR, src0_idx=0, vdst_idx=0, PC=None):"]

  # Registers that need special handling (aliases or init)
  def needs_init(name): return name in combined and not re.search(rf'^\s*{name}\s*=\s*Reg\(', code, re.MULTILINE)
  special_regs = []
  if is_ds: special_regs = [('DATA', 'DATA0'), ('DATA2', 'DATA1'), ('OFFSET', 'OFFSET0'), ('ADDR_BASE', 'ADDR')]
  elif is_flat: special_regs = [('DATA', 'VDATA')]
  else:
    special_regs = [('D1', 'Reg(0)'), ('SIMM16', 'Reg(literal)'), ('SIMM32', 'Reg(literal)'),
                    ('SRC0', 'Reg(src0_idx)'), ('VDST', 'Reg(vdst_idx)')]
    if needs_init('tmp'): special_regs.insert(0, ('tmp', 'Reg(0)'))
    if needs_init('saveexec'): special_regs.insert(0, ('saveexec', 'Reg(EXEC._val)'))

  used = {name for name, _ in special_regs if name in combined}

  # Detect which registers are modified (not just read) - look for assignments
  modifies_d0 = is_div_scale or bool(re.search(r'\bD0\b[.\[]', combined))
  modifies_exec = is_cmpx or bool(re.search(r'EXEC\.(u32|u64|b32|b64)\s*=', combined))
  modifies_vcc = has_sdst or bool(re.search(r'VCC\.(u32|u64|b32|b64)\s*=|VCC\.u64\[laneId\]\s*=', combined))
  modifies_scc = bool(re.search(r'\bSCC\s*=', combined))
  modifies_pc = bool(re.search(r'\bPC\s*=', combined))
  # DS/FLAT ops: detect memory writes (MEM[...] = ...)
  modifies_mem = (is_ds or is_flat) and bool(re.search(r'MEM\[.*\]\.[a-z0-9]+\s*=', combined))
  # FLAT ops: detect VDST writes
  modifies_vdst = is_flat and bool(re.search(r'VDST[\.\[].*=', combined))

  # Build init code for special registers
  init_lines = []
  if is_div_scale: init_lines.append("  D0 = Reg(S0._val)")
  for name, init in special_regs:
    if name in used: init_lines.append(f"  {name} = {init}")
  if 'EXEC_LO' in code: init_lines.append("  EXEC_LO = SliceProxy(EXEC, 31, 0)")
  if 'EXEC_HI' in code: init_lines.append("  EXEC_HI = SliceProxy(EXEC, 63, 32)")
  if 'VCCZ' in code and not re.search(r'^\s*VCCZ\s*=', code, re.MULTILINE): init_lines.append("  VCCZ = Reg(1 if VCC._val == 0 else 0)")
  if 'EXECZ' in code and not re.search(r'^\s*EXECZ\s*=', code, re.MULTILINE): init_lines.append("  EXECZ = Reg(1 if EXEC._val == 0 else 0)")
  code_lines = [line for line in code.split('\n') if line.strip()]
  if init_lines:
    lines.extend(init_lines)
    if code_lines: lines.append("  # --- compiled pseudocode ---")
  for line in code_lines:
    lines.append(f"  {line}")

  # Build result dict - only include registers that are modified
  result_items = []
  if modifies_d0: result_items.append("'D0': D0")
  if modifies_scc: result_items.append("'SCC': SCC")
  if modifies_vcc: result_items.append("'VCC': VCC")
  if modifies_exec: result_items.append("'EXEC': EXEC")
  if has_d1: result_items.append("'D1': D1")
  if modifies_pc: result_items.append("'PC': PC")
  # DS ops: return RETURN_DATA if it was written (left side of assignment)
  if is_ds and 'RETURN_DATA' in combined and re.search(r'^\s*RETURN_DATA[\.\[].*=', code, re.MULTILINE):
    result_items.append("'RETURN_DATA': RETURN_DATA")
  # FLAT ops: return RETURN_DATA for atomics, VDATA for loads (only if written to)
  if is_flat:
    if 'RETURN_DATA' in combined and re.search(r'^\s*RETURN_DATA[\.\[].*=', code, re.MULTILINE):
      result_items.append("'RETURN_DATA': RETURN_DATA")
    if re.search(r'^\s*VDATA[\.\[].*=', code, re.MULTILINE):
      result_items.append("'VDATA': VDATA")
  lines.append(f"  return {{{', '.join(result_items)}}}\n")
  return fn_name, '\n'.join(lines)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_arch(arch: str) -> dict:
  """Generate enum.py, ins.py and gen_pcode.py for a single architecture."""
  urls = PDF_URLS[arch]
  if isinstance(urls, str): urls = [urls]

  print(f"\n{'='*60}\nGenerating {arch}...")
  print(f"Parsing {len(urls)} PDF(s)...")
  results = [_parse_single_pdf(url) for url in urls]
  merged = _merge_results(results) if len(results) > 1 else results[0]
  doc_name = "+".join(merged["doc_names"]) if len(results) > 1 else merged["doc_name"]

  base_path = Path(f"extra/assembly/amd/autogen/{arch}")
  base_path.mkdir(parents=True, exist_ok=True)
  (base_path / "__init__.py").touch()

  # Write enum.py (enums only, no dsl.py dependency)
  enum_path = base_path / "enum.py"
  enum_content = _generate_enum_py(merged["enums"], merged["src_enum"], doc_name)
  enum_path.write_text(enum_content)
  print(f"Generated {enum_path}: SrcEnum ({len(merged['src_enum'])}) + {len(merged['enums'])} enums")

  # Write ins.py (instruction formats and helpers, imports dsl.py and enum.py)
  ins_path = base_path / "ins.py"
  ins_content = _generate_ins_py(merged["formats"], merged["enums"], merged["src_enum"], doc_name).replace("{arch}", arch)
  ins_path.write_text(ins_content)
  print(f"Generated {ins_path}: {len(merged['formats'])} formats")

  # Write gen_pcode.py (needs enum.py to exist first for imports)
  pcode_path = base_path / "gen_pcode.py"
  pcode_content = _generate_gen_pcode_py(merged["enums"], merged["pseudocode"], arch)
  pcode_path.write_text(pcode_content)
  print(f"Generated {pcode_path}: {len(merged['pseudocode'])} instructions")

  return merged

def _generate_arch_wrapper(arch: str):
  """Wrapper for multiprocessing - returns arch name for ordering."""
  generate_arch(arch)
  return arch

def generate_all():
  """Generate all architectures in parallel."""
  with ProcessPoolExecutor() as executor:
    list(executor.map(_generate_arch_wrapper, PDF_URLS.keys()))

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description="Generate AMD ISA autogen files from PDF documentation")
  parser.add_argument("--arch", choices=list(PDF_URLS.keys()) + ["all"], default="rdna3")
  args = parser.parse_args()
  if args.arch == "all": generate_all()
  else: generate_arch(args.arch)
