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
      desc = row[2]
      # Handle shared FLAT/GLOBAL/SCRATCH table: look for format-specific encoding
      fmt_key = fmt.lstrip('V').lower().capitalize()  # VFLAT -> Flat, VGLOBAL -> Global
      if m := re.search(rf"{fmt_key}='b([01_]+)", desc):
        enc_bits = m.group(1).replace('_', '')
      elif m := re.search(r"(?:'b|Must be:\s*)([01_]+)", desc):
        enc_bits = m.group(1).replace('_', '')
      else:
        enc_bits = None
      if enc_bits:
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
    # RDNA4: Look for "Table X. Y Fields" patterns (e.g., VIMAGE, VSAMPLE, or shared FLAT/GLOBAL/SCRATCH)
    for m in re.finditer(r'Table \d+\.\s+([\w,\s]+?)\s+Fields', text):
      table_name = m.group(1).strip()
      # Handle shared table like "FLAT, GLOBAL and SCRATCH"
      if ',' in table_name or ' and ' in table_name:
        for part in re.split(r',\s*|\s+and\s+', table_name):
          fmt_name = 'V' + part.strip()
          if fmt_name not in [h[0] for h in format_headers]: format_headers.append((fmt_name, i, m.start()))
      elif table_name.startswith('V'):
        if table_name not in [h[0] for h in format_headers]: format_headers.append((table_name, i, m.start()))

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
  # RDNA4: VFLAT/VGLOBAL/VSCRATCH OP field is [20:14] not [20:13] (PDF documentation error)
  for fmt_name in ['VFLAT', 'VGLOBAL', 'VSCRATCH']:
    if fmt_name in formats:
      formats[fmt_name] = [(n, h, 14 if n == 'OP' else l, e, t) for n, h, l, e, t in formats[fmt_name]]
  if doc_name in ('RDNA3', 'RDNA3.5'):
    if 'SOPPOp' in enums:
      for k, v in {8: 'S_WAITCNT_DEPCTR', 58: 'S_TTRACEDATA', 59: 'S_TTRACEDATA_IMM'}.items():
        assert k not in enums['SOPPOp']; enums['SOPPOp'][k] = v
    if 'SOPKOp' in enums:
      for k, v in {22: 'S_SUBVECTOR_LOOP_BEGIN', 23: 'S_SUBVECTOR_LOOP_END'}.items():
        assert k not in enums['SOPKOp']; enums['SOPKOp'][k] = v
    if 'SMEMOp' in enums:
      for k, v in {34: 'S_ATC_PROBE', 35: 'S_ATC_PROBE_BUFFER'}.items():
        assert k not in enums['SMEMOp']; enums['SMEMOp'][k] = v
    if 'DSOp' in enums:
      for k, v in {24: 'DS_GWS_SEMA_RELEASE_ALL', 25: 'DS_GWS_INIT', 26: 'DS_GWS_SEMA_V', 27: 'DS_GWS_SEMA_BR', 28: 'DS_GWS_SEMA_P', 29: 'DS_GWS_BARRIER'}.items():
        assert k not in enums['DSOp']; enums['DSOp'][k] = v
    if 'FLATOp' in enums:
      for k, v in {40: 'GLOBAL_LOAD_ADDTID_B32', 41: 'GLOBAL_STORE_ADDTID_B32', 55: 'FLAT_ATOMIC_CSUB_U32'}.items():
        assert k not in enums['FLATOp']; enums['FLATOp'][k] = v
  # CDNA SDWA/DPP: PDF only has modifier fields, need VOP1/VOP2 overlay for correct encoding
  if is_cdna:
    if 'SDWA' in formats:
      formats['SDWA'] = [('ENCODING', 8, 0, 0xf9, None), ('VOP_OP', 16, 9, None, None), ('VDST', 24, 17, None, 'VGPRField'), ('VOP2_OP', 31, 25, None, None)] + \
                        [f for f in formats['SDWA'] if f[0] not in ('ENCODING', 'SDST', 'SD', 'ROW_MASK')]
    if 'DPP' in formats:
      formats['DPP'] = [('ENCODING', 8, 0, 0xfa, None), ('VOP_OP', 16, 9, None, None), ('VDST', 24, 17, None, 'VGPRField'), ('VOP2_OP', 31, 25, None, None),
        ('SRC0', 39, 32, None, 'Src'), ('DPP_CTRL', 48, 40, None, None), ('BOUND_CTRL', 51, 51, None, None), ('SRC0_NEG', 52, 52, None, None), ('SRC0_ABS', 53, 53, None, None),
        ('SRC1_NEG', 54, 54, None, None), ('SRC1_ABS', 55, 55, None, None), ('BANK_MASK', 59, 56, None, None), ('ROW_MASK', 63, 60, None, None)]

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
           "from extra.assembly.amd.dsl import bits, BitField, Inst32, Inst64, Inst96, SGPR, VGPR, TTMP as TTMP, s as s, v as v, ttmp as ttmp, SSrc, Src, SImm, Imm, VDSTYEnc, SGPRField, VGPRField",
           "from extra.assembly.amd.autogen.{arch}.enum import *",
           "import functools", ""]
  format_defaults = {'VOP3P': {'opsel_hi': 3, 'opsel_hi2': 1}}
  lines.append("# instruction formats")
  # MIMG has optional NSA (Non-Sequential Address) fields that extend beyond 64 bits, but base encoding is 64-bit
  inst64_override = {'MIMG'}
  for fmt_name, fields in sorted(formats.items()):
    max_bit = max(f[1] for f in fields)
    if fmt_name in inst64_override: base = "Inst64"
    else: base = "Inst96" if max_bit > 63 else "Inst64" if max_bit > 31 or fmt_name == 'VOP3SD' else "Inst32"
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

def _generate_str_pcode_py(enums, pseudocode, arch) -> str:
  """Generate str_pcode.py content (raw pseudocode strings)."""
  # Get op enums for this arch (import from .ins which re-exports from .enum)
  import importlib
  autogen = importlib.import_module(f"extra.assembly.amd.autogen.{arch}.ins")
  OP_ENUMS = [getattr(autogen, name) for name in ['SOP1Op', 'SOP2Op', 'SOPCOp', 'SOPKOp', 'SOPPOp', 'SMEMOp', 'VOP1Op', 'VOP2Op', 'VOP3Op', 'VOP3SDOp', 'VOP3POp', 'VOPCOp', 'VOP3AOp', 'VOP3BOp', 'DSOp', 'FLATOp', 'GLOBALOp', 'SCRATCHOp'] if hasattr(autogen, name)]

  # Build defined ops mapping
  defined_ops: dict[tuple, list] = {}
  for enum_cls in OP_ENUMS:
    for op in enum_cls:
      if op.name.startswith(('S_', 'V_', 'DS_', 'FLAT_', 'GLOBAL_', 'SCRATCH_')): defined_ops.setdefault((op.name, op.value), []).append((enum_cls, op))

  enum_names = [e.__name__ for e in OP_ENUMS]
  instructions: dict = {cls: {} for cls in OP_ENUMS}
  for key, pc in pseudocode.items():
    if key in defined_ops:
      for enum_cls, enum_val in defined_ops[key]: instructions[enum_cls][enum_val] = pc

  # Build string dictionaries for each enum
  lines = [f'''# autogenerated by pdf.py - do not edit
# to regenerate: python -m extra.assembly.amd.pdf --arch {arch}
# ruff: noqa: E501
from extra.assembly.amd.autogen.{arch}.enum import {", ".join(enum_names)}
''']
  all_dict_entries: dict = {}
  for enum_cls in OP_ENUMS:
    cls_name = enum_cls.__name__
    if not instructions.get(enum_cls): continue
    dict_entries = [(op, repr(pc)) for op, pc in instructions[enum_cls].items()]
    if dict_entries:
      all_dict_entries[enum_cls] = dict_entries
      lines.append(f'{cls_name}_PCODE = {{')
      for op, escaped in dict_entries: lines.append(f"  {cls_name}.{op.name}: {escaped},")
      lines.append('}\n')

  lines.append('PSEUDOCODE_STRINGS = {')
  for enum_cls in OP_ENUMS:
    if all_dict_entries.get(enum_cls): lines.append(f'  {enum_cls.__name__}: {enum_cls.__name__}_PCODE,')
  lines.append('}')
  return '\n'.join(lines)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_arch(arch: str) -> dict:
  """Generate enum.py, ins.py and str_pcode.py for a single architecture."""
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

  # Write str_pcode.py (needs enum.py to exist first for imports)
  pcode_path = base_path / "str_pcode.py"
  pcode_content = _generate_str_pcode_py(merged["enums"], merged["pseudocode"], arch)
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
