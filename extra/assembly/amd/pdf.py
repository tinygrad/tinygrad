# Generic PDF text extractor - no external dependencies
import re, zlib
from tinygrad.helpers import fetch, merge_dicts

PDF_URLS = {
  "rdna3": "https://docs.amd.com/api/khub/documents/UVVZM22UN7tMUeiW_4ShTQ/content",
  "rdna4": "https://docs.amd.com/api/khub/documents/uQpkEvk3pv~kfAb2x~j4uw/content",
  "cdna": "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf",
}

# ═══════════════════════════════════════════════════════════════════════════════
# Generic PDF extraction tools
# ═══════════════════════════════════════════════════════════════════════════════

def extract(url: str) -> list[list[tuple[float, float, str, str]]]:
  """Extract positioned text from PDF. Returns list of text elements (x, y, text, font) per page."""
  data = fetch(url).read_bytes()

  # Parse xref table to locate objects
  xref: dict[int, int] = {}
  pos = int(re.search(rb'startxref\s+(\d+)', data).group(1)) + 4
  while data[pos:pos+7] != b'trailer':
    while data[pos:pos+1] in b' \r\n': pos += 1
    line_end = data.find(b'\n', pos)
    start_obj, count = map(int, data[pos:line_end].split()[:2])
    pos = line_end + 1
    for i in range(count):
      if data[pos+17:pos+18] == b'n' and (off := int(data[pos:pos+10])) > 0: xref[start_obj + i] = off
      pos += 20

  def get_stream(n: int) -> bytes:
    obj = data[xref[n]:data.find(b'endobj', xref[n])]
    raw = obj[obj.find(b'stream\n') + 7:obj.find(b'\nendstream')]
    return zlib.decompress(raw) if b'/FlateDecode' in obj else raw

  # Find page content streams and extract text
  pages = []
  for n in sorted(xref):
    if b'/Type /Page' not in data[xref[n]:xref[n]+500]: continue
    if not (m := re.search(rb'/Contents (\d+) 0 R', data[xref[n]:xref[n]+500])): continue
    stream = get_stream(int(m.group(1))).decode('latin-1')
    elements, font = [], ''
    for bt in re.finditer(r'BT(.*?)ET', stream, re.S):
      x, y = 0.0, 0.0
      for m in re.finditer(r'(/F[\d.]+) [\d.]+ Tf|([\d.+-]+) ([\d.+-]+) Td|[\d.+-]+ [\d.+-]+ [\d.+-]+ [\d.+-]+ ([\d.+-]+) ([\d.+-]+) Tm|<([0-9A-Fa-f]+)>.*?Tj|\[([^\]]+)\] TJ', bt.group(1)):
        if m.group(1): font = m.group(1)
        elif m.group(2): x, y = x + float(m.group(2)), y + float(m.group(3))
        elif m.group(4): x, y = float(m.group(4)), float(m.group(5))
        elif m.group(6) and (t := bytes.fromhex(m.group(6)).decode('latin-1')).strip(): elements.append((x, y, t, font))
        elif m.group(7) and (t := ''.join(bytes.fromhex(h).decode('latin-1') for h in re.findall(r'<([0-9A-Fa-f]+)>', m.group(7)))).strip(): elements.append((x, y, t, font))
    pages.append(sorted(elements, key=lambda e: (-e[1], e[0])))
  return pages

def extract_tables(pages: list[list[tuple[float, float, str, str]]]) -> dict[int, tuple[str, list[list[str]]]]:
  """Extract numbered tables from PDF pages. Returns {table_num: (title, rows)} where rows is list of cells per row."""
  def group_by_y(texts, key=lambda y: round(y)):
    by_y: dict[int, list[tuple[float, float, str]]] = {}
    for x, y, t, _ in texts:
      by_y.setdefault(key(y), []).append((x, y, t))
    return by_y

  # Find all table headers by merging text on same line
  table_positions = []
  for page_idx, texts in enumerate(pages):
    for items in group_by_y(texts).values():
      line = ''.join(t for _, t in sorted((x, t) for x, _, t in items))
      if m := re.search(r'Table (\d+)\. (.+)', line):
        table_positions.append((int(m.group(1)), m.group(2).strip(), page_idx, items[0][1]))
  table_positions.sort(key=lambda t: (t[2], -t[3]))

  # For each table, find rows with matching X positions
  result: dict[int, tuple[str, list[list[str]]]] = {}
  for num, title, start_page, header_y in table_positions:
    rows, col_xs = [], None
    for page_idx in range(start_page, len(pages)):
      page_texts = [(x, y, t) for x, y, t, _ in pages[page_idx] if 30 < y < 760 and (page_idx > start_page or y < header_y)]
      for items in sorted(group_by_y([(x, y, t, '') for x, y, t in page_texts], key=lambda y: round(y / 5)).values(), key=lambda items: -items[0][1]):
        xs = tuple(sorted(round(x) for x, _, _ in items))
        if col_xs is None:
          if len(xs) < 2: continue  # Skip single-column rows before table starts
          col_xs = xs
        elif len(xs) == 1 and xs[0] in col_xs: continue  # Skip continuation rows at known column positions
        elif not any(c in xs for c in col_xs[:2]): break  # Row missing first columns = end of table
        rows.append([t for _, t in sorted((x, t) for x, _, t in items)])
      else: continue
      break
    if rows: result[num] = (title, rows)
  return result

# ═══════════════════════════════════════════════════════════════════════════════
# AMD specific extraction
# ═══════════════════════════════════════════════════════════════════════════════

def extract_enums(tables: dict[int, tuple[str, list[list[str]]]], arch: str) -> dict[str, dict[int, str]]:
  """Extract all enums from tables. Returns {enum_name: {value: name}}."""
  enums: dict[str, dict[int, str]] = {}
  for num, (title, rows) in tables.items():
    # Opcode enums from "XXX Opcodes" tables
    if m := re.match(r'(\w+) (?:Y-)?Opcodes', title):
      fmt_name = 'VOPD' if 'Y-Opcodes' in title else m.group(1)
      ops: dict[int, str] = {}
      for row in rows:
        for i in range(0, len(row) - 1, 2):
          if row[i].isdigit() and re.match(r'^[A-Z][A-Z0-9_]+$', row[i + 1]):
            ops[int(row[i])] = row[i + 1]
      if ops: enums[fmt_name] = ops
    # BufFmt from "Data Format" tables
    if 'Data Format' in title:
      for row in rows:
        for i in range(0, len(row) - 1, 2):
          if row[i].isdigit() and re.match(r'^[\dA-Z_]+$', row[i + 1]) and 'INVALID' not in row[i + 1]:
            enums.setdefault('BufFmt', {})[int(row[i])] = row[i + 1]
  # Fix missing opcodes not in PDF tables (documented in ISA but missing from opcode tables)
  if arch == 'rdna3':
    fixes = {'SOPP': {8: 'S_WAITCNT_DEPCTR', 58: 'S_TTRACEDATA', 59: 'S_TTRACEDATA_IMM'},
             'SOPK': {22: 'S_SUBVECTOR_LOOP_BEGIN', 23: 'S_SUBVECTOR_LOOP_END'},
             'SMEM': {34: 'S_ATC_PROBE', 35: 'S_ATC_PROBE_BUFFER'},
             'DS': {24: 'DS_GWS_SEMA_RELEASE_ALL', 25: 'DS_GWS_INIT', 26: 'DS_GWS_SEMA_V', 27: 'DS_GWS_SEMA_BR', 28: 'DS_GWS_SEMA_P', 29: 'DS_GWS_BARRIER'},
             'FLAT': {40: 'GLOBAL_LOAD_ADDTID_B32', 41: 'GLOBAL_STORE_ADDTID_B32', 55: 'FLAT_ATOMIC_CSUB_U32'}}
    for fmt, ops in fixes.items(): enums[fmt] = merge_dicts([enums[fmt], ops])
  return enums

def extract_ins(tables: dict[int, tuple[str, list[list[str]]]], arch: str) -> tuple[dict[str, list[tuple[str, int, int]]], dict[str, str]]:
  """Extract formats and encodings from 'XXX Fields' tables. Returns (formats, encodings)."""
  formats: dict[str, list[tuple[str, int, int]]] = {}
  encodings: dict[str, str] = {}
  for num, (title, rows) in tables.items():
    if not (m := re.match(r'(\w+) Fields$', title)): continue
    fmt_name = m.group(1)
    fields = []
    for row in rows:
      if len(row) < 2: continue
      if (bits := re.match(r'\[?(\d+):(\d+)\]?$', row[1])) or (bits := re.match(r'\[(\d+)\]$', row[1])):
        field_name = row[0].lower()
        hi, lo = int(bits.group(1)), int(bits.group(2)) if bits.lastindex >= 2 else int(bits.group(1))
        if field_name == 'encoding' and len(row) >= 3:
          enc_bits = None
          if "'b" in row[2]: enc_bits = row[2].split("'b")[-1].replace('_', '')
          elif (enc := re.search(r':\s*([01_]+)', row[2])): enc_bits = enc.group(1).replace('_', '')
          if enc_bits:
            # If encoding bits exceed field width, extend field to match (AMD docs sometimes have this)
            declared_width, actual_width = hi - lo + 1, len(enc_bits)
            if actual_width > declared_width: lo = hi - actual_width + 1
            encodings[fmt_name] = enc_bits
        fields.append((field_name, hi, lo))
    if fields: formats[fmt_name] = fields
  # Fix known PDF errors
  if arch in ('rdna3', 'rdna4'):
    # RDNA SMEM: PDF says DLC=[14], GLC=[16] but hardware uses DLC=[13], GLC=[14]
    if 'SMEM' in formats:
      formats['SMEM'] = [(n, 13 if n == 'dlc' else 14 if n == 'glc' else h, 13 if n == 'dlc' else 14 if n == 'glc' else l)
                         for n, h, l in formats['SMEM']]
  if arch == 'cdna':
    # CDNA DS: PDF is missing the GDS field (bit 16)
    if 'DS' in formats and not any(n == 'gds' for n, _, _ in formats['DS']):
      formats['DS'].append(('gds', 16, 16))
    # CDNA DPP/SDWA: PDF only documents modifier fields (bits[63:32]), need to add VOP overlay fields (bits[31:0])
    vop_overlay = [('encoding', 8, 0), ('vop_op', 16, 9), ('vdst', 24, 17), ('vop2_op', 31, 25)]
    if 'DPP' in formats and not any(n == 'encoding' for n, _, _ in formats['DPP']):
      formats['DPP'] = vop_overlay + [('bc' if n == 'bound_ctrl' else n, h, l) for n, h, l in formats['DPP']]
      encodings['DPP'] = '11111010'
    if 'SDWA' in formats and not any(n == 'encoding' for n, _, _ in formats['SDWA']):
      formats['SDWA'] = vop_overlay + [(n, h, l) for n, h, l in formats['SDWA']]
      encodings['SDWA'] = '11111001'
  return formats, encodings

def extract_pcode(pages: list[list[tuple[float, float, str, str]]], enums: dict[str, dict[int, str]]) -> dict[tuple[str, int], str]:
  """Extract pseudocode for instructions. Returns {(name, opcode): pseudocode}."""
  # Build lookup from instruction name to opcode
  name_to_op = {name: op for ops in enums.values() for op, name in ops.items()}

  # First pass: find all instruction headers across all pages
  all_instructions: list[tuple[int, float, str, int]] = []  # (page_idx, y, name, opcode)
  for page_idx, page in enumerate(pages):
    by_y: dict[int, list[tuple[float, str]]] = {}
    for x, y, t, _ in page:
      by_y.setdefault(round(y), []).append((x, t))
    for y, items in sorted(by_y.items(), reverse=True):
      left = [(x, t) for x, t in items if 55 < x < 65]
      right = [(x, t) for x, t in items if 535 < x < 550]
      if left and right and left[0][1] in name_to_op and right[0][1].isdigit():
        all_instructions.append((page_idx, y, left[0][1], int(right[0][1])))

  # Second pass: extract pseudocode between consecutive instructions
  pcode: dict[tuple[str, int], str] = {}
  for i, (page_idx, y, name, opcode) in enumerate(all_instructions):
    # Get end boundary from next instruction
    if i + 1 < len(all_instructions):
      next_page, next_y = all_instructions[i + 1][0], all_instructions[i + 1][1]
    else:
      next_page, next_y = page_idx, 0
    # Collect F6 text from current position to next instruction
    lines = []
    for p in range(page_idx, next_page + 1):
      start_y = y if p == page_idx else 800
      end_y = next_y if p == next_page else 0
      lines.extend((p, y2, t) for x, y2, t, f in pages[p] if f in ('/F6.0', '/F7.0') and end_y < y2 < start_y)
    if lines:
      # Sort by page first, then by y descending within each page (higher y = earlier text in PDF)
      pcode_lines = [t.replace('Ê', '').strip() for _, _, t in sorted(lines, key=lambda x: (x[0], -x[1]))]
      if pcode_lines: pcode[(name, opcode)] = '\n'.join(pcode_lines)
  return pcode

# ═══════════════════════════════════════════════════════════════════════════════
# Write autogen files
# ═══════════════════════════════════════════════════════════════════════════════

def write_enums(enums: dict[str, dict[int, str]], arch: str, path: str):
  """Write enum.py file from extracted enums."""
  doc_name = {"rdna3": "RDNA3.5", "rdna4": "RDNA4", "cdna": "CDNA4"}[arch]
  lines = [f"# autogenerated from AMD {doc_name} ISA PDF by pdf.py - do not edit", "from enum import IntEnum", ""]
  for name, values in sorted(enums.items()):
    suffix = "Op" if name not in ('Src', 'BufFmt') else ("Enum" if name == 'Src' else "")
    prefix = "BUF_FMT_" if name == 'BufFmt' else ""
    lines.append(f"class {name}{suffix}(IntEnum):")
    for val, member in sorted(values.items()):
      lines.append(f"  {prefix}{member} = {val}")
    lines.append("")
  with open(path, "w") as f:
    f.write("\n".join(lines))

def write_ins(formats: dict[str, list[tuple[str, int, int]]], encodings: dict[str, str], enums: dict[str, dict[int, str]], arch: str, path: str):
  """Write ins.py file from extracted formats and enums."""
  doc_name = {"rdna3": "RDNA3.5", "rdna4": "RDNA4", "cdna": "CDNA4"}[arch]

  # Field types and ordering
  def field_type(name, fmt):
    if name == 'op' and fmt in enums: return f'Annotated[BitField, {fmt}Op]'
    if name in ('opx', 'opy'): return 'Annotated[BitField, VOPDOp]'
    if name == 'vdsty': return 'VDSTYEnc'
    if name in ('vdst', 'vsrc1', 'vaddr', 'vdata', 'data', 'data0', 'data1', 'addr', 'vsrc0', 'vsrc2', 'vsrc3'): return 'VGPRField'
    if name in ('sdst', 'sbase', 'sdata', 'srsrc', 'ssamp'): return 'SGPRField'
    if name.startswith('ssrc') or name in ('saddr', 'soffset'): return 'SSrc'
    if name in ('src0', 'srcx0', 'srcy0') or name.startswith('src') and name[3:].isdigit(): return 'Src'
    if name.startswith('simm'): return 'SImm'
    if name == 'offset' or name.startswith('imm'): return 'Imm'
    return None
  field_priority = ['encoding', 'op', 'opx', 'opy', 'vdst', 'vdstx', 'vdsty', 'sdst', 'vdata', 'sdata', 'addr', 'vaddr', 'data', 'data0', 'data1',
                    'src0', 'srcx0', 'srcy0', 'vsrc0', 'ssrc0', 'src1', 'vsrc1', 'vsrcx1', 'vsrcy1', 'ssrc1', 'src2', 'vsrc2', 'src3', 'vsrc3',
                    'saddr', 'sbase', 'srsrc', 'ssamp', 'soffset', 'offset', 'simm16', 'en', 'target', 'attr', 'attr_chan',
                    'omod', 'neg', 'neg_hi', 'abs', 'clmp', 'opsel', 'opsel_hi', 'waitexp', 'wait_va',
                    'dmask', 'dim', 'seg', 'format', 'offen', 'idxen', 'glc', 'dlc', 'slc', 'tfe', 'unrm', 'done', 'row']
  def sort_fields(fields):
    order = {name: i for i, name in enumerate(field_priority)}
    return sorted(fields, key=lambda f: (order.get(f[0], 1000), f[2]))

  # Generate format classes
  lines = [f"# autogenerated from AMD {doc_name} ISA PDF by pdf.py - do not edit", "# ruff: noqa: F401,F403",
           "from typing import Annotated",
           "from extra.assembly.amd.dsl import *",
           f"from extra.assembly.amd.autogen.{arch}.enum import *", "import functools", ""]
  for fmt_name, fields in sorted(formats.items()):
    lines.append(f"class {fmt_name}(Inst):")
    for name, hi, lo in sort_fields(fields):
      bits_str = f"bits[{hi}:{lo}]" if hi != lo else f"bits[{hi}]"
      if name == 'encoding' and fmt_name in encodings: lines.append(f"  encoding = {bits_str} == 0b{encodings[fmt_name]}")
      else:
        ftype = field_type(name, fmt_name)
        lines.append(f"  {name}{f':{ftype}' if ftype else ''} = {bits_str}")
    lines.append("")

  # Generate instruction helpers
  lines.append("# instruction helpers")
  for fmt_name, ops in sorted(enums.items()):
    seg = {"GLOBAL": ", seg=2", "SCRATCH": ", seg=1"}.get(fmt_name, "")
    tgt = {"GLOBAL": "FLAT, GLOBALOp", "SCRATCH": "FLAT, SCRATCHOp"}.get(fmt_name, f"{fmt_name}, {fmt_name}Op")
    suffix = "_e32" if fmt_name in ("VOP1", "VOP2", "VOPC") else "_e64" if fmt_name == "VOP3" and len(ops) > 0 else ""
    if fmt_name in formats or fmt_name in ("GLOBAL", "SCRATCH"):
      for op_val, name in sorted(ops.items()):
        fn_suffix = suffix if fmt_name != "VOP3" or op_val < 512 else ""
        # Special handling for instructions with literal constants
        if name in ('V_FMAMK_F32', 'V_FMAMK_F16'):
          lines.append(f"def {name.lower()}{fn_suffix}(vdst, src0, K, vsrc1): return {fmt_name}({fmt_name}Op.{name}, vdst, src0, vsrc1, literal=K)")
        elif name in ('V_FMAAK_F32', 'V_FMAAK_F16'):
          lines.append(f"def {name.lower()}{fn_suffix}(vdst, src0, vsrc1, K): return {fmt_name}({fmt_name}Op.{name}, vdst, src0, vsrc1, literal=K)")
        else:
          lines.append(f"{name.lower()}{fn_suffix} = functools.partial({tgt}.{name}{seg})")

  with open(path, "w") as f:
    f.write("\n".join(lines))

def write_pcode(pcode: dict[tuple[str, int], str], enums: dict[str, dict[int, str]], arch: str, path: str):
  """Write str_pcode.py file from extracted pseudocode."""
  # Build mapping from (name, opcode) to list of enum class names (can be multiple)
  op_to_enums: dict[tuple[str, int], list[str]] = {}
  for fmt_name, ops in enums.items():
    for opcode, name in ops.items():
      op_to_enums.setdefault((name, opcode), []).append(f"{fmt_name}Op")

  # Group pseudocode by enum class (same pcode can go to multiple enums)
  by_enum: dict[str, list[tuple[str, int, str]]] = {}
  for (name, opcode), code in pcode.items():
    for enum_name in op_to_enums.get((name, opcode), []):
      by_enum.setdefault(enum_name, []).append((name, opcode, code))

  # Generate file
  enum_names = sorted(by_enum.keys())
  lines = [f"# autogenerated by pdf.py - do not edit", f"# to regenerate: python -m extra.assembly.amd.pdf",
           "# ruff: noqa: E501", f"from extra.assembly.amd.autogen.{arch}.enum import {', '.join(enum_names)}", ""]
  for enum_name in enum_names:
    lines.append(f"{enum_name}_PCODE = {{")
    for name, opcode, code in sorted(by_enum[enum_name], key=lambda x: x[1]):
      lines.append(f"  {enum_name}.{name}: {code!r},")
    lines.append("}\n")
  lines.append("PSEUDOCODE_STRINGS = {")
  for enum_name in enum_names:
    lines.append(f"  {enum_name}: {enum_name}_PCODE,")
  lines.append("}")
  with open(path, "w") as f:
    f.write("\n".join(lines))

if __name__ == "__main__":
  import pathlib
  for arch, url in PDF_URLS.items():
    print(f"Processing {arch}...")
    pages = extract(url)
    tables = extract_tables(pages)
    enums = extract_enums(tables, arch)
    formats, encodings = extract_ins(tables, arch)
    pcode = extract_pcode(pages, enums)
    base = pathlib.Path(__file__).parent / "autogen" / arch
    write_enums(enums, arch, base / "enum.py")
    write_ins(formats, encodings, enums, arch, base / "ins.py")
    write_pcode(pcode, enums, arch, base / "str_pcode.py")
    print(f"  {len(tables)} tables, {len(pcode)} pcode -> {base}")
