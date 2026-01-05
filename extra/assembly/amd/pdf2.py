# Generic PDF text extractor - no external dependencies
import re, zlib
from tinygrad.helpers import fetch

PDF_URLS = {
  "rdna3": "https://docs.amd.com/api/khub/documents/UVVZM22UN7tMUeiW_4ShTQ/content",
  "rdna4": "https://docs.amd.com/api/khub/documents/uQpkEvk3pv~kfAb2x~j4uw/content",
  "cdna": "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf",
}

# ═══════════════════════════════════════════════════════════════════════════════
# Generic PDF extraction tools
# ═══════════════════════════════════════════════════════════════════════════════

def extract(url: str) -> list[list[tuple[float, float, str]]]:
  """Extract positioned text from PDF. Returns list of text elements (x, y, text) per page."""
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
    elements = []
    for bt in re.finditer(r'BT(.*?)ET', stream, re.S):
      x, y = 0.0, 0.0
      for m in re.finditer(r'([\d.+-]+) ([\d.+-]+) Td|[\d.+-]+ [\d.+-]+ [\d.+-]+ [\d.+-]+ ([\d.+-]+) ([\d.+-]+) Tm|<([0-9A-Fa-f]+)>.*?Tj|\[([^\]]+)\] TJ', bt.group(1)):
        if m.group(1): x, y = x + float(m.group(1)), y + float(m.group(2))
        elif m.group(3): x, y = float(m.group(3)), float(m.group(4))
        elif m.group(5) and (t := bytes.fromhex(m.group(5)).decode('latin-1')).strip(): elements.append((x, y, t))
        elif m.group(6) and (t := ''.join(bytes.fromhex(h).decode('latin-1') for h in re.findall(r'<([0-9A-Fa-f]+)>', m.group(6)))).strip(): elements.append((x, y, t))
    pages.append(sorted(elements, key=lambda e: (-e[1], e[0])))
  return pages

def extract_tables(pages: list[list[tuple[float, float, str]]]) -> dict[int, tuple[str, list[list[str]]]]:
  """Extract numbered tables from PDF pages. Returns {table_num: (title, rows)} where rows is list of cells per row."""
  def group_by_y(texts, key=lambda y: round(y)):
    by_y: dict[int, list[tuple[float, float, str]]] = {}
    for x, y, t in texts:
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
      page_texts = [(x, y, t) for x, y, t in pages[page_idx] if 30 < y < 760 and (page_idx > start_page or y < header_y)]
      for items in sorted(group_by_y(page_texts, key=lambda y: round(y / 5)).values(), key=lambda items: -items[0][1]):
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
# AMD autogen
# ═══════════════════════════════════════════════════════════════════════════════

def extract_enums(tables: dict[int, tuple[str, list[list[str]]]]) -> dict[str, dict[int, str]]:
  """Extract opcode enums from tables. Returns {fmt_name: {opcode: opname}}."""
  enums: dict[str, dict[int, str]] = {}
  for num, (title, rows) in tables.items():
    if not (m := re.match(r'(\w+) (?:Y-)?Opcodes', title)): continue
    fmt_name = 'VOPD' if 'Y-Opcodes' in title else m.group(1)
    ops: dict[int, str] = {}
    for row in rows:
      for i in range(0, len(row) - 1, 2):
        if row[i].isdigit() and re.match(r'^[A-Z][A-Z0-9_]+$', row[i + 1]):
          ops[int(row[i])] = row[i + 1]
    if ops: enums[fmt_name] = ops
  return enums

def generate_enums(enums: dict[str, dict[int, str]], arch: str, path: str):
  """Write enum.py file from extracted enums."""
  doc_name = {"rdna3": "RDNA3.5", "rdna4": "RDNA4", "cdna": "CDNA4"}[arch]
  lines = [f"# autogenerated from AMD {doc_name} ISA PDF by pdf2.py - do not edit", "from enum import IntEnum", ""]
  for fmt_name, ops in sorted(enums.items()):
    lines.append(f"class {fmt_name}Op(IntEnum):")
    for opcode, opname in sorted(ops.items()):
      lines.append(f"  {opname} = {opcode}")
    lines.append("")
  with open(path, "w") as f:
    f.write("\n".join(lines))

def generate_ins(pages: list[list[tuple[float, float, str]]], tables: dict[int, tuple[str, list[list[str]]]], enums: dict[str, dict[int, str]], arch: str, path: str):
  """Parse AMD ISA format tables and write ins.py file."""
  doc_name = {"rdna3": "RDNA3.5", "rdna4": "RDNA4", "cdna": "CDNA4"}[arch]

  # Find all 'XXX Fields' tables and extract field definitions
  format_tables = []
  for num, (title, rows) in tables.items():
    if m := re.match(r'(\w+) Fields$', title):
      for page_idx, texts in enumerate(pages):
        for x, y, t in texts:
          if f'Table {num}.' in t and title in t:
            format_tables.append((m.group(1), page_idx, y, num))
            break
  format_tables.sort(key=lambda t: (t[1], -t[2]))

  formats: dict[str, list[tuple[str, int, int]]] = {}
  for i, (fmt_name, page_idx, header_y, table_num) in enumerate(format_tables):
    next_page, next_y = (format_tables[i + 1][1], format_tables[i + 1][2]) if i + 1 < len(format_tables) else (len(pages), 0)
    fields = []
    for pg in range(page_idx, min(next_page + 1, len(pages))):
      by_y: dict[int, list[tuple[float, str]]] = {}
      for x, y, t in pages[pg]:
        if pg == page_idx and y >= header_y: continue
        if pg == next_page and y <= next_y: continue
        by_y.setdefault(round(y), []).append((x, t))
      for items in by_y.values():
        field_name, bit_range = None, None
        for x, t in items:
          if 55 < x < 80 and re.match(r'^[A-Z][A-Z0-9_]*$', t): field_name = t.lower()
          if 110 < x < 180 and (m := re.match(r'\[(\d+):?(\d+)?\]', t)): bit_range = (int(m.group(1)), int(m.group(2)) if m.group(2) else int(m.group(1)))
        if field_name and bit_range: fields.append((field_name, bit_range[0], bit_range[1]))
    if fields: formats[fmt_name] = fields

  # Determine field types
  def field_type(name, fmt):
    if name == 'op': return f'Annotated[BitField, {fmt}Op]'
    if name in ('vdst', 'vsrc1', 'vaddr', 'vdata', 'data', 'data0', 'data1', 'addr', 'vsrc0', 'vsrc2', 'vsrc3'): return 'VGPRField'
    if name in ('sdst', 'sbase', 'sdata', 'srsrc', 'ssamp'): return 'SGPRField'
    if name.startswith('ssrc') or name in ('saddr', 'soffset'): return 'SSrc'
    if name == 'src0' or name.startswith('src') and name[3:].isdigit(): return 'Src'
    if name.startswith('simm'): return 'SImm'
    if name == 'offset' or name.startswith('imm'): return 'Imm'
    return None

  # Find encoding values from tables
  encodings = {}
  for num, (title, rows) in tables.items():
    if m := re.match(r'(\w+) Fields$', title):
      for row in rows:
        if len(row) >= 3 and row[0].upper() == 'ENCODING':
          if "'b" in row[2]: encodings[m.group(1)] = row[2].split("'b")[-1].replace('_', '')
          elif (enc := re.search(r':\s*([01_]+)', row[2])): encodings[m.group(1)] = enc.group(1).replace('_', '')  # "Must be: 110110"

  # Field ordering - important fields first, then rest by bit position
  field_priority = ['encoding', 'op', 'opx', 'opy', 'vdst', 'vdstx', 'vdsty', 'sdst', 'vdata', 'sdata', 'addr', 'vaddr', 'data', 'data0', 'data1',
                    'src0', 'srcx0', 'srcy0', 'vsrc0', 'ssrc0', 'src1', 'vsrc1', 'vsrcx1', 'vsrcy1', 'ssrc1', 'src2', 'vsrc2', 'src3', 'vsrc3',
                    'saddr', 'sbase', 'srsrc', 'ssamp', 'soffset', 'offset', 'simm16', 'en', 'target', 'attr', 'attr_chan',
                    'omod', 'neg', 'neg_hi', 'abs', 'clmp', 'opsel', 'opsel_hi', 'waitexp', 'wait_va',
                    'dmask', 'dim', 'seg', 'format', 'offen', 'idxen', 'glc', 'dlc', 'slc', 'tfe', 'unrm', 'done', 'row']
  def sort_fields(fields):
    order = {name: i for i, name in enumerate(field_priority)}
    return sorted(fields, key=lambda f: (order.get(f[0], 1000), f[2]))  # by priority then by lo bit

  lines = [f"# autogenerated from AMD {doc_name} ISA PDF by pdf2.py - do not edit", "# ruff: noqa: F401,F403",
           "import functools", "from typing import Annotated",
           "from extra.assembly.amd.dsl import bits, BitField, Inst32, Inst64, Inst96, SSrc, Src, SImm, Imm, SGPRField, VGPRField",
           f"from extra.assembly.amd.autogen.{arch}.enum import *", ""]

  for fmt_name, fields in sorted(formats.items()):
    max_bit = max(hi for _, hi, _ in fields)
    size = 96 if max_bit > 63 else 64 if max_bit > 31 else 32
    lines.append(f"class {fmt_name}(Inst{size}):")
    for name, hi, lo in sort_fields(fields):
      if name == 'encoding' and fmt_name in encodings:
        bits_str = f"bits[{hi}:{lo}]" if hi != lo else f"bits[{hi}]"
        lines.append(f"  encoding = {bits_str} == 0b{encodings[fmt_name]}")
      else:
        ftype = field_type(name, fmt_name)
        type_ann = f':{ftype}' if ftype else ''
        bits_str = f"bits[{hi}:{lo}]" if hi != lo else f"bits[{hi}]"
        lines.append(f"  {name}{type_ann} = {bits_str}")
    lines.append("")

  # Generate instruction helpers
  lines.append("# instruction helpers")
  for fmt_name, ops in sorted(enums.items()):
    fmt = fmt_name
    seg = {"GLOBAL": ", seg=2", "SCRATCH": ", seg=1"}.get(fmt, "")
    tgt = {"GLOBAL": "FLAT, GLOBALOp", "SCRATCH": "FLAT, SCRATCHOp"}.get(fmt, f"{fmt}, {fmt_name}Op")
    if fmt in formats or fmt in ("GLOBAL", "SCRATCH"):
      for op_val, name in sorted(ops.items()):
        lines.append(f"{name.lower()} = functools.partial({tgt}.{name}{seg})")

  with open(path, "w") as f:
    f.write("\n".join(lines))

if __name__ == "__main__":
  import pathlib
  for arch, url in PDF_URLS.items():
    print(f"Processing {arch}...")
    pages = extract(url)
    tables = extract_tables(pages)
    enums = extract_enums(tables)
    base = pathlib.Path(__file__).parent / "autogen" / arch
    generate_enums(enums, arch, base / "enum.py")
    generate_ins(pages, tables, enums, arch, base / "ins.py")
    print(f"  {len(tables)} tables -> {base}")
