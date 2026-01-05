# Generic PDF text extractor - no external dependencies
import re, zlib
from tinygrad.helpers import fetch

PDF_URLS = {
  "rdna3": "https://docs.amd.com/api/khub/documents/UVVZM22UN7tMUeiW_4ShTQ/content",
  "rdna4": "https://docs.amd.com/api/khub/documents/uQpkEvk3pv~kfAb2x~j4uw/content",
  "cdna": "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf",
}

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
  # Merge text elements on same line for finding headers
  def merge_lines(texts):
    by_y: dict[int, list[tuple[float, str]]] = {}
    for x, y, t in texts:
      by_y.setdefault(round(y), []).append((x, t))
    return [(float(y), ''.join(t for _, t in sorted(items))) for y, items in by_y.items()]

  # Group elements into rows by Y position, return list of (y, x_positions, row_texts) sorted by y descending
  def group_rows(texts):
    by_y: dict[int, list[tuple[float, float, str]]] = {}
    for x, y, t in texts:
      by_y.setdefault(round(y / 5), []).append((x, y, t))
    rows = []
    for items in by_y.values():
      avg_y = sum(y for _, y, _ in items) / len(items)
      xs = tuple(sorted(round(x) for x, _, _ in items))
      row = [t for _, t in sorted((x, t) for x, _, t in items)]
      rows.append((avg_y, xs, row))
    return sorted(rows, key=lambda r: -r[0])  # Sort by Y descending (top to bottom)

  # Find all table headers
  table_positions = []
  for page_idx, texts in enumerate(pages):
    for y, line in merge_lines(texts):
      if m := re.search(r'Table (\d+)\. (.+)', line):
        table_positions.append((int(m.group(1)), m.group(2).strip(), page_idx, y))
  table_positions.sort(key=lambda t: (t[2], -t[3]))

  # For each table, find rows with matching X positions
  result: dict[int, tuple[str, list[list[str]]]] = {}
  for num, title, start_page, header_y in table_positions:
    rows, col_xs = [], None
    for page_idx in range(start_page, len(pages)):
      texts = pages[page_idx]
      # Filter to elements below header on start page, exclude page header/footer
      page_texts = [(x, y, t) for x, y, t in texts if 30 < y < 760 and (page_idx > start_page or y < header_y)]
      for y, xs, row in group_rows(page_texts):
        if col_xs is None:
          if len(xs) < 2: continue  # Skip single-column rows before table starts
          col_xs = xs  # First multi-column row defines column positions
        elif not any(c in xs for c in col_xs[:2]): break  # Row missing first columns = end of table
        rows.append(row)
      else:
        continue  # No break = continue to next page
      break  # Break from inner loop = break from outer loop
    if rows: result[num] = (title, rows)
  return result

def generate_enums(tables: dict[int, tuple[str, list[list[str]]]], arch: str, path: str):
  """Parse AMD ISA opcode tables and write enum.py file."""
  doc_name = {"rdna3": "RDNA3.5", "rdna4": "RDNA4", "cdna": "CDNA4"}[arch]
  lines = [f"# autogenerated from AMD {doc_name} ISA PDF by pdf2.py - do not edit", "from enum import IntEnum", ""]
  for num, (title, rows) in sorted(tables.items()):
    if not (m := re.match(r'(\w+) (?:Y-)?Opcodes', title)): continue
    fmt_name = 'VOPD' if 'Y-Opcodes' in title else m.group(1)
    ops: dict[int, str] = {}
    for row in rows:
      for i in range(0, len(row) - 1, 2):
        if row[i].isdigit() and re.match(r'^[A-Z][A-Z0-9_]+$', row[i + 1]):
          ops[int(row[i])] = row[i + 1]
    if ops:
      lines.append(f"class {fmt_name}Op(IntEnum):")
      for opcode, opname in sorted(ops.items()):
        lines.append(f"  {opname} = {opcode}")
      lines.append("")
  with open(path, "w") as f:
    f.write("\n".join(lines))

if __name__ == "__main__":
  import pathlib
  for arch, url in PDF_URLS.items():
    print(f"Processing {arch}...")
    pages = extract(url)
    tables = extract_tables(pages)
    out_path = pathlib.Path(__file__).parent / "autogen" / arch / "enum.py"
    generate_enums(tables, arch, out_path)
    print(f"  {len(tables)} tables -> {out_path}")
