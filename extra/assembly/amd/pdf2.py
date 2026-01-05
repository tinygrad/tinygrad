# Generic PDF text extractor - no external dependencies
import re, zlib
from tinygrad.helpers import fetch

PDF_URLS = {
  "rdna3": "https://docs.amd.com/api/khub/documents/UVVZM22UN7tMUeiW_4ShTQ/content",
  "rdna4": "https://docs.amd.com/api/khub/documents/uQpkEvk3pv~kfAb2x~j4uw/content",
  "cdna": "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf",
}

def extract(url: str) -> list[list[tuple[float, float, str]]]:
  """Extract text elements from PDF grouped by page. Returns list of pages, each page is list of (x, y, text)."""
  data = fetch(url).read_bytes()

  # Parse xref table
  xref: dict[int, int] = {}
  pos = int(re.search(rb'startxref\s+(\d+)', data).group(1)) + 4
  while data[pos:pos+7] != b'trailer':
    while data[pos:pos+1] in b' \r\n': pos += 1
    line_end = data.find(b'\n', pos)
    start_obj, count = int(data[pos:line_end].split()[0]), int(data[pos:line_end].split()[1])
    pos = line_end + 1
    for i in range(count):
      if data[pos+17:pos+18] == b'n' and (off := int(data[pos:pos+10])) > 0: xref[start_obj + i] = off
      pos += 20

  def get_stream(n: int) -> bytes:
    obj = data[xref[n]:data.find(b'endobj', xref[n])]
    start = obj.find(b'stream\n') + 7
    raw = obj[start:obj.find(b'\nendstream')]
    return zlib.decompress(raw) if b'/FlateDecode' in obj else raw

  # Find page content streams
  pages = [int(m.group(1)) for n in sorted(xref) if b'/Type /Page' in data[xref[n]:xref[n]+500] and (m := re.search(rb'/Contents (\d+) 0 R', data[xref[n]:xref[n]+500]))]

  # Extract text from each page
  result = []
  for page_obj in pages:
    elements = []
    for bt in re.finditer(r'BT(.*?)ET', get_stream(page_obj).decode('latin-1'), re.S):
      x, y = 0.0, 0.0
      for m in re.finditer(r'([\d.+-]+) ([\d.+-]+) Td|[\d.+-]+ [\d.+-]+ [\d.+-]+ [\d.+-]+ ([\d.+-]+) ([\d.+-]+) Tm|<([0-9A-Fa-f]+)>.*?Tj|\[([^\]]+)\] TJ', bt.group(1)):
        if m.group(1): x, y = x + float(m.group(1)), y + float(m.group(2))
        elif m.group(3): x, y = float(m.group(3)), float(m.group(4))
        elif m.group(5) and (t := bytes.fromhex(m.group(5)).decode('latin-1')).strip(): elements.append((x, y, t))
        elif m.group(6) and (t := ''.join(bytes.fromhex(h).decode('latin-1') for h in re.findall(r'<([0-9A-Fa-f]+)>', m.group(6)))).strip(): elements.append((x, y, t))
    result.append(sorted(elements, key=lambda e: (-e[1], e[0])))
  return result

def extract_tables(pages: list[list[tuple[float, float, str]]]) -> dict[int, tuple[str, list[list[str]]]]:
  """Extract numbered tables from PDF pages. Returns {table_num: (title, rows)} where rows is list of cells per row."""
  # Find all table headers with their positions
  table_positions: list[tuple[int, str, int, float]] = []  # (num, title, page, y)
  for page_idx, page in enumerate(pages):
    for x, y, t in page:
      if m := re.match(r'Table (\d+)\. (.+)', t):
        table_positions.append((int(m.group(1)), m.group(2), page_idx, y))

  table_positions.sort(key=lambda t: (t[2], -t[3]))  # sort by page, then by y descending

  result: dict[int, tuple[str, list[list[str]]]] = {}
  for i, (num, title, start_page, start_y) in enumerate(table_positions):
    # Find end boundary
    if i + 1 < len(table_positions):
      _, _, end_page, end_y = table_positions[i + 1]
    else:
      end_page, end_y = len(pages) - 1, 0

    # Collect elements for this table (below header, above next table)
    elements: list[tuple[float, float, str]] = []
    for page_idx in range(start_page, min(end_page + 1, len(pages))):
      for x, y, t in pages[page_idx]:
        # Skip headers/footers
        if y < 30 or y > 760: continue
        if re.match(r'^\d+ of \d+$', t): continue
        if t.startswith('"RDNA') or t.startswith('CDNA') or 'Instruction Set Architecture' in t: continue

        # On start page: only elements below the table header
        if page_idx == start_page and y >= start_y: continue
        # On end page: only elements above the next table header
        if page_idx == end_page and y <= end_y: continue

        elements.append((x, y, t))

    if not elements: continue

    # Group into rows by Y position
    elements.sort(key=lambda e: (-e[1], e[0]))
    rows: list[list[tuple[float, str]]] = []
    current_row: list[tuple[float, str]] = []
    current_y: float | None = None
    for x, y, t in elements:
      if current_y is None or abs(y - current_y) < 5:
        current_row.append((x, t))
        current_y = y if current_y is None else current_y
      else:
        if current_row: rows.append(sorted(current_row, key=lambda e: e[0]))
        current_row, current_y = [(x, t)], y
    if current_row: rows.append(sorted(current_row, key=lambda e: e[0]))

    result[num] = (title, [[t for _, t in row] for row in rows])

  return result

if __name__ == "__main__":
  import sys, json
  pages = extract(sys.argv[1])
  if len(sys.argv) > 2 and sys.argv[2] == "--tables":
    tables = extract_tables(pages)
    print(f"Found {len(tables)} tables")
    for num, (title, rows) in sorted(tables.items()):
      print(f"\nTable {num}. {title} ({len(rows)} rows)")
      for row in rows[:5]: print(f"  {row}")
      if len(rows) > 5: print(f"  ... ({len(rows) - 5} more)")
  else:
    print(json.dumps([{"page": i, "elements": [{"x": x, "y": y, "text": t} for x, y, t in p]} for i, p in enumerate(pages)]))
