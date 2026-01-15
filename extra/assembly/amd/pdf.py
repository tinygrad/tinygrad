# PDF pseudocode extractor - extracts instruction pseudocode from AMD ISA PDFs
# Enums are imported from amdxml.py generated files
import re, zlib
from tinygrad.helpers import fetch

PDF_URLS = {
  "rdna3": "https://docs.amd.com/api/khub/documents/UVVZM22UN7tMUeiW_4ShTQ/content",
  "rdna4": "https://docs.amd.com/api/khub/documents/uQpkEvk3pv~kfAb2x~j4uw/content",
  "cdna": "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf",
}

def extract_pdf_text(url: str) -> list[list[tuple[float, float, str, str]]]:
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

def extract_pcode(pages: list[list[tuple[float, float, str, str]]], name_to_op: dict[str, int]) -> dict[tuple[str, int], str]:
  """Extract pseudocode for instructions. Returns {(name, opcode): pseudocode}."""
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
    if i + 1 < len(all_instructions):
      next_page, next_y = all_instructions[i + 1][0], all_instructions[i + 1][1]
    else:
      next_page, next_y = page_idx, 0
    # Collect F6 text from current position to next instruction (pseudocode is at x ≈ 69)
    lines = []
    for p in range(page_idx, next_page + 1):
      start_y = y if p == page_idx else 800
      end_y = next_y if p == next_page else 0
      lines.extend((p, y2, t) for x, y2, t, f in pages[p] if f in ('/F6.0', '/F7.0') and end_y < y2 < start_y and 60 < x < 80)
    if lines:
      sorted_lines = sorted(lines, key=lambda x: (x[0], -x[1]))
      # Stop at large Y gaps (>30) - indicates section break
      filtered = [sorted_lines[0]]
      for j in range(1, len(sorted_lines)):
        prev_page, prev_y, _ = sorted_lines[j-1]
        curr_page, curr_y, _ = sorted_lines[j]
        if curr_page == prev_page and prev_y - curr_y > 30: break
        if curr_page != prev_page and prev_y > 60 and curr_y < 730: break
        filtered.append(sorted_lines[j])
      pcode_lines = [t.replace('Ê', '').strip() for _, _, t in filtered]
      if pcode_lines: pcode[(name, opcode)] = '\n'.join(pcode_lines)
  return pcode

def write_pcode(pcode: dict[tuple[str, int], str], enums: dict[str, dict[int, str]], arch: str, path: str):
  """Write str_pcode.py file from extracted pseudocode."""
  entries: list[tuple[str, str, int, str]] = []
  for fmt_name, ops in enums.items():
    member_suffix = "_E32" if fmt_name in ("VOP1", "VOP2", "VOPC") else "_E64" if fmt_name == "VOP3" else ""
    for opcode, name in ops.items():
      if (name, opcode) in pcode:
        msuf = member_suffix if fmt_name != "VOP3" or opcode < 512 else ""
        entries.append((f"{fmt_name}Op", f"{name}{msuf}", opcode, pcode[(name, opcode)]))
  enum_names = sorted(set(e[0] for e in entries))
  lines = ["# autogenerated by pdf.py - do not edit", "# to regenerate: python -m extra.assembly.amd.pdf",
           "# ruff: noqa: E501", f"from extra.assembly.amd.autogen.{arch}.enum import {', '.join(enum_names)}", "",
           "PCODE = {"]
  for enum_name, name, opcode, code in sorted(entries, key=lambda x: (x[0], x[2])):
    lines.append(f"  {enum_name}.{name}: {code!r},")
  lines.append("}")
  with open(path, "w") as f:
    f.write("\n".join(lines))

def load_enums(arch: str) -> dict[str, dict[int, str]]:
  """Load enums from amdxml-generated files."""
  import importlib
  enum_module = importlib.import_module(f"extra.assembly.amd.autogen.{arch}.enum")
  enums: dict[str, dict[int, str]] = {}
  for name in dir(enum_module):
    if name.endswith("Op"):
      cls = getattr(enum_module, name)
      if hasattr(cls, '__members__'):
        fmt_name = name[:-2]  # Strip "Op" suffix
        enums[fmt_name] = {m.value: m.name.removesuffix("_E32").removesuffix("_E64") for m in cls}
  return enums

if __name__ == "__main__":
  import pathlib
  for arch, url in PDF_URLS.items():
    print(f"Processing {arch}...")
    pages = extract_pdf_text(url)
    enums = load_enums(arch)
    name_to_op = {name: op for ops in enums.values() for op, name in ops.items()}
    pcode = extract_pcode(pages, name_to_op)
    base = pathlib.Path(__file__).parent / "autogen" / arch
    write_pcode(pcode, enums, arch, base / "str_pcode.py")
    print(f"  {len(pcode)} pcode entries -> {base / 'str_pcode.py'}")
