#!/usr/bin/env python3
# generates autogen/__init__.py by parsing the AMD RDNA3.5 ISA PDF
import re, pdfplumber

FIELD_TYPES = {'OP': lambda fmt, enums: f"{fmt}Op" if f"{fmt}Op" in enums else None,
  'SSRC0': 'SSrc', 'SSRC1': 'SSrc', 'SOFFSET': 'SSrc', 'SADDR': 'SSrc', 'SRC0': 'Src', 'SRC1': 'Src', 'SRC2': 'Src',
  'SDST': 'SGPR', 'SBASE': 'SGPR', 'SDATA': 'SGPR', 'SRSRC': 'SGPR', 'VDST': 'VGPR', 'VSRC1': 'VGPR', 'VDATA': 'VGPR',
  'VADDR': 'VGPR', 'ADDR': 'VGPR', 'DATA': 'VGPR', 'DATA0': 'VGPR', 'DATA1': 'VGPR', 'SIMM16': 'SImm', 'OFFSET': 'Imm'}
FIELD_ORDER = {'SOP2': ['op', 'sdst', 'ssrc0', 'ssrc1'], 'SOP1': ['op', 'sdst', 'ssrc0'], 'SOPC': ['op', 'ssrc0', 'ssrc1'],
  'SOPK': ['op', 'sdst', 'simm16'], 'SOPP': ['op', 'simm16'], 'SMEM': ['op', 'sdata', 'sbase', 'soffset', 'offset', 'glc', 'dlc'],
  'VOP1': ['op', 'vdst', 'src0'], 'VOP2': ['op', 'vdst', 'src0', 'vsrc1'], 'VOPC': ['op', 'src0', 'vsrc1'],
  'VOP3': ['op', 'vdst', 'src0', 'src1', 'src2', 'omod', 'neg', 'abs', 'clmp', 'opsel'], 'VOP3SD': ['op', 'vdst', 'sdst', 'src0', 'src1', 'src2', 'clmp'],
  'DS': ['op', 'vdst', 'addr', 'data0', 'data1', 'offset0', 'offset1', 'gds'], 'FLAT': ['op', 'vdst', 'addr', 'data', 'saddr', 'offset', 'seg', 'dlc', 'glc', 'slc'],
  'MUBUF': ['op', 'vdata', 'vaddr', 'srsrc', 'soffset', 'offset', 'offen', 'idxen', 'glc', 'dlc', 'slc', 'tfe'],
  'MTBUF': ['op', 'vdata', 'vaddr', 'srsrc', 'soffset', 'offset', 'format', 'offen', 'idxen', 'glc', 'dlc', 'slc', 'tfe'],
  'MIMG': ['op', 'vdata', 'vaddr', 'srsrc', 'ssamp', 'dmask', 'dim', 'unrm', 'dlc', 'glc', 'slc', 'r128', 'a16', 'd16', 'tfe', 'lwe', 'nsa'],
  'EXP': ['en', 'target', 'vsrc0', 'vsrc1', 'vsrc2', 'vsrc3', 'done', 'row']}
SRC_EXTRAS = {233: 'DPP8', 234: 'DPP8FI', 250: 'DPP16', 251: 'VCCZ', 252: 'EXECZ', 254: 'LDS_DIRECT'}
FLOAT_MAP = {'0.5': 'POS_HALF', '-0.5': 'NEG_HALF', '1.0': 'POS_ONE', '-1.0': 'NEG_ONE', '2.0': 'POS_TWO', '-2.0': 'NEG_TWO', '4.0': 'POS_FOUR', '-4.0': 'NEG_FOUR', '1/(2*PI)': 'INV_2PI', '0': 'ZERO'}

def parse_bits(s: str) -> tuple[int, int] | None:
  if m := re.match(r'\[(\d+):(\d+)\]', s): return int(m.group(1)), int(m.group(2))
  if m := re.match(r'\[(\d+)\]', s): return int(m.group(1)), int(m.group(1))
  return None

def parse_fields_table(table: list, fmt: str, enums: set[str]) -> list[tuple]:
  fields = []
  for row in table[1:]:
    if not row or not row[0]: continue
    name, bits_str = row[0].split('\n')[0].strip(), (row[1] or '').split('\n')[0].strip()
    if not (bits := parse_bits(bits_str)): continue
    enc_val = int(m.group(1).replace('_', ''), 2) if name == 'ENCODING' and row[2] and (m := re.search(r"'b([01_]+)", row[2])) else None
    ftype = FIELD_TYPES.get(name.upper())
    fields.append((name, bits[0], bits[1], enc_val, ftype(fmt, enums) if callable(ftype) else ftype))
  return fields

def is_fields_table(t) -> bool: return t and len(t) > 1 and t[0] and 'Field' in str(t[0][0] or '')

if __name__ == "__main__":
  pdf = pdfplumber.open("extra/assembly/rdna3/autogen/rdna35_instruction_set_architecture.pdf")
  full_text = '\n'.join(page.extract_text() or '' for page in pdf.pages)

  # parse SSRC encoding
  src_enum = dict(SRC_EXTRAS)
  for page in pdf.pages[150:160]:
    text = page.extract_text() or ''
    if 'SSRC0' in text and 'VCC_LO' in text:
      for m in re.finditer(r'^(\d+)\s+(\S+)', text, re.M):
        val, name = int(m.group(1)), m.group(2).rstrip('.:')
        if name in FLOAT_MAP: src_enum[val] = FLOAT_MAP[name]
        elif re.match(r'^[A-Z][A-Z0-9_]*$', name): src_enum[val] = name
      break

  # parse opcode tables
  enums: dict[str, dict[int, str]] = {}
  for m in re.finditer(r'Table \d+\. (\w+) Opcodes(.*?)(?=Table \d+\.|\n\d+\.\d+\.\d+\.\s+\w+\s*\nDescription|$)', full_text, re.S):
    if ops := {int(x.group(1)): x.group(2) for x in re.finditer(r'(\d+)\s+([A-Z][A-Z0-9_]+)', m.group(2))}:
      enums[m.group(1) + "Op"] = ops

  # parse instruction formats
  formats: dict[str, list] = {}
  for i, page in enumerate(pdf.pages[150:200]):
    text = page.extract_text() or ''
    for m in re.finditer(r'\d+\.\d+\.\d+\.\s+(\w+)\s*\n?Description', text):
      fmt_name, header_pos = m.group(1), m.start()
      if fmt_name in formats: continue
      fields, field_pos = None, text.find('Field Name', header_pos)
      if field_pos > header_pos:
        for t in page.find_tables():
          if is_fields_table(t_data := t.extract()) and (fields := parse_fields_table(t_data, fmt_name, set(enums.keys()))) and any(f[0] == 'ENCODING' for f in fields): break
          fields = None
      if not fields and 150+i+1 < len(pdf.pages):
        for t in pdf.pages[150+i+1].find_tables():
          if is_fields_table(t_data := t.extract()) and (fields := parse_fields_table(t_data, fmt_name, set(enums.keys()))) and any(f[0] == 'ENCODING' for f in fields): break
          fields = None
      if fields:
        if field_pos > header_pos and 150+i+1 < len(pdf.pages):
          for nt in pdf.pages[150+i+1].extract_tables():
            if is_fields_table(nt) and (extra := parse_fields_table(nt, fmt_name, set(enums.keys()))) and not any(f[0] == 'ENCODING' for f in extra): fields.extend(extra); break
        formats[fmt_name] = fields

  # generate output
  lines = ["# autogenerated from AMD RDNA3.5 ISA PDF - do not edit", "from enum import IntEnum",
           "from extra.assembly.rdna3.lib import bits, Inst32, Inst64, SGPR, VGPR, TTMP, s, v, SSrc, Src, SImm, Imm", "import functools", ""]
  lines.append("class SrcEnum(IntEnum):")
  for val, name in sorted(src_enum.items()): lines.append(f"  {name} = {val}")
  lines.append("")
  for cls_name, ops in sorted(enums.items()):
    lines.append(f"class {cls_name}(IntEnum):")
    for opcode, name in sorted(ops.items()): lines.append(f"  {name} = {opcode}")
    lines.append("")
  lines.append("# instruction formats")
  for fmt_name, fields in sorted(formats.items()):
    base = "Inst64" if max(f[1] for f in fields) > 31 or fmt_name == 'VOP3SD' else "Inst32"
    lines.append(f"class {fmt_name}({base}):")
    if enc := next((f for f in fields if f[0] == 'ENCODING'), None):
      lines.append(f"  encoding = bits[{enc[1]}:{enc[2]}] == 0b{enc[3]:b}" if enc[1] != enc[2] else f"  encoding = bits[{enc[1]}] == 0b{enc[3]:b}" if enc[3] else f"  encoding = bits[{enc[1]}] == 0")
    order = FIELD_ORDER.get(fmt_name, [])
    for name, hi, lo, _, ftype in sorted([f for f in fields if f[0] != 'ENCODING'], key=lambda f: order.index(f[0].lower()) if f[0].lower() in order else 1000):
      lines.append(f"  {name.lower()}{f':{ftype}' if ftype else ''} = bits[{hi}]" if hi == lo else f"  {name.lower()}{f':{ftype}' if ftype else ''} = bits[{hi}:{lo}]")
    lines.append("")
  lines.append("# instruction helpers")
  for cls_name, ops in sorted(enums.items()):
    fmt = cls_name[:-2]
    for _, name in sorted(ops.items()):
      if fmt == "GLOBAL": lines.append(f"{name.lower()} = functools.partial(FLAT, GLOBALOp.{name}, seg=2)")
      elif fmt == "SCRATCH": lines.append(f"{name.lower()} = functools.partial(FLAT, SCRATCHOp.{name}, seg=2)")
      elif fmt in formats: lines.append(f"{name.lower()}{'_e32' if fmt in ('VOP1', 'VOP2') else ''} = functools.partial({fmt}, {cls_name}.{name})")
  lines.append("")
  for _, name in sorted(src_enum.items()): lines.append(f"{name} = SrcEnum.{name}")
  lines.append("OFF = NULL\n")
  with open("extra/assembly/rdna3/autogen/__init__.py", "w") as f: f.write('\n'.join(lines))
  print(f"generated SrcEnum ({len(src_enum)}) + {len(enums)} opcode enums + {len(formats)} format classes")
