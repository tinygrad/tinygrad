#!/usr/bin/env python3
# generates autogen_rdna3_enum.py by parsing the AMD RDNA3.5 ISA PDF
import re, pdfplumber

def parse_opcode_table(text: str) -> dict[int, str]:
  """Parse a two-column opcode table like '0 S_ADD_U32 35 S_AND_NOT1_B64'"""
  ops = {}
  for m in re.finditer(r'(\d+)\s+([A-Z][A-Z0-9_]+)', text):
    ops[int(m.group(1))] = m.group(2)
  return ops

def infer_field_type(name: str, desc: str, fmt_name: str, known_enums: set[str] | None = None) -> str | None:
  """Infer field type from name and description."""
  name = name.upper()
  desc_lower = desc.lower() if desc else ''
  # Opcode fields - type is FormatOp enum (only if enum exists)
  if name == 'OP':
    op_enum = f"{fmt_name}Op"
    if known_enums is None or op_enum in known_enums:
      return op_enum
    return None
  # Scalar source operands (SSRC0, SSRC1) - uses full source encoding
  if name in ('SSRC0', 'SSRC1'):
    return 'SSrc'
  # Vector/scalar source operands (SRC0, SRC1, SRC2 in VOP formats)
  if name in ('SRC0', 'SRC1', 'SRC2'):
    return 'Src'
  # Scalar destination - SGPR register
  if name == 'SDST':
    return 'SGPR'
  # Vector destination - VGPR register (index stored directly, no 256 offset)
  if name == 'VDST':
    return 'VGPR'
  # Vector source (VSRC1 is always a VGPR, index stored directly)
  if name == 'VSRC1':
    return 'VGPR'
  # SGPR pair/quad fields (SBASE, SDATA, SRSRC)
  if name in ('SBASE', 'SDATA', 'SRSRC'):
    return 'SGPR'
  # SOFFSET can be SGPR or special value
  if name == 'SOFFSET':
    return 'SSrc'
  # VGPR fields for addresses and data
  if name in ('VDATA', 'VADDR'):
    return 'VGPR'
  if name == 'ADDR' and ('vgpr' in desc_lower or fmt_name in ('FLAT', 'DS')):
    return 'VGPR'
  if name == 'DATA' and fmt_name in ('FLAT', 'DS'):
    return 'VGPR'
  if name in ('DATA0', 'DATA1'):
    return 'VGPR'
  # SADDR uses source encoding (can be SGPR or OFF/NULL)
  if name == 'SADDR':
    return 'SSrc'
  # Immediate fields - signed or unsigned based on description
  if name == 'SIMM16':
    return 'SImm'
  if name == 'OFFSET':
    if 'signed' in desc_lower:
      return 'SImm'
    return 'Imm'
  return None

def parse_fields_table(table: list, fmt_name: str = '', known_enums: set[str] | None = None) -> list[tuple[str, int, int, int | None, str | None]]:
  """Parse fields table, return list of (name, hi, lo, encoding_value, type)"""
  fields = []
  for row in table[1:]:  # skip header
    if not row or not row[0]: continue
    name = row[0].split('\n')[0].strip()
    bits_str = row[1].split('\n')[0].strip() if row[1] else ''
    if m := re.match(r'\[(\d+):(\d+)\]', bits_str): hi, lo = int(m.group(1)), int(m.group(2))
    elif m := re.match(r'\[(\d+)\]', bits_str): hi = lo = int(m.group(1))
    else: continue
    enc_val = None
    desc = row[2].split('\n')[0].strip() if row[2] else ''
    if name == 'ENCODING' and row[2]:
      if m := re.search(r"'b([01_]+)", row[2]): enc_val = int(m.group(1).replace('_', ''), 2)
    field_type = infer_field_type(name, desc, fmt_name, known_enums)
    fields.append((name, hi, lo, enc_val, field_type))
  return fields

def parse_src_encoding(text: str) -> dict[int, str]:
  """Parse SSRC encoding from the SOP2 Fields table"""
  src = {}
  float_map = {'0.5': 'POS_HALF', '-0.5': 'NEG_HALF', '1.0': 'POS_ONE', '-1.0': 'NEG_ONE',
               '2.0': 'POS_TWO', '-2.0': 'NEG_TWO', '4.0': 'POS_FOUR', '-4.0': 'NEG_FOUR'}
  for line in text.split('\n'):
    if m := re.match(r'^(\d+)\s+(\S+)', line.strip()):
      val, name = int(m.group(1)), m.group(2).rstrip('.').rstrip(':')
      if name in float_map: src[val] = float_map[name]
      elif name == '1/(2*PI)': src[val] = 'INV_2PI'
      elif name == '0': src[val] = 'ZERO'
      elif re.match(r'^[A-Z][A-Z0-9_]*$', name): src[val] = name
  return src

if __name__ == "__main__":
  pdf = pdfplumber.open("extra/assembly/rdna3/rdna35_instruction_set_architecture.pdf")
  full_text = '\n'.join(page.extract_text() or '' for page in pdf.pages)

  # parse SSRC encoding from SOP2 Fields table
  src_enum = {}
  for page in pdf.pages[150:160]:
    text = page.extract_text() or ''
    if 'SSRC0' in text and 'VCC_LO' in text:
      src_enum = parse_src_encoding(text)
      break
  src_enum.update({233: 'DPP8', 234: 'DPP8FI', 250: 'DPP16', 251: 'VCCZ', 252: 'EXECZ', 254: 'LDS_DIRECT'})

  # parse all opcode tables FIRST (needed for field type inference)
  enums: dict[str, dict[int, str]] = {}
  table_positions = [(m.start(), m.group(1)) for m in re.finditer(r'Table \d+\. (\w+) Opcodes', full_text)]
  section_positions = [m.start() for m in re.finditer(r'\n\d+\.\d+\.\d+\.\s+\w+\s*\nDescription', full_text)]
  for i, (pos, enc_name) in enumerate(table_positions):
    next_table = table_positions[i+1][0] if i+1 < len(table_positions) else len(full_text)
    next_section = min((s for s in section_positions if s > pos), default=len(full_text))
    table_text = full_text[pos:min(next_table, next_section)]
    if ops := parse_opcode_table(table_text): enums[enc_name + "Op"] = ops
  known_enums = set(enums.keys())

  # parse instruction formats
  formats: dict[str, list[tuple[str, int, int, int | None, str | None]]] = {}
  for i, page in enumerate(pdf.pages[150:200]):
    text = page.extract_text() or ''
    for m in re.finditer(r'\d+\.\d+\.\d+\.\s+(\w+)\s*\n?Description', text):
      fmt_name = m.group(1)
      header_pos = m.start()
      # Look for a Fields table that appears AFTER the section header
      # First check tables on this page that are after the header
      found = False
      tables = page.find_tables()
      for t in tables:
        table_text_start = text.find('Field Name', header_pos)
        if table_text_start > header_pos:  # Table is after header
          t_data = t.extract()
          if t_data and len(t_data) > 1 and t_data[0] and 'Field' in str(t_data[0][0] if t_data[0] else ''):
            fields = parse_fields_table(t_data, fmt_name, known_enums)
            if fields and any(f[0] == 'ENCODING' for f in fields):
              # Check next page for continuation of fields table (only if no ENCODING in continuation)
              if 150+i+1 < len(pdf.pages):
                next_page = pdf.pages[150+i+1]
                for nt in next_page.extract_tables():
                  if nt and len(nt) > 0 and nt[0] and 'Field' in str(nt[0][0] if nt[0] else ''):
                    extra_fields = parse_fields_table(nt, fmt_name, known_enums)
                    # Only merge if this is a continuation (no ENCODING field)
                    if extra_fields and not any(f[0] == 'ENCODING' for f in extra_fields):
                      fields.extend(extra_fields)
                    break
              formats[fmt_name] = fields
              found = True
              break
      # If not found on same page after header, check next page
      if not found and 150+i+1 < len(pdf.pages):
        next_page = pdf.pages[150+i+1]
        for t in next_page.extract_tables():
          if t and len(t) > 1 and t[0] and 'Field' in str(t[0][0] if t[0] else ''):
            fields = parse_fields_table(t, fmt_name, known_enums)
            if fields and any(f[0] == 'ENCODING' for f in fields):
              formats[fmt_name] = fields
              break

  # generate output
  lines = ["# autogenerated from AMD RDNA3.5 ISA PDF - do not edit"]
  lines.append("from enum import IntEnum")
  lines.append("from extra.assembly.rdna3.lib import bits, Inst32, Inst64, SGPR, VGPR, TTMP, s, v")
  lines.append("from extra.assembly.rdna3.lib import SSrc, Src, SImm, Imm")
  lines.append("import functools")
  lines.append("")

  # SrcEnum
  lines.append("class SrcEnum(IntEnum):")
  for val, name in sorted(src_enum.items()): lines.append(f"  {name} = {val}")
  lines.append("")

  # opcode enums
  for cls_name, ops in sorted(enums.items()):
    lines.append(f"class {cls_name}(IntEnum):")
    for opcode, name in sorted(ops.items()): lines.append(f"  {name} = {opcode}")
    lines.append("")

  # determine base class for each format (32-bit vs 64-bit)
  def get_base_class(fields):
    max_bit = max(f[1] for f in fields) if fields else 31
    return "Inst64" if max_bit > 31 else "Inst32"

  # instruction format classes matching rdna3fun.py syntax: bits[hi:lo] == val
  # field order follows assembly syntax (op first, then dst, then sources in order)
  lines.append("# instruction formats")

  # define field order for each format to match assembly syntax
  field_order = {
    'SOP2': ['op', 'sdst', 'ssrc0', 'ssrc1'],
    'SOP1': ['op', 'sdst', 'ssrc0'],
    'SOPC': ['op', 'ssrc0', 'ssrc1'],
    'SOPK': ['op', 'sdst', 'simm16'],
    'SOPP': ['op', 'simm16'],
    'SMEM': ['op', 'sdata', 'sbase', 'soffset', 'offset', 'glc', 'dlc'],
    'VOP1': ['op', 'vdst', 'src0'],
    'VOP2': ['op', 'vdst', 'src0', 'vsrc1'],
    'VOP3': ['op', 'vdst', 'src0', 'src1', 'src2', 'omod', 'neg', 'abs', 'clmp', 'opsel'],
    'VOP3SD': ['op', 'vdst', 'sdst', 'src0', 'src1', 'src2', 'clmp'],
    'VOPC': ['op', 'src0', 'vsrc1'],
    'DS': ['op', 'vdst', 'addr', 'data0', 'data1', 'offset0', 'offset1', 'gds'],
    'FLAT': ['op', 'vdst', 'addr', 'data', 'saddr', 'offset', 'seg', 'dlc', 'glc', 'slc'],
    'MUBUF': ['op', 'vdata', 'vaddr', 'srsrc', 'soffset', 'offset', 'offen', 'idxen', 'glc', 'dlc', 'slc', 'tfe'],
    'MTBUF': ['op', 'vdata', 'vaddr', 'srsrc', 'soffset', 'offset', 'format', 'offen', 'idxen', 'glc', 'dlc', 'slc', 'tfe'],
  }

  def sort_fields(fmt_name, fields):
    order = field_order.get(fmt_name, [])
    def key(f):
      name = f[0].lower()
      if name in order: return order.index(name)
      return 1000 + fields.index(f)  # unknown fields at end
    return sorted(fields, key=key)

  for fmt_name, fields in sorted(formats.items()):
    base = get_base_class(fields)
    enc = next((f for f in fields if f[0] == 'ENCODING'), None)
    # VOP3SD needs special handling - it has src fields in second DWORD but PDF splits across pages
    if fmt_name == 'VOP3SD':
      base = 'Inst64'  # VOP3SD is 64-bit
    lines.append(f"class {fmt_name}({base}):")
    if enc:
      enc_val = f"0b{enc[3]:b}" if enc[3] else "0"
      if enc[1] == enc[2]:
        lines.append(f"  encoding = bits[{enc[1]}] == {enc_val}")
      else:
        lines.append(f"  encoding = bits[{enc[1]}:{enc[2]}] == {enc_val}")
    sorted_fields = sort_fields(fmt_name, [f for f in fields if f[0] != 'ENCODING'])
    # VOP3SD has src0/src1/src2 in second DWORD (not parsed from PDF due to page break)
    # Insert them after sdst, before clmp
    if fmt_name == 'VOP3SD':
      new_fields = []
      for f in sorted_fields:
        new_fields.append(f)
        if f[0].lower() == 'sdst':
          new_fields.append(('src0', 40, 32, None, 'Src'))
          new_fields.append(('src1', 49, 41, None, 'Src'))
          new_fields.append(('src2', 58, 50, None, 'Src'))
      sorted_fields = new_fields
    for field in sorted_fields:
      name, hi, lo = field[0], field[1], field[2]
      field_type = field[4] if len(field) > 4 else None
      # Build the field definition with optional type annotation
      type_ann = f":{field_type}" if field_type else ""
      if hi == lo:
        lines.append(f"  {name.lower()}{type_ann} = bits[{hi}]")
      else:
        lines.append(f"  {name.lower()}{type_ann} = bits[{hi}:{lo}]")
    lines.append("")

  # instruction helper functions using functools.partial
  lines.append("# instruction helpers: functools.partial(Format, Op.OPCODE)")
  for cls_name, ops in sorted(enums.items()):
    fmt_name = cls_name[:-2]  # SOP2Op -> SOP2
    # GLOBAL/SCRATCH use FLAT format with different segment bits
    if fmt_name == "GLOBAL":
      for opcode, name in sorted(ops.items()):
        lines.append(f"{name.lower()} = functools.partial(FLAT, GLOBALOp.{name}, seg=2)")
      continue
    if fmt_name == "SCRATCH":
      for opcode, name in sorted(ops.items()):
        lines.append(f"{name.lower()} = functools.partial(FLAT, SCRATCHOp.{name}, seg=2)")
      continue
    if fmt_name not in formats: continue
    for opcode, name in sorted(ops.items()):
      # VOP1/VOP2 instructions use _e32 suffix (32-bit encoding) to avoid collisions with VOP3
      suffix = "_e32" if fmt_name in ("VOP1", "VOP2") else ""
      lines.append(f"{name.lower()}{suffix} = functools.partial({fmt_name}, {cls_name}.{name})")
  lines.append("")

  # export all SrcEnum values as module-level names
  for val, name in sorted(src_enum.items()):
    lines.append(f"{name} = SrcEnum.{name}")
  lines.append("OFF = NULL")
  lines.append("")

  with open("extra/assembly/rdna3/autogen_rdna3_enum.py", "w") as f: f.write('\n'.join(lines))
  print(f"generated SrcEnum ({len(src_enum)}) + {len(enums)} opcode enums + {len(formats)} format classes")
  for name, ops in sorted(enums.items()): print(f"  {name}: {len(ops)} opcodes")
