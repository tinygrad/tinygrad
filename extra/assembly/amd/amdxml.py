# AMD machine-readable ISA XML parser - generates enum.py and ins.py
# XML: https://gpuopen.com/download/machine-readable-isa/latest/
import xml.etree.ElementTree as ET, zipfile
from tinygrad.helpers import fetch

XML_URL = "https://gpuopen.com/download/machine-readable-isa/latest/"
ARCH_MAP = {"amdgpu_isa_rdna3_5.xml": "rdna3", "amdgpu_isa_rdna4.xml": "rdna4", "amdgpu_isa_cdna4.xml": "cdna"}
# Map XML encoding names to codebase names (arch-specific overrides in ARCH_NAME_MAP)
NAME_MAP = {"VOP3_SDST_ENC": "VOP3SD", "VOPDXY": "VOPD", "VDS": "DS"}
ARCH_NAME_MAP = {"cdna": {"VOP3": "VOP3A", "VOP3_SDST_ENC": "VOP3B"}}
# Instructions missing from XML but present in PDF
FIXES = {"rdna3": {"SOPP": {8: "S_WAITCNT_DEPCTR", 58: "S_TTRACEDATA", 59: "S_TTRACEDATA_IMM"},
                   "SOPK": {22: "S_SUBVECTOR_LOOP_BEGIN", 23: "S_SUBVECTOR_LOOP_END"},
                   "SMEM": {34: "S_ATC_PROBE", 35: "S_ATC_PROBE_BUFFER"},
                   "FLAT": {40: "GLOBAL_LOAD_ADDTID_B32", 41: "GLOBAL_STORE_ADDTID_B32", 55: "FLAT_ATOMIC_CSUB_U32"}},
         "rdna4": {"SOP1": {80: "S_GET_BARRIER_STATE", 81: "S_BARRIER_INIT", 82: "S_BARRIER_JOIN"},
                   "SOPP": {9: "S_WAITCNT", 21: "S_BARRIER_LEAVE", 58: "S_TTRACEDATA", 59: "S_TTRACEDATA_IMM"},
                   "SMEM": {34: "S_ATC_PROBE", 35: "S_ATC_PROBE_BUFFER"}}}
# Encoding suffixes to strip (variants we don't generate separate classes for)
_ENC_SUFFIXES = ("_INST_LITERAL", "_VOP_DPP16", "_VOP_DPP8", "_VOP_DPP", "_VOP_SDWA", "_NSA1", "_MFMA")
# Field name normalization
_FIELD_RENAMES = {"opsel_hi_2": "opsel_hi2", "op_sel_hi_2": "opsel_hi2", "op_sel": "opsel", "bound_ctrl": "bc",
                  "tgt": "target", "row_en": "row", "unorm": "unrm", "clamp": "clmp", "wait_exp": "waitexp"}

def _strip_enc(name: str) -> str:
  """Strip ENC_ prefix and encoding variant suffixes."""
  name = name.removeprefix("ENC_")
  for sfx in _ENC_SUFFIXES: name = name.replace(sfx, "")
  return name

def _map_flat(enc_name: str, instr_name: str) -> str:
  """Map FLAT/GLOBAL/SCRATCH encoding to correct enum based on instruction prefix."""
  if enc_name in ("FLAT_GLBL", "FLAT_GLOBAL"): return "GLOBAL"
  if enc_name == "FLAT_SCRATCH": return "SCRATCH"
  if enc_name in ("FLAT", "VFLAT", "VGLOBAL", "VSCRATCH"):
    v = "V" if enc_name.startswith("V") else ""
    if instr_name.startswith("GLOBAL_"): return f"{v}GLOBAL"
    if instr_name.startswith("SCRATCH_"): return f"{v}SCRATCH"
    return f"{v}FLAT"
  return enc_name

def parse_xml(filename: str, arch: str):
  root = ET.fromstring(zipfile.ZipFile(fetch(XML_URL)).read(filename))
  name_map = {**NAME_MAP, **ARCH_NAME_MAP.get(arch, {})}
  encodings, enums, types, fmts, op_types_set = {}, {}, {}, {}, set()
  # Extract DataFormats with BitCount
  for df in root.findall("ISA/DataFormats/DataFormat"):
    name, bits = df.findtext("DataFormatName"), df.findtext("BitCount")
    if name and bits: fmts[name] = int(bits)
  # Extract instruction operand info keyed by (instruction_name, encoding_name)
  for instr in root.findall("ISA/Instructions/Instruction"):
    name = instr.findtext("InstructionName")
    for enc in instr.findall("InstructionEncodings/InstructionEncoding"):
      if enc.findtext("EncodingCondition") != "default": continue
      base = _map_flat(_strip_enc(enc.findtext("EncodingName")), name)
      enc_name = name_map.get(base, base)
      op_info = {}
      for op in enc.findall("Operands/Operand"):
        field = op.findtext("FieldName")
        if not field: continue
        fmt, size, otype = op.findtext("DataFormatName"), op.findtext("OperandSize"), op.findtext("OperandType")
        if fmt and fmt not in fmts: fmts[fmt] = 0
        if otype: op_types_set.add(otype)
        op_info[field.lower()] = (fmt, int(size) if size else 0, otype)
      if op_info: types[(name, enc_name)] = op_info
  # Extract encoding definitions
  def norm_field(n):
    for old, new in _FIELD_RENAMES.items(): n = n.replace(old, new)
    return n
  for enc in root.findall("ISA/Encodings/Encoding"):
    name = enc.findtext("EncodingName")
    if not name.startswith("ENC_") and name not in ("VOP3_SDST_ENC", "VOPDXY"): continue
    if any(s in name for s in ("LITERAL", "NSA", "DPP16", "DPP8")): continue
    fields = [(norm_field(f.findtext("FieldName").lower()), int(f.find("BitLayout/Range").findtext("BitOffset") or 0) + int(f.find("BitLayout/Range").findtext("BitCount") or 0) - 1,
               int(f.find("BitLayout/Range").findtext("BitOffset") or 0))
              for f in enc.findall(".//MicrocodeFormat/BitMap/Field") if f.find("BitLayout/Range") is not None]
    ident = (enc.findall("EncodingIdentifiers/EncodingIdentifier") or [None])[0]
    enc_field = next((f for f in fields if f[0] == "encoding"), None)
    enc_bits = "".join(ident.text[len(ident.text)-1-b] for b in range(enc_field[1], enc_field[2]-1, -1)) if ident is not None and enc_field else None
    base_name = _strip_enc(name)
    encodings[name_map.get(base_name, base_name)] = (fields, enc_bits)
  # Extract instruction opcodes into enums
  for instr in root.findall("ISA/Instructions/Instruction"):
    name = instr.findtext("InstructionName")
    for enc in instr.findall("InstructionEncodings/InstructionEncoding"):
      if enc.findtext("EncodingCondition") != "default": continue
      base = _map_flat(_strip_enc(enc.findtext("EncodingName")), name)
      # ADDTID instructions go in both FLAT and GLOBAL enums
      if base == "GLOBAL" and "ADDTID" in name: enums.setdefault("FLAT", {})[int(enc.findtext("Opcode") or 0)] = name
      if base == "VGLOBAL" and "ADDTID" in name: enums.setdefault("VFLAT", {})[int(enc.findtext("Opcode") or 0)] = name
      enums.setdefault(name_map.get(base, base), {})[int(enc.findtext("Opcode") or 0)] = name
  return encodings, enums, types, fmts, op_types_set

def write_common(all_fmts, all_op_types, path):
  lines = ["# autogenerated from AMD ISA XML - do not edit", "from enum import Enum, auto", ""]
  # Add Fmt enum (union of all architectures)
  lines.append("class Fmt(Enum):")
  for fmt in sorted(all_fmts.keys()): lines.append(f"  {fmt} = auto()")
  lines.append("")
  # Add FMT_BITS dict mapping Fmt -> BitCount
  lines.append("FMT_BITS = {")
  for fmt, bits in sorted(all_fmts.items()): lines.append(f"  Fmt.{fmt}: {bits},")
  lines.append("}")
  lines.append("")
  # Add OpType enum (union of all architectures)
  lines.append("class OpType(Enum):")
  for ot in sorted(all_op_types): lines.append(f"  {ot} = auto()")
  with open(path, "w") as f: f.write("\n".join(lines))

def write_enum(enums, path):
  lines = ["# autogenerated from AMD ISA XML - do not edit", "from enum import Enum", "from extra.assembly.amd.autogen.common import Fmt, FMT_BITS, OpType  # noqa: F401", ""]
  for name, ops in sorted(enums.items()):
    if not ops: continue
    suffix = "_E32" if name in ("VOP1", "VOP2", "VOPC") else "_E64" if name == "VOP3" else ""
    lines.append(f"class {name}Op(Enum):")
    aliases = []
    for op, mem in sorted(ops.items()):
      msuf = suffix if name != "VOP3" or op < 512 else ""
      lines.append(f"  {mem}{msuf} = {op}")
      if msuf: aliases.append((mem, msuf))
    for mem, msuf in aliases: lines.append(f"  {mem} = {mem}{msuf}")
    lines.append("")
  with open(path, "w") as f: f.write("\n".join(lines))

def write_ins(encodings, enums, arch, path):
  def field_def(name, hi, lo, fmt, enc_bits=None):
    bits = hi - lo + 1
    if name == "encoding" and enc_bits: return f"FixedBitField({hi}, {lo}, 0b{enc_bits})"
    if name == "op" and fmt not in ("DPP", "SDWA"): return f"EnumBitField({hi}, {lo}, {fmt}Op)"
    if name in ("opx", "opy"): return f"EnumBitField({hi}, {lo}, VOPDOp)"
    if name == "vdsty": return f"VDSTYField({hi}, {lo})"
    if name in ("vdst", "vdstx", "vsrc0", "vsrc1", "vsrc2", "vsrc3", "vsrcx1", "vsrcy1", "vaddr", "vdata", "data", "data0", "data1", "addr") and bits == 8: return f"VGPRField({hi}, {lo})"
    if name == "sbase" and bits == 6: return f"SBaseField({hi}, {lo})"
    if name in ("srsrc", "ssamp") and bits == 5: return f"SRsrcField({hi}, {lo})"
    if name in ("sdst", "sdata") and bits == 7: return f"SGPRField({hi}, {lo})"
    if name in ("soffset", "saddr") and bits == 7: return f"SGPRField({hi}, {lo}, default=NULL)"
    if name.startswith("ssrc") and bits == 8: return f"SSrcField({hi}, {lo})"
    if name in ("saddr", "soffset") and bits == 8: return f"SSrcField({hi}, {lo}, default=NULL)"
    if name.startswith("src") and bits == 9: return f"SrcField({hi}, {lo})"
    if fmt == "VOP3P" and name == "opsel_hi": return f"BitField({hi}, {lo}, default=3)"
    if fmt == "VOP3P" and name == "opsel_hi2": return f"BitField({hi}, {lo}, default=1)"
    return f"BitField({hi}, {lo})"
  ORDER = ['encoding', 'op', 'opx', 'opy', 'vdst', 'vdstx', 'vdsty', 'sdst', 'vdata', 'sdata', 'addr', 'vaddr', 'data', 'data0', 'data1',
           'src0', 'srcx0', 'srcy0', 'vsrc0', 'ssrc0', 'src1', 'vsrc1', 'vsrcx1', 'vsrcy1', 'ssrc1', 'src2', 'vsrc2', 'src3', 'vsrc3',
           'saddr', 'sbase', 'srsrc', 'ssamp', 'soffset', 'offset', 'simm16', 'en', 'target', 'attr', 'attr_chan',
           'omod', 'neg', 'neg_hi', 'abs', 'clmp', 'opsel', 'opsel_hi', 'waitexp', 'wait_va',
           'dmask', 'dim', 'seg', 'format', 'offen', 'idxen', 'glc', 'dlc', 'slc', 'tfe', 'unrm', 'done', 'row']
  sort_fields = lambda fields: sorted(fields, key=lambda f: (ORDER.index(f[0]) if f[0] in ORDER else 999, f[2]))

  lines = ["# autogenerated from AMD ISA XML - do not edit", "# ruff: noqa: F401,F403",
           "from extra.assembly.amd.dsl import *", f"from extra.assembly.amd.autogen.{arch}.enum import *", "import functools", ""]
  for enc_name, (fields, enc_bits) in sorted(encodings.items()):
    # FLAT/VFLAT variants
    if enc_name in ("FLAT", "VFLAT"):
      prefix = "V" if enc_name == "VFLAT" else ""
      for cls, seg, op_enum in [(f"{prefix}FLAT", 0, f"{prefix}FLATOp"), (f"{prefix}GLOBAL", 2, f"{prefix}GLOBALOp"), (f"{prefix}SCRATCH", 1, f"{prefix}SCRATCHOp")]:
        lines.append(f"class {cls}(Inst):")
        for fn, hi, lo in sort_fields(fields):
          if fn == "seg": lines.append(f"  seg = FixedBitField({hi}, {lo}, {seg})")
          elif fn == "op": lines.append(f"  op = EnumBitField({hi}, {lo}, {op_enum})")
          else: lines.append(f"  {fn} = {field_def(fn, hi, lo, cls, enc_bits)}")
        lines.append("")
    elif enc_name not in ("FLAT_GLOBAL", "FLAT_SCRATCH", "FLAT_GLBL", "VGLOBAL", "VSCRATCH", "DPP", "SDWA"):
      lines.append(f"class {enc_name}(Inst):")
      for fn, hi, lo in sort_fields(fields):
        lines.append(f"  {fn} = {field_def(fn, hi, lo, enc_name, enc_bits if fn == 'encoding' else None)}")
      lines.append("")
  # SDST variants
  for base, field in [("VOP1", "vdst = SSrcField(24, 17)"), ("VOP3", "vdst = SSrcField(7, 0)")]:
    if base in encodings: lines += [f"class {base}_SDST({base}):", f"  {field}", ""]
  # Instruction helpers
  lines.append("# instruction helpers")
  SDST_OPS = {"V_READFIRSTLANE_B32", "V_READLANE_B32"}
  for fmt, ops in sorted(enums.items()):
    if fmt not in encodings and fmt not in ("GLOBAL", "SCRATCH", "VGLOBAL", "VSCRATCH"): continue
    suffix = "_E32" if fmt in ("VOP1", "VOP2", "VOPC") else "_E64" if fmt == "VOP3" else ""
    for op, name in sorted(ops.items()):
      msuf = suffix if fmt != "VOP3" or op < 512 else ""
      cls = "VOP1_SDST" if fmt == "VOP1" and name in SDST_OPS else "VOP3_SDST" if fmt == "VOP3" and (name in SDST_OPS or op < 256) else fmt
      lines.append(f"{name.lower()}{msuf.lower()} = functools.partial({cls}, {fmt}Op.{name}{msuf})")
  with open(path, "w") as f: f.write("\n".join(lines))

def write_operands(types, enums, arch, path):
  # Build set of valid (name, fmt) pairs from enums
  valid = {(name, fmt) for fmt, ops in enums.items() for name in ops.values()}
  lines = ["# autogenerated from AMD ISA XML - do not edit",
           "from extra.assembly.amd.autogen.common import Fmt, OpType",
           f"from extra.assembly.amd.autogen.{arch}.enum import *", ""]
  lines.append("# instruction operand info: {Op: {field: (Fmt, size_bits, OpType)}}")
  lines.append("OPERANDS = {")
  def fmt_val(v):
    fmt, size, otype = v
    return f"({f'Fmt.{fmt}' if fmt else 'None'}, {size}, {f'OpType.{otype}' if otype else 'None'})"
  for (name, enc_base), fields in sorted(types.items()):
    if (name, enc_base) not in valid: continue
    fstr = ", ".join(f'"{k}": {fmt_val(v)}' for k, v in sorted(fields.items()))
    lines.append(f'  {enc_base}Op.{name}: {{{fstr}}},')
  lines.append("}")
  with open(path, "w") as f: f.write("\n".join(lines))

if __name__ == "__main__":
  import pathlib
  # First pass: collect all fmts and op_types across all architectures
  all_fmts, all_op_types, arch_data = {}, set(), {}
  for filename, arch in ARCH_MAP.items():
    print(f"Processing {filename} -> {arch}")
    encodings, enums, types, fmts, op_types_set = parse_xml(filename, arch)
    for fmt, ops in FIXES.get(arch, {}).items(): enums.setdefault(fmt, {}).update(ops)
    arch_data[arch] = (encodings, enums, types)
    for fmt, bits in fmts.items():
      assert fmt not in all_fmts or all_fmts[fmt] == bits, f"FMT_BITS mismatch for {fmt}: {all_fmts[fmt]} vs {bits}"
      all_fmts[fmt] = bits
    all_op_types.update(op_types_set)
  # Write common.py with merged Fmt and OpType enums
  common_path = pathlib.Path(__file__).parent / "autogen" / "common.py"
  write_common(all_fmts, all_op_types, common_path)
  print(f"Wrote common.py: {len(all_fmts)} formats, {len(all_op_types)} op types")
  # Write per-arch files
  for arch, (encodings, enums, types) in arch_data.items():
    base = pathlib.Path(__file__).parent / "autogen" / arch
    write_enum(enums, base / "enum.py")
    write_ins(encodings, enums, arch, base / "ins.py")
    write_operands(types, enums, arch, base / "operands.py")
    print(f"  {arch}: {len(encodings)} encodings, {sum(len(v) for v in enums.values())} instructions, {len(types)} types")
