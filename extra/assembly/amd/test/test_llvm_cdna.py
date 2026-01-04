#!/usr/bin/env python3
"""Test CDNA assembler/disassembler against LLVM test vectors."""
import unittest, re, subprocess
from tinygrad.helpers import fetch
from extra.assembly.amd.autogen.cdna.ins import *
from extra.assembly.amd.asm import disasm
from extra.assembly.amd.test.helpers import get_llvm_mc

LLVM_BASE = "https://raw.githubusercontent.com/llvm/llvm-project/main/llvm/test/MC/AMDGPU"

def parse_llvm_tests(text: str, mnemonic_filter: str = None, size_filter: int = None) -> list[tuple[str, bytes]]:
  """Parse LLVM test format into (asm, expected_bytes) pairs."""
  tests, lines = [], text.split('\n')
  for i, line in enumerate(lines):
    line = line.strip()
    if not line or line.startswith(('//', '.', ';')): continue
    asm_text = line.split('//')[0].strip()
    if not asm_text or (mnemonic_filter and not asm_text.startswith(mnemonic_filter)): continue
    for j in list(range(max(0, i - 3), i)) + list(range(i, min(i + 3, len(lines)))):
      if m := re.search(r'(?:VI9|GFX9|CHECK)[^:]*:.*?encoding:\s*\[(.*?)\]', lines[j]):
        hex_bytes = m.group(1).replace('0x', '').replace(',', '').replace(' ', '')
      elif m := re.search(r'CHECK[^:]*:\s*\[(0x[0-9a-fA-F,x\s]+)\]', lines[j]):
        hex_bytes = m.group(1).replace('0x', '').replace(',', '').replace(' ', '')
      else: continue
      try:
        data = bytes.fromhex(hex_bytes)
        if size_filter is None or len(data) == size_filter: tests.append((asm_text, data))
      except ValueError: pass
      break
  return tests

# Use gfx9 tests for compatible scalar/vector formats and gfx90a/gfx942 tests for CDNA-specific instructions
# Format: (filename, format_class, op_enum, mcpu, mnemonic_filter, size_filter)
CDNA_TEST_FILES = {
  # Scalar ALU - encoding is stable across GFX9/CDNA
  'sop1': ('gfx9_asm_sop1.s', SOP1, SOP1Op, 'gfx940', None, None),
  'sop2': ('gfx9_asm_sop2.s', SOP2, SOP2Op, 'gfx940', None, None),
  'sopp': ('gfx9_asm_sopp.s', SOPP, SOPPOp, 'gfx940', None, None),
  'sopp_gfx9': ('sopp-gfx9.s', SOPP, SOPPOp, 'gfx940', None, None),
  'sopk': ('gfx9_asm_sopk.s', SOPK, SOPKOp, 'gfx940', None, None),
  'sopc': ('gfx9_asm_sopc.s', SOPC, SOPCOp, 'gfx940', None, None),
  # Vector ALU - encoding is mostly stable
  'vop1': ('gfx9_asm_vop1.s', VOP1, VOP1Op, 'gfx940', None, None),
  'vop1_gfx9': ('vop1-gfx9.s', VOP1, VOP1Op, 'gfx940', None, None),
  'vop2': ('gfx9_asm_vop2.s', VOP2, VOP2Op, 'gfx940', None, None),
  'vopc': ('gfx9_asm_vopc.s', VOPC, VOPCOp, 'gfx940', None, None),
  'vop3p': ('gfx9_asm_vop3p.s', VOP3P, VOP3POp, 'gfx940', None, None),
  'vop3_gfx9': ('vop3-gfx9.s', VOP3A, VOP3AOp, 'gfx940', None, 8),  # Only 64-bit VOP3 instructions
  # Memory instructions
  'ds': ('gfx9_asm_ds.s', DS, DSOp, 'gfx940', None, None),
  'ds_gfx9': ('ds-gfx9.s', DS, DSOp, 'gfx940', None, None),
  # CDNA memory instructions (gfx90a has correct FLAT/MUBUF encodings with acc registers)
  'flat_gfx90a': ('gfx90a_ldst_acc.s', FLAT, FLATOp, 'gfx90a', 'flat_', None),
  'global_gfx90a': ('gfx90a_ldst_acc.s', FLAT, FLATOp, 'gfx90a', 'global_', None),
  'mubuf_gfx90a': ('gfx90a_ldst_acc.s', MUBUF, MUBUFOp, 'gfx90a', 'buffer_', None),
  'mubuf_gfx9': ('mubuf-gfx9.s', MUBUF, MUBUFOp, 'gfx940', None, None),
  'scratch_gfx942': ('flat-scratch-gfx942.s', FLAT, FLATOp, 'gfx942', 'scratch_', None),
  # CDNA-specific: MFMA/MAI instructions
  'mai': ('mai-gfx942.s', VOP3P, VOP3POp, 'gfx942', None, None),
  # SDWA and DPP format tests for VOP1 (VOP2 has different bit layout, tested separately)
  'sdwa_vop1': ('gfx9_asm_vop1.s', SDWA, VOP1Op, 'gfx940', None, None),
  'dpp_vop1': ('gfx9_asm_vop1.s', DPP, VOP1Op, 'gfx940', None, None),
}

class TestLLVMCDNA(unittest.TestCase):
  """Test CDNA instruction format decode/encode roundtrip and disassembly."""
  tests: dict[str, list[tuple[str, bytes]]] = {}

  @classmethod
  def setUpClass(cls):
    for name, (filename, _, _, _, mnemonic_filter, size_filter) in CDNA_TEST_FILES.items():
      try:
        data = fetch(f"{LLVM_BASE}/{filename}").read_bytes()
        cls.tests[name] = parse_llvm_tests(data.decode('utf-8', errors='ignore'), mnemonic_filter, size_filter)
      except Exception as e:
        print(f"Warning: couldn't fetch {filename}: {e}")
        cls.tests[name] = []

def _get_val(v): return v.val if hasattr(v, 'val') else v

def _filter_and_decode(tests, fmt_cls, op_enum):
  """Filter tests and decode instructions, yielding (asm_text, data, decoded, error)."""
  fn, is_sdwa, is_dpp = fmt_cls.__name__, fmt_cls.__name__ == 'SDWA', fmt_cls.__name__ == 'DPP'
  for asm_text, data in tests:
    has_lit = False
    # SDWA/DPP format tests: only accept matching 8-byte instructions
    if is_sdwa:
      if len(data) != 8 or data[0] != 0xf9: continue
    elif is_dpp:
      if len(data) != 8 or data[0] != 0xfa: continue
    elif fmt_cls._size() == 4 and len(data) == 8:
      if data[0] in (0xf9, 0xfa): continue  # Skip SDWA/DPP (tested separately)
      has_lit = data[0] == 255 or (len(data) >= 2 and data[1] == 255 and fn in ('SOP2', 'SOPC'))
      if fn == 'SOPK': has_lit = has_lit or ((int.from_bytes(data[:4], 'little') >> 23) & 0x1f) == 20
      if fn == 'VOP2': has_lit = has_lit or ((int.from_bytes(data[:4], 'little') >> 25) & 0x3f) in (23, 24, 36, 37)
      if not has_lit: continue
    if len(data) > fmt_cls._size() + (4 if has_lit else 0): continue
    try:
      decoded = fmt_cls.from_bytes(data)
      # For SDWA/DPP, opcode location depends on VOP1 vs VOP2
      if is_sdwa or is_dpp:
        vop2_op = _get_val(decoded._values.get('vop2_op', 0))
        op_val = _get_val(decoded._values.get('vop_op', 0)) if vop2_op == 0x3f else vop2_op
      else:
        op_val = _get_val(decoded._values.get('op', 0))
      try: op_enum(op_val)
      except ValueError: continue
      yield asm_text, data, decoded, None
    except Exception as e:
      yield asm_text, data, None, str(e)

def _make_roundtrip_test(name):
  def test(self):
    _, fmt_cls, op_enum, _, _, _ = CDNA_TEST_FILES[name]
    passed, failed, failures = 0, 0, []
    for asm_text, data, decoded, error in _filter_and_decode(self.tests.get(name, []), fmt_cls, op_enum):
      if error: failed += 1; failures.append(f"'{asm_text}': {error}"); continue
      if decoded.to_bytes()[:len(data)] == data: passed += 1
      else: failed += 1; failures.append(f"'{asm_text}': orig={data.hex()} reenc={decoded.to_bytes()[:len(data)].hex()}")
    print(f"CDNA {name.upper()} roundtrip: {passed} passed, {failed} failed")
    if failures[:5]: print("  " + "\n  ".join(failures[:5]))
    self.assertEqual(failed, 0)
  return test

def _make_disasm_test(name):
  def test(self):
    _, fmt_cls, op_enum, _, _, _ = CDNA_TEST_FILES[name]
    passed, failed, failures = 0, 0, []
    for asm_text, data, decoded, error in _filter_and_decode(self.tests.get(name, []), fmt_cls, op_enum):
      if error: failed += 1; failures.append(f"'{asm_text}': {error}"); continue
      if decoded.to_bytes()[:len(data)] != data: failed += 1; failures.append(f"'{asm_text}': roundtrip failed"); continue
      if not (disasm_text := disasm(decoded)) or not disasm_text.strip(): failed += 1; failures.append(f"'{asm_text}': empty disassembly"); continue
      passed += 1
    print(f"CDNA {name.upper()} disasm: {passed} passed, {failed} failed")
    if failures[:5]: print("  " + "\n  ".join(failures[:5]))
    self.assertEqual(failed, 0)
  return test

for name in CDNA_TEST_FILES:
  setattr(TestLLVMCDNA, f'test_{name}_roundtrip', _make_roundtrip_test(name))
  setattr(TestLLVMCDNA, f'test_{name}_disasm', _make_disasm_test(name))

if __name__ == "__main__":
  unittest.main()
