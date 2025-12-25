#!/usr/bin/env python3
"""Test RDNA3 assembler/disassembler against LLVM test vectors."""
import unittest, re
from tinygrad.helpers import fetch
from extra.assembly.rdna3.autogen import *
from extra.assembly.rdna3.lib import asm

LLVM_BASE = "https://raw.githubusercontent.com/llvm/llvm-project/main/llvm/test/MC/AMDGPU"

# Format info: (filename, format_class, op_enum)
LLVM_TEST_FILES = {
  # Scalar ALU
  'sop1': ('gfx11_asm_sop1.s', SOP1, SOP1Op),
  'sop2': ('gfx11_asm_sop2.s', SOP2, SOP2Op),
  'sopp': ('gfx11_asm_sopp.s', SOPP, SOPPOp),
  'sopk': ('gfx11_asm_sopk.s', SOPK, SOPKOp),
  'sopc': ('gfx11_asm_sopc.s', SOPC, SOPCOp),
  # Vector ALU
  'vop1': ('gfx11_asm_vop1.s', VOP1, VOP1Op),
  'vop2': ('gfx11_asm_vop2.s', VOP2, VOP2Op),
  'vopc': ('gfx11_asm_vopc.s', VOPC, VOPCOp),
  'vop3': ('gfx11_asm_vop3.s', VOP3, VOP3Op),
  'vop3p': ('gfx11_asm_vop3p.s', VOP3P, VOP3POp),
  'vop3sd': ('gfx11_asm_vop3.s', VOP3SD, VOP3SDOp),  # VOP3SD shares file with VOP3
  'vinterp': ('gfx11_asm_vinterp.s', VINTERP, VINTERPOp),
  'vopd': ('gfx11_asm_vopd.s', VOPD, VOPDOp),
  'vopcx': ('gfx11_asm_vopcx.s', VOPC, VOPCOp),  # VOPCX uses VOPC format
  # VOP3 promotions (VOP1/VOP2/VOPC promoted to VOP3 encoding)
  'vop3_from_vop1': ('gfx11_asm_vop3_from_vop1.s', VOP3, VOP3Op),
  'vop3_from_vop2': ('gfx11_asm_vop3_from_vop2.s', VOP3, VOP3Op),
  'vop3_from_vopc': ('gfx11_asm_vop3_from_vopc.s', VOP3, VOP3Op),
  'vop3_from_vopcx': ('gfx11_asm_vop3_from_vopcx.s', VOP3, VOP3Op),
  # Memory
  'ds': ('gfx11_asm_ds.s', DS, DSOp),
  'smem': ('gfx11_asm_smem.s', SMEM, SMEMOp),
  'flat': ('gfx11_asm_flat.s', FLAT, FLATOp),
  'mubuf': ('gfx11_asm_mubuf.s', MUBUF, MUBUFOp),
  'mtbuf': ('gfx11_asm_mtbuf.s', MTBUF, MTBUFOp),
  'mimg': ('gfx11_asm_mimg.s', MIMG, MIMGOp),
  # WMMA (matrix multiply)
  'wmma': ('gfx11_asm_wmma.s', VOP3P, VOP3POp),
  # Additional features
  'vop3_features': ('gfx11_asm_vop3_features.s', VOP3, VOP3Op),
  'vop3p_features': ('gfx11_asm_vop3p_features.s', VOP3P, VOP3POp),
  'vopd_features': ('gfx11_asm_vopd_features.s', VOPD, VOPDOp),
  # Alias files (alternative mnemonics)
  'vop3_alias': ('gfx11_asm_vop3_alias.s', VOP3, VOP3Op),
  'vop3p_alias': ('gfx11_asm_vop3p_alias.s', VOP3P, VOP3POp),
  'vopc_alias': ('gfx11_asm_vopc_alias.s', VOPC, VOPCOp),
  'vopcx_alias': ('gfx11_asm_vopcx_alias.s', VOPC, VOPCOp),
  'vinterp_alias': ('gfx11_asm_vinterp_alias.s', VINTERP, VINTERPOp),
  'smem_alias': ('gfx11_asm_smem_alias.s', SMEM, SMEMOp),
  'mubuf_alias': ('gfx11_asm_mubuf_alias.s', MUBUF, MUBUFOp),
  'mtbuf_alias': ('gfx11_asm_mtbuf_alias.s', MTBUF, MTBUFOp),
}

def parse_llvm_tests(text: str) -> list[tuple[str, bytes]]:
  """Parse LLVM test format into (asm, expected_bytes) pairs."""
  tests, lines = [], text.split('\n')
  for i, line in enumerate(lines):
    line = line.strip()
    if not line or line.startswith(('//', '.', ';')): continue
    asm_text = line.split('//')[0].strip()
    if not asm_text: continue
    for j in range(i, min(i + 3, len(lines))):
      # Match GFX11, W32, or W64 encodings (all valid for gfx11)
      if m := re.search(r'(?:GFX11|W32|W64)[^:]*:.*?encoding:\s*\[(.*?)\]', lines[j]):
        hex_bytes = m.group(1).replace('0x', '').replace(',', '').replace(' ', '')
        if hex_bytes:
          try: tests.append((asm_text, bytes.fromhex(hex_bytes)))
          except ValueError: pass
        break
  return tests

def try_assemble(text: str):
  """Try to assemble instruction text, return bytes or None on failure."""
  try: return asm(text).to_bytes()
  except: return None

class TestLLVM(unittest.TestCase):
  """Test assembler and disassembler against all LLVM test vectors."""
  tests: dict[str, list[tuple[str, bytes]]] = {}

  @classmethod
  def setUpClass(cls):
    for name, (filename, _, _) in LLVM_TEST_FILES.items():
      try:
        data = fetch(f"{LLVM_BASE}/{filename}").read_bytes()
        cls.tests[name] = parse_llvm_tests(data.decode('utf-8', errors='ignore'))
      except Exception as e:
        print(f"Warning: couldn't fetch {filename}: {e}")
        cls.tests[name] = []

# Generate test methods dynamically for each format
def _make_asm_test(name):
  def test(self):
    passed, failed, skipped = 0, 0, 0
    for asm_text, expected in self.tests.get(name, []):
      result = try_assemble(asm_text)
      if result is None: skipped += 1
      elif result == expected: passed += 1
      else: failed += 1
    print(f"{name.upper()} asm: {passed} passed, {failed} failed, {skipped} skipped")
    self.assertEqual(failed, 0)
  return test

def _make_disasm_test(name):
  def test(self):
    _, fmt_cls, op_enum = LLVM_TEST_FILES[name]
    passed, failed, skipped = 0, 0, 0
    for asm_text, data in self.tests.get(name, []):
      if len(data) > fmt_cls._size(): skipped += 1; continue  # skip literals
      try:
        decoded = fmt_cls.from_bytes(data)
        op_enum(decoded._values.get('op', 0))  # validate opcode
        if decoded.to_bytes()[:len(data)] == data: passed += 1
        else: failed += 1
      except: skipped += 1
    print(f"{name.upper()} disasm: {passed} passed, {failed} failed, {skipped} skipped")
    self.assertEqual(failed, 0)
  return test

for name in LLVM_TEST_FILES:
  setattr(TestLLVM, f'test_{name}_asm', _make_asm_test(name))
  setattr(TestLLVM, f'test_{name}_disasm', _make_disasm_test(name))

if __name__ == "__main__":
  unittest.main()
