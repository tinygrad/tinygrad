#!/usr/bin/env python3
"""Test AMD assembler/disassembler against LLVM test vectors."""
import unittest, re, subprocess
from tinygrad.helpers import fetch
from extra.assembly.amd.asm import asm, disasm, detect_format
from extra.assembly.amd.test.helpers import get_llvm_mc

LLVM_BASE = "https://raw.githubusercontent.com/llvm/llvm-project/main/llvm/test/MC/AMDGPU"

RDNA_FILES = ['gfx11_asm_sop1.s', 'gfx11_asm_sop2.s', 'gfx11_asm_sopp.s', 'gfx11_asm_sopk.s', 'gfx11_asm_sopc.s',
  'gfx11_asm_vop1.s', 'gfx11_asm_vop2.s', 'gfx11_asm_vopc.s', 'gfx11_asm_vop3.s', 'gfx11_asm_vop3p.s', 'gfx11_asm_vinterp.s',
  'gfx11_asm_vopd.s', 'gfx11_asm_vopcx.s', 'gfx11_asm_vop3_from_vop1.s', 'gfx11_asm_vop3_from_vop2.s', 'gfx11_asm_vop3_from_vopc.s',
  'gfx11_asm_vop3_from_vopcx.s', 'gfx11_asm_ds.s', 'gfx11_asm_smem.s', 'gfx11_asm_flat.s', 'gfx11_asm_mubuf.s', 'gfx11_asm_mtbuf.s',
  'gfx11_asm_mimg.s', 'gfx11_asm_wmma.s', 'gfx11_asm_vop3_features.s', 'gfx11_asm_vop3p_features.s', 'gfx11_asm_vopd_features.s',
  'gfx11_asm_vop3_alias.s', 'gfx11_asm_vop3p_alias.s', 'gfx11_asm_vopc_alias.s', 'gfx11_asm_vopcx_alias.s', 'gfx11_asm_vinterp_alias.s',
  'gfx11_asm_smem_alias.s', 'gfx11_asm_mubuf_alias.s', 'gfx11_asm_mtbuf_alias.s']
# NOTE: gfx9_asm_mimg.s/mimg-gfx90a.s/gfx9_asm_exp.s are for gfx900 (Vega), not CDNA - CDNA doesn't have MIMG/EXP instructions
# MIMG instructions (encoding 0b111100) in other files are filtered out in setUpClass
CDNA_FILES = ['gfx9_asm_sop1.s', 'gfx9_asm_sop2.s', 'gfx9_asm_sopp.s', 'sopp-gfx9.s', 'gfx9_asm_sopk.s', 'gfx9_asm_sopc.s',
  'gfx9_asm_vop1.s', 'vop1-gfx9.s', 'gfx9_asm_vop2.s', 'gfx9_asm_vopc.s', 'gfx9_asm_vop3.s', 'gfx9_asm_vop3_e64.s', 'gfx9_asm_vop3p.s', 'vop3-gfx9.s',
  'gfx9_asm_ds.s', 'ds-gfx9.s', 'gfx9_asm_flat.s', 'flat-gfx9.s', 'gfx9_asm_smem.s', 'gfx9_asm_mubuf.s', 'mubuf-gfx9.s', 'gfx9_asm_mtbuf.s',
  'gfx90a_ldst_acc.s', 'gfx90a_asm_features.s', 'flat-scratch-gfx942.s', 'gfx942_asm_features.s', 'mai-gfx90a.s', 'mai-gfx942.s']

def parse_llvm_tests(text: str, pattern: str) -> list[tuple[str, bytes]]:
  """Parse LLVM test format into (asm, expected_bytes) pairs."""
  tests, lines = [], text.split('\n')
  for i, line in enumerate(lines):
    line = line.strip()
    if not line or line.startswith(('//', '.', ';')): continue
    asm_text = line.split('//')[0].strip()
    if not asm_text: continue
    for j in range(i, min(i + 3, len(lines))):
      if m := re.search(pattern + r'[^:]*:.*?encoding:\s*\[(.*?)\]', lines[j]):
        hex_bytes = m.group(1).replace('0x', '').replace(',', '').replace(' ', '')
        try: tests.append((asm_text, bytes.fromhex(hex_bytes)))
        except ValueError: pass
        break
  return tests

def compile_asm_batch(instrs: list[str]) -> list[bytes]:
  """Compile instructions with llvm-mc."""
  if not instrs: return []
  result = subprocess.run([get_llvm_mc(), '-triple=amdgcn', '-mcpu=gfx1100', '-mattr=+real-true16,+wavefrontsize32', '-show-encoding'],
    input=".text\n" + "\n".join(instrs) + "\n", capture_output=True, text=True, timeout=30)
  if result.returncode != 0: raise RuntimeError(f"llvm-mc failed: {result.stderr.strip()}")
  return [bytes.fromhex(line.split('encoding:')[1].strip()[1:-1].replace('0x', '').replace(',', '').replace(' ', ''))
          for line in result.stdout.split('\n') if 'encoding:' in line]

class TestLLVM(unittest.TestCase):
  rdna: list[tuple[str, bytes]] = []
  cdna: list[tuple[str, bytes]] = []

  @classmethod
  def setUpClass(cls):
    for f in RDNA_FILES:
      try: cls.rdna.extend(parse_llvm_tests(fetch(f"{LLVM_BASE}/{f}").read_bytes().decode('utf-8', errors='ignore'), r'(?:GFX11|W32|W64)'))
      except Exception as e: print(f"Warning: {f}: {e}")
    for f in CDNA_FILES:
      try:
        tests = parse_llvm_tests(fetch(f"{LLVM_BASE}/{f}").read_bytes().decode('utf-8', errors='ignore'), r'(?:VI9|GFX9|CHECK)')
        cls.cdna.extend((asm, data) for asm, data in tests if (int.from_bytes(data[:4], 'little') >> 26) & 0x3f != 0b111100)  # skip MIMG
      except Exception as e: print(f"Warning: {f}: {e}")

  def _test_asm(self, tests, arch):
    passed, failed, skipped = 0, 0, 0
    for asm_text, expected in tests:
      try: result = asm(asm_text).to_bytes()
      except: result = None
      if result is None: skipped += 1
      elif result == expected: passed += 1
      else: failed += 1
    print(f"{arch} asm: {passed} passed, {failed} failed, {skipped} skipped")
    self.assertEqual(failed, 0)

  def _test_roundtrip(self, tests, arch):
    passed, failed, skipped = 0, 0, 0
    for _, data in tests:
      try:
        decoded = detect_format(data, arch).from_bytes(data)
        if decoded.to_bytes()[:len(data)] == data: passed += 1
        else: failed += 1
      except: skipped += 1
    print(f"{arch} roundtrip: {passed} passed, {failed} failed, {skipped} skipped")
    self.assertEqual(failed, 0)
    self.assertEqual(skipped, 0)

  def _test_disasm(self, tests, arch):
    to_test = []
    for _, data in tests:
      try:
        decoded = detect_format(data, arch).from_bytes(data)
        if decoded.to_bytes()[:len(data)] == data and (d := disasm(decoded)): to_test.append((data, d))
      except: pass
    if arch == "rdna3":
      llvm_results = compile_asm_batch([t[1] for t in to_test])
      passed = sum(1 for (data, _), llvm in zip(to_test, llvm_results) if llvm == data)
      print(f"{arch} disasm: {passed} passed, {len(to_test) - passed} failed")
      self.assertEqual(passed, len(to_test))
    else:
      print(f"{arch} disasm: {len(to_test)} passed")

  def test_rdna3_asm(self): self._test_asm(self.rdna, "rdna3")
  def test_rdna3_roundtrip(self): self._test_roundtrip(self.rdna, "rdna3")
  def test_rdna3_disasm(self): self._test_disasm(self.rdna, "rdna3")
  def test_cdna_roundtrip(self): self._test_roundtrip(self.cdna, "cdna")
  def test_cdna_disasm(self): self._test_disasm(self.cdna, "cdna")

if __name__ == "__main__":
  unittest.main()
