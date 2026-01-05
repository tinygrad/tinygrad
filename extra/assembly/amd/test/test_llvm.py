#!/usr/bin/env python3
"""Test AMD assembler/disassembler against LLVM test vectors."""
import unittest, re, subprocess, functools
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
# CDNA test files - includes gfx9 files for shared instructions, plus gfx90a/gfx942 specific files
# gfx90a_ldst_acc.s has MIMG mixed in, filtered via is_mimg check
CDNA_FILES = ['gfx9_asm_sop1.s', 'gfx9_asm_sop2.s', 'gfx9_asm_sopp.s', 'gfx9_asm_sopk.s', 'gfx9_asm_sopc.s',
  'gfx9_asm_vop1.s', 'gfx9_asm_vop2.s', 'gfx9_asm_vopc.s', 'gfx9_asm_vop3.s', 'gfx9_asm_vop3p.s',
  'gfx9_asm_ds.s', 'gfx9_asm_flat.s', 'gfx9_asm_smem.s', 'gfx9_asm_mubuf.s', 'gfx9_asm_mtbuf.s',
  'gfx90a_ldst_acc.s', 'gfx90a_asm_features.s', 'flat-scratch-gfx942.s', 'gfx942_asm_features.s',
  'mai-gfx90a.s', 'mai-gfx942.s']

def _is_mimg(data: bytes) -> bool: return (int.from_bytes(data[:4], 'little') >> 26) & 0x3f == 0b111100

def _parse_llvm_tests(text: str, pattern: str) -> list[tuple[str, bytes]]:
  tests = []
  for block in text.split('\n\n'):
    asm_text, encoding = None, None
    for line in block.split('\n'):
      line = line.strip()
      if not line or line.startswith(('.', ';')): continue
      if not line.startswith('//'):
        asm_text = line.split('//')[0].strip() or asm_text
      if m := re.search(pattern + r'[^:]*:.*?(?:encoding:\s*)?\[(0x[0-9a-f,x\s]+)\]', line, re.I):
        encoding = m.group(1).replace('0x', '').replace(',', '').replace(' ', '')
    if asm_text and encoding:
      try: tests.append((asm_text, bytes.fromhex(encoding)))
      except ValueError: pass
  return tests

@functools.cache
def _get_tests(f: str, arch: str) -> list[tuple[str, bytes]]:
  text = fetch(f"{LLVM_BASE}/{f}").read_bytes().decode('utf-8', errors='ignore')
  if arch == "rdna3":
    tests = _parse_llvm_tests(text, r'(?:GFX11|W32|W64)')
  elif 'gfx90a' in f or 'gfx942' in f:
    tests = _parse_llvm_tests(text, r'(?:GFX90A|GFX942)')
  else:
    tests = _parse_llvm_tests(text, r'(?:VI9|GFX9|CHECK)')
  return [(a, d) for a, d in tests if not _is_mimg(d)] if arch == "cdna" else tests

def _compile_asm_batch(instrs: list[str]) -> list[bytes]:
  if not instrs: return []
  result = subprocess.run([get_llvm_mc(), '-triple=amdgcn', '-mcpu=gfx1100', '-mattr=+real-true16,+wavefrontsize32', '-show-encoding'],
    input=".text\n" + "\n".join(instrs) + "\n", capture_output=True, text=True, timeout=30)
  if result.returncode != 0: raise RuntimeError(f"llvm-mc failed: {result.stderr.strip()}")
  return [bytes.fromhex(line.split('encoding:')[1].strip()[1:-1].replace('0x', '').replace(',', '').replace(' ', ''))
          for line in result.stdout.split('\n') if 'encoding:' in line]

def _make_test(f: str, arch: str, test_type: str):
  def test(self):
    tests = _get_tests(f, arch)
    name = f"{arch}_{test_type}_{f}"
    if test_type == "roundtrip":
      for _, data in tests:
        decoded = detect_format(data, arch).from_bytes(data)
        self.assertEqual(decoded.to_bytes()[:len(data)], data)
      print(f"{name}: {len(tests)} passed")
    elif test_type == "asm":
      passed, failed, skipped = 0, 0, 0
      for asm_text, expected in tests:
        try:
          result = asm(asm_text, arch=arch)
          if result.to_bytes() == expected:
            passed += 1
          else:
            failed += 1
        except:
          skipped += 1
      print(f"{name}: {passed} passed, {failed} failed, {skipped} skipped")
    elif test_type == "disasm":
      to_test = []
      for _, data in tests:
        try:
          decoded = detect_format(data, arch).from_bytes(data)
          if decoded.to_bytes()[:len(data)] == data and (d := disasm(decoded)): to_test.append((data, d))
        except: pass
      print(f"{name}: {len(to_test)} passed, {len(tests) - len(to_test)} skipped")
      if arch == "rdna3":
        for (data, _), llvm in zip(to_test, _compile_asm_batch([t[1] for t in to_test])): self.assertEqual(llvm, data)
  return test

class TestLLVM(unittest.TestCase): pass

for f in RDNA_FILES:
  setattr(TestLLVM, f"test_rdna3_roundtrip_{f.replace('.s', '').replace('-', '_')}", _make_test(f, "rdna3", "roundtrip"))
  setattr(TestLLVM, f"test_rdna3_asm_{f.replace('.s', '').replace('-', '_')}", _make_test(f, "rdna3", "asm"))
  setattr(TestLLVM, f"test_rdna3_disasm_{f.replace('.s', '').replace('-', '_')}", _make_test(f, "rdna3", "disasm"))
for f in CDNA_FILES:
  setattr(TestLLVM, f"test_cdna_roundtrip_{f.replace('.s', '').replace('-', '_')}", _make_test(f, "cdna", "roundtrip"))
  setattr(TestLLVM, f"test_cdna_asm_{f.replace('.s', '').replace('-', '_')}", _make_test(f, "cdna", "asm"))
  setattr(TestLLVM, f"test_cdna_disasm_{f.replace('.s', '').replace('-', '_')}", _make_test(f, "cdna", "disasm"))

if __name__ == "__main__":
  unittest.main()
