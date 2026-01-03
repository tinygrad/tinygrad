#!/usr/bin/env python3
"""Test CDNA assembler/disassembler against LLVM test vectors.

NOTE: CDNA (gfx908/gfx90a/gfx940/gfx942) instruction encodings differ from GFX9 (gfx900/Vega)
in several ways, particularly for memory instructions. We use CDNA-specific test files (gfx90a/gfx942)
for memory instructions to ensure correct bit-level compatibility.
"""
import unittest, re, subprocess
from tinygrad.helpers import fetch
from extra.assembly.amd.autogen.cdna.ins import *
from extra.assembly.amd.asm import disasm
from extra.assembly.amd.test.helpers import get_llvm_mc

LLVM_BASE = "https://raw.githubusercontent.com/llvm/llvm-project/main/llvm/test/MC/AMDGPU"

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
}

def parse_llvm_tests(text: str, mnemonic_filter: str = None, size_filter: int = None) -> list[tuple[str, bytes]]:
  """Parse LLVM test format into (asm, expected_bytes) pairs.
  
  Args:
    text: LLVM test file content
    mnemonic_filter: Optional prefix filter (e.g. 'flat_', 'buffer_', 'global_', 'scratch_')
    size_filter: Optional instruction size filter (4 for 32-bit, 8 for 64-bit)
  """
  tests, lines = [], text.split('\n')
  for i, line in enumerate(lines):
    line = line.strip()
    if not line or line.startswith(('//', '.', ';')): continue
    asm_text = line.split('//')[0].strip()
    if not asm_text: continue
    # Apply mnemonic filter if specified
    if mnemonic_filter and not asm_text.startswith(mnemonic_filter): continue
    for j in range(i, min(i + 3, len(lines))):
      # Match various CHECK formats
      if m := re.search(r'CHECK[^:]*:\s*\[(0x[0-9a-fA-F,x\s]+)\]', lines[j]):
        hex_bytes = m.group(1).replace('0x', '').replace(',', '').replace(' ', '')
      elif m := re.search(r'GFX\d+[^:]*:.*?encoding:\s*\[(.*?)\]', lines[j]):
        hex_bytes = m.group(1).replace('0x', '').replace(',', '').replace(' ', '')
      elif m := re.search(r'GFX\d+[^:]*:\s*\[(0x[0-9a-fA-F,x\s]+)\]', lines[j]):
        hex_bytes = m.group(1).replace('0x', '').replace(',', '').replace(' ', '')
      else:
        continue
      if hex_bytes:
        try:
          data = bytes.fromhex(hex_bytes)
          # Apply size filter if specified
          if size_filter and len(data) != size_filter: continue
          tests.append((asm_text, data))
        except ValueError: pass
      break
  return tests

def compile_asm_batch(instrs: list[str], mcpu: str = 'gfx940') -> list[bytes]:
  """Compile multiple instructions with a single llvm-mc call."""
  if not instrs: return []
  asm_text = ".text\n" + "\n".join(instrs) + "\n"
  result = subprocess.run(
    [get_llvm_mc(), '-triple=amdgcn', f'-mcpu={mcpu}', '-show-encoding'],
    input=asm_text, capture_output=True, text=True, timeout=30)
  if result.returncode != 0: raise RuntimeError(f"llvm-mc batch failed: {result.stderr.strip()}")
  results = []
  for line in result.stdout.split('\n'):
    if 'encoding:' not in line: continue
    enc = line.split('encoding:')[1].strip()
    if enc.startswith('[') and enc.endswith(']'):
      results.append(bytes.fromhex(enc[1:-1].replace('0x', '').replace(',', '').replace(' ', '')))
  if len(results) != len(instrs): raise RuntimeError(f"expected {len(instrs)} encodings, got {len(results)}")
  return results

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

def _make_roundtrip_test(name):
  def test(self):
    _, fmt_cls, op_enum, mcpu, _, _ = CDNA_TEST_FILES[name]
    passed, failed, skipped, enum_missing = 0, 0, 0, 0
    failures: list[str] = []

    for asm_text, data in self.tests.get(name, []):
      if len(data) > fmt_cls._size():
        skipped += 1
        continue
      try:
        decoded = fmt_cls.from_bytes(data)
        op_val = decoded._values.get('op', 0)
        op_val = op_val.val if hasattr(op_val, 'val') else op_val
        try:
          op_enum(op_val)
        except ValueError:
          enum_missing += 1
          continue  # Skip opcodes not in our enum
        reencoded = decoded.to_bytes()[:len(data)]
        if reencoded == data:
          passed += 1
        else:
          failed += 1
          failures.append(f"'{asm_text}': orig={data.hex()} reenc={reencoded.hex()}")
      except Exception as e:
        failed += 1
        failures.append(f"'{asm_text}': {e}")

    print(f"CDNA {name.upper()} roundtrip: {passed} passed, {failed} failed, {skipped} skipped, {enum_missing} missing")
    if failures[:5]: print("  " + "\n  ".join(failures[:5]))
    self.assertEqual(failed, 0)
  return test

def _make_disasm_test(name):
  def test(self):
    _, fmt_cls, op_enum, mcpu, _, _ = CDNA_TEST_FILES[name]
    passed, failed, skipped, enum_missing = 0, 0, 0, 0
    failures: list[str] = []

    for asm_text, data in self.tests.get(name, []):
      if len(data) > fmt_cls._size():
        skipped += 1
        continue
      try:
        decoded = fmt_cls.from_bytes(data)
        op_val = decoded._values.get('op', 0)
        op_val = op_val.val if hasattr(op_val, 'val') else op_val
        try:
          op_enum(op_val)
        except ValueError:
          enum_missing += 1
          continue  # Skip opcodes not in our enum
        # Check roundtrip first
        reencoded = decoded.to_bytes()[:len(data)]
        if reencoded != data:
          failed += 1
          failures.append(f"'{asm_text}': roundtrip failed orig={data.hex()} reenc={reencoded.hex()}")
          continue
        # Try to disassemble - just check it doesn't crash and produces valid output
        disasm_text = disasm(decoded)
        if not disasm_text or not disasm_text.strip():
          failed += 1
          failures.append(f"'{asm_text}': empty disassembly")
          continue
        passed += 1
      except Exception as e:
        failed += 1
        failures.append(f"'{asm_text}': {e}")

    print(f"CDNA {name.upper()} disasm: {passed} passed, {failed} failed, {skipped} skipped, {enum_missing} missing")
    if failures[:5]: print("  " + "\n  ".join(failures[:5]))
    self.assertEqual(failed, 0)
  return test

for name in CDNA_TEST_FILES:
  setattr(TestLLVMCDNA, f'test_{name}_roundtrip', _make_roundtrip_test(name))
  setattr(TestLLVMCDNA, f'test_{name}_disasm', _make_disasm_test(name))

if __name__ == "__main__":
  unittest.main()
