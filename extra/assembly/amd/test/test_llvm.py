#!/usr/bin/env python3
"""Test RDNA3 assembler/disassembler against LLVM test vectors."""
import unittest, re, subprocess
from tinygrad.helpers import fetch
from extra.assembly.amd.autogen.rdna3.ins import *
from extra.assembly.amd.asm import asm
from extra.assembly.amd.test.helpers import get_llvm_mc

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

# Undocumented opcodes that we skip (reserved/internal to AMD)
UNDOCUMENTED: dict = {}

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
      # Format 1: "// GFX11: v_foo ... ; encoding: [0x01,0x02,...]"
      # Format 2: "// GFX11: [0x01,0x02,...]" (used by DS, older files)
      if m := re.search(r'(?:GFX11|W32|W64)[^:]*:.*?encoding:\s*\[(.*?)\]', lines[j]):
        hex_bytes = m.group(1).replace('0x', '').replace(',', '').replace(' ', '')
      elif m := re.search(r'(?:GFX11|W32|W64)[^:]*:\s*\[(0x[0-9a-fA-F,x\s]+)\]', lines[j]):
        hex_bytes = m.group(1).replace('0x', '').replace(',', '').replace(' ', '')
      else:
        continue
      if hex_bytes:
        try: tests.append((asm_text, bytes.fromhex(hex_bytes)))
        except ValueError: pass
      break
  return tests

def _get_op(inst) -> int:
  """Extract opcode value from instruction."""
  op = inst._values.get('op', 0)
  return op.val if hasattr(op, 'val') else op

def compile_asm_batch(instrs: list[str]) -> list[bytes]:
  """Compile multiple instructions with a single llvm-mc call."""
  if not instrs: return []
  asm_text = ".text\n" + "\n".join(instrs) + "\n"
  result = subprocess.run(
    [get_llvm_mc(), '-triple=amdgcn', '-mcpu=gfx1100', '-mattr=+real-true16,+wavefrontsize32', '-show-encoding'],
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

class TestLLVM(unittest.TestCase):
  """Test instruction format decode/encode roundtrip and disassembly against LLVM test vectors."""
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

def _has_literal(data: bytes, fmt_cls) -> bool:
  """Check if instruction has a trailing 32-bit literal constant."""
  if len(data) <= fmt_cls._size(): return False
  # Check for literal marker (src0=0xff or ssrc0/ssrc1=0xff)
  fn = fmt_cls.__name__
  if fn in ('SOP1', 'VOP1', 'VOPC'): return data[0] == 0xff  # ssrc0 / src0
  if fn in ('SOP2', 'SOPC'): return data[0] == 0xff or data[1] == 0xff  # ssrc0 or ssrc1
  if fn == 'VOP2':
    op = (int.from_bytes(data[:4], 'little') >> 25) & 0x3f
    return data[0] == 0xff or op in (44, 45, 55, 56)  # src0=literal or FMAMK/FMAAK F32/F16 (RDNA3 opcodes)
  if fn == 'SOPK': return ((int.from_bytes(data[:4], 'little') >> 23) & 0x1f) == 19  # S_SETREG_IMM32_B32 (RDNA3 opcode)
  if fn == 'VOPD':
    # Check srcx0/srcy0 for literal, or opx/opy for FMAAK/FMAMK (1, 2)
    word0 = int.from_bytes(data[:4], 'little')
    opx, opy = (word0 >> 22) & 0xf, (word0 >> 17) & 0x1f
    return data[0] == 0xff or data[4] == 0xff or opx in (1, 2) or opy in (1, 2)
  # VOP3/VOP3SD/VOP3P: check src0/src1/src2 fields for literal marker (0xff)
  if fn in ('VOP3', 'VOP3SD', 'VOP3P'):
    word1 = int.from_bytes(data[4:8], 'little')
    src0, src1, src2 = word1 & 0x1ff, (word1 >> 9) & 0x1ff, (word1 >> 18) & 0x1ff
    return src0 == 0xff or src1 == 0xff or src2 == 0xff
  return False

def _make_roundtrip_test(name):
  """Test decode → encode roundtrip for all instructions."""
  def test(self):
    _, fmt_cls, op_enum = LLVM_TEST_FILES[name]
    vop3sd_opcodes = {288, 289, 290, 764, 765, 766, 767, 768, 769, 770}
    is_vopc_promotion = name in ('vop3_from_vopc', 'vop3_from_vopcx')
    passed, failed, skipped, enum_missing = 0, 0, 0, 0
    failures: list[str] = []

    for asm_text, data in self.tests.get(name, []):
      # Handle literal constants: formats can have trailing 32-bit literal
      has_lit = _has_literal(data, fmt_cls)
      expected_size = fmt_cls._size() + (4 if has_lit else 0)
      if len(data) > expected_size: skipped += 1; continue
      try:
        # Decode instruction
        if fmt_cls.__name__ in ('VOP3', 'VOP3SD'):
          temp = VOP3.from_bytes(data)
          is_vop3sd = (_get_op(temp) in vop3sd_opcodes) and not is_vopc_promotion
          decoded = VOP3SD.from_bytes(data) if is_vop3sd else VOP3.from_bytes(data)
          try: (VOP3SDOp if is_vop3sd else VOP3Op)(_get_op(decoded))
          except ValueError: enum_missing += 1; continue
        else:
          decoded = fmt_cls.from_bytes(data)
          op_val = _get_op(decoded)
          if op_val in UNDOCUMENTED.get(name, set()): skipped += 1; continue
          try: op_enum(op_val)
          except ValueError: enum_missing += 1; continue
        # Check roundtrip
        reencoded = decoded.to_bytes()[:len(data)]
        if reencoded == data: passed += 1
        else: failed += 1; failures.append(f"'{asm_text}': orig={data.hex()} reenc={reencoded.hex()}")
      except Exception as e:
        failed += 1; failures.append(f"'{asm_text}': {e}")

    print(f"{name.upper()} roundtrip: {passed} passed, {failed} failed, {skipped} skipped, {enum_missing} missing")
    if failures[:5]: print("  " + "\n  ".join(failures[:5]))
    self.assertEqual(failed, 0)
  return test

def _make_disasm_test(name):
  """Test decode → disasm → re-asm → compare for all instructions."""
  def test(self):
    _, fmt_cls, op_enum = LLVM_TEST_FILES[name]
    vop3sd_opcodes = {288, 289, 290, 764, 765, 766, 767, 768, 769, 770}
    is_vopc_promotion = name in ('vop3_from_vopc', 'vop3_from_vopcx')
    to_test: list[tuple[str, bytes, str | None, str | None]] = []
    skipped = 0

    for asm_text, data in self.tests.get(name, []):
      # Handle literal constants
      has_lit = _has_literal(data, fmt_cls)
      expected_size = fmt_cls._size() + (4 if has_lit else 0)
      if len(data) > expected_size: continue
      try:
        # Decode instruction
        if fmt_cls.__name__ in ('VOP3', 'VOP3SD'):
          temp = VOP3.from_bytes(data)
          is_vop3sd = (_get_op(temp) in vop3sd_opcodes) and not is_vopc_promotion
          decoded = VOP3SD.from_bytes(data) if is_vop3sd else VOP3.from_bytes(data)
          try: (VOP3SDOp if is_vop3sd else VOP3Op)(_get_op(decoded))
          except ValueError: skipped += 1; continue
        else:
          decoded = fmt_cls.from_bytes(data)
          op_val = _get_op(decoded)
          if op_val in UNDOCUMENTED.get(name, set()): skipped += 1; continue
          try: op_enum(op_val)
          except ValueError: skipped += 1; continue
        # Check roundtrip first
        if decoded.to_bytes()[:len(data)] != data:
          to_test.append((asm_text, data, None, "roundtrip failed")); continue
        to_test.append((asm_text, data, decoded.disasm(), None))
      except Exception as e:
        to_test.append((asm_text, data, None, f"exception: {e}"))

    # Batch compile all disasm strings
    disasm_strs = [(i, t[2]) for i, t in enumerate(to_test) if t[2] is not None]
    llvm_results = compile_asm_batch([s for _, s in disasm_strs]) if disasm_strs else []
    llvm_map = {i: llvm_results[j] for j, (i, _) in enumerate(disasm_strs)}

    passed, failed, failures = 0, 0, []
    for idx, (asm_text, data, disasm_str, error) in enumerate(to_test):
      if error: failed += 1; failures.append(f"{error} for {data.hex()}")
      elif disasm_str is not None and idx in llvm_map:
        if llvm_map[idx] == data: passed += 1
        else: failed += 1; failures.append(f"'{disasm_str}': expected={data.hex()} got={llvm_map[idx].hex()}")

    print(f"{name.upper()} disasm: {passed} passed, {failed} failed" + (f", {skipped} skipped" if skipped else ""))
    if failures[:10]: print("  " + "\n  ".join(failures[:10]))
    self.assertEqual(failed, 0)
  return test

for name in LLVM_TEST_FILES:
  setattr(TestLLVM, f'test_{name}_roundtrip', _make_roundtrip_test(name))
  setattr(TestLLVM, f'test_{name}_disasm', _make_disasm_test(name))

if __name__ == "__main__":
  unittest.main()
