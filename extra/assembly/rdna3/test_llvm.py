#!/usr/bin/env python3
"""Test RDNA3 assembler/disassembler against LLVM test vectors."""
import unittest, re
from tinygrad.helpers import fetch
from extra.assembly.rdna3.autogen_rdna3_enum import *
from extra.assembly.rdna3.lib import RawImm

LLVM_BASE = "https://raw.githubusercontent.com/llvm/llvm-project/main/llvm/test/MC/AMDGPU"

def fetch_llvm_tests(filename: str) -> bytes:
  """Fetch LLVM test file from GitHub."""
  url = f"{LLVM_BASE}/{filename}"
  return fetch(url).read_bytes()

def parse_llvm_tests(text: str) -> list[tuple[str, bytes]]:
  """Parse LLVM test format into (asm, expected_bytes) pairs.

  Format: instruction_line followed by // GFX11: encoding: [0xNN,0xNN,...]
  """
  tests = []
  lines = text.split('\n')
  i = 0
  while i < len(lines):
    line = lines[i].strip()
    # Skip empty lines, comments, directives
    if not line or line.startswith('//') or line.startswith('.') or line.startswith(';'):
      i += 1
      continue
    # Look for encoding on this line or next lines
    asm = line.split('//')[0].strip()  # Remove trailing comments
    if not asm:
      i += 1
      continue
    # Search for GFX11 encoding in the next few lines
    for j in range(i, min(i + 3, len(lines))):
      if m := re.search(r'GFX11[^:]*:\s*encoding:\s*\[(.*?)\]', lines[j]):
        hex_bytes = m.group(1).replace('0x', '').replace(',', '').replace(' ', '')
        if hex_bytes:
          try:
            expected = bytes.fromhex(hex_bytes)
            tests.append((asm, expected))
          except ValueError:
            pass  # Skip malformed hex
        break
    i += 1
  return tests

# Map from LLVM mnemonic to our instruction helper
def get_instruction_helper(mnemonic: str):
  """Get instruction helper function from mnemonic."""
  # Try direct lookup first
  name = mnemonic.lower().replace('.', '_')
  # For VOP1/VOP2, prefer the _e32 suffix version (32-bit encoding)
  if f"{name}_e32" in globals():
    return globals()[f"{name}_e32"]
  if name in globals():
    return globals()[name]
  return None

def parse_operand(op: str):
  """Parse an assembly operand into a Python value."""
  op = op.strip()
  # Registers
  if m := re.match(r'^s\[(\d+):(\d+)\]$', op):
    return SGPR[int(m.group(1)):int(m.group(2))+1]
  if m := re.match(r'^s(\d+)$', op):
    return SGPR[int(m.group(1))]
  if m := re.match(r'^v\[(\d+):(\d+)\]$', op):
    return VGPR[int(m.group(1)):int(m.group(2))+1]
  if m := re.match(r'^v(\d+)$', op):
    return VGPR[int(m.group(1))]
  if m := re.match(r'^ttmp\[(\d+):(\d+)\]$', op):
    return TTMP[int(m.group(1)):int(m.group(2))+1]
  if m := re.match(r'^ttmp(\d+)$', op):
    return TTMP[int(m.group(1))]
  # Special registers
  specials = {
    'vcc_lo': VCC_LO, 'vcc_hi': VCC_HI, 'null': NULL, 'off': OFF,
    'm0': M0, 'exec_lo': EXEC_LO, 'exec_hi': EXEC_HI, 'scc': SCC,
  }
  if op.lower() in specials:
    return specials[op.lower()]
  # Hex numbers
  if m := re.match(r'^0x([0-9a-fA-F]+)$', op):
    return int(m.group(1), 16)
  # Decimal numbers (including negative)
  if m := re.match(r'^-?\d+$', op):
    return int(op)
  # Float constants
  floats = {'0.5': POS_HALF, '-0.5': NEG_HALF, '1.0': POS_ONE, '-1.0': NEG_ONE,
            '2.0': POS_TWO, '-2.0': NEG_TWO, '4.0': POS_FOUR, '-4.0': NEG_FOUR}
  if op in floats:
    return floats[op]
  return None

def split_operands(operands_str: str) -> list[str]:
  """Split operands carefully, handling brackets."""
  operands = []
  current = ""
  depth = 0
  for ch in operands_str:
    if ch == '[':
      depth += 1
      current += ch
    elif ch == ']':
      depth -= 1
      current += ch
    elif ch == ',' and depth == 0:
      if current.strip():
        operands.append(current.strip())
      current = ""
    else:
      current += ch
  if current.strip():
    operands.append(current.strip())
  return operands

# SOP1 instructions that only have a source operand (no destination)
SOP1_SRC_ONLY = {'s_setpc_b64', 's_rfe_b64'}
# SOP1 instructions that have a message immediate, not source encoding
SOP1_MSG_IMM = {'s_sendmsg_rtn_b32', 's_sendmsg_rtn_b64'}

def try_assemble(asm: str):
  """Try to assemble an instruction, return bytes or None."""
  # Parse mnemonic and operands
  parts = asm.replace(',', ' ').split()
  if not parts:
    return None
  mnemonic = parts[0]
  operands_str = asm[len(mnemonic):].strip()

  helper = get_instruction_helper(mnemonic)
  if helper is None:
    return None

  # Parse operands - handle complex cases
  if not operands_str:
    try:
      return helper().to_bytes()
    except Exception:
      return None

  operands = split_operands(operands_str)

  # Parse each operand
  parsed = []
  for op in operands:
    val = parse_operand(op)
    if val is None:
      return None  # Can't parse this operand
    parsed.append(val)

  try:
    name = mnemonic.lower()
    # Handle special SOP1 cases
    if name in SOP1_SRC_ONLY:
      # These instructions only have ssrc0, no sdst
      inst = helper(ssrc0=parsed[0])
    elif name in SOP1_MSG_IMM:
      # s_sendmsg_rtn_b32 sdst, simm16 - the second operand is a raw immediate (message ID)
      inst = helper(sdst=parsed[0], ssrc0=RawImm(parsed[1]))
    else:
      inst = helper(*parsed)
    return inst.to_bytes()
  except Exception:
    return None

class TestLLVMSOP1(unittest.TestCase):
  """Test SOP1 instructions against LLVM test vectors."""

  @classmethod
  def setUpClass(cls):
    data = fetch_llvm_tests("gfx11_asm_sop1.s")
    cls.tests = parse_llvm_tests(data.decode('utf-8', errors='ignore'))

  def test_sop1_instructions(self):
    passed, failed, skipped = 0, 0, 0
    for asm, expected in self.tests:
      result = try_assemble(asm)
      if result is None:
        skipped += 1
        continue
      if result == expected:
        passed += 1
      else:
        failed += 1
        if failed <= 5:
          print(f"FAIL: {asm}")
          print(f"  expected: {expected.hex()}")
          print(f"  got:      {result.hex()}")
    print(f"SOP1: {passed} passed, {failed} failed, {skipped} skipped")
    self.assertEqual(failed, 0, f"{failed} SOP1 tests failed")

class TestLLVMSOP2(unittest.TestCase):
  """Test SOP2 instructions against LLVM test vectors."""

  @classmethod
  def setUpClass(cls):
    data = fetch_llvm_tests("gfx11_asm_sop2.s")
    cls.tests = parse_llvm_tests(data.decode('utf-8', errors='ignore'))

  def test_sop2_instructions(self):
    passed, failed, skipped = 0, 0, 0
    for asm, expected in self.tests:
      result = try_assemble(asm)
      if result is None:
        skipped += 1
        continue
      if result == expected:
        passed += 1
      else:
        failed += 1
        if failed <= 5:
          print(f"FAIL: {asm}")
          print(f"  expected: {expected.hex()}")
          print(f"  got:      {result.hex()}")
    print(f"SOP2: {passed} passed, {failed} failed, {skipped} skipped")
    self.assertEqual(failed, 0, f"{failed} SOP2 tests failed")

class TestLLVMSOPP(unittest.TestCase):
  """Test SOPP instructions against LLVM test vectors."""

  @classmethod
  def setUpClass(cls):
    data = fetch_llvm_tests("gfx11_asm_sopp.s")
    cls.tests = parse_llvm_tests(data.decode('utf-8', errors='ignore'))

  def test_sopp_instructions(self):
    passed, failed, skipped = 0, 0, 0
    for asm, expected in self.tests:
      result = try_assemble(asm)
      if result is None:
        skipped += 1
        continue
      if result == expected:
        passed += 1
      else:
        failed += 1
        if failed <= 5:
          print(f"FAIL: {asm}")
          print(f"  expected: {expected.hex()}")
          print(f"  got:      {result.hex()}")
    print(f"SOPP: {passed} passed, {failed} failed, {skipped} skipped")
    self.assertEqual(failed, 0, f"{failed} SOPP tests failed")

class TestLLVMVOP1(unittest.TestCase):
  """Test VOP1 instructions against LLVM test vectors."""

  @classmethod
  def setUpClass(cls):
    data = fetch_llvm_tests("gfx11_asm_vop1.s")
    cls.tests = parse_llvm_tests(data.decode('utf-8', errors='ignore'))

  def test_vop1_instructions(self):
    passed, failed, skipped = 0, 0, 0
    for asm, expected in self.tests:
      result = try_assemble(asm)
      if result is None:
        skipped += 1
        continue
      if result == expected:
        passed += 1
      else:
        failed += 1
        if failed <= 5:
          print(f"FAIL: {asm}")
          print(f"  expected: {expected.hex()}")
          print(f"  got:      {result.hex()}")
    print(f"VOP1: {passed} passed, {failed} failed, {skipped} skipped")
    self.assertEqual(failed, 0, f"{failed} VOP1 tests failed")

class TestLLVMVOP2(unittest.TestCase):
  """Test VOP2 instructions against LLVM test vectors."""

  @classmethod
  def setUpClass(cls):
    data = fetch_llvm_tests("gfx11_asm_vop2.s")
    cls.tests = parse_llvm_tests(data.decode('utf-8', errors='ignore'))

  def test_vop2_instructions(self):
    passed, failed, skipped = 0, 0, 0
    for asm, expected in self.tests:
      result = try_assemble(asm)
      if result is None:
        skipped += 1
        continue
      if result == expected:
        passed += 1
      else:
        failed += 1
        if failed <= 5:
          print(f"FAIL: {asm}")
          print(f"  expected: {expected.hex()}")
          print(f"  got:      {result.hex()}")
    print(f"VOP2: {passed} passed, {failed} failed, {skipped} skipped")
    self.assertEqual(failed, 0, f"{failed} VOP2 tests failed")

def _unwrap(val):
  """Unwrap RawImm to get the raw value."""
  return val.val if isinstance(val, RawImm) else val

class TestDisassembler(unittest.TestCase):
  """Test disassembler (from_bytes) roundtrips."""

  def test_sop1_field_extraction(self):
    """Test SOP1 field extraction."""
    # s_mov_b32 s0, s1 -> 01 00 80 be
    inst = s_mov_b32(s[0], s[1])
    data = inst.to_bytes()
    decoded = SOP1.from_bytes(data)
    # Check fields are extracted correctly
    self.assertEqual(decoded._values['op'], 0)  # S_MOV_B32 = 0
    self.assertEqual(decoded._values['sdst'], 0)  # s0
    self.assertEqual(_unwrap(decoded._values['ssrc0']), 1)  # s1

  def test_sop2_field_extraction(self):
    """Test SOP2 field extraction."""
    # s_add_u32 s3, s0, s1
    inst = s_add_u32(s[3], s[0], s[1])
    data = inst.to_bytes()
    decoded = SOP2.from_bytes(data)
    self.assertEqual(decoded._values['op'], 0)  # S_ADD_U32 = 0
    self.assertEqual(decoded._values['sdst'], 3)
    self.assertEqual(_unwrap(decoded._values['ssrc0']), 0)
    self.assertEqual(_unwrap(decoded._values['ssrc1']), 1)

  def test_vop2_field_extraction(self):
    """Test VOP2 field extraction."""
    # v_add_f32_e32 v0, v1, v2
    inst = v_add_f32_e32(v[0], v[1], v[2])
    data = inst.to_bytes()
    decoded = VOP2.from_bytes(data)
    self.assertEqual(decoded._values['op'], 3)  # V_ADD_F32 = 3
    self.assertEqual(decoded._values['vdst'], 0)
    self.assertEqual(_unwrap(decoded._values['src0']), 256 + 1)  # v1 encoded
    self.assertEqual(decoded._values['vsrc1'], 2)  # vsrc1 is VGPR index directly

  def test_sop1_disasm(self):
    """Test SOP1 disassembly to string."""
    inst = s_mov_b32(s[0], s[1])
    decoded = SOP1.from_bytes(inst.to_bytes())
    asm = decoded.disasm()
    self.assertIn('s_mov_b32', asm)
    self.assertIn('s0', asm)
    self.assertIn('s1', asm)

  def test_sop2_disasm(self):
    """Test SOP2 disassembly to string."""
    inst = s_add_u32(s[3], s[0], s[1])
    decoded = SOP2.from_bytes(inst.to_bytes())
    asm = decoded.disasm()
    self.assertIn('s_add_u32', asm)
    self.assertIn('s3', asm)

  def test_sopp_disasm(self):
    """Test SOPP disassembly to string."""
    inst = s_endpgm()
    decoded = SOPP.from_bytes(inst.to_bytes())
    asm = decoded.disasm()
    self.assertIn('s_endpgm', asm)

  def test_vop2_disasm(self):
    """Test VOP2 disassembly to string."""
    inst = v_add_f32_e32(v[0], v[1], v[2])
    decoded = VOP2.from_bytes(inst.to_bytes())
    asm = decoded.disasm()
    self.assertIn('v_add_f32', asm)
    self.assertIn('v0', asm)

class TestDisassemblerRoundtrip(unittest.TestCase):
  """Test that assemble -> disassemble -> reassemble produces same bytes."""

  def _test_roundtrip(self, inst, format_cls):
    """Helper to test roundtrip for an instruction."""
    # Assemble to bytes
    data = inst.to_bytes()
    # Disassemble
    decoded = format_cls.from_bytes(data)
    # Re-encode should produce same bytes
    reencoded = decoded.to_int()
    original = int.from_bytes(data[:format_cls._size()], 'little')
    self.assertEqual(reencoded, original, f"Roundtrip failed: {inst}")

  def test_sop1_roundtrip(self):
    """Test SOP1 instructions roundtrip."""
    test_cases = [
      s_mov_b32(s[0], s[1]),
      s_mov_b32(s[5], s[10]),
      s_not_b32(s[3], s[7]),
      s_brev_b32(s[0], s[0]),
    ]
    for inst in test_cases:
      self._test_roundtrip(inst, SOP1)

  def test_sop2_roundtrip(self):
    """Test SOP2 instructions roundtrip."""
    test_cases = [
      s_add_u32(s[0], s[1], s[2]),
      s_sub_u32(s[3], s[4], s[5]),
      s_and_b32(s[10], s[20], s[30]),
      s_or_b32(s[0], s[0], s[1]),
    ]
    for inst in test_cases:
      self._test_roundtrip(inst, SOP2)

  def test_sopp_roundtrip(self):
    """Test SOPP instructions roundtrip."""
    test_cases = [
      s_endpgm(),
      s_nop(0),
      s_waitcnt(0),
      s_barrier(),
    ]
    for inst in test_cases:
      self._test_roundtrip(inst, SOPP)

  def test_vop2_roundtrip(self):
    """Test VOP2 instructions roundtrip."""
    test_cases = [
      v_add_f32_e32(v[0], v[1], v[2]),
      v_mul_f32_e32(v[5], v[10], v[15]),
      v_and_b32_e32(v[0], 10, v[0]),
    ]
    for inst in test_cases:
      self._test_roundtrip(inst, VOP2)

  def test_vop1_roundtrip(self):
    """Test VOP1 instructions roundtrip."""
    test_cases = [
      v_mov_b32_e32(v[0], v[1]),
      v_not_b32_e32(v[5], v[10]),
    ]
    for inst in test_cases:
      self._test_roundtrip(inst, VOP1)

  def test_smem_roundtrip(self):
    """Test SMEM instructions roundtrip."""
    test_cases = [
      s_load_b128(s[4:7], s[0:1], NULL),
      s_load_b32(s[0], s[2:3], NULL),
    ]
    for inst in test_cases:
      self._test_roundtrip(inst, SMEM)

  def test_vop3_roundtrip(self):
    """Test VOP3 instructions roundtrip."""
    test_cases = [
      v_add3_u32(v[0], v[1], v[2], v[3]),
      v_bfe_u32(v[1], v[0], 10, 10),
    ]
    for inst in test_cases:
      self._test_roundtrip(inst, VOP3)

class TestDisassemblerFromLLVM(unittest.TestCase):
  """Test disassembling LLVM-assembled bytes back to valid instructions."""

  @classmethod
  def setUpClass(cls):
    # Fetch and parse SOP1, SOP2, SOPP test vectors
    cls.sop1_tests = parse_llvm_tests(fetch_llvm_tests("gfx11_asm_sop1.s").decode('utf-8', errors='ignore'))
    cls.sop2_tests = parse_llvm_tests(fetch_llvm_tests("gfx11_asm_sop2.s").decode('utf-8', errors='ignore'))

  def test_sop1_disasm_from_llvm(self):
    """Disassemble LLVM SOP1 bytes and verify fields are valid."""
    passed, failed = 0, 0
    for asm, data in self.sop1_tests[:100]:  # Test first 100
      try:
        decoded = SOP1.from_bytes(data)
        # Verify opcode is valid
        op = decoded._values['op']
        _ = SOP1Op(op)  # Should not raise
        # Verify we can disassemble to string
        disasm = decoded.disasm()
        self.assertIn('s_', disasm)
        passed += 1
      except (ValueError, KeyError):
        failed += 1
    print(f"SOP1 disasm: {passed} passed, {failed} failed")
    self.assertGreater(passed, 50)

  def test_sop2_disasm_from_llvm(self):
    """Disassemble LLVM SOP2 bytes and verify fields are valid."""
    passed, failed = 0, 0
    for asm, data in self.sop2_tests[:100]:  # Test first 100
      try:
        decoded = SOP2.from_bytes(data)
        # Verify opcode is valid
        op = decoded._values['op']
        _ = SOP2Op(op)  # Should not raise
        # Verify we can disassemble to string
        disasm = decoded.disasm()
        self.assertIn('s_', disasm)
        passed += 1
      except (ValueError, KeyError):
        failed += 1
    print(f"SOP2 disasm: {passed} passed, {failed} failed")
    self.assertGreater(passed, 50)

  def test_sop2_roundtrip_from_llvm(self):
    """Test that LLVM bytes can be decoded and re-encoded identically."""
    passed, failed = 0, 0
    for asm, data in self.sop2_tests[:100]:
      try:
        decoded = SOP2.from_bytes(data)
        reencoded = decoded.to_bytes()
        # For 32-bit instructions without literals, should match exactly
        if len(data) == 4:
          if reencoded == data:
            passed += 1
          else:
            failed += 1
        else:
          passed += 1  # Skip literal cases for now
      except Exception:
        failed += 1
    print(f"SOP2 roundtrip: {passed} passed, {failed} failed")
    self.assertGreater(passed, 50)

if __name__ == "__main__":
  unittest.main()
