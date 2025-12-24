#!/usr/bin/env python3
"""Test RDNA3 SOP1 instructions against LLVM test vectors."""
import unittest, re
from extra.assembly.rdna3.autogen_rdna3_enum import *

# Parse LLVM test format: "instruction\n// GFX11: encoding: [bytes]"
LLVM_TESTS = """
s_mov_b32 s0, s1
// GFX11: encoding: [0x01,0x00,0x80,0xbe]

s_mov_b32 s105, s104
// GFX11: encoding: [0x68,0x00,0xe9,0xbe]

s_mov_b32 s0, 0
// GFX11: encoding: [0x80,0x00,0x80,0xbe]

s_mov_b32 s0, -1
// GFX11: encoding: [0xc1,0x00,0x80,0xbe]

s_mov_b32 s0, null
// GFX11: encoding: [0x7c,0x00,0x80,0xbe]

s_not_b32 s0, s1
// GFX11: encoding: [0x01,0x1e,0x80,0xbe]

s_not_b32 s105, s104
// GFX11: encoding: [0x68,0x1e,0xe9,0xbe]

s_brev_b32 s0, s1
// GFX11: encoding: [0x01,0x04,0x80,0xbe]

s_abs_i32 s0, s1
// GFX11: encoding: [0x01,0x15,0x80,0xbe]
"""

def parse_llvm_tests(text):
  """Parse LLVM test format into (asm, expected_bytes) pairs."""
  tests = []
  lines = text.strip().split('\n')
  i = 0
  while i < len(lines):
    line = lines[i].strip()
    if line and not line.startswith('//'):
      # This is an instruction
      asm = line
      # Next line should have encoding
      if i + 1 < len(lines):
        enc_line = lines[i + 1]
        if m := re.search(r'encoding: \[(.*?)\]', enc_line):
          hex_bytes = m.group(1).replace('0x', '').replace(',', '')
          expected = bytes.fromhex(hex_bytes)
          tests.append((asm, expected))
    i += 1
  return tests

class TestSOP1(unittest.TestCase):
  def test_s_mov_b32_reg_reg(self):
    """s_mov_b32 s0, s1"""
    inst = s_mov_b32(s[0], s[1])
    expected = bytes([0x01, 0x00, 0x80, 0xbe])
    self.assertEqual(inst.to_bytes(), expected)

  def test_s_mov_b32_high_regs(self):
    """s_mov_b32 s105, s104"""
    inst = s_mov_b32(s[105], s[104])
    expected = bytes([0x68, 0x00, 0xe9, 0xbe])
    self.assertEqual(inst.to_bytes(), expected)

  def test_s_mov_b32_inline_zero(self):
    """s_mov_b32 s0, 0"""
    inst = s_mov_b32(s[0], 0)
    expected = bytes([0x80, 0x00, 0x80, 0xbe])
    self.assertEqual(inst.to_bytes(), expected)

  def test_s_mov_b32_inline_neg1(self):
    """s_mov_b32 s0, -1"""
    inst = s_mov_b32(s[0], -1)
    expected = bytes([0xc1, 0x00, 0x80, 0xbe])
    self.assertEqual(inst.to_bytes(), expected)

  def test_s_mov_b32_null(self):
    """s_mov_b32 s0, null"""
    inst = s_mov_b32(s[0], NULL)
    expected = bytes([0x7c, 0x00, 0x80, 0xbe])
    self.assertEqual(inst.to_bytes(), expected)

  def test_s_not_b32(self):
    """s_not_b32 s0, s1"""
    inst = s_not_b32(s[0], s[1])
    expected = bytes([0x01, 0x1e, 0x80, 0xbe])
    self.assertEqual(inst.to_bytes(), expected)

  def test_s_brev_b32(self):
    """s_brev_b32 s0, s1"""
    inst = s_brev_b32(s[0], s[1])
    expected = bytes([0x01, 0x04, 0x80, 0xbe])
    self.assertEqual(inst.to_bytes(), expected)

  def test_s_abs_i32(self):
    """s_abs_i32 s0, s1"""
    inst = s_abs_i32(s[0], s[1])
    expected = bytes([0x01, 0x15, 0x80, 0xbe])
    self.assertEqual(inst.to_bytes(), expected)

if __name__ == "__main__":
  unittest.main()
