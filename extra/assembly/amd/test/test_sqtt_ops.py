#!/usr/bin/env python3
"""Tests validating SQTT packet definitions against the reference implementation.

Verifies that:
1. Encoding patterns produce the correct STATE_TO_OPCODE table
2. Packet sizes (derived from fields) match expected budget values
3. Field extractions match attempt_sqtt_parse.py
"""
import unittest
from extra.assembly.amd.sqtt import (
  VALUINST, VMEMEXEC, ALUEXEC, IMMEDIATE, IMMEDIATE_MASK, WAVERDY,
  WAVEEND, WAVESTART, PERF, TS_WAVE_STATE, EVENT, EVENT_BIG, REG, SNAPSHOT,
  TS_DELTA_OR_MARK, LAYOUT_HEADER, INST, UTILCTR, TS_DELTA_SHORT, NOP,
  TS_DELTA_S8_W3, TS_DELTA_S5_W2, TS_DELTA_S5_W3, WAVEALLOC,
  decode, encode, OPCODE_TO_CLASS, STATE_TO_OPCODE, PACKET_TYPES, BUDGET,
  AluSrc, MemSrc, InstOp
)

# Reference table from rocprof trace decoder (attempt_sqtt_parse.py)
REFERENCE_STATE_TABLE = bytes([
  0x10, 0x16, 0x18, 0x01, 0x05, 0x0b, 0x0c, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x09, 0x04, 0x03, 0x02,
  0x10, 0x17, 0x18, 0x01, 0x06, 0x08, 0x0d, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x0a, 0x04, 0x03, 0x02,
  0x10, 0x07, 0x18, 0x01, 0x05, 0x0b, 0x0c, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x09, 0x04, 0x03, 0x02,
  0x10, 0x19, 0x18, 0x01, 0x06, 0x08, 0x0d, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x0a, 0x04, 0x03, 0x02,
  0x10, 0x00, 0x18, 0x01, 0x05, 0x0b, 0x0c, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x09, 0x04, 0x03, 0x02,
  0x10, 0x11, 0x18, 0x01, 0x06, 0x08, 0x0d, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x0a, 0x04, 0x03, 0x02,
  0x10, 0x12, 0x18, 0x01, 0x05, 0x0b, 0x0c, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x09, 0x04, 0x03, 0x02,
  0x10, 0x15, 0x18, 0x01, 0x06, 0x08, 0x0d, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x0a, 0x04, 0x03, 0x02,
  0x10, 0x16, 0x18, 0x01, 0x05, 0x0b, 0x0c, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x09, 0x04, 0x03, 0x02,
  0x10, 0x17, 0x18, 0x01, 0x06, 0x08, 0x0d, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x0a, 0x04, 0x03, 0x02,
  0x10, 0x07, 0x18, 0x01, 0x05, 0x0b, 0x0c, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x09, 0x04, 0x03, 0x02,
  0x10, 0x19, 0x18, 0x01, 0x06, 0x08, 0x0d, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x0a, 0x04, 0x03, 0x02,
  0x10, 0x00, 0x18, 0x01, 0x05, 0x0b, 0x0c, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x09, 0x04, 0x03, 0x02,
  0x10, 0x11, 0x18, 0x01, 0x06, 0x08, 0x0d, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x0a, 0x04, 0x03, 0x02,
  0x10, 0x13, 0x18, 0x01, 0x05, 0x0b, 0x0c, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x09, 0x04, 0x03, 0x02,
  0x10, 0x15, 0x18, 0x01, 0x06, 0x08, 0x0d, 0x00, 0x0f, 0x14, 0x18, 0x01, 0x0a, 0x04, 0x03, 0x02,
])

# Reference opcode -> name mapping (old opcode values from rocprof)
OLD_OPCODE_TO_NAME = {
  0x01: 'VALUINST', 0x02: 'VMEMEXEC', 0x03: 'ALUEXEC', 0x04: 'IMMEDIATE',
  0x05: 'IMMEDIATE_MASK', 0x06: 'WAVERDY', 0x07: 'TS_DELTA_S8_W3',
  0x08: 'WAVEEND', 0x09: 'WAVESTART', 0x0A: 'TS_DELTA_S5_W2',
  0x0B: 'WAVEALLOC', 0x0C: 'TS_DELTA_S5_W3', 0x0D: 'PERF',
  0x0F: 'TS_DELTA_SHORT', 0x10: 'NOP', 0x11: 'TS_WAVE_STATE',
  0x12: 'EVENT', 0x13: 'EVENT_BIG', 0x14: 'REG', 0x15: 'SNAPSHOT',
  0x16: 'TS_DELTA_OR_MARK', 0x17: 'LAYOUT_HEADER', 0x18: 'INST',
  0x19: 'UTILCTR', 0x00: 'NOP',
}

# Reference budget values (nibbles for NEXT packet) from rocprof
REFERENCE_BUDGET_NIBBLES = {
  'VALUINST': 3, 'VMEMEXEC': 2, 'ALUEXEC': 2, 'IMMEDIATE': 3,
  'IMMEDIATE_MASK': 6, 'WAVERDY': 6, 'TS_DELTA_S8_W3': 16,
  'WAVEEND': 5, 'WAVESTART': 8, 'TS_DELTA_S5_W2': 12,
  'WAVEALLOC': 5, 'TS_DELTA_S5_W3': 13, 'PERF': 7,
  'TS_DELTA_SHORT': 2, 'NOP': 1, 'TS_WAVE_STATE': 6,
  'EVENT': 6, 'EVENT_BIG': 8, 'REG': 16, 'SNAPSHOT': 16,
  'TS_DELTA_OR_MARK': 12, 'LAYOUT_HEADER': 16, 'INST': 5,
  'UTILCTR': 12,
}


class TestEncodingsMatchStateTable(unittest.TestCase):
  """Verify encoding patterns produce the correct state decode table."""

  def test_all_256_bytes_decode_correctly(self):
    """Each byte value should decode to the same packet type as reference."""
    mismatches = []
    for byte_val in range(256):
      ref_opcode = REFERENCE_STATE_TABLE[byte_val]
      ref_name = OLD_OPCODE_TO_NAME.get(ref_opcode, f"UNK_{ref_opcode:02x}")

      our_opcode = STATE_TO_OPCODE[byte_val]
      our_name = OPCODE_TO_CLASS[our_opcode].__name__

      if ref_name != our_name:
        mismatches.append((byte_val, ref_name, our_name))

    if mismatches:
      msg = "\n".join(f"  0x{b:02x}: expected {r}, got {o}" for b, r, o in mismatches[:10])
      self.fail(f"State table mismatches ({len(mismatches)} total):\n{msg}")


class TestPacketSizesMatchBudget(unittest.TestCase):
  """Verify packet sizes (from field definitions) match expected budget values."""

  def test_all_packet_sizes(self):
    """Each packet type's size should match the reference budget."""
    for pkt_cls in PACKET_TYPES:
      name = pkt_cls.__name__
      expected = REFERENCE_BUDGET_NIBBLES.get(name)
      if expected is None:
        continue

      actual = pkt_cls.size_nibbles()
      self.assertEqual(expected, actual,
        f"{name}: expected {expected} nibbles, got {actual} (size_bits={pkt_cls.size_bits()})")


class TestFieldExtraction(unittest.TestCase):
  """Test that field values are extracted correctly."""

  def test_valuinst(self):
    reg = 0b11110_1_001_011  # wave=0x1E, flag=1, delta=1
    pkt = VALUINST.from_raw(reg)
    self.assertEqual(pkt.delta, 1)
    self.assertEqual(pkt.flag, 1)
    self.assertEqual(pkt.wave, 0x1E)

  def test_vmemexec_enum(self):
    reg = 0b11_00_1111  # src=3 (VMEM_ALT), delta=0
    pkt = VMEMEXEC.from_raw(reg)
    self.assertEqual(pkt.src, MemSrc.VMEM_ALT)

  def test_aluexec_enum(self):
    reg = 0b10_01_1110  # src=2 (VALU), delta=1
    pkt = ALUEXEC.from_raw(reg)
    self.assertEqual(pkt.src, AluSrc.VALU)

  def test_waveend(self):
    reg = (0x15 << 15) | (0x7 << 11) | (0x3 << 9) | (1 << 8) | 0b10101
    pkt = WAVEEND.from_raw(reg)
    self.assertEqual(pkt.flag7, 1)
    self.assertEqual(pkt.simd, 3)
    self.assertEqual(pkt.cu_lo, 7)
    self.assertEqual(pkt.wave, 0x15)
    self.assertEqual(pkt.cu, 0xF)  # cu_lo | (flag7 << 3) = 7 | 8 = 15

  def test_wavestart(self):
    reg = (0x7F << 18) | (0x15 << 13) | (0x7 << 10) | (0x3 << 8) | (1 << 7) | 0b01100
    pkt = WAVESTART.from_raw(reg)
    self.assertEqual(pkt.flag7, 1)
    self.assertEqual(pkt.simd, 3)
    self.assertEqual(pkt.cu_lo, 7)
    self.assertEqual(pkt.wave, 0x15)
    self.assertEqual(pkt.id7, 0x7F)
    self.assertEqual(pkt.cu, 0xF)

  def test_inst_enum(self):
    reg = (0x21 << 13) | (0x15 << 8) | (1 << 7) | (1 << 3) | 0b010
    pkt = INST.from_raw(reg)
    self.assertEqual(pkt.flag1, 1)
    self.assertEqual(pkt.flag2, 1)
    self.assertEqual(pkt.wave, 0x15)
    self.assertEqual(pkt.op, InstOp.VMEM_LOAD)

  def test_layout_header(self):
    reg = (0b101 << 33) | (0b1010 << 28) | (0b111 << 15) | (0b11 << 13) | (0b101010 << 7) | 0b0010001
    pkt = LAYOUT_HEADER.from_raw(reg)
    self.assertEqual(pkt.layout, 0b101010)
    self.assertEqual(pkt.simd, 0b11)
    self.assertEqual(pkt.group, 0b111)
    self.assertEqual(pkt.sel_a, 0b1010)
    self.assertEqual(pkt.sel_b, 0b101)

  def test_ts_delta_or_mark_modes(self):
    # delta mode: bit9=0, bit8=0
    pkt_delta = TS_DELTA_OR_MARK.from_raw(0b0000001)  # just the encoding pattern
    self.assertFalse(pkt_delta.is_marker)

    # marker mode: bit9=1, bit8=0
    pkt_marker = TS_DELTA_OR_MARK.from_raw(0b0000001 | (1 << 9))  # bit9=1, bit8=0
    self.assertTrue(pkt_marker.is_marker)

    # other mode: bit9=1, bit8=1 (not marker)
    pkt_other = TS_DELTA_OR_MARK.from_raw(0b0000001 | (1 << 8) | (1 << 9))
    self.assertFalse(pkt_other.is_marker)

  def test_reg(self):
    # REG fields: slot=bits[9:7], hi_byte=bits[15:8], subop=bits[31:16], val32=bits[63:32]
    # Note: slot[2:1] overlaps with hi_byte[1:0], so we need to set them consistently
    # hi_byte=0x55 means bits 8-15 = 0b01010101, so slot bits 8-9 = 0b01
    # slot bit 7 = 1, so slot = 0b011 = 3
    reg = (0xDEADBEEF << 32) | (0xCAFE << 16) | (0x55 << 8) | (1 << 7) | 0b1001
    pkt = REG.from_raw(reg)
    self.assertEqual(pkt.slot, 0b011)  # bit7=1, bits 8-9 from hi_byte low 2 bits = 01
    self.assertEqual(pkt.hi_byte, 0x55)
    self.assertEqual(pkt.subop, 0xCAFE)
    self.assertEqual(pkt.val32, 0xDEADBEEF)


class TestRoundtrip(unittest.TestCase):
  """Test encode/decode roundtrip."""

  def test_simple_roundtrip(self):
    """Test encode/decode roundtrip preserves packet types."""
    test_packets = [
      LAYOUT_HEADER.from_raw(0x100),
      WAVESTART.from_raw(0x0),
      INST.from_raw(0x10),
      INST.from_raw(0x10),
      WAVEEND.from_raw(0x40),
    ]
    encoded = encode(test_packets)
    decoded = decode(encoded)

    self.assertGreaterEqual(len(decoded), len(test_packets))
    for i, (orig, dec) in enumerate(zip(test_packets, decoded)):
      self.assertEqual(type(orig), type(dec), f"type mismatch at {i}")


if __name__ == "__main__":
  unittest.main()
