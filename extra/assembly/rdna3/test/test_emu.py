#!/usr/bin/env python3
"""Regression tests for the RDNA3 emulator instruction execution.

Run with emulator (default):
  PYTHONPATH="." python3 extra/assembly/rdna3/test/test_emu.py

Run on real AMD hardware:
  PYTHONPATH="." REAL_AMD=1 python3 extra/assembly/rdna3/test/test_emu.py
"""
import unittest
import struct
import os
from extra.assembly.rdna3.autogen import *
from extra.assembly.rdna3.emu import WaveState, decode_program, exec_wave, VCC_LO

USE_REAL_AMD = os.getenv("REAL_AMD", "0") == "1"
VCC = SrcEnum.VCC_LO  # For VOP3SD sdst field

def f2i(f: float) -> int:
  """Convert float32 to its bit representation."""
  return struct.unpack('I', struct.pack('f', f))[0]

def i2f(i: int) -> float:
  """Convert bit representation to float32."""
  return struct.unpack('f', struct.pack('I', i & 0xffffffff))[0]

def assemble(instructions: list) -> bytes:
  """Assemble instructions to bytes, handling literals properly."""
  code = b''
  for inst in instructions:
    inst_bytes = inst.to_bytes()
    code += inst_bytes
    # Check if instruction uses literal (src=255) and has a float/int value that needs appending
    if hasattr(inst, 'src0') and inst.src0 == 255:
      # Literal value should be set via _literal or we need to extract from context
      if hasattr(inst, '_literal') and inst._literal is not None:
        code += struct.pack('<I', inst._literal)
    elif hasattr(inst, 'src') and inst.src == 255:
      if hasattr(inst, '_literal') and inst._literal is not None:
        code += struct.pack('<I', inst._literal)
  return code

def run_program(instructions: list, n_lanes: int = 1) -> WaveState:
  """Assemble instructions, set up state, and run until s_endpgm."""
  instructions = instructions + [s_endpgm()]
  code = assemble(instructions)
  program = decode_program(code)
  st = WaveState()
  lds = bytearray(65536)
  exec_wave(program, st, lds, n_lanes)
  return st


class TestVDivScale(unittest.TestCase):
  """Tests for V_DIV_SCALE_F32 VCC handling."""

  def test_div_scale_f32_vcc_zero_single_lane(self):
    """V_DIV_SCALE_F32 sets VCC=0 when no scaling needed."""
    instructions = [
      v_mov_b32_e32(v[0], 1.0),  # uses inline constant
      v_mov_b32_e32(v[1], 4.0),  # uses inline constant
      v_div_scale_f32(v[2], VCC, v[0], v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc, 0, "VCC should be 0 when no scaling needed")

  def test_div_scale_f32_vcc_zero_multiple_lanes(self):
    """V_DIV_SCALE_F32 sets VCC=0 for all lanes when no scaling needed."""
    instructions = [
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 4.0),
      v_div_scale_f32(v[2], VCC, v[0], v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=4)
    self.assertEqual(st.vcc & 0xf, 0, "VCC should be 0 for all lanes")

  def test_div_scale_f32_preserves_input(self):
    """V_DIV_SCALE_F32 outputs S0 when no scaling needed."""
    instructions = [
      v_mov_b32_e32(v[0], 2.0),  # numerator - use inline constant
      v_mov_b32_e32(v[1], 4.0),  # denominator
      v_div_scale_f32(v[2], VCC, v[0], v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 2.0, places=5)


class TestVCmpClass(unittest.TestCase):
  """Tests for V_CMP_CLASS_F32 float classification."""

  def test_cmp_class_quiet_nan(self):
    """V_CMP_CLASS_F32 detects quiet NaN."""
    quiet_nan = 0x7fc00000
    instructions = [
      s_mov_b32(s[0], quiet_nan),  # large int encodes as literal
      v_mov_b32_e32(v[0], s[0]),  # value to classify
      v_mov_b32_e32(v[1], 0b0000000010),  # bit 1 = quiet NaN (mask in VGPR for VOPC)
      v_cmp_class_f32_e32(v[0], v[1]),  # VOPC: src0=value, vsrc1=mask, writes VCC
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect quiet NaN")

  def test_cmp_class_signaling_nan(self):
    """V_CMP_CLASS_F32 detects signaling NaN."""
    signal_nan = 0x7f800001
    instructions = [
      s_mov_b32(s[0], signal_nan),  # large int encodes as literal
      v_mov_b32_e32(v[0], s[0]),  # value to classify
      v_mov_b32_e32(v[1], 0b0000000001),  # bit 0 = signaling NaN
      v_cmp_class_f32_e32(v[0], v[1]),  # VOPC: src0=value, vsrc1=mask, writes VCC
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect signaling NaN")

  def test_cmp_class_quiet_nan_not_signaling(self):
    """Quiet NaN does not match signaling NaN mask."""
    quiet_nan = 0x7fc00000
    instructions = [
      s_mov_b32(s[0], quiet_nan),  # large int encodes as literal
      v_mov_b32_e32(v[0], s[0]),  # value to classify
      v_mov_b32_e32(v[1], 0b0000000001),  # bit 0 = signaling NaN only
      v_cmp_class_f32_e32(v[0], v[1]),  # VOPC: src0=value, vsrc1=mask, writes VCC
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 0, "Quiet NaN should not match signaling mask")

  def test_cmp_class_signaling_nan_not_quiet(self):
    """Signaling NaN does not match quiet NaN mask."""
    signal_nan = 0x7f800001
    instructions = [
      s_mov_b32(s[0], signal_nan),  # large int encodes as literal
      v_mov_b32_e32(v[0], s[0]),  # value to classify
      v_mov_b32_e32(v[1], 0b0000000010),  # bit 1 = quiet NaN only
      v_cmp_class_f32_e32(v[0], v[1]),  # VOPC: src0=value, vsrc1=mask, writes VCC
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 0, "Signaling NaN should not match quiet mask")

  def test_cmp_class_positive_inf(self):
    """V_CMP_CLASS_F32 detects +inf."""
    pos_inf = 0x7f800000
    instructions = [
      s_mov_b32(s[0], pos_inf),  # large int encodes as literal
      s_mov_b32(s[1], 0b1000000000),  # bit 9 = +inf (512 is outside inline range)
      v_mov_b32_e32(v[0], s[0]),  # value to classify
      v_mov_b32_e32(v[1], s[1]),  # mask in VGPR
      v_cmp_class_f32_e32(v[0], v[1]),  # VOPC: src0=value, vsrc1=mask, writes VCC
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect +inf")

  def test_cmp_class_negative_inf(self):
    """V_CMP_CLASS_F32 detects -inf."""
    neg_inf = 0xff800000
    instructions = [
      s_mov_b32(s[0], neg_inf),  # large int encodes as literal
      v_mov_b32_e32(v[0], s[0]),  # value to classify
      v_mov_b32_e32(v[1], 0b0000000100),  # bit 2 = -inf
      v_cmp_class_f32_e32(v[0], v[1]),  # VOPC: src0=value, vsrc1=mask, writes VCC
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect -inf")

  def test_cmp_class_normal_positive(self):
    """V_CMP_CLASS_F32 detects positive normal."""
    instructions = [
      v_mov_b32_e32(v[0], 1.0),  # inline constant - value to classify
      s_mov_b32(s[1], 0b0100000000),  # bit 8 = positive normal (256 is outside inline range)
      v_mov_b32_e32(v[1], s[1]),  # mask in VGPR
      v_cmp_class_f32_e32(v[0], v[1]),  # VOPC: src0=value, vsrc1=mask, writes VCC
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect positive normal")

  def test_cmp_class_normal_negative(self):
    """V_CMP_CLASS_F32 detects negative normal."""
    instructions = [
      v_mov_b32_e32(v[0], -1.0),  # inline constant - value to classify
      v_mov_b32_e32(v[1], 0b0000001000),  # bit 3 = negative normal
      v_cmp_class_f32_e32(v[0], v[1]),  # VOPC: src0=value, vsrc1=mask, writes VCC
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "Should detect negative normal")


class TestBasicOps(unittest.TestCase):
  """Basic instruction tests."""

  def test_v_add_f32(self):
    """V_ADD_F32 adds two floats."""
    instructions = [
      v_mov_b32_e32(v[0], 1.0),  # inline constant
      v_mov_b32_e32(v[1], 2.0),  # inline constant
      v_add_f32_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 3.0, places=5)

  def test_v_mul_f32(self):
    """V_MUL_F32 multiplies two floats."""
    instructions = [
      v_mov_b32_e32(v[0], 2.0),  # inline constant
      v_mov_b32_e32(v[1], 4.0),  # inline constant
      v_mul_f32_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 8.0, places=5)

  def test_v_mov_b32(self):
    """V_MOV_B32 moves a value."""
    instructions = [
      s_mov_b32(s[0], 42),
      v_mov_b32_e32(v[0], s[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][0], 42)

  def test_s_add_u32(self):
    """S_ADD_U32 adds two scalar values."""
    instructions = [
      s_mov_b32(s[0], 100),
      s_mov_b32(s[1], 200),
      s_add_u32(s[2], s[0], s[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[2], 300)

  def test_s_add_u32_carry(self):
    """S_ADD_U32 sets SCC on overflow."""
    instructions = [
      s_mov_b32(s[0], 64),  # use inline constant for max
      s_not_b32(s[0], s[0]),  # s0 = ~64 = 0xffffffbf, close to max
      s_mov_b32(s[1], 64),
      s_add_u32(s[2], s[0], s[1]),  # 0xffffffbf + 64 = 0xffffffff
      s_mov_b32(s[3], 1),
      s_add_u32(s[4], s[2], s[3]),  # 0xffffffff + 1 = overflow
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.sgpr[4], 0)
    self.assertEqual(st.scc, 1)

  def test_v_alignbit_b32(self):
    """V_ALIGNBIT_B32 extracts bits from concatenated sources."""
    instructions = [
      s_mov_b32(s[0], 0x12),  # small values as inline constants
      s_mov_b32(s[1], 0x34),
      s_mov_b32(s[2], 4),  # shift amount
      v_mov_b32_e32(v[0], s[2]),
      v_alignbit_b32(v[1], s[0], s[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    # {0x12, 0x34} >> 4 = 0x0000001200000034 >> 4 = 0x20000003
    expected = ((0x12 << 32) | 0x34) >> 4
    self.assertEqual(st.vgpr[0][1], expected & 0xffffffff)


class TestMultiLane(unittest.TestCase):
  """Tests for multi-lane execution."""

  def test_v_mov_all_lanes(self):
    """V_MOV_B32 sets all lanes to the same value."""
    instructions = [
      s_mov_b32(s[0], 42),
      v_mov_b32_e32(v[0], s[0]),
    ]
    st = run_program(instructions, n_lanes=4)
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][0], 42)

  def test_v_cmp_sets_vcc_bits(self):
    """V_CMP_EQ sets VCC bits based on per-lane comparison."""
    instructions = [
      s_mov_b32(s[0], 5),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[0]),
      v_cmp_eq_u32_e32(v[0], v[1]),  # VOPC: src0, vsrc1 - writes VCC implicitly
    ]
    st = run_program(instructions, n_lanes=4)
    self.assertEqual(st.vcc & 0xf, 0xf, "All lanes should match")


if __name__ == '__main__':
  unittest.main()
