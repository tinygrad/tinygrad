#!/usr/bin/env python3
"""Tests for FLAT and GLOBAL memory instructions - atomics, stores, loads."""
import unittest
from extra.assembly.amd.autogen.rdna3.ins import *
from extra.assembly.amd.test.hw.helpers import run_program


class TestFLATAtomic(unittest.TestCase):
  """Tests for FLAT and GLOBAL atomic instructions."""

  # Helper to set up address in v[0:1] and clear after test
  def _make_test(self, setup_instrs, atomic_instr, check_fn, test_offset=2000):
    """Helper to create atomic test instructions."""
    instructions = [
      # Load output buffer address from args (saved in s[80:81] by prologue)
      s_load_b64(s[2:3], s[80], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      v_mov_b32_e32(v[0], s[2]),  # addr low
      v_mov_b32_e32(v[1], s[3]),  # addr high
    ] + setup_instrs + [atomic_instr, s_waitcnt(vmcnt=0),
      # Clear address registers that differ between emu/hw
      v_mov_b32_e32(v[0], 0),
      v_mov_b32_e32(v[1], 0),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    check_fn(st)

  def test_flat_atomic_add_u32(self):
    """FLAT_ATOMIC_ADD_U32 adds to memory and returns old value."""
    TEST_OFFSET = 2000
    setup = [
      s_mov_b32(s[0], 100),
      v_mov_b32_e32(v[2], s[0]),
      global_store_b32(addr=v[0], data=v[2], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      s_mov_b32(s[0], 50),
      v_mov_b32_e32(v[3], s[0]),  # add 50
    ]
    atomic = FLAT(FLATOp.FLAT_ATOMIC_ADD_U32, addr=v[0], data=v[3], vdst=v[4], saddr=SrcEnum.NULL, offset=TEST_OFFSET, glc=1)
    def check(st):
      self.assertEqual(st.vgpr[0][4], 100, "v4 should have old value (100)")
    self._make_test(setup, atomic, check, TEST_OFFSET)

  def test_flat_atomic_swap_b32(self):
    """FLAT_ATOMIC_SWAP_B32 swaps memory value and returns old value."""
    TEST_OFFSET = 2000
    setup = [
      s_mov_b32(s[0], 0xAAAAAAAA),
      v_mov_b32_e32(v[2], s[0]),
      global_store_b32(addr=v[0], data=v[2], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      s_mov_b32(s[0], 0xBBBBBBBB),
      v_mov_b32_e32(v[3], s[0]),  # new value
    ]
    atomic = FLAT(FLATOp.FLAT_ATOMIC_SWAP_B32, addr=v[0], data=v[3], vdst=v[4], saddr=SrcEnum.NULL, offset=TEST_OFFSET, glc=1)
    def check(st):
      self.assertEqual(st.vgpr[0][4], 0xAAAAAAAA, "v4 should have old value")
    self._make_test(setup, atomic, check, TEST_OFFSET)

  def test_flat_atomic_and_b32(self):
    """FLAT_ATOMIC_AND_B32 ANDs with memory and returns old value."""
    TEST_OFFSET = 2000
    setup = [
      s_mov_b32(s[0], 0xFF00FF00),
      v_mov_b32_e32(v[2], s[0]),
      global_store_b32(addr=v[0], data=v[2], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      s_mov_b32(s[0], 0xFFFF0000),
      v_mov_b32_e32(v[3], s[0]),  # AND mask
    ]
    atomic = FLAT(FLATOp.FLAT_ATOMIC_AND_B32, addr=v[0], data=v[3], vdst=v[4], saddr=SrcEnum.NULL, offset=TEST_OFFSET, glc=1)
    def check(st):
      self.assertEqual(st.vgpr[0][4], 0xFF00FF00, "v4 should have old value")
    self._make_test(setup, atomic, check, TEST_OFFSET)

  def test_flat_atomic_or_b32(self):
    """FLAT_ATOMIC_OR_B32 ORs with memory and returns old value."""
    TEST_OFFSET = 2000
    setup = [
      s_mov_b32(s[0], 0x00FF0000),
      v_mov_b32_e32(v[2], s[0]),
      global_store_b32(addr=v[0], data=v[2], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      s_mov_b32(s[0], 0x0000FF00),
      v_mov_b32_e32(v[3], s[0]),  # OR mask
    ]
    atomic = FLAT(FLATOp.FLAT_ATOMIC_OR_B32, addr=v[0], data=v[3], vdst=v[4], saddr=SrcEnum.NULL, offset=TEST_OFFSET, glc=1)
    def check(st):
      self.assertEqual(st.vgpr[0][4], 0x00FF0000, "v4 should have old value")
    self._make_test(setup, atomic, check, TEST_OFFSET)

  def test_flat_atomic_min_u32(self):
    """FLAT_ATOMIC_MIN_U32 stores min and returns old value."""
    TEST_OFFSET = 2000
    setup = [
      s_mov_b32(s[0], 100),
      v_mov_b32_e32(v[2], s[0]),
      global_store_b32(addr=v[0], data=v[2], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      s_mov_b32(s[0], 50),
      v_mov_b32_e32(v[3], s[0]),  # compare value (smaller)
    ]
    atomic = FLAT(FLATOp.FLAT_ATOMIC_MIN_U32, addr=v[0], data=v[3], vdst=v[4], saddr=SrcEnum.NULL, offset=TEST_OFFSET, glc=1)
    def check(st):
      self.assertEqual(st.vgpr[0][4], 100, "v4 should have old value (100)")
    self._make_test(setup, atomic, check, TEST_OFFSET)

  def test_flat_atomic_max_u32(self):
    """FLAT_ATOMIC_MAX_U32 stores max and returns old value."""
    TEST_OFFSET = 2000
    setup = [
      s_mov_b32(s[0], 50),
      v_mov_b32_e32(v[2], s[0]),
      global_store_b32(addr=v[0], data=v[2], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      s_mov_b32(s[0], 100),
      v_mov_b32_e32(v[3], s[0]),  # compare value (larger)
    ]
    atomic = FLAT(FLATOp.FLAT_ATOMIC_MAX_U32, addr=v[0], data=v[3], vdst=v[4], saddr=SrcEnum.NULL, offset=TEST_OFFSET, glc=1)
    def check(st):
      self.assertEqual(st.vgpr[0][4], 50, "v4 should have old value (50)")
    self._make_test(setup, atomic, check, TEST_OFFSET)


class TestGlobalStoreB64(unittest.TestCase):
  """Tests for GLOBAL_STORE_B64 instruction."""

  def test_global_store_load_b64(self):
    """GLOBAL_STORE_B64 and GLOBAL_LOAD_B64 for 64-bit values."""
    TEST_OFFSET = 2000
    instructions = [
      s_load_b64(s[2:3], s[80], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      v_mov_b32_e32(v[0], s[2]),
      v_mov_b32_e32(v[1], s[3]),
      # Store 64-bit value
      s_mov_b32(s[0], 0xDEADBEEF),
      v_mov_b32_e32(v[2], s[0]),
      s_mov_b32(s[0], 0xCAFEBABE),
      v_mov_b32_e32(v[3], s[0]),
      global_store_b64(addr=v[0], data=v[2], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      # Load back
      global_load_b64(addr=v[0], vdst=v[4], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], 0),
      v_mov_b32_e32(v[1], 0),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][4], 0xDEADBEEF, "Low dword")
    self.assertEqual(st.vgpr[0][5], 0xCAFEBABE, "High dword")


class TestD16HiLoads(unittest.TestCase):
  """Tests for D16_HI/D16_LO load variants."""

  def test_global_load_d16_hi_b16(self):
    """GLOBAL_LOAD_D16_HI_B16 loads 16-bit value to high half of destination."""
    TEST_OFFSET = 2000
    instructions = [
      s_load_b64(s[2:3], s[80], 0, soffset=SrcEnum.NULL),
      s_waitcnt(lgkmcnt=0),
      v_mov_b32_e32(v[0], s[2]),
      v_mov_b32_e32(v[1], s[3]),
      # Store a 32-bit value containing f16 in low bits
      s_mov_b32(s[0], 0x12343c00),  # hi=0x1234, lo=f16 1.0 (0x3c00)
      v_mov_b32_e32(v[2], s[0]),
      global_store_b32(addr=v[0], data=v[2], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      # Pre-fill destination with known value
      s_mov_b32(s[0], 0x0000DEAD),
      v_mov_b32_e32(v[4], s[0]),
      # Load D16_HI - should load 16-bit value to HIGH half, preserving low
      global_load_d16_hi_b16(addr=v[0], vdst=v[4], saddr=SrcEnum.NULL, offset=TEST_OFFSET),
      s_waitcnt(vmcnt=0),
      v_mov_b32_e32(v[0], 0),
      v_mov_b32_e32(v[1], 0),
      s_mov_b32(s[2], 0),
      s_mov_b32(s[3], 0),
    ]
    st = run_program(instructions, n_lanes=1)
    result_hi = (st.vgpr[0][4] >> 16) & 0xFFFF
    result_lo = st.vgpr[0][4] & 0xFFFF
    self.assertEqual(result_hi, 0x3c00, f"High half should have loaded 0x3c00, got 0x{result_hi:04x}")
    self.assertEqual(result_lo, 0xDEAD, f"Low half should be preserved as 0xDEAD, got 0x{result_lo:04x}")


if __name__ == "__main__":
  unittest.main()
