#!/usr/bin/env python3
"""Tests for DS (Data Share / LDS) instructions."""
import unittest
from extra.assembly.amd.autogen.rdna3.ins import *
from extra.assembly.amd.test.hw.helpers import run_program


class TestDS2Addr(unittest.TestCase):
  """Regression tests for DS_LOAD_2ADDR and DS_STORE_2ADDR instructions.
  These ops use offset scaling: offset * sizeof(data) for address calculation.
  Bug: Emulator was using offset*4 for both B32 and B64, but B64 needs offset*8."""

  def test_ds_store_load_2addr_b32(self):
    """DS_STORE_2ADDR_B32 and DS_LOAD_2ADDR_B32 with offset scaling by 4."""
    # Store 0x12345678 at offset0=0 (*4=0) and 0xDEADBEEF at offset1=1 (*4=4)
    instructions = [
      v_mov_b32_e32(v[10], 0),  # addr base = 0
      s_mov_b32(s[2], 0x12345678),
      v_mov_b32_e32(v[0], s[2]),  # data0
      s_mov_b32(s[2], 0xDEADBEEF),
      v_mov_b32_e32(v[1], s[2]),  # data1
      DS(DSOp.DS_STORE_2ADDR_B32, addr=v[10], data0=v[0], data1=v[1], vdst=v[0], offset0=0, offset1=1),
      s_waitcnt(lgkmcnt=0),
      DS(DSOp.DS_LOAD_2ADDR_B32, addr=v[10], vdst=v[2], offset0=0, offset1=1),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 0x12345678, "v2 should have value from offset 0")
    self.assertEqual(st.vgpr[0][3], 0xDEADBEEF, "v3 should have value from offset 4")

  def test_ds_store_load_2addr_b32_nonzero_offsets(self):
    """DS_STORE_2ADDR_B32 with non-zero offsets (offset*4 scaling)."""
    # Store at offset0=2 (*4=8) and offset1=5 (*4=20)
    instructions = [
      v_mov_b32_e32(v[10], 0),  # addr base = 0
      s_mov_b32(s[2], 0x11111111),
      v_mov_b32_e32(v[0], s[2]),
      s_mov_b32(s[2], 0x22222222),
      v_mov_b32_e32(v[1], s[2]),
      DS(DSOp.DS_STORE_2ADDR_B32, addr=v[10], data0=v[0], data1=v[1], vdst=v[0], offset0=2, offset1=5),
      s_waitcnt(lgkmcnt=0),
      DS(DSOp.DS_LOAD_2ADDR_B32, addr=v[10], vdst=v[2], offset0=2, offset1=5),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 0x11111111, "v2 should have value from offset 8 (2*4)")
    self.assertEqual(st.vgpr[0][3], 0x22222222, "v3 should have value from offset 20 (5*4)")

  def test_ds_store_load_2addr_b64(self):
    """DS_STORE_2ADDR_B64 and DS_LOAD_2ADDR_B64 with offset scaling by 8."""
    # For B64: each value is 8 bytes (2 dwords), offsets scaled by 8
    instructions = [
      v_mov_b32_e32(v[10], 0),  # addr base = 0
      # First 64-bit value: 0x123456789ABCDEF0
      s_mov_b32(s[2], 0x9ABCDEF0),
      v_mov_b32_e32(v[0], s[2]),  # low dword
      s_mov_b32(s[2], 0x12345678),
      v_mov_b32_e32(v[1], s[2]),  # high dword
      # Second 64-bit value: 0xDEADBEEFCAFEBABE
      s_mov_b32(s[2], 0xCAFEBABE),
      v_mov_b32_e32(v[2], s[2]),  # low dword
      s_mov_b32(s[2], 0xDEADBEEF),
      v_mov_b32_e32(v[3], s[2]),  # high dword
      DS(DSOp.DS_STORE_2ADDR_B64, addr=v[10], data0=v[0], data1=v[2], vdst=v[0], offset0=0, offset1=1),
      s_waitcnt(lgkmcnt=0),
      DS(DSOp.DS_LOAD_2ADDR_B64, addr=v[10], vdst=v[4], offset0=0, offset1=1),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    # v4,v5 = first 64-bit value from offset 0
    self.assertEqual(st.vgpr[0][4], 0x9ABCDEF0, "v4 should have low dword of first value")
    self.assertEqual(st.vgpr[0][5], 0x12345678, "v5 should have high dword of first value")
    # v6,v7 = second 64-bit value from offset 8 (1*8)
    self.assertEqual(st.vgpr[0][6], 0xCAFEBABE, "v6 should have low dword of second value")
    self.assertEqual(st.vgpr[0][7], 0xDEADBEEF, "v7 should have high dword of second value")


class TestDSAtomic(unittest.TestCase):
  """Tests for DS atomic instructions (add, max, min, and, or, xor, cmpstore, etc.)."""

  def test_ds_max_rtn_u32(self):
    """DS_MAX_RTN_U32: atomically store max(mem, data) and return old value."""
    instructions = [
      v_mov_b32_e32(v[10], 0),  # addr = 0
      s_mov_b32(s[2], 100),
      v_mov_b32_e32(v[0], s[2]),  # initial value = 100
      ds_store_b32(addr=v[10], data0=v[0], offset0=0),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[2], 200),
      v_mov_b32_e32(v[1], s[2]),  # data = 200 (greater than 100)
      ds_max_rtn_u32(addr=v[10], data0=v[1], vdst=v[2], offset0=0),
      s_waitcnt(lgkmcnt=0),
      ds_load_b32(addr=v[10], vdst=v[3], offset0=0),  # read result
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 100, "v2 should have old value (100)")
    self.assertEqual(st.vgpr[0][3], 200, "v3 should have max(100, 200) = 200")

  def test_ds_add_u32_no_rtn_preserves_vdst(self):
    """DS_ADD_U32 (no RTN) should NOT write to vdst - vdst should preserve sentinel value."""
    instructions = [
      v_mov_b32_e32(v[10], 0),
      # Set sentinel value in vdst
      s_mov_b32(s[2], 0xDEADBEEF),
      v_mov_b32_e32(v[2], s[2]),  # sentinel in v2
      # Store initial value
      s_mov_b32(s[2], 100),
      v_mov_b32_e32(v[0], s[2]),
      ds_store_b32(addr=v[10], data0=v[0], offset0=0),
      s_waitcnt(lgkmcnt=0),
      # Do non-RTN add (should NOT write to v2)
      s_mov_b32(s[2], 50),
      v_mov_b32_e32(v[1], s[2]),
      ds_add_u32(addr=v[10], data0=v[1], vdst=v[2], offset0=0),
      s_waitcnt(lgkmcnt=0),
      # Load result to verify add worked
      ds_load_b32(addr=v[10], vdst=v[3], offset0=0),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 0xDEADBEEF, "v2 should preserve sentinel (no RTN)")
    self.assertEqual(st.vgpr[0][3], 150, "v3 should have 100 + 50 = 150")

  def test_ds_add_rtn_u32_writes_vdst(self):
    """DS_ADD_RTN_U32 should write old value to vdst."""
    instructions = [
      v_mov_b32_e32(v[10], 0),
      s_mov_b32(s[2], 100),
      v_mov_b32_e32(v[0], s[2]),
      ds_store_b32(addr=v[10], data0=v[0], offset0=0),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[2], 50),
      v_mov_b32_e32(v[1], s[2]),
      ds_add_rtn_u32(addr=v[10], data0=v[1], vdst=v[2], offset0=0),
      s_waitcnt(lgkmcnt=0),
      ds_load_b32(addr=v[10], vdst=v[3], offset0=0),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 100, "v2 should have old value (100)")
    self.assertEqual(st.vgpr[0][3], 150, "v3 should have 100 + 50 = 150")


class TestDSRegisterWidth(unittest.TestCase):
  """Tests for DS instructions with different register widths (B32, B64, B128)."""

  def test_ds_load_b32(self):
    """DS_LOAD_B32 reads 4 bytes into one VGPR."""
    instructions = [
      v_mov_b32_e32(v[10], 0),
      s_mov_b32(s[2], 0xCAFEBABE),
      v_mov_b32_e32(v[0], s[2]),
      ds_store_b32(addr=v[10], data0=v[0], offset0=0),
      s_waitcnt(lgkmcnt=0),
      ds_load_b32(addr=v[10], vdst=v[1], offset0=0),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0xCAFEBABE)

  def test_ds_load_b64(self):
    """DS_LOAD_B64 reads 8 bytes into two VGPRs."""
    instructions = [
      v_mov_b32_e32(v[10], 0),
      s_mov_b32(s[2], 0xDEADBEEF),
      v_mov_b32_e32(v[0], s[2]),
      s_mov_b32(s[2], 0xCAFEBABE),
      v_mov_b32_e32(v[1], s[2]),
      ds_store_b64(addr=v[10], data0=v[0], offset0=0),
      s_waitcnt(lgkmcnt=0),
      ds_load_b64(addr=v[10], vdst=v[2], offset0=0),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 0xDEADBEEF, "Low dword")
    self.assertEqual(st.vgpr[0][3], 0xCAFEBABE, "High dword")


class TestDSStorexchg(unittest.TestCase):
  """Tests for DS exchange instructions."""

  def test_ds_storexchg_rtn_b32(self):
    """DS_STOREXCHG_RTN_B32 writes new value and returns old."""
    instructions = [
      v_mov_b32_e32(v[10], 0),
      s_mov_b32(s[2], 0xAAAAAAAA),
      v_mov_b32_e32(v[0], s[2]),  # initial
      ds_store_b32(addr=v[10], data0=v[0], offset0=0),
      s_waitcnt(lgkmcnt=0),
      s_mov_b32(s[2], 0xBBBBBBBB),
      v_mov_b32_e32(v[1], s[2]),  # new value
      ds_storexchg_rtn_b32(addr=v[10], data0=v[1], vdst=v[2], offset0=0),
      s_waitcnt(lgkmcnt=0),
      ds_load_b32(addr=v[10], vdst=v[3], offset0=0),
      s_waitcnt(lgkmcnt=0),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][2], 0xAAAAAAAA, "v2 should have old value")
    self.assertEqual(st.vgpr[0][3], 0xBBBBBBBB, "v3 should have new value")


if __name__ == "__main__":
  unittest.main()
