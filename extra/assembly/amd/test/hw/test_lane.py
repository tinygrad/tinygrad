#!/usr/bin/env python3
"""Tests for cross-lane instructions: readlane, writelane, readfirstlane."""
import unittest
from extra.assembly.amd.autogen.rdna3.ins import *
from extra.assembly.amd.dsl import RawImm
from extra.assembly.amd.test.hw.helpers import run_program

class TestLaneInstructions(unittest.TestCase):
  """Tests for cross-lane instructions - critical for reductions and WMMA."""

  def _readlane(self, sdst_idx, vsrc, lane_idx):
    """Helper to create V_READLANE_B32 with SGPR destination."""
    return VOP3(VOP3Op.V_READLANE_B32, vdst=RawImm(sdst_idx), src0=vsrc, src1=lane_idx)

  def _readfirstlane(self, sdst_idx, vsrc):
    """Helper to create V_READFIRSTLANE_B32 with SGPR destination."""
    return VOP1(VOP1Op.V_READFIRSTLANE_B32, vdst=RawImm(sdst_idx), src0=vsrc)

  def test_readlane_basic(self):
    """V_READLANE_B32 reads a value from a specific lane's VGPR."""
    st = run_program([
      v_lshlrev_b32_e32(v[0], 1, v[255]),
      v_lshlrev_b32_e32(v[1], 3, v[255]),
      v_add_nc_u32_e32(v[0], v[0], v[1]),
      self._readlane(0, v[0], 2),
      v_mov_b32_e32(v[2], s[0]),
    ], n_lanes=4)
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][2], 20)

  def test_readlane_lane_0(self):
    """V_READLANE_B32 reading from lane 0."""
    st = run_program([
      v_lshlrev_b32_e32(v[0], 2, v[255]),
      v_add_nc_u32_e32(v[0], 100, v[0]),
      self._readlane(0, v[0], 0),
      v_mov_b32_e32(v[1], s[0]),
    ], n_lanes=4)
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][1], 100)

  def test_readlane_last_lane(self):
    """V_READLANE_B32 reading from the last active lane."""
    st = run_program([
      v_lshlrev_b32_e32(v[0], 2, v[255]),
      v_add_nc_u32_e32(v[0], 100, v[0]),
      self._readlane(0, v[0], 3),
      v_mov_b32_e32(v[1], s[0]),
    ], n_lanes=4)
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][1], 112)

  def test_readlane_different_vgpr(self):
    """V_READLANE_B32 reading from different VGPR indices."""
    st = run_program([
      v_lshlrev_b32_e32(v[5], 3, v[255]),
      v_add_nc_u32_e32(v[5], 50, v[5]),
      self._readlane(0, v[5], 1),
      v_mov_b32_e32(v[6], s[0]),
    ], n_lanes=4)
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][6], 58)

  def test_readfirstlane_basic(self):
    """V_READFIRSTLANE_B32 reads from the first active lane."""
    st = run_program([
      v_lshlrev_b32_e32(v[0], 2, v[255]),
      v_add_nc_u32_e32(v[0], 1000, v[0]),
      self._readfirstlane(0, v[0]),
      v_mov_b32_e32(v[1], s[0]),
    ], n_lanes=4)
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][1], 1000)

  def test_readfirstlane_different_vgpr(self):
    """V_READFIRSTLANE_B32 reading from different VGPR index."""
    st = run_program([
      v_lshlrev_b32_e32(v[7], 5, v[255]),
      v_add_nc_u32_e32(v[7], 200, v[7]),
      self._readfirstlane(0, v[7]),
      v_mov_b32_e32(v[8], s[0]),
    ], n_lanes=4)
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][8], 200)

  def test_writelane_basic(self):
    """V_WRITELANE_B32 writes a scalar to a specific lane's VGPR."""
    st = run_program([
      v_mov_b32_e32(v[0], 0),
      s_mov_b32(s[0], 999),
      v_writelane_b32(v[0], s[0], 2),
    ], n_lanes=4)
    for lane in range(4):
      if lane == 2:
        self.assertEqual(st.vgpr[lane][0], 999)
      else:
        self.assertEqual(st.vgpr[lane][0], 0)

  def test_writelane_then_readlane(self):
    """V_WRITELANE followed by V_READLANE round-trip."""
    st = run_program([
      v_mov_b32_e32(v[0], 0),
      s_mov_b32(s[0], 0xdeadbeef),
      v_writelane_b32(v[0], s[0], 1),
      self._readlane(1, v[0], 1),
      v_mov_b32_e32(v[1], s[1]),
    ], n_lanes=4)
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][1], 0xdeadbeef)

  def test_writelane_different_vgpr(self):
    """V_WRITELANE_B32 writes to a non-zero VGPR index."""
    st = run_program([
      v_mov_b32_e32(v[0], 0),
      v_mov_b32_e32(v[5], 0),
      s_mov_b32(s[0], 0x12345678),
      v_writelane_b32(v[5], s[0], 1),
    ], n_lanes=4)
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][0], 0)
    for lane in range(4):
      if lane == 1:
        self.assertEqual(st.vgpr[lane][5], 0x12345678)
      else:
        self.assertEqual(st.vgpr[lane][5], 0)

  def test_writelane_high_vgpr(self):
    """V_WRITELANE_B32 writes to a high VGPR index (v[15])."""
    st = run_program([
      v_mov_b32_e32(v[0], 0),
      v_mov_b32_e32(v[15], 0),
      s_mov_b32(s[0], 0xCAFEBABE),
      v_writelane_b32(v[15], s[0], 0),
    ], n_lanes=4)
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][0], 0)
    self.assertEqual(st.vgpr[0][15], 0xCAFEBABE)
    for lane in range(1, 4):
      self.assertEqual(st.vgpr[lane][15], 0)

  def test_writelane_multiple_vgprs(self):
    """V_WRITELANE_B32 writes to multiple different VGPRs."""
    st = run_program([
      v_mov_b32_e32(v[0], 0),
      v_mov_b32_e32(v[3], 0),
      v_mov_b32_e32(v[7], 0),
      v_mov_b32_e32(v[10], 0),
      s_mov_b32(s[0], 100),
      v_writelane_b32(v[3], s[0], 0),
      s_mov_b32(s[0], 200),
      v_writelane_b32(v[7], s[0], 1),
      s_mov_b32(s[0], 300),
      v_writelane_b32(v[10], s[0], 2),
    ], n_lanes=4)
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][0], 0)
    self.assertEqual(st.vgpr[0][3], 100)
    self.assertEqual(st.vgpr[1][7], 200)
    self.assertEqual(st.vgpr[2][10], 300)

  def test_readlane_for_reduction(self):
    """Simulate a wave reduction using readlane."""
    st = run_program([
      v_add_nc_u32_e32(v[0], 1, v[255]),
      self._readlane(0, v[0], 0),
      self._readlane(1, v[0], 1),
      s_add_u32(s[0], s[0], s[1]),
      self._readlane(1, v[0], 2),
      s_add_u32(s[0], s[0], s[1]),
      self._readlane(1, v[0], 3),
      s_add_u32(s[0], s[0], s[1]),
      v_mov_b32_e32(v[1], s[0]),
    ], n_lanes=4)
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][1], 10)

  def test_writelane_accumulate(self):
    """V_WRITELANE_B32 used to accumulate values across lanes."""
    st = run_program([
      v_mov_b32_e32(v[6], 0),
      s_mov_b32(s[0], 10),
      v_writelane_b32(v[6], s[0], 0),
      s_mov_b32(s[0], 20),
      v_writelane_b32(v[6], s[0], 1),
      s_mov_b32(s[0], 30),
      v_writelane_b32(v[6], s[0], 2),
      s_mov_b32(s[0], 40),
      v_writelane_b32(v[6], s[0], 3),
      self._readlane(0, v[6], 0),
      self._readlane(1, v[6], 1),
      s_add_u32(s[0], s[0], s[1]),
      self._readlane(1, v[6], 2),
      s_add_u32(s[0], s[0], s[1]),
      self._readlane(1, v[6], 3),
      s_add_u32(s[0], s[0], s[1]),
      v_mov_b32_e32(v[7], s[0]),
    ], n_lanes=4)
    self.assertEqual(st.vgpr[0][6], 10)
    self.assertEqual(st.vgpr[1][6], 20)
    self.assertEqual(st.vgpr[2][6], 30)
    self.assertEqual(st.vgpr[3][6], 40)
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][7], 100)


if __name__ == '__main__':
  unittest.main()
