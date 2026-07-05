#!/usr/bin/env python3
"""Regression tests for RDNA4 v_s_*_f16 scalar-VALU ops in the emulator.

These ops write the f16 result to the low 16 bits of the SGPR and zero the high
16 bits, which the pcode expresses as two separate writes (D0.f16 and D0[31:16]).
The emulator must combine these into a single SGPR store, otherwise the generated
program has two stores to the same register and tripping the coalescer assertion
"attempting multiple stores". See _compile_vop3 in test/mockgpu/amd/emu.py.
"""
import unittest
from test.mockgpu.amd.emu import _get_runner
from tinygrad.runtime.autogen.amd.rdna4.ins import s, v_s_rcp_f16, v_s_sqrt_f16, v_s_rsq_f16, v_s_log_f16, v_s_exp_f16, v_s_rcp_f32

class TestEmuVSF16(unittest.TestCase):
  def _compiles(self, inst):
    # _get_runner raises if the instruction can't be lowered+compiled (e.g. multiple stores to one reg)
    _get_runner(inst.to_bytes(), 'rdna4')

  def test_v_s_rcp_f16(self): self._compiles(v_s_rcp_f16(s[2], s[2]))
  def test_v_s_sqrt_f16(self): self._compiles(v_s_sqrt_f16(s[2], s[2]))
  def test_v_s_rsq_f16(self): self._compiles(v_s_rsq_f16(s[2], s[2]))
  def test_v_s_log_f16(self): self._compiles(v_s_log_f16(s[2], s[2]))
  def test_v_s_exp_f16(self): self._compiles(v_s_exp_f16(s[2], s[2]))
  def test_v_s_rcp_f32(self): self._compiles(v_s_rcp_f32(s[2], s[2]))  # f32 single-write control

if __name__ == "__main__":
  unittest.main()
