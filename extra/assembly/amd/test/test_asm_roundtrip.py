#!/usr/bin/env python3
"""Test asm.py instruction parsing and disassembly roundtrip."""
import unittest
from extra.assembly.amd.autogen.rdna3.ins import *
from extra.assembly.amd.dsl import RawImm, VCC_LO, EXEC_LO, NULL, OFF
from extra.assembly.amd.asm import asm, get_dsl, disasm

class TestVOP2VCCInstructions(unittest.TestCase):
  """Test VOP2 carry instructions with VCC operands."""

  def test_v_add_co_ci_u32_e32_disasm(self):
    # v_add_co_ci_u32_e32 uses VCC implicitly for both carry in and carry out
    inst = v_add_co_ci_u32_e32(v[6], s[11], v[6])
    asm_str = inst.disasm()
    self.assertIn('v_add_co_ci_u32_e32', asm_str)
    self.assertIn('vcc_lo', asm_str)
    # Should have format: v_add_co_ci_u32_e32 vdst, vcc_lo, src0, vsrc1, vcc_lo
    self.assertEqual(asm_str.count('vcc_lo'), 2)

  def test_v_sub_co_ci_u32_e32_disasm(self):
    inst = v_sub_co_ci_u32_e32(v[10], s[5], v[3])
    asm_str = inst.disasm()
    self.assertIn('v_sub_co_ci_u32_e32', asm_str)
    self.assertEqual(asm_str.count('vcc_lo'), 2)

  def test_v_add_co_ci_u32_roundtrip(self):
    # Test that we can parse and reassemble
    original = 'v_add_co_ci_u32_e32 v6, vcc_lo, s11, v6, vcc_lo'
    inst = asm(original)
    self.assertEqual(inst.disasm(), original)


class TestVOP3SDInstructions(unittest.TestCase):
  """Test VOP3SD (carry out) instructions."""

  def test_v_add_co_u32_disasm(self):
    # v_add_co_u32 (VOP3SD) should NOT have _e64 suffix
    inst = v_add_co_u32(v[5], VCC_LO, s[10], v[5])
    asm_str = inst.disasm()
    self.assertIn('v_add_co_u32', asm_str)
    self.assertNotIn('_e64', asm_str)
    self.assertIn('vcc_lo', asm_str)

  def test_v_add_co_u32_roundtrip(self):
    original = 'v_add_co_u32 v5, vcc_lo, s10, v5'
    inst = asm(original)
    self.assertEqual(inst.disasm(), original)


class TestSMEMOffset(unittest.TestCase):
  """Test SMEM offset handling."""

  def test_smem_null_offset_zero(self):
    # When soffset is null (124) and offset is 0, should output 0x0 not null
    # s_load_b128 needs 4 regs: s[20:23] means s20,s21,s22,s23 (inclusive end in assembly syntax)
    inst = s_load_b128(sdata=s[20:23], sbase=s[0:1], offset=0, soffset=RawImm(124))
    asm_str = inst.disasm()
    self.assertIn('0x0', asm_str)
    self.assertNotIn('null', asm_str)

  def test_smem_with_offset(self):
    # s_load_b64 needs 2 regs: s[0:1] means s0,s1 (inclusive end in assembly syntax)
    inst = s_load_b64(sdata=s[0:1], sbase=s[2:3], offset=0x10, soffset=RawImm(124))
    asm_str = inst.disasm()
    self.assertIn('0x10', asm_str)

  def test_smem_roundtrip(self):
    original = 's_load_b128 s[20:23], s[0:1], 0x0'
    inst = asm(original)
    self.assertEqual(inst.disasm(), original)


class TestDSOffsets(unittest.TestCase):
  """Test DS instruction offset parsing."""

  def test_ds_load_with_offset(self):
    inst = ds_load_b64(vdst=v[190:192], addr=v[183], offset0=8)
    asm_str = inst.disasm()
    self.assertIn('offset:8', asm_str)

  def test_ds_load_offset_parsing_with_space(self):
    # Reference assembly uses "offset: 8" with space after colon
    dsl = get_dsl('ds_load_b64 v[190:191], v183 offset: 8')
    self.assertIn('offset0=8', dsl)

  def test_ds_load_offset_parsing_no_space(self):
    dsl = get_dsl('ds_load_b64 v[190:191], v183 offset:8')
    self.assertIn('offset0=8', dsl)

  def test_ds_2addr_offsets(self):
    # ds_store_2addr uses offset0: and offset1: separately
    dsl = get_dsl('ds_store_2addr_stride64_b32 v8, v23, v24 offset0:16 offset1:18')
    self.assertIn('offset0=16', dsl)
    self.assertIn('offset1=18', dsl)

  def test_ds_2addr_roundtrip(self):
    inst = ds_store_2addr_stride64_b32(addr=v[8], data0=v[23], data1=v[24], offset0=16, offset1=18)
    asm_str = inst.disasm()
    self.assertIn('offset0:16', asm_str)
    self.assertIn('offset1:18', asm_str)


class TestSOPPInstructions(unittest.TestCase):
  """Test SOPP instruction parsing."""

  def test_s_clause_parsing(self):
    dsl = get_dsl('s_clause 0x5')
    self.assertEqual(dsl, 's_clause(simm16=5)')

  def test_s_clause_roundtrip(self):
    inst = asm('s_clause 0x5')
    self.assertEqual(inst.disasm(), 's_clause 0x5')

  def test_s_delay_alu_parsing(self):
    dsl = get_dsl('s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)')
    # VALU_DEP_2=2, SKIP_3=4, VALU_DEP_3=3 -> simm16 = 2 | (4 << 4) | (3 << 7) = 450
    self.assertEqual(dsl, 's_delay_alu(simm16=450)')

  def test_s_delay_alu_roundtrip(self):
    inst = asm('s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)')
    self.assertEqual(inst.disasm(), 's_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)')

  def test_s_delay_alu_salu_cycle(self):
    dsl = get_dsl('s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_3)')
    # SALU_CYCLE_1=9, NEXT=1, VALU_DEP_3=3 -> simm16 = 9 | (1 << 4) | (3 << 7) = 409
    self.assertEqual(dsl, 's_delay_alu(simm16=409)')


class TestInstructionNameMapping(unittest.TestCase):
  """Test old instruction names are mapped to new RDNA3 names."""

  def test_v_add_u32_to_v_add_nc_u32(self):
    # Old name v_add_u32 should map to v_add_nc_u32
    dsl = get_dsl('v_add_u32_e32 v215, v1, v215')
    self.assertIn('v_add_nc_u32', dsl)

  def test_v_sub_u32_to_v_sub_nc_u32(self):
    dsl = get_dsl('v_sub_u32_e32 v5, v1, v2')
    self.assertIn('v_sub_nc_u32', dsl)


class TestKernel8Roundtrip(unittest.TestCase):
  """Test that kernel8_batched_gmem.s can be parsed and regenerated."""

  def test_kernel8_instructions_parse(self):
    """Test that all non-trivial instructions from kernel8 can be parsed."""
    test_instructions = [
      's_load_b128 s[20:23], s[0:1], 0x0',
      's_waitcnt lgkmcnt(0)',
      's_add_u32 s24, s22, 0x4000',
      's_addc_u32 s25, s23, 0',
      's_lshl_b32 s19, s14, 7',
      'v_add_nc_u32_e32 v203, s19, v0',
      'v_lshlrev_b32_e32 v203, 2, v203',
      'v_lshrrev_b32_e32 v1, 3, v0',
      'v_and_b32_e32 v215, 7, v0',
      's_mov_b32 s4, 0x1000',
      'v_add_co_u32 v5, vcc_lo, s10, v5',
      'v_add_co_ci_u32_e32 v6, vcc_lo, s11, v6, vcc_lo',
      'v_lshlrev_b64 v[5:6], 2, v[1:2]',
      'v_ashrrev_i32_e32 v4, 31, v3',
      'v_mul_lo_u32 v119, v22, s4',
      'global_load_b32 v23, v[5:6], off',
      'ds_load_b64 v[186:187], v183',
      'ds_load_b64 v[190:191], v183 offset:8',
      'ds_store_2addr_stride64_b32 v8, v23, v24 offset0:16 offset1:18',
      's_barrier',
      's_setprio 0',
      's_clause 0x5',
      's_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)',
      'v_dual_mov_b32 v2, s12 :: v_dual_mov_b32 v3, s12',
      'v_dual_fmac_f32 v5, v186, v184 :: v_dual_fmac_f32 v2, v187, v185',
    ]
    for instr in test_instructions:
      with self.subTest(instr=instr):
        try:
          inst = asm(instr)
          # Verify it produces valid bytes
          self.assertGreater(len(inst.to_bytes()), 0)
        except Exception as e:
          self.fail(f"Failed to parse '{instr}': {e}")

  def test_kernel8_critical_instructions_roundtrip(self):
    """Test roundtrip for critical instructions."""
    roundtrip_instructions = [
      's_load_b128 s[20:23], s[0:1], 0x0',
      's_waitcnt lgkmcnt(0)',
      'v_add_co_u32 v5, vcc_lo, s10, v5',
      'v_add_co_ci_u32_e32 v6, vcc_lo, s11, v6, vcc_lo',
      'ds_load_b64 v[186:187], v183 offset:8',
      's_clause 0x5',
      's_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)',
      's_barrier',
    ]
    for instr in roundtrip_instructions:
      with self.subTest(instr=instr):
        inst = asm(instr)
        result = inst.disasm()
        # Parse again and compare bytes
        inst2 = asm(result)
        self.assertEqual(inst.to_bytes(), inst2.to_bytes(),
                        f"Roundtrip failed: '{instr}' -> '{result}'")


class TestSendmsgInstruction(unittest.TestCase):
  """Test s_sendmsg instruction parsing and disassembly."""

  def test_sendmsg_dealloc_vgprs_parsing(self):
    dsl = get_dsl('s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)')
    self.assertEqual(dsl, 's_sendmsg(simm16=3)')

  def test_sendmsg_dealloc_vgprs_roundtrip(self):
    inst = asm('s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)')
    self.assertEqual(inst.disasm(), 's_sendmsg sendmsg(MSG_DEALLOC_VGPRS)')

  def test_sendmsg_binary_roundtrip(self):
    inst1 = asm('s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)')
    inst2 = asm(inst1.disasm())
    self.assertEqual(inst1.to_bytes(), inst2.to_bytes())

  def test_sendmsg_other_messages(self):
    # Test other message types
    for msg, simm16 in [('MSG_INTERRUPT', 1), ('MSG_GS', 2), ('MSG_SAVEWAVE', 4)]:
      with self.subTest(msg=msg):
        inst = asm(f's_sendmsg sendmsg({msg})')
        self.assertEqual(inst.to_bytes(), asm(f's_sendmsg 0x{simm16:x}').to_bytes())


class TestKernel8FullRoundtrip(unittest.TestCase):
  """Test full roundtrip of kernel8_batched_gmem.s assembly file."""

  KERNEL8_PATH = 'extra/gemm/amd_seb/kernel8_batched_gmem.s'

  # Instructions that can't be parsed directly (need label resolution)
  SKIP_PATTERNS = ['s_cbranch_', 's_branch ', '.LBB']

  @classmethod
  def setUpClass(cls):
    """Load and parse the kernel8 assembly file."""
    import os
    # Find the kernel file relative to repo root
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    cls.kernel_path = os.path.join(repo_root, cls.KERNEL8_PATH)
    if not os.path.exists(cls.kernel_path):
      raise unittest.SkipTest(f"Kernel file not found: {cls.kernel_path}")

    with open(cls.kernel_path) as f:
      cls.kernel_source = f.read()

    # Extract instruction lines
    cls.instructions = []
    for line in cls.kernel_source.splitlines():
      # Strip comments
      if ';' in line:
        line = line.split(';')[0]
      line = line.strip()
      # Skip empty lines, directives, labels, metadata
      if not line or line.startswith('.') or line.endswith(':'):
        continue
      if any(line.startswith(x) for x in ['---', 'amdhsa', '...', '-', 'kernel:']):
        continue
      # Skip unparseable instructions
      if any(p in line for p in cls.SKIP_PATTERNS):
        continue
      cls.instructions.append(line)

  def test_kernel8_all_instructions_parse(self):
    """Test that all instructions from kernel8 can be parsed."""
    failed = []
    for i, instr in enumerate(self.instructions):
      try:
        inst = asm(instr)
        self.assertIsNotNone(inst.to_bytes())
      except Exception as e:
        failed.append((i, instr, str(e)))

    if failed:
      msg = f"Failed to parse {len(failed)}/{len(self.instructions)} instructions:\n"
      for i, instr, err in failed[:10]:  # Show first 10 failures
        msg += f"  Line {i}: {instr[:60]}: {err[:40]}\n"
      if len(failed) > 10:
        msg += f"  ... and {len(failed) - 10} more\n"
      self.fail(msg)

  def test_kernel8_binary_roundtrip(self):
    """Test that all instructions produce identical bytes after roundtrip."""
    mismatches = []
    for i, instr in enumerate(self.instructions):
      try:
        inst1 = asm(instr)
        bytes1 = inst1.to_bytes()
        disasm1 = inst1.disasm()
        inst2 = asm(disasm1)
        bytes2 = inst2.to_bytes()
        if bytes1 != bytes2:
          mismatches.append((i, instr, disasm1, bytes1.hex(), bytes2.hex()))
      except Exception:
        pass  # Skip parse failures (tested separately)

    if mismatches:
      msg = f"Binary mismatch after roundtrip for {len(mismatches)} instructions:\n"
      for i, orig, disasm_str, b1, b2 in mismatches[:10]:
        msg += f"  {i}: '{orig[:40]}' -> '{disasm_str[:40]}'\n"
        msg += f"      bytes: {b1} != {b2}\n"
      self.fail(msg)

  def test_kernel8_instruction_count(self):
    """Verify we're testing a reasonable number of instructions."""
    # kernel8 has ~1992 instructions, we skip ~9 (branches + sendmsg)
    self.assertGreater(len(self.instructions), 1900,
                       f"Expected >1900 instructions, got {len(self.instructions)}")

  def test_kernel8_dsl_compiles_successfully(self):
    """Test that DSL-generated kernel compiles successfully.
    Note: The DSL kernel is simplified (removes unnecessary loads/FMACs since beta=0),
    so it won't match the reference binary exactly. Correctness is verified by amd_asm_matmul.py."""
    try:
      from tinygrad import Device
      dev = Device[Device.DEFAULT]
      if 'gfx1100' not in dev.arch:
        raise unittest.SkipTest("Test requires gfx1100 device")
    except Exception as e:
      raise unittest.SkipTest(f"Could not initialize device: {e}")

    # Compile DSL-generated kernel
    from extra.gemm.amd_asm_kernel_dsl import build_kernel
    dsl_asm = build_kernel('gfx1100')
    dsl_bin = dev.compiler.compile(dsl_asm)

    # Verify compilation produced a valid binary
    self.assertGreater(len(dsl_bin), 10000, "DSL kernel binary should be at least 10KB")
    self.assertLess(len(dsl_bin), 20000, "DSL kernel binary should be at most 20KB")


if __name__ == "__main__":
  unittest.main()
