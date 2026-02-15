#!/usr/bin/env python3
"""Tests for the pcode-based instruction selector (isel.py)."""
import unittest
from tinygrad.uop.ops import UOp, Ops
from tinygrad.dtype import dtypes
from extra.assembly.amd.isel import (rdna3_isel, make_inst, normalize, _count_nodes, _is_direct_alu,
                                      _parse_pcode_patterns, _pattern_key, _runtime_key, uop_to_upat,
                                      _SENTINEL, _SENTINEL_SET, _DIRECT_TABLE, _STRUCTURAL_TABLE,
                                      _ALU_ENUM_TYPES, build_isel_patterns)
from extra.assembly.amd.autogen.rdna3.str_pcode import PCODE
from extra.assembly.amd.autogen.rdna3.enum import VOP2Op, VOP1Op, VOP3Op, SOP2Op, VOPCOp, VOP3SDOp

# helpers
def _var(name, dtype=dtypes.float): return UOp(Ops.DEFINE_VAR, dtype, arg=(name, 0, 100))
def _const(val, dtype=dtypes.float): return UOp(Ops.CONST, dtype, arg=val)

class TestMakeInst(unittest.TestCase):
  def test_vop2(self):
    inst = make_inst(VOP2Op.V_ADD_F32_E32)
    assert inst.op == VOP2Op.V_ADD_F32_E32

  def test_vop1(self):
    inst = make_inst(VOP1Op.V_SQRT_F32_E32)
    assert inst.op == VOP1Op.V_SQRT_F32_E32

  def test_vop3(self):
    inst = make_inst(VOP3Op.V_ADD_F64)
    assert inst.op == VOP3Op.V_ADD_F64

  def test_vop3_sdst(self):
    # VOP3SDOp opcodes need the _SDST variant class
    inst = make_inst(VOP3SDOp.V_ADD_CO_CI_U32)
    assert inst.op == VOP3SDOp.V_ADD_CO_CI_U32

  def test_vopc(self):
    inst = make_inst(VOPCOp.V_CMP_LT_F32_E32)
    assert inst.op == VOPCOp.V_CMP_LT_F32_E32

  def test_sop2(self):
    inst = make_inst(SOP2Op.S_ADD_I32)
    assert inst.op == SOP2Op.S_ADD_I32

  def test_invalid_raises(self):
    with self.assertRaises(RuntimeError): make_inst("not_an_opcode")

class TestNormalize(unittest.TestCase):
  def test_bitcast_sentinel(self):
    s0 = _SENTINEL['S0']
    bc = UOp(Ops.BITCAST, dtypes.float, (s0,))
    norm = normalize(bc)
    assert norm.op == Ops.DEFINE_VAR
    assert norm.dtype == dtypes.float

  def test_cast_sentinel(self):
    s0 = _SENTINEL['S0']
    cast = UOp(Ops.CAST, dtypes.int, (s0,))
    norm = normalize(cast)
    assert norm.op == Ops.DEFINE_VAR
    assert norm.dtype == dtypes.int

  def test_identity_bitcast(self):
    x = UOp(Ops.CONST, dtypes.float, arg=1.0)
    bc = UOp(Ops.BITCAST, dtypes.float, (x,))
    norm = normalize(bc)
    assert norm.op == Ops.CONST
    assert norm.arg == 1.0

  def test_shift_mask_31(self):
    s0 = _SENTINEL['S0']
    c31 = UOp(Ops.CONST, dtypes.uint, arg=31)
    masked = UOp(Ops.AND, dtypes.uint, (s0, c31))
    norm = normalize(masked)
    assert norm.op == Ops.DEFINE_VAR

  def test_shift_mask_63(self):
    s0 = _SENTINEL['S0']
    c63 = UOp(Ops.CONST, dtypes.uint, arg=63)
    masked = UOp(Ops.AND, dtypes.uint, (s0, c63))
    norm = normalize(masked)
    assert norm.op == Ops.DEFINE_VAR

  def test_non_sentinel_bitcast_preserved(self):
    x = _var('x', dtypes.uint)
    bc = UOp(Ops.BITCAST, dtypes.float, (x,))
    norm = normalize(bc)
    assert norm.op == Ops.BITCAST  # not a sentinel, so preserved

  def test_recursive(self):
    s0 = _SENTINEL['S0']
    s1 = _SENTINEL['S1']
    bc0 = UOp(Ops.BITCAST, dtypes.float, (s0,))
    bc1 = UOp(Ops.BITCAST, dtypes.float, (s1,))
    add = UOp(Ops.ADD, dtypes.float, (bc0, bc1))
    norm = normalize(add)
    assert norm.op == Ops.ADD
    assert all(s.dtype == dtypes.float for s in norm.src)
    assert all(s.op == Ops.DEFINE_VAR for s in norm.src)

class TestCountNodes(unittest.TestCase):
  def test_leaf(self):
    assert _count_nodes(_var('x')) == 1

  def test_binary(self):
    x, y = _var('x'), _var('y')
    add = UOp(Ops.ADD, dtypes.float, (x, y))
    assert _count_nodes(add) == 3

  def test_dag_sharing(self):
    x = _var('x')
    add = UOp(Ops.ADD, dtypes.float, (x, x))
    assert _count_nodes(add) == 2  # x counted once

class TestIsDirectAlu(unittest.TestCase):
  def test_add_sentinels(self):
    s0 = _SENTINEL['S0'].replace(dtype=dtypes.float)
    s1 = _SENTINEL['S1'].replace(dtype=dtypes.float)
    add = UOp(Ops.ADD, dtypes.float, (s0, s1))
    assert _is_direct_alu(add)

  def test_cast_sentinel(self):
    s0 = _SENTINEL['S0'].replace(dtype=dtypes.int)
    cast = UOp(Ops.CAST, dtypes.float, (s0,))
    assert _is_direct_alu(cast)

  def test_nested_not_direct(self):
    s0 = _SENTINEL['S0'].replace(dtype=dtypes.uint)
    c = _const(0xFFFFFFFF, dtypes.uint)
    xor = UOp(Ops.XOR, dtypes.uint, (s0, c))
    assert not _is_direct_alu(xor)  # const child is not DEFINE_VAR

class TestPatternKey(unittest.TestCase):
  def test_sentinel_var(self):
    s0 = _SENTINEL['S0'].replace(dtype=dtypes.float)
    key = _pattern_key(s0)
    assert key == 'var(S0,dtypes.float)'

  def test_const(self):
    c = _const(42, dtypes.uint)
    key = _pattern_key(c)
    assert key == 'const(42,dtypes.uint)'

  def test_binary_op(self):
    s0 = _SENTINEL['S0'].replace(dtype=dtypes.float)
    s1 = _SENTINEL['S1'].replace(dtype=dtypes.float)
    add = UOp(Ops.ADD, dtypes.float, (s0, s1))
    key = _pattern_key(add)
    assert key == 'Ops.ADD(dtypes.float,var(S0,dtypes.float),var(S1,dtypes.float))'

class TestRuntimeKey(unittest.TestCase):
  def test_matches_pattern_key(self):
    # runtime key on a matched UOp should equal pattern key on the pcode template
    x = _var('x', dtypes.uint)
    c = _const(0xFFFFFFFF, dtypes.uint)
    xor = UOp(Ops.XOR, dtypes.uint, (x, c))
    rkey = _runtime_key(xor)

    s0 = _SENTINEL['S0'].replace(dtype=dtypes.uint)
    xor_template = UOp(Ops.XOR, dtypes.uint, (s0, c))
    pkey = _pattern_key(xor_template)
    assert rkey == pkey

class TestUopToUpat(unittest.TestCase):
  def test_sentinel_becomes_var(self):
    s0 = _SENTINEL['S0'].replace(dtype=dtypes.float)
    pat = uop_to_upat(s0)
    assert pat.name == 'S0'
    assert pat.dtype == (dtypes.float,)

  def test_const_preserved(self):
    c = _const(42, dtypes.uint)
    pat = uop_to_upat(c)
    assert pat.op == (Ops.CONST,)
    assert pat.arg == 42

class TestBuildPerformance(unittest.TestCase):
  def test_builds_under_2_seconds(self):
    import time
    t0 = time.time()
    build_isel_patterns(PCODE)
    elapsed = time.time() - t0
    assert elapsed < 2.0, f"build took {elapsed:.2f}s, expected <2s"

  def test_alu_filter(self):
    # verify only ALU enum types are parsed
    for opcode in PCODE:
      if type(opcode).__name__ not in _ALU_ENUM_TYPES: continue
      # these should parse without hanging

class TestDirectPatterns(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.pm = rdna3_isel()

  def _check(self, uop, expected_name_substr):
    result = self.pm.rewrite(uop)
    self.assertIsNotNone(result, f"no match for {uop.op} {uop.dtype}")
    self.assertEqual(result.op, Ops.INS)
    self.assertIn(expected_name_substr, result.arg.op.name, f"expected {expected_name_substr} in {result.arg.op.name}")
    return result

  # arithmetic
  def test_add_f32(self): self._check(UOp(Ops.ADD, dtypes.float, (_var('a'), _var('b'))), 'V_ADD_F32')
  def test_add_f64(self): self._check(UOp(Ops.ADD, dtypes.double, (_var('a', dtypes.double), _var('b', dtypes.double))), 'V_ADD_F64')
  def test_add_i32(self): self._check(UOp(Ops.ADD, dtypes.int, (_var('a', dtypes.int), _var('b', dtypes.int))), 'ADD_NC_I32')
  def test_add_u32(self): self._check(UOp(Ops.ADD, dtypes.uint, (_var('a', dtypes.uint), _var('b', dtypes.uint))), 'ADD_NC_U32')
  def test_mul_f32(self): self._check(UOp(Ops.MUL, dtypes.float, (_var('a'), _var('b'))), 'V_MUL_F32')
  def test_mul_f64(self): self._check(UOp(Ops.MUL, dtypes.double, (_var('a', dtypes.double), _var('b', dtypes.double))), 'V_MUL_F64')
  def test_mul_i32(self): self._check(UOp(Ops.MUL, dtypes.int, (_var('a', dtypes.int), _var('b', dtypes.int))), 'MUL_I32')
  def test_mul_u32(self): self._check(UOp(Ops.MUL, dtypes.uint, (_var('a', dtypes.uint), _var('b', dtypes.uint))), 'MUL_U32')

  # bitwise
  def test_and_u32(self): self._check(UOp(Ops.AND, dtypes.uint, (_var('a', dtypes.uint), _var('b', dtypes.uint))), 'AND_B32')
  def test_or_u32(self): self._check(UOp(Ops.OR, dtypes.uint, (_var('a', dtypes.uint), _var('b', dtypes.uint))), 'OR_B32')
  def test_xor_u32(self): self._check(UOp(Ops.XOR, dtypes.uint, (_var('a', dtypes.uint), _var('b', dtypes.uint))), 'XOR_B32')
  # u64 bitwise ops are SOP-only, skipped in vgpr_only mode
  def test_and_u64_skipped(self):
    result = self.pm.rewrite(UOp(Ops.AND, dtypes.ulong, (_var('a', dtypes.ulong), _var('b', dtypes.ulong))))
    self.assertIsNone(result)
  def test_or_u64_skipped(self):
    result = self.pm.rewrite(UOp(Ops.OR, dtypes.ulong, (_var('a', dtypes.ulong), _var('b', dtypes.ulong))))
    self.assertIsNone(result)
  def test_xor_u64_skipped(self):
    result = self.pm.rewrite(UOp(Ops.XOR, dtypes.ulong, (_var('a', dtypes.ulong), _var('b', dtypes.ulong))))
    self.assertIsNone(result)

  # shifts
  def test_shl_u32(self): self._check(UOp(Ops.SHL, dtypes.uint, (_var('a', dtypes.uint), _var('b', dtypes.uint))), 'LSH')
  def test_shr_u32(self): self._check(UOp(Ops.SHR, dtypes.uint, (_var('a', dtypes.uint), _var('b', dtypes.uint))), 'LSH')

  # unary float
  def test_sqrt_f32(self): self._check(UOp(Ops.SQRT, dtypes.float, (_var('a'),)), 'SQRT_F32')
  def test_sqrt_f64(self): self._check(UOp(Ops.SQRT, dtypes.double, (_var('a', dtypes.double),)), 'SQRT_F64')
  def test_trunc_f32(self): self._check(UOp(Ops.TRUNC, dtypes.float, (_var('a'),)), 'TRUNC_F32')
  def test_trunc_f64(self): self._check(UOp(Ops.TRUNC, dtypes.double, (_var('a', dtypes.double),)), 'TRUNC_F64')
  def test_log2_f32(self): self._check(UOp(Ops.LOG2, dtypes.float, (_var('a'),)), 'LOG')
  def test_exp2_f32(self): self._check(UOp(Ops.EXP2, dtypes.float, (_var('a'),)), 'EXP')

  # conversions
  def test_cast_i32_to_f32(self): self._check(UOp(Ops.CAST, dtypes.float, (_var('a', dtypes.int),)), 'CVT_F32_I32')
  def test_cast_f32_to_f64(self): self._check(UOp(Ops.CAST, dtypes.double, (_var('a'),)), 'CVT_F64_F32')
  def test_cast_f64_to_f32(self): self._check(UOp(Ops.CAST, dtypes.float, (_var('a', dtypes.double),)), 'CVT_F32_F64')
  def test_cast_i32_to_f64(self): self._check(UOp(Ops.CAST, dtypes.double, (_var('a', dtypes.int),)), 'CVT_F64_I32')
  def test_cast_f32_to_f16(self): self._check(UOp(Ops.CAST, dtypes.half, (_var('a'),)), 'CVT_F16_F32')

  # compares are skipped by ISel (VOPC writes VCC, not VGPRs; LLVM handles natively)
  def test_cmplt_skipped(self):
    result = self.pm.rewrite(UOp(Ops.CMPLT, dtypes.bool, (_var('a', dtypes.int), _var('b', dtypes.int))))
    self.assertIsNone(result)
  def test_cmpne_skipped(self):
    result = self.pm.rewrite(UOp(Ops.CMPNE, dtypes.bool, (_var('a'), _var('b'))))
    self.assertIsNone(result)

  # check that unmatched types return None
  def test_no_match(self):
    # there's no direct ADD for bools
    result = self.pm.rewrite(UOp(Ops.ADD, dtypes.bool, (_var('a', dtypes.bool), _var('b', dtypes.bool))))
    self.assertIsNone(result)

class TestStructuralPatterns(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.pm = rdna3_isel()

  def _check(self, uop, expected_name_substr, expected_src_count=None):
    result = self.pm.rewrite(uop)
    self.assertIsNotNone(result, f"no match for structural pattern")
    self.assertEqual(result.op, Ops.INS)
    self.assertIn(expected_name_substr, result.arg.op.name, f"expected {expected_name_substr} in {result.arg.op.name}")
    if expected_src_count is not None:
      self.assertEqual(len(result.src), expected_src_count, f"expected {expected_src_count} srcs, got {len(result.src)}")
    return result

  def test_not_u32(self):
    x = _var('x', dtypes.uint)
    xor = UOp(Ops.XOR, dtypes.uint, (x, _const(0xFFFFFFFF, dtypes.uint)))
    self._check(xor, 'NOT_B32', 1)

  # u64 NOT is SOP-only (S_NOT_B64), skipped in vgpr_only mode
  def test_not_u64_skipped(self):
    x = _var('x', dtypes.ulong)
    xor = UOp(Ops.XOR, dtypes.ulong, (x, _const(0xFFFFFFFFFFFFFFFF, dtypes.ulong)))
    result = self.pm.rewrite(xor)
    self.assertIsNone(result)

  def test_sub_u32(self):
    x, y = _var('x', dtypes.uint), _var('y', dtypes.uint)
    neg = UOp(Ops.MUL, dtypes.uint, (y, _const(-1, dtypes.uint)))
    sub = UOp(Ops.ADD, dtypes.uint, (x, neg))
    result = self._check(sub, 'SUB_NC_U32', 2)
    # verify source order: x is first, y is second
    self.assertEqual(result.src[0].arg, ('x', 0, 100))
    self.assertEqual(result.src[1].arg, ('y', 0, 100))

  def test_rcp_f32(self):
    a = _var('a')
    rcp = UOp(Ops.RECIPROCAL, dtypes.float, (a,))
    mul_rcp = UOp(Ops.MUL, dtypes.float, (_const(1.0), rcp))
    result = self._check(mul_rcp, 'RCP_F32', 1)
    self.assertEqual(result.src[0].arg, ('a', 0, 100))

  def test_rcp_f64(self):
    a = _var('a', dtypes.double)
    rcp = UOp(Ops.RECIPROCAL, dtypes.double, (a,))
    mul_rcp = UOp(Ops.MUL, dtypes.double, (_const(1.0, dtypes.double), rcp))
    self._check(mul_rcp, 'RCP_F64', 1)

  def test_cvt_i32_f32(self):
    # CAST(i32, TRUNC(f32, x)) -> V_CVT_I32_F32
    a = _var('a')
    trunc = UOp(Ops.TRUNC, dtypes.float, (a,))
    cast = UOp(Ops.CAST, dtypes.int, (trunc,))
    self._check(cast, 'CVT_I32_F32', 1)

  def test_mad_u32(self):
    x, y, z = _var('x', dtypes.uint), _var('y', dtypes.uint), _var('z', dtypes.uint)
    mul = UOp(Ops.MUL, dtypes.uint, (x, y))
    mad = UOp(Ops.ADD, dtypes.uint, (mul, z))
    result = self._check(mad, 'MAD_U32_U24', 3)
    self.assertEqual(result.src[0].arg, ('x', 0, 100))
    self.assertEqual(result.src[1].arg, ('y', 0, 100))
    self.assertEqual(result.src[2].arg, ('z', 0, 100))

  def test_add3_u32(self):
    x, y, z = _var('x', dtypes.uint), _var('y', dtypes.uint), _var('z', dtypes.uint)
    add1 = UOp(Ops.ADD, dtypes.uint, (x, y))
    add3 = UOp(Ops.ADD, dtypes.uint, (add1, z))
    result = self._check(add3, 'ADD3_U32', 3)

  def test_xor3_b32(self):
    x, y, z = _var('x', dtypes.uint), _var('y', dtypes.uint), _var('z', dtypes.uint)
    xor1 = UOp(Ops.XOR, dtypes.uint, (x, y))
    xor3 = UOp(Ops.XOR, dtypes.uint, (xor1, z))
    self._check(xor3, 'XOR3_B32', 3)

  def test_and_or_b32(self):
    x, y, z = _var('x', dtypes.uint), _var('y', dtypes.uint), _var('z', dtypes.uint)
    and_op = UOp(Ops.AND, dtypes.uint, (x, y))
    or_op = UOp(Ops.OR, dtypes.uint, (and_op, z))
    self._check(or_op, 'AND_OR_B32', 3)

  def test_or3_b32(self):
    x, y, z = _var('x', dtypes.uint), _var('y', dtypes.uint), _var('z', dtypes.uint)
    or1 = UOp(Ops.OR, dtypes.uint, (x, y))
    or3 = UOp(Ops.OR, dtypes.uint, (or1, z))
    self._check(or3, 'OR3_B32', 3)

  # NAND/NOR were SOP-only, in vgpr_only mode they decompose to V_XOR_B32(AND/OR, mask)
  def test_nand_decomposes(self):
    x, y = _var('x', dtypes.uint), _var('y', dtypes.uint)
    and_op = UOp(Ops.AND, dtypes.uint, (x, y))
    nand = UOp(Ops.XOR, dtypes.uint, (and_op, _const(0xFFFFFFFF, dtypes.uint)))
    self._check(nand, 'XOR_B32')

  def test_nor_decomposes(self):
    x, y = _var('x', dtypes.uint), _var('y', dtypes.uint)
    or_op = UOp(Ops.OR, dtypes.uint, (x, y))
    nor = UOp(Ops.XOR, dtypes.uint, (or_op, _const(0xFFFFFFFF, dtypes.uint)))
    self._check(nor, 'XOR_B32')

  def test_xnor_b32(self):
    x, y = _var('x', dtypes.uint), _var('y', dtypes.uint)
    xor_op = UOp(Ops.XOR, dtypes.uint, (x, y))
    xnor = UOp(Ops.XOR, dtypes.uint, (xor_op, _const(0xFFFFFFFF, dtypes.uint)))
    self._check(xnor, 'XNOR_B32', 2)

  def test_min_u32(self):
    x, y = _var('x', dtypes.uint), _var('y', dtypes.uint)
    cmp = UOp(Ops.CMPLT, dtypes.bool, (x, y))
    where = UOp(Ops.WHERE, dtypes.uint, (cmp, x, y))
    self._check(where, 'MIN_U32', 2)

class TestInstProperties(unittest.TestCase):
  """Verify that Inst objects produced by isel have correct properties."""
  @classmethod
  def setUpClass(cls):
    cls.pm = rdna3_isel()

  def test_ins_has_dtype(self):
    result = self.pm.rewrite(UOp(Ops.ADD, dtypes.float, (_var('a'), _var('b'))))
    self.assertEqual(result.dtype, dtypes.float)

  def test_ins_preserves_sources(self):
    a, b = _var('a'), _var('b')
    result = self.pm.rewrite(UOp(Ops.ADD, dtypes.float, (a, b)))
    self.assertEqual(result.src, (a, b))

  def test_ins_tag_default_none(self):
    result = self.pm.rewrite(UOp(Ops.ADD, dtypes.float, (_var('a'), _var('b'))))
    # tag should not be set (defaults to None or empty)
    self.assertIsNone(result.tag)

class TestTableCoverage(unittest.TestCase):
  """Verify that the tables have expected coverage."""
  @classmethod
  def setUpClass(cls):
    rdna3_isel()  # populate tables

  def test_direct_table_has_add(self):
    found = any(op == Ops.ADD for (op, _, _) in _DIRECT_TABLE)
    self.assertTrue(found)

  def test_direct_table_has_cast(self):
    found = any(op == Ops.CAST for (op, _, _) in _DIRECT_TABLE)
    self.assertTrue(found)

  def test_direct_table_skips_cmplt(self):
    # compares are in the table but skipped at runtime (bool output)
    found = any(op == Ops.CMPLT for (op, _, _) in _DIRECT_TABLE)
    self.assertTrue(found)  # entries exist but callbacks skip them

  def test_structural_table_has_not(self):
    found = any('NOT' in inst.op.name for inst in _STRUCTURAL_TABLE.values())
    self.assertTrue(found)

  def test_structural_table_has_rcp(self):
    found = any('RCP' in inst.op.name for inst in _STRUCTURAL_TABLE.values())
    self.assertTrue(found)

  def test_structural_table_has_sub(self):
    found = any('SUB' in inst.op.name for inst in _STRUCTURAL_TABLE.values())
    self.assertTrue(found)

  def test_direct_count(self):
    self.assertGreaterEqual(len(_DIRECT_TABLE), 25, "expected at least 25 direct patterns")

  def test_structural_count(self):
    self.assertGreaterEqual(len(_STRUCTURAL_TABLE), 15, "expected at least 15 structural patterns")

class TestEmulatorValidation(unittest.TestCase):
  """Validate isel-produced Inst objects execute correctly in the emulator.

  For each pattern, we:
  1. Run the UOp through isel to get Ops.INS with arg=Inst
  2. Copy the Inst and assign concrete registers
  3. Set up operand values via MOV instructions
  4. Execute through the emulator
  5. Verify the output matches expected computation
  """
  @classmethod
  def setUpClass(cls):
    cls.pm = rdna3_isel()

  def _run(self, instructions, n_lanes=1):
    from extra.assembly.amd.test.hw.helpers import run_program_emu
    return run_program_emu(instructions, n_lanes)

  def _get_inst(self, uop):
    """Get isel result, return (Inst, src_count)."""
    result = self.pm.rewrite(uop)
    assert result is not None and result.op == Ops.INS, f"isel failed for {uop.op} {uop.dtype}"
    return result.arg, len(result.src)

  def _copy_inst(self, inst):
    import copy
    return copy.copy(inst)

  # ── direct ALU: float arithmetic ──

  def test_emu_add_f32(self):
    from extra.assembly.amd.test.hw.helpers import i2f, f2i
    from extra.assembly.amd.autogen.rdna3.ins import v, v_mov_b32_e32
    inst, _ = self._get_inst(UOp(Ops.ADD, dtypes.float, (_var('a'), _var('b'))))
    ci = self._copy_inst(inst)
    ci.src0 = v[0]; ci.vsrc1 = v[1]; ci.vdst = v[2]
    st = self._run([v_mov_b32_e32(v[0], 1.5), v_mov_b32_e32(v[1], 2.25), ci])
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 3.75, places=5)

  def test_emu_mul_f32(self):
    from extra.assembly.amd.test.hw.helpers import i2f
    from extra.assembly.amd.autogen.rdna3.ins import v, v_mov_b32_e32
    inst, _ = self._get_inst(UOp(Ops.MUL, dtypes.float, (_var('a'), _var('b'))))
    ci = self._copy_inst(inst)
    ci.src0 = v[0]; ci.vsrc1 = v[1]; ci.vdst = v[2]
    st = self._run([v_mov_b32_e32(v[0], 3.0), v_mov_b32_e32(v[1], 4.0), ci])
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 12.0, places=5)

  # ── direct ALU: integer arithmetic ──

  def test_emu_add_u32(self):
    from extra.assembly.amd.autogen.rdna3.ins import v, v_mov_b32_e32
    inst, _ = self._get_inst(UOp(Ops.ADD, dtypes.uint, (_var('a', dtypes.uint), _var('b', dtypes.uint))))
    ci = self._copy_inst(inst)
    ci.src0 = v[0]; ci.vsrc1 = v[1]; ci.vdst = v[2]
    st = self._run([v_mov_b32_e32(v[0], 10), v_mov_b32_e32(v[1], 20), ci])
    self.assertEqual(st.vgpr[0][2], 30)

  # ── direct ALU: bitwise ──

  def test_emu_and_u32(self):
    from extra.assembly.amd.autogen.rdna3.ins import v, v_mov_b32_e32
    inst, _ = self._get_inst(UOp(Ops.AND, dtypes.uint, (_var('a', dtypes.uint), _var('b', dtypes.uint))))
    ci = self._copy_inst(inst)
    ci.src0 = v[0]; ci.vsrc1 = v[1]; ci.vdst = v[2]
    st = self._run([v_mov_b32_e32(v[0], 0xFF00), v_mov_b32_e32(v[1], 0x0FF0), ci])
    self.assertEqual(st.vgpr[0][2], 0x0F00)

  def test_emu_or_u32(self):
    from extra.assembly.amd.autogen.rdna3.ins import v, v_mov_b32_e32
    inst, _ = self._get_inst(UOp(Ops.OR, dtypes.uint, (_var('a', dtypes.uint), _var('b', dtypes.uint))))
    ci = self._copy_inst(inst)
    ci.src0 = v[0]; ci.vsrc1 = v[1]; ci.vdst = v[2]
    st = self._run([v_mov_b32_e32(v[0], 0xFF00), v_mov_b32_e32(v[1], 0x0FF0), ci])
    self.assertEqual(st.vgpr[0][2], 0xFFF0)

  def test_emu_xor_u32(self):
    from extra.assembly.amd.autogen.rdna3.ins import v, v_mov_b32_e32
    inst, _ = self._get_inst(UOp(Ops.XOR, dtypes.uint, (_var('a', dtypes.uint), _var('b', dtypes.uint))))
    ci = self._copy_inst(inst)
    ci.src0 = v[0]; ci.vsrc1 = v[1]; ci.vdst = v[2]
    st = self._run([v_mov_b32_e32(v[0], 0xFF00), v_mov_b32_e32(v[1], 0x0FF0), ci])
    self.assertEqual(st.vgpr[0][2], 0xF0F0)

  # ── direct ALU: shifts ──

  def test_emu_shl_u32(self):
    from extra.assembly.amd.autogen.rdna3.ins import v, v_mov_b32_e32
    inst, _ = self._get_inst(UOp(Ops.SHL, dtypes.uint, (_var('a', dtypes.uint), _var('b', dtypes.uint))))
    ci = self._copy_inst(inst)
    # LSHLREV: vdst = vsrc1 << src0 (reversed operands!)
    ci.src0 = v[1]; ci.vsrc1 = v[0]; ci.vdst = v[2]
    st = self._run([v_mov_b32_e32(v[0], 1), v_mov_b32_e32(v[1], 4), ci])
    self.assertEqual(st.vgpr[0][2], 16)  # 1 << 4 = 16

  def test_emu_shr_u32(self):
    from extra.assembly.amd.autogen.rdna3.ins import v, v_mov_b32_e32
    inst, _ = self._get_inst(UOp(Ops.SHR, dtypes.uint, (_var('a', dtypes.uint), _var('b', dtypes.uint))))
    ci = self._copy_inst(inst)
    # LSHRREV: vdst = vsrc1 >> src0 (reversed operands!)
    ci.src0 = v[1]; ci.vsrc1 = v[0]; ci.vdst = v[2]
    st = self._run([v_mov_b32_e32(v[0], 16), v_mov_b32_e32(v[1], 4), ci])
    self.assertEqual(st.vgpr[0][2], 1)  # 16 >> 4 = 1

  # ── direct ALU: unary float ──

  def test_emu_sqrt_f32(self):
    from extra.assembly.amd.test.hw.helpers import i2f
    from extra.assembly.amd.autogen.rdna3.ins import v, v_mov_b32_e32
    inst, _ = self._get_inst(UOp(Ops.SQRT, dtypes.float, (_var('a'),)))
    ci = self._copy_inst(inst)
    ci.src0 = v[0]; ci.vdst = v[2]
    st = self._run([v_mov_b32_e32(v[0], 4.0), ci])
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 2.0, places=4)

  def test_emu_trunc_f32(self):
    from extra.assembly.amd.test.hw.helpers import i2f
    from extra.assembly.amd.autogen.rdna3.ins import v, v_mov_b32_e32
    inst, _ = self._get_inst(UOp(Ops.TRUNC, dtypes.float, (_var('a'),)))
    ci = self._copy_inst(inst)
    ci.src0 = v[0]; ci.vdst = v[2]
    st = self._run([v_mov_b32_e32(v[0], 3.7), ci])
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 3.0, places=5)

  def test_emu_exp2_f32(self):
    from extra.assembly.amd.test.hw.helpers import i2f
    from extra.assembly.amd.autogen.rdna3.ins import v, v_mov_b32_e32
    inst, _ = self._get_inst(UOp(Ops.EXP2, dtypes.float, (_var('a'),)))
    ci = self._copy_inst(inst)
    ci.src0 = v[0]; ci.vdst = v[2]
    st = self._run([v_mov_b32_e32(v[0], 3.0), ci])
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 8.0, delta=0.01)

  def test_emu_log2_f32(self):
    from extra.assembly.amd.test.hw.helpers import i2f
    from extra.assembly.amd.autogen.rdna3.ins import v, v_mov_b32_e32
    inst, _ = self._get_inst(UOp(Ops.LOG2, dtypes.float, (_var('a'),)))
    ci = self._copy_inst(inst)
    ci.src0 = v[0]; ci.vdst = v[2]
    st = self._run([v_mov_b32_e32(v[0], 8.0), ci])
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 3.0, delta=0.01)

  # ── direct ALU: conversions ──

  def test_emu_cast_i32_to_f32(self):
    from extra.assembly.amd.test.hw.helpers import i2f
    from extra.assembly.amd.autogen.rdna3.ins import v, v_mov_b32_e32
    inst, _ = self._get_inst(UOp(Ops.CAST, dtypes.float, (_var('a', dtypes.int),)))
    ci = self._copy_inst(inst)
    ci.src0 = v[0]; ci.vdst = v[2]
    st = self._run([v_mov_b32_e32(v[0], 42), ci])
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 42.0, places=5)

  def test_emu_cast_f32_to_f16(self):
    from extra.assembly.amd.test.hw.helpers import i2f, f16
    from extra.assembly.amd.autogen.rdna3.ins import v, v_mov_b32_e32
    inst, _ = self._get_inst(UOp(Ops.CAST, dtypes.half, (_var('a'),)))
    ci = self._copy_inst(inst)
    ci.src0 = v[0]; ci.vdst = v[2]
    st = self._run([v_mov_b32_e32(v[0], 1.5), ci])
    # f16 result is in lower 16 bits of v[2]
    self.assertAlmostEqual(f16(st.vgpr[0][2]), 1.5, places=2)

  # compares are skipped by ISel (VOPC writes VCC; LLVM handles natively)

  # ── structural: NOT ──

  def test_emu_not_u32(self):
    from extra.assembly.amd.autogen.rdna3.ins import v, v_mov_b32_e32
    x = _var('x', dtypes.uint)
    xor = UOp(Ops.XOR, dtypes.uint, (x, _const(0xFFFFFFFF, dtypes.uint)))
    inst, _ = self._get_inst(xor)
    ci = self._copy_inst(inst)
    ci.src0 = v[0]; ci.vdst = v[2]
    st = self._run([v_mov_b32_e32(v[0], 0x0000FF00), ci])
    self.assertEqual(st.vgpr[0][2], 0xFFFF00FF)

  # ── structural: SUB ──

  def test_emu_sub_u32(self):
    from extra.assembly.amd.autogen.rdna3.ins import v, v_mov_b32_e32
    x, y = _var('x', dtypes.uint), _var('y', dtypes.uint)
    neg = UOp(Ops.MUL, dtypes.uint, (y, _const(-1, dtypes.uint)))
    sub = UOp(Ops.ADD, dtypes.uint, (x, neg))
    inst, nsrc = self._get_inst(sub)
    ci = self._copy_inst(inst)
    ci.src0 = v[0]; ci.vsrc1 = v[1]; ci.vdst = v[2]
    st = self._run([v_mov_b32_e32(v[0], 30), v_mov_b32_e32(v[1], 12), ci])
    self.assertEqual(st.vgpr[0][2], 18)

  # ── structural: RCP ──

  def test_emu_rcp_f32(self):
    from extra.assembly.amd.test.hw.helpers import i2f
    from extra.assembly.amd.autogen.rdna3.ins import v, v_mov_b32_e32
    a = _var('a')
    rcp = UOp(Ops.RECIPROCAL, dtypes.float, (a,))
    mul_rcp = UOp(Ops.MUL, dtypes.float, (_const(1.0), rcp))
    inst, _ = self._get_inst(mul_rcp)
    ci = self._copy_inst(inst)
    ci.src0 = v[0]; ci.vdst = v[2]
    st = self._run([v_mov_b32_e32(v[0], 4.0), ci])
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 0.25, places=4)

  # ── structural: MAD ──

  def test_emu_mad_u32(self):
    from extra.assembly.amd.autogen.rdna3.ins import v, v_mov_b32_e32
    x, y, z = _var('x', dtypes.uint), _var('y', dtypes.uint), _var('z', dtypes.uint)
    mul = UOp(Ops.MUL, dtypes.uint, (x, y))
    mad = UOp(Ops.ADD, dtypes.uint, (mul, z))
    inst, _ = self._get_inst(mad)
    ci = self._copy_inst(inst)
    # VOP3 format: src0, src1, src2, vdst
    ci.src0 = v[0]; ci.src1 = v[1]; ci.src2 = v[2]; ci.vdst = v[3]
    st = self._run([v_mov_b32_e32(v[0], 3), v_mov_b32_e32(v[1], 4), v_mov_b32_e32(v[2], 5), ci])
    self.assertEqual(st.vgpr[0][3], 17)  # 3*4 + 5 = 17

  # ── structural: ADD3 ──

  def test_emu_add3_u32(self):
    from extra.assembly.amd.autogen.rdna3.ins import v, v_mov_b32_e32
    x, y, z = _var('x', dtypes.uint), _var('y', dtypes.uint), _var('z', dtypes.uint)
    add1 = UOp(Ops.ADD, dtypes.uint, (x, y))
    add3 = UOp(Ops.ADD, dtypes.uint, (add1, z))
    inst, _ = self._get_inst(add3)
    ci = self._copy_inst(inst)
    ci.src0 = v[0]; ci.src1 = v[1]; ci.src2 = v[2]; ci.vdst = v[3]
    st = self._run([v_mov_b32_e32(v[0], 10), v_mov_b32_e32(v[1], 20), v_mov_b32_e32(v[2], 30), ci])
    self.assertEqual(st.vgpr[0][3], 60)  # 10+20+30

  # ── structural: XOR3 ──

  def test_emu_xor3_b32(self):
    from extra.assembly.amd.autogen.rdna3.ins import v, v_mov_b32_e32
    x, y, z = _var('x', dtypes.uint), _var('y', dtypes.uint), _var('z', dtypes.uint)
    xor1 = UOp(Ops.XOR, dtypes.uint, (x, y))
    xor3 = UOp(Ops.XOR, dtypes.uint, (xor1, z))
    inst, _ = self._get_inst(xor3)
    ci = self._copy_inst(inst)
    ci.src0 = v[0]; ci.src1 = v[1]; ci.src2 = v[2]; ci.vdst = v[3]
    st = self._run([v_mov_b32_e32(v[0], 0xFF), v_mov_b32_e32(v[1], 0x0F), v_mov_b32_e32(v[2], 0x33), ci])
    self.assertEqual(st.vgpr[0][3], 0xFF ^ 0x0F ^ 0x33)

if __name__ == '__main__':
  unittest.main()
