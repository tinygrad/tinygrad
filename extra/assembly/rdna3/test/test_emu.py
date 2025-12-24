# Unit tests for RDNA3 Python emulator
import unittest
import ctypes
import struct
import math
from extra.assembly.rdna3.emu import (
  WaveState, decode_program, exec_wave, exec_workgroup, run_asm,
  i32, f32, sext, WAVE_SIZE, set_valid_mem_ranges
)
from extra.assembly.rdna3.autogen import *

def run_kernel(kernel: bytes, n_threads: int = 1, n_outputs: int = 1) -> list[int]:
  """Helper to run a kernel and return output values."""
  output = (ctypes.c_uint32 * (n_threads * n_outputs))(*[0xdead] * (n_threads * n_outputs))
  output_ptr = ctypes.addressof(output)
  args = (ctypes.c_uint64 * 1)(output_ptr)
  args_ptr = ctypes.addressof(args)
  kernel_buf = (ctypes.c_char * len(kernel))(*kernel)
  kernel_ptr = ctypes.addressof(kernel_buf)
  # Register valid memory ranges for bounds checking
  set_valid_mem_ranges({
    (output_ptr, ctypes.sizeof(output)),
    (args_ptr, ctypes.sizeof(args)),
    (kernel_ptr, len(kernel)),
  })
  result = run_asm(kernel_ptr, len(kernel), 1, 1, 1, n_threads, 1, 1, args_ptr)
  assert result == 0, f"run_asm failed with {result}"
  return [output[i] for i in range(n_threads * n_outputs)]

def make_store_kernel(setup_instrs: list, store_vreg: int = 1) -> bytes:
  """Create a kernel that runs setup instructions then stores v[store_vreg] to output[tid]."""
  kernel = b''
  # Load output pointer
  kernel += s_load_b64(s[2:3], s[0:1], soffset=NULL, offset=0).to_bytes()
  kernel += s_waitcnt(lgkmcnt=0).to_bytes()
  # Run setup instructions
  for instr in setup_instrs:
    kernel += instr.to_bytes()
  # Compute offset: v3 = tid * 4
  kernel += v_lshlrev_b32_e32(v[3], 2, v[0]).to_bytes()
  # Store result
  kernel += global_store_b32(addr=v[3], data=v[store_vreg], saddr=s[2]).to_bytes()
  kernel += s_endpgm().to_bytes()
  return kernel

class TestScalarOps(unittest.TestCase):
  def test_s_mov_b32(self):
    state = WaveState()
    kernel = s_mov_b32(s[5], 42).to_bytes() + s_endpgm().to_bytes()
    prog = decode_program(kernel)
    exec_wave(prog, state, bytearray(65536), 1)
    self.assertEqual(state.sgpr[5], 42)

  def test_s_add_u32(self):
    state = WaveState()
    state.sgpr[0], state.sgpr[1] = 100, 50
    kernel = s_add_u32(s[2], s[0], s[1]).to_bytes() + s_endpgm().to_bytes()
    prog = decode_program(kernel)
    exec_wave(prog, state, bytearray(65536), 1)
    self.assertEqual(state.sgpr[2], 150)
    self.assertEqual(state.scc, 0)  # no carry

  def test_s_add_u32_carry(self):
    state = WaveState()
    state.sgpr[0], state.sgpr[1] = 0xffffffff, 1
    kernel = s_add_u32(s[2], s[0], s[1]).to_bytes() + s_endpgm().to_bytes()
    prog = decode_program(kernel)
    exec_wave(prog, state, bytearray(65536), 1)
    self.assertEqual(state.sgpr[2], 0)
    self.assertEqual(state.scc, 1)  # carry

  def test_s_sub_u32(self):
    state = WaveState()
    state.sgpr[0], state.sgpr[1] = 100, 30
    kernel = s_sub_u32(s[2], s[0], s[1]).to_bytes() + s_endpgm().to_bytes()
    prog = decode_program(kernel)
    exec_wave(prog, state, bytearray(65536), 1)
    self.assertEqual(state.sgpr[2], 70)
    self.assertEqual(state.scc, 0)  # no borrow

  def test_s_and_b32(self):
    state = WaveState()
    state.sgpr[0], state.sgpr[1] = 0xff00, 0x0ff0
    kernel = s_and_b32(s[2], s[0], s[1]).to_bytes() + s_endpgm().to_bytes()
    prog = decode_program(kernel)
    exec_wave(prog, state, bytearray(65536), 1)
    self.assertEqual(state.sgpr[2], 0x0f00)

  def test_s_or_b32(self):
    state = WaveState()
    state.sgpr[0], state.sgpr[1] = 0xff00, 0x00ff
    kernel = s_or_b32(s[2], s[0], s[1]).to_bytes() + s_endpgm().to_bytes()
    prog = decode_program(kernel)
    exec_wave(prog, state, bytearray(65536), 1)
    self.assertEqual(state.sgpr[2], 0xffff)

  def test_s_lshl_b32(self):
    state = WaveState()
    state.sgpr[0], state.sgpr[1] = 1, 4
    kernel = s_lshl_b32(s[2], s[0], s[1]).to_bytes() + s_endpgm().to_bytes()
    prog = decode_program(kernel)
    exec_wave(prog, state, bytearray(65536), 1)
    self.assertEqual(state.sgpr[2], 16)

  def test_s_lshr_b32(self):
    state = WaveState()
    state.sgpr[0], state.sgpr[1] = 256, 4
    kernel = s_lshr_b32(s[2], s[0], s[1]).to_bytes() + s_endpgm().to_bytes()
    prog = decode_program(kernel)
    exec_wave(prog, state, bytearray(65536), 1)
    self.assertEqual(state.sgpr[2], 16)

  def test_s_mul_i32(self):
    state = WaveState()
    state.sgpr[0], state.sgpr[1] = 7, 6
    kernel = s_mul_i32(s[2], s[0], s[1]).to_bytes() + s_endpgm().to_bytes()
    prog = decode_program(kernel)
    exec_wave(prog, state, bytearray(65536), 1)
    self.assertEqual(state.sgpr[2], 42)

  def test_s_cmp_eq_u32(self):
    state = WaveState()
    state.sgpr[0], state.sgpr[1] = 42, 42
    kernel = s_cmp_eq_u32(s[0], s[1]).to_bytes() + s_endpgm().to_bytes()
    prog = decode_program(kernel)
    exec_wave(prog, state, bytearray(65536), 1)
    self.assertEqual(state.scc, 1)

  def test_s_cmp_lg_u32(self):
    state = WaveState()
    state.sgpr[0], state.sgpr[1] = 42, 43
    kernel = s_cmp_lg_u32(s[0], s[1]).to_bytes() + s_endpgm().to_bytes()
    prog = decode_program(kernel)
    exec_wave(prog, state, bytearray(65536), 1)
    self.assertEqual(state.scc, 1)

class TestVectorOps(unittest.TestCase):
  def test_v_mov_b32(self):
    kernel = make_store_kernel([v_mov_b32_e32(v[1], 42)])
    out = run_kernel(kernel, n_threads=1)
    self.assertEqual(out, [42])

  def test_v_add_nc_u32(self):
    kernel = make_store_kernel([
      v_mov_b32_e32(v[1], 10),
      v_mov_b32_e32(v[2], 32),
      v_add_nc_u32_e32(v[1], v[1], v[2]),
    ])
    out = run_kernel(kernel, n_threads=1)
    self.assertEqual(out, [42])

  def test_v_sub_nc_u32(self):
    kernel = make_store_kernel([
      v_mov_b32_e32(v[1], 50),
      v_mov_b32_e32(v[2], 8),
      v_sub_nc_u32_e32(v[1], v[1], v[2]),
    ])
    out = run_kernel(kernel, n_threads=1)
    self.assertEqual(out, [42])

  def test_v_mul_lo_u32(self):
    kernel = make_store_kernel([
      v_mov_b32_e32(v[1], 6),
      v_mov_b32_e32(v[2], 7),
      v_mul_lo_u32(v[1], v[1], v[2]),
    ])
    out = run_kernel(kernel, n_threads=1)
    self.assertEqual(out, [42])

  def test_v_and_b32(self):
    kernel = make_store_kernel([
      v_mov_b32_e32(v[1], 0xff0f),
      v_mov_b32_e32(v[2], 0x0fff),
      v_and_b32_e32(v[1], v[1], v[2]),
    ])
    out = run_kernel(kernel, n_threads=1)
    self.assertEqual(out, [0x0f0f])

  def test_v_or_b32(self):
    kernel = make_store_kernel([
      v_mov_b32_e32(v[1], 0xf000),
      v_mov_b32_e32(v[2], 0x000f),
      v_or_b32_e32(v[1], v[1], v[2]),
    ])
    out = run_kernel(kernel, n_threads=1)
    self.assertEqual(out, [0xf00f])

  def test_v_lshlrev_b32(self):
    kernel = make_store_kernel([
      v_mov_b32_e32(v[1], 1),
      v_lshlrev_b32_e32(v[1], 5, v[1]),
    ])
    out = run_kernel(kernel, n_threads=1)
    self.assertEqual(out, [32])

  def test_v_lshrrev_b32(self):
    kernel = make_store_kernel([
      v_mov_b32_e32(v[1], 128),
      v_lshrrev_b32_e32(v[1], 3, v[1]),
    ])
    out = run_kernel(kernel, n_threads=1)
    self.assertEqual(out, [16])

  def test_v_add_f32(self):
    kernel = make_store_kernel([
      v_mov_b32_e32(v[1], i32(1.5)),
      v_mov_b32_e32(v[2], i32(2.5)),
      v_add_f32_e32(v[1], v[1], v[2]),
    ])
    out = run_kernel(kernel, n_threads=1)
    self.assertEqual(f32(out[0]), 4.0)

  def test_v_mul_f32(self):
    kernel = make_store_kernel([
      v_mov_b32_e32(v[1], i32(3.0)),
      v_mov_b32_e32(v[2], i32(4.0)),
      v_mul_f32_e32(v[1], v[1], v[2]),
    ])
    out = run_kernel(kernel, n_threads=1)
    self.assertEqual(f32(out[0]), 12.0)

  def test_v_max_f32(self):
    kernel = make_store_kernel([
      v_mov_b32_e32(v[1], i32(3.0)),
      v_mov_b32_e32(v[2], i32(5.0)),
      v_max_f32_e32(v[1], v[1], v[2]),
    ])
    out = run_kernel(kernel, n_threads=1)
    self.assertEqual(f32(out[0]), 5.0)

  def test_v_min_f32(self):
    kernel = make_store_kernel([
      v_mov_b32_e32(v[1], i32(3.0)),
      v_mov_b32_e32(v[2], i32(5.0)),
      v_min_f32_e32(v[1], v[1], v[2]),
    ])
    out = run_kernel(kernel, n_threads=1)
    self.assertEqual(f32(out[0]), 3.0)

class TestThreading(unittest.TestCase):
  def test_thread_id(self):
    """Each thread should get its own thread ID in v0."""
    kernel = make_store_kernel([v_mov_b32_e32(v[1], v[0])], store_vreg=1)
    out = run_kernel(kernel, n_threads=4)
    self.assertEqual(out, [0, 1, 2, 3])

  def test_thread_local_ops(self):
    """Each thread computes tid * 10."""
    kernel = make_store_kernel([
      v_mov_b32_e32(v[2], 10),
      v_mul_lo_u32(v[1], v[0], v[2]),
    ])
    out = run_kernel(kernel, n_threads=4)
    self.assertEqual(out, [0, 10, 20, 30])

  def test_exec_mask(self):
    """Test that exec mask controls which lanes execute."""
    kernel = b''
    kernel += s_load_b64(s[2:3], s[0:1], 0, soffset=NULL).to_bytes()
    kernel += s_waitcnt(lgkmcnt=0).to_bytes()
    kernel += v_mov_b32_e32(v[1], 100).to_bytes()  # default value
    kernel += s_mov_b32(EXEC_LO, 0b0101).to_bytes()  # only lanes 0 and 2
    kernel += v_mov_b32_e32(v[1], 42).to_bytes()   # only for active lanes
    kernel += s_mov_b32(EXEC_LO, 0xf).to_bytes()   # restore all lanes
    kernel += v_lshlrev_b32_e32(v[3], 2, v[0]).to_bytes()
    kernel += global_store_b32(addr=v[3], data=v[1], saddr=s[2]).to_bytes()
    kernel += s_endpgm().to_bytes()
    out = run_kernel(kernel, n_threads=4)
    self.assertEqual(out, [42, 100, 42, 100])

class TestBranching(unittest.TestCase):
  def test_s_branch(self):
    """Test unconditional branch."""
    state = WaveState()
    kernel = b''
    kernel += s_mov_b32(s[0], 1).to_bytes()
    kernel += s_branch(1).to_bytes()  # skip next instruction
    kernel += s_mov_b32(s[0], 2).to_bytes()  # should be skipped
    kernel += s_mov_b32(s[1], 3).to_bytes()
    kernel += s_endpgm().to_bytes()
    prog = decode_program(kernel)
    exec_wave(prog, state, bytearray(65536), 1)
    self.assertEqual(state.sgpr[0], 1)  # not overwritten
    self.assertEqual(state.sgpr[1], 3)

  def test_s_cbranch_scc0(self):
    """Test conditional branch on SCC=0."""
    state = WaveState()
    state.scc = 0
    kernel = b''
    kernel += s_mov_b32(s[0], 1).to_bytes()
    kernel += s_cbranch_scc0(1).to_bytes()  # branch if scc=0
    kernel += s_mov_b32(s[0], 2).to_bytes()  # should be skipped
    kernel += s_endpgm().to_bytes()
    prog = decode_program(kernel)
    exec_wave(prog, state, bytearray(65536), 1)
    self.assertEqual(state.sgpr[0], 1)

  def test_s_cbranch_scc1(self):
    """Test conditional branch on SCC=1."""
    state = WaveState()
    state.scc = 1
    kernel = b''
    kernel += s_mov_b32(s[0], 1).to_bytes()
    kernel += s_cbranch_scc1(1).to_bytes()  # branch if scc=1
    kernel += s_mov_b32(s[0], 2).to_bytes()  # should be skipped
    kernel += s_endpgm().to_bytes()
    prog = decode_program(kernel)
    exec_wave(prog, state, bytearray(65536), 1)
    self.assertEqual(state.sgpr[0], 1)

  def test_unknown_sopp_opcode(self):
    """Regression test: unknown SOPP opcodes should be ignored, not crash."""
    state = WaveState()
    # Create a raw SOPP instruction with opcode 8 (undefined in our enum)
    # SOPP format: bits[31:23] = 0b101111111, bits[22:16] = op, bits[15:0] = simm16
    unknown_sopp = (0b101111111 << 23) | (8 << 16) | 0  # op=8, simm16=0
    kernel = unknown_sopp.to_bytes(4, 'little') + s_endpgm().to_bytes()
    prog = decode_program(kernel)
    # Should not raise an exception
    exec_wave(prog, state, bytearray(65536), 1)

class TestMemory(unittest.TestCase):
  def test_global_load_store(self):
    """Test global load followed by store."""
    # Create input buffer
    input_buf = (ctypes.c_uint32 * 4)(10, 20, 30, 40)
    input_ptr = ctypes.addressof(input_buf)
    output_buf = (ctypes.c_uint32 * 4)(*[0]*4)
    output_ptr = ctypes.addressof(output_buf)
    args = (ctypes.c_uint64 * 2)(output_ptr, input_ptr)
    args_ptr = ctypes.addressof(args)

    # Kernel: load from input[tid], add 1, store to output[tid]
    kernel = b''
    kernel += s_load_b64(s[2:3], s[0:1], soffset=NULL, offset=0).to_bytes()  # output ptr
    kernel += s_load_b64(s[4:5], s[0:1], soffset=NULL, offset=8).to_bytes()  # input ptr
    kernel += s_waitcnt(lgkmcnt=0).to_bytes()
    kernel += v_lshlrev_b32_e32(v[2], 2, v[0]).to_bytes()  # offset = tid * 4
    kernel += global_load_b32(vdst=v[1], addr=v[2], saddr=s[4]).to_bytes()
    kernel += s_waitcnt(vmcnt=0).to_bytes()
    kernel += v_add_nc_u32_e32(v[1], 1, v[1]).to_bytes()  # add 1
    kernel += global_store_b32(addr=v[2], data=v[1], saddr=s[2]).to_bytes()
    kernel += s_endpgm().to_bytes()

    kernel_buf = (ctypes.c_char * len(kernel))(*kernel)
    kernel_ptr = ctypes.addressof(kernel_buf)
    set_valid_mem_ranges({
      (input_ptr, ctypes.sizeof(input_buf)),
      (output_ptr, ctypes.sizeof(output_buf)),
      (args_ptr, ctypes.sizeof(args)),
      (kernel_ptr, len(kernel)),
    })
    result = run_asm(kernel_ptr, len(kernel), 1, 1, 1, 4, 1, 1, args_ptr)
    self.assertEqual(result, 0)
    self.assertEqual([output_buf[i] for i in range(4)], [11, 21, 31, 41])

class TestFloatOps(unittest.TestCase):
  def test_v_rcp_f32(self):
    kernel = make_store_kernel([
      v_mov_b32_e32(v[1], i32(4.0)),
      v_rcp_f32_e32(v[1], v[1]),
    ])
    out = run_kernel(kernel, n_threads=1)
    self.assertAlmostEqual(f32(out[0]), 0.25, places=5)

  def test_v_sqrt_f32(self):
    kernel = make_store_kernel([
      v_mov_b32_e32(v[1], i32(16.0)),
      v_sqrt_f32_e32(v[1], v[1]),
    ])
    out = run_kernel(kernel, n_threads=1)
    self.assertAlmostEqual(f32(out[0]), 4.0, places=5)

  def test_v_floor_f32(self):
    kernel = make_store_kernel([
      v_mov_b32_e32(v[1], i32(3.7)),
      v_floor_f32_e32(v[1], v[1]),
    ])
    out = run_kernel(kernel, n_threads=1)
    self.assertEqual(f32(out[0]), 3.0)

  def test_v_ceil_f32(self):
    kernel = make_store_kernel([
      v_mov_b32_e32(v[1], i32(3.2)),
      v_ceil_f32_e32(v[1], v[1]),
    ])
    out = run_kernel(kernel, n_threads=1)
    self.assertEqual(f32(out[0]), 4.0)

  def test_v_cvt_f32_i32(self):
    kernel = make_store_kernel([
      v_mov_b32_e32(v[1], 42),
      v_cvt_f32_i32_e32(v[1], v[1]),
    ])
    out = run_kernel(kernel, n_threads=1)
    self.assertEqual(f32(out[0]), 42.0)

  def test_v_cvt_i32_f32(self):
    kernel = make_store_kernel([
      v_mov_b32_e32(v[1], i32(42.9)),
      v_cvt_i32_f32_e32(v[1], v[1]),
    ])
    out = run_kernel(kernel, n_threads=1)
    self.assertEqual(out[0], 42)

class TestVOP3(unittest.TestCase):
  def test_v_fma_f32(self):
    """Test fused multiply-add: a*b + c"""
    kernel = make_store_kernel([
      v_mov_b32_e32(v[1], i32(2.0)),
      v_mov_b32_e32(v[2], i32(3.0)),
      v_mov_b32_e32(v[4], i32(4.0)),
      v_fma_f32(v[1], v[1], v[2], v[4]),  # 2*3+4 = 10
    ])
    out = run_kernel(kernel, n_threads=1)
    self.assertEqual(f32(out[0]), 10.0)

  def test_v_add3_u32(self):
    """Test 3-operand add."""
    kernel = make_store_kernel([
      v_mov_b32_e32(v[1], 10),
      v_mov_b32_e32(v[2], 20),
      v_mov_b32_e32(v[4], 12),
      v_add3_u32(v[1], v[1], v[2], v[4]),  # 10+20+12 = 42
    ])
    out = run_kernel(kernel, n_threads=1)
    self.assertEqual(out[0], 42)

  def test_v_neg_modifier(self):
    """Test VOP3 negation modifier."""
    kernel = make_store_kernel([
      v_mov_b32_e32(v[1], i32(5.0)),
      v_mov_b32_e32(v[2], i32(3.0)),
      # v_add_f32 with neg on src1: 5 + (-3) = 2
      v_add_f32(v[1], v[1], v[2], neg=0b010),
    ])
    out = run_kernel(kernel, n_threads=1)
    self.assertEqual(f32(out[0]), 2.0)

  def test_v_ldexp_f32(self):
    """Regression test: V_LDEXP_F32 used by exp()."""
    kernel = make_store_kernel([
      v_mov_b32_e32(v[1], i32(1.5)),
      v_mov_b32_e32(v[2], 3),  # exponent
      v_ldexp_f32(v[1], v[1], v[2]),  # 1.5 * 2^3 = 12.0
    ])
    out = run_kernel(kernel, n_threads=1)
    self.assertEqual(f32(out[0]), 12.0)

  def test_v_xad_u32(self):
    """Regression test: V_XAD_U32 (multiply-add) used by matmul address calculation."""
    kernel = make_store_kernel([
      v_mov_b32_e32(v[1], 3),
      v_mov_b32_e32(v[2], 4),
      v_mov_b32_e32(v[4], 5),
      v_xad_u32(v[1], v[1], v[2], v[4]),  # 3*4+5 = 17
    ])
    out = run_kernel(kernel, n_threads=1)
    self.assertEqual(out[0], 17)

  def test_v_lshl_or_b32(self):
    """Regression test: V_LSHL_OR_B32 operand order is (s0 << s1) | s2, not (s0 << s2) | s1."""
    kernel = make_store_kernel([
      v_mov_b32_e32(v[1], 5),   # s0 = value to shift
      v_mov_b32_e32(v[2], 2),   # s1 = shift amount
      v_mov_b32_e32(v[4], 3),   # s2 = value to OR
      v_lshl_or_b32(v[1], v[1], v[2], v[4]),  # (5 << 2) | 3 = 20 | 3 = 23
    ])
    out = run_kernel(kernel, n_threads=1)
    self.assertEqual(out[0], 23)

  def test_v_sqrt_f32_negative(self):
    """Regression test: V_SQRT_F32 should return NaN for negative inputs, not 0."""
    kernel = make_store_kernel([
      v_mov_b32_e32(v[1], i32(-1.0)),
      v_sqrt_f32(v[1], v[1]),
    ])
    out = run_kernel(kernel, n_threads=1)
    self.assertTrue(math.isnan(f32(out[0])))

  def test_v_rsq_f32_negative(self):
    """Regression test: V_RSQ_F32 should return NaN for negative inputs, not inf."""
    kernel = make_store_kernel([
      v_mov_b32_e32(v[1], i32(-1.0)),
      v_rsq_f32(v[1], v[1]),
    ])
    out = run_kernel(kernel, n_threads=1)
    self.assertTrue(math.isnan(f32(out[0])))

class TestVOPD(unittest.TestCase):
  def test_vopd_add_nc_u32(self):
    """Test VOPD V_DUAL_ADD_NC_U32."""
    state = WaveState()
    state.vgpr[0][1] = 100
    state.vgpr[0][2] = 50
    # vdsty = (vdsty_enc << 1) | ((vdstx & 1) ^ 1), so for vdstx=3 (odd), vdsty_enc=2 gives vdsty=4
    kernel = VOPD(opx=VOPDOp.V_DUAL_MOV_B32, srcx0=256+1, vsrcx1=0, vdstx=3,
                  opy=VOPDOp.V_DUAL_ADD_NC_U32, srcy0=256+1, vsrcy1=2, vdsty=2).to_bytes()
    kernel += s_endpgm().to_bytes()
    prog = decode_program(kernel)
    exec_wave(prog, state, bytearray(65536), 1)
    self.assertEqual(state.vgpr[0][3], 100)  # MOV result
    self.assertEqual(state.vgpr[0][4], 150)  # 100 + 50

  def test_vopd_lshlrev(self):
    """Test VOPD V_DUAL_LSHLREV_B32."""
    state = WaveState()
    state.vgpr[0][1] = 0x10
    state.vgpr[0][2] = 0
    # vdsty = (vdsty_enc << 1) | ((vdstx & 1) ^ 1), so for vdstx=3 (odd), vdsty_enc=2 gives vdsty=4
    kernel = VOPD(opx=VOPDOp.V_DUAL_MOV_B32, srcx0=256+1, vsrcx1=0, vdstx=3,
                  opy=VOPDOp.V_DUAL_LSHLREV_B32, srcy0=132, vsrcy1=1, vdsty=2).to_bytes()  # V4 = V1 << 4
    kernel += s_endpgm().to_bytes()
    prog = decode_program(kernel)
    exec_wave(prog, state, bytearray(65536), 1)
    self.assertEqual(state.vgpr[0][3], 0x10)  # MOV result
    self.assertEqual(state.vgpr[0][4], 0x100)  # 0x10 << 4 = 0x100

  def test_vopd_and(self):
    """Test VOPD V_DUAL_AND_B32."""
    state = WaveState()
    state.vgpr[0][1] = 0xff
    state.vgpr[0][2] = 0x0f
    # vdsty = (vdsty_enc << 1) | ((vdstx & 1) ^ 1), so for vdstx=3 (odd), vdsty_enc=2 gives vdsty=4
    kernel = VOPD(opx=VOPDOp.V_DUAL_MOV_B32, srcx0=256+1, vsrcx1=0, vdstx=3,
                  opy=VOPDOp.V_DUAL_AND_B32, srcy0=256+1, vsrcy1=2, vdsty=2).to_bytes()
    kernel += s_endpgm().to_bytes()
    prog = decode_program(kernel)
    exec_wave(prog, state, bytearray(65536), 1)
    self.assertEqual(state.vgpr[0][3], 0xff)
    self.assertEqual(state.vgpr[0][4], 0x0f)  # 0xff & 0x0f = 0x0f

class TestDecoder(unittest.TestCase):
  def test_vopd_literal_handling(self):
    """Regression test: VOPD srcx0/srcy0 with literal (255) wasn't consuming the literal dword."""
    state = WaveState()
    # Create VOPD with srcx0=255 (literal), followed by literal value 0x12345678
    vopd_bytes = VOPD(opx=8, srcx0=255, vsrcx1=0, vdstx=1,  # MOV: V1 = literal
                      opy=8, srcy0=128, vsrcy1=0, vdsty=2).to_bytes()  # MOV: V2 = 0
    literal_bytes = (0x12345678).to_bytes(4, 'little')
    kernel = vopd_bytes + literal_bytes + s_endpgm().to_bytes()
    prog = decode_program(kernel)
    # Should decode as 3 instructions: VOPD (with literal), then S_ENDPGM
    # The literal should NOT be decoded as a separate instruction
    self.assertEqual(len(prog), 2)  # VOPD + S_ENDPGM
    exec_wave(prog, state, bytearray(65536), 1)
    self.assertEqual(state.vgpr[0][1], 0x12345678)

  def test_s_endpgm_stops_decode(self):
    """Regression test: decoder should stop at S_ENDPGM, not read past into metadata."""
    # Create a kernel followed by garbage that looks like an invalid instruction
    kernel = s_mov_b32(s[0], 42).to_bytes() + s_endpgm().to_bytes()
    garbage = bytes([0xff] * 16)  # garbage after kernel
    prog = decode_program(kernel + garbage)
    # Should only have 2 instructions (s_mov_b32 and s_endpgm)
    self.assertEqual(len(prog), 2)

class TestFloatConversion(unittest.TestCase):
  """Unit tests for i32/i16/f32/f16 float conversion functions."""

  def test_i32_preserves_nan_sign(self):
    """NaN sign bit should be preserved when converting float to int bits."""
    from extra.assembly.rdna3.emu import i32, f32
    # 0 * -inf produces a negative NaN
    neg_nan = 0.0 * float('-inf')
    bits = i32(neg_nan)
    # Should have sign bit set (0xffc00000), not canonical positive NaN (0x7fc00000)
    self.assertEqual(bits & 0x80000000, 0x80000000, f"Expected negative NaN, got 0x{bits:08x}")
    self.assertTrue(math.isnan(f32(bits)))

  def test_i32_preserves_positive_nan(self):
    """Positive NaN should remain positive."""
    from extra.assembly.rdna3.emu import i32, f32
    pos_nan = float('nan')
    bits = i32(pos_nan)
    # Standard Python NaN is positive (0x7fc00000)
    self.assertEqual(bits & 0x80000000, 0, f"Expected positive NaN, got 0x{bits:08x}")
    self.assertTrue(math.isnan(f32(bits)))

  def test_i32_overflow_to_inf(self):
    """Values too large for f32 should become inf."""
    from extra.assembly.rdna3.emu import i32, f32
    big = 2.0 ** 200
    self.assertEqual(i32(big), 0x7f800000)  # +inf
    self.assertEqual(i32(-big), 0xff800000)  # -inf

  def test_i32_inf(self):
    """Infinity should be preserved."""
    from extra.assembly.rdna3.emu import i32
    self.assertEqual(i32(float('inf')), 0x7f800000)
    self.assertEqual(i32(float('-inf')), 0xff800000)

  def test_i32_normal_values(self):
    """Normal float values should round-trip correctly (within f32 precision)."""
    from extra.assembly.rdna3.emu import i32, f32
    # Use values exactly representable in float32
    for val in [0.0, 1.0, -1.0, 0.5, -0.5, 100.0, -100.0, 1e10]:
      bits = i32(val)
      self.assertAlmostEqual(f32(bits), val, places=5)

  def test_i16_overflow_to_inf(self):
    """Values too large for f16 should become inf."""
    from extra.assembly.rdna3.emu import i16
    big = 100000.0  # way larger than f16 max (65504)
    self.assertEqual(i16(big), 0x7c00)  # +inf
    self.assertEqual(i16(-big), 0xfc00)  # -inf

  def test_i16_inf(self):
    """Infinity should be preserved."""
    from extra.assembly.rdna3.emu import i16
    self.assertEqual(i16(float('inf')), 0x7c00)
    self.assertEqual(i16(float('-inf')), 0xfc00)

  def test_fma_nan_sign_preserved(self):
    """FMA producing NaN should preserve the correct sign bit."""
    from extra.assembly.rdna3.emu import i32, f32
    # 0 * (-inf) + 1.0 = NaN (from 0 * -inf)
    a, b, c = 0.0, float('-inf'), 1.0
    result = i32(a * b + c)
    # The NaN should be negative since 0 * -inf produces negative NaN
    self.assertEqual(result & 0x80000000, 0x80000000, f"Expected negative NaN, got 0x{result:08x}")

class TestMultiWave(unittest.TestCase):
  def test_all_waves_execute(self):
    """Regression test: all waves in a workgroup must execute, not just the first."""
    n_threads = 64  # 2 waves of 32 threads each
    output = (ctypes.c_uint32 * n_threads)(*[0xdead] * n_threads)
    output_ptr = ctypes.addressof(output)
    args = (ctypes.c_uint64 * 1)(output_ptr)
    args_ptr = ctypes.addressof(args)

    # Simple kernel: store tid to output[tid]
    kernel = b''
    kernel += s_load_b64(s[2:3], s[0:1], soffset=NULL, offset=0).to_bytes()
    kernel += s_waitcnt(lgkmcnt=0).to_bytes()
    kernel += v_lshlrev_b32_e32(v[1], 2, v[0]).to_bytes()  # offset = tid * 4
    kernel += global_store_b32(addr=v[1], data=v[0], saddr=s[2]).to_bytes()
    kernel += s_endpgm().to_bytes()

    kernel_buf = (ctypes.c_char * len(kernel))(*kernel)
    kernel_ptr = ctypes.addressof(kernel_buf)
    set_valid_mem_ranges({
      (output_ptr, ctypes.sizeof(output)),
      (args_ptr, ctypes.sizeof(args)),
      (kernel_ptr, len(kernel)),
    })
    result = run_asm(kernel_ptr, len(kernel), 1, 1, 1, n_threads, 1, 1, args_ptr)
    self.assertEqual(result, 0)
    # All threads should have written their tid
    for i in range(n_threads):
      self.assertEqual(output[i], i, f"Thread {i} didn't execute")

if __name__ == "__main__":
  unittest.main()
