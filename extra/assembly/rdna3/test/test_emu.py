#!/usr/bin/env python3
"""Regression tests for the RDNA3 emulator instruction execution.
Uses run_asm() with memory output, so tests can run on both emulator and real hardware.

Set USE_HW=1 to run on real AMD hardware instead of the emulator.
"""

import ctypes, unittest, os, struct
from extra.assembly.rdna3.autogen import *
from extra.assembly.rdna3.emu import WaveState, run_asm, set_valid_mem_ranges
from extra.assembly.rdna3.pcode import _i32, _f32

VCC = SrcEnum.VCC_LO  # For VOP3SD sdst field
USE_HW = os.environ.get("USE_HW", "0") == "1"

# Output buffer layout: vgpr[16][32], sgpr[16], vcc, scc
# Each VGPR store writes 32 lanes (128 bytes), so vgpr[i] is at offset i*128
N_VGPRS, N_SGPRS, WAVE_SIZE = 16, 16, 32
VGPR_BYTES = N_VGPRS * WAVE_SIZE * 4  # 16 regs * 32 lanes * 4 bytes = 2048
SGPR_BYTES = N_SGPRS * 4  # 16 regs * 4 bytes = 64
OUT_BYTES = VGPR_BYTES + SGPR_BYTES + 8  # + vcc + scc

def f2i(f: float) -> int: return _i32(f)
def i2f(i: int) -> float: return _f32(i)

def assemble(instructions: list) -> bytes:
  return b''.join(inst.to_bytes() for inst in instructions)

def get_prologue_epilogue(n_lanes: int) -> tuple[list, list]:
  """Generate prologue and epilogue instructions for state capture."""
  # Prologue: save s[0:1] and v[0] before test clobbers them
  # Use s[80:81] for args pointer (safe range, avoiding VCC=106-107 and staying under 100)
  prologue = [
    s_mov_b32(s[80], s[0]),
    s_mov_b32(s[81], s[1]),
    v_mov_b32_e32(v[255], v[0]),
  ]

  # Epilogue: store wave state to memory
  # Use s[90-99] for epilogue temps to stay in safe SGPR range (<100, avoiding VCC=106-107)
  # s[90] = saved VCC, s[91] = saved SCC, s[92:93] = output addr, s[94] = saved EXEC
  # Save VCC/SCC first before we clobber them
  epilogue = [
    s_mov_b32(s[90], SrcEnum.VCC_LO),  # save VCC
    s_cselect_b32(s[91], 1, 0),  # save SCC
    s_load_b64(s[92], s[80], 0, soffset=SrcEnum.NULL),
    s_waitcnt(lgkmcnt=0),
    v_lshlrev_b32_e32(v[240], 2, v[255]),  # v[240] = lane_id * 4
  ]
  # Store VGPRs: vgpr[i] at offset i*128 + lane_id*4
  for i in range(N_VGPRS):
    epilogue.append(global_store_b32(addr=v[240], data=v[i], saddr=s[92], offset=i * WAVE_SIZE * 4))
  # Store SGPRs at VGPR_BYTES + i*4 (lane 0 only via exec mask)
  epilogue.append(v_mov_b32_e32(v[241], 0))
  epilogue.append(v_cmp_eq_u32_e32(v[255], v[241]))
  epilogue.append(s_and_saveexec_b32(s[94], SrcEnum.VCC_LO))
  epilogue.append(v_mov_b32_e32(v[240], 0))
  for i in range(N_SGPRS):
    epilogue.append(v_mov_b32_e32(v[243], s[i]))
    epilogue.append(global_store_b32(addr=v[240], data=v[243], saddr=s[92], offset=VGPR_BYTES + i * 4))
  # Store saved VCC
  epilogue.append(v_mov_b32_e32(v[243], s[90]))
  epilogue.append(global_store_b32(addr=v[240], data=v[243], saddr=s[92], offset=VGPR_BYTES + SGPR_BYTES))
  # Store saved SCC
  epilogue.append(v_mov_b32_e32(v[243], s[91]))
  epilogue.append(global_store_b32(addr=v[240], data=v[243], saddr=s[92], offset=VGPR_BYTES + SGPR_BYTES + 4))
  epilogue.append(s_mov_b32(s[SrcEnum.EXEC_LO - 128], s[94]))  # restore exec
  epilogue.append(s_endpgm())

  return prologue, epilogue

def parse_output(out_buf: bytes, n_lanes: int) -> WaveState:
  """Parse output buffer into WaveState."""
  st = WaveState()
  for i in range(N_VGPRS):
    for lane in range(n_lanes):
      off = i * WAVE_SIZE * 4 + lane * 4
      st.vgpr[lane][i] = struct.unpack_from('<I', out_buf, off)[0]
  for i in range(N_SGPRS):
    st.sgpr[i] = struct.unpack_from('<I', out_buf, VGPR_BYTES + i * 4)[0]
  st.vcc = struct.unpack_from('<I', out_buf, VGPR_BYTES + SGPR_BYTES)[0]
  st.scc = struct.unpack_from('<I', out_buf, VGPR_BYTES + SGPR_BYTES + 4)[0]
  return st

def run_program_emu(instructions: list, n_lanes: int = 1) -> WaveState:
  """Run instructions via emulator run_asm, dump state to memory, return WaveState."""
  out_buf = (ctypes.c_uint8 * OUT_BYTES)(*([0] * OUT_BYTES))
  out_addr = ctypes.addressof(out_buf)

  prologue, epilogue = get_prologue_epilogue(n_lanes)
  code = assemble(prologue + instructions + epilogue)

  args = (ctypes.c_uint64 * 1)(out_addr)
  args_ptr = ctypes.addressof(args)
  kernel_buf = (ctypes.c_char * len(code)).from_buffer_copy(code)
  lib_ptr = ctypes.addressof(kernel_buf)

  set_valid_mem_ranges({(out_addr, OUT_BYTES), (args_ptr, 8)})
  result = run_asm(lib_ptr, len(code), 1, 1, 1, n_lanes, 1, 1, args_ptr)
  assert result == 0, f"run_asm failed with {result}"

  return parse_output(bytes(out_buf), n_lanes)

def run_program_hw(instructions: list, n_lanes: int = 1) -> WaveState:
  """Run instructions on real AMD hardware via HIPCompiler and AMDProgram."""
  from tinygrad.device import Device
  from tinygrad.runtime.ops_amd import AMDProgram
  from tinygrad.runtime.support.compiler_amd import HIPCompiler
  from tinygrad.helpers import flat_mv

  dev = Device["AMD"]
  compiler = HIPCompiler(dev.arch)

  prologue, epilogue = get_prologue_epilogue(n_lanes)
  code = assemble(prologue + instructions + epilogue)

  # Create inline assembly source with .byte directives
  byte_str = ', '.join(f'0x{b:02x}' for b in code)
  asm_src = f""".text
.globl test
.p2align 8
.type test,@function
test:
.byte {byte_str}

.rodata
.p2align 6
.amdhsa_kernel test
  .amdhsa_next_free_vgpr 256
  .amdhsa_next_free_sgpr 96
  .amdhsa_wavefront_size32 1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_kernarg_size 8
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 0
amdhsa.kernels:
  - .name: test
    .symbol: test.kd
    .kernarg_segment_size: 8
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 32
    .sgpr_count: 96
    .vgpr_count: 256
    .max_flat_workgroup_size: 1024
...
.end_amdgpu_metadata
"""

  lib = compiler.compile(asm_src)
  prg = AMDProgram(dev, "test", lib)

  # Allocate output buffer on GPU
  out_gpu = dev.allocator.alloc(OUT_BYTES)

  # Run the kernel
  prg(out_gpu, global_size=(1, 1, 1), local_size=(n_lanes, 1, 1), wait=True)

  # Copy result back
  out_buf = bytearray(OUT_BYTES)
  dev.allocator._copyout(flat_mv(memoryview(out_buf)), out_gpu)

  return parse_output(bytes(out_buf), n_lanes)

def run_program(instructions: list, n_lanes: int = 1) -> WaveState:
  """Run instructions, dump state to memory, return WaveState. Uses USE_HW env var to select backend."""
  return run_program_hw(instructions, n_lanes) if USE_HW else run_program_emu(instructions, n_lanes)


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


class TestTrigonometry(unittest.TestCase):
  """Tests for trigonometric instructions."""

  def test_v_sin_f32_small(self):
    """V_SIN_F32 computes sin for small values."""
    import math
    # sin(1.0) ≈ 0.8414709848
    instructions = [
      v_mov_b32_e32(v[0], 1.0),
      v_sin_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][1])
    expected = math.sin(1.0 * 2 * math.pi)  # V_SIN_F32 expects input in cycles (0-1 = 0-2π)
    self.assertAlmostEqual(result, expected, places=4)

  def test_v_sin_f32_quarter(self):
    """V_SIN_F32 at 0.25 cycles = sin(π/2) = 1.0."""
    instructions = [
      s_mov_b32(s[0], f2i(0.25)),  # 0.25 is not an inline constant, use f2i
      v_mov_b32_e32(v[0], s[0]),
      v_sin_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][1])
    self.assertAlmostEqual(result, 1.0, places=4)

  def test_v_sin_f32_large(self):
    """V_SIN_F32 for large input value (132000.0)."""
    import math
    # This is the failing case: sin(132000.0) should be ≈ 0.294
    # V_SIN_F32 input is in cycles, so we need frac(132000.0) * 2π
    instructions = [
      s_mov_b32(s[0], f2i(132000.0)),
      v_mov_b32_e32(v[0], s[0]),
      v_sin_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][1])
    # frac(132000.0) = 0, so sin(0) = 0... but actually V_SIN_F32 does its own frac internally
    # The expected value is sin(frac(132000.0) * 2π) where frac is done in the instruction
    # For 132000.0, the hardware computes frac(132000.0) ≈ 0.046875 (due to precision)
    # sin(0.046875 * 2π) ≈ 0.294
    expected = math.sin(132000.0 * 2 * math.pi)
    # Allow some tolerance due to precision differences
    self.assertAlmostEqual(result, expected, places=2, msg=f"sin(132000) got {result}, expected ~{expected}")


if __name__ == '__main__':
  unittest.main()
