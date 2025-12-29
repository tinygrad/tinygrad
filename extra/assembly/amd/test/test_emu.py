#!/usr/bin/env python3
"""Regression tests for the RDNA3 emulator instruction execution.
Uses run_asm() with memory output, so tests can run on both emulator and real hardware.

Set USE_HW=1 to run on both emulator and real hardware, comparing results.
"""

import ctypes, unittest, os, struct
from extra.assembly.amd.autogen.rdna3 import *
from extra.assembly.amd.dsl import RawImm
from extra.assembly.amd.emu import WaveState, run_asm, set_valid_mem_ranges
from extra.assembly.amd.pcode import _i32, _f32

VCC = SrcEnum.VCC_LO  # For VOP3SD sdst field
USE_HW = os.environ.get("USE_HW", "0") == "1"
# Tolerance for float comparisons (in ULPs or absolute)
FLOAT_TOLERANCE = 1e-5

# Output buffer layout: vgpr[16][32], sgpr[16], vcc, scc
# Each VGPR store writes 32 lanes (128 bytes), so vgpr[i] is at offset i*128
N_VGPRS, N_SGPRS, WAVE_SIZE = 16, 16, 32
VGPR_BYTES = N_VGPRS * WAVE_SIZE * 4  # 16 regs * 32 lanes * 4 bytes = 2048
SGPR_BYTES = N_SGPRS * 4  # 16 regs * 4 bytes = 64
OUT_BYTES = VGPR_BYTES + SGPR_BYTES + 8  # + vcc + scc

def f2i(f: float) -> int: return _i32(f)
def i2f(i: int) -> float: return _f32(i)
def f2i64(f: float) -> int: return struct.unpack('<Q', struct.pack('<d', f))[0]
def i642f(i: int) -> float: return struct.unpack('<d', struct.pack('<Q', i))[0]

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
  # Zero out test registers (v0-v15, s0-s15, vcc) so emu and hw start from same state
  for i in range(N_VGPRS):
    prologue.append(v_mov_b32_e32(v[i], 0))
  for i in range(N_SGPRS):
    prologue.append(s_mov_b32(s[i], 0))
  prologue.append(s_mov_b32(s[SrcEnum.VCC_LO - 128], 0))  # zero VCC

  # Epilogue: store wave state to memory
  # Use s[90-99] for epilogue temps to stay in safe SGPR range (<100, avoiding VCC=106-107)
  # s[90] = saved VCC, s[91] = saved SCC, s[92:93] = output addr, s[94] = saved EXEC
  # Save VCC/SCC first before we clobber them
  epilogue = [
    s_mov_b32(s[90], SrcEnum.VCC_LO),  # save VCC
    s_cselect_b32(s[91], 1, 0),  # save SCC
    s_load_b64(s[92:93], s[80], 0, soffset=SrcEnum.NULL),
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

def compare_wave_states(emu_st: WaveState, hw_st: WaveState, n_lanes: int, n_vgprs: int = N_VGPRS) -> list[str]:
  """Compare two WaveStates and return list of differences."""
  import math
  diffs = []
  # Compare VGPRs - vgpr is list[lane][reg]
  for i in range(n_vgprs):
    for lane in range(n_lanes):
      emu_val = emu_st.vgpr[lane][i]
      hw_val = hw_st.vgpr[lane][i]
      if emu_val != hw_val:
        emu_f, hw_f = _f32(emu_val), _f32(hw_val)
        # Handle NaN comparison
        if math.isnan(emu_f) and math.isnan(hw_f):
          continue
        diffs.append(f"v[{i}] lane {lane}: emu=0x{emu_val:08x} ({emu_f:.6g}) hw=0x{hw_val:08x} ({hw_f:.6g})")
  # Compare SGPRs - sgpr is list
  for i in range(N_SGPRS):
    emu_val = emu_st.sgpr[i]
    hw_val = hw_st.sgpr[i]
    if emu_val != hw_val:
      diffs.append(f"s[{i}]: emu=0x{emu_val:08x} hw=0x{hw_val:08x}")
  # Compare VCC
  if emu_st.vcc != hw_st.vcc:
    diffs.append(f"vcc: emu=0x{emu_st.vcc:08x} hw=0x{hw_st.vcc:08x}")
  # Compare SCC
  if emu_st.scc != hw_st.scc:
    diffs.append(f"scc: emu={emu_st.scc} hw={hw_st.scc}")
  return diffs

def run_program(instructions: list, n_lanes: int = 1) -> WaveState:
  """Run instructions and return WaveState.

  If USE_HW=1, runs on both emulator and hardware, compares results, and raises if they differ.
  Otherwise, runs only on emulator.
  """
  emu_st = run_program_emu(instructions, n_lanes)
  if USE_HW:
    hw_st = run_program_hw(instructions, n_lanes)
    diffs = compare_wave_states(emu_st, hw_st, n_lanes)
    if diffs:
      raise AssertionError(f"Emulator vs Hardware mismatch:\n" + "\n".join(diffs))
    return hw_st  # Return hardware result when both match
  return emu_st


class TestVDivScale(unittest.TestCase):
  """Tests for V_DIV_SCALE_F32 edge cases.

  V_DIV_SCALE_F32 is used in the Newton-Raphson division sequence to handle
  denormals and near-overflow cases. It scales operands and sets VCC when
  the final result needs to be unscaled.

  Pseudocode cases:
  1. Zero operands -> NaN
  2. exp(S2) - exp(S1) >= 96 -> scale denom, VCC=1
  3. S1 is denorm -> scale by 2^64
  4. 1/S1 is f64 denorm AND S2/S1 is f32 denorm -> scale denom, VCC=1
  5. 1/S1 is f64 denorm -> scale by 2^-64
  6. S2/S1 is f32 denorm -> scale numer, VCC=1
  7. exp(S2) <= 23 -> scale by 2^64 (tiny numerator)
  """

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

  def test_div_scale_f32_zero_denom_gives_nan(self):
    """V_DIV_SCALE_F32: zero denominator -> NaN, VCC=1."""
    instructions = [
      v_mov_b32_e32(v[0], 1.0),  # numerator
      v_mov_b32_e32(v[1], 0.0),  # denominator = 0
      v_div_scale_f32(v[2], VCC, v[0], v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    import math
    self.assertTrue(math.isnan(i2f(st.vgpr[0][2])), "Should be NaN for zero denom")
    self.assertEqual(st.vcc & 1, 1, "VCC should be 1 for zero denom")

  def test_div_scale_f32_zero_numer_gives_nan(self):
    """V_DIV_SCALE_F32: zero numerator -> NaN, VCC=1."""
    instructions = [
      v_mov_b32_e32(v[0], 0.0),  # numerator = 0
      v_mov_b32_e32(v[1], 1.0),  # denominator
      v_div_scale_f32(v[2], VCC, v[0], v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    import math
    self.assertTrue(math.isnan(i2f(st.vgpr[0][2])), "Should be NaN for zero numer")
    self.assertEqual(st.vcc & 1, 1, "VCC should be 1 for zero numer")

  def test_div_scale_f32_large_exp_diff_scales_denom(self):
    """V_DIV_SCALE_F32: exp(numer) - exp(denom) >= 96 -> scale denom, VCC=1."""
    # Need exp difference >= 96. Use MAX_FLOAT / tiny_normal
    # MAX_FLOAT exp=254, tiny_normal with exp <= 254-96=158
    # Let's use exp=127 (1.0) for denom, exp=254 for numer -> diff = 127 (>96)
    max_float = 0x7f7fffff  # 3.4028235e+38, exp=254
    instructions = [
      s_mov_b32(s[0], max_float),
      v_mov_b32_e32(v[0], s[0]),  # numer = MAX_FLOAT (S2)
      v_mov_b32_e32(v[1], 1.0),   # denom = 1.0 (S1), exp=127. diff = 254-127 = 127 >= 96
      # S0=denom (what we're scaling), S1=denom, S2=numer
      v_div_scale_f32(v[2], VCC, v[1], v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vcc & 1, 1, "VCC should be 1 when scaling denom for large exp diff")
    # Result should be denom * 2^64
    expected = 1.0 * (2.0 ** 64)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), expected, delta=expected * 1e-6)

  def test_div_scale_f32_denorm_denom(self):
    """V_DIV_SCALE_F32: denormalized denominator -> NaN, VCC=1.

    Hardware returns NaN when denominator is denormalized (different from PDF pseudocode).
    """
    # Smallest positive denorm: 0x00000001 = 1.4e-45
    denorm = 0x00000001
    instructions = [
      s_mov_b32(s[0], denorm),
      v_mov_b32_e32(v[0], 1.0),   # numer = 1.0 (S2)
      v_mov_b32_e32(v[1], s[0]), # denom = denorm (S1)
      # S0=denom, S1=denom, S2=numer -> scale denom
      v_div_scale_f32(v[2], VCC, v[1], v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    import math
    self.assertTrue(math.isnan(i2f(st.vgpr[0][2])), "Hardware returns NaN for denorm denom")
    self.assertEqual(st.vcc & 1, 1, "VCC should be 1 for denorm denom")

  def test_div_scale_f32_tiny_numer_exp_le_23(self):
    """V_DIV_SCALE_F32: exponent(numer) <= 23 -> scale by 2^64, VCC=1."""
    # exp <= 23 means exponent field is 0..23
    # exp=23 corresponds to float value around 2^(23-127) = 2^-104 ≈ 4.9e-32
    # Use exp=1 (smallest normal), which is 2^(1-127) = 2^-126 ≈ 1.18e-38
    smallest_normal = 0x00800000  # exp=1, mantissa=0
    instructions = [
      s_mov_b32(s[0], smallest_normal),
      v_mov_b32_e32(v[0], s[0]),  # numer = smallest_normal (S2), exp=1 <= 23
      v_mov_b32_e32(v[1], 1.0),   # denom = 1.0 (S1)
      # S0=numer, S1=denom, S2=numer -> scale numer
      v_div_scale_f32(v[2], VCC, v[0], v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    # Numer scaled by 2^64, VCC=1 to indicate scaling was done
    numer_f = i2f(smallest_normal)
    expected = numer_f * (2.0 ** 64)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), expected, delta=abs(expected) * 1e-5)
    self.assertEqual(st.vcc & 1, 1, "VCC should be 1 when scaling tiny numer")

  def test_div_scale_f32_result_would_be_denorm(self):
    """V_DIV_SCALE_F32: result would be denorm -> no scaling applied, VCC=1.

    When the result of numer/denom would be denormalized, hardware sets VCC=1
    but does NOT scale the input (returns it unchanged). The scaling happens
    elsewhere in the division sequence.
    """
    # If S2/S1 would be denorm, set VCC but don't scale
    # Denorm result: exp < 1, i.e., |result| < 2^-126
    # Use 1.0 / 2^127 ≈ 5.9e-39 (result would be denorm)
    large_denom = 0x7f000000  # 2^127
    instructions = [
      s_mov_b32(s[0], large_denom),
      v_mov_b32_e32(v[0], 1.0),   # numer = 1.0 (S2)
      v_mov_b32_e32(v[1], s[0]), # denom = 2^127 (S1)
      # S0=numer, S1=denom, S2=numer -> check if we need to scale numer
      v_div_scale_f32(v[2], VCC, v[0], v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    # Hardware returns input unchanged but sets VCC=1
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 1.0, places=5)
    self.assertEqual(st.vcc & 1, 1, "VCC should be 1 when result would be denorm")


class TestVDivFmas(unittest.TestCase):
  """Tests for V_DIV_FMAS_F32 edge cases.

  V_DIV_FMAS_F32 performs FMA with optional scaling based on VCC.
  The scale direction depends on S2's exponent (the addend):
  - If exponent(S2) > 127 (i.e., S2 >= 2.0): scale by 2^+64
  - Otherwise: scale by 2^-64

  NOTE: The PDF (page 449) incorrectly says just 2^32.
  """

  def test_div_fmas_f32_no_scale(self):
    """V_DIV_FMAS_F32: VCC=0 -> normal FMA."""
    instructions = [
      s_mov_b32(s[SrcEnum.VCC_LO - 128], 0),  # VCC = 0
      v_mov_b32_e32(v[0], 2.0),   # S0
      v_mov_b32_e32(v[1], 3.0),   # S1
      v_mov_b32_e32(v[2], 1.0),   # S2
      v_div_fmas_f32(v[3], v[0], v[1], v[2]),  # 2*3+1 = 7
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][3]), 7.0, places=5)

  def test_div_fmas_f32_scale_up(self):
    """V_DIV_FMAS_F32: VCC=1 with S2 >= 2.0 -> scale by 2^+64."""
    instructions = [
      s_mov_b32(s[SrcEnum.VCC_LO - 128], 1),  # VCC = 1
      v_mov_b32_e32(v[0], 1.0),   # S0
      v_mov_b32_e32(v[1], 1.0),   # S1
      v_mov_b32_e32(v[2], 2.0),   # S2 >= 2.0, so scale UP
      v_div_fmas_f32(v[3], v[0], v[1], v[2]),  # 2^+64 * (1*1+2) = 2^+64 * 3
    ]
    st = run_program(instructions, n_lanes=1)
    expected = 3.0 * (2.0 ** 64)
    self.assertAlmostEqual(i2f(st.vgpr[0][3]), expected, delta=abs(expected) * 1e-6)

  def test_div_fmas_f32_scale_down(self):
    """V_DIV_FMAS_F32: VCC=1 with S2 < 2.0 -> scale by 2^-64."""
    instructions = [
      s_mov_b32(s[SrcEnum.VCC_LO - 128], 1),  # VCC = 1
      v_mov_b32_e32(v[0], 2.0),   # S0
      v_mov_b32_e32(v[1], 3.0),   # S1
      v_mov_b32_e32(v[2], 1.0),   # S2 < 2.0, so scale DOWN
      v_div_fmas_f32(v[3], v[0], v[1], v[2]),  # 2^-64 * (2*3+1) = 2^-64 * 7
    ]
    st = run_program(instructions, n_lanes=1)
    expected = 7.0 * (2.0 ** -64)
    self.assertAlmostEqual(i2f(st.vgpr[0][3]), expected, delta=abs(expected) * 1e-6)

  def test_div_fmas_f32_per_lane_vcc(self):
    """V_DIV_FMAS_F32: different VCC per lane with S2 < 2.0."""
    instructions = [
      s_mov_b32(s[SrcEnum.VCC_LO - 128], 0b0101),  # VCC: lanes 0,2 set
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 1.0),
      v_mov_b32_e32(v[2], 1.0),  # S2 < 2.0, so scale DOWN
      v_div_fmas_f32(v[3], v[0], v[1], v[2]),  # fma(1,1,1) = 2, scaled = 2^-64 * 2
    ]
    st = run_program(instructions, n_lanes=4)
    scaled = 2.0 * (2.0 ** -64)
    unscaled = 2.0
    self.assertAlmostEqual(i2f(st.vgpr[0][3]), scaled, delta=abs(scaled) * 1e-6)  # lane 0: VCC=1
    self.assertAlmostEqual(i2f(st.vgpr[1][3]), unscaled, places=5)                 # lane 1: VCC=0
    self.assertAlmostEqual(i2f(st.vgpr[2][3]), scaled, delta=abs(scaled) * 1e-6)  # lane 2: VCC=1
    self.assertAlmostEqual(i2f(st.vgpr[3][3]), unscaled, places=5)                 # lane 3: VCC=0


class TestVDivFixup(unittest.TestCase):
  """Tests for V_DIV_FIXUP_F32 edge cases.

  V_DIV_FIXUP_F32 is the final step of Newton-Raphson division.
  It handles special cases: NaN, Inf, zero, overflow, underflow.

  Args: S0=quotient from NR iteration, S1=denominator, S2=numerator
  """

  def test_div_fixup_f32_normal(self):
    """V_DIV_FIXUP_F32: normal division passes through quotient."""
    # 6.0 / 2.0 = 3.0
    instructions = [
      v_mov_b32_e32(v[0], 3.0),   # S0 = quotient
      v_mov_b32_e32(v[1], 2.0),   # S1 = denominator
      v_mov_b32_e32(v[2], 6.0),   # S2 = numerator
      v_div_fixup_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][3]), 3.0, places=5)

  def test_div_fixup_f32_nan_numer(self):
    """V_DIV_FIXUP_F32: NaN numerator -> quiet NaN."""
    nan = 0x7fc00000  # quiet NaN
    instructions = [
      s_mov_b32(s[0], nan),
      v_mov_b32_e32(v[0], 1.0),   # S0 = quotient
      v_mov_b32_e32(v[1], 1.0),   # S1 = denominator
      v_mov_b32_e32(v[2], s[0]), # S2 = numerator = NaN
      v_div_fixup_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    import math
    self.assertTrue(math.isnan(i2f(st.vgpr[0][3])), "Should be NaN")

  def test_div_fixup_f32_nan_denom(self):
    """V_DIV_FIXUP_F32: NaN denominator -> quiet NaN."""
    nan = 0x7fc00000  # quiet NaN
    instructions = [
      s_mov_b32(s[0], nan),
      v_mov_b32_e32(v[0], 1.0),   # S0 = quotient
      v_mov_b32_e32(v[1], s[0]), # S1 = denominator = NaN
      v_mov_b32_e32(v[2], 1.0),   # S2 = numerator
      v_div_fixup_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    import math
    self.assertTrue(math.isnan(i2f(st.vgpr[0][3])), "Should be NaN")

  def test_div_fixup_f32_zero_div_zero(self):
    """V_DIV_FIXUP_F32: 0/0 -> NaN (0xffc00000)."""
    instructions = [
      v_mov_b32_e32(v[0], 1.0),   # S0 = quotient (doesn't matter)
      v_mov_b32_e32(v[1], 0.0),   # S1 = denominator = 0
      v_mov_b32_e32(v[2], 0.0),   # S2 = numerator = 0
      v_div_fixup_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    import math
    self.assertTrue(math.isnan(i2f(st.vgpr[0][3])), "0/0 should be NaN")

  def test_div_fixup_f32_inf_div_inf(self):
    """V_DIV_FIXUP_F32: inf/inf -> NaN."""
    pos_inf = 0x7f800000
    instructions = [
      s_mov_b32(s[0], pos_inf),
      v_mov_b32_e32(v[0], 1.0),   # S0 = quotient
      v_mov_b32_e32(v[1], s[0]), # S1 = denominator = +inf
      v_mov_b32_e32(v[2], s[0]), # S2 = numerator = +inf
      v_div_fixup_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    import math
    self.assertTrue(math.isnan(i2f(st.vgpr[0][3])), "inf/inf should be NaN")

  def test_div_fixup_f32_x_div_zero(self):
    """V_DIV_FIXUP_F32: x/0 -> +/-inf based on sign."""
    instructions = [
      v_mov_b32_e32(v[0], 1.0),   # S0 = quotient
      v_mov_b32_e32(v[1], 0.0),   # S1 = denominator = 0
      v_mov_b32_e32(v[2], 1.0),   # S2 = numerator = 1.0
      v_div_fixup_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    import math
    self.assertTrue(math.isinf(i2f(st.vgpr[0][3])), "x/0 should be inf")
    self.assertGreater(i2f(st.vgpr[0][3]), 0, "1/0 should be +inf")

  def test_div_fixup_f32_neg_x_div_zero(self):
    """V_DIV_FIXUP_F32: -x/0 -> -inf."""
    instructions = [
      v_mov_b32_e32(v[0], 1.0),   # S0 = quotient
      v_mov_b32_e32(v[1], 0.0),   # S1 = denominator = 0
      v_mov_b32_e32(v[2], -1.0),  # S2 = numerator = -1.0
      v_div_fixup_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    import math
    self.assertTrue(math.isinf(i2f(st.vgpr[0][3])), "-x/0 should be inf")
    self.assertLess(i2f(st.vgpr[0][3]), 0, "-1/0 should be -inf")

  def test_div_fixup_f32_zero_div_x(self):
    """V_DIV_FIXUP_F32: 0/x -> 0."""
    instructions = [
      v_mov_b32_e32(v[0], 1.0),   # S0 = quotient
      v_mov_b32_e32(v[1], 2.0),   # S1 = denominator = 2.0
      v_mov_b32_e32(v[2], 0.0),   # S2 = numerator = 0
      v_div_fixup_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(i2f(st.vgpr[0][3]), 0.0, "0/x should be 0")

  def test_div_fixup_f32_x_div_inf(self):
    """V_DIV_FIXUP_F32: x/inf -> 0."""
    pos_inf = 0x7f800000
    instructions = [
      s_mov_b32(s[0], pos_inf),
      v_mov_b32_e32(v[0], 1.0),   # S0 = quotient
      v_mov_b32_e32(v[1], s[0]), # S1 = denominator = +inf
      v_mov_b32_e32(v[2], 1.0),   # S2 = numerator = 1.0
      v_div_fixup_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(i2f(st.vgpr[0][3]), 0.0, "x/inf should be 0")

  def test_div_fixup_f32_inf_div_x(self):
    """V_DIV_FIXUP_F32: inf/x -> inf."""
    pos_inf = 0x7f800000
    instructions = [
      s_mov_b32(s[0], pos_inf),
      v_mov_b32_e32(v[0], 1.0),   # S0 = quotient
      v_mov_b32_e32(v[1], 1.0),   # S1 = denominator = 1.0
      v_mov_b32_e32(v[2], s[0]), # S2 = numerator = +inf
      v_div_fixup_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    import math
    self.assertTrue(math.isinf(i2f(st.vgpr[0][3])), "inf/x should be inf")

  def test_div_fixup_f32_sign_propagation(self):
    """V_DIV_FIXUP_F32: sign is XOR of numer and denom signs."""
    instructions = [
      v_mov_b32_e32(v[0], 3.0),   # S0 = |quotient|
      v_mov_b32_e32(v[1], -2.0),  # S1 = denominator (negative)
      v_mov_b32_e32(v[2], 6.0),   # S2 = numerator (positive)
      v_div_fixup_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    # pos / neg = neg
    self.assertAlmostEqual(i2f(st.vgpr[0][3]), -3.0, places=5)

  def test_div_fixup_f32_neg_neg(self):
    """V_DIV_FIXUP_F32: neg/neg -> positive."""
    instructions = [
      v_mov_b32_e32(v[0], 3.0),   # S0 = |quotient|
      v_mov_b32_e32(v[1], -2.0),  # S1 = denominator (negative)
      v_mov_b32_e32(v[2], -6.0),  # S2 = numerator (negative)
      v_div_fixup_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    # neg / neg = pos
    self.assertAlmostEqual(i2f(st.vgpr[0][3]), 3.0, places=5)

  def test_div_fixup_f32_nan_estimate_overflow(self):
    """V_DIV_FIXUP_F32: NaN estimate returns overflow (inf).

    PDF doesn't check isNAN(S0), but hardware returns OVERFLOW if S0 is NaN.
    This happens when division fails (e.g., denorm denominator in V_DIV_SCALE).
    """
    quiet_nan = 0x7fc00000
    instructions = [
      s_mov_b32(s[0], quiet_nan),
      v_mov_b32_e32(v[0], s[0]),  # S0 = NaN (failed estimate)
      v_mov_b32_e32(v[1], 1.0),   # S1 = denominator = 1.0
      v_mov_b32_e32(v[2], 1.0),   # S2 = numerator = 1.0
      v_div_fixup_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    import math
    self.assertTrue(math.isinf(i2f(st.vgpr[0][3])), "NaN estimate should return inf")
    self.assertEqual(st.vgpr[0][3], 0x7f800000, "Should be +inf (pos/pos)")

  def test_div_fixup_f32_nan_estimate_sign(self):
    """V_DIV_FIXUP_F32: NaN estimate with negative sign returns -inf."""
    quiet_nan = 0x7fc00000
    instructions = [
      s_mov_b32(s[0], quiet_nan),
      v_mov_b32_e32(v[0], s[0]),  # S0 = NaN (failed estimate)
      v_mov_b32_e32(v[1], -1.0),  # S1 = denominator = -1.0
      v_mov_b32_e32(v[2], 1.0),   # S2 = numerator = 1.0
      v_div_fixup_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    import math
    self.assertTrue(math.isinf(i2f(st.vgpr[0][3])), "NaN estimate should return inf")
    self.assertEqual(st.vgpr[0][3], 0xff800000, "Should be -inf (pos/neg)")


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


class TestLaneInstructions(unittest.TestCase):
  """Tests for cross-lane instructions (readlane, writelane, readfirstlane).

  These are critical for wave-level reductions and WMMA matrix operations.

  Note: V_READLANE_B32 and V_READFIRSTLANE_B32 write to SGPR, but the VOP1/VOP3
  encoding has a 'vdst' field. We use RawImm to encode SGPR indices directly.
  """

  def _readlane(self, sdst_idx, vsrc, lane_idx):
    """Helper to create V_READLANE_B32 with SGPR destination."""
    return VOP3(VOP3Op.V_READLANE_B32, vdst=RawImm(sdst_idx), src0=vsrc, src1=lane_idx)

  def _readfirstlane(self, sdst_idx, vsrc):
    """Helper to create V_READFIRSTLANE_B32 with SGPR destination."""
    return VOP1(VOP1Op.V_READFIRSTLANE_B32, vdst=RawImm(sdst_idx), src0=vsrc)

  def test_v_readlane_b32_basic(self):
    """V_READLANE_B32 reads a value from a specific lane's VGPR."""
    # v[255] = lane_id from prologue; compute v[0] = lane_id * 10
    instructions = [
      v_lshlrev_b32_e32(v[0], 1, v[255]),  # v0 = lane_id * 2
      v_lshlrev_b32_e32(v[1], 3, v[255]),  # v1 = lane_id * 8
      v_add_nc_u32_e32(v[0], v[0], v[1]),  # v0 = lane_id * 10
      # Now read lane 2's value (should be 20) into s0
      self._readlane(0, v[0], 2),          # s0 = v0 from lane 2 = 20
      v_mov_b32_e32(v[2], s[0]),           # broadcast to all lanes
    ]
    st = run_program(instructions, n_lanes=4)
    # All lanes should have the value 20 (lane 2's value)
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][2], 20, f"Lane {lane}: expected 20, got {st.vgpr[lane][2]}")

  def test_v_readlane_b32_lane_0(self):
    """V_READLANE_B32 reading from lane 0."""
    instructions = [
      v_lshlrev_b32_e32(v[0], 2, v[255]),  # v0 = lane_id * 4
      v_add_nc_u32_e32(v[0], 100, v[0]),   # v0 = 100 + lane_id * 4
      self._readlane(0, v[0], 0),          # s0 = lane 0's v0 = 100
      v_mov_b32_e32(v[1], s[0]),
    ]
    st = run_program(instructions, n_lanes=4)
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][1], 100)

  def test_v_readlane_b32_last_lane(self):
    """V_READLANE_B32 reading from the last active lane (lane 3 in 4-lane test)."""
    instructions = [
      v_lshlrev_b32_e32(v[0], 2, v[255]),  # v0 = lane_id * 4
      v_add_nc_u32_e32(v[0], 100, v[0]),   # v0 = 100 + lane_id * 4
      self._readlane(0, v[0], 3),          # s0 = lane 3's v0 = 112
      v_mov_b32_e32(v[1], s[0]),
    ]
    st = run_program(instructions, n_lanes=4)
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][1], 112)

  def test_v_readlane_b32_different_vgpr(self):
    """V_READLANE_B32 reading from different VGPR indices.

    Regression test for bug where rd_lane was checked against VGPR values
    instead of being used as an index (using 'in' operator on list instead
    of checking if index is within bounds).
    """
    instructions = [
      # Set up v[5] with per-lane values
      v_lshlrev_b32_e32(v[5], 3, v[255]),  # v5 = lane_id * 8
      v_add_nc_u32_e32(v[5], 50, v[5]),    # v5 = 50 + lane_id * 8
      # Read lane 1's v[5] (should be 58)
      self._readlane(0, v[5], 1),
      v_mov_b32_e32(v[6], s[0]),
    ]
    st = run_program(instructions, n_lanes=4)
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][6], 58, f"Lane {lane}: expected 58 from v[5] lane 1")

  def test_v_readfirstlane_b32_basic(self):
    """V_READFIRSTLANE_B32 reads from the first active lane."""
    instructions = [
      v_lshlrev_b32_e32(v[0], 2, v[255]),  # v0 = lane_id * 4
      v_add_nc_u32_e32(v[0], 1000, v[0]),  # v0 = 1000 + lane_id * 4
      self._readfirstlane(0, v[0]),        # s0 = first lane's v0 = 1000
      v_mov_b32_e32(v[1], s[0]),
    ]
    st = run_program(instructions, n_lanes=4)
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][1], 1000)

  def test_v_readfirstlane_b32_different_vgpr(self):
    """V_READFIRSTLANE_B32 reading from different VGPR index.

    Regression test for bug where src0_idx bounds check was incorrect.
    """
    instructions = [
      v_lshlrev_b32_e32(v[7], 5, v[255]),  # v7 = lane_id * 32
      v_add_nc_u32_e32(v[7], 200, v[7]),   # v7 = 200 + lane_id * 32
      self._readfirstlane(0, v[7]),        # s0 = first lane's v7 = 200
      v_mov_b32_e32(v[8], s[0]),
    ]
    st = run_program(instructions, n_lanes=4)
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][8], 200)

  def test_v_writelane_b32_basic(self):
    """V_WRITELANE_B32 writes a scalar to a specific lane's VGPR."""
    instructions = [
      v_mov_b32_e32(v[0], 0),              # Initialize v0 = 0 for all lanes
      s_mov_b32(s[0], 999),                # Value to write
      v_writelane_b32(v[0], s[0], 2),      # Write 999 to lane 2's v0
    ]
    st = run_program(instructions, n_lanes=4)
    for lane in range(4):
      if lane == 2:
        self.assertEqual(st.vgpr[lane][0], 999, f"Lane 2 should have 999")
      else:
        self.assertEqual(st.vgpr[lane][0], 0, f"Lane {lane} should have 0")

  def test_v_writelane_then_readlane(self):
    """V_WRITELANE followed by V_READLANE to verify round-trip."""
    instructions = [
      v_mov_b32_e32(v[0], 0),
      s_mov_b32(s[0], 0xdeadbeef),
      v_writelane_b32(v[0], s[0], 1),      # Write to lane 1
      self._readlane(1, v[0], 1),          # Read back from lane 1 into s1
      v_mov_b32_e32(v[1], s[1]),
    ]
    st = run_program(instructions, n_lanes=4)
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][1], 0xdeadbeef)

  def test_v_readlane_for_reduction(self):
    """Simulate a wave reduction using readlane - common pattern in WMMA/reductions.

    This pattern is used when reducing across lanes, e.g., for computing
    the sum of all elements in a wave.
    """
    # Each lane computes lane_id + 1, then we sum lanes 0-3 using readlane
    instructions = [
      v_add_nc_u32_e32(v[0], 1, v[255]),   # v0 = lane_id + 1 (1, 2, 3, 4)
      # Read all 4 lanes and sum in scalar registers
      self._readlane(0, v[0], 0),          # s0 = 1
      self._readlane(1, v[0], 1),          # s1 = 2
      s_add_u32(s[0], s[0], s[1]),         # s0 = 3
      self._readlane(1, v[0], 2),          # s1 = 3
      s_add_u32(s[0], s[0], s[1]),         # s0 = 6
      self._readlane(1, v[0], 3),          # s1 = 4
      s_add_u32(s[0], s[0], s[1]),         # s0 = 10
      v_mov_b32_e32(v[1], s[0]),           # Broadcast sum to all lanes
    ]
    st = run_program(instructions, n_lanes=4)
    for lane in range(4):
      self.assertEqual(st.vgpr[lane][1], 10, f"Sum 1+2+3+4 should be 10")


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


class TestFMA(unittest.TestCase):
  """Tests for FMA instructions - key for OCML sin argument reduction."""

  def test_v_fma_f32_basic(self):
    """V_FMA_F32: a*b+c basic case using inline constants only."""
    # Inline float constants: 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0
    instructions = [
      v_mov_b32_e32(v[0], 2.0),  # inline constant
      v_mov_b32_e32(v[1], 4.0),  # inline constant
      v_mov_b32_e32(v[2], 1.0),  # inline constant
      v_fma_f32(v[3], v[0], v[1], v[2]),  # 2*4+1 = 9
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][3]), 9.0, places=5)

  def test_v_fma_f32_negative(self):
    """V_FMA_F32 with negative multiplier (used in sin reduction)."""
    instructions = [
      v_mov_b32_e32(v[0], -2.0),  # inline constant
      v_mov_b32_e32(v[1], 4.0),   # inline constant
      v_mov_b32_e32(v[2], 1.0),   # inline constant
      v_fma_f32(v[3], v[0], v[1], v[2]),  # -2*4+1 = -7
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][3]), -7.0, places=5)

  def test_v_fmac_f32(self):
    """V_FMAC_F32: d = d + a*b using inline constants."""
    instructions = [
      v_mov_b32_e32(v[0], 2.0),  # inline constant
      v_mov_b32_e32(v[1], 4.0),  # inline constant
      v_mov_b32_e32(v[2], 1.0),  # inline constant
      v_fmac_f32_e32(v[2], v[0], v[1]),  # v2 = v2 + v0*v1 = 1 + 2*4 = 9
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 9.0, places=5)

  def test_v_fmaak_f32(self):
    """V_FMAAK_F32: d = a * b + K using inline constants."""
    instructions = [
      v_mov_b32_e32(v[0], 2.0),  # inline constant
      v_mov_b32_e32(v[1], 4.0),  # inline constant
      v_fmaak_f32_e32(v[2], v[0], v[1], 0x3f800000),  # v2 = v0 * v1 + 1.0 = 2*4+1 = 9
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][2]), 9.0, places=5)

  def test_v_fma_f32_with_sgpr(self):
    """V_FMA_F32: using SGPR for non-inline constant."""
    # Use SGPR to load 3.0 which is not an inline constant
    instructions = [
      s_mov_b32(s[0], f2i(3.0)),  # 3.0 via literal in SGPR
      v_mov_b32_e32(v[0], 2.0),   # inline constant
      v_mov_b32_e32(v[1], s[0]),  # 3.0 from SGPR
      v_mov_b32_e32(v[2], 4.0),   # inline constant
      v_fma_f32(v[3], v[0], v[1], v[2]),  # 2*3+4 = 10
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][3]), 10.0, places=5)


class TestRounding(unittest.TestCase):
  """Tests for rounding instructions - used in sin argument reduction."""

  def test_v_rndne_f32_half_even(self):
    """V_RNDNE_F32 rounds to nearest even."""
    instructions = [
      s_mov_b32(s[0], f2i(2.5)),
      v_mov_b32_e32(v[0], s[0]),
      v_rndne_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 2.0, places=5)  # rounds to even

  def test_v_rndne_f32_half_odd(self):
    """V_RNDNE_F32 rounds 3.5 to 4 (nearest even)."""
    instructions = [
      s_mov_b32(s[0], f2i(3.5)),
      v_mov_b32_e32(v[0], s[0]),
      v_rndne_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 4.0, places=5)

  def test_v_rndne_f32_large(self):
    """V_RNDNE_F32 with large value (like sin reduction uses)."""
    # sin(1e5) reduction: 1e5 * (1/2pi) ≈ 15915.49...
    val = 100000.0 * 0.15915494309189535  # 1/(2*pi)
    instructions = [
      s_mov_b32(s[0], f2i(val)),
      v_mov_b32_e32(v[0], s[0]),
      v_rndne_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    expected = round(val)  # Python's round does banker's rounding
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), expected, places=0)

  def test_v_floor_f32(self):
    """V_FLOOR_F32 floors to integer."""
    instructions = [
      s_mov_b32(s[0], f2i(3.7)),
      v_mov_b32_e32(v[0], s[0]),
      v_floor_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 3.0, places=5)

  def test_v_trunc_f32(self):
    """V_TRUNC_F32 truncates toward zero."""
    instructions = [
      s_mov_b32(s[0], f2i(-3.7)),
      v_mov_b32_e32(v[0], s[0]),
      v_trunc_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), -3.0, places=5)

  def test_v_fract_f32(self):
    """V_FRACT_F32 returns fractional part."""
    instructions = [
      s_mov_b32(s[0], f2i(3.75)),
      v_mov_b32_e32(v[0], s[0]),
      v_fract_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 0.75, places=5)

  def test_v_fract_f32_large(self):
    """V_FRACT_F32 with large value - precision matters here."""
    instructions = [
      s_mov_b32(s[0], f2i(132000.25)),
      v_mov_b32_e32(v[0], s[0]),
      v_fract_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][1])
    # For large floats, fract precision degrades
    self.assertGreaterEqual(result, 0.0)
    self.assertLess(result, 1.0)


class TestConversion(unittest.TestCase):
  """Tests for conversion instructions."""

  def test_v_cvt_i32_f32_positive(self):
    """V_CVT_I32_F32 converts float to signed int."""
    instructions = [
      s_mov_b32(s[0], f2i(42.7)),
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_i32_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 42)

  def test_v_cvt_i32_f32_negative(self):
    """V_CVT_I32_F32 converts negative float to signed int."""
    instructions = [
      s_mov_b32(s[0], f2i(-42.7)),
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_i32_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    # Result is signed, stored as unsigned
    self.assertEqual(st.vgpr[0][1] & 0xffffffff, (-42) & 0xffffffff)

  def test_v_cvt_i32_f32_large(self):
    """V_CVT_I32_F32 with large float (used in sin for quadrant)."""
    # sin reduction converts round(x * 1/2pi) to int for quadrant selection
    instructions = [
      s_mov_b32(s[0], f2i(15915.0)),  # ~1e5 / (2*pi)
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_i32_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 15915)

  def test_v_cvt_f32_i32(self):
    """V_CVT_F32_I32 converts signed int to float."""
    instructions = [
      s_mov_b32(s[0], 42),
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_f32_i32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 42.0, places=5)

  def test_v_cvt_f32_u32(self):
    """V_CVT_F32_U32 converts unsigned int to float."""
    instructions = [
      s_mov_b32(s[0], 0xffffffff),  # max u32
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_f32_u32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 4294967296.0, places=-5)


class TestBitManipulation(unittest.TestCase):
  """Tests for bit manipulation - used in sin for quadrant selection."""

  def test_v_and_b32(self):
    """V_AND_B32 bitwise and."""
    instructions = [
      s_mov_b32(s[0], 0xff),
      s_mov_b32(s[1], 0x0f),
      v_mov_b32_e32(v[0], s[0]),
      v_and_b32_e32(v[1], s[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0x0f)

  def test_v_and_b32_quadrant(self):
    """V_AND_B32 for quadrant extraction (n & 3)."""
    instructions = [
      s_mov_b32(s[0], 15915),  # some large number
      v_mov_b32_e32(v[0], s[0]),
      v_and_b32_e32(v[1], 3, v[0]),  # n & 3 for quadrant
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 15915 & 3)

  def test_v_lshrrev_b32(self):
    """V_LSHRREV_B32 logical shift right."""
    instructions = [
      s_mov_b32(s[0], 0xff00),
      v_mov_b32_e32(v[0], s[0]),
      v_lshrrev_b32_e32(v[1], 8, v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0xff)

  def test_v_lshlrev_b32(self):
    """V_LSHLREV_B32 logical shift left."""
    instructions = [
      s_mov_b32(s[0], 0xff),
      v_mov_b32_e32(v[0], s[0]),
      v_lshlrev_b32_e32(v[1], 8, v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0xff00)

  def test_v_xor_b32(self):
    """V_XOR_B32 bitwise xor (used in sin for sign)."""
    instructions = [
      s_mov_b32(s[0], 0x80000000),  # sign bit
      s_mov_b32(s[1], f2i(1.0)),
      v_mov_b32_e32(v[0], s[1]),
      v_xor_b32_e32(v[1], s[0], v[0]),  # flip sign
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), -1.0, places=5)


class TestOCMLSinSequence(unittest.TestCase):
  """Test the specific instruction sequence used in OCML sin."""

  def test_sin_reduction_step1_mul(self):
    """First step: v12 = |x| * (1/2pi)."""
    import math
    one_over_2pi = 1.0 / (2.0 * math.pi)  # 0x3e22f983 in hex
    x = 100000.0
    instructions = [
      s_mov_b32(s[0], f2i(x)),
      s_mov_b32(s[1], f2i(one_over_2pi)),
      v_mov_b32_e32(v[0], s[0]),
      v_mul_f32_e32(v[1], s[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][1])
    expected = x * one_over_2pi
    self.assertAlmostEqual(result, expected, places=0)

  def test_sin_reduction_step2_round(self):
    """Second step: round to nearest integer."""
    import math
    one_over_2pi = 1.0 / (2.0 * math.pi)
    x = 100000.0
    val = x * one_over_2pi  # ~15915.49
    instructions = [
      s_mov_b32(s[0], f2i(val)),
      v_mov_b32_e32(v[0], s[0]),
      v_rndne_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][1])
    expected = round(val)
    self.assertAlmostEqual(result, expected, places=0)

  def test_sin_reduction_step3_fma(self):
    """Third step: x - n * (pi/2) via FMA."""
    import math
    # This is where precision matters - the FMA does: |x| + (-pi/2) * n
    neg_half_pi = -math.pi / 2.0  # 0xbfc90fda
    x = 100000.0
    n = 15915.0
    instructions = [
      s_mov_b32(s[0], f2i(neg_half_pi)),
      s_mov_b32(s[1], f2i(n)),
      s_mov_b32(s[2], f2i(x)),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]),
      v_fma_f32(v[3], v[0], v[1], v[2]),  # x + (-pi/2) * n
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][3])
    expected = x + neg_half_pi * n
    # Allow some tolerance due to float precision
    self.assertAlmostEqual(result, expected, places=2)

  def test_sin_1e5_full_reduction(self):
    """Full reduction sequence for sin(1e5)."""
    import math
    x = 100000.0
    one_over_2pi = 1.0 / (2.0 * math.pi)
    neg_half_pi = -math.pi / 2.0

    instructions = [
      # Load constants
      s_mov_b32(s[0], f2i(x)),
      s_mov_b32(s[1], f2i(one_over_2pi)),
      s_mov_b32(s[2], f2i(neg_half_pi)),
      # Step 1: v1 = x * (1/2pi)
      v_mov_b32_e32(v[0], s[0]),
      v_mul_f32_e32(v[1], s[1], v[0]),
      # Step 2: v2 = round(v1)
      v_rndne_f32_e32(v[2], v[1]),
      # Step 3: v3 = x + (-pi/2) * round_val (FMA)
      v_fma_f32(v[3], s[2], v[2], v[0]),
      # Step 4: convert to int for quadrant
      v_cvt_i32_f32_e32(v[4], v[2]),
      # Step 5: quadrant = n & 3
      v_and_b32_e32(v[5], 3, v[4]),
    ]
    st = run_program(instructions, n_lanes=1)

    # Check intermediate values
    mul_result = i2f(st.vgpr[0][1])
    round_result = i2f(st.vgpr[0][2])
    reduced = i2f(st.vgpr[0][3])
    quadrant = st.vgpr[0][5]

    # Verify results match expected
    expected_mul = x * one_over_2pi
    expected_round = round(expected_mul)
    expected_reduced = x + neg_half_pi * expected_round
    expected_quadrant = int(expected_round) & 3

    self.assertAlmostEqual(mul_result, expected_mul, places=0, msg=f"mul: got {mul_result}, expected {expected_mul}")
    self.assertAlmostEqual(round_result, expected_round, places=0, msg=f"round: got {round_result}, expected {expected_round}")
    self.assertEqual(quadrant, expected_quadrant, f"quadrant: got {quadrant}, expected {expected_quadrant}")


class TestMad64(unittest.TestCase):
  """Tests for V_MAD_U64_U32 - critical for OCML Payne-Hanek sin reduction."""

  def test_v_mad_u64_u32_simple(self):
    """V_MAD_U64_U32: D = S0 * S1 + S2 (64-bit result)."""
    # 3 * 4 + 5 = 17
    instructions = [
      s_mov_b32(s[0], 3),
      s_mov_b32(s[1], 4),
      v_mov_b32_e32(v[2], 5),  # S2 lo
      v_mov_b32_e32(v[3], 0),  # S2 hi
      v_mad_u64_u32(v[4], SrcEnum.NULL, s[0], s[1], v[2]),  # result in v[4:5]
    ]
    st = run_program(instructions, n_lanes=1)
    result_lo = st.vgpr[0][4]
    result_hi = st.vgpr[0][5]
    result = result_lo | (result_hi << 32)
    self.assertEqual(result, 17)

  def test_v_mad_u64_u32_large_mult(self):
    """V_MAD_U64_U32 with large values that overflow 32 bits."""
    # 0x80000000 * 2 + 0 = 0x100000000
    instructions = [
      s_mov_b32(s[0], 0x80000000),
      s_mov_b32(s[1], 2),
      v_mov_b32_e32(v[2], 0),
      v_mov_b32_e32(v[3], 0),
      v_mad_u64_u32(v[4], SrcEnum.NULL, s[0], s[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result_lo = st.vgpr[0][4]
    result_hi = st.vgpr[0][5]
    result = result_lo | (result_hi << 32)
    self.assertEqual(result, 0x100000000)

  def test_v_mad_u64_u32_with_add(self):
    """V_MAD_U64_U32 with 64-bit addend."""
    # 1000 * 1000 + 0x100000000 = 1000000 + 0x100000000 = 0x1000F4240
    instructions = [
      s_mov_b32(s[0], 1000),
      s_mov_b32(s[1], 1000),
      v_mov_b32_e32(v[2], 0),  # S2 lo
      v_mov_b32_e32(v[3], 1),  # S2 hi = 0x100000000
      v_mad_u64_u32(v[4], SrcEnum.NULL, s[0], s[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result_lo = st.vgpr[0][4]
    result_hi = st.vgpr[0][5]
    result = result_lo | (result_hi << 32)
    expected = 1000 * 1000 + 0x100000000
    self.assertEqual(result, expected)

  def test_v_mad_u64_u32_max_values(self):
    """V_MAD_U64_U32 with max u32 values."""
    # 0xFFFFFFFF * 0xFFFFFFFF + 0 = 0xFFFFFFFE00000001
    instructions = [
      s_mov_b32(s[0], 0xFFFFFFFF),
      s_mov_b32(s[1], 0xFFFFFFFF),
      v_mov_b32_e32(v[2], 0),
      v_mov_b32_e32(v[3], 0),
      v_mad_u64_u32(v[4], SrcEnum.NULL, s[0], s[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result_lo = st.vgpr[0][4]
    result_hi = st.vgpr[0][5]
    result = result_lo | (result_hi << 32)
    expected = 0xFFFFFFFF * 0xFFFFFFFF
    self.assertEqual(result, expected)


class TestClz(unittest.TestCase):
  """Tests for V_CLZ_I32_U32 - count leading zeros, used in Payne-Hanek."""

  def test_v_clz_i32_u32_zero(self):
    """V_CLZ_I32_U32 of 0 returns -1 (all bits are 0)."""
    instructions = [
      v_mov_b32_e32(v[0], 0),
      v_clz_i32_u32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    # -1 as unsigned 32-bit
    self.assertEqual(st.vgpr[0][1], 0xFFFFFFFF)

  def test_v_clz_i32_u32_one(self):
    """V_CLZ_I32_U32 of 1 returns 31 (31 leading zeros)."""
    instructions = [
      v_mov_b32_e32(v[0], 1),
      v_clz_i32_u32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 31)

  def test_v_clz_i32_u32_msb_set(self):
    """V_CLZ_I32_U32 of 0x80000000 returns 0 (no leading zeros)."""
    instructions = [
      s_mov_b32(s[0], 0x80000000),
      v_mov_b32_e32(v[0], s[0]),
      v_clz_i32_u32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0)

  def test_v_clz_i32_u32_half(self):
    """V_CLZ_I32_U32 of 0x8000 (bit 15) returns 16."""
    instructions = [
      s_mov_b32(s[0], 0x8000),
      v_mov_b32_e32(v[0], s[0]),
      v_clz_i32_u32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 16)

  def test_v_clz_i32_u32_all_ones(self):
    """V_CLZ_I32_U32 of 0xFFFFFFFF returns 0."""
    instructions = [
      s_mov_b32(s[0], 0xFFFFFFFF),
      v_mov_b32_e32(v[0], s[0]),
      v_clz_i32_u32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0)


class TestCtz(unittest.TestCase):
  """Tests for V_CTZ_I32_B32 - count trailing zeros."""

  def test_v_ctz_i32_b32_zero(self):
    """V_CTZ_I32_B32 of 0 returns -1 (all bits are 0)."""
    instructions = [
      v_mov_b32_e32(v[0], 0),
      v_ctz_i32_b32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0xFFFFFFFF)

  def test_v_ctz_i32_b32_one(self):
    """V_CTZ_I32_B32 of 1 returns 0 (no trailing zeros)."""
    instructions = [
      v_mov_b32_e32(v[0], 1),
      v_ctz_i32_b32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0)

  def test_v_ctz_i32_b32_msb_set(self):
    """V_CTZ_I32_B32 of 0x80000000 returns 31."""
    instructions = [
      s_mov_b32(s[0], 0x80000000),
      v_mov_b32_e32(v[0], s[0]),
      v_ctz_i32_b32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 31)

  def test_v_ctz_i32_b32_half(self):
    """V_CTZ_I32_B32 of 0x8000 (bit 15) returns 15."""
    instructions = [
      s_mov_b32(s[0], 0x8000),
      v_mov_b32_e32(v[0], s[0]),
      v_ctz_i32_b32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 15)

  def test_v_ctz_i32_b32_all_ones(self):
    """V_CTZ_I32_B32 of 0xFFFFFFFF returns 0."""
    instructions = [
      s_mov_b32(s[0], 0xFFFFFFFF),
      v_mov_b32_e32(v[0], s[0]),
      v_ctz_i32_b32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0)


class TestDivision(unittest.TestCase):
  """Tests for division instructions - V_RCP, V_DIV_SCALE, V_DIV_FMAS, V_DIV_FIXUP."""

  def test_v_rcp_f32_normal(self):
    """V_RCP_F32 of 2.0 returns 0.5."""
    instructions = [
      v_mov_b32_e32(v[0], 2.0),
      v_rcp_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 0.5, places=5)

  def test_v_rcp_f32_inf(self):
    """V_RCP_F32 of +inf returns 0."""
    instructions = [
      s_mov_b32(s[0], 0x7f800000),  # +inf
      v_mov_b32_e32(v[0], s[0]),
      v_rcp_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(i2f(st.vgpr[0][1]), 0.0)

  def test_v_rcp_f32_neg_inf(self):
    """V_RCP_F32 of -inf returns -0."""
    instructions = [
      s_mov_b32(s[0], 0xff800000),  # -inf
      v_mov_b32_e32(v[0], s[0]),
      v_rcp_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][1])
    self.assertEqual(result, 0.0)
    # Check it's negative zero
    self.assertEqual(st.vgpr[0][1], 0x80000000)

  def test_v_rcp_f32_zero(self):
    """V_RCP_F32 of 0 returns +inf."""
    instructions = [
      v_mov_b32_e32(v[0], 0),
      v_rcp_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    import math
    self.assertTrue(math.isinf(i2f(st.vgpr[0][1])))

  def test_v_div_fixup_f32_normal(self):
    """V_DIV_FIXUP_F32 normal division 1.0/2.0."""
    # S0 = approximation (from rcp * scale), S1 = denominator, S2 = numerator
    instructions = [
      s_mov_b32(s[0], f2i(0.5)),   # approximation
      s_mov_b32(s[1], f2i(2.0)),   # denominator
      s_mov_b32(s[2], f2i(1.0)),   # numerator
      v_mov_b32_e32(v[0], s[0]),
      v_div_fixup_f32(v[1], v[0], s[1], s[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertAlmostEqual(i2f(st.vgpr[0][1]), 0.5, places=5)

  def test_v_div_fixup_f32_one_div_inf(self):
    """V_DIV_FIXUP_F32: 1.0 / +inf = 0."""
    # For x/inf: S0=approx(~0), S1=inf, S2=x
    instructions = [
      s_mov_b32(s[0], 0),           # approximation (rcp of inf = 0)
      s_mov_b32(s[1], 0x7f800000),  # denominator = +inf
      s_mov_b32(s[2], f2i(1.0)),    # numerator = 1.0
      v_mov_b32_e32(v[0], s[0]),
      v_div_fixup_f32(v[1], v[0], s[1], s[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(i2f(st.vgpr[0][1]), 0.0)

  def test_v_div_fixup_f32_one_div_neg_inf(self):
    """V_DIV_FIXUP_F32: 1.0 / -inf = -0."""
    instructions = [
      s_mov_b32(s[0], 0x80000000),  # approximation (rcp of -inf = -0)
      s_mov_b32(s[1], 0xff800000),  # denominator = -inf
      s_mov_b32(s[2], f2i(1.0)),    # numerator = 1.0
      v_mov_b32_e32(v[0], s[0]),
      v_div_fixup_f32(v[1], v[0], s[1], s[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertEqual(st.vgpr[0][1], 0x80000000)  # -0.0

  def test_v_div_fixup_f32_inf_div_inf(self):
    """V_DIV_FIXUP_F32: inf / inf = NaN."""
    import math
    instructions = [
      s_mov_b32(s[0], 0),           # approximation
      s_mov_b32(s[1], 0x7f800000),  # denominator = +inf
      s_mov_b32(s[2], 0x7f800000),  # numerator = +inf
      v_mov_b32_e32(v[0], s[0]),
      v_div_fixup_f32(v[1], v[0], s[1], s[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isnan(i2f(st.vgpr[0][1])))

  def test_v_div_fixup_f32_zero_div_zero(self):
    """V_DIV_FIXUP_F32: 0 / 0 = NaN."""
    import math
    instructions = [
      s_mov_b32(s[0], 0),  # approximation
      s_mov_b32(s[1], 0),  # denominator = 0
      s_mov_b32(s[2], 0),  # numerator = 0
      v_mov_b32_e32(v[0], s[0]),
      v_div_fixup_f32(v[1], v[0], s[1], s[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isnan(i2f(st.vgpr[0][1])))

  def test_v_div_fixup_f32_x_div_zero(self):
    """V_DIV_FIXUP_F32: 1.0 / 0 = +inf."""
    import math
    instructions = [
      s_mov_b32(s[0], 0x7f800000),  # approximation (rcp of 0 = inf)
      s_mov_b32(s[1], 0),           # denominator = 0
      s_mov_b32(s[2], f2i(1.0)),    # numerator = 1.0
      v_mov_b32_e32(v[0], s[0]),
      v_div_fixup_f32(v[1], v[0], s[1], s[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][1])
    self.assertTrue(math.isinf(result) and result > 0)

  def test_v_div_fixup_f32_neg_x_div_zero(self):
    """V_DIV_FIXUP_F32: -1.0 / 0 = -inf."""
    import math
    instructions = [
      s_mov_b32(s[0], 0xff800000),  # approximation (rcp of 0 = inf, with sign)
      s_mov_b32(s[1], 0),           # denominator = 0
      s_mov_b32(s[2], f2i(-1.0)),   # numerator = -1.0
      v_mov_b32_e32(v[0], s[0]),
      v_div_fixup_f32(v[1], v[0], s[1], s[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][1])
    self.assertTrue(math.isinf(result) and result < 0)


class TestSpecialValues(unittest.TestCase):
  """Tests for special float values - inf, nan, zero handling."""

  def test_v_mul_f32_zero_times_inf(self):
    """V_MUL_F32: 0 * inf = NaN."""
    import math
    instructions = [
      v_mov_b32_e32(v[0], 0),
      s_mov_b32(s[0], 0x7f800000),  # +inf
      v_mov_b32_e32(v[1], s[0]),
      v_mul_f32_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isnan(i2f(st.vgpr[0][2])))

  def test_v_add_f32_inf_minus_inf(self):
    """V_ADD_F32: inf + (-inf) = NaN."""
    import math
    instructions = [
      s_mov_b32(s[0], 0x7f800000),  # +inf
      s_mov_b32(s[1], 0xff800000),  # -inf
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_add_f32_e32(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    self.assertTrue(math.isnan(i2f(st.vgpr[0][2])))

  def test_v_fma_f32_with_inf(self):
    """V_FMA_F32: 1.0 * inf + 0 = inf."""
    import math
    instructions = [
      v_mov_b32_e32(v[0], 1.0),
      s_mov_b32(s[0], 0x7f800000),  # +inf
      v_mov_b32_e32(v[1], s[0]),
      v_mov_b32_e32(v[2], 0),
      v_fma_f32(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = i2f(st.vgpr[0][3])
    self.assertTrue(math.isinf(result) and result > 0)

  def test_v_exp_f32_large_negative(self):
    """V_EXP_F32 of large negative value (2^-100) returns very small number."""
    instructions = [
      s_mov_b32(s[0], f2i(-100.0)),
      v_mov_b32_e32(v[0], s[0]),
      v_exp_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    # V_EXP_F32 computes 2^x, so 2^-100 is ~7.9e-31 (very small but not 0)
    result = i2f(st.vgpr[0][1])
    self.assertLess(result, 1e-20)  # Just verify it's very small

  def test_v_exp_f32_large_positive(self):
    """V_EXP_F32 of large positive value (2^100) returns very large number."""
    instructions = [
      s_mov_b32(s[0], f2i(100.0)),
      v_mov_b32_e32(v[0], s[0]),
      v_exp_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    # V_EXP_F32 computes 2^x, so 2^100 is ~1.27e30 (very large)
    result = i2f(st.vgpr[0][1])
    self.assertGreater(result, 1e20)  # Just verify it's very large


class TestF16Conversions(unittest.TestCase):
  """Tests for f16 conversion and packing instructions."""

  def test_v_cvt_f16_f32_basic(self):
    """V_CVT_F16_F32 converts f32 to f16 in low 16 bits."""
    from extra.assembly.amd.pcode import _f16
    instructions = [
      v_mov_b32_e32(v[0], 1.0),  # f32 1.0 = 0x3f800000
      v_cvt_f16_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1]
    # f16 1.0 = 0x3c00, should be in low 16 bits
    lo_bits = result & 0xffff
    self.assertEqual(lo_bits, 0x3c00, f"Expected 0x3c00, got 0x{lo_bits:04x}")

  def test_v_cvt_f16_f32_negative(self):
    """V_CVT_F16_F32 converts negative f32 to f16."""
    from extra.assembly.amd.pcode import _f16
    instructions = [
      v_mov_b32_e32(v[0], -2.0),  # f32 -2.0 = 0xc0000000
      v_cvt_f16_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1]
    lo_bits = result & 0xffff
    # f16 -2.0 = 0xc000
    self.assertEqual(lo_bits, 0xc000, f"Expected 0xc000, got 0x{lo_bits:04x}")

  def test_v_cvt_f16_f32_small(self):
    """V_CVT_F16_F32 converts small f32 value."""
    from extra.assembly.amd.pcode import _f16, f32_to_f16
    instructions = [
      v_mov_b32_e32(v[0], 0.5),
      v_cvt_f16_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1]
    lo_bits = result & 0xffff
    expected = f32_to_f16(0.5)  # Should be 0x3800
    self.assertEqual(lo_bits, expected, f"Expected 0x{expected:04x}, got 0x{lo_bits:04x}")

  def test_v_cvt_f16_f32_preserves_high_bits(self):
    """V_CVT_F16_F32 preserves high 16 bits of destination.

    Hardware verified: V_CVT_F16_F32 only writes to the low 16 bits of the
    destination register, preserving the high 16 bits. This is important for
    the common pattern of converting two f32 values and packing them.
    """
    instructions = [
      s_mov_b32(s[0], 0xdead0000),  # Pre-fill with garbage in high bits
      v_mov_b32_e32(v[1], s[0]),
      v_mov_b32_e32(v[0], 1.0),
      v_cvt_f16_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1]
    hi_bits = (result >> 16) & 0xffff
    lo_bits = result & 0xffff
    self.assertEqual(lo_bits, 0x3c00, f"Low bits should be 0x3c00, got 0x{lo_bits:04x}")
    self.assertEqual(hi_bits, 0xdead, f"High bits should be preserved as 0xdead, got 0x{hi_bits:04x}")

  def test_v_cvt_f16_f32_same_src_dst_preserves_high_bits(self):
    """V_CVT_F16_F32 with same src/dst preserves high bits of source.

    Regression test: When converting v0 in-place (v_cvt_f16_f32 v0, v0),
    the high 16 bits of the original f32 value are preserved in the result.
    For f32 1.0 (0x3f800000), the result should be 0x3f803c00:
    - Low 16 bits: 0x3c00 (f16 1.0)
    - High 16 bits: 0x3f80 (preserved from original f32)
    """
    instructions = [
      v_mov_b32_e32(v[0], 1.0),      # v0 = 0x3f800000
      v_cvt_f16_f32_e32(v[0], v[0]), # convert v0 in-place
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][0]
    # Hardware preserves high bits: 0x3f800000 -> 0x3f803c00
    self.assertEqual(result, 0x3f803c00, f"Expected 0x3f803c00, got 0x{result:08x}")

  def test_v_cvt_f16_f32_reads_full_32bit_source(self):
    """V_CVT_F16_F32 must read full 32-bit f32 source, not just low 16 bits.

    Regression test for a bug where V_CVT_F16_F32 was incorrectly treated as having
    a 16-bit source because '_F16' is in the instruction name. The CVT naming convention
    is V_CVT_DST_SRC, so V_CVT_F16_F32 has a 32-bit f32 source and 16-bit f16 destination.

    The bug caused the emulator to only read the low 16 bits of the source register,
    which would produce wrong results when the significant bits of the f32 value are
    in the upper bits (as they are for most f32 values > 1.0 or < -1.0).
    """
    from extra.assembly.amd.pcode import _f16
    # Use f32 value 1.5 = 0x3fc00000. If only low 16 bits (0x0000) are read, result is wrong.
    # Correct f16 result: 0x3e00 (1.5 in half precision)
    instructions = [
      s_mov_b32(s[0], 0x3fc00000),  # f32 1.5
      v_mov_b32_e32(v[0], s[0]),
      v_cvt_f16_f32_e32(v[1], v[0]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1]
    lo_bits = result & 0xffff
    # f16(1.5) = 0x3e00
    self.assertEqual(lo_bits, 0x3e00, f"Expected f16(1.5)=0x3e00, got 0x{lo_bits:04x} ({_f16(lo_bits)})")

  def test_v_cvt_f16_f32_then_pack_for_wmma(self):
    """Regression test: f32->f16 conversion followed by pack for WMMA input.

    This sequence is used in fused fp16 GEMM kernels where f32 data is loaded,
    converted to f16, packed into pairs, and fed to WMMA instructions.

    The bug was: V_CVT_F16_F32 was treated as having 16-bit source (because '_F16'
    is in the name), causing it to read only low 16 bits of the f32 input.
    This resulted in WMMA receiving zero inputs and producing zero outputs.
    """
    from extra.assembly.amd.pcode import _f16
    # Simulate loading two f32 values and converting/packing for WMMA
    # f32 1.5 = 0x3fc00000, f32 2.5 = 0x40200000
    # After CVT: f16 1.5 = 0x3e00, f16 2.5 = 0x4100
    # After PACK: 0x41003e00 (hi=2.5, lo=1.5)
    instructions = [
      s_mov_b32(s[0], 0x3fc00000),  # f32 1.5
      s_mov_b32(s[1], 0x40200000),  # f32 2.5
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_cvt_f16_f32_e32(v[2], v[0]),  # v2 = f16(1.5) = 0x3e00
      v_cvt_f16_f32_e32(v[3], v[1]),  # v3 = f16(2.5) = 0x4100
      v_pack_b32_f16(v[4], v[2], v[3]),  # v4 = pack(v2, v3) = 0x41003e00
    ]
    st = run_program(instructions, n_lanes=1)

    # Check intermediate CVT results
    v2_lo = st.vgpr[0][2] & 0xffff
    v3_lo = st.vgpr[0][3] & 0xffff
    self.assertEqual(v2_lo, 0x3e00, f"v2 should be f16(1.5)=0x3e00, got 0x{v2_lo:04x} ({_f16(v2_lo)})")
    self.assertEqual(v3_lo, 0x4100, f"v3 should be f16(2.5)=0x4100, got 0x{v3_lo:04x} ({_f16(v3_lo)})")

    # Check packed result
    result = st.vgpr[0][4]
    self.assertEqual(result, 0x41003e00, f"Expected packed 0x41003e00, got 0x{result:08x}")

  def test_v_pack_b32_f16_basic(self):
    """V_PACK_B32_F16 packs two f16 values into one 32-bit register."""
    from extra.assembly.amd.pcode import _f16
    instructions = [
      # First convert two f32 values to f16
      v_mov_b32_e32(v[0], 1.0),   # Will become f16 0x3c00
      v_mov_b32_e32(v[2], -2.0),  # Will become f16 0xc000
      v_cvt_f16_f32_e32(v[1], v[0]),  # v1 low = 0x3c00
      v_cvt_f16_f32_e32(v[3], v[2]),  # v3 low = 0xc000
      # Now pack them: v4 = (v3.f16 << 16) | v1.f16
      v_pack_b32_f16(v[4], v[1], v[3]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][4]
    lo_bits = result & 0xffff
    hi_bits = (result >> 16) & 0xffff
    # Expected: lo=0x3c00 (1.0), hi=0xc000 (-2.0)
    self.assertEqual(lo_bits, 0x3c00, f"Lo should be 0x3c00 (1.0), got 0x{lo_bits:04x} ({_f16(lo_bits)})")
    self.assertEqual(hi_bits, 0xc000, f"Hi should be 0xc000 (-2.0), got 0x{hi_bits:04x} ({_f16(hi_bits)})")

  def test_v_pack_b32_f16_both_positive(self):
    """V_PACK_B32_F16 packs two positive f16 values."""
    from extra.assembly.amd.pcode import _f16
    instructions = [
      v_mov_b32_e32(v[0], 0.5),   # f16 0x3800
      v_mov_b32_e32(v[2], 2.0),   # f16 0x4000
      v_cvt_f16_f32_e32(v[1], v[0]),
      v_cvt_f16_f32_e32(v[3], v[2]),
      v_pack_b32_f16(v[4], v[1], v[3]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][4]
    lo_bits = result & 0xffff
    hi_bits = (result >> 16) & 0xffff
    self.assertEqual(lo_bits, 0x3800, f"Lo should be 0x3800 (0.5), got 0x{lo_bits:04x}")
    self.assertEqual(hi_bits, 0x4000, f"Hi should be 0x4000 (2.0), got 0x{hi_bits:04x}")

  def test_v_pack_b32_f16_zeros(self):
    """V_PACK_B32_F16 packs two zero values."""
    instructions = [
      v_mov_b32_e32(v[0], 0),
      v_mov_b32_e32(v[2], 0),
      v_cvt_f16_f32_e32(v[1], v[0]),
      v_cvt_f16_f32_e32(v[3], v[2]),
      v_pack_b32_f16(v[4], v[1], v[3]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][4]
    self.assertEqual(result, 0, f"Expected 0x00000000, got 0x{result:08x}")


class TestPackInstructions(unittest.TestCase):
  """Tests for pack instructions."""

  def test_v_pack_b32_f16(self):
    """V_PACK_B32_F16 packs two f16 values into one 32-bit register."""
    instructions = []
    # f16 1.0 = 0x3c00, f16 2.0 = 0x4000
    instructions.append(s_mov_b32(s[0], 0x3c00))  # f16 1.0
    instructions.append(s_mov_b32(s[1], 0x4000))  # f16 2.0
    instructions.append(v_mov_b32_e32(v[0], s[0]))
    instructions.append(v_mov_b32_e32(v[1], s[1]))
    # Pack: v[2] = (v[1].f16 << 16) | v[0].f16
    instructions.append(v_pack_b32_f16(v[2], v[0], v[1]))

    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2]
    # Expected: hi=0x4000 (2.0), lo=0x3c00 (1.0) -> 0x40003c00
    self.assertEqual(result, 0x40003c00, f"Expected 0x40003c00, got 0x{result:08x}")

  def test_v_pack_b32_f16_with_cvt(self):
    """V_PACK_B32_F16 after V_CVT_F16_F32 conversions."""
    instructions = []
    # f32 1.0 = 0x3f800000
    instructions.append(s_mov_b32(s[0], 0x3f800000))
    instructions.append(v_mov_b32_e32(v[0], s[0]))  # f32 1.0
    instructions.append(v_mov_b32_e32(v[1], s[0]))  # f32 1.0
    # Convert to f16
    instructions.append(v_cvt_f16_f32_e32(v[2], v[0]))  # v[2].f16 = 1.0
    instructions.append(v_cvt_f16_f32_e32(v[3], v[1]))  # v[3].f16 = 1.0
    # Pack
    instructions.append(v_pack_b32_f16(v[4], v[2], v[3]))

    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][4]
    # Expected: 0x3c003c00 (two f16 1.0 values)
    self.assertEqual(result, 0x3c003c00, f"Expected 0x3c003c00, got 0x{result:08x}")

  def test_v_pack_b32_f16_packed_sources(self):
    """V_PACK_B32_F16 with sources that have packed f16 pairs (both hi and lo used).
    This mimics what happens in matmul kernels where VGPRs contain packed f16 data.
    """
    instructions = []
    # v0 = 0x40003c00 (hi=f16 2.0, lo=f16 1.0)
    # v1 = 0x44004200 (hi=f16 4.0, lo=f16 3.0)
    # V_PACK_B32_F16 with default opsel=0 reads low halves from each source
    # Result should be: hi=v1.lo=0x4200 (3.0), lo=v0.lo=0x3c00 (1.0) -> 0x42003c00
    instructions.append(s_mov_b32(s[0], 0x40003c00))  # packed: hi=2.0, lo=1.0
    instructions.append(s_mov_b32(s[1], 0x44004200))  # packed: hi=4.0, lo=3.0
    instructions.append(v_mov_b32_e32(v[0], s[0]))
    instructions.append(v_mov_b32_e32(v[1], s[1]))
    instructions.append(v_pack_b32_f16(v[2], v[0], v[1]))

    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2]
    # Expected: hi=0x4200 (3.0), lo=0x3c00 (1.0) -> 0x42003c00
    self.assertEqual(result, 0x42003c00, f"Expected 0x42003c00, got 0x{result:08x}")

  def test_v_pack_b32_f16_opsel_hi_hi(self):
    """V_PACK_B32_F16 with opsel=0b0011 to read high halves from both sources.
    This is used when extracting the high f16 values from packed registers.
    """
    # v0 = 0x40003c00 (hi=f16 2.0, lo=f16 1.0)
    # v1 = 0x44004200 (hi=f16 4.0, lo=f16 3.0)
    # With opsel=0b0011: read hi from v0 (0x4000=2.0) and hi from v1 (0x4400=4.0)
    # Result should be: hi=v1.hi=0x4400 (4.0), lo=v0.hi=0x4000 (2.0) -> 0x44004000
    inst = v_pack_b32_f16(v[2], v[0], v[1])
    inst._values['opsel'] = 0b0011  # opsel[0]=1 for src0 hi, opsel[1]=1 for src1 hi

    instructions = [
      s_mov_b32(s[0], 0x40003c00),  # packed: hi=2.0, lo=1.0
      s_mov_b32(s[1], 0x44004200),  # packed: hi=4.0, lo=3.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      inst,
    ]

    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2]
    # Expected: hi=0x4400 (4.0), lo=0x4000 (2.0) -> 0x44004000
    self.assertEqual(result, 0x44004000, f"Expected 0x44004000, got 0x{result:08x}")

  def test_v_pack_b32_f16_opsel_lo_hi(self):
    """V_PACK_B32_F16 with opsel=0b0010 to read lo from src0, hi from src1."""
    # v0 = 0x40003c00 (hi=f16 2.0, lo=f16 1.0)
    # v1 = 0x44004200 (hi=f16 4.0, lo=f16 3.0)
    # With opsel=0b0010: read lo from v0 (0x3c00=1.0), hi from v1 (0x4400=4.0)
    # Result should be: hi=v1.hi=0x4400 (4.0), lo=v0.lo=0x3c00 (1.0) -> 0x44003c00
    inst = v_pack_b32_f16(v[2], v[0], v[1])
    inst._values['opsel'] = 0b0010  # opsel[0]=0 for src0 lo, opsel[1]=1 for src1 hi

    instructions = [
      s_mov_b32(s[0], 0x40003c00),
      s_mov_b32(s[1], 0x44004200),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      inst,
    ]

    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2]
    # Expected: hi=0x4400 (4.0), lo=0x3c00 (1.0) -> 0x44003c00
    self.assertEqual(result, 0x44003c00, f"Expected 0x44003c00, got 0x{result:08x}")

  def test_v_pack_b32_f16_opsel_hi_lo(self):
    """V_PACK_B32_F16 with opsel=0b0001 to read hi from src0, lo from src1."""
    # v0 = 0x40003c00 (hi=f16 2.0, lo=f16 1.0)
    # v1 = 0x44004200 (hi=f16 4.0, lo=f16 3.0)
    # With opsel=0b0001: read hi from v0 (0x4000=2.0), lo from v1 (0x4200=3.0)
    # Result should be: hi=v1.lo=0x4200 (3.0), lo=v0.hi=0x4000 (2.0) -> 0x42004000
    inst = v_pack_b32_f16(v[2], v[0], v[1])
    inst._values['opsel'] = 0b0001  # opsel[0]=1 for src0 hi, opsel[1]=0 for src1 lo

    instructions = [
      s_mov_b32(s[0], 0x40003c00),
      s_mov_b32(s[1], 0x44004200),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      inst,
    ]

    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2]
    # Expected: hi=0x4200 (3.0), lo=0x4000 (2.0) -> 0x42004000
    self.assertEqual(result, 0x42004000, f"Expected 0x42004000, got 0x{result:08x}")


class TestWMMA(unittest.TestCase):
  """Tests for WMMA (Wave Matrix Multiply-Accumulate) instructions."""

  def test_v_wmma_f32_16x16x16_f16_basic(self):
    """V_WMMA_F32_16X16X16_F16 basic test - verify emulator matches hardware."""
    # WMMA does D = A @ B + C where A,B are 16x16 f16, C,D are 16x16 f32
    # Use: A=v[16:23], B=v[24:31], C=D=v[0:7] (output in captured range v[0:15])
    instructions = []

    # f16 1.0 = 0x3c00, packed pair = 0x3c003c00
    instructions.append(s_mov_b32(s[0], 0x3c003c00))

    # Set A (v16-v23) and B (v24-v31) to all 1.0s
    for i in range(16, 32):
      instructions.append(v_mov_b32_e32(v[i], s[0]))

    # Set C (v0-v7) to all 0s (will also be output D)
    for i in range(8):
      instructions.append(v_mov_b32_e32(v[i], 0))

    # Execute WMMA: v[0:7] = A @ B + C
    instructions.append(v_wmma_f32_16x16x16_f16(v[0], v[16], v[24], v[0]))

    # Just run and compare - USE_HW=1 will verify emulator matches hardware
    st = run_program(instructions, n_lanes=32)

    # Verify at least some output is non-zero (actual values depend on WMMA layout)
    # Output should be 16.0 (16 x 1.0 x 1.0) for each element
    any_nonzero = any(st.vgpr[lane][0] != 0 for lane in range(32))
    self.assertTrue(any_nonzero, "WMMA should produce non-zero output")

  def test_v_wmma_f32_16x16x16_f16_all_ones(self):
    """V_WMMA_F32_16X16X16_F16 with all ones should produce 16.0 for each output element.
    This verifies the matrix multiply is computing the correct sum.
    """
    instructions = []

    # f16 1.0 = 0x3c00, packed pair = 0x3c003c00
    instructions.append(s_mov_b32(s[0], 0x3c003c00))

    # Set A (v16-v23) and B (v24-v31) to all 1.0s
    for i in range(16, 32):
      instructions.append(v_mov_b32_e32(v[i], s[0]))

    # Set C (v0-v7) to all 0s (will also be output D)
    for i in range(8):
      instructions.append(v_mov_b32_e32(v[i], 0))

    # Execute WMMA: v[0:7] = A @ B + C
    instructions.append(v_wmma_f32_16x16x16_f16(v[0], v[16], v[24], v[0]))

    st = run_program(instructions, n_lanes=32)

    # All output elements should be 16.0 (sum of 16 * 1.0 * 1.0)
    expected = f2i(16.0)
    for lane in range(32):
      for reg in range(8):
        result = st.vgpr[lane][reg]
        self.assertEqual(result, expected, f"v[{reg}] lane {lane}: expected 0x{expected:08x} (16.0), got 0x{result:08x} ({i2f(result)})")

  def test_v_wmma_f32_16x16x16_f16_with_accumulator(self):
    """V_WMMA_F32_16X16X16_F16 with non-zero accumulator.
    Verifies that C matrix is properly added to the product.
    """
    instructions = []

    # f16 1.0 = 0x3c00, packed pair = 0x3c003c00
    instructions.append(s_mov_b32(s[0], 0x3c003c00))
    # f32 5.0 = 0x40a00000
    instructions.append(s_mov_b32(s[1], f2i(5.0)))

    # Set A (v16-v23) and B (v24-v31) to all 1.0s
    for i in range(16, 32):
      instructions.append(v_mov_b32_e32(v[i], s[0]))

    # Set C (v0-v7) to all 5.0s
    for i in range(8):
      instructions.append(v_mov_b32_e32(v[i], s[1]))

    # Execute WMMA: v[0:7] = A @ B + C = 16.0 + 5.0 = 21.0
    instructions.append(v_wmma_f32_16x16x16_f16(v[0], v[16], v[24], v[0]))

    st = run_program(instructions, n_lanes=32)

    # All output elements should be 21.0 (16.0 + 5.0)
    expected = f2i(21.0)
    for lane in range(32):
      for reg in range(8):
        result = st.vgpr[lane][reg]
        self.assertEqual(result, expected, f"v[{reg}] lane {lane}: expected 0x{expected:08x} (21.0), got 0x{result:08x} ({i2f(result)})")


class TestVOP3P(unittest.TestCase):
  """Tests for VOP3P packed 16-bit operations."""

  def test_v_pk_add_f16_basic(self):
    """V_PK_ADD_F16 adds two packed f16 values."""
    from extra.assembly.amd.pcode import _f16
    # v0 = packed (1.0, 2.0), v1 = packed (3.0, 4.0)
    # Result should be packed (4.0, 6.0)
    instructions = [
      s_mov_b32(s[0], 0x40003c00),  # packed f16: hi=2.0, lo=1.0
      s_mov_b32(s[1], 0x44004200),  # packed f16: hi=4.0, lo=3.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_pk_add_f16(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2]
    # Expected: lo=1.0+3.0=4.0 (0x4400), hi=2.0+4.0=6.0 (0x4600) -> 0x46004400
    lo = _f16(result & 0xffff)
    hi = _f16((result >> 16) & 0xffff)
    self.assertAlmostEqual(lo, 4.0, places=2, msg=f"lo: expected 4.0, got {lo}")
    self.assertAlmostEqual(hi, 6.0, places=2, msg=f"hi: expected 6.0, got {hi}")

  def test_v_pk_add_f16_with_inline_constant(self):
    """V_PK_ADD_F16 with inline constant POS_ONE (1.0).
    Inline constants for VOP3P are f16 values in the low 16 bits only.
    The opsel_hi bits (default=0b11) select lo half for hi result, so both halves use the constant.
    """
    from extra.assembly.amd.pcode import _f16
    # v0 = packed (1.0, 1.0), add POS_ONE
    # With default opsel_hi=0b11: both lo and hi results use lo half of src1 (the constant)
    # But opsel_hi=1 means src1 hi comes from lo half - wait, let me check the actual encoding
    # Default opsel_hi=3 means: bit0=1 (src0 hi from hi), bit1=1 (src1 hi from hi)
    # Since inline constant has 0 in hi half, hi result = v0.hi + 0 = 1.0
    instructions = [
      s_mov_b32(s[0], 0x3c003c00),  # packed f16: hi=1.0, lo=1.0
      v_mov_b32_e32(v[0], s[0]),
      v_pk_add_f16(v[1], v[0], SrcEnum.POS_ONE),  # Add inline constant 1.0
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1]
    lo = _f16(result & 0xffff)
    hi = _f16((result >> 16) & 0xffff)
    # lo = 1.0 + 1.0 = 2.0, hi = 1.0 + 0.0 = 1.0 (inline const hi half is 0)
    self.assertAlmostEqual(lo, 2.0, places=2, msg=f"lo: expected 2.0, got {lo} (result=0x{result:08x})")
    self.assertAlmostEqual(hi, 1.0, places=2, msg=f"hi: expected 1.0, got {hi} (result=0x{result:08x})")

  def test_v_pk_mul_f16_basic(self):
    """V_PK_MUL_F16 multiplies two packed f16 values."""
    from extra.assembly.amd.pcode import _f16
    # v0 = packed (2.0, 3.0), v1 = packed (4.0, 5.0)
    # Result should be packed (8.0, 15.0)
    instructions = [
      s_mov_b32(s[0], 0x42004000),  # packed f16: hi=3.0, lo=2.0
      s_mov_b32(s[1], 0x45004400),  # packed f16: hi=5.0, lo=4.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_pk_mul_f16(v[2], v[0], v[1]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][2]
    lo = _f16(result & 0xffff)
    hi = _f16((result >> 16) & 0xffff)
    self.assertAlmostEqual(lo, 8.0, places=1, msg=f"lo: expected 8.0, got {lo}")
    self.assertAlmostEqual(hi, 15.0, places=1, msg=f"hi: expected 15.0, got {hi}")

  def test_v_pk_mul_f16_with_inline_constant(self):
    """V_PK_MUL_F16 with inline constant POS_TWO (2.0).
    Inline constant has value only in low 16 bits, hi is 0.
    """
    from extra.assembly.amd.pcode import _f16
    # v0 = packed (3.0, 4.0), multiply by POS_TWO
    # lo = 3.0 * 2.0 = 6.0, hi = 4.0 * 0.0 = 0.0 (inline const hi is 0)
    instructions = [
      s_mov_b32(s[0], 0x44004200),  # packed f16: hi=4.0, lo=3.0
      v_mov_b32_e32(v[0], s[0]),
      v_pk_mul_f16(v[1], v[0], SrcEnum.POS_TWO),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][1]
    lo = _f16(result & 0xffff)
    hi = _f16((result >> 16) & 0xffff)
    self.assertAlmostEqual(lo, 6.0, places=1, msg=f"lo: expected 6.0, got {lo}")
    self.assertAlmostEqual(hi, 0.0, places=1, msg=f"hi: expected 0.0, got {hi}")

  def test_v_pk_fma_f16_basic(self):
    """V_PK_FMA_F16: D = A * B + C for packed f16."""
    from extra.assembly.amd.pcode import _f16
    # A = packed (2.0, 3.0), B = packed (4.0, 5.0), C = packed (1.0, 1.0)
    # Result should be packed (2*4+1=9.0, 3*5+1=16.0)
    instructions = [
      s_mov_b32(s[0], 0x42004000),  # A: hi=3.0, lo=2.0
      s_mov_b32(s[1], 0x45004400),  # B: hi=5.0, lo=4.0
      s_mov_b32(s[2], 0x3c003c00),  # C: hi=1.0, lo=1.0
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_mov_b32_e32(v[2], s[2]),
      v_pk_fma_f16(v[3], v[0], v[1], v[2]),
    ]
    st = run_program(instructions, n_lanes=1)
    result = st.vgpr[0][3]
    lo = _f16(result & 0xffff)
    hi = _f16((result >> 16) & 0xffff)
    self.assertAlmostEqual(lo, 9.0, places=1, msg=f"lo: expected 9.0, got {lo}")
    self.assertAlmostEqual(hi, 16.0, places=0, msg=f"hi: expected 16.0, got {hi}")


class TestF64Conversions(unittest.TestCase):
  """Tests for 64-bit float operations and conversions."""

  def test_v_add_f64_inline_constant(self):
    """V_ADD_F64 with inline constant POS_ONE (1.0) as f64."""
    one_f64 = f2i64(1.0)
    instructions = [
      s_mov_b32(s[0], one_f64 & 0xffffffff),
      s_mov_b32(s[1], one_f64 >> 32),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_add_f64(v[2:4], v[0:2], SrcEnum.POS_ONE),  # 1.0 + 1.0 = 2.0
    ]
    st = run_program(instructions, n_lanes=1)
    result = i642f(st.vgpr[0][2] | (st.vgpr[0][3] << 32))
    self.assertAlmostEqual(result, 2.0, places=5)

  def test_v_ldexp_f64_negative_exponent(self):
    """V_LDEXP_F64 with negative exponent (-32)."""
    val = -8.0
    val_bits = f2i64(val)
    expected = -8.0 * (2.0 ** -32)  # -1.862645149230957e-09
    instructions = [
      s_mov_b32(s[0], val_bits & 0xffffffff),
      s_mov_b32(s[1], val_bits >> 32),
      v_mov_b32_e32(v[0], s[0]),
      v_mov_b32_e32(v[1], s[1]),
      v_ldexp_f64(v[2:4], v[0:2], 0xffffffe0),  # -32
    ]
    st = run_program(instructions, n_lanes=1)
    result = i642f(st.vgpr[0][2] | (st.vgpr[0][3] << 32))
    self.assertAlmostEqual(result, expected, places=15)

  def test_f64_to_i64_conversion_sequence(self):
    """Test the f64->i64 conversion sequence used by the compiler.

    The compiler generates:
      v_trunc_f64 -> v_ldexp_f64 (by -32) -> v_floor_f64 -> v_fma_f64 (by -2^32)
      -> v_cvt_u32_f64 (low bits) -> v_cvt_i32_f64 (high bits)

    The FMA computes: trunc + (-2^32) * floor = trunc - floor * 2^32
    which gives the low 32 bits as a positive float (for proper u32 conversion).
    """
    val = -8.0
    val_bits = f2i64(val)
    lit = -4294967296.0  # -2^32 (note: NEGATIVE, so FMA does trunc - floor * 2^32)
    lit_bits = f2i64(lit)

    instructions = [
      s_mov_b32(s[0], val_bits & 0xffffffff),
      s_mov_b32(s[1], val_bits >> 32),
      v_trunc_f64_e32(v[0:2], s[0:2]),
      v_ldexp_f64(v[2:4], v[0:2], 0xffffffe0),  # -32
      v_floor_f64_e32(v[2:4], v[2:4]),
      s_mov_b32(s[2], lit_bits & 0xffffffff),
      s_mov_b32(s[3], lit_bits >> 32),
      v_fma_f64(v[0:2], s[2:4], v[2:4], v[0:2]),
      v_cvt_u32_f64_e32(v[4], v[0:2]),
      v_cvt_i32_f64_e32(v[5], v[2:4]),
    ]
    st = run_program(instructions, n_lanes=1)
    # v4 = low 32 bits, v5 = high 32 bits (sign extended)
    lo = st.vgpr[0][4]
    hi = st.vgpr[0][5]
    # For -8: lo should be 0xfffffff8, hi should be 0xffffffff
    result = struct.unpack('<q', struct.pack('<II', lo, hi))[0]
    self.assertEqual(result, -8, f"Expected -8, got {result} (lo=0x{lo:08x}, hi=0x{hi:08x})")


if __name__ == '__main__':
  unittest.main()
