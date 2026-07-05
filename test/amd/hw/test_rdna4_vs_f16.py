"""RDNA4 v_s_*_f16 scalar transcendental coverage.

v_s_{rcp,sqrt,rsq,log,exp}_f16 write the f16 result to the low 16 bits of the
destination SGPR and zero the high 16 bits (the pcode expresses this as two
writes, D0.f16 and D0[31:16]=0). Exercises the generated pcode end-to-end in the
emulator and compares against real RDNA4 hardware when USE_HW=1. Inputs are powers
of two so the results are exactly representable and the approximate hardware ops
still match bit-for-bit.
"""
import ctypes, unittest
import tinygrad.runtime.autogen.amd.rdna4.ins as r4
from tinygrad.helpers import flat_mv
from tinygrad.renderer.amd.dsl import NULL
from test.amd.hw.helpers import USE_HW, assemble, f32_to_f16, f2i

LANES = 1

def _code(instructions: list, out_reg: int = 2) -> bytes:
  return assemble([
    r4.s_mov_b32(r4.s[80], r4.s[0]),
    r4.s_mov_b32(r4.s[81], r4.s[1]),
    r4.v_mov_b32_e32(r4.v[255], r4.v[0]),
    *instructions,
    r4.s_load_b64(r4.s[92:93], r4.s[80:81], soffset=NULL),
    r4.s_wait_kmcnt(simm16=0),
    r4.v_lshlrev_b32_e32(r4.v[240], 2, r4.v[255]),
    r4.v_mov_b32_e32(r4.v[241], 0),
    r4.global_store_b32(vaddr=r4.v[240:241], saddr=r4.s[92:93], vsrc=r4.v[out_reg]),
    r4.s_endpgm(),
  ])

def _run_emu(instructions: list, out_reg: int = 2) -> int:
  from test.mockgpu.amd.emu import run_asm
  out_buf = (ctypes.c_uint32 * LANES)(*([0] * LANES))
  args = (ctypes.c_uint64 * 1)(ctypes.addressof(out_buf))
  code = _code(instructions, out_reg)
  kernel_buf = (ctypes.c_char * len(code)).from_buffer_copy(code)
  result = run_asm(ctypes.addressof(kernel_buf), len(code), 1, 1, 1, LANES, 1, 1, ctypes.addressof(args), arch='rdna4')
  assert result == 0, f"run_asm failed with {result}"
  return out_buf[0]

def _run_hw(instructions: list, out_reg: int = 2) -> int:
  from tinygrad.device import Device
  from tinygrad.runtime.ops_amd import AMDProgram
  from tinygrad.runtime.support.compiler_amd import HIPCompiler

  dev = Device['AMD']
  if not dev.arch.startswith('gfx12'): raise unittest.SkipTest('requires RDNA4 hardware')
  code = _code(instructions, out_reg)
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
  .amdhsa_group_segment_fixed_size 65536
  .amdhsa_private_segment_fixed_size 65536
  .amdhsa_enable_private_segment 1
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
    .group_segment_fixed_size: 65536
    .private_segment_fixed_size: 65536
    .kernarg_segment_align: 8
    .wavefront_size: 32
    .sgpr_count: 96
    .vgpr_count: 256
    .max_flat_workgroup_size: 1024
...
.end_amdgpu_metadata
"""
  prg = AMDProgram(dev, 'test', HIPCompiler(dev.arch).compile(asm_src))
  out_gpu = dev.allocator.alloc(LANES * 4)
  prg(out_gpu, global_size=(1, 1, 1), local_size=(LANES, 1, 1), wait=True)
  out = bytearray(LANES * 4)
  dev.allocator._copyout(flat_mv(memoryview(out)), out_gpu)
  return int.from_bytes(out[:4], 'little')

def run_rdna4(instructions: list, out_reg: int = 2) -> int:
  emu = _run_emu(instructions, out_reg)
  if not USE_HW: return emu
  hw = _run_hw(instructions, out_reg)
  if emu != hw: raise AssertionError(f"Emulator vs Hardware mismatch: emu=0x{emu:08x} hw=0x{hw:08x}")
  return hw

class TestEmuVSF16(unittest.TestCase):
  def _f16_op(self, op, in_val: float) -> int:
    """Run a scalar f16 op: load in_val into s[3], write result to s[2], move to v[2]."""
    return run_rdna4([r4.s_mov_b32(r4.s[3], f32_to_f16(in_val)), op(r4.s[2], r4.s[3]), r4.v_mov_b32_e32(r4.v[2], r4.s[2])])

  def _check(self, op, in_val: float, expected: float):
    res = self._f16_op(op, in_val)
    self.assertEqual(res >> 16, 0, f"high 16 bits must be zero, got 0x{res:08x}")
    self.assertEqual(res & 0xffff, f32_to_f16(expected), f"got 0x{res & 0xffff:04x}, want 0x{f32_to_f16(expected):04x}")

  def test_v_s_rcp_f16(self): self._check(r4.v_s_rcp_f16, 4.0, 0.25)
  def test_v_s_sqrt_f16(self): self._check(r4.v_s_sqrt_f16, 4.0, 2.0)
  def test_v_s_rsq_f16(self): self._check(r4.v_s_rsq_f16, 4.0, 0.5)
  def test_v_s_log_f16(self): self._check(r4.v_s_log_f16, 4.0, 2.0)
  def test_v_s_exp_f16(self): self._check(r4.v_s_exp_f16, 2.0, 4.0)

  def test_v_s_rcp_f32(self):  # f32 single-write control
    res = run_rdna4([r4.s_mov_b32(r4.s[3], f2i(4.0)), r4.v_s_rcp_f32(r4.s[2], r4.s[3]), r4.v_mov_b32_e32(r4.v[2], r4.s[2])])
    self.assertEqual(res, f2i(0.25))

if __name__ == "__main__":
  unittest.main()
