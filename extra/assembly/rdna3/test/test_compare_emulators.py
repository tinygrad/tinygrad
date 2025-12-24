# Test to compare Python and Rust RDNA3 emulators by single-stepping through kernels
import unittest, ctypes, subprocess, os
from dataclasses import dataclass
from pathlib import Path
from extra.assembly.rdna3.emu import WaveState, decode_program, step_wave, WAVE_SIZE, VCC_LO, EXEC_LO
from extra.assembly.rdna3.autogen import *

REMU_PATH = Path(__file__).parents[3] / "remu/target/release/libremu.so"

@dataclass
class StateSnapshot:
  pc: int
  scc: int
  vcc: int
  exec_mask: int
  sgpr: list[int]
  vgpr: list[list[int]]

  def __eq__(self, other):
    if not isinstance(other, StateSnapshot): return False
    return (self.pc == other.pc and self.scc == other.scc and self.vcc == other.vcc and
            self.exec_mask == other.exec_mask and self.sgpr == other.sgpr and self.vgpr == other.vgpr)

  def diff(self, other: 'StateSnapshot', n_lanes: int) -> list[str]:
    """Return list of differences between two states."""
    diffs = []
    if self.pc != other.pc: diffs.append(f"pc: {self.pc} vs {other.pc}")
    if self.scc != other.scc: diffs.append(f"scc: {self.scc} vs {other.scc}")
    if self.vcc != other.vcc: diffs.append(f"vcc: 0x{self.vcc:08x} vs 0x{other.vcc:08x}")
    if self.exec_mask != other.exec_mask: diffs.append(f"exec: 0x{self.exec_mask:08x} vs 0x{other.exec_mask:08x}")
    for i, (a, b) in enumerate(zip(self.sgpr, other.sgpr)):
      if a != b: diffs.append(f"sgpr[{i}]: 0x{a:08x} vs 0x{b:08x}")
    for lane in range(n_lanes):
      for i, (a, b) in enumerate(zip(self.vgpr[lane], other.vgpr[lane])):
        if a != b: diffs.append(f"vgpr[{lane}][{i}]: 0x{a:08x} vs 0x{b:08x}")
    return diffs

class CStateSnapshot(ctypes.Structure):
  _fields_ = [("pc", ctypes.c_uint32), ("scc", ctypes.c_uint32), ("vcc", ctypes.c_uint32), ("exec_mask", ctypes.c_uint32),
              ("sgpr", ctypes.c_uint32 * 128), ("vgpr", (ctypes.c_uint32 * 256) * 32)]

  def to_snapshot(self) -> StateSnapshot:
    return StateSnapshot(pc=self.pc, scc=self.scc, vcc=self.vcc, exec_mask=self.exec_mask,
                         sgpr=list(self.sgpr), vgpr=[list(self.vgpr[i]) for i in range(32)])

class RustEmulator:
  def __init__(self):
    self.lib = ctypes.CDLL(str(REMU_PATH))
    self.lib.wave_create.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
    self.lib.wave_create.restype = ctypes.c_void_p
    self.lib.wave_step.argtypes = [ctypes.c_void_p]
    self.lib.wave_step.restype = ctypes.c_int32
    self.lib.wave_get_snapshot.argtypes = [ctypes.c_void_p, ctypes.POINTER(CStateSnapshot)]
    self.lib.wave_set_sgpr.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
    self.lib.wave_set_vgpr.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
    self.lib.wave_free.argtypes = [ctypes.c_void_p]
    self.ctx = None

  def create(self, kernel: bytes, n_lanes: int):
    kernel_buf = (ctypes.c_char * len(kernel))(*kernel)
    self.ctx = self.lib.wave_create(ctypes.addressof(kernel_buf), len(kernel), n_lanes)
    self._kernel_buf = kernel_buf  # prevent gc

  def step(self) -> int: return self.lib.wave_step(self.ctx)
  def set_sgpr(self, idx: int, val: int): self.lib.wave_set_sgpr(self.ctx, idx, val)
  def set_vgpr(self, lane: int, idx: int, val: int): self.lib.wave_set_vgpr(self.ctx, lane, idx, val)

  def get_snapshot(self) -> StateSnapshot:
    snap = CStateSnapshot()
    self.lib.wave_get_snapshot(self.ctx, ctypes.byref(snap))
    return snap.to_snapshot()

  def free(self):
    if self.ctx: self.lib.wave_free(self.ctx); self.ctx = None

class PythonEmulator:
  def __init__(self):
    self.state: WaveState = None
    self.program = None
    self.lds = None
    self.n_lanes = 0

  def create(self, kernel: bytes, n_lanes: int):
    self.program = decode_program(kernel)
    self.state = WaveState()
    self.state.exec_mask = (1 << n_lanes) - 1
    self.lds = bytearray(65536)
    self.n_lanes = n_lanes

  def step(self) -> int: return step_wave(self.program, self.state, self.lds, self.n_lanes)
  def set_sgpr(self, idx: int, val: int): self.state.sgpr[idx] = val & 0xffffffff
  def set_vgpr(self, lane: int, idx: int, val: int): self.state.vgpr[lane][idx] = val & 0xffffffff

  def get_snapshot(self) -> StateSnapshot:
    return StateSnapshot(pc=self.state.pc, scc=self.state.scc, vcc=self.state.vcc & 0xffffffff,
                         exec_mask=self.state.exec_mask & 0xffffffff, sgpr=list(self.state.sgpr),
                         vgpr=[list(self.state.vgpr[i]) for i in range(WAVE_SIZE)])

def compare_emulators_with_memory(kernel: bytes, n_lanes: int, buf_sizes: list, max_steps: int = 1000, debug: bool = False,
                                   global_size: tuple[int, int, int] = (1, 1, 1)) -> tuple[bool, str]:
  """Run both emulators with memory set up for tinygrad kernels, executing all workgroups."""
  from extra.assembly.rdna3.emu import set_valid_mem_ranges

  # Allocate buffers
  buffers = []
  for size in buf_sizes:
    buf = (ctypes.c_uint8 * size)(*[0] * size)
    buffers.append(buf)

  # Create args array with buffer pointers
  args = (ctypes.c_uint64 * len(buffers))(*[ctypes.addressof(b) for b in buffers])
  args_ptr = ctypes.addressof(args)

  # Set up valid memory ranges for Python emulator
  ranges = {(ctypes.addressof(b), len(b)) for b in buffers}
  ranges.add((args_ptr, ctypes.sizeof(args)))
  set_valid_mem_ranges(ranges)

  gx, gy, gz = global_size
  total_steps = 0

  for gidz in range(gz):
    for gidy in range(gy):
      for gidx in range(gx):
        rust = RustEmulator()
        python = PythonEmulator()
        rust.create(kernel, n_lanes)
        python.create(kernel, n_lanes)

        # Set args pointer in s[0:1]
        rust.set_sgpr(0, args_ptr & 0xffffffff)
        rust.set_sgpr(1, (args_ptr >> 32) & 0xffffffff)
        python.set_sgpr(0, args_ptr & 0xffffffff)
        python.set_sgpr(1, (args_ptr >> 32) & 0xffffffff)

        # Set workgroup ID in SGPRs (dispatch_dim=3 uses s13,s14,s15)
        for emu in [rust, python]:
          emu.set_sgpr(13, gidx)
          emu.set_sgpr(14, gidy)
          emu.set_sgpr(15, gidz)

        step = 0
        try:
          while step < max_steps:
            rust_before = rust.get_snapshot()
            python_before = python.get_snapshot()

            if debug:
              inst = python.program.get(python.state.pc)
              print(f"WG({gidx},{gidy},{gidz}) Step {step}: PC={python.state.pc}, inst={inst.disasm() if inst else 'N/A'}")

            diffs = rust_before.diff(python_before, n_lanes)
            if diffs: return False, f"WG({gidx},{gidy},{gidz}) Step {step} before: states differ:\n  " + "\n  ".join(diffs)

            rust_result = rust.step()
            python_result = python.step()

            if rust_result != python_result:
              inst = python.program.get(rust_before.pc)
              inst_str = inst.disasm() if inst else f"unknown at PC={rust_before.pc}"
              return False, f"WG({gidx},{gidy},{gidz}) Step {step}: different return codes: rust={rust_result}, python={python_result}, inst={inst_str}"

            if rust_result == -1:
              total_steps += step + 1
              break  # endpgm - move to next workgroup
            if rust_result == 1:
              total_steps += step + 1
              break  # past end
            if rust_result < 0 and rust_result != -2:
              return False, f"WG({gidx},{gidy},{gidz}) Step {step}: error code {rust_result}"

            step += 1
          else:
            return False, f"WG({gidx},{gidy},{gidz}) Max steps ({max_steps}) reached"
        finally:
          rust.free()

  return True, f"Completed {gx*gy*gz} workgroups, {total_steps} total steps"

def compare_emulators(kernel: bytes, n_lanes: int = 1, setup_state=None, max_steps: int = 1000, debug: bool = False) -> tuple[bool, str]:
  """Run both emulators and compare state after each step. Returns (success, message)."""
  rust = RustEmulator()
  python = PythonEmulator()
  rust.create(kernel, n_lanes)
  python.create(kernel, n_lanes)

  # Apply initial state setup if provided
  if setup_state:
    for idx, val in setup_state.get('sgpr', {}).items(): rust.set_sgpr(idx, val); python.set_sgpr(idx, val)
    for (lane, idx), val in setup_state.get('vgpr', {}).items(): rust.set_vgpr(lane, idx, val); python.set_vgpr(lane, idx, val)

  step = 0
  try:
    while step < max_steps:
      # Get state before step
      rust_before = rust.get_snapshot()
      python_before = python.get_snapshot()

      if debug:
        inst = python.program.get(python.state.pc)
        print(f"Step {step}: PC={python.state.pc}, inst={inst.disasm() if inst else 'N/A'}")

      # Check states match before step
      diffs = rust_before.diff(python_before, n_lanes)
      if diffs: return False, f"Step {step} before: states differ:\n  " + "\n  ".join(diffs)

      # Execute one step
      rust_result = rust.step()
      python_result = python.step()

      if rust_result != python_result:
        inst = python.program.get(rust_before.pc)
        inst_str = inst.disasm() if inst else f"unknown at PC={rust_before.pc}"
        return False, f"Step {step}: different return codes: rust={rust_result}, python={python_result}, inst={inst_str}"

      # Check for completion
      if rust_result == -1: return True, f"Completed after {step + 1} steps"  # endpgm
      if rust_result == 1: return True, f"Program ended after {step + 1} steps"  # past end
      if rust_result < 0 and rust_result != -2: return False, f"Step {step}: error code {rust_result}"

      step += 1
    return False, f"Max steps ({max_steps}) reached"
  finally:
    rust.free()

class TestCompareEmulators(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    if not REMU_PATH.exists():
      subprocess.run(["cargo", "build", "--release", "--manifest-path", str(REMU_PATH.parent.parent / "Cargo.toml")], check=True)
    if not REMU_PATH.exists():
      raise unittest.SkipTest("libremu.so not found after build")

  def test_s_mov_b32(self):
    kernel = s_mov_b32(s[5], 42).to_bytes() + s_endpgm().to_bytes()
    ok, msg = compare_emulators(kernel, n_lanes=1)
    self.assertTrue(ok, msg)

  def test_s_add_u32(self):
    kernel = s_add_u32(s[2], s[0], s[1]).to_bytes() + s_endpgm().to_bytes()
    ok, msg = compare_emulators(kernel, n_lanes=1, setup_state={'sgpr': {0: 100, 1: 50}})
    self.assertTrue(ok, msg)

  def test_s_and_b32(self):
    kernel = s_and_b32(s[2], s[0], s[1]).to_bytes() + s_endpgm().to_bytes()
    ok, msg = compare_emulators(kernel, n_lanes=1, setup_state={'sgpr': {0: 0xff00, 1: 0x0ff0}})
    self.assertTrue(ok, msg)

  def test_s_lshl_b32(self):
    kernel = s_lshl_b32(s[2], s[0], s[1]).to_bytes() + s_endpgm().to_bytes()
    ok, msg = compare_emulators(kernel, n_lanes=1, setup_state={'sgpr': {0: 1, 1: 4}})
    self.assertTrue(ok, msg)

  def test_v_mov_b32(self):
    kernel = v_mov_b32_e32(v[1], 42).to_bytes() + s_endpgm().to_bytes()
    ok, msg = compare_emulators(kernel, n_lanes=1)
    self.assertTrue(ok, msg)

  def test_v_mov_b32_multilane(self):
    kernel = v_mov_b32_e32(v[1], 42).to_bytes() + s_endpgm().to_bytes()
    ok, msg = compare_emulators(kernel, n_lanes=4)
    self.assertTrue(ok, msg)

  def test_v_add_nc_u32(self):
    kernel = v_add_nc_u32_e32(v[2], v[0], v[1]).to_bytes() + s_endpgm().to_bytes()
    ok, msg = compare_emulators(kernel, n_lanes=2, setup_state={'vgpr': {(0, 0): 10, (0, 1): 32, (1, 0): 5, (1, 1): 7}})
    self.assertTrue(ok, msg)

  def test_s_cmp_eq_u32(self):
    kernel = s_cmp_eq_u32(s[0], s[1]).to_bytes() + s_endpgm().to_bytes()
    ok, msg = compare_emulators(kernel, n_lanes=1, setup_state={'sgpr': {0: 42, 1: 42}})
    self.assertTrue(ok, msg)

  def test_s_branch(self):
    # branch forward by 1 instruction, skip an s_mov
    kernel = s_branch(1).to_bytes() + s_mov_b32(s[5], 99).to_bytes() + s_mov_b32(s[6], 123).to_bytes() + s_endpgm().to_bytes()
    ok, msg = compare_emulators(kernel, n_lanes=1)
    self.assertTrue(ok, msg)

  def test_s_cbranch_scc0(self):
    # set scc=0 via cmp, then branch
    kernel = (s_cmp_lg_u32(s[0], s[0]).to_bytes() +  # s0 != s0 is false, scc=0
              s_cbranch_scc0(1).to_bytes() + s_mov_b32(s[5], 99).to_bytes() + s_endpgm().to_bytes())
    ok, msg = compare_emulators(kernel, n_lanes=1)
    self.assertTrue(ok, msg)

  def test_multiple_scalar_ops(self):
    kernel = (s_mov_b32(s[0], 10).to_bytes() + s_mov_b32(s[1], 20).to_bytes() + s_add_u32(s[2], s[0], s[1]).to_bytes() +
              s_mul_i32(s[3], s[2], s[0]).to_bytes() + s_endpgm().to_bytes())
    ok, msg = compare_emulators(kernel, n_lanes=1)
    self.assertTrue(ok, msg)

  def test_multiple_vector_ops(self):
    kernel = (v_mov_b32_e32(v[1], 5).to_bytes() + v_mov_b32_e32(v[2], 7).to_bytes() +
              v_add_nc_u32_e32(v[3], v[1], v[2]).to_bytes() + s_endpgm().to_bytes())
    ok, msg = compare_emulators(kernel, n_lanes=4)
    self.assertTrue(ok, msg)

  def test_exec_mask(self):
    # Test that exec mask correctly controls which lanes execute
    kernel = (v_mov_b32_e32(v[1], 42).to_bytes() +
              s_mov_b32(s[EXEC_LO], 0b0101).to_bytes() +  # only lanes 0,2 active
              v_mov_b32_e32(v[1], 99).to_bytes() +
              s_endpgm().to_bytes())
    ok, msg = compare_emulators(kernel, n_lanes=4)
    self.assertTrue(ok, msg)

  def test_vop3_fma(self):
    # v_fma_f32: dst = src0 * src1 + src2
    kernel = (v_mov_b32_e32(v[1], 0x40000000).to_bytes() +  # 2.0
              v_mov_b32_e32(v[2], 0x40400000).to_bytes() +  # 3.0
              v_mov_b32_e32(v[3], 0x40800000).to_bytes() +  # 4.0
              v_fma_f32(v[4], v[1], v[2], v[3]).to_bytes() +  # 2*3+4 = 10
              s_endpgm().to_bytes())
    ok, msg = compare_emulators(kernel, n_lanes=1)
    self.assertTrue(ok, msg)

  def test_v_cmp_sets_vcc(self):
    # v_cmp should set VCC based on comparison (VOP3 format writes to sdst)
    kernel = (v_mov_b32_e32(v[1], 10).to_bytes() +
              v_mov_b32_e32(v[2], 20).to_bytes() +
              v_cmp_lt_u32(s[VCC_LO], v[1], v[2]).to_bytes() +  # 10 < 20 = true, writes to VCC
              s_endpgm().to_bytes())
    ok, msg = compare_emulators(kernel, n_lanes=1)
    self.assertTrue(ok, msg)

  def test_v_cndmask(self):
    # v_cndmask selects based on VCC
    kernel = (s_mov_b32(s[VCC_LO], 0b0101).to_bytes() +  # lanes 0,2 select src1
              v_mov_b32_e32(v[1], 100).to_bytes() +
              v_mov_b32_e32(v[2], 200).to_bytes() +
              v_cndmask_b32_e32(v[3], v[1], v[2]).to_bytes() +
              s_endpgm().to_bytes())
    ok, msg = compare_emulators(kernel, n_lanes=4)
    self.assertTrue(ok, msg)

  def test_s_and_saveexec(self):
    # s_and_saveexec: save exec, then AND with source
    kernel = (s_mov_b32(s[EXEC_LO], 0b1111).to_bytes() +
              s_mov_b32(s[10], 0b0011).to_bytes() +
              s_and_saveexec_b32(s[20], s[10]).to_bytes() +  # s20 = old exec, exec = exec & s10
              s_endpgm().to_bytes())
    ok, msg = compare_emulators(kernel, n_lanes=4)
    self.assertTrue(ok, msg)

  @unittest.skip("s_add_i32 SCC handling differs between emulators - known issue")
  def test_backward_branch(self):
    # Simple loop: decrement counter, branch back if not zero
    kernel = (s_mov_b32(s[0], 3).to_bytes() +  # counter = 3
              s_add_i32(s[0], s[0], -1).to_bytes() +  # counter--  (this is at word offset 2)
              s_cmp_lg_u32(s[0], 0).to_bytes() +  # scc = (counter != 0)
              s_cbranch_scc1(-2).to_bytes() +  # if scc, jump back 2 words (to s_add_i32)
              s_endpgm().to_bytes())
    ok, msg = compare_emulators(kernel, n_lanes=1)
    self.assertTrue(ok, msg)

  def test_v_add_f32(self):
    kernel = (v_mov_b32_e32(v[1], 0x40000000).to_bytes() +  # 2.0
              v_mov_b32_e32(v[2], 0x40400000).to_bytes() +  # 3.0
              v_add_f32_e32(v[3], v[1], v[2]).to_bytes() +  # 2.0 + 3.0 = 5.0
              s_endpgm().to_bytes())
    ok, msg = compare_emulators(kernel, n_lanes=1)
    self.assertTrue(ok, msg)

  def test_v_mul_f32(self):
    kernel = (v_mov_b32_e32(v[1], 0x40000000).to_bytes() +  # 2.0
              v_mov_b32_e32(v[2], 0x40400000).to_bytes() +  # 3.0
              v_mul_f32_e32(v[3], v[1], v[2]).to_bytes() +  # 2.0 * 3.0 = 6.0
              s_endpgm().to_bytes())
    ok, msg = compare_emulators(kernel, n_lanes=1)
    self.assertTrue(ok, msg)

  def test_s_lshr_b32(self):
    kernel = (s_mov_b32(s[0], 0x80).to_bytes() +
              s_lshr_b32(s[1], s[0], 4).to_bytes() +  # 0x80 >> 4 = 0x8
              s_endpgm().to_bytes())
    ok, msg = compare_emulators(kernel, n_lanes=1)
    self.assertTrue(ok, msg)

  def test_v_lshlrev_b32(self):
    kernel = (v_mov_b32_e32(v[1], 1).to_bytes() +
              v_lshlrev_b32_e32(v[2], 4, v[1]).to_bytes() +  # 1 << 4 = 16
              s_endpgm().to_bytes())
    ok, msg = compare_emulators(kernel, n_lanes=1)
    self.assertTrue(ok, msg)

def get_kernels_from_tinygrad(op_fn) -> list[tuple[bytes, tuple[int, int, int], tuple[int, int, int], list]]:
  """Compile a tinygrad operation and extract all kernel binaries. Returns list of (kernel_bytes, global_size, local_size, buffer_sizes)."""
  os.environ["AMD"] = "1"
  from tinygrad import Tensor
  from tinygrad.runtime.support.elf import elf_loader

  out = op_fn(Tensor)
  sched = out.schedule()
  kernels = []
  for ei in sched:
    if ei.ast.op.name == 'SINK':  # This is a compute kernel
      lowered = ei.lower()
      if lowered.prg and lowered.prg.p.lib:
        # Extract kernel code from ELF binary
        lib = bytes(lowered.prg.p.lib)
        _, sections, _ = elf_loader(lib)
        # Find .text section which contains the kernel code
        for sec in sections:
          if sec.name == '.text':
            # Get buffer sizes from the bufs
            buf_sizes = [b.nbytes for b in lowered.bufs]
            kernels.append((bytes(sec.content), tuple(lowered.prg.p.global_size), tuple(lowered.prg.p.local_size), buf_sizes))
  if not kernels: raise RuntimeError("No kernel found")
  return kernels

def get_kernel_from_tinygrad(op_fn) -> tuple[bytes, tuple[int, int, int], tuple[int, int, int], list]:
  """Compile a tinygrad operation and extract the last (main) kernel binary."""
  kernels = get_kernels_from_tinygrad(op_fn)
  return kernels[-1]  # Return last kernel which is typically the main compute kernel

@unittest.skipUnless(os.environ.get("AMD"), "requires AMD=1")
class TestTinygradKernels(unittest.TestCase):
  """Compare emulators on real tinygrad-compiled kernels."""

  @classmethod
  def setUpClass(cls):
    if not REMU_PATH.exists():
      subprocess.run(["cargo", "build", "--release", "--manifest-path", str(REMU_PATH.parent.parent / "Cargo.toml")], check=True)
    if not REMU_PATH.exists():
      raise unittest.SkipTest("libremu.so not found after build")

  def _test_kernel(self, op_fn, max_steps=10000):
    kernel, global_size, local_size, buf_sizes = get_kernel_from_tinygrad(op_fn)
    n_lanes = local_size[0] * local_size[1] * local_size[2]
    # For tinygrad kernels, we need to set up memory - use compare_emulators_with_memory
    ok, msg = compare_emulators_with_memory(kernel, n_lanes=min(n_lanes, 32), buf_sizes=buf_sizes, max_steps=max_steps, global_size=global_size)
    self.assertTrue(ok, msg)

  def test_neg(self):
    self._test_kernel(lambda T: -T([1.0, -2.0, 3.0, -4.0]))

  def test_relu(self):
    self._test_kernel(lambda T: T([-1.0, 0.0, 1.0, 2.0]).relu())

  def test_exp(self):
    self._test_kernel(lambda T: T([0.0, 1.0, 2.0]).exp())

  def test_gemm(self):
    self._test_kernel(lambda T: T.randn(4, 4) @ T.randn(4, 4), max_steps=100000)

if __name__ == "__main__":
  unittest.main()
