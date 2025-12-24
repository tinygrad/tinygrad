# Test to compare Python and Rust RDNA3 emulators by running real tinygrad kernels
import unittest, ctypes, os
from dataclasses import dataclass
from pathlib import Path
from extra.assembly.rdna3.emu import WaveState, decode_program, step_wave, WAVE_SIZE

REMU_PATH = Path(__file__).parents[3] / "remu/target/release/libremu.so"

@dataclass
class StateSnapshot:
  pc: int
  scc: int
  vcc: int
  exec_mask: int
  sgpr: list[int]
  vgpr: list[list[int]]

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
    self._kernel_buf = kernel_buf

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

        for emu in [rust, python]:
          emu.set_sgpr(0, args_ptr & 0xffffffff)
          emu.set_sgpr(1, (args_ptr >> 32) & 0xffffffff)
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

def get_kernel_from_tinygrad(op_fn) -> tuple[bytes, tuple[int, int, int], tuple[int, int, int], list]:
  """Compile a tinygrad operation and extract the last (main) kernel binary."""
  os.environ["AMD"] = "1"
  from tinygrad import Tensor
  from tinygrad.runtime.support.elf import elf_loader

  out = op_fn(Tensor)
  sched = out.schedule()
  kernels = []
  for ei in sched:
    if ei.ast.op.name == 'SINK':
      lowered = ei.lower()
      if lowered.prg and lowered.prg.p.lib:
        lib = bytes(lowered.prg.p.lib)
        _, sections, _ = elf_loader(lib)
        for sec in sections:
          if sec.name == '.text':
            buf_sizes = [b.nbytes for b in lowered.bufs]
            kernels.append((bytes(sec.content), tuple(lowered.prg.p.global_size), tuple(lowered.prg.p.local_size), buf_sizes))
  if not kernels: raise RuntimeError("No kernel found")
  return kernels[-1]

@unittest.skipUnless(REMU_PATH.exists(), "libremu.so not found")
class TestTinygradKernels(unittest.TestCase):
  """Compare emulators on real tinygrad-compiled kernels."""

  def _test_kernel(self, op_fn, max_steps=10000):
    kernel, global_size, local_size, buf_sizes = get_kernel_from_tinygrad(op_fn)
    n_lanes = local_size[0] * local_size[1] * local_size[2]
    ok, msg = compare_emulators_with_memory(kernel, n_lanes=min(n_lanes, 32), buf_sizes=buf_sizes, max_steps=max_steps, global_size=global_size)
    self.assertTrue(ok, msg)

  # Basic unary ops
  def test_neg(self): self._test_kernel(lambda T: -T([1.0, -2.0, 3.0, -4.0]))
  def test_relu(self): self._test_kernel(lambda T: T([-1.0, 0.0, 1.0, 2.0]).relu())
  def test_exp(self): self._test_kernel(lambda T: T([0.0, 1.0, 2.0]).exp())
  def test_log(self): self._test_kernel(lambda T: T([1.0, 2.0, 3.0]).log())
  def test_sin(self): self._test_kernel(lambda T: T([0.0, 1.0, 2.0]).sin())
  def test_sqrt(self): self._test_kernel(lambda T: T([1.0, 4.0, 9.0]).sqrt())
  def test_recip(self): self._test_kernel(lambda T: T([1.0, 2.0, 4.0]).reciprocal())

  # Binary ops
  def test_add(self): self._test_kernel(lambda T: T([1.0, 2.0]) + T([3.0, 4.0]))
  def test_sub(self): self._test_kernel(lambda T: T([5.0, 6.0]) - T([1.0, 2.0]))
  def test_mul(self): self._test_kernel(lambda T: T([2.0, 3.0]) * T([4.0, 5.0]))
  def test_div(self): self._test_kernel(lambda T: T([10.0, 20.0]) / T([2.0, 4.0]))
  def test_max_binary(self): self._test_kernel(lambda T: T([1.0, 5.0]).maximum(T([3.0, 2.0])))

  # Reductions
  def test_sum_reduce(self): self._test_kernel(lambda T: T.randn(64).sum())
  def test_max_reduce(self): self._test_kernel(lambda T: T.randn(64).max())
  def test_mean_reduce(self): self._test_kernel(lambda T: T.randn(32).mean())

  # Matmul - various sizes
  def test_gemm_4x4(self): self._test_kernel(lambda T: T.randn(4, 4) @ T.randn(4, 4), max_steps=100000)
  def test_gemm_8x8(self): self._test_kernel(lambda T: T.randn(8, 8) @ T.randn(8, 8), max_steps=200000)
  def test_gemm_16x16(self): self._test_kernel(lambda T: T.randn(16, 16) @ T.randn(16, 16), max_steps=500000)
  def test_gemv(self): self._test_kernel(lambda T: T.randn(1, 16) @ T.randn(16, 16), max_steps=100000)

  # Complex ops
  def test_softmax(self): self._test_kernel(lambda T: T.randn(16).softmax())
  def test_layernorm(self): self._test_kernel(lambda T: T.randn(8, 8).layernorm())

  # Memory patterns
  def test_contiguous(self): self._test_kernel(lambda T: T.randn(4, 4).permute(1, 0).contiguous())
  def test_reshape(self): self._test_kernel(lambda T: T.randn(16).reshape(4, 4).contiguous())
  def test_expand(self): self._test_kernel(lambda T: T.randn(4, 1).expand(4, 4).contiguous())

  # Cast ops
  def test_cast_int(self): self._test_kernel(lambda T: T.randn(16).int().float())
  def test_cast_half(self): self._test_kernel(lambda T: T.randn(16).half().float())

if __name__ == "__main__":
  unittest.main()
