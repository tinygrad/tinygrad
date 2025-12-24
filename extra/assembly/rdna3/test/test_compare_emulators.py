# Test to compare Python and Rust RDNA3 emulators by running real tinygrad kernels
import unittest, ctypes, os
from dataclasses import dataclass
from pathlib import Path
from extra.assembly.rdna3.emu import WaveState, decode_program, step_wave, WAVE_SIZE

REMU_PATH = Path(__file__).parents[3] / "remu/target/release/libremu.so"

@dataclass
class KernelInfo:
  code: bytes
  global_size: tuple[int, int, int]
  local_size: tuple[int, int, int]
  buf_idxs: list[int]  # indices into shared buffer pool
  buf_sizes: list[int]  # sizes for each buffer index

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
    self.lib.wave_init_lds.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    self.lib.wave_free.argtypes = [ctypes.c_void_p]
    self.ctx = None

  def create(self, kernel: bytes, n_lanes: int):
    kernel_buf = (ctypes.c_char * len(kernel))(*kernel)
    self.ctx = self.lib.wave_create(ctypes.addressof(kernel_buf), len(kernel), n_lanes)
    self._kernel_buf = kernel_buf

  def step(self) -> int: return self.lib.wave_step(self.ctx)
  def set_sgpr(self, idx: int, val: int): self.lib.wave_set_sgpr(self.ctx, idx, val)
  def set_vgpr(self, lane: int, idx: int, val: int): self.lib.wave_set_vgpr(self.ctx, lane, idx, val)
  def init_lds(self, size: int): self.lib.wave_init_lds(self.ctx, size)

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

def run_single_kernel(kernel: bytes, n_lanes: int, args_ptr: int, global_size: tuple[int, int, int],
                      program, max_steps: int, debug: bool, trace_len: int, kernel_idx: int = 0) -> tuple[bool, str, int]:
  """Run a single kernel through both emulators. Returns (success, message, total_steps)."""
  gx, gy, gz = global_size
  total_steps = 0

  for gidz in range(gz):
    for gidy in range(gy):
      for gidx in range(gx):
        rust = RustEmulator()
        python = PythonEmulator()
        rust.create(kernel, n_lanes)
        python.create(kernel, n_lanes)

        # Initialize LDS (64KB, standard size for AMD GPUs)
        rust.init_lds(65536)

        for emu in [rust, python]:
          emu.set_sgpr(0, args_ptr & 0xffffffff)
          emu.set_sgpr(1, (args_ptr >> 32) & 0xffffffff)
          emu.set_sgpr(13, gidx)
          emu.set_sgpr(14, gidy)
          emu.set_sgpr(15, gidz)

        step = 0
        trace: list[tuple[int, int, str, StateSnapshot, StateSnapshot]] = []
        try:
          while step < max_steps:
            rust_before = rust.get_snapshot()
            python_before = python.get_snapshot()

            inst = program.get(python_before.pc)
            inst_str = inst.disasm() if inst else f"unknown at PC={python_before.pc}"
            trace.append((step, python_before.pc, inst_str, rust_before, python_before))
            if len(trace) > trace_len: trace.pop(0)

            if debug: print(f"K{kernel_idx} WG({gidx},{gidy},{gidz}) Step {step}: PC={python_before.pc}, inst={inst_str}")

            diffs = rust_before.diff(python_before, n_lanes)
            if diffs:
              trace_lines = []
              for s, pc, d, rb, pb in trace[:-1]:
                trace_lines.append(f"    step {s}: PC={pc:3d} {d}")
                if trace.index((s, pc, d, rb, pb)) < len(trace) - 2:
                  next_rb, next_pb = trace[trace.index((s, pc, d, rb, pb)) + 1][3:5]
                  inst_diffs = rb.diff(next_rb, n_lanes)
                  if inst_diffs: trace_lines.append(f"             rust changes: {', '.join(inst_diffs[:3])}")
              trace_str = "\n".join(trace_lines)
              return False, f"K{kernel_idx} WG({gidx},{gidy},{gidz}) Step {step} before inst '{inst_str}': states differ:\n  " + "\n  ".join(diffs[:10]) + f"\n  Recent instructions:\n{trace_str}", total_steps

            rust_result = rust.step()
            python_result = python.step()

            if rust_result != python_result:
              trace_str = "\n".join(f"    step {s}: PC={pc:3d} {d}" for s, pc, d, _, _ in trace)
              return False, f"K{kernel_idx} WG({gidx},{gidy},{gidz}) Step {step}: different return codes: rust={rust_result}, python={python_result}, inst={inst_str}\n  Recent instructions:\n{trace_str}", total_steps

            if rust_result == -1:
              total_steps += step + 1
              break
            if rust_result == 1:
              total_steps += step + 1
              break
            if rust_result < 0 and rust_result != -2:
              return False, f"K{kernel_idx} WG({gidx},{gidy},{gidz}) Step {step}: error code {rust_result}", total_steps

            step += 1
          else:
            return False, f"K{kernel_idx} WG({gidx},{gidy},{gidz}) Max steps ({max_steps}) reached", total_steps
        finally:
          rust.free()

  return True, f"Completed {gx*gy*gz} workgroups", total_steps

def compare_emulators_multi_kernel(kernels: list[KernelInfo], buf_pool: dict[int, int], max_steps: int = 1000,
                                    debug: bool = False, trace_len: int = 10) -> tuple[bool, str]:
  """Run all kernels through both emulators with shared buffer pool."""
  from extra.assembly.rdna3.emu import set_valid_mem_ranges, decode_program

  # Allocate shared buffer pool with padding for over-reads (GPU loads up to 16 bytes at once)
  buf_id_to_ptr: dict[int, int] = {}
  buffers = []
  for buf_id, size in buf_pool.items():
    padded_size = ((size + 15) // 16) * 16 + 16  # round up to 16 bytes + extra padding
    buf = (ctypes.c_uint8 * padded_size)(*[0] * padded_size)
    buffers.append((buf, padded_size))
    buf_id_to_ptr[buf_id] = ctypes.addressof(buf)

  # Set up valid memory ranges
  ranges = {(ctypes.addressof(b), size) for b, size in buffers}

  total_steps = 0
  for ki, kernel in enumerate(kernels):
    # Create args array for this kernel's buffers
    args = (ctypes.c_uint64 * len(kernel.buf_idxs))(*[buf_id_to_ptr[bid] for bid in kernel.buf_idxs])
    args_ptr = ctypes.addressof(args)

    # Update valid ranges to include this args array
    kernel_ranges = ranges | {(args_ptr, ctypes.sizeof(args))}
    set_valid_mem_ranges(kernel_ranges)

    program = decode_program(kernel.code)
    n_lanes = kernel.local_size[0] * kernel.local_size[1] * kernel.local_size[2]

    ok, msg, steps = run_single_kernel(
      kernel.code, min(n_lanes, 32), args_ptr, kernel.global_size,
      program, max_steps, debug, trace_len, ki
    )
    total_steps += steps
    if not ok:
      return False, msg

  return True, f"Completed {len(kernels)} kernels, {total_steps} total steps"

def compare_emulators_with_memory(kernel: bytes, n_lanes: int, buf_sizes: list, max_steps: int = 1000, debug: bool = False,
                                   global_size: tuple[int, int, int] = (1, 1, 1), trace_len: int = 10) -> tuple[bool, str]:
  """Run both emulators with memory set up for tinygrad kernels, executing all workgroups. Legacy wrapper."""
  from extra.assembly.rdna3.emu import set_valid_mem_ranges, decode_program

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

  program = decode_program(kernel)
  ok, msg, _ = run_single_kernel(kernel, n_lanes, args_ptr, global_size, program, max_steps, debug, trace_len)
  return ok, msg

def get_kernels_from_tinygrad(op_fn) -> tuple[list[KernelInfo], dict[int, int]]:
  """Compile a tinygrad operation and extract all kernels with their buffer mappings."""
  os.environ["AMD"] = "1"
  from tinygrad import Tensor
  from tinygrad.runtime.support.elf import elf_loader

  out = op_fn(Tensor)
  sched = out.schedule()
  kernels = []
  buf_pool: dict[int, int] = {}  # buffer id -> size

  for ei in sched:
    if ei.ast.op.name == 'SINK':
      lowered = ei.lower()
      if lowered.prg and lowered.prg.p.lib:
        lib = bytes(lowered.prg.p.lib)
        _, sections, _ = elf_loader(lib)
        for sec in sections:
          if sec.name == '.text':
            buf_idxs = []
            buf_sizes = []
            for b in lowered.bufs:
              buf_id = id(b)
              if buf_id not in buf_pool:
                buf_pool[buf_id] = b.nbytes
              buf_idxs.append(buf_id)
              buf_sizes.append(b.nbytes)
            kernels.append(KernelInfo(
              code=bytes(sec.content),
              global_size=tuple(lowered.prg.p.global_size),
              local_size=tuple(lowered.prg.p.local_size),
              buf_idxs=buf_idxs,
              buf_sizes=buf_sizes
            ))
  if not kernels: raise RuntimeError("No kernel found")
  return kernels, buf_pool

def get_kernel_from_tinygrad(op_fn) -> tuple[bytes, tuple[int, int, int], tuple[int, int, int], list]:
  """Compile a tinygrad operation and extract the last (main) kernel binary. Legacy wrapper."""
  kernels, _ = get_kernels_from_tinygrad(op_fn)
  k = kernels[-1]
  return k.code, k.global_size, k.local_size, k.buf_sizes

@unittest.skipUnless(REMU_PATH.exists(), "libremu.so not found")
class TestTinygradKernels(unittest.TestCase):
  """Compare emulators on real tinygrad-compiled kernels."""

  def _test_kernel(self, op_fn, max_steps=10000):
    kernels, buf_pool = get_kernels_from_tinygrad(op_fn)
    ok, msg = compare_emulators_multi_kernel(kernels, buf_pool, max_steps=max_steps)
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
  def test_sum_reduce(self): self._test_kernel(lambda T: T.empty(64).sum())
  def test_max_reduce(self): self._test_kernel(lambda T: T.empty(64).max())
  def test_mean_reduce(self): self._test_kernel(lambda T: T.empty(32).mean())

  # Matmul - various sizes
  def test_gemm_4x4(self): self._test_kernel(lambda T: T.empty(4, 4) @ T.empty(4, 4), max_steps=100000)
  def test_gemm_8x8(self): self._test_kernel(lambda T: T.empty(8, 8) @ T.empty(8, 8), max_steps=200000)
  @unittest.skip("too slow")
  def test_gemm_16x16(self): self._test_kernel(lambda T: T.empty(16, 16) @ T.empty(16, 16), max_steps=500000)
  def test_gemv(self): self._test_kernel(lambda T: T.empty(1, 16) @ T.empty(16, 16), max_steps=100000)

  # Complex ops
  def test_softmax(self): self._test_kernel(lambda T: T.empty(16).softmax())
  def test_layernorm(self): self._test_kernel(lambda T: T.empty(8, 8).layernorm())

  # Memory patterns
  def test_contiguous(self): self._test_kernel(lambda T: T.empty(4, 4).permute(1, 0).contiguous())
  def test_reshape(self): self._test_kernel(lambda T: (T.empty(16) + 1).reshape(4, 4).contiguous())
  def test_expand(self): self._test_kernel(lambda T: T.empty(4, 1).expand(4, 4).contiguous())

  # Cast ops
  def test_cast_int(self): self._test_kernel(lambda T: T.empty(16).int().float())
  def test_cast_half(self): self._test_kernel(lambda T: T.empty(16).half().float())

  # Min/max (uses comparison internally)
  def test_min_binary(self): self._test_kernel(lambda T: T([1.0, 5.0, 3.0]).minimum(T([3.0, 2.0, 4.0])))

  # Comparison ops (test VOPC instructions) - use 32+ elements to force vector instructions
  def test_cmp_lt(self): self._test_kernel(lambda T: (T.empty(64) < T.empty(64)).where(T.empty(64), T.empty(64)))
  def test_cmp_eq(self): self._test_kernel(lambda T: (T.empty(64) == T.empty(64)).where(T.empty(64), T.empty(64)))
  def test_where(self): self._test_kernel(lambda T: (T.empty(64) > 0).where(T.empty(64), T.empty(64)))

  # Bitwise ops
  def test_bitwise_and(self): self._test_kernel(lambda T: T([0xF0, 0x0F, 0xFF]).int() & T([0x0F, 0x0F, 0x00]).int())
  def test_bitwise_or(self): self._test_kernel(lambda T: T([0xF0, 0x0F, 0x00]).int() | T([0x0F, 0x0F, 0xFF]).int())
  def test_bitwise_xor(self): self._test_kernel(lambda T: T([0xFF, 0x0F, 0xF0]).int() ^ T([0x0F, 0xF0, 0xF0]).int())

  # Integer ops - use 32+ elements to force vector instructions
  def test_int_add(self): self._test_kernel(lambda T: (T.empty(64).int() + T.empty(64).int()).float())
  def test_int_mul(self): self._test_kernel(lambda T: (T.empty(64).int() * T.empty(64).int()).float())
  def test_int_mod(self): self._test_kernel(lambda T: (T.empty(64).int().abs() % (T.empty(64).int().abs() + 1)).float())

  # More math ops - use 32+ elements to force vector instructions
  def test_abs(self): self._test_kernel(lambda T: T.empty(64).abs())
  def test_floor(self): self._test_kernel(lambda T: T.empty(64).floor())
  def test_ceil(self): self._test_kernel(lambda T: T.empty(64).ceil())
  def test_trunc(self): self._test_kernel(lambda T: T.empty(64).trunc())

  # Fused ops
  def test_fma(self): self._test_kernel(lambda T: (T([1.0, 2.0]) * T([3.0, 4.0]) + T([5.0, 6.0])))

  # Argmax/argmin (tests different reduction pattern) - use 32+ elements to force vector instructions
  def test_argmax(self): self._test_kernel(lambda T: T.empty(64).argmax())
  def test_argmin(self): self._test_kernel(lambda T: T.empty(64).argmin())

  # Exact value tests - use 32+ elements to force vector instructions (small tensors use scalar ops which Rust emu doesn't fully support)
  def test_abs_exact(self): self._test_kernel(lambda T: T([-1., 0., 1.]*11).abs())  # 33 elements
  def test_neg_exact(self): self._test_kernel(lambda T: -T([-1., 0., 1.]*11))
  def test_log_special(self): self._test_kernel(lambda T: T([1., 2., 0.5]*11).log())
  def test_exp_exact(self): self._test_kernel(lambda T: T([0., 1., -1.]*11).exp())
  def test_reciprocal_exact(self): self._test_kernel(lambda T: T([1., 2., 0.5]*11).reciprocal())

  # Integer division and mod - use 32+ elements
  def test_int_div(self): self._test_kernel(lambda T: (T([10, 20, 30]*11).int() // T([3, 4, 5]*11).int()).float())
  def test_int_neg(self): self._test_kernel(lambda T: (-T([1, -2, 3]*11).int()).float())

  # Mixed precision - use 32+ elements
  def test_half_add(self): self._test_kernel(lambda T: (T([1., 2.]*16).half() + T([3., 4.]*16).half()).float())
  def test_half_mul(self): self._test_kernel(lambda T: (T([2., 3.]*16).half() * T([4., 5.]*16).half()).float())

  # Matrix ops - patterns from test_ops.py failures
  def test_cat(self): self._test_kernel(lambda T: T.empty(32, 64).cat(T.empty(32, 64), dim=1))
  def test_gather(self): self._test_kernel(lambda T: T.empty(64).gather(0, T.arange(32).int()))

if __name__ == "__main__":
  unittest.main()
