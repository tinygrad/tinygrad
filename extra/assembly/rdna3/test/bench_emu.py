#!/usr/bin/env python3
"""Benchmark comparing Python vs Rust RDNA3 emulators using full execution (not single-step)."""
import ctypes, time, os, struct
from pathlib import Path

# Set AMD=1 before importing tinygrad
os.environ["AMD"] = "1"

from extra.assembly.rdna3.emu import run_asm as python_run_asm, set_valid_mem_ranges, decode_program

REMU_PATH = Path(__file__).parents[3] / "remu/target/release/libremu.so"
if not REMU_PATH.exists():
  REMU_PATH = Path(__file__).parents[3] / "remu/target/release/libremu.dylib"

def get_rust_remu():
  """Load the Rust libremu shared library."""
  if not REMU_PATH.exists():
    return None
  remu = ctypes.CDLL(str(REMU_PATH))
  remu.run_asm.restype = ctypes.c_int32
  remu.run_asm.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                           ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
  return remu

def count_instructions(kernel: bytes) -> int:
  """Count instructions in a kernel."""
  program = decode_program(kernel)
  return len(program)

def setup_buffers(buf_sizes: list[int]):
  """Allocate buffers and return args pointer + valid ranges."""
  buffers = [(ctypes.c_uint8 * size)(*[0] * size) for size in buf_sizes]
  args = (ctypes.c_uint64 * len(buffers))(*[ctypes.addressof(b) for b in buffers])
  args_ptr = ctypes.addressof(args)
  ranges = {(ctypes.addressof(b), len(b)) for b in buffers}
  ranges.add((args_ptr, ctypes.sizeof(args)))
  return buffers, args, args_ptr, ranges

def benchmark_emulator(name: str, run_fn, kernel: bytes, global_size, local_size, args_ptr, iterations: int = 5):
  """Benchmark an emulator and return average time."""
  gx, gy, gz = global_size
  lx, ly, lz = local_size
  kernel_buf = (ctypes.c_char * len(kernel))(*kernel)
  lib_ptr = ctypes.addressof(kernel_buf)

  # Warmup
  run_fn(lib_ptr, len(kernel), gx, gy, gz, lx, ly, lz, args_ptr)

  # Timed runs
  times = []
  for _ in range(iterations):
    start = time.perf_counter()
    result = run_fn(lib_ptr, len(kernel), gx, gy, gz, lx, ly, lz, args_ptr)
    end = time.perf_counter()
    if result != 0:
      print(f"  {name} returned error: {result}")
      return None
    times.append(end - start)

  avg = sum(times) / len(times)
  return avg

def create_test_kernel(n_ops: int) -> bytes:
  """Create a synthetic kernel with n_ops vector operations.

  This creates a kernel that does:
    v0 = v0 + v1 (repeated n_ops times with various ops)
    s_endpgm
  """
  instructions = []

  # VOP2 V_ADD_F32: encoding = 0x03 (bits 25-30), vdst=v0, src0=v0 (256), vsrc1=v1 (1)
  # Format: [31:25]=0b0000011, [24:17]=vsrc1, [16:9]=vdst, [8:0]=src0
  # v_add_f32 v0, v0, v1
  v_add_f32 = (0b0000011 << 25) | (1 << 17) | (0 << 9) | 256

  # VOP2 V_MUL_F32: encoding = 0x08
  v_mul_f32 = (0b0001000 << 25) | (1 << 17) | (0 << 9) | 256

  # VOP2 V_MAX_F32: encoding = 0x10
  v_max_f32 = (0b0010000 << 25) | (1 << 17) | (0 << 9) | 256

  # VOP2 V_MIN_F32: encoding = 0x0F
  v_min_f32 = (0b0001111 << 25) | (1 << 17) | (0 << 9) | 256

  ops = [v_add_f32, v_mul_f32, v_max_f32, v_min_f32]

  for i in range(n_ops):
    instructions.append(ops[i % len(ops)])

  # S_ENDPGM: SOPP format, op=48 (0x30)
  # [31:23]=0b101111111, [22:16]=op(48), [15:0]=simm16(0)
  s_endpgm = (0b101111111 << 23) | (48 << 16) | 0
  instructions.append(s_endpgm)

  return b''.join(struct.pack('<I', inst) for inst in instructions)

# Synthetic test cases: (name, n_ops, n_workgroups, local_size)
SYNTHETIC_TESTS = [
  ("10_ops_1wg", 10, (1, 1, 1), (32, 1, 1)),
  ("100_ops_1wg", 100, (1, 1, 1), (32, 1, 1)),
  ("1000_ops_1wg", 1000, (1, 1, 1), (32, 1, 1)),
  ("100_ops_4wg", 100, (4, 1, 1), (32, 1, 1)),
  ("100_ops_16wg", 100, (16, 1, 1), (32, 1, 1)),
  ("500_ops_1wg", 500, (1, 1, 1), (32, 1, 1)),
  ("500_ops_4wg", 500, (4, 1, 1), (32, 1, 1)),
]

def main():
  rust_remu = get_rust_remu()
  if rust_remu is None:
    print("Rust libremu not found. Build it with: cargo build --release --manifest-path extra/remu/Cargo.toml")
    print("Running Python-only benchmark...")

  print("=" * 80)
  print("RDNA3 Emulator Benchmark: Python vs Rust (Synthetic Workloads)")
  print("=" * 80)
  print()

  results = []

  for name, n_ops, global_size, local_size in SYNTHETIC_TESTS:
    print(f"Test: {name}")
    kernel = create_test_kernel(n_ops)
    n_insts = count_instructions(kernel)
    n_workgroups = global_size[0] * global_size[1] * global_size[2]
    n_threads = local_size[0] * local_size[1] * local_size[2]
    total_work = n_insts * n_workgroups * n_threads

    print(f"  {n_insts} instructions, {n_workgroups} workgroup(s), {n_threads} threads/wg")
    print(f"  Total work: {total_work:,} instruction-thread executions")

    # Setup minimal buffers for args
    buf_sizes = [4096]  # 4KB dummy buffer
    buffers, args, args_ptr, ranges = setup_buffers(buf_sizes)
    set_valid_mem_ranges(ranges)

    # Benchmark Python
    print(f"  Benchmarking Python...", end=" ", flush=True)
    py_time = benchmark_emulator("Python", python_run_asm, kernel, global_size, local_size, args_ptr, iterations=3)
    if py_time is not None:
      print(f"{py_time*1000:.3f} ms ({total_work/py_time/1e6:.2f}M ops/s)")
    else:
      print("error")
      py_time = float('inf')

    # Benchmark Rust (if available)
    if rust_remu:
      print(f"  Benchmarking Rust...", end=" ", flush=True)
      rust_time = benchmark_emulator("Rust", rust_remu.run_asm, kernel, global_size, local_size, args_ptr, iterations=3)
      if rust_time is not None:
        print(f"{rust_time*1000:.3f} ms ({total_work/rust_time/1e6:.2f}M ops/s)")
      else:
        print("error")
        rust_time = float('inf')
    else:
      rust_time = None

    results.append((name, n_insts, n_workgroups, py_time, rust_time))
    print()

  # Summary table
  print("=" * 80)
  print("SUMMARY")
  print("=" * 80)
  print(f"{'Test':<20} {'Insts':<8} {'WGs':<6} {'Python (ms)':<15} {'Rust (ms)':<15} {'Speedup':<10}")
  print("-" * 80)

  for name, n_insts, n_wgs, py_time, rust_time in results:
    py_ms = f"{py_time*1000:.3f}" if py_time != float('inf') else "error"
    if rust_time is not None:
      rust_ms = f"{rust_time*1000:.3f}" if rust_time != float('inf') else "error"
      if py_time != float('inf') and rust_time != float('inf'):
        speedup = f"{py_time/rust_time:.1f}x"
      else:
        speedup = "N/A"
    else:
      rust_ms = "N/A"
      speedup = "N/A"
    print(f"{name:<20} {n_insts:<8} {n_wgs:<6} {py_ms:<15} {rust_ms:<15} {speedup:<10}")

  print()
  print("To improve Python emulator speed, focus on the hot path in step_wave() and exec_* functions.")
  print()

if __name__ == "__main__":
  main()
