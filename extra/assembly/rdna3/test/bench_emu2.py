#!/usr/bin/env python3
"""Benchmark for RDNA3 Python emulator (emu2.py) - measures instruction throughput and identifies bottlenecks."""
import ctypes, time, struct, cProfile, pstats, io
from pathlib import Path

from extra.assembly.rdna3.emu2 import run_asm, set_valid_mem_ranges, decode_program, step_wave, WaveState, WAVE_SIZE

def count_instructions(kernel: bytes) -> int:
  """Count instructions in a kernel."""
  return len(decode_program(kernel))

def setup_buffers(buf_sizes: list[int], init_data: dict[int, bytes] | None = None):
  """Allocate buffers and return args pointer + valid ranges."""
  if init_data is None: init_data = {}
  buffers = []
  for i, size in enumerate(buf_sizes):
    padded = ((size + 15) // 16) * 16 + 16
    data = init_data.get(i, b'\x00' * padded)
    data_list = list(data) + [0] * (padded - len(data))
    buf = (ctypes.c_uint8 * padded)(*data_list[:padded])
    buffers.append(buf)
  args = (ctypes.c_uint64 * len(buffers))(*[ctypes.addressof(b) for b in buffers])
  args_ptr = ctypes.addressof(args)
  ranges = {(ctypes.addressof(b), len(b)) for b in buffers}
  ranges.add((args_ptr, ctypes.sizeof(args)))
  return buffers, args, args_ptr, ranges

def benchmark_emulator(kernel: bytes, global_size: tuple, local_size: tuple, args_ptr: int, iterations: int = 3) -> float | None:
  """Benchmark the emulator and return average time."""
  gx, gy, gz = global_size
  lx, ly, lz = local_size
  kernel_buf = (ctypes.c_char * len(kernel))(*kernel)
  lib_ptr = ctypes.addressof(kernel_buf)

  # Warmup
  run_asm(lib_ptr, len(kernel), gx, gy, gz, lx, ly, lz, args_ptr)

  # Timed runs
  times = []
  for _ in range(iterations):
    start = time.perf_counter()
    result = run_asm(lib_ptr, len(kernel), gx, gy, gz, lx, ly, lz, args_ptr)
    end = time.perf_counter()
    if result != 0:
      print(f"  Error: returned {result}")
      return None
    times.append(end - start)

  return sum(times) / len(times)

def create_synthetic_kernel(n_ops: int) -> bytes:
  """Create a synthetic kernel with n_ops vector operations."""
  instructions = []
  # VOP2 instructions: v_add_f32, v_mul_f32, v_max_f32, v_min_f32
  ops = [
    (0b0000011 << 25) | (1 << 17) | (0 << 9) | 256,  # v_add_f32 v0, v0, v1
    (0b0001000 << 25) | (1 << 17) | (0 << 9) | 256,  # v_mul_f32 v0, v0, v1
    (0b0010000 << 25) | (1 << 17) | (0 << 9) | 256,  # v_max_f32 v0, v0, v1
    (0b0001111 << 25) | (1 << 17) | (0 << 9) | 256,  # v_min_f32 v0, v0, v1
  ]
  for i in range(n_ops):
    instructions.append(ops[i % len(ops)])
  # S_ENDPGM
  instructions.append((0b101111111 << 23) | (48 << 16) | 0)
  return b''.join(struct.pack('<I', inst) for inst in instructions)

def profile_emulator(kernel: bytes, global_size: tuple, local_size: tuple, args_ptr: int, n_runs: int = 1) -> str:
  """Profile the emulator to find bottlenecks."""
  gx, gy, gz = global_size
  lx, ly, lz = local_size
  kernel_buf = (ctypes.c_char * len(kernel))(*kernel)
  lib_ptr = ctypes.addressof(kernel_buf)

  pr = cProfile.Profile()
  pr.enable()
  for _ in range(n_runs):
    run_asm(lib_ptr, len(kernel), gx, gy, gz, lx, ly, lz, args_ptr)
  pr.disable()

  s = io.StringIO()
  ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
  ps.print_stats(30)
  return s.getvalue()

def measure_step_rate(kernel: bytes, n_steps: int = 1000) -> float:
  """Measure raw step_wave() performance (steps per second)."""
  program = decode_program(kernel)
  if not program: return 0.0

  st = WaveState()
  st.exec_mask = 0xffffffff
  lds = bytearray(65536)
  n_lanes = 32

  start = time.perf_counter()
  for _ in range(n_steps):
    st.pc = 0
    while st.pc in program:
      result = step_wave(program, st, lds, n_lanes)
      if result == -1: break
  elapsed = time.perf_counter() - start
  return n_steps / elapsed if elapsed > 0 else 0

# Test configurations
SYNTHETIC_TESTS = [
  ("synthetic_10ops", 10, (1, 1, 1), (32, 1, 1)),
  ("synthetic_100ops", 100, (1, 1, 1), (32, 1, 1)),
  ("synthetic_500ops", 500, (1, 1, 1), (32, 1, 1)),
  ("synthetic_100ops_4wg", 100, (4, 1, 1), (32, 1, 1)),
  ("synthetic_100ops_16wg", 100, (16, 1, 1), (32, 1, 1)),
]

def main():
  import argparse
  parser = argparse.ArgumentParser(description="Benchmark RDNA3 emu2.py emulator")
  parser.add_argument("--profile", action="store_true", help="Profile emulator to find bottlenecks")
  parser.add_argument("--iterations", type=int, default=3, help="Number of iterations per benchmark")
  parser.add_argument("--step-rate", action="store_true", help="Measure raw step_wave() rate")
  args = parser.parse_args()

  print("=" * 80)
  print("RDNA3 emu2.py Benchmark")
  print("=" * 80)

  results = []

  print("\n[SYNTHETIC WORKLOADS]")
  print("-" * 80)

  for name, n_ops, global_size, local_size in SYNTHETIC_TESTS:
    kernel = create_synthetic_kernel(n_ops)
    n_insts = count_instructions(kernel)
    n_workgroups = global_size[0] * global_size[1] * global_size[2]
    n_threads = local_size[0] * local_size[1] * local_size[2]
    total_work = n_insts * n_workgroups * n_threads

    print(f"\n{name}: {n_insts} insts × {n_workgroups} WGs × {n_threads} threads = {total_work:,} ops")

    buf_sizes = [4096]
    buffers, args_arr, args_ptr, ranges = setup_buffers(buf_sizes)
    set_valid_mem_ranges(ranges)

    if args.step_rate:
      step_rate = measure_step_rate(kernel, 100)
      print(f"  Step rate: {step_rate:.0f} steps/s")

    emu_time = benchmark_emulator(kernel, global_size, local_size, args_ptr, args.iterations)
    if emu_time:
      ops_rate = total_work / emu_time / 1e6
      print(f"  Time: {emu_time*1000:8.3f} ms  ({ops_rate:7.2f} M ops/s)")
      results.append((name, n_insts, n_workgroups, emu_time, ops_rate))

    if args.profile:
      print("\n  [PROFILE - Top 20 functions]")
      profile_output = profile_emulator(kernel, global_size, local_size, args_ptr)
      for line in profile_output.split('\n')[5:25]:
        if line.strip(): print(f"    {line}")

  # Summary
  print("\n" + "=" * 80)
  print("SUMMARY")
  print("=" * 80)
  print(f"{'Name':<25} {'Insts':<8} {'WGs':<6} {'Time (ms)':<12} {'M ops/s':<12}")
  print("-" * 80)
  for name, n_insts, n_wgs, emu_time, ops_rate in results:
    print(f"{name:<25} {n_insts:<8} {n_wgs:<6} {emu_time*1000:<12.3f} {ops_rate:<12.2f}")

if __name__ == "__main__":
  main()
