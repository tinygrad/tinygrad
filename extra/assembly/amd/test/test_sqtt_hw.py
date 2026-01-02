#!/usr/bin/env python3
"""Hardware tests for SQTT decoder - validates decoding of real SQTT streams.

Run with: python -m pytest extra/assembly/amd/test/test_sqtt_hw.py -v -s
Requires AMD GPU with SQTT support.

For pretty trace output: DEBUG=2 python -m pytest extra/assembly/amd/test/test_sqtt_hw.py -v -s
"""
import os
os.environ["SQTT"] = "1"
os.environ["PROFILE"] = "1"
os.environ["SQTT_ITRACE_SE_MASK"] = "1"  # Enable instruction tracing on SE0
os.environ["SQTT_LIMIT_SE"] = "2"        # Force work to traced SE only

import unittest
from tinygrad.helpers import DEBUG, colored
from tinygrad.device import Device
from tinygrad.runtime.ops_amd import AMDProgram, ProfileSQTTEvent
from tinygrad.runtime.support.compiler_amd import HIPCompiler

from extra.assembly.amd.autogen.rdna3.ins import v_mov_b32_e32, v_add_f32_e32, v_mul_f32_e32, s_mov_b32, s_add_u32, s_nop, s_waitcnt, s_endpgm
from extra.assembly.amd.dsl import v, s
from extra.assembly.amd.sqtt import decode, LAYOUT_HEADER, WAVESTART, WAVEEND, INST, VALUINST, ALUEXEC, VMEMEXEC, InstOp, AluSrc, MemSrc

dev = Device["AMD"]

# ═══════════════════════════════════════════════════════════════════════════════
# PRETTY PRINTING
# ═══════════════════════════════════════════════════════════════════════════════

PACKET_COLORS = {
  "INST": "WHITE", "VALUINST": "BLACK",
  "VMEMEXEC": "yellow", "ALUEXEC": "yellow",
  "IMMEDIATE": "YELLOW", "IMMEDIATE_MASK": "YELLOW",
  "WAVERDY": "cyan", "WAVEALLOC": "cyan",
  "WAVEEND": "blue", "WAVESTART": "blue",
  "PERF": "magenta",
  "EVENT": "red", "EVENT_BIG": "red",
  "REG": "green",
  "LAYOUT_HEADER": "white",
  "TS_DELTA_SHORT": "BLACK", "NOP": "BLACK", "TS_WAVE_STATE": "BLACK",
  "SNAPSHOT": "white", "TS_DELTA_OR_MARK": "BLACK",
  "TS_DELTA_S8_W3": "BLACK", "TS_DELTA_S5_W2": "BLACK", "TS_DELTA_S5_W3": "BLACK",
  "UTILCTR": "green",
}

def format_packet(p, last_time: int = 0, time_offset: int = 0) -> str:
  """Format a packet for pretty printing."""
  name = type(p).__name__
  color = PACKET_COLORS.get(name, "white")

  fields = []
  if isinstance(p, INST):
    op = p.op
    op_name = op.name if isinstance(op, InstOp) else f"0x{op:02x}"
    fields = [f"wave={p.wave}", f"op={op_name}"]
    if p.flag1: fields.append("flag1")
    if p.flag2: fields.append("flag2")
  elif isinstance(p, VALUINST):
    fields = [f"wave={p.wave}"]
    if p.flag: fields.append("flag")
  elif isinstance(p, ALUEXEC):
    src_name = p.src.name if isinstance(p.src, AluSrc) else f"{p.src}"
    fields = [f"src={src_name}"]
  elif isinstance(p, VMEMEXEC):
    src_name = p.src.name if isinstance(p.src, MemSrc) else f"{p.src}"
    fields = [f"src={src_name}"]
  elif isinstance(p, WAVESTART):
    fields = [f"wave={p.wave}", f"simd={p.simd}", f"cu={p.cu}"]
  elif isinstance(p, WAVEEND):
    fields = [f"wave={p.wave}", f"simd={p.simd}", f"cu={p.cu}"]
  elif hasattr(p, '_values'):
    # Format hex fields appropriately
    hex_fields = {'snap', 'val32'}
    fields = [f"{k}=0x{v:x}" if k in hex_fields else f"{k}={v}" for k, v in p._values.items() if not k.startswith('_') and k != 'delta']

  return colored(f"{name:18s}", color) + " " + ", ".join(fields)

def get_wave_packets(packets: list) -> list:
  """Extract packets from WAVESTART to WAVEEND, filtering pure timing packets."""
  skip_types = {"NOP", "TS_DELTA_SHORT", "TS_WAVE_STATE", "TS_DELTA_OR_MARK", "TS_DELTA_S5_W2", "TS_DELTA_S5_W3", "TS_DELTA_S8_W3"}
  result = []
  in_wave = False
  for p in packets:
    name = type(p).__name__
    if isinstance(p, WAVESTART):
      in_wave = True
    if in_wave and name not in skip_types:
      result.append(p)
    if isinstance(p, WAVEEND):
      in_wave = False
  return result



def print_wave_trace(packets: list) -> None:
  """Print packets from WAVESTART to WAVEEND with normalized time."""
  wave_packets = get_wave_packets(packets)
  if not wave_packets:
    return
  time_offset = wave_packets[0]._time
  last_time = time_offset
  for p in wave_packets:
    print(format_packet(p, last_time, time_offset))
    last_time = p._time

def print_blobs(blobs: list[bytes], wave_only: bool = True) -> None:
  """Print traces for all blobs. wave_only=True filters to WAVESTART..WAVEEND only."""
  for i, blob in enumerate(blobs):
    packets = decode(blob)
    print(f"\n--- Blob {i}: {len(blob)} bytes, {len(packets)} packets ---")
    if wave_only:
      print_wave_trace(packets)
    else:
      print_all_packets(packets)

def print_all_packets(packets: list) -> None:
  """Print all packets, filtering out pure timing packets."""
  skip_types = {"NOP", "TS_DELTA_SHORT", "TS_WAVE_STATE", "TS_DELTA_OR_MARK", "TS_DELTA_S5_W2", "TS_DELTA_S5_W3", "TS_DELTA_S8_W3"}
  if not packets: return
  time_offset = packets[0]._time
  last_time = time_offset
  for p in packets:
    if type(p).__name__ not in skip_types:
      print(format_packet(p, last_time, time_offset))
    last_time = p._time

# ═══════════════════════════════════════════════════════════════════════════════
# ASSEMBLY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def assemble(instructions: list) -> bytes:
  return b''.join(inst.to_bytes() for inst in instructions)

def wrap_with_nops(instructions: list) -> list:
  """Add epilogue for clean SQTT timing."""
  return instructions + [s_nop(0)]*32 + [s_endpgm()]

def compile_asm_sqtt(instructions: list, alu_only: bool = False) -> AMDProgram:
  """Compile instructions to an AMDProgram for SQTT tracing.

  Args:
    instructions: List of instructions to compile
    alu_only: If True, use minimal kernel config with no kernargs/LDS/scratch
  Returns:
    Compiled AMDProgram ready to run
  """
  compiler = HIPCompiler(dev.arch)
  # Add NOPs before s_endpgm to flush pipeline and get clean timing
  code = assemble(wrap_with_nops(instructions))
  byte_str = ', '.join(f'0x{b:02x}' for b in code)

  if alu_only:
    asm_src = f""".text
.globl test
.p2align 8
.type test,@function
test:
.byte {byte_str}

.rodata
.p2align 6
.amdhsa_kernel test
  .amdhsa_next_free_vgpr 64
  .amdhsa_next_free_sgpr 8
  .amdhsa_wavefront_size32 1
  .amdhsa_group_segment_fixed_size 0
  .amdhsa_private_segment_fixed_size 0
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 0
amdhsa.kernels:
  - .name: test
    .symbol: test.kd
    .kernarg_segment_size: 0
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: 32
    .sgpr_count: 8
    .vgpr_count: 64
    .max_flat_workgroup_size: 1024
...
.end_amdgpu_metadata
"""
  else:
    asm_src = f""".text
.globl test
.p2align 8
.type test,@function
test:
.byte {byte_str}

.rodata
.p2align 6
.amdhsa_kernel test
  .amdhsa_next_free_vgpr 8
  .amdhsa_next_free_sgpr 16
  .amdhsa_wavefront_size32 1
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_kernarg_size 8
  .amdhsa_group_segment_fixed_size 0
  .amdhsa_private_segment_fixed_size 0
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
    .sgpr_count: 16
    .vgpr_count: 8
    .max_flat_workgroup_size: 1024
...
.end_amdgpu_metadata
"""

  lib = compiler.compile(asm_src)
  return AMDProgram(dev, "test", lib)

def run_asm_sqtt(instructions: list, n_lanes: int = 1, alu_only: bool = False) -> list[bytes]:
  """Compile and run instructions on AMD hardware, return SQTT blobs.

  Args:
    instructions: List of instructions to run
    n_lanes: Number of lanes to use
    alu_only: If True, use minimal kernel config with no kernargs/LDS/scratch
  """
  prg = compile_asm_sqtt(instructions, alu_only=alu_only)
  return run_prg_sqtt(prg, n_lanes=n_lanes, alu_only=alu_only)

def run_prg_sqtt(prg: AMDProgram, n_lanes: int = 1, alu_only: bool = False) -> list[bytes]:
  """Run a compiled AMDProgram and return SQTT blobs.

  Args:
    prg: Compiled AMDProgram to run
    n_lanes: Number of lanes to use
    alu_only: If True, don't allocate kernarg buffer
  """
  dev.profile_events.clear()
  if alu_only:
    prg(global_size=(1, 1, 1), local_size=(n_lanes, 1, 1), wait=True)
  else:
    out_gpu = dev.allocator.alloc(2048)
    prg(out_gpu, global_size=(1, 1, 1), local_size=(n_lanes, 1, 1), wait=True)
  return [ev.blob for ev in dev.profile_events if isinstance(ev, ProfileSQTTEvent)]

def run_prg_sqtt_batch(prg: AMDProgram, n_runs: int, n_lanes: int = 1) -> list[bytes]:
  """Run a compiled AMDProgram N times in a single queue submission and return SQTT blobs.

  This builds one queue with N kernel executions, submits it once, and collects SQTT.
  All N runs are captured in the same SQTT trace, reducing startup jitter.

  Args:
    prg: Compiled AMDProgram to run
    n_runs: Number of times to execute the kernel in the queue
    n_lanes: Number of lanes to use
  Returns:
    List of SQTT blobs (one per shader engine)
  """
  from typing import cast
  from tinygrad.runtime.ops_amd import AMDComputeQueue, SQTT_ITRACE_SE_MASK
  from tinygrad.device import Compiled
  import struct

  dev.profile_events.clear()

  # Build queue with sqtt_start, N kernel executions, sqtt_stop
  kernargs = prg.fill_kernargs([], ())
  q = cast(AMDComputeQueue, dev.hw_compute_queue_t())
  q.wait(dev.timeline_signal, dev.timeline_value - 1).memory_barrier()
  q.sqtt_start(dev.sqtt_buffers)

  # Execute kernel N times
  for _ in range(n_runs):
    q.exec(prg, kernargs, (1, 1, 1), (n_lanes, 1, 1))

  q.sqtt_stop(dev.sqtt_wptrs)
  q.signal(dev.timeline_signal, dev.next_timeline())
  q.submit(dev)
  dev.synchronize()

  # Collect SQTT blobs
  blobs = []
  for se, buf in enumerate(dev.sqtt_buffers):
    wptr = (dev.sqtt_wptrs.cpu_view().view(fmt='I')[se] & 0x1FFFFFFF) * 32
    if dev.target[:2] == (11, 0): wptr -= ((buf.va_addr // 32) & 0x1FFFFFFF) * 32
    if wptr > 0 and wptr <= buf.size:
      dev.allocator._copyout(sqtt_mv:=memoryview(bytearray(wptr)), buf)
      resbuf = (struct.pack('<Q', 0x11 | (4 << 13) | (0xf << 16) | (se << 24)) + bytes(sqtt_mv)) if dev.target[0] == 9 else bytes(sqtt_mv)
      blobs.append(resbuf)

  return blobs

def decode_all_blobs(blobs: list[bytes]) -> list:
  """Decode all blobs and combine packets."""
  all_packets = []
  for blob in blobs:
    all_packets.extend(decode(blob))
  return all_packets

def get_inst_ops(packets: list, traced_simd: int | None = None) -> set:
  """Extract all InstOp values from INST packets within WAVESTART..WAVEEND on traced SIMD."""
  ops = set()
  in_wave = False
  for p in packets:
    if isinstance(p, WAVESTART):
      in_wave = traced_simd is None or p.simd == traced_simd
    if in_wave and isinstance(p, INST):
      ops.add(p.op if isinstance(p.op, int) else p.op.value)
    if isinstance(p, WAVEEND):
      in_wave = False
  return ops

def count_valuinst(packets: list, traced_simd: int | None = None) -> int:
  """Count VALUINST packets within WAVESTART..WAVEEND on traced SIMD."""
  count = 0
  in_wave = False
  for p in packets:
    if isinstance(p, WAVESTART):
      in_wave = traced_simd is None or p.simd == traced_simd
    if in_wave and isinstance(p, VALUINST):
      count += 1
    if isinstance(p, WAVEEND):
      in_wave = False
  return count

# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

@unittest.skipIf(not hasattr(dev, 'profile_events'), "AMD device required")
class TestSQTTDecode(unittest.TestCase):
  """Test SQTT decoder with real hardware traces."""

  def test_basic_structure(self):
    """Verify basic SQTT stream structure: LAYOUT_HEADER, WAVESTART, instructions, WAVEEND."""
    blobs = run_asm_sqtt([v_mov_b32_e32(v[0], 0)])

    self.assertGreater(len(blobs), 0, "No SQTT data captured")
    packets = decode_all_blobs(blobs)

    self.assertGreater(len(packets), 0, "No packets decoded")
    self.assertGreater(len([p for p in packets if isinstance(p, LAYOUT_HEADER)]), 0, "No LAYOUT_HEADER packets")
    self.assertGreater(len([p for p in packets if isinstance(p, WAVESTART)]), 0, "No WAVESTART packets")
    self.assertGreater(len([p for p in packets if isinstance(p, WAVEEND)]), 0, "No WAVEEND packets")

    if DEBUG >= 2:
      print("\n=== Basic structure trace ===")
      print_trace(packets)

  def test_valu_instructions(self):
    """Verify VALU instructions produce INST or VALUINST packets."""
    instructions = [
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 2.0),
      v_add_f32_e32(v[2], v[0], v[1]),
      v_add_f32_e32(v[3], v[2], v[1]),
      v_mul_f32_e32(v[4], v[2], v[3]),
    ]
    blobs = run_asm_sqtt(instructions)

    self.assertGreater(len(blobs), 0, "No SQTT data captured")
    packets = decode_all_blobs(blobs)

    inst_packets = [p for p in packets if isinstance(p, (INST, VALUINST))]
    self.assertGreater(len(inst_packets), 0, "No INST/VALUINST packets for VALU instructions")

    if DEBUG >= 2:
      print("\n=== VALU instructions trace ===")
      print_trace(packets)

  def test_salu_instructions(self):
    """Verify SALU instructions produce appropriate packets."""
    instructions = [
      s_mov_b32(s[0], 0),
      s_mov_b32(s[1], 1),
      s_add_u32(s[2], s[0], s[1]),
      s_add_u32(s[3], s[2], s[1]),
      s_nop(0),
    ]
    blobs = run_asm_sqtt(instructions)

    self.assertGreater(len(blobs), 0, "No SQTT data captured")
    packets = decode_all_blobs(blobs)

    if DEBUG >= 2:
      print("\n=== SALU instructions trace ===")
      print_trace(packets)

  def test_timing_increases(self):
    """Verify time increases monotonically through packets within each blob."""
    instructions = [
      v_mov_b32_e32(v[0], 1.0),
      v_mov_b32_e32(v[1], 2.0),
      v_add_f32_e32(v[2], v[0], v[1]),
      v_mul_f32_e32(v[3], v[2], v[1]),
    ]
    blobs = run_asm_sqtt(instructions)

    self.assertGreater(len(blobs), 0, "No SQTT data captured")
    for blob in blobs:
      packets = decode(blob)
      prev_time = 0
      for p in packets:
        self.assertGreaterEqual(p._time, prev_time, f"Time decreased: {prev_time} -> {p._time}")
        prev_time = p._time

  def test_wave_id_consistency(self):
    """Verify wave IDs are consistent between WAVESTART/WAVEEND."""
    blobs = run_asm_sqtt([v_mov_b32_e32(v[0], 0)])

    self.assertGreater(len(blobs), 0, "No SQTT data captured")
    packets = decode_all_blobs(blobs)

    wavestarts = [p for p in packets if isinstance(p, WAVESTART)]
    waveends = [p for p in packets if isinstance(p, WAVEEND)]

    if wavestarts and waveends:
      start_waves = {p.wave for p in wavestarts}
      end_waves = {p.wave for p in waveends}
      self.assertTrue(start_waves & end_waves, "No matching wave IDs between WAVESTART and WAVEEND")

  def test_nop_sequence(self):
    """Test a sequence of NOP instructions."""
    blobs = run_asm_sqtt([s_nop(0), s_nop(0), s_nop(0)])

    self.assertGreater(len(blobs), 0, "No SQTT data captured")
    packets = decode_all_blobs(blobs)
    self.assertGreater(len(packets), 0, "No packets decoded")

    if DEBUG >= 2:
      print("\n=== NOP sequence trace ===")
      print_trace(packets, filter_timing=False)


if __name__ == "__main__":
  unittest.main()
