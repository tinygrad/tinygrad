#!/usr/bin/env python3
"""SQTT InstOp discovery tool - finds instruction opcodes by running different instructions.

Run with: DEBUG=1 python extra/assembly/amd/test/discover_instops.py
For full traces: DEBUG=2 python extra/assembly/amd/test/discover_instops.py
"""
import os
os.environ["SQTT"] = "1"
os.environ["PROFILE"] = "1"
os.environ["SQTT_ITRACE_SE_MASK"] = "2"  # Enable instruction tracing on SE1
os.environ["SQTT_LIMIT_SE"] = "2"        # Force work to traced SE only

from tinygrad.helpers import DEBUG, colored
from tinygrad.runtime.ops_amd import SQTT_SIMD_SEL

from extra.assembly.amd.autogen.rdna3.ins import (
  # VALU - basic (these are safe, just register ops)
  v_mov_b32_e32, v_add_f32_e32, v_mul_f32_e32,
  v_and_b32_e32, v_or_b32_e32, v_xor_b32_e32,
  v_lshlrev_b32_e32, v_lshrrev_b32_e32,
  # VALU - transcendental
  v_exp_f32_e32, v_log_f32_e32, v_rcp_f32_e32, v_sqrt_f32_e32,
  # VALU - 64-bit
  v_lshlrev_b64, v_lshrrev_b64,
  # VALU - compare (writes to VCC, safe)
  v_cmp_eq_u32_e32,
  v_cmpx_eq_u32_e32,
  # SALU - basic (safe, just register ops)
  s_mov_b32, s_add_u32, s_and_b32, s_or_b32,
  s_lshl_b32, s_lshr_b32,
  s_nop, s_endpgm,
  # SALU - branch (safe if offset is 0 = next instruction)
  s_branch, s_cbranch_scc0, s_cbranch_execz,
  # SALU - message
  s_sendmsg,
)
from extra.assembly.amd.dsl import v, s
from extra.assembly.amd.sqtt import InstOp, INST

from extra.assembly.amd.test.test_sqtt_hw import (
  run_asm_sqtt, decode_all_blobs, get_inst_ops, print_blobs
)

# ═══════════════════════════════════════════════════════════════════════════════
# INSTRUCTION TEST CASES - only safe instructions that don't access memory
# ═══════════════════════════════════════════════════════════════════════════════

INSTRUCTION_TESTS: dict[str, tuple[str, list]] = {
  # SALU (0x0) - scalar ALU, just register operations
  "SALU_mov": ("s_mov_b32", [s_mov_b32(s[0], 0), s_mov_b32(s[1], 1)]),
  "SALU_add": ("s_add_u32", [s_mov_b32(s[0], 1), s_mov_b32(s[1], 2), s_add_u32(s[2], s[0], s[1])]),
  "SALU_logic": ("s_and/or", [s_and_b32(s[2], s[0], s[1]), s_or_b32(s[3], s[0], s[1])]),
  "SALU_shift": ("s_lshl/lshr", [s_lshl_b32(s[2], s[0], 1), s_lshr_b32(s[3], s[0], 1)]),
  "SALU_nop": ("s_nop", [s_nop(0)]),

  # JUMP (0x3) - branch to next instruction (offset 0)
  "JUMP_branch": ("s_branch", [s_branch(0)]),

  # VALU (0xb) - vector ALU, just register operations
  "VALU_mov": ("v_mov_b32", [v_mov_b32_e32(v[0], 0), v_mov_b32_e32(v[1], 1.0)]),
  "VALU_add": ("v_add_f32", [v_mov_b32_e32(v[0], 1.0), v_mov_b32_e32(v[1], 2.0), v_add_f32_e32(v[2], v[0], v[1])]),
  "VALU_mul": ("v_mul_f32", [v_mul_f32_e32(v[2], v[0], v[1])]),
  "VALU_logic": ("v_and/or/xor", [v_and_b32_e32(v[2], v[0], v[1]), v_or_b32_e32(v[3], v[0], v[1]), v_xor_b32_e32(v[4], v[0], v[1])]),
  "VALU_shift": ("v_lshl/lshr", [v_lshlrev_b32_e32(v[2], 1, v[0]), v_lshrrev_b32_e32(v[3], 1, v[0])]),

  # VALU transcendental - still just register ops
  "VALU_exp": ("v_exp_f32", [v_mov_b32_e32(v[0], 1.0), v_exp_f32_e32(v[1], v[0])]),
  "VALU_log": ("v_log_f32", [v_mov_b32_e32(v[0], 1.0), v_log_f32_e32(v[1], v[0])]),
  "VALU_rcp": ("v_rcp_f32", [v_mov_b32_e32(v[0], 1.0), v_rcp_f32_e32(v[1], v[0])]),
  "VALU_sqrt": ("v_sqrt_f32", [v_mov_b32_e32(v[0], 1.0), v_sqrt_f32_e32(v[1], v[0])]),

  # VALU 64-bit (0xd)
  "VALU64_lshl": ("v_lshlrev_b64", [v_lshlrev_b64(v[0:1], 1, v[2:3])]),

  # VALU MAD64 (0xe) - commented out, needs proper clamp arg
  # "VALU_mad64": ("v_mad_u64_u32", [v_mad_u64_u32(v[0:1], None, v[2], v[3], v[4:5])]),

  # VALU compare - writes to VCC
  "VALU_cmp": ("v_cmp_eq_u32", [v_cmp_eq_u32_e32(v[0], v[1])]),

  # VALU CMPX (0x73) - modifies EXEC
  "VALU_cmpx": ("v_cmpx_eq_u32", [v_cmpx_eq_u32_e32(v[0], v[1])]),
}


def run_with_simd_retry(instructions: list, max_retries: int = 4) -> tuple[list[bytes], list, set]:
  """Run instructions and retry with different SIMD selections until we get INST packets."""
  for simd in range(max_retries):
    SQTT_SIMD_SEL.value = simd
    blobs = run_asm_sqtt(instructions)
    packets = decode_all_blobs(blobs)
    ops = get_inst_ops(packets)
    if ops:
      return blobs, packets, ops
  # Return last attempt even if no ops found
  return blobs, packets, ops

def discover_all_instops() -> tuple[dict[int, set[str]], dict[str, Exception]]:
  """Run all instruction tests and collect InstOp values."""
  discovered: dict[int, set[str]] = {}
  failures: dict[str, Exception] = {}

  for test_name, (instr_name, instructions) in INSTRUCTION_TESTS.items():
    try:
      blobs, packets, ops = run_with_simd_retry(instructions)

      for op in ops:
        if op not in discovered:
          discovered[op] = set()
        discovered[op].add(f"{test_name}")

      if DEBUG >= 2:
        print(f"\n{'─'*60}")
        print(f"{test_name} ({instr_name}): ops={[hex(op) for op in sorted(ops)]} simd_sel={SQTT_SIMD_SEL.value}")
        print_blobs(blobs, filter_timing=True)
      if DEBUG >= 1:
        status = colored("✓", "green") if ops else colored("∅", "yellow")
        ops_str = ", ".join(hex(op) for op in sorted(ops)) if ops else "none"
        print(f"  {status} {test_name:25s} ops=[{ops_str}]")

    except Exception as e:
      failures[test_name] = e
      if DEBUG >= 1:
        print(f"  {colored('✗', 'red')} {test_name:25s} FAILED: {e}")

  return discovered, failures


def print_summary(discovered: dict[int, set[str]], failures: dict[str, Exception]) -> None:
  """Print discovery summary."""
  known_ops = {e.value for e in InstOp}
  discovered_ops = set(discovered.keys())

  print("\n" + "=" * 60)
  print("DISCOVERED INSTOP VALUES")
  print("=" * 60)

  for op in sorted(discovered_ops):
    try:
      name = InstOp(op).name
      status = colored("known", "green")
    except ValueError:
      name = f"UNKNOWN"
      status = colored("NEW!", "yellow")

    sources = ", ".join(sorted(discovered[op]))
    print(f"  0x{op:02x} {name:20s} ({status}) <- {sources}")

  # Missing from enum
  missing = known_ops - discovered_ops
  if missing:
    print("\n" + "=" * 60)
    print("ENUM VALUES NOT DISCOVERED")
    print("=" * 60)
    print("(need memory ops: SMEM, VMEM, LDS)")
    for op in sorted(missing):
      print(f"  0x{op:02x} {InstOp(op).name}")

  # New values to add
  new_ops = discovered_ops - known_ops
  if new_ops:
    print("\n" + "=" * 60)
    print(colored("NEW INSTOP VALUES TO ADD TO ENUM", "yellow"))
    print("=" * 60)
    for op in sorted(new_ops):
      sources = ", ".join(sorted(discovered[op]))
      print(f"  {op:#04x}: \"{sources}\",")

  # Stats
  print("\n" + "=" * 60)
  print("STATISTICS")
  print("=" * 60)
  print(f"  Tests run:      {len(INSTRUCTION_TESTS)}")
  print(f"  Tests passed:   {len(INSTRUCTION_TESTS) - len(failures)}")
  print(f"  Tests failed:   {len(failures)}")
  print(f"  Known ops:      {len(known_ops)}")
  print(f"  Discovered:     {len(discovered_ops)}")
  if known_ops:
    print(f"  Coverage:       {len(discovered_ops & known_ops)}/{len(known_ops)} ({100*len(discovered_ops & known_ops)//len(known_ops)}%)")
  print(f"  New ops found:  {len(new_ops)}")


if __name__ == "__main__":
  print("=" * 60)
  print("SQTT InstOp Discovery Tool")
  print("=" * 60)
  print(f"Testing {len(INSTRUCTION_TESTS)} instruction categories...\n")

  discovered, failures = discover_all_instops()
  print_summary(discovered, failures)
