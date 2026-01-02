#!/usr/bin/env python3
"""SQTT InstOp discovery tool - finds instruction opcodes by running different instructions.

Run with: DEBUG=1 python extra/assembly/amd/test/discover_instops.py
For full traces: DEBUG=2 python extra/assembly/amd/test/discover_instops.py
"""
import os
os.environ["SQTT"] = "1"
os.environ["PROFILE"] = "1"
os.environ["SQTT_LIMIT_SE"] = "2"  # Force work to traced SE only
os.environ["SQTT_TOKEN_EXCLUDE"] = "3784"  # Exclude WAVERDY, REG, EVENT, UTILCTR, WAVEALLOC, PERF

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
  # VALU - MAD64
  v_mad_u64_u32,
  # VALU - compare (writes to VCC, safe)
  v_cmp_eq_u32_e32,
  v_cmpx_eq_u32_e32,
  # SALU - basic (safe, just register ops)
  s_mov_b32, s_add_u32, s_and_b32, s_or_b32,
  s_lshl_b32, s_lshr_b32,
  s_nop, s_endpgm, s_waitcnt,
  # SALU - branch (safe if offset is 0 = next instruction)
  s_branch, s_cbranch_scc0, s_cbranch_execz,
  # SALU - message
  s_sendmsg,
  # SMEM - scalar memory (load from kernarg pointer in s[0:1])
  s_load_b32, s_load_b64,
  # VMEM - vector memory (global load/store) - various widths
  global_load_b32, global_load_b64, global_load_b96, global_load_b128,
  global_store_b32, global_store_b64, global_store_b96, global_store_b128,
  # LDS - local data share - various widths
  ds_load_b32, ds_load_b64, ds_load_b128,
  ds_store_b32, ds_store_b64, ds_store_b128,
  # SrcEnum for NULL soffset
  SrcEnum,
)
from extra.assembly.amd.dsl import v, s
from extra.assembly.amd.sqtt import InstOp, INST

from extra.assembly.amd.test.test_sqtt_hw import (
  run_asm_sqtt, decode_all_blobs, get_inst_ops, print_blobs, get_wave_packets, format_packet
)
from extra.assembly.amd.sqtt import WAVESTART

# ═══════════════════════════════════════════════════════════════════════════════
# INSTRUCTION TEST CASES - only safe instructions that don't access memory
# ═══════════════════════════════════════════════════════════════════════════════

# Helper: load buffer address from kernarg (s[0:1] -> s[2:3])
# The runtime passes kernarg pointer in s[0:1], kernarg contains buffer address
def _load_buf_addr():
  return [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),  # load buf addr from kernarg
    s_waitcnt(lgkmcnt=0),  # wait for SMEM load
  ]

INSTRUCTION_TESTS: dict[str, tuple[str, list]] = {
  # SALU (0x0) - scalar ALU, just register operations
  "SALU_mov": ("s_mov_b32", [s_mov_b32(s[4], 0), s_mov_b32(s[5], 1)]),
  "SALU_add": ("s_add_u32", [s_mov_b32(s[4], 1), s_mov_b32(s[5], 2), s_add_u32(s[6], s[4], s[5])]),
  "SALU_logic": ("s_and/or", [s_and_b32(s[6], s[4], s[5]), s_or_b32(s[7], s[4], s[5])]),
  "SALU_shift": ("s_lshl/lshr", [s_lshl_b32(s[6], s[4], 1), s_lshr_b32(s[7], s[4], 1)]),
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

  # VALU MAD64 (0xe)
  "VALU_mad64": ("v_mad_u64_u32", [
    v_mov_b32_e32(v[2], 2),
    v_mov_b32_e32(v[3], 3),
    v_mov_b32_e32(v[4], 0),
    v_mov_b32_e32(v[5], 0),
    v_mad_u64_u32(v[0:1], SrcEnum.NULL, v[2], v[3], v[4:5]),  # 2*3+0 = 6
  ]),

  # VALU compare - writes to VCC
  "VALU_cmp": ("v_cmp_eq_u32", [v_cmp_eq_u32_e32(v[0], v[1])]),

  # VALU CMPX (0x73) - modifies EXEC
  "VALU_cmpx": ("v_cmpx_eq_u32", [v_cmpx_eq_u32_e32(v[0], v[1])]),

  # ═══════════════════════════════════════════════════════════════════════════════
  # MEMORY INSTRUCTIONS - access real buffer passed via kernarg
  # ═══════════════════════════════════════════════════════════════════════════════

  # SMEM (0x1) - scalar memory load from buffer
  "SMEM_load": ("s_load_b32", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),  # load buf addr from kernarg
    s_waitcnt(lgkmcnt=0),
    s_load_b32(s[4], s[2], 0, soffset=SrcEnum.NULL),  # load from buffer
    s_waitcnt(lgkmcnt=0),
  ]),

  # VMEM load (0x21 VMEM_LOAD) - global load
  "VMEM_load": ("global_load_b32", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),  # load buf addr from kernarg
    s_waitcnt(lgkmcnt=0),
    v_mov_b32_e32(v[0], 0),  # offset = 0
    global_load_b32(v[1], addr=v[0], saddr=s[2], offset=0),  # load from buffer
    s_waitcnt(vmcnt=0),
  ]),

  # VMEM store (0x24 VMEM_STORE) - global store
  "VMEM_store": ("global_store_b32", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),  # load buf addr from kernarg
    s_waitcnt(lgkmcnt=0),
    v_mov_b32_e32(v[0], 0),  # offset = 0
    v_mov_b32_e32(v[1], 42),  # data to store
    global_store_b32(addr=v[0], data=v[1], saddr=s[2], offset=0),  # store to buffer
    s_waitcnt(vmcnt=0),
  ]),

  # LDS load (0x29 LDS_LOAD) - local data share read
  "LDS_load": ("ds_load_b32", [
    v_mov_b32_e32(v[0], 0),  # LDS address = 0
    ds_load_b32(v[1], v[0], offset=0),  # read from LDS
    s_waitcnt(lgkmcnt=0),
  ]),

  # LDS store (0x2b LDS_STORE) - local data share write
  "LDS_store": ("ds_store_b32", [
    v_mov_b32_e32(v[0], 0),  # LDS address = 0
    v_mov_b32_e32(v[1], 42),  # data to store
    ds_store_b32(v[0], v[1], offset=0),  # write to LDS
    s_waitcnt(lgkmcnt=0),
  ]),

  # ═══════════════════════════════════════════════════════════════════════════════
  # WIDER MEMORY OPERATIONS - to discover more InstOp variants
  # ═══════════════════════════════════════════════════════════════════════════════

  # VMEM 64-bit load
  "VMEM_load64": ("global_load_b64", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),
    s_waitcnt(lgkmcnt=0),
    v_mov_b32_e32(v[0], 0),
    global_load_b64(v[2:3], addr=v[0], saddr=s[2], offset=0),
    s_waitcnt(vmcnt=0),
  ]),

  # VMEM 96-bit load
  "VMEM_load96": ("global_load_b96", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),
    s_waitcnt(lgkmcnt=0),
    v_mov_b32_e32(v[0], 0),
    global_load_b96(v[4:6], addr=v[0], saddr=s[2], offset=0),
    s_waitcnt(vmcnt=0),
  ]),

  # VMEM 128-bit load
  "VMEM_load128": ("global_load_b128", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),
    s_waitcnt(lgkmcnt=0),
    v_mov_b32_e32(v[0], 0),
    global_load_b128(v[4:7], addr=v[0], saddr=s[2], offset=0),
    s_waitcnt(vmcnt=0),
  ]),

  # VMEM 64-bit store
  "VMEM_store64": ("global_store_b64", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),
    s_waitcnt(lgkmcnt=0),
    v_mov_b32_e32(v[0], 0),
    v_mov_b32_e32(v[2], 42),
    v_mov_b32_e32(v[3], 43),
    global_store_b64(addr=v[0], data=v[2:3], saddr=s[2], offset=0),
    s_waitcnt(vmcnt=0),
  ]),

  # VMEM 96-bit store
  "VMEM_store96": ("global_store_b96", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),
    s_waitcnt(lgkmcnt=0),
    v_mov_b32_e32(v[0], 0),
    v_mov_b32_e32(v[4], 42),
    v_mov_b32_e32(v[5], 43),
    v_mov_b32_e32(v[6], 44),
    global_store_b96(addr=v[0], data=v[4:6], saddr=s[2], offset=0),
    s_waitcnt(vmcnt=0),
  ]),

  # VMEM 128-bit store
  "VMEM_store128": ("global_store_b128", [
    s_load_b64(s[2:3], s[0], 0, soffset=SrcEnum.NULL),
    s_waitcnt(lgkmcnt=0),
    v_mov_b32_e32(v[0], 0),
    v_mov_b32_e32(v[4], 42),
    v_mov_b32_e32(v[5], 43),
    v_mov_b32_e32(v[6], 44),
    v_mov_b32_e32(v[7], 45),
    global_store_b128(addr=v[0], data=v[4:7], saddr=s[2], offset=0),
    s_waitcnt(vmcnt=0),
  ]),

  # LDS 64-bit load
  "LDS_load64": ("ds_load_b64", [
    v_mov_b32_e32(v[0], 0),
    ds_load_b64(v[2:3], v[0], offset=0),
    s_waitcnt(lgkmcnt=0),
  ]),

  # LDS 128-bit load
  "LDS_load128": ("ds_load_b128", [
    v_mov_b32_e32(v[0], 0),
    ds_load_b128(v[4:7], v[0], offset=0),
    s_waitcnt(lgkmcnt=0),
  ]),

  # LDS 64-bit store
  "LDS_store64": ("ds_store_b64", [
    v_mov_b32_e32(v[0], 0),
    v_mov_b32_e32(v[2], 42),
    v_mov_b32_e32(v[3], 43),
    ds_store_b64(v[0], v[2:3], offset=0),
    s_waitcnt(lgkmcnt=0),
  ]),

  # LDS 128-bit store
  "LDS_store128": ("ds_store_b128", [
    v_mov_b32_e32(v[0], 0),
    v_mov_b32_e32(v[4], 42),
    v_mov_b32_e32(v[5], 43),
    v_mov_b32_e32(v[6], 44),
    v_mov_b32_e32(v[7], 45),
    ds_store_b128(v[0], v[4:7], offset=0),
    s_waitcnt(lgkmcnt=0),
  ]),

  # MESSAGE (0x9) - s_sendmsg
  "MESSAGE": ("s_sendmsg", [
    s_sendmsg(0),  # send message 0 (NOP message)
  ]),
}


def run_with_retry(instructions: list, max_attempts: int = 10) -> tuple[list[tuple[int, list[bytes]]], list[list], set]:
  """Run instructions multiple times with both SIMD selections to collect all InstOp variants.

  Memory ops produce different InstOp values (0x2x vs 0x5x) depending on which SIMD executes them:
  - 0x2x range: wave ran on traced SIMD (matched)
  - 0x5x range: wave ran on other SIMD (not matched)

  We trace SIMD 0 and SIMD 2, and collect ALL runs to capture both matched and unmatched cases.
  Returns list of (traced_simd, blobs) tuples.
  """
  all_ops = set()
  all_runs: list[tuple[int, list[bytes]]] = []
  all_packets = []
  for simd_sel in [0, 2]:  # trace SIMD 0 and SIMD 2
    SQTT_SIMD_SEL.value = simd_sel
    for _ in range(max_attempts):
      blobs = run_asm_sqtt(instructions)
      packets = decode_all_blobs(blobs)
      ops = get_inst_ops(packets)
      all_runs.append((simd_sel, blobs))
      all_packets.append(packets)
      all_ops.update(ops)
  return all_runs, all_packets, all_ops

def discover_all_instops() -> tuple[dict[int, set[str]], dict[str, Exception]]:
  """Run all instruction tests and collect InstOp values."""
  discovered: dict[int, set[str]] = {}
  failures: dict[str, Exception] = {}

  for test_name, (instr_name, instructions) in INSTRUCTION_TESTS.items():
    try:
      all_runs, _, ops = run_with_retry(instructions)

      for op in ops:
        if op not in discovered:
          discovered[op] = set()
        discovered[op].add(f"{test_name}")

      if DEBUG >= 2:
        print(f"\n{'─'*60}")
        print(f"{test_name} ({instr_name}): ops={[hex(op) for op in sorted(ops)]}")

        # collect wave patterns from traced SIMD runs (group by packet type sequence, ignore timing)
        patterns: dict[tuple, list] = {}  # pattern (types only) -> list of (wave_packets, t0)
        for traced_simd, blobs in all_runs:
          for blob in blobs:
            packets = decode_all_blobs([blob])
            wave_packets = get_wave_packets(packets)
            # only include runs where wave ran on traced SIMD
            ws = next((p for p in wave_packets if isinstance(p, WAVESTART)), None)
            if ws and ws.simd == traced_simd and wave_packets:
              t0 = wave_packets[0]._time
              pattern = tuple(type(p).__name__ for p in wave_packets)  # types only, no timing
              if pattern not in patterns:
                patterns[pattern] = []
              patterns[pattern].append((wave_packets, t0))

        if patterns:
          counts = {p: len(runs) for p, runs in patterns.items()}
          most_common = max(counts, key=counts.get)
          count = counts[most_common]
          total = sum(counts.values())
          print(f"\n=== most common pattern ({count}/{total} runs) ===")
          # print using actual packets from one of the matching runs
          wave_packets, t0 = patterns[most_common][0]
          last_time = t0
          for p in wave_packets:
            print(format_packet(p, last_time, t0))
            last_time = p._time
          if len(patterns) > 1:
            print(f"\n  variations: {len(patterns)} unique patterns")

      if DEBUG >= 3:
        for traced_simd, blobs in all_runs:
          print(f"\n=== traced simd={traced_simd} ===")
          print_blobs(blobs)
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
