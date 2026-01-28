#!/usr/bin/env python3
from __future__ import annotations
import enum, collections
from typing import Iterator
from tinygrad.helpers import colored
from extra.assembly.amd.sqtt import PacketType, bits

# ═══════════════════════════════════════════════════════════════════════════════
# STALL REASONS
# ═══════════════════════════════════════════════════════════════════════════════

class StallReason(enum.IntEnum):
  # Based on CUpti_ActivityPCSamplingStallReason
  INVALID = 0
  NONE = 1              # selected, selected_not_issued
  INST_FETCH = 2        # branch_resolving, no_instructions
  EXEC_DEPENDENCY = 3   # short_scoreboard, wait
  MEMORY_DEPENDENCY = 4 # long_scoreboard
  TEXTURE = 5           # tex_throttle
  SYNC = 6              # barrier, membar
  CONSTANT_MEMORY = 7   # imc_miss
  PIPE_BUSY = 8         # mio_throttle, math_pipe_throttle
  MEMORY_THROTTLE = 9   # drain, lg_throttle
  NOT_SELECTED = 10     # not_selected
  OTHER = 11            # misc, dispatch_stall
  SLEEPING = 12         # sleeping

STALL_KEY_MAP: dict[int, StallReason] = {
  1: StallReason.MEMORY_THROTTLE, 15: StallReason.MEMORY_THROTTLE,
  2: StallReason.CONSTANT_MEMORY,
  3: StallReason.SYNC,
  6: StallReason.INST_FETCH, 11: StallReason.INST_FETCH,
  7: StallReason.EXEC_DEPENDENCY, 10: StallReason.EXEC_DEPENDENCY,
  9: StallReason.MEMORY_DEPENDENCY,
  12: StallReason.PIPE_BUSY,
  17: StallReason.OTHER, 20: StallReason.OTHER,
  18: StallReason.NONE,
}

# ═══════════════════════════════════════════════════════════════════════════════
# AMPERE PACKET DEFINITIONS (8-byte aligned)
# ═══════════════════════════════════════════════════════════════════════════════

# Lookup table for extracting sample bytes from 32-byte packet
LOOKUP_8B = [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

class PMAHeaderAmpere8B(PacketType):
  num_bytes  = bits[4:0]    # number of sample bytes in this packet
  tpc_id_lo  = bits[15:8]   # TPC identifier low 8 bits
  tpc_id_hi  = bits[27:25]  # TPC identifier high 3 bits
  dropped    = bits[28:28]  # dropped flag (resets byte accumulator)
  @property
  def tpc_id(self) -> int: return self.tpc_id_lo | (self.tpc_id_hi << 8)

class PMASampleAmpere8B(PacketType):
  pc_raw     = bits[44:0]   # raw PC value (actual PC = pc_raw << 4)
  stall_lo   = bits[47:45]  # stall key low 3 bits
  stall_hi   = bits[49:48]  # stall key high 2 bits
  wave_id    = bits[55:50]  # warp/wave identifier (0-63)
  active     = bits[62:62]  # active flag (warp was executing)
  @property
  def pc_offset(self) -> int: return self.pc_raw << 4
  @property
  def stall_key(self) -> int: return self.stall_lo | (self.stall_hi << 3)
  @property
  def stall_reason(self) -> StallReason: return STALL_KEY_MAP.get(self.stall_key, StallReason.OTHER)

def decode(data: bytes) -> Iterator[tuple[PMASampleAmpere8B, int]]:
  tpc_state: dict[int, list[int]] = collections.defaultdict(list)
  for pkt_idx in range(len(data) // 32):
    pkt = data[pkt_idx * 32:(pkt_idx + 1) * 32]
    hdr = PMAHeaderAmpere8B.from_raw(int.from_bytes(pkt[4:8], 'little'))

    if hdr.dropped: tpc_state[hdr.tpc_id].clear()

    for i in range(hdr.num_bytes):
      tpc_state[hdr.tpc_id].append(pkt[LOOKUP_8B[i]])

    while len(tpc_state[hdr.tpc_id]) >= 8:
      yield PMASampleAmpere8B.from_raw(int.from_bytes(bytes(tpc_state[hdr.tpc_id][:8]), 'little')), hdr.tpc_id
      del tpc_state[hdr.tpc_id][:8]

# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

STALL_COLORS = {
  StallReason.NONE: "green", StallReason.INST_FETCH: "yellow", StallReason.EXEC_DEPENDENCY: "cyan",
  StallReason.MEMORY_DEPENDENCY: "red", StallReason.SYNC: "magenta", StallReason.CONSTANT_MEMORY: "blue",
  StallReason.PIPE_BUSY: "yellow", StallReason.MEMORY_THROTTLE: "RED", StallReason.OTHER: "white",
}

def decode_tpc_id(tpc_id: int) -> tuple[int, int, int]:
  # NOTE: valid only for ops_nv, cuda encoding is different
  return (tpc_id >> 5, (tpc_id >> 1) & 0xf, tpc_id & 1)

def print_samples(samples: list[tuple[PMASampleAmpere8B, int]]) -> None:
  if not samples: return
  base_pc = min(s.pc_offset for s, _ in samples)
  for s, tpc_id in samples:
    gpc, tpc, sm = decode_tpc_id(tpc_id)
    stall_str = colored(f"{s.stall_reason.name:17}", STALL_COLORS.get(s.stall_reason, "white"))
    print(f"pc=0x{s.pc_offset - base_pc:06x} {stall_str} ev={s.stall_key:2d} active={s.active} wave={s.wave_id:2d} gpc={gpc} tpc={tpc} sm={sm}")

def print_packets(data: bytes) -> None:
  for i in range(len(data) // 32):
    pkt = data[i * 32:(i + 1) * 32]
    hdr = PMAHeaderAmpere8B.from_raw(int.from_bytes(pkt[4:8], 'little'))
    print(f"Pkt {i:3d}: tpc={hdr.tpc_id} bytes={hdr.num_bytes} drop={hdr.dropped} | {pkt.hex()}")

def print_aggregated(samples: list[tuple[PMASampleAmpere8B, int]]) -> None:
  if not samples: return
  base_pc = min(s.pc_offset for s, _ in samples)
  counter: collections.Counter[tuple[int, int]] = collections.Counter((s.pc_offset, s.stall_key) for s, _ in samples)
  print(f"\nAggregated samples (base_pc=0x{base_pc:x}):")
  for (pc, key), cnt in sorted(counter.items()):
    reason = STALL_KEY_MAP.get(key, StallReason.OTHER)
    stall_str = colored(f"{reason.name:17}", STALL_COLORS.get(reason, "white"))
    print(f"  pc=0x{pc - base_pc:06x} {stall_str} ev={key:2d} samples={cnt:4d}")

if __name__ == "__main__":
  import sys, pickle

  if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

  with open(sys.argv[1], "rb") as f:
    data = pickle.load(f)

  if isinstance(data, dict): dumps = list(enumerate(data["pma_raw_dumps"]))
  else: dumps = [(i, e.blob) for i, e in enumerate(e for e in data if type(e).__name__ == "ProfilePMAEvent")]

  for dump_idx, raw in dumps:
    print(f"\n{'='*60}\nDump {dump_idx} ({len(raw)} bytes, {len(raw)//32} packets)\n{'='*60}")
    if "--raw" in sys.argv: print_packets(raw)
    else:
      samples = list(decode(raw))
      print(f"\nDecoded {len(samples)} samples:")
      print_samples(samples)
      print_aggregated(samples)
