#!/usr/bin/env python3
from __future__ import annotations
import enum, collections, dataclasses
from typing import Iterator
from tinygrad.helpers import colored

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

# Mapping from 5-bit stall_key to CUPTI StallReason
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

LOOKUP_8b = [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]

@dataclasses.dataclass
class PacketHeader:
  num_bytes: int   # number of sample bytes in this packet (bits 4:0 of byte 4)
  tpc_id: int      # TPC identifier (byte 5 | bits 3:1 of byte 7 << 8)
  dropped: bool    # dropped flag (bit 4 of byte 7)
  sync: bool       # sync flag (bit 6 of byte 7)

  @classmethod
  def from_packet(cls, pkt: bytes) -> PacketHeader:
    return cls(num_bytes=pkt[4] & 0x1f, tpc_id=pkt[5] | (((pkt[7] >> 1) & 0x07) << 8),
               dropped=bool((pkt[7] >> 4) & 1), sync=bool((pkt[7] >> 6) & 1))

@dataclasses.dataclass
class PCSample:
  pc_offset: int            # absolute PC offset (pc_raw << 4)
  stall_key: int            # raw 5-bit stall key from hardware
  stall_reason: StallReason # mapped CUPTI stall reason
  wave_id: int              # warp/wave identifier (0-63)
  active: bool              # active flag (warp was executing)
  tpc_id: int               # TPC that generated this sample

  @classmethod
  def from_record(cls, rec: bytes, tpc_id: int) -> PCSample:
    pc_lo = rec[0] | (rec[1] << 8) | (rec[2] << 16) | (rec[3] << 24) | (rec[4] << 32)
    pc_offset = (pc_lo | ((rec[5] & 0x1f) << 40)) << 4
    stall_key = ((rec[6] & 0x03) << 3) | (rec[5] >> 5)
    return cls(pc_offset=pc_offset, stall_key=stall_key, stall_reason=STALL_KEY_MAP.get(stall_key, StallReason.OTHER),
               wave_id=rec[6] >> 2, active=bool((rec[7] >> 6) & 1), tpc_id=tpc_id)

def decode(data: bytes) -> Iterator[PCSample]:
  tpc_state: dict[int, list[int]] = {}
  for pkt_idx in range(len(data) // 32):
    pkt = data[pkt_idx * 32:(pkt_idx + 1) * 32]
    hdr = PacketHeader.from_packet(pkt)
    if hdr.tpc_id not in tpc_state: tpc_state[hdr.tpc_id] = []
    acc = tpc_state[hdr.tpc_id]
    if hdr.dropped: acc.clear()
    for i in range(hdr.num_bytes): acc.append(pkt[LOOKUP_8b[i]])
    while len(acc) >= 8:
      yield PCSample.from_record(bytes(acc[:8]), hdr.tpc_id)
      del acc[:8]

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

def print_samples(samples: list[PCSample]) -> None:
  if not samples: return
  base_pc = min(s.pc_offset for s in samples)
  for s in samples:
    gpc, tpc, sm = decode_tpc_id(s.tpc_id)
    stall_str = colored(f"{s.stall_reason.name:17}", STALL_COLORS.get(s.stall_reason, "white"))
    issued_str = "" if s.active else colored(" not_issued", "white")
    print(f"pc=0x{s.pc_offset - base_pc:06x} {stall_str} ev={s.stall_key:2d} active={int(s.active)} wave={s.wave_id:2d} gpc={gpc} tpc={tpc} sm={sm}")

def print_packets(data: bytes) -> None:
  for i in range(len(data) // 32):
    pkt = data[i * 32:(i + 1) * 32]
    hdr = PacketHeader.from_packet(pkt)
    print(f"Pkt {i:3d}: tpc={hdr.tpc_id} bytes={hdr.num_bytes} drop={int(hdr.dropped)} sync={int(hdr.sync)} | {pkt.hex()}")

def print_aggregated(samples: list[PCSample]) -> None:
  if not samples: return
  base_pc = min(s.pc_offset for s in samples)
  counter: collections.Counter[tuple[int, int]] = collections.Counter((s.pc_offset, s.stall_key) for s in samples)
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
