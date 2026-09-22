import struct
from typing import NamedTuple
from test.mockgpu.qcom.errors import input_boundary

class Packet(NamedTuple):
  offset: int
  type: int
  target: int
  payload: tuple[int, ...]

@input_boundary
def decode(data: bytes) -> tuple[Packet, ...]:
  # Framing only: targets and payloads are opaque, never executed.
  if len(data) % 4: raise ValueError(f"PM4 dword {len(data)//4}: unaligned byte length {len(data)}")
  words = struct.unpack(f"<{len(data)//4}I", data)
  packets, pos = [], 0
  while pos < len(words):
    header = words[pos]
    kind = header >> 28
    ctx = f"PM4 dword {pos} header={header:#010x}"
    if kind == 4:
      count, target = header & 0x7f, (header >> 8) & 0x3ffff
      count_parity, target_parity = (header >> 7) & 1, (header >> 27) & 1
      reserved = header & 0x04000000
    elif kind == 7:
      count, target = header & 0x3fff, (header >> 16) & 0x7f
      count_parity, target_parity = (header >> 15) & 1, (header >> 23) & 1
      reserved = header & 0x0f004000
    else: raise ValueError(f"{ctx}: unsupported packet type {kind}")
    # Canonical producer fields; stricter than Mesa's permissive parser masks.
    if reserved: raise ValueError(f"{ctx}: unsupported header bits {reserved:#x}")
    if count_parity != (1 ^ (count.bit_count() & 1)): raise ValueError(f"{ctx}: invalid count parity")
    if target_parity != (1 ^ (target.bit_count() & 1)): raise ValueError(f"{ctx}: invalid target parity")
    if kind == 4 and target + count > 0x40000: raise ValueError(f"{ctx}: register span {target:#x}+{count} exceeds 18 bits")
    if pos + 1 + count > len(words): raise ValueError(f"{ctx}: truncated payload: need {count}, have {len(words)-pos-1}")
    packets.append(Packet(pos, kind, target, words[pos+1:pos+1+count]))
    pos += 1 + count
  return tuple(packets)
