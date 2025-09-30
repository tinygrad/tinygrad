from __future__ import annotations
import io
from dataclasses import dataclass
from typing import BinaryIO, Iterator, Sequence

_START_CODE_SHORT = b"\x00\x00\x01"
_START_CODE_LONG = b"\x00\x00\x00\x01"

@dataclass(frozen=True)
class HEVCNalHeader:
  forbidden_zero_bit: int
  nal_unit_type: int
  nuh_layer_id: int
  nuh_temporal_id_plus1: int

  @property
  def temporal_id(self) -> int:
    return self.nuh_temporal_id_plus1 - 1

class AnnexBParserError(RuntimeError):
  pass

def _locate_start_code(buf: Sequence[int], start: int) -> tuple[int, int]:
  idx = bytes(buf).find(_START_CODE_SHORT, start)
  while idx != -1:
    prefix_len = 3
    if idx > 0 and buf[idx - 1] == 0:
      idx -= 1
      prefix_len = 4
    return idx, prefix_len
  return -1, 0

def _strip_trailing_zeroes(data: bytes) -> bytes:
  return data.rstrip(b"\x00")

def _ensure_stream(source: BinaryIO | bytes | bytearray | memoryview) -> BinaryIO:
  if isinstance(source, (bytes, bytearray, memoryview)):
    return io.BytesIO(bytes(source))
  return source

def iter_annexb_nalus(source: BinaryIO | bytes | bytearray | memoryview, *, chunk_size: int = 1 << 16) -> Iterator[bytes]:
  if chunk_size <= 0:
    raise ValueError("chunk_size must be positive")
  stream = _ensure_stream(source)
  buffer = bytearray()
  prev_start = -1
  prev_prefix = 0
  saw_start = False
  while True:
    chunk = stream.read(chunk_size)
    if chunk:
      buffer.extend(chunk)
    eof = not chunk
    search_start = prev_start + prev_prefix if prev_start >= 0 else 0
    while True:
      next_start, prefix_len = _locate_start_code(buffer, search_start)
      if next_start == -1:
        break
      if prev_start == -1:
        prev_start, prev_prefix = next_start, prefix_len
        saw_start = True
      else:
        nalu = bytes(buffer[prev_start + prev_prefix:next_start])
        if nalu:
          yield _strip_trailing_zeroes(nalu)
        prev_start, prev_prefix = next_start, prefix_len
      search_start = next_start + prefix_len
    if eof:
      if prev_start == -1:
        if not saw_start:
          raise AnnexBParserError("No Annex B start code found in stream")
        break
      nalu = bytes(buffer[prev_start + prev_prefix:])
      if nalu:
        yield _strip_trailing_zeroes(nalu)
      break
    if prev_start >= 0:
      buffer = buffer[prev_start:]
      prev_start = 0
    else:
      buffer.clear()
  if not saw_start:
    raise AnnexBParserError("No Annex B start code found in stream")


def with_annexb_start_code(nalu: bytes, *, long_start_code: bool = False) -> bytes:
  if not nalu:
    raise ValueError("NAL unit must be non-empty")
  prefix = _START_CODE_LONG if long_start_code else _START_CODE_SHORT
  return prefix + nalu

_HEVC_TYPE_NAMES = {
  0: "TRAIL_N",
  1: "TRAIL_R",
  19: "IDR_W_RADL",
  20: "IDR_N_LP",
  32: "VPS_NUT",
  33: "SPS_NUT",
  34: "PPS_NUT",
}

def nal_unit_type_name(nal_type: int) -> str:
  return _HEVC_TYPE_NAMES.get(nal_type, f"TYPE_{nal_type}")

def parse_nal_header(nalu: bytes) -> HEVCNalHeader:
  if len(nalu) < 2:
    raise AnnexBParserError("NAL unit too short")
  first = nalu[0]
  second = nalu[1]
  forbidden_zero_bit = (first >> 7) & 0x1
  nal_unit_type = (first >> 1) & 0x3F
  nuh_layer_id = ((first & 0x1) << 5) | (second >> 3)
  nuh_temporal_id_plus1 = second & 0x7
  if forbidden_zero_bit != 0:
    raise AnnexBParserError("Forbidden zero bit set in NAL header")
  if nuh_temporal_id_plus1 == 0:
    raise AnnexBParserError("Invalid temporal_id_plus1 (must be >= 1)")
  return HEVCNalHeader(forbidden_zero_bit, nal_unit_type, nuh_layer_id, nuh_temporal_id_plus1)

__all__ = [
  "AnnexBParserError",
  "HEVCNalHeader",
  "iter_annexb_nalus",
  "with_annexb_start_code",
  "parse_nal_header",
  "nal_unit_type_name",
]
