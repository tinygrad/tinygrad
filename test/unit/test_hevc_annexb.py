from __future__ import annotations
import io
import pytest
from tinygrad.runtime.hevc_annexb import (
  AnnexBParserError,
  HEVCNalHeader,
  iter_annexb_nalus,
  with_annexb_start_code,
  nal_unit_type_name,
  parse_nal_header,
)


def _header(nal_unit_type: int, layer_id: int = 0, temporal_id_plus1: int = 1) -> bytes:
  if not 0 <= nal_unit_type < 64:
    raise ValueError("invalid nal type")
  if not 0 <= layer_id < 64:
    raise ValueError("invalid layer id")
  if not 1 <= temporal_id_plus1 < 8:
    raise ValueError("invalid temporal id")
  first = (nal_unit_type << 1) | ((layer_id >> 5) & 0x1)
  second = ((layer_id & 0x1F) << 3) | (temporal_id_plus1 & 0x7)
  return bytes([first, second])


def test_iter_annexb_nalus_basic():
  payloads = [b"abc", b"defg", b"xyz"]
  nal_types = [32, 33, 34]
  data = b"".join([
    b"\x00\x00\x00\x01" + _header(nal_types[0]) + payloads[0],
    b"\x00\x00\x01" + _header(nal_types[1], temporal_id_plus1=2) + payloads[1],
    b"\x00\x00\x00\x01" + _header(nal_types[2], layer_id=3) + payloads[2],
  ])
  nalus = list(iter_annexb_nalus(data))
  assert [parse_nal_header(n).nal_unit_type for n in nalus] == nal_types
  assert [n[2:] for n in nalus] == payloads
  assert nal_unit_type_name(nal_types[1]) == "SPS_NUT"


def test_iter_annexb_nalus_chunk_boundaries():
  payload = b"hello world" * 3
  data = b"\x00\x00\x01" + _header(19) + payload + b"\x00\x00\x00\x01" + _header(20) + b"tail"
  stream = io.BytesIO(data)
  nalus = list(iter_annexb_nalus(stream, chunk_size=5))
  assert len(nalus) == 2
  headers = [parse_nal_header(n) for n in nalus]
  assert [h.nal_unit_type for h in headers] == [19, 20]
  assert headers[0].temporal_id == 0


def test_iter_annexb_nalus_strips_trailing_zeroes():
  data = b"\x00\x00\x01" + _header(32) + b"data\x00\x00\x00" + b"\x00\x00\x01" + _header(33) + b"next"
  nalus = list(iter_annexb_nalus(data))
  assert nalus[0][2:] == b"data"
  assert nalus[1][2:] == b"next"


def test_iter_annexb_nalus_requires_start_code():
  with pytest.raises(AnnexBParserError):
    list(iter_annexb_nalus(b"no start code"))


def test_parse_nal_header_validates_bits():
  good = parse_nal_header(b"\x40\x01" + b"payload")
  assert isinstance(good, HEVCNalHeader)
  with pytest.raises(AnnexBParserError):
    parse_nal_header(b"\x80\x01")
  with pytest.raises(AnnexBParserError):
    parse_nal_header(b"\x00\x00")
  with pytest.raises(AnnexBParserError):
    parse_nal_header(b"\x40")


def test_with_annexb_start_code():
  nalu = b"\x40\x01payload"
  assert with_annexb_start_code(nalu, long_start_code=False).startswith(b"\x00\x00\x01")
  assert with_annexb_start_code(nalu, long_start_code=True).startswith(b"\x00\x00\x00\x01")
  with pytest.raises(ValueError):
    with_annexb_start_code(b"")
