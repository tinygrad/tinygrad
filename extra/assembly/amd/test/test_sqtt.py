#!/usr/bin/env python3
"""Tests for SQTT packet codec (no hardware required)."""
import unittest
from extra.assembly.amd.sqtt import (
  LAYOUT_HEADER, WAVESTART, WAVEEND, INST, NOP,
  decode, encode, PACKET_TYPES, OPCODE_TO_CLASS
)


class TestSQTTCodec(unittest.TestCase):
  """Tests for SQTT encoder/decoder roundtrip."""

  def test_roundtrip_simple(self):
    """Test encode/decode roundtrip for simple packets."""
    test_packets = [
      LAYOUT_HEADER.from_raw(0x100),
      WAVESTART.from_raw(0x0),
      INST.from_raw(0x10),  # delta=1
      INST.from_raw(0x10),  # delta=1
      WAVEEND.from_raw(0x40),  # delta=2
    ]
    encoded = encode(test_packets)
    decoded = decode(encoded)

    self.assertGreaterEqual(len(decoded), len(test_packets))
    for i, (orig, dec) in enumerate(zip(test_packets, decoded)):
      self.assertEqual(type(orig), type(dec), f"type mismatch at {i}")

  def test_decode_empty(self):
    """Test decoding empty data."""
    packets = decode(b'')
    self.assertEqual(packets, [])

  def test_encode_empty(self):
    """Test encoding empty list."""
    data = encode([])
    self.assertEqual(data, b'')

  def test_all_packet_types_have_encoding(self):
    """All packet types should have an encoding defined."""
    for pkt_cls in PACKET_TYPES:
      self.assertIsNotNone(pkt_cls._encoding, f"{pkt_cls.__name__} missing encoding")

  def test_packet_from_raw(self):
    """Test creating packets from raw values."""
    # INST with wave=5, op=0x21, delta=2
    raw = (0x21 << 13) | (5 << 8) | (2 << 4) | 0b010
    pkt = INST.from_raw(raw)
    self.assertEqual(pkt.wave, 5)
    self.assertEqual(pkt.op, 0x21)
    self.assertEqual(pkt.delta, 2)


class TestDecodeRealBlob(unittest.TestCase):
  """Test decoding real SQTT blobs from examples."""

  def test_decode_example_file(self):
    """Test decoding a real SQTT blob from examples."""
    import pickle
    from pathlib import Path
    example_path = Path(__file__).parent.parent.parent.parent / "sqtt/examples/profile_plus_run_0.pkl"
    if not example_path.exists():
      self.skipTest(f"Example file not found: {example_path}")

    from tinygrad.runtime.ops_amd import ProfileSQTTEvent
    with open(example_path, "rb") as f:
      data = pickle.load(f)

    sqtt_events = [e for e in data if isinstance(e, ProfileSQTTEvent)]
    self.assertGreater(len(sqtt_events), 0, "No SQTT events in example")

    packets = decode(sqtt_events[0].blob)
    self.assertGreater(len(packets), 0, "No packets decoded")
    # First packet should be LAYOUT_HEADER
    self.assertIsInstance(packets[0], LAYOUT_HEADER)


if __name__ == "__main__":
  unittest.main()
