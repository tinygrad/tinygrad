#!/usr/bin/env python3
"""Tests comparing hardware SQTT traces to emulator SQTT output.

Run with: python -m pytest extra/assembly/amd/test/test_sqtt_compare.py -v
Requires AMD GPU with SQTT support.
"""
import os
os.environ["SQTT"] = "1"
os.environ["PROFILE"] = "1"
os.environ["AMD_LLVM"] = "0"

import unittest, sys, contextlib
from tinygrad import Tensor
from tinygrad.device import Device
from tinygrad.uop.ops import UOp, Ops, KernelInfo
from tinygrad.runtime.ops_amd import ProfileSQTTEvent

from extra.assembly.amd.sqtt import decode, encode, LAYOUT_HEADER, WAVESTART, WAVEEND, INST, PacketType

# ═══════════════════════════════════════════════════════════════════════════════
# HARDWARE SQTT CAPTURE
# ═══════════════════════════════════════════════════════════════════════════════

dev = Device["AMD"]

def custom(arg: str, s: UOp | None = None) -> UOp:
  return UOp(Ops.CUSTOM, src=(s,) if s is not None else (), arg=arg)

def asm_kernel(instrs: list[str], local_size: int = 1, global_size: int = 1) -> Tensor:
  """Create a kernel from inline assembly instructions."""
  name = sys._getframe(1).f_code.co_name
  def fxn(_):
    L = UOp.special(local_size, "lidx0")
    G = UOp.special(global_size, "gidx0")
    op = custom("asm volatile (")
    for inst in instrs:
      op = custom(f'  "{inst}\\n\\t"', op)
    op = custom(");", op)
    return UOp.sink(op, L, G, arg=KernelInfo(name=name))
  return Tensor.custom_kernel(Tensor.empty(1), fxn=fxn)[0]

@contextlib.contextmanager
def capture_hw_sqtt():
  """Capture raw SQTT blobs from hardware execution."""
  dev.profile_events.clear()
  result: dict[str, list[bytes]] = {}
  yield result
  for ev in dev.profile_events:
    if isinstance(ev, ProfileSQTTEvent):
      result.setdefault(ev.kern, []).append(ev.blob)

# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

@unittest.skipIf(not hasattr(dev, 'profile_events'), "AMD device required")
class TestHardwareSQTT(unittest.TestCase):
  """Tests that verify hardware SQTT capture and parsing."""

  def test_capture_and_parse(self):
    """Verify we can capture and parse SQTT for simple VALU instructions."""
    with capture_hw_sqtt() as sqtt:
      asm_kernel(["v_add_f32 v10, v10, v11", "v_add_f32 v11, v11, v12", "v_add_f32 v12, v12, v13"]).realize()
    self.assertGreater(len(sqtt), 0, "No SQTT data captured")
    kern = list(sqtt.keys())[0]
    blob = sqtt[kern][0]
    packets = decode(blob)
    self.assertGreater(len(packets), 0, "No packets parsed")
    pkt_types = {type(p) for p in packets}
    # LAYOUT_HEADER should always be present
    self.assertIn(LAYOUT_HEADER, pkt_types, "Missing LAYOUT_HEADER packet")

  def test_print_raw_packets(self):
    """Debug test to print raw SQTT packets."""
    with capture_hw_sqtt() as sqtt:
      asm_kernel([
        "v_mov_b32 v1, 1.0",
        "v_add_f32 v2, v1, v1",
        "s_nop 0",
        "v_mul_f32 v3, v2, v2",
      ]).realize()
    kern = list(sqtt.keys())[0]
    blob = sqtt[kern][0]
    packets = decode(blob)
    print(f"\n=== Raw SQTT packets for {kern} ({len(blob)} bytes, {len(packets)} packets) ===")
    for i, p in enumerate(packets):
      delta = p.delta if p.delta is not None else 0
      print(f"  [{i:3d}] time={p._time:6d} delta={delta:4d} {type(p).__name__:18s} raw=0x{p._raw:x}")

  def test_valu_timing(self):
    """Check VALU instruction timing from raw packets."""
    with capture_hw_sqtt() as sqtt:
      asm_kernel([
        "v_add_f32 v10, v10, v11",
        "v_add_f32 v11, v11, v12",
        "v_add_f32 v12, v12, v13",
      ]).realize()
    kern = list(sqtt.keys())[0]
    packets = decode(sqtt[kern][0])
    inst_packets = [p for p in packets if isinstance(p, INST)]
    print(f"\n=== INST packets ===")
    for i, p in enumerate(inst_packets):
      print(f"  [{i}] time={p._time} delta={p.delta} raw=0x{p._raw:x}")


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
      self.assertEqual(type(orig), type(dec), f"type mismatch at {i}: {orig} vs {dec}")

  def test_decode_real_blob(self):
    """Test decoding a real SQTT blob from examples."""
    import pickle
    from pathlib import Path
    example_path = Path(__file__).parent.parent.parent.parent / "sqtt/examples/profile_plus_run_0.pkl"
    if not example_path.exists():
      self.skipTest(f"Example file not found: {example_path}")

    with open(example_path, "rb") as f:
      data = pickle.load(f)

    sqtt_events = [e for e in data if isinstance(e, ProfileSQTTEvent)]
    self.assertGreater(len(sqtt_events), 0, "No SQTT events in example")

    packets = decode(sqtt_events[0].blob)
    self.assertGreater(len(packets), 0, "No packets decoded")
    # Should see common packet types
    pkt_types = {type(p) for p in packets}
    self.assertIn(LAYOUT_HEADER, pkt_types)


@unittest.skip("Emulator SQTT not yet implemented")
class TestEmulatorSQTT(unittest.TestCase):
  """Tests comparing emulator SQTT to hardware SQTT."""

  def test_simple_valu_match(self):
    """Simple VALU chain should produce matching SQTT packets."""
    pass


if __name__ == "__main__":
  unittest.main()
