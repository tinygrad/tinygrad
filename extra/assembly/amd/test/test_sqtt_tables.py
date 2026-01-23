"""Tests comparing sqtt.py PACKET_TYPES against AMD's rocprof-trace-decoder binary."""
import unittest
from pathlib import Path

ROCPROF_LIB = "/usr/lib/librocprof-trace-decoder.so"

def extract_bit_tables():
  """Extract bit budget tables from librocprof-trace-decoder.so."""
  lib_path = Path(ROCPROF_LIB)
  if not lib_path.exists():
    return None, None, None

  with open(lib_path, 'rb') as f:
    f.seek(0x2d220)
    layout2 = list(f.read(32))
    f.seek(0x2d280)
    layout3 = list(f.read(32))
    f.seek(0x2d2c0)
    layout4 = list(f.read(32))

  return layout2, layout3, layout4

# Mapping from sqtt.py class name to rocprof type_id (verified via Ghidra analysis)
NAME_TO_TYPE_ID = {
  "VALUINST": 1, "VMEMEXEC": 2, "ALUEXEC": 3, "IMMEDIATE": 4, "IMMEDIATE_MASK": 5,
  "WAVERDY": 6, "TS_DELTA_S8_W3": 7, "WAVEEND": 8, "WAVESTART": 9, "TS_DELTA_S5_W2": 10,
  "WAVEALLOC": 11, "TS_DELTA_S5_W3": 12, "PERF": 13, "UTILCTR": 14, "TS_DELTA_SHORT": 15,
  "NOP": 16, "TS_WAVE_STATE": 17, "EVENT": 18, "EVENT_BIG": 19, "REG": 20,
  "SNAPSHOT": 21, "TS_DELTA_OR_MARK": 22, "LAYOUT_HEADER": 23,
}

@unittest.skipUnless(Path(ROCPROF_LIB).exists(), "rocprof-trace-decoder not installed")
class TestSQTTMatchesBinary(unittest.TestCase):
  def test_bit_counts_match_layout3(self):
    """Verify PACKET_TYPES bit counts match rocprof-trace-decoder layout 3."""
    from extra.assembly.amd.sqtt import PACKET_TYPES

    _, layout3, _ = extract_bit_tables()

    for pkt_cls in PACKET_TYPES:
      name = pkt_cls.__name__
      if name not in NAME_TO_TYPE_ID:
        continue
      type_id = NAME_TO_TYPE_ID[name]
      expected_bits = layout3[type_id]
      actual_bits = pkt_cls._size_nibbles * 4
      with self.subTest(packet=name):
        self.assertEqual(actual_bits, expected_bits,
          f"{name}: {actual_bits} bits != expected {expected_bits}")

if __name__ == "__main__":
  unittest.main()
