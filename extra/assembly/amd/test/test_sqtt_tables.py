"""Tests comparing sqtt.py PACKET_TYPES_L3/L4 against AMD's rocprof-trace-decoder binary."""
import unittest
from pathlib import Path
import ctypes

ROCPROF_LIB = "/usr/lib/librocprof-trace-decoder.so"

def _find_rw_segment():
  """Find the rw- segment of the loaded library."""
  with open('/proc/self/maps', 'r') as f:
    for line in f:
      if 'librocprof-trace-decoder.so' in line and ' rw-p ' in line:
        parts = line.split()
        rw_base = int(parts[0].split('-')[0], 16)
        rw_file_offset = int(parts[2], 16)
        return rw_base, rw_file_offset
  return None, None

def extract_bit_tables():
  """Extract bit budget tables by loading librocprof-trace-decoder.so at runtime.

  Returns (layout0, layout2, layout3, layout4) - Layout 0 (CDNA) shares Layout 2's bit counts.
  """
  if not Path(ROCPROF_LIB).exists():
    return None, None, None, None

  ctypes.CDLL(ROCPROF_LIB)
  rw_base, rw_file_offset = _find_rw_segment()
  if rw_base is None:
    return None, None, None, None

  # Bit tables at file offsets 0x2d220, 0x2d280, 0x2d2c0
  # Layout 0 (CDNA) uses the same bit counts as Layout 2
  layout2 = list((ctypes.c_uint8 * 32).from_address(rw_base + (0x2d220 - rw_file_offset)))
  layout3 = list((ctypes.c_uint8 * 32).from_address(rw_base + (0x2d280 - rw_file_offset)))
  layout4 = list((ctypes.c_uint8 * 32).from_address(rw_base + (0x2d2c0 - rw_file_offset)))
  return layout2, layout2, layout3, layout4  # L0 shares L2's bit counts

def _find_ro_segment():
  """Find the r--p segment containing .rodata of the loaded library."""
  with open('/proc/self/maps', 'r') as f:
    for line in f:
      if 'librocprof-trace-decoder.so' in line and ' r--p ' in line:
        parts = line.split()
        base = int(parts[0].split('-')[0], 16)
        file_offset = int(parts[2], 16)
        # The delta table is at file offset 0x26dc0, which is in .rodata at 0x26000
        if file_offset <= 0x26dc0:
          return base, file_offset
  return None, None

def extract_delta_fields():
  """Extract delta bitfield tables from .rodata section.

  Returns (layout0_table, layout2_table, layout3_table, layout4_table) where each is dict mapping type_id -> (delta_lo, delta_hi).
  The delta field is at bits[delta_hi-1:delta_lo], extracted as: (reg >> delta_lo) & ((1 << (delta_hi - delta_lo)) - 1)
  """
  if not Path(ROCPROF_LIB).exists():
    return None, None, None, None

  ctypes.CDLL(ROCPROF_LIB)
  ro_base, ro_file_offset = _find_ro_segment()
  if ro_base is None:
    return None, None, None, None

  import struct

  def read_table(file_offset, num_entries=25):
    table_addr = ro_base + (file_offset - ro_file_offset)
    table_size = num_entries * 12
    data = bytes((ctypes.c_uint8 * table_size).from_address(table_addr))
    delta_fields = {}
    for j in range(0, table_size, 12):
      type_id, delta_lo, delta_hi = struct.unpack('<III', data[j:j+12])
      if type_id < 32:
        delta_fields[type_id] = (delta_lo, delta_hi)
    return delta_fields

  # Delta tables: Layout 0 at 0x26800, Layout 2 at 0x26a80, Layout 3 at 0x26dc0, Layout 4 at 0x27300
  layout0 = read_table(0x26800, 24)  # L0 has 24 entries (no type 25)
  layout2 = read_table(0x26a80, 25)
  layout3 = read_table(0x26dc0, 25)
  layout4 = read_table(0x27300, 27)  # L4 has more entries
  return layout0, layout2, layout3, layout4

def extract_packet_encodings():
  """Extract packet type encodings from runtime packet type registrations.

  Returns dict mapping type_id -> (mask, value).
  """
  if not Path(ROCPROF_LIB).exists():
    return None

  ctypes.CDLL(ROCPROF_LIB)
  rw_base, rw_file_offset = _find_rw_segment()
  if rw_base is None:
    return None

  # Packet registrations vector at file offset 0x2d340 (ghidra DAT_0012e340 -> vaddr 0x2e340)
  vec_start_addr = rw_base + (0x2d340 - rw_file_offset)
  vec_end_addr = rw_base + (0x2d348 - rw_file_offset)

  vec_start = ctypes.c_void_p.from_address(vec_start_addr).value
  vec_end = ctypes.c_void_p.from_address(vec_end_addr).value
  if not vec_start or not vec_end:
    return None

  # Each entry is 32 bytes: type_id at offset 0, pattern_start at 8, pattern_end at 16
  encodings = {}
  for i in range((vec_end - vec_start) // 32):
    entry_addr = vec_start + i * 32
    type_id = ctypes.c_uint8.from_address(entry_addr).value
    pattern_start = ctypes.c_void_p.from_address(entry_addr + 8).value
    pattern_end = ctypes.c_void_p.from_address(entry_addr + 16).value

    if pattern_start and pattern_end:
      pattern_len = pattern_end - pattern_start
      if 0 < pattern_len <= 8:
        pattern = list((ctypes.c_uint8 * pattern_len).from_address(pattern_start))
        mask = sum(1 << j for j in range(pattern_len))
        value = sum(b << j for j, b in enumerate(pattern))
        encodings[type_id] = (mask, value)

  return encodings

@unittest.skipUnless(Path(ROCPROF_LIB).exists(), "rocprof-trace-decoder not installed")
class TestSQTTMatchesBinary(unittest.TestCase):
  def _test_bit_counts_match_layout(self, layout_num: int):
    from extra.assembly.amd.sqtt import PACKET_TYPES_L0, PACKET_TYPES_L3, PACKET_TYPES_L4
    layout0, layout2, layout3, layout4 = extract_bit_tables()
    layout = {0: layout0, 2: layout2, 3: layout3, 4: layout4}[layout_num]
    packet_types = {0: PACKET_TYPES_L0, 3: PACKET_TYPES_L3, 4: PACKET_TYPES_L4}[layout_num]

    for type_id, pkt_cls in packet_types.items():
      expected_bits, actual_bits = layout[type_id], pkt_cls._size_nibbles * 4
      with self.subTest(packet=pkt_cls.__name__):
        self.assertEqual(actual_bits, expected_bits, f"{pkt_cls.__name__}: {actual_bits} bits != expected {expected_bits}")

  def test_bit_counts_match_layout0(self): self._test_bit_counts_match_layout(0)
  def test_bit_counts_match_layout3(self): self._test_bit_counts_match_layout(3)
  def test_bit_counts_match_layout4(self): self._test_bit_counts_match_layout(4)

  def test_encodings_exist_in_binary(self):
    """Verify each PACKET_TYPE encoding exists in rocprof-trace-decoder."""
    from extra.assembly.amd.sqtt import PACKET_TYPES_L3

    encodings = extract_packet_encodings()

    for type_id, pkt_cls in PACKET_TYPES_L3.items():
      enc = (pkt_cls.encoding.mask, pkt_cls.encoding.default)
      with self.subTest(packet=pkt_cls.__name__):
        self.assertIn(type_id, encodings, f"{pkt_cls.__name__}: type_id {type_id} not in binary")
        self.assertEqual(enc, encodings[type_id],
          f"{pkt_cls.__name__}: encoding mismatch (ours=0x{enc[0]:02x}/0x{enc[1]:02x}, binary=0x{encodings[type_id][0]:02x}/0x{encodings[type_id][1]:02x})")

  def _test_delta_fields_match_layout(self, layout_num: int):
    from extra.assembly.amd.sqtt import PACKET_TYPES_L0, PACKET_TYPES_L3, PACKET_TYPES_L4
    packet_types = {0: PACKET_TYPES_L0, 3: PACKET_TYPES_L3, 4: PACKET_TYPES_L4}[layout_num]

    layout0_deltas, layout2_deltas, layout3_deltas, layout4_deltas = extract_delta_fields()
    delta_fields = {0: layout0_deltas, 2: layout2_deltas, 3: layout3_deltas, 4: layout4_deltas}[layout_num]

    for type_id, pkt_cls in packet_types.items():
      if type_id not in delta_fields:
        continue
      expected_lo, expected_hi = delta_fields[type_id]
      delta_field = getattr(pkt_cls, 'delta', None)
      if delta_field is None:
        # NOP has no delta field, rocprof has (0, 0)
        actual_lo, actual_hi = 0, 0
      else:
        actual_lo = delta_field.lo
        # Our BitField hi is inclusive, rocprof's is exclusive, so convert
        actual_hi = delta_field.hi + 1
      with self.subTest(packet=pkt_cls.__name__):
        self.assertEqual((actual_lo, actual_hi), (expected_lo, expected_hi),
          f"{pkt_cls.__name__}: delta bits[{actual_hi}:{actual_lo}] != expected bits[{expected_hi}:{expected_lo}]")

  def test_delta_fields_match_layout0(self): self._test_delta_fields_match_layout(0)
  def test_delta_fields_match_layout3(self): self._test_delta_fields_match_layout(3)
  def test_delta_fields_match_layout4(self): self._test_delta_fields_match_layout(4)

if __name__ == "__main__":
  layout0, layout2, layout3, layout4 = extract_bit_tables()
  encodings = extract_packet_encodings()
  delta0, delta2, delta3, delta4 = extract_delta_fields()

  TYPE_NAMES = {
    1: 'VALUINST', 2: 'VMEMEXEC', 3: 'ALUEXEC', 4: 'IMMEDIATE', 5: 'IMMEDIATE_MASK',
    6: 'WAVERDY', 7: 'TS_DELTA_S8_W3', 8: 'WAVEEND', 9: 'WAVESTART', 10: 'TS_DELTA_S5_W2',
    11: 'WAVEALLOC', 12: 'TS_DELTA_S5_W3', 13: 'PERF', 14: 'UTILCTR', 15: 'TS_DELTA_SHORT',
    16: 'NOP', 17: 'TS_WAVE_STATE', 18: 'EVENT', 19: 'EVENT_BIG', 20: 'REG',
    21: 'SNAPSHOT', 22: 'TS_DELTA_OR_MARK', 23: 'LAYOUT_HEADER', 24: 'INST',
  }

  print("L0:", layout0)
  print("L2:", layout2)
  print("L3:", layout3)
  print("L4:", layout4)

  if encodings and layout3:
    print("\nPacket type registrations from rocprof-trace-decoder:\n")
    print(f"{'TypeID':>6} {'Name':>18} {'Mask':>6} {'Value':>6} {'L0':>4} {'L3':>4} {'L4':>4} {'L0 delta':>12} {'L3 delta':>12} {'L4 delta':>12} {'Pattern'}")
    print("-" * 130)
    for type_id in sorted(encodings.keys()):
      mask, value = encodings[type_id]
      name = TYPE_NAMES.get(type_id, f'UNK_{type_id}')
      l0 = layout0[type_id] if type_id < len(layout0) else 0
      l3 = layout3[type_id] if type_id < len(layout3) else 0
      l4 = layout4[type_id] if type_id < len(layout4) else 0
      d0 = delta0.get(type_id, (0, 0)) if delta0 else (0, 0)
      d3 = delta3.get(type_id, (0, 0)) if delta3 else (0, 0)
      d4 = delta4.get(type_id, (0, 0)) if delta4 else (0, 0)
      d0_str = f"[{d0[1]-1}:{d0[0]}]" if d0[1] > d0[0] else "-"
      d3_str = f"[{d3[1]-1}:{d3[0]}]" if d3[1] > d3[0] else "-"
      d4_str = f"[{d4[1]-1}:{d4[0]}]" if d4[1] > d4[0] else "-"
      # Reconstruct pattern from mask/value
      pattern = [(value >> i) & 1 for i in range(mask.bit_length())]
      print(f"{type_id:6d} {name:>18} 0x{mask:04x} 0x{value:04x} {l0:4d} {l3:4d} {l4:4d} {d0_str:>12} {d3_str:>12} {d4_str:>12} {pattern}")

  unittest.main()
