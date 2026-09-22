import struct, unittest
from test.mockgpu.qcom.pm4 import Packet, decode

def raw(*words): return struct.pack(f"<{len(words)}I", *words)

class TestPM4(unittest.TestCase):
  # Synthetic little-endian fixtures, independent of tinygrad's packet builders.
  def test_mixed(self):
    data = bytes.fromhex('02010040 efbeadde 78563412 00802670 01001070 00000000')
    self.assertEqual(decode(data), (Packet(0, 4, 1, (0xdeadbeef, 0x12345678)), Packet(3, 7, 0x26, ()), Packet(4, 7, 0x10, (0,))))

  def test_empty(self): self.assertEqual(decode(b''), ())

  def test_zero_counts(self):
    self.assertEqual(decode(raw(0x40000180, 0x70268000)), (Packet(0, 4, 1, ()), Packet(1, 7, 0x26, ())))

  def test_max_counts(self):
    for header, kind, target, count in ((0x4000017f, 4, 1, 127), (0x7026bfff, 7, 0x26, 16383)):
      with self.subTest(kind=kind): self.assertEqual(decode(raw(header, *range(count))), (Packet(0, kind, target, tuple(range(count))),))

  def test_register_boundary(self):
    self.assertEqual(decode(raw(0x4bffff01, 123)), (Packet(0, 4, 0x3ffff, (123,)),))
    with self.assertRaisesRegex(ValueError, r'dword 0.*register span 0x3ffff\+2'): decode(raw(0x4bffff02, 123, 456))

  def test_payload_is_opaque(self):
    self.assertEqual(decode(raw(0x40000102, 0xffffffff, 0x70268000))[0].payload, (0xffffffff, 0x70268000))

  def test_opcode_is_not_execution(self):
    self.assertEqual(decode(raw(0x70808000)), (Packet(0, 7, 0, ()),))

  def test_unaligned(self):
    for tail in (b'\x00', b'\x00\x00', b'\x00\x00\x00'):
      with self.subTest(size=len(tail)), self.assertRaisesRegex(ValueError, 'dword 1: unaligned'): decode(raw(0x70268000) + tail)

  def test_unsupported_types(self):
    for kind in set(range(16)) - {4, 7}:
      with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, f'dword 1.*unsupported packet type {kind}'):
        decode(raw(0x70268000, kind << 28))

  def test_parity(self):
    for header, bits in ((0x40000101, (7, 27)), (0x70100001, (15, 23))):
      for bit in bits:
        with self.subTest(header=header, bit=bit), self.assertRaisesRegex(ValueError, 'dword 1.*invalid .* parity'):
          decode(raw(0x70268000, header ^ (1 << bit), 0))

  def test_reserved_bits(self):
    for header, bits in ((0x40000101, (26,)), (0x70100001, (14, 24, 25, 26, 27))):
      for bit in bits:
        with self.subTest(header=header, bit=bit), self.assertRaisesRegex(ValueError, 'dword 0.*unsupported header bits'):
          decode(raw(header | (1 << bit), 0))

  def test_truncated(self):
    for header in (0x40000102, 0x70100002):
      for count in range(2):
        with self.subTest(header=header, count=count), self.assertRaisesRegex(ValueError, f'dword 1.*need 2, have {count}'):
          decode(raw(0x70268000, header, *([0] * count)))

  def test_no_partial_result(self):
    result = None
    with self.assertRaisesRegex(ValueError, 'dword 1.*truncated'):
      result = decode(raw(0x70268000, 0x40000102, 0))
    self.assertIsNone(result)

if __name__ == '__main__': unittest.main()
