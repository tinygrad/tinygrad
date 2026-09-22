import struct, unittest
from test.mockgpu.qcom.errors import ErrorCode, ModelInputError
from test.mockgpu.qcom.state import QCOMGPU

def packet(control, lo=0, hi=0, *inline):
  count = 3 + len(inline)
  header = 0x70340000 | count | ((1 ^ (count.bit_count() & 1)) << 15)
  return struct.pack(f'<{count+1}I', header, control, lo, hi, *inline)

class TestState(unittest.TestCase):
  def setUp(self): self.gpu = QCOMGPU()

  def test_raw_shader_and_constants(self):
    # Independent raw CS packets: one 128-byte shader unit and one 16-byte constant unit.
    raw = bytes.fromhex('03803470 00007600 00100000 00000000 03803470 00407600 00200000 00000000')
    shader = bytes(range(128))
    self.gpu.submit(raw, {0x1000: shader, 0x2000: struct.pack('<4I', 0x80000000, 0x7fc00001, 0xffffffff, 1)})
    self.assertEqual(self.gpu.shader, shader)
    self.assertEqual(self.gpu.constants, {0: 0x80000000, 1: 0x7fc00001, 2: 0xffffffff, 3: 1})

  def test_direct_offset_and_overwrite(self):
    self.gpu.submit(packet(0x00744002, 0, 0, 1, 2, 3, 4), {})
    self.gpu.submit(packet(0x00744002, 0, 0, 5, 6, 7, 8) + packet(0x00744004, 0, 0, 9, 10, 11, 12), {})
    self.assertEqual(self.gpu.constants, dict(zip((8, 9, 10, 11, 16, 17, 18, 19), range(5, 13))))
    self.assertIsNone(self.gpu.shader)

  def test_high_address(self):
    self.gpu.submit(packet(0x00764001, 0x1004, 1), {0x100001000: bytes(4) + struct.pack('<4I', 1, 2, 3, 4)})
    self.assertEqual(self.gpu.constants, dict(enumerate((1, 2, 3, 4), 4)))

  def test_shader_replacement_and_snapshot(self):
    buffers = {0x1000: bytes(range(128))}
    self.gpu.submit(packet(0x00760000, 0x1000), buffers)
    buffers[0x1000] = bytes(128)
    self.assertEqual(self.gpu.shader, bytes(range(128)))
    self.gpu.submit(packet(0x00760000, 0x1000), buffers)
    self.assertEqual(self.gpu.shader, bytes(128))

  def test_direct_indirect_equivalence(self):
    values = (0, 0x80000000, 0x7fc00001, 0xffffffff, 1, 2, 3, 4)
    direct = QCOMGPU()
    direct.submit(packet(0x00b44003, 0, 0, *values), {})
    self.gpu.submit(packet(0x00b64003, 0x1000), {0x1000: struct.pack('<8I', *values)})
    self.assertEqual(self.gpu.constants, direct.constants)

  def test_error_in_second_packet_has_offset(self):
    with self.assertRaisesRegex(ValueError, 'dword 4.*address 0x2000\\+16 has 0 backing buffers'):
      self.gpu.submit(packet(0x00760000, 0x1000) + packet(0x00764000, 0x2000), {0x1000: bytes(128)})
    self.assertIsNone(self.gpu.shader)

  def test_instances_are_isolated(self):
    other = QCOMGPU()
    self.gpu.submit(packet(0x00744000, 0, 0, 1, 2, 3, 4), {})
    self.assertEqual(other.constants, {})
    self.assertIsNone(other.shader)

  def test_snapshot_restores_complete_binding_state(self):
    self.gpu.submit(packet(0x00744002, 0, 0, 1, 2, 3, 4), {})
    snapshot = self.gpu.snapshot()
    self.gpu.constants[0] = 99
    self.gpu.registers[0xa9b0] = 7
    self.gpu.shader, self.gpu.shader_addr = b'changed', 0x4000
    self.gpu.restore(snapshot)
    self.assertEqual(self.gpu.snapshot(), snapshot)
    self.assertEqual(self.gpu.constants, {8: 1, 9: 2, 10: 3, 11: 4})
    self.assertEqual(self.gpu.registers, {})

  def test_model_input_error_has_stable_code(self):
    with self.assertRaises(ModelInputError) as raised:
      self.gpu.submit(packet(0x00764000, 0x2000), {})
    self.assertEqual(raised.exception.code, ErrorCode.INPUT)

  def test_ten_bit_count(self):
    for count in (511, 512, 1023):
      with self.subTest(count=count):
        gpu = QCOMGPU()
        data = bytes(range(256)) * (count//16) + bytes(range((count % 16)*16))
        gpu.submit(packet((count << 22) | 0x364000, 0x1000), {0x1000: data})
        self.assertEqual(len(gpu.constants), count*4)
        self.assertEqual(struct.pack(f'<{count*4}I', *gpu.constants.values()), data)

  def test_unknown_states(self):
    for control in (0x00720000, 0x00768000, 0x0076c000, 0x00750000, 0x00770000):
      with self.subTest(control=hex(control)), self.assertRaisesRegex(ValueError, 'dword 0.*unsupported state'):
        self.gpu.submit(packet(control), {})

  def test_zero_count(self):
    with self.assertRaisesRegex(ValueError, 'count=0.*zero count'): self.gpu.submit(packet(0x00364000), {})

  def test_shader_modes(self):
    for control in (0x00740000, 0x00760001):
      with self.subTest(control=control), self.assertRaisesRegex(ValueError, 'only indirect shader preload'):
        self.gpu.submit(packet(control), {})

  def test_direct_length_and_address(self):
    for raw in (packet(0x00744000), packet(0x00744000, 0, 0, 1, 2, 3, 4, 5), packet(0x00744000, 4, 0, 1, 2, 3, 4)):
      with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, 'invalid direct payload or address'): self.gpu.submit(raw, {})

  def test_indirect_ranges(self):
    for lo, hi, blobs, error in ((0x1001, 0, {}, 'invalid indirect address'), (0xfffffffc, 0xffffffff, {}, 'invalid indirect address'),
                                (0x1000, 0, {}, '0 backing buffers'), (0x1000, 0, {0x1000: bytes(15)}, '0 backing buffers'),
                                (0x1000, 0, {0xffc: bytes(20), 0x1000: bytes(16)}, '2 backing buffers')):
      with self.subTest(lo=lo, hi=hi, error=error), self.assertRaisesRegex(ValueError, error):
        self.gpu.submit(packet(0x00764000, lo, hi), blobs)

  def test_indirect_extra_payload(self):
    with self.assertRaisesRegex(ValueError, 'invalid indirect address .* or payload'):
      self.gpu.submit(packet(0x00764000, 0x1000, 0, 1), {0x1000: bytes(16)})

  def test_destination_boundary(self):
    self.gpu.submit(packet(0x00747fff, 0, 0, 1, 2, 3, 4), {})
    self.assertEqual(self.gpu.constants, dict(enumerate((1, 2, 3, 4), 0x3fff*4)))
    with self.assertRaisesRegex(ValueError, 'destination exceeds'): self.gpu.submit(packet(0x00b47fff), {})

  def test_submission_atomicity(self):
    self.gpu.submit(packet(0x00744000, 0, 0, 1, 2, 3, 4), {})
    previous = self.gpu.constants
    for bad in (packet(0x00764000, 0x2000), bytes.fromhex('00802670'), b'\x00', packet(0x00744000)):
      with self.subTest(bad=bad), self.assertRaises(ValueError):
        self.gpu.submit(packet(0x00760000, 0x1000) + packet(0x00744000, 0, 0, 5, 6, 7, 8) + bad, {0x1000: bytes(128)})
      self.assertIsNone(self.gpu.shader)
      self.assertIs(self.gpu.constants, previous)
      self.assertEqual(previous, {0: 1, 1: 2, 2: 3, 3: 4})

  def test_short_control(self):
    with self.assertRaisesRegex(ValueError, 'needs three control dwords'):
      self.gpu.submit(bytes.fromhex('01003470 00407600'), {})

  def test_type4_rejected(self):
    with self.assertRaisesRegex(ValueError, 'type=4.*unsupported register'):
      self.gpu.submit(bytes.fromhex('01010040 00000000'), {})

  def test_empty_register_write(self):
    # Same dword test_pm4 accepts at framing level; the state layer rejects it.
    with self.assertRaisesRegex(ValueError, 'type=4.*empty register write'):
      self.gpu.submit(struct.pack('<I', 0x40000180), {})

if __name__ == '__main__': unittest.main()
