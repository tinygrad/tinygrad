import struct, unittest
from test.mockgpu.qcom.state import QCOMGPU
from test.mockgpu.qcom.test_state import packet

def reg(addr, *values):
  n = len(values)
  header = 0x40000000 | addr << 8 | n | (1 ^ (n.bit_count() & 1)) << 7 | (1 ^ (addr.bit_count() & 1)) << 27
  return struct.pack(f'<{n+1}I', header, *values)

class TestBinding(unittest.TestCase):
  def setUp(self):
    self.gpu = QCOMGPU()
    self.buffers = {0x1000: bytes(range(256)), 0x2000: struct.pack('<4I', 1, 2, 3, 4)}
    self.setup = reg(0xa9b4, 0x1000, 0) + reg(0xa9bc, 2) + reg(0xb987, 0x101)
    self.loads = packet(0x00760000, 0x1000) + packet(0x00764000, 0x2000)

  def test_raw_register_encoding(self):
    # Independent dwords, sourced from a6xx.xml register addresses and field widths.
    raw = bytes.fromhex('02b4a948 00100000 00000000 01bca940 02000000 0187b940 01010000')
    self.assertEqual(self.setup, raw)
    self.gpu.submit(raw + self.loads, self.buffers)
    self.assertEqual(self.gpu.registers, {0xa9b4: 0x1000, 0xa9b5: 0, 0xa9bc: 2, 0xb987: 0x101})
    self.assertEqual(self.gpu.bind(self.buffers, (0, 1, 2, 3)), self.buffers[0x1000])

  def test_partial_preload_is_valid(self):
    self.gpu.submit(self.setup + self.loads, self.buffers)
    self.assertEqual(self.gpu.shader, self.buffers[0x1000][:128])
    self.assertEqual(len(self.gpu.bind(self.buffers)), 256)

  def test_split_base_and_load_order(self):
    self.gpu.submit(self.loads, self.buffers)
    for raw in (reg(0xa9bc, 2), reg(0xa9b5, 0), reg(0xb987, 0x101), reg(0xa9b4, 0x1000)): self.gpu.submit(raw, {})
    self.assertEqual(self.gpu.bind(self.buffers), self.buffers[0x1000])

  def test_constlen_shift_and_holes(self):
    self.gpu.submit(self.setup + packet(0x00744003, 0, 0, 5, 6, 7, 8) + packet(0x00760000, 0x1000), self.buffers)
    self.gpu.bind(self.buffers, (12, 15))
    with self.assertRaisesRegex(ValueError, 'constant 0 uninitialized'): self.gpu.bind(self.buffers, (0,))
    with self.assertRaisesRegex(ValueError, 'constant 16 exceeds'): self.gpu.bind(self.buffers, (16,))
    self.gpu.submit(packet(0x00744004, 0, 0, 1, 2, 3, 4), {})
    with self.assertRaisesRegex(ValueError, 'loaded constants exceed'): self.gpu.bind(self.buffers)

  def test_no_constants_program(self):
    self.gpu.submit(self.setup + reg(0xb987, 0x100) + packet(0x00760000, 0x1000), self.buffers)
    self.assertEqual(self.gpu.bind(self.buffers), self.buffers[0x1000])

  def test_missing_registers(self):
    for missing in (0xa9b4, 0xa9b5, 0xa9bc, 0xb987):
      gpu = QCOMGPU()
      gpu.submit(self.setup + self.loads, self.buffers)
      del gpu.registers[missing]
      with self.subTest(missing=hex(missing)), self.assertRaisesRegex(ValueError, f'missing register {missing:#x}'): gpu.bind(self.buffers)

  def test_bad_program_binding(self):
    for change, error in ((reg(0xa9b4, 0x1004), 'not 32-byte aligned'), (reg(0xa9bc, 0), 'invalid range'),
                          (reg(0xa9bc, 3), '0 backing buffers'), (reg(0xb987, 1), 'constants disabled'),
                          (reg(0xa9b4, 0x1020) + reg(0xa9bc, 1), 'preload address differs')):
      gpu = QCOMGPU()
      gpu.submit(self.setup + self.loads + change, self.buffers)
      with self.subTest(error=error), self.assertRaisesRegex(ValueError, error): gpu.bind(self.buffers)

  def test_shader_missing_and_oversized(self):
    self.gpu.submit(self.setup, {})
    with self.assertRaisesRegex(ValueError, 'preload missing'): self.gpu.bind(self.buffers)
    self.gpu.submit(packet(0x00b60000, 0x1000) + reg(0xa9bc, 1), self.buffers)
    with self.assertRaisesRegex(ValueError, 'preload exceeds'): self.gpu.bind(self.buffers)

  def test_mutated_backing_and_revocation(self):
    self.gpu.submit(self.setup + self.loads, self.buffers)
    with self.assertRaisesRegex(ValueError, '0 backing buffers'): self.gpu.bind({})
    changed = dict(self.buffers)
    changed[0x1000] = b'\xff' + self.buffers[0x1000][1:]
    with self.assertRaisesRegex(ValueError, 'preload differs from program bytes'): self.gpu.bind(changed)
    self.assertEqual(self.gpu.bind(self.buffers), self.buffers[0x1000])

  def test_partial_overlapping_backing_rejected(self):
    self.gpu.submit(self.setup + self.loads, self.buffers)
    ambiguous = dict(self.buffers)
    ambiguous[0x10fc] = bytes(8)
    with self.assertRaisesRegex(ValueError, '2 backing buffers'): self.gpu.bind(ambiguous)

  def test_zero_length_backing_ignored(self):
    self.gpu.submit(self.setup + self.loads, self.buffers)
    padded = dict(self.buffers)
    padded[0x1080] = b''
    self.assertEqual(self.gpu.bind(padded), self.buffers[0x1000])

  def test_register_packet_atomicity(self):
    self.gpu.submit(self.setup + self.loads, self.buffers)
    old = self.gpu.registers
    with self.assertRaisesRegex(ValueError, 'unsupported register 0xa9b6'):
      self.gpu.submit(reg(0xa9b4, 0x3000, 1, 0), {})
    self.assertIs(self.gpu.registers, old)
    self.assertEqual(self.gpu.bind(self.buffers), self.buffers[0x1000])

  def test_invalid_register_bits(self):
    for address, value in ((0xa9bc, 1 << 28), (0xb987, 0x200), (0xb987, 1 << 31)):
      with self.subTest(address=address, value=value), self.assertRaisesRegex(ValueError, 'unsupported bits in register'):
        self.gpu.submit(reg(address, value), {})
    self.assertEqual(self.gpu.registers, {})

  def test_high_program_base(self):
    address = (1 << 32) + 0x1000
    buffers = {address: self.buffers[0x1000]}
    self.gpu.submit(self.setup + reg(0xa9b5, 1) + packet(0x00760000, 0x1000, 1), buffers)
    self.assertEqual(self.gpu.bind(buffers), buffers[address])

  def test_second_program_rebind(self):
    self.gpu.submit(self.setup + self.loads, self.buffers)
    buffers = {0x3000: b'\xab'*128}
    self.gpu.submit(reg(0xa9b4, 0x3000, 0) + reg(0xa9bc, 1), {})
    with self.assertRaisesRegex(ValueError, 'preload address differs'): self.gpu.bind(buffers)
    self.gpu.submit(packet(0x00760000, 0x3000), buffers)
    self.assertEqual(self.gpu.bind(buffers, (0,)), b'\xab'*128)

  def test_failed_load_rolls_back_registers(self):
    self.gpu.submit(self.setup + self.loads, self.buffers)
    old = self.gpu.registers
    with self.assertRaisesRegex(ValueError, 'backing buffers'):
      self.gpu.submit(reg(0xa9bc, 1) + packet(0x00760000, 0x3000), {})
    self.assertIs(self.gpu.registers, old)

if __name__ == '__main__': unittest.main()
