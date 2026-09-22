import struct, unittest
from pathlib import Path
from test.mockgpu.qcom.errors import ErrorCode, ModelInputError
from test.mockgpu.qcom.compute import QCOMCompute, FIXED

def packet(op, *values):
  count = len(values)
  return struct.pack(f'<{count+1}I', 0x70000000 | (op << 16) | count |
                     ((1 ^ (op.bit_count() & 1)) << 23) | ((1 ^ (count.bit_count() & 1)) << 15), *values)

def register(reg, *values):
  count = len(values)
  return struct.pack(f'<{count+1}I', 0x40000000 | (reg << 8) | count |
                     ((1 ^ (reg.bit_count() & 1)) << 27) | ((1 ^ (count.bit_count() & 1)) << 7), *values)

def scalar_state():
  raw = packet(0x65, 8) + register(0xbb08, 0x60) + register(0xbb08, 0)
  # Explicit raw producer configuration, independently transcribed from its scalar inventory.
  for reg, value in ((0xa9b0, 0x100), (0xa9b1, 0x41), (0xa9b2, 0), (0xa9b3, 0), (0xa9b4, 0x4000), (0xa9b5, 0),
                     (0xa9b6, 0), (0xa9b7, 0x5000), (0xa9b8, 0), (0xa9b9, 0), (0xa9ba, 0x80), (0xa9bb, 0x100),
                     (0xa9bc, 2), (0xa9bd, 0x1000), (0xaa00, 0x40), (0xab00, 5), (0xae0f, 0x20), (0xb309, 2), (0xb600, 0),
                     (0xb983, 0xfcfcfcfc), (0xb984, 0xfcfcfcfc), (0xb985, 0xfcfcfcfc), (0xb986, 0xfc), (0xb987, 0x140)):
    raw += register(reg, value)
  raw += register(0xb990, 3, 1, 0, 1, 0, 1, 0, 0xfcfcfcfc, 0xfc, 1, 1, 1)
  return raw + packet(0x34, 0xb60000, 0x4000, 0) + packet(0x34, 0xb44000, 0, 0, 0x3000, 0, 0x1000, 0, 0x2000, 0, 0, 0)

def scalar_memory():
  return {0x1000: bytearray.fromhex('ffffffff'), 0x2000: bytearray.fromhex('02000000'), 0x3000: bytearray.fromhex('aaaaaaaa'),
          0x4000: bytearray((Path(__file__).parent / 'fixtures/compute/scalar.bin').read_bytes())}

class TestComputeControl(unittest.TestCase):
  def test_event_wait_visibility(self):
    gpu, memory = QCOMCompute(), {0x100001000: bytearray.fromhex('aaaaaaaa bbbbbbbb')}
    # Literal captured-producer forms with independent address/value inputs.
    raw = bytes.fromhex('04004670 04000000 00100000 01000000 07000000 00809270 00802670')
    raw += packet(0x3c, 0x15, 0x1000, 1, 7, 0xffffffff, 32) + packet(0x46, 0x31)
    gpu.run(raw, memory)
    self.assertEqual(memory[0x100001000], bytes.fromhex('07000000 bbbbbbbb'))
    self.assertEqual((gpu.flushes, gpu.invalidations, gpu.idle_waits, gpu.memory_waits), (1, 1, 1, 1))
    gpu.run(packet(0x3c, 0x15, 0x1000, 1, 6, 0xffffffff, 32), memory)

  def test_wait_unsigned_and_failure_rollback(self):
    gpu, memory = QCOMCompute(), {0x1000: bytearray.fromhex('ffffffff')}
    gpu.run(packet(0x3c, 0x15, 0x1000, 0, 0x80000000, 0xffffffff, 32), memory)
    raw = packet(0x46, 4, 0x1000, 0, 3) + packet(0x3c, 0x15, 0x1000, 0, 4, 0xffffffff, 32)
    with self.assertRaisesRegex(ValueError, 'dword 5.*unsatisfied wait 3 < 4'): gpu.run(raw, memory)
    self.assertEqual(memory[0x1000], bytes.fromhex('ffffffff'))
    self.assertEqual(gpu.flushes, 0)

  def test_events_in_adjacent_buffers(self):
    gpu, memory = QCOMCompute(), {0x1000: bytearray(b'abcdefgh'), 0x1008: bytearray(b'ijklmnop')}
    gpu.run(packet(0x46, 4, 0x1004, 0, 7) + packet(0x46, 4, 0x1008, 0, 9), memory)
    self.assertEqual(memory, {0x1000: b'abcd\x07\0\0\0', 0x1008: b'\x09\0\0\0mnop'})
    self.assertEqual(gpu.flushes, 2)
    with self.assertRaisesRegex(ValueError, 'backing buffers'):
      gpu.run(packet(0x46, 4, 0x100c, 0, 11) + packet(0x46, 4, 0x1010, 0, 13), memory)
    self.assertEqual(memory[0x1008], b'\x09\0\0\0mnop')
    self.assertEqual(gpu.flushes, 2)

  def test_marker_and_bad_packets(self):
    gpu = QCOMCompute()
    gpu.run(packet(0x65, 8), {})
    self.assertEqual(gpu.mode, 8)
    for raw in (packet(0x65, 1), packet(0x65), packet(0x26, 0), packet(0x12, 0), packet(0x46),
                packet(0x46, 4), packet(0x46, 0x40000004, 0x1000, 0, 7), packet(0x46, 0x80000004, 0x1000, 0, 7),
                packet(0x46, 0x31, 0), packet(0x3c, 0x15), packet(0x10)):
      with self.subTest(raw=raw.hex()):
        with self.assertRaisesRegex(ValueError, 'PM4 dword 0'): gpu.run(raw, {})
    self.assertEqual(gpu.mode, 8)

  def test_wait_fields(self):
    values = [0x15, 0x1000, 0, 0, 0xffffffff, 32]
    for index, value in ((0, 0x1d), (0, 5), (4, 0xffff), (5, 0)):
      with self.subTest(index=index, value=value):
        changed = values.copy()
        changed[index] = value
        with self.assertRaisesRegex(ValueError, 'unsupported WAIT_REG_MEM'): QCOMCompute().run(packet(0x3c, *changed), {})

  def test_memory_bounds_and_alias(self):
    for addr in (0x1001, 0xffc, 0x1004):
      for raw in (packet(0x46, 4, addr, 0, 7), packet(0x3c, 0x15, addr, 0, 0, 0xffffffff, 32)):
        with self.subTest(addr=addr, raw=raw.hex()):
          with self.assertRaisesRegex(ValueError, '(unaligned|backing buffers)'): QCOMCompute().run(raw, {0x1000: bytearray(4)})
    shared = bytearray(4)
    for memory in ({0: shared, 4: shared}, {0: bytearray(8), 4: bytearray(4)}, {-4: bytearray(4)}, {0: bytearray()}):
      with self.subTest(memory=memory):
        with self.assertRaisesRegex(ValueError, 'QCOM memory'): QCOMCompute().run(b'', memory)

class TestComputeDispatch(unittest.TestCase):
  def test_dispatch_instrumentation_is_read_only(self):
    events: list[dict[str, object]] = []
    gpu, memory = QCOMCompute(events.append), scalar_memory()
    gpu.run(scalar_state() + packet(0x33, 0, 1, 1, 1), memory)
    self.assertEqual(len(events), 1)
    self.assertEqual(events[0]['kind'], 'dispatch')
    self.assertEqual(events[0]['groups'], [1, 1, 1])
    self.assertEqual(events[0]['local'], [1, 1, 1])
    self.assertEqual(events[0]['global'], [1, 1, 1])
    self.assertRegex(str(events[0]['program_sha256']), r'^[0-9a-f]{64}$')

  def test_scalar_dispatch_and_repeated_submission(self):
    gpu, memory = QCOMCompute(), scalar_memory()
    gpu.run(scalar_state() + packet(0x33, 0, 1, 1, 1), memory)
    self.assertEqual(memory[0x3000], bytes.fromhex('01000000'))
    self.assertEqual(gpu.dispatches, 1)
    memory[0x1000][:] = bytes.fromhex('05000000')
    gpu.run(packet(0x33, 0, 1, 1, 1), memory)
    self.assertEqual(memory[0x3000], bytes.fromhex('07000000'))
    self.assertEqual(gpu.dispatches, 2)

  def test_dispatch_rolls_back_outputs_and_state(self):
    gpu, memory = QCOMCompute(), scalar_memory()
    raw = scalar_state() + packet(0x33, 0, 1, 1, 1) + packet(0x10)
    before = gpu.snapshot()
    with self.assertRaisesRegex(ValueError, 'unsupported packet'): gpu.run(raw, memory)
    self.assertEqual(memory[0x3000], bytes.fromhex('aaaaaaaa'))
    self.assertEqual((gpu.dispatches, gpu.mode, gpu.registers), (0, 0, {}))
    self.assertEqual(gpu.snapshot(), before)

  def test_compute_snapshot_includes_dispatch_state(self):
    gpu = QCOMCompute()
    gpu.mode, gpu.dispatches, gpu.flushes = 8, 3, 2
    snapshot = gpu.snapshot()
    gpu.mode, gpu.dispatches, gpu.flushes = 0, 0, 0
    gpu.restore(snapshot)
    self.assertEqual(gpu.snapshot(), snapshot)

  def test_compute_input_error_code(self):
    with self.assertRaises(ModelInputError) as raised:
      QCOMCompute().run(packet(0x10), {})
    self.assertEqual(raised.exception.code, ErrorCode.INPUT)

  def test_reset_clears_pending_state(self):
    gpu, memory = QCOMCompute(), scalar_memory()
    gpu.run(scalar_state(), memory)
    gpu.run(register(0xbb08, 0x60) + register(0xbb08, 0), memory)
    self.assertIsNone(gpu.shader)
    self.assertEqual(gpu.constants, {})
    with self.assertRaisesRegex(ValueError, 'shader preload missing'): gpu.run(packet(0x33, 0, 1, 1, 1), memory)

  def test_invalid_dispatch_state(self):
    for change, message in ((packet(0x33, 1, 1, 1, 1), 'EXEC_CS form'), (packet(0x33, 0, 0, 1, 1), 'dispatch dimensions'),
                            (packet(0x33, 0, 2, 1, 1), 'group registers disagree'),
                            (register(0xb991, 2) + packet(0x33, 0, 1, 1, 1), 'global sizes disagree'),
                            (register(0xb990, 2) + packet(0x33, 0, 1, 1, 1), 'kernel dimension'),
                            (register(0xb987, 0x141) + packet(0x33, 0, 1, 1, 1), 'constant RAM mode'),
                            (register(0xb987, 0x100) + packet(0x33, 0, 1, 1, 1), 'loaded constants exceed CONSTLEN'),
                            (register(0xb997, 0xf3fcfcfc) + packet(0x33, 0, 1, 1, 1), 'system registers'),
                            (register(0xb997, 0x00fcfc00) + packet(0x33, 0, 1, 1, 1), 'system registers')):
      with self.subTest(message=message):
        gpu, memory = QCOMCompute(), scalar_memory()
        gpu.run(scalar_state(), memory)
        before = gpu.registers.copy()
        with self.assertRaisesRegex(ValueError, message): gpu.run(change, memory)
        self.assertEqual(gpu.registers, before)
        self.assertEqual(memory[0x3000], bytes.fromhex('aaaaaaaa'))

  def test_missing_registers_and_unsupported_values(self):
    gpu, memory = QCOMCompute(), scalar_memory()
    gpu.run(scalar_state(), memory)
    for reg in list(gpu.registers):
      with self.subTest(reg=reg):
        value = gpu.registers.pop(reg)
        with self.assertRaisesRegex(ValueError, 'missing compute register'): gpu.run(packet(0x33, 0, 1, 1, 1), memory)
        gpu.registers[reg] = value
    for reg, values in FIXED.items():
      with self.subTest(reg=reg):
        with self.assertRaisesRegex(ValueError, 'unsupported register'): gpu.run(register(reg, max(values)+1), memory)
    with self.assertRaisesRegex(ValueError, 'unsupported register'): gpu.run(register(0x1234, 0), memory)
    with self.assertRaisesRegex(ValueError, 'empty register'): gpu.run(register(0xa9b0), memory)

  def test_dimension_limits_and_system_profile(self):
    for raw in (packet(0x33, 0, 1, 0, 1), packet(0x33, 0, 1, 1, 0), packet(0x33, 0, 65537, 1, 1),
                register(0xb990, 3 | (1023 << 2) | (1 << 12)) + packet(0x33, 0, 1, 1, 1)):
      with self.subTest(raw=raw.hex()):
        with self.assertRaisesRegex(ValueError, 'dispatch dimensions'):
          QCOMCompute().run(scalar_state() + raw, scalar_memory())
    for ids in (0xfcfcfcc1, 0x01fcfcc0, 0xfcfcc0fc, 0xfcc0fcfc):
      with self.subTest(ids=ids):
        with self.assertRaisesRegex(ValueError, 'unsupported system registers'):
          QCOMCompute().run(scalar_state() + register(0xb997, ids) + packet(0x33, 0, 1, 1, 1), scalar_memory())

  def test_three_dimensional_group_and_local_ids(self):
    def shift(dst, src, amount): return 0x46d0000020000000 | (dst << 32) | (amount << 16) | src
    def add(dst, a, b): return 0x4210000000000000 | (dst << 32) | (b << 16) | a
    words = [shift(20, 192, 1), add(20, 20, 0), shift(21, 193, 1), add(21, 21, 1), shift(21, 21, 2),
             shift(22, 194, 1), add(22, 22, 2), shift(22, 22, 4), add(20, 20, 21), add(20, 20, 22), shift(20, 20, 5),
             add(40, 20, 0x1000), 0x202cc02900000001, 0xc0c6010003800000 | (40 << 41) | (192 << 1),
             add(40, 40, 0x1002), 0xc0c6010003800000 | (40 << 41), 0x0300000000000000]
    program = struct.pack(f'<{len(words)}Q', *words).ljust(256, b'\0')
    memory = {0x4000: bytearray(program), 0x10000: bytearray(b'\xaa' * 2048)}
    raw = scalar_state() + packet(0x34, 0x744000, 0, 0, 0x10000, 0, 16, 0)
    raw += register(0xb990, 3 | (1 << 2) | (1 << 12) | (1 << 22), 4, 0, 4, 0, 4, 0, 0x00fcfcc0, 0xfc, 2, 2, 2)
    QCOMCompute().run(raw + packet(0x33, 0, 2, 2, 2), memory)
    for z in range(4):
      for y in range(4):
        for x in range(4):
          offset = (z*16+y*4+x)*32
          expected = struct.pack('<4I', x//2, y//2, z//2, 0xaaaaaaaa) + struct.pack('<4I', x%2, y%2, z%2, 0xaaaaaaaa)
          self.assertEqual(memory[0x10000][offset:offset+32], expected)

if __name__ == '__main__': unittest.main()
