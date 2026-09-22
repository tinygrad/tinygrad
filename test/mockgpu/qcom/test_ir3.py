import struct, unittest
from pathlib import Path
from test.mockgpu.qcom.ir3 import execute

ADD = bytes.fromhex('0000010002001042')
END = bytes.fromhex('0000000000000003')

class TestIR3(unittest.TestCase):
  def test_sample(self):
    program = ADD + END
    # Independent literal sums include wraparound and signed-boundary crossings.
    for a, b, expected in ((0, 0, 0), (1, 2, 3), (0xffffffff, 1, 0), (0x7fffffff, 1, 0x80000000), (0xffffffff, 0xffffffff, 0xfffffffe)):
      with self.subTest(a=a, b=b):
        inputs = {0: a, 1: b, 7: 0xdeadbeef}
        self.assertEqual(execute(program, inputs), {**inputs, 2: expected})
        self.assertEqual(inputs, {0: a, 1: b, 7: 0xdeadbeef})

  def test_carry_each_bit(self):
    for bit in range(32):
      with self.subTest(bit=bit):
        self.assertEqual(execute(ADD + END, {0: (1 << bit)-1, 1: 1})[2], 1 << bit)

  def test_aliasing(self):
    for raw, inputs, expected in (
      ('0000010000001042', {0: 5, 1: 7}, {0: 12, 1: 7}),
      ('0000010001001042', {0: 5, 1: 7}, {0: 5, 1: 12}),
      ('0000000000001042', {0: 0x80000001}, {0: 2}),
    ):
      with self.subTest(raw=raw): self.assertEqual(execute(bytes.fromhex(raw) + END, inputs), expected)

  def test_sequential_writeback(self):
    # add.u r0.w, r0.z, r0.y observes the preceding r0.z result in this sequential model.
    program = ADD + bytes.fromhex('0200010003001042') + END
    self.assertEqual(execute(program, {0: 5, 1: 7}), {0: 5, 1: 7, 2: 12, 3: 19})

  def test_register_components(self):
    for reg in range(244):
      with self.subTest(reg=reg):
        raw = struct.pack('<Q', 0x4210000000000000 | reg | (reg << 16) | (reg << 32))
        self.assertEqual(execute(raw + END, {reg: 3}), {reg: 6})

  def test_special_registers(self):
    for reg in range(244, 256):
      for shift in (0, 16, 32):
        with self.subTest(reg=reg, shift=shift):
          raw = struct.pack('<Q', 0x4210000000000000 | (reg << shift))
          with self.assertRaisesRegex(ValueError, 'pc=0x0.*add.u.*special/reserved'):
            execute(raw + END, {0: 1})

  def test_reject_every_non_register_bit(self):
    for bit in set(range(64)) - set(range(8)) - set(range(16, 24)) - set(range(32, 40)) - {12, 13, 28, 29, 40, 41, 43, 44, 51, 54, 55, 56, 60}:
      with self.subTest(bit=bit):
        raw = struct.pack('<Q', int.from_bytes(ADD, 'little') ^ (1 << bit))
        with self.assertRaisesRegex(ValueError, 'pc=0x0.*word=.*unsupported'): execute(raw + END, {0: 1, 1: 2})

  def test_end_modifiers(self):
    for bit in set(range(64)) - {57}:  # This toggles end into jump, tested in test_ir3_flow.
      with self.subTest(bit=bit):
        raw = struct.pack('<Q', int.from_bytes(END, 'little') ^ (1 << bit))
        with self.assertRaisesRegex(ValueError, 'pc=0x0.*word=.*unsupported'): execute(raw, {})

  def test_uninitialized(self):
    for inputs, missing in (({}, 0), ({0: 1}, 1), ({1: 1}, 0)):
      with self.subTest(inputs=inputs):
        with self.assertRaisesRegex(ValueError, f'pc=0x0.*add.u.*uninitialized register {missing}'):
          execute(ADD + END, inputs)

  def test_invalid_initial_state(self):
    for inputs in ({-1: 0}, {244: 0}, {True: 0}, {0.0: 0}, {0: -1}, {0: 1 << 32}, {0: True}, {0: 1.5}):
      with self.subTest(inputs=inputs):
        with self.assertRaisesRegex(ValueError, 'pc=0x0.*invalid'): execute(END, inputs)  # type: ignore[arg-type]

  def test_length_and_termination(self):
    for size in range(16):
      if size == 8: continue
      with self.subTest(size=size):
        with self.assertRaisesRegex(ValueError, 'empty or truncated'): execute((ADD + END)[:size], {})
    with self.assertRaisesRegex(ValueError, 'pc=0x8.*missing end'): execute(ADD, {0: 1, 1: 2})
    with self.assertRaisesRegex(ValueError, 'pc=0x0.*trailing'): execute(END + ADD, {})
    with self.assertRaisesRegex(ValueError, 'pc=0x8.*trailing'): execute(ADD + END + bytes(8), {0: 1, 1: 2})

  def test_bounded_program(self):
    self.assertEqual(execute(ADD*4095 + END, {0: 1, 1: 2}), {0: 1, 1: 2, 2: 3})
    with self.assertRaisesRegex(ValueError, 'pc=0x8000.*missing end'): execute(ADD*4096, {0: 1, 1: 2})
    with self.assertRaisesRegex(ValueError, 'pc=0x0.*4096 instruction limit'): execute(ADD*4096 + END, {0: 1, 1: 2})

  def test_unsupported_categories(self):
    for cat in (0, 1, 3, 4, 5, 6, 7):
      with self.subTest(cat=cat):
        word = (cat << 61) | 1 | ((7 << 50) if cat == 1 else 0)
        with self.assertRaisesRegex(ValueError, f'pc=0x0.*cat={cat}.*unsupported'):
          execute(struct.pack('<Q', word) + END, {})

  def test_failure_isolation_and_pc(self):
    inputs = {0: 1, 1: 2}
    with self.assertRaisesRegex(ValueError, 'pc=0x8 word=0xe000000000000000 cat=7.*unsupported'):
      execute(ADD + bytes.fromhex('00000000000000e0') + END, inputs)
    self.assertEqual(inputs, {0: 1, 1: 2})
    self.assertEqual(execute(END, {}), {})
    result = execute(ADD + END, inputs)
    result[0] = 99
    self.assertEqual(execute(ADD + END, inputs), {0: 1, 1: 2, 2: 3})

class TestIR3Constants(unittest.TestCase):
  def test_bit_patterns(self):
    for value in (0, 1, 0x7fffffff, 0x80000000, 0xffffffff, 0x7fc00001):
      with self.subTest(value=value):
        consts = {2: value}
        self.assertEqual(execute(bytes.fromhex('0200000000c02c20') + END, {7: 99}, consts), {0: value, 7: 99})
        self.assertEqual(consts, {2: value})

  def test_full_constant_index(self):
    for index in range(2048):
      with self.subTest(index=index):
        raw = struct.pack('<Q', 0x202cc00000000000 | index)
        self.assertEqual(execute(raw + END, {}, {index: 0xdeadbeef}), {0: 0xdeadbeef})

  def test_destination_boundaries(self):
    for dst in range(256):
      with self.subTest(dst=dst):
        raw = struct.pack('<Q', 0x202cc00000000000 | (dst << 32)) + END
        if dst < 244: self.assertEqual(execute(raw, {}, {0: 42}), {dst: 42})
        else:
          with self.assertRaisesRegex(ValueError, 'pc=0x0.*mov.u32u32.*special/reserved'): execute(raw, {}, {0: 42})

  def test_non_operand_bits(self):
    for bit in set(range(64)) - set(range(11)) - set(range(32, 40)) - {53}:
      with self.subTest(bit=bit):
        raw = struct.pack('<Q', 0x202cc00000000000 ^ (1 << bit))
        with self.assertRaisesRegex(ValueError, 'pc=0x0.*unsupported'): execute(raw + END, {0: 1}, {0: 1})

  def test_missing_and_invalid_constants(self):
    raw = bytes.fromhex('ff07000000c02c20') + END
    with self.assertRaisesRegex(ValueError, 'pc=0x0.*const=2047.*uninitialized constant 2047'): execute(raw, {}, {0: 1})
    with self.assertRaisesRegex(ValueError, 'uninitialized constant 2047'): execute(raw, {})
    for consts in ({-1: 0}, {2048: 0}, {True: 0}, {0.0: 0}, {0: -1}, {0: 1 << 32}, {0: True}, {0: 1.5}):
      with self.subTest(consts=consts):
        with self.assertRaisesRegex(ValueError, 'invalid.*constant'): execute(END, {}, consts)  # type: ignore[arg-type]

  def test_move_add_and_failure_isolation(self):
    raw = bytes.fromhex('0000000000c02c20 0100000001c02c20')
    inputs, constants = {2: 99}, {0: 0xffffffff, 1: 2}
    self.assertEqual(execute(raw + ADD + END, inputs, constants), {0: 0xffffffff, 1: 2, 2: 1})
    with self.assertRaisesRegex(ValueError, 'pc=0x10.*unsupported'): execute(raw + bytes.fromhex('00000000000000e0') + END, inputs, constants)
    self.assertEqual(inputs, {2: 99})
    self.assertEqual(constants, {0: 0xffffffff, 1: 2})

  def test_compiled_prefix(self):
    code = (Path(__file__).parent / 'fixtures/compute/scalar.bin').read_bytes()
    # Six real compiler moves plus a synthetic terminator, not the complete workload.
    constants = {0: 10, 1: 11, 2: 12, 3: 13, 4: 14, 5: 15}
    self.assertEqual(execute(code[:48] + END, {}, constants), {0: 12, 1: 13, 3: 14, 4: 15, 5: 10, 6: 11})
    with self.assertRaisesRegex(ValueError, 'pc=0x38.*unmapped range 0xd0000000c'): execute(code, {}, constants)

if __name__ == '__main__': unittest.main()
