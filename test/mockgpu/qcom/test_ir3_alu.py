import struct, unittest
from test.mockgpu.qcom.ir3 import execute

END = bytes.fromhex('0000000000000003')
def instruction(word): return struct.pack('<Q', word)

class TestIR3ALU(unittest.TestCase):
  def test_floor_float(self):
    for value, expected in ((0x3fc00000, 0x3f800000), (0xbfc00000, 0xc0000000), (0x00000000, 0x00000000)):
      with self.subTest(value=value):
        raw = struct.pack('<QQ', 0x4130000000000000 | (2 << 32) | 1, 0x0300000000000000)
        self.assertEqual(execute(raw, {0: 0, 1: value})[2], expected)

  def test_sign_float(self):
    raw = instruction(0x4090000000000000 | (2 << 32) | 1) + END
    for value, expected in ((0x3fc00000, 0x3f800000), (0xbfc00000, 0xbf800000), (0, 0)):
      with self.subTest(value=value): self.assertEqual(execute(raw, {0: 0, 1: value})[2], expected)

  def test_absneg_signed(self):
    raw = instruction(0x4350000200004001) + END
    self.assertEqual(execute(raw, {0: 0, 1: 7})[2], 0xfffffff9)

  def test_clz(self):
    for base in (0x4690000000000000, 0x46b0000000000000):
      with self.subTest(base=base):
        self.assertEqual(execute(instruction(base | (2 << 32) | 1) + END, {1: 0x00100000})[2], 11)
        self.assertEqual(execute(instruction(base | (2 << 32) | 1) + END, {1: 0})[2], 32)

  def test_trunc_float(self):
    raw = instruction(0x41b0000000000000 | (2 << 32) | 1) + END
    for value, expected in ((0x3fc00000, 0x3f800000), (0xbfc00000, 0xbf800000), (0x3f800000, 0x3f800000)):
      with self.subTest(value=value): self.assertEqual(execute(raw, {0: 0, 1: value})[2], expected)

  def test_moves_and_immediate_sign(self):
    self.assertEqual(execute(bytes.fromhex('c000000001c00c20') + END, {192: 0xdeadbeef})[1], 0xdeadbeef)
    self.assertEqual(execute(instruction(0x204cc003ffffffff) + END, {})[3], 0xffffffff)
    for code, expected in ((0x2000, 1), (0x23ff, 1024), (0x2400, 0xfffffc01), (0x27ff, 0)):
      with self.subTest(code=code):
        self.assertEqual(execute(instruction(0x4210000100000000 | (code << 16)) + END, {0: 1})[1], expected)

  def test_multiply_low_and_cross_terms(self):
    raw = bytes.fromhex('0b00100011085046 0b00110011088861 1080110010888561')
    for a, b, expected in ((2, 4, 8), (0x10000, 0x10000, 0), (0xffffffff, 0xffffffff, 1),
                            (0x12345678, 0x00010001, 0x68ac5678), (0x80000000, 3, 0x80000000)):
      with self.subTest(a=a, b=b): self.assertEqual(execute(raw + END, {11: a, 16: b})[16], expected)

  def test_xor_bits(self):
    raw = instruction(0x43f0080200010000) + END
    self.assertEqual(execute(raw, {0: 0xaaaa5555, 1: 0x0f0ff0f0})[2], 0xa5a5a5a5)

  def test_or_bits(self):
    raw = instruction(0x43b0000200010000) + END
    self.assertEqual(execute(raw, {0: 0xaaaa0000, 1: 0x00005555})[2], 0xaaaa5555)

  def test_cat3_shlg_immediate(self):
    raw = instruction(0x65878423000e3018) + END
    self.assertEqual(execute(raw, {14: 3, 15: 1})[35], (1 << 24) | 3)

  def test_cat3_shrg_masks_shift_count(self):
    shrg = (0x6514042d002d3001 & ~0x1fff) | 0x2010
    for count, expected in ((32, 0x12345678), (33, 0x091a2b3c)):
      with self.subTest(count=count): self.assertEqual(execute(instruction(shrg) + END, {16: count, 40: 0x12345678, 45: 0})[45], expected)

  def test_cat3_shrg_accepts_r0_source(self):
    raw = instruction(0x650084010000301e) + END
    self.assertEqual(execute(raw, {0: 0x12345678, 1: 0x80000000})[1], 0x1234567a)

  def test_half_and_immediate_destination(self):
    with self.assertRaisesRegex(ValueError, 'uninitialized half'):
      execute(instruction(0x4380000008000000) + END, {0: 1})

  def test_sub_unsigned(self):
    raw = instruction(0x4250080200010000) + END
    self.assertEqual(execute(raw, {0: 3, 1: 5})[2], 0xfffffffe)

  def test_float_rounding_and_sign(self):
    raw = instruction(0x4010000200010000) + END
    for a, b, expected in ((0x3f800000, 0x40000000, 0x40400000), (0x3f800000, 0xbf800000, 0),
                            (0x80000000, 0x80000000, 0x80000000), (0x3f800000, 0x33800000, 0x3f800000),
                            (0x3f800001, 0x33800000, 0x3f800002)):
      with self.subTest(a=a, b=b): self.assertEqual(execute(raw, {0: a, 1: b})[2], expected)
    for a, b in ((1, 0), (0x00800001, 0x80800000)):
      with self.subTest(a=a, b=b):
        with self.assertRaisesRegex(ValueError, 'float'): execute(raw, {0: a, 1: b})

  def test_new_forms_reject_modifiers(self):
    for word in (0x200cc00100000000 | (1 << 42), 0x204cc00100000000 | (1 << 45),
                 0x6180000100000000 | (1 << 42), 0x4650000100000000 | (1 << 46)):
      with self.subTest(word=word):
        with self.assertRaisesRegex(ValueError, 'unsupported'): execute(instruction(word) + END, {0: 1})

  def test_float_multiply(self):
    raw = instruction(0x4070000200010000) + END
    for a, b, expected in ((0x3fc00000, 0x40000000, 0x40400000), (0xc0000000, 0x40400000, 0xc0c00000),
                            (0x80000000, 0x40000000, 0x80000000), (0x3f800001, 0x3f800001, 0x3f800002)):
      with self.subTest(a=a, b=b): self.assertEqual(execute(raw, {0: a, 1: b})[2], expected)
    self.assertEqual(execute(raw, {0: 0x7f7fffff, 1: 0x40000000})[2], 0x7f800000)

  def test_unsigned_compare_sy(self):
    load = instruction(0xc006000401800001)
    compare = instruction(0x5290400200030004)
    convert = instruction(0x2009400500000002)
    self.assertEqual(execute(load + compare + convert + END, {0: 0x1000, 1: 0, 3: 9},
                             memory={0x1000: bytearray(b'\x07\0\0\0')})[5], 1)
    with self.assertRaisesRegex(ValueError, 'needs sy/ss'):
      execute(load + instruction(0x4290400200030004) + convert + END, {0: 0x1000, 1: 0, 3: 9},
              memory={0x1000: bytearray(b'\x07\0\0\0')})

  def test_float_source_modifiers_and_lookup(self):
    for modifier, expected in ((0, 0xc0000000), (0x4000, 0x40000000), (0x8000, 0x40000000), (0xc000, 0xc0000000)):
      with self.subTest(modifier=modifier):
        self.assertEqual(execute(instruction(0x40d0000200000000 | modifier) + END, {0: 0xc0000000})[2], expected)
    self.assertEqual(execute(instruction(0x4010000200014000) + END, {0: 0x40000000, 1: 0x3f800000})[2], 0xbf800000)
    for code, expected in ((0, 0), (1, 0x3f000000), (2, 0x3f800000), (3, 0x40000000), (11, 0x40800000)):
      with self.subTest(code=code):
        self.assertEqual(execute(instruction(0x4010000228000000 | (code << 16)) + END, {0: 0})[2], expected)
    for code in (0x2804, 0x2c00, 0x0800):
      with self.subTest(code=code), self.assertRaisesRegex(ValueError, 'unsupported'):
        execute(instruction(0x4010000200000000 | (code << 16)) + END, {0: 0})
    with self.assertRaisesRegex(ValueError, 'unsupported'):
      execute(instruction(0x4210000240000000) + END, {0: 1})

  def test_float_maximum(self):
    raw = instruction(0x4050000200010000) + END
    for a, b, expected in ((0xbf800000, 0, 0), (0x40000000, 0x3f800000, 0x40000000),
                            (0x80000000, 0, 0), (0, 0x80000000, 0), (0x80000000, 0x80000000, 0x80000000)):
      with self.subTest(a=a, b=b): self.assertEqual(execute(raw, {0: a, 1: b})[2], expected)
    for a, b, expected in ((0xff800000, 0x3f800000, 0x3f800000), (0, 0x7f800000, 0x7f800000),
                            (0xff800000, 0xff800000, 0xff800000), (0x7f800000, 0xff800000, 0x7f800000)):
      with self.subTest(a=a, b=b): self.assertEqual(execute(raw, {0: a, 1: b})[2], expected)
    for value in (0x7fc00000, 0xffc00000): self.assertEqual(execute(raw, {0: value, 1: 0})[2], 0)
    for value in (1, 0x80000001):
      with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'non-normal'):
        execute(raw, {0: value, 1: 0})

  def test_integer_maximum(self):
    for a, b, unsigned, signed in ((0xffffffff, 0, 0xffffffff, 0), (0x80000000, 0x7fffffff, 0x80000000, 0x7fffffff),
                                   (5, 5, 5, 5), (0x80000000, 0xffffffff, 0xffffffff, 0xffffffff)):
      with self.subTest(a=a, b=b):
        self.assertEqual(execute(instruction(0x4310000200010000) + END, {0: a, 1: b})[2], unsigned)
        self.assertEqual(execute(instruction(0x4330000200010000) + END, {0: a, 1: b})[2], signed)

  def test_store_sy_publishes_global_load(self):
    memory = {0x1000: bytearray(b'\x07\0\0\0'), 0x2000: bytearray(4)}
    raw = instruction(0xc006000401800001) + instruction(0xd0c6050001800008) + END
    execute(raw, {0: 0x1000, 1: 0, 2: 0x2000, 3: 0}, memory=memory)
    self.assertEqual(memory[0x2000], b'\x07\0\0\0')

  def test_float_compare_repeat(self):
    raw = instruction(0x40b84b0228000000) + b''.join(instruction(0x2009400800000000 | (i << 32) | (2+i)) for i in range(4)) + END
    regs = execute(raw, {0: 0xbf800000, 1: 0, 2: 0x3f800000, 3: 0x80000000})
    self.assertEqual([regs[i] for i in range(8, 12)], [1, 0, 0, 0])
    floating = instruction(0x40b84b0228000000) + instruction(0x20084b0800000002) + END
    self.assertEqual([execute(floating, {0: 0xbf800000, 1: 0, 2: 0x3f800000, 3: 0x80000000})[i]
                      for i in range(8, 12)], [0x3f800000, 0, 0, 0])

if __name__ == '__main__': unittest.main()
