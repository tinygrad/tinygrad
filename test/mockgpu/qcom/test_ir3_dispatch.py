import struct, unittest
from test.mockgpu.qcom.ir3 import execute

END = bytes.fromhex('0000000000000003')
def instruction(word): return struct.pack('<Q', word)

class TestIR3Dispatch(unittest.TestCase):
  def test_shift_values(self):
    for base, value, shift, expected in ((0x46d0000020000000, 0x80000001, 1, 2),
                                         (0x46d0000020000000, 1, 31, 0x80000000),
                                         (0x4710000020000000, 0x80000000, 31, 0xffffffff),
                                         (0x4710000020000000, 0x80000000, 1, 0xc0000000),
                                         (0x4710000020000000, 0x7fffffff, 31, 0),
                                         (0x46d0000020000000, 0xffffffff, 0, 0xffffffff)):
      with self.subTest(base=base, value=value, shift=shift):
        self.assertEqual(execute(instruction(base | (shift << 16)) + END, {0: value})[0], expected)
    for shift in range(32):
      with self.subTest(shift=shift):
        self.assertEqual(execute(instruction(0x46d0000020000000 | (shift << 16)) + END, {0: 1})[0], 1 << shift)

  def test_shrg_alias_and_or(self):
    raw = bytes.fromhex('1e30040004840165')  # shrg r1.x, 30, r0.w, r1.x
    self.assertEqual(execute(raw + END, {3: 0x80000000, 4: 0x80000000})[4], 0x80000002)
    self.assertEqual(execute(raw + END, {3: 0xffffffff, 4: 4})[4], 7)

  def test_unsigned_compare_and_half_conversion(self):
    raw = bytes.fromhex('0500021000409042 000000000b400920')  # cmps.u.lt hr0.x,r1.y,c0.z; cov.u16s32 r2.w,hr0.x
    for a, b, expected in ((0, 1, 1), (1, 1, 0), (0xffffffff, 0, 0), (0x7fffffff, 0x80000000, 1)):
      with self.subTest(a=a, b=b):
        result = execute(raw + END, {0: 99, 5: a}, {2: b})
        self.assertEqual((result[0], result[11]), (99, expected))
    with self.assertRaisesRegex(ValueError, 'uninitialized half register'):
      execute(raw[8:] + END, {})

  def test_constant_add_indices_and_repeat(self):
    for index in (0, 255, 256, 1023, 2047):
      with self.subTest(index=index):
        raw = instruction(0x4210000200011000 | index)
        self.assertEqual(execute(raw + END, {1: 2}, {index: 0xffffffff})[2], 1)
    # Four destinations and independently incremented source ranges; literal sums include carry.
    raw = bytes.fromhex('040008000c0b1842')
    regs = {4: 0, 5: 1, 6: 0xffffffff, 7: 0x80000000, 8: 0, 9: 2, 10: 1, 11: 0x80000000}
    result = execute(raw + END, regs)
    self.assertEqual([result[index] for index in range(12, 16)], [0, 3, 0, 0])
    self.assertNotIn(12, regs)

  def test_vector_memory_counts_and_pending(self):
    for count in range(1, 5):
      with self.subTest(count=count):
        load = instruction(0xc006000a00800001 | (count << 24))
        sync = instruction(0x52100014000a000a)
        store = instruction(0xc0c60b0000800014 | (count << 24))
        data = bytes.fromhex('01000000 02000000 03000000 04000000')[:count*4]
        memory = {0x1000: bytearray(data), 0x2000: bytearray(count*4)}
        execute(load + sync + store + END, {0: 0x1000, 1: 0, 5: 0x2000, 6: 0}, memory=memory)
        self.assertEqual(memory[0x2000], data)
        with self.assertRaisesRegex(ValueError, 'need sy'): execute(load + END, {0: 0x1000, 1: 0}, memory=memory)
        with self.assertRaisesRegex(ValueError, 'unmapped range'):
          execute(load + END, {0: 0x1000, 1: 0}, memory={0x1000: bytearray(count*4-1)})

  def test_new_encoding_rejections(self):
    forms = ((0x46d0000020000000, set(range(8)) | set(range(16, 21)) | set(range(32, 40)) | {43, 44, 51, 54, 55}),
             (0x4710000020000000, set(range(8)) | set(range(16, 21)) | set(range(32, 40)) | {43, 44, 51, 58}),
             (0x6500040000003000, set(range(11)) | {12, 55, 60} | set(range(16, 24)) | set(range(32, 40)) | set(range(47, 55)) | {15, 43}),
             (0x2009400000000000, set(range(8)) | set(range(32, 42)) | {43, 44, 48, 50, 60}))
    for word, variable in forms:
      for bit in set(range(64)) - variable:
        with self.subTest(word=word, bit=bit):
          with self.assertRaisesRegex(ValueError, 'pc=0x0.*(unsupported|uninitialized half register)'):
            execute(instruction(word ^ (1 << bit)) + END, {0: 1}, {0: 1})
    for count in (5, 6, 7):
      with self.subTest(count=count):
        with self.assertRaisesRegex(ValueError, 'unsupported memory count'):
          execute(instruction(0xc006000a00800001 | (count << 24)) + END, {})

  def test_vector_register_span(self):
    raw = instruction(0xc00600f204800001)
    with self.assertRaisesRegex(ValueError, 'special/reserved'):
      execute(raw + END, {0: 0x1000, 1: 0}, memory={0x1000: bytearray(16)})
    with self.assertRaisesRegex(ValueError, 'special/reserved'):
      execute(instruction(0x421003f300000000) + END, {0: 1})

  def test_128_byte_alignment(self):
    self.assertEqual(execute(END + bytes(120), {}, padded=True), {})
    with self.assertRaisesRegex(ValueError, 'trailing'): execute(END + bytes(376), {}, padded=True)

if __name__ == '__main__': unittest.main()
