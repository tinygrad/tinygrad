import struct, unittest
from test.mockgpu.qcom.ir3 import execute

END = 0x0300000000000000
def code(*words): return struct.pack(f'<{len(words)}Q', *words)
def convert(src, dst, flags=0): return 0x2000000100000000 | (src << 50) | (dst << 46) | flags

class TestIR3Convert(unittest.TestCase):
  def test_shr_half_destination(self):
    for count in (0, 1, 8, 16, 31):
      for value in (0, 0xffffffff, 0x80000000, 0x12345678):
        with self.subTest(count=count, value=value):
          shift = 0x46f0400220000000 | (count << 16)
          self.assertEqual(execute(code(shift, 0x2008c00300000002, END), {0: value})[3], (value >> count) & 65535)

  def test_shl_multisrc_immediate(self):
    self.assertEqual(execute(code(0x46d0002800252001, END), {37: 7})[40], 14)

  def test_multisrc_shift_direction_and_count(self):
    for count, expected in ((0, (0x80000001, 0x80000001, 0x80000001)), (1, (2, 0x40000000, 0xc0000000)),
                            (31, (0x80000000, 1, 0xffffffff)), (32, (0x80000001, 0x80000001, 0x80000001)),
                            (33, (2, 0x40000000, 0xc0000000))):
      for base, result in zip((0x46d0000000000000, 0x46f0000000000000, 0x4710000000000000), expected):
        with self.subTest(base=base, count=count):
          self.assertEqual(execute(code(base | (8 << 32) | (2 << 16) | 1, END), {1: count, 2: 0x80000001})[8], result)

  def test_shr_half_sync_and_delays(self):
    load, shift, widen = 0xc006000401800001, 0x46f0400220080004, 0x2008c00300000002
    for delay in (0, 1 << 43, 1 << 51, (1 << 43) | (1 << 51)):
      with self.subTest(delay=delay):
        memory = {0x1000: bytearray.fromhex('78563412')}
        self.assertEqual(execute(code(load, shift | delay | (1 << 60), widen, END), {0: 0x1000, 1: 0}, memory=memory)[3], 0x3456)
        with self.assertRaisesRegex(ValueError, 'needs sy/ss'):
          execute(code(load, shift | delay, widen, END), {0: 0x1000, 1: 0}, memory=memory)

  def test_shr_half_rejects_unsupported_forms(self):
    shift = 0x46f0400220080000
    for bits in (1 << 40, 1 << 41, 1 << 45, 1 << 21, 1 << 59):
      with self.subTest(bits=bits), self.assertRaisesRegex(ValueError, 'unsupported'):
        execute(code(shift | bits, END), {0: 1})
    with self.assertRaisesRegex(ValueError, 'reserved'):
      execute(code((shift & ~(255 << 32)) | (244 << 32), END), {0: 1})

  def test_float_to_integer_truncates(self):
    for src, expected in ((0xbf666666, 0), (0xbe99999a, 0), (0x3f99999a, 1), (0x4effffff, 0x7fffff80),
                          (0xcf000000, 0x80000000), (0xbfc00000, 0xffffffff)):
      with self.subTest(src=src): self.assertEqual(execute(code(convert(1, 5), END), {0: src})[1], expected)
    for src, expected in ((0xbf666666, 0), (0x3f99999a, 1), (0x4f7fffff, 0xffffff00)):
      with self.subTest(src=src): self.assertEqual(execute(code(convert(1, 3), END), {0: src})[1], expected)

  def test_u32_to_f16_and_back(self):
    narrow = convert(3, 0, (1 << 60))
    widen = (convert(0, 1) & ~(255 << 32)) | (2 << 32) | 1
    for value, expected in ((0, 0), (1, 0x3f800000), (2, 0x40000000), (65504, 0x477fe000)):
      with self.subTest(value=value): self.assertEqual(execute(code(narrow, widen, END), {0: value})[2], expected)
    with self.assertRaisesRegex(ValueError, 'subnormal half'):
      execute(code(convert(3, 2), widen, END), {0: 1})

  def test_integer_to_float_rounding(self):
    for dtype, src, expected in ((5, 0xffffffff, 0xbf800000), (5, 0x80000000, 0xcf000000),
                                 (3, 0xffffffff, 0x4f800000), (3, 16777217, 0x4b800000), (3, 16777219, 0x4b800002)):
      with self.subTest(dtype=dtype, src=src): self.assertEqual(execute(code(convert(dtype, 1), END), {0: src})[1], expected)

  def test_integer_cast_and_float_move_bits(self):
    for src_type, dst_type in ((3, 5), (5, 3), (1, 1)):
      with self.subTest(src_type=src_type, dst_type=dst_type):
        self.assertEqual(execute(code(convert(src_type, dst_type), END), {0: 0xffc12345})[1], 0xffc12345)

  def test_conversion_rejects_undefined_and_modifiers(self):
    for dst, value in ((5, 0x4f000000), (3, 0x4f800000), (3, 0xbf800000), (5, 0x7f800000), (5, 1), (5, 0x7fc00000)):
      with self.subTest(dst=dst, value=value), self.assertRaisesRegex(ValueError, 'unsupported'):
        execute(code(convert(1, dst), END), {0: value})
    for bit in (42, 45, 55, 56, 59):
      with self.subTest(bit=bit), self.assertRaisesRegex(ValueError, 'unsupported'):
        execute(code(convert(5, 1, 1 << bit), END), {0: 1})

  def test_sy_repeat_conversion(self):
    memory = {0x1000: bytearray(struct.pack('<4i', -1, 0, 1, 2))}
    load = 0xc006000804800001
    cov = (convert(5, 1, (1 << 60) | (3 << 40) | (1 << 43)) & ~(255 << 32)) | (12 << 32) | 8
    regs = execute(code(load, cov, END), {0: 0x1000, 1: 0}, memory=memory)
    self.assertEqual([regs[i] for i in range(12, 16)], [0xbf800000, 0, 0x3f800000, 0x40000000])
    with self.assertRaisesRegex(ValueError, 'needs sy/ss'):
      execute(code(load, cov & ~(1 << 60), END), {0: 0x1000, 1: 0}, memory=memory)

  def test_comparison_conditions(self):
    for base, regs, results in ((0x42b0400000000000, {0: 0xffffffff, 1: 1}, (1, 1, 0, 0, 0, 1)),
                               (0x4290400000000000, {0: 0xffffffff, 1: 1}, (0, 0, 1, 1, 0, 1)),
                               (0x40b0400000000000, {0: 0x80000000, 1: 0}, (0, 1, 0, 1, 1, 0))):
      for condition, expected in enumerate(results):
        with self.subTest(base=base, condition=condition):
          raw = code(base | (condition << 48) | (2 << 32) | (1 << 16), 0x2009400300000002, END)
          self.assertEqual(execute(raw, regs)[3], expected)
    for condition in (6, 7):
      with self.subTest(condition=condition), self.assertRaisesRegex(ValueError, 'unsupported'):
        execute(code(0x42b0400200010000 | (condition << 48), END), {0: 1, 1: 2})

  def test_shrg_nop_delays(self):
    for delay in (0, 1 << 43, 1 << 15, (1 << 43) | (1 << 15)):
      with self.subTest(delay=delay):
        raw = 0x650004040002301e | (1 << 47) | delay
        self.assertEqual(execute(code(raw, END), {1: 0x80000000, 2: 1})[4], 3)
        load = 0xc006000101800001
        with self.assertRaisesRegex(ValueError, 'needs sy/ss'):
          execute(code(load, raw, END), {0: 0x1000, 1: 0, 2: 1}, memory={0x1000: bytearray(4)})

if __name__ == '__main__': unittest.main()
