import struct, unittest
from test.mockgpu.qcom.ir3 import execute
from test.mockgpu.qcom.test_ir3_convert import code, END

SS = 1 << 44

class TestIR3Math(unittest.TestCase):
  def test_sin_sfu(self):
    for value, expected in ((0x00000000, 0x00000000), (0x3fc90fdb, 0x3f800000), (0xbfc90fdb, 0xbf800000)):
      with self.subTest(value=value):
        raw = struct.pack('<QQQ', 0x8090000000000000 | (2 << 32) | 1, SS, 0x0300000000000000)
        self.assertEqual(execute(raw, {0: 0, 1: value})[2], expected)

  def test_select_bits(self):
    for condition, expected in ((0, 0x80000000), (1, 0x7fc00123), (0x80000000, 0x7fc00123)):
      with self.subTest(condition=condition):
        self.assertEqual(execute(code(0x6480800300020000, END), {0: 0x7fc00123, 1: condition, 2: 0x80000000})[3], expected)
    with self.assertRaisesRegex(ValueError, 'unsupported'):
      execute(code(0x6480800300024000, END), {0: 1, 1: 1, 2: 2})

  def test_select_repeat_and_sync(self):
    word = 0x6480000c00080000 | (4 << 47) | (3 << 40) | (1 << 43) | (1 << 15) | (1 << 29)
    regs = dict(enumerate([10,11,12,13,0,1,0,1,20,21,22,23]))
    result = execute(code(word, END), regs)
    self.assertEqual([result[i] for i in range(12,16)], [20,11,22,13])
    sfu, select = 0x8010000200000000, 0x6480800300040002
    with self.assertRaisesRegex(ValueError, 'ss'):
      execute(code(sfu, select, END), {0: 0x40000000, 1: 1, 4: 0})
    self.assertEqual(execute(code(sfu, select | SS, END), {0: 0x40000000, 1: 1, 4: 0})[3], 0x3f000000)

  def test_select_condition_banks_and_pending(self):
    narrow, select = 0x200c800100000004, 0x6480800300020000
    for full, half, expected in ((None, 0, 20), (None, 1, 10), (0, 1, 20), (1, 0, 10)):
      with self.subTest(full=full, half=half):
        regs = {0: 10, 2: 20, 4: half} | ({} if full is None else {1: full})
        self.assertEqual(execute(code(narrow, select, END), regs)[3], expected)
    for producer, sync in ((0xc006000101810001, 1 << 60), (0x8010000100000006, SS)):
      for full in (None, 0):
        with self.subTest(producer=producer, full=full):
          regs = {0: 10, 2: 20, 4: 0, 5: 0, 6: 0x40000000} | ({} if full is None else {1: full})
          memory = {0: bytearray.fromhex('01000000')}
          with self.assertRaisesRegex(ValueError, 'pc=0x10.*register 1 needs sy/ss'):
            execute(code(narrow, producer, select, SS, 1 << 60 | 0x200cc00700000000, END), regs, memory=memory)
          self.assertEqual(execute(code(narrow, producer, select | sync, END), regs, memory=memory)[3], 10)

  def test_select_half_fallback_ignores_unrelated_full_bank(self):
    narrow, select = 0x200c800100000004, 0x6480800300020000
    base = {0: 10, 2: 20, 4: 1}
    self.assertEqual(execute(code(narrow, select, END), base)[3], 10)
    with_unrelated_full = base | {99: 0xffffffff}
    self.assertEqual(execute(code(narrow, select, END), with_unrelated_full)[3], 10)

  def test_andg_uses_all_three_sources(self):
    # Mesa ir3-cat3.xml: (src2 & src1) | src3; high mask bits are not a shift count.
    for full in (False, True):
      for mask, value, addend, expected in ((0xf0, 0xff, 0x100, 0x1f0), (0xf0, 0x0f, 0x100, 0x100),
                                            (0x8000, 0xffff, 0, 0x8000), (0xffff, 0x8001, 0x10, 0x8011)):
        with self.subTest(full=full, mask=mask, value=value, addend=addend):
          prefix = () if full else tuple(0x200c800000000000 | (reg << 32) | reg for reg in (15, 16, 19))
          suffix = () if full else (0x2008c00300000010,)
          raw = code(*prefix, 0x6607901000132010 | (int(full) << 42), *suffix, END)
          self.assertEqual(execute(raw, {16: mask, 15: value, 19: addend})[16 if full else 3], expected)

  def test_sfu_failed_submit_preserves_memory(self):
    memory = {0x1000: bytearray.fromhex('12345678')}
    store = 0xc0c6010001800008
    with self.assertRaisesRegex(ValueError, 'float'):
      execute(code(store, 0x8010000300000002, SS, END), {0: 0x1000, 1: 0, 2: 1, 4: 7}, memory=memory)
    self.assertEqual(memory[0x1000], bytearray.fromhex('12345678'))

  def test_float_special_arithmetic(self):
    for word, a, b, expected in ((0x4010000200010000, 0x7f800000, 0x3f800000, 0x7f800000),
                                 (0x4010000200010000, 0x7f800000, 0xff800000, 0x7fc00000),
                                 (0x4070000200010000, 0x7f800000, 0, 0x7fc00000),
                                 (0x4070000200010000, 0xff800000, 0x40000000, 0xff800000)):
      with self.subTest(word=word, a=a, b=b): self.assertEqual(execute(code(word, END), {0: a, 1: b})[2], expected)

  def test_half_and_full_bitwise(self):
    # Convert full inputs into half booleans, then AND into a full register.
    for a, b, expected in ((0,0,1), (0,1,0), (1,0,0), (1,1,0)):
      raw = code(0x4294400200002000, 0x4294400300012000, 0x4380400400030002, END)
      self.assertEqual(execute(raw, {0: a, 1: b})[4], expected)

  def test_sfu_values(self):
    for op, value, expected in ((0, 0x40400000, 0x3eaaaaab), (0, 0xc0000000, 0xbf000000),
                                (1, 0x40800000, 0x3f000000), (3, 0x40000000, 0x40800000),
                                (3, 0x3f000000, 0x3fb504f3), (6, 0x41100000, 0x40400000)):
      with self.subTest(op=op, value=value):
        self.assertEqual(execute(code(0x8010000200000000 | (op << 53), SS, END), {0: value})[2], expected)

  def test_sfu_specials(self):
    for op, value, expected in ((0, 0, 0x7f800000), (0, 0x80000000, 0xff800000), (0, 0xff800000, 0x80000000),
                                (1, 0x80000000, 0xff800000), (1, 0xbf800000, 0x7fc00000),
                                (3, 0xff800000, 0), (3, 0x7f800000, 0x7f800000), (3, 0x43000000, 0x7f800000),
                                (6, 0x80000000, 0x80000000), (6, 0xbf800000, 0x7fc00000), (6, 0x7fc00001, 0x7fc00000)):
      with self.subTest(op=op, value=value):
        self.assertEqual(execute(code(0x8010000200000000 | (op << 53), SS, END), {0: value})[2], expected)

  def test_sfu_dependencies(self):
    sfu = 0x8010000200000000
    for consumer in (END, 0x4010000300020001, 0x204cc00200000000, 0x204cc00000000000):
      with self.subTest(consumer=consumer), self.assertRaisesRegex(ValueError, 'ss'):
        execute(code(sfu, consumer) if consumer == END else code(sfu, consumer, END), {0: 0x40000000, 1: 0})
    with self.assertRaisesRegex(ValueError, 'ss'):
      execute(code(sfu, 0x5010000300020001, END), {0: 0x40000000, 1: 0})
    self.assertEqual(execute(code(sfu, 0x4010100300020001, END), {0: 0x40000000, 1: 0})[3], 0x3f000000)
    load = 0xc006000001810001
    memory = {0x1000: bytearray.fromhex('00000040')}
    self.assertEqual(execute(code(load, sfu | (1 << 60), SS, END), {4: 0x1000, 5: 0}, memory=memory)[2], 0x3f000000)

  def test_cat3_alt_ss_releases_sfu_source(self):
    sfu, andg, add = 0x8070000700000007, 0x6607901000132010, 0x4010000700010017
    def narrow(dst): return 0x200c800000000000 | (dst << 32) | dst
    registers = {1: 0x40000000, 7: 0x3f800000, 15: 2, 16: 1, 19: 4, 23: 0x3f800000}
    raw = code(sfu, narrow(16), narrow(15), narrow(19), andg, add, END)
    self.assertEqual(execute(raw, registers)[7], 0x40400000)
    with self.assertRaisesRegex(ValueError, 'register 7 needs sy/ss before overwrite'):
      execute(code(sfu, narrow(16), narrow(15), narrow(19), andg & ~SS, add, END), registers)

  def test_cat3_alt_half_sources(self):
    def narrow(dst, src): return 0x200c800000000000 | (dst << 32) | src
    widen = 0x2008c00300000010
    andg = 0x6607901000132010
    raw = code(narrow(16, 0), narrow(15, 1), narrow(19, 2), andg, widen, END)
    self.assertEqual(execute(raw, {0: 0xf0, 1: 0x0f, 2: 0x100})[3], 0x100)
    with self.assertRaisesRegex(ValueError, 'uninitialized register'):
      execute(code(narrow(16, 0), narrow(15, 1), narrow(19, 2), andg | (1 << 42), END), {0: 0xf0, 1: 0x0f, 2: 0x100})

  def test_repeat_modifiers_and_rejection(self):
    word = 0x80100b0800008000
    result = execute(code(word, SS, END), {0: 0xc0000000, 1: 0xc0800000, 2: 0xc1000000, 3: 0xc1800000})
    self.assertEqual([result[i] for i in range(8,12)], [0x3f000000, 0x3e800000, 0x3e000000, 0x3d800000])
    for bit in (16, 42, 45, 46, 47, 59):
      with self.subTest(bit=bit), self.assertRaisesRegex(ValueError, 'unsupported'):
        execute(code(0x8010000200000000 | (1 << bit), SS, END), {0: 0x40000000})
    with self.assertRaisesRegex(ValueError, 'float'):
      execute(code(0x8010000200000000, SS, END), {0: 1})

  def test_compare_specials_and_flut(self):
    for a, b, expected in ((0x3f800000, 0xff800000, (0,0,1,1,0,1)), (0x7fc00000, 0, (0,0,0,0,0,1)),
                            (0x7f800000, 0x7f800000, (0,1,0,1,1,0))):
      for condition, result in enumerate(expected):
        with self.subTest(a=a, b=b, condition=condition):
          self.assertEqual(execute(code(0x40b0400200010000 | (condition << 48), 0x2009400300000002, END),
                                   {0: a, 1: b})[3], result)
    self.assertEqual(execute(code(0x4070000228080000, END), {0: 0x3f800000})[2], 0x3fb8aa3b)

if __name__ == '__main__': unittest.main()
