import unittest
from test.mockgpu.qcom.ir3 import execute
from test.mockgpu.qcom.test_ir3_convert import code, END

MAD = 0x6380000300020000 | (1 << 47)

class TestIR3Mad(unittest.TestCase):
  def test_unfused_rounding(self):
    # (1+2^-23)*(1-2^-23) rounds to 1 before subtracting 1 on A6xx.
    self.assertEqual(execute(code(MAD, END), {0: 0x3f800001, 1: 0x3f7ffffe, 2: 0xbf800000})[3], 0)
    self.assertEqual(execute(code(MAD, END), {0: 0x40000000, 1: 0x40400000, 2: 0x40800000})[3], 0x41200000)

  def test_negation_constants_and_delays(self):
    for flags, expected in ((0, 0x41200000), (1 << 14, 0xc0000000), (1 << 30, 0xc0000000), (1 << 31, 0x40000000)):
      for delay in (0, 1 << 43, 1 << 15, (1 << 43) | (1 << 15)):
        with self.subTest(flags=flags, delay=delay):
          word = MAD | 0x1000 | (0x1000 << 16) | flags | delay
          self.assertEqual(execute(code(word, END), {1: 0x40400000}, {0: 0x40000000, 2: 0x40800000})[3], expected)

  def test_repeat(self):
    word = 0x6380000c00080000 | (4 << 47) | (3 << 40) | (1 << 43) | (1 << 15) | (1 << 29)
    regs = dict(enumerate([0x3f800000, 0x40000000, 0x40400000, 0x40800000]*3))
    result = execute(code(word, END), regs)
    self.assertEqual([result[i] for i in range(12,16)], [0x40000000, 0x40c00000, 0x41400000, 0x41a00000])

  def test_sy_and_unsupported(self):
    load = 0xc006000001800001 | (4 << 14)
    regs = {1: 0x40000000, 2: 0x3f800000, 4: 0x1000, 5: 0}
    memory = {0x1000: bytearray.fromhex('00004040')}
    self.assertEqual(execute(code(load, MAD | (1 << 60), END), regs, memory=memory)[3], 0x40e00000)
    with self.assertRaisesRegex(ValueError, 'needs sy'):
      execute(code(load, MAD, END), regs, memory=memory)
    for bit in (13, 42, 45, 46, 59):
      with self.subTest(bit=bit), self.assertRaisesRegex(ValueError, 'unsupported'):
        execute(code(MAD | (1 << bit), END), {0: 0, 1: 0, 2: 0})

if __name__ == '__main__': unittest.main()
