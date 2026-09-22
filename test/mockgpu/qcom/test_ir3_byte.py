import unittest
from test.mockgpu.qcom.ir3 import execute
from test.mockgpu.qcom.test_ir3_convert import code, END

LOAD = 0xc00c000401800001
WIDEN = 0x3008c00600000004
STORE = 0xc0cc050001800008

class TestIR3Byte(unittest.TestCase):
  def test_unaligned_byte_and_guards(self):
    for value in (0, 127, 128, 255):
      with self.subTest(value=value):
        memory = {0x1000: bytearray([99, value, 88]), 0x2000: bytearray([77, 66, 55])}
        regs = execute(code(LOAD, WIDEN, STORE, END), {0: 0x1001, 1: 0, 2: 0x2001, 3: 0}, memory=memory)
        self.assertEqual(regs[6], value)
        self.assertEqual(memory, {0x1000: bytes([99, value, 88]), 0x2000: bytes([77, value, 55])})

  def test_store_low_byte(self):
    memory = {0x2001: bytearray([0])}
    execute(code(0x46f0400420000000, STORE, END), {0: 0x1234abcd, 2: 0x2001, 3: 0}, memory=memory)
    self.assertEqual(memory[0x2001], b'\xcd')

  def test_pending_half_hazards(self):
    for tail in ((END,), (WIDEN & ~(1 << 60), END), (STORE, END), (LOAD, END),
                 (0x46f0400420000002, END), (0x4290400400020002, END),
                 (1 << 44, WIDEN & ~(1 << 60), END)):
      with self.subTest(tail=tail), self.assertRaisesRegex(ValueError, 'need.*sy'):
        execute(code(LOAD, *tail), {0: 0x1000, 1: 0, 2: 0x1000, 3: 0}, memory={0x1000: bytearray([255])})

  def test_vector_and_rollback(self):
    load = (LOAD & ~(7 << 24)) | (4 << 24)
    store = (STORE & ~(7 << 24)) | (4 << 24) | (1 << 60)
    regs = {0: 0x1001, 1: 0, 2: 0x2001, 3: 0}
    memory = {0x1001: bytearray([1, 128, 255, 0]), 0x2001: bytearray([9]*4)}
    with self.assertRaisesRegex(ValueError, 'unsupported'):
      execute(code(load, store, 0xe000000000000000, END), regs, memory=memory)
    self.assertEqual(memory[0x2001], bytes([9]*4))
    execute(code(load, store, END), regs, memory=memory)
    self.assertEqual(memory[0x2001], memory[0x1001])
    with self.assertRaisesRegex(ValueError, 'unmapped'):
      execute(code(load, WIDEN, END), {0: 0x1002, 1: 0}, memory=memory)
    with self.assertRaisesRegex(ValueError, 'reserved'):
      execute(code((load & ~(255 << 32)) | (242 << 32), END), regs, memory=memory)

if __name__ == '__main__': unittest.main()
