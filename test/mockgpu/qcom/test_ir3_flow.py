import unittest
from test.mockgpu.qcom.ir3 import execute, execute_group
from test.mockgpu.qcom.test_ir3_convert import code, END

ADD = 0x4210000020010000
GE = 0x42b308f820030000
BR = 0x0080000000000002
BACK = 0x01000000fffffffd

class TestIR3Flow(unittest.TestCase):
  def test_loop_and_forward_branch(self):
    self.assertEqual(execute(code(ADD, GE, BR, BACK, END), {0: 0})[0], 3)
    self.assertEqual([r[0] for r in execute_group(code(ADD, GE, BR, BACK, END), [{0: 0}]*2, {}, {}, 0)], [3,3])
    self.assertEqual(execute(code(0x0100000000000002, 0xffffffffffffffff, END), {0: 7})[0], 7)

  def test_divergence_rejected(self):
    with self.assertRaisesRegex(ValueError, 'divergent.*branch'):
      execute_group(code(ADD, GE, BR, BACK, END), [{0: 0}, {0: 2}], {}, {}, 0)

  def test_loop_failure_rolls_back_stores(self):
    memory = {0x1000: bytearray.fromhex('aaaaaaaa')}
    with self.assertRaisesRegex(ValueError, 'step limit'):
      execute(code(0xc0c6050001800000, 0x01000000ffffffff, END), {0: 7, 2: 0x1000, 3: 0}, memory=memory)
    self.assertEqual(memory[0x1000], bytes.fromhex('aaaaaaaa'))

  def test_invalid_targets_predicate_and_budget(self):
    for jump in (0x01000000ffffffff, 0x0100000000000002):
      with self.subTest(jump=jump), self.assertRaisesRegex(ValueError, 'branch target'):
        execute(code(jump, END), {})
    with self.assertRaisesRegex(ValueError, 'uninitialized predicate'):
      execute(code(BR, 0, END), {})
    with self.assertRaisesRegex(ValueError, 'step limit'):
      execute(code(0x0100000000000000, END), {})

if __name__ == '__main__': unittest.main()
