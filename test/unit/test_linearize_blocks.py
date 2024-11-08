import unittest, random
from tinygrad.codegen.linearize import order_blocks

# the tuples are (<inside loop>, <after loop>)

class TestLinearizeBlocks(unittest.TestCase):
  def _order_blocks(self, blocks):
    rng_blocks = blocks[:]
    random.shuffle(rng_blocks)
    self.assertListEqual(order_blocks(rng_blocks), blocks)

  def test_simple(self):
    self._order_blocks([
      ((), ()),
      ((4,), ()),
      ((), (4,))])

  def test_two_loops(self):
    self._order_blocks([
      ((), ()),
      ((4,), ()),
      ((5,), ()),
      ((4, 5), ()),
      ((), (4, 5))])

  def test_three_loops(self):
    self._order_blocks([
      ((), ()),
      ((5,), ()),
      ((6,), ()),
      ((5, 6), ()),
      ((7,), ()),
      ((6, 7), ()),
      ((5, 6, 7), ()),
      ((), (5, 6, 7))])

  def test_non_nested_loops(self):
    self._order_blocks([
      ((), ()),
      ((2,), ()),
      ((), (2,)),
      ((1001,), ()),
      ((1001,), (2,)),
      ((), (2, 1001))])

  def test_non_nested_two_loops(self):
    self._order_blocks([
      ((), ()),
      ((3,), ()),
      ((4,), ()),
      ((3, 4), ()),
      ((), (3, 4)),
      ((1002,), ()),
      ((1002,), (3, 4)),
      ((), (3, 4, 1002))])

  def test_nested_final_loop(self):
    self._order_blocks([
      ((), ()),
      ((2,), ()),
      ((2, 1001), ()),
      ((2,), (1001,)),
      ((), (2, 1001))])

  def test_complex_nesting(self):
    self._order_blocks([
      ((), ()),
      ((5,), ()),
      ((), (5,)),
      ((1003,), ()),
      ((1003,), (5,)),
      ((), (5, 1003)),
      ((4,), ()),
      ((4,), (5, 1003)),
      ((), (4, 5, 1003)),
      ((1002,), ()),
      ((1002,), (4, 5, 1003)),
      ((), (4, 5, 1002, 1003))])

if __name__ == '__main__':
  unittest.main()