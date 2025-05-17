import unittest
from tinygrad.helpers import getbits

class TestGetBits(unittest.TestCase):
  def test_low_bits(self):
    self.assertEqual(getbits(0b11010110, 0, 3), 0b0110)

  def test_high_bits(self):
    self.assertEqual(getbits(0b11010110, 4, 7), 0b1101)

  def test_middle_bits(self):
    self.assertEqual(getbits(0b11010110, 3, 5), 0b010)

  def test_full_range(self):
    self.assertEqual(getbits(0b11010110, 0, 7), 0b11010110)

  def test_single_bit(self):
    self.assertEqual(getbits(0b100000000, 8, 8), 1)

if __name__ == "__main__":
  unittest.main()
