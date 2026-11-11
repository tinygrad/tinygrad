import unittest
from tinygrad.runtime.support.am.amdev import card_power_limit

class TestAMPowerLimits(unittest.TestCase):
  def test_default(self):
    self.assertEqual(card_power_limit("0000:06:00.0", 130.0, ""), 130.0)
    self.assertEqual(card_power_limit("0000:06:00.0", 0.0, ""), 0.0)
  def test_own(self):
    lim = "0000:03:00.0=139, 0000:06:00.0=156,0000:C4:00.0=100"
    self.assertEqual(card_power_limit("0000:06:00.0", 130.0, lim), 156.0)
    self.assertEqual(card_power_limit("0000:c4:00.0", 130.0, lim), 100.0)
    self.assertEqual(card_power_limit("0000:43:00.0", 130.0, lim), 130.0)

if __name__ == "__main__":
  unittest.main()
