import unittest

class Test(unittest.TestCase):
  def test(self):
    a = [bytearray(1024*1024*1024) for _ in range(100)]
