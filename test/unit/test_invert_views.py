import unittest
from tinygrad.shape.view import View, invert_view

class TestInvertViews(unittest.TestCase):
  def test_invert_simple(self):
    v = View.create((2,3,4))
    nv = v.reshape((6,4))
    print(invert_view(nv))

if __name__ == '__main__':
  unittest.main()
