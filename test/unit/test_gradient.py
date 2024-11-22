import unittest
import jax
from tinygrad.ops import UOp
from tinygrad.gradient import gradient

class TestGradient(unittest.TestCase):
  def test_add(self):
    def f(x,y): return x+y
    UOp.variable()



