import unittest
from tinygrad.tensor import Tensor
from tinygrad.helpers import Context
from test.test_schedule import check_schedule

class TestTranscendentalSchedule(unittest.TestCase):
  # w/ payne_hanek_reduction (fp32)
  def test_transcendental_sin_fusion(self):
    with Context(TRANSCENDENTAL=2):
      a = Tensor.empty(10)
      b = Tensor.empty(10)
      c = a.sin() + b.sin()
      c = c.sin()
      check_schedule(c, 1)

  def test_transcendental_log2_fusion(self):
    with Context(TRANSCENDENTAL=2):
      a = Tensor.empty(10)
      b = Tensor.empty(10)
      c = a.log2() + b.log2()
      c = c.log2()
      check_schedule(c, 1)

  def test_transcendental_exp2_fusion(self):
    with Context(TRANSCENDENTAL=2):
      a = Tensor.empty(10)
      b = Tensor.empty(10)
      c = a.exp2() + b.exp2()
      c = c.exp2()
      check_schedule(c, 1)

