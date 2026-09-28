import unittest
from tinygrad import Tensor
from tinygrad.engine.realize import compile_linear

class TestCompileFailures(unittest.TestCase):
  def compile(self, out:Tensor):
    compile_linear(out.schedule_linear())

  def test_interpolate_atari(self):
    self.compile(Tensor.empty(210, 160, dtype='uint8').interpolate((64, 64)))

  def test_add_max_uchar(self):
    self.compile((Tensor.empty(1024, dtype='uint8') + Tensor.empty(1024, dtype='uint8')).max())

if __name__ == '__main__':
  unittest.main()
