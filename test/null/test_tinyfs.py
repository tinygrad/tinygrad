import unittest
from tinygrad import Tensor
from tinygrad.nn.state import fs_store, fs_load

class TestLoadStore(unittest.TestCase):
  def test_load_shape(self):
    t = fs_load(Tensor(bytes(16)), 1024)
    assert t.shape == (1024,), t.shape
    t.schedule_linear()

  def test_store_shape(self):
    t = fs_store(Tensor.zeros(1024))
    assert t.shape == (16,), t.shape
    t.schedule_linear()

  def test_load_large_shape(self):
    t = fs_load(Tensor(bytes(16)), 10_000_000)
    assert t.shape == (10_000_000,), t.shape
    t.schedule_linear()

  def test_store_large_shape(self):
    t = fs_store(Tensor.zeros(10_000_000))
    assert t.shape == (16,), t.shape
    t.schedule_linear()

if __name__ == "__main__":
  unittest.main()
