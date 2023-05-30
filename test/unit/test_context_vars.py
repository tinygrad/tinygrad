import unittest
from tinygrad.helpers import Context, ContextVar
from tinygrad.tensor import Tensor
from tinygrad.lazy import Device, DEFAULT_DEVICE

class TestContextStack(unittest.TestCase):
  def test_default_device(self):
    assert Tensor.empty(1).device == Device.DEFAULT
    assert Tensor.ones(1).device == Device.DEFAULT
    with Context(DEVICE="CPU"):
      assert Tensor.empty(1).device == "CPU"
      assert Tensor.ones(1).device == "CPU"
      DEFAULT_DEVICE("gpu")
      assert Tensor.empty(1).device == "GPU"
      assert Tensor.ones(1).device == "GPU"

if __name__ == "__main__":
  unittest.main()