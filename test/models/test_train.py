import unittest
import numpy as np
from tinygrad import Device
from tinygrad.nn.state import get_parameters
from tinygrad.nn import optim
from tinygrad.helpers import getenv
from extra.training import train
from extra.models.convnext import ConvNeXt

BS = getenv("BS", 2)

class TestTrain(unittest.TestCase):
  def test_convnext(self):
    model = ConvNeXt(depths=[1], dims=[16])
    X = np.zeros((BS,3,64,64), dtype=np.float32)
    Y = np.zeros((BS), dtype=np.int32)
    optimizer = optim.SGD(get_parameters(model), lr=0.001)
    train(model, X, Y, optimizer, steps=1, BS=BS)
    if Device.DEFAULT == "CL":
      from extra.introspection import print_objects
      assert print_objects() == 0

if __name__ == '__main__':
  unittest.main()
