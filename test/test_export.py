import unittest
from tinygrad import Tensor, Device, TinyJit
from tinygrad.nn.state import get_parameters

class Model:
  def __init__(self):
    self.weight1 = Tensor([1,2,3])

  def mutate_weight(self, x:Tensor):
    self.weight1 = self.weight1 + x
    return self.weight1

@unittest.skipUnless(Device.DEFAULT == "WEBGPU", "only used for WebGPU export currently")
class TestExportWebGPU(unittest.TestCase):
  def test_export_mutate_weight(self):
    model = Model()
    for t in get_parameters(model): t.realize()
    _, state_dict = TinyJit(model.mutate_weight).export_webgpu(Tensor([7]))
    assert len(state_dict) == 1 and list(state_dict.values())[0].tolist() == [1,2,3]
