import unittest
from tinygrad import Tensor, Device, TinyJit

class Model:
  def __init__(self):
    self.w = Tensor([0])

  def mutate_implicit_input(self, x:Tensor):
    self.w = self.w + x

@unittest.skipUnless(Device.DEFAULT == "WEBGPU", "only used for WebGPU export currently")
class TestExportWebGPU(unittest.TestCase):
  def test_export_unmutated_implicit_input(self):
    model = Model()
    _, state_dict = TinyJit(model.mutate_implicit_input).export_webgpu(Tensor([7]))
    self.assertEqual(len(state_dict), 1)
    self.assertEqual(list(state_dict.values())[0].tolist(), [0])
    self.assertEqual(model.w.tolist(), [7])

if __name__ == "__main__":
  unittest.main()
