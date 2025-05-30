import unittest
from tinygrad import Tensor, Device, TinyJit
from tinygrad.nn.state import get_state_dict

class Model:
  def __init__(self):
    self.w = Tensor([0])

  def mutate_implicit_input(self, x:Tensor):
    self.w = self.w + x

@unittest.skipUnless(Device.DEFAULT == "WEBGPU", "only used for WebGPU export currently")
class TestExportWebGPU(unittest.TestCase):
  def test_export_unmutated_implicit_input(self):
    model = Model()
    _, state_dict = TinyJit(model.mutate_implicit_input).export_webgpu(Tensor([7]), tensor_names=get_state_dict(model))

    name_t_u_b = "\n"
    # debugging test failure in CI that doesn't reproduce locally
    for name, t in state_dict.items():
      name_t_u_b += str(name) + "\n"
      name_t_u_b += str(t) + "\n"
      name_t_u_b += str(t.lazydata) + "\n"
      name_t_u_b += str(t.lazydata.base.realized) + "\n\n"
    self.assertEqual(len(state_dict), 1, name_t_u_b)
    self.assertEqual(list(state_dict.values())[0].tolist(), [0])
    self.assertEqual(model.w.tolist(), [7])

if __name__ == "__main__":
  unittest.main()
