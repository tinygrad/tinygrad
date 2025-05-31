import unittest
from tinygrad import Tensor
from tinygrad.renderer.graph import GraphRenderer
from tinygrad.nn.state import get_state_dict

class TestGraphRenderer(unittest.TestCase):
  def test_capture_unmutated_implicit_input(self):

    class Model:
      def __init__(self):
        self.w = Tensor([42])

      def mutate_implicit_input(self, x:Tensor):
        self.w = self.w + x

    model = Model()
    r = GraphRenderer(model.mutate_implicit_input, Tensor([7]), tensor_names=get_state_dict(model))

    self.assertIn("w", r.state_dict)
    self.assertEqual(r.state_dict["w"].tolist(), [42])
    self.assertEqual(model.w.lazydata.base.is_realized, False)
    self.assertEqual(model.w.tolist(), [49])

if __name__ == "__main__":
  unittest.main()
