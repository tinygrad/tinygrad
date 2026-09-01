import unittest
from tinygrad import Tensor
from tinygrad.nn.onnx import Domain, OnnxRunner, OpSetId, onnx_ops
from tinygrad.uop.ops import Ops

class TestTinygradOnnxOps(unittest.TestCase):
  def setUp(self):
    self.runner = OnnxRunner.__new__(OnnxRunner)
    self.runner.onnx_ops = onnx_ops

  def test_tinygrad_domain(self):
    self.assertIs(Domain.from_onnx("org.tinygrad"), Domain.TINYGRAD)

  def test_tinygrad_contiguous(self):
    op = self.runner._select_op("TinygradContiguous", OpSetId(Domain.TINYGRAD, 1))
    self.assertIs(op(Tensor.empty(4) + 1).uop.op, Ops.CONTIGUOUS)

  def test_tinygrad_contiguous_requires_domain(self):
    with self.assertRaises(NotImplementedError):
      self.runner._select_op("TinygradContiguous", OpSetId(Domain.ONNX, 1))

if __name__ == '__main__': unittest.main()
