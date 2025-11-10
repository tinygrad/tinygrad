from tinygrad import Tensor, UOp
from tinygrad.uop.ops import Ops, AxisType
import unittest
# this test is only focused on transformers and using range for the layers

class TestOuterworldTransformer(unittest.TestCase):
  def test_three_mats(self):
    w = Tensor.empty(3, 1024, 1024)
    inp = Tensor.empty(1, 1024)
    i = UOp.range(3, -1, AxisType.OUTER)
    inp_after = Tensor(inp.uop.after(i))
    inp_gemm = inp_after@w[i]
    inp = Tensor(inp_gemm.uop.reduce(i, arg=Ops.MAX))
    inp.realize()

if __name__ == "__main__":
  unittest.main()
