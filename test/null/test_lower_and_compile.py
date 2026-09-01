import unittest
from tinygrad import Tensor
from tinygrad.uop.ops import UOp, Ops
from tinygrad.engine.realize import lower_and_compile

class TestLowerAndCompile(unittest.TestCase):
  def test_program_goes_inside_hcq_wrapper(self):
    linear = (Tensor.empty(16) + Tensor.empty(16)).schedule_linear()
    call = linear.src[0]
    wrapped = call.replace(src=(UOp.custom_function("hcq", call.src[0]), *call.src[1:]))
    body = lower_and_compile(linear.replace(src=(wrapped,))).src[0].src[0]
    self.assertIs(body.op, Ops.CUSTOM_FUNCTION)
    self.assertIs(body.src[0].op, Ops.PROGRAM)

if __name__ == '__main__':
  unittest.main()
