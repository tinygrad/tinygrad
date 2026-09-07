"""Compact compiled signatures must select the corresponding raw call arguments."""
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import KernelInfo, UOp


def test_unused_middle_buffer_does_not_replace_the_live_input():
  # Program.globals is (0,2), whereas the call still carries all three buffers.
  # Distinct input values expose accidentally binding compact slot1 to raw slot1.
  def kernel(output:UOp, unused:UOp, source:UOp):
    index = UOp.range(5, 0)
    return output[index].store(source[index] * 3 - 1).end(index).sink(arg=KernelInfo(name='sparse_arguments'))

  output = Tensor.empty(5, dtype=dtypes.int32)
  unused = Tensor(np.array([100, 200, 300, 400, 500], dtype=np.int32))
  source = Tensor(np.array([-7, 0, 2, 9, 23], dtype=np.int32))
  actual = Tensor.custom_kernel(output, unused, source, fxn=kernel)[0].numpy()
  np.testing.assert_array_equal(actual, [-22, -1, 5, 26, 68])
