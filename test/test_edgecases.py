# end to end tests of tinygrad that you think might be edge cases.
# like using the documentation, write code you think should work.
# you can compare the outputs to torch or numpy, or just try your best to make tinygrad assert while doing things you think should be valid

# i'm only interested in tests that are failing but you think should pass
# mark them with @unittest.expectedFailure
# i'm not interested in tests that currently pass, i'm only interested in tests that you think should pass but don't.
# all the tests in here didn't pass until bugs were fixed
# get creative! think about things that failed in pytorch or tensorflow for a long time until they were fixed.
# the tests don't have to test the same parts of the code that these current ones test

# focus on making tinygrad throw runtime errors or assertions for valid things.
# confirm they are valid by doing the same thing in pytorch in the test.
# for any failing tests, explain why tinygrad is wrong and what the desired behavior should be.

# don't worry about running mypy for types, it's slow. focus on your tests only

import unittest
import numpy as np
import torch
from tinygrad import Tensor

class TestEdgeCases(unittest.TestCase):
  @unittest.expectedFailure
  def test_sort_empty(self):
    # Sorting an empty tensor works in PyTorch and should return empty
    # values and indices. tinygrad raises an error instead.
    torch_vals, torch_idxs = torch.tensor([]).sort()
    values, indices = Tensor([]).sort()
    np.testing.assert_equal(values.numpy(), torch_vals.numpy())
    np.testing.assert_equal(indices.numpy(), torch_idxs.numpy().astype(np.int32))

  @unittest.expectedFailure
  def test_dropout_rate_one(self):
    # out is full of NaNs it should be 0s
    with Tensor.train():
      out = Tensor.ones(100).dropout(1.0)
      np.testing.assert_allclose(out.numpy(), np.zeros(100))

if __name__ == "__main__":
  unittest.main()