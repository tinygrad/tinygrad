import numpy as np
import pytest
from tinygrad import Tensor

@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4), (3, 0), (0, 3)])
@pytest.mark.parametrize("offset", [-10, -4, 0, 4, 10])
def test_diagonal_outside_matrix(shape, offset):
  a = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
  expected = np.diagonal(a, offset=offset, axis1=-2, axis2=-1)
  out = Tensor(a).diagonal(offset=offset, dim1=-2, dim2=-1)
  assert out.shape == expected.shape
  np.testing.assert_array_equal(out.numpy(), expected)
