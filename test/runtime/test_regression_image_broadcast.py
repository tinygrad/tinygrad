import numpy as np
import pytest
from tinygrad import Tensor, Context

@pytest.mark.parametrize("shapes", [((2, 1, 3, 4), (1, 5, 4, 6)), ((3, 4), (2, 4, 5)), ((2, 3, 4), (1, 4, 5)), ((4,), (2, 4, 5))])
def test_image_dot_batch_broadcast(shapes):
  a, b = [np.arange(np.prod(s), dtype=np.float32).reshape(s) / 10 for s in shapes]
  with Context(FLOAT16=0): np.testing.assert_allclose(Tensor(a).image_dot(Tensor(b)).numpy(), a @ b, rtol=1e-5, atol=1e-5)


def test_image_dot_incompatible_batches():
  with pytest.raises(IndexError): Tensor.ones(2, 3, 4).image_dot(Tensor.ones(5, 4, 6))
