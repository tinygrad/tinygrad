import numpy as np
import pytest
from tinygrad import Tensor, Context, dtypes

@pytest.mark.parametrize("shape", [(4,), (3, 4), (2, 3, 4)])
def test_image_dot_right_vector(shape):
  a, b = np.arange(np.prod(shape), dtype=np.float32).reshape(shape), np.arange(4, dtype=np.float32)
  with Context(FLOAT16=0):
    out = Tensor(a).image_dot(Tensor(b), dtype=dtypes.float32)
    np.testing.assert_allclose(out.numpy(), a @ b)
    assert out.dtype == dtypes.float32
