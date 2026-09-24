import unittest
from typing import List, Callable
from tinygrad import Tensor

class TestOps(unittest.TestCase):
  def test_negative_dims(self):
    creation_methods: List[Callable[..., Tensor]] = [
      Tensor.empty,
      Tensor.rand,
      Tensor.zeros,
      Tensor.ones,
      Tensor.randn,
      Tensor.randint,
      Tensor.normal,
      Tensor.uniform,
      Tensor.scaled_uniform,
      Tensor.glorot_uniform
    ]

    for method in creation_methods:
      with self.assertRaises(ValueError): method(-3, 2)
      with self.assertRaises(ValueError): method((2, -3))
      with self.assertRaises(ValueError): method((2, -3, 0))

  def test_negative_dims_full(self):
    with self.assertRaises(ValueError): Tensor.full((-3,), 2)
    with self.assertRaises(ValueError): Tensor.full((2, -3), 4)
    with self.assertRaises(ValueError): Tensor.full((2, -3, 0), 4)

  def test_negative_dims_eye(self):
    with self.assertRaises(ValueError): Tensor.eye(-3, 3)
    with self.assertRaises(ValueError): Tensor.eye(3, -3)
    with self.assertRaises(ValueError): Tensor.eye(-3, -3)

  def test_negative_dims_kaiming(self):
    creation_methods = [Tensor.kaiming_uniform, Tensor.kaiming_normal]
    for method in creation_methods:
      with self.assertRaises(ValueError): method(-3, 3)
      with self.assertRaises(ValueError): method((-3, 3), 3)
      with self.assertRaises(ValueError): method((-3, -3), 3)

  def test_prod_dtype_arg(self):
    with self.assertRaises(AttributeError): Tensor([1.0, 2.0]).prod(dtype="")

  def test_slice_errors(self):
    a = Tensor.ones(4, 3)
    b = Tensor(2)
    with self.assertRaisesRegex(IndexError, "too many"): a[1, 77, 77, 77] # IndexError: (finds too many indices before the out of bounds)
    with self.assertRaisesRegex(IndexError, "out of bounds"): a[1, 3] # IndexError: (out of bounds).
    with self.assertRaisesRegex(IndexError, "out of bounds"): a[1, -4]
    with self.assertRaisesRegex(IndexError, "single ellipsis"): a[..., ...] # IndexError: only single ellipsis
    with self.assertRaises(ValueError): a[::0, 1] # no 0 strides
    with self.assertRaises(TypeError): a[:Tensor([3]), 1] # Tensor can't be used as a slice parameter
    with self.assertRaises(IndexError): b[:] # slice cannot be applied to a 0-dim tensor

  def test_scatter_no_reduce_tensor_src(self):
    with self.assertRaises(TypeError):
      Tensor.ones(4).scatter(dim=1, index=Tensor([0]), src=Tensor.ones(4), reduce="add")

if __name__ == '__main__':
  unittest.main()
