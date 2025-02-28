import os
import unittest
from typing import List, Dict, Optional, Union

os.environ["TYPED"] = "1"

from tinygrad import Tensor, typechecked
from tinygrad.dtype import dtypes

@typechecked
def func_with_tensor_param(x: Tensor) -> Tensor:
  """Test function with tensor parameter annotation."""
  return x

@typechecked
def func_with_float_param(x: Tensor, scale: float) -> Tensor:
  """Test function with float parameter annotation."""
  return x * scale

@typechecked
def func_with_complex_types(x: Tensor, y: List[Tensor], opts: Dict[str, Union[int, float]]) -> Tensor:
  """Test function with complex type annotations."""
  result = x 
  for t in y:
    result += t
  if "scale" in opts:
    # convert to float tensor to avoid dtype mismatch
    result = result.float() * opts["scale"]
  return result

@typechecked
def func_with_optional(x: Tensor, y: Optional[float] = None) -> Tensor:
  """Test function with optional parameter."""
  if y is not None:
    return x * y
  return x

class TestTypeChecking(unittest.TestCase):
  """Test suite for runtime type checking in tinygrad."""

  def setUp(self):
    """Set up test fixtures."""
    self.tensor = Tensor([1.0, 2.0, 3.0, 4.0], dtype=dtypes.float)  # use float dtype
    self.tensor_list = [
      Tensor([1.0, 2.0, 3.0, 4.0], dtype=dtypes.float),
      Tensor([5.0, 6.0, 7.0, 8.0], dtype=dtypes.float)
    ]
    self.opts_dict = {"scale": 2.0, "offset": 1}

  def test_tensor_param_valid(self):
    """Test valid tensor parameter."""
    result = func_with_tensor_param(self.tensor)
    self.assertEqual(result.shape, (4,))

  def test_tensor_param_invalid(self):
    """Test invalid tensor parameter raises TypeError."""
    with self.assertRaises(Exception):
      func_with_tensor_param("not a tensor")

  def test_float_param_valid(self):
    """Test valid float parameter."""
    result = func_with_float_param(self.tensor, 2.5)
    self.assertEqual(result.shape, (4,))

  def test_float_param_invalid(self):
    """Test invalid float parameter raises TypeError."""
    with self.assertRaises(Exception):
      func_with_float_param(self.tensor, "not a float")

  def test_complex_types_valid(self):
    """Test valid complex type parameters."""
    result = func_with_complex_types(self.tensor, self.tensor_list, self.opts_dict)
    self.assertEqual(result.shape, (4,))

  def test_complex_types_invalid_list(self):
    """Test invalid list parameter raises TypeError."""
    with self.assertRaises(Exception):
      func_with_complex_types(self.tensor, "not a list", self.opts_dict)

  def test_complex_types_invalid_dict(self):
    """Test invalid dict parameter raises TypeError."""
    with self.assertRaises(Exception):
      func_with_complex_types(self.tensor, self.tensor_list, "not a dict")

  def test_optional_param_none(self):
    """Test optional parameter with None value."""
    result = func_with_optional(self.tensor)
    self.assertEqual(result.shape, (4,))

  def test_optional_param_value(self):
    """Test optional parameter with a value."""
    result = func_with_optional(self.tensor, 2.5)
    self.assertEqual(result.shape, (4,))

  def test_optional_param_invalid(self):
    """Test invalid optional parameter raises TypeError."""
    with self.assertRaises(Exception):
      func_with_optional(self.tensor, "not a float")

if __name__ == "__main__":
  unittest.main()
