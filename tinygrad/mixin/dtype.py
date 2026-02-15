from tinygrad.dtype import DType, dtypes

class DTypeMixin:
  @property
  def dtype(self) -> DType: raise NotImplementedError

  def element_size(self) -> int:
    """Returns the number of bytes of a single element in the tensor."""
    return self.dtype.itemsize

  def is_floating_point(self) -> bool:
    """Returns `True` if the tensor contains floating point types, i.e. is one of `bool`, `float16`, `bfloat16`, `float32`, `float64`."""
    return dtypes.is_float(self.dtype)
