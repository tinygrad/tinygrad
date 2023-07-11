from tinygrad.DType import DType
from typing import Tuple
# dependent typing?
class ImageDType(DType):
  sz = DType.sz

  def __new__(cls, priority, itemsize, name, np, shape):
    return super().__new__(cls, priority, itemsize, name, np)

  def __init__(self, priority, itemsize, name, np, shape):
    self.shape: Tuple[int, ...] = shape  # arbitrary arg for the dtype, used in image for the shape
    super().__init__()

  def __repr__(self): return f"dtypes.{self.name}({self.shape})"
