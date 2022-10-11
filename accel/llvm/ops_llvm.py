from typing import Tuple, Union
from tinygrad.shapetracker import ShapeTracker

# TODO: write this
class LLVMBuffer:
  def __init__(self, shape:Union[ShapeTracker, Tuple[int, ...]]):
    self.st = shape if isinstance(shape, ShapeTracker) else ShapeTracker(tuple(shape))
    self.shape = self.st.shape
