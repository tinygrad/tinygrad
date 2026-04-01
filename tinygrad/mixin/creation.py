from typing import Any, Self
from tinygrad.dtype import ConstType

class CreationMixin:
  def const_like(self, b: ConstType) -> Self: raise NotImplementedError

  def full_like(self, fill_value: ConstType, **kwargs: Any) -> Self: return self.const_like(fill_value)
  def zeros_like(self, **kwargs: Any) -> Self: return self.const_like(0)
  def ones_like(self, **kwargs: Any) -> Self: return self.const_like(1)
