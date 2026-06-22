from typing import Self
from tinygrad.dtype import ConstType, DType, DTypeLike
from tinygrad.mixin.dtype import DTypeMixin

class CreationMixin(DTypeMixin):
  def const_like(self, b: ConstType) -> Self: return self._wrap_uop(self._uop.const_like(b))

  def empty_like(self, dtype: DTypeLike|None=None, device: str|tuple[str, ...]|None=None) -> Self:
    """
    Creates an empty tensor with the same shape as `self`.
    If `dtype` is not specified, the dtype of `self` is used.
    """
    return self._wrap_uop(self._uop.empty_like(dtype, device))

  def full_like(self, fill_value: ConstType, dtype: DType|None=None) -> Self:
    """Creates a tensor with the same shape as `self`, filled with the given value."""
    return self.const_like(fill_value) if dtype is None else self.const_like(fill_value).cast(dtype)

  def zeros_like(self, **kwargs) -> Self:
    """
    Creates a tensor with the same shape as `self`, filled with zeros.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.ones(2, 3)
    print(Tensor.zeros_like(t).numpy())
    ```
    """
    return self.full_like(0, **kwargs)

  def ones_like(self, **kwargs) -> Self:
    """
    Creates a tensor with the same shape as `self`, filled with ones.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.zeros(2, 3)
    print(Tensor.ones_like(t).numpy())
    ```
    """
    return self.full_like(1, **kwargs)
