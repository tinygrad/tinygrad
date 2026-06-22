from typing import TYPE_CHECKING, Self
from tinygrad.dtype import ConstType, DType, DTypeLike, Invalid, dtypes, to_dtype
from tinygrad.helpers import argfix
from tinygrad.mixin.dtype import DTypeMixin

if TYPE_CHECKING:
  from tinygrad.uop.ops import sint, UOp

class CreationMixin(DTypeMixin):
  @staticmethod
  def const(dtype, b): raise NotImplementedError

  def const_like(self, b: ConstType) -> Self: return self._wrap_uop(self._uop.const_like(b))

  def empty_like(self, dtype: DTypeLike|None=None, device: str|tuple[str, ...]|None=None) -> Self:
    """
    Creates an empty tensor with the same shape as `self`.
    If `dtype` is not specified, the dtype of `self` is used.
    """
    return self._wrap_uop(self._uop.empty_like(dtype, device))

  @classmethod
  def invalids(cls, *shape, device:str|tuple[str, ...]|None=None, dtype:DTypeLike|None=None) -> Self:
    """
    Creates a tensor with the given shape, filled with Invalid.

    This is an alternative to Tensor.empty when you want an "anonymous" buffer.

    Eventually Tensor.empty will be replaced by this.
    """
    return cls.full(argfix(*shape), Invalid, dtype=dtype, device=device)

  @classmethod
  def full(cls, shape:'tuple[sint, ...]', fill_value:'ConstType|UOp', dtype:DTypeLike|None=None,
           device:str|tuple[str, ...]|None=None, buffer=True) -> Self:
    """
    Creates a tensor with the given shape, filled with the given value.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Pass `buffer=False` to get a broadcast const value instead of a materialized buffer.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.full((2, 3), 42).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.full((2, 3), False).numpy())
    ```
    """
    # TODO: enable this check
    # if not buffer: assert device is None, "buffer=False does not support device specification"
    from tinygrad.uop.ops import UOp
    new_shape = argfix(shape)
    dt = to_dtype(dtype) if dtype is not None else None
    val = cls.const(dt or (fill_value.dtype if isinstance(fill_value, UOp) else dtypes.from_py(fill_value)), fill_value)
    val = val.reshape((1,)*len(new_shape)).expand(new_shape)
    return val.clone(device=device) if buffer else val

  def full_like(self, fill_value: ConstType, dtype: DType|None=None) -> Self:
    """Creates a tensor with the same shape as `self`, filled with the given value."""
    return self.const_like(fill_value) if dtype is None else self.const_like(fill_value).cast(dtype)

  @classmethod
  def zeros(cls, *shape, **kwargs) -> Self:
    """
    Creates a tensor with the given shape, filled with zeros.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.zeros(2, 3).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.zeros(2, 3, dtype=dtypes.int32).numpy())
    ```
    """
    return cls.full(argfix(*shape), 0.0, **kwargs)

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

  @classmethod
  def ones(cls, *shape, **kwargs) -> Self:
    """
    Creates a tensor with the given shape, filled with ones.

    You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
    Additionally, all other keyword arguments are passed to the constructor of the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.ones(2, 3).numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(Tensor.ones(2, 3, dtype=dtypes.int32).numpy())
    ```
    """
    return cls.full(argfix(*shape), 1.0, **kwargs)

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
