# mixins add syntactic sugar to Tensor and UOp
from typing import TypeAlias, TYPE_CHECKING, Self
from tinygrad.uop import Ops
from tinygrad.helpers import prod, argfix
if TYPE_CHECKING:
  from tinygrad.uop.ops import UOp
  sint:TypeAlias = UOp|int

class MovementMixin:
  # required to implement
  def _mop(self, op:Ops, arg) -> Self: raise NotImplementedError
  @property
  def shape(self) -> tuple["sint", ...]: raise NotImplementedError

  # great functions you get!
  @property
  def ndim(self) -> int:
    """
    Returns the number of dimensions in the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[1, 2], [3, 4]])
    print(t.ndim)
    ```
    """
    return len(self.shape)

  def numel(self) -> "sint":
    """
    Returns the total number of elements in the tensor.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print(t.numel())
    ```
    """
    return prod(self.shape)

  def _resolve_dim(self, dim:int, *, extra:bool=False) -> int:
    total = self.ndim + int(extra)
    if not -max(1, total) <= dim <= max(1, total)-1: raise IndexError(f"{dim=} out of range {[-max(1, total), max(1, total)-1]}")
    return dim + total if dim < 0 else dim

  def view(self, shape, *args) -> Self:
    """`.view` is an alias for `.reshape`."""
    return self.reshape(shape, *args)

  def reshape(self, shape, *args) -> Self:
    """
    Returns a tensor with the same data as the original tensor but with a different shape.
    `shape` can be passed as a tuple or as separate arguments.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(6)
    print(t.reshape(2, 3).numpy())
    ```
    """
    # resolve None and args
    new_shape = tuple([s if s is not None else self.shape[i] for i,s in enumerate(argfix(shape, *args))])
    # resolve -1
    if (c := new_shape.count(-1)) > 1: raise RuntimeError(f"only one dimension can be inferred using -1, getting {new_shape}")
    if c: new_shape = tuple([-prod(self.shape) // prod(new_shape) if s == -1 else s for s in new_shape])
    if prod(self.shape) != prod(new_shape): raise ValueError(f"size mismatch, can't reshape ({self.shape}) -> ({new_shape})")
    return self._mop(Ops.RESHAPE, arg=new_shape) if new_shape != self.shape else self

  def permute(self, order, *args) -> Self:
    """
    Returns a tensor that is a permutation of the original tensor.
    The new tensor has the same data as the original tensor but with the dimensions permuted according to the order specified.
    `order` can be passed as a tuple or as separate arguments.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.empty(2, 3, 5)
    print(t.shape)
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.permute(2, 0, 1).shape)
    ```
    """
    order_arg = tuple(self._resolve_dim(x) for x in argfix(order, *args))
    if sorted(order_arg) != list(range(self.ndim)): raise RuntimeError(f"order is not a valid permutation, getting {order_arg}")
    return self._mop(Ops.PERMUTE, arg=order_arg) if order_arg != tuple(range(self.ndim)) else self

  def flatten(self, start_dim=0, end_dim=-1) -> Self:
    """
    Flattens the tensor by reshaping it into a one-dimensional tensor.
    If `start_dim` or `end_dim` are passed, only dimensions starting with `start_dim` and ending with `end_dim` are flattened.

    ```python exec="true" source="above" session="tensor" result="python"
    t = Tensor.arange(8).reshape(2, 2, 2)
    print(t.flatten().numpy())
    ```
    ```python exec="true" source="above" session="tensor" result="python"
    print(t.flatten(start_dim=1).numpy())
    ```
    """
    start_dim, end_dim = self._resolve_dim(start_dim), self._resolve_dim(end_dim)
    return self.reshape(self.shape[:start_dim] + (prod(self.shape[start_dim:end_dim+1]), ) + self.shape[end_dim+1:])