from typing import Self
from tinygrad.mixin.elementwise import ElementwiseMixin
from tinygrad.mixin.movement import MovementMixin
from tinygrad.uop.ops import _broadcast_shape
from tinygrad.dtype import least_upper_dtype


class OpMixin(ElementwiseMixin, MovementMixin):
  def _broadcasted(self, y, reverse=False) -> tuple[Self, Self]:
    if not isinstance(y, type(self)): y = self.ufix(y)
    x, y = (self, y) if not reverse else (y, self)
    out_shape, out_dtype = _broadcast_shape(x.shape, y.shape), least_upper_dtype(x.dtype, y.dtype)
    return x._broadcast_to(out_shape).cast(out_dtype), y._broadcast_to(out_shape).cast(out_dtype)
