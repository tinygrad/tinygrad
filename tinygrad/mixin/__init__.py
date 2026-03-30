from typing import Self
from tinygrad.mixin.elementwise import ElementwiseMixin
from tinygrad.mixin.reduce import ReduceMixin
from tinygrad.uop.ops import _broadcast_shape
from tinygrad.dtype import least_upper_dtype


class OpMixin(ElementwiseMixin, ReduceMixin):
  def _broadcasted(self, y, reverse=False) -> tuple[Self, Self]:
    if not isinstance(y, type(self)): y = self.ufix(y)
    x, y = (self, y) if not reverse else (y, self)
    try:
      out_shape = _broadcast_shape(x.shape, y.shape)
      x, y = x._broadcast_to(out_shape), y._broadcast_to(out_shape)
    except RuntimeError: pass
    out_dtype = least_upper_dtype(x.dtype, y.dtype)
    return x.cast(out_dtype), y.cast(out_dtype)
