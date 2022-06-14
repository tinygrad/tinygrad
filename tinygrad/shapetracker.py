# ShapeTracker allows many operations to a buffer that don't require a copy to be made.
# When initted, you assume the underlying buffer is contiguous in the shape.

from tinygrad.helpers import prod
from functools import cached_property

def strides_for_shape(shape):
  strides = [1]
  for d in shape[::-1][:-1]:
    strides = [d*strides[0]] + strides
  return strides

class View:
  def __init__(self, shape, strides, offset=0):
    assert len(shape) == len(strides)
    self.shape, self.strides, self.offset = shape, [], offset

    self.realshape = [shape[0]]
    self.strides = [strides[0]]
    for i in range(1, len(shape)):
      if strides[i] != 0 and self.strides[-1]//strides[i] == shape[i]:
        self.realshape[-1] *= shape[i]
        self.strides[-1] = strides[i]
      else:
        self.realshape.append(shape[i])
        self.strides.append(strides[i])

  @cached_property
  def expr(self):
    ret = [f"{self.offset}"] if self.offset != 0 else []
    acc = 1
    for i,(d,s) in enumerate(zip(self.realshape[::-1], self.strides[::-1])):
      if d != 1 and s != 0:
        lr = f"(idx//{acc})" if acc != 1 else "idx"
        lr = f"({lr}%{d})" if i != len(self.realshape)-1 else lr  # don't mod the top shape dimension
        lr = f"({lr}*{s})" if s != 1 else lr
        ret.append(lr)
      acc *= d
    return '+'.join(ret) if len(ret) > 0 else "0"
  
  def __getitem__(self, idx):
    return eval(self.expr())

class ShapeTracker:
  def __init__(self, *shape):
    self.views = [View(shape, strides_for_shape(shape))]

  def __getitem__(self, val):
    locals = {"idx": val}
    exec(self.expr(), None, locals)
    return locals["idx"]

  def expr(self):
    return ';'.join([f"idx={v.expr}" for v in self.views[::-1] if v.expr != 'idx'])

  @property
  def shape(self):
    return tuple(self.views[-1].shape)

  def reshape(self, *new_shape):
    assert all([isinstance(x, int) for x in new_shape])
    assert prod(self.shape) == prod(new_shape)
    self.views.append(View(new_shape, strides_for_shape(new_shape)))

  def permute(self, *axis):
    assert all([isinstance(x, int) and x >= 0 and x < len(self.shape) for x in axis])
    assert len(set(axis)) == len(axis)
    shape = [self.shape[a] for a in axis]
    strides = strides_for_shape(self.shape)
    strides = [strides[a] for a in axis]
    self.views.append(View(shape, strides))

  def expand(self, *new_shape):
    assert all([isinstance(x, int) for x in new_shape])
    assert all([x == y or x == 1 for x,y in zip(self.shape, new_shape)])
    strides = strides_for_shape(self.shape)
    strides = [s if x == y else 0 for s,(x,y) in zip(strides, zip(self.shape, new_shape))]
    self.views.append(View(new_shape, strides))

  def flip(self, *axis):
    assert all([isinstance(x, int) and x >= 0 and x < len(self.shape) for x in axis])
    strides = strides_for_shape(self.shape)
    offset = 0
    for a in axis:
      offset += (self.shape[a]-1) * strides[a]
      strides[a] *= -1
    self.views.append(View(self.shape, strides, offset))

  def slice(self, arg):  # NOTE: this slice can only shrink
    assert len(arg) == len(self.shape)
    assert all([x>=0 and y<=self.shape[i] for i,(x,y) in enumerate(arg)])

    strides = strides_for_shape(self.shape)
    offset = sum([strides[i]*x for i,(x,_) in enumerate(arg)])
    new_shape = [y-x for x,y in arg]
    self.views.append(View(new_shape, strides, offset))
