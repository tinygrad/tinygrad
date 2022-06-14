# ShapeTracker allows many operations to a buffer that don't
# require a copy to be made. When initted, you assume the underlying
# buffer is contiguous in the shape. Then movement operations modify the
# buffer. ReduceOps and ProcessingOps "realize" the buffer.

from tinygrad.helpers import prod

def strides_for_shape(shape):
  strides = [1]
  for d in shape[::-1][:-1]:
    strides = [d*strides[0]] + strides
  return strides

class View:
  def __init__(self, shape, strides, offset=0):
    self.shape = shape
    self.strides = strides
    self.offset = offset
  
  def __getitem__(self, val):
    ret = self.offset
    for d,s in zip(self.shape[::-1], self.strides[::-1]):
      ret += (val%d) * s
      val //= d
    return ret

class ShapeTracker:
  def __init__(self, *shape):
    self.views = []
    self.views.append(View(shape, strides_for_shape(shape)))

  def __getitem__(self, val):
    for v in self.views[::-1]:
      val = v[val]
    return val

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
    strides = strides_for_shape(self.shape)
    for i,(x,y) in enumerate(zip(self.shape, new_shape)):
      if x != y:
        assert x == 1
        strides[i] = 0
    self.views.append(View(new_shape, strides))

  def flip(self, *axis):
    assert all([isinstance(x, int) and x >= 0 and x < len(self.shape) for x in axis])
    strides = strides_for_shape(self.shape)
    offset = 0
    for a in axis:
      offset += (self.shape[a]-1) * strides[a]
      strides[a] *= -1
    self.views.append(View(self.shape, strides, offset))

  def slice(self, arg):
    # NOTE: this slice can only shrink
    assert len(arg) == len(self.shape)
    assert all([x>=0 and y<=self.shape[i] for i,(x,y) in enumerate(arg)])

    strides = strides_for_shape(self.shape)
    offset = sum([strides[i]*x for i,(x,_) in enumerate(arg)])
    new_shape = [y-x for x,y in arg]
    self.views.append(View(new_shape, strides, offset))
