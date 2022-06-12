# ShapeTracker allows many operations to a buffer that don't
# require a copy to be made. When initted, you assume the underlying
# buffer is contiguous in the shape. Then movement operations modify the
# buffer. ReduceOps and ProcessingOps "realize" the buffer.

# Buffers should extend this
class ShapeTracker:
  def __init__(self, *shape):
    self._shape = tuple(shape)
    self.strides = [1]
    for d in self.shape[::-1][:-1]:
      self.strides = [d*self.strides[0]] + self.strides

  @property
  def shape(self):
    return tuple(self._shape)

  def reshape(self, *new_shape):
    self._shape = tuple(new_shape)
    # restride (this is wrong if permuted)
    self.strides = [1]
    for d in self.shape[::-1][:-1]:
      self.strides = [d*self.strides[0]] + self.strides

  def permute(self, *axis):
    self._shape = [self.shape[a] for a in axis]
    self.strides = [self.strides[a] for a in axis]

  def slice(self, arg):
    pass

  def expand(self, new_shape):
    pass

  # this returns the index
  def __getitem__(self, val):
    ret = 0
    for i,v in enumerate(val):
      assert isinstance(v, int)
      assert v>=0 and v < self.shape[i]
      ret += self.strides[i]*v
    return ret
