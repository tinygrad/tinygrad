# ShapeTracker allows movement operations to a buffer that don't require a copy to be made.
from tinygrad.helpers import prod
from functools import cached_property

def divmodidx(acc, d, mod=True):
  lr = f"(idx//{acc})" if acc != 1 else "idx"
  return f"({lr}%{d})" if mod else lr  # don't mod the top shape dimension

class View:
  def __init__(self, shape, strides, offset=0):
    assert len(shape) == len(strides)
    self.shape, self.offset = shape, offset

    self.shape_strides = [(shape[0], strides[0])]
    for i in range(1, len(shape)):
      if strides[i] != 0 and self.shape_strides[-1][1]//strides[i] == shape[i]:
        self.shape_strides[-1] = (self.shape_strides[-1][0] * shape[i], strides[i])
      else:
        self.shape_strides.append((shape[i], strides[i]))

  @cached_property
  def expr(self):
    ret = [f"{self.offset}"] if self.offset != 0 else []
    acc = 1
    for i,(d,s) in enumerate(self.shape_strides[::-1]):
      if d != 1 and s != 0:
        lr = divmodidx(acc, d, i != len(self.shape_strides)-1)
        lr = f"({lr}*{s})" if s != 1 else lr
        ret.append(lr)
      acc *= d
    return 'idx=' + ('+'.join(ret) if len(ret) > 0 else "0")

class ZeroView:
  def __init__(self, old_shape, arg):
    expr = ['valid']
    self.shape = []
    acc = 1
    for s,(x,y) in list(zip(old_shape, arg))[::-1]:
      self.shape = [y-x] + self.shape
      base = divmodidx(acc, self.shape[0], len(self.shape) != len(old_shape)) + f"+{x}"
      if x < 0: expr.append(f"(({base}) >= 0)")
      if y > s: expr.append(f"(({base}) < {s})")
      acc *= self.shape[0]
    self.expr = 'valid=' + ' && '.join(expr)

def strides_for_shape(shape):
  strides = [1]
  for d in shape[::-1][:-1]:
    strides = [d*strides[0]] + strides
  return strides

# TODO: support "contiguous" property and "simple_strided" property and make them work
# TODO: support simplification across views
class ShapeTracker:
  def __init__(self, *shape, strides=None):
    if len(shape) == 0: shape = (1,)
    self.views = [View(shape, strides_for_shape(shape) if strides == None else strides)]

  @property
  def shape(self): return tuple(self.views[-1].shape)

  def expr(self): return ';'.join([v.expr for v in self.views[::-1] if v.expr != 'idx=idx' and v.expr != 'valid=valid'])
  def movement_op(self, op, arg): getattr(self, str(op).split(".")[1].lower())(*arg); return self

  def __getitem__(self, val):
    locals = {"idx": val}
    exec(self.expr(), None, locals)
    return locals["idx"]

  def reshape(self, *new_shape):
    assert all([isinstance(x, int) for x in new_shape])
    assert prod(self.shape) == prod(new_shape)
    self.views.append(View(new_shape, strides_for_shape(new_shape)))

  def permute(self, *axis):
    assert all([isinstance(x, int) and x >= 0 and x < len(self.shape) for x in axis])
    assert len(set(axis)) == len(axis)
    strides = strides_for_shape(self.shape)
    self.views.append(View([self.shape[a] for a in axis], [strides[a] for a in axis]))

  def slice(self, *arg):  # NOTE: this slice cannot pad the edges
    assert len(arg) == len(self.shape)
    # if you pass in an expansion, it will be correct for the shrink, but return junk in the expanded region
    #assert all([x>=0 and y<=self.shape[i] for i,(x,y) in enumerate(arg)])
    strides = strides_for_shape(self.shape)
    offset = sum([strides[i]*x for i,(x,_) in enumerate(arg)])
    self.views += [View([y-x for x,y in arg], strides, offset), ZeroView(self.shape, arg)]

  def expand(self, *new_shape):
    assert all([isinstance(x, int) for x in new_shape])
    assert all([x == y or x == 1 for x,y in zip(self.shape, new_shape)])
    strides = [s if x == y else 0 for s,(x,y) in zip(strides_for_shape(self.shape), zip(self.shape, new_shape))]
    self.views.append(View(new_shape, strides))

  def flip(self, *axis):
    assert all([isinstance(x, int) and x >= 0 and x < len(self.shape) for x in axis])
    strides = [-s if i in axis else s for i,s in enumerate(strides_for_shape(self.shape))]
    offset = sum([(self.shape[a] - 1) * -strides[a] for a in axis])
    self.views.append(View(self.shape, strides, offset))
